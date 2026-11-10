"""Standalone TLE split-N scheduling probe; not part of default dispatch."""

import argparse
import importlib
import statistics

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.tools.tensor_descriptor import TensorDescriptor



@triton.jit
def _probe_producer(writer, a_desc, b_desc, n_offset, k_tiles: tl.constexpr,
                    block_m: tl.constexpr, block_k: tl.constexpr):
    for k_iter in range(k_tiles):
        k_offset = k_iter * block_k
        slot = writer.acquire(k_iter)
        tle.gpu.copy(a_desc, slot.a, [block_m, block_k], [0, k_offset])
        tle.gpu.copy(b_desc, slot.b, [block_k, 256], [k_offset, n_offset])
        writer.commit(k_iter)


@triton.jit
def _probe_consumer(reader, c_ptr, n_offset, stride_cm, stride_cn, m, n,
                    k_tiles: tl.constexpr, block_m: tl.constexpr,
                    mma_group: tl.constexpr):
    acc_left = tl.zeros((block_m, 128), dtype=tl.float32)
    acc_right = tl.zeros((block_m, 128), dtype=tl.float32)
    for group_start in tl.static_range(0, k_tiles, mma_group):
        for group_offset in tl.static_range(mma_group):
            k_iter = group_start + group_offset
            if k_iter < k_tiles:
                ready = reader.wait(k_iter)
                b_left = ready.slot.b.slice(0, 128, dim=1)
                b_right = ready.slot.b.slice(128, 128, dim=1)
                acc_left = tle.gpu.wgmma(ready.slot.a, b_left, acc_left)
                acc_right = tle.gpu.wgmma(ready.slot.a, b_right, acc_right)
        acc_left = tle.gpu.wgmma_wait(0, acc_left)
        acc_right = tle.gpu.wgmma_wait(0, acc_right)
        for group_offset in tl.static_range(mma_group):
            k_iter = group_start + group_offset
            if k_iter < k_tiles:
                reader.release(k_iter)

    rm = tl.arange(0, block_m)
    rn = tl.arange(0, 128)
    row_mask = (rm < m)[:, None]
    left_ptr = c_ptr + rm[:, None] * stride_cm + (rn[None, :] + n_offset) * stride_cn
    right_ptr = c_ptr + rm[:, None] * stride_cm + (rn[None, :] + n_offset + 128) * stride_cn
    tl.store(left_ptr, acc_left.to(c_ptr.dtype.element_ty),
             mask=row_mask & ((rn + n_offset) < n)[None, :])
    tl.store(right_ptr, acc_right.to(c_ptr.dtype.element_ty),
             mask=row_mask & ((rn + n_offset + 128) < n)[None, :])


@triton.jit
def one_pipe_kernel(a_desc, b_desc, c_ptr, m, n, stride_cm, stride_cn,
                    k_tiles: tl.constexpr, block_m: tl.constexpr,
                    block_k: tl.constexpr, slots: tl.constexpr,
                    mma_group: tl.constexpr, worker_warps: tl.constexpr):
    n_offset = (tl.program_id(0) * 256).to(tl.int32)
    a_smem = tle.gpu.alloc([slots, block_m, block_k], dtype=a_desc.dtype,
                            layout=None, scope=tle.gpu.smem,
                            nv_mma_shared_layout=True)
    b_smem = tle.gpu.alloc([slots, block_k, 256], dtype=b_desc.dtype,
                            layout=None, scope=tle.gpu.smem,
                            nv_mma_shared_layout=True)
    pipe = tle.pipe(capacity=slots, scope="cta", name="probe_one_pipe",
                    a=a_smem, b=b_smem)
    tle.gpu.warp_specialize(
        [(_probe_consumer, (pipe.reader(), c_ptr, n_offset, stride_cm, stride_cn,
                            m, n, k_tiles, block_m, mma_group)),
         (_probe_producer, (pipe.writer(), a_desc, b_desc, n_offset, k_tiles,
                            block_m, block_k))],
        [worker_warps], [24],
    )


@triton.jit
def _fused_k128_producer(writer, a_desc, b_desc, n_offset,
                          groups: tl.constexpr, block_m: tl.constexpr):
    # Store two legal K=64 tiles in one pipe slot.  This emulates the
    # bandwidth/synchronization behavior of muDNN's B128 kernels without
    # requesting the unsupported generic BF16/FP16 K=128 SQMMA intrinsic.
    for group in range(groups):
        k_offset = group * 128
        slot = writer.acquire(group)
        # The MTT pipe contract permits one TMA transaction per payload.  Use
        # descriptors with a K=128 block and split the shared tile only at the
        # SQMMA consumer, where both halves are legal K=64 operations.
        tle.gpu.copy(a_desc, slot.a, [block_m, 128], [0, k_offset])
        tle.gpu.copy(b_desc, slot.b, [128, 256], [k_offset, n_offset])
        writer.commit(group)


@triton.jit
def _fused_k128_consumer(reader, c_ptr, n_offset, stride_cm, stride_cn,
                          m, n, groups: tl.constexpr, block_m: tl.constexpr):
    acc_left = tl.zeros((block_m, 128), dtype=tl.float32)
    acc_right = tl.zeros((block_m, 128), dtype=tl.float32)
    for group in range(groups):
        ready = reader.wait(group)
        a0 = ready.slot.a.slice(0, 64, dim=1)
        a1 = ready.slot.a.slice(64, 64, dim=1)
        b_left = ready.slot.b.slice(0, 128, dim=1)
        b_right = ready.slot.b.slice(128, 128, dim=1)
        b0_left = b_left.slice(0, 64, dim=0)
        b1_left = b_left.slice(64, 64, dim=0)
        b0_right = b_right.slice(0, 64, dim=0)
        b1_right = b_right.slice(64, 64, dim=0)
        acc_left = tle.gpu.wgmma(a0, b0_left, acc_left)
        acc_left = tle.gpu.wgmma(a1, b1_left, acc_left)
        acc_right = tle.gpu.wgmma(a0, b0_right, acc_right)
        acc_right = tle.gpu.wgmma(a1, b1_right, acc_right)
        acc_left = tle.gpu.wgmma_wait(0, acc_left)
        acc_right = tle.gpu.wgmma_wait(0, acc_right)
        reader.release(group)
    rm = tl.arange(0, block_m)
    rn = tl.arange(0, 128)
    row_mask = (rm < m)[:, None]
    left_ptr = c_ptr + rm[:, None] * stride_cm + (rn[None, :] + n_offset) * stride_cn
    right_ptr = c_ptr + rm[:, None] * stride_cm + (rn[None, :] + n_offset + 128) * stride_cn
    tl.store(left_ptr, acc_left.to(c_ptr.dtype.element_ty),
             mask=row_mask & ((rn + n_offset) < n)[None, :])
    tl.store(right_ptr, acc_right.to(c_ptr.dtype.element_ty),
             mask=row_mask & ((rn + n_offset + 128) < n)[None, :])


@triton.jit
def fused_k128_kernel(a_desc, b_desc, c_ptr, m, n, stride_cm, stride_cn,
                      groups: tl.constexpr, block_m: tl.constexpr,
                      slots: tl.constexpr, worker_warps: tl.constexpr):
    n_offset = (tl.program_id(0) * 256).to(tl.int32)
    a_smem = tle.gpu.alloc([slots, block_m, 128], dtype=a_desc.dtype,
                            layout=None, scope=tle.gpu.smem,
                            nv_mma_shared_layout=True)
    b_smem = tle.gpu.alloc([slots, 128, 256], dtype=b_desc.dtype,
                            layout=None, scope=tle.gpu.smem,
                            nv_mma_shared_layout=True)
    pipe = tle.pipe(capacity=slots, scope="cta", name="probe_fused_k128",
                    a=a_smem, b=b_smem)
    tle.gpu.warp_specialize(
        [(_fused_k128_consumer, (pipe.reader(), c_ptr, n_offset,
                                 stride_cm, stride_cn, m, n, groups, block_m)),
         (_fused_k128_producer, (pipe.writer(), a_desc, b_desc, n_offset,
                                 groups, block_m))],
        [worker_warps], [24],
    )


def bench(fn, warmup=10, rep=40, repeats=9):
    for _ in range(warmup):
        fn()
    torch.musa.synchronize()
    vals = []
    for _ in range(repeats):
        st = torch.musa.Event(enable_timing=True)
        en = torch.musa.Event(enable_timing=True)
        st.record()
        for _ in range(rep):
            fn()
        en.record()
        torch.musa.synchronize()
        vals.append(st.elapsed_time(en) / rep)
    return statistics.median(vals), vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", nargs=3, type=int, required=True)
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--group", type=int, default=1)
    ap.add_argument("--worker-warps", type=int, default=8)
    ap.add_argument("--rep", type=int, default=40)
    ap.add_argument("--repeats", type=int, default=9)
    ap.add_argument("--only-probe", action="store_true")
    ap.add_argument("--fused-k128", action="store_true")
    args = ap.parse_args()
    m, n, k = args.shape
    block_m = 64 if m <= 64 else (128 if m <= 128 else 256)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    a = torch.randn((m, k), device="musa", dtype=dtype)
    b = torch.randn((k, n), device="musa", dtype=dtype)
    out = torch.empty((m, n), device="musa", dtype=dtype)
    ref = torch.mm(a, b)
    mm_mod = importlib.import_module("flag_gems.runtime.backend._mthreads.ops.mm")
    ad = TensorDescriptor.from_tensor(a, [block_m, 64])
    bd = TensorDescriptor.from_tensor(b, [64, 256])
    grid = (triton.cdiv(n, 256),)

    def probe():
        one_pipe_kernel[grid](ad, bd, out, m, n, out.stride(0), out.stride(1),
                              triton.cdiv(k, 64), block_m, 64,
                              args.slots, args.group, args.worker_warps,
                              num_warps=args.worker_warps,
                              enable_backend_opt=True)

    def fused_k128():
        # The experimental path requires K to be a multiple of 128 and uses
        # one slot by default to stay below the S5000 shared-memory limit.
        if k % 128:
            raise ValueError("fused K128 probe requires K % 128 == 0")
        ad128 = TensorDescriptor.from_tensor(a, [block_m, 128])
        bd128 = TensorDescriptor.from_tensor(b, [128, 256])
        fused_k128_kernel[grid](ad128, bd128, out, m, n, out.stride(0), out.stride(1),
                                triton.cdiv(k, 128), block_m, args.slots,
                                args.worker_warps, num_warps=args.worker_warps,
                                enable_backend_opt=True)

    def baseline():
        mm_mod.mm_tle_split_n_pipe(a, b, out, m, n, k, block_k=64,
                                   num_slots=2, mma_group=1)

    def single_direct():
        mm_mod.mm_tle_split_n_single_pipe(a, b, out, m, n, k)

    def integrated():
        mm_mod.mm_out(a, b, out=out)

    def dispatch():
        mm_mod.mm_tle_pipe(a, b, out, m, n, k)

    probe()
    torch.musa.synchronize()
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
    if args.fused_k128:
        fused_k128()
        torch.musa.synchronize()
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
        torch_ms, _ = bench(lambda: torch.mm(a, b), rep=args.rep)
        fused_ms, fused_samples = bench(fused_k128, rep=args.rep)
        print(f"RESULT fused_k128 {m} {n} {k} {args.dtype} slots={args.slots} "
              f"torch_ms={torch_ms:.6f} fused_ms={fused_ms:.6f} "
              f"ratio={torch_ms/fused_ms:.4f} samples={fused_samples}", flush=True)
        return
    if args.only_probe:
        torch_ms, torch_vals = bench(lambda: torch.mm(a, b), rep=args.rep, repeats=args.repeats)
        probe_ms, vals = bench(probe, rep=args.rep, repeats=args.repeats)
        print(f"RESULT probe_only {m} {n} {k} {args.dtype} "
              f"torch_ms={torch_ms:.6f} probe_ms={probe_ms:.6f} "
              f"ratio={torch_ms/probe_ms:.4f} torch_samples={torch_vals} samples={vals}", flush=True)
        return
    torch_ms, _ = bench(lambda: torch.mm(a, b), rep=args.rep)
    baseline()
    torch.musa.synchronize()
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
    baseline_ms, baseline_samples = bench(baseline, rep=args.rep)
    probe_ms, vals = bench(probe, rep=args.rep)
    single_direct()
    torch.musa.synchronize()
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
    single_ms, single_samples = bench(single_direct, rep=args.rep)
    integrated()
    torch.musa.synchronize()
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
    integrated_ms, integrated_samples = bench(integrated, rep=args.rep)
    dispatch()
    torch.musa.synchronize()
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
    dispatch_ms, dispatch_samples = bench(dispatch, rep=args.rep)
    print(f"RESULT {m} {n} {k} {args.dtype} slots={args.slots} group={args.group} "
          f"torch_ms={torch_ms:.6f} baseline_ms={baseline_ms:.6f} "
          f"probe_ms={probe_ms:.6f} single_ms={single_ms:.6f} integrated_ms={integrated_ms:.6f} "
          f"baseline_ratio={torch_ms/baseline_ms:.4f} probe_ratio={torch_ms/probe_ms:.4f} "
          f"single_ratio={torch_ms/single_ms:.4f} integrated_ratio={torch_ms/integrated_ms:.4f} dispatch_ms={dispatch_ms:.6f} "
          f"dispatch_ratio={torch_ms/dispatch_ms:.4f} "
          f"samples={baseline_samples} {vals} {single_samples} {integrated_samples} {dispatch_samples}", flush=True)


if __name__ == "__main__":
    main()
