"""Probe a 256-column ordered two-consumer tile for BN320 tail splitting."""
import argparse
import importlib
import statistics

import torch
import torch_musa  # noqa: F401
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from triton.tools.tensor_descriptor import TensorDescriptor

tle = importlib.import_module("triton.experimental.tle.language")


@triton.jit
def _producer(writer, a_desc, b_desc, m_offset, n_offset, k_tiles: tl.constexpr):
    for k_iter in range(k_tiles):
        k_offset = k_iter * 64
        slot = writer.acquire(k_iter)
        tle.gpu.copy(a_desc, slot.a, [256, 64], [m_offset, k_offset])
        tle.gpu.copy(b_desc, slot.b, [64, 256], [k_offset, n_offset])
        writer.commit(k_iter)


@triton.jit
def _consumer(reader, to_top, to_bottom, c_ptr, m_offset, n_offset,
              consumer_id: tl.constexpr, row_offset: tl.constexpr,
              block_m: tl.constexpr, stride_cm, stride_cn, m, n,
              k_tiles: tl.constexpr):
    acc = tl.zeros((block_m, 256), dtype=tl.float32)
    for k_iter in range(k_tiles):
        ready = reader.wait(k_iter)
        if consumer_id == 0:
            tle.gpu.barrier_wait(to_top, phaseIdx=(k_iter + 1) & 1)
        else:
            tle.gpu.barrier_wait(to_bottom, phaseIdx=k_iter & 1)
        a_tile = ready.slot.a.slice(row_offset, block_m, dim=0)
        acc = tle.gpu.wgmma(a_tile, ready.slot.b, acc)
        if consumer_id == 0:
            tle.gpu.barrier_arrive(to_bottom, phaseIdx=k_iter & 1)
        else:
            tle.gpu.barrier_arrive(to_top, phaseIdx=k_iter & 1)
        acc = tle.gpu.wgmma_wait(0, acc)
        reader.release(k_iter)
    rm = m_offset + row_offset + tl.arange(0, block_m)
    rn = n_offset + tl.arange(0, 256)
    ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(ptrs, acc.to(c_ptr.dtype.element_ty),
             mask=(rm < m)[:, None] & (rn < n)[None, :])


@triton.jit
def bn256_ordered_kernel(a_desc, b_desc, c_ptr, m, n, stride_cm, stride_cn,
                         grid_m: tl.constexpr, grid_n: tl.constexpr,
                         k_tiles: tl.constexpr, slots: tl.constexpr):
    pid = tl.program_id(0)
    group_m: tl.constexpr = min(grid_m, 2)
    group_width: tl.constexpr = group_m * grid_n
    group_id = pid // group_width
    first_m = group_id * group_m
    actual_group_m = min(grid_m - first_m, group_m)
    pid_in_group = pid % group_width
    pid_m = first_m + pid_in_group % actual_group_m
    pid_n = pid_in_group // actual_group_m
    m_offset = (pid_m * 256).to(tl.int32)
    n_offset = (pid_n * 256).to(tl.int32)
    a_smem = tle.gpu.alloc([slots, 256, 64], dtype=a_desc.dtype,
                            layout=None, scope=tle.gpu.smem,
                            nv_mma_shared_layout=True)
    b_smem = tle.gpu.alloc([slots, 64, 256], dtype=b_desc.dtype,
                            layout=None, scope=tle.gpu.smem,
                            nv_mma_shared_layout=True)
    pipe = tle.pipe(capacity=slots, scope="cta", name="bn256_probe",
                    a=a_smem, b=b_smem)
    to_top = tle.gpu.alloc_barrier(arrive_count=8, init=tle.gpu.PENDING)
    to_bottom = tle.gpu.alloc_barrier(arrive_count=8, init=tle.gpu.PENDING)
    tle.gpu.warp_specialize(
        [(_consumer, (pipe.reader(), to_top, to_bottom, c_ptr, m_offset,
                      n_offset, 0, 0, 128, stride_cm, stride_cn, m, n,
                      k_tiles)),
         (_consumer, (pipe.reader(), to_top, to_bottom, c_ptr, m_offset,
                      n_offset, 1, 128, 128, stride_cm, stride_cn, m, n,
                      k_tiles)),
         (_producer, (pipe.writer(), a_desc, b_desc, m_offset, n_offset,
                      k_tiles))],
        [8, 4], [168, 24],
    )


def run(a, b, out, slots):
    m, k = a.shape
    n = b.shape[1]
    da = TensorDescriptor.from_tensor(a, [256, 64])
    db = TensorDescriptor.from_tensor(b, [64, 256])
    gm = triton.cdiv(m, 256)
    gn = triton.cdiv(n, 256)
    with torch_device_fn.device(a.device):
        bn256_ordered_kernel[(gm * gn,)](
            da, db, out, m, n, out.stride(0), out.stride(1), gm, gn,
            triton.cdiv(k, 64), slots, num_warps=8, enable_backend_opt=True)


def bench(fn, warmup, rep):
    for _ in range(warmup):
        fn()
    torch.musa.synchronize()
    vals = []
    for _ in range(3):
        s, e = torch.musa.Event(enable_timing=True), torch.musa.Event(enable_timing=True)
        s.record()
        for _ in range(rep):
            fn()
        e.record(); torch.musa.synchronize()
        vals.append(s.elapsed_time(e) / rep)
    return statistics.median(vals), vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=448); ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--k", type=int, default=2048); ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument("--slots", type=int, default=2); ap.add_argument("--warmup", type=int, default=20); ap.add_argument("--rep", type=int, default=100)
    args = ap.parse_args(); dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    a = torch.randn((args.m, args.k), device="musa", dtype=dt); b = torch.randn((args.k, args.n), device="musa", dtype=dt)
    out = torch.empty((args.m, args.n), device="musa", dtype=dt); ref = torch.mm(a, b)
    fn = lambda: run(a, b, out, args.slots)
    fn(); torch.musa.synchronize(); torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
    base, _ = bench(lambda: torch.mm(a, b, out=out), args.warmup, args.rep)
    got, vals = bench(fn, args.warmup, args.rep)
    print(f"RESULT m={args.m} n={args.n} k={args.k} dtype={args.dtype} slots={args.slots} torch_ms={base:.6f} bn256_ms={got:.6f} ratio={base/got:.4f} samples={vals}", flush=True)


if __name__ == "__main__":
    main()
