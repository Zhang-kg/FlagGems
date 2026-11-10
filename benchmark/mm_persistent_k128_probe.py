"""Benchmark-only persistent BM256/BN256 with two K64 SQMMA per K128 slot."""

import argparse
import importlib
import statistics

import torch
import torch_musa  # noqa: F401
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems.runtime import torch_device_fn

tle = importlib.import_module("triton.experimental.tle.language")


@triton.jit
def _producer(writer, a_desc, b_desc, pid, total_tiles, grid_n, num_sms,
              tile_iters, groups: tl.constexpr):
    for tile_iter in range(tile_iters):
        tile_id = pid + tile_iter * num_sms
        if tile_id < total_tiles:
            pid_m = tile_id // grid_n
            pid_n = tile_id % grid_n
            m_offset = (pid_m * 256).to(tl.int32)
            n_offset = (pid_n * 256).to(tl.int32)
            for group in range(groups):
                token = tile_iter * groups + group
                k_offset = group * 128
                slot = writer.acquire(token)
                tle.gpu.copy(a_desc, slot.a, [256, 128], [m_offset, k_offset])
                tle.gpu.copy(b_desc, slot.b, [128, 256], [k_offset, n_offset])
                writer.commit(token)


@triton.jit
def _consumer(reader, c_ptr, pid, M, N, stride_cm, stride_cn,
              total_tiles, grid_n, num_sms, tile_iters,
              groups: tl.constexpr):
    for tile_iter in range(tile_iters):
        tile_id = pid + tile_iter * num_sms
        if tile_id < total_tiles:
            pid_m = tile_id // grid_n
            pid_n = tile_id % grid_n
            m_offset = (pid_m * 256).to(tl.int32)
            n_offset = (pid_n * 256).to(tl.int32)
            acc = tl.zeros((256, 256), dtype=tl.float32)
            for group in range(groups):
                token = tile_iter * groups + group
                ready = reader.wait(token)
                a0 = ready.slot.a.slice(0, 64, dim=1)
                a1 = ready.slot.a.slice(64, 64, dim=1)
                b0 = ready.slot.b.slice(0, 64, dim=0)
                b1 = ready.slot.b.slice(64, 64, dim=0)
                acc = tle.gpu.wgmma(a0, b0, acc)
                acc = tle.gpu.wgmma(a1, b1, acc)
                acc = tle.gpu.wgmma_wait(0, acc)
                reader.release(token)
            rm = m_offset + tl.arange(0, 256)
            rn = n_offset + tl.arange(0, 256)
            ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
            mask = (rm < M)[:, None] & (rn < N)[None, :]
            tl.store(ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)


@triton.jit
def persistent_k128_kernel(a_desc, b_desc, c_ptr, M, N, stride_cm, stride_cn,
                           TOTAL_TILES: tl.constexpr, GRID_N: tl.constexpr,
                           NUM_SMS: tl.constexpr, TILE_ITERS: tl.constexpr,
                           GROUPS: tl.constexpr, NUM_SLOTS: tl.constexpr):
    pid = tl.program_id(0)
    a_smem = tle.gpu.alloc([NUM_SLOTS, 256, 128], dtype=a_desc.dtype,
                           layout=None, scope=tle.gpu.smem,
                           nv_mma_shared_layout=True)
    b_smem = tle.gpu.alloc([NUM_SLOTS, 128, 256], dtype=b_desc.dtype,
                           layout=None, scope=tle.gpu.smem,
                           nv_mma_shared_layout=True)
    pipe = tle.pipe(capacity=NUM_SLOTS, scope="cta", name="probe_persistent_k128",
                    a=a_smem, b=b_smem)
    tle.gpu.warp_specialize(
        [(_consumer, (pipe.reader(), c_ptr, pid, M, N, stride_cm, stride_cn,
                      TOTAL_TILES, GRID_N, NUM_SMS, TILE_ITERS, GROUPS)),
         (_producer, (pipe.writer(), a_desc, b_desc, pid, TOTAL_TILES,
                      GRID_N, NUM_SMS, TILE_ITERS, GROUPS))],
        worker_num_warps=[4], worker_num_regs=[24],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", nargs=3, type=int, required=True)
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument("--num-sms", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--rep", type=int, default=30)
    args = ap.parse_args()
    m, n, k = args.shape
    if k % 128:
        ap.error("K must be divisible by 128")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    a = torch.randn((m, k), device="musa", dtype=dtype)
    b = torch.randn((k, n), device="musa", dtype=dtype)
    out = torch.empty((m, n), device="musa", dtype=dtype)
    ref = torch.mm(a, b)
    grid_n = triton.cdiv(n, 256)
    total_tiles = triton.cdiv(m, 256) * grid_n
    tile_iters = triton.cdiv(total_tiles, args.num_sms)
    groups = k // 128
    da = TensorDescriptor.from_tensor(a, [256, 128])
    db = TensorDescriptor.from_tensor(b, [128, 256])

    def run():
        with torch_device_fn.device(a.device):
            persistent_k128_kernel[(args.num_sms,)](
                da, db, out, m, n, out.stride(0), out.stride(1), total_tiles,
                grid_n, args.num_sms, tile_iters, groups, 1,
                num_warps=16, enable_backend_opt=True)

    run()
    torch.musa.synchronize()
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)

    def bench(fn):
        for _ in range(args.warmup):
            fn()
        torch.musa.synchronize()
        vals = []
        for _ in range(5):
            st = torch.musa.Event(enable_timing=True)
            en = torch.musa.Event(enable_timing=True)
            st.record()
            for _ in range(args.rep):
                fn()
            en.record()
            torch.musa.synchronize()
            vals.append(st.elapsed_time(en) / args.rep)
        return statistics.median(vals), vals

    torch_ms, _ = bench(lambda: torch.mm(a, b, out=out))
    probe_ms, vals = bench(run)
    print(f"RESULT shape={m}x{n}x{k} dtype={args.dtype} num_sms={args.num_sms} "
          f"slots=1 torch_ms={torch_ms:.6f} probe_ms={probe_ms:.6f} "
          f"ratio={torch_ms/probe_ms:.4f} samples={vals}", flush=True)


if __name__ == "__main__":
    main()
