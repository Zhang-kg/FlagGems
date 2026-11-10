"""Probe muDNN-like 128x32/128x64 TLE tiles for small-M GEMM.

This is deliberately a benchmark-only kernel.  It maps both M and N tiles so
tail rows are handled correctly, while keeping the accumulator narrow enough
to measure the occupancy benefit of muDNN's small-N schedule.
"""

import argparse
import statistics

import torch
import torch_musa  # noqa: F401
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems.runtime import torch_device_fn
from triton.experimental.tle.language import gpu as tle_gpu
import triton.experimental.tle.language as tle


@triton.jit
def _producer(writer, a_desc, b_desc, m_offset, n_offset,
              k_tiles: tl.constexpr, block_m: tl.constexpr,
              block_n: tl.constexpr, block_k: tl.constexpr):
    for k_iter in range(k_tiles):
        slot = writer.acquire(k_iter)
        k_offset = k_iter * block_k
        tle_gpu.copy(a_desc, slot.a, [block_m, block_k],
                     [m_offset, k_offset])
        tle_gpu.copy(b_desc, slot.b, [block_k, block_n],
                     [k_offset, n_offset])
        writer.commit(k_iter)


@triton.jit
def _consumer(reader, c_ptr, m_offset, n_offset, M, N, stride_cm,
              stride_cn, k_tiles: tl.constexpr, block_m: tl.constexpr,
              block_n: tl.constexpr, mma_group: tl.constexpr):
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for group_start in tl.static_range(0, k_tiles, mma_group):
        for group_offset in tl.static_range(mma_group):
            k_iter = group_start + group_offset
            if k_iter < k_tiles:
                ready = reader.wait(k_iter)
                acc = tle_gpu.wgmma(ready.slot.a, ready.slot.b, acc)
        acc = tle_gpu.wgmma_wait(0, acc)
        for group_offset in tl.static_range(mma_group):
            k_iter = group_start + group_offset
            if k_iter < k_tiles:
                reader.release(k_iter)
    rows = m_offset + tl.arange(0, block_m)
    cols = n_offset + tl.arange(0, block_n)
    ptrs = c_ptr + rows[:, None] * stride_cm + cols[None, :] * stride_cn
    mask = (rows < M)[:, None] & (cols < N)[None, :]
    tl.store(ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)


@triton.jit
def smallm_narrow_kernel(a_desc, b_desc, c_ptr, M, N, stride_cm, stride_cn,
                         GRID_N: tl.constexpr, K_TILES: tl.constexpr,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                         BLOCK_K: tl.constexpr, NUM_SLOTS: tl.constexpr,
                         MMA_GROUP: tl.constexpr):
    pid = tl.program_id(0)
    pid_m = pid // GRID_N
    pid_n = pid % GRID_N
    m_offset = (pid_m * BLOCK_M).to(tl.int32)
    n_offset = (pid_n * BLOCK_N).to(tl.int32)
    a_smem = tle_gpu.alloc([NUM_SLOTS, BLOCK_M, BLOCK_K], dtype=a_desc.dtype,
                           layout=None, scope=tle_gpu.smem,
                           nv_mma_shared_layout=True)
    b_smem = tle_gpu.alloc([NUM_SLOTS, BLOCK_K, BLOCK_N], dtype=b_desc.dtype,
                           layout=None, scope=tle_gpu.smem,
                           nv_mma_shared_layout=True)
    pipe = tle.pipe(capacity=NUM_SLOTS, scope="cta", name="smallm_narrow",
                    a=a_smem, b=b_smem)
    tle_gpu.warp_specialize(
        [(_consumer, (pipe.reader(), c_ptr, m_offset, n_offset, M, N,
                      stride_cm, stride_cn, K_TILES, BLOCK_M, BLOCK_N,
                      MMA_GROUP)),
         (_producer, (pipe.writer(), a_desc, b_desc, m_offset, n_offset,
                      K_TILES, BLOCK_M, BLOCK_N, BLOCK_K))],
        [4], [24],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--n", type=int, default=9216)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument("--bm", type=int, default=128)
    ap.add_argument("--bn", type=int, default=32)
    ap.add_argument("--bk", type=int, default=64)
    ap.add_argument("--slots", type=int, default=3)
    ap.add_argument("--mma-group", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--rep", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    a = torch.randn((args.m, args.k), device="musa", dtype=dtype)
    b = torch.randn((args.k, args.n), device="musa", dtype=dtype)
    out = torch.empty((args.m, args.n), device="musa", dtype=dtype)
    ref = torch.mm(a, b)
    grid_n = triton.cdiv(args.n, args.bn)
    grid_m = triton.cdiv(args.m, args.bm)
    da = TensorDescriptor.from_tensor(a, [args.bm, args.bk])
    db = TensorDescriptor.from_tensor(b, [args.bk, args.bn])

    def run():
        with torch_device_fn.device(a.device):
            smallm_narrow_kernel[(grid_m * grid_n,)](
                da, db, out, args.m, args.n, out.stride(0), out.stride(1),
                grid_n, triton.cdiv(args.k, args.bk), args.bm, args.bn,
                args.bk, args.slots, args.mma_group,
                num_warps=4, enable_backend_opt=True)

    run()
    torch.musa.synchronize()
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)

    def bench(fn):
        values = []
        for _ in range(args.repeats):
            start = torch.musa.Event(enable_timing=True)
            end = torch.musa.Event(enable_timing=True)
            start.record()
            for _ in range(args.rep):
                fn()
            end.record()
            torch.musa.synchronize()
            values.append(start.elapsed_time(end) / args.rep)
        return statistics.median(values), values

    base, base_values = bench(lambda: torch.mm(a, b, out=out))
    tested, tested_values = bench(run)
    print(f"RESULT shape={args.m}x{args.n}x{args.k} dtype={args.dtype} "
          f"bm={args.bm} bn={args.bn} bk={args.bk} slots={args.slots} "
          f"mma_group={args.mma_group} torch_ms={base:.6f} "
          f"tested_ms={tested:.6f} ratio={base/tested:.4f} "
          f"torch_samples={base_values} samples={tested_values}", flush=True)


if __name__ == "__main__":
    main()
