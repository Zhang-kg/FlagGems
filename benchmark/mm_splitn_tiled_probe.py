"""Correct 2-D-grid one-pipe probe for narrow M split-N tiles.

The older small-M probes launched only along N, which silently omitted rows
when BLOCK_M < M.  This probe maps both M and N tile IDs and is suitable for
testing BM=16/32/64 without masking out an entire row tile.
"""

import argparse
import importlib
import statistics

import torch
import torch_musa  # noqa: F401
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems.runtime import torch_device_fn
from triton.experimental.tle.language import gpu as tle_gpu


tle = importlib.import_module("triton.experimental.tle.language")


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
              block_n: tl.constexpr):
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)
    for k_iter in range(k_tiles):
        ready = reader.wait(k_iter)
        acc = tle_gpu.wgmma(ready.slot.a, ready.slot.b, acc)
        acc = tle_gpu.wgmma_wait(0, acc)
        reader.release(k_iter)
    rm = m_offset + tl.arange(0, block_m)
    rn = n_offset + tl.arange(0, block_n)
    ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)


@triton.jit
def tiled_one_pipe_kernel(a_desc, b_desc, c_ptr, M, N, stride_cm,
                          stride_cn, GRID_N: tl.constexpr,
                          K_TILES: tl.constexpr, BLOCK_M: tl.constexpr,
                          BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                          NUM_SLOTS: tl.constexpr, WORKER_WARPS: tl.constexpr):
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
    pipe = tle.pipe(capacity=NUM_SLOTS, scope="cta", name="tiled_one_pipe",
                    a=a_smem, b=b_smem)
    tle_gpu.warp_specialize(
        [
            (_consumer, (pipe.reader(), c_ptr, m_offset, n_offset, M, N,
                         stride_cm, stride_cn, K_TILES, BLOCK_M, BLOCK_N)),
            (_producer, (pipe.writer(), a_desc, b_desc, m_offset, n_offset,
                         K_TILES, BLOCK_M, BLOCK_N, BLOCK_K)),
        ],
        [WORKER_WARPS], [24],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--n", type=int, default=12288)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument("--bm", type=int, default=32)
    ap.add_argument("--bn", type=int, default=128)
    ap.add_argument("--bk", type=int, default=64)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--worker-warps", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--rep", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    a = torch.randn((args.m, args.k), device="musa", dtype=dt)
    b = torch.randn((args.k, args.n), device="musa", dtype=dt)
    out = torch.empty((args.m, args.n), device="musa", dtype=dt)
    ref = torch.mm(a, b)
    grid_n = triton.cdiv(args.n, args.bn)
    grid_m = triton.cdiv(args.m, args.bm)
    total_tiles = grid_m * grid_n
    da = TensorDescriptor.from_tensor(a, [args.bm, args.bk])
    db = TensorDescriptor.from_tensor(b, [args.bk, args.bn])

    def run():
        with torch_device_fn.device(a.device):
            tiled_one_pipe_kernel[(total_tiles,)](
                da, db, out, args.m, args.n, out.stride(0), out.stride(1),
                grid_n, triton.cdiv(args.k, args.bk), args.bm, args.bn,
                args.bk, args.slots, args.worker_warps,
                num_warps=args.worker_warps, enable_backend_opt=True)

    run()
    torch.musa.synchronize()
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
    for _ in range(args.warmup):
        run()
    torch.musa.synchronize()

    def bench(fn):
        vals = []
        for _ in range(args.repeats):
            st, en = torch.musa.Event(enable_timing=True), torch.musa.Event(enable_timing=True)
            st.record()
            for _ in range(args.rep):
                fn()
            en.record()
            torch.musa.synchronize()
            vals.append(st.elapsed_time(en) / args.rep)
        return statistics.median(vals), vals

    base, base_vals = bench(lambda: torch.mm(a, b, out=out))
    tested, vals = bench(run)
    print(f"RESULT shape={args.m}x{args.n}x{args.k} dtype={args.dtype} "
          f"bm={args.bm} bn={args.bn} bk={args.bk} slots={args.slots} "
          f"warps={args.worker_warps} tiles={total_tiles} "
          f"torch_ms={base:.6f} tested_ms={tested:.6f} ratio={base/tested:.4f} "
          f"torch_samples={base_vals} samples={vals}", flush=True)


if __name__ == "__main__":
    main()
