"""Persistent TLE GEMM probe for small-M, wide-N workloads on MTT.

Unlike the regular split-N kernel, this launch uses one CTA per MTT SM and
lets each CTA consume multiple output tiles.  It is intentionally a benchmark
probe; production dispatch should only be changed after independent timing.
"""

import argparse
import importlib

import torch
import torch_musa  # noqa: F401
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems.runtime import torch_device_fn
from triton.experimental.tle.language import gpu as tle_gpu


tle = importlib.import_module("triton.experimental.tle.language")


@triton.jit
def _producer(writer, a_desc, b_desc, pid, total_tiles, grid_n,
              num_sms, tile_iters, k_tiles, BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    for tile_iter in range(tile_iters):
        tile_id = pid + tile_iter * num_sms
        if tile_id < total_tiles:
            pid_m = tile_id // grid_n
            pid_n = tile_id % grid_n
            m_offset = (pid_m * BLOCK_M).to(tl.int32)
            n_offset = (pid_n * BLOCK_N).to(tl.int32)
            for k_iter in range(k_tiles):
                token = tile_iter * k_tiles + k_iter
                slot = writer.acquire(token)
                k_offset = k_iter * BLOCK_K
                tle_gpu.copy(a_desc, slot.a, [BLOCK_M, BLOCK_K],
                             [m_offset, k_offset])
                tle_gpu.copy(b_desc, slot.b, [BLOCK_K, BLOCK_N],
                             [k_offset, n_offset])
                writer.commit(token)


@triton.jit
def _consumer(reader, c_ptr, pid, M, N, stride_cm, stride_cn,
              total_tiles, grid_n, num_sms, tile_iters, k_tiles,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
              BLOCK_K: tl.constexpr):
    for tile_iter in range(tile_iters):
        tile_id = pid + tile_iter * num_sms
        if tile_id < total_tiles:
            pid_m = tile_id // grid_n
            pid_n = tile_id % grid_n
            m_offset = (pid_m * BLOCK_M).to(tl.int32)
            n_offset = (pid_n * BLOCK_N).to(tl.int32)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_iter in range(k_tiles):
                token = tile_iter * k_tiles + k_iter
                ready = reader.wait(token)
                acc = tle_gpu.wgmma(ready.slot.a, ready.slot.b, acc)
                acc = tle_gpu.wgmma_wait(0, acc)
                reader.release(token)
            offs_m = m_offset + tl.arange(0, BLOCK_M)
            offs_n = n_offset + tl.arange(0, BLOCK_N)
            ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
            tl.store(ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)


@triton.jit
def persistent_smallm_kernel(a_desc, b_desc, c_ptr, M, N, stride_cm,
                             stride_cn, TOTAL_TILES: tl.constexpr,
                             GRID_N: tl.constexpr, NUM_SMS: tl.constexpr,
                             TILE_ITERS: tl.constexpr, K_TILES: tl.constexpr,
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                             BLOCK_K: tl.constexpr, NUM_SLOTS: tl.constexpr):
    pid = tl.program_id(0)
    a_smem = tle_gpu.alloc([NUM_SLOTS, BLOCK_M, BLOCK_K], dtype=a_desc.dtype,
                           layout=None, scope=tle_gpu.smem,
                           nv_mma_shared_layout=True)
    b_smem = tle_gpu.alloc([NUM_SLOTS, BLOCK_K, BLOCK_N], dtype=b_desc.dtype,
                           layout=None, scope=tle_gpu.smem,
                           nv_mma_shared_layout=True)
    pipe = tle.pipe(capacity=NUM_SLOTS, scope="cta", name="smallm_persistent",
                    a=a_smem, b=b_smem)
    tle_gpu.warp_specialize(
        [
            (_consumer, (pipe.reader(), c_ptr, pid, M, N, stride_cm, stride_cn,
                         TOTAL_TILES, GRID_N, NUM_SMS, TILE_ITERS, K_TILES,
                         BLOCK_M, BLOCK_N, BLOCK_K)),
            (_producer, (pipe.writer(), a_desc, b_desc, pid, TOTAL_TILES,
                         GRID_N, NUM_SMS, TILE_ITERS, K_TILES, BLOCK_M,
                         BLOCK_N, BLOCK_K)),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--n", type=int, default=12288)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument("--bm", type=int, default=64)
    ap.add_argument("--bn", type=int, default=128)
    ap.add_argument("--bk", type=int, default=64)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--sms", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--rep", type=int, default=80)
    args = ap.parse_args()
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    a = torch.randn((args.m, args.k), device="musa", dtype=dt)
    b = torch.randn((args.k, args.n), device="musa", dtype=dt)
    out = torch.empty((args.m, args.n), device="musa", dtype=dt)
    ref = torch.mm(a, b)
    grid_n = triton.cdiv(args.n, args.bn)
    total_tiles = triton.cdiv(args.m, args.bm) * grid_n
    tile_iters = triton.cdiv(total_tiles, args.sms)
    da = TensorDescriptor.from_tensor(a, [args.bm, args.bk])
    db = TensorDescriptor.from_tensor(b, [args.bk, args.bn])

    def run():
        with torch_device_fn.device(a.device):
            persistent_smallm_kernel[(args.sms,)](
                da, db, out, args.m, args.n, out.stride(0), out.stride(1),
                total_tiles, grid_n, args.sms, tile_iters,
                triton.cdiv(args.k, args.bk), args.bm, args.bn, args.bk,
                args.slots, num_warps=16, enable_backend_opt=True)

    run()
    torch.musa.synchronize()
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
    base = triton.testing.do_bench(lambda: torch.mm(a, b, out=out),
                                   warmup=args.warmup, rep=args.rep)
    tested = triton.testing.do_bench(run, warmup=args.warmup, rep=args.rep)
    print(f"RESULT shape={args.m}x{args.n}x{args.k} dtype={args.dtype} "
          f"bm={args.bm} bn={args.bn} bk={args.bk} slots={args.slots} "
          f"sms={args.sms} tiles={total_tiles} tile_iters={tile_iters} "
          f"torch_ms={base:.6f} fixed_ms={tested:.6f} ratio={base/tested:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
