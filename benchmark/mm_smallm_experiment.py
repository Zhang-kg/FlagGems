"""Experimental single-consumer TLE GEMM for small-M split-N shapes.

This file intentionally does not modify mm dispatch.  It is copied to the
MTT host and used to compare a single 256-column consumer against the current
two 128-column ``split_n`` kernel.
"""

import argparse
import importlib
import time

import torch
import torch_musa  # noqa: F401
import triton
import triton.language as tl

from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems.runtime import torch_device_fn
from triton.experimental.tle.language import gpu as tle_gpu


tle = importlib.import_module("triton.experimental.tle.language")
mm_mod = importlib.import_module("flag_gems.runtime.backend._mthreads.ops.mm")


@triton.jit
def _producer(a_writer, b_writer, a_desc, b_desc, n_offset,
              K_TILES: tl.constexpr, BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    for k_iter in range(K_TILES):
        k_offset = k_iter * BLOCK_K
        a_slot = a_writer.acquire(k_iter)
        b_slot = b_writer.acquire(k_iter)
        tle_gpu.copy(a_desc, a_slot.a, [BLOCK_M, BLOCK_K], [0, k_offset])
        tle_gpu.copy(b_desc, b_slot.b, [BLOCK_K, BLOCK_N], [k_offset, n_offset])
        a_writer.commit(k_iter)
        b_writer.commit(k_iter)


@triton.jit
def _consumer(a_reader, b_reader, c_ptr, n_offset, stride_cm, stride_cn,
              M, N, K_TILES: tl.constexpr, BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr):
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_iter in range(K_TILES):
        a_wait = a_reader.wait(k_iter)
        b_wait = b_reader.wait(k_iter)
        acc = tle_gpu.wgmma(a_wait.slot.a, b_wait.slot.b, acc)
        acc = tle_gpu.wgmma_wait(0, acc)
        a_reader.release(k_iter)
        b_reader.release(k_iter)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = n_offset + tl.arange(0, BLOCK_N)
    ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    tl.store(ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)


@triton.jit
def single_consumer_kernel(a_desc, b_desc, c_ptr, M, N, stride_cm, stride_cn,
                           K_TILES: tl.constexpr, BLOCK_M: tl.constexpr,
                           BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                           NUM_SLOTS: tl.constexpr):
    n_offset = (tl.program_id(0) * BLOCK_N).to(tl.int32)
    a_smem = tle_gpu.alloc([NUM_SLOTS, BLOCK_M, BLOCK_K], dtype=a_desc.dtype,
                           layout=None, scope=tle_gpu.smem,
                           nv_mma_shared_layout=True)
    b_smem = tle_gpu.alloc([NUM_SLOTS, BLOCK_K, BLOCK_N], dtype=b_desc.dtype,
                           layout=None, scope=tle_gpu.smem,
                           nv_mma_shared_layout=True)
    a_pipe = tle.pipe(capacity=NUM_SLOTS, scope="cta", name="smallm_a", a=a_smem)
    b_pipe = tle.pipe(capacity=NUM_SLOTS, scope="cta", name="smallm_b", b=b_smem)
    tle_gpu.warp_specialize(
        [
            (_consumer, (a_pipe.reader(), b_pipe.reader(), c_ptr, n_offset,
                         stride_cm, stride_cn, M, N, K_TILES, BLOCK_M, BLOCK_N)),
            (_producer, (a_pipe.writer(), b_pipe.writer(), a_desc, b_desc,
                         n_offset, K_TILES, BLOCK_M, BLOCK_N, BLOCK_K)),
        ],
        # The first entry is the default partition (consumer); only workers
        # after it receive explicit warp/register counts.
        [4],
        [24],
    )


def run(a, b, out, block_m, block_n, block_k, slots):
    m, k = a.shape
    n = b.shape[1]
    da = TensorDescriptor.from_tensor(a, [block_m, block_k])
    db = TensorDescriptor.from_tensor(b, [block_k, block_n])
    grid = (triton.cdiv(n, block_n),)
    with torch_device_fn.device(a.device):
        single_consumer_kernel[grid](
            da, db, out, m, n, out.stride(0), out.stride(1),
            triton.cdiv(k, block_k), block_m, block_n, block_k, slots,
            num_warps=8, enable_backend_opt=True,
        )


def bench(fn, warmup=5, rep=20):
    return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--n", type=int, default=12288)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument("--block-m", type=int, default=64)
    ap.add_argument("--block-n", type=int, default=256)
    ap.add_argument("--block-k", type=int, default=64)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--rep", type=int, default=20)
    args = ap.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    a = torch.randn((args.m, args.k), device="musa", dtype=dtype)
    b = torch.randn((args.k, args.n), device="musa", dtype=dtype)
    ref = torch.mm(a, b)
    out = torch.empty_like(ref)
    run(a, b, out, args.block_m, args.block_n, args.block_k, args.slots)
    torch.musa.synchronize()
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
    # Warm compilation/cache before timings.
    base = bench(lambda: torch.mm(a, b), rep=args.rep)
    exp = bench(lambda: run(a, b, out, args.block_m, args.block_n,
                            args.block_k, args.slots), rep=args.rep)
    cur = bench(lambda: mm_mod.mm_tle_split_n_pipe(a, b, out, args.m,
                                                    args.n, args.k), rep=args.rep)
    print(f"shape={args.m}x{args.n}x{args.k} dtype={args.dtype} "
          f"cfg=({args.block_m},{args.block_n},{args.block_k},s{args.slots}) "
          f"torch={base:.6f} single={exp:.6f} split_n={cur:.6f} "
          f"single_ratio={base / exp:.4f} split_ratio={base / cur:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
