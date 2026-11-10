"""Probe production multifield split-N single-pipe kernel with smaller BM."""
import argparse
import importlib

import torch
import torch_musa  # noqa: F401
import triton
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems.runtime import torch_device_fn

mm = importlib.import_module("flag_gems.runtime.backend._mthreads.ops.mm")


def run(a, b, out, bm, bn, bk, slots):
    m, k = a.shape
    n = b.shape[1]
    da = TensorDescriptor.from_tensor(a, [bm, bk])
    db = TensorDescriptor.from_tensor(b, [bk, bn])
    grid = (triton.cdiv(n, bn),)
    with torch_device_fn.device(a.device):
        mm.mm_tle_split_n_single_pipe_kernel[grid](
            da, db, out, m, n, out.stride(0), out.stride(1),
            triton.cdiv(k, bk), bm, bk, slots, 1,
            num_warps=4, enable_backend_opt=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--n", type=int, default=12288)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument("--bm", type=int, default=64)
    ap.add_argument("--bn", type=int, default=256)
    ap.add_argument("--bk", type=int, default=64)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--rep", type=int, default=200)
    args = ap.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    a = torch.randn((args.m, args.k), device="musa", dtype=dtype)
    b = torch.randn((args.k, args.n), device="musa", dtype=dtype)
    out = torch.empty((args.m, args.n), device="musa", dtype=dtype)
    ref = torch.mm(a, b)
    run(a, b, out, args.bm, args.bn, args.bk, args.slots)
    torch.musa.synchronize()
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
    torch_fn = lambda: torch.mm(a, b, out=out)
    mm_fn = lambda: run(a, b, out, args.bm, args.bn, args.bk, args.slots)
    t = triton.testing.do_bench(torch_fn, warmup=args.warmup, rep=args.rep)
    x = triton.testing.do_bench(mm_fn, warmup=args.warmup, rep=args.rep)
    print(f"RESULT shape={args.m}x{args.n}x{args.k} dtype={args.dtype} "
          f"bm={args.bm} bn={args.bn} bk={args.bk} slots={args.slots} "
          f"torch_ms={t:.6f} mm_ms={x:.6f} ratio={t/x:.4f}", flush=True)


if __name__ == "__main__":
    main()
