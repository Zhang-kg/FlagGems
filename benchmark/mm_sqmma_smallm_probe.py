"""Fixed-config probe for the non-pipe SQMMA GEMM on small-M shapes."""

import argparse
import importlib

import torch
import torch_musa  # noqa: F401
import triton
from triton.tools.tensor_descriptor import TensorDescriptor


mm = importlib.import_module("flag_gems.runtime.backend._mthreads.ops.mm")


def run(a, b, c, bm, bn, bk, group_m, warps):
    m, k = a.shape
    n = b.shape[1]
    da = TensorDescriptor.from_tensor(a, [bm, bk])
    db = TensorDescriptor.from_tensor(b, [bk, bn])
    dc = TensorDescriptor.from_tensor(c, [bm, bn])
    cfg = triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": group_m},
        num_stages=1,
        num_warps=warps,
        pre_hook=mm.sqmma_descriptor_pre_hook,
    )
    tuner = mm.mm_sqmma_kernel.fn
    old_configs = tuner.configs
    tuner.configs = [cfg]
    try:
        grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn), 1, 1)
        tuner.run(
            da,
            db,
            dc,
            m,
            n,
            k,
            str(a.dtype).split(".")[-1],
            grid=grid,
            warmup=False,
        )
    finally:
        tuner.configs = old_configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", nargs=3, type=int, action="append", required=True)
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument("--bm", type=int, default=128)
    ap.add_argument("--bn", type=int, default=256)
    ap.add_argument("--bk", type=int, default=64)
    ap.add_argument("--group-m", type=int, default=1)
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--rep", type=int, default=30)
    args = ap.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    for m, n, k in args.shape:
        a = torch.randn((m, k), device="musa", dtype=dtype)
        b = torch.randn((k, n), device="musa", dtype=dtype)
        ref = torch.mm(a, b)
        out_sqmma = torch.empty_like(ref)
        out_dispatch = torch.empty_like(ref)

        run(a, b, out_sqmma, args.bm, args.bn, args.bk, args.group_m, args.warps)
        mm.mm_out(a, b, out=out_dispatch)
        torch.musa.synchronize()
        torch.testing.assert_close(out_sqmma, ref, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(out_dispatch, ref, atol=3e-2, rtol=3e-2)

        torch_fn = lambda: torch.mm(a, b, out=out_dispatch)
        sqmma_fn = lambda: run(
            a, b, out_sqmma, args.bm, args.bn, args.bk, args.group_m, args.warps
        )
        dispatch_fn = lambda: mm.mm_out(a, b, out=out_dispatch)
        torch_ms = triton.testing.do_bench(torch_fn, warmup=args.warmup, rep=args.rep)
        sqmma_ms = triton.testing.do_bench(sqmma_fn, warmup=args.warmup, rep=args.rep)
        dispatch_ms = triton.testing.do_bench(
            dispatch_fn, warmup=args.warmup, rep=args.rep
        )
        print(
            f"RESULT shape={m}x{n}x{k} dtype={args.dtype} "
            f"bm={args.bm} bn={args.bn} bk={args.bk} group_m={args.group_m} "
            f"warps={args.warps} torch_ms={torch_ms:.6f} "
            f"sqmma_ms={sqmma_ms:.6f} sqmma_ratio={torch_ms/sqmma_ms:.4f} "
            f"dispatch_ms={dispatch_ms:.6f} dispatch_ratio={torch_ms/dispatch_ms:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
