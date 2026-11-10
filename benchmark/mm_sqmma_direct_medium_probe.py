"""Direct-output SQMMA probe for medium-M GEMM shapes.

Unlike ``mm_sqmma``, this times only the descriptor SQMMA kernel and reuses
the output tensor, so allocation/copy overhead does not obscure tile quality.
"""
import argparse
import importlib

import torch
import torch_musa  # noqa: F401
import triton
from triton.tools.tensor_descriptor import TensorDescriptor

mm = importlib.import_module("flag_gems.runtime.backend._mthreads.ops.mm")


def run(a, b, c, bm, bn, bk, group_m, warps):
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
        tuner.run(
            da,
            db,
            dc,
            a.shape[0],
            b.shape[1],
            a.shape[1],
            str(a.dtype).split(".")[-1],
            grid=(triton.cdiv(a.shape[0], bm) * triton.cdiv(b.shape[1], bn), 1, 1),
            warmup=False,
        )
    finally:
        tuner.configs = old_configs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument("--bm", type=int, default=128)
    ap.add_argument("--bn", type=int, default=128)
    ap.add_argument("--bk", type=int, default=64)
    ap.add_argument("--group-m", type=int, default=1)
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--rep", type=int, default=30)
    args = ap.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    a = torch.randn((args.m, args.k), device="musa", dtype=dtype)
    b = torch.randn((args.k, args.n), device="musa", dtype=dtype)
    out = torch.empty((args.m, args.n), device="musa", dtype=dtype)
    ref = torch.mm(a, b)
    run(a, b, out, args.bm, args.bn, args.bk, args.group_m, args.warps)
    torch.musa.synchronize()
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
    torch_fn = lambda: torch.mm(a, b, out=out)
    sqmma_fn = lambda: run(a, b, out, args.bm, args.bn, args.bk, args.group_m, args.warps)
    torch_ms = float(triton.testing.do_bench(torch_fn, warmup=args.warmup, rep=args.rep))
    sqmma_ms = float(triton.testing.do_bench(sqmma_fn, warmup=args.warmup, rep=args.rep))
    print(
        f"RESULT shape={args.m}x{args.n}x{args.k} dtype={args.dtype} "
        f"bm={args.bm} bn={args.bn} bk={args.bk} group_m={args.group_m} "
        f"warps={args.warps} torch_ms={torch_ms:.6f} direct_ms={sqmma_ms:.6f} "
        f"ratio={torch_ms/sqmma_ms:.4f}", flush=True
    )


if __name__ == "__main__":
    main()
