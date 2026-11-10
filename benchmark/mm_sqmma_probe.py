import argparse
import importlib

import torch
import torch_musa  # noqa: F401
import triton


mm = importlib.import_module("flag_gems.runtime.backend._mthreads.ops.mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--rep", type=int, default=100)
    args = ap.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    a = torch.randn((args.m, args.k), device="musa", dtype=dtype)
    b = torch.randn((args.k, args.n), device="musa", dtype=dtype)
    out = torch.empty((args.m, args.n), device="musa", dtype=dtype)
    ref = torch.mm(a, b)

    def torch_fn():
        torch.mm(a, b, out=out)

    def sqmma_fn():
        result = mm.mm_sqmma(a, b, args.m, args.n, args.k)
        out.copy_(result)

    sqmma_fn()
    torch.musa.synchronize()
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
    torch_ms = float(triton.testing.do_bench(torch_fn, warmup=args.warmup, rep=args.rep))
    sqmma_ms = float(triton.testing.do_bench(sqmma_fn, warmup=args.warmup, rep=args.rep))
    print(
        f"RESULT shape={args.m}x{args.n}x{args.k} dtype={args.dtype} "
        f"torch_ms={torch_ms:.6f} sqmma_ms={sqmma_ms:.6f} "
        f"ratio={torch_ms/sqmma_ms:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
