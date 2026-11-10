"""Single-path split384 benchmark; isolates it from poisoned variants."""

import argparse
import importlib

import torch
import torch_musa  # noqa: F401
import triton

mm = importlib.import_module("flag_gems.runtime.backend._mthreads.ops.mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", nargs=3, type=int, required=True)
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument("--wide-warps", action="store_true",
                    help="use experimental 24-warp split384 schedule")
    ap.add_argument("--block-k", type=int, choices=(32, 64, 128), default=32)
    ap.add_argument("--num-slots", type=int, choices=(1, 2, 3), default=3)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--rep", type=int, default=30)
    args = ap.parse_args()
    m, n, k = args.shape
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    a = torch.randn((m, k), device="musa", dtype=dt)
    b = torch.randn((k, n), device="musa", dtype=dt)
    out = torch.empty((m, n), device="musa", dtype=dt)
    ref = torch.mm(a, b)

    def torch_fn():
        torch.mm(a, b, out=out)

    def split_fn():
        mm.mm_tle_split384_pipe(a, b, out, m, n, k,
                                block_k=args.block_k,
                                num_slots=args.num_slots,
                                wide_warps=args.wide_warps)

    split_fn()
    torch.musa.synchronize()
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
    base = float(triton.testing.do_bench(torch_fn, warmup=args.warmup, rep=args.rep))
    tested = float(triton.testing.do_bench(split_fn, warmup=args.warmup, rep=args.rep))
    print(f"RESULT shape={m}x{n}x{k} dtype={args.dtype} "
          f"wide_warps={int(args.wide_warps)} block_k={args.block_k} "
          f"num_slots={args.num_slots} torch_ms={base:.6f} "
          f"split384_ms={tested:.6f} ratio={base/tested:.4f}", flush=True)


if __name__ == "__main__":
    main()
