"""Probe the mm.py full-B single-pipe experiment without changing dispatch."""
import argparse
import importlib
import statistics

import torch
import torch_musa  # noqa: F401
import triton

mm = importlib.import_module("flag_gems.runtime.backend._mthreads.ops.mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--dtype", choices=("bf16", "fp16"), required=True)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--rep", type=int, default=60)
    args = ap.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    a = torch.randn((args.m, args.k), device="musa", dtype=dtype)
    b = torch.randn((args.k, args.n), device="musa", dtype=dtype)
    out = torch.empty((args.m, args.n), device="musa", dtype=dtype)
    ref = torch.mm(a, b)

    def fullb():
        mm.mm_tle_split_n_full_pipe(a, b, out, args.m, args.n, args.k)

    fullb()
    torch.musa.synchronize()
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
    def bench(fn):
        for _ in range(args.warmup):
            fn()
        torch.musa.synchronize()
        vals = []
        for _ in range(5):
            start = torch.musa.Event(enable_timing=True)
            end = torch.musa.Event(enable_timing=True)
            start.record()
            for _ in range(args.rep):
                fn()
            end.record()
            torch.musa.synchronize()
            vals.append(start.elapsed_time(end) / args.rep)
        return statistics.median(vals)

    torch_ms = bench(lambda: torch.mm(a, b, out=out))
    full_ms = bench(fullb)
    print(
        f"RESULT shape={args.m}x{args.n}x{args.k} dtype={args.dtype} "
        f"torch_ms={torch_ms:.6f} fullb_ms={full_ms:.6f} "
        f"ratio={torch_ms/full_ms:.4f}", flush=True
    )


if __name__ == "__main__":
    main()
