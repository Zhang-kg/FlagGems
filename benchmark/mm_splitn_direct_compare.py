"""Direct high-repetition comparison of tiled and legacy split-N wrappers."""
import argparse
import importlib
import statistics

import torch
import torch_musa  # noqa: F401

mm = importlib.import_module("flag_gems.runtime.backend._mthreads.ops.mm")


def bench(fn, warmup=20, rep=80):
    for _ in range(warmup):
        fn()
    torch.musa.synchronize()
    vals = []
    for _ in range(7):
        s = torch.musa.Event(enable_timing=True)
        e = torch.musa.Event(enable_timing=True)
        s.record()
        for _ in range(rep):
            fn()
        e.record()
        torch.musa.synchronize()
        vals.append(s.elapsed_time(e) / rep)
    return statistics.median(vals), vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=("bf16", "fp16"), required=True)
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--n", type=int, default=12288)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--rep", type=int, default=80)
    args = ap.parse_args()
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    a = torch.randn((args.m, args.k), device="musa", dtype=dt)
    b = torch.randn((args.k, args.n), device="musa", dtype=dt)
    out = torch.empty((args.m, args.n), device="musa", dtype=dt)
    ref = torch.mm(a, b)
    print("flags", mm.is_tle_split_n_compatible(args.m, args.n, args.k),
          mm.is_tle_split_n_single_compatible(args.m, args.n, args.k, dt), flush=True)
    fns = {
        "torch": lambda: torch.mm(a, b, out=out),
        "splitn": lambda: mm.mm_tle_split_n_pipe(a, b, out, args.m, args.n, args.k),
        "legacy": lambda: mm.mm_tle_split_n_single_pipe(a, b, out, args.m, args.n, args.k),
        "wide": lambda: mm.mm_tle_split_n_wide_single_pipe(a, b, out, args.m, args.n, args.k),
    }
    results = {}
    for name, fn in fns.items():
        fn(); torch.musa.synchronize()
        torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
        results[name] = bench(fn, args.warmup, args.rep)
    base = results["torch"][0]
    print(f"RESULT shape={args.m}x{args.n}x{args.k} dtype={args.dtype} "
          + " ".join(f"{name}_ms={v[0]:.6f} {name}_ratio={base/v[0]:.4f}" for name, v in results.items())
          + " samples=" + repr({name: v[1] for name, v in results.items()}), flush=True)


if __name__ == "__main__":
    main()
