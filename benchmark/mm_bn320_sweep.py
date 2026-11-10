"""Focused BN320 parameter sweep for MTT boundary shapes."""
import argparse
import importlib
import statistics

import torch
import torch_musa  # noqa: F401

mm = importlib.import_module("flag_gems.runtime.backend._mthreads.ops.mm")


def bench(fn, warmup, rep):
    for _ in range(warmup):
        fn()
    torch.musa.synchronize()
    vals = []
    for _ in range(3):
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
    ap.add_argument("--m", type=int, default=448)
    ap.add_argument("--n", type=int, default=9216)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument("--bottom", type=int, default=128)
    ap.add_argument("--tail", type=int, default=64)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--stages", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--rep", type=int, default=80)
    args = ap.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    a = torch.randn((args.m, args.k), device="musa", dtype=dtype)
    b = torch.randn((args.k, args.n), device="musa", dtype=dtype)
    out = torch.empty((args.m, args.n), device="musa", dtype=dtype)
    ref = torch.mm(a, b)
    fn = lambda: mm.mm_tle_bn320_ordered(
        a, b, out, args.m, args.n, args.k,
        block_m_bottom=args.bottom, block_n_tail=args.tail,
        num_slots=args.slots, num_stages=args.stages,
    )
    try:
        fn()
        torch.musa.synchronize()
        torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
        base, _ = bench(lambda: torch.mm(a, b, out=out), args.warmup, args.rep)
        got, vals = bench(fn, args.warmup, args.rep)
        print(f"RESULT m={args.m} n={args.n} k={args.k} dtype={args.dtype} "
              f"bottom={args.bottom} tail={args.tail} slots={args.slots} stages={args.stages} "
              f"torch_ms={base:.6f} bn320_ms={got:.6f} ratio={base/got:.4f} samples={vals}", flush=True)
    except Exception as exc:
        print(f"ERROR m={args.m} n={args.n} k={args.k} dtype={args.dtype} "
              f"bottom={args.bottom} tail={args.tail} slots={args.slots} stages={args.stages} "
              f"{type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
