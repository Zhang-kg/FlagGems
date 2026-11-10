"""High-repetition benchmark for the guarded split-N single-pipe path."""

import argparse
import importlib
import statistics

import torch
mm_mod = importlib.import_module("flag_gems.runtime.backend._mthreads.ops.mm")


def bench(fn, warmup, rep):
    for _ in range(warmup):
        fn()
    torch.musa.synchronize()
    samples = []
    for _ in range(5):
        start = torch.musa.Event(enable_timing=True)
        end = torch.musa.Event(enable_timing=True)
        start.record()
        for _ in range(rep):
            fn()
        end.record()
        torch.musa.synchronize()
        samples.append(start.elapsed_time(end) / rep)
    return statistics.median(samples), samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=("bf16", "fp16"), required=True)
    parser.add_argument("--m", type=int, default=64)
    parser.add_argument("--n", type=int, default=12288)
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument(
        "--variant", choices=("old", "single", "pipe", "dispatch"), default="single"
    )
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--rep", type=int, default=200)
    args = parser.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    M, N, K = args.m, args.n, args.k
    a = torch.randn((M, K), device="musa", dtype=dtype)
    b = torch.randn((K, N), device="musa", dtype=dtype)
    out = torch.empty((M, N), device="musa", dtype=dtype)
    ref = torch.mm(a, b)

    def torch_fn():
        torch.mm(a, b, out=out)

    def mm_fn():
        if args.variant == "old":
            mm_mod.mm_tle_split_n_pipe(a, b, out, M, N, K)
        elif args.variant == "single":
            mm_mod.mm_tle_split_n_single_pipe(
                a, b, out, M, N, K, num_slots=args.slots
            )
        elif args.variant == "pipe":
            mm_mod.mm_tle_pipe(a, b, out, M, N, K)
        else:
            mm_mod.mm_out(a, b, out=out)

    mm_fn()
    torch.musa.synchronize()
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
    torch_ms, torch_samples = bench(torch_fn, args.warmup, args.rep)
    mm_ms, mm_samples = bench(mm_fn, args.warmup, args.rep)
    print(
        f"RESULT variant={args.variant} dtype={args.dtype} shape={M}x{N}x{K} "
        f"torch_ms={torch_ms:.6f} mm_out_ms={mm_ms:.6f} "
        f"ratio={torch_ms/mm_ms:.4f} torch_samples={torch_samples} "
        f"mm_samples={mm_samples}",
        flush=True,
    )


if __name__ == "__main__":
    main()
