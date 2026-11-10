"""High-repetition benchmark for the MTT bn320 ordered path."""
import argparse
import importlib
import statistics

import torch
import torch_musa  # noqa: F401

mm = importlib.import_module("flag_gems.runtime.backend._mthreads.ops.mm")


def bench(fn, warmup, rep, rounds=5):
    for _ in range(warmup):
        fn()
    torch.musa.synchronize()
    vals = []
    for _ in range(rounds):
        start = torch.musa.Event(enable_timing=True)
        end = torch.musa.Event(enable_timing=True)
        start.record()
        for _ in range(rep):
            fn()
        end.record()
        torch.musa.synchronize()
        vals.append(start.elapsed_time(end) / rep)
    return statistics.median(vals), vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--dtype", choices=("bf16", "fp16"), required=True)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--rep", type=int, default=200)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--bottom", type=int, default=0)
    ap.add_argument("--stages", type=int, default=0)
    ap.add_argument("--tail", type=int, default=64)
    args = ap.parse_args()
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    m, n, k = args.m, args.n, args.k
    a = torch.randn((m, k), device="musa", dtype=dt)
    b = torch.randn((k, n), device="musa", dtype=dt)
    out = torch.empty((m, n), device="musa", dtype=dt)
    ref = torch.mm(a, b)

    def torch_fn():
        torch.mm(a, b, out=out)

    def bn_fn():
        mm.mm_tle_bn320_ordered(
            a,
            b,
            out,
            m,
            n,
            k,
            num_slots=args.slots,
            block_m_bottom=args.bottom or None,
            block_n_tail=args.tail,
            num_stages=args.stages or None,
        )

    bn_fn()
    torch.musa.synchronize()
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
    t_ms, ts = bench(torch_fn, args.warmup, args.rep)
    b_ms, bs = bench(bn_fn, args.warmup, args.rep)
    print(f"RESULT shape={m}x{n}x{k} dtype={args.dtype} torch_ms={t_ms:.6f} bn320_ms={b_ms:.6f} ratio={t_ms/b_ms:.4f} torch_samples={ts} bn_samples={bs}", flush=True)


if __name__ == "__main__":
    main()
