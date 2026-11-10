"""Compare medium-M ordered TLE paths on the bn320 target shapes."""
import argparse
import importlib
import statistics

import torch
import torch_musa  # noqa: F401

mm = importlib.import_module("flag_gems.runtime.backend._mthreads.ops.mm")


def bench(fn, warmup=20, rep=100):
    for _ in range(warmup):
        fn()
    torch.musa.synchronize()
    vals = []
    for _ in range(5):
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
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--dtype", choices=("bf16", "fp16"), required=True)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--rep", type=int, default=100)
    ap.add_argument("--bn-tail", type=int, default=64,
                    help="BN320 tail columns (probe only)")
    ap.add_argument("--bn-bottom", type=int, default=None,
                    help="BN320 bottom rows (defaults to production heuristic)")
    ap.add_argument("--bn-slots", type=int, default=2,
                    help="BN320 TLE pipe slots")
    ap.add_argument("--bn-stages", type=int, default=None,
                    help="optional Triton launch num_stages")
    args = ap.parse_args()
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    m, n, k = args.m, args.n, args.k
    a = torch.randn((m, k), device="musa", dtype=dt)
    b = torch.randn((k, n), device="musa", dtype=dt)
    out = torch.empty((m, n), device="musa", dtype=dt)
    ref = torch.mm(a, b)
    fns = {
        "torch": lambda: torch.mm(a, b, out=out),
        "bn320": lambda: mm.mm_tle_bn320_ordered(
            a, b, out, m, n, k, block_n_tail=args.bn_tail,
            block_m_bottom=args.bn_bottom, num_slots=args.bn_slots,
            num_stages=args.bn_stages,
        ),
        "split384": lambda: mm.mm_tle_split384_pipe(a, b, out, m, n, k),
        "splitm": lambda: mm.mm_tle_split_m_pipe(a, b, out, m, n, k),
        "split256": lambda: mm.mm_tle_split256_ordered(a, b, out, m, n, k),
        "persistent": lambda: mm.mm_tle_persistent_multifield(a, b, out, m, n, k),
        "persistent16w": lambda: mm.mm_tle_persistent_16w(a, b, out, m, n, k),
    }
    results = {}
    for name, fn in fns.items():
        fn()
        torch.musa.synchronize()
        torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
        results[name] = bench(fn, args.warmup, args.rep)
    base = results["torch"][0]
    print(f"RESULT shape={m}x{n}x{k} dtype={args.dtype} "
          + " ".join(f"{n}_ms={v[0]:.6f} {n}_ratio={base/v[0]:.4f}" for n, v in results.items())
          + f" samples=" + repr({n: v[1] for n, v in results.items()}), flush=True)


if __name__ == "__main__":
    main()
