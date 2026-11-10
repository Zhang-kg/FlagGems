"""Compare the normal WS route with a temporary non-WS route."""

import argparse
import importlib

import torch
import torch_musa  # noqa: F401
import triton


mm = importlib.import_module("flag_gems.runtime.backend._mthreads.ops.mm")


def bench(fn, warmup, rep):
    return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", nargs=3, type=int, required=True)
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--rep", type=int, default=40)
    args = ap.parse_args()
    m, n, k = args.shape
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    a = torch.randn((m, k), device="musa", dtype=dtype)
    b = torch.randn((k, n), device="musa", dtype=dtype)
    ref = torch.mm(a, b)
    out = torch.empty_like(ref)

    def normal():
        mm.mm_tle_pipe(a, b, out, m, n, k)

    def nonws():
        old = mm.use_tle_ws_pipe
        mm.use_tle_ws_pipe = lambda *_args: False
        try:
            mm.mm_tle_pipe(a, b, out, m, n, k)
        finally:
            mm.use_tle_ws_pipe = old

    normal()
    torch.musa.synchronize()
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
    torch_ms = bench(lambda: torch.mm(a, b, out=out), args.warmup, args.rep)
    print(f"torch_ms={torch_ms:.6f}")
    for name, fn in (("normal", normal), ("nonws", nonws)):
        try:
            fn()
            torch.musa.synchronize()
            torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
            ms = bench(fn, args.warmup, args.rep)
            print(f"{name}_ms={ms:.6f} ratio={torch_ms/ms:.4f}")
        except Exception as exc:
            print(f"{name}_ERROR={type(exc).__name__}:{str(exc).replace(chr(10), ' ')[:180]}")


if __name__ == "__main__":
    main()
