import argparse
import importlib

import torch
import torch_musa  # noqa: F401
import triton


mm = importlib.import_module("flag_gems.runtime.backend._mthreads.ops.mm")


def bench(fn):
    return float(triton.testing.do_bench(fn, warmup=10, rep=30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", nargs=3, type=int, required=True)
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    args = ap.parse_args()
    m, n, k = args.shape
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    a = torch.randn((m, k), device="musa", dtype=dtype)
    b = torch.randn((k, n), device="musa", dtype=dtype)
    ref = torch.mm(a, b)
    out = torch.empty_like(ref)
    impls = {
        "torch": lambda: torch.mm(a, b, out=out),
        "split_or_dispatch": lambda: mm.mm_out(a, b, out=out),
        "generic": lambda: mm._generic_mm_out(a, b, out=out),
    }
    for name, fn in impls.items():
        try:
            fn()
            torch.musa.synchronize()
            torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
            print(name, bench(fn))
        except Exception as exc:
            print(name, "ERROR", type(exc).__name__, str(exc).replace("\n", " ")[:180])


if __name__ == "__main__":
    main()
