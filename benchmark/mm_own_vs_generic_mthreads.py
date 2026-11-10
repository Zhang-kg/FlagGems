import argparse
import importlib
import os

import torch
import torch_musa  # noqa: F401
import triton


mm = importlib.import_module("flag_gems.runtime.backend._mthreads.ops.mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", nargs=3, type=int, required=True)
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    args = ap.parse_args()
    m, n, k = args.shape
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    a = torch.randn((m, k), device="musa", dtype=dtype)
    b = torch.randn((k, n), device="musa", dtype=dtype)
    out = torch.empty((m, n), device="musa", dtype=dtype)
    ref = torch.mm(a, b)

    def own():
        grid = lambda meta: (
            triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
        )
        return mm.mm_kernel[grid](
            a,
            b,
            out,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
            dtype=args.dtype,
            GROUP_M=8,
            IS_FP64=False,
        )

    impls = {
        "torch": lambda: torch.mm(a, b, out=out),
        "current": lambda: mm.mm_out(a, b, out=out),
        "generic": lambda: mm._generic_mm_out(a, b, out=out),
        "own_mm_kernel": own,
    }
    only = os.environ.get("MM_AB_ONLY")
    if only:
        impls = {name: fn for name, fn in impls.items() if name in only.split(",")}
    for name, fn in impls.items():
        try:
            fn()
            torch.musa.synchronize()
            torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
            elapsed = float(triton.testing.do_bench(fn, warmup=5, rep=15))
            print(f"{name}\t{elapsed:.6f}", flush=True)
        except Exception as exc:
            print(f"{name}\tERROR\t{type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    main()
