import argparse
import importlib

import torch
import torch_musa  # noqa: F401
import triton
from triton.tools.tensor_descriptor import TensorDescriptor


mm = importlib.import_module("flag_gems.runtime.backend._mthreads.ops.mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--n", type=int, default=12288)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    args = ap.parse_args()
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    a = torch.randn((args.m, args.k), device="musa", dtype=dt)
    b = torch.randn((args.k, args.n), device="musa", dtype=dt)
    ref = torch.mm(a, b)
    out = torch.empty_like(ref)
    da = TensorDescriptor.from_tensor(a, [64, 64])
    db = TensorDescriptor.from_tensor(b, [64, 64])
    cfg = triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "NUM_SLOTS": 1,
         "MMA_GROUP": 1, "GROUP_M": 1}, num_warps=4, num_stages=3
    )
    grid = (triton.cdiv(args.m, 64) * triton.cdiv(args.n, 64),)

    def fixed():
        tuner = mm.mm_tle_non_ws_pipe_kernel.fn
        old_configs = tuner.configs
        tuner.configs = [cfg]
        try:
            tuner.run(
            da,
            db,
            out,
            args.m,
            args.n,
            args.k,
            out.stride(0),
            out.stride(1),
            grid=grid,
            warmup=0,
            )
        finally:
            tuner.configs = old_configs

    fixed()
    torch.musa.synchronize()
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
    print("torch", triton.testing.do_bench(lambda: torch.mm(a, b, out=out), warmup=10, rep=30))
    print("fixed64", triton.testing.do_bench(fixed, warmup=10, rep=30))


if __name__ == "__main__":
    main()
