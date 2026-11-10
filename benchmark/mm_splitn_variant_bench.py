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
    ap.add_argument("--bn", type=int, choices=(64, 128, 256), default=128)
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--rep", type=int, default=80)
    args = ap.parse_args()
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    a = torch.randn((args.m, args.k), device="musa", dtype=dt)
    b = torch.randn((args.k, args.n), device="musa", dtype=dt)
    ref = torch.mm(a, b)
    out = torch.empty_like(ref)
    bm, bk = args.m, 64
    da = TensorDescriptor.from_tensor(a, [bm, bk])
    db = TensorDescriptor.from_tensor(b, [bk, args.bn])
    cfg = triton.Config({"BLOCK_M": bm, "BLOCK_N": args.bn, "BLOCK_K": bk,
                         "NUM_SLOTS": 2, "MMA_GROUP": 1, "GROUP_M": 1},
                        num_warps=args.warps, num_stages=3)
    grid = (triton.cdiv(args.m, bm) * triton.cdiv(args.n, args.bn),)

    def fixed():
        tuner = mm.mm_tle_non_ws_pipe_kernel.fn
        old = tuner.configs
        tuner.configs = [cfg]
        try:
            tuner.run(da, db, out, args.m, args.n, args.k,
                      out.stride(0), out.stride(1), grid=grid, warmup=0)
        finally:
            tuner.configs = old

    fixed()
    torch.musa.synchronize()
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
    t = triton.testing.do_bench(lambda: torch.mm(a, b, out=out), warmup=args.warmup, rep=args.rep)
    x = triton.testing.do_bench(fixed, warmup=args.warmup, rep=args.rep)
    print(f"shape={args.m}x{args.n}x{args.k} dtype={args.dtype} bn={args.bn} warps={args.warps} torch={t:.6f} triton={x:.6f} ratio={t/x:.4f}", flush=True)


if __name__ == "__main__":
    main()
