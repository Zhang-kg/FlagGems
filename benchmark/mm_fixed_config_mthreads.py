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
    ap.add_argument("--bm", type=int, default=64)
    ap.add_argument("--bn", type=int, default=64)
    ap.add_argument("--bk", type=int, default=64)
    ap.add_argument("--slots", type=int, default=1)
    ap.add_argument("--group", type=int, default=1)
    ap.add_argument("--group-m", type=int, default=1)
    ap.add_argument("--warps", type=int, default=4)
    ap.add_argument("--ws", action="store_true")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--rep", type=int, default=30)
    ap.add_argument("--bypass-cache", action="store_true")
    args = ap.parse_args()
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    a = torch.randn((args.m, args.k), device="musa", dtype=dt)
    b = torch.randn((args.k, args.n), device="musa", dtype=dt)
    ref = torch.mm(a, b)
    out = torch.empty_like(ref)
    da = TensorDescriptor.from_tensor(a, [args.bm, args.bk])
    db = TensorDescriptor.from_tensor(b, [args.bk, args.bn])
    cfg = triton.Config(
        {"BLOCK_M": args.bm, "BLOCK_N": args.bn, "BLOCK_K": args.bk,
         "NUM_SLOTS": args.slots, "MMA_GROUP": args.group,
         "GROUP_M": args.group_m}, num_warps=args.warps, num_stages=3
    )
    grid = (triton.cdiv(args.m, args.bm) * triton.cdiv(args.n, args.bn),)

    def fixed():
        tuner = (mm.mm_tle_pipe_kernel if args.ws else mm.mm_tle_non_ws_pipe_kernel).fn
        old_configs = tuner.configs
        tuner.configs = [cfg]
        try:
            run_kwargs = dict(
                grid=grid,
                warmup=0,
            )
            if args.bypass_cache:
                from flag_gems.utils.libentry import LibTunerRunMode

                with tuner.use_run_mode(LibTunerRunMode.FORCE_POLICY):
                    tuner.run(
                        da, db, out, args.m, args.n, args.k,
                        out.stride(0), out.stride(1), **run_kwargs
                    )
            else:
                tuner.run(
                    da, db, out, args.m, args.n, args.k,
                    out.stride(0), out.stride(1), **run_kwargs
                )
        finally:
            tuner.configs = old_configs

    fixed()
    torch.musa.synchronize()
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
    torch_ms = triton.testing.do_bench(
        lambda: torch.mm(a, b, out=out), warmup=args.warmup, rep=args.rep
    )
    fixed_ms = triton.testing.do_bench(fixed, warmup=args.warmup, rep=args.rep)
    print(
        f"RESULT shape={args.m}x{args.n}x{args.k} dtype={args.dtype} "
        f"ws={args.ws} bm={args.bm} bn={args.bn} bk={args.bk} "
        f"slots={args.slots} group={args.group} group_m={args.group_m} "
        f"warps={args.warps} torch_ms={torch_ms:.6f} fixed_ms={fixed_ms:.6f} "
        f"ratio={torch_ms/fixed_ms:.4f}", flush=True
    )


if __name__ == "__main__":
    main()
