"""Benchmark one explicit MTT TLE pipe configuration.

This probe bypasses autotune only for experiments; it does not change the
production dispatch or its expanded candidate set.
"""

import argparse
import importlib

import torch
import torch_musa  # noqa: F401
import triton
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems.runtime import torch_device_fn


mm = importlib.import_module("flag_gems.runtime.backend._mthreads.ops.mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", nargs=3, type=int, required=True)
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument("--kernel", choices=("ws", "nonws"), default="ws")
    ap.add_argument("--block-m", type=int, required=True)
    ap.add_argument("--block-n", type=int, required=True)
    ap.add_argument("--block-k", type=int, default=64)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--group", type=int, default=1)
    ap.add_argument("--mma-group", type=int, default=1)
    ap.add_argument("--warps", type=int, default=8)
    ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--rep", type=int, default=100)
    args = ap.parse_args()

    m, n, k = args.shape
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    a = torch.randn((m, k), device="musa", dtype=dtype)
    b = torch.randn((k, n), device="musa", dtype=dtype)
    out = torch.empty((m, n), device="musa", dtype=dtype)
    ref = torch.mm(a, b)
    desc_a = TensorDescriptor.from_tensor(a, [args.block_m, args.block_k])
    desc_b = TensorDescriptor.from_tensor(b, [args.block_k, args.block_n])
    kernel = (
        mm.mm_tle_pipe_kernel
        if args.kernel == "ws"
        else mm.mm_tle_non_ws_pipe_kernel
    )
    cfg = triton.Config(
        {
            "BLOCK_M": args.block_m,
            "BLOCK_N": args.block_n,
            "BLOCK_K": args.block_k,
            "NUM_SLOTS": args.slots,
            "MMA_GROUP": args.mma_group,
            "GROUP_M": args.group,
        },
        num_stages=args.stages,
        num_warps=args.warps,
        pre_hook=mm.tle_pipe_descriptor_pre_hook,
    )
    grid = (triton.cdiv(m, args.block_m) * triton.cdiv(n, args.block_n),)
    tuner = kernel.fn

    def run():
        old_configs = tuner.configs
        tuner.configs = [cfg]
        try:
            with torch_device_fn.device(a.device):
                tuner.run(
                    desc_a,
                    desc_b,
                    out,
                    m,
                    n,
                    k,
                    out.stride(0),
                    out.stride(1),
                    grid=grid,
                    warmup=0,
                )
        finally:
            tuner.configs = old_configs

    run()
    torch.musa.synchronize()
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
    torch_ms = float(triton.testing.do_bench(lambda: torch.mm(a, b, out=out), warmup=args.warmup, rep=args.rep))
    t_ms = float(triton.testing.do_bench(run, warmup=args.warmup, rep=args.rep))
    print(
        f"RESULT shape={m}x{n}x{k} dtype={args.dtype} kernel={args.kernel} "
        f"bm={args.block_m} bn={args.block_n} bk={args.block_k} slots={args.slots} "
        f"group={args.group} mma_group={args.mma_group} warps={args.warps} "
        f"torch_ms={torch_ms:.6f} triton_ms={t_ms:.6f} ratio={torch_ms/t_ms:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
