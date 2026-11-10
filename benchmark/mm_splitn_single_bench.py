import argparse
import importlib

import torch
import torch_musa  # noqa: F401
import triton
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems.runtime import torch_device_fn


mm = importlib.import_module("flag_gems.runtime.backend._mthreads.ops.mm")


def run(a, b, c, m, n, k, block_k=64, slots=2, group=1):
    da = TensorDescriptor.from_tensor(a, [128, block_k])
    db = TensorDescriptor.from_tensor(b, [block_k, 256])
    grid = (triton.cdiv(n, 256),)
    with torch_device_fn.device(a.device):
        mm.mm_tle_split_n_single_pipe_kernel[grid](
            da, db, c, m, n, c.stride(0), c.stride(1),
            triton.cdiv(k, block_k), 128, block_k, slots, group,
            num_warps=16, enable_backend_opt=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--group", type=int, default=1)
    ap.add_argument("--block-k", type=int, default=64)
    args = ap.parse_args()
    dt = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    for m, n, k in ((128, 9216, 2048),):
        a = torch.randn((m, k), device="musa", dtype=dt)
        b = torch.randn((k, n), device="musa", dtype=dt)
        ref = torch.mm(a, b)
        c = torch.empty_like(ref)
        run(a, b, c, m, n, k, args.block_k, args.slots, args.group)
        torch.musa.synchronize()
        err = (c - ref).abs().max().item()
        base = triton.testing.do_bench(lambda: torch.mm(a, b), warmup=10, rep=30)
        ms = triton.testing.do_bench(lambda: run(a, b, c, m, n, k, args.block_k, args.slots, args.group), warmup=10, rep=30)
        print(args.dtype, args.slots, args.group, base, ms, base / ms, err, flush=True)


if __name__ == "__main__":
    main()
