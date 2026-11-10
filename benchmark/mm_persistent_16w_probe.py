"""Independent fixed-launch probe for the 16-warp persistent MTT GEMM."""

import argparse
import importlib
import statistics

import torch
import torch_musa  # noqa: F401
import triton
from triton.tools.tensor_descriptor import TensorDescriptor

mm = importlib.import_module("flag_gems.runtime.backend._mthreads.ops.mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", nargs=3, type=int, required=True)
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument("--num-sms", type=int, default=60)
    ap.add_argument("--slots", type=int, choices=(1, 2, 3), default=2)
    ap.add_argument("--group-m", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--rep", type=int, default=30)
    args = ap.parse_args()

    m, n, k = args.shape
    if k % 64:
        ap.error("K must be divisible by 64")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    a = torch.randn((m, k), device="musa", dtype=dtype)
    b = torch.randn((k, n), device="musa", dtype=dtype)
    out = torch.empty((m, n), device="musa", dtype=dtype)
    ref = torch.mm(a, b)
    da = TensorDescriptor.from_tensor(a, [256, 64])
    db = TensorDescriptor.from_tensor(b, [64, 256])
    grid_m = triton.cdiv(m, 256)
    grid_n = triton.cdiv(n, 256)
    total_tiles = grid_m * grid_n
    num_sms = args.num_sms
    launch_tiles = triton.cdiv(total_tiles, num_sms) * num_sms

    def fixed():
        mm.mm_tle_persistent_16w_kernel[(num_sms,)](
            da,
            db,
            out,
            m,
            n,
            k,
            launch_tiles,
            grid_n,
            num_sms,
            args.slots,
            args.group_m,
            num_warps=4,
            enable_backend_opt=True,
            disable_max_ilp_scheduler=True,
        )

    fixed()
    torch.musa.synchronize()
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)

    def bench(fn):
        for _ in range(args.warmup):
            fn()
        torch.musa.synchronize()
        vals = []
        for _ in range(5):
            start = torch.musa.Event(enable_timing=True)
            end = torch.musa.Event(enable_timing=True)
            start.record()
            for _ in range(args.rep):
                fn()
            end.record()
            torch.musa.synchronize()
            vals.append(start.elapsed_time(end) / args.rep)
        return statistics.median(vals), vals

    torch_ms, _ = bench(lambda: torch.mm(a, b, out=out))
    fixed_ms, vals = bench(fixed)
    print(
        f"RESULT shape={m}x{n}x{k} dtype={args.dtype} num_sms={num_sms} "
        f"slots={args.slots} group_m={args.group_m} torch_ms={torch_ms:.6f} "
        f"fixed_ms={fixed_ms:.6f} ratio={torch_ms/fixed_ms:.4f} samples={vals}",
        flush=True,
    )


if __name__ == "__main__":
    main()
