"""Fixed-launch probe for the MTT persistent multifield GEMM."""

import argparse
import importlib

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
    ap.add_argument("--slots", type=int, choices=(2, 3), default=2)
    ap.add_argument("--group-m", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--rep", type=int, default=100)
    ap.add_argument(
        "--manual-events",
        action="store_true",
        help="time repeated launches without Triton cache clearing",
    )
    args = ap.parse_args()

    m, n, k = args.shape
    if k % 64:
        ap.error("K must be divisible by 64")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    a = torch.randn((m, k), device="musa", dtype=dtype)
    b = torch.randn((k, n), device="musa", dtype=dtype)
    ref = torch.mm(a, b)
    out = torch.empty_like(ref)
    desc_a = TensorDescriptor.from_tensor(a, [256, 64])
    desc_b = TensorDescriptor.from_tensor(b, [64, 256])
    grid_m = triton.cdiv(m, 256)
    grid_n = triton.cdiv(n, 256)
    total_tiles = grid_m * grid_n

    def fixed():
        mm.mm_tle_persistent_multifield_kernel[(args.num_sms,)](
            desc_a,
            desc_b,
            out,
            m,
            n,
            k,
            out.stride(0),
            out.stride(1),
            total_tiles,
            grid_n,
            args.num_sms,
            args.slots,
            args.group_m,
            num_warps=16,
            enable_backend_opt=True,
        )

    fixed()
    torch.musa.synchronize()
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
    torch_ms = float(
        triton.testing.do_bench(
            lambda: torch.mm(a, b, out=out),
            warmup=args.warmup,
            rep=args.rep,
        )
    )
    if args.manual_events:
        for _ in range(args.warmup):
            fixed()
        torch.musa.synchronize()
        start = torch.musa.Event(enable_timing=True)
        end = torch.musa.Event(enable_timing=True)
        start.record()
        for _ in range(args.rep):
            fixed()
        end.record()
        torch.musa.synchronize()
        fixed_ms = float(start.elapsed_time(end)) / args.rep
    else:
        fixed_ms = float(
            triton.testing.do_bench(fixed, warmup=args.warmup, rep=args.rep)
        )
    print(
        f"RESULT shape={m}x{n}x{k} dtype={args.dtype} "
        f"num_sms={args.num_sms} slots={args.slots} group_m={args.group_m} "
        f"torch_ms={torch_ms:.6f} fixed_ms={fixed_ms:.6f} "
        f"ratio={torch_ms/fixed_ms:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
