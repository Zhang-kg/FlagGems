import argparse
import gc
import importlib
import sys
import time

import torch
import torch_musa  # noqa: F401
import triton
import flag_gems


mm_module = importlib.import_module("flag_gems.runtime.backend._mthreads.ops.mm")


def dispatch_name(m, n, k, dtype):
    if n == 1:
        return "gemv"
    if mm_module.is_tle_split_n_small_compatible(m, n, k, dtype):
        return "split_n_small"
    # Keep this exact MuBLAS-matched route in sync with mm_tle_pipe(), where
    # split384 is selected before the broader persistent multifield fallback.
    if (
        m == 16384
        and n == 1024
        and k == 2048
        and mm_module.is_tle_split384_compatible(m, n, k)
    ):
        return "split384"
    if mm_module.is_tle_persistent_multifield_compatible(m, n, k, dtype):
        return "persistent_multifield"
    if mm_module.is_tle_persistent_16w_compatible(m, n, k, dtype):
        return "persistent_16w"
    if mm_module.is_tle_bn320_compatible(m, n, k):
        return "bn320_ordered"
    if mm_module.is_tle_split128_compatible(m, n, k):
        return "split128"
    if mm_module.is_tle_split_n_full_compatible(m, n, k, dtype):
        return "split_n_full"
    if mm_module.is_tle_split_n_single_compatible(m, n, k, dtype):
        return "split_n_single"
    if mm_module.is_tle_split_n_compatible(m, n, k):
        return "split_n"
    if mm_module.is_tle_split256_ordered_compatible(m, n, k):
        return "split256_ordered"
    if mm_module.is_tle_split384_compatible(m, n, k):
        return "split384"
    if mm_module.is_tle_split_m_fused_compatible(m, n, k):
        return "split_m_fused"
    if mm_module.is_tle_split_m_compatible(m, n, k):
        return "split_m"
    # Mirror mm()'s is_tle_pipe_compatible() guard. Inputs created by this
    # benchmark are contiguous, so only the dtype/shape capability checks are
    # needed here. Shapes that fail this guard use the generic Triton kernel,
    # not a TLE non-WS kernel.
    if (
        getattr(mm_module, "HAS_MTHREADS_REGISTER_FAILURE_RECOVERY", False)
        and getattr(mm_module, "HAS_MTHREADS_TLE_PIPE_SQMMA", False)
        and getattr(mm_module, "HAS_MTHREADS_TLE_MULTIFIELD_PIPE", False)
        and getattr(mm_module, "HAS_MTHREADS_TLE_DYNAMIC_LOOPS", False)
        and dtype in (torch.float16, torch.bfloat16)
        and m >= 32
        and n > 1
        and n % 8 == 0
        and k % 64 == 0
    ):
        return (
            "general_tle_ws"
            if mm_module.use_tle_ws_pipe(m, n, k)
            else "general_tle_nonws"
        )
    return "generic"


def read_shapes(path):
    out = set()
    # The batch runner accepts the compact four-column export produced from
    # the model YAML as well as the original YAML itself.
    with open(path) as probe:
        first = probe.readline()
    if first and not first.startswith("addmm:") and len(first.split()) >= 4:
        with open(path) as f:
            for line in f:
                fields = line.split()
                if len(fields) >= 3 and all(x.lstrip("-").isdigit() for x in fields[:3]):
                    out.add(tuple(int(x) for x in fields[:3]))
        return sorted(out)
    with open(path) as f:
        lines = iter(f)
        in_mm = False
        for line in lines:
            if line.startswith("mm:"):
                in_mm = True
                continue
            if in_mm and line.startswith("  shape_desc:"):
                break
            if in_mm and line.startswith("  - -"):
                vals = [line]
                # The compact shape entry is a four-column B,M,N,K list;
                # the first value is already present on the ``- -`` line.
                vals.extend(next(lines) for _ in range(3))
                nums = [int(x.split("-")[-1].strip()) for x in vals]
                out.add(tuple(nums[1:4]))
    return sorted(out)


def bench(fn, warmup, rep):
    return float(triton.testing.do_bench(fn, warmup=warmup, rep=rep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--rep", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--shape",
        action="append",
        default=None,
        metavar="MxNxK",
        help="benchmark only the requested MxNxK shape; may be repeated",
    )
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--dtype", choices=("bf16", "fp16", "both"), default="both")
    args = ap.parse_args()
    shapes = read_shapes(args.shapes)
    if args.shape:
        selected = []
        for item in args.shape:
            try:
                dims = tuple(int(x) for x in item.lower().split("x"))
            except ValueError:
                ap.error(f"invalid --shape {item!r}; expected MxNxK")
            if len(dims) != 3 or any(x <= 0 for x in dims):
                ap.error(f"invalid --shape {item!r}; expected positive MxNxK")
            selected.append(dims)
        shapes = selected
    if args.limit:
        shapes = shapes[: args.limit]
    if not 0 <= args.shard_index < args.shard_count:
        ap.error("shard-index must be in [0, shard-count)")
    shapes = shapes[args.shard_index :: args.shard_count]
    print(f"shapes={len(shapes)}", flush=True)
    print("dtype\tM\tN\tK\ttorch_ms\ttriton_ms\tratio\tdispatch\tstatus", flush=True)
    dtype_items = (("bf16", torch.bfloat16), ("fp16", torch.float16))
    if args.dtype != "both":
        dtype_items = tuple(x for x in dtype_items if x[0] == args.dtype)
    with open(args.out, "w", buffering=1) as fout:
        fout.write("dtype\tM\tN\tK\ttorch_ms\ttriton_ms\tratio\tdispatch\tstatus\n")
        for dtype_name, dtype in dtype_items:
            for m, n, k in shapes:
                dispatch = dispatch_name(m, n, k, dtype)
                status = "ok"
                base_ms = tri_ms = ratio = float("nan")
                try:
                    a = torch.randn((m, k), device="musa", dtype=dtype)
                    b = torch.randn((k, n), device="musa", dtype=dtype)
                    ref = torch.mm(a, b)
                    out = torch.empty_like(ref)
                    torch_out = torch.empty_like(ref)
                    # Call the backend entry point directly.  This avoids
                    # repeatedly installing the global FlagGems overrides
                    # while preserving the same dispatch as torch.mm(...,
                    # out=...) under use_gems().
                    mm_module.mm_out(a, b, out=out)
                    torch.musa.synchronize()
                    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
                    base_ms = bench(lambda: torch.mm(a, b, out=torch_out), args.warmup, args.rep)
                    tri_ms = bench(lambda: mm_module.mm_out(a, b, out=out), args.warmup, args.rep)
                    ratio = base_ms / tri_ms if tri_ms else float("nan")
                except Exception as exc:  # keep the full scan going
                    status = type(exc).__name__ + ":" + str(exc).replace("\n", " ")[:180]
                row = f"{dtype_name}\t{m}\t{n}\t{k}\t{base_ms:.6f}\t{tri_ms:.6f}\t{ratio:.6f}\t{dispatch}\t{status}"
                print(row, flush=True)
                fout.write(row + "\n")
                try:
                    del a, b, ref, out, torch_out
                    torch.musa.empty_cache()
                except Exception:
                    pass
                gc.collect()


if __name__ == "__main__":
    main()
