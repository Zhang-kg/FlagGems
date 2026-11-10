"""Probe a persistent BM=256, BN=128 MTT GEMM tile.

This intentionally remains benchmark-only.  It mirrors the production
multifield persistent protocol while exposing the narrower B tile suggested by
muDNN's 128x128 kernels.
"""

import argparse
import importlib

import torch
import torch_musa  # noqa: F401
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems.runtime import torch_device_fn


tle = importlib.import_module("triton.experimental.tle.language")


@triton.jit
def _producer(writer, a_desc, b_desc, pid, total_tiles, grid_n, num_sms,
              tile_iters, k_tiles: tl.constexpr):
    for tile_iter in range(tile_iters):
        tile_id = pid + tile_iter * num_sms
        if tile_id < total_tiles:
            pid_m = tile_id // grid_n
            pid_n = tile_id % grid_n
            m_offset = (pid_m * 256).to(tl.int32)
            n_offset = (pid_n * 128).to(tl.int32)
            for k_iter in range(k_tiles):
                token = tile_iter * k_tiles + k_iter
                slot = writer.acquire(token)
                k_offset = k_iter * 64
                tle.gpu.copy(a_desc, slot.a, [256, 64], [m_offset, k_offset])
                tle.gpu.copy(b_desc, slot.b, [64, 128], [k_offset, n_offset])
                writer.commit(token)


@triton.jit
def _consumer(reader, c_ptr, pid, M, N, stride_cm, stride_cn,
              total_tiles, grid_n, num_sms, tile_iters,
              k_tiles: tl.constexpr):
    for tile_iter in range(tile_iters):
        tile_id = pid + tile_iter * num_sms
        if tile_id < total_tiles:
            pid_m = tile_id // grid_n
            pid_n = tile_id % grid_n
            m_offset = (pid_m * 256).to(tl.int32)
            n_offset = (pid_n * 128).to(tl.int32)
            acc = tl.zeros((256, 128), dtype=tl.float32)
            for k_iter in range(k_tiles):
                token = tile_iter * k_tiles + k_iter
                ready = reader.wait(token)
                acc = tle.gpu.wgmma(ready.slot.a, ready.slot.b, acc)
                acc = tle.gpu.wgmma_wait(0, acc)
                reader.release(token)
            offs_m = m_offset + tl.arange(0, 256)
            offs_n = n_offset + tl.arange(0, 128)
            ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
            tl.store(ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)


@triton.jit
def persistent_bn128_kernel(a_desc, b_desc, c_ptr, M, N, stride_cm, stride_cn,
                            TOTAL_TILES: tl.constexpr, GRID_N: tl.constexpr,
                            NUM_SMS: tl.constexpr, TILE_ITERS: tl.constexpr,
                            K_TILES: tl.constexpr, NUM_SLOTS: tl.constexpr):
    pid = tl.program_id(0)
    a_smem = tle.gpu.alloc([NUM_SLOTS, 256, 64], dtype=a_desc.dtype,
                           layout=None, scope=tle.gpu.smem,
                           nv_mma_shared_layout=True)
    b_smem = tle.gpu.alloc([NUM_SLOTS, 64, 128], dtype=b_desc.dtype,
                           layout=None, scope=tle.gpu.smem,
                           nv_mma_shared_layout=True)
    pipe = tle.pipe(capacity=NUM_SLOTS, scope="cta", name="probe_persistent_bn128",
                    a=a_smem, b=b_smem)
    tle.gpu.warp_specialize(
        [
            (_consumer, (pipe.reader(), c_ptr, pid, M, N, stride_cm, stride_cn,
                         TOTAL_TILES, GRID_N, NUM_SMS, TILE_ITERS, K_TILES)),
            (_producer, (pipe.writer(), a_desc, b_desc, pid, TOTAL_TILES,
                         GRID_N, NUM_SMS, TILE_ITERS, K_TILES)),
        ],
        worker_num_warps=[4], worker_num_regs=[24],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", nargs=3, type=int, required=True)
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument("--num-sms", type=int, default=60)
    ap.add_argument("--slots", type=int, choices=(1, 2, 3), default=3)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--rep", type=int, default=30)
    ap.add_argument("--manual-events", action="store_true")
    args = ap.parse_args()
    m, n, k = args.shape
    if k % 64 or n % 128:
        ap.error("K must be divisible by 64 and N by 128")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    a = torch.randn((m, k), device="musa", dtype=dtype)
    b = torch.randn((k, n), device="musa", dtype=dtype)
    out = torch.empty((m, n), device="musa", dtype=dtype)
    ref = torch.mm(a, b)
    grid_n = triton.cdiv(n, 128)
    total_tiles = triton.cdiv(m, 256) * grid_n
    tile_iters = triton.cdiv(total_tiles, args.num_sms)
    da = TensorDescriptor.from_tensor(a, [256, 64])
    db = TensorDescriptor.from_tensor(b, [64, 128])

    def run():
        with torch_device_fn.device(a.device):
            persistent_bn128_kernel[(args.num_sms,)](
                da, db, out, m, n, out.stride(0), out.stride(1),
                total_tiles, grid_n, args.num_sms, tile_iters, k // 64,
                args.slots, num_warps=16, enable_backend_opt=True,
            )

    run()
    torch.musa.synchronize()
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
    torch_ms = float(triton.testing.do_bench(
        lambda: torch.mm(a, b, out=out), warmup=args.warmup, rep=args.rep))
    if args.manual_events:
        for _ in range(args.warmup):
            run()
        torch.musa.synchronize()
        start = torch.musa.Event(enable_timing=True)
        end = torch.musa.Event(enable_timing=True)
        start.record()
        for _ in range(args.rep):
            run()
        end.record()
        torch.musa.synchronize()
        probe_ms = float(start.elapsed_time(end)) / args.rep
    else:
        probe_ms = float(triton.testing.do_bench(run, warmup=args.warmup, rep=args.rep))
    print(f"RESULT shape={m}x{n}x{k} dtype={args.dtype} bm=256 bn=128 bk=64 "
          f"num_sms={args.num_sms} slots={args.slots} tiles={total_tiles} "
          f"tile_iters={tile_iters} torch_ms={torch_ms:.6f} "
          f"probe_ms={probe_ms:.6f} ratio={torch_ms/probe_ms:.4f}", flush=True)


if __name__ == "__main__":
    main()
