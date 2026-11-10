"""Benchmark-only BM512 persistent SQMMA/TLE GEMM probe.

MuDNN exposes a ``512x256xB128`` persistent family for wide matrices.  The
production MThreads implementation currently uses 256x256xB64.  This probe
keeps the production dispatch untouched and tests the closest legal Triton
schedule: one persistent CTA owns a 512-row by 256-column output tile, with
two 256-row consumers sharing each A/B pipe slot.  BK128 is decomposed into
two legal K64 SQMMA operations at the consumer.
"""

import argparse
import importlib
import statistics

import torch
import torch_musa  # noqa: F401
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems.runtime import torch_device_fn

tle = importlib.import_module("triton.experimental.tle.language")


@triton.jit
def _producer(
    writer,
    a_desc,
    b_desc,
    pid,
    total_tiles: tl.constexpr,
    grid_n: tl.constexpr,
    num_sms: tl.constexpr,
    tile_iters: tl.constexpr,
    k_groups: tl.constexpr,
    block_k: tl.constexpr,
):
    for tile_iter in range(tile_iters):
        tile_id = pid + tile_iter * num_sms
        if tile_id < total_tiles:
            pid_m = tile_id // grid_n
            pid_n = tile_id % grid_n
            m_offset = (pid_m * 512).to(tl.int32)
            n_offset = (pid_n * 256).to(tl.int32)
            for group in range(k_groups):
                token = tile_iter * k_groups + group
                k_offset = group * block_k
                slot = writer.acquire(token)
                tle.gpu.copy(a_desc, slot.a, [512, block_k], [m_offset, k_offset])
                tle.gpu.copy(b_desc, slot.b, [block_k, 256], [k_offset, n_offset])
                writer.commit(token)


@triton.jit
def _consumer(
    reader,
    c_ptr,
    pid,
    M,
    N,
    stride_cm,
    stride_cn,
    total_tiles: tl.constexpr,
    grid_n: tl.constexpr,
    num_sms: tl.constexpr,
    tile_iters: tl.constexpr,
    k_groups: tl.constexpr,
    block_k: tl.constexpr,
    row_offset: tl.constexpr,
):
    acc = tl.zeros((256, 256), dtype=tl.float32)
    for tile_iter in range(tile_iters):
        tile_id = pid + tile_iter * num_sms
        if tile_id < total_tiles:
            # Each consumer owns a fixed half of the 512-row payload.  The
            # pipe's multi-reader release accounting keeps the slot live until
            # both consumers have finished their SQMMA reads.
            acc = tl.zeros((256, 256), dtype=tl.float32)
            pid_m = tile_id // grid_n
            pid_n = tile_id % grid_n
            m_offset = (pid_m * 512).to(tl.int32)
            n_offset = (pid_n * 256).to(tl.int32)
            for group in range(k_groups):
                token = tile_iter * k_groups + group
                ready = reader.wait(token)
                a_tile = ready.slot.a.slice(row_offset, 256, dim=0)
                if block_k == 64:
                    acc = tle.gpu.wgmma(a_tile, ready.slot.b, acc)
                else:
                    # PH1 does not have a reliable native BF16/FP16 K128
                    # intrinsic.  Keep the descriptor K128 for the DMA, then
                    # issue two legal K64 operations.
                    a0 = a_tile.slice(0, 64, dim=1)
                    a1 = a_tile.slice(64, 64, dim=1)
                    b0 = ready.slot.b.slice(0, 64, dim=0)
                    b1 = ready.slot.b.slice(64, 64, dim=0)
                    acc = tle.gpu.wgmma(a0, b0, acc)
                    acc = tle.gpu.wgmma(a1, b1, acc)
                # Keep the wait local to this slot.  The bundled MTT ABI only
                # exposes wait-all, so this is intentionally not a MuDNN-level
                # asynchronous pending-group pipeline.
                acc = tle.gpu.wgmma_wait(0, acc)
                reader.release(token)

            offs_m = m_offset + row_offset + tl.arange(0, 256)
            offs_n = n_offset + tl.arange(0, 256)
            ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
            tl.store(ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)


@triton.jit
def persistent_bm512_kernel(
    a_desc,
    b_desc,
    c_ptr,
    M,
    N,
    stride_cm,
    stride_cn,
    TOTAL_TILES: tl.constexpr,
    GRID_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
    TILE_ITERS: tl.constexpr,
    K_GROUPS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
):
    pid = tl.program_id(0)
    a_smem = tle.gpu.alloc(
        [NUM_SLOTS, 512, BLOCK_K],
        dtype=a_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    b_smem = tle.gpu.alloc(
        [NUM_SLOTS, BLOCK_K, 256],
        dtype=b_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    pipe = tle.pipe(
        capacity=NUM_SLOTS,
        scope="cta",
        name="probe_persistent_bm512",
        a=a_smem,
        b=b_smem,
    )
    tle.gpu.warp_specialize(
        [
            (
                _consumer,
                (
                    pipe.reader(), c_ptr, pid, M, N, stride_cm, stride_cn,
                    TOTAL_TILES, GRID_N, NUM_SMS, TILE_ITERS, K_GROUPS,
                    BLOCK_K, 0,
                ),
            ),
            (
                _consumer,
                (
                    pipe.reader(), c_ptr, pid, M, N, stride_cm, stride_cn,
                    TOTAL_TILES, GRID_N, NUM_SMS, TILE_ITERS, K_GROUPS,
                    BLOCK_K, 256,
                ),
            ),
            (
                _producer,
                (
                    pipe.writer(), a_desc, b_desc, pid, TOTAL_TILES, GRID_N,
                    NUM_SMS, TILE_ITERS, K_GROUPS, BLOCK_K,
                ),
            ),
        ],
        # The second consumer is intentionally given the same register
        # budget as the first.  ``worker_num_warps`` describes the two worker
        # partitions; the default consumer gets the launch ``num_warps``.
        # With the default launch value 8 this is a legal 8+8+8=24-warp
        # static specialization (each partition remains power-of-two).
        worker_num_warps=[8, 8],
        worker_num_regs=[168, 168],
    )


def run(a, b, out, *, slots, num_sms, block_k, default_warps):
    m, k = a.shape
    n = b.shape[1]
    if k % block_k:
        raise ValueError("K must be divisible by block_k")
    grid_n = triton.cdiv(n, 256)
    total_tiles = triton.cdiv(m, 512) * grid_n
    tile_iters = triton.cdiv(total_tiles, num_sms)
    da = TensorDescriptor.from_tensor(a, [512, block_k])
    db = TensorDescriptor.from_tensor(b, [block_k, 256])
    with torch_device_fn.device(a.device):
        persistent_bm512_kernel[(num_sms,)](
            da, db, out, m, n, out.stride(0), out.stride(1), total_tiles,
            grid_n, num_sms, tile_iters, k // block_k, block_k, slots,
            num_warps=default_warps, enable_backend_opt=True,
        )


def bench(fn, warmup, rep):
    for _ in range(warmup):
        fn()
    torch.musa.synchronize()
    values = []
    for _ in range(5):
        start = torch.musa.Event(enable_timing=True)
        end = torch.musa.Event(enable_timing=True)
        start.record()
        for _ in range(rep):
            fn()
        end.record()
        torch.musa.synchronize()
        values.append(start.elapsed_time(end) / rep)
    return statistics.median(values), values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", nargs=3, type=int, required=True)
    ap.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    ap.add_argument("--block-k", type=int, choices=(64, 128), default=64)
    ap.add_argument("--slots", type=int, choices=(1, 2), default=2)
    ap.add_argument("--num-sms", type=int, default=60)
    ap.add_argument(
        "--default-warps",
        type=int,
        choices=(4, 8, 16),
        default=8,
        help="warps for the default consumer partition (workers stay 8+8)",
    )
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--rep", type=int, default=40)
    args = ap.parse_args()
    if args.block_k == 128 and args.slots != 1:
        ap.error("BK128 uses 196608 B per slot; choose --slots 1")
    if args.block_k == 64 and args.slots > 2:
        ap.error("BM512/BK64 exceeds the S5000 shared-memory budget at slots > 2")
    m, n, k = args.shape
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    a = torch.randn((m, k), device="musa", dtype=dtype)
    b = torch.randn((k, n), device="musa", dtype=dtype)
    out = torch.empty((m, n), device="musa", dtype=dtype)
    ref = torch.mm(a, b)
    fn = lambda: run(a, b, out, slots=args.slots, num_sms=args.num_sms,
                     block_k=args.block_k, default_warps=args.default_warps)
    fn()
    torch.musa.synchronize()
    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)
    torch_ms, torch_samples = bench(lambda: torch.mm(a, b, out=out), args.warmup, args.rep)
    probe_ms, probe_samples = bench(fn, args.warmup, args.rep)
    print(
        f"RESULT shape={m}x{n}x{k} dtype={args.dtype} block_k={args.block_k} "
        f"slots={args.slots} sms={args.num_sms} default_warps={args.default_warps} "
        f"torch_ms={torch_ms:.6f} probe_ms={probe_ms:.6f} ratio={torch_ms/probe_ms:.4f} "
        f"torch_samples={torch_samples} probe_samples={probe_samples}",
        flush=True,
    )


if __name__ == "__main__":
    main()
