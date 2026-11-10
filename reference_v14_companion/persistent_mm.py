"""Version-local MThreads persistent BF16 MM/bias-MM kernel.

This copied module intentionally contains only the two-dimensional compute
primitive.  Operator wrappers for BMM and baddbmm keep their own validation,
fallback, and public semantics.  Keeping the copy beside v10 prevents the
implementation from depending on a mutable cross-version ``common`` module.

The schedule is derived from the stable addmm-v4 persistent kernel:

* BM256 x BN256 x BK64;
* a 4-warp 64-column tail consumer in the default partition;
* an 8- or 16-warp 192-column main consumer;
* one 4-warp TMA producer;
* real output-tile count, with no synthetic drain tokens or exit handshake.

``HAS_BIAS`` seeds the FP32 accumulators with a broadcast N-vector.  With it
disabled, the exact same MM main loop serves BMM.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems.runtime import torch_device_fn


BLOCK_M = tl.constexpr(256)
BLOCK_N = tl.constexpr(256)
BLOCK_K = tl.constexpr(64)


@triton.jit
def _producer(
    writer,
    a_desc,
    b_desc,
    pid,
    total_tiles: tl.constexpr,
    grid_m: tl.constexpr,
    grid_n: tl.constexpr,
    num_sms: tl.constexpr,
    tile_iters: tl.constexpr,
    group_m: tl.constexpr,
    k_tiles: tl.constexpr,
):
    group_width: tl.constexpr = group_m * grid_n
    for tile_iter in range(tile_iters):
        tile_id = pid + tile_iter * num_sms
        if tile_id < total_tiles:
            group_id = tile_id // group_width
            first_m = group_id * group_m
            actual_group_m = tl.minimum(grid_m - first_m, group_m)
            pid_in_group = tile_id % group_width
            pid_m = first_m + pid_in_group % actual_group_m
            pid_n = pid_in_group // actual_group_m
            m_offset = (pid_m * BLOCK_M).to(tl.int32)
            n_offset = (pid_n * BLOCK_N).to(tl.int32)
            for k_iter in range(k_tiles):
                token = tile_iter * k_tiles + k_iter
                k_offset = k_iter * BLOCK_K
                slot = writer.acquire(token)
                tle.gpu.copy(
                    a_desc, slot.a, [BLOCK_M, BLOCK_K], [m_offset, k_offset]
                )
                tle.gpu.copy(
                    b_desc, slot.b, [BLOCK_K, BLOCK_N], [k_offset, n_offset]
                )
                writer.commit(token)


@triton.jit
def _tail_consumer(
    reader,
    to_tail,
    to_main,
    bias_ptr,
    c_ptr,
    pid,
    M: tl.constexpr,
    N: tl.constexpr,
    total_tiles: tl.constexpr,
    grid_m: tl.constexpr,
    grid_n: tl.constexpr,
    num_sms: tl.constexpr,
    tile_iters: tl.constexpr,
    group_m: tl.constexpr,
    k_tiles: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    group_width: tl.constexpr = group_m * grid_n
    for tile_iter in range(tile_iters):
        tile_id = pid + tile_iter * num_sms
        if tile_id < total_tiles:
            group_id = tile_id // group_width
            first_m = group_id * group_m
            actual_group_m = tl.minimum(grid_m - first_m, group_m)
            pid_in_group = tile_id % group_width
            pid_m = first_m + pid_in_group % actual_group_m
            pid_n = pid_in_group // actual_group_m
            m_offset = (pid_m * BLOCK_M).to(tl.int32)
            n_offset = (pid_n * BLOCK_N).to(tl.int32)
            columns = n_offset + 192 + tl.arange(0, 64)
            accumulator = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
            if HAS_BIAS:
                bias = tl.load(
                    bias_ptr + columns, mask=columns < N, other=0.0
                ).to(tl.float32)
                accumulator += bias[None, :]

            for k_iter in range(k_tiles):
                token = tile_iter * k_tiles + k_iter
                ready = reader.wait(token)
                tle.gpu.barrier_wait(to_tail, phaseIdx=(token + 1) & 1)
                b_tail = ready.slot.b.slice(192, 64, dim=1)
                accumulator = tle.gpu.wgmma(
                    ready.slot.a, b_tail, accumulator
                )
                tle.gpu.barrier_arrive(to_main, phaseIdx=token & 1)
                accumulator = tle.gpu.wgmma_wait(0, accumulator)
                reader.release(token)

            rows = m_offset + tl.arange(0, BLOCK_M)
            tl.store(
                c_ptr + rows[:, None] * N + columns[None, :],
                accumulator.to(c_ptr.dtype.element_ty),
                mask=(rows < M)[:, None] & (columns < N)[None, :],
            )


@triton.jit
def _main_consumer(
    reader,
    to_tail,
    to_main,
    bias_ptr,
    c_ptr,
    pid,
    M: tl.constexpr,
    N: tl.constexpr,
    total_tiles: tl.constexpr,
    grid_m: tl.constexpr,
    grid_n: tl.constexpr,
    num_sms: tl.constexpr,
    tile_iters: tl.constexpr,
    group_m: tl.constexpr,
    k_tiles: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    group_width: tl.constexpr = group_m * grid_n
    for tile_iter in range(tile_iters):
        tile_id = pid + tile_iter * num_sms
        if tile_id < total_tiles:
            group_id = tile_id // group_width
            first_m = group_id * group_m
            actual_group_m = tl.minimum(grid_m - first_m, group_m)
            pid_in_group = tile_id % group_width
            pid_m = first_m + pid_in_group % actual_group_m
            pid_n = pid_in_group // actual_group_m
            m_offset = (pid_m * BLOCK_M).to(tl.int32)
            n_offset = (pid_n * BLOCK_N).to(tl.int32)
            columns_main = n_offset + tl.arange(0, 128)
            columns_mid = n_offset + 128 + tl.arange(0, 64)
            accumulator_main = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
            accumulator_mid = tl.zeros((BLOCK_M, 64), dtype=tl.float32)
            if HAS_BIAS:
                bias_main = tl.load(
                    bias_ptr + columns_main,
                    mask=columns_main < N,
                    other=0.0,
                ).to(tl.float32)
                bias_mid = tl.load(
                    bias_ptr + columns_mid,
                    mask=columns_mid < N,
                    other=0.0,
                ).to(tl.float32)
                accumulator_main += bias_main[None, :]
                accumulator_mid += bias_mid[None, :]

            for k_iter in range(k_tiles):
                token = tile_iter * k_tiles + k_iter
                ready = reader.wait(token)
                tle.gpu.barrier_wait(to_main, phaseIdx=token & 1)
                b_main = ready.slot.b.slice(0, 128, dim=1)
                accumulator_main = tle.gpu.wgmma(
                    ready.slot.a, b_main, accumulator_main
                )
                b_mid = ready.slot.b.slice(128, 64, dim=1)
                accumulator_mid = tle.gpu.wgmma(
                    ready.slot.a, b_mid, accumulator_mid
                )
                tle.gpu.barrier_arrive(to_tail, phaseIdx=token & 1)
                accumulator_main = tle.gpu.wgmma_wait(0, accumulator_main)
                accumulator_mid = tle.gpu.wgmma_wait(0, accumulator_mid)
                reader.release(token)

            rows = m_offset + tl.arange(0, BLOCK_M)
            tl.store(
                c_ptr + rows[:, None] * N + columns_mid[None, :],
                accumulator_mid.to(c_ptr.dtype.element_ty),
                mask=(rows < M)[:, None] & (columns_mid < N)[None, :],
            )
            tl.store(
                c_ptr + rows[:, None] * N + columns_main[None, :],
                accumulator_main.to(c_ptr.dtype.element_ty),
                mask=(rows < M)[:, None] & (columns_main < N)[None, :],
            )


@triton.jit
def _persistent_mm_kernel(
    a_desc,
    b_desc,
    bias_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TOTAL_TILES: tl.constexpr,
    GRID_M: tl.constexpr,
    GRID_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    GROUP_M: tl.constexpr,
    MAIN_WARPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(0)
    a_smem = tle.gpu.alloc(
        [NUM_SLOTS, BLOCK_M, BLOCK_K],
        dtype=a_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    b_smem = tle.gpu.alloc(
        [NUM_SLOTS, BLOCK_K, BLOCK_N],
        dtype=b_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    pipe = tle.pipe(
        capacity=NUM_SLOTS,
        scope="cta",
        name="shared_persistent_mm_ab",
        a=a_smem,
        b=b_smem,
    )
    to_tail = tle.gpu.alloc_barrier(
        arrive_count=MAIN_WARPS, init=tle.gpu.PENDING
    )
    to_main = tle.gpu.alloc_barrier(arrive_count=4, init=tle.gpu.PENDING)
    tile_iters: tl.constexpr = tl.cdiv(TOTAL_TILES, NUM_SMS)
    k_tiles: tl.constexpr = K // BLOCK_K

    if MAIN_WARPS == 16:
        tle.gpu.warp_specialize(
            [
                (
                    _tail_consumer,
                    (
                        pipe.reader(), to_tail, to_main, bias_ptr, c_ptr, pid,
                        M, N, TOTAL_TILES, GRID_M, GRID_N, NUM_SMS,
                        tile_iters, GROUP_M, k_tiles, HAS_BIAS,
                    ),
                ),
                (
                    _main_consumer,
                    (
                        pipe.reader(), to_tail, to_main, bias_ptr, c_ptr, pid,
                        M, N, TOTAL_TILES, GRID_M, GRID_N, NUM_SMS,
                        tile_iters, GROUP_M, k_tiles, HAS_BIAS,
                    ),
                ),
                (
                    _producer,
                    (
                        pipe.writer(), a_desc, b_desc, pid, TOTAL_TILES,
                        GRID_M, GRID_N, NUM_SMS, tile_iters, GROUP_M, k_tiles,
                    ),
                ),
            ],
            worker_num_warps=[16, 4],
            worker_num_regs=[112, 24],
        )
    else:
        tle.gpu.warp_specialize(
            [
                (
                    _tail_consumer,
                    (
                        pipe.reader(), to_tail, to_main, bias_ptr, c_ptr, pid,
                        M, N, TOTAL_TILES, GRID_M, GRID_N, NUM_SMS,
                        tile_iters, GROUP_M, k_tiles, HAS_BIAS,
                    ),
                ),
                (
                    _main_consumer,
                    (
                        pipe.reader(), to_tail, to_main, bias_ptr, c_ptr, pid,
                        M, N, TOTAL_TILES, GRID_M, GRID_N, NUM_SMS,
                        tile_iters, GROUP_M, k_tiles, HAS_BIAS,
                    ),
                ),
                (
                    _producer,
                    (
                        pipe.writer(), a_desc, b_desc, pid, TOTAL_TILES,
                        GRID_M, GRID_N, NUM_SMS, tile_iters, GROUP_M, k_tiles,
                    ),
                ),
            ],
            worker_num_warps=[8, 4],
            worker_num_regs=[192, 24],
        )


def persistent_mm_out(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    num_sms: int = 60,
    num_slots: int = 3,
    group_m: int = 2,
    main_warps: int = 16,
) -> torch.Tensor:
    """Run the shared persistent kernel into a preallocated 2-D output."""

    if a.ndim != 2 or b.ndim != 2 or out.ndim != 2:
        raise ValueError("a, b, and out must be rank-2")
    m, k = a.shape
    if b.shape[0] != k:
        raise ValueError("incompatible matrix dimensions")
    n = b.shape[1]
    if tuple(out.shape) != (m, n):
        raise ValueError("output shape mismatch")
    tensors = (a, b, out) if bias is None else (a, b, out, bias)
    if any(t.dtype != torch.bfloat16 for t in tensors):
        raise TypeError("shared persistent kernel requires BF16")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("shared persistent kernel requires contiguous tensors")
    if k % 64:
        raise ValueError("K must be divisible by 64")
    if bias is not None and (bias.ndim != 1 or bias.numel() != n):
        raise ValueError("bias must be an N-vector")
    if num_slots not in (2, 3):
        raise ValueError("num_slots must be 2 or 3")
    if main_warps not in (8, 16):
        raise ValueError("main_warps must be 8 or 16")
    if group_m not in (1, 2, 4):
        raise ValueError("group_m must be 1, 2, or 4")

    a_desc = TensorDescriptor.from_tensor(a, [256, 64])
    b_desc = TensorDescriptor.from_tensor(b, [64, 256])
    grid_m = triton.cdiv(m, 256)
    grid_n = triton.cdiv(n, 256)
    total_tiles = grid_m * grid_n
    bias_arg = out if bias is None else bias
    with torch_device_fn.device(a.device):
        _persistent_mm_kernel[(num_sms,)](
            a_desc,
            b_desc,
            bias_arg,
            out,
            M=m,
            N=n,
            K=k,
            TOTAL_TILES=total_tiles,
            GRID_M=grid_m,
            GRID_N=grid_n,
            NUM_SMS=num_sms,
            NUM_SLOTS=num_slots,
            GROUP_M=group_m,
            MAIN_WARPS=main_warps,
            HAS_BIAS=bias is not None,
            num_warps=4,
            enable_backend_opt=True,
            disable_max_ilp_scheduler=True,
        )
        out.record_stream(torch_device_fn.current_stream(a.device))
        if bias is not None:
            bias.record_stream(torch_device_fn.current_stream(a.device))
    return out


__all__ = ["persistent_mm_out"]
