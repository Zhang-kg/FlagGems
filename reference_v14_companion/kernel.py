"""Self-contained experimental baddbmm v14 dispatch for MThreads.

This version intentionally has no dependency on an earlier ``optimize/vN``
directory or on ``common``.  Every compute kernel used by this version lives
beside this file.  The five v10 BF16/B=1/alpha=beta=1 fast paths are retained
unchanged.  Every other valid FP16/BF16/FP32 input uses a shape-tuned direct
Triton fallback with a three-dimensional batch grid and storage-sharing bias
broadcast views.  Its epilogue remains fused in FP32 to match FlagGems' CPU
reference semantics.  Case 1 uses the 300-launch-validated S3/U2 schedule: one
16-warp consumer computes two BN128 halves of a BM256xBN256 tile, one 4-warp
producer feeds a BK64 pipe, and 60 persistent CTAs are protected by an
explicit consumer-epoch barrier.  Case 2 uses the explicit
barrier BM256xBN256xBK64/S2 kernel compiled with the required private
FlagTree layout-conversion synchronization fix.  Case 3 uses an exact-A
BM384xBN256xBK32/S3 schedule with compile-time group-M=2 and a synthetic
SQMMA token for the fused vector bias.  Case 0 uses a single 16-warp
consumer for both BN128 halves of its BM256xBN256 tile, with BK64/S3/U2,
group-M=4, 60 persistent CTAs, and an explicit consumer-epoch barrier.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext


HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_persistent = _load("baddbmm_v13_local_persistent", HERE / "persistent_mm.py")
_c0 = _load(
    "baddbmm_v13_c0_single16_s3u2g4", HERE / "c0_single16_s3u2g4.py"
)
_c1 = _load("baddbmm_v13_c1_single2x128", HERE / "c1_single2x128.py")
_synthetic = _load(
    "baddbmm_v13_synthetic_bias", HERE / "synthetic_bias_split384.py"
)
_c2 = _load("baddbmm_v13_c2_explicit", HERE / "c2_explicit.py")
_c3 = _load("baddbmm_v13_c3_exactA_group2", HERE / "c3_exactA_group2.py")
_fg_hotspots = _load("baddbmm_v14_flaggems_hotspots", HERE / "flaggems_hotspots.py")


_C0_SINGLE16_PERSISTENT = (16384, 7168, 1024)
_C1_PERSISTENT = (16384, 2112, 7168)
_C2_EXPLICIT = (448, 7168, 256)
_C3_EXACT_A_GROUP2 = (14429, 7168, 1024)
_SYNTHETIC_SHAPES = {
    (14429, 7168, 1024),
    (14429, 2112, 7168),
}
SUPPORTED_SHAPES = {
    _C0_SINGLE16_PERSISTENT,
    _C1_PERSISTENT,
    _C2_EXPLICIT,
} | _SYNTHETIC_SHAPES

_MAX_UNROLL_TILES = 64
_BLOCK_K = 32
_NUM_SLOTS = 3
_GENERIC_BLOCK_M = 32
_GENERIC_BLOCK_N = 32
_GENERIC_BLOCK_K = 32
_GENERIC_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _shape(batch1: torch.Tensor, batch2: torch.Tensor):
    if batch1.ndim != 3 or batch2.ndim != 3:
        return None
    if batch1.shape[0] != batch2.shape[0]:
        return None
    if batch1.shape[2] != batch2.shape[1]:
        return None
    return (batch1.shape[1], batch2.shape[2], batch1.shape[2])


def _default_fast_eligible(
    input: torch.Tensor,
    batch1: torch.Tensor,
    batch2: torch.Tensor,
    *,
    beta,
    alpha,
) -> bool:
    return (
        input.ndim == 1
        and batch1.shape[0] == 1
        and _shape(batch1, batch2) in SUPPORTED_SHAPES
        and input.numel() == batch2.shape[2]
        and float(alpha) == 1.0
        and float(beta) == 1.0
        and input.dtype == batch1.dtype == batch2.dtype == torch.bfloat16
        and input.is_contiguous()
        and batch1.is_contiguous()
        and batch2.is_contiguous()
        and input.device == batch1.device == batch2.device
        and batch1.device.type == "musa"
    )


def _validate(
    input: torch.Tensor,
    batch1: torch.Tensor,
    batch2: torch.Tensor,
    out: torch.Tensor,
) -> tuple[torch.Tensor, int, int, int, int]:
    if batch1.ndim != 3 or batch2.ndim != 3 or out.ndim != 3:
        raise ValueError("batch1, batch2 and out must be rank-3")
    if batch1.shape[0] != batch2.shape[0]:
        raise ValueError("batch1 and batch2 batch dimensions must match")
    if batch1.shape[2] != batch2.shape[1]:
        raise ValueError("incompatible matrix dimensions")
    batch = batch1.shape[0]
    m, k = batch1.shape[1:]
    n = batch2.shape[2]
    expected_shape = (batch, m, n)
    if tuple(out.shape) != expected_shape:
        raise ValueError(
            f"out shape must be {expected_shape}, got {tuple(out.shape)}"
        )
    tensors = (input, batch1, batch2, out)
    if any(t.dtype != batch1.dtype for t in tensors):
        raise TypeError("input, batch1, batch2 and out must have one dtype")
    if batch1.dtype not in _GENERIC_DTYPES:
        raise TypeError(f"unsupported baddbmm dtype {batch1.dtype}")
    if any(t.device != batch1.device for t in tensors):
        raise ValueError("all tensors must share one device")
    if batch1.device.type != "musa":
        raise ValueError("baddbmm v13 requires MUSA tensors")
    try:
        bias_view = torch.broadcast_to(input, expected_shape)
    except RuntimeError as exc:
        raise ValueError(
            f"input shape {tuple(input.shape)} is not broadcastable to {expected_shape}"
        ) from exc
    return bias_view, batch, m, n, k


@triton.jit(do_not_specialize=["alpha", "beta"])
def _generic_fallback_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    out_ptr,
    alpha,
    beta,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_bias_b,
    stride_bias_m,
    stride_bias_n,
    stride_out_b,
    stride_out_m,
    stride_out_n,
    BETA_ZERO: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = ext.program_id(0)
    pid_n = ext.program_id(1)
    pid_b = ext.program_id(2)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)
    a_ptrs = (
        a_ptr
        + pid_b * stride_ab
        + rows[:, None] * stride_am
        + k_offsets[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + pid_b * stride_bb
        + k_offsets[:, None] * stride_bk
        + cols[None, :] * stride_bn
    )
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_iter in range(tl.cdiv(K, BLOCK_K)):
        current_k = k_iter * BLOCK_K + k_offsets
        a = tl.load(
            a_ptrs,
            mask=(rows[:, None] < M) & (current_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(current_k[:, None] < K) & (cols[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # FlagGems validates against an FP64 CPU reference and casts only the final
    # value to the requested output dtype.  Keep the complete epilogue fused in
    # FP32; in particular, do not emulate the double-rounding behavior of the
    # non-aliasing torch_musa functional path for matrix/batched bias.
    scaled_product = alpha * accumulator
    if BETA_ZERO:
        result = scaled_product
    else:
        out_mask = (rows[:, None] < M) & (cols[None, :] < N)
        bias_ptrs = (
            bias_ptr
            + pid_b * stride_bias_b
            + rows[:, None] * stride_bias_m
            + cols[None, :] * stride_bias_n
        )
        bias = tl.load(bias_ptrs, mask=out_mask, other=0.0).to(tl.float32)
        result = scaled_product + beta * bias
    out_ptrs = (
        out_ptr
        + pid_b * stride_out_b
        + rows[:, None] * stride_out_m
        + cols[None, :] * stride_out_n
    )
    tl.store(
        out_ptrs,
        result.to(out_ptr.dtype.element_ty),
        mask=(rows[:, None] < M) & (cols[None, :] < N),
    )


def _generic_fallback_out(
    input, batch1, batch2, bias_view, out, *, beta, alpha
):
    batch, m, k = batch1.shape
    n = batch2.shape[2]
    if out.numel() == 0:
        return out
    block_m = _GENERIC_BLOCK_M
    block_n = _GENERIC_BLOCK_N
    block_k = _GENERIC_BLOCK_K
    num_warps = 4
    # MThreads' direct tl.dot lowering strongly favors its 32x32 output tile,
    # but the reduction tile is shape-sensitive.  These routes are the winners
    # of a fresh-cache sweep over every active FlagGems forward/backward shape.
    if m <= 32 and n <= 256 and k >= 512:
        block_k = 64
    elif k <= 128 and m >= 128 and n >= 512:
        block_k = 16
    elif m >= 128 and n <= 128 and k >= 1024:
        block_m, block_n = 16, 16
    elif m <= 128 and n >= 512 and k >= 128:
        block_k = 16
    grid = (
        triton.cdiv(m, block_m),
        triton.cdiv(n, block_n),
        batch,
    )
    alpha_value, beta_value = float(alpha), float(beta)
    with torch_device_fn.device(batch1.device):
        _generic_fallback_kernel[grid](
            batch1,
            batch2,
            bias_view,
            out,
            alpha_value,
            beta_value,
            batch1.stride(0),
            batch1.stride(1),
            batch1.stride(2),
            batch2.stride(0),
            batch2.stride(1),
            batch2.stride(2),
            bias_view.stride(0),
            bias_view.stride(1),
            bias_view.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            BETA_ZERO=beta_value == 0.0,
            M=m,
            N=n,
            K=k,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=1,
        )
        out.record_stream(torch_device_fn.current_stream(batch1.device))
    return out


@triton.jit(do_not_specialize=["beta"])
def _bias_grad_exact_kernel(
    grad_output_ptr,
    grad_input_ptr,
    beta,
    stride_gb,
    stride_gm,
    stride_gn,
    M: tl.constexpr,
    N: tl.constexpr,
    NUMEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Scale a non-broadcasted bias gradient without a PyTorch fallback."""
    offsets = ext.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    batch_indices = offsets // (M * N)
    within_batch = offsets % (M * N)
    row_indices = within_batch // N
    col_indices = within_batch % N
    mask = offsets < NUMEL
    values = tl.load(
        grad_output_ptr
        + batch_indices * stride_gb
        + row_indices * stride_gm
        + col_indices * stride_gn,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        grad_input_ptr + offsets,
        (beta * values).to(grad_input_ptr.dtype.element_ty),
        mask=mask,
    )


@triton.jit(do_not_specialize=["beta"])
def _bias_grad_broadcast_kernel(
    grad_output_ptr,
    grad_input_ptr,
    beta,
    stride_gb,
    stride_gm,
    stride_gn,
    B: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    INPUT_B: tl.constexpr,
    INPUT_M: tl.constexpr,
    INPUT_N: tl.constexpr,
    REDUCE_B: tl.constexpr,
    REDUCE_M: tl.constexpr,
    REDUCE_N: tl.constexpr,
    INPUT_NUMEL: tl.constexpr,
    REDUCE_SIZE: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_REDUCE: tl.constexpr,
):
    """Reduce a right-aligned broadcast bias entirely in Triton."""
    output_offsets = (
        ext.program_id(0) * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    )
    valid_output = output_offsets < INPUT_NUMEL
    input_b = output_offsets // (INPUT_M * INPUT_N)
    within_input_b = output_offsets % (INPUT_M * INPUT_N)
    input_m = within_input_b // INPUT_N
    input_n = within_input_b % INPUT_N
    accumulator = tl.zeros((BLOCK_REDUCE, BLOCK_OUT), dtype=tl.float32)
    for reduction_block in range(tl.cdiv(REDUCE_SIZE, BLOCK_REDUCE)):
        reduction_offsets = (
            reduction_block * BLOCK_REDUCE
            + tl.arange(0, BLOCK_REDUCE)
        )
        reduce_b = reduction_offsets // (REDUCE_M * REDUCE_N)
        within_reduce_b = reduction_offsets % (REDUCE_M * REDUCE_N)
        reduce_m = within_reduce_b // REDUCE_N
        reduce_n = within_reduce_b % REDUCE_N
        if INPUT_B == 1:
            batch_indices = reduce_b[:, None] + tl.zeros(
                (1, BLOCK_OUT), dtype=tl.int32
            )
        else:
            batch_indices = input_b[None, :] + tl.zeros(
                (BLOCK_REDUCE, 1), dtype=tl.int32
            )
        if INPUT_M == 1:
            row_indices = reduce_m[:, None] + tl.zeros(
                (1, BLOCK_OUT), dtype=tl.int32
            )
        else:
            row_indices = input_m[None, :] + tl.zeros(
                (BLOCK_REDUCE, 1), dtype=tl.int32
            )
        if INPUT_N == 1:
            col_indices = reduce_n[:, None] + tl.zeros(
                (1, BLOCK_OUT), dtype=tl.int32
            )
        else:
            col_indices = input_n[None, :] + tl.zeros(
                (BLOCK_REDUCE, 1), dtype=tl.int32
            )
        load_mask = (
            (reduction_offsets[:, None] < REDUCE_SIZE)
            & valid_output[None, :]
            & (batch_indices < B)
            & (row_indices < M)
            & (col_indices < N)
        )
        values = tl.load(
            grad_output_ptr
            + batch_indices * stride_gb
            + row_indices * stride_gm
            + col_indices * stride_gn,
            mask=load_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += values
    totals = tl.sum(accumulator, axis=0)
    tl.store(
        grad_input_ptr + output_offsets,
        (beta * totals).to(grad_input_ptr.dtype.element_ty),
        mask=valid_output,
    )


def _bias_gradient_triton(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    *,
    beta,
) -> torch.Tensor:
    """Return beta * sum_to_size(grad_output, input.shape) via Triton."""
    batch, m, n = grad_output.shape
    padded_shape = (1,) * (3 - input.ndim) + tuple(input.shape)
    if len(padded_shape) != 3:
        raise RuntimeError(
            "baddbmm v13 autograd requires a bias with at most three dimensions"
        )
    input_b, input_m, input_n = padded_shape
    for input_dim, output_dim in zip(padded_shape, (batch, m, n)):
        if input_dim not in (1, output_dim):
            raise RuntimeError(
                f"cannot reduce baddbmm gradient {(batch, m, n)} "
                f"to bias shape {tuple(input.shape)}"
            )
    grad_input = torch.empty_like(input, memory_format=torch.contiguous_format)
    if grad_input.numel() == 0:
        return grad_input
    beta_value = float(beta)
    with torch_device_fn.device(grad_output.device):
        if padded_shape == (batch, m, n):
            block = 256
            _bias_grad_exact_kernel[(triton.cdiv(input.numel(), block),)](
                grad_output,
                grad_input,
                beta_value,
                grad_output.stride(0),
                grad_output.stride(1),
                grad_output.stride(2),
                M=m,
                N=n,
                NUMEL=input.numel(),
                BLOCK=block,
                num_warps=4,
            )
        else:
            reduce_b = batch if input_b == 1 else 1
            reduce_m = m if input_m == 1 else 1
            reduce_n = n if input_n == 1 else 1
            reduce_size = reduce_b * reduce_m * reduce_n
            # Common vector and matrix broadcasts need at most B*M elements.
            # Keep each program bounded while accumulating additional chunks.
            block_out = 32
            block_reduce = min(32, triton.next_power_of_2(reduce_size))
            _bias_grad_broadcast_kernel[
                (triton.cdiv(input.numel(), block_out),)
            ](
                grad_output,
                grad_input,
                beta_value,
                grad_output.stride(0),
                grad_output.stride(1),
                grad_output.stride(2),
                B=batch,
                M=m,
                N=n,
                INPUT_B=input_b,
                INPUT_M=input_m,
                INPUT_N=input_n,
                REDUCE_B=reduce_b,
                REDUCE_M=reduce_m,
                REDUCE_N=reduce_n,
                INPUT_NUMEL=input.numel(),
                REDUCE_SIZE=reduce_size,
                BLOCK_OUT=block_out,
                BLOCK_REDUCE=block_reduce,
                num_warps=4,
            )
        grad_input.record_stream(torch_device_fn.current_stream(grad_output.device))
    return grad_input


def _matmul_gradient_triton(
    left: torch.Tensor,
    right: torch.Tensor,
    shape: tuple[int, int, int],
    *,
    alpha,
) -> torch.Tensor:
    """Allocate and compute an alpha-scaled batched matmul gradient."""
    out = torch.empty(shape, dtype=left.dtype, device=left.device)
    # beta=0 makes the generic kernel skip the dummy bias read.  Reusing out as
    # that dummy does not create an aliasing hazard.
    return _generic_fallback_out(
        out,
        left,
        right,
        out,
        out,
        beta=0.0,
        alpha=alpha,
    )


@triton.jit
def _consumer_top_unrolled(
    a_reader,
    b_reader,
    c_ptr,
    m_offset,
    n_offset,
    M: tl.constexpr,
    N: tl.constexpr,
    K_TILES: tl.constexpr,
):
    accumulator = tl.zeros((256, 256), dtype=tl.float32)
    for token in tl.static_range(K_TILES + 1):
        a_wait = a_reader.wait(token)
        b_wait = b_reader.wait(token)
        a_tile = a_wait.slot.a.slice(0, 256, dim=0)
        accumulator = tle.gpu.wgmma(a_tile, b_wait.slot.b, accumulator)
        accumulator = tle.gpu.wgmma_wait(0, accumulator)
        a_reader.release(token)
        b_reader.release(token)
    rows = m_offset + tl.arange(0, 256)
    cols = n_offset + tl.arange(0, 256)
    tl.store(
        c_ptr + rows[:, None] * N + cols[None, :],
        accumulator.to(c_ptr.dtype.element_ty),
        mask=(rows < M)[:, None] & (cols < N)[None, :],
    )


@triton.jit
def _consumer_bottom_unrolled(
    a_reader,
    b_reader,
    c_ptr,
    m_offset,
    n_offset,
    M: tl.constexpr,
    N: tl.constexpr,
    K_TILES: tl.constexpr,
):
    accumulator = tl.zeros((128, 256), dtype=tl.float32)
    for token in tl.static_range(K_TILES + 1):
        a_wait = a_reader.wait(token)
        b_wait = b_reader.wait(token)
        a_tile = a_wait.slot.a.slice(256, 128, dim=0)
        accumulator = tle.gpu.wgmma(a_tile, b_wait.slot.b, accumulator)
        accumulator = tle.gpu.wgmma_wait(0, accumulator)
        a_reader.release(token)
        b_reader.release(token)
    rows = m_offset + 256 + tl.arange(0, 128)
    cols = n_offset + tl.arange(0, 256)
    tl.store(
        c_ptr + rows[:, None] * N + cols[None, :],
        accumulator.to(c_ptr.dtype.element_ty),
        mask=(rows < M)[:, None] & (cols < N)[None, :],
    )


@libentry()
@triton.jit
def _split384_unrolled_kernel(
    a_desc,
    b_desc,
    synth_a_desc,
    synth_b_desc,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    GRID_N: tl.constexpr,
    K_TILES: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
):
    pid = ext.program_id(0)
    m_offset = ((pid // GRID_N) * 384).to(tl.int32)
    n_offset = ((pid % GRID_N) * 256).to(tl.int32)
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
    a_pipe = tle.pipe(
        capacity=NUM_SLOTS, scope="cta", name="baddbmm_v13_s384_a", a=a_smem
    )
    b_pipe = tle.pipe(
        capacity=NUM_SLOTS, scope="cta", name="baddbmm_v13_s384_b", b=b_smem
    )
    tle.gpu.warp_specialize(
        [
            (
                _consumer_top_unrolled,
                (a_pipe.reader(), b_pipe.reader(), c_ptr, m_offset, n_offset,
                 M, N, K_TILES),
            ),
            (
                _consumer_bottom_unrolled,
                (a_pipe.reader(), b_pipe.reader(), c_ptr, m_offset, n_offset,
                 M, N, K_TILES),
            ),
            (
                _synthetic._producer,
                (a_pipe.writer(), b_pipe.writer(), a_desc, b_desc,
                 synth_a_desc, synth_b_desc, m_offset, n_offset, K_TILES,
                 BLOCK_K),
            ),
        ],
        worker_num_warps=[8, 4],
        worker_num_regs=[168, 24],
    )


def _launch_unrolled(bias, batch1, batch2, out):
    m, n, k = _synthetic._validate(bias, batch1, batch2, out)
    a, b, c = batch1[0], batch2[0], out[0]
    a_desc = TensorDescriptor.from_tensor(a, [512, _BLOCK_K])
    b_desc = TensorDescriptor.from_tensor(b, [_BLOCK_K, 256])
    cache_key = (str(a.device), a.dtype, _BLOCK_K)
    synth_a_base = _synthetic._SYNTHETIC_A_BASE.get(cache_key)
    if synth_a_base is None:
        synth_a_base = torch.zeros((512, 8), dtype=a.dtype, device=a.device)
        synth_a_base[:, 0] = 1.0
        _synthetic._SYNTHETIC_A_BASE[cache_key] = synth_a_base
    synth_a_desc = TensorDescriptor.from_tensor(synth_a_base, [512, _BLOCK_K])
    synth_b_desc = TensorDescriptor.from_tensor(
        bias.reshape(1, n), [_BLOCK_K, 256]
    )
    grid_n = triton.cdiv(n, 256)
    with torch_device_fn.device(a.device):
        _split384_unrolled_kernel[(triton.cdiv(m, 384) * grid_n,)](
            a_desc,
            b_desc,
            synth_a_desc,
            synth_b_desc,
            c,
            M=m,
            N=n,
            GRID_N=grid_n,
            K_TILES=triton.cdiv(k, _BLOCK_K),
            BLOCK_K=_BLOCK_K,
            NUM_SLOTS=_NUM_SLOTS,
            num_warps=16,
            enable_backend_opt=True,
        )
        c.record_stream(torch_device_fn.current_stream(a.device))
        synth_a_base.record_stream(torch_device_fn.current_stream(a.device))
        bias.record_stream(torch_device_fn.current_stream(a.device))
    return out


def _takes_unrolled_path(
    input: torch.Tensor,
    batch1: torch.Tensor,
    batch2: torch.Tensor,
    beta,
    alpha,
) -> bool:
    shape = _shape(batch1, batch2)
    return (
        shape in _SYNTHETIC_SHAPES
        and _default_fast_eligible(
            input, batch1, batch2, beta=beta, alpha=alpha
        )
        and shape[2] % _BLOCK_K == 0
        and triton.cdiv(shape[2], _BLOCK_K) + 1 <= _MAX_UNROLL_TILES
    )


def baddbmm_v14_dispatch_path(
    input: torch.Tensor,
    batch1: torch.Tensor,
    batch2: torch.Tensor,
    *,
    beta=1.0,
    alpha=1.0,
) -> str:
    shape = _shape(batch1, batch2)
    if not _default_fast_eligible(
        input, batch1, batch2, beta=beta, alpha=alpha
    ):
        return "invalid" if shape is None else "triton_generic_fallback"
    if shape == _C2_EXPLICIT:
        return "explicit_bm256_bn256_bk64_s2_4c1p_direct_bias"
    if shape == _C1_PERSISTENT:
        return "persistent_single16_2xbn128_bk64_s3_u2_epoch16_g4_sm60"
    if shape == _C0_SINGLE16_PERSISTENT:
        return "c0_single16_2xbn128_bk64_s3_u2_epoch16_g4_sm60"
    if shape == _C3_EXACT_A_GROUP2:
        return "split384_exactA_group2_synthetic_bias_wgmma_unrolled"
    if _takes_unrolled_path(input, batch1, batch2, beta, alpha):
        return "split384_synthetic_bias_wgmma_unrolled"
    return "split384_synthetic_bias_wgmma"


def baddbmm_v14_out(
    input: torch.Tensor,
    batch1: torch.Tensor,
    batch2: torch.Tensor,
    out: torch.Tensor,
    *,
    beta=1.0,
    alpha=1.0,
) -> torch.Tensor:
    bias_view, _, _, _, _ = _validate(input, batch1, batch2, out)
    path = baddbmm_v14_dispatch_path(
        input, batch1, batch2, beta=beta, alpha=alpha
    )
    # The retained v10 fast kernels use contiguous output addressing.  A legal
    # strided out tensor therefore takes the generic stride-aware path.
    if path != "triton_generic_fallback" and not out.is_contiguous():
        path = "triton_generic_fallback"
    if path == "triton_generic_fallback":
        hotspot_path = _fg_hotspots.dispatch_path(input, batch1, batch2, out)
        if hotspot_path is not None:
            return _fg_hotspots.launch(
                hotspot_path,
                input,
                batch1,
                batch2,
                out,
                beta=beta,
                alpha=alpha,
            )
    if path.startswith("explicit_bm256"):
        return _c2.baddbmm_c2_explicit_out(input, batch1, batch2, out)
    if path.startswith("c0_single16_2xbn128"):
        _c0.persistent_mm_out(
            batch1[0],
            batch2[0],
            out[0],
            bias=input,
            num_sms=60,
            num_slots=3,
            group_m=4,
            unroll_k=2,
        )
        return out
    if path.startswith("persistent_single16_2xbn128"):
        _c1.persistent_mm_out(
            batch1[0],
            batch2[0],
            out[0],
            bias=input,
            num_sms=60,
            num_slots=3,
            group_m=4,
            unroll_k=2,
        )
        return out
    if path.startswith("shared_persistent_bias"):
        _persistent.persistent_mm_out(
            batch1[0],
            batch2[0],
            out[0],
            bias=input,
            num_sms=60,
            num_slots=3,
            group_m=4,
            main_warps=8,
        )
        return out
    if path.startswith("split384_exactA_group2"):
        return _c3.baddbmm_out(input, batch1, batch2, out)
    if path.endswith("_unrolled"):
        return _launch_unrolled(input, batch1, batch2, out)
    if path == "split384_synthetic_bias_wgmma":
        return _synthetic.baddbmm_synthetic_bias_split384_out(
            input, batch1, batch2, out, beta=beta, alpha=alpha
        )
    if path == "triton_generic_fallback":
        return _generic_fallback_out(
            input,
            batch1,
            batch2,
            bias_view,
            out,
            beta=beta,
            alpha=alpha,
        )
    raise ValueError(f"unsupported v13 path {path}")


def _baddbmm_v14_forward(
    input: torch.Tensor,
    batch1: torch.Tensor,
    batch2: torch.Tensor,
    *,
    beta=1.0,
    alpha=1.0,
) -> torch.Tensor:
    shape = _shape(batch1, batch2)
    if shape is None:
        raise ValueError(
            f"incompatible baddbmm operands {tuple(batch1.shape)} and "
            f"{tuple(batch2.shape)}"
        )
    out = torch.empty(
        (batch1.shape[0], batch1.shape[1], batch2.shape[2]),
        dtype=batch1.dtype,
        device=batch1.device,
    )
    return baddbmm_v14_out(
        input, batch1, batch2, out, beta=beta, alpha=alpha
    )


class _BaddbmmV14Autograd(torch.autograd.Function):
    """First-order autograd adapter whose arithmetic remains in Triton."""

    @staticmethod
    def forward(ctx, input, batch1, batch2, beta, alpha):
        ctx.save_for_backward(input, batch1, batch2)
        ctx.beta = float(beta)
        ctx.alpha = float(alpha)
        return _baddbmm_v14_forward(
            input,
            batch1,
            batch2,
            beta=ctx.beta,
            alpha=ctx.alpha,
        )

    @staticmethod
    def backward(ctx, grad_output):
        input, batch1, batch2 = ctx.saved_tensors
        if (
            ctx.needs_input_grad[0]
            and ctx.needs_input_grad[1]
            and ctx.needs_input_grad[2]
            and _fg_hotspots.tiny_backward_eligible(
                input, batch1, batch2, grad_output
            )
        ):
            grad_input, grad_batch1, grad_batch2 = (
                _fg_hotspots.launch_tiny_backward(
                    grad_output,
                    input,
                    batch1,
                    batch2,
                    alpha=ctx.alpha,
                    beta=ctx.beta,
                )
            )
            return grad_input, grad_batch1, grad_batch2, None, None
        grad_input = grad_batch1 = grad_batch2 = None
        if ctx.needs_input_grad[0]:
            grad_input = _bias_gradient_triton(
                grad_output,
                input,
                beta=ctx.beta,
            )
        if ctx.needs_input_grad[1]:
            grad_batch1 = _matmul_gradient_triton(
                grad_output,
                batch2.transpose(1, 2),
                tuple(batch1.shape),
                alpha=ctx.alpha,
            )
        if ctx.needs_input_grad[2]:
            grad_batch2 = _matmul_gradient_triton(
                batch1.transpose(1, 2),
                grad_output,
                tuple(batch2.shape),
                alpha=ctx.alpha,
            )
        return grad_input, grad_batch1, grad_batch2, None, None


def baddbmm_v14(
    input: torch.Tensor,
    batch1: torch.Tensor,
    batch2: torch.Tensor,
    *,
    beta=1.0,
    alpha=1.0,
) -> torch.Tensor:
    if torch.is_grad_enabled() and (
        input.requires_grad or batch1.requires_grad or batch2.requires_grad
    ):
        return _BaddbmmV14Autograd.apply(
            input,
            batch1,
            batch2,
            float(beta),
            float(alpha),
        )
    return _baddbmm_v14_forward(
        input,
        batch1,
        batch2,
        beta=beta,
        alpha=alpha,
    )


__all__ = ["baddbmm_v14", "baddbmm_v14_dispatch_path", "baddbmm_v14_out"]
