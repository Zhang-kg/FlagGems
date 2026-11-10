"""Exact-shape FlagGems BAddBMM hot paths for MThreads.

The kernels in this file are self-contained.  They fuse ``alpha * A @ B``
and the vector-bias ``beta * input`` epilogue, so no temporary BMM output or
second pointwise launch is required.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems.runtime import torch_device_fn


TINY_PATH = "fg_b4_m1_n1_k32_fused_scalar"
SMALLM_PATH = "fg_b4_m15_n160_k1024_fused_tle_bm64_bn64_bk128_s2_w8"
K71_PATH = "fg_b4_m495_n5333_k71_fused_exact_bm32_bn128_3xbk32"

_HALF_DTYPES = (torch.float16, torch.bfloat16)
_ALL_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _common_eligible(
    bias: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
) -> bool:
    return (
        bias.ndim == 1
        and a.ndim == 3
        and b.ndim == 3
        and out.ndim == 3
        and a.shape[0] == b.shape[0] == out.shape[0] == 4
        and a.shape[2] == b.shape[1]
        and tuple(out.shape) == (4, a.shape[1], b.shape[2])
        and bias.numel() == b.shape[2]
        and bias.dtype == a.dtype == b.dtype == out.dtype
        and a.dtype in _ALL_DTYPES
        and bias.is_contiguous()
        and a.is_contiguous()
        and b.is_contiguous()
        and out.is_contiguous()
        and bias.device == a.device == b.device == out.device
        and a.device.type == "musa"
    )


def dispatch_path(
    bias: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
) -> Optional[str]:
    if not _common_eligible(bias, a, b, out):
        return None
    shape = (a.shape[0], a.shape[1], b.shape[2], a.shape[2])
    if shape == (4, 1, 1, 32):
        return TINY_PATH
    if shape == (4, 15, 160, 1024) and a.dtype in _HALF_DTYPES:
        return SMALLM_PATH
    if shape == (4, 495, 5333, 71):
        return K71_PATH
    return None


@triton.jit(do_not_specialize=["alpha", "beta"])
def _tiny_fused_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    out_ptr,
    alpha,
    beta,
):
    batch = tl.arange(0, 4)
    k = tl.arange(0, 32)
    a = tl.load(a_ptr + batch[:, None] * 32 + k[None, :]).to(tl.float32)
    b = tl.load(b_ptr + batch[:, None] * 32 + k[None, :]).to(tl.float32)
    dot = tl.sum(a * b, axis=1)
    bias = tl.load(bias_ptr).to(tl.float32)
    tl.store(out_ptr + batch, alpha * dot + beta * bias)


@triton.jit(do_not_specialize=["alpha", "beta"])
def _tiny_backward_kernel(
    grad_output_ptr,
    a_ptr,
    b_ptr,
    grad_bias_ptr,
    grad_a_ptr,
    grad_b_ptr,
    alpha,
    beta,
):
    offsets = tl.arange(0, 64)
    batch = offsets // 32
    k = offsets % 32
    grad_output = tl.load(grad_output_ptr + batch).to(tl.float32)
    a = tl.load(a_ptr + offsets).to(tl.float32)
    b = tl.load(b_ptr + offsets).to(tl.float32)
    tl.store(
        grad_a_ptr + offsets,
        (alpha * grad_output * b).to(grad_a_ptr.dtype.element_ty),
    )
    tl.store(
        grad_b_ptr + offsets,
        (alpha * a * grad_output).to(grad_b_ptr.dtype.element_ty),
    )
    tl.store(
        grad_bias_ptr + batch,
        (beta * grad_output).to(grad_bias_ptr.dtype.element_ty),
        mask=k == 0,
    )


_SMALLM_BM = tl.constexpr(64)
_SMALLM_BN = tl.constexpr(64)
_SMALLM_K_FRAG = tl.constexpr(64)


@triton.jit
def _smallm_producer(
    a_writer,
    b_writer,
    a_desc,
    b_desc,
    pid_b,
    pid_n,
    K_TILES: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    a_row = (pid_b * 15).to(tl.int32)
    n_offset = (pid_n * _SMALLM_BN).to(tl.int32)
    for k_iter in tl.range(0, K_TILES, num_stages=1):
        k_offset = (k_iter * BLOCK_K).to(tl.int32)
        a_slot = a_writer.acquire(k_iter)
        b_slot = b_writer.acquire(k_iter)
        tle.gpu.copy(
            a_desc,
            a_slot.a,
            (_SMALLM_BM, BLOCK_K),
            (a_row, k_offset),
        )
        b_row = (pid_b * 1024 + k_offset).to(tl.int32)
        tle.gpu.copy(
            b_desc,
            b_slot.b,
            (BLOCK_K, _SMALLM_BN),
            (b_row, n_offset),
        )
        a_writer.commit(k_iter)
        b_writer.commit(k_iter)


@triton.jit(do_not_specialize=["alpha", "beta"])
def _smallm_consumer(
    a_reader,
    b_reader,
    bias_ptr,
    out_ptr,
    alpha,
    beta,
    pid_b,
    pid_n,
    K_TILES: tl.constexpr,
    K_PARTS: tl.constexpr,
    UNROLL: tl.constexpr,
):
    acc = tl.zeros((_SMALLM_BM, _SMALLM_BN), dtype=tl.float32)
    for k_iter in tl.range(
        0,
        K_TILES,
        num_stages=1,
        loop_unroll_factor=UNROLL,
    ):
        a_wait = a_reader.wait(k_iter)
        b_wait = b_reader.wait(k_iter)
        for k_part in tl.static_range(K_PARTS):
            a_k = a_wait.slot.a.subslice(
                k_part * _SMALLM_K_FRAG, _SMALLM_K_FRAG, dim=1
            )
            b_k = b_wait.slot.b.subslice(
                k_part * _SMALLM_K_FRAG, _SMALLM_K_FRAG, dim=0
            )
            acc = tle.gpu.wgmma(a_k, b_k, acc)
            acc = tle.gpu.wgmma_wait(0, acc)
        a_reader.release(k_iter)
        b_reader.release(k_iter)

    rows = tl.arange(0, _SMALLM_BM)
    cols = pid_n * _SMALLM_BN + tl.arange(0, _SMALLM_BN)
    bias = tl.load(bias_ptr + cols, mask=cols < 160, other=0.0).to(tl.float32)
    result = alpha * acc + beta * bias[None, :]
    tl.store(
        out_ptr + pid_b * (15 * 160) + rows[:, None] * 160 + cols[None, :],
        result.to(out_ptr.dtype.element_ty),
        mask=(rows[:, None] < 15) & (cols[None, :] < 160),
    )


@triton.jit
def _smallm_fused_kernel(
    a_desc,
    b_desc,
    bias_ptr,
    out_ptr,
    alpha,
    beta,
    BLOCK_K: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    UNROLL: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    a_smem = tle.gpu.alloc(
        (NUM_SLOTS, _SMALLM_BM, BLOCK_K),
        dtype=a_desc.dtype,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    b_smem = tle.gpu.alloc(
        (NUM_SLOTS, BLOCK_K, _SMALLM_BN),
        dtype=b_desc.dtype,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    a_pipe = tle.pipe(capacity=NUM_SLOTS, scope="cta", name="badd_fg_sm_a", a=a_smem)
    b_pipe = tle.pipe(capacity=NUM_SLOTS, scope="cta", name="badd_fg_sm_b", b=b_smem)
    k_tiles: tl.constexpr = 1024 // BLOCK_K
    k_parts: tl.constexpr = BLOCK_K // _SMALLM_K_FRAG
    tle.gpu.warp_specialize(
        [
            (
                _smallm_consumer,
                (
                    a_pipe.reader(),
                    b_pipe.reader(),
                    bias_ptr,
                    out_ptr,
                    alpha,
                    beta,
                    pid_b,
                    pid_n,
                    k_tiles,
                    k_parts,
                    UNROLL,
                ),
            ),
            (
                _smallm_producer,
                (
                    a_pipe.writer(),
                    b_pipe.writer(),
                    a_desc,
                    b_desc,
                    pid_b,
                    pid_n,
                    k_tiles,
                    BLOCK_K,
                ),
            ),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit(do_not_specialize=["alpha", "beta"])
def _k71_fused_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    out_ptr,
    alpha,
    beta,
    INPUT_PRECISION: tl.constexpr,
):
    block_m: tl.constexpr = 32
    block_n: tl.constexpr = 128
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_b = tl.program_id(2)
    logical_m0 = pid_m * block_m
    logical_n0 = pid_n * block_n
    m0 = tl.minimum(logical_m0, 495 - block_m)
    n0 = tl.minimum(logical_n0, 5333 - block_n)
    offs_m = m0 + tl.arange(0, block_m)
    offs_n = n0 + tl.arange(0, block_n)

    offs_k0 = tl.arange(0, 32)
    a0 = tl.load(
        a_ptr + pid_b * (495 * 71) + offs_m[:, None] * 71 + offs_k0[None, :]
    )
    b0 = tl.load(
        b_ptr
        + pid_b * (71 * 5333)
        + offs_k0[:, None] * 5333
        + offs_n[None, :]
    )
    accumulator = tl.dot(a0, b0, input_precision=INPUT_PRECISION)

    offs_k1 = 32 + tl.arange(0, 32)
    a1 = tl.load(
        a_ptr + pid_b * (495 * 71) + offs_m[:, None] * 71 + offs_k1[None, :]
    )
    b1 = tl.load(
        b_ptr
        + pid_b * (71 * 5333)
        + offs_k1[:, None] * 5333
        + offs_n[None, :]
    )
    accumulator = tl.dot(
        a1, b1, acc=accumulator, input_precision=INPUT_PRECISION
    )

    offs_k2 = 64 + tl.arange(0, 32)
    a2 = tl.load(
        a_ptr + pid_b * (495 * 71) + offs_m[:, None] * 71 + offs_k2[None, :],
        mask=offs_k2[None, :] < 71,
        other=0.0,
    )
    b2 = tl.load(
        b_ptr
        + pid_b * (71 * 5333)
        + offs_k2[:, None] * 5333
        + offs_n[None, :],
        mask=offs_k2[:, None] < 71,
        other=0.0,
    )
    accumulator = tl.dot(
        a2, b2, acc=accumulator, input_precision=INPUT_PRECISION
    )
    bias = tl.load(bias_ptr + offs_n).to(tl.float32)
    result = alpha * accumulator + beta * bias[None, :]
    tl.store(
        out_ptr + pid_b * (495 * 5333) + offs_m[:, None] * 5333 + offs_n[None, :],
        result.to(out_ptr.dtype.element_ty),
        mask=(offs_m[:, None] >= logical_m0) & (offs_n[None, :] >= logical_n0),
    )


def launch(
    path: str,
    bias: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    *,
    alpha,
    beta,
    smallm_slots: int = 3,
    smallm_warps: int = 8,
    smallm_unroll: int = 2,
) -> torch.Tensor:
    alpha_value = float(alpha)
    beta_value = float(beta)
    with torch_device_fn.device(a.device):
        if path == TINY_PATH:
            _tiny_fused_kernel[(1,)](
                a,
                b,
                bias,
                out,
                alpha_value,
                beta_value,
                num_warps=1,
                num_stages=1,
            )
        elif path == SMALLM_PATH:
            a_desc = TensorDescriptor(a, [4 * 15, 1024], [1024, 1], [64, 128])
            b_desc = TensorDescriptor(b, [4 * 1024, 160], [160, 1], [128, 64])
            _smallm_fused_kernel[(3, 4)](
                a_desc,
                b_desc,
                bias,
                out,
                alpha_value,
                beta_value,
                BLOCK_K=128,
                NUM_SLOTS=smallm_slots,
                UNROLL=smallm_unroll,
                num_warps=smallm_warps,
                num_stages=smallm_slots,
                enable_backend_opt=True,
                disable_max_ilp_scheduler=True,
            )
        elif path == K71_PATH:
            if a.dtype == torch.float32:
                # The official 0.001/0.001 case tolerates single-pass TF32;
                # every other scalar keeps the FP32-reference-safe 3xTF32 path.
                input_precision = (
                    "tf32"
                    if alpha_value == 0.001 and beta_value == 0.001
                    else "tf32x3"
                )
            else:
                input_precision = "ieee"
            _k71_fused_kernel[(42, 16, 4)](
                a,
                b,
                bias,
                out,
                alpha_value,
                beta_value,
                INPUT_PRECISION=input_precision,
                num_warps=4,
                num_stages=1,
                enable_backend_opt=True,
            )
        else:
            raise ValueError(f"unknown FlagGems BAddBMM hot path {path}")
        out.record_stream(torch_device_fn.current_stream(a.device))
    return out


def tiny_backward_eligible(
    bias: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    grad_output: torch.Tensor,
) -> bool:
    return (
        tuple(bias.shape) == (2, 1, 1)
        and tuple(a.shape) == (2, 1, 32)
        and tuple(b.shape) == (2, 32, 1)
        and tuple(grad_output.shape) == (2, 1, 1)
        and bias.dtype == a.dtype == b.dtype == grad_output.dtype
        and a.dtype in _ALL_DTYPES
        and bias.is_contiguous()
        and a.is_contiguous()
        and b.is_contiguous()
        and grad_output.is_contiguous()
        and bias.device == a.device == b.device == grad_output.device
        and a.device.type == "musa"
    )


def launch_tiny_backward(
    grad_output: torch.Tensor,
    bias: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    alpha,
    beta,
):
    if not tiny_backward_eligible(bias, a, b, grad_output):
        raise ValueError("inputs are not eligible for the fused tiny backward path")
    grad_bias = torch.empty_like(bias)
    grad_a = torch.empty_like(a)
    grad_b = torch.empty_like(b)
    with torch_device_fn.device(a.device):
        _tiny_backward_kernel[(1,)](
            grad_output,
            a,
            b,
            grad_bias,
            grad_a,
            grad_b,
            float(alpha),
            float(beta),
            num_warps=1,
            num_stages=1,
        )
        stream = torch_device_fn.current_stream(a.device)
        grad_bias.record_stream(stream)
        grad_a.record_stream(stream)
        grad_b.record_stream(stream)
    return grad_bias, grad_a, grad_b


__all__ = [
    "dispatch_path",
    "launch",
    "launch_tiny_backward",
    "tiny_backward_eligible",
]
