# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PPU-specialized batched GEMM with an optional fused bias epilogue."""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.bmm import bmm as _generic_bmm
from flag_gems.ops.bmm import bmm_out as _generic_bmm_out
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

from ._matmul_utils import (
    EXPAND_CONFIG_FILENAME,
    HAS_PPU_TLE,
    _LOAD_A_AIU,
    _LOAD_B_AIU,
    _LOAD_BOTH_AIU,
    _LOAD_REGULAR,
    _SMALL_M_TILE,
    _configs_from_specs,
    _ppu_gemm_tile,
    _ppu_bucket_strategy,
    _prune_gemm_configs,
    _prune_split_k_configs,
    tle,
)

logger = logging.getLogger(__name__)


def _ppu_bmm_configs():
    """Bounded PPU search covering narrow, balanced, and wide batched GEMMs."""
    specs = (
        (32, 64, 32, 4, 2, 4, _LOAD_BOTH_AIU),
        # Expanded winner for batch-2 short-M projections.
        (32, 128, 64, 8, 5, 2, _LOAD_A_AIU),
        (64, 128, 64, 4, 2, 8, _LOAD_BOTH_AIU),
        (128, 128, 64, 8, 2, 8, _LOAD_BOTH_AIU),
        # Expanded winner for batch-1 long-K projections.
        (128, 128, 64, 8, 4, 4, _LOAD_A_AIU),
        (128, 256, 64, 8, 2, 8, _LOAD_BOTH_AIU),
        (256, 128, 64, 8, 2, 8, _LOAD_BOTH_AIU),
        (128, 64, 64, 8, 3, 8, _LOAD_BOTH_AIU),
        (64, 128, 64, 8, 3, 8, _LOAD_REGULAR),
        (64, 256, 64, 8, 3, 8, _LOAD_REGULAR),
        (64, 256, 32, 4, 4, 1, _LOAD_BOTH_AIU),
        (64, 256, 32, 4, 4, 8, _LOAD_BOTH_AIU),
        # Expanded FP16 fused-bias winner for wide-N, K=1024 projections.
        (64, 256, 32, 8, 4, 1, _LOAD_A_AIU),
        (128, 128, 32, 4, 5, 1, _LOAD_BOTH_AIU),
        (128, 128, 32, 4, 5, 8, _LOAD_REGULAR),
        (128, 256, 32, 8, 5, 4, _LOAD_REGULAR),
        (128, 256, 32, 8, 5, 8, _LOAD_REGULAR),
        (256, 128, 32, 8, 5, 1, _LOAD_REGULAR),
        (256, 128, 32, 8, 5, 8, _LOAD_REGULAR),
        (512, 64, 32, 8, 4, 1, _LOAD_REGULAR),
        (256, 64, 32, 8, 4, 1, _LOAD_REGULAR),
        (256, 64, 32, 8, 4, 4, _LOAD_BOTH_AIU),
        (256, 64, 64, 8, 3, 1, _LOAD_BOTH_AIU),
        (128, 64, 32, 4, 3, 1, _LOAD_REGULAR),
        (64, 512, 32, 8, 4, 2, _LOAD_REGULAR),
        (64, 512, 32, 8, 4, 8, _LOAD_REGULAR),
        (128, 64, 64, 4, 5, 1, _LOAD_BOTH_AIU),
    )
    return _configs_from_specs(
        specs,
        (
            "BLOCK_M",
            "BLOCK_N",
            "BLOCK_K",
            "num_warps",
            "num_stages",
            "GROUP_M",
            "LOAD_MODE",
        ),
    )


def _ppu_small_m_bmm_configs():
    """Configs for a fixed legal 16-row PPU dot tile."""
    specs = (
        (64, 32, 4, 4, _LOAD_REGULAR),
        (64, 32, 4, 4, _LOAD_B_AIU),
        (64, 64, 4, 3, _LOAD_BOTH_AIU),
        (64, 64, 4, 5, _LOAD_REGULAR),
        (128, 32, 4, 4, _LOAD_REGULAR),
        (128, 32, 4, 4, _LOAD_B_AIU),
        (128, 64, 4, 3, _LOAD_BOTH_AIU),
        (128, 64, 4, 3, _LOAD_A_AIU),
        (256, 32, 8, 4, _LOAD_REGULAR),
        (256, 32, 8, 4, _LOAD_B_AIU),
        (256, 64, 8, 3, _LOAD_BOTH_AIU),
        (256, 64, 8, 3, _LOAD_B_AIU),
        (512, 32, 8, 4, _LOAD_REGULAR),
        (512, 32, 8, 4, _LOAD_B_AIU),
        (512, 64, 8, 2, _LOAD_BOTH_AIU),
        (512, 64, 8, 2, _LOAD_B_AIU),
    )
    return _configs_from_specs(
        specs,
        ("BLOCK_N", "BLOCK_K", "num_warps", "num_stages", "LOAD_MODE"),
    )


def _ppu_split_k_bmm_configs():
    """Candidate set formerly selected by the batch/shape dispatch table."""
    specs = (
        (4, 32, 128, 32, 5, 4, _LOAD_REGULAR, False),
        (6, 64, 256, 32, 4, 4, _LOAD_B_AIU, True),
        (4, 64, 256, 32, 4, 4, _LOAD_B_AIU, True),
        (2, 128, 64, 64, 3, 4, _LOAD_BOTH_AIU, False),
    )
    return _configs_from_specs(
        specs,
        (
            "SPLIT_K",
            "BLOCK_M",
            "BLOCK_N",
            "BLOCK_K",
            "num_stages",
            "num_warps",
            "LOAD_MODE",
            "INTERLEAVED",
        ),
    )


def _ppu_split_k_bmm_reduce_configs():
    return [
        triton.Config({"BLOCK": block, "VEC": vec}, num_warps=warps, num_stages=1)
        for block, vec, warps in ((128, 8, 4), (256, 4, 4))
    ]


if HAS_PPU_TLE:

    @libentry()
    @libtuner(
        configs=_ppu_bmm_configs(),
        key=["FUSE_BIAS", "batch", "M", "N", "K"],
        strategy=["default", _ppu_bucket_strategy, _ppu_bucket_strategy, _ppu_bucket_strategy, _ppu_bucket_strategy],
        prune_configs_by={"early_config_prune": _prune_gemm_configs},
        warmup=5,
        rep=10,
        flagtune_op_name="bmm",
        flagtune_expand_op_name="bmm_ppu",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    )
    @triton.jit(do_not_specialize=["alpha", "beta"])
    def bmm_kernel_ppu(
        A,
        B,
        C,
        Bias,
        alpha,
        beta,
        batch,
        M,
        N,
        K,
        stride_ab,
        stride_am,
        stride_ak,
        stride_bb,
        stride_bk,
        stride_bn,
        stride_cb,
        stride_cm,
        stride_cn,
        stride_bias_b,
        stride_bias_m,
        stride_bias_n,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        LOAD_MODE: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
        EVEN_K: tl.constexpr,
        EVEN_M: tl.constexpr,
        EVEN_N: tl.constexpr,
        FUSE_BIAS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pid_batch = tl.program_id(1)
        A += pid_batch * stride_ab
        B += pid_batch * stride_bb
        C += pid_batch * stride_cb
        Bias += pid_batch * stride_bias_b

        grid_m = tl.cdiv(M, BLOCK_M)
        grid_n = tl.cdiv(N, BLOCK_N)
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + pid % group_size
        pid_n = (pid % width) // group_size
        _ppu_gemm_tile(
            A,
            B,
            C,
            Bias,
            alpha,
            beta,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            stride_bias_m,
            stride_bias_n,
            pid_m,
            pid_n,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            LOAD_MODE,
            PIPE_STAGES,
            False,
            EVEN_K,
            EVEN_M,
            EVEN_N,
            FUSE_BIAS,
        )


    @libentry()
    @libtuner(
        configs=_ppu_small_m_bmm_configs(),
        key=["FUSE_BIAS", "batch", "M", "N", "K"],
        strategy=["default", _ppu_bucket_strategy, _ppu_bucket_strategy, _ppu_bucket_strategy, _ppu_bucket_strategy],
        prune_configs_by={"early_config_prune": _prune_gemm_configs},
        warmup=5,
        rep=10,
        flagtune_op_name="bmm",
        flagtune_expand_op_name="bmm_ppu_small_m",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    )
    @triton.jit(do_not_specialize=["alpha", "beta"])
    def bmm_small_m_kernel_ppu(
        A,
        B,
        C,
        Bias,
        alpha,
        beta,
        batch,
        M,
        N,
        K,
        stride_ab,
        stride_am,
        stride_ak,
        stride_bb,
        stride_bk,
        stride_bn,
        stride_cb,
        stride_cm,
        stride_cn,
        stride_bias_b,
        stride_bias_m,
        stride_bias_n,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        LOAD_MODE: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
        EVEN_K: tl.constexpr,
        EVEN_N: tl.constexpr,
        FUSE_BIAS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pid_batch = tl.program_id(1)
        A += pid_batch * stride_ab
        B += pid_batch * stride_bb
        C += pid_batch * stride_cb
        Bias += pid_batch * stride_bias_b

        grid_n = tl.cdiv(N, BLOCK_N)
        pid_m = pid // grid_n
        pid_n = pid % grid_n
        _ppu_gemm_tile(
            A,
            B,
            C,
            Bias,
            alpha,
            beta,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            stride_bias_m,
            stride_bias_n,
            pid_m,
            pid_n,
            16,
            BLOCK_N,
            BLOCK_K,
            LOAD_MODE,
            PIPE_STAGES,
            True,
            EVEN_K,
            False,
            EVEN_N,
            FUSE_BIAS,
        )


    # Keep FP32 workspace reduction here. Direct global-output atomics retained
    # only 21-72% of this path's throughput and BF16 partial sums lost accuracy.
    @libtuner(
        configs=_ppu_split_k_bmm_configs(),
        key=["batch", "M", "N", "K"],
        strategy=[_ppu_bucket_strategy] * 4,
        prune_configs_by={"early_config_prune": _prune_split_k_configs},
        warmup=5,
        rep=10,
        flagtune_op_name="bmm",
        flagtune_expand_op_name="bmm_ppu_split_k",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    )
    @triton.jit
    def bmm_split_k_kernel_ppu(
        A,
        B,
        Workspace,
        M,
        N,
        K,
        batch,
        stride_ab,
        stride_am,
        stride_ak,
        stride_bb,
        stride_bk,
        stride_bn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SPLIT_K: tl.constexpr,
        INTERLEAVED: tl.constexpr,
        LOAD_MODE: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
        EVEN_MN: tl.constexpr,
    ):
        tile_id = tl.program_id(0)
        split_id = tl.program_id(1)
        batch_id = tl.program_id(2)
        A += batch_id * stride_ab
        B += batch_id * stride_bb

        grid_n = tl.cdiv(N, BLOCK_N)
        pid_m = tile_id // grid_n
        pid_n = tile_id % grid_n
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        wrapped_m = tl.max_contiguous(
            tl.multiple_of(offs_m % M, BLOCK_M), BLOCK_M
        ).to(tl.int64)
        wrapped_n = tl.max_contiguous(
            tl.multiple_of(offs_n % N, BLOCK_N), BLOCK_N
        ).to(tl.int64)

        k_per_split = K // SPLIT_K
        if INTERLEAVED:
            k_per_split = tl.cdiv(K, BLOCK_K * SPLIT_K) * BLOCK_K
            k_begin = split_id * BLOCK_K
            k_advance = BLOCK_K * SPLIT_K
        else:
            k_begin = split_id * k_per_split
            k_advance = BLOCK_K
        a_block_ptr = tl.make_block_ptr(
            base=A,
            shape=(M, K),
            strides=(stride_am, stride_ak),
            offsets=(pid_m * BLOCK_M, k_begin),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        b_block_ptr = tl.make_block_ptr(
            base=B,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            offsets=(k_begin, pid_n * BLOCK_N),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(1, 0),
        )
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_offset in tl.range(
            0, k_per_split, BLOCK_K, num_stages=PIPE_STAGES
        ):
            if INTERLEAVED:
                offs_k = (
                    k_begin
                    + k_offset * SPLIT_K
                    + tl.arange(0, BLOCK_K)
                )
            else:
                offs_k = k_begin + k_offset + tl.arange(0, BLOCK_K)
            if LOAD_MODE == 0 or LOAD_MODE == 1:
                a = tle.load(
                    a_block_ptr,
                    boundary_check=(0, 1),
                    padding_option="zero",
                    is_async=True,
                )
            else:
                a_ptrs = (
                    A
                    + wrapped_m[:, None] * stride_am
                    + offs_k[None, :] * stride_ak
                )
                if INTERLEAVED:
                    a = tl.load(
                        a_ptrs, mask=offs_k[None, :] < K, other=0.0
                    )
                else:
                    a = tl.load(a_ptrs, mask=offs_k[None, :] < K, other=0.0)
            if LOAD_MODE == 0 or LOAD_MODE == 2:
                b = tle.load(
                    b_block_ptr,
                    boundary_check=(0, 1),
                    padding_option="zero",
                    is_async=True,
                )
            else:
                b_ptrs = (
                    B
                    + offs_k[:, None] * stride_bk
                    + wrapped_n[None, :] * stride_bn
                )
                if INTERLEAVED:
                    b = tl.load(
                        b_ptrs, mask=offs_k[:, None] < K, other=0.0
                    )
                else:
                    b = tl.load(b_ptrs, mask=offs_k[:, None] < K, other=0.0)
            acc = tl.dot(a, b, acc=acc, out_dtype=tl.float32)
            a_block_ptr = tl.advance(a_block_ptr, (0, k_advance))
            b_block_ptr = tl.advance(b_block_ptr, (k_advance, 0))

        matrix_elements = M * N
        workspace_ptrs = (
            Workspace
            + batch_id * SPLIT_K * matrix_elements
            + split_id * matrix_elements
            + offs_m[:, None] * N
            + offs_n[None, :]
        )
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        if EVEN_MN:
            tl.store(workspace_ptrs, acc)
        else:
            tl.store(workspace_ptrs, acc, mask=mask)


    @libtuner(
        configs=_ppu_split_k_bmm_reduce_configs(),
        key=["M", "N", "total_elements"],
        strategy=[_ppu_bucket_strategy] * 3,
        warmup=5,
        rep=10,
        flagtune_op_name="bmm",
        flagtune_expand_op_name="bmm_ppu_split_k_reduce",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    )
    @triton.jit(do_not_specialize=["alpha", "beta"])
    def bmm_split_k_reduce_kernel_ppu(
        Workspace,
        C,
        Bias,
        alpha,
        beta,
        M,
        N,
        stride_cb,
        stride_cm,
        stride_cn,
        stride_bias_b,
        stride_bias_m,
        stride_bias_n,
        total_elements,
        SPLIT_K: tl.constexpr,
        BLOCK: tl.constexpr,
        VEC: tl.constexpr,
        EVEN_N: tl.constexpr,
        FUSE_BIAS: tl.constexpr,
    ):
        offsets = (
            tl.program_id(0) * BLOCK * VEC
            + tl.arange(0, BLOCK)[:, None] * VEC
            + tl.arange(0, VEC)[None, :]
        )
        offsets = tl.max_contiguous(offsets, (1, VEC))
        mask = offsets < total_elements
        matrix_elements = M * N
        batch_ids = offsets // matrix_elements
        matrix_offsets = offsets % matrix_elements
        workspace_ptrs = (
            Workspace
            + batch_ids * SPLIT_K * matrix_elements
            + matrix_offsets
        )
        acc = tl.zeros((BLOCK, VEC), dtype=tl.float32)
        for _ in range(SPLIT_K):
            if EVEN_N:
                acc += tl.load(workspace_ptrs)
            else:
                acc += tl.load(workspace_ptrs, mask=mask, other=0.0)
            workspace_ptrs += matrix_elements

        offs_m = matrix_offsets // N
        offs_n = matrix_offsets % N
        c_ptrs = (
            C
            + batch_ids * stride_cb
            + offs_m * stride_cm
            + offs_n * stride_cn
        )
        acc *= alpha
        if FUSE_BIAS:
            bias_ptrs = (
                Bias
                + batch_ids * stride_bias_b
                + offs_m * stride_bias_m
                + offs_n * stride_bias_n
            )
            bias_value = tl.load(bias_ptrs, mask=mask, other=0.0)
            acc += beta * bias_value
        if EVEN_N:
            tl.store(c_ptrs, acc.to(C.dtype.element_ty))
        else:
            tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=mask)


def _can_use_ppu_bmm_inputs(A: torch.Tensor, B: torch.Tensor) -> bool:
    if not (
        HAS_PPU_TLE
        and A.ndim == B.ndim == 3
        and A.dtype in (torch.float16, torch.bfloat16)
        and B.dtype == A.dtype
        and A.device == B.device
        and A.is_contiguous()
        and B.is_contiguous()
    ):
        return False

    batch, M, K = A.shape
    batch_b, b_k, N = B.shape
    return (
        batch == batch_b
        and batch > 0
        and M > 0
        and K == b_k
        and (N in (8, 16, 32, 64) or N >= 128)
        and K >= 32
        and A.data_ptr() % 32 == 0
        and B.data_ptr() % 32 == 0
    )


def _can_use_ppu_bmm(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> bool:
    return (
        _can_use_ppu_bmm_inputs(A, B)
        and out.ndim == 3
        and out.shape == (A.shape[0], A.shape[1], B.shape[2])
        and out.dtype == A.dtype
        and out.device == A.device
        and out.is_contiguous()
        and out.data_ptr() % 32 == 0
    )


def _should_use_split_k_bmm(batch: int, M: int, N: int, K: int) -> bool:
    """Select split-K only for the measured long-K batch regimes."""
    if N == 512 and K == 4096 and M == 144 and batch == 1:
        return True
    if N == 512 and K == 4096 and M == 384:
        # At batch=2/4, the ordinary batched grid already fills the PPU and is
        # faster than paying for a workspace and reduction launch.  A single
        # matrix is K-parallelism limited, while batch>=8 amortizes reduction
        # and still benefits from the extra K slices.
        if batch == 1:
            return True
        if batch >= 8:
            return True
    if N == 64 and K == 2048 and M == 16384 and batch <= 2:
        return True
    return False


def _run_ppu_split_k_bmm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    bias: torch.Tensor,
    alpha,
    beta,
) -> torch.Tensor:
    batch, M, K = A.shape
    _, _, N = B.shape
    matrix_elements = M * N
    total_elements = batch * matrix_elements
    fuse_bias = bias is not out
    # Expanded mode includes SPLIT_K=8; reserve that capacity even though the
    # default Pareto shortlist tops out at six slices.
    max_split_k = max(
        8,
        max(config.kwargs["SPLIT_K"] for config in _ppu_split_k_bmm_configs()),
    )
    workspace = torch.empty(
        (batch, max_split_k, M, N), device=out.device, dtype=torch.float32
    )
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        META["SPLIT_K"],
        batch,
    )
    with torch_device_fn.device(A.device):
        bmm_split_k_kernel_ppu[grid](
            A,
            B,
            workspace,
            M,
            N,
            K,
            batch,
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            EVEN_MN=False,
        )
        split_k = bmm_split_k_kernel_ppu.best_config.kwargs["SPLIT_K"]
        reduce_grid = lambda META: (
            triton.cdiv(total_elements, META["BLOCK"] * META["VEC"]),
        )
        bmm_split_k_reduce_kernel_ppu[
            reduce_grid
        ](
            workspace,
            out,
            bias,
            alpha,
            beta,
            M,
            N,
            out.stride(0),
            out.stride(1),
            out.stride(2),
            bias.stride(0),
            bias.stride(1),
            bias.stride(2),
            total_elements,
            SPLIT_K=split_k,
            FUSE_BIAS=fuse_bias,
            EVEN_N=False,
        )
    return out


def _run_ppu_bmm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    alpha=1.0,
    beta=0.0,
) -> torch.Tensor:
    batch, M, K = A.shape
    _, _, N = B.shape
    fuse_bias = bias is not None
    expanded_bias = bias.broadcast_to((batch, M, N)) if fuse_bias else out
    if _should_use_split_k_bmm(batch, M, N, K):
        return _run_ppu_split_k_bmm(
            A,
            B,
            out,
            expanded_bias,
            alpha,
            beta,
        )
    kernel = bmm_small_m_kernel_ppu if M < 32 else bmm_kernel_ppu

    def grid(meta):
        block_m = _SMALL_M_TILE if M < 32 else meta["BLOCK_M"]
        return (
            triton.cdiv(M, block_m) * triton.cdiv(N, meta["BLOCK_N"]),
            batch,
        )

    with torch_device_fn.device(A.device):
        kernel[grid](
            A,
            B,
            out,
            expanded_bias,
            alpha,
            beta,
            batch,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            A.stride(2),
            B.stride(0),
            B.stride(1),
            B.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            expanded_bias.stride(0),
            expanded_bias.stride(1),
            expanded_bias.stride(2),
            EVEN_K=K % 64 == 0,
            EVEN_N=N % 512 == 0,
            FUSE_BIAS=fuse_bias,
            **(
                {"EVEN_M": M % 512 == 0}
                if M >= 32
                else {}
            ),
        )
    return out


def bmm(A, B):
    logger.debug("GEMS_THEAD BMM")
    if A.ndim == B.ndim == 3 and A.shape[0] == B.shape[0] and A.shape[2] == B.shape[1]:
        out = torch.empty(
            (A.shape[0], A.shape[1], B.shape[2]),
            dtype=A.dtype,
            device=A.device,
        )
        if _can_use_ppu_bmm(A, B, out):
            return _run_ppu_bmm(A, B, out)
    return _generic_bmm(A, B)


def bmm_out(A, B, out):
    logger.debug("GEMS_THEAD BMM_OUT")
    if _can_use_ppu_bmm(A, B, out):
        return _run_ppu_bmm(A, B, out)
    return _generic_bmm_out(A, B, out)


__all__ = ["bmm", "bmm_out"]
