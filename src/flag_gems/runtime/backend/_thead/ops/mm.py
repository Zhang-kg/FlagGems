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

"""PPU-specialized matrix multiplication.

The tuner compares a no-wrap software-pipelined kernel with FlagTree TLE AIU
and mixed-load variants. Unsupported layouts retain the generic FlagGems
implementation; partial M/N/K tiles are handled in the PPU kernels.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.mm import mm as _generic_mm
from flag_gems.ops.mm import mm_out as _generic_mm_out
from flag_gems.ops.mv import mv as _generic_mv
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

def _ppu_mm_configs():
    """A bounded search space covering PPU latency and throughput regimes."""

    # Retained Pareto set from the broad PPU search.  Tuple order is
    # BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, GROUP_M, LOAD_MODE.
    pareto_specs = (
        (32, 64, 32, 4, 2, 4, _LOAD_BOTH_AIU),
        (64, 128, 64, 4, 2, 8, _LOAD_BOTH_AIU),
        (64, 128, 64, 4, 2, 8, _LOAD_A_AIU),
        (64, 128, 64, 4, 2, 8, _LOAD_B_AIU),
        (32, 128, 64, 4, 3, 4, _LOAD_REGULAR),
        (32, 256, 32, 8, 4, 4, _LOAD_REGULAR),
        (64, 128, 64, 4, 3, 8, _LOAD_REGULAR),
        (64, 256, 64, 8, 3, 8, _LOAD_REGULAR),
        (64, 512, 32, 8, 4, 2, _LOAD_REGULAR),
        (64, 512, 32, 8, 4, 8, _LOAD_REGULAR),
        (64, 512, 32, 8, 4, 16, _LOAD_REGULAR),
        (128, 256, 32, 8, 4, 8, _LOAD_REGULAR),
        (64, 16, 32, 4, 4, 8, _LOAD_REGULAR),
        (128, 16, 32, 4, 4, 8, _LOAD_REGULAR),
        (256, 16, 32, 8, 4, 8, _LOAD_REGULAR),
        (64, 8, 32, 4, 4, 8, _LOAD_REGULAR),
        (128, 8, 32, 4, 4, 8, _LOAD_REGULAR),
        (256, 8, 32, 8, 4, 8, _LOAD_REGULAR),
        (64, 32, 32, 4, 4, 8, _LOAD_REGULAR),
        (128, 32, 32, 4, 4, 8, _LOAD_REGULAR),
        (256, 32, 32, 8, 4, 8, _LOAD_REGULAR),
    )
    # Winners and near-winners from the broad and second-order sweeps.  These
    # cover deep-pipeline balanced GEMMs, long-K/narrow-N GEMMs, small-K/wide-N
    # GEMMs, and launch-bound small-M GEMMs without retaining the exploration
    # space in production autotuning.
    search_specs = (
        # Actlize's PPU0010 AIU shortlist.  ZW810E is a PPU0010 target; these
        # configurations use the vendor library's native K=64 granularity and
        # much shallower two/three-stage pipeline.
        (256, 128, 64, 8, 2, 8, _LOAD_BOTH_AIU),
        (256, 128, 64, 8, 2, 8, _LOAD_REGULAR),
        (128, 256, 64, 8, 2, 8, _LOAD_BOTH_AIU),
        (128, 256, 64, 8, 2, 8, _LOAD_REGULAR),
        (64, 256, 64, 8, 2, 8, _LOAD_BOTH_AIU),
        (64, 256, 64, 8, 2, 8, _LOAD_REGULAR),
        (128, 128, 64, 8, 2, 8, _LOAD_BOTH_AIU),
        (128, 128, 64, 8, 2, 8, _LOAD_REGULAR),
        (128, 64, 64, 8, 3, 8, _LOAD_BOTH_AIU),
        (128, 64, 64, 8, 3, 8, _LOAD_REGULAR),
        (64, 128, 64, 8, 3, 8, _LOAD_BOTH_AIU),
        (64, 128, 64, 8, 3, 8, _LOAD_REGULAR),
        # BK=32 reduces the live dot-operand footprint enough for the AIU path
        # to outperform both K=64 AIU and regular loads on large PPU GEMMs.
        (64, 256, 32, 4, 4, 1, _LOAD_BOTH_AIU),
        (64, 256, 32, 4, 4, 4, _LOAD_BOTH_AIU),
        (64, 256, 32, 4, 4, 8, _LOAD_BOTH_AIU),
        (128, 128, 32, 4, 5, 8, _LOAD_REGULAR),
        (64, 256, 32, 4, 4, 2, _LOAD_REGULAR),
        (64, 256, 32, 4, 4, 8, _LOAD_REGULAR),
        (128, 256, 32, 8, 5, 4, _LOAD_REGULAR),
        (128, 256, 32, 8, 5, 8, _LOAD_REGULAR),
        (256, 128, 32, 8, 5, 1, _LOAD_REGULAR),
        (256, 128, 32, 8, 5, 8, _LOAD_REGULAR),
        (512, 64, 32, 8, 4, 1, _LOAD_REGULAR),
        (16, 64, 32, 4, 5, 8, _LOAD_REGULAR),
        (16, 128, 32, 4, 5, 8, _LOAD_REGULAR),
        (32, 64, 32, 4, 5, 8, _LOAD_REGULAR),
    )

    # Winners for N=64 from the full Qwen3.5 expanded sweep.  Narrow N benefits
    # from tall-M/short-N AIU tiles at large M and 16-row tiles at small M.
    narrow_n_specs = (
        (16, 32, 64, 4, 5, 2, _LOAD_REGULAR),
        (16, 32, 64, 4, 5, 4, _LOAD_REGULAR),
        (16, 32, 64, 4, 5, 8, _LOAD_REGULAR),
        (16, 64, 64, 4, 5, 1, _LOAD_REGULAR),
        (16, 64, 64, 4, 5, 2, _LOAD_BOTH_AIU),
        (16, 64, 64, 4, 5, 4, _LOAD_BOTH_AIU),
        (16, 64, 64, 4, 5, 4, _LOAD_REGULAR),
        (16, 64, 64, 4, 5, 8, _LOAD_BOTH_AIU),
        (16, 64, 64, 4, 5, 8, _LOAD_REGULAR),
        (32, 16, 64, 4, 5, 1, _LOAD_REGULAR),
        (32, 16, 64, 4, 5, 4, _LOAD_REGULAR),
        (32, 64, 64, 4, 5, 8, _LOAD_REGULAR),
        (32, 64, 64, 8, 5, 4, _LOAD_BOTH_AIU),
        (64, 16, 64, 4, 5, 1, _LOAD_BOTH_AIU),
        (64, 16, 64, 4, 5, 2, _LOAD_BOTH_AIU),
        (64, 16, 64, 4, 5, 4, _LOAD_BOTH_AIU),
        (64, 16, 64, 4, 5, 8, _LOAD_BOTH_AIU),
        (64, 64, 64, 4, 5, 2, _LOAD_REGULAR),
        (128, 64, 32, 4, 3, 1, _LOAD_REGULAR),
        (128, 64, 64, 4, 5, 1, _LOAD_BOTH_AIU),
        (128, 64, 64, 4, 5, 2, _LOAD_BOTH_AIU),
        (128, 64, 64, 4, 5, 8, _LOAD_BOTH_AIU),
    )

    return _configs_from_specs(
        (*pareto_specs, *search_specs, *narrow_n_specs),
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


def _ppu_small_m_configs():
    """Production shortlist for 2 <= M < 32 with a fixed legal M tile."""

    # Tuple order is BLOCK_N, BLOCK_K, num_warps, num_stages, LOAD_MODE.
    # Keeping the physical M tile fixed at 16 prevents AABS from shrinking a
    # dot dimension below the PPU AIU compiler's legal minimum.
    specs = (
        (64, 32, 4, 4, _LOAD_REGULAR),
        (64, 32, 4, 4, _LOAD_B_AIU),
        (64, 64, 4, 3, _LOAD_BOTH_AIU),
        (64, 64, 4, 3, _LOAD_B_AIU),
        (64, 64, 4, 5, _LOAD_BOTH_AIU),
        (64, 64, 4, 5, _LOAD_B_AIU),
        (64, 64, 4, 5, _LOAD_REGULAR),
        (128, 32, 4, 4, _LOAD_REGULAR),
        (128, 32, 4, 4, _LOAD_B_AIU),
        (128, 64, 4, 3, _LOAD_BOTH_AIU),
        (128, 64, 4, 3, _LOAD_B_AIU),
        (256, 32, 4, 4, _LOAD_REGULAR),
        (256, 32, 4, 4, _LOAD_B_AIU),
        (256, 32, 8, 3, _LOAD_REGULAR),
        (256, 64, 8, 3, _LOAD_BOTH_AIU),
        (256, 64, 8, 3, _LOAD_B_AIU),
        (512, 32, 8, 4, _LOAD_REGULAR),
        (512, 32, 8, 4, _LOAD_B_AIU),
        (512, 64, 8, 2, _LOAD_BOTH_AIU),
        (512, 64, 8, 2, _LOAD_B_AIU),
    )
    expanded_winners = (
        (64, 64, 4, 4, _LOAD_BOTH_AIU),
        (64, 64, 4, 5, _LOAD_A_AIU),
        (128, 32, 4, 5, _LOAD_REGULAR),
        (128, 64, 4, 3, _LOAD_REGULAR),
        (128, 64, 8, 3, _LOAD_A_AIU),
        (128, 64, 8, 5, _LOAD_A_AIU),
        (256, 32, 8, 4, _LOAD_REGULAR),
    )
    return _configs_from_specs(
        (*specs, *expanded_winners),
        ("BLOCK_N", "BLOCK_K", "num_warps", "num_stages", "LOAD_MODE"),
    )


def _ppu_mid_m_configs():
    """Production configs for 16 < M < 32 with one fixed 32-row tile."""

    return [
        triton.Config(
            {"BLOCK_M": 32, **config.kwargs},
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
        for config in _ppu_small_m_configs()
    ]


def _ppu_gemv_configs():
    """A PPU reduction search space spanning latency and throughput regimes."""

    # Tuple order is output rows, reduction width, warps, stages.  Wide
    # reduction tiles remove Python/Triton loop overhead for latency-bound
    # decode shapes; multi-row tiles amortize vector loads for larger M.
    specs = (
        (1, 256, 4, 3),
        (1, 512, 4, 3),
        (1, 1024, 4, 3),
        (1, 2048, 8, 3),
        (1, 4096, 8, 3),
        (2, 256, 4, 3),
        (2, 512, 4, 3),
        (2, 1024, 8, 3),
        (2, 2048, 8, 3),
        (4, 256, 4, 3),
        (4, 512, 8, 3),
        (4, 1024, 8, 3),
        (8, 128, 4, 3),
        (8, 256, 8, 3),
        (8, 512, 8, 3),
        (16, 128, 8, 3),
        (16, 256, 8, 3),
    )
    return _configs_from_specs(
        specs,
        ("BLOCK_M", "BLOCK_K", "num_warps", "num_stages"),
    )


def _ppu_split_k_configs():
    """Candidate set formerly selected by the shape dispatch table."""
    specs = (
        (7, 16, 128, 64, 5, 4, _LOAD_BOTH_AIU, False),
        (7, 32, 128, 64, 5, 4, _LOAD_BOTH_AIU, False),
        (4, 64, 128, 64, 5, 4, _LOAD_BOTH_AIU, False),
        (2, 32, 128, 64, 5, 4, _LOAD_BOTH_AIU, False),
        (2, 64, 128, 64, 5, 4, _LOAD_BOTH_AIU, False),
        (4, 128, 128, 32, 5, 4, _LOAD_BOTH_AIU, True),
        (6, 128, 128, 32, 5, 4, _LOAD_BOTH_AIU, True),
        (5, 128, 128, 32, 5, 4, _LOAD_BOTH_AIU, True),
        (2, 128, 128, 32, 5, 4, _LOAD_BOTH_AIU, False),
        (8, 16, 64, 32, 5, 4, _LOAD_REGULAR, False),
        (2, 32, 128, 32, 5, 4, _LOAD_REGULAR, False),
        (6, 64, 256, 32, 4, 4, _LOAD_B_AIU, True),
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


def _ppu_split_k_reduce_configs():
    return [
        triton.Config({"BLOCK": block, "VEC": vec}, num_warps=warps, num_stages=1)
        for block, vec, warps in ((128, 8, 4), (256, 4, 4))
    ]


if HAS_PPU_TLE:

    @libentry()
    @libtuner(
        configs=_ppu_gemv_configs(),
        key=["M", "K"],
        strategy=["default", "default"],
        warmup=5,
        rep=20,
        flagtune_op_name="mm",
        flagtune_expand_op_name="gemv_ppu",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    )
    @triton.jit
    def gemv_kernel_ppu(
        A,
        X,
        Y,
        M,
        K,
        stride_am,
        stride_ak,
        stride_xk,
        stride_ym,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_m = rows[:, None]
        offs_k = tl.arange(0, BLOCK_K)[None, :]
        a_ptrs = A + offs_m * stride_am + offs_k * stride_ak
        x_ptrs = X + offs_k * stride_xk
        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

        for k_start in tl.range(0, K, BLOCK_K, num_stages=PIPE_STAGES):
            mask = (offs_m < M) & (k_start + offs_k < K)
            a = tl.load(a_ptrs, mask=mask, other=0.0).to(tl.float32)
            x = tl.load(
                x_ptrs,
                mask=k_start + offs_k < K,
                other=0.0,
            ).to(tl.float32)
            acc += a * x
            a_ptrs += BLOCK_K * stride_ak
            x_ptrs += BLOCK_K * stride_xk

        result = tl.sum(acc, axis=1)
        tl.store(
            Y + rows * stride_ym,
            result.to(Y.dtype.element_ty),
            mask=rows < M,
        )

    @libentry()
    @libtuner(
        configs=_ppu_mm_configs(),
        key=["FUSE_ADDMM", "M", "N", "K"],
        strategy=["default", _ppu_bucket_strategy, _ppu_bucket_strategy, _ppu_bucket_strategy],
        prune_configs_by={"early_config_prune": _prune_gemm_configs},
        warmup=5,
        rep=10,
        flagtune_op_name="mm",
        flagtune_expand_op_name="mm_ppu",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    )
    @triton.jit
    def mm_kernel_ppu(
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
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        LOAD_MODE: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
        EVEN_K: tl.constexpr,
        EVEN_M: tl.constexpr,
        EVEN_N: tl.constexpr,
        FUSE_ADDMM: tl.constexpr,
    ):
        pid = tl.program_id(0)
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
            FUSE_ADDMM,
        )


    @libentry()
    @libtuner(
        configs=_ppu_small_m_configs(),
        key=["FUSE_ADDMM", "M", "N", "K"],
        strategy=["default", _ppu_bucket_strategy, _ppu_bucket_strategy, _ppu_bucket_strategy],
        prune_configs_by={"early_config_prune": _prune_gemm_configs},
        warmup=5,
        rep=10,
        flagtune_op_name="mm",
        flagtune_expand_op_name="mm_ppu_small_m",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    )
    @triton.jit
    def mm_small_m_kernel_ppu(
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
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        LOAD_MODE: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
        EVEN_K: tl.constexpr,
        EVEN_N: tl.constexpr,
        FUSE_ADDMM: tl.constexpr,
    ):
        # M is deliberately represented by a literal physical tile.  AABS may
        # shrink tunable block sizes to the runtime tensor extent; for M < 16
        # that creates an illegal PPU dot tile.  Boundary padding and the final
        # store mask preserve the logical M without changing the physical dot.
        pid = tl.program_id(0)
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
            FUSE_ADDMM,
        )


    @libentry()
    @libtuner(
        configs=_ppu_mid_m_configs(),
        key=["FUSE_ADDMM", "M", "N", "K"],
        strategy=["default", _ppu_bucket_strategy, _ppu_bucket_strategy, _ppu_bucket_strategy],
        prune_configs_by={"early_config_prune": _prune_gemm_configs},
        warmup=5,
        rep=10,
        flagtune_op_name="mm",
        flagtune_expand_op_name="mm_ppu_mid_m",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    )
    @triton.jit
    def mm_mid_m_kernel_ppu(
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
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        LOAD_MODE: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
        EVEN_K: tl.constexpr,
        EVEN_N: tl.constexpr,
        FUSE_ADDMM: tl.constexpr,
    ):
        # This route fixes BLOCK_M at the legal 32-row PPU dot tile.  It covers
        # every 16 < M < 32 input with one program, so a very wide B tile is
        # loaded only once.
        pid_n = tl.program_id(0)
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
            0,
            pid_n,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            LOAD_MODE,
            PIPE_STAGES,
            True,
            EVEN_K,
            False,
            EVEN_N,
            FUSE_ADDMM,
        )


    # PPU global-output atomics compile, but direct FP16 accumulation retained
    # only 20-77% of this workspace path's throughput in the Qwen split-K set.
    # BF16 also rounds every partial sum before the atomic and missed tolerance.
    @libtuner(
        configs=_ppu_split_k_configs(),
        key=["M", "N", "K"],
        strategy=[_ppu_bucket_strategy, _ppu_bucket_strategy, _ppu_bucket_strategy],
        prune_configs_by={"early_config_prune": _prune_split_k_configs},
        warmup=5,
        rep=10,
        flagtune_op_name="mm",
        flagtune_expand_op_name="mm_ppu_split_k",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    )
    @triton.jit
    def mm_split_k_kernel_ppu(
        A,
        B,
        Workspace,
        M,
        N,
        K,
        stride_am,
        stride_ak,
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
            # Actlize/acBLAS distribute K tiles round-robin.  The final slices
            # may own one fewer tile (for example 64 K-tiles over six slices),
            # so every slice executes the ceiling count and pads the final
            # out-of-range tile with zero.
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

        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        workspace_ptrs = (
            Workspace
            + split_id * M * N
            + offs_m[:, None] * N
            + offs_n[None, :]
        )
        if EVEN_MN:
            tl.store(workspace_ptrs, acc)
        else:
            tl.store(workspace_ptrs, acc, mask=mask)


    @libtuner(
        configs=_ppu_split_k_reduce_configs(),
        key=["n_elements"],
        strategy=["default"],
        warmup=5,
        rep=10,
        flagtune_op_name="mm",
        flagtune_expand_op_name="mm_ppu_split_k_reduce",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    )
    @triton.jit
    def mm_split_k_reduce_kernel_ppu(
        Workspace,
        C,
        n_elements,
        SPLIT_K: tl.constexpr,
        BLOCK: tl.constexpr,
        VEC: tl.constexpr,
        EVEN_N: tl.constexpr,
    ):
        offsets = (
            tl.program_id(0) * BLOCK * VEC
            + tl.arange(0, BLOCK)[:, None] * VEC
            + tl.arange(0, VEC)[None, :]
        )
        offsets = tl.max_contiguous(offsets, (1, VEC))
        mask = offsets < n_elements
        acc = tl.zeros((BLOCK, VEC), dtype=tl.float32)
        workspace_ptrs = Workspace + offsets
        for _ in range(SPLIT_K // 2):
            if EVEN_N:
                x0 = tl.load(workspace_ptrs)
                x1 = tl.load(workspace_ptrs + n_elements)
            else:
                x0 = tl.load(workspace_ptrs, mask=mask, other=0.0)
                x1 = tl.load(
                    workspace_ptrs + n_elements, mask=mask, other=0.0
                )
            acc += x0 + x1
            workspace_ptrs += 2 * n_elements
        if SPLIT_K % 2:
            if EVEN_N:
                acc += tl.load(workspace_ptrs)
            else:
                acc += tl.load(workspace_ptrs, mask=mask, other=0.0)
        if EVEN_N:
            tl.store(C + offsets, acc.to(C.dtype.element_ty))
        else:
            tl.store(C + offsets, acc.to(C.dtype.element_ty), mask=mask)


def _can_use_ppu_mm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> bool:
    if not (
        HAS_PPU_TLE
        and a.ndim == b.ndim == out.ndim == 2
        and a.dtype in (torch.float16, torch.bfloat16)
        and b.dtype == out.dtype == a.dtype
        and a.is_contiguous()
        and b.is_contiguous()
        and out.is_contiguous()
    ):
        return False

    M, K = a.shape
    b_k, N = b.shape
    # Keep true GEMV (M == 1) on its existing fallback, but admit skinny GEMMs.
    # The PPU kernel pads the final M/N tile on AIU loads, wraps regular-load
    # lanes, and masks stores, so M smaller than BLOCK_M is supported.  N=64 is
    # also a native production tile and must not be routed to the generic GEMM.
    # The PPU AIU copy path requires at least 32-byte rows. BLOCK_K is always a
    # legal physical dot tile; boundary-checked AIU loads and masked regular
    # loads zero-fill the final partial K tile.
    return (
        K == b_k
        and M >= 2
        and (N in (8, 16, 32, 64) or N >= 128)
        and K >= 32
        and a.data_ptr() % 32 == 0
        and b.data_ptr() % 32 == 0
        and out.data_ptr() % 32 == 0
    )


def _run_ppu_mm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(a.device):
        mm_kernel_ppu[grid](
            a,
            b,
            out,
            out,
            1.0,
            0.0,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(0),
            out.stride(1),
            EVEN_K=K % 64 == 0,
            # All production tiles are at most 512 in M/N.  This conservative
            # predicate lets aligned large GEMMs omit the output mask without
            # coupling launch code to the autotuner-selected tile.
            EVEN_M=M % 512 == 0,
            EVEN_N=N % 512 == 0,
            FUSE_ADDMM=False,
        )
    return out


def _run_ppu_addmm(
    bias: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    alpha,
    beta,
) -> torch.Tensor:
    """Run the PPU GEMM with an in-kernel affine bias epilogue."""
    M, K = a.shape
    _, N = b.shape
    if M < 32:
        return _run_small_m_ppu_mm(
            a,
            b,
            out,
            bias=bias,
            alpha=alpha,
            beta=beta,
        )
    expanded_bias = bias.broadcast_to((M, N))
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(a.device):
        mm_kernel_ppu[grid](
            a,
            b,
            out,
            expanded_bias,
            alpha,
            beta,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
            expanded_bias.stride(0),
            expanded_bias.stride(1),
            EVEN_K=K % 64 == 0,
            EVEN_M=M % 512 == 0,
            EVEN_N=N % 512 == 0,
            FUSE_ADDMM=True,
        )
    return out


def _run_small_m_ppu_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    alpha=1.0,
    beta=0.0,
) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    fuse_addmm = bias is not None
    expanded_bias = bias.broadcast_to((M, N)) if fuse_addmm else out
    kernel = mm_small_m_kernel_ppu if M <= _SMALL_M_TILE else mm_mid_m_kernel_ppu
    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)
    with torch_device_fn.device(a.device):
        kernel[grid](
            a,
            b,
            out,
            expanded_bias,
            alpha,
            beta,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
            expanded_bias.stride(0),
            expanded_bias.stride(1),
            EVEN_K=K % 64 == 0,
            EVEN_N=N % 512 == 0,
            FUSE_ADDMM=fuse_addmm,
        )
    return out


def _should_use_split_k_mm(M: int, N: int, K: int) -> bool:
    """Select split-K for known long-K wave regimes.

    Tile, pipeline, load-mode, and split-count values are selected by the
    split-K LibTuner candidates; this predicate only chooses the implementation
    family.
    """
    # Kimi K2.5's projection GEMMs use K=7168.  Timeline profiling shows that
    # acBLAS parallelizes K for these shapes, while the ordinary Triton route
    # leaves too few independent output tiles and serializes 112 BK=64 steps
    # in every program.  The measured PPU winners use fewer K slices than
    # acBLAS at small M because Triton's reduction kernel has a higher fixed
    # launch cost.  Wider/larger outputs switch to two or five slices so the
    # workspace traffic does not dominate.
    if K == 7168 and N in (384, 512):
        # Workspace allocation dominates the split-K launch in this middle
        # M band on PPU.  The regular kernel has the same four output waves
        # without the reduction overhead, so keep split-K for the smaller and
        # larger wave regimes where it provides real parallelism.
        if N == 512 and 400 <= M < 512:
            return False
        if M < 32:
            return True
        # M=32 already has a separately tuned eight-way route below.
        if 32 < M <= 64:
            return True
        if 64 < M <= 128:
            return True
        if 128 < M <= 256:
            # For N=512, eight 32-row M tiles times four N tiles times two
            # slices lands on a slow 64-program wave at M=232..256.  A 64-row
            # tile halves that grid and recovers the measured ~97% of acBLAS.
            return True
        if N == 512 and 256 < M <= 320:
            return True
        # N=384 keeps three N tiles.  Four K slices fill the device efficiently
        # through M=384; after the next M-grid boundary six slices are faster.
        if N == 384 and 257 <= M <= 384:
            return True
        if N == 384 and 432 <= M <= 512:
            return True
        # N=512 has a different grid-wave boundary: six slices win while there
        # are three M tiles, whereas five win again once there are four.
        if N == 512 and 336 <= M <= 384:
            return True
        if N == 512 and 464 <= M < 512:
            return True

    # For Kimi's wider gate projection, the output grid already supplies most
    # of the parallelism.  Two K slices with a larger M tile match the profiled
    # acBLAS schedule and remove the remaining long-K tail.
    if K == 7168 and N == 2112 and 200 <= M <= 256:
        return True

    if K % 256 != 0 or N > 512:
        return False
    if M == 32 and K >= 4096:
        return True
    # Torch PPU uses parallel split-K for both shapes.  Replaying its large AIU
    # tiles directly is slower in Triton because the backend does not reproduce
    # acBLAS's explicit shared-memory swizzle/register pipeline.  These smaller
    # tiles are the measured Triton winners around the vendor slice strategy.
    if M == 144 and N == 512 and K == 4096:
        return True
    if M == 384 and N == 512 and K == 4096:
        # Match acBLAS's six-way K parallelism.  Keeping A on the regular load
        # path and B on AIU avoids the small-A AIU setup cost while retaining
        # asynchronous reuse of the much wider B tile.
        return True
    return False


def _run_split_k_mm(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    # The tuner updates ``best_config`` inside the kernel launch.  It may still
    # contain the winner for a previous shape before this call, so allocate the
    # bounded maximum up front and read the active split count afterwards.
    max_split_k = max(
        config.kwargs["SPLIT_K"] for config in _ppu_split_k_configs()
    )
    workspace = torch.empty(
        (max_split_k, M, N), device=out.device, dtype=torch.float32
    )
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        META["SPLIT_K"],
    )
    with torch_device_fn.device(a.device):
        mm_split_k_kernel_ppu[grid](
            a,
            b,
            workspace,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            EVEN_MN=False,
        )
        split_k = mm_split_k_kernel_ppu.best_config.kwargs["SPLIT_K"]
        reduce_grid = lambda META: (
            triton.cdiv(M * N, META["BLOCK"] * META["VEC"]),
        )
        mm_split_k_reduce_kernel_ppu[
            reduce_grid
        ](
            workspace,
            out,
            M * N,
            SPLIT_K=split_k,
            EVEN_N=False,
        )
    return out


def mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_THEAD MM")
    if a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0] and a.dtype == b.dtype:
        out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
        if _can_use_ppu_mm(a, b, out):
            if _should_use_split_k_mm(a.shape[0], b.shape[1], a.shape[1]):
                return _run_split_k_mm(a, b, out)
            if a.shape[0] < 32:
                return _run_small_m_ppu_mm(a, b, out)
            return _run_ppu_mm(a, b, out)
    return _generic_mm(a, b)


def mm_out(a: torch.Tensor, b: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_THEAD MM_OUT")
    if _can_use_ppu_mm(a, b, out):
        if _should_use_split_k_mm(a.shape[0], b.shape[1], a.shape[1]):
            return _run_split_k_mm(a, b, out)
        if a.shape[0] < 32:
            return _run_small_m_ppu_mm(a, b, out)
        return _run_ppu_mm(a, b, out)
    return _generic_mm_out(a, b, out=out)


def _can_use_ppu_gemv(inp: torch.Tensor, vec: torch.Tensor) -> bool:
    return (
        HAS_PPU_TLE
        and inp.ndim == 2
        and vec.ndim == 1
        and inp.shape[1] == vec.shape[0]
        and inp.dtype in (torch.float16, torch.bfloat16)
        and vec.dtype == inp.dtype
        and inp.is_contiguous()
        and vec.is_contiguous()
        and inp.shape[0] > 0
        and inp.shape[1] >= 32
        and inp.shape[1] % 16 == 0
        and inp.data_ptr() % 32 == 0
        and vec.data_ptr() % 32 == 0
    )


def mv(inp: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_THEAD MV")
    if not _can_use_ppu_gemv(inp, vec):
        return _generic_mv(inp, vec)

    M, K = inp.shape
    out = torch.empty((M,), device=inp.device, dtype=inp.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
    with torch_device_fn.device(inp.device):
        gemv_kernel_ppu[grid](
            inp,
            vec,
            out,
            M,
            K,
            inp.stride(0),
            inp.stride(1),
            vec.stride(0),
            out.stride(0),
        )
    return out


__all__ = ["mm", "mm_out", "mv"]
