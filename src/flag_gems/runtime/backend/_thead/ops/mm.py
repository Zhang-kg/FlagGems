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
and mixed-load variants. Unsupported layouts and unaligned problems retain the
generic FlagGems implementation.
"""

import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems.ops.mm import mm as _generic_mm
from flag_gems.ops.mm import mm_out as _generic_mm_out
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

try:
    import triton.experimental.tle.language as tle

    HAS_PPU_TLE = True
except ImportError:
    tle = None
    HAS_PPU_TLE = False


logger = logging.getLogger(__name__)
EXPAND_CONFIG_FILENAME = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "mm_ppu_expand.yaml")
)

_LOAD_BOTH_AIU = 0
_LOAD_A_AIU = 1
_LOAD_B_AIU = 2
_LOAD_REGULAR = 3


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

    # Preserve ordering while removing candidates already in the Pareto set.
    specs = tuple(dict.fromkeys((*pareto_specs, *search_specs)))
    configs = [
        triton.Config(
            {
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "BLOCK_K": block_k,
                "GROUP_M": group_m,
                "LOAD_MODE": load_mode,
                "PIPE_STAGES": num_stages,
            },
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for (
            block_m,
            block_n,
            block_k,
            num_warps,
            num_stages,
            group_m,
            load_mode,
        ) in specs
    ]
    return configs


if HAS_PPU_TLE:

    @libentry()
    @libtuner(
        configs=_ppu_mm_configs(),
        key=["M", "N", "K"],
        strategy=["default", "default", "default"],
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
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        LOAD_MODE: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
        EVEN_K: tl.constexpr,
        EVEN_MN: tl.constexpr,
    ):
        pid = tl.program_id(0)
        grid_m = tl.cdiv(M, BLOCK_M)
        grid_n = tl.cdiv(N, BLOCK_N)

        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + pid % group_size
        pid_n = (pid % width) // group_size

        a_block_ptr = tl.make_block_ptr(
            base=A,
            shape=(M, K),
            strides=(stride_am, stride_ak),
            offsets=(pid_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        b_block_ptr = tl.make_block_ptr(
            base=B,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            offsets=(0, pid_n * BLOCK_N),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(1, 0),
        )
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        # For regular loads, wrap only the final M/N tile and mask the store.
        # This keeps every K-loop load in-bounds without carrying M/N masks
        # through the PPU dot pipeline.  It is the same boundary strategy used
        # by FlagGems' generic GEMM kernel.
        wrapped_m = tl.max_contiguous(
            tl.multiple_of(offs_m % M, BLOCK_M), BLOCK_M
        ).to(tl.int64)
        wrapped_n = tl.max_contiguous(
            tl.multiple_of(offs_n % N, BLOCK_N), BLOCK_N
        ).to(tl.int64)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # PPU requires an explicit staged range for the load/dot loop.  Keeping
        # PIPE_STAGES equal to the launch option makes the intended software
        # pipeline visible to both Triton IR and the PPU backend.
        for k_start in tl.range(0, K, BLOCK_K, num_stages=PIPE_STAGES):
            if LOAD_MODE == 0 or LOAD_MODE == 1:
                a = tle.load(
                    a_block_ptr,
                    boundary_check=(0, 1),
                    padding_option="zero",
                    is_async=True,
                )
            else:
                offs_k = k_start + tl.arange(0, BLOCK_K)
                a_ptrs = (
                    A
                    + wrapped_m[:, None] * stride_am
                    + offs_k[None, :] * stride_ak
                )
                if EVEN_K:
                    a = tl.load(a_ptrs)
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
                offs_k = k_start + tl.arange(0, BLOCK_K)
                b_ptrs = (
                    B
                    + offs_k[:, None] * stride_bk
                    + wrapped_n[None, :] * stride_bn
                )
                if EVEN_K:
                    b = tl.load(b_ptrs)
                else:
                    b = tl.load(b_ptrs, mask=offs_k[:, None] < K, other=0.0)

            acc = tl.dot(a, b, acc=acc, out_dtype=tl.float32)
            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

        c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        if EVEN_MN:
            tl.store(c_ptrs, acc.to(C.dtype.element_ty))
        else:
            tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=c_mask)


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
                    a = tl.load(a_ptrs)
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
                    b = tl.load(b_ptrs)
            acc = tl.dot(a, b, acc=acc, out_dtype=tl.float32)
            a_block_ptr = tl.advance(a_block_ptr, (0, k_advance))
            b_block_ptr = tl.advance(b_block_ptr, (k_advance, 0))

        workspace_ptrs = (
            Workspace
            + split_id * M * N
            + offs_m[:, None] * N
            + offs_n[None, :]
        )
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        if EVEN_MN:
            tl.store(workspace_ptrs, acc)
        else:
            tl.store(workspace_ptrs, acc, mask=mask)


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
        # Mirror acBLAS's reduction schedule: process two adjacent slices per
        # iteration and advance one base pointer.  This avoids rebuilding
        # split_id * n_elements for every vector load in the generated ISA.
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
    if not HAS_PPU_TLE:
        return False
    if a.ndim != 2 or b.ndim != 2 or out.ndim != 2:
        return False
    if a.dtype not in (torch.float16, torch.bfloat16) or b.dtype != a.dtype:
        return False
    if out.dtype != a.dtype:
        return False
    if not (a.is_contiguous() and b.is_contiguous() and out.is_contiguous()):
        return False

    M, K = a.shape
    b_k, N = b.shape
    if K != b_k or M < 32 or (N not in (8, 16, 32) and N < 128) or K < 32:
        return False

    # The PPU AIU copy path requires at least 32-byte rows.  Keeping K aligned
    # also prevents the compiler from shrinking BLOCK_K below the dot minimum.
    if K % 16 != 0:
        return False
    return (
        a.data_ptr() % 32 == 0
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
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
            EVEN_K=K % 64 == 0,
            # All production tiles are at most 512 in M/N.  This conservative
            # predicate lets aligned large GEMMs omit the output mask without
            # coupling launch code to the autotuner-selected tile.
            EVEN_MN=M % 512 == 0 and N % 512 == 0,
        )
    return out


def _split_k_params(
    M: int, N: int, K: int
) -> tuple[int, int, int, int, int, int, int, bool] | None:
    if K % 256 != 0 or N > 512:
        return None
    if M == 32 and K >= 4096:
        return 8, 16, 64, 32, 5, 4, _LOAD_REGULAR, False
    # Torch PPU uses parallel split-K for both shapes.  Replaying its large AIU
    # tiles directly is slower in Triton because the backend does not reproduce
    # acBLAS's explicit shared-memory swizzle/register pipeline.  These smaller
    # tiles are the measured Triton winners around the vendor slice strategy.
    if M == 144 and N == 512 and K == 4096:
        return 2, 32, 128, 32, 5, 4, _LOAD_REGULAR, False
    if M == 384 and N == 512 and K == 4096:
        # Match acBLAS's six-way K parallelism.  Keeping A on the regular load
        # path and B on AIU avoids the small-A AIU setup cost while retaining
        # asynchronous reuse of the much wider B tile.
        return 6, 64, 256, 32, 4, 4, _LOAD_B_AIU, True
    return None


def _run_split_k_mm(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    params = _split_k_params(M, N, K)
    assert params is not None
    (
        split_k,
        block_m,
        block_n,
        block_k,
        stages,
        warps,
        load_mode,
        interleaved,
    ) = params
    workspace = torch.empty(
        (split_k, M, N), device=out.device, dtype=torch.float32
    )
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n), split_k)
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
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            SPLIT_K=split_k,
            INTERLEAVED=interleaved,
            LOAD_MODE=load_mode,
            PIPE_STAGES=stages,
            EVEN_MN=M % block_m == 0 and N % block_n == 0,
            num_warps=warps,
            num_stages=stages,
        )
        mm_split_k_reduce_kernel_ppu[(triton.cdiv(M * N, 128 * 8),)](
            workspace,
            out,
            M * N,
            SPLIT_K=split_k,
            BLOCK=128,
            VEC=8,
            EVEN_N=(M * N) % (128 * 8) == 0,
            num_warps=4,
            num_stages=1,
        )
    return out


def mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_THEAD MM")
    if a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0] and a.dtype == b.dtype:
        out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
        if _can_use_ppu_mm(a, b, out):
            if _split_k_params(a.shape[0], b.shape[1], a.shape[1]) is not None:
                return _run_split_k_mm(a, b, out)
            return _run_ppu_mm(a, b, out)
    return _generic_mm(a, b)


def mm_out(a: torch.Tensor, b: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_THEAD MM_OUT")
    if _can_use_ppu_mm(a, b, out):
        if _split_k_params(a.shape[0], b.shape[1], a.shape[1]) is not None:
            return _run_split_k_mm(a, b, out)
        return _run_ppu_mm(a, b, out)
    return _generic_mm_out(a, b, out=out)


__all__ = ["mm", "mm_out"]
