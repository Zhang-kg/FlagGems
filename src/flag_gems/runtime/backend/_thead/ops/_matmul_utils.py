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

"""Shared compile-time building blocks for THead PPU matrix multiplies."""

import os
import math

import triton
import triton.language as tl

try:
    import triton.experimental.tle.language as tle

    HAS_PPU_TLE = True
except ImportError:
    tle = None
    HAS_PPU_TLE = False


_LOAD_BOTH_AIU = 0
_LOAD_A_AIU = 1
_LOAD_B_AIU = 2
_LOAD_REGULAR = 3
_SMALL_M_TILE = 16
EXPAND_CONFIG_FILENAME = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "mm_ppu_expand.yaml")
)


def _ppu_bucket_strategy(value):
    """Coarsen PPU GEMM keys at tile-wave boundaries.

    This keeps nearby model trace shapes on one libtuner result while
    preserving the boundaries that change the number of output waves.  It is
    intentionally dtype-independent; dtype remains part of LibTuner's key.
    """
    value = int(value)
    if value <= 0:
        return value
    if value <= 32:
        return 1 << (value - 1).bit_length()
    if value <= 256:
        quantum = 32
    elif value <= 1024:
        quantum = 256
    elif value <= 4096:
        quantum = 512
    else:
        quantum = 1024
    return int(math.ceil(value / quantum) * quantum)


def _configs_from_specs(specs, fields):
    """Build Triton configs while keeping tuple shortlists easy to audit."""
    configs = []
    for spec in dict.fromkeys(specs):
        kwargs = dict(zip(fields, spec))
        num_warps = kwargs.pop("num_warps")
        num_stages = kwargs.pop("num_stages")
        kwargs["PIPE_STAGES"] = num_stages
        configs.append(
            triton.Config(
                kwargs, num_warps=num_warps, num_stages=num_stages
            )
        )
    return configs


def _prune_split_k_configs(configs, named_args, **kwargs):
    """Keep non-interleaved split-K slices exactly tile-aligned.

    The interleaved path explicitly zero-pads its final K tile.  The ordinary
    path partitions K into contiguous slices, so a candidate is safe only when
    every slice has an integral number of ``BLOCK_K`` tiles.  This is an early
    tuner prune and therefore applies equally to Default and Expanded spaces.
    """
    K = named_args.get("K")
    if K is None:
        return configs
    pruned = []
    for config in configs:
        split_k = config.kwargs["SPLIT_K"]
        block_k = config.kwargs["BLOCK_K"]
        interleaved = config.kwargs.get("INTERLEAVED", False)
        if interleaved or (
            K % split_k == 0 and (K // split_k) % block_k == 0
        ):
            pruned.append(config)
    return pruned or configs


def _prune_gemm_configs(configs, named_args, **kwargs):
    """Avoid compiling tiles that cannot fit the current GEMM shape.

    PPU's async-load compiler allocates resources for the full constexpr tile,
    even when the launch only has a small ragged tail.  Limiting tile extents
    to the next power of two of the runtime problem keeps autotuning safe while
    preserving every candidate that can cover at least one complete output
    tile.  The shared-memory bound is a conservative guard for deep pipelines.
    """
    M = named_args.get("M")
    N = named_args.get("N")
    K = named_args.get("K")
    if M is None or N is None or K is None:
        return configs
    max_m = max(16, triton.next_power_of_2(M))
    max_n = max(16, triton.next_power_of_2(N))
    max_k = max(32, triton.next_power_of_2(K))
    filtered = []
    expanded = len(configs) > 100
    for config in configs:
        bm = config.kwargs.get("BLOCK_M", 16)
        bn = config.kwargs.get("BLOCK_N", 16)
        bk = config.kwargs.get("BLOCK_K", 32)
        load_mode = config.kwargs.get("LOAD_MODE", _LOAD_REGULAR)
        stages = config.kwargs.get("PIPE_STAGES", config.num_stages or 1)
        if bm > max_m or bn > max_n or bk > max_k:
            continue
        # The PPU lowering cannot reliably form an async two-operand AIU
        # descriptor for sub-16-wide MMA tiles, especially with a two-stage
        # software pipeline.  Keep narrow-N candidates on regular loads (or a
        # deeper pipeline) so Expanded tuning remains compilable across the
        # full space while still exposing the useful skinny-tile choices.
        if bn < 16 and load_mode != _LOAD_REGULAR:
            continue
        if bn < 16 and stages < 3:
            continue
        if expanded:
            # Expanded tuning keeps a broad space, but each runtime shape only
            # needs the tile families that can contribute a complete wave.
            # These bucket rules are shape-generic and retain all production
            # winners while avoiding thousands of compilations per key.
            if bn > N:
                continue
            if N >= 64 and bn < 64:
                continue
            grid_m = triton.cdiv(M, bm)
            group_m = config.kwargs.get("GROUP_M", 1)
            if group_m > 1 and grid_m < group_m * 2:
                continue
            if K >= 4096 and stages < 4:
                # acBLAS uses a shallow two-stage AIU pipeline for its
                # 256x128x64 family.  Keep that family visible for the
                # long-K, N=512 wave where it is a useful alternative to the
                # deeper Triton candidates.  The exception is shape-generic
                # (it applies to the whole N=512 bucket), and libtuner still
                # decides whether it wins for the concrete dtype/shape.
                if not (N == 512 and stages >= 2):
                    continue
            if K < 4096 and stages > 4:
                continue
            if bm == 256 and M < 256:
                continue
            if group_m not in (1, 8):
                continue
            # A-only and regular loads cover the latency-sensitive regimes;
            # retain B-only AIU only for genuinely wide tiles where its setup
            # cost is amortized.
            if load_mode == _LOAD_B_AIU and bn < 256:
                continue
            if bm * bn <= 4096 and config.num_warps > 4:
                continue
        # Two operands, two bytes per element; leave headroom below the 256 KiB
        # device limit for compiler metadata and asynchronous descriptors.
        if stages * (bm + bn) * bk * 2 > 192 * 1024:
            continue
        filtered.append(config)
    return filtered or configs


if HAS_PPU_TLE:

    @triton.jit
    def _ppu_gemm_tile(
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
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        LOAD_MODE: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
        MASK_M: tl.constexpr,
        EVEN_K: tl.constexpr,
        EVEN_M: tl.constexpr,
        EVEN_N: tl.constexpr,
        FUSE_BIAS: tl.constexpr,
    ):
        """Compute and store one general GEMM tile; inlined by Triton."""
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
        if MASK_M:
            load_m = offs_m.to(tl.int64)
        elif EVEN_M:
            load_m = offs_m.to(tl.int64)
        else:
            load_m = tl.max_contiguous(
                tl.multiple_of(offs_m % M, BLOCK_M), BLOCK_M
            ).to(tl.int64)
        if EVEN_N:
            load_n = offs_n.to(tl.int64)
        else:
            load_n = tl.max_contiguous(
                tl.multiple_of(offs_n % N, BLOCK_N), BLOCK_N
            ).to(tl.int64)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in tl.range(0, K, BLOCK_K, num_stages=PIPE_STAGES):
            offs_k = k_start + tl.arange(0, BLOCK_K)
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
                    + load_m[:, None] * stride_am
                    + offs_k[None, :] * stride_ak
                )
                if MASK_M:
                    a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
                    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
                elif EVEN_K:
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
                b_ptrs = (
                    B
                    + offs_k[:, None] * stride_bk
                    + load_n[None, :] * stride_bn
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
        if FUSE_BIAS:
            bias_ptrs = (
                Bias
                + offs_m[:, None] * stride_bias_m
                + offs_n[None, :] * stride_bias_n
            )
            bias_value = tl.load(bias_ptrs, mask=c_mask, other=0.0)
            acc = alpha * acc + beta * bias_value
        if not MASK_M and EVEN_M and EVEN_N:
            tl.store(c_ptrs, acc.to(C.dtype.element_ty))
        else:
            tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=c_mask)

else:
    _ppu_gemm_tile = None
