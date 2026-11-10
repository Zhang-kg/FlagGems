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

import logging
import os

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems import runtime
from flag_gems.ops.mm import mm as _generic_mm
from flag_gems.ops.mm import mm_out as _generic_mm_out
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

try:
    import triton.experimental.tle.language as tle
    from triton.experimental.tle.language.gpu.mthreads import (
        MTHREADS_TLE_16_WARP_PERSISTENT_VERSION,
        MTHREADS_TLE_DYNAMIC_PARTITION_SYNC_VERSION,
        MTHREADS_TLE_DYNAMIC_LOOPS_VERSION,
        MTHREADS_TLE_MULTIFIELD_PIPE_VERSION,
        MTHREADS_TLE_PIPE_SQMMA_VERSION,
        MTHREADS_TLE_PERSISTENT_ORDERED_SQMMA_VERSION,
        MTHREADS_TLE_SPLIT128_SQMMA_VERSION,
        MTHREADS_TLE_SPLIT_M_SQMMA_VERSION,
    )

    HAS_MTHREADS_TLE_PIPE_SQMMA = MTHREADS_TLE_PIPE_SQMMA_VERSION >= 2
    HAS_MTHREADS_TLE_MULTIFIELD_PIPE = MTHREADS_TLE_MULTIFIELD_PIPE_VERSION >= 1
    HAS_MTHREADS_TLE_DYNAMIC_PARTITION_SYNC = (
        MTHREADS_TLE_DYNAMIC_PARTITION_SYNC_VERSION >= 1
    )
    HAS_MTHREADS_TLE_DYNAMIC_LOOPS = MTHREADS_TLE_DYNAMIC_LOOPS_VERSION >= 1
    HAS_MTHREADS_TLE_PERSISTENT_ORDERED_SQMMA = (
        MTHREADS_TLE_PERSISTENT_ORDERED_SQMMA_VERSION >= 1
    )
    HAS_MTHREADS_TLE_BN320_SQMMA = (
        HAS_MTHREADS_TLE_MULTIFIELD_PIPE
        and HAS_MTHREADS_TLE_PERSISTENT_ORDERED_SQMMA
        and HAS_MTHREADS_TLE_DYNAMIC_PARTITION_SYNC
    )
    HAS_MTHREADS_TLE_SPLIT128_SQMMA = MTHREADS_TLE_SPLIT128_SQMMA_VERSION >= 1
    HAS_MTHREADS_TLE_SPLIT_M_SQMMA = MTHREADS_TLE_SPLIT_M_SQMMA_VERSION >= 1
    HAS_MTHREADS_TLE_SPLIT_N_SINGLE_PIPE = HAS_MTHREADS_TLE_PIPE_SQMMA
    HAS_MTHREADS_TLE_16_WARP_PERSISTENT = (
        MTHREADS_TLE_16_WARP_PERSISTENT_VERSION >= 1
        and HAS_MTHREADS_TLE_PIPE_SQMMA
        and HAS_MTHREADS_TLE_MULTIFIELD_PIPE
        and HAS_MTHREADS_TLE_PERSISTENT_ORDERED_SQMMA
        and HAS_MTHREADS_TLE_DYNAMIC_PARTITION_SYNC
    )
except ImportError:
    tle = None
    HAS_MTHREADS_TLE_DYNAMIC_PARTITION_SYNC = False
    HAS_MTHREADS_TLE_DYNAMIC_LOOPS = False
    HAS_MTHREADS_TLE_PIPE_SQMMA = False
    HAS_MTHREADS_TLE_MULTIFIELD_PIPE = False
    HAS_MTHREADS_TLE_BN320_SQMMA = False
    HAS_MTHREADS_TLE_PERSISTENT_ORDERED_SQMMA = False
    HAS_MTHREADS_TLE_SPLIT128_SQMMA = False
    HAS_MTHREADS_TLE_SPLIT_M_SQMMA = False
    HAS_MTHREADS_TLE_SPLIT_N_SINGLE_PIPE = False
    HAS_MTHREADS_TLE_16_WARP_PERSISTENT = False

# Old FlagTree versions propagate the MTGPU register allocator abort as a
# process-level compiler failure.  Keep the guard narrow and fail closed: the
# generic Torch/Triton path remains available on those installations, while
# the updated backend can safely let expanded autotune prune bad candidates.
try:
    from triton.backends.mthreads.compiler import (
        MTHREADS_REGISTER_FAILURE_RECOVERY_VERSION,
    )
except (ImportError, AttributeError):
    MTHREADS_REGISTER_FAILURE_RECOVERY_VERSION = 0
HAS_MTHREADS_REGISTER_FAILURE_RECOVERY = (
    MTHREADS_REGISTER_FAILURE_RECOVERY_VERSION >= 1
)

logger = logging.getLogger(__name__)

EXPAND_CONFIG_FILENAME = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "mm_mthreads_expand.yaml")
)

# Module-level capability flag: evaluated once at import time, then reused as
# a constant for the entire process lifetime with no repeated parsing overhead.
# False when Triton < 3.2 (e.g. 3.1), True when Triton >= 3.2.
SQMMA_ON = tuple(int(x) for x in triton.__version__.split(".")[:2]) >= (3, 2)


def is_supported_sqmma_layout(tensor):
    return tensor.is_contiguous() or (
        tensor.stride(0) == 1 and tensor.stride(1) == tensor.shape[0]
    )


def is_sqmma_compatible(a, b, N, K):
    return (
        SQMMA_ON
        and a.dim() == 2
        and b.dim() == 2
        and a.dtype == b.dtype
        and a.dtype in (torch.float16, torch.bfloat16)
        and is_supported_sqmma_layout(a)
        and is_supported_sqmma_layout(b)
        and N % 8 == 0
        and K % 8 == 0
    )


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mm"),
    key=["M", "N", "K", "stride_am", "stride_bk"],
    strategy=["align32", "align32", "align32", "align32", "align32"],
    warmup=5,
    rep=5,
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
)
@triton.jit
def mm_kernel(
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
    dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    IS_FP64: tl.constexpr = False,
):
    # matrix multiplication
    pid = ext.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M).to(tl.int64)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N).to(tl.int64)
    rm = rm.to(tl.int64)
    rn = rn.to(tl.int64)
    prev_multiple = prev_multiple_of(K, BLOCK_K)

    if IS_FP64:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for start_k in range(0, prev_multiple, BLOCK_K):
        rk = (start_k + tl.arange(0, BLOCK_K)).to(tl.int64)
        a = tl.load(A + (ram[:, None] * stride_am + rk[None, :] * stride_ak))
        b = tl.load(B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn))
        if a.dtype != b.dtype:
            a = a.to(C.dtype.element_ty)
            b = b.to(C.dtype.element_ty)
        if IS_FP64:
            acc += tl.dot(a, b, allow_tf32=False)
        else:
            acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    # loop peeling
    rk = (prev_multiple + tl.arange(0, BLOCK_K)).to(tl.int64)
    mask_k = rk < K
    a = tl.load(
        A + (ram[:, None] * stride_am + rk[None, :] * stride_ak), mask=mask_k[None, :]
    )
    b = tl.load(
        B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn), mask=mask_k[:, None]
    )
    if a.dtype != b.dtype:
        a = a.to(C.dtype.element_ty)
        b = b.to(C.dtype.element_ty)
    if IS_FP64:
        acc += tl.dot(a, b, allow_tf32=False)
    else:
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    acc = acc.to(C.dtype.element_ty)
    # rematerialize rm and rn to save registers
    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    rn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    # handles write-back with reduction-splitting
    tl.store(C, acc, mask=mask)


@libentry()
@triton.jit
def gemv_kernel(
    A,
    B,
    C,
    M,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_cm,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = ext.program_id(0)

    row_start = pid * BLOCK_M
    row_offset = row_start + tl.arange(0, BLOCK_M)
    row_mask = row_offset < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offset = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offset < K

        a_ptrs = A + row_offset[:, None] * stride_am + k_offset[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)

        b_ptrs = B + k_offset * stride_bk
        b = tl.load(b_ptrs, mask=k_mask, other=0.0)

        # Keep the reduction in fp32 so N=1 GEMV matches the mm path more closely.
        a = a.to(tl.float32)
        b = b.to(tl.float32)
        acc += tl.sum(a * b[None, :], axis=1)

    c_ptrs = C + row_offset * stride_cm
    acc = acc.to(C.dtype.element_ty)
    tl.store(c_ptrs, acc, mask=row_mask)


@triton.jit
def mm_small_m_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Low-live-range GEMM for very small M.

    The general MTT GEMM keeps a BLOCK_M x BLOCK_N accumulator, which is
    disproportionately expensive when M is one or a few rows.  This kernel
    computes one row tile per program and therefore avoids the large register
    class pressure that makes the expanded generic candidates fail or run
    several times slower than torch for decode-sized matrices.
    """
    pid_m = ext.program_id(0)
    pid_n = ext.program_id(1)
    rows = pid_m
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < N
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        k_mask = ks < K
        a = tl.load(
            a_ptr + rows * stride_am + ks * stride_ak,
            mask=k_mask,
            other=0.0,
        )
        b = tl.load(
            b_ptr + ks[:, None] * stride_bk + cols[None, :] * stride_bn,
            mask=k_mask[:, None] & col_mask[None, :],
            other=0.0,
        )
        # Explicitly widen operands before the reduction.  On MTT this keeps
        # the reduction in FP32 without invoking the tensor-core dot path for
        # a 1-row tile, whose setup cost dominates these decode-sized GEMMs.
        a = a.to(tl.float32)
        b = b.to(tl.float32)
        acc += tl.sum(a[:, None] * b, axis=0)
    tl.store(
        c_ptr + rows * stride_cm + cols * stride_cn,
        acc.to(c_ptr.dtype.element_ty),
        mask=col_mask,
    )


def mm_small_m(a, b, c, M, N, K):
    if N <= 16:
        block_n = 16
    elif N <= 32:
        block_n = 32
    elif N <= 64:
        block_n = 64
    else:
        block_n = 128
    # Larger K tiles amortize the loop and descriptor arithmetic for decode
    # GEMMs while the accumulator remains only 1 x BLOCK_N, so they do not
    # trigger the rolling-kernel register-pressure failure.
    block_k = 256 if K % 256 == 0 else 64
    grid = (M, triton.cdiv(N, block_n))
    with torch_device_fn.device(a.device):
        mm_small_m_kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=4,
            num_stages=2,
        )
    return c


_ordered_datatypes = [torch.float16, torch.bfloat16, torch.float32, torch.float64]


def get_higher_dtype(a, b):
    if a is b:
        return a

    assert a in _ordered_datatypes
    assert b in _ordered_datatypes

    for d in _ordered_datatypes:
        if a is d:
            return b
        if b is d:
            return a


def mm_fma(a, b):
    logger.debug("GEMS_MTHREADS MM_FMA")
    device = a.device
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # allocates output
    c_dtype = get_higher_dtype(a.dtype, b.dtype)
    c = torch.empty((M, N), device=device, dtype=c_dtype)
    # launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(a.device):
        mm_kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            dtype=str(a.dtype).split(".")[-1],
            GROUP_M=8,
            IS_FP64=a.dtype == torch.float64,
        )
    return c


def gemv_mm(a, b, c, M, K):
    logger.debug(
        "GEMS_MTHREADS MM_GEMV_, [shape info]: [%s, %s, 1](M, K, N)",
        M,
        K,
    )
    if M <= 480:
        block_m = 1
    elif M < 3072:
        block_m = 8
    elif M < 8192:
        block_m = 16
    else:
        block_m = 64
    if M < 8192 and K % 512 == 0:
        block_k = 512
    elif K % 256 == 0:
        block_k = 256
    elif K % 128 == 0:
        block_k = 128
    else:
        block_k = 64
    grid = (triton.cdiv(M, block_m),)
    with torch_device_fn.device(a.device):
        gemv_kernel[grid](
            a,
            b,
            c,
            M,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            c.stride(0),
            block_m,
            block_k,
            num_warps=8,
            num_stages=4,
        )
    return c


def mm_out(a, b, *, out):
    logger.debug("GEMS_MTHREADS MM_OUT")
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # allocates output
    c = out
    if N == 1:
        return gemv_mm(a, b, c, M, K)
    if is_tle_split_n_small_compatible(M, N, K, a.dtype):
        return mm_tle_split_n_small_pipe(a, b, c, M, N, K)
    # Keep the experimental scalar small-M kernel out of the production
    # dispatch. It does not use SQMMA and regresses the measured Qwen buckets.
    # Keep the already-tuned M=64 wide-N path on a short user-entry route.
    # ``mm_tle_pipe`` checks the same family after several persistent/split
    # guards; avoiding that second Python dispatch matters at sub-50 us
    # latencies while preserving the full capability/contiguity check here.
    tle_compatible = is_tle_pipe_compatible(a, b, M, N, K)
    if tle_compatible:
        if is_tle_split_n_full_compatible(M, N, K, a.dtype):
            return mm_tle_split_n_full_pipe(a, b, c, M, N, K)
        if is_tle_split_n_single_compatible(M, N, K, a.dtype):
            return mm_tle_split_n_single_pipe(a, b, c, M, N, K)
        return mm_tle_pipe(a, b, c, M, N, K)
    return _generic_mm_out(a, b, out=c)


def sqmma_descriptor_pre_hook(nargs):
    nargs["a_desc"].block_shape = [nargs["BLOCK_M"], nargs["BLOCK_K"]]
    nargs["b_desc"].block_shape = [nargs["BLOCK_K"], nargs["BLOCK_N"]]
    nargs["c_desc"].block_shape = [nargs["BLOCK_M"], nargs["BLOCK_N"]]


def tle_pipe_descriptor_pre_hook(nargs):
    nargs["a_desc"].block_shape = [nargs["BLOCK_M"], nargs["BLOCK_K"]]
    nargs["b_desc"].block_shape = [nargs["BLOCK_K"], nargs["BLOCK_N"]]


def _tle_pipe_configs(specs, num_warps):
    return [
        triton.Config(
            {
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "BLOCK_K": block_k,
                "NUM_SLOTS": num_slots,
                "MMA_GROUP": 1,
                "GROUP_M": group_m,
            },
            num_stages=3,
            num_warps=num_warps,
            pre_hook=tle_pipe_descriptor_pre_hook,
        )
        for block_m, block_n, block_k, num_slots, group_m in dict.fromkeys(specs)
    ]


_TLE_NON_WS_PIPE_CONFIGS = _tle_pipe_configs(
    [
        (64, 64, 64, 1, 1),
        # Measured best tile for the high-count N=64,K=2048 buckets.
        (64, 64, 64, 3, 1),
        (64, 64, 128, 2, 1),
        (64, 64, 128, 3, 1),
        (64, 64, 256, 3, 1),
        (64, 64, 256, 3, 2),
        (64, 128, 64, 3, 2),
        (64, 128, 64, 3, 1),
        (64, 128, 64, 4, 1),
        (64, 128, 64, 3, 8),
        (64, 128, 128, 1, 8),
        (64, 128, 128, 2, 2),
        (64, 128, 128, 2, 8),
        (128, 64, 128, 2, 1),
        (128, 64, 256, 2, 1),
        (128, 128, 64, 3, 1),
        (128, 128, 64, 3, 4),
        (128, 128, 64, 3, 8),
        (128, 128, 128, 1, 4),
        (128, 128, 128, 2, 4),
        (128, 256, 64, 2, 1),
        (128, 256, 128, 2, 1),
    ],
    num_warps=4,
) + _tle_pipe_configs(
    [
        # Verified on MTT for the small-M N=1024 bucket. Keep this as a
        # fallback even when expanded FlagTune mode is unavailable.
        (64, 128, 64, 3, 1),
    ],
    num_warps=8,
)

_TLE_WS_PIPE_SPECS = [
    (128, 128, 64, 1, 1),
    (128, 128, 64, 2, 1),
    (128, 256, 64, 2, 1),
    (256, 128, 64, 2, 1),
    (256, 128, 64, 4, 1),
    (256, 256, 64, 1, 1),
    (256, 256, 64, 2, 1),
    (256, 256, 64, 2, 2),
    (256, 256, 64, 3, 1),
    (256, 256, 64, 3, 8),
]

_TLE_WS_PIPE_CONFIGS = _tle_pipe_configs(
    _TLE_WS_PIPE_SPECS,
    num_warps=16,
) + _tle_pipe_configs(
    [
        (128, 128, 64, 1, 1),
        (128, 128, 64, 2, 1),
        (128, 256, 64, 2, 1),
        (256, 128, 64, 2, 1),
    ],
    num_warps=8,
)


def _prune_tle_pipe_configs(configs, named_args, warp_specialized=False, **kwargs):
    M = named_args["M"]
    N = named_args["N"]
    K = named_args["K"]
    max_block_m = max(64, triton.next_power_of_2(M))
    max_block_n = max(64, triton.next_power_of_2(N))
    filtered = []

    # These ragged large-M/N=64 shapes are frequent in the Qwen traces.  The
    # measured 64-row tile avoids the register/occupancy loss of BM=128 while
    # retaining enough CTAs to saturate MTT.  Keep this exact guard narrow:
    # neighboring M values have not all been benchmarked yet.
    verified_n64_shape = (
        not warp_specialized
        and N == 64
        and K == 2048
        and M in (1035, 1036, 2048, 4138, 4434, 4435, 16384)
    )
    verified_ws_512_k512 = (
        warp_specialized and M == 512 and N == 2048 and K == 512
    )

    for config in configs:
        block_m = config.kwargs["BLOCK_M"]
        block_n = config.kwargs["BLOCK_N"]
        block_k = config.kwargs["BLOCK_K"]
        num_slots = config.kwargs["NUM_SLOTS"]
        mma_group = config.kwargs.get("MMA_GROUP", 1)
        group_m = config.kwargs["GROUP_M"]
        if verified_n64_shape and (
            block_m,
            block_n,
            block_k,
            num_slots,
            mma_group,
            group_m,
            config.num_warps,
        ) != (64, 64, 64, 3, 1, 1, 4):
            continue
        if verified_ws_512_k512 and (
            block_m,
            block_n,
            block_k,
            num_slots,
            mma_group,
            group_m,
            config.num_warps,
        ) != (256, 128, 64, 4, 1, 1, 16):
            continue
        # MTT SQMMA currently accepts K=16/32/64 for bf16/fp16.  Expanded
        # YAML intentionally keeps wider values for portability, but sending
        # them to this backend only causes compile/runtime failures.
        if HAS_MTHREADS_TLE_PIPE_SQMMA and block_k > 64:
            continue
        if block_m > max_block_m or block_n > max_block_n:
            continue
        # Narrow-N accumulators are register-bound once rolling pipe refills
        # are present.  Keep the fallback search below 16K output elements;
        # otherwise the generic safe-list can select BM128xBN256 and fail in
        # LLVM register allocation before autotune has a chance to reject it.
        if N <= 256 and block_m * block_n > 16384:
            continue
        k_tiles = K // block_k
        if K % block_k != 0 or num_slots > k_tiles:
            continue
        if mma_group > num_slots:
            continue
        # A non-divisible MMA group creates one predicated scf.if per group
        # offset and places reader.release in sibling conditionals.  The
        # current TLE pipe protocol cannot prove those cross-region lifetimes;
        # keep only groups that map to one straight-line wait/release block.
        if k_tiles % mma_group != 0:
            continue
        if block_k == 32 and k_tiles > 64:
            continue
        if group_m > triton.cdiv(M, block_m):
            continue
        if k_tiles >= 4 and num_slots == 1:
            continue
        if min(M, N) >= 256 and block_m * block_n < 8192:
            output_tiles = triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
            if output_tiles > 64:
                continue
        shared_memory = num_slots * (block_m + block_n) * block_k * 2
        if shared_memory > 192 * 1024:
            continue
        if warp_specialized:
            if (
                config.num_warps not in (8, 16)
                or block_m < 128
                or block_n < 128
            ):
                continue
            if M == 512 and N >= 2048 and K >= 4096:
                if (
                    block_m * block_n != 32768
                    or block_k != 64
                    or num_slots != 4
                    # K=4096 benefits from issuing multiple SQMMA operations
                    # before the wait on some dtypes/shapes.  Keep the
                    # expanded search bounded to the measured group sizes;
                    # group=3 has no supporting benchmark and group=4 is
                    # retained only for autotune to reject when slower.
                    or mma_group not in (1, 2, 4)
                ):
                    continue
            if config.num_warps == 8:
                output_tiles = triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
                if block_m * block_n > 32768 or output_tiles > 64:
                    continue
        else:
            if config.num_warps not in (4, 8):
                continue
            if config.num_warps == 4 and block_m > 128:
                continue
            # Non-WS 8-warp candidates are useful for the larger output tiles
            # in the small-M regime, but become register/occupancy limited by
            # M=1035 on MTT. Keep the expanded search bounded to the measured
            # M<=512 range and avoid high-register configurations with deep
            # pipes.
            verified_small_8warp = (
                block_m == 64
                and block_n == 128
                and block_k == 64
                and num_slots == 3
                and group_m == 1
            )
            if config.num_warps == 8 and (
                M > 512
                or (block_m * block_n < 16384 and not verified_small_8warp)
                or (
                    num_slots > 2
                    # This exact low-register tile was measured at about
                    # 109% of torch for 512x1024x2048. Keep it available to
                    # expanded tuning without opening every deep pipe.
                    and not verified_small_8warp
                )
            ):
                continue
        filtered.append(config)

    if filtered:
        return filtered

    # A prune hook must still enforce backend capability constraints when all
    # shape-specific candidates were rejected.  The old ``or list(configs)``
    # fallback could re-introduce K=128/256 SQMMA configs on MTT and turn a
    # compile-time guard into a backend failure.  Keep the historical fallback
    # for non-MTT consumers, where the wider K contract remains valid.
    if HAS_MTHREADS_TLE_PIPE_SQMMA:
        # Keep the fallback bounded.  Empty shape-specific results are common
        # for ragged M/N (the output-tile heuristic intentionally rejects all
        # candidates), but handing the full expanded space back to FlagTune
        # defeats pruning and can retry unsupported K=128/256 kernels.  Apply
        # only backend/resource constraints here, then retain the largest
        # eight tiles so the tuner still has a small, useful search space.
        safe = []
        for config in configs:
            block_m = config.kwargs.get("BLOCK_M", 0)
            block_n = config.kwargs.get("BLOCK_N", 0)
            block_k = config.kwargs.get("BLOCK_K", 0)
            num_slots = config.kwargs.get("NUM_SLOTS", 0)
            mma_group = config.kwargs.get("MMA_GROUP", 1)
            group_m = config.kwargs.get("GROUP_M", 1)
            if block_m <= 0 or block_n <= 0:
                continue
            if N <= 256 and block_m * block_n > 16384:
                continue
            if block_k <= 0 or block_k > 64 or K % block_k:
                continue
            k_tiles = K // block_k
            if (
                num_slots <= 0
                or mma_group <= 0
                or num_slots > k_tiles
                or mma_group > num_slots
            ):
                continue
            if k_tiles % mma_group != 0:
                continue
            if block_k == 32 and k_tiles > 64:
                continue
            if group_m <= 0 or group_m > triton.cdiv(M, block_m):
                continue
            if num_slots * (block_m + block_n) * block_k * 2 > 192 * 1024:
                continue
            if warp_specialized:
                if (
                    config.num_warps not in (8, 16)
                    or block_m < 128
                    or block_n < 128
                ):
                    continue
            elif config.num_warps not in (4, 8):
                continue
            safe.append(config)
        safe.sort(
            key=lambda config: (
                config.kwargs.get("BLOCK_M", 0) * config.kwargs.get("BLOCK_N", 0),
                config.kwargs.get("BLOCK_K", 0),
                -config.kwargs.get("NUM_SLOTS", 0),
            ),
            reverse=True,
        )
        return safe[:8]
    return list(configs)


def _prune_tle_non_ws_pipe_configs(configs, named_args, **kwargs):
    return _prune_tle_pipe_configs(configs, named_args, **kwargs)


def _prune_tle_ws_pipe_configs(configs, named_args, **kwargs):
    return _prune_tle_pipe_configs(
        configs, named_args, warp_specialized=True, **kwargs
    )


@triton.jit
def _tle_mm_pipe_producer(
    writer,
    a_desc,
    b_desc,
    m_offset,
    n_offset,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_TILES: tl.constexpr,
):
    # Keep the K loop dynamic.  Python range/constexpr expansion makes
    # K=4096 (64 tiles at BLOCK_K=64) generate a very large TLE IR module and
    # can exceed the compiler/runtime watchdog; tl.range preserves the same
    # protocol while allowing the backend to lower one loop body.
    for k_iter in tl.range(0, K_TILES, num_stages=1):
        k_offset = k_iter * BLOCK_K
        slot = writer.acquire(k_iter)
        tle.gpu.copy(
            a_desc,
            slot.a,
            [BLOCK_M, BLOCK_K],
            [m_offset, k_offset],
        )
        tle.gpu.copy(
            b_desc,
            slot.b,
            [BLOCK_K, BLOCK_N],
            [k_offset, n_offset],
        )
        writer.commit(k_iter)


@triton.jit
def _tle_mm_pipe_consumer(
    reader,
    c_ptr,
    m_offset,
    n_offset,
    stride_cm,
    stride_cn,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K_TILES: tl.constexpr,
    MMA_GROUP: tl.constexpr,
):
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Expanded pruning requires K_TILES % MMA_GROUP == 0, so no tail guard is
    # needed in this WS-only helper.  Keeping a dynamic loop avoids expanding
    # up to 64 K iterations into every autotune candidate while leaving the
    # wait/release pair in one lexical block for LowerPipe.
    for group_start in tl.range(0, K_TILES, MMA_GROUP, num_stages=1):
        for group_offset in tl.static_range(MMA_GROUP):
            k_iter = group_start + group_offset
            ready = reader.wait(k_iter)
            accumulator = tle.gpu.wgmma(
                ready.slot.a,
                ready.slot.b,
                accumulator,
            )
        accumulator = tle.gpu.wgmma_wait(0, accumulator)
        for group_offset in tl.static_range(MMA_GROUP):
            k_iter = group_start + group_offset
            reader.release(k_iter)

    offs_m = m_offset + tl.arange(0, BLOCK_M)
    offs_n = n_offset + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=mask)


@libentry()
@libtuner(
    configs=_TLE_NON_WS_PIPE_CONFIGS,
    key=["M", "N", "K"],
    strategy=["mthreads_mm_bucket", "mthreads_mm_bucket", "mthreads_mm_bucket"],
    prune_configs_by={"early_config_prune": _prune_tle_non_ws_pipe_configs},
    warmup=5,
    rep=5,
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm_tle_non_ws_pipe",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    flagtune_pre_hook=tle_pipe_descriptor_pre_hook,
    flagtune_default_mode=runtime.TuningMode.EXPANDED,
)
@triton.jit
def mm_tle_non_ws_pipe_kernel(
    a_desc,
    b_desc,
    c_ptr,
    M,
    N,
    K: tl.constexpr,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    MMA_GROUP: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # The launch grid is bounded by the MTT tile count; keep pid native i32
    # to avoid widening it before descriptor offsets are narrowed below.
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    group_width = GROUP_M * grid_n
    group_id = pid // group_width
    first_m = group_id * GROUP_M
    actual_group_m = min(grid_m - first_m, GROUP_M)
    pid_in_group = pid % group_width
    pid_m = first_m + pid_in_group % actual_group_m
    pid_n = pid_in_group // actual_group_m
    m_offset = (pid_m * BLOCK_M).to(tl.int32)
    n_offset = (pid_n * BLOCK_N).to(tl.int32)

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
        name="mm_non_ws",
        a=a_smem,
        b=b_smem,
    )
    writer = pipe.writer()
    reader = pipe.reader()

    # Keep the preload index distinct from the dynamic K-loop temporary.
    # The latter is intentionally tensor-valued inside the pipelined loop;
    # older FlagTree frontends treated reuse of the same source name as a
    # constexpr/tensor loop-carried type change.
    for preload_iter in tl.static_range(NUM_SLOTS):
        k_offset = preload_iter * BLOCK_K
        slot = writer.acquire(preload_iter)
        tle.gpu.copy(
            a_desc, slot.a, [BLOCK_M, BLOCK_K], [m_offset, k_offset]
        )
        tle.gpu.copy(
            b_desc, slot.b, [BLOCK_K, BLOCK_N], [k_offset, n_offset]
        )
        writer.commit(preload_iter)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    k_tiles: tl.constexpr = K // BLOCK_K
    # Keep one lexical producer block for both complete and ragged K groups.
    # The compile-time branch only changes SQMMA/release predication; placing
    # A/B refill below it prevents grouped TME copies from being split across
    # two SCF regions in the FlagTree lowering.
    for group_start in tl.range(0, k_tiles, MMA_GROUP, num_stages=1):
        if k_tiles % MMA_GROUP == 0:
            for group_offset in tl.static_range(MMA_GROUP):
                k_iter = group_start + group_offset
                ready = reader.wait(k_iter)
                accumulator = tle.gpu.wgmma(
                    ready.slot.a,
                    ready.slot.b,
                    accumulator,
                )
        else:
            for group_offset in tl.static_range(MMA_GROUP):
                k_iter = group_start + group_offset
                if k_iter < k_tiles:
                    ready = reader.wait(k_iter)
                    accumulator = tle.gpu.wgmma(
                        ready.slot.a,
                        ready.slot.b,
                        accumulator,
                    )
        accumulator = tle.gpu.wgmma_wait(0, accumulator)
        if k_tiles % MMA_GROUP == 0:
            for group_offset in tl.static_range(MMA_GROUP):
                k_iter = group_start + group_offset
                reader.release(k_iter)
        else:
            for group_offset in tl.static_range(MMA_GROUP):
                k_iter = group_start + group_offset
                if k_iter < k_tiles:
                    reader.release(k_iter)

        for group_offset in tl.static_range(MMA_GROUP):
            k_iter = group_start + group_offset
            if k_iter + NUM_SLOTS < k_tiles:
                next_iter = k_iter + NUM_SLOTS
                k_offset = next_iter * BLOCK_K
                next_slot = writer.acquire(next_iter)
                tle.gpu.copy(
                    a_desc,
                    next_slot.a,
                    [BLOCK_M, BLOCK_K],
                    [m_offset, k_offset],
                )
                tle.gpu.copy(
                    b_desc,
                    next_slot.b,
                    [BLOCK_K, BLOCK_N],
                    [k_offset, n_offset],
                )
                writer.commit(next_iter)

    offs_m = m_offset + tl.arange(0, BLOCK_M)
    offs_n = n_offset + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=mask)


@libentry()
@libtuner(
    configs=_TLE_WS_PIPE_CONFIGS,
    key=["M", "N", "K"],
    strategy=["mthreads_mm_bucket", "mthreads_mm_bucket", "mthreads_mm_bucket"],
    prune_configs_by={"early_config_prune": _prune_tle_ws_pipe_configs},
    warmup=5,
    rep=5,
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm_tle_ws_pipe",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    flagtune_pre_hook=tle_pipe_descriptor_pre_hook,
    flagtune_default_mode=runtime.TuningMode.EXPANDED,
)
@triton.jit
def mm_tle_pipe_kernel(
    a_desc,
    b_desc,
    c_ptr,
    M,
    N,
    K: tl.constexpr,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    MMA_GROUP: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # The launch grid is bounded by the MTT tile count; keep pid native i32
    # to avoid widening it before descriptor offsets are narrowed below.
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    group_width = GROUP_M * grid_n
    group_id = pid // group_width
    first_m = group_id * GROUP_M
    actual_group_m = min(grid_m - first_m, GROUP_M)
    pid_in_group = pid % group_width
    pid_m = first_m + pid_in_group % actual_group_m
    pid_n = pid_in_group // actual_group_m
    # MThreads TMA descriptors require 32-bit dynamic offsets. program_id is
    # widened by ext.program_id, so make the descriptor contract explicit.
    m_offset = (pid_m * BLOCK_M).to(tl.int32)
    n_offset = (pid_n * BLOCK_N).to(tl.int32)

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
        name="mm_ws",
        a=a_smem,
        b=b_smem,
    )

    k_tiles: tl.constexpr = K // BLOCK_K
    tle.gpu.warp_specialize(
        [
            (
                _tle_mm_pipe_consumer,
                (
                    pipe.reader(),
                    c_ptr,
                    m_offset,
                    n_offset,
                    stride_cm,
                    stride_cn,
                    M,
                    N,
                    BLOCK_M,
                    BLOCK_N,
                    k_tiles,
                    MMA_GROUP,
                ),
            ),
            (
                _tle_mm_pipe_producer,
                (
                    pipe.writer(),
                    a_desc,
                    b_desc,
                    m_offset,
                    n_offset,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                    k_tiles,
                ),
            ),
        ],
        [8],
        [24],
    )



@triton.jit
def _tle_mm_split_m_producer(
    a_writer,
    b_writer,
    a_desc,
    b_desc,
    m_offset,
    n_offset,
    K_TILES: tl.constexpr,
    BLOCK_M_STORAGE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    for k_iter in range(K_TILES):
        k_offset = k_iter * BLOCK_K
        a_slot = a_writer.acquire(k_iter)
        b_slot = b_writer.acquire(k_iter)
        tle.gpu.copy(
            a_desc,
            a_slot.a,
            [BLOCK_M_STORAGE, BLOCK_K],
            [m_offset, k_offset],
        )
        tle.gpu.copy(
            b_desc, b_slot.b, [BLOCK_K, 256], [k_offset, n_offset]
        )
        a_writer.commit(k_iter)
        b_writer.commit(k_iter)


@triton.jit
def _tle_mm_split_m_consumer_top(
    a_reader,
    b_reader,
    c_ptr,
    m_offset,
    n_offset,
    stride_cm,
    stride_cn,
    M,
    N,
    K_TILES: tl.constexpr,
):
    accumulator = tl.zeros((256, 256), dtype=tl.float32)
    for k_iter in range(K_TILES):
        a_wait = a_reader.wait(k_iter)
        b_wait = b_reader.wait(k_iter)
        a_tile = a_wait.slot.a.slice(0, 256, dim=0)
        accumulator = tle.gpu.wgmma(a_tile, b_wait.slot.b, accumulator)
        accumulator = tle.gpu.wgmma_wait(0, accumulator)
        a_reader.release(k_iter)
        b_reader.release(k_iter)

    offs_m = m_offset + tl.arange(0, 256)
    offs_n = n_offset + tl.arange(0, 256)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _tle_mm_split_m_consumer_bottom(
    a_reader,
    b_reader,
    c_ptr,
    m_offset,
    n_offset,
    stride_cm,
    stride_cn,
    M,
    N,
    K_TILES: tl.constexpr,
    BLOCK_M_BOTTOM: tl.constexpr,
):
    accumulator = tl.zeros((BLOCK_M_BOTTOM, 256), dtype=tl.float32)
    for k_iter in range(K_TILES):
        a_wait = a_reader.wait(k_iter)
        b_wait = b_reader.wait(k_iter)
        a_tile = a_wait.slot.a.slice(256, BLOCK_M_BOTTOM, dim=0)
        accumulator = tle.gpu.wgmma(a_tile, b_wait.slot.b, accumulator)
        accumulator = tle.gpu.wgmma_wait(0, accumulator)
        a_reader.release(k_iter)
        b_reader.release(k_iter)

    offs_m = m_offset + 256 + tl.arange(0, BLOCK_M_BOTTOM)
    offs_n = n_offset + tl.arange(0, 256)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _tle_mm_split384_bk128_consumer_top(
    a_reader,
    b_reader,
    c_ptr,
    m_offset,
    n_offset,
    stride_cm,
    stride_cn,
    M,
    N,
    K_TILES: tl.constexpr,
):
    """Split each BK=128 payload into two PH1-supported K=64 issues."""
    accumulator = tl.zeros((256, 256), dtype=tl.float32)
    for k_iter in range(K_TILES):
        a_wait = a_reader.wait(k_iter)
        b_wait = b_reader.wait(k_iter)
        a_tile = a_wait.slot.a.slice(0, 256, dim=0)
        for k_part in tl.static_range(2):
            a_k = a_tile.slice(k_part * 64, 64, dim=1)
            b_k = b_wait.slot.b.slice(k_part * 64, 64, dim=0)
            accumulator = tle.gpu.wgmma(a_k, b_k, accumulator)
            accumulator = tle.gpu.wgmma_wait(0, accumulator)
        a_reader.release(k_iter)
        b_reader.release(k_iter)

    offs_m = m_offset + tl.arange(0, 256)
    offs_n = n_offset + tl.arange(0, 256)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _tle_mm_split384_bk128_consumer_bottom(
    a_reader,
    b_reader,
    c_ptr,
    m_offset,
    n_offset,
    stride_cm,
    stride_cn,
    M,
    N,
    K_TILES: tl.constexpr,
):
    """Split the bottom 128-row consumer's BK=128 payload into K=64 issues."""
    accumulator = tl.zeros((128, 256), dtype=tl.float32)
    for k_iter in range(K_TILES):
        a_wait = a_reader.wait(k_iter)
        b_wait = b_reader.wait(k_iter)
        a_tile = a_wait.slot.a.slice(256, 128, dim=0)
        for k_part in tl.static_range(2):
            a_k = a_tile.slice(k_part * 64, 64, dim=1)
            b_k = b_wait.slot.b.slice(k_part * 64, 64, dim=0)
            accumulator = tle.gpu.wgmma(a_k, b_k, accumulator)
            accumulator = tle.gpu.wgmma_wait(0, accumulator)
        a_reader.release(k_iter)
        b_reader.release(k_iter)

    offs_m = m_offset + 256 + tl.arange(0, 128)
    offs_n = n_offset + tl.arange(0, 256)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit
def mm_tle_split_m_pipe_kernel(
    a_desc,
    b_desc,
    c_ptr,
    M,
    N,
    stride_cm,
    stride_cn,
    GRID_N: tl.constexpr,
    K_TILES: tl.constexpr,
    BLOCK_M_STORAGE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
):
    pid = ext.program_id(0)
    pid_m = pid // GRID_N
    pid_n = pid % GRID_N
    m_offset = (pid_m * 320).to(tl.int32)
    n_offset = (pid_n * 256).to(tl.int32)

    a_smem = tle.gpu.alloc(
        [NUM_SLOTS, BLOCK_M_STORAGE, BLOCK_K],
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
        capacity=NUM_SLOTS, scope="cta", name="mm_split_m_a", a=a_smem
    )
    b_pipe = tle.pipe(
        capacity=NUM_SLOTS, scope="cta", name="mm_split_m_b", b=b_smem
    )
    tle.gpu.warp_specialize(
        [
            (
                _tle_mm_split_m_consumer_top,
                (
                    a_pipe.reader(),
                    b_pipe.reader(),
                    c_ptr,
                    m_offset,
                    n_offset,
                    stride_cm,
                    stride_cn,
                    M,
                    N,
                    K_TILES,
                ),
            ),
            (
                _tle_mm_split_m_consumer_bottom,
                (
                    a_pipe.reader(),
                    b_pipe.reader(),
                    c_ptr,
                    m_offset,
                    n_offset,
                    stride_cm,
                    stride_cn,
                    M,
                    N,
                    K_TILES,
                    64,
                ),
            ),
            (
                _tle_mm_split_m_producer,
                (
                    a_pipe.writer(),
                    b_pipe.writer(),
                    a_desc,
                    b_desc,
                    m_offset,
                    n_offset,
                    K_TILES,
                    BLOCK_M_STORAGE,
                    BLOCK_K,
                ),
            ),
        ],
        [4, 4],
        [168, 24],
    )


@triton.jit
def _tle_mm_split_m_fused_consumer(
    reader,
    c_ptr,
    m_offset,
    n_offset,
    stride_cm,
    stride_cn,
    M,
    N,
    K_TILES: tl.constexpr,
    ROW_OFFSET: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    accumulator = tl.zeros((BLOCK_M, 256), dtype=tl.float32)
    for k_iter in range(K_TILES):
        ready = reader.wait(k_iter)
        a_tile = ready.slot.a.slice(ROW_OFFSET, BLOCK_M, dim=0)
        accumulator = tle.gpu.wgmma(a_tile, ready.slot.b, accumulator)
        accumulator = tle.gpu.wgmma_wait(0, accumulator)
        reader.release(k_iter)

    offs_m = m_offset + ROW_OFFSET + tl.arange(0, BLOCK_M)
    offs_n = n_offset + tl.arange(0, 256)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _tle_mm_split_m_fused_producer(
    writer,
    a_desc,
    b_desc,
    m_offset,
    n_offset,
    K_TILES: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    for k_iter in range(K_TILES):
        k_offset = k_iter * BLOCK_K
        slot = writer.acquire(k_iter)
        tle.gpu.copy(a_desc, slot.a, [512, BLOCK_K], [m_offset, k_offset])
        tle.gpu.copy(b_desc, slot.b, [BLOCK_K, 256], [k_offset, n_offset])
        writer.commit(k_iter)


@libentry()
@triton.jit
def mm_tle_split_m_fused_pipe_kernel(
    a_desc,
    b_desc,
    c_ptr,
    M,
    N,
    stride_cm,
    stride_cn,
    GRID_N: tl.constexpr,
    K_TILES: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
):
    pid = tl.program_id(0)
    m_offset = ((pid // GRID_N) * 320).to(tl.int32)
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
    pipe = tle.pipe(
        capacity=NUM_SLOTS,
        scope="cta",
        name="mm_split_m_fused",
        a=a_smem,
        b=b_smem,
    )
    k_tiles: tl.constexpr = K_TILES
    tle.gpu.warp_specialize(
        [
            (
                _tle_mm_split_m_fused_consumer,
                (
                    pipe.reader(),
                    c_ptr,
                    m_offset,
                    n_offset,
                    stride_cm,
                    stride_cn,
                    M,
                    N,
                    k_tiles,
                    0,
                    256,
                ),
            ),
            (
                _tle_mm_split_m_fused_consumer,
                (
                    pipe.reader(),
                    c_ptr,
                    m_offset,
                    n_offset,
                    stride_cm,
                    stride_cn,
                    M,
                    N,
                    k_tiles,
                    256,
                    64,
                ),
            ),
            (
                _tle_mm_split_m_fused_producer,
                (
                    pipe.writer(),
                    a_desc,
                    b_desc,
                    m_offset,
                    n_offset,
                    k_tiles,
                    BLOCK_K,
                ),
            ),
        ],
        [4, 4],
        [168, 24],
    )


def _get_tle_split384_schedule(wide_warps=False):
    """Return the static TLE partition sizes for the split384 probe.

    ``num_warps`` is the default (top-consumer) partition; worker warp counts
    are appended by the TLE lowering.  Keeping this arithmetic in one Python
    helper makes the experimental CTA-size contract easy to test without
    compiling a MUSA kernel.
    """
    if wide_warps:
        # 16 top consumer + 4 bottom consumer + 4 producer = 24 warps.
        return 16, (4, 4), (168, 24)
    # Existing production route: 16 top + 8 bottom + 4 producer = 28 warps.
    return 16, (8, 4), (168, 24)


@libentry()
@triton.jit
def mm_tle_split384_pipe_kernel(
    a_desc,
    b_desc,
    c_ptr,
    M,
    N,
    stride_cm,
    stride_cn,
    GRID_N: tl.constexpr,
    K_TILES: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    WIDE_WARPS: tl.constexpr,
):
    pid = ext.program_id(0)
    pid_m = pid // GRID_N
    pid_n = pid % GRID_N
    m_offset = (pid_m * 384).to(tl.int32)
    n_offset = (pid_n * 256).to(tl.int32)

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
        capacity=NUM_SLOTS, scope="cta", name="mm_split384_a", a=a_smem
    )
    b_pipe = tle.pipe(
        capacity=NUM_SLOTS, scope="cta", name="mm_split384_b", b=b_smem
    )
    # MuBLASLt's 768-thread 384x256 family has 24 warps per CTA.  The default
    # partition is the 16-warp top consumer; the two worker partitions are the
    # bottom consumer and producer.  Keep the experimental variant at
    # 16 + 4 + 4 = 24 warps.  (The old experimental [16, 8] plus
    # ``num_warps=32`` accidentally requested 56 warps, so it could not model
    # the 768-thread family and often exhausted the allocator.)
    # Keep the choice compile-time so the production 28-warp variant retains
    # its existing register footprint.
    # Keep both the worker list and argument tuples literal: the TLE frontend
    # recognizes JITFunction entries/constexpr arguments only when they are
    # directly nested in warp_specialize's AST.
    if BLOCK_K == 128:
        if WIDE_WARPS:
            tle.gpu.warp_specialize(
                [
                    (_tle_mm_split384_bk128_consumer_top,
                     (a_pipe.reader(), b_pipe.reader(), c_ptr, m_offset,
                      n_offset, stride_cm, stride_cn, M, N, K_TILES)),
                    (_tle_mm_split384_bk128_consumer_bottom,
                     (a_pipe.reader(), b_pipe.reader(), c_ptr, m_offset,
                      n_offset, stride_cm, stride_cn, M, N, K_TILES)),
                    (_tle_mm_split_m_producer,
                     (a_pipe.writer(), b_pipe.writer(), a_desc, b_desc,
                      m_offset, n_offset, K_TILES, 512, BLOCK_K)),
                ],
                [4, 4], [168, 24],
            )
        else:
            tle.gpu.warp_specialize(
                [
                    (_tle_mm_split384_bk128_consumer_top,
                     (a_pipe.reader(), b_pipe.reader(), c_ptr, m_offset,
                      n_offset, stride_cm, stride_cn, M, N, K_TILES)),
                    (_tle_mm_split384_bk128_consumer_bottom,
                     (a_pipe.reader(), b_pipe.reader(), c_ptr, m_offset,
                      n_offset, stride_cm, stride_cn, M, N, K_TILES)),
                    (_tle_mm_split_m_producer,
                     (a_pipe.writer(), b_pipe.writer(), a_desc, b_desc,
                      m_offset, n_offset, K_TILES, 512, BLOCK_K)),
                ],
                [8, 4], [168, 24],
            )
    else:
        if WIDE_WARPS:
            tle.gpu.warp_specialize(
                [
                    (_tle_mm_split_m_consumer_top,
                     (a_pipe.reader(), b_pipe.reader(), c_ptr, m_offset,
                      n_offset, stride_cm, stride_cn, M, N, K_TILES)),
                    (_tle_mm_split_m_consumer_bottom,
                     (a_pipe.reader(), b_pipe.reader(), c_ptr, m_offset,
                      n_offset, stride_cm, stride_cn, M, N, K_TILES, 128)),
                    (_tle_mm_split_m_producer,
                     (a_pipe.writer(), b_pipe.writer(), a_desc, b_desc,
                      m_offset, n_offset, K_TILES, 512, BLOCK_K)),
                ],
                [4, 4], [168, 24],
            )
        else:
            tle.gpu.warp_specialize(
                [
                    (_tle_mm_split_m_consumer_top,
                     (a_pipe.reader(), b_pipe.reader(), c_ptr, m_offset,
                      n_offset, stride_cm, stride_cn, M, N, K_TILES)),
                    (_tle_mm_split_m_consumer_bottom,
                     (a_pipe.reader(), b_pipe.reader(), c_ptr, m_offset,
                      n_offset, stride_cm, stride_cn, M, N, K_TILES, 128)),
                    (_tle_mm_split_m_producer,
                     (a_pipe.writer(), b_pipe.writer(), a_desc, b_desc,
                      m_offset, n_offset, K_TILES, 512, BLOCK_K)),
                ],
                [8, 4], [168, 24],
            )


@triton.jit
def _tle_mm_split128_producer(
    a_writer,
    b_writer,
    a_desc,
    b_desc,
    n_offset,
    K_TILES: tl.constexpr,
):
    for k_iter in range(K_TILES):
        k_offset = k_iter * 64
        a_slot = a_writer.acquire(k_iter)
        b_slot = b_writer.acquire(k_iter)
        tle.gpu.copy(a_desc, a_slot.a, [128, 64], [0, k_offset])
        tle.gpu.copy(b_desc, b_slot.b, [64, 512], [k_offset, n_offset])
        a_writer.commit(k_iter)
        b_writer.commit(k_iter)


@triton.jit
def _tle_mm_split128_consumer(
    a_reader,
    b_reader,
    c_ptr,
    n_offset,
    row_offset: tl.constexpr,
    stride_cm,
    stride_cn,
    M,
    N,
    K_TILES: tl.constexpr,
):
    accumulator = tl.zeros((64, 512), dtype=tl.float32)
    for k_iter in range(K_TILES):
        a_wait = a_reader.wait(k_iter)
        b_wait = b_reader.wait(k_iter)
        a_tile = a_wait.slot.a.slice(row_offset, 64, dim=0)
        accumulator = tle.gpu.wgmma(a_tile, b_wait.slot.b, accumulator)
        accumulator = tle.gpu.wgmma_wait(0, accumulator)
        a_reader.release(k_iter)
        b_reader.release(k_iter)

    offs_m = row_offset + tl.arange(0, 64)
    offs_n = n_offset + tl.arange(0, 512)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit
def mm_tle_split128_pipe_kernel(
    a_desc,
    b_desc,
    c_ptr,
    M,
    N,
    stride_cm,
    stride_cn,
    K_TILES: tl.constexpr,
):
    n_offset = (ext.program_id(0) * 512).to(tl.int32)
    a_smem = tle.gpu.alloc(
        [2, 128, 64],
        dtype=a_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    b_smem = tle.gpu.alloc(
        [2, 64, 512],
        dtype=b_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    a_pipe = tle.pipe(capacity=2, scope="cta", name="mm_split128_a", a=a_smem)
    b_pipe = tle.pipe(capacity=2, scope="cta", name="mm_split128_b", b=b_smem)
    tle.gpu.warp_specialize(
        [
            (
                _tle_mm_split128_consumer,
                (
                    a_pipe.reader(),
                    b_pipe.reader(),
                    c_ptr,
                    n_offset,
                    0,
                    stride_cm,
                    stride_cn,
                    M,
                    N,
                    K_TILES,
                ),
            ),
            (
                _tle_mm_split128_consumer,
                (
                    a_pipe.reader(),
                    b_pipe.reader(),
                    c_ptr,
                    n_offset,
                    64,
                    stride_cm,
                    stride_cn,
                    M,
                    N,
                    K_TILES,
                ),
            ),
            (
                _tle_mm_split128_producer,
                (
                    a_pipe.writer(),
                    b_pipe.writer(),
                    a_desc,
                    b_desc,
                    n_offset,
                    K_TILES,
                ),
            ),
        ],
        [8, 4],
        [168, 24],
    )


@triton.jit
def _tle_mm_split_n_producer(
    a_writer,
    b_left_writer,
    b_right_writer,
    a_desc,
    b_desc,
    n_offset,
    K_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    for k_iter in range(K_TILES):
        k_offset = k_iter * BLOCK_K
        a_slot = a_writer.acquire(k_iter)
        b_left_slot = b_left_writer.acquire(k_iter)
        b_right_slot = b_right_writer.acquire(k_iter)
        tle.gpu.copy(
            a_desc, a_slot.a, [BLOCK_M, BLOCK_K], [0, k_offset]
        )
        tle.gpu.copy(
            b_desc,
            b_left_slot.b,
            [BLOCK_K, 128],
            [k_offset, n_offset],
        )
        tle.gpu.copy(
            b_desc,
            b_right_slot.b,
            [BLOCK_K, 128],
            [k_offset, n_offset + 128],
        )
        a_writer.commit(k_iter)
        b_left_writer.commit(k_iter)
        b_right_writer.commit(k_iter)


@triton.jit
def _tle_mm_split_n_consumer(
    a_reader,
    b_reader,
    c_ptr,
    n_offset,
    column_offset: tl.constexpr,
    stride_cm,
    stride_cn,
    M,
    N,
    K_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    MMA_GROUP: tl.constexpr,
):
    accumulator = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    for group_start in tl.static_range(0, K_TILES, MMA_GROUP):
        for group_offset in tl.static_range(MMA_GROUP):
            k_iter = group_start + group_offset
            if k_iter < K_TILES:
                a_wait = a_reader.wait(k_iter)
                b_wait = b_reader.wait(k_iter)
                accumulator = tle.gpu.wgmma(
                    a_wait.slot.a, b_wait.slot.b, accumulator
                )
        accumulator = tle.gpu.wgmma_wait(0, accumulator)
        for group_offset in tl.static_range(MMA_GROUP):
            k_iter = group_start + group_offset
            if k_iter < K_TILES:
                a_reader.release(k_iter)
                b_reader.release(k_iter)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = n_offset + column_offset + tl.arange(0, 128)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit
def mm_tle_split_n_pipe_kernel(
    a_desc,
    b_desc,
    c_ptr,
    M,
    N,
    stride_cm,
    stride_cn,
    K_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    MMA_GROUP: tl.constexpr,
):
    n_offset = (ext.program_id(0) * 256).to(tl.int32)
    a_smem = tle.gpu.alloc(
        [NUM_SLOTS, BLOCK_M, BLOCK_K],
        dtype=a_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    b_left_smem = tle.gpu.alloc(
        [NUM_SLOTS, BLOCK_K, 128],
        dtype=b_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    b_right_smem = tle.gpu.alloc(
        [NUM_SLOTS, BLOCK_K, 128],
        dtype=b_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    a_pipe = tle.pipe(
        capacity=NUM_SLOTS, scope="cta", name="mm_split_n_a", a=a_smem
    )
    b_left_pipe = tle.pipe(
        capacity=NUM_SLOTS,
        scope="cta",
        name="mm_split_n_b_left",
        b=b_left_smem,
    )
    b_right_pipe = tle.pipe(
        capacity=NUM_SLOTS,
        scope="cta",
        name="mm_split_n_b_right",
        b=b_right_smem,
    )
    tle.gpu.warp_specialize(
        [
            (
                _tle_mm_split_n_consumer,
                (
                    a_pipe.reader(),
                    b_left_pipe.reader(),
                    c_ptr,
                    n_offset,
                    0,
                    stride_cm,
                    stride_cn,
                    M,
                    N,
                    K_TILES,
                    BLOCK_M,
                    MMA_GROUP,
                ),
            ),
            (
                _tle_mm_split_n_consumer,
                (
                    a_pipe.reader(),
                    b_right_pipe.reader(),
                    c_ptr,
                    n_offset,
                    128,
                    stride_cm,
                    stride_cn,
                    M,
                    N,
                    K_TILES,
                    BLOCK_M,
                    MMA_GROUP,
                ),
            ),
            (
                _tle_mm_split_n_producer,
                (
                    a_pipe.writer(),
                    b_left_pipe.writer(),
                    b_right_pipe.writer(),
                    a_desc,
                    b_desc,
                    n_offset,
                    K_TILES,
                    BLOCK_M,
                    BLOCK_K,
                ),
            ),
        ],
        [8, 4],
        [168, 24],
    )


@triton.jit
def _tle_mm_split_n_single_producer(
    writer,
    a_desc,
    b_desc,
    n_offset,
    K_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    for k_iter in range(K_TILES):
        k_offset = k_iter * BLOCK_K
        slot = writer.acquire(k_iter)
        tle.gpu.copy(a_desc, slot.a, [BLOCK_M, BLOCK_K], [0, k_offset])
        tle.gpu.copy(b_desc, slot.b, [BLOCK_K, 256], [k_offset, n_offset])
        writer.commit(k_iter)


@triton.jit
def _tle_mm_split_n_single_consumer(
    reader,
    c_ptr,
    n_offset,
    stride_cm,
    stride_cn,
    M,
    N,
    K_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    MMA_GROUP: tl.constexpr,
):
    accumulator_left = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    accumulator_right = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    for group_start in tl.static_range(0, K_TILES, MMA_GROUP):
        for group_offset in tl.static_range(MMA_GROUP):
            k_iter = group_start + group_offset
            if k_iter < K_TILES:
                ready = reader.wait(k_iter)
                b_left = ready.slot.b.slice(0, 128, dim=1)
                b_right = ready.slot.b.slice(128, 128, dim=1)
                accumulator_left = tle.gpu.wgmma(
                    ready.slot.a, b_left, accumulator_left
                )
                accumulator_right = tle.gpu.wgmma(
                    ready.slot.a, b_right, accumulator_right
                )
        accumulator_left = tle.gpu.wgmma_wait(0, accumulator_left)
        accumulator_right = tle.gpu.wgmma_wait(0, accumulator_right)
        for group_offset in tl.static_range(MMA_GROUP):
            k_iter = group_start + group_offset
            if k_iter < K_TILES:
                reader.release(k_iter)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, 128)
    row_mask = (offs_m < M)[:, None]
    left_n = n_offset + offs_n
    right_n = n_offset + 128 + offs_n
    left_ptrs = c_ptr + offs_m[:, None] * stride_cm + left_n[None, :] * stride_cn
    right_ptrs = c_ptr + offs_m[:, None] * stride_cm + right_n[None, :] * stride_cn
    tl.store(
        left_ptrs,
        accumulator_left.to(c_ptr.dtype.element_ty),
        mask=row_mask & (left_n < N)[None, :],
    )
    tl.store(
        right_ptrs,
        accumulator_right.to(c_ptr.dtype.element_ty),
        mask=row_mask & (right_n < N)[None, :],
    )


@triton.jit
def _tle_mm_split_n_full_consumer(
    reader,
    c_ptr,
    n_offset,
    stride_cm,
    stride_cn,
    M,
    N,
    K_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    MMA_GROUP: tl.constexpr,
):
    """Consume a complete 256-column tile with one SQMMA accumulator.

    The regular single-pipe consumer splits B into two 128-column fields to
    reduce register pressure.  S5000's N=12288 small-M shapes are faster with
    one full-width SQMMA, so keep this as a separate experimental kernel until
    all relevant M values have independent measurements.
    """
    accumulator = tl.zeros((BLOCK_M, 256), dtype=tl.float32)
    for group_start in tl.static_range(0, K_TILES, MMA_GROUP):
        for group_offset in tl.static_range(MMA_GROUP):
            k_iter = group_start + group_offset
            if k_iter < K_TILES:
                ready = reader.wait(k_iter)
                accumulator = tle.gpu.wgmma(
                    ready.slot.a, ready.slot.b, accumulator
                )
        accumulator = tle.gpu.wgmma_wait(0, accumulator)
        for group_offset in tl.static_range(MMA_GROUP):
            k_iter = group_start + group_offset
            if k_iter < K_TILES:
                reader.release(k_iter)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = n_offset + tl.arange(0, 256)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=mask)


@triton.jit
def mm_tle_split_n_full_pipe_kernel(
    a_desc,
    b_desc,
    c_ptr,
    M,
    N,
    stride_cm,
    stride_cn,
    K_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    MMA_GROUP: tl.constexpr,
):
    n_offset = (tl.program_id(0) * 256).to(tl.int32)
    a_smem = tle.gpu.alloc(
        [NUM_SLOTS, BLOCK_M, BLOCK_K], dtype=a_desc.dtype,
        layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=True,
    )
    b_smem = tle.gpu.alloc(
        [NUM_SLOTS, BLOCK_K, 256], dtype=b_desc.dtype,
        layout=None, scope=tle.gpu.smem, nv_mma_shared_layout=True,
    )
    pipe = tle.pipe(
        capacity=NUM_SLOTS, scope="cta", name="mm_split_n_full",
        a=a_smem, b=b_smem,
    )
    tle.gpu.warp_specialize(
        [
            (
                _tle_mm_split_n_full_consumer,
                (
                    pipe.reader(), c_ptr, n_offset, stride_cm, stride_cn,
                    M, N, K_TILES, BLOCK_M, MMA_GROUP,
                ),
            ),
            (
                _tle_mm_split_n_single_producer,
                (pipe.writer(), a_desc, b_desc, n_offset, K_TILES,
                 BLOCK_M, BLOCK_K),
            ),
        ],
        [4],
        [24],
    )


@triton.jit
def mm_tle_split_n_single_pipe_kernel(
    a_desc,
    b_desc,
    c_ptr,
    M,
    N,
    stride_cm,
    stride_cn,
    K_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    MMA_GROUP: tl.constexpr,
):
    n_offset = (tl.program_id(0) * 256).to(tl.int32)
    a_smem = tle.gpu.alloc(
        [NUM_SLOTS, BLOCK_M, BLOCK_K],
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
        name="mm_split_n_single",
        a=a_smem,
        b=b_smem,
    )
    tle.gpu.warp_specialize(
        [
            (
                _tle_mm_split_n_single_consumer,
                (
                    pipe.reader(),
                    c_ptr,
                    n_offset,
                    stride_cm,
                    stride_cn,
                    M,
                    N,
                    K_TILES,
                    BLOCK_M,
                    MMA_GROUP,
                ),
            ),
            (
                _tle_mm_split_n_single_producer,
                (
                    pipe.writer(),
                    a_desc,
                    b_desc,
                    n_offset,
                    K_TILES,
                    BLOCK_M,
                    BLOCK_K,
                ),
            ),
        ],
        [4],
        [24],
    )


@triton.jit
def mm_tle_split_n_wide_single_pipe_kernel(
    a_desc,
    b_desc,
    c_ptr,
    M,
    N,
    stride_cm,
    stride_cn,
    K_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    MMA_GROUP: tl.constexpr,
):
    n_offset = (tl.program_id(0) * 256).to(tl.int32)
    a_smem = tle.gpu.alloc(
        [NUM_SLOTS, BLOCK_M, BLOCK_K],
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
        name="mm_split_n_wide_single",
        a=a_smem,
        b=b_smem,
    )
    tle.gpu.warp_specialize(
        [
            (
                _tle_mm_split_n_single_consumer,
                (
                    pipe.reader(),
                    c_ptr,
                    n_offset,
                    stride_cm,
                    stride_cn,
                    M,
                    N,
                    K_TILES,
                    BLOCK_M,
                    MMA_GROUP,
                ),
            ),
            (
                _tle_mm_split_n_single_producer,
                (
                    pipe.writer(),
                    a_desc,
                    b_desc,
                    n_offset,
                    K_TILES,
                    BLOCK_M,
                    BLOCK_K,
                ),
            ),
        ],
        [8],
        [24],
    )


@triton.jit
def _tle_mm_split256_ordered_producer(
    writer,
    a_desc,
    b_desc,
    m_offset,
    n_offset,
    K_TILES: tl.constexpr,
):
    for k_iter in range(K_TILES):
        k_offset = k_iter * 64
        slot = writer.acquire(k_iter)
        tle.gpu.copy(a_desc, slot.a, [256, 64], [m_offset, k_offset])
        tle.gpu.copy(b_desc, slot.b, [64, 256], [k_offset, n_offset])
        writer.commit(k_iter)


@triton.jit
def _tle_mm_split256_ordered_consumer(
    reader,
    to_top,
    to_bottom,
    c_ptr,
    m_offset,
    n_offset,
    consumer_id: tl.constexpr,
    row_offset: tl.constexpr,
    stride_cm,
    stride_cn,
    M,
    N,
    K_TILES: tl.constexpr,
):
    accumulator = tl.zeros((128, 256), dtype=tl.float32)
    for k_iter in range(K_TILES):
        ready = reader.wait(k_iter)
        if consumer_id == 0:
            tle.gpu.barrier_wait(to_top, phaseIdx=(k_iter + 1) & 1)
        else:
            tle.gpu.barrier_wait(to_bottom, phaseIdx=k_iter & 1)
        a_tile = ready.slot.a.slice(row_offset, 128, dim=0)
        accumulator = tle.gpu.wgmma(a_tile, ready.slot.b, accumulator)
        if consumer_id == 0:
            tle.gpu.barrier_arrive(to_bottom, phaseIdx=k_iter & 1)
        else:
            tle.gpu.barrier_arrive(to_top, phaseIdx=k_iter & 1)
        accumulator = tle.gpu.wgmma_wait(0, accumulator)
        reader.release(k_iter)

    offs_m = m_offset + row_offset + tl.arange(0, 128)
    offs_n = n_offset + tl.arange(0, 256)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit
def mm_tle_split256_ordered_kernel(
    a_desc,
    b_desc,
    c_ptr,
    M,
    N,
    stride_cm,
    stride_cn,
    GRID_M: tl.constexpr,
    K_TILES: tl.constexpr,
):
    pid = ext.program_id(0)
    # Dispatch all M tiles for one N tile back-to-back so adjacent CTAs can
    # reuse B from cache. The original medium-M fast path has GRID_M == 2.
    pid_m = pid % GRID_M
    pid_n = pid // GRID_M
    m_offset = (pid_m * 256).to(tl.int32)
    n_offset = (pid_n * 256).to(tl.int32)

    a_smem = tle.gpu.alloc(
        [3, 256, 64],
        dtype=a_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    b_smem = tle.gpu.alloc(
        [3, 64, 256],
        dtype=b_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    to_top = tle.gpu.alloc_barrier(arrive_count=8, init=tle.gpu.PENDING)
    to_bottom = tle.gpu.alloc_barrier(arrive_count=8, init=tle.gpu.PENDING)
    pipe = tle.pipe(
        capacity=3,
        scope="cta",
        name="mm_split256_ordered",
        a=a_smem,
        b=b_smem,
    )
    tle.gpu.warp_specialize(
        [
            (
                _tle_mm_split256_ordered_consumer,
                (
                    pipe.reader(),
                    to_top,
                    to_bottom,
                    c_ptr,
                    m_offset,
                    n_offset,
                    0,
                    0,
                    stride_cm,
                    stride_cn,
                    M,
                    N,
                    K_TILES,
                ),
            ),
            (
                _tle_mm_split256_ordered_consumer,
                (
                    pipe.reader(),
                    to_top,
                    to_bottom,
                    c_ptr,
                    m_offset,
                    n_offset,
                    1,
                    128,
                    stride_cm,
                    stride_cn,
                    M,
                    N,
                    K_TILES,
                ),
            ),
            (
                _tle_mm_split256_ordered_producer,
                (
                    pipe.writer(),
                    a_desc,
                    b_desc,
                    m_offset,
                    n_offset,
                    K_TILES,
                ),
            ),
        ],
        worker_num_warps=[8, 4],
        worker_num_regs=[168, 24],
    )


@triton.jit
def _tle_mm_persistent_multifield_producer(
    writer,
    a_desc,
    b_desc,
    pid,
    total_tiles,
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
            pid_in_group = tile_id % group_width
            pid_m = group_id * group_m + pid_in_group % group_m
            pid_n = pid_in_group // group_m
            m_offset = (pid_m * 256).to(tl.int32)
            n_offset = (pid_n * 256).to(tl.int32)
            for k_iter in range(k_tiles):
                token = tile_iter * k_tiles + k_iter
                k_offset = k_iter * 64
                slot = writer.acquire(token)
                tle.gpu.copy(
                    a_desc, slot.a, [256, 64], [m_offset, k_offset]
                )
                tle.gpu.copy(
                    b_desc, slot.b, [64, 256], [k_offset, n_offset]
                )
                writer.commit(token)


@triton.jit
def _tle_mm_persistent_multifield_consumer(
    reader,
    c_ptr,
    pid,
    stride_cm,
    stride_cn,
    M,
    N,
    total_tiles,
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
            pid_in_group = tile_id % group_width
            pid_m = group_id * group_m + pid_in_group % group_m
            pid_n = pid_in_group // group_m
            m_offset = (pid_m * 256).to(tl.int32)
            n_offset = (pid_n * 256).to(tl.int32)
            accumulator = tl.zeros((256, 256), dtype=tl.float32)
            for k_iter in range(k_tiles):
                token = tile_iter * k_tiles + k_iter
                ready = reader.wait(token)
                accumulator = tle.gpu.wgmma(
                    ready.slot.a,
                    ready.slot.b,
                    accumulator,
                )
                accumulator = tle.gpu.wgmma_wait(0, accumulator)
                reader.release(token)

            offs_m = m_offset + tl.arange(0, 256)
            offs_n = n_offset + tl.arange(0, 256)
            c_ptrs = (
                c_ptr
                + offs_m[:, None] * stride_cm
                + offs_n[None, :] * stride_cn
            )
            mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
            tl.store(
                c_ptrs,
                accumulator.to(c_ptr.dtype.element_ty),
                mask=mask,
            )

# libentry's benchmark replay can corrupt MUSA pipe barrier state for this
# persistent multifield kernel. Keep its offline-tuned launch parameters fixed.
@triton.jit
def mm_tle_persistent_multifield_kernel(
    a_desc,
    b_desc,
    c_ptr,
    M,
    N,
    K: tl.constexpr,
    stride_cm,
    stride_cn,
    TOTAL_TILES: tl.constexpr,
    GRID_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Persistent tile IDs stay within the launch grid (<= 60 on S5000).
    # Keep them in the native i32 representation instead of the generic
    # extension's i64 conversion; this removes repeated 64-bit index extends
    # from both producer and consumer loops.
    pid = tl.program_id(0)
    a_smem = tle.gpu.alloc(
        [NUM_SLOTS, 256, 64],
        dtype=a_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    b_smem = tle.gpu.alloc(
        [NUM_SLOTS, 64, 256],
        dtype=b_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    pipe = tle.pipe(
        capacity=NUM_SLOTS,
        scope="cta",
        name="mm_persistent_multifield",
        a=a_smem,
        b=b_smem,
    )
    tile_iters: tl.constexpr = tl.cdiv(TOTAL_TILES, NUM_SMS)
    k_tiles: tl.constexpr = K // 64
    tle.gpu.warp_specialize(
        [
            (
                _tle_mm_persistent_multifield_consumer,
                (
                    pipe.reader(),
                    c_ptr,
                    pid,
                    stride_cm,
                    stride_cn,
                    M,
                    N,
                    TOTAL_TILES,
                    GRID_N,
                    NUM_SMS,
                    tile_iters,
                    GROUP_M,
                    k_tiles,
                ),
            ),
            (
                _tle_mm_persistent_multifield_producer,
                (
                    pipe.writer(),
                    a_desc,
                    b_desc,
                    pid,
                    TOTAL_TILES,
                    GRID_N,
                    NUM_SMS,
                    tile_iters,
                    GROUP_M,
                    k_tiles,
                ),
            ),
        ],
        worker_num_warps=[4],
        worker_num_regs=[24],
    )


@triton.jit
def _tle_mm_persistent_16w_producer(
    ab_writer,
    a_desc,
    b_desc,
    pid,
    total_tiles: tl.constexpr,
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
            pid_in_group = tile_id % group_width
            pid_m = group_id * group_m + pid_in_group % group_m
            pid_n = pid_in_group // group_m
            m_offset = (pid_m * 256).to(tl.int32)
            n_offset = (pid_n * 256).to(tl.int32)
            for k_iter in range(k_tiles):
                token = tile_iter * k_tiles + k_iter
                k_offset = k_iter * 64
                slot = ab_writer.acquire(token)
                tle.gpu.copy(
                    a_desc, slot.a, [256, 64], [m_offset, k_offset]
                )
                tle.gpu.copy(
                    b_desc,
                    slot.b,
                    [64, 256],
                    [k_offset, n_offset],
                )
                ab_writer.commit(token)


    drain_base: tl.constexpr = tile_iters * k_tiles
    for drain_offset in tl.static_range(2):
        token = drain_base + drain_offset
        k_offset = drain_offset * 64
        slot = ab_writer.acquire(token)
        tle.gpu.copy(a_desc, slot.a, [256, 64], [0, k_offset])
        tle.gpu.copy(b_desc, slot.b, [64, 256], [k_offset, 0])
        ab_writer.commit(token)


@triton.jit
def _tle_mm_persistent_16w_tail_consumer(
    ab_reader,
    to_tail,
    to_main,
    c_ptr,
    pid,
    M: tl.constexpr,
    N: tl.constexpr,
    total_tiles: tl.constexpr,
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
            pid_in_group = tile_id % group_width
            pid_m = group_id * group_m + pid_in_group % group_m
            pid_n = pid_in_group // group_m
            m_offset = (pid_m * 256).to(tl.int32)
            n_offset = (pid_n * 256).to(tl.int32)
            accumulator = tl.zeros((256, 64), dtype=tl.float32)
            for k_iter in range(k_tiles):
                token = tile_iter * k_tiles + k_iter
                ready = ab_reader.wait(token)
                tle.gpu.barrier_wait(to_tail, phaseIdx=(token + 1) & 1)
                b_tail = ready.slot.b.slice(192, 64, dim=1)
                accumulator = tle.gpu.wgmma(
                    ready.slot.a,
                    b_tail,
                    accumulator,
                )
                tle.gpu.barrier_arrive(to_main, phaseIdx=token & 1)
                accumulator = tle.gpu.wgmma_wait(0, accumulator)
                ab_reader.release(token)

            offs_m = m_offset + tl.arange(0, 256)
            offs_n = n_offset + 192 + tl.arange(0, 64)
            c_ptrs = (
                c_ptr
                + offs_m[:, None] * N
                + offs_n[None, :]
            )
            tl.store(
                c_ptrs,
                accumulator.to(c_ptr.dtype.element_ty),
                mask=(offs_m < M)[:, None],
            )


    final_token: tl.constexpr = tile_iters * k_tiles
    for drain_offset in tl.static_range(2):
        token = final_token + drain_offset
        ab_reader.wait(token)
        ab_reader.release(token)

    if final_token % 2 == 0:
        tle.gpu.barrier_wait(to_tail, phaseIdx=1)
        tle.gpu.barrier_arrive(to_main, phaseIdx=0)


@triton.jit
def _tle_mm_persistent_16w_main_consumer(
    ab_reader,
    to_tail,
    to_main,
    c_ptr,
    pid,
    M: tl.constexpr,
    N: tl.constexpr,
    total_tiles: tl.constexpr,
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
            pid_in_group = tile_id % group_width
            pid_m = group_id * group_m + pid_in_group % group_m
            pid_n = pid_in_group // group_m
            m_offset = (pid_m * 256).to(tl.int32)
            n_offset = (pid_n * 256).to(tl.int32)
            accumulator_main = tl.zeros((256, 128), dtype=tl.float32)
            accumulator_mid = tl.zeros((256, 64), dtype=tl.float32)
            for k_iter in range(k_tiles):
                token = tile_iter * k_tiles + k_iter
                ready = ab_reader.wait(token)
                tle.gpu.barrier_wait(to_main, phaseIdx=token & 1)
                b_main = ready.slot.b.slice(0, 128, dim=1)
                accumulator_main = tle.gpu.wgmma(
                    ready.slot.a,
                    b_main,
                    accumulator_main,
                )
                b_mid = ready.slot.b.slice(128, 64, dim=1)
                accumulator_mid = tle.gpu.wgmma(
                    ready.slot.a,
                    b_mid,
                    accumulator_mid,
                )
                tle.gpu.barrier_arrive(to_tail, phaseIdx=token & 1)
                accumulator_main = tle.gpu.wgmma_wait(0, accumulator_main)
                # Each async SQMMA result must have a matching TLE wait.  The
                # MTT lowering coalesces adjacent zero-pending waits into one
                # hardware flush, so this does not add a second flush.
                accumulator_mid = tle.gpu.wgmma_wait(0, accumulator_mid)
                ab_reader.release(token)

            offs_m = m_offset + tl.arange(0, 256)
            offs_n_main = n_offset + tl.arange(0, 128)
            main_ptrs = (
                c_ptr
                + offs_m[:, None] * N
                + offs_n_main[None, :]
            )
            tl.store(
                main_ptrs,
                accumulator_main.to(c_ptr.dtype.element_ty),
                mask=(offs_m < M)[:, None],
            )

            offs_n_mid = n_offset + 128 + tl.arange(0, 64)
            mid_ptrs = (
                c_ptr
                + offs_m[:, None] * N
                + offs_n_mid[None, :]
            )
            tl.store(
                mid_ptrs,
                accumulator_mid.to(c_ptr.dtype.element_ty),
                mask=(offs_m < M)[:, None],
            )

    final_token: tl.constexpr = tile_iters * k_tiles
    for drain_offset in tl.static_range(2):
        token = final_token + drain_offset
        ab_reader.wait(token)
        ab_reader.release(token)

    if final_token % 2 == 0:
        tle.gpu.barrier_wait(to_main, phaseIdx=0)
        tle.gpu.barrier_arrive(to_tail, phaseIdx=0)


@triton.jit
def mm_tle_persistent_16w_kernel(
    a_desc,
    b_desc,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    TOTAL_TILES: tl.constexpr,
    GRID_N: tl.constexpr,
    NUM_SMS: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    a_smem = tle.gpu.alloc(
        [NUM_SLOTS, 256, 64],
        dtype=a_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    b_smem = tle.gpu.alloc(
        [NUM_SLOTS, 64, 256],
        dtype=b_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    ab_pipe = tle.pipe(
        capacity=NUM_SLOTS,
        scope="cta",
        name="mm_persistent_16w_ab",
        a=a_smem,
        b=b_smem,
    )
    to_tail = tle.gpu.alloc_barrier(arrive_count=8, init=tle.gpu.PENDING)
    to_main = tle.gpu.alloc_barrier(arrive_count=4, init=tle.gpu.PENDING)
    tile_iters: tl.constexpr = tl.cdiv(TOTAL_TILES, NUM_SMS)
    k_tiles: tl.constexpr = K // 64
    tle.gpu.warp_specialize(
        [
            (
                _tle_mm_persistent_16w_tail_consumer,
                (
                    ab_pipe.reader(),
                    to_tail,
                    to_main,
                    c_ptr,
                    pid,
                    M,
                    N,
                    TOTAL_TILES,
                    GRID_N,
                    NUM_SMS,
                    tile_iters,
                    GROUP_M,
                    k_tiles,
                ),
            ),
            (
                _tle_mm_persistent_16w_main_consumer,
                (
                    ab_pipe.reader(),
                    to_tail,
                    to_main,
                    c_ptr,
                    pid,
                    M,
                    N,
                    TOTAL_TILES,
                    GRID_N,
                    NUM_SMS,
                    tile_iters,
                    GROUP_M,
                    k_tiles,
                ),
            ),
            (
                _tle_mm_persistent_16w_producer,
                (
                    ab_pipe.writer(),
                    a_desc,
                    b_desc,
                    pid,
                    TOTAL_TILES,
                    GRID_N,
                    NUM_SMS,
                    tile_iters,
                    GROUP_M,
                    k_tiles,
                ),
            ),
        ],
        worker_num_warps=[8, 4],
        worker_num_regs=[192, 24],
    )
@triton.jit
def _tle_mm_bn320_producer(
    a_writer,
    b_main_writer,
    b_tail_writer,
    a_desc,
    b_main_desc,
    b_tail_desc,
    m_offset,
    n_offset,
    K_TILES: tl.constexpr,
    BLOCK_N_TAIL: tl.constexpr,
):
    for k_iter in range(K_TILES):
        k_offset = k_iter * 64
        a_slot = a_writer.acquire(k_iter)
        b_main_slot = b_main_writer.acquire(k_iter)
        b_tail_slot = b_tail_writer.acquire(k_iter)
        tle.gpu.copy(a_desc, a_slot.a, [256, 64], [m_offset, k_offset])
        tle.gpu.copy(
            b_main_desc,
            b_main_slot.b,
            [64, 256],
            [k_offset, n_offset],
        )
        tle.gpu.copy(
            b_tail_desc,
            b_tail_slot.b,
            [64, BLOCK_N_TAIL],
            [k_offset, n_offset + 256],
        )
        a_writer.commit(k_iter)
        b_main_writer.commit(k_iter)
        b_tail_writer.commit(k_iter)


@triton.jit
def _tle_mm_bn320_consumer(
    a_reader,
    b_main_reader,
    b_tail_reader,
    to_top,
    to_bottom,
    c_ptr,
    m_offset,
    n_offset,
    consumer_id: tl.constexpr,
    row_offset: tl.constexpr,
    BLOCK_M: tl.constexpr,
    stride_cm,
    stride_cn,
    M,
    N,
    K_TILES: tl.constexpr,
    BLOCK_N_TAIL: tl.constexpr,
):
    accumulator_main = tl.zeros((BLOCK_M, 256), dtype=tl.float32)
    accumulator_tail = tl.zeros((BLOCK_M, BLOCK_N_TAIL), dtype=tl.float32)
    for k_iter in range(K_TILES):
        a_ready = a_reader.wait(k_iter)
        b_main_ready = b_main_reader.wait(k_iter)
        b_tail_ready = b_tail_reader.wait(k_iter)
        if consumer_id == 0:
            tle.gpu.barrier_wait(to_top, phaseIdx=(k_iter + 1) & 1)
        else:
            tle.gpu.barrier_wait(to_bottom, phaseIdx=k_iter & 1)
        a_tile = a_ready.slot.a.slice(row_offset, BLOCK_M, dim=0)
        accumulator_main = tle.gpu.wgmma(
            a_tile, b_main_ready.slot.b, accumulator_main
        )
        accumulator_tail = tle.gpu.wgmma(
            a_tile, b_tail_ready.slot.b, accumulator_tail
        )
        if consumer_id == 0:
            tle.gpu.barrier_arrive(to_bottom, phaseIdx=k_iter & 1)
        else:
            tle.gpu.barrier_arrive(to_top, phaseIdx=k_iter & 1)
        accumulator_main = tle.gpu.wgmma_wait(0, accumulator_main)
        accumulator_tail = tle.gpu.wgmma_wait(0, accumulator_tail)
        a_reader.release(k_iter)
        b_main_reader.release(k_iter)
        b_tail_reader.release(k_iter)

    offs_m = m_offset + row_offset + tl.arange(0, BLOCK_M)
    offs_n_main = n_offset + tl.arange(0, 256)
    c_ptrs_main = (
        c_ptr + offs_m[:, None] * stride_cm + offs_n_main[None, :] * stride_cn
    )
    mask_main = (offs_m < M)[:, None] & (offs_n_main < N)[None, :]
    tl.store(
        c_ptrs_main,
        accumulator_main.to(c_ptr.dtype.element_ty),
        mask=mask_main,
    )

    offs_n_tail = n_offset + 256 + tl.arange(0, BLOCK_N_TAIL)
    c_ptrs_tail = (
        c_ptr + offs_m[:, None] * stride_cm + offs_n_tail[None, :] * stride_cn
    )
    mask_tail = (offs_m < M)[:, None] & (offs_n_tail < N)[None, :]
    tl.store(
        c_ptrs_tail,
        accumulator_tail.to(c_ptr.dtype.element_ty),
        mask=mask_tail,
    )


@libentry()
@triton.jit
def mm_tle_bn320_ordered_kernel(
    a_desc,
    b_main_desc,
    b_tail_desc,
    c_ptr,
    M,
    N,
    stride_cm,
    stride_cn,
    GRID_M: tl.constexpr,
    GRID_N: tl.constexpr,
    K_TILES: tl.constexpr,
    BLOCK_M_BOTTOM: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
    BLOCK_N_TAIL: tl.constexpr,
):
    pid = ext.program_id(0)
    group_m: tl.constexpr = min(GRID_M, 2)
    group_width: tl.constexpr = group_m * GRID_N
    group_id = pid // group_width
    first_m = group_id * group_m
    actual_group_m = min(GRID_M - first_m, group_m)
    pid_in_group = pid % group_width
    pid_m = first_m + pid_in_group % actual_group_m
    pid_n = pid_in_group // actual_group_m
    m_offset = (pid_m * (128 + BLOCK_M_BOTTOM)).to(tl.int32)
    n_offset = (pid_n * (256 + BLOCK_N_TAIL)).to(tl.int32)

    a_smem = tle.gpu.alloc(
        [NUM_SLOTS, 256, 64],
        dtype=a_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    b_main_smem = tle.gpu.alloc(
        [NUM_SLOTS, 64, 256],
        dtype=b_main_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    b_tail_smem = tle.gpu.alloc(
        [NUM_SLOTS, 64, BLOCK_N_TAIL],
        dtype=b_tail_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    to_top = tle.gpu.alloc_barrier(arrive_count=8, init=tle.gpu.PENDING)
    to_bottom = tle.gpu.alloc_barrier(arrive_count=8, init=tle.gpu.PENDING)
    a_pipe = tle.pipe(
        capacity=NUM_SLOTS, scope="cta", name="mm_bn320_a", a=a_smem
    )
    b_main_pipe = tle.pipe(
        capacity=NUM_SLOTS, scope="cta", name="mm_bn320_b_main", b=b_main_smem
    )
    b_tail_pipe = tle.pipe(
        capacity=NUM_SLOTS, scope="cta", name="mm_bn320_b_tail", b=b_tail_smem
    )
    tle.gpu.warp_specialize(
        [
            (
                _tle_mm_bn320_consumer,
                (
                    a_pipe.reader(),
                    b_main_pipe.reader(),
                    b_tail_pipe.reader(),
                    to_top,
                    to_bottom,
                    c_ptr,
                    m_offset,
                    n_offset,
                    0,
                    0,
                    128,
                    stride_cm,
                    stride_cn,
                    M,
                    N,
                    K_TILES,
                    BLOCK_N_TAIL,
                ),
            ),
            (
                _tle_mm_bn320_consumer,
                (
                    a_pipe.reader(),
                    b_main_pipe.reader(),
                    b_tail_pipe.reader(),
                    to_top,
                    to_bottom,
                    c_ptr,
                    m_offset,
                    n_offset,
                    1,
                    128,
                    BLOCK_M_BOTTOM,
                    stride_cm,
                    stride_cn,
                    M,
                    N,
                    K_TILES,
                    BLOCK_N_TAIL,
                ),
            ),
            (
                _tle_mm_bn320_producer,
                (
                    a_pipe.writer(),
                    b_main_pipe.writer(),
                    b_tail_pipe.writer(),
                    a_desc,
                    b_main_desc,
                    b_tail_desc,
                    m_offset,
                    n_offset,
                    K_TILES,
                    BLOCK_N_TAIL,
                ),
            ),
        ],
        worker_num_warps=[8, 4],
        worker_num_regs=[168, 24],
    )


@triton.jit
def _tle_mm_bn384_single_producer(
    writer,
    a_desc,
    b_main_desc,
    b_tail_desc,
    m_offset,
    n_offset,
    K_TILES: tl.constexpr,
):
    """Fill one A/B-main/B-tail slot and publish it as one transaction."""
    for k_iter in range(K_TILES):
        k_offset = k_iter * 64
        slot = writer.acquire(k_iter)
        tle.gpu.copy(a_desc, slot.a, [256, 64], [m_offset, k_offset])
        tle.gpu.copy(
            b_main_desc,
            slot.b_main,
            [64, 256],
            [k_offset, n_offset],
        )
        tle.gpu.copy(
            b_tail_desc,
            slot.b_tail,
            [64, 128],
            [k_offset, n_offset + 256],
        )
        writer.commit(k_iter)


@triton.jit
def _tle_mm_bn384_single_consumer(
    reader,
    to_top,
    to_bottom,
    c_ptr,
    m_offset,
    n_offset,
    consumer_id: tl.constexpr,
    row_offset: tl.constexpr,
    BLOCK_M: tl.constexpr,
    stride_cm,
    stride_cn,
    M,
    N,
    K_TILES: tl.constexpr,
):
    """Consume all three payloads with one wait/release per K token."""
    accumulator_main = tl.zeros((BLOCK_M, 256), dtype=tl.float32)
    accumulator_tail = tl.zeros((BLOCK_M, 128), dtype=tl.float32)
    for k_iter in range(K_TILES):
        ready = reader.wait(k_iter)
        if consumer_id == 0:
            tle.gpu.barrier_wait(to_top, phaseIdx=(k_iter + 1) & 1)
        else:
            tle.gpu.barrier_wait(to_bottom, phaseIdx=k_iter & 1)
        a_tile = ready.slot.a.slice(row_offset, BLOCK_M, dim=0)
        accumulator_main = tle.gpu.wgmma(
            a_tile, ready.slot.b_main, accumulator_main
        )
        accumulator_tail = tle.gpu.wgmma(
            a_tile, ready.slot.b_tail, accumulator_tail
        )
        if consumer_id == 0:
            tle.gpu.barrier_arrive(to_bottom, phaseIdx=k_iter & 1)
        else:
            tle.gpu.barrier_arrive(to_top, phaseIdx=k_iter & 1)
        accumulator_main = tle.gpu.wgmma_wait(0, accumulator_main)
        accumulator_tail = tle.gpu.wgmma_wait(0, accumulator_tail)
        reader.release(k_iter)

    offs_m = m_offset + row_offset + tl.arange(0, BLOCK_M)
    offs_n_main = n_offset + tl.arange(0, 256)
    c_ptrs_main = (
        c_ptr + offs_m[:, None] * stride_cm + offs_n_main[None, :] * stride_cn
    )
    mask_main = (offs_m < M)[:, None] & (offs_n_main < N)[None, :]
    tl.store(
        c_ptrs_main,
        accumulator_main.to(c_ptr.dtype.element_ty),
        mask=mask_main,
    )

    offs_n_tail = n_offset + 256 + tl.arange(0, 128)
    c_ptrs_tail = (
        c_ptr + offs_m[:, None] * stride_cm + offs_n_tail[None, :] * stride_cn
    )
    mask_tail = (offs_m < M)[:, None] & (offs_n_tail < N)[None, :]
    tl.store(
        c_ptrs_tail,
        accumulator_tail.to(c_ptr.dtype.element_ty),
        mask=mask_tail,
    )


@libentry()
@triton.jit
def mm_tle_bn384_single_pipe_kernel(
    a_desc,
    b_main_desc,
    b_tail_desc,
    c_ptr,
    M,
    N,
    stride_cm,
    stride_cn,
    GRID_M: tl.constexpr,
    GRID_N: tl.constexpr,
    K_TILES: tl.constexpr,
    BLOCK_M_BOTTOM: tl.constexpr,
    NUM_SLOTS: tl.constexpr,
):
    """Experimental BN384 schedule using one three-field TLE pipe."""
    pid = ext.program_id(0)
    group_m: tl.constexpr = min(GRID_M, 2)
    group_width: tl.constexpr = group_m * GRID_N
    group_id = pid // group_width
    first_m = group_id * group_m
    actual_group_m = min(GRID_M - first_m, group_m)
    pid_in_group = pid % group_width
    pid_m = first_m + pid_in_group % actual_group_m
    pid_n = pid_in_group // actual_group_m
    m_offset = (pid_m * (128 + BLOCK_M_BOTTOM)).to(tl.int32)
    n_offset = (pid_n * 384).to(tl.int32)

    a_smem = tle.gpu.alloc(
        [NUM_SLOTS, 256, 64],
        dtype=a_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    b_main_smem = tle.gpu.alloc(
        [NUM_SLOTS, 64, 256],
        dtype=b_main_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    b_tail_smem = tle.gpu.alloc(
        [NUM_SLOTS, 64, 128],
        dtype=b_tail_desc.dtype,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=True,
    )
    to_top = tle.gpu.alloc_barrier(arrive_count=8, init=tle.gpu.PENDING)
    to_bottom = tle.gpu.alloc_barrier(arrive_count=8, init=tle.gpu.PENDING)
    pipe = tle.pipe(
        capacity=NUM_SLOTS,
        scope="cta",
        name="mm_bn384_single",
        a=a_smem,
        b_main=b_main_smem,
        b_tail=b_tail_smem,
    )
    tle.gpu.warp_specialize(
        [
            (
                _tle_mm_bn384_single_consumer,
                (
                    pipe.reader(),
                    to_top,
                    to_bottom,
                    c_ptr,
                    m_offset,
                    n_offset,
                    0,
                    0,
                    128,
                    stride_cm,
                    stride_cn,
                    M,
                    N,
                    K_TILES,
                ),
            ),
            (
                _tle_mm_bn384_single_consumer,
                (
                    pipe.reader(),
                    to_top,
                    to_bottom,
                    c_ptr,
                    m_offset,
                    n_offset,
                    1,
                    128,
                    BLOCK_M_BOTTOM,
                    stride_cm,
                    stride_cn,
                    M,
                    N,
                    K_TILES,
                ),
            ),
            (
                _tle_mm_bn384_single_producer,
                (
                    pipe.writer(),
                    a_desc,
                    b_main_desc,
                    b_tail_desc,
                    m_offset,
                    n_offset,
                    K_TILES,
                ),
            ),
        ],
        worker_num_warps=[8, 4],
        worker_num_regs=[168, 24],
    )


@libentry()
@libtuner(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=1,
            num_warps=4,
            pre_hook=sqmma_descriptor_pre_hook,
        )
    ],
    key=["M", "N", "K", "dtype"],
    strategy=["align32", "align32", "align32", "default"],
    warmup=5,
    rep=5,
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm_sqmma",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    flagtune_pre_hook=sqmma_descriptor_pre_hook,
)
@triton.jit
def mm_sqmma_kernel(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    dtype: tl.constexpr,
    GROUP_M: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = ext.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    offs_am = (pid_m * BLOCK_M).to(tl.int32)
    offs_bn = (pid_n * BLOCK_N).to(tl.int32)
    offs_k = 0
    offs_k = offs_k.to(tl.int32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load_tensor_descriptor(a_desc, [offs_am, offs_k])
        b = tl.load_tensor_descriptor(b_desc, [offs_k, offs_bn])
        accumulator = tl.dot(a, b, acc=accumulator)
        offs_k += BLOCK_K
    tl.store_tensor_descriptor(c_desc, [offs_am, offs_bn], accumulator.to(c_desc.dtype))


def mm_sqmma(A, B, M, N, K):
    logger.debug("GEMS_MTHREADS MM_SQMMA")
    device = A.device
    if not A.is_contiguous():
        A = A.contiguous()
    if not B.is_contiguous():
        B = B.contiguous()
    a_type = A.dtype
    b_type = B.dtype
    assert a_type == b_type, "Mat A and Mat B should have the same dtype"
    c_dtype = get_higher_dtype(a_type, b_type)
    C = torch.empty((M, N), dtype=c_dtype, device=device)
    desc_a = TensorDescriptor.from_tensor(A, [1, 1])
    desc_b = TensorDescriptor.from_tensor(B, [1, 1])
    desc_c = TensorDescriptor.from_tensor(C, [1, 1])
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
        1,
        1,
    )
    mm_sqmma_kernel[grid](
        desc_a,
        desc_b,
        desc_c,
        M,
        N,
        K,
        str(a_type).split(".")[-1],
    )
    return C


def is_tle_pipe_compatible(a, b, M, N, K):
    return (
        HAS_MTHREADS_REGISTER_FAILURE_RECOVERY
        and HAS_MTHREADS_TLE_PIPE_SQMMA
        and HAS_MTHREADS_TLE_MULTIFIELD_PIPE
        and HAS_MTHREADS_TLE_DYNAMIC_LOOPS
        and a.dtype == b.dtype
        and a.dtype in (torch.float16, torch.bfloat16)
        and a.is_contiguous()
        and b.is_contiguous()
        # The generic rolling TLE kernel carries a full accumulator and pipe
        # state even for tiny row tiles.  On MTT this exceeds the available
        # register classes (and llc aborts) for M<32; specialized split-N
        # routes are checked before this predicate for the wide-N families.
        and M >= 32
        and N > 1
        # MTT SQMMA descriptors require the output-N dimension to be 8-wide.
        # Keep non-conforming inputs on the generic backend path instead of
        # allowing a compile/runtime layout failure.
        and N % 8 == 0
        and K == 2048
    )


def is_tle_split_m_compatible(M, N, K):
    return (
        HAS_MTHREADS_TLE_SPLIT_M_SQMMA
        and (
            (4096 < M <= 4480 and N == 1024 and K == 2048)
            or (2048 <= M <= 16384 and N == 2048 and K == 512)
        )
    )


def is_tle_split_m_fused_compatible(M, N, K):
    return (
        HAS_MTHREADS_TLE_SPLIT_M_SQMMA
        and M in (4138, 4434, 4435)
        and N == 2048
        and K == 512
    )


def is_tle_split384_compatible(M, N, K):
    return (
        HAS_MTHREADS_TLE_SPLIT_M_SQMMA
        and M == 16384
        and (
            (N == 2048 and K == 512)
            or (N in (1024, 9216, 12288) and K == 2048)
        )
    )


def is_tle_split128_compatible(M, N, K):
    return (
        HAS_MTHREADS_TLE_SPLIT128_SQMMA
        and M == 100
        and N == 248320
        and K == 2048
    )


def is_tle_split_n_compatible(M, N, K):
    return (
        HAS_MTHREADS_TLE_SPLIT128_SQMMA
        and 4 <= M <= 256
        and N in (9216, 12288)
        and K == 2048
    )


def is_tle_split_n_single_compatible(M, N, K, dtype=None):
    return (
        HAS_MTHREADS_TLE_SPLIT_N_SINGLE_PIPE
        and HAS_MTHREADS_TLE_MULTIFIELD_PIPE
        and dtype in (torch.bfloat16, torch.float16)
        # The fixed 64-row consumer only amortizes its TLE launch/barrier and
        # dispatch cost for the already validated M=64 wide-N shape.  Keep
        # neighboring ragged M values on the generic split-N route until the
        # full mm_out dispatch path is independently validated.
        and M == 64
        and N == 12288
        and K == 2048
    )


def is_tle_split_n_small_compatible(M, N, K, dtype=None):
    """Use one low-register 16-row consumer for decode-sized GEMMs.

    The regular rolling kernel carries a full BLOCK_M accumulator and is not
    viable for M<32 on MTT.  The split-N producer/consumer already has the
    required two 128-column SQMMA fields; with BLOCK_M=16 it keeps the live
    range small enough for the allocator and amortizes the launch over the
    wide N tiles.  Tensor descriptors are not tail-safe for arbitrary N on
    the current TLE ABI, hence the exact 256-column alignment guard.
    """
    return (
        HAS_MTHREADS_REGISTER_FAILURE_RECOVERY
        and HAS_MTHREADS_TLE_SPLIT_N_SINGLE_PIPE
        and HAS_MTHREADS_TLE_MULTIFIELD_PIPE
        and dtype in (torch.bfloat16, torch.float16)
        and 1 <= M <= 16
        and N >= 256
        and N % 256 == 0
        and K % 64 == 0
    )


def is_tle_split_n_full_compatible(M, N, K, dtype=None):
    """Exact guard for the experimentally validated full-B tile family."""
    return (
        HAS_MTHREADS_TLE_SPLIT_N_SINGLE_PIPE
        and HAS_MTHREADS_TLE_MULTIFIELD_PIPE
        and dtype in (torch.bfloat16, torch.float16)
        and M in (48, 56, 64)
        and N == 12288
        and K == 2048
    )


def is_tle_split_n_wide_single_compatible(M, N, K, dtype=None):
    """Guard the M=128 one-pipe schedule to measured shape families.

    A wide tile raises register/shared-memory pressure substantially, so it
    must not become a generic replacement for the existing split-N path.
    Keeping this predicate exact also lets older FlagTree versions safely
    fall back to the generic implementation when the pipe feature is absent.
    """
    return (
        HAS_MTHREADS_TLE_SPLIT_N_SINGLE_PIPE
        and HAS_MTHREADS_TLE_MULTIFIELD_PIPE
        and dtype in (torch.bfloat16, torch.float16)
        and M == 128
        and N in (9216, 12288)
        and K == 2048
    )


def is_tle_split256_ordered_compatible(M, N, K):
    return (
        HAS_MTHREADS_TLE_PERSISTENT_ORDERED_SQMMA
        and HAS_MTHREADS_TLE_DYNAMIC_PARTITION_SYNC
        and (
            (256 < M <= 496 and N in (9216, 12288) and K == 2048)
            or (M == 2048 and N == 1024 and K == 2048)
            or (512 <= M <= 1040 and N == 2048 and K == 512)
        )
    )


def is_tle_bn320_compatible(M, N, K):
    return (
        HAS_MTHREADS_TLE_BN320_SQMMA
        and HAS_MTHREADS_TLE_DYNAMIC_PARTITION_SYNC
        and 256 < M <= 512
        and N == 9216
        and K == 2048
    )


def is_tle_persistent_multifield_compatible(M, N, K, dtype=None):
    return (
        HAS_MTHREADS_TLE_MULTIFIELD_PIPE
        and HAS_MTHREADS_TLE_PERSISTENT_ORDERED_SQMMA
        and (
            (
                M == 512
                and N == 12288
                and K == 2048
                and dtype in (None, torch.bfloat16, torch.float16)
            )
            or (
                M == 384
                and N == 4096
                and K == 1024
                and dtype in (torch.bfloat16, torch.float16)
            )
            or (
                M == 16384
                and N == 1024
                and K == 2048
                and dtype in (torch.bfloat16, torch.float16)
            )
            or (
                dtype in (torch.bfloat16, torch.float16)
                and M == 16384
                and N in (9216, 12288)
                and K == 2048
            )
        )
    )


def is_tle_persistent_16w_compatible(M, N, K, dtype=None):
    return (
        HAS_MTHREADS_TLE_16_WARP_PERSISTENT
        and dtype in (None, torch.bfloat16)
        and M == 16384
        and N == 12288
        and K == 2048
    )


def get_tle_bn320_block_m_bottom(M):
    return 64 if M <= 384 else 128


def use_tle_ws_pipe(M, N, K):
    if M < 512 or N <= 256:
        return False
    if N == 512 and K == 4096 and (M <= 1040 or M == 2048):
        return False
    if N == 1024 and K == 2048 and M <= 1040:
        return False
    if N == 4096 and K in (128, 1024) and M <= 512:
        # PH1 BF16/FP16 K=128 is still lowered as two K=64 operations, and
        # K=1024 has the same fixed-partition overhead for low-M trace
        # buckets. Keep both families on the non-WS route (including the
        # M=512 boundary) until a native schedule is validated independently.
        return False
    return True


def mm_tle_split_m_pipe(
    a,
    b,
    c,
    M,
    N,
    K,
    *,
    block_m_storage=512,
    block_k=32,
    num_slots=4,
):
    desc_a = TensorDescriptor.from_tensor(a, [block_m_storage, block_k])
    desc_b = TensorDescriptor.from_tensor(b, [block_k, 256])
    grid_n = triton.cdiv(N, 256)
    grid = (triton.cdiv(M, 320) * grid_n,)
    logger.debug(
        "GEMS_MTHREADS MM_TLE_SPLIT_M_PIPE, [shape info]: [%s, %s, %s](M, N, K)",
        M,
        N,
        K,
    )
    with torch_device_fn.device(a.device):
        mm_tle_split_m_pipe_kernel[grid](
            desc_a,
            desc_b,
            c,
            M,
            N,
            c.stride(0),
            c.stride(1),
            grid_n,
            triton.cdiv(K, block_k),
            block_m_storage,
            block_k,
            num_slots,
            num_warps=16,
            enable_backend_opt=True,
        )
    return c


def mm_tle_split_m_fused_pipe(a, b, c, M, N, K, *, block_k=32, num_slots=4):
    desc_a = TensorDescriptor.from_tensor(a, [512, block_k])
    desc_b = TensorDescriptor.from_tensor(b, [block_k, 256])
    grid_n = triton.cdiv(N, 256)
    grid = (triton.cdiv(M, 320) * grid_n,)
    logger.debug(
        "GEMS_MTHREADS MM_TLE_SPLIT_M_FUSED_PIPE, "
        "[shape info]: [%s, %s, %s](M, N, K)",
        M,
        N,
        K,
    )
    with torch_device_fn.device(a.device):
        mm_tle_split_m_fused_pipe_kernel[grid](
            desc_a,
            desc_b,
            c,
            M,
            N,
            c.stride(0),
            c.stride(1),
            grid_n,
            triton.cdiv(K, block_k),
            block_k,
            num_slots,
            num_warps=16,
            enable_backend_opt=True,
        )
    return c


def mm_tle_split384_pipe(
    a,
    b,
    c,
    M,
    N,
    K,
    *,
    block_k=32,
    num_slots=3,
    wide_warps=False,
):
    desc_a = TensorDescriptor.from_tensor(a, [512, block_k])
    desc_b = TensorDescriptor.from_tensor(b, [block_k, 256])
    grid_n = triton.cdiv(N, 256)
    grid = (triton.cdiv(M, 384) * grid_n,)
    default_num_warps, _worker_warps, _worker_regs = _get_tle_split384_schedule(
        wide_warps
    )
    with torch_device_fn.device(a.device):
        mm_tle_split384_pipe_kernel[grid](
            desc_a,
            desc_b,
            c,
            M,
            N,
            c.stride(0),
            c.stride(1),
            grid_n,
            triton.cdiv(K, block_k),
            block_k,
            num_slots,
            wide_warps,
            # Both variants use the 16-warp default partition.  The
            # experimental 24-warp total is formed by its [4, 4] workers;
            # passing 32 here would inflate the CTA to 56 warps.
            num_warps=default_num_warps,
            enable_backend_opt=True,
        )
    return c


def mm_tle_split128_pipe(a, b, c, M, N, K):
    desc_a = TensorDescriptor.from_tensor(a, [128, 64])
    desc_b = TensorDescriptor.from_tensor(b, [64, 512])
    grid = (triton.cdiv(N, 512),)
    logger.debug(
        "GEMS_MTHREADS MM_TLE_SPLIT128_PIPE, [shape info]: [%s, %s, %s](M, N, K)",
        M,
        N,
        K,
    )
    with torch_device_fn.device(a.device):
        mm_tle_split128_pipe_kernel[grid](
            desc_a,
            desc_b,
            c,
            M,
            N,
            c.stride(0),
            c.stride(1),
            triton.cdiv(K, 64),
            num_warps=16,
            enable_backend_opt=True,
        )
    return c


def mm_tle_split_n_pipe(
    a,
    b,
    c,
    M,
    N,
    K,
    *,
    block_k=64,
    num_slots=2,
    mma_group=1,
):
    if M <= 64:
        block_m = 64
    elif M <= 128:
        block_m = 128
    else:
        block_m = 256
    desc_a = TensorDescriptor.from_tensor(a, [block_m, block_k])
    desc_b = TensorDescriptor.from_tensor(b, [block_k, 128])
    grid = (triton.cdiv(N, 256),)
    logger.debug(
        "GEMS_MTHREADS MM_TLE_SPLIT_N_PIPE, [shape info]: [%s, %s, %s](M, N, K)",
        M,
        N,
        K,
    )
    with torch_device_fn.device(a.device):
        mm_tle_split_n_pipe_kernel[grid](
            desc_a,
            desc_b,
            c,
            M,
            N,
            c.stride(0),
            c.stride(1),
            triton.cdiv(K, block_k),
            block_m,
            block_k,
            num_slots,
            mma_group,
            num_warps=8,
            enable_backend_opt=True,
        )
    return c


def mm_tle_split_n_single_pipe(
    a, b, c, M, N, K, *, block_m=64, block_k=64, num_slots=3
):
    desc_a = TensorDescriptor.from_tensor(a, [block_m, block_k])
    desc_b = TensorDescriptor.from_tensor(b, [block_k, 256])
    grid = (triton.cdiv(N, 256),)
    logger.debug(
        "GEMS_MTHREADS MM_TLE_SPLIT_N_SINGLE_PIPE, "
        "[shape info]: [%s, %s, %s](M, N, K)",
        M,
        N,
        K,
    )
    with torch_device_fn.device(a.device):
        mm_tle_split_n_single_pipe_kernel[grid](
            desc_a,
            desc_b,
            c,
            M,
            N,
            c.stride(0),
            c.stride(1),
            triton.cdiv(K, block_k),
            block_m,
            block_k,
            num_slots,
            1,
            # muDNN uses a 4-warp tile for this M=64 case.  Three slots give
            # the producer enough distance to overlap TMA and SQMMA on the
            # validated N=12288 shape; four slots reduce occupancy on S5000.
            num_warps=4,
            enable_backend_opt=True,
        )
    return c


def mm_tle_split_n_small_pipe(a, b, c, M, N, K):
    """Low-live-range split-N launch for MTT decode-sized small-M GEMMs."""
    return mm_tle_split_n_single_pipe(
        a, b, c, M, N, K, block_m=16, block_k=64, num_slots=3
    )


def mm_tle_split_n_full_pipe(
    a, b, c, M, N, K, *, block_k=64, num_slots=3, mma_group=1
):
    """Experimental full-B single-pipe wrapper; callers must use its guard."""
    block_m = 64
    desc_a = TensorDescriptor.from_tensor(a, [block_m, block_k])
    desc_b = TensorDescriptor.from_tensor(b, [block_k, 256])
    grid = (triton.cdiv(N, 256),)
    with torch_device_fn.device(a.device):
        mm_tle_split_n_full_pipe_kernel[grid](
            desc_a,
            desc_b,
            c,
            M,
            N,
            c.stride(0),
            c.stride(1),
            triton.cdiv(K, block_k),
            block_m,
            block_k,
            num_slots,
            mma_group,
            num_warps=4,
            enable_backend_opt=True,
        )
    return c


def mm_tle_split_n_wide_single_pipe(
    a, b, c, M, N, K, *, block_k=64, num_slots=2
):
    """Experimental M=128 one-pipe wrapper.

    This is intentionally kept out of ``mm_tle_pipe`` until an MThreads
    benchmark confirms that the wider accumulator beats the existing split-N
    implementation.  Callers must satisfy
    ``is_tle_split_n_wide_single_compatible`` themselves.
    """
    block_m = 128
    desc_a = TensorDescriptor.from_tensor(a, [block_m, block_k])
    desc_b = TensorDescriptor.from_tensor(b, [block_k, 256])
    grid = (triton.cdiv(N, 256),)
    logger.debug(
        "GEMS_MTHREADS MM_TLE_SPLIT_N_WIDE_SINGLE_PIPE, "
        "[shape info]: [%s, %s, %s](M, N, K)",
        M,
        N,
        K,
    )
    with torch_device_fn.device(a.device):
        mm_tle_split_n_wide_single_pipe_kernel[grid](
            desc_a,
            desc_b,
            c,
            M,
            N,
            c.stride(0),
            c.stride(1),
            triton.cdiv(K, block_k),
            block_m,
            block_k,
            num_slots,
            1,
            num_warps=8,
            enable_backend_opt=True,
        )
    return c


def mm_tle_split256_ordered(a, b, c, M, N, K):
    desc_a = TensorDescriptor.from_tensor(a, [256, 64])
    desc_b = TensorDescriptor.from_tensor(b, [64, 256])
    grid_m = triton.cdiv(M, 256)
    grid_n = triton.cdiv(N, 256)
    grid = (grid_m * grid_n,)
    logger.debug(
        "GEMS_MTHREADS MM_TLE_SPLIT256_ORDERED, "
        "[shape info]: [%s, %s, %s](M, N, K)",
        M,
        N,
        K,
    )
    with torch_device_fn.device(a.device):
        mm_tle_split256_ordered_kernel[grid](
            desc_a,
            desc_b,
            c,
            M,
            N,
            c.stride(0),
            c.stride(1),
            grid_m,
            triton.cdiv(K, 64),
            num_warps=8,
            enable_backend_opt=True,
        )
    return c


def mm_tle_bn320_ordered(
    a,
    b,
    c,
    M,
    N,
    K,
    *,
    num_slots=2,
    block_m_bottom=None,
    block_n_tail=64,
    num_stages=None,
):
    desc_a = TensorDescriptor.from_tensor(a, [256, 64])
    desc_b_main = TensorDescriptor.from_tensor(b, [64, 256])
    desc_b_tail = TensorDescriptor.from_tensor(b, [64, block_n_tail])
    if block_m_bottom is None:
        block_m_bottom = get_tle_bn320_block_m_bottom(M)
    grid_m = triton.cdiv(M, 128 + block_m_bottom)
    grid_n = triton.cdiv(N, 256 + block_n_tail)
    grid = (grid_m * grid_n,)
    logger.debug(
        "GEMS_MTHREADS MM_TLE_BN320_ORDERED, "
        "[shape info]: [%s, %s, %s](M, N, K)",
        M,
        N,
        K,
    )
    with torch_device_fn.device(a.device):
        launch_kwargs = dict(
            num_warps=8,
            enable_backend_opt=True,
        )
        if num_stages is not None:
            launch_kwargs["num_stages"] = num_stages
        mm_tle_bn320_ordered_kernel[grid](
            desc_a,
            desc_b_main,
            desc_b_tail,
            c,
            M,
            N,
            c.stride(0),
            c.stride(1),
            grid_m,
            grid_n,
            triton.cdiv(K, 64),
            block_m_bottom,
            num_slots,
            block_n_tail,
            **launch_kwargs,
        )
    return c


def mm_tle_bn384_single_pipe(
    a,
    b,
    c,
    M,
    N,
    K,
    *,
    num_slots=2,
    block_m_bottom=None,
):
    """Experimental A+B-main256+B-tail128 single-pipe launch.

    This entry point deliberately has no dispatch predicate yet.  It is used
    by focused correctness/compile probes while the three-field schedule is
    being compared with the existing BN320 and muDNN paths.
    """
    desc_a = TensorDescriptor.from_tensor(a, [256, 64])
    desc_b_main = TensorDescriptor.from_tensor(b, [64, 256])
    desc_b_tail = TensorDescriptor.from_tensor(b, [64, 128])
    if block_m_bottom is None:
        block_m_bottom = get_tle_bn320_block_m_bottom(M)
    grid_m = triton.cdiv(M, 128 + block_m_bottom)
    grid_n = triton.cdiv(N, 384)
    grid = (grid_m * grid_n,)
    logger.debug(
        "GEMS_MTHREADS MM_TLE_BN384_SINGLE_PIPE, "
        "[shape info]: [%s, %s, %s](M, N, K)",
        M,
        N,
        K,
    )
    with torch_device_fn.device(a.device):
        mm_tle_bn384_single_pipe_kernel[grid](
            desc_a,
            desc_b_main,
            desc_b_tail,
            c,
            M,
            N,
            c.stride(0),
            c.stride(1),
            grid_m,
            grid_n,
            triton.cdiv(K, 64),
            block_m_bottom,
            num_slots,
            num_warps=8,
            enable_backend_opt=True,
        )
    return c


def mm_tle_persistent_multifield(a, b, c, M, N, K):
    desc_a = TensorDescriptor.from_tensor(a, [256, 64])
    desc_b = TensorDescriptor.from_tensor(b, [64, 256])
    grid_m = triton.cdiv(M, 256)
    grid_n = triton.cdiv(N, 256)
    total_tiles = grid_m * grid_n
    # Wide workloads have enough tiles to keep all 60 MTT SMs busy; split384
    # under-fills the device for the 9216-column case.  The 512x12288 family
    # also benefits from the 60-CTA persistent schedule; retain two slots to
    # stay within the current TLE shared-memory budget.
    is_large_persistent = (
        M == 16384 and N in (9216, 12288) and K == 2048
    )
    # muDNN uses a persistent 256x256/BK64 schedule for this ragged decode
    # bucket. The generic rolling route exhausts the MTGPU register class for
    # it; this exact shape has been checked for BF16/FP16 correctness.
    is_ragged_384_persistent = M == 384 and N == 4096 and K == 1024
    is_narrow_16384_persistent = M == 16384 and N == 1024 and K == 2048
    is_medium_persistent = M == 512 and N == 12288 and K == 2048
    num_sms = (
        60
        if (is_large_persistent or is_medium_persistent or is_narrow_16384_persistent)
        else 48
    )
    num_slots = (
        3
        if (is_large_persistent or is_ragged_384_persistent or is_narrow_16384_persistent)
        else 2
    )
    grid = (num_sms,)
    logger.debug(
        "GEMS_MTHREADS MM_TLE_PERSISTENT_MULTIFIELD, "
        "[shape info]: [%s, %s, %s](M, N, K)",
        M,
        N,
        K,
    )
    with torch_device_fn.device(a.device):
        mm_tle_persistent_multifield_kernel[grid](
            desc_a,
            desc_b,
            c,
            M,
            N,
            K,
            c.stride(0),
            c.stride(1),
            total_tiles,
            grid_n,
            num_sms,
            num_slots,
            2,
            num_warps=16,
            enable_backend_opt=True,
        )
        # Triton's stream is not visible to the MUSA caching allocator here.
        c.record_stream(torch_device_fn.current_stream(a.device))
    return c


def mm_tle_persistent_16w(a, b, c, M, N, K, *, llc_options=None):
    desc_a = TensorDescriptor.from_tensor(a, [256, 64])
    desc_b = TensorDescriptor.from_tensor(b, [64, 256])
    grid_m = triton.cdiv(M, 256)
    grid_n = triton.cdiv(N, 256)
    total_tiles = grid_m * grid_n
    num_sms = 60
    launch_tiles = triton.cdiv(total_tiles, num_sms) * num_sms
    grid = (num_sms,)
    logger.debug(
        "GEMS_MTHREADS MM_TLE_PERSISTENT_16W, "
        "[shape info]: [%s, %s, %s](M, N, K)",
        M,
        N,
        K,
    )
    with torch_device_fn.device(a.device):
        mm_tle_persistent_16w_kernel[grid](
            desc_a,
            desc_b,
            c,
            M,
            N,
            K,
            launch_tiles,
            grid_n,
            num_sms,
            2,
            2,
            num_warps=4,
            enable_backend_opt=True,
            disable_max_ilp_scheduler=True,
            llc_options=llc_options,
        )
        c.record_stream(torch_device_fn.current_stream(a.device))
    return c


def mm_tle_pipe(a, b, c, M, N, K):
    # MuBLASLt exposes a 768-thread 384x256 persistent family for this
    # narrow-N bucket.  The existing split384 TLE kernel has the same
    # two-consumer topology and is materially faster than BM256 multifield
    # here (the latter remains a fallback when split384 is unavailable).
    if (
        M == 16384
        and N == 1024
        and K == 2048
        and is_tle_split384_compatible(M, N, K)
    ):
        return mm_tle_split384_pipe(a, b, c, M, N, K)
    if is_tle_persistent_multifield_compatible(
        M, N, K, getattr(a, "dtype", None)
    ):
        return mm_tle_persistent_multifield(a, b, c, M, N, K)
    if is_tle_persistent_16w_compatible(M, N, K, getattr(a, "dtype", None)):
        return mm_tle_persistent_16w(a, b, c, M, N, K)
    if is_tle_bn320_compatible(M, N, K):
        return mm_tle_bn320_ordered(a, b, c, M, N, K)
    if is_tle_split128_compatible(M, N, K):
        return mm_tle_split128_pipe(a, b, c, M, N, K)
    if is_tle_split_n_full_compatible(M, N, K, getattr(a, "dtype", None)):
        return mm_tle_split_n_full_pipe(a, b, c, M, N, K)
    if is_tle_split_n_single_compatible(M, N, K, getattr(a, "dtype", None)):
        return mm_tle_split_n_single_pipe(a, b, c, M, N, K)
    if is_tle_split_n_compatible(M, N, K):
        return mm_tle_split_n_pipe(a, b, c, M, N, K)
    if is_tle_split256_ordered_compatible(M, N, K):
        return mm_tle_split256_ordered(a, b, c, M, N, K)
    if is_tle_split384_compatible(M, N, K):
        return mm_tle_split384_pipe(a, b, c, M, N, K)
    if is_tle_split_m_fused_compatible(M, N, K):
        return mm_tle_split_m_fused_pipe(a, b, c, M, N, K)
    if is_tle_split_m_compatible(M, N, K):
        return mm_tle_split_m_pipe(a, b, c, M, N, K)
    desc_a = TensorDescriptor.from_tensor(a, [1, 1])
    desc_b = TensorDescriptor.from_tensor(b, [1, 1])
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    logger.debug(
        "GEMS_MTHREADS MM_TLE_PIPE, [shape info]: [%s, %s, %s](M, N, K)",
        M,
        N,
        K,
    )
    with torch_device_fn.device(a.device):
        kernel = mm_tle_pipe_kernel if use_tle_ws_pipe(M, N, K) else mm_tle_non_ws_pipe_kernel
        kernel[grid](
            desc_a,
            desc_b,
            c,
            M,
            N,
            K,
            c.stride(0),
            c.stride(1),
            enable_backend_opt=True,
        )
    return c


def mm(a, b):
    a_dtype = a.dtype
    M, K = a.shape
    _, N = b.shape
    if N == 1:
        c_dtype = get_higher_dtype(a_dtype, b.dtype)
        c = torch.empty((M, N), device=a.device, dtype=c_dtype)
        return gemv_mm(a, b, c, M, K)
    if is_tle_split_n_small_compatible(M, N, K, a_dtype):
        c = torch.empty((M, N), device=a.device, dtype=a_dtype)
        return mm_tle_split_n_small_pipe(a, b, c, M, N, K)
    # The experimental scalar small-M kernel is intentionally not dispatched:
    # it uses FP32 scalar FMAs instead of SQMMA and measured only 0.20-0.42x
    # Torch on the Qwen M<32 buckets. Keep the implementation available for
    # future compiler-backed variants, but use the normal path in production.
    if is_tle_pipe_compatible(a, b, M, N, K):
        c = torch.empty((M, N), device=a.device, dtype=a_dtype)
        return mm_tle_pipe(a, b, c, M, N, K)
    return _generic_mm(a, b)
