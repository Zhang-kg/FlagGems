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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

_CATALOG_MNK = {
    (64, 4, 16384),
    (16384, 1, 4096),
}


def _supported_b_layout(tensor, k):
    return tensor.is_contiguous() or tensor.stride() == (1, k)


def _is_one(value):
    try:
        return float(value) == 1.0
    except (TypeError, ValueError):
        return False


def _is_eligible(bias, mat1, mat2, alpha, beta, out=None):
    if (
        not _is_one(alpha)
        or not _is_one(beta)
        or mat1.ndim != 2
        or mat2.ndim != 2
        or bias.ndim != 2
        or mat1.shape[1] != mat2.shape[0]
        or (mat1.shape[0], mat2.shape[1], mat1.shape[1]) not in _CATALOG_MNK
        or tuple(bias.shape) != (mat1.shape[0], mat2.shape[1])
        or mat1.dtype not in (torch.float16, torch.bfloat16)
        or bias.dtype != mat1.dtype
        or mat2.dtype != mat1.dtype
        or not bias.is_contiguous()
        or not mat1.is_contiguous()
        or not _supported_b_layout(mat2, mat1.shape[1])
        or bias.device != mat1.device
        or mat2.device != mat1.device
        or mat1.device.type != "musa"
    ):
        return False
    return out is None or (
        tuple(out.shape) == tuple(bias.shape)
        and out.dtype == mat1.dtype
        and out.device == mat1.device
        and out.is_contiguous()
        and all(not torch._C._overlaps(out, tensor) for tensor in (bias, mat1, mat2))
    )


@libentry()
@triton.jit(do_not_specialize=["alpha", "beta"])
def _addmm_strided_b_kernel(
    bias,
    mat1,
    mat2,
    out,
    alpha,
    beta,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_im: tl.constexpr,
    stride_in: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n
    pid_n = pid % num_n

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    mask_m = offsets_m < M
    mask_n = offsets_n < N
    a_ptrs = mat1 + offsets_m[:, None] * stride_am + offsets_k[None, :] * stride_ak
    b_ptrs = mat2 + offsets_k[:, None] * stride_bk + offsets_n[None, :] * stride_bn
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = k_start * BLOCK_K + offsets_k < K
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias_ptrs = bias + offsets_m[:, None] * stride_im + offsets_n[None, :] * stride_in
    bias_tile = tl.load(
        bias_ptrs,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    )
    out_ptrs = out + offsets_m[:, None] * N + offsets_n[None, :]
    tl.store(
        out_ptrs,
        alpha * accumulator + beta * bias_tile,
        mask=mask_m[:, None] & mask_n[None, :],
    )


def _config(m, n, k):
    if (m, n, k) == (64, 4, 16384):
        return (64, 16, 128, 4)
    return (128, 16, 64, 4)


def try_addmm(bias, mat1, mat2, out=None, *, alpha=1.0, beta=1.0):
    if not _is_eligible(bias, mat1, mat2, alpha, beta, out):
        return None
    m, k = mat1.shape
    n = mat2.shape[1]
    if out is None:
        out = torch.empty((m, n), dtype=mat1.dtype, device=mat1.device)
    block_m, block_n, block_k, num_warps = _config(m, n, k)
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    with torch_device_fn.device(mat1.device):
        _addmm_strided_b_kernel[grid](
            bias,
            mat1,
            mat2,
            out,
            alpha,
            beta,
            M=m,
            N=n,
            K=k,
            stride_am=mat1.stride(0),
            stride_ak=mat1.stride(1),
            stride_bk=mat2.stride(0),
            stride_bn=mat2.stride(1),
            stride_im=bias.stride(0),
            stride_in=bias.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=1,
        )
    return out


__all__ = ["try_addmm"]
