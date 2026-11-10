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

import pytest
import torch

import flag_gems
from flag_gems.runtime import backend

from . import accuracy_utils as utils


pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "thead", reason="requires the THead PPU backend"
)


def _unexpected_fallback(*args, **kwargs):
    raise AssertionError("eligible batched GEMM unexpectedly used the generic path")


@pytest.mark.addmm
@pytest.mark.parametrize("M", [4, 17, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_addmm_thead_reuses_ppu_mm_epilogue(M, dtype):
    N, K = 128, 128
    bias = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    ref = torch.addmm(
        utils.to_reference(bias, True),
        utils.to_reference(mat1, True),
        utils.to_reference(mat2, True),
        beta=0.5,
        alpha=1.25,
    )

    with flag_gems.use_gems():
        actual = torch.addmm(bias, mat1, mat2, beta=0.5, alpha=1.25)

    utils.gems_assert_close(actual, ref, dtype, reduce_dim=K)

    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        returned = torch.addmm(
            bias, mat1, mat2, beta=0.5, alpha=1.25, out=out
        )
    assert returned.data_ptr() == out.data_ptr()
    utils.gems_assert_close(out, ref, dtype, reduce_dim=K)


@pytest.mark.addmm
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_addmm_thead_unaligned_tiles_use_ppu(dtype, monkeypatch):
    M, N, K = 67, 537, 538
    bias = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    mat1 = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    ref = torch.addmm(
        utils.to_reference(bias, True),
        utils.to_reference(mat1, True),
        utils.to_reference(mat2, True),
        beta=0.5,
        alpha=1.25,
    )
    addmm_globals = backend.get_backend_state().ops_module.addmm.__globals__
    monkeypatch.setitem(addmm_globals, "_generic_addmm", _unexpected_fallback)

    with flag_gems.use_gems():
        actual = torch.addmm(bias, mat1, mat2, beta=0.5, alpha=1.25)

    utils.gems_assert_close(actual, ref, dtype, reduce_dim=K)


def test_thead_bmm_config_lists_are_bounded():
    bmm_globals = backend.get_backend_state().ops_module.bmm.__globals__
    configs = bmm_globals["_ppu_bmm_configs"]()
    small_m_configs = bmm_globals["_ppu_small_m_bmm_configs"]()

    assert len(configs) == 27
    assert len(small_m_configs) == 16
    assert all("PIPE_STAGES" in config.kwargs for config in configs)
    assert all("PIPE_STAGES" in config.kwargs for config in small_m_configs)
    assert all("BLOCK_M" not in config.kwargs for config in small_m_configs)


def test_thead_bmm_split_k_policy_respects_batch_parallelism():
    bmm_globals = backend.get_backend_state().ops_module.bmm.__globals__
    should_split_k = bmm_globals["_should_use_split_k_bmm"]

    assert should_split_k(1, 144, 512, 4096)
    assert not should_split_k(2, 144, 512, 4096)
    assert not should_split_k(4, 144, 512, 4096)
    assert not should_split_k(8, 144, 512, 4096)
    assert should_split_k(1, 384, 512, 4096)
    assert not should_split_k(2, 384, 512, 4096)
    assert not should_split_k(4, 384, 512, 4096)
    assert should_split_k(8, 384, 512, 4096)
    assert should_split_k(1, 16384, 64, 2048)


@pytest.mark.bmm
@pytest.mark.parametrize("batch", [1, 2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bmm_thead_native_batch_kernel(batch, dtype, monkeypatch):
    M, N, K = 64, 128, 128
    mat1 = torch.randn((batch, M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((batch, K, N), dtype=dtype, device=flag_gems.device)
    ref = torch.bmm(utils.to_reference(mat1, True), utils.to_reference(mat2, True))
    bmm_globals = backend.get_backend_state().ops_module.bmm.__globals__
    monkeypatch.setitem(bmm_globals, "_generic_bmm", _unexpected_fallback)
    monkeypatch.setitem(bmm_globals, "_generic_bmm_out", _unexpected_fallback)

    with flag_gems.use_gems():
        actual = torch.bmm(mat1, mat2)

    utils.gems_assert_close(actual, ref, dtype, reduce_dim=K)

    out = torch.empty((batch, M, N), dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        returned = torch.bmm(mat1, mat2, out=out)
    assert returned.data_ptr() == out.data_ptr()
    utils.gems_assert_close(out, ref, dtype, reduce_dim=K)


@pytest.mark.bmm
@pytest.mark.parametrize("batch,M", [(1, 4), (4, 8), (2, 17)])
def test_bmm_thead_native_fixed_small_m_tile(batch, M, monkeypatch):
    N, K = 128, 128
    dtype = torch.bfloat16
    mat1 = torch.randn((batch, M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((batch, K, N), dtype=dtype, device=flag_gems.device)
    ref = torch.bmm(utils.to_reference(mat1, True), utils.to_reference(mat2, True))
    bmm_globals = backend.get_backend_state().ops_module.bmm.__globals__
    monkeypatch.setitem(bmm_globals, "_generic_bmm", _unexpected_fallback)

    with flag_gems.use_gems():
        actual = torch.bmm(mat1, mat2)

    utils.gems_assert_close(actual, ref, dtype, reduce_dim=K)


@pytest.mark.bmm
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_bmm_thead_unaligned_tiles_use_ppu(dtype, monkeypatch):
    batch, M, N, K = 2, 67, 537, 538
    mat1 = torch.randn((batch, M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((batch, K, N), dtype=dtype, device=flag_gems.device)
    ref = torch.bmm(utils.to_reference(mat1, True), utils.to_reference(mat2, True))
    bmm_globals = backend.get_backend_state().ops_module.bmm.__globals__
    monkeypatch.setitem(bmm_globals, "_generic_bmm", _unexpected_fallback)

    with flag_gems.use_gems():
        actual = torch.bmm(mat1, mat2)

    utils.gems_assert_close(actual, ref, dtype, reduce_dim=K)


@pytest.mark.baddbmm
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_baddbmm_thead_unaligned_tiles_use_ppu(dtype, monkeypatch):
    batch, M, N, K = 2, 67, 537, 538
    bias = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    mat1 = torch.randn((batch, M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((batch, K, N), dtype=dtype, device=flag_gems.device)
    ref = torch.baddbmm(
        utils.to_reference(bias, True),
        utils.to_reference(mat1, True),
        utils.to_reference(mat2, True),
        beta=0.5,
        alpha=1.25,
    )
    baddbmm_globals = backend.get_backend_state().ops_module.baddbmm.__globals__
    monkeypatch.setitem(
        baddbmm_globals, "_generic_baddbmm", _unexpected_fallback
    )
    with flag_gems.use_gems():
        actual = torch.baddbmm(
            bias, mat1, mat2, beta=0.5, alpha=1.25
        )

    utils.gems_assert_close(actual, ref, dtype, reduce_dim=K)


@pytest.mark.bmm
@pytest.mark.baddbmm
@pytest.mark.parametrize("batch,M", [(1, 144), (1, 384), (8, 384)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_thead_native_batched_split_k_and_fused_reduction(
    batch, M, dtype, monkeypatch
):
    N, K = 512, 4096
    bias = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    mat1 = torch.randn((batch, M, K), dtype=dtype, device=flag_gems.device)
    mat2 = torch.randn((batch, K, N), dtype=dtype, device=flag_gems.device)
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    ref_bmm = torch.bmm(ref_mat1, ref_mat2)
    ref_baddbmm = torch.baddbmm(
        utils.to_reference(bias, True),
        ref_mat1,
        ref_mat2,
        beta=0.5,
        alpha=1.25,
    )
    bmm_globals = backend.get_backend_state().ops_module.bmm.__globals__
    baddbmm_globals = backend.get_backend_state().ops_module.baddbmm.__globals__
    monkeypatch.setitem(bmm_globals, "_generic_bmm", _unexpected_fallback)
    monkeypatch.setitem(
        baddbmm_globals, "_generic_baddbmm", _unexpected_fallback
    )

    with flag_gems.use_gems():
        actual_bmm = torch.bmm(mat1, mat2)
        actual_baddbmm = torch.baddbmm(
            bias, mat1, mat2, beta=0.5, alpha=1.25
        )

    utils.gems_assert_close(actual_bmm, ref_bmm, dtype, reduce_dim=K)
    utils.gems_assert_close(actual_baddbmm, ref_baddbmm, dtype, reduce_dim=K)


def _make_bias(shape_kind, batch, M, N, dtype):
    shapes = {
        "vector": (N,),
        "shared": (1, M, N),
        "batched": (batch, M, N),
    }
    return torch.randn(
        shapes[shape_kind],
        dtype=dtype,
        device=flag_gems.device,
        requires_grad=True,
    )


@pytest.mark.baddbmm
@pytest.mark.parametrize("batch", [1, 2, 4, 8])
@pytest.mark.parametrize("shape_kind", ["vector", "shared", "batched"])
def test_baddbmm_thead_native_fused_batch_kernel(batch, shape_kind, monkeypatch):
    M, N, K = 64, 128, 128
    dtype = torch.bfloat16
    bias = _make_bias(shape_kind, batch, M, N, dtype)
    mat1 = torch.randn(
        (batch, M, K), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    mat2 = torch.randn(
        (batch, K, N), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    ref_bias = utils.to_reference(bias, True)
    ref_mat1 = utils.to_reference(mat1, True)
    ref_mat2 = utils.to_reference(mat2, True)
    ref = torch.baddbmm(
        ref_bias,
        ref_mat1,
        ref_mat2,
        beta=0.5,
        alpha=1.25,
    )
    baddbmm_globals = backend.get_backend_state().ops_module.baddbmm.__globals__
    monkeypatch.setitem(
        baddbmm_globals, "_generic_baddbmm", _unexpected_fallback
    )
    monkeypatch.setitem(
        baddbmm_globals, "_generic_baddbmm_out", _unexpected_fallback
    )

    with flag_gems.use_gems():
        actual = torch.baddbmm(bias, mat1, mat2, beta=0.5, alpha=1.25)

    utils.gems_assert_close(actual, ref, dtype, reduce_dim=K)

    ref_grads = torch.autograd.grad(ref.sum(), (ref_bias, ref_mat1, ref_mat2))
    actual_grads = torch.autograd.grad(actual.sum(), (bias, mat1, mat2))
    utils.gems_assert_close(actual_grads[0], ref_grads[0], dtype)
    utils.gems_assert_close(
        actual_grads[1], ref_grads[1], dtype, reduce_dim=N
    )
    utils.gems_assert_close(
        actual_grads[2], ref_grads[2], dtype, reduce_dim=M
    )

    out = torch.empty((batch, M, N), dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        returned = torch.baddbmm(
            bias.detach(),
            mat1.detach(),
            mat2.detach(),
            beta=0.5,
            alpha=1.25,
            out=out,
        )
    assert returned.data_ptr() == out.data_ptr()
    utils.gems_assert_close(out, ref, dtype, reduce_dim=K)
