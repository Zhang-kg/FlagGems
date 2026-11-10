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


def _unexpected_mm_fallback(*args, **kwargs):
    raise AssertionError("eligible matrix multiplication used the generic path")


PPU_SHAPES = [
    (2, 128, 64),
    (5, 256, 128),
    (8, 64, 128),
    (16, 64, 128),
    (32, 64, 32),
    (33, 65, 48),
    (63, 127, 64),
    (64, 128, 128),
    (65, 129, 96),
    (100, 256, 128),
    (127, 256, 128),
    (128, 256, 128),
    (129, 256, 128),
    (336, 512, 256),
    (512, 1024, 512),
    (1024, 1024, 1024),
    (100, 4096, 1024),
    (104, 64, 128),
    (32, 384, 7168),
    (384, 512, 4096),
]


def test_mm_thead_default_config_list_is_retained():
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    configs = mm_globals["_ppu_mm_configs"]()

    # The production Pareto list may contain duplicate specs after adding
    # medium-M long-K candidates; retain the original coverage floor without
    # making the test depend on tuple de-duplication details.
    assert len(configs) >= 69
    assert all("PIPE_STAGES" in config.kwargs for config in configs)

    small_m_configs = mm_globals["_ppu_small_m_configs"]()
    assert len(small_m_configs) == 27
    assert all("BLOCK_M" not in config.kwargs for config in small_m_configs)
    assert all("PIPE_STAGES" in config.kwargs for config in small_m_configs)

    mid_m_configs = mm_globals["_ppu_mid_m_configs"]()
    assert len(mid_m_configs) == 27
    assert {config.kwargs["BLOCK_M"] for config in mid_m_configs} == {32}
    assert all("PIPE_STAGES" in config.kwargs for config in mid_m_configs)

    gemv_configs = mm_globals["_ppu_gemv_configs"]()
    assert len(gemv_configs) == 17
    assert {config.kwargs["BLOCK_M"] for config in gemv_configs} == {
        1,
        2,
        4,
        8,
        16,
    }
    assert all("PIPE_STAGES" in config.kwargs for config in gemv_configs)


@pytest.mark.mv
@pytest.mark.parametrize("M,K", [(1, 2048), (136, 4096), (1035, 2048), (16384, 4096)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mv_thead_ppu_gemv(M, K, dtype):
    matrix = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    vector = torch.randn((K,), dtype=dtype, device=flag_gems.device)
    ref = torch.mv(utils.to_reference(matrix, True), utils.to_reference(vector, True))

    with flag_gems.use_gems():
        actual = torch.mv(matrix, vector)

    utils.gems_assert_close(actual, ref, dtype, reduce_dim=K)


@pytest.mark.mm
@pytest.mark.parametrize("M,N,K", PPU_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mm_thead_ppu_contiguous_nn(M, N, K, dtype):
    torch.manual_seed(42)
    a = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    b = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    ref = torch.mm(utils.to_reference(a, True), utils.to_reference(b, True))

    with flag_gems.use_gems():
        actual = torch.mm(a, b)

    utils.gems_assert_close(actual, ref, dtype, reduce_dim=K)


@pytest.mark.mm_out
@pytest.mark.parametrize(
    "M,N,K", [(5, 256, 128), (24, 256, 128), (104, 64, 128)]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mm_out_thead_ppu_contiguous_nn(M, N, K, dtype):
    a = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    b = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)
    ref = torch.mm(utils.to_reference(a, True), utils.to_reference(b, True))

    with flag_gems.use_gems():
        returned = torch.mm(a, b, out=out)

    assert returned.data_ptr() == out.data_ptr()
    utils.gems_assert_close(out, ref, dtype, reduce_dim=K)


@pytest.mark.mm
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mm_thead_m1_keeps_gemv_fallback(dtype):
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    a = torch.randn((1, 128), dtype=dtype, device=flag_gems.device)
    b = torch.randn((128, 128), dtype=dtype, device=flag_gems.device)
    out = torch.empty((1, 128), dtype=dtype, device=flag_gems.device)

    assert not mm_globals["_can_use_ppu_mm"](a, b, out)


@pytest.mark.mm_out
@pytest.mark.parametrize(
    "M,N,K",
    [
        (2, 512, 7168),
        (24, 384, 7168),
        (64, 512, 7168),
        (128, 512, 7168),
        (136, 512, 7168),
        (256, 384, 7168),
        (320, 512, 7168),
        (384, 512, 7168),
        (480, 512, 7168),
        (512, 384, 7168),
        (208, 2112, 7168),
        (32, 384, 7168),
        (144, 512, 4096),
        (384, 512, 4096),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mm_out_thead_ppu_split_k(M, N, K, dtype):
    a = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    b = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)
    ref = torch.mm(utils.to_reference(a, True), utils.to_reference(b, True))

    with flag_gems.use_gems():
        returned = torch.mm(a, b, out=out)

    assert returned.data_ptr() == out.data_ptr()
    utils.gems_assert_close(out, ref, dtype, reduce_dim=K)


def test_mm_thead_k7168_split_k_policy():
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    should_split_k = mm_globals["_should_use_split_k_mm"]
    for M, N in (
        (2, 512),
        (24, 384),
        (32, 384),
        (32, 512),
        (128, 512),
        (136, 512),
        (240, 512),
        (240, 384),
        (320, 384),
        (320, 512),
        (384, 512),
        (480, 512),
        (512, 384),
        (208, 2112),
    ):
        assert should_split_k(M, N, 7168)
    assert not should_split_k(400, 512, 7168)


@pytest.mark.mm
@pytest.mark.parametrize("M,N,K", [(33, 131, 47), (65, 129, 71), (67, 537, 538)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mm_thead_unaligned_tiles_use_ppu(M, N, K, dtype, monkeypatch):
    a = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    b = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    ref = torch.mm(utils.to_reference(a, True), utils.to_reference(b, True))
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    monkeypatch.setitem(mm_globals, "_generic_mm", _unexpected_mm_fallback)

    with flag_gems.use_gems():
        actual = torch.mm(a, b)

    utils.gems_assert_close(actual, ref, dtype, reduce_dim=K)


@pytest.mark.mm
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mm_thead_column_major_b_falls_back(dtype):
    M, N, K = 64, 128, 128
    a = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    b = torch.randn((N, K), dtype=dtype, device=flag_gems.device).t()
    assert not b.is_contiguous()
    ref = torch.mm(utils.to_reference(a, True), utils.to_reference(b, True))

    with flag_gems.use_gems():
        actual = torch.mm(a, b)

    utils.gems_assert_close(actual, ref, dtype, reduce_dim=K)
