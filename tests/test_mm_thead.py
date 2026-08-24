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


PPU_SHAPES = [
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
    (32, 384, 7168),
    (384, 512, 4096),
]


def test_mm_thead_default_config_list_is_retained():
    mm_globals = backend.get_backend_state().ops_module.mm.__globals__
    configs = mm_globals["_ppu_mm_configs"]()

    assert len(configs) == 44
    assert all("PIPE_STAGES" in config.kwargs for config in configs)


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
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mm_out_thead_ppu_contiguous_nn(dtype):
    M, N, K = 100, 256, 128
    a = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    b = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    out = torch.empty((M, N), dtype=dtype, device=flag_gems.device)
    ref = torch.mm(utils.to_reference(a, True), utils.to_reference(b, True))

    with flag_gems.use_gems():
        returned = torch.mm(a, b, out=out)

    assert returned.data_ptr() == out.data_ptr()
    utils.gems_assert_close(out, ref, dtype, reduce_dim=K)


@pytest.mark.mm_out
@pytest.mark.parametrize(
    "M,N,K", [(32, 384, 7168), (144, 512, 4096), (384, 512, 4096)]
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


@pytest.mark.mm
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mm_thead_unaligned_k_falls_back(dtype):
    M, N, K = 65, 129, 71
    a = torch.randn((M, K), dtype=dtype, device=flag_gems.device)
    b = torch.randn((K, N), dtype=dtype, device=flag_gems.device)
    ref = torch.mm(utils.to_reference(a, True), utils.to_reference(b, True))

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
