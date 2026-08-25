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

from typing import Generator

import pytest
import torch
import yaml

from . import base, consts


class MvBenchmark(base.GenericBenchmark2DOnly):
    def set_shapes(self, shape_file_path=None):
        with open(shape_file_path, "r", encoding="utf-8") as shape_file:
            shape_config = yaml.safe_load(shape_file) or {}

        if "mv" in shape_config or "mm" not in shape_config:
            return super().set_shapes(shape_file_path)

        # Model traces commonly record GEMV as mm(M, K) x (K, 1).  Accept that
        # schema directly so benchmark/test_mv.py can consume trace YAMLs
        # without maintaining a transformed copy.
        mm_shapes = shape_config.get("mm", {}).get("shapes", [])
        self.shapes = list(
            dict.fromkeys(
                (int(shape[1]), int(shape[3]))
                for shape in mm_shapes
                if len(shape) >= 4 and int(shape[2]) == 1
            )
        )
        self.shape_desc = "M, K"

    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype) -> Generator:
        for m, n in self.shapes:
            yield from self.input_fn(m, n, dtype, self.device)


def _input_fn(m, n, cur_dtype, device):
    inp1 = torch.randn([m, n], dtype=cur_dtype, device=device)
    inp2 = torch.randn([n], dtype=cur_dtype, device=device)
    yield inp1, inp2


@pytest.mark.mv
def test_mv():
    bench = MvBenchmark(
        op_name="mv",
        input_fn=_input_fn,
        torch_op=torch.Tensor.mv,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
