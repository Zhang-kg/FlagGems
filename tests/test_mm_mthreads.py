import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import flag_gems
from flag_gems.runtime.configs_loader import TunedConfigLoader


mm_mod = importlib.import_module("flag_gems.runtime.backend._mthreads.ops.mm")


class _FakeTensor:
    def __init__(self, shape, dtype=torch.bfloat16):
        self.shape = shape
        self.dtype = dtype

    def is_contiguous(self):
        return True


def test_mm_mthreads_old_flagtree_falls_back(monkeypatch):
    sentinel = object()
    a = _FakeTensor((4434, 2048))
    b = _FakeTensor((2048, 1024))

    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_PIPE_SQMMA", False)
    monkeypatch.setattr(mm_mod, "_generic_mm", lambda lhs, rhs: sentinel)

    assert mm_mod.mm(a, b) is sentinel


def test_mm_mthreads_small_m_experiment_is_not_dispatched(monkeypatch):
    """The scalar small-M probe must not regress the generic backend path."""
    sentinel = object()
    a = _FakeTensor((16, 4096), torch.bfloat16)
    b = _FakeTensor((4096, 64), torch.bfloat16)
    monkeypatch.setattr(mm_mod, "is_tle_pipe_compatible", lambda *args: False)
    monkeypatch.setattr(
        mm_mod,
        "mm_small_m",
        lambda *args: pytest.fail("experimental route"),
    )
    monkeypatch.setattr(mm_mod, "_generic_mm", lambda lhs, rhs: sentinel)

    assert mm_mod.mm(a, b) is sentinel


def test_mm_mthreads_multifield_pipe_capability_guard(monkeypatch):
    a = _FakeTensor((64, 4096))
    b = _FakeTensor((4096, 2560))

    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_PIPE_SQMMA", True)
    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_MULTIFIELD_PIPE", False)
    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_DYNAMIC_LOOPS", True)
    assert not mm_mod.is_tle_pipe_compatible(a, b, 64, 2560, 4096)

    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_MULTIFIELD_PIPE", True)
    assert mm_mod.is_tle_pipe_compatible(a, b, 64, 2560, 4096)


def test_mm_mthreads_tle_pipe_requires_n_alignment(monkeypatch):
    a = _FakeTensor((64, 4096))
    b = _FakeTensor((4096, 10))
    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_PIPE_SQMMA", True)
    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_MULTIFIELD_PIPE", True)
    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_DYNAMIC_LOOPS", True)
    assert not mm_mod.is_tle_pipe_compatible(a, b, 64, 10, 4096)


def test_mm_mthreads_tle_pipe_rejects_tiny_rows(monkeypatch):
    a = _FakeTensor((16, 4096))
    b = _FakeTensor((4096, 256))
    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_PIPE_SQMMA", True)
    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_MULTIFIELD_PIPE", True)
    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_DYNAMIC_LOOPS", True)
    assert not mm_mod.is_tle_pipe_compatible(a, b, 16, 256, 4096)


def test_mm_mthreads_dynamic_loop_capability_guard(monkeypatch):
    a = _FakeTensor((64, 4096))
    b = _FakeTensor((4096, 2560))
    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_PIPE_SQMMA", True)
    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_MULTIFIELD_PIPE", True)
    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_DYNAMIC_LOOPS", False)
    assert not mm_mod.is_tle_pipe_compatible(a, b, 64, 2560, 4096)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((48, 12288, 2048), True),
        ((56, 12288, 2048), True),
        ((64, 12288, 2048), True),
        ((47, 12288, 2048), False),
        ((65, 12288, 2048), False),
        ((64, 9216, 2048), False),
        ((64, 12288, 4096), False),
    ],
)
def test_mm_mthreads_split_n_full_capability_guard(monkeypatch, shape, expected):
    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_SPLIT_N_SINGLE_PIPE", True)
    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_MULTIFIELD_PIPE", True)
    for dtype in (torch.bfloat16, torch.float16):
        assert mm_mod.is_tle_split_n_full_compatible(*shape, dtype) is expected

    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_MULTIFIELD_PIPE", False)
    assert not mm_mod.is_tle_split_n_full_compatible(*shape, torch.bfloat16)

    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_SPLIT_N_SINGLE_PIPE", False)
    assert not mm_mod.is_tle_split_n_full_compatible(*shape, torch.float16)


def test_mm_mthreads_split_n_full_dispatch_precedes_split_n_single(monkeypatch):
    sentinel = object()
    a = _FakeTensor((64, 2048), torch.bfloat16)
    monkeypatch.setattr(mm_mod, "is_tle_split_n_full_compatible", lambda *args: True)
    monkeypatch.setattr(mm_mod, "mm_tle_split_n_full_pipe", lambda *args: sentinel)
    monkeypatch.setattr(
        mm_mod,
        "is_tle_split_n_single_compatible",
        lambda *args: pytest.fail("full-B route should precede split-N single"),
    )

    assert mm_mod.mm_tle_pipe(a, None, None, 64, 12288, 2048) is sentinel


def test_mm_mthreads_mm_out_full_route_precedes_single(monkeypatch):
    class _MmOutTensor:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype

        def stride(self, dim):
            return self.shape[1] if dim == 0 else 1

        def contiguous(self):
            return self

    a = _MmOutTensor((64, 2048), torch.bfloat16)
    b = _MmOutTensor((2048, 12288), torch.bfloat16)
    out = object()
    sentinel = object()
    monkeypatch.setattr(mm_mod, "is_tle_pipe_compatible", lambda *args: True)
    monkeypatch.setattr(mm_mod, "is_tle_split_n_full_compatible", lambda *args: True)
    monkeypatch.setattr(mm_mod, "mm_tle_split_n_full_pipe", lambda *args: sentinel)
    monkeypatch.setattr(
        mm_mod,
        "is_tle_split_n_single_compatible",
        lambda *args: pytest.fail("full-B route should precede short single route"),
    )

    assert mm_mod.mm_out(a, b, out=out) is sentinel


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((4138, 1024, 2048), True),
        ((4434, 1024, 2048), True),
        ((4435, 1024, 2048), True),
        ((2048, 2048, 512), True),
        ((16384, 2048, 512), True),
        ((4096, 1024, 2048), False),
        ((4481, 1024, 2048), False),
        ((2047, 2048, 512), False),
        ((16385, 2048, 512), False),
        ((4434, 256, 2048), False),
        ((4434, 1024, 4096), False),
    ],
)
def test_mm_mthreads_split_m_capability_guard(monkeypatch, shape, expected):
    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_SPLIT_M_SQMMA", True)
    assert mm_mod.is_tle_split_m_compatible(*shape) is expected

    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_SPLIT_M_SQMMA", False)
    assert not mm_mod.is_tle_split_m_compatible(*shape)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((4138, 2048, 512), True),
        ((4434, 2048, 512), True),
        ((4435, 2048, 512), True),
        ((4436, 2048, 512), False),
        ((4434, 1024, 512), False),
        ((4434, 2048, 2048), False),
    ],
)
def test_mm_mthreads_split_m_fused_capability_guard(monkeypatch, shape, expected):
    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_SPLIT_M_SQMMA", True)
    assert mm_mod.is_tle_split_m_fused_compatible(*shape) is expected

    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_SPLIT_M_SQMMA", False)
    assert not mm_mod.is_tle_split_m_fused_compatible(*shape)


def test_mm_mthreads_split_m_fused_dispatch_precedes_split_m(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        mm_mod, "is_tle_split_m_fused_compatible", lambda *args: True
    )
    monkeypatch.setattr(
        mm_mod, "mm_tle_split_m_fused_pipe", lambda *args: sentinel
    )
    monkeypatch.setattr(
        mm_mod, "is_tle_split_m_compatible", lambda *args: True
    )

    assert mm_mod.mm_tle_pipe(None, None, None, 4434, 2048, 512) is sentinel


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((16384, 2048, 512), True),
        ((16384, 1024, 2048), True),
        ((16384, 9216, 2048), True),
        ((16384, 12288, 2048), True),
        ((16383, 2048, 512), False),
        ((16384, 1024, 512), False),
        ((16384, 2048, 2048), False),
        ((16384, 8192, 2048), False),
        ((16384, 12288, 4096), False),
    ],
)
def test_mm_mthreads_split384_capability_guard(monkeypatch, shape, expected):
    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_SPLIT_M_SQMMA", True)
    assert mm_mod.is_tle_split384_compatible(*shape) is expected

    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_SPLIT_M_SQMMA", False)
    assert not mm_mod.is_tle_split384_compatible(*shape)


def test_mm_mthreads_split384_dispatch_precedes_split_m(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        mm_mod, "is_tle_split384_compatible", lambda *args: True
    )
    monkeypatch.setattr(
        mm_mod, "mm_tle_split384_pipe", lambda *args: sentinel
    )

    assert mm_mod.mm_tle_pipe(None, None, None, 16384, 2048, 512) is sentinel


def test_mm_mthreads_narrow_split384_dispatch_precedes_persistent(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(mm_mod, "is_tle_split384_compatible", lambda *args: True)
    monkeypatch.setattr(mm_mod, "mm_tle_split384_pipe", lambda *args: sentinel)
    monkeypatch.setattr(
        mm_mod,
        "is_tle_persistent_multifield_compatible",
        lambda *args: pytest.fail("narrow split384 route should precede persistent"),
    )

    assert mm_mod.mm_tle_pipe(None, None, None, 16384, 1024, 2048) is sentinel


@pytest.mark.parametrize(
    ("wide_warps", "expected"),
    [
        (False, (16, (8, 4), (168, 24))),
        (True, (16, (4, 4), (168, 24))),
    ],
)
def test_mm_mthreads_split384_experimental_warp_schedule(wide_warps, expected):
    """The wide probe must model MuBLASLt's 24-warp CTA exactly.

    TLE adds worker partitions to the kernel's default ``num_warps``.  This
    pure-Python contract test prevents accidentally reintroducing the former
    56-warp configuration while leaving the production schedule unchanged.
    """
    assert mm_mod._get_tle_split384_schedule(wide_warps) == expected


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((100, 248320, 2048), True),
        ((99, 248320, 2048), False),
        ((100, 31040, 4096), False),
    ],
)
def test_mm_mthreads_split128_capability_guard(monkeypatch, shape, expected):
    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_SPLIT128_SQMMA", True)
    assert mm_mod.is_tle_split128_compatible(*shape) is expected

    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_SPLIT128_SQMMA", False)
    assert not mm_mod.is_tle_split128_compatible(*shape)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((4, 9216, 2048), True),
        ((64, 12288, 2048), True),
        ((128, 9216, 2048), True),
        ((256, 12288, 2048), True),
        ((3, 9216, 2048), False),
        ((257, 9216, 2048), False),
        ((64, 8192, 2048), False),
        ((64, 9216, 4096), False),
    ],
)
def test_mm_mthreads_split_n_capability_guard(monkeypatch, shape, expected):
    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_SPLIT128_SQMMA", True)
    assert mm_mod.is_tle_split_n_compatible(*shape) is expected

    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_SPLIT128_SQMMA", False)
    assert not mm_mod.is_tle_split_n_compatible(*shape)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((64, 12288, 2048), True),
        ((4, 12288, 2048), False),
        ((32, 12288, 2048), False),
        ((64, 9216, 2048), False),
        ((65, 12288, 2048), False),
        ((64, 8192, 2048), False),
        ((64, 12288, 4096), False),
    ],
)
def test_mm_mthreads_split_n_single_capability_guard(
    monkeypatch, dtype, shape, expected
):
    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_SPLIT_N_SINGLE_PIPE", True)
    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_MULTIFIELD_PIPE", True)
    assert mm_mod.is_tle_split_n_single_compatible(*shape, dtype) is expected

    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_MULTIFIELD_PIPE", False)
    assert not mm_mod.is_tle_split_n_single_compatible(*shape, dtype)

    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_SPLIT_N_SINGLE_PIPE", False)
    assert not mm_mod.is_tle_split_n_single_compatible(*shape, dtype)


def test_mm_mthreads_split_n_single_dispatch_precedes_split_n(monkeypatch):
    sentinel = object()
    a = _FakeTensor((64, 2048), torch.bfloat16)
    monkeypatch.setattr(mm_mod, "is_tle_split_n_full_compatible", lambda *args: False)
    monkeypatch.setattr(
        mm_mod,
        "is_tle_split_n_single_compatible",
        lambda *args: True,
    )
    monkeypatch.setattr(mm_mod, "mm_tle_split_n_single_pipe", lambda *args: sentinel)

    assert mm_mod.mm_tle_pipe(a, None, None, 64, 12288, 2048) is sentinel


def test_mm_mthreads_mm_out_uses_short_single_pipe_route(monkeypatch):
    class _MmOutTensor:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype

        def stride(self, dim):
            return self.shape[1] if dim == 0 else 1

        def contiguous(self):
            return self

    a = _MmOutTensor((64, 2048), torch.bfloat16)
    b = _MmOutTensor((2048, 12288), torch.bfloat16)
    out = object()
    sentinel = object()
    monkeypatch.setattr(mm_mod, "is_tle_pipe_compatible", lambda *args: True)
    monkeypatch.setattr(mm_mod, "is_tle_split_n_full_compatible", lambda *args: False)
    monkeypatch.setattr(
        mm_mod, "is_tle_split_n_single_compatible", lambda *args: True
    )
    monkeypatch.setattr(
        mm_mod, "mm_tle_split_n_single_pipe", lambda *args: sentinel
    )
    monkeypatch.setattr(
        mm_mod,
        "mm_tle_pipe",
        lambda *args: pytest.fail("short route should bypass mm_tle_pipe"),
    )

    assert mm_mod.mm_out(a, b, out=out) is sentinel


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((128, 9216, 2048), True),
        ((128, 12288, 2048), True),
        ((64, 12288, 2048), False),
        ((128, 8192, 2048), False),
        ((128, 12288, 4096), False),
    ],
)
def test_mm_mthreads_split_n_wide_single_capability_guard(
    monkeypatch, dtype, shape, expected
):
    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_SPLIT_N_SINGLE_PIPE", True)
    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_MULTIFIELD_PIPE", True)
    assert mm_mod.is_tle_split_n_wide_single_compatible(*shape, dtype) is expected

    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_MULTIFIELD_PIPE", False)
    assert not mm_mod.is_tle_split_n_wide_single_compatible(*shape, dtype)


@pytest.mark.mm
@pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads"
    or getattr(torch, "musa", None) is None,
    reason="requires the MTT MUSA backend",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mm_mthreads_split_n_single_correctness(dtype):
    M, N, K = 64, 12288, 2048
    a = torch.randn((M, K), device=flag_gems.device, dtype=dtype)
    b = torch.randn((K, N), device=flag_gems.device, dtype=dtype)
    expected = torch.mm(a, b)
    actual = mm_mod.mm(a, b)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.mm
@pytest.mark.skipif(
    flag_gems.vendor_name != "mthreads"
    or getattr(torch, "musa", None) is None,
    reason="requires the MTT MUSA backend",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mm_mthreads_split_n_wide_single_correctness(dtype):
    """Compile and validate the experimental M=128 one-pipe schedule."""
    M, N, K = 128, 12288, 2048
    a = torch.randn((M, K), device=flag_gems.device, dtype=dtype)
    b = torch.randn((K, N), device=flag_gems.device, dtype=dtype)
    actual = torch.empty((M, N), device=flag_gems.device, dtype=dtype)
    expected = torch.mm(a, b)
    mm_mod.mm_tle_split_n_wide_single_pipe(a, b, actual, M, N, K)
    torch.musa.synchronize()
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((257, 9216, 2048), True),
        ((384, 12288, 2048), True),
        ((496, 9216, 2048), True),
        ((2048, 1024, 2048), True),
        ((512, 2048, 512), True),
        ((513, 2048, 512), True),
        ((1036, 2048, 512), True),
        ((256, 9216, 2048), False),
        ((497, 12288, 2048), False),
        ((2047, 1024, 2048), False),
        ((2048, 1024, 4096), False),
        ((511, 2048, 512), False),
        ((1041, 2048, 512), False),
        ((384, 8192, 2048), False),
        ((384, 9216, 4096), False),
    ],
)
def test_mm_mthreads_split256_ordered_capability_guard(
    monkeypatch, shape, expected
):
    monkeypatch.setattr(
        mm_mod, "HAS_MTHREADS_TLE_PERSISTENT_ORDERED_SQMMA", True
    )
    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_DYNAMIC_PARTITION_SYNC", True)
    assert mm_mod.is_tle_split256_ordered_compatible(*shape) is expected

    monkeypatch.setattr(
        mm_mod, "HAS_MTHREADS_TLE_PERSISTENT_ORDERED_SQMMA", False
    )
    assert not mm_mod.is_tle_split256_ordered_compatible(*shape)

    monkeypatch.setattr(
        mm_mod, "HAS_MTHREADS_TLE_PERSISTENT_ORDERED_SQMMA", True
    )
    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_DYNAMIC_PARTITION_SYNC", False)
    assert not mm_mod.is_tle_split256_ordered_compatible(*shape)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((257, 9216, 2048), True),
        ((384, 9216, 2048), True),
        ((480, 9216, 2048), True),
        ((512, 9216, 2048), True),
        ((256, 9216, 2048), False),
        ((513, 9216, 2048), False),
        ((496, 12288, 2048), False),
        ((480, 12288, 2048), False),
        ((480, 9216, 4096), False),
    ],
)
def test_mm_mthreads_bn320_capability_guard(monkeypatch, shape, expected):
    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_DYNAMIC_PARTITION_SYNC", True)
    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_BN320_SQMMA", True)
    assert mm_mod.is_tle_bn320_compatible(*shape) is expected

    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_BN320_SQMMA", False)
    assert not mm_mod.is_tle_bn320_compatible(*shape)

    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_BN320_SQMMA", True)
    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_DYNAMIC_PARTITION_SYNC", False)
    assert not mm_mod.is_tle_bn320_compatible(*shape)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((256, 9216, 2048), False),
        ((256, 12288, 2048), False),
        ((512, 12288, 2048), True),
        ((384, 4096, 1024), True),
        ((511, 12288, 2048), False),
        ((512, 9216, 2048), False),
        ((512, 12288, 4096), False),
    ],
)
def test_mm_mthreads_persistent_multifield_capability_guard(monkeypatch, shape, expected):
    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_MULTIFIELD_PIPE", True)
    monkeypatch.setattr(
        mm_mod, "HAS_MTHREADS_TLE_PERSISTENT_ORDERED_SQMMA", True
    )
    dtype = torch.bfloat16 if shape == (384, 4096, 1024) else None
    assert mm_mod.is_tle_persistent_multifield_compatible(*shape, dtype) is expected
    if shape == (384, 4096, 1024):
        assert not mm_mod.is_tle_persistent_multifield_compatible(
            *shape, torch.float32
        )

    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_MULTIFIELD_PIPE", False)
    assert not mm_mod.is_tle_persistent_multifield_compatible(*shape)

    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_MULTIFIELD_PIPE", True)
    monkeypatch.setattr(
        mm_mod, "HAS_MTHREADS_TLE_PERSISTENT_ORDERED_SQMMA", False
    )
    assert not mm_mod.is_tle_persistent_multifield_compatible(*shape)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mm_mthreads_large_persistent_multifield_supports_low_precision(
    monkeypatch, dtype
):
    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_MULTIFIELD_PIPE", True)
    monkeypatch.setattr(
        mm_mod, "HAS_MTHREADS_TLE_PERSISTENT_ORDERED_SQMMA", True
    )

    assert mm_mod.is_tle_persistent_multifield_compatible(
        16384, 12288, 2048, dtype
    )
    assert mm_mod.is_tle_persistent_multifield_compatible(
        16384, 9216, 2048, dtype
    )
    assert mm_mod.is_tle_persistent_multifield_compatible(
        16384, 1024, 2048, dtype
    )
    assert not mm_mod.is_tle_persistent_multifield_compatible(
        16384, 12288, 2048, torch.float32
    )
    assert not mm_mod.is_tle_persistent_multifield_compatible(
        16384, 9216, 2048, torch.float32
    )
    assert not mm_mod.is_tle_persistent_multifield_compatible(
        16384, 1024, 2048, torch.float32
    )


def test_mm_mthreads_persistent_multifield_dispatch_precedes_general_pipe(
    monkeypatch,
):
    sentinel = object()
    monkeypatch.setattr(
        mm_mod, "is_tle_persistent_multifield_compatible", lambda *args: True
    )
    monkeypatch.setattr(
        mm_mod,
        "mm_tle_persistent_multifield",
        lambda *args: sentinel,
    )

    assert mm_mod.mm_tle_pipe(None, None, None, 512, 12288, 2048) is sentinel


def test_mm_mthreads_large_multifield_dispatch_precedes_16w(monkeypatch):
    sentinel = object()
    a = _FakeTensor((16384, 2048), torch.bfloat16)

    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_MULTIFIELD_PIPE", True)
    monkeypatch.setattr(
        mm_mod, "HAS_MTHREADS_TLE_PERSISTENT_ORDERED_SQMMA", True
    )
    monkeypatch.setattr(
        mm_mod, "is_tle_persistent_16w_compatible", lambda *args: True
    )
    monkeypatch.setattr(
        mm_mod,
        "mm_tle_persistent_multifield",
        lambda *args: sentinel,
    )

    assert mm_mod.mm_tle_pipe(a, None, None, 16384, 12288, 2048) is sentinel


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((16384, 12288, 2048), True),
        ((16383, 12288, 2048), False),
        ((16384, 9216, 2048), False),
        ((16384, 12288, 4096), False),
    ],
)
def test_mm_mthreads_persistent_16w_capability_guard(
    monkeypatch, shape, expected
):
    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_16_WARP_PERSISTENT", True)
    assert mm_mod.is_tle_persistent_16w_compatible(*shape) is expected

    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_16_WARP_PERSISTENT", False)
    assert not mm_mod.is_tle_persistent_16w_compatible(*shape)


def test_mm_mthreads_persistent_16w_rejects_fp16(monkeypatch):
    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_16_WARP_PERSISTENT", True)
    shape = (16384, 12288, 2048)

    assert mm_mod.is_tle_persistent_16w_compatible(*shape, torch.bfloat16)
    assert not mm_mod.is_tle_persistent_16w_compatible(*shape, torch.float16)


def test_mm_mthreads_persistent_16w_dispatch_precedes_other_fast_paths(
    monkeypatch,
):
    sentinel = object()
    monkeypatch.setattr(
        mm_mod, "is_tle_persistent_16w_compatible", lambda *args: True
    )
    monkeypatch.setattr(
        mm_mod,
        "mm_tle_persistent_16w",
        lambda *args: sentinel,
    )

    assert mm_mod.mm_tle_pipe(None, None, None, 16384, 12288, 2048) is sentinel


@pytest.mark.parametrize(
    ("M", "expected"),
    [(257, 64), (384, 64), (385, 128), (512, 128)],
)
def test_mm_mthreads_bn320_selects_bottom_consumer_height(M, expected):
    assert mm_mod.get_tle_bn320_block_m_bottom(M) == expected


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ((192, 4096, 1024), (64, 128, 128, 2, 4, 8, False)),
        ((193, 4096, 1024), (128, 128, 64, 2, 4, 8, False)),
        ((271, 4096, 128), (64, 64, 128, 1, 4, 1, False)),
        ((272, 4096, 128), (64, 128, 128, 1, 4, 8, False)),
        ((320, 4096, 128), (64, 128, 128, 1, 4, 8, False)),
        ((321, 4096, 128), (128, 128, 128, 1, 4, 4, False)),
        ((512, 4096, 128), (128, 128, 128, 1, 4, 4, False)),
        ((192, 1024, 2048), (64, 64, 256, 3, 4, 1, False)),
        ((193, 1024, 2048), (64, 64, 128, 3, 4, 1, False)),
        ((320, 1024, 2048), (64, 64, 128, 3, 4, 1, False)),
        ((321, 1024, 2048), (128, 64, 128, 2, 4, 1, False)),
        ((511, 1024, 2048), (128, 64, 128, 2, 4, 1, False)),
        ((512, 1024, 2048), (128, 128, 128, 2, 4, 4, False)),
        ((513, 1024, 2048), (128, 128, 64, 3, 4, 4, False)),
        ((1040, 1024, 2048), (128, 128, 64, 3, 4, 4, False)),
        ((1041, 1024, 2048), (256, 256, 64, 3, 16, 1, True)),
        ((1, 2048, 512), (64, 128, 128, 2, 4, 2, False)),
        ((511, 2048, 512), (64, 128, 128, 2, 4, 2, False)),
        ((512, 2048, 512), (256, 128, 64, 4, 1, 1, True)),
        ((1, 512, 4096), (64, 64, 256, 3, 4, 2, False)),
        ((511, 512, 4096), (64, 64, 256, 3, 4, 2, False)),
        ((512, 512, 4096), (128, 64, 256, 2, 4, 1, False)),
        ((513, 512, 4096), (128, 128, 128, 2, 4, 4, False)),
        ((1040, 512, 4096), (128, 128, 128, 2, 4, 4, False)),
        ((1041, 512, 4096), (256, 256, 64, 3, 16, 1, True)),
        ((2048, 512, 4096), (128, 128, 64, 3, 4, 4, False)),
        ((71, 2048, 4096), (64, 64, 256, 3, 4, 1, False)),
        ((72, 2048, 4096), (64, 128, 128, 2, 4, 2, False)),
        ((272, 2048, 4096), (64, 128, 128, 2, 4, 2, False)),
        ((273, 2048, 4096), (64, 128, 64, 3, 4, 2, False)),
        ((448, 2048, 4096), (64, 128, 64, 3, 4, 2, False)),
        ((449, 2048, 4096), (128, 128, 64, 3, 4, 4, False)),
        ((512, 2048, 4096), (256, 128, 64, 4, 8, 2, True)),
        ((64, 2560, 4096), (64, 64, 256, 3, 4, 1, False)),
        ((65, 2560, 4096), (64, 128, 128, 2, 4, 8, False)),
        ((224, 2560, 4096), (64, 128, 128, 2, 4, 8, False)),
        ((225, 2560, 4096), (64, 128, 64, 3, 4, 8, False)),
        ((384, 2560, 4096), (64, 128, 64, 3, 4, 8, False)),
        ((385, 2560, 4096), (128, 128, 64, 3, 4, 8, False)),
        ((512, 2560, 4096), (256, 128, 64, 4, 8, 2, True)),
        ((4138, 64, 2048), (64, 64, 128, 2, 4, 1, False)),
        ((512, 256, 2048), (64, 64, 256, 3, 4, 1, False)),
        ((1036, 256, 2048), (64, 64, 128, 2, 4, 1, False)),
        ((4434, 256, 2048), (128, 128, 64, 3, 4, 1, False)),
        ((16384, 9216, 2048), (256, 256, 64, 2, 16, 2, True)),
        ((16384, 12288, 2048), (256, 256, 64, 2, 16, 2, True)),
    ],
)
def test_mm_mthreads_tle_pipe_topology_dispatch(shape, expected):
    assert mm_mod.use_tle_ws_pipe(*shape) is expected[-1]


@pytest.mark.parametrize(
    "shape",
    [
        # 397B trace: K=128/1024, N=4096, all low-M buckets.
        (1, 4096, 128),
        (144, 4096, 128),
        (512, 4096, 128),
        (1, 4096, 1024),
        (144, 4096, 1024),
        (512, 4096, 1024),
        # 35B trace: K=512, N=2048, low-M buckets.
        (1, 2048, 512),
        (256, 2048, 512),
        (511, 2048, 512),
    ],
)
def test_mm_mthreads_low_m_trace_stays_non_ws(shape):
    """Do not route unvalidated low-M trace families to WS.

    The MTT backend currently has no stable native BF16/FP16 K=128 contract,
    and WS adds fixed partition overhead for these small row counts.  This
    test intentionally checks topology only; it does not claim a performance
    result for a replacement kernel.
    """
    assert mm_mod.use_tle_ws_pipe(*shape) is False


@pytest.mark.parametrize(
    "shape",
    [(1, 4096, 128), (144, 4096, 1024), (512, 2048, 512)],
)
def test_mm_mthreads_low_m_prune_keeps_backend_k_contract(monkeypatch, shape):
    """Expanded candidates must use K<=64 on the current MTT backend."""
    yaml_path = Path(mm_mod.__file__).parents[1] / "mm_mthreads_expand.yaml"
    configs = TunedConfigLoader().ops_get_configs(
        "mm_tle_non_ws_pipe", yaml_path=str(yaml_path)
    )
    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_PIPE_SQMMA", True)
    M, N, K = shape
    filtered = mm_mod._prune_tle_non_ws_pipe_configs(
        configs, {"M": M, "N": N, "K": K}
    )
    assert filtered
    assert all(config.kwargs["BLOCK_K"] <= 64 for config in filtered)
    assert all(K % config.kwargs["BLOCK_K"] == 0 for config in filtered)


def test_mm_mthreads_tle_pipe_expanded_configs_cover_pipeline_axes():
    yaml_path = Path(mm_mod.__file__).parents[1] / "mm_mthreads_expand.yaml"
    loader = TunedConfigLoader()

    non_ws = loader.ops_get_configs(
        "mm_tle_non_ws_pipe", yaml_path=str(yaml_path)
    )
    ws = loader.ops_get_configs("mm_tle_ws_pipe", yaml_path=str(yaml_path))

    assert len(non_ws) == 3840
    assert len(ws) == 1920
    assert {config.kwargs["BLOCK_K"] for config in non_ws} == {32, 64, 128, 256}
    assert {config.kwargs["NUM_SLOTS"] for config in non_ws} == {1, 2, 3, 4}
    assert {config.kwargs["MMA_GROUP"] for config in non_ws} == {1, 2, 3, 4}
    assert {config.kwargs["GROUP_M"] for config in non_ws} == {1, 2, 4, 8, 16}
    assert {config.num_warps for config in non_ws} == {4, 8}
    assert {config.kwargs["BLOCK_K"] for config in ws} == {32, 64, 128}
    assert {config.kwargs["NUM_SLOTS"] for config in ws} == {1, 2, 3, 4}
    assert {config.kwargs["MMA_GROUP"] for config in ws} == {1, 2, 3, 4}
    assert {config.kwargs["GROUP_M"] for config in ws} == {1, 2, 4, 8, 16}
    assert {config.num_warps for config in ws} == {8, 16}


def test_mm_mthreads_default_non_ws_keeps_verified_8warp_candidate():
    matches = [
        config
        for config in mm_mod._TLE_NON_WS_PIPE_CONFIGS
        if config.num_warps == 8
        and config.kwargs.get("BLOCK_M") == 64
        and config.kwargs.get("BLOCK_N") == 128
        and config.kwargs.get("BLOCK_K") == 64
        and config.kwargs.get("NUM_SLOTS") == 3
        and config.kwargs.get("GROUP_M") == 1
    ]
    assert len(matches) == 1


@pytest.mark.parametrize("shape", [(1035, 64, 2048), (1036, 64, 2048), (2048, 64, 2048), (4138, 64, 2048), (4434, 64, 2048), (4435, 64, 2048), (16384, 64, 2048)])
def test_mm_mthreads_prune_uses_verified_n64_tile(shape):
    yaml_path = Path(mm_mod.__file__).parents[1] / "mm_mthreads_expand.yaml"
    configs = TunedConfigLoader().ops_get_configs(
        "mm_tle_non_ws_pipe", yaml_path=str(yaml_path)
    )
    filtered = mm_mod._prune_tle_non_ws_pipe_configs(
        configs, dict(zip(("M", "N", "K"), shape))
    )
    assert [
        (
            config.kwargs["BLOCK_M"],
            config.kwargs["BLOCK_N"],
            config.kwargs["BLOCK_K"],
            config.kwargs["NUM_SLOTS"],
            config.kwargs["MMA_GROUP"],
            config.kwargs["GROUP_M"],
            config.num_warps,
        )
        for config in filtered
    ] == [(64, 64, 64, 3, 1, 1, 4)]


def test_mm_mthreads_prune_uses_verified_ws_k512_tile():
    yaml_path = Path(mm_mod.__file__).parents[1] / "mm_mthreads_expand.yaml"
    configs = TunedConfigLoader().ops_get_configs(
        "mm_tle_ws_pipe", yaml_path=str(yaml_path)
    )
    filtered = mm_mod._prune_tle_ws_pipe_configs(
        configs, {"M": 512, "N": 2048, "K": 512}
    )
    assert [
        (
            config.kwargs["BLOCK_M"],
            config.kwargs["BLOCK_N"],
            config.kwargs["BLOCK_K"],
            config.kwargs["NUM_SLOTS"],
            config.kwargs["MMA_GROUP"],
            config.kwargs["GROUP_M"],
            config.num_warps,
        )
        for config in filtered
    ] == [(256, 128, 64, 4, 1, 1, 16)]


@pytest.mark.parametrize(
    ("op_name", "shape", "expected"),
    [
        ("mm_tle_non_ws_pipe", (512, 1024, 2048), (64, 128, 64, 3, 1, 1, 8)),
        ("mm_tle_non_ws_pipe", (1035, 1024, 2048), (128, 128, 64, 3, 1, 1, 4)),
        ("mm_tle_ws_pipe", (2048, 1024, 2048), (256, 256, 64, 3, 1, 1, 16)),
        ("mm_tle_ws_pipe", (512, 2048, 4096), (256, 128, 64, 4, 1, 1, 16)),
        ("mm_tle_ws_pipe", (2048, 2048, 4096), (256, 128, 64, 4, 1, 1, 16)),
    ],
)
def test_mm_mthreads_prune_keeps_verified_fixed_configs(
    monkeypatch, op_name, shape, expected
):
    yaml_path = Path(mm_mod.__file__).parents[1] / "mm_mthreads_expand.yaml"
    configs = TunedConfigLoader().ops_get_configs(op_name, yaml_path=str(yaml_path))
    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_PIPE_SQMMA", True)
    prune = (
        mm_mod._prune_tle_ws_pipe_configs
        if op_name == "mm_tle_ws_pipe"
        else mm_mod._prune_tle_non_ws_pipe_configs
    )
    pruned = prune(configs, dict(zip(("M", "N", "K"), shape)))
    assert any(
        (
            config.kwargs["BLOCK_M"],
            config.kwargs["BLOCK_N"],
            config.kwargs["BLOCK_K"],
            config.kwargs["NUM_SLOTS"],
            config.kwargs["MMA_GROUP"],
            config.kwargs["GROUP_M"],
            config.num_warps,
        )
        == expected
        for config in pruned
    )


def test_mm_mthreads_prune_empty_fallback_keeps_mtt_k_capability(monkeypatch):
    # Both candidates fail the shape-specific tile constraints (BLOCK_M=128
    # does not fit M=64), so this exercises the prune hook's fallback branch.
    configs = [
        SimpleNamespace(
            kwargs={
                "BLOCK_M": 128,
                "BLOCK_N": 64,
                "BLOCK_K": block_k,
                "NUM_SLOTS": 1,
                "MMA_GROUP": 1,
                "GROUP_M": 1,
            },
            num_warps=4,
        )
        for block_k in (128, 64)
    ]
    named_args = {"M": 64, "N": 64, "K": 64}

    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_PIPE_SQMMA", True)
    pruned = mm_mod._prune_tle_non_ws_pipe_configs(configs, named_args)
    assert pruned == [configs[1]]
    assert all(config.kwargs["BLOCK_K"] <= 64 for config in pruned)

    # If the caller supplied only unsupported wide-K candidates, returning an
    # empty list is preferable to handing an invalid config to the compiler.
    assert mm_mod._prune_tle_non_ws_pipe_configs([configs[0]], named_args) == []

    # Other backends retain the historical all-config fallback when no
    # candidate survives shape-specific pruning.
    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_PIPE_SQMMA", False)
    assert mm_mod._prune_tle_non_ws_pipe_configs(configs, named_args) == configs


def test_mm_mthreads_prune_empty_ws_fallback_is_bounded(monkeypatch):
    # The output-tile heuristic rejects every normal WS candidate for this
    # ragged shape.  The MTT fallback must still return only a small set of
    # resource-safe K<=64 candidates instead of the full expanded space.
    configs = []
    for block_m in (128, 256):
        for block_n in (128, 256):
            for num_slots in (2, 3):
                configs.append(
                    SimpleNamespace(
                        kwargs={
                            "BLOCK_M": block_m,
                            "BLOCK_N": block_n,
                            "BLOCK_K": 64,
                            "NUM_SLOTS": num_slots,
                            "MMA_GROUP": 1,
                            "GROUP_M": 1,
                        },
                        num_warps=16,
                    )
                )

    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_PIPE_SQMMA", True)
    pruned = mm_mod._prune_tle_ws_pipe_configs(
        configs, {"M": 1035, "N": 9216, "K": 2048}
    )
    assert 0 < len(pruned) <= 8
    assert all(config.kwargs["BLOCK_K"] <= 64 for config in pruned)
    assert all(config.kwargs["NUM_SLOTS"] <= 3 for config in pruned)


def test_mm_mthreads_prune_keeps_verified_non_ws_8warp_tile(monkeypatch):
    configs = []
    for group_m in (1, 2):
        configs.append(
            SimpleNamespace(
                kwargs={
                    "BLOCK_M": 64,
                    "BLOCK_N": 128,
                    "BLOCK_K": 64,
                    "NUM_SLOTS": 3,
                    "MMA_GROUP": 1,
                    "GROUP_M": group_m,
                },
                num_warps=8,
            )
        )

    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_PIPE_SQMMA", True)
    pruned = mm_mod._prune_tle_non_ws_pipe_configs(
        configs, {"M": 512, "N": 1024, "K": 2048}
    )
    assert pruned == [configs[0]]


def test_mm_mthreads_non_ws_prune_retains_small_tiles_for_low_occupancy():
    yaml_path = Path(mm_mod.__file__).parents[1] / "mm_mthreads_expand.yaml"
    configs = TunedConfigLoader().ops_get_configs(
        "mm_tle_non_ws_pipe", yaml_path=str(yaml_path)
    )

    low_occupancy = mm_mod._prune_tle_non_ws_pipe_configs(
        configs, {"M": 512, "N": 512, "K": 4096}
    )
    high_occupancy = mm_mod._prune_tle_non_ws_pipe_configs(
        configs, {"M": 512, "N": 1024, "K": 2048}
    )

    assert any(
        config.kwargs["BLOCK_M"] == 64 and config.kwargs["BLOCK_N"] == 64
        for config in low_occupancy
    )
    assert all(
        config.kwargs["BLOCK_M"] * config.kwargs["BLOCK_N"] >= 8192
        for config in high_occupancy
    )


def test_mm_mthreads_tle_pipe_uses_expanded_configs_by_default(monkeypatch):
    monkeypatch.delenv("USE_FLAGTUNE", raising=False)
    monkeypatch.delenv("USE_FLAGTUNE_COST_MODEL", raising=False)
    monkeypatch.delenv("FLAGTUNE_INCLUDE", raising=False)

    for tuner in (mm_mod.mm_tle_non_ws_pipe_kernel, mm_mod.mm_tle_pipe_kernel):
        assert tuner.fn._flagtune_default_mode is mm_mod.runtime.TuningMode.EXPANDED


@pytest.mark.parametrize(
    ("op_name", "prune", "shape", "expected_warps"),
    [
        (
            "mm_tle_non_ws_pipe",
            mm_mod._prune_tle_non_ws_pipe_configs,
            (512, 512, 4096),
            {4, 8},
        ),
        (
            "mm_tle_ws_pipe",
            mm_mod._prune_tle_ws_pipe_configs,
            (16384, 9216, 2048),
            {16},
        ),
    ],
)
def test_mm_mthreads_tle_pipe_expanded_configs_are_shape_pruned(
    op_name, prune, shape, expected_warps
):
    yaml_path = Path(mm_mod.__file__).parents[1] / "mm_mthreads_expand.yaml"
    configs = TunedConfigLoader().ops_get_configs(op_name, yaml_path=str(yaml_path))
    M, N, K = shape
    filtered = prune(configs, {"M": M, "N": N, "K": K})

    assert filtered
    assert len(filtered) < len(configs)
    assert all(K % config.kwargs["BLOCK_K"] == 0 for config in filtered)
    assert all(
        config.kwargs["NUM_SLOTS"] <= K // config.kwargs["BLOCK_K"]
        for config in filtered
    )
    assert all(
        config.kwargs["MMA_GROUP"] <= config.kwargs["NUM_SLOTS"]
        for config in filtered
    )
    assert all(
        config.kwargs["GROUP_M"]
        <= (M + config.kwargs["BLOCK_M"] - 1) // config.kwargs["BLOCK_M"]
        for config in filtered
    )
    assert all(
        K // config.kwargs["BLOCK_K"] < 4
        or config.kwargs["NUM_SLOTS"] > 1
        for config in filtered
    )
    assert all(
        config.kwargs["BLOCK_K"] != 32 or K // 32 <= 64
        for config in filtered
    )
    assert all(
        config.kwargs["NUM_SLOTS"]
        * (config.kwargs["BLOCK_M"] + config.kwargs["BLOCK_N"])
        * config.kwargs["BLOCK_K"]
        * 2
        <= 192 * 1024
        for config in filtered
    )
    assert {config.num_warps for config in filtered} == expected_warps

    if op_name == "mm_tle_non_ws_pipe":
        assert all(
            config.num_warps != 8
            or (
                (
                    config.kwargs["BLOCK_M"] * config.kwargs["BLOCK_N"] >= 16384
                    and config.kwargs["NUM_SLOTS"] <= 2
                    and shape[0] <= 512
                )
                or (
                    config.kwargs["BLOCK_M"] == 64
                    and config.kwargs["BLOCK_N"] == 128
                    and config.kwargs["BLOCK_K"] == 64
                    and config.kwargs["NUM_SLOTS"] == 3
                    and config.kwargs["GROUP_M"] == 1
                )
            )
            for config in filtered
        )


def test_mm_mthreads_ws8_prune_limits_search_to_productive_tiles():
    yaml_path = Path(mm_mod.__file__).parents[1] / "mm_mthreads_expand.yaml"
    configs = TunedConfigLoader().ops_get_configs(
        "mm_tle_ws_pipe", yaml_path=str(yaml_path)
    )
    filtered = mm_mod._prune_tle_ws_pipe_configs(
        configs, {"M": 512, "N": 2560, "K": 4096}
    )
    ws8 = [config for config in filtered if config.num_warps == 8]

    assert ws8
    assert all(
        config.kwargs["BLOCK_M"] * config.kwargs["BLOCK_N"] <= 32768
        for config in ws8
    )
    assert all(
        ((512 + config.kwargs["BLOCK_M"] - 1) // config.kwargs["BLOCK_M"])
        * ((2560 + config.kwargs["BLOCK_N"] - 1) // config.kwargs["BLOCK_N"])
        <= 64
        for config in ws8
    )

    assert all(
        config.kwargs["BLOCK_M"] * config.kwargs["BLOCK_N"] == 32768
        and config.kwargs["BLOCK_K"] == 64
        and config.kwargs["NUM_SLOTS"] == 4
        and config.kwargs["MMA_GROUP"] in (1, 2, 4)
        for config in filtered
    )


def test_mm_mthreads_ws_k4096_prune_exposes_mma_group_search():
    yaml_path = Path(mm_mod.__file__).parents[1] / "mm_mthreads_expand.yaml"
    configs = TunedConfigLoader().ops_get_configs(
        "mm_tle_ws_pipe", yaml_path=str(yaml_path)
    )
    filtered = mm_mod._prune_tle_ws_pipe_configs(
        configs, {"M": 512, "N": 2048, "K": 4096}
    )
    groups = {config.kwargs["MMA_GROUP"] for config in filtered}
    assert groups == {1, 2, 4}


def test_mm_mthreads_prune_drops_unsupported_wide_k(monkeypatch):
    yaml_path = Path(mm_mod.__file__).parents[1] / "mm_mthreads_expand.yaml"
    configs = TunedConfigLoader().ops_get_configs(
        "mm_tle_non_ws_pipe", yaml_path=str(yaml_path)
    )
    monkeypatch.setattr(mm_mod, "HAS_MTHREADS_TLE_PIPE_SQMMA", True)
    filtered = mm_mod._prune_tle_non_ws_pipe_configs(
        configs, {"M": 64, "N": 12288, "K": 2048}
    )
    assert filtered
    assert all(config.kwargs["BLOCK_K"] <= 64 for config in filtered)
