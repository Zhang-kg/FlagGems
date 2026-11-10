from flag_gems.utils.libentry import LibTuner, mthreads_mm_bucket_strategy


def test_mthreads_bucket_keeps_pruning_boundaries_separate():
    assert all(
        isinstance(mthreads_mm_bucket_strategy(value), int)
        for value in (1, 256, 512, 2048)
    )
    assert mthreads_mm_bucket_strategy(255) != mthreads_mm_bucket_strategy(256)
    assert mthreads_mm_bucket_strategy(512) != mthreads_mm_bucket_strategy(513)
    assert mthreads_mm_bucket_strategy(2047) != mthreads_mm_bucket_strategy(2048)


def test_mthreads_bucket_reuses_same_tile_class():
    # These pairs have identical cdiv() results for every TLE tile size and
    # therefore can safely share a ConfigCache entry.
    for lhs, rhs in ((65, 96), (129, 160), (513, 544), (2049, 2080)):
        assert mthreads_mm_bucket_strategy(lhs) == mthreads_mm_bucket_strategy(rhs)


def test_mthreads_bucket_is_registered_for_expanded_yaml():
    assert LibTuner.get_strategy("mthreads_mm_bucket") is mthreads_mm_bucket_strategy
