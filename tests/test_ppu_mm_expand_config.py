# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from pathlib import Path

from flag_gems.runtime.configs_loader import TunedConfigLoader


def test_ppu_mm_expand_config_is_filtered_and_stage_coupled():
    loader = TunedConfigLoader()
    yaml_path = (
        Path(__file__).parents[1]
        / "src"
        / "flag_gems"
        / "runtime"
        / "backend"
        / "_thead"
        / "mm_ppu_expand.yaml"
    )

    expand = loader.get_expand_config("mm_ppu", yaml_path=str(yaml_path))
    configs = loader.ops_get_configs("mm_ppu", yaml_path=str(yaml_path))

    assert expand != -1
    assert expand["strategy"] == ["default"] * 4
    assert len(configs) == 4512
    assert len(configs) <= 5000
    assert {config.kwargs["LOAD_MODE"] for config in configs} == {0, 1, 2, 3}
    assert all(
        config.kwargs["PIPE_STAGES"] == config.num_stages for config in configs
    )
    assert all(
        config.kwargs["BLOCK_M"]
        * config.kwargs["BLOCK_N"]
        // (config.num_warps * 32)
        <= 128
        for config in configs
    )
    assert all(
        (
            config.kwargs["BLOCK_M"] + config.kwargs["BLOCK_N"]
        )
        * config.kwargs["BLOCK_K"]
        * 2
        * max(config.num_stages - 1, 1)
        <= 240 * 1024
        for config in configs
    )


def test_ppu_small_m_expand_config_keeps_a_legal_fixed_m_tile():
    loader = TunedConfigLoader()
    yaml_path = (
        Path(__file__).parents[1]
        / "src"
        / "flag_gems"
        / "runtime"
        / "backend"
        / "_thead"
        / "mm_ppu_expand.yaml"
    )

    expand = loader.get_expand_config("mm_ppu_small_m", yaml_path=str(yaml_path))
    configs = loader.ops_get_configs("mm_ppu_small_m", yaml_path=str(yaml_path))

    assert expand != -1
    assert expand["strategy"] == ["default"] * 4
    assert len(configs) == 248
    assert {config.kwargs["LOAD_MODE"] for config in configs} == {0, 1, 2, 3}
    assert all("BLOCK_M" not in config.kwargs for config in configs)
    assert all(
        config.kwargs["PIPE_STAGES"] == config.num_stages for config in configs
    )
    assert all(
        (16 + config.kwargs["BLOCK_N"])
        * config.kwargs["BLOCK_K"]
        * 2
        * max(config.num_stages - 1, 1)
        <= 240 * 1024
        for config in configs
    )


def test_ppu_mid_m_expand_config_has_a_stable_cache_schema():
    loader = TunedConfigLoader()
    yaml_path = (
        Path(__file__).parents[1]
        / "src"
        / "flag_gems"
        / "runtime"
        / "backend"
        / "_thead"
        / "mm_ppu_expand.yaml"
    )

    expand = loader.get_expand_config("mm_ppu_mid_m", yaml_path=str(yaml_path))
    configs = loader.ops_get_configs("mm_ppu_mid_m", yaml_path=str(yaml_path))

    assert expand != -1
    assert expand["strategy"] == ["default"] * 4
    assert len(configs) == 248
    assert {config.kwargs["BLOCK_M"] for config in configs} == {32}
    assert all(
        config.kwargs["PIPE_STAGES"] == config.num_stages for config in configs
    )
    assert all(
        (32 + config.kwargs["BLOCK_N"])
        * config.kwargs["BLOCK_K"]
        * 2
        * max(config.num_stages - 1, 1)
        <= 240 * 1024
        for config in configs
    )


def test_ppu_bmm_expand_configs_cover_main_and_fixed_small_m_tiles():
    loader = TunedConfigLoader()
    yaml_path = (
        Path(__file__).parents[1]
        / "src"
        / "flag_gems"
        / "runtime"
        / "backend"
        / "_thead"
        / "mm_ppu_expand.yaml"
    )

    main_expand = loader.get_expand_config("bmm_ppu", yaml_path=str(yaml_path))
    main_configs = loader.ops_get_configs("bmm_ppu", yaml_path=str(yaml_path))
    small_expand = loader.get_expand_config(
        "bmm_ppu_small_m", yaml_path=str(yaml_path)
    )
    small_configs = loader.ops_get_configs(
        "bmm_ppu_small_m", yaml_path=str(yaml_path)
    )

    assert main_expand["strategy"] == ["default"] * 5
    assert len(main_configs) == 4512
    assert len(main_configs) <= 5000
    assert {config.kwargs["LOAD_MODE"] for config in main_configs} == {0, 1, 2, 3}
    assert small_expand["strategy"] == ["default"] * 5
    assert len(small_configs) == 248
    assert all("BLOCK_M" not in config.kwargs for config in small_configs)


def test_ppu_split_k_expand_configs_are_bounded_and_complete():
    loader = TunedConfigLoader()
    yaml_path = (
        Path(__file__).parents[1]
        / "src"
        / "flag_gems"
        / "runtime"
        / "backend"
        / "_thead"
        / "mm_ppu_expand.yaml"
    )

    for op_name, expected_count, expected_strategy, expected_splits in (
        ("mm_ppu_split_k", 4320, ["default"] * 3, {2, 4, 6, 7, 8}),
        ("bmm_ppu_split_k", 1728, ["default"] * 4, {2, 4, 6, 8}),
    ):
        expand = loader.get_expand_config(op_name, yaml_path=str(yaml_path))
        configs = loader.ops_get_configs(op_name, yaml_path=str(yaml_path))
        assert expand["strategy"] == expected_strategy
        assert len(configs) == expected_count
        assert len(configs) <= 5000
        assert {config.kwargs["SPLIT_K"] for config in configs} == expected_splits
        assert all("INTERLEAVED" in config.kwargs for config in configs)
