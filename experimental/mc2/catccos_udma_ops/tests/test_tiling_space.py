# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import importlib.util
import json
from pathlib import Path

import pytest

from validation.tiling_space import (
    ALL_TO_ALLV, GROUPED_MATMUL, MMAD_SHAPES, TILING_FIELDS,
    candidate_id, candidates_for_case, default_tiling, validate_tiling,
)


def case(operator=ALL_TO_ALLV, export=True, n=1024, k=2048):
    return {
        "operator": operator, "output_all_to_allv": export,
        "shape": {"m": 131072, "n": n, "k": k},
        "world_size": 8, "expert_num": 256,
    }


@pytest.mark.parametrize("export,n,expected", [
    (True, 1024, (256, 256, 2048, 2, 16, 1, 16, 8)),
    (True, 512, (256, 256, 3072, 3, 16, 2, 32, 4)),
    (False, 1024, (128, 256, 1536, 2, 64, 1, 0, 1)),
    (False, 512, (256, 128, 1536, 2, 64, 1, 0, 1)),
])
def test_delivery_business_profiles_remain_search_baselines(export, n, expected):
    profile = default_tiling(case(export=export, n=n))
    assert tuple(profile[key] for key in (
        "m0", "n0", "stage_rows", "workspace_stages", "local_copy_tile_rows",
        "m_split", "export_core_count", "self_copy_cores",
    )) == expected


@pytest.mark.parametrize("k,expected", [(1024, (128, 256, 14, 31)), (512, (256, 128, 12, 8))])
def test_inverse_profiles_use_effective_ub_clamped_default(k, expected):
    profile = default_tiling(case(GROUPED_MATMUL, False, 2048, k))
    assert tuple(profile[key] for key in ("m0", "n0", "comm_interval", "local_copy_tile_rows")) == expected


def test_validate_does_not_mutate_input_and_preserves_wire_order():
    profile = default_tiling(case())
    reordered = dict(reversed(list(profile.items())))
    result = validate_tiling(ALL_TO_ALLV, reordered, output_all_to_allv=True, aic_cores=32, world_size=8, n=1024, k=2048)
    assert result == profile
    assert tuple(result) == TILING_FIELDS[ALL_TO_ALLV]
    assert tuple(reordered) != tuple(result)


@pytest.mark.parametrize("field,value,match", [
    ("m0", True, "integer"), ("stage_rows", 0, "must be in"),
    ("stage_rows", 3000, "divisible"), ("m_split", 3, "compiled"),
    ("workspace_stages", 4, "compiled"), ("k0", 128, "compiled"),
    ("export_core_count", 33, "AI-core"), ("self_copy_cores", 26, "worker pool"),
    ("local_copy_tile_rows", 256, "UB capacity"),
    ("comm_block_m", 1 << 31, "must be in"),
])
def test_invalid_dispatch_and_resource_profiles_are_rejected(field, value, match):
    profile = default_tiling(case())
    profile[field] = value
    with pytest.raises(ValueError, match=match):
        validate_tiling(ALL_TO_ALLV, profile, output_all_to_allv=True, aic_cores=32, world_size=8, n=1024, k=2048)


def test_workspace_boundary_and_valid_copy_tails():
    profile = default_tiling(case())
    profile.update(stage_rows=6400, comm_block_m=97, local_copy_tile_rows=31)
    # 2 * 6400 * 8 * 2048 * 2 == exactly 400 MiB; tail rows are legal.
    validate_tiling(ALL_TO_ALLV, profile, output_all_to_allv=True, aic_cores=32, world_size=8, k=2048)
    profile["stage_rows"] += 256
    with pytest.raises(ValueError, match="workspace"):
        validate_tiling(ALL_TO_ALLV, profile, output_all_to_allv=True, world_size=8, k=2048)


def test_non_export_cannot_select_export_only_instances_or_unused_worker_knob():
    profile = default_tiling(case(export=False))
    for override in ({"workspace_stages": 3}, {"m_split": 2}, {"export_core_count": 8}):
        with pytest.raises(ValueError, match="non-export"):
            validate_tiling(ALL_TO_ALLV, dict(profile, **override))


def test_local_rows_are_checked_against_actual_ub_and_full_inverse_n():
    profile = default_tiling(case())
    profile["local_copy_tile_rows"] = 128
    with pytest.raises(ValueError, match="UB capacity"):
        validate_tiling(ALL_TO_ALLV, profile, output_all_to_allv=True, ub_bytes=128 * 1024)
    inverse = default_tiling(case(GROUPED_MATMUL, False, 2048, 1024))
    inverse["local_copy_tile_rows"] = 32
    with pytest.raises(ValueError, match="UB capacity"):
        validate_tiling(GROUPED_MATMUL, inverse, n=2048)
    inverse["local_copy_tile_rows"] = 31
    validate_tiling(GROUPED_MATMUL, inverse, n=2048)


def test_inverse_interval_guards_actual_core_count_and_scheduler_geometry():
    profile = default_tiling(case(GROUPED_MATMUL, False, 2048, 1024))
    with pytest.raises(ValueError, match="divisible"):
        validate_tiling(GROUPED_MATMUL, profile, n=2048, aic_cores=30, world_size=8)
    profile["comm_interval"] = 1
    with pytest.raises(ValueError, match="too small"):
        validate_tiling(GROUPED_MATMUL, profile, n=2048, aic_cores=32, world_size=8)
    profile["comm_interval"] = 1 << 30
    with pytest.raises(ValueError, match="uint32"):
        validate_tiling(GROUPED_MATMUL, profile, n=2048, aic_cores=32, world_size=8)
    profile["comm_interval"] = 1000
    with pytest.raises(ValueError, match="workspace"):
        validate_tiling(GROUPED_MATMUL, profile, n=2048, aic_cores=32, world_size=8)


def test_completion_counter_and_mmad_grid_overflow_are_rejected_before_launch():
    profile = default_tiling(case())
    profile["comm_block_m"] = 1
    with pytest.raises(ValueError, match="completion counter"):
        validate_tiling(ALL_TO_ALLV, profile, output_all_to_allv=True, m=1 << 29, world_size=8)
    profile["stage_rows"] = 1 << 20
    with pytest.raises(ValueError, match="task grid"):
        validate_tiling(ALL_TO_ALLV, profile, output_all_to_allv=True, n=1 << 28, world_size=8, aic_cores=32)


@pytest.mark.parametrize("operator,export,n,k", [(ALL_TO_ALLV, True, 1024, 2048), (ALL_TO_ALLV, False, 512, 2048), (GROUPED_MATMUL, False, 2048, 1024)])
def test_reproducible_search_covers_compiled_variants_and_only_valid_profiles(operator, export, n, k):
    workload = case(operator, export, n, k)
    arguments = dict(max_candidates=40, seed=42, aic_cores=24)
    first = candidates_for_case(workload, **arguments)
    second = candidates_for_case(workload, **arguments)
    assert first == second
    assert len(first) == 40
    assert len({candidate_id(item) for item in first}) == len(first)
    assert {tuple(item[key] for key in ("m0", "n0", "k0")) for item in first} == set(MMAD_SHAPES[operator])
    for item in first:
        validate_tiling(operator, item, output_all_to_allv=export, n=n, k=k, world_size=8, aic_cores=24)


def test_custom_space_keeps_baseline_and_omits_equivalent_all_core_export():
    workload = case()
    space = {
        "mmad": [[256, 256, 256]], "comm_block_m": [32], "stage_rows": [2048],
        "workspace_stages": [2], "local_copy_tile_rows": [16], "m_split": [1],
        "export_core_count": [0, 32], "self_copy_cores": [8],
    }
    result = candidates_for_case(workload, space, strategy="exhaustive", max_candidates=None, aic_cores=32)
    assert result[0] == default_tiling(workload)
    assert len(result) == 2  # Baseline16cores, then one effective all-core candidate.
    assert result[1]["export_core_count"] == 0


def test_inverse_search_deduplicates_intervals_with_same_complete_m_groups():
    workload = case(GROUPED_MATMUL, False, 2048, 1024)
    result = candidates_for_case(workload, {"mmad": [[128, 256, 256]], "comm_interval": [2, 3], "local_copy_tile_rows": [8]}, strategy="exhaustive", max_candidates=None, aic_cores=32)
    assert len(result) == 2  # Baseline plus interval2/3, both producing one group.


@pytest.mark.parametrize("space", [{"comm_npu_split": [4]}, {"mmad": [[64, 64, 64]]}, {"stage_rows": []}, {"stage_rows": [True]}, {"mmad": "128,256,256"}])
def test_bad_or_ineffective_search_dimensions_fail_early(space):
    with pytest.raises(ValueError):
        candidates_for_case(case(), space)


def test_unsafe_old_hardware_profile_does_not_prevent_search():
    workload = case(n=512)
    assert default_tiling(workload)["export_core_count"] == 32
    result = candidates_for_case(workload, aic_cores=16, max_candidates=8)
    assert all(item["export_core_count"] <= 16 for item in result)


def _loader():
    path = Path(__file__).parents[1] / "ops_transformer_catccos_udma" / "tiling.py"
    spec = importlib.util.spec_from_file_location("_tiling_loader_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_tiling_config


def test_saved_best_profile_can_be_reused_and_missing_or_failed_case_rejected(tmp_path):
    workload = case()
    entry = dict(
        workload, case_id="customer_case", passed=True, valid=True,
        search_mode="performance_only", accuracy_checked=False, tiling=default_tiling(workload),
    )
    path = tmp_path / "best_tilings.json"
    path.write_text(json.dumps({"schema_version": 1, "cases": [entry]}), encoding="utf-8")
    assert _loader()(path, "customer_case") == entry["tiling"]
    with pytest.raises(ValueError, match="exactly one"):
        _loader()(path, "missing")
    entry["passed"] = False
    path.write_text(json.dumps({"schema_version": 1, "cases": [entry]}), encoding="utf-8")
    with pytest.raises(ValueError, match="no successfully measured"):
        _loader()(path, "customer_case")
