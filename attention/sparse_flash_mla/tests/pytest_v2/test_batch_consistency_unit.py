#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CPU-only tests for the pytest_v2 consistency data path."""

from pathlib import Path

import pytest
import torch

from batch.consistency.adapter import CaseAdapter
from batch.consistency.compare import compare_pair, tensor_md5
from batch.consistency.config import CaseConfigParser, ConsistencyMode, ModeRegistry
from batch.consistency.flow import ConsistencyFlow, DeterministicLevelGuard
from batch.consistency.model import RunResult
from batch.consistency.schema import check_invariants, get_q_lengths
from batch.consistency.transform import (
    ActualInputSemanticOracle,
    InvalidTransformError,
    generate_split_groups,
    groups_from_split_sizes,
    transform_mode1_reorder,
    transform_mode2_split,
    transform_mode3_token_partition,
    transform_mode4_shape_change,
)
from batch.consistency.metadata_coverage import CoverageContext, MetadataCoverageAnalyzer, S2Category
from batch.consistency.metadata_layout import MetadataLayout
from batch.consistency.offline import mode_options_from_mapping
from batch.consistency.reporting import attach_provenance, build_input_summary


def make_case(batch=2, q_length=4, ori_length=8, physical_ori=None, mask_mode=0):
    physical_ori = physical_ori or ori_length
    q = torch.arange(batch * q_length * 4, dtype=torch.float32).reshape(batch, q_length, 1, 4)
    ori_kv = torch.arange(batch * physical_ori * 4, dtype=torch.float32).reshape(
        batch, physical_ori, 1, 4
    )
    cmp_physical = max((physical_ori + 3) // 4, 1)
    cmp_kv = torch.arange(batch * cmp_physical * 4, dtype=torch.float32).reshape(
        batch, cmp_physical, 1, 4
    )
    cu_q = torch.arange(0, (batch + 1) * q_length, q_length, dtype=torch.int32)
    seqused_q = torch.full((batch,), q_length, dtype=torch.int32)
    seqused_ori = torch.full((batch,), ori_length, dtype=torch.int32)
    seqused_cmp = torch.full((batch,), ori_length // 4, dtype=torch.int32)
    residual = torch.full((batch,), ori_length % 4, dtype=torch.int32)
    params = {
        "B": batch,
        "S1": q_length,
        "S2": physical_ori,
        "layout_q": "BSND",
        "layout_kv": "BSND",
        "seqused_q": [q_length] * batch,
        "seqused_ori_kv": [ori_length] * batch,
        "seqused_cmp_kv": [ori_length // 4] * batch,
        "cmp_residual_kv": [ori_length % 4] * batch,
        "return_softmax_lse": False,
    }
    metadata = {
        "N1": 1,
        "N2": 1,
        "D": 4,
        "B": batch,
        "max_seqlen_q": q_length,
        "max_seqlen_ori_kv": ori_length,
        "max_seqlen_cmp_kv": ori_length // 4,
        "K": 0,
        "cmp_ratio": 4,
        "ori_mask_mode": mask_mode,
        "cmp_mask_mode": 3 if mask_mode else 0,
        "ori_win_left": 127,
        "ori_win_right": 0,
        "layout_q": "BSND",
        "layout_kv": "BSND",
    }
    tensors = {
        "q": q,
        "ori_kv": ori_kv,
        "cmp_kv": cmp_kv,
        "ori_sparse_indices": None,
        "cmp_sparse_indices": None,
        "ori_block_table": None,
        "cmp_block_table": None,
        "cu_seqlens_q": cu_q,
        "cu_seqlens_ori_kv": None,
        "cu_seqlens_cmp_kv": None,
        "seqused_q": seqused_q,
        "seqused_ori_kv": seqused_ori,
        "seqused_cmp_kv": seqused_cmp,
        "cmp_residual_kv": residual,
        "sinks": torch.zeros(1),
        "softmax_scale": 1.0,
        "cmp_ratio": 4,
        "ori_mask_mode": mask_mode,
        "cmp_mask_mode": 3 if mask_mode else 0,
        "ori_win_left": 127,
        "ori_win_right": 0,
        "layout_q": "BSND",
        "layout_kv": "BSND",
    }
    return {
        "B": batch,
        "S1": q_length,
        "S2": physical_ori,
        "T1": batch * q_length,
        "T2": batch * physical_ori,
        "T3": batch * cmp_physical,
        "N1": 1,
        "N2": 1,
        "K": 0,
        "layout_q": "BSND",
        "layout_kv": "BSND",
        "template_run_mode": "HCA",
        "case_name": "unit",
        "params": params,
        "metadata_input": metadata,
        "input": tensors,
        "cpu_output": q.clone(),
        "softmax_lse": None,
    }


def make_mq_case():
    case = make_case()
    tensors = case.pop("input")
    metadata = case["metadata_input"]
    case.pop("B")
    case.pop("layout_q")
    case.pop("layout_kv")
    case["metadata_input"] = {
        **metadata,
        "num_heads_q": metadata.pop("N1"),
        "num_heads_kv": metadata.pop("N2"),
        "head_dim": metadata.pop("D"),
        "batch_size": metadata.pop("B"),
    }
    case["op_input"] = tensors
    return case


class FakeExecutor:
    def __call__(self, input_data):
        tensors = CaseAdapter(input_data).tensors
        return RunResult(output=tensors["q"].clone())


@pytest.fixture
def bypass_level_guard(monkeypatch):
    monkeypatch.setattr(DeterministicLevelGuard, "__enter__", lambda guard: guard)
    monkeypatch.setattr(DeterministicLevelGuard, "__exit__", lambda guard, *args: False)


def test_case_adapter_detects_both_operator_formats():
    assert CaseAdapter(make_case()).name == "smla"
    assert CaseAdapter(make_mq_case()).name == "mqsmla"


def test_mode1_reorders_all_batch_fields():
    case = transform_mode1_reorder(make_case(batch=3), order=[2, 0, 1])
    assert case.input_data["input"]["q"][:, 0, 0, 0].tolist() == [32.0, 0.0, 16.0]
    assert [origin.source_batch for origin in case.output_origins[::4]] == [2, 0, 1]
    check_invariants(case.input_data)


def test_requirement_mode2_default_is_three_plus_five():
    assert generate_split_groups(8) == [list(range(3)), list(range(3, 8))]


def test_explicit_split_sizes_and_groups():
    assert groups_from_split_sizes(8, [3, 5]) == [list(range(3)), list(range(3, 8))]
    cases = transform_mode2_split(make_case(batch=8), groups=[[0, 1, 2], [3, 4, 5, 6, 7]])
    assert [item.input_data["B"] for item in cases] == [3, 5]
    assert all(item.transform_meta["complete_partition"] for item in cases)


def test_mode3_complete_partition_no_mask():
    baseline = make_case(batch=1, q_length=16)
    cases = transform_mode3_token_partition(baseline, batch_id=0, split_sizes=[4, 12])
    assert [get_q_lengths(item.input_data, valid_only=True) for item in cases] == [[4], [12]]
    assert [item.transform_meta["token_range"] for item in cases] == [[0, 4], [4, 16]]
    assert all(item.transform_meta["semantic_oracle"]["pass"] for item in cases)


def test_mode3_position_sensitive_context_is_right_aligned():
    baseline = make_case(batch=1, q_length=4, ori_length=8, mask_mode=4)
    cases = transform_mode3_token_partition(baseline, batch_id=0, split_sizes=[1, 3])
    first = cases[0].input_data["input"]
    last = cases[1].input_data["input"]
    assert first["seqused_ori_kv"].tolist() == [5]
    assert last["seqused_ori_kv"].tolist() == [8]
    assert cases[0].transform_meta["semantic_oracle"]["checked_token_count"] == 1


def test_mode3_rejects_wrong_total():
    with pytest.raises(ValueError, match="sum"):
        transform_mode3_token_partition(make_case(batch=1), 0, [1, 2])


def test_mode4_common_prefix_grows_q_and_logical_kv():
    baseline = make_case(batch=1, q_length=4, ori_length=8, physical_ori=12, mask_mode=4)
    case = transform_mode4_shape_change(baseline, 0, common_tokens=3, derived_extra_tokens=2)
    assert get_q_lengths(case.input_data, valid_only=True) == [5]
    assert case.input_data["input"]["seqused_ori_kv"].tolist() == [9]
    assert case.transform_meta["baseline_extra_tokens"] == 1
    assert case.transform_meta["semantic_oracle"]["checked_token_count"] == 3


def test_mode4_no_mask_does_not_change_kv_context():
    baseline = make_case(batch=1, q_length=4, ori_length=8, physical_ori=12, mask_mode=0)
    case = transform_mode4_shape_change(baseline, 0, common_tokens=3, derived_extra_tokens=2)
    assert case.input_data["input"]["seqused_ori_kv"].tolist() == [8]


def test_actual_input_oracle_detects_changed_q():
    baseline = make_case(batch=1)
    derived = make_case(batch=1)
    derived["input"]["q"][0, 0, 0, 0] += 1
    with pytest.raises(InvalidTransformError, match="Q changed"):
        ActualInputSemanticOracle.validate_mapped_tokens(baseline, derived, [(0, 0, 0, 0)])


def test_actual_input_oracle_uses_bits_for_identical_float8_nan_payloads():
    payload = torch.tensor([0x7F], dtype=torch.uint8).view(torch.float8_e4m3fn)
    assert ActualInputSemanticOracle._tensor_bits_equal(payload, payload.clone())


def test_named_modes_and_explicit_config_parsing():
    assert ModeRegistry.parse("reorder,shape-change", warn_legacy=False) == (
        ConsistencyMode.REORDER,
        ConsistencyMode.SHAPE_CHANGE,
    )
    assert ModeRegistry.unsupported((ConsistencyMode.SHAPE_CHANGE,)) == ()
    assert CaseConfigParser.split_groups("0,1,2;3,4") == [[0, 1, 2], [3, 4]]
    assert CaseConfigParser.split_sizes("4096,12288", "token") == [4096, 12288]
    assert mode_options_from_mapping(
        {
            "consistency_modes": "reorder, split",
            "split_groups": "0,1;2,3",
            "token_split_sizes": "4,12",
            "mode_batch": 1,
            "common_tokens": 3,
            "derived_extra": 0,
        }
    ) == {
        "modes": ("reorder", "split"),
        "split_groups": [[0, 1], [2, 3]],
        "token_split_sizes": [4, 12],
        "mode_batch": 1,
        "common_tokens": 3,
        "derived_extra": 0,
    }


def test_md5_and_numeric_compare_are_both_reported():
    tensor = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    result = compare_pair(tensor, tensor.clone(), require_bitwise=True)
    assert result["pass"] and result["bitwise_equal"]
    assert tensor_md5(tensor) == result["expected_md5"]


def test_whitebox_input_summary_contains_source_shapes_values_and_digests(tmp_path):
    case = make_case()
    source = tmp_path / "baseline.pt"
    attach_provenance(case, source_pt=str(source), source_excel="cases.xlsx", source_sheet="SMLA")
    summary = build_input_summary(case, role="baseline", mode="reorder")
    assert summary["operator"] == "smla"
    assert summary["provenance"]["source_pt"] == str(source)
    assert summary["dimensions"]["S1_effective"] == [4, 4]
    assert summary["tensor_inputs"]["q"]["shape"] == [2, 4, 1, 4]
    assert len(summary["tensor_inputs"]["q"]["sha256"]) == 64
    assert summary["tensor_inputs"]["seqused_q"]["values"] == [4, 4]
    assert summary["tensor_inputs"]["seqused_q"]["values_complete"]
    assert summary["metadata_input_sha256"]


def test_pr8737_floor_and_requirement_ceil_formula_are_both_reported():
    boundary = MetadataCoverageAnalyzer.classify_s2(4097)
    assert boundary.pr8737_floor_tile == 128
    assert boundary.requirement_ceil_tile == 256
    assert boundary.quotient_formula_conflict
    assert boundary.tile_formula_conflict

    same_aligned_tile = MetadataCoverageAnalyzer.classify_s2(5121)
    assert same_aligned_tile.quotient_formula_conflict
    assert not same_aligned_tile.tile_formula_conflict


@pytest.mark.parametrize(
    "s2,category",
    [
        (4096, S2Category.EQUAL_128),
        (4095, S2Category.LESS_THAN_128),
        (5120, S2Category.GREATER_DIVISIBLE),
        (5121, S2Category.GREATER_REMAINDER),
    ],
)
def test_level1_s2_categories(s2, category):
    assert MetadataCoverageAnalyzer.classify_s2(s2).category == category


def metadata_tensor(fa_rows, fd_rows=()):
    raw = torch.zeros(1024, dtype=torch.int32)
    for core, row in fa_rows:
        raw[core * 9:core * 9 + 9] = torch.tensor(row, dtype=torch.int32)
    for core, row in fd_rows:
        start = 36 * 9 + core * 8
        raw[start:start + 8] = torch.tensor(row, dtype=torch.int32)
    return raw


def test_whitebox_expands_single_core_tokens():
    case = make_case(batch=1, q_length=2, ori_length=256)
    raw = metadata_tensor([(0, [1, 0, 0, 0, 1, 0, 0, 0, 4])])
    parsed = MetadataLayout().parse(raw)
    report = MetadataCoverageAnalyzer.analyze(
        parsed,
        context=CoverageContext.from_input_data(case),
    )
    assert report["assigned_token_count"] == 2
    assert report["single_core_token_count"] == 2
    assert report["level2_targets"]["all_tokens_single_core"]
    assert not report["errors"]


def test_whitebox_correlates_cross_core_fa_and_fd():
    case = make_case(batch=1, q_length=1, ori_length=256)
    raw = metadata_tensor(
        [
            (0, [1, 0, 0, 0, 0, 0, 1, 0, 1]),
            (1, [1, 0, 0, 1, 1, 0, 0, 1, 1]),
        ],
        [(0, [1, 0, 0, 0, 3, 0, 1, 0])],
    )
    report = MetadataCoverageAnalyzer.analyze(
        MetadataLayout().parse(raw),
        context=CoverageContext.from_input_data(case),
    )
    assert report["cross_core_token_count"] == 1
    assert report["level2_targets"]["has_cross_core_token"]
    assert report["level2_targets"]["core_one_cross_only"]
    assert not report["errors"]


def test_split_g_duplicate_ranges_are_not_false_cross_core():
    case = make_case(batch=1, q_length=1, ori_length=128)
    row = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    report = MetadataCoverageAnalyzer.analyze(
        MetadataLayout().parse(metadata_tensor([(0, row), (1, row)])),
        context=CoverageContext.from_input_data(case),
    )
    assert report["single_core_token_count"] == 1
    assert report["cross_core_token_count"] == 0


def test_flow_runs_all_modes_with_original_cpu_golden(bypass_level_guard):
    flow = ConsistencyFlow(FakeExecutor())
    assert flow.run_mode1(make_case(batch=2), order=[1, 0])["pass"]
    assert flow.run_mode2(make_case(batch=2), [[0], [1]])["pass"]
    assert flow.run_mode3(make_case(batch=1), 0, [1, 3])["pass"]
    assert flow.run_mode4(make_case(batch=1), 0, 3, 2)["pass"]


def test_dual_operator_consistency_core_is_an_exact_mirror():
    repo = next(
        parent
        for parent in Path(__file__).resolve().parents
        if parent.name == "ops-transformer_0716"
    )
    smla = repo / "attention/sparse_flash_mla/tests/pytest_v2/batch/consistency"
    mqsmla = repo / "attention/mixed_quant_sparse_flash_mla/tests/pytest_v2/batch/consistency"
    names = {
        "adapter.py",
        "compare.py",
        "config.py",
        "flow.py",
        "layouts.py",
        "metadata_coverage.py",
        "metadata_layout.py",
        "model.py",
        "offline.py",
        "reporting.py",
        "schema.py",
        "transform.py",
    }
    mismatched = [
        name
        for name in sorted(names)
        if (smla / name).read_bytes() != (mqsmla / name).read_bytes()
    ]
    assert not mismatched, f"pytest_v2 consistency mirror drift: {mismatched}"
