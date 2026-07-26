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

import json
from pathlib import Path

import pytest
import torch

from batch.consistency.adapter import CaseAdapter
from batch.consistency.compare import compare_pair, tensor_md5
from batch.consistency.config import CaseConfigParser, ConsistencyMode, ModeRegistry
from batch.consistency.flow import ConsistencyFlow, DeterministicLevelGuard
from batch.consistency.model import ConsistencyCase, RunResult, TokenOrigin
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
from batch.consistency.metadata_stub import (
    MetadataStubPlanError,
    apply_metadata_transform,
    plan_single_to_multi_core_stub,
)
from batch.consistency.offline import (
    build_mode_cases,
    materialize_expected_output,
    mode_options_from_mapping,
)
from batch.consistency.relation_manifest import (
    RelationManifestError,
    build_case_relation,
    build_relation_manifest,
    make_baseline_id,
    validate_relation_manifest,
)
from batch.consistency.result_artifact import (
    ResultArtifactError,
    ResultStatus,
    aggregate_result_artifacts,
    write_result_artifact,
)
from batch.consistency.runtime_capability import (
    KIND as CAPABILITY_KIND,
    SCHEMA_VERSION as CAPABILITY_SCHEMA_VERSION,
    RuntimeCapabilityProbe,
    RuntimeCapabilityError,
    _custom_opp_identity_contract,
    is_capability_configuration_error,
    load_runtime_capability_evidence,
    require_runtime_capability_evidence,
)
from batch.consistency.reporting import (
    attach_provenance,
    build_input_summary,
    input_content_digest,
    stable_json_digest,
)
from batch.consistency.suite_config import (
    CONFIG_KIND,
    SuiteConfig,
    SuiteConfigError,
    resolve_case_values,
    resolve_runtime_case_config,
)


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
    with pytest.raises(ValueError, match="canonical names"):
        mode_options_from_mapping({"consistency_modes": "1,2"})
    with pytest.raises(ValueError, match="consistency_modes must be text"):
        mode_options_from_mapping({"consistency_modes": 1234})


def write_suite_config(tmp_path, document):
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_suite_config_merges_defaults_with_one_exact_case_override(tmp_path):
    case = make_case(batch=2, q_length=16)
    path = write_suite_config(
        tmp_path,
        {
            "schema_version": 1,
            "kind": CONFIG_KIND,
            "unmatched_case_policy": "error",
            "defaults": {"profile": "blackbox-basic", "seed": 7},
            "cases": [
                {
                    "id": "unit-token-modes",
                    "match": {"operator": "smla", "case_name": "unit"},
                    "config": {
                        "modes": ["token-split", "shape-change"],
                        "token_split_sizes": [4, 12],
                        "common_tokens": 12,
                        "derived_extra": 8,
                    },
                }
            ],
        },
    )
    resolved = SuiteConfig.load(str(path)).resolve(case, "/tmp/unit.pt")
    assert resolved.config_id == "unit-token-modes"
    assert resolved.seed == 7
    assert resolved.modes == (
        ConsistencyMode.TOKEN_SPLIT,
        ConsistencyMode.SHAPE_CHANGE,
    )
    assert resolved.token_split_sizes == (4, 12)
    assert (resolved.common_tokens, resolved.derived_extra) == (12, 8)


def test_suite_config_rejects_unmatched_and_ambiguous_cases(tmp_path):
    base = {
        "schema_version": 1,
        "kind": CONFIG_KIND,
        "defaults": {"modes": ["reorder"]},
    }
    unmatched = write_suite_config(
        tmp_path,
        {
            **base,
            "cases": [
                {
                    "id": "other",
                    "match": {"case_name": "other"},
                    "config": {},
                }
            ],
        },
    )
    with pytest.raises(SuiteConfigError, match="no suite config match"):
        SuiteConfig.load(str(unmatched)).resolve(make_case(), "/tmp/unit.pt")

    ambiguous = write_suite_config(
        tmp_path,
        {
            **base,
            "cases": [
                {"id": "by-op", "match": {"operator": "smla"}, "config": {}},
                {"id": "by-name", "match": {"case_name": "unit"}, "config": {}},
            ],
        },
    )
    with pytest.raises(SuiteConfigError, match="matches multiple configs"):
        SuiteConfig.load(str(ambiguous)).resolve(make_case(), "/tmp/unit.pt")


def test_suite_config_rejects_numeric_modes_and_invalid_shape_values():
    with pytest.raises(SuiteConfigError, match="canonical names"):
        resolve_case_values(
            {"modes": ["1", "2"]},
            make_case(),
            config_id="bad",
            source="unit",
        )
    with pytest.raises(SuiteConfigError, match="complete partition"):
        resolve_case_values(
            {"modes": ["split"], "split_groups": [[0], [1]]},
            make_case(batch=3),
            config_id="bad",
            source="unit",
        )
    with pytest.raises(SuiteConfigError, match="outside"):
        resolve_case_values(
            {"modes": ["split"], "split_groups": [[0], [3]]},
            make_case(batch=3),
            config_id="bad",
            source="unit",
        )
    with pytest.raises(SuiteConfigError, match="sum to 3"):
        resolve_case_values(
            {"modes": ["token-split"], "token_split_sizes": [1, 2]},
            make_case(q_length=4),
            config_id="bad",
            source="unit",
        )


def test_suite_config_computes_defaults_for_each_case_shape():
    first = resolve_case_values(
        {"modes": ["split", "token-split"]},
        make_case(batch=8, q_length=16),
        config_id="first",
        source="unit",
    )
    second = resolve_case_values(
        {"modes": ["split", "token-split"]},
        make_case(batch=6, q_length=10),
        config_id="second",
        source="unit",
    )
    assert first.split_groups == (tuple(range(3)), tuple(range(3, 8)))
    assert second.split_groups == ((0, 1, 2), (3, 4, 5))
    assert first.token_split_sizes == (8, 8)
    assert second.token_split_sizes == (5, 5)

    single = resolve_case_values(
        {"profile": "blackbox-supported"},
        make_case(batch=1, q_length=1),
        config_id="single",
        source="unit",
    )
    assert single.split_groups is None
    assert single.token_split_sizes is None


def test_runtime_config_rejects_file_mixed_with_legacy_environment(tmp_path, monkeypatch):
    path = write_suite_config(
        tmp_path,
        {
            "schema_version": 1,
            "kind": CONFIG_KIND,
            "unmatched_case_policy": "defaults",
            "defaults": {"modes": ["reorder"]},
            "cases": [],
        },
    )
    monkeypatch.setenv("UNIT_CONSISTENCY_CONFIG_FILE", str(path))
    monkeypatch.setenv("UNIT_CONSISTENCY_MODES", "reorder")
    with pytest.raises(SuiteConfigError, match="cannot be combined"):
        resolve_runtime_case_config(
            make_case(), "/tmp/unit.pt", "UNIT_CONSISTENCY"
        )


def test_legacy_numeric_modes_are_normalized_before_formal_validation(monkeypatch):
    monkeypatch.setenv("LEGACY_CONSISTENCY_MODES", "1,2")
    with pytest.warns(DeprecationWarning):
        resolved = resolve_runtime_case_config(
            make_case(), "/tmp/unit.pt", "LEGACY_CONSISTENCY"
        )
    assert resolved.modes == (ConsistencyMode.REORDER, ConsistencyMode.SPLIT)
    assert resolved.source == "deprecated process-wide CLI/environment"


def level3_report(passed=True, observed=3):
    return {
        "pass": passed,
        "deterministic_level": {"requested": 3, "observed": observed},
    }


class FakeNpuState:
    def __init__(self):
        self.level = 0

    def set_deterministic_level(self, level):
        self.level = level

    def _get_deterministic_level(self):
        return self.level

    def set_device(self, _device_id):
        return None


def test_runtime_capability_probe_detects_delayed_rejection_and_restores_level():
    npu = FakeNpuState()

    def reject_smoke():
        raise RuntimeError(
            "torch_npu.npu.set_deterministic_level currently only supports configuring 0/1/2"
        )

    attempt = RuntimeCapabilityProbe(npu=npu, smoke=reject_smoke).run_attempt()
    assert attempt["status"] == "UNSUPPORTED"
    assert attempt["failure_stage"] == "npu_smoke"
    assert attempt["reason_code"] == "LEVEL3_ACCEPTED_BUT_NPU_REJECTED"
    assert attempt["set_call_returned"] and attempt["readback_matches"]
    assert not attempt["npu_smoke_passed"]
    assert attempt["levels"] == {"before": 0, "observed": 3, "restored": 0}
    assert attempt["restoration"]["status"] == "PASS"
    assert npu.level == 0


def test_runtime_capability_probe_requires_real_smoke_success():
    npu = FakeNpuState()
    attempt = RuntimeCapabilityProbe(
        npu=npu,
        smoke=lambda: {"op": "fake-test-smoke"},
    ).run_attempt()
    assert attempt["status"] == "SUPPORTED"
    assert attempt["npu_smoke_passed"]
    assert attempt["restoration"]["status"] == "PASS"
    assert npu.level == 0


def test_custom_opp_identity_contract_binds_path_order_and_manifest(monkeypatch, tmp_path):
    root = tmp_path / "custom_transformer"
    op_api_lib = root / "op_api" / "lib"
    op_api_lib.mkdir(parents=True)
    manifest = "a" * 64
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", f"{root}:/old/custom")
    monkeypatch.setenv("LD_LIBRARY_PATH", f"{op_api_lib}:/old/lib")
    environment = {
        "custom_opp_packages": [
            {
                "root": str(root.resolve()),
                "content_manifest_sha256": manifest,
                "identity_status": "CONTENT_HASHED_SOURCE_COMMIT_UNPROVEN",
                "errors": [],
            }
        ]
    }
    contract = _custom_opp_identity_contract(environment, str(root), manifest)
    assert contract["status"] == "PASS"
    monkeypatch.setenv("ASCEND_CUSTOM_OPP_PATH", f"/old/custom:{root}")
    contract = _custom_opp_identity_contract(environment, str(root), "b" * 64)
    assert contract["status"] == "ERROR"
    assert "expected custom OPP root is not first" in "; ".join(contract["errors"])
    assert "manifest does not match" in "; ".join(contract["errors"])


def test_capability_configuration_error_only_matches_custom_opp_identity():
    identity_error = {
        "status": "ERROR",
        "failure_stage": "environment_identity",
        "reason_code": "CUSTOM_OPP_IDENTITY_MISMATCH",
    }
    assert is_capability_configuration_error(identity_error)
    assert not is_capability_configuration_error(
        {**identity_error, "failure_stage": "npu_smoke"}
    )
    assert not is_capability_configuration_error(
        {**identity_error, "status": "UNSUPPORTED"}
    )


def capability_evidence(tmp_path, status, marker="default"):
    supported = status == "SUPPORTED"
    report = {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "kind": CAPABILITY_KIND,
        "claim_scope": "unit",
        "requested_level": 3,
        "status": status,
        "failure_stage": None if supported else "npu_smoke",
        "reason_code": None if supported else "LEVEL3_ACCEPTED_BUT_NPU_REJECTED",
        "reason": None if supported else "unit rejection",
        "attempt_count_requested": 1,
        "attempt_count_executed": 1,
        "restoration_pass": True,
        "device_init": {"status": "PASS", "device_id": 0},
        "attempts": [
            {
                "status": status,
                "readback_matches": True,
                "npu_smoke_passed": supported,
            }
        ],
        "environment": {"marker": marker},
    }
    path = tmp_path / f"runtime_capability_{status.lower()}_{marker}.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return load_runtime_capability_evidence(str(path))


def supported_capability_evidence(tmp_path):
    return capability_evidence(tmp_path, "SUPPORTED")


def unsupported_capability_evidence(tmp_path):
    return capability_evidence(tmp_path, "UNSUPPORTED")


def test_required_capability_binds_probe_to_test_device(tmp_path):
    evidence = supported_capability_evidence(tmp_path)
    path = evidence["source_report"]["path"]
    assert require_runtime_capability_evidence(path, 0)["probe"]["status"] == "SUPPORTED"
    with pytest.raises(RuntimeCapabilityError, match="does not match test device 1"):
        require_runtime_capability_evidence(path, 1)
    with pytest.raises(RuntimeCapabilityError, match="required before any NPU"):
        require_runtime_capability_evidence("", 0)


def test_result_artifact_requires_observed_level3_for_pass(tmp_path):
    with pytest.raises(ResultArtifactError, match="observed deterministic level 3"):
        write_result_artifact(
            str(tmp_path),
            operator="smla",
            input_data=make_case(),
            pt_path="/tmp/unit.pt",
            mode="reorder",
            status=ResultStatus.PASS,
            report=level3_report(observed=2),
            runtime_capability=supported_capability_evidence(tmp_path),
        )
    assert not list((tmp_path / "results").glob("*.json"))


def test_result_artifact_rejects_pass_conflicting_with_capability_probe(tmp_path):
    with pytest.raises(ResultArtifactError, match="non-supported runtime capability"):
        write_result_artifact(
            str(tmp_path),
            operator="smla",
            input_data=make_case(),
            pt_path="/tmp/unit.pt",
            mode="reorder",
            status=ResultStatus.PASS,
            report=level3_report(),
            runtime_capability=unsupported_capability_evidence(tmp_path),
        )


def test_result_artifact_requires_capability_evidence_for_pass(tmp_path):
    with pytest.raises(ResultArtifactError, match="requires runtime capability evidence"):
        write_result_artifact(
            str(tmp_path),
            operator="smla",
            input_data=make_case(),
            pt_path="/tmp/unit.pt",
            mode="reorder",
            status=ResultStatus.PASS,
            report=level3_report(),
        )


def test_result_artifact_rechecks_capability_source_file(tmp_path):
    capability = supported_capability_evidence(tmp_path)
    source_path = Path(capability["source_report"]["path"])
    source_path.write_text(source_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ResultArtifactError, match="source report digest mismatch"):
        write_result_artifact(
            str(tmp_path),
            operator="smla",
            input_data=make_case(),
            pt_path="/tmp/unit.pt",
            mode="reorder",
            status=ResultStatus.PASS,
            report=level3_report(),
            runtime_capability=capability,
        )


def test_result_artifact_aggregation_uses_json_statuses_only(tmp_path):
    case = make_case()
    capability = supported_capability_evidence(tmp_path)
    statuses = {
        "reorder": (ResultStatus.PASS, level3_report(), None),
        "split": (ResultStatus.UNSUPPORTED, None, "RUNTIME_LEVEL3_UNSUPPORTED"),
        "token-split": (ResultStatus.NOT_APPLICABLE, None, "MODE_NOT_SELECTED"),
        "shape-change": (ResultStatus.NOT_APPLICABLE, None, "MODE_NOT_SELECTED"),
    }
    for mode, (status, report, reason_code) in statuses.items():
        write_result_artifact(
            str(tmp_path),
            operator="smla",
            input_data=case,
            pt_path="/tmp/unit.pt",
            mode=mode,
            status=status,
            report=report,
            reason_code=reason_code,
            reason=reason_code,
            runtime_capability=capability,
        )
    summary = aggregate_result_artifacts(
        str(tmp_path), operator="smla", expected_case_count=1
    )
    assert summary["overall_status"] == "PARTIAL"
    assert summary["status_counts"] == {
        "PASS": 1,
        "FAIL": 0,
        "UNSUPPORTED": 1,
        "INVALID_TRANSFORM": 0,
        "NOT_APPLICABLE": 2,
        "CONFIG_ERROR": 0,
    }
    assert summary["missing_artifact_count"] == 0
    failed_harness = aggregate_result_artifacts(
        str(tmp_path),
        operator="smla",
        expected_case_count=1,
        execution_failure_count=1,
    )
    assert failed_harness["overall_status"] == "FAIL"
    assert failed_harness["pytest_process_failure_count"] == 1


def test_result_artifact_aggregation_detects_missing_mode_records(tmp_path):
    write_result_artifact(
        str(tmp_path),
        operator="smla",
        input_data=make_case(),
        pt_path="/tmp/unit.pt",
        mode="reorder",
        status=ResultStatus.PASS,
        report=level3_report(),
        runtime_capability=supported_capability_evidence(tmp_path),
    )
    summary = aggregate_result_artifacts(
        str(tmp_path), operator="smla", expected_case_count=1
    )
    assert summary["overall_status"] == "INCOMPLETE"
    assert summary["missing_artifact_count"] == 3


def test_result_aggregation_rejects_mixed_or_partial_capability_evidence(tmp_path):
    case = make_case()
    first = capability_evidence(tmp_path, "SUPPORTED", "first")
    second = capability_evidence(tmp_path, "SUPPORTED", "second")
    for mode, capability in (
        ("reorder", first),
        ("split", second),
        ("token-split", None),
    ):
        write_result_artifact(
            str(tmp_path),
            operator="smla",
            input_data=case,
            pt_path="/tmp/unit.pt",
            mode=mode,
            status=ResultStatus.UNSUPPORTED,
            reason_code="UNIT_UNSUPPORTED",
            reason="unit",
            runtime_capability=capability,
        )
    summary = aggregate_result_artifacts(str(tmp_path), operator="smla")
    errors = {item["error"] for item in summary["artifact_validation_errors"]}
    assert summary["overall_status"] == "FAIL"
    assert errors == {
        "Runtime capability evidence is missing from part of the result matrix",
        "Result matrix is bound to multiple runtime capability probes",
    }


def test_result_aggregation_enforces_case_matrix_and_reports_invalid_json(tmp_path):
    case = make_case()
    capability = supported_capability_evidence(tmp_path)
    per_case_modes = {
        "/tmp/first.pt": ("reorder", "split"),
        "/tmp/second.pt": ("token-split", "shape-change"),
    }
    for pt_path, modes in per_case_modes.items():
        for mode in modes:
            write_result_artifact(
                str(tmp_path),
                operator="smla",
                input_data=case,
                pt_path=pt_path,
                mode=mode,
                status=ResultStatus.PASS,
                report=level3_report(),
                runtime_capability=capability,
            )

    summary = aggregate_result_artifacts(
        str(tmp_path), operator="smla", expected_case_count=1
    )
    assert summary["artifact_count"] == 4
    assert summary["missing_artifact_count"] == 0
    assert summary["case_count"] == 2
    assert summary["unexpected_case_count"] == 1
    assert summary["overall_status"] == "INCOMPLETE"

    invalid_path = tmp_path / "results" / "invalid.json"
    invalid_path.write_text("{", encoding="utf-8")
    summary = aggregate_result_artifacts(
        str(tmp_path), operator="smla", expected_case_count=1
    )
    assert summary["overall_status"] == "FAIL"
    assert summary["artifact_validation_error_count"] == 1
    assert summary["artifact_validation_errors"][0]["artifact"] == str(invalid_path)
    assert (tmp_path / "consistency_result_summary.json").is_file()


def test_relation_manifest_prechecks_origins_and_reports_coverage():
    baseline = make_case(batch=3, q_length=4)
    baseline_summary = build_input_summary(baseline, role="baseline")
    fingerprint = input_content_digest(baseline_summary)
    baseline_id = make_baseline_id(baseline, fingerprint)
    cases = build_mode_cases(
        baseline,
        modes=("reorder", "split", "token-split", "shape-change"),
        split_groups=[[0], [1, 2]],
        token_split_sizes=[1, 3],
        common_tokens=3,
        derived_extra=2,
    )
    relations = []
    for index, case in enumerate(cases):
        materialize_expected_output(baseline, case)
        relations.append(
            build_case_relation(
                baseline,
                baseline_id,
                case,
                f"/tmp/derived_{index}.pt",
                f"fingerprint_{index}",
            )
        )
    manifest = build_relation_manifest(
        baseline,
        "/tmp/baseline.pt",
        fingerprint,
        "cpu-md5",
        "/tmp/cases.xlsx",
        "SMLA",
        relations,
    )
    validate_relation_manifest(manifest)
    assert manifest["precheck"]["status"] == "PASS"
    assert manifest["coverage"]["mode_counts"] == {
        "reorder": 1,
        "split": 2,
        "token-split": 2,
        "shape-change": 1,
    }
    assert manifest["coverage"]["missing_modes"] == []
    assert manifest["scenario_coverage"] == {
        "implemented": ["same-source-cross-call"],
        "supplemental_gaps": [
            "independent-cross-case",
            "single-call-intra-case",
        ],
    }
    assert all(
        relation["scenario"] == "same-source-cross-call"
        for relation in manifest["relations"]
    )
    assert manifest["coverage"]["source_landmark_hits"]["batch"]["max"] > 0
    assert all(
        relation["comparison_identity"]["origin_spans"]
        for relation in manifest["relations"]
    )


def test_relation_manifest_rejects_out_of_bounds_origin_and_tampering():
    baseline = make_case(batch=1)
    invalid = ConsistencyCase(
        name="invalid",
        input_data=make_case(batch=1),
        output_origins=[TokenOrigin(1, 0)],
        transform_meta={"mode_name": "reorder"},
    )
    invalid.input_data["_consistency_expected_scope"] = {"compared_q_lengths": [1]}
    with pytest.raises(RelationManifestError, match="source_batch"):
        build_case_relation(baseline, "baseline", invalid, "/tmp/invalid.pt", "digest")

    missing_oracle = ConsistencyCase(
        name="missing_oracle",
        input_data=make_case(batch=1),
        output_origins=[TokenOrigin(0, 0)],
        transform_meta={"mode_name": "token-split"},
    )
    missing_oracle.input_data["_consistency_expected_scope"] = {
        "compared_q_lengths": [1]
    }
    with pytest.raises(RelationManifestError, match="requires a semantic oracle"):
        build_case_relation(
            baseline, "baseline", missing_oracle, "/tmp/missing.pt", "digest"
        )

    case = build_mode_cases(
        baseline, modes=("token-split",), token_split_sizes=[1, 3]
    )[0]
    materialize_expected_output(baseline, case)
    fingerprint = input_content_digest(build_input_summary(baseline, role="baseline"))
    baseline_id = make_baseline_id(baseline, fingerprint)
    relation = build_case_relation(
        baseline, baseline_id, case, "/tmp/derived.pt", "derived-digest"
    )
    manifest = build_relation_manifest(
        baseline,
        "/tmp/baseline.pt",
        fingerprint,
        "cpu-md5",
        "/tmp/cases.xlsx",
        "SMLA",
        [relation],
    )
    manifest["relations"][0]["comparison_identity"]["origin_spans_sha256"] = "tampered"
    with pytest.raises(RelationManifestError, match="digest mismatch"):
        validate_relation_manifest(manifest)

    identity = manifest["relations"][0]["comparison_identity"]
    identity["origin_spans_sha256"] = stable_json_digest(identity["origin_spans"])
    manifest["relations"][0]["precheck"]["status"] = "FAIL"
    with pytest.raises(RelationManifestError, match="precheck is not PASS"):
        validate_relation_manifest(manifest)


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

    same_content = make_case()
    attach_provenance(same_content, source_pt=str(tmp_path / "other.pt"))
    other_summary = build_input_summary(same_content, role="baseline", mode="reorder")
    assert stable_json_digest(summary) != stable_json_digest(other_summary)
    assert input_content_digest(summary) == input_content_digest(other_summary)


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


def test_metadata_stub_planner_finds_candidate_without_mutating_metadata():
    case = make_case(batch=1, q_length=1, ori_length=256)
    case["input"]["cmp_kv"] = None
    natural = metadata_tensor([(0, [1, 0, 0, 0, 1, 0, 0, 0, 2])])
    before = natural.clone()
    plan = plan_single_to_multi_core_stub(natural, case)
    assert plan["status"] == "CANDIDATE_FOUND_BLOCKED"
    assert plan["candidate_count"] == 1
    assert plan["candidates"][0]["proposed_two_core_s2_block_ranges"] == [
        [0, 1],
        [1, 2],
    ]
    assert plan["mutation"] == {
        "generated": False,
        "applied": False,
        "allowed": False,
        "synthetic_metadata_must_not_count_as_natural_coverage": True,
    }
    assert "FD_RECORD_SLOT7_CONTRACT_UNVERIFIED" in plan["activation_blockers"]
    assert torch.equal(natural, before)


def test_metadata_stub_planner_reports_no_candidate_for_one_s2_block():
    case = make_case(batch=1, q_length=1, ori_length=128)
    case["input"]["cmp_kv"] = None
    natural = metadata_tensor([(0, [1, 0, 0, 0, 1, 0, 0, 0, 1])])
    plan = plan_single_to_multi_core_stub(natural, case)
    assert plan["status"] == "NO_CANDIDATE"
    assert plan["candidates"] == []


def test_metadata_transform_is_default_off_and_enforces_tensor_contract():
    natural = torch.arange(8, dtype=torch.int32)
    assert apply_metadata_transform(natural, {}, None) is natural

    before = natural.clone()

    def transform(cloned, input_data):
        assert input_data["marker"] == "unit"
        assert cloned.data_ptr() != natural.data_ptr()
        cloned[0] = 99
        return cloned

    transformed = apply_metadata_transform(natural, {"marker": "unit"}, transform)
    assert torch.equal(natural, before)
    assert transformed[0].item() == 99

    with pytest.raises(MetadataStubPlanError, match="changed shape"):
        apply_metadata_transform(natural, {}, lambda metadata, _: metadata[:4])
    with pytest.raises(MetadataStubPlanError, match="must return a torch.Tensor"):
        apply_metadata_transform(natural, {}, lambda _metadata, _: None)


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
        "metadata_stub.py",
        "model.py",
        "offline.py",
        "reporting.py",
        "relation_manifest.py",
        "result_artifact.py",
        "runtime_capability.py",
        "schema.py",
        "suite_config.py",
        "transform.py",
    }
    mismatched = [
        name
        for name in sorted(names)
        if (smla / name).read_bytes() != (mqsmla / name).read_bytes()
    ]
    assert not mismatched, f"pytest_v2 consistency mirror drift: {mismatched}"
