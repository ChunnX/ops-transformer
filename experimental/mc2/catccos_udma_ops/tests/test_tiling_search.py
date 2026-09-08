# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import copy
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from validation import search_tiling
from validation.case_schema import load_case_manifest
from validation.tiling_space import default_tiling
from validation.tiling_worker import request_digest, score_rank_samples, validate_rank_hardware


PROJECT = Path(__file__).parents[1]
CASES = PROJECT / "validation" / "cases" / "alltoallv_grouped_mat_mul.yaml"


def _measured_result(request, latency=10.0):
    case = request["case"]
    return dict(
        score_rank_samples([[latency] * case["performance"]["repeats"] for _ in range(case["world_size"])]),
        status="measured", request_digest=request_digest(request),
        environment=request["environment"], tiling=request["tiling"],
        search_mode="performance_only", accuracy_checked=False,
    )


def _candidate_request():
    case = load_case_manifest(CASES)[0]
    return {
        "mode": "candidate", "case": case, "world_size": case["world_size"],
        "search_mode": "performance_only",
        "tiling": default_tiling(case),
        "environment": {"ranks": [{
            "rank": rank, "aic_cores": 24, "hardware": [24, 256 * 1024],
        } for rank in range(case["world_size"])]},
    }


def test_scoring_uses_max_of_each_aligned_batch_before_median():
    # max(median(each rank)) would report 2 and hide alternating stragglers.
    score = score_rank_samples([[100.0, 1.0, 2.0], [1.0, 100.0, 2.0]])
    assert score["latency_us"] == 100.0
    assert score["worst_rank_samples_us"] == [100.0, 100.0, 2.0]


@pytest.mark.parametrize("samples", [[], [[]], [[1.0], [2.0, 3.0]], [[0.0]], [[float("nan")]], [[float("inf")]]])
def test_scoring_rejects_incomplete_or_invalid_measurements(samples):
    with pytest.raises(ValueError):
        score_rank_samples(samples)


def test_request_digest_detects_rank_disagreement():
    request = {"case": {"shape": {"n": 256}}, "tiling": {"stage_rows": 128}, "run_id": "one"}
    reordered = {"run_id": "one", "tiling": {"stage_rows": 128}, "case": {"shape": {"n": 256}}}
    assert request_digest(request) == request_digest(reordered)
    reordered["tiling"]["stage_rows"] = 256
    assert request_digest(request) != request_digest(reordered)


def test_rank_hardware_requires_same_schedule_and_build():
    ranks = [{"aic_cores": 24, "extension_sha256": "same"} for _ in range(2)]
    validate_rank_hardware(ranks)
    ranks[1]["aic_cores"] = 25
    with pytest.raises(ValueError, match="AIC core count"):
        validate_rank_hardware(ranks)
    ranks[1]["aic_cores"] = 24
    ranks[1]["extension_sha256"] = "different"
    with pytest.raises(ValueError, match="same operator build"):
        validate_rank_hardware(ranks)


@pytest.mark.parametrize("changed", ["plan", "cases", "environment", "settings", "run_id", "search_mode"])
def test_resume_rejects_workload_software_hardware_or_plan_change(changed):
    previous = {key: {"value": 1} for key in ("plan", "cases", "environment", "settings", "run_id", "search_mode")}
    current = copy.deepcopy(previous)
    current[changed] = {"value": 2}
    with pytest.raises(ValueError, match=changed):
        search_tiling.validate_resume(previous, current)
    search_tiling.validate_resume(previous, copy.deepcopy(previous))


def test_best_selection_accepts_unchecked_measurements_and_preserves_default_baseline():
    case = load_case_manifest(CASES)[0]
    tiling = default_tiling(case)
    flags = {"search_mode": "performance_only", "accuracy_checked": False}
    results = [
        dict(flags, case_id=case["id"], candidate_id="broken", status="launch_failed", latency_us=1.0, tiling=tiling),
        dict(flags, case_id=case["id"], candidate_id="default", status="measured", latency_us=20.0, tiling=tiling, is_default=True),
        dict(flags, case_id=case["id"], candidate_id="faster", status="measured", latency_us=10.0, tiling=tiling),
    ]
    best = search_tiling.build_best_results([case], results, {"machine": "customer"})
    assert best["cases"][0]["candidate_id"] == "faster"
    assert best["cases"][0]["speedup_vs_default"] == 2.0
    assert best["cases"][0]["measured_candidates"] == 2
    assert best["cases"][0]["passed"] is True
    assert best["cases"][0]["valid"] is True
    assert best["cases"][0]["accuracy_checked"] is False
    assert best["cases"][0]["search_mode"] == "performance_only"
    assert best["accuracy_checked"] is False
    assert best["search_mode"] == "performance_only"
    assert best["cases"][0]["output_all_to_allv"] is True
    assert best["environment"] == {"machine": "customer"}


def _launch_args(**overrides):
    values = dict(
        nnodes=2, node_rank=1, nproc_per_node=2,
        master_addr="10.0.0.1", master_port=29500, udma_master_port=29601,
        timeout=600,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_multi_node_command_uses_global_world_size_and_distinct_trial_ports(tmp_path):
    args = _launch_args()
    command, environment = search_tiling.launch_command(args, 4, tmp_path / "request.json", tmp_path, 3)
    assert "--nnodes=2" in command
    assert "--node-rank=1" in command
    assert "--nproc-per-node=2" in command
    assert "--master-addr=10.0.0.1" in command
    assert "--master-port=29506" in command
    assert "--max-restarts=0" in command
    assert environment["UDMA_MASTER_ADDR"] == "10.0.0.1"
    assert environment["UDMA_MASTER_PORT"] == "29607"


def test_same_parity_port_bases_are_rejected_before_the_probe(tmp_path):
    # Although trial 0's ports differ, trial 50 would otherwise reuse 29600 as
    # its torchrun port while a lagging node could still have SHMEM on it.
    with pytest.raises(ValueError, match="opposite parity"):
        search_tiling.launch_command(
            _launch_args(master_port=29500, udma_master_port=29600), 4, "probe.json", tmp_path, 0,
        )


@pytest.mark.parametrize("arguments,world_size", [
    ({"nnodes": 3}, 4), ({"nproc_per_node": 4}, 4),
    ({"master_port": 65535}, 4), ({"udma_master_port": 29500}, 4),
])
def test_launch_rejects_inconsistent_world_or_ports(arguments, world_size, tmp_path):
    with pytest.raises(ValueError):
        search_tiling.launch_command(_launch_args(**arguments), world_size, "request", tmp_path, 1)


def test_dry_run_generates_both_operator_plans_without_torch_import(tmp_path, monkeypatch):
    output = tmp_path / "plan"
    monkeypatch.setattr(sys, "argv", [
        "search_tiling", "--dry-run", "--output-dir", str(output), "--max-candidates", "2",
    ])
    monkeypatch.setattr(search_tiling, "run_isolated", lambda *args: pytest.fail("dry run launched a worker"))
    assert search_tiling.main() == 0
    plan = json.loads((output / "search_manifest.json").read_text(encoding="utf-8"))
    assert len(plan["cases"]) == 4
    assert len(plan["plan"]) == 8
    assert {case["operator"] for case in plan["cases"]} == {
        "all_to_allv_grouped_mat_mul", "grouped_mat_mul_all_to_allv",
    }
    assert plan["environment"]["world_sizes"] == {}
    assert plan["search_mode"] == "performance_only"
    assert plan["settings"]["search_mode"] == "performance_only"


def test_multi_node_resume_is_rejected_before_launch(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "search_tiling", "--nnodes", "2", "--run-id", "job", "--resume", "--output-dir", "results",
    ])
    with pytest.raises(SystemExit):
        search_tiling._parse_args()


def test_resume_reprobes_hardware_and_reuses_only_identical_trials(tmp_path, monkeypatch):
    output = tmp_path / "search"
    arguments = [
        "search_tiling", "--cases", str(CASES),
        "--case-id", "aligned_m128_n256_k256_ep2_uniform",
        "--max-candidates", "2", "--output-dir", str(output), "--run-id", "test-run",
    ]
    launches = []
    hardware_generation = ["initial"]

    def run(command, environment, output_dir, timeout):
        request_path = command[command.index("--request") + 1]
        request = json.loads(Path(request_path).read_text(encoding="utf-8"))
        launches.append(request["mode"])
        if request["mode"] == "probe":
            return {
                "status": "probed", "request_digest": request_digest(request),
                "search_mode": "performance_only", "accuracy_checked": False,
                "environment": {
                    "topology_label": "",
                    "ranks": [{
                        "rank": rank,
                        "aic_cores": 24, "hardware": [24, 256 * 1024],
                        "hostname": hardware_generation[0],
                    } for rank in range(2)],
                },
            }
        assert request["search_mode"] == "performance_only"
        return _measured_result(request)

    monkeypatch.setattr(search_tiling, "run_isolated", run)
    monkeypatch.setattr(sys, "argv", arguments)
    assert search_tiling.main() == 0
    assert launches == ["probe", "candidate", "candidate"]
    monkeypatch.setattr(sys, "argv", arguments + ["--resume"])
    assert search_tiling.main() == 0
    assert launches == ["probe", "candidate", "candidate", "probe"]
    best = json.loads((output / "best_tilings.json").read_text(encoding="utf-8"))
    assert best["cases"][0]["passed"] is True
    assert best["cases"][0]["accuracy_checked"] is False
    assert best["cases"][0]["search_mode"] == "performance_only"
    loader = runpy.run_path(str(PROJECT / "ops_transformer_catccos_udma" / "tiling.py"))["load_tiling_config"]
    assert loader(output / "best_tilings.json", best["cases"][0]["case_id"]) == best["cases"][0]["tiling"]
    hardware_generation[0] = "another-machine"
    with pytest.raises(ValueError, match="environment"):
        search_tiling.main()
    assert launches[-1] == "probe"


def test_expected_core_count_cannot_silently_limit_actual_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "search_tiling", "--cases", str(CASES), "--max-candidates", "1",
        "--output-dir", str(tmp_path / "search"), "--aic-cores", "20",
    ])

    def probe(command, environment, output_dir, timeout):
        request = json.loads(Path(command[command.index("--request") + 1]).read_text(encoding="utf-8"))
        assert request["mode"] == "probe"
        return {
            "status": "probed", "request_digest": request_digest(request),
            "search_mode": "performance_only", "accuracy_checked": False,
            "environment": {"ranks": [{"rank": rank, "aic_cores": 24, "hardware": [24, 256 * 1024]} for rank in range(2)]},
        }

    monkeypatch.setattr(search_tiling, "run_isolated", probe)
    with pytest.raises(ValueError, match="physical AIC count"):
        search_tiling.main()


@pytest.mark.parametrize("defect", [
    "missing_digest", "wrong_digest", "missing_accuracy_flag", "wrong_accuracy_flag",
    "missing_search_mode", "wrong_search_mode", "legacy_status", "legacy_accuracy",
    "duplicate_rank", "missing_samples", "missing_timing_rank", "missing_timing_repeat",
    "nan_latency", "wrong_score", "wrong_tiling", "wrong_metric",
])
def test_successful_result_requires_complete_matching_worker_evidence(defect):
    request = _candidate_request()
    result = _measured_result(request)
    search_tiling.validate_worker_result(result, request)
    result = copy.deepcopy(result)
    if defect == "missing_digest":
        result.pop("request_digest")
    elif defect == "wrong_digest":
        result["request_digest"] = "another-trial"
    elif defect == "missing_accuracy_flag":
        result.pop("accuracy_checked")
    elif defect == "wrong_accuracy_flag":
        result["accuracy_checked"] = True
    elif defect == "missing_search_mode":
        result.pop("search_mode")
    elif defect == "wrong_search_mode":
        result["search_mode"] = "accuracy_and_performance"
    elif defect == "legacy_status":
        result["status"] = "passed"
    elif defect == "legacy_accuracy":
        result["accuracy"] = [{"rank": 0, "passed": True}]
    elif defect == "duplicate_rank":
        result["environment"]["ranks"][1]["rank"] = 0
    elif defect == "missing_samples":
        result.pop("rank_samples_us")
    elif defect == "missing_timing_rank":
        result["rank_samples_us"].pop()
    elif defect == "missing_timing_repeat":
        result["rank_samples_us"][0].pop()
    elif defect == "nan_latency":
        result["latency_us"] = float("nan")
    elif defect == "wrong_score":
        result["latency_us"] = 1.0
    elif defect == "wrong_tiling":
        result["tiling"]["stage_rows"] *= 2
    elif defect == "wrong_metric":
        result["metric"] = "mean_of_rank_medians"
    with pytest.raises(ValueError):
        search_tiling.validate_worker_result(result, request)


def test_complete_performance_result_is_eligible_without_accuracy_configuration_or_reports():
    request = _candidate_request()
    request["case"].pop("accuracy")
    result = _measured_result(request)
    assert "accuracy" not in result
    search_tiling.validate_worker_result(result, request)


@pytest.mark.parametrize("latency", [None, 0.0, -1.0, True, float("nan"), float("inf")])
def test_best_selection_excludes_invalid_measurements(latency):
    request = _candidate_request()
    result = dict(
        _measured_result(request), case_id=request["case"]["id"],
        candidate_id="invalid", latency_us=latency,
    )
    best = search_tiling.build_best_results([request["case"]], [result], request["environment"])
    assert best["cases"] == []


def test_probe_success_cannot_omit_request_digest():
    request = {"mode": "probe", "world_size": 2}
    with pytest.raises(ValueError, match="request digest"):
        search_tiling.validate_worker_result({"status": "probed"}, request)


@pytest.mark.parametrize("missing_field", ["search_mode", "accuracy_checked"])
def test_probe_success_cannot_reuse_legacy_search_protocol(missing_field):
    request = {"mode": "probe", "search_mode": "performance_only", "world_size": 2}
    result = {
        "status": "probed", "request_digest": request_digest(request),
        "environment": _candidate_request()["environment"],
        "search_mode": "performance_only", "accuracy_checked": False,
    }
    search_tiling.validate_worker_result(result, request)
    result.pop(missing_field)
    with pytest.raises(ValueError, match="performance-only search protocol"):
        search_tiling.validate_worker_result(result, request)


@pytest.mark.skipif(os.name == "nt", reason="POSIX worker process-group cleanup")
def test_timeout_signals_the_worker_process_group(tmp_path, monkeypatch):
    class Process:
        pid = 12345

        def __init__(self):
            self.waits = 0

        def wait(self, timeout):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("torchrun", timeout)
            return -15

    process = Process()
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    result = search_tiling.run_isolated(["torchrun"], {}, tmp_path, 3)
    assert result["status"] == "timeout"
    assert [pid for pid, _ in calls] == [12345, 12345]
    assert calls[-1][1] == search_tiling.signal.SIGKILL


def test_nonzero_teardown_cannot_make_a_candidate_eligible(tmp_path, monkeypatch):
    class Process:
        def wait(self, timeout):
            return 1

    (tmp_path / "result.json").write_text(json.dumps({"status": "measured", "latency_us": 1.0}))
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: Process())
    result = search_tiling.run_isolated(["torchrun"], {}, tmp_path, 3)
    assert result["status"] == "launch_failed"
