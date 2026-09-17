# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Exercise the real controller and CSV parser with isolated mock worker processes.

The mock replaces only accelerator launch commands. It emits the same rank
configuration/accuracy/profiler artifact schemas as the production workers.
These tests do not represent NPU accuracy or performance measurements.
"""

import csv
import json
import os
from pathlib import Path
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ops_transformer_catccos_mte.tuning import controller
from ops_transformer_catccos_mte.tuning.schema import CommConfig, Platform, enumerate_candidates
from ops_transformer_catccos_mte.tuning.worker import install_override


def business_case(operator="all_to_allv_grouped_mat_mul"):
    return {"id": "model_case_1", "operator": operator,
            "output_all_to_allv": operator == "all_to_allv_grouped_mat_mul",
            "world_size": 8, "expert_num": 256, "shape": {"m": 131072, "k": 2048, "n": 1024},
            "performance": {"warmup": 1, "align_iterations": 1, "iterations": 2,
                            "repeats": 2, "max_rank_skew_ratio": 1.0},
            "accuracy": {"repeats": 2}}


SPACE = {"commAivCount": [7, 14], "workersPerPeer": [1, 2],
         "commChunkBytes": [4096], "commUbStages": [2]}


@pytest.fixture
def mock_accelerator(monkeypatch, tmp_path):
    program = tmp_path / "mock_worker.py"
    program.write_text('''
import csv, json, os, sys
from pathlib import Path
stage, payload_path, output_path = sys.argv[1:]
payload = json.loads(Path(payload_path).read_text())
case, config = payload['case'], payload['config']
output = Path(output_path)
output.mkdir(parents=True, exist_ok=True)
with Path(os.environ['MTE_TEST_STAGE_LOG']).open('a') as log:
    log.write(json.dumps({'stage': stage, 'pid': os.getpid(), 'config': config}) + '\\n')
if stage == 'accuracy' and config['commAivCount'] == int(os.environ.get('MTE_TEST_FAIL_AIV', '-1')):
    raise SystemExit(3)
for rank in range(case['world_size']):
    if rank != int(os.environ.get('MTE_TEST_MISSING_RANK', '-1')):
        (output / f'effective_rank_{rank}.json').write_text(json.dumps({
            'rank': rank, 'config': config, 'platform_verified': True}))
    if stage == 'accuracy':
        folder = output / case['id'] / f'rank_{rank}'
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'summary.json').write_text(json.dumps({'rank': rank, 'passed': True}))
if stage == 'performance':
    folder = output / 'msprof'
    folder.mkdir(parents=True, exist_ok=True)
    name = ('KernelAdapter<Catccos::DGemm::Kernel::Ascend950AllToAllVGroupedMatmulMte>'
            if case['operator'] == 'all_to_allv_grouped_mat_mul' else
            'KernelAdapter<Catccos::DGemm::Kernel::GroupedMatmulAllToAllVMte>')
    perf = case['performance']
    with (folder / 'op_summary_mock.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Op Name', 'Device_id', 'Task Duration(us)', 'Task Start Time(us)'])
        for rank in range(case['world_size']):
            durations = [9000] * perf['warmup']
            for repeat in range(perf['repeats']):
                durations.extend([8000] * perf['align_iterations'])
                for iteration in range(perf['iterations']):
                    # Max rank alternates: maximum-of-medians is the wrong metric.
                    duration = (10 if iteration % 2 == 0 else 30) if rank == 0 else (20 if iteration % 2 == 0 else 25)
                    durations.append(duration * 7 / max(config['commAivCount'], 1))
            if rank == 0 and os.environ.get('MTE_TEST_EXTRA_LAUNCH') == '1':
                durations.append(1)
            for index, duration in enumerate(durations):
                writer.writerow([name, rank, duration, index * 10000])
            if os.environ.get('MTE_TEST_NOISE') == '1':
                # The baseline UDMA classes lack the Mte suffix. Neither
                # those rows nor the initialization probe belong in timing.
                writer.writerow([name.replace('Mte>', '>'), rank, 123456, 0])
                writer.writerow(['KernelAdapter<MtePeerProbe>', rank, 234567, 0])
''', encoding="utf-8")
    lifecycle = tmp_path / "lifecycle.jsonl"
    monkeypatch.setenv("MTE_TEST_STAGE_LOG", str(lifecycle))
    original_process = controller.run_process_group
    original_which = controller.shutil.which
    launched = []

    def command(case, stage, payload, output, port):
        return [sys.executable, str(program), stage, str(payload), str(output)]

    def execute(argv, environment, logfile, timeout):
        # Keep the controller's real msprof wrapper construction and remove
        # only that executable in this CPU test; run a genuine fresh process.
        launched.append(argv)
        if argv[0] == "test-msprof":
            argv = argv[argv.index(sys.executable):]
        return original_process(argv, environment, logfile, timeout)

    monkeypatch.setattr(controller, "_worker_command", command)
    monkeypatch.setattr(controller, "run_process_group", execute)
    monkeypatch.setattr(controller.shutil, "which", lambda name: "test-msprof" if name == "msprof" else original_which(name))
    monkeypatch.setattr(controller, "environment_fingerprint", lambda: {"mock_build": "stable"})
    return lifecycle, launched


@pytest.mark.parametrize("operator", ["all_to_allv_grouped_mat_mul", "grouped_mat_mul_all_to_allv"])
def test_evaluate_real_process_artifacts_and_iteration_maxrank_metric(mock_accelerator, tmp_path, operator):
    plan = controller.make_plan([business_case(operator)], Platform(), SPACE)
    entry = plan["cases"][0]
    result = controller.evaluate(entry, entry["candidates"][0], plan, tmp_path / "run", timeout=5, port=29500)
    assert result["status"] == "passed", result
    assert result["latency_us"]["p50"] == 25
    assert result["metric_definition"] == "p50 of iteration-wise maximum device duration"
    lifecycle, commands = mock_accelerator
    launches = [json.loads(line) for line in lifecycle.read_text().splitlines()]
    assert [item["stage"] for item in launches] == ["accuracy", "performance"]
    assert launches[0]["pid"] != launches[1]["pid"]
    assert commands[1][0] == "test-msprof"
    report = controller.read_json(tmp_path / "run/performance/profile_report.json")
    assert len(report["iterations"]) == 4
    assert report["occurrence_selection"]["warmup_skipped"] == 1
    assert report["occurrence_selection"]["primer_skipped_per_repeat"] == 1


def test_real_worker_missing_rank_stops_before_profiling(mock_accelerator, monkeypatch, tmp_path):
    monkeypatch.setenv("MTE_TEST_MISSING_RANK", "6")
    plan = controller.make_plan([business_case()], Platform(), SPACE)
    entry = plan["cases"][0]
    result = controller.evaluate(entry, entry["candidates"][0], plan, tmp_path / "run", timeout=5, port=29500)
    assert result["status"] == "accuracy_validation_failure"
    assert "rank 6" in result["errors"][0]
    assert len(mock_accelerator[1]) == 1


def test_real_profiler_rejects_extra_kernel_occurrence(mock_accelerator, monkeypatch, tmp_path):
    monkeypatch.setenv("MTE_TEST_EXTRA_LAUNCH", "1")
    plan = controller.make_plan([business_case()], Platform(), SPACE)
    entry = plan["cases"][0]
    result = controller.evaluate(entry, entry["candidates"][0], plan, tmp_path / "run", timeout=5, port=29500)
    assert result["status"] == "performance_validation_failure"
    assert "expected" in result["errors"][0] and "found" in result["errors"][0]


@pytest.mark.parametrize("operator", ["all_to_allv_grouped_mat_mul", "grouped_mat_mul_all_to_allv"])
def test_profiler_excludes_udma_and_peer_probe_rows(mock_accelerator, monkeypatch, tmp_path, operator):
    monkeypatch.setenv("MTE_TEST_NOISE", "1")
    plan = controller.make_plan([business_case(operator)], Platform(), SPACE)
    entry = plan["cases"][0]
    result = controller.evaluate(entry, entry["candidates"][0], plan, tmp_path / "run", timeout=5, port=29500)
    assert result["status"] == "passed", result
    assert result["latency_us"]["p50"] == 25
    report = controller.read_json(tmp_path / "run/performance/profile_report.json")
    assert report["expected_target_launches_per_device"] == 7


def test_real_search_resume_failure_retry_and_replay(mock_accelerator, monkeypatch, tmp_path):
    plan = controller.make_plan([business_case()], Platform(), SPACE)
    output = tmp_path / "search"
    monkeypatch.setenv("MTE_TEST_FAIL_AIV", "14")
    first = controller.search(plan, output, budget=2, topk=0)
    assert first["complete"]
    assert first["cases"][0]["config"]["commAivCount"] == 7
    records = controller.read_json(output / "results.json")
    assert [item["status"] for item in records] == ["passed", "accuracy_process_failure"]
    count = len(mock_accelerator[1])
    monkeypatch.delenv("MTE_TEST_FAIL_AIV")
    controller.search(plan, output, budget=2, topk=0, resume=True)
    assert len(mock_accelerator[1]) == count  # failed contexts are not implicitly reused/retried
    second = controller.search(plan, output, budget=2, topk=0, resume=True, retry_failed=True)
    assert second["cases"][0]["config"]["commAivCount"] == 14
    assert second["cases"][0]["latency_us"]["p50"] == 12.5
    assert len(mock_accelerator[1]) == count + 2
    assert (output / "results.csv").is_file()
    code = controller.main(["replay", "--best", str(output / "best.json"),
                            "--output", str(tmp_path / "replay"), "--topk", "0"])
    assert code == 0
    replay = controller.read_json(tmp_path / "replay/best.json")
    assert replay["cases"][0]["config"] == second["cases"][0]["config"]
    assert replay["cases"][0]["latency_us"]["p50"] == 12.5
    assert len(mock_accelerator[1]) == count + 4


def test_retest_and_stress_use_fresh_real_processes(mock_accelerator, tmp_path):
    plan = controller.make_plan([business_case()], Platform(), SPACE)
    best = controller.search(plan, tmp_path / "run", budget=1, topk=1, retest_factor=3, stress_repeats=5)
    assert best["cases"][0]["retested"] and best["cases"][0]["stress_passed"]
    lifecycle, commands = mock_accelerator
    launched = [json.loads(line) for line in lifecycle.read_text().splitlines()]
    assert [item["stage"] for item in launched] == ["accuracy", "performance", "accuracy", "performance", "accuracy"]
    assert len({item["pid"] for item in launched}) == 5
    records = controller.read_json(tmp_path / "run/results.json")
    assert [item["phase"] for item in records] == ["search", "retest", "stress"]
    payload = controller.read_json(Path(records[-1]["artifact_dir"]) / "candidate.json")
    assert payload["case"]["accuracy"]["repeats"] == 5
    payload = controller.read_json(Path(records[1]["artifact_dir"]) / "candidate.json")
    assert payload["case"]["performance"]["iterations"] == 6


def test_resume_preserves_orphan_attempt_and_uses_clean_profiler_directory(mock_accelerator, monkeypatch, tmp_path):
    plan = controller.make_plan([business_case()], Platform(), SPACE)
    original = controller.evaluate
    orphan = []

    def interrupted(entry, candidate, plan, root, **kwargs):
        stale = root / "performance/msprof/op_summary_interrupted.csv"
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_text("stale rows from interrupted run")
        orphan.append(stale)
        raise KeyboardInterrupt()

    monkeypatch.setattr(controller, "evaluate", interrupted)
    with pytest.raises(KeyboardInterrupt):
        controller.search(plan, tmp_path / "run", budget=1, topk=0)
    assert not (tmp_path / "run/results.json").exists()
    monkeypatch.setattr(controller, "evaluate", original)
    result = controller.search(plan, tmp_path / "run", budget=1, topk=0, resume=True)
    assert result["complete"]
    records = controller.read_json(tmp_path / "run/results.json")
    assert records[0]["attempt"] == 1
    assert Path(records[0]["artifact_dir"]).name == "search_1"
    assert orphan[0].read_text() == "stale rows from interrupted run"


@pytest.mark.parametrize("raw_ub", [248 * 1024, 256 * 1024])
def test_ub_boundary_and_actual_report_are_equivalent(monkeypatch, tmp_path, raw_ub):
    from types import SimpleNamespace
    declared = Platform()
    actual = Platform(ub_bytes=raw_ub)
    assert declared.payload_budget_bytes == actual.payload_budget_bytes == 246 * 1024
    edge = CommConfig(7, 1, 123 * 1024, 2)
    edge.validate(business_case(), actual)
    with pytest.raises(ValueError, match="UB budget"):
        CommConfig(7, 1, 123 * 1024 + 32, 2).validate(business_case(), actual)
    package = SimpleNamespace(init_mte_group=lambda: None)
    runtime = SimpleNamespace(_get_tuning_platform=lambda: {"block_dim": 32, "ub_bytes": raw_ub},
                              _set_tuning_override=lambda values: None,
                              _get_requested_comm_config=lambda: edge.as_tuple())
    monkeypatch.setenv("RANK", "0")
    install_override(package, runtime, {"case": business_case(), "config": edge.to_dict(),
                                        "platform": declared.to_dict()}, tmp_path)
    package.init_mte_group()
    assert controller.read_json(tmp_path / "effective_rank_0.json")["platform"]["ub_bytes"] == raw_ub


def test_worker_limits_match_host_constraints():
    with pytest.raises(ValueError, match="64"):
        CommConfig(65, 65, 4096, 1).validate(business_case(), Platform(block_dim=128))
    small = business_case()
    small["output_all_to_allv"] = False
    with pytest.raises(ValueError, match="rankSize"):
        CommConfig(1, 1, 4096, 1).validate(small, Platform(block_dim=4))


def test_packaged_business_manifest_covers_six_frozen_modes():
    from ops_transformer_catccos_mte.validation.case_schema import load_case_manifest
    manifest = ROOT / "ops_transformer_catccos_mte/validation/cases/mte_frozen_profiles.yaml"
    cases = load_case_manifest(manifest)
    plan = controller.make_plan(cases, Platform())
    controller.validate_plan(plan)
    assert len(plan["cases"]) == 6
    modes = {(entry["case"]["operator"], entry["case"]["shape"]["k"], entry["case"]["shape"]["n"],
              entry["case"]["output_all_to_allv"]) for entry in plan["cases"]}
    assert len(modes) == 6
    for entry in plan["cases"]:
        case = entry["case"]
        accepted, _ = enumerate_candidates(case, Platform(ub_bytes=248 * 1024))
        assert {c.as_tuple() for c in accepted} == {CommConfig.from_dict(c).as_tuple() for c in entry["candidates"]}
        if case["operator"] == "grouped_mat_mul_all_to_allv":
            expected = (896, 31) if case["shape"]["k"] == 1024 else (768, 8)
            assert (entry["frozen_geometry"]["stageRows"], entry["frozen_geometry"]["localCopyTileRows"]) == expected


def test_worker_uses_distinct_resolved_json_for_each_rank(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import ops_transformer_catccos_mte as package
    from ops_transformer_catccos_mte import _build_versions
    from ops_transformer_catccos_mte.tuning import worker
    payload = {"case": business_case(), "config": CommConfig(7, 1, 4096, 2).to_dict(),
               "platform": Platform().to_dict()}
    candidate = tmp_path / "candidate.json"
    controller.write_json(candidate, payload)
    monkeypatch.setattr(_build_versions, "WITH_TUNING", True)
    monkeypatch.setattr(package, "init_mte_group", lambda: None)
    original_import = worker.importlib.import_module
    read_paths = []

    def main():
        path = Path(sys.argv[sys.argv.index("--resolved-cases") + 1])
        assert controller.read_json(path)["cases"][0] == payload["case"]
        read_paths.append(path)
        return 0

    fake = SimpleNamespace(__file__="test_accuracy_worker.py", main=main)
    monkeypatch.setattr(worker.importlib, "import_module", lambda name, *args:
                        fake if name.endswith("validation.accuracy_worker") else original_import(name, *args))
    for rank in (0, 1):
        monkeypatch.setenv("RANK", str(rank))
        assert worker.main(["--stage", "accuracy", "--payload", str(candidate),
                            "--output-dir", str(tmp_path / "accuracy")]) == 0
    assert read_paths[0].name == "resolved_case_rank_0.json"
    assert read_paths[1].name == "resolved_case_rank_1.json"
    assert all(path.exists() for path in read_paths)


def test_timeout_kills_descendant_worker_process(tmp_path):
    marker = tmp_path / "descendant_survived.txt"
    child = "import pathlib,time; time.sleep(1.5); pathlib.Path({!r}).write_text('survived')".format(str(marker))
    launcher = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{!r}]); time.sleep(30)".format(child)
    result = controller.run_process_group([sys.executable, "-c", launcher], os.environ.copy(),
                                          tmp_path / "process.log", 0.4)
    assert result["status"] == "timeout"
    time.sleep(1.7)
    assert not marker.exists(), "external timeout leaked a descendant rank process"
