# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""CPU contracts for legal MTE candidates, scheduling and resumable execution."""

import copy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ops_transformer_catccos_mte.tuning import controller
from ops_transformer_catccos_mte.tuning.schema import (
    CommConfig, Platform, enumerate_candidates, frozen_geometry, remote_worker_budget, worker_tasks,
)


def case(n=1024, export=True, operator="all_to_allv_grouped_mat_mul"):
    return {"id": "model_case_1", "operator": operator, "output_all_to_allv": export,
            "world_size": 8, "expert_num": 256, "shape": {"m": 131072, "k": 2048, "n": n},
            "performance": {"warmup": 1, "align_iterations": 1, "iterations": 2, "repeats": 1,
                            "max_rank_skew_ratio": 1.0}, "accuracy": {"repeats": 2}}


def space(count=(7, 14), chunk=(4096,)):
    return {"commAivCount": list(count), "workersPerPeer": [1, 2],
            "commChunkBytes": list(chunk), "commUbStages": [2]}


def test_frozen_geometry_and_remote_worker_budget():
    assert frozen_geometry(case())["stageRows"] == 2048
    assert frozen_geometry(case())["localCopyTileRows"] == 16
    assert frozen_geometry(case(512))["workspaceStages"] == 3
    assert frozen_geometry(case(512))["mSplit"] == 2
    assert frozen_geometry(case(export=False))["localCopyTileRows"] == 64
    assert remote_worker_budget(case(), Platform()) == 24
    assert remote_worker_budget(case(512), Platform()) == 28
    assert remote_worker_budget(case(export=False), Platform()) == 31
    assert remote_worker_budget(case(operator="grouped_mat_mul_all_to_allv", export=False), Platform()) == 31


@pytest.mark.parametrize("config,message", [
    (CommConfig(7, 2, 4096, 2), "divisible"),
    (CommConfig(28, 4, 4096, 2), "commAivCount"),
    (CommConfig(7, 1, 4097, 2), "aligned"),
    (CommConfig(7, 1, 131072, 2), "UB budget"),
    (CommConfig(7, 1, 4096, 3), "1 or 2"),
    (CommConfig(8, 1, 4096, 2), "idle"),
    (CommConfig(True, 1, 4096, 2), "integer"),
])
def test_rejects_invalid_configuration(config, message):
    with pytest.raises(ValueError, match=message):
        config.validate(case(), Platform())


def test_schema_rejects_legacy_or_joint_dimensions():
    value = CommConfig(7, 1, 4096, 2).to_dict()
    for name in ("commNpuSplit", "commDataSplit", "m0", "stageRows"):
        with pytest.raises(ValueError, match="exactly"):
            CommConfig.from_dict(dict(value, **{name: 1}))


def test_prunes_and_deduplicates_without_changing_candidate_values():
    accepted, rejected = enumerate_candidates(case(), Platform(), space(count=(7, 7, 14, 28)))
    assert [c.as_tuple() for c in accepted] == [(7, 1, 4096, 2), (14, 2, 4096, 2)]
    assert len(rejected) == 4
    assert all(item["reason"] for item in rejected)


def test_ep1_zero_worker_case_is_explicit():
    one = dict(case(export=False), world_size=1, expert_num=32)
    CommConfig(0, 1, 4096, 1).validate(one, Platform())
    with pytest.raises(ValueError, match="EP1"):
        CommConfig(1, 1, 4096, 1).validate(one, Platform())
    accepted, _ = enumerate_candidates(one, Platform())
    assert len(accepted) == 12 and all(c.commAivCount == 0 for c in accepted)
    assert {c.commUbStages for c in accepted} == {1, 2}
    assert len({c.commChunkBytes for c in accepted}) == 6


def test_group_lane_assignment_covers_each_chunk_once_with_nondivisible_peer_count():
    config = CommConfig(6, 2, 4096, 2).validate(case(), Platform())
    peers = [0, 1, 2, 4, 5, 6, 7]
    counts = {peer: peer + 1 for peer in peers}
    assignments = worker_tasks(config, peers, counts)
    work = [pair for values in assignments.values() for pair in values]
    assert sorted(work) == sorted((peer, chunk) for peer in peers for chunk in range(counts[peer]))
    assert len(work) == len(set(work))
    assert set(peer for peer, _ in assignments[0]) == {peers[i] for i in (0, 3, 6)}


def test_plan_isolates_operator_export_mode_and_detects_editing():
    cases = [case(), case(export=False), case(export=False, operator="grouped_mat_mul_all_to_allv")]
    plan = controller.make_plan(cases, Platform(), space())
    controller.validate_plan(plan)
    assert len({entry["key"] for entry in plan["cases"]}) == 3
    modified = copy.deepcopy(plan)
    modified["cases"][0]["candidates"][0]["commChunkBytes"] = 8192
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        controller.validate_plan(modified)


def test_plan_cli_never_imports_torch_or_extension(tmp_path):
    script = """
import sys
class Guard:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'torch_npu') or fullname.endswith('._C'):
            raise AssertionError('accelerator imported: ' + fullname)
sys.meta_path.insert(0, Guard())
from ops_transformer_catccos_mte.tuning.controller import main
raise SystemExit(main(sys.argv[1:]))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    manifest = ROOT / "ops_transformer_catccos_mte/validation/cases/alltoallv_grouped_mat_mul.yaml"
    run = subprocess.run([sys.executable, "-c", script, "plan", "--cases", str(manifest),
                          "--case-id", "aligned_m128_n256_k256_ep2_uniform", "--output", str(tmp_path / "plan.json")],
                         env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert controller.read_json(tmp_path / "plan.json")["cases"][0]["case"]["id"] == "aligned_m128_n256_k256_ep2_uniform"


def test_process_group_has_external_timeout_and_log(tmp_path):
    result = controller.run_process_group([sys.executable, "-c", "import time; time.sleep(30)"],
                                          os.environ.copy(), tmp_path / "process.log", 0.2)
    assert result["status"] == "timeout"
    assert (tmp_path / "process.log").read_text()


def test_search_budget_resume_retests_and_fingerprint(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(controller, "environment_fingerprint", lambda: {"build": "one"})

    def evaluate(entry, candidate, plan, root, **kwargs):
        calls.append((candidate, kwargs))
        return {"case_key": entry["key"], "case_id": entry["case"]["id"],
                "candidate_id": controller.digest(candidate)[:16], "config": candidate,
                "status": "passed", "errors": [], "latency_us": {"p50": 100 / candidate["commAivCount"], "p90": 20},
                "artifact_dir": str(root)}

    monkeypatch.setattr(controller, "evaluate", evaluate)
    plan = controller.make_plan([case()], Platform(), space())
    result = controller.search(plan, tmp_path, budget=1, topk=1)
    assert result["complete"] and not result["search_exhaustive"]
    assert len(calls) == 2 and calls[-1][1]["factor"] == 2
    result = controller.search(plan, tmp_path, budget=1, topk=1, resume=True)
    assert result["search_exhaustive"]
    assert len(calls) == 4
    assert result["cases"][0]["config"]["commAivCount"] == 14
    monkeypatch.setattr(controller, "environment_fingerprint", lambda: {"build": "two"})
    with pytest.raises(ValueError, match="resume fingerprint mismatch"):
        controller.search(plan, tmp_path, resume=True, topk=1)


def test_accuracy_failure_never_profiles(monkeypatch, tmp_path):
    commands = []

    def failed(command, environment, logfile, timeout):
        commands.append(command)
        return {"status": "process_failure", "returncode": 1}

    monkeypatch.setattr(controller, "run_process_group", failed)
    plan = controller.make_plan([case()], Platform(), space())
    entry = plan["cases"][0]
    result = controller.evaluate(entry, entry["candidates"][0], plan, tmp_path, timeout=2, port=29500)
    assert result["status"] == "accuracy_process_failure"
    assert len(commands) == 1
    assert "--max-restarts=0" in commands[0]


def test_override_bridge_runs_after_init_and_checks_all_fields(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from ops_transformer_catccos_mte.tuning.worker import install_override
    order = []
    config = CommConfig(7, 1, 4096, 2)
    package = SimpleNamespace(init_mte_group=lambda: order.append("init"))
    runtime = SimpleNamespace(
        _get_tuning_platform=lambda: {"block_dim": 32, "ub_bytes": 262144},
        _set_tuning_override=lambda values: order.append(values),
        _get_requested_comm_config=lambda: config.as_tuple(),
    )
    monkeypatch.setenv("RANK", "0")
    install_override(package, runtime, {"config": config.to_dict(), "case": case(),
                                        "platform": Platform().to_dict()}, tmp_path)
    package.init_mte_group()
    assert order == ["init", config.as_tuple()]
    record = controller.read_json(tmp_path / "effective_rank_0.json")
    assert record["platform_verified"] and record["config"] == config.to_dict()
