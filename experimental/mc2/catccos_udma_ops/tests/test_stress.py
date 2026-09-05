# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import argparse
import json
import subprocess
import sys
import types
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import pytest

from validation.case_schema import CaseManifestError, load_case_manifest, select_cases
from validation import run_stress, stress_worker


PROJECT_ROOT = Path(__file__).parents[1]
CASES = PROJECT_ROOT / "validation" / "cases" / "alltoallv_grouped_mat_mul.yaml"


def workload(mode="chain", export=True):
    case = select_cases(load_case_manifest(CASES), mode=None, case_ids=["model_case_1"])[0]
    # Distinct widths catch accidental weight/activation transposes in chaining.
    case.update(world_size=2, expert_num=8, shape={"m": 128, "k": 512, "n": 256})
    if mode == "gmma2av":
        case.update(operator="grouped_mat_mul_all_to_allv", output_all_to_allv=False)
    else:
        case["output_all_to_allv"] = export
    return case


class FakeTensor:
    def __init__(self, backend, shape, dtype, device):
        self.backend = backend
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device

    def uniform_(self, *args, **kwargs):
        self.backend.random_generations += 1
        return self

    normal_ = uniform_

    def to(self, *, dtype, device):
        return self.backend.empty(self.shape, dtype=dtype, device=device)

    def data_ptr(self):
        assert not self.backend.calls, "tensor inspection must precede launches"
        return id(self)

    def cpu(self):
        raise AssertionError("stress must not read device outputs")

    def item(self):
        raise AssertionError("stress must not read device scalars")


class Backend:
    """Record call order and fail on allocations/synchronization inside a run."""

    def __init__(
        self, monkeypatch, mode, iterations, fail_drain=False, fail_call=None,
        sync_each_call=False, fail_monitored_barrier=None,
    ):
        self.calls = []
        self.sync_positions = []
        self.barrier_positions = []
        self.monitored_barrier_positions = []
        self.new_groups = []
        self.random_generations = 0
        self.route_generations = 0
        self.total_calls = iterations * (2 if mode == "chain" else 1)
        self.fail_drain = fail_drain
        self.fail_call = fail_call
        self.sync_each_call = sync_each_call
        self.fail_monitored_barrier = fail_monitored_barrier
        self.allocations = []
        self.stream = types.SimpleNamespace(synchronize=self.synchronize)
        self.torch = types.ModuleType("torch")
        self.torch.__path__ = []
        self.torch.empty = self.empty
        self.torch.float32 = "float32"
        self.torch.bfloat16 = "bfloat16"
        self.torch.Generator = lambda **kwargs: types.SimpleNamespace(manual_seed=lambda seed: None)
        self.torch.no_grad = nullcontext
        dist = types.ModuleType("torch.distributed")
        dist.is_gloo_available = lambda: True
        dist.init_process_group = lambda *args, **kwargs: None
        dist.destroy_process_group = lambda *args, **kwargs: None
        dist.new_group = self.new_group
        dist.monitored_barrier = self.monitored_barrier
        dist.barrier = self.barrier
        self.torch.distributed = dist
        torch_npu = types.ModuleType("torch_npu")
        torch_npu.npu = types.SimpleNamespace(
            set_device=lambda device: None,
            current_stream=lambda: self.stream,
            stream=self.stream_context,
            synchronize=self.synchronize,
        )
        udma = types.ModuleType("ops_transformer_catccos_udma")
        udma.init_udma_group = lambda **kwargs: None
        udma.destroy_udma_group = lambda: None
        udma.build_rank_route = self.build_route
        udma.all_to_allv_grouped_mat_mul = lambda *args, **kwargs: self.launch("a2avgmm", args, kwargs)
        udma.grouped_mat_mul_all_to_allv = lambda *args, **kwargs: self.launch("gmma2av", args, kwargs)
        for name, module in (("torch", self.torch), ("torch.distributed", dist),
                             ("torch_npu", torch_npu), ("ops_transformer_catccos_udma", udma)):
            monkeypatch.setitem(sys.modules, name, module)
        for key, value in {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "2",
                           "CATCCOS_UDMA_VALIDATE_ROUTES": "0", "UDMA_MASTER_PORT": "29601",
                           "UDMA_MASTER_ADDR": "127.0.0.1"}.items():
            monkeypatch.setenv(key, value)
        monkeypatch.delenv("UDMA_HEAP_SIZE", raising=False)

    def empty(self, shape, *, dtype, device="cpu"):
        assert not self.calls, "device buffers and inputs must be allocated once before the loop"
        result = FakeTensor(self, shape, dtype, device)
        self.allocations.append(result)
        return result

    def build_route(self, m, rank, world_size, expert_num, device):
        self.route_generations += 1
        return (self.empty((expert_num,), dtype="int64", device=device),
                self.empty((expert_num,), dtype="int64", device=device), m)

    def stream_context(self, stream):
        assert stream is self.stream
        return nullcontext()

    def synchronize(self):
        self.sync_positions.append(len(self.calls))
        if not self.sync_each_call:
            assert len(self.calls) in (0, self.total_calls), "synchronization between operator calls"
        if self.fail_drain and self.calls:
            raise RuntimeError("device drain failed")

    def barrier(self):
        self.barrier_positions.append(len(self.calls))
        assert not self.calls, "rank barrier inside stress loop"

    def new_group(self, **kwargs):
        group = object()
        self.new_groups.append((group, kwargs))
        return group

    def monitored_barrier(self, *, group, timeout):
        assert self.new_groups and group is self.new_groups[0][0]
        assert timeout > timedelta(0)
        self.monitored_barrier_positions.append(len(self.calls))
        if len(self.monitored_barrier_positions) == self.fail_monitored_barrier:
            raise RuntimeError("monitored barrier failed")

    def launch(self, name, args, kwargs):
        if len(self.calls) == self.fail_call:
            raise RuntimeError("submission failed")
        self.calls.append((name, args, kwargs))
        assert kwargs["out"].device == "npu:0"
        assert args[2].dtype == args[3].dtype == "int64"
        assert args[2].device == args[3].device == "npu:0"
        if name == "a2avgmm":
            assert kwargs["received_rows"] == 128  # Host integer, never device sum().item().
            assert kwargs["out"].shape == (128, 256)
            if kwargs["output_all_to_allv"]:
                assert kwargs["all_to_allv_out"].shape == (128, 512)
                return kwargs["out"], kwargs["all_to_allv_out"]
        return kwargs["out"]


@pytest.mark.parametrize("mode,export", [("a2avgmm", True), ("a2avgmm", False),
                                         ("gmma2av", False), ("chain", True), ("chain", False)])
@pytest.mark.parametrize("iterations", [1, 7])
def test_worker_reuses_inputs_and_outputs_without_inter_call_sync(monkeypatch, tmp_path, mode, export, iterations):
    backend = Backend(monkeypatch, mode, iterations)
    report = stress_worker.run_worker(workload(mode, export), mode, iterations, tmp_path)
    pattern = ["a2avgmm", "gmma2av"] if mode == "chain" else [mode]
    assert [call[0] for call in backend.calls] == pattern * iterations
    assert backend.sync_positions == [0, len(pattern) * iterations]
    assert backend.barrier_positions == [0]
    assert backend.random_generations == (3 if mode == "chain" else 2)
    assert backend.route_generations == 1
    for name in pattern:
        calls = [call for call in backend.calls if call[0] == name]
        first_args, first_kwargs = calls[0][1:]
        for _, args, kwargs in calls:
            assert all(current is initial for current, initial in zip(args, first_args))
            assert kwargs["out"] is first_kwargs["out"]
            assert kwargs.get("all_to_allv_out") is first_kwargs.get("all_to_allv_out")
    if mode == "chain":
        _, a_args, a_kwargs = backend.calls[0]
        _, b_args, b_kwargs = backend.calls[1]
        assert b_args[0] is a_kwargs["out"]
        assert b_args[0] is not a_kwargs["all_to_allv_out"]
        assert b_args[1].shape == (4, 256, 512)
        assert b_kwargs["out"].shape == a_args[0].shape == (128, 512)
        assert b_kwargs["out"] is not a_args[0]
        assert b_args[2] is a_args[2] and b_args[3] is a_args[3]
    assert report["passed"]
    assert report["completed_iterations"] == iterations
    assert report["completed_calls"] == {name: iterations for name in pattern}


def test_worker_can_diagnose_every_operator_call(monkeypatch, tmp_path, capsys):
    backend = Backend(monkeypatch, "chain", 2, sync_each_call=True)
    report = stress_worker.run_worker(
        workload(), "chain", 2, tmp_path,
        sync_each_call=True, sync_timeout_seconds=17,
    )
    assert [call[0] for call in backend.calls] == ["a2avgmm", "gmma2av"] * 2
    assert backend.sync_positions == [0, 1, 2, 3, 4]
    assert backend.barrier_positions == [0]
    assert backend.monitored_barrier_positions == [1, 2, 3, 4]
    assert len(backend.new_groups) == 1
    assert backend.new_groups[0][1] == {
        "backend": "gloo", "timeout": timedelta(seconds=17),
    }
    output = capsys.readouterr().out
    for sequence, (iteration, name) in enumerate([
        (1, "a2avgmm"), (1, "gmma2av"),
        (2, "a2avgmm"), (2, "gmma2av"),
    ], start=1):
        prefix = "rank=0 iteration={} seq={} op={}".format(iteration, sequence, name)
        for stage in (
            "before_launch", "before_npu_sync", "after_npu_sync", "after_gloo_barrier",
        ):
            assert "{} {}".format(prefix, stage) in output
    assert report["synchronized_calls"] == {"a2avgmm": 2, "gmma2av": 2}
    assert report["last_debug_event"] == {
        "rank": 0, "iteration": 2, "sequence": 4,
        "operator": "gmma2av", "stage": "after_gloo_barrier",
    }


def test_debug_barrier_failure_records_the_last_reached_stage(monkeypatch, tmp_path, capsys):
    Backend(
        monkeypatch, "chain", 2, sync_each_call=True,
        fail_monitored_barrier=2,
    )
    with pytest.raises(RuntimeError, match="monitored barrier failed"):
        stress_worker.run_worker(
            workload(), "chain", 2, tmp_path,
            sync_each_call=True, sync_timeout_seconds=17,
        )
    report = json.loads((tmp_path / "worker_rank_0.json").read_text())
    assert report["submitted_calls"] == {"a2avgmm": 1, "gmma2av": 1}
    assert report["synchronized_calls"] == {"a2avgmm": 1, "gmma2av": 0}
    assert report["last_debug_event"]["sequence"] == 2
    assert report["last_debug_event"]["stage"] == "after_npu_sync"
    output = capsys.readouterr().out
    assert "seq=2 op=gmma2av after_npu_sync" in output
    assert "seq=2 op=gmma2av after_gloo_barrier" not in output


def test_final_device_failure_is_not_reported_as_completed(monkeypatch, tmp_path):
    Backend(monkeypatch, "chain", 3, fail_drain=True)
    with pytest.raises(RuntimeError, match="device drain failed"):
        stress_worker.run_worker(workload(), "chain", 3, tmp_path)
    report = json.loads((tmp_path / "worker_rank_0.json").read_text())
    assert report["submitted_iterations"] == 3
    assert report["submitted_calls"] == {"a2avgmm": 3, "gmma2av": 3}
    assert report["completed_iterations"] is None
    assert report["completed_calls"] is None
    assert not report["passed"]
    assert report["stage"] == "draining"


def test_partial_pair_submission_preserves_unknown_completion(monkeypatch, tmp_path):
    Backend(monkeypatch, "chain", 3, fail_call=3)
    with pytest.raises(RuntimeError, match="submission failed"):
        stress_worker.run_worker(workload(), "chain", 3, tmp_path)
    report = json.loads((tmp_path / "worker_rank_0.json").read_text())
    assert report["submitted_iterations"] == 1
    assert report["submitted_calls"] == {"a2avgmm": 2, "gmma2av": 1}
    assert report["completed_calls"] is None
    assert not report["passed"]


def test_teardown_failure_is_reported_without_retrying_cleanup(monkeypatch, tmp_path):
    Backend(monkeypatch, "chain", 1)
    attempts = []

    def fail_teardown():
        attempts.append(1)
        raise RuntimeError("teardown failed")

    monkeypatch.setattr(sys.modules["ops_transformer_catccos_udma"], "destroy_udma_group", fail_teardown)
    with pytest.raises(RuntimeError, match="teardown failed"):
        stress_worker.run_worker(workload(), "chain", 1, tmp_path)
    report = json.loads((tmp_path / "worker_rank_0.json").read_text())
    assert attempts == [1]
    assert report["completed_iterations"] == 1
    assert not report["passed"]
    assert report["stage"] == "finalizing"
    assert "UDMA teardown failed" in report["errors"][0]


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "NaN", "abc"])
def test_invalid_iterations_rejected(value):
    with pytest.raises(argparse.ArgumentTypeError):
        stress_worker.positive_iterations(value)


@pytest.mark.parametrize("value", ["0", "-1", "NaN", "inf", "abc"])
def test_invalid_sync_timeout_rejected(value):
    with pytest.raises(argparse.ArgumentTypeError):
        stress_worker.positive_timeout_seconds(value)


def test_stress_reuses_accuracy_only_case_without_inheriting_run_phases():
    cases = load_case_manifest(CASES)
    selected = select_cases(cases, mode=None, include_tags=["normal"])
    assert len(selected) == 1
    assert selected[0]["run"] == ["accuracy"]
    stress_worker.validate_stress_case(selected[0], "chain")


def test_invalid_stress_workloads_fail_before_launch():
    case = workload()
    case["route"] = "unbalanced"
    with pytest.raises(CaseManifestError, match="balanced"):
        stress_worker.validate_stress_case(case, "chain")
    case["route"] = "balanced"
    case["expert_num"] = 128
    with pytest.raises(CaseManifestError, match="32 local experts"):
        stress_worker.validate_stress_case(case, "chain")
    with pytest.raises(CaseManifestError, match="requires a case for grouped_mat_mul_all_to_allv"):
        stress_worker.validate_stress_case(case, "gmma2av")
    with pytest.raises(ValueError, match="D2H"):
        stress_worker.validate_stress_environment({"CATCCOS_UDMA_VALIDATE_ROUTES": "1"})


@pytest.mark.parametrize("sync_args,sync_each_call,sync_timeout", [
    ([], False, None),
    (["--sync-each-call", "--sync-timeout-seconds", "17"], True, 17.0),
])
def test_launcher_selects_a_case_and_passes_exact_chain_count(
    monkeypatch, tmp_path, sync_args, sync_each_call, sync_timeout,
):
    monkeypatch.setenv("CATCCOS_UDMA_VALIDATE_ROUTES", "0")
    monkeypatch.setattr(run_stress, "environment_fingerprint", lambda: {})
    monkeypatch.setattr(run_stress, "runtime_log_roots", lambda environment: [])
    monkeypatch.setattr(run_stress, "snapshot_runtime_logs", lambda roots: {})
    monkeypatch.setattr(run_stress, "collect_runtime_logs", lambda *args: {})

    def fake_run(command, environment, case_root, timeout_seconds):
        assert command[command.index("--iterations") + 1] == "13"
        assert command[command.index("--mode") + 1] == "chain"
        assert "--max-restarts=0" in command
        assert not any("msprof" in item for item in command)
        assert ("--sync-each-call" in command) is sync_each_call
        if sync_each_call:
            assert command[command.index("--sync-timeout-seconds") + 1] == "17.0"
        assert environment["CATCCOS_UDMA_VALIDATE_ROUTES"] == "0"
        assert timeout_seconds == 120
        for rank in range(8):
            run_stress.write_json(case_root / "worker_rank_{}.json".format(rank), {
                "case_id": "model_case_1", "rank": rank, "mode": "chain", "passed": True,
                "completed_iterations": 13, "completed_calls": {"a2avgmm": 13, "gmma2av": 13},
                "sync_each_call": sync_each_call, "sync_timeout_seconds": sync_timeout,
                "elapsed_seconds": 1.0,
            })
        return {"returncode": 0, "timed_out": False}

    monkeypatch.setattr(run_stress, "run_command", fake_run)
    assert run_stress.main([
        "--mode", "chain", "--iterations", "13", "--case-id", "model_case_1",
        "--output-dir", str(tmp_path), "--timeout-seconds", "120",
    ] + sync_args) == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["passed"]
    assert summary["sync_each_call"] is sync_each_call
    assert summary["sync_timeout_seconds"] == sync_timeout
    assert summary["cases"][0]["requested_calls_per_rank"] == {"a2avgmm": 13, "gmma2av": 13}


def test_summary_requires_every_rank_and_device_completion(tmp_path):
    case = workload()
    result = {"returncode": 0, "timed_out": False}
    report = {"case_id": case["id"], "mode": "chain", "passed": True,
              "sync_each_call": False,
              "completed_iterations": 3, "completed_calls": {"a2avgmm": 3, "gmma2av": 3}}
    run_stress.write_json(tmp_path / "worker_rank_0.json", dict(report, rank=0))
    assert not run_stress.summarize_case(case, "chain", 3, tmp_path, result)["passed"]
    run_stress.write_json(tmp_path / "worker_rank_1.json", dict(report, rank=1))
    assert run_stress.summarize_case(case, "chain", 3, tmp_path, result)["passed"]
    timed_out = dict(result, timed_out=True)
    assert not run_stress.summarize_case(case, "chain", 3, tmp_path, timed_out)["passed"]
    report["completed_calls"] = None
    run_stress.write_json(tmp_path / "worker_rank_1.json", dict(report, rank=1))
    assert not run_stress.summarize_case(case, "chain", 3, tmp_path, result)["passed"]


def test_parent_deadline_terminates_workers_and_records_timeout(monkeypatch, tmp_path):
    process = types.SimpleNamespace(returncode=None)
    process.stdout = iter(())

    def wait(*, timeout):
        raise subprocess.TimeoutExpired("torchrun", timeout)

    process.wait = wait
    monkeypatch.setattr(run_stress.subprocess, "Popen", lambda *args, **kwargs: process)
    stopped = []

    def terminate(child):
        stopped.append(child)
        child.returncode = -9

    monkeypatch.setattr(run_stress, "_terminate_process_tree", terminate)
    result = run_stress.run_command(["torchrun"], {}, tmp_path, 2)
    assert stopped == [process]
    assert result["timed_out"] and result["returncode"] == -9


def test_parent_shows_worker_output_live_and_saves_it(monkeypatch, tmp_path, capsys):
    process = types.SimpleNamespace(
        returncode=0, stdout=iter(("rank 0 before_npu_sync\n", "rank 1 after_npu_sync\n")),
        wait=lambda **kwargs: None,
    )
    monkeypatch.setattr(run_stress.subprocess, "Popen", lambda *args, **kwargs: process)
    result = run_stress.run_command(["torchrun"], {}, tmp_path, 2)
    expected = "rank 0 before_npu_sync\nrank 1 after_npu_sync\n"
    assert capsys.readouterr().out == expected
    assert (tmp_path / "torchrun.log").read_text() == expected
    assert not result["timed_out"] and result["returncode"] == 0
