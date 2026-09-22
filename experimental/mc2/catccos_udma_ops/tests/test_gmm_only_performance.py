# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from validation import run_performance
from validation.case_schema import (
    load_case_manifest, performance_workload_digest, read_resolved_cases, write_resolved_cases,
)


PROJECT = Path(__file__).parents[1]
CASES = PROJECT / "validation" / "cases" / "grouped_mat_mul_all_to_allv.yaml"


@pytest.fixture
def worker(monkeypatch):
    """Load the real worker with device APIs replaced; keep the real route builder."""
    spec = importlib.util.spec_from_file_location(
        "gmm_test_routing", PROJECT / "ops_transformer_catccos_udma" / "routing.py",
    )
    routing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(routing)
    dist = SimpleNamespace(
        is_gloo_available=lambda: True, init_process_group=Mock(),
        barrier=Mock(), destroy_process_group=Mock(),
    )
    torch = SimpleNamespace(
        distributed=dist, int64="int64", bfloat16="bfloat16",
        Generator=Mock(return_value=SimpleNamespace(manual_seed=Mock())),
        tensor=Mock(side_effect=lambda values, **kw: SimpleNamespace(values=values, **kw)),
    )
    npu = SimpleNamespace(set_device=Mock(), synchronize=Mock())
    torch_npu = SimpleNamespace(npu=npu, npu_grouped_matmul=Mock(return_value=[object()]))
    # Deliberately expose no UDMA runtime/communication functions: pure GMM must
    # not call them, including in the finally block.
    udma = SimpleNamespace(build_rank_route_tables=routing.build_rank_route_tables)
    for name, module in (("torch", torch), ("torch.distributed", dist),
                         ("torch_npu", torch_npu), ("ops_transformer_catccos_udma", udma)):
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "validation.gmm_test_worker", PROJECT / "validation" / "performance_worker.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_make_tensor", Mock(
        side_effect=lambda shape, *args: SimpleNamespace(shape=shape),
    ))
    return module


@pytest.mark.parametrize("rank", [0, 7])
def test_gmm_worker_preserves_case_and_launch_schedule(worker, monkeypatch, tmp_path, rank):
    case = load_case_manifest(CASES)[-1]
    case["performance"]["gmm_only"] = True
    resolved = tmp_path / "resolved_case.json"
    write_resolved_cases(resolved, [case])
    monkeypatch.setattr(sys, "argv", [
        "performance_worker.py", "--resolved-case", str(resolved), "--output-dir", str(tmp_path),
    ])
    for name, value in (("RANK", rank), ("LOCAL_RANK", rank), ("WORLD_SIZE", 8)):
        monkeypatch.setenv(name, str(value))
    assert worker.main() == 0
    assert worker._make_tensor.call_count == 2
    assert worker._make_tensor.call_args_list[0].args[:3] == (
        (131072, 512), case["input_distribution"], "npu:{}".format(rank),
    )
    worker.torch.Generator.return_value.manual_seed.assert_called_once_with(2026 + rank)
    calls = worker.torch_npu.npu_grouped_matmul.call_args_list
    assert len(calls) == 169
    x, weight = calls[0].args
    assert x[0].shape == (131072, 512)
    assert weight[0].shape == (32, 512, 2048)
    groups = calls[0].kwargs["group_list"]
    assert groups.values == [4096 * (i + 1) for i in range(32)]
    assert groups.dtype == "int64" and groups.device == "npu:{}".format(rank)
    assert calls[0].kwargs["split_item"] == 3
    assert calls[0].kwargs["group_type"] == 0
    assert calls[0].kwargs["group_list_type"] == 0
    assert all(call.args[0] is x and call.args[1] is weight for call in calls)
    assert all(call.kwargs["group_list"] is groups for call in calls)
    assert worker.torch.tensor.call_count == 1
    assert worker.torch_npu.npu.synchronize.call_count == 4
    assert worker.dist.barrier.call_count == 5
    worker.dist.init_process_group.assert_called_once_with("gloo", rank=rank, world_size=8)
    worker.dist.destroy_process_group.assert_called_once()
    report = json.loads((tmp_path / "worker_rank_{}.json".format(rank)).read_text())
    assert report["gmm_group_sizes"] == [4096] * 32
    assert report["gmm_output_shape"] == [131072, 2048]
    assert report["performance_target"] == "grouped_matmul"


@pytest.mark.parametrize("m", [1, 31, 128])
def test_group_sizes_follow_route_including_empty_experts(worker, m):
    case = {"world_size": 4, "expert_num": 16, "shape": {"m": m}}
    for rank in range(4):
        sizes = worker._gmm_group_sizes(case, rank)
        assert len(sizes) == 4 and sum(sizes) == m
        if m == 128:
            assert sizes == [32] * 4
        elif m == 1:
            assert sizes.count(0) == 3
        else:
            # Compute counts independently from each source rank's local table.
            expected = [0] * 4
            for source in range(4):
                local, _ = worker.udma.build_rank_route_tables(m, source, 4, 16)
                for expert in range(4):
                    expected[expert] += local[rank * 4 + expert]
            assert sizes == expected


@pytest.mark.parametrize("gmm_only", [False, True])
def test_launcher_selects_target_and_rejects_cross_target_baseline(monkeypatch, tmp_path, gmm_only):
    case = load_case_manifest(CASES)[-1]
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"cases": [{
        "case_id": case["id"], "performance_workload_digest": performance_workload_digest(case),
        "valid": True, "latency_us": {"p50": 20}, "environment": {},
    }]}))
    output = tmp_path / "capture"
    argv = ["run_performance.py", "--cases", str(CASES), "--case-id", case["id"],
            "--output-dir", str(output), "--baseline", str(baseline)]
    if gmm_only:
        argv.append("--gmm-only")
        # No UDMA bootstrap should be consulted in the compute-only experiment.
        monkeypatch.setenv("UDMA_MASTER_PORT", "unused")
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(run_performance.shutil, "which", lambda name: "/fake/msprof")
    monkeypatch.setattr(run_performance, "_task_time_value", lambda *args: "l1")
    monkeypatch.setattr(run_performance, "environment_fingerprint", lambda: {})

    def capture(command, **kwargs):
        assert "--nproc-per-node=8" in command
        resolved = read_resolved_cases(command[command.index("--resolved-case") + 1])[0]
        assert resolved["shape"] == case["shape"]
        assert resolved["performance"].get("gmm_only", False) == gmm_only
        profile = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--output=")))
        with (profile / "op_summary_test.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["Op Name", "OP Type", "Device_id", "Task Start Time(us)", "Task Duration(us)"])
            name = "kernel_0" if gmm_only else "GroupedMatmulAllToAllVTla"
            kind = "GroupedMatmul" if gmm_only else "MIX_AIC"
            for device in range(8):
                for occurrence in range(169):
                    writer.writerow([name, kind, device, occurrence * 30, 10 + device * 0.1])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_performance.subprocess, "run", capture)
    assert run_performance.main() == (1 if gmm_only else 0)
    summary = json.loads((output / "summary.json").read_text())
    report = summary["cases"][0]
    assert report["valid"]
    assert len(report["iterations"]) == 150
    assert report["performance_target"] == ("grouped_matmul" if gmm_only else case["operator"])
    assert report["passed"] is not gmm_only
    if gmm_only:
        assert report["errors"] == ["baseline performance workload digest does not match"]
