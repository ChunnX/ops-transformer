# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import copy
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from validation import common, tiling_worker
from validation.case_schema import load_case_manifest
from validation.tiling_space import default_tiling
from validation.tiling_protocol import EXECUTION_MODE, validate_batch_payload


PROJECT = Path(__file__).parents[1]


@pytest.mark.parametrize("operator,export", [
    ("all_to_allv_grouped_mat_mul", False),
    ("all_to_allv_grouped_mat_mul", True),
    ("grouped_mat_mul_all_to_allv", False),
])
def test_search_worker_measures_baseline_and_candidates_without_accuracy_imports(
    operator, export, tmp_path, monkeypatch,
):
    # Exercise the real worker entry point without Torch/NPU dependencies. Any
    # attempt to restore a hidden accuracy pass must fail at import time.
    for name in ("accuracy_worker", "validation.accuracy_worker", "precision", "validation.precision"):
        monkeypatch.setitem(sys.modules, name, None)
    calls = []
    torch = ModuleType("torch")
    dist = ModuleType("torch.distributed")
    torch.distributed = dist
    torch.empty = lambda *args, **kwargs: object()
    torch.ops = SimpleNamespace(ops_transformer_catccos_udma=SimpleNamespace(
        _tiling_hardware=lambda anchor: [24, 256 * 1024],
    ))
    dist.is_gloo_available = lambda: True
    dist.init_process_group = lambda *args, **kwargs: calls.append("init_dist")
    dist.destroy_process_group = lambda: calls.append("destroy_dist")
    dist.barrier = lambda: calls.append("barrier")

    def all_gather_object(results, payload):
        if isinstance(payload, dict):
            calls.append("gather_environment")
            for rank in range(len(results)):
                item = copy.deepcopy(payload)
                item["rank"] = rank
                if "environment" in item:
                    item["environment"].update(rank=rank, local_rank=rank)
                results[rank] = item
        else:
            calls.append("gather_timings")
            assert payload == [10.0, 1.0, 2.0]
            results[:] = [[10.0, 1.0, 2.0], [1.0, 10.0, 2.0]]

    dist.all_gather_object = all_gather_object
    torch_npu = ModuleType("torch_npu")
    torch_npu.npu = SimpleNamespace(
        set_device=lambda index: None,
        get_device_name=lambda index: "test-npu",
    )
    udma = ModuleType("ops_transformer_catccos_udma")
    udma.CATCCOS_COMMIT = "catccos-build"
    udma.SHMEM_COMMIT = "shmem-build"
    udma._C = SimpleNamespace(__file__="extension.so")
    udma.init_udma_group = lambda **kwargs: calls.append("init_udma")
    udma.destroy_udma_group = lambda: calls.append("destroy_udma")
    for module in (torch, dist, torch_npu, udma):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    for name, value in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "2")):
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(common, "environment_fingerprint", lambda: {"operator_package": "test-build"})
    monkeypatch.setattr(common, "udma_init_method_from_env", lambda: "tcp://127.0.0.1:12345")
    monkeypatch.setattr(common, "validation_udma_heap_size", lambda: 1 << 30)
    monkeypatch.setattr(tiling_worker, "_artifact_digest", lambda path: "extension-digest")
    monkeypatch.setattr(tiling_worker, "_source_digests", lambda package: {"tiling_worker.py": "source-digest"})

    def measure(case, tiling, rank, world_size, device, *modules):
        assert "accuracy" not in case
        assert calls[-1] == "init_udma"
        assert case["operator"] == operator
        assert case["output_all_to_allv"] is export
        calls.append("measure")
        return [10.0, 1.0, 2.0]

    monkeypatch.setattr(tiling_worker, "_measure", measure)

    def run(request, name):
        request["execution_mode"] = EXECUTION_MODE
        request_path = tmp_path / (name + ".json")
        output = tmp_path / name
        request_path.write_text(json.dumps(request), encoding="utf-8")
        monkeypatch.setattr(sys, "argv", [
            "tiling_worker", "--request", str(request_path), "--output-dir", str(output),
        ])
        assert tiling_worker.main() == 0
        return json.loads((output / "result.json").read_text(encoding="utf-8"))

    probe = run({"mode": "probe", "search_mode": "performance_only", "world_size": 2}, "probe")
    assert probe["status"] == "probed"
    assert "measure" not in calls
    case = load_case_manifest(PROJECT / "validation" / "cases" / "alltoallv_grouped_mat_mul.yaml")[0]
    case.update(operator=operator, output_all_to_allv=export)
    case.pop("accuracy")
    for is_default in (True, False):
        result = run({
            "mode": "candidate", "search_mode": "performance_only", "world_size": 2,
            "case": case, "tiling": default_tiling(case), "environment": probe["environment"],
            "is_default": is_default,
        }, "default" if is_default else "candidate")
        assert result["status"] == "measured"
        assert result["search_mode"] == "performance_only"
        assert result["accuracy_checked"] is False
        assert "accuracy" not in result
        assert result["latency_us"] == 10.0
        assert result["worst_rank_samples_us"] == [10.0, 10.0, 2.0]
    assert calls.count("measure") == 2
    assert calls.count("gather_timings") == 2
    assert calls.count("init_udma") == calls.count("destroy_udma") == 2
    assert calls.count("init_dist") == calls.count("destroy_dist") == 3
    assert not list(tmp_path.rglob("accuracy"))


@pytest.mark.parametrize("operator,export", [
    ("all_to_allv_grouped_mat_mul", False),
    ("all_to_allv_grouped_mat_mul", True),
    ("grouped_mat_mul_all_to_allv", False),
])
def test_persistent_worker_reuses_resources_and_events_through_partial_last_batch(
    operator, export, tmp_path, monkeypatch,
):
    from validation.tiling_space import candidate_id

    calls = []
    launches = []
    ticks = [0.0]
    for name in ("accuracy_worker", "validation.accuracy_worker", "precision", "validation.precision"):
        monkeypatch.setitem(sys.modules, name, None)
    torch = ModuleType("torch")
    dist = ModuleType("torch.distributed")
    torch.distributed = dist
    torch.bfloat16 = "bf16"
    torch.Generator = lambda **kwargs: SimpleNamespace(manual_seed=lambda seed: calls.append("seed"))

    def empty(*args, **kwargs):
        calls.append("empty")
        return object()

    torch.empty = empty
    torch.ops = SimpleNamespace(ops_transformer_catccos_udma=SimpleNamespace(
        _tiling_hardware=lambda anchor: [24, 256 * 1024],
    ))
    dist.is_gloo_available = lambda: True
    dist.init_process_group = lambda *args, **kwargs: calls.append("init_dist")
    dist.destroy_process_group = lambda: calls.append("destroy_dist")
    dist.barrier = lambda: calls.append("barrier")

    def gather(results, payload):
        for rank in range(len(results)):
            item = copy.deepcopy(payload)
            item["rank"] = rank
            if "environment" in item:
                item["environment"].update(rank=rank, local_rank=rank)
            if "samples" in item:
                item["samples"] = [[value + rank for value in samples] for samples in item["samples"]]
            results[rank] = item
        if "samples" in payload:
            calls.append("gather_timings")

    dist.all_gather_object = gather
    torch_npu = ModuleType("torch_npu")

    class Event:
        def __init__(self, **kwargs):
            calls.append("new_event")
            self.tick = None

        def record(self):
            self.tick = ticks[0]

        def synchronize(self):
            calls.append("event_sync")

        def elapsed_time(self, end):
            return (end.tick - self.tick) / 1000.0

    torch_npu.npu = SimpleNamespace(
        set_device=lambda index: None, get_device_name=lambda index: "test-npu",
        synchronize=lambda: calls.append("device_sync"), Event=Event,
    )
    udma = ModuleType("ops_transformer_catccos_udma")
    udma.CATCCOS_COMMIT = "catccos-build"
    udma.SHMEM_COMMIT = "shmem-build"
    udma._C = SimpleNamespace(__file__="extension.so")
    udma.init_udma_group = lambda **kwargs: calls.append("init_udma")
    udma.destroy_udma_group = lambda: calls.append("destroy_udma")

    def route(m, *args):
        calls.append("route")
        return object(), object(), m

    udma.build_rank_route = route

    def launch(*args, **kwargs):
        launches.append((args, kwargs))
        ticks[0] += 2.0  # deterministic two-microsecond kernel

    udma.all_to_allv_grouped_mat_mul = launch
    udma.grouped_mat_mul_all_to_allv = launch
    performance = ModuleType("validation.performance_worker")

    def make_tensor(*args):
        calls.append("input")
        return object()

    performance._make_tensor = make_tensor
    for module in (torch, dist, torch_npu, udma, performance):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    for name, value in (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "2")):
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(common, "environment_fingerprint", lambda: {"operator_package": "test-build"})
    monkeypatch.setattr(common, "udma_init_method_from_env", lambda: "tcp://127.0.0.1:12345")
    monkeypatch.setattr(common, "validation_udma_heap_size", lambda: 1 << 30)
    monkeypatch.setattr(tiling_worker, "_artifact_digest", lambda path: "extension-digest")
    monkeypatch.setattr(tiling_worker, "_source_digests", lambda package: {"worker": "same"})

    def run(request, name):
        request.update(search_mode="performance_only", execution_mode=EXECUTION_MODE, world_size=2)
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(request), encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["worker", "--request", str(path), "--output-dir", str(tmp_path / name)])
        assert tiling_worker.main() == 0
        return json.loads((tmp_path / name / "result.json").read_text(encoding="utf-8"))

    probe = run({"mode": "probe"}, "probe")
    case = load_case_manifest(PROJECT / "validation" / "cases" / "alltoallv_grouped_mat_mul.yaml")[0]
    case.update(operator=operator, output_all_to_allv=export)
    case.pop("accuracy")
    case["performance"].update(warmup=10, align_iterations=3, iterations=3, repeats=3)
    baseline = default_tiling(case)
    items = []
    for index, rows in enumerate((baseline["local_copy_tile_rows"], 1, 2, 4, 8)):
        tiling = dict(baseline, local_copy_tile_rows=rows)
        items.append(dict(case_id=case["id"], candidate_id=candidate_id(tiling), tiling=tiling, is_default=index == 0))
    request = dict(mode="session", case=case, candidates=items, batch_size=2, run_id="test",
                   plan_digest="plan", environment=probe["environment"])
    result = run(request, "session")
    assert result["status"] == "completed"
    assert result["completed_candidates"] == 5
    assert result["completed_batches"] == 3
    records = []
    for index in range(3):
        payload = json.loads(tiling_worker.batch_path(tmp_path / "session", index).read_text(encoding="utf-8"))
        records.extend(validate_batch_payload(payload, request, index))
    assert [item["candidate_id"] for item in records] == [item["candidate_id"] for item in items]
    assert all(item["latency_us"] == 3.0 for item in records)  # slower rank adds one microsecond
    assert all("environment" not in item and "accuracy" not in item for item in records)
    assert calls.count("input") == 2  # x and weight prepared once for the whole session
    assert calls.count("route") == calls.count("seed") == 1
    assert calls.count("empty") == 2 + 1 + int(export)  # probe/session anchors plus shared outputs
    assert calls.count("new_event") == 2 * 3 * 2  # bounded pool, including partial last batch
    assert calls.count("event_sync") == calls.count("gather_timings") == 3
    assert calls.count("device_sync") == 2  # prepare and warmup only
    assert calls.count("init_udma") == calls.count("destroy_udma") == 1
    assert calls.count("init_dist") == calls.count("destroy_dist") == 2  # probe + persistent session
    assert len(launches) == 10 + 5 * 3 * (3 + 3)
    original_args, original_kwargs = launches[0]
    assert all(args == original_args and kwargs["out"] is original_kwargs["out"] for args, kwargs in launches)
    if export:
        assert all(kwargs["all_to_allv_out"] is original_kwargs["all_to_allv_out"] for _, kwargs in launches)
    progress = json.loads((tmp_path / "session" / "progress.json").read_text(encoding="utf-8"))
    assert progress["completed_batches"] == 3
    assert progress["request_digest"] == tiling_worker.request_digest(request)
