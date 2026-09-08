# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Persistent distributed tiling measurements with shared inputs and batched events."""

from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import json
import math
import os
import socket
import statistics
import sys
import time
from pathlib import Path

try:
    from .tiling_protocol import EXECUTION_MODE, batch_path, candidate_request, request_digest
except ImportError:
    from tiling_protocol import EXECUTION_MODE, batch_path, candidate_request, request_digest


def score_rank_samples(rank_samples):
    """Use the slowest rank in each aligned batch, then take their median."""
    if not rank_samples or not rank_samples[0]:
        raise ValueError("timing samples must be non-empty")
    repeats = len(rank_samples[0])
    if any(len(samples) != repeats for samples in rank_samples):
        raise ValueError("all ranks must report the same number of timing batches")
    if any(
        isinstance(value, bool) or not math.isfinite(value) or value <= 0
        for samples in rank_samples for value in samples
    ):
        raise ValueError("timing samples must be finite and positive")
    worst_rank = [max(samples[index] for samples in rank_samples) for index in range(repeats)]
    return {
        "latency_us": statistics.median(worst_rank),
        "worst_rank_samples_us": worst_rank,
        "rank_samples_us": rank_samples,
        "metric": "median_of_batch_max_rank_npu_event_us_per_call",
    }


def _artifact_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _gather(dist, world_size, payload):
    results = [None] * world_size
    dist.all_gather_object(results, payload)
    return results


def validate_rank_hardware(rank_environments):
    # Target-order GMM derives groups/stage rows from the physical AIC count.
    # Identical tiling integers alone therefore do not make heterogeneous core
    # counts safe for its peer completion protocol.
    if len({item["aic_cores"] for item in rank_environments}) != 1:
        raise ValueError("all ranks must have the same AIC core count for the communication schedule")
    software_keys = (
        "operator_package", "catccos_commit", "shmem_commit", "extension_sha256",
        "python_sources_sha256", "torch", "torch_npu", "cann_version",
    )
    identities = {
        request_digest({key: item.get(key) for key in software_keys})
        for item in rank_environments
    }
    if len(identities) != 1:
        raise ValueError("all ranks must use the same operator build, validation sources and software versions")


def _source_digests(udma):
    result = {}
    for prefix, directory, names in (
        ("operator", Path(udma.__file__).parent, ("ops.py", "runtime.py", "routing.py", "tiling.py")),
        ("validation", Path(__file__).parent, (
            "tiling_worker.py", "tiling_protocol.py", "performance_worker.py", "tiling_space.py", "common.py",
        )),
    ):
        for name in names:
            result[prefix + "/" + name] = _artifact_digest(directory / name)
    return result


def _prepare_case(case, rank, world_size, device, torch, udma):
    """Allocate inputs, routes and outputs once; return a reusable tiling launch."""
    try:
        from .performance_worker import _make_tensor
    except ImportError:
        from performance_worker import _make_tensor

    shape = case["shape"]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(case["seed"] + rank)
    x = _make_tensor((shape["m"], shape["k"]), case["input_distribution"], device, generator)
    weight = _make_tensor(
        (case["expert_num"] // world_size, shape["k"], shape["n"]),
        case["input_distribution"], device, generator,
    )
    local_table, global_table, received_rows = udma.build_rank_route(
        shape["m"], rank, world_size, case["expert_num"], device
    )
    all_to_allv_out = None
    if case["operator"] == "grouped_mat_mul_all_to_allv":
        out = torch.empty((shape["m"], shape["n"]), dtype=torch.bfloat16, device=device)

        def launch(tiling):
            udma.grouped_mat_mul_all_to_allv(
                x, weight, local_table, global_table,
                ep_size=world_size, expert_num=case["expert_num"], out=out, tiling=tiling,
            )
    else:
        out = torch.empty((received_rows, shape["n"]), dtype=torch.bfloat16, device=device)
        if case["output_all_to_allv"]:
            all_to_allv_out = torch.empty(
                (received_rows, shape["k"]), dtype=torch.bfloat16, device=device
            )

        def launch(tiling):
            udma.all_to_allv_grouped_mat_mul(
                x, weight, local_table, global_table,
                ep_size=world_size, expert_num=case["expert_num"], out=out,
                received_rows=received_rows, tiling=tiling,
                output_all_to_allv=case["output_all_to_allv"],
                all_to_allv_out=all_to_allv_out,
            )

    return launch


def _event_pool(size, repeats, torch_npu):
    return [
        [(torch_npu.npu.Event(enable_timing=True), torch_npu.npu.Event(enable_timing=True))
         for _ in range(repeats)]
        for _ in range(size)
    ]


def _measure_batch(launch, items, performance, events, torch_npu, dist):
    """Enqueue a bounded batch, then wait once and read all per-candidate events.

    Every rank uses the same candidate and repeat order. Untimed primer calls
    precede each sample to reduce entry skew; collective kernels preserve the
    rank rendezvous while the host queues subsequent candidates asynchronously.
    """
    dist.barrier()
    started = time.perf_counter()
    for item, pairs in zip(items, events):
        for start, end in pairs:
            for _ in range(performance["align_iterations"]):
                launch(item["tiling"])
            start.record()
            for _ in range(performance["iterations"]):
                launch(item["tiling"])
            end.record()
    enqueued = time.perf_counter()
    # All launches and event records use the current stream. The last end event
    # completes every preceding candidate before the event pool is reused.
    events[len(items) - 1][-1][1].synchronize()
    finished = time.perf_counter()
    samples = [
        [float(start.elapsed_time(end)) * 1000.0 / performance["iterations"]
         for start, end in pairs]
        for pairs in events[:len(items)]
    ]
    return samples, {"enqueue": enqueued - started, "device_wait": finished - enqueued,
                     "measurement": finished - started}


def _measure(case, tiling, rank, world_size, device, torch, torch_npu, dist, udma):
    """Exceptional single-candidate isolation uses the same sampling contract."""
    launch = _prepare_case(case, rank, world_size, device, torch, udma)
    for _ in range(case["performance"]["warmup"]):
        launch(tiling)
    torch_npu.npu.synchronize()
    events = _event_pool(1, case["performance"]["repeats"], torch_npu)
    samples, _ = _measure_batch(launch, [{"tiling": tiling}], case["performance"], events, torch_npu, dist)
    return samples[0]


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run_session(request, output_root, rank, local_rank, world_size, device,
                 torch, torch_npu, dist, udma, progress):
    case = request["case"]
    performance = case["performance"]
    items = request["candidates"]
    size = request["batch_size"]
    started = time.perf_counter()
    launch = _prepare_case(case, rank, world_size, device, torch, udma)
    torch_npu.npu.synchronize()
    prepared = time.perf_counter()
    progress("prepared", 0)
    for _ in range(performance["warmup"]):
        launch(items[0]["tiling"])
    torch_npu.npu.synchronize()
    warmed = time.perf_counter()
    progress("warmed", 0)
    events = _event_pool(min(size, len(items)), performance["repeats"], torch_npu)
    digest = request_digest(request)
    environment_digest = request_digest(request["environment"])
    total_batches = 0
    for offset in range(0, len(items), size):
        batch_index = offset // size
        batch = items[offset:offset + size]
        progress("batch_started", batch_index)
        samples, timings = _measure_batch(launch, batch, performance, events, torch_npu, dist)
        gathered = _gather(dist, world_size, {"rank": rank, "samples": samples, "timings": timings})
        if (
            any(item.get("rank") != index for index, item in enumerate(gathered))
            or any(len(item["samples"]) != len(batch) for item in gathered)
        ):
            raise ValueError("batch measurements do not cover all ranks and candidates")
        records = []
        for index, item in enumerate(batch):
            records.append({
                "case_id": case["id"], "candidate_id": item["candidate_id"],
                "is_default": item["is_default"], "tiling": item["tiling"],
                "status": "measured", "search_mode": "performance_only",
                "execution_mode": EXECUTION_MODE, "accuracy_checked": False,
                "request_digest": request_digest(candidate_request(request, item)),
                **score_rank_samples([item["samples"][index] for item in gathered]),
            })
        payload = {
            "schema_version": 1, "search_mode": "performance_only",
            "execution_mode": EXECUTION_MODE, "accuracy_checked": False,
            "request_digest": digest, "environment_digest": environment_digest,
            "batch_index": batch_index, "results": records,
            "phase_timings_s": [item["timings"] for item in gathered],
        }
        if local_rank == 0:
            _atomic_json(batch_path(output_root, batch_index), payload)
        progress("batch_completed", batch_index + 1)
        total_batches += 1
    # The closure owns every NPU tensor. Release it before SHMEM teardown.
    del launch, events
    return {
        "status": "completed", "completed_candidates": len(items),
        "completed_batches": total_batches,
        "phase_timings_s": {"prepare_inputs": prepared - started,
                            "warmup": warmed - prepared, "session": time.perf_counter() - started},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    output_root = Path(args.output_dir).resolve()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    digest = request_digest(request)
    sequence = 0

    def progress(phase, completed_batches=0):
        nonlocal sequence
        sequence += 1
        if local_rank == 0 and request.get("mode") == "session":
            _atomic_json(output_root / "progress.json", {
                "request_digest": digest, "execution_mode": EXECUTION_MODE,
                "sequence": sequence, "phase": phase, "completed_batches": completed_batches,
            })

    started = time.perf_counter()
    progress("starting")

    # Keep the module's score helpers importable for CPU-only validation.
    import torch
    import torch.distributed as dist
    import torch_npu
    import ops_transformer_catccos_udma as udma
    imported = time.perf_counter()
    progress("imported")

    try:
        from .common import environment_fingerprint, udma_init_method_from_env, validation_udma_heap_size
        from .tiling_space import candidate_id, validate_tiling
    except ImportError:
        from common import environment_fingerprint, udma_init_method_from_env, validation_udma_heap_size
        from tiling_space import candidate_id, validate_tiling

    initialized_dist = False
    initialized_udma = False
    result = None
    try:
        torch_npu.npu.set_device(local_rank)
        if not dist.is_gloo_available():
            raise RuntimeError("tiling search requires Gloo for host-side rank coordination")
        dist.init_process_group(
            "gloo", rank=rank, world_size=world_size,
            timeout=datetime.timedelta(seconds=args.timeout),
        )
        initialized_dist = True
        progress("gloo_initialized")
        local = {"rank": rank, "request_digest": digest, "error": None}
        try:
            if (
                request.get("search_mode") != "performance_only"
                or request.get("execution_mode") != EXECUTION_MODE
                or request.get("mode") not in ("probe", "candidate", "session")
            ):
                raise ValueError("tiling search requires the persistent performance-only execution protocol")
            if request["world_size"] != world_size:
                raise ValueError("request world size does not match torchrun")
            anchor = torch.empty(0, device="npu:{}".format(local_rank))
            hardware = list(torch.ops.ops_transformer_catccos_udma._tiling_hardware(anchor))
            if not hardware or hardware[0] <= 0:
                raise RuntimeError("native hardware query did not return a positive AIC core count")
            local["environment"] = dict(
                environment_fingerprint(), rank=rank, local_rank=local_rank,
                hostname=socket.gethostname(),
                device_name=torch_npu.npu.get_device_name(local_rank),
                aic_cores=int(hardware[0]), hardware=hardware,
                visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                ascend_visible_devices=os.environ.get("ASCEND_VISIBLE_DEVICES"),
                catccos_commit=udma.CATCCOS_COMMIT, shmem_commit=udma.SHMEM_COMMIT,
                extension_sha256=_artifact_digest(udma._C.__file__),
                python_sources_sha256=_source_digests(udma),
            )
            if request["mode"] != "probe":
                case = request["case"]
                for key in ("warmup", "align_iterations", "iterations", "repeats"):
                    value = case["performance"][key]
                    if type(value) is not int or value < (0 if key in ("warmup", "align_iterations") else 1):
                        raise ValueError("invalid search sampling parameter: {}".format(key))
                items = [{"tiling": request["tiling"]}] if request["mode"] == "candidate" else request["candidates"]
                if request["mode"] == "session":
                    if not isinstance(items, list) or not items or type(request.get("batch_size")) is not int or request["batch_size"] < 1:
                        raise ValueError("session must contain candidates and a positive batch size")
                    identities = set()
                    for item in items:
                        if (
                            item.get("case_id") != case["id"]
                            or item.get("candidate_id") != candidate_id(item["tiling"])
                            or type(item.get("is_default")) is not bool
                            or item["candidate_id"] in identities
                        ):
                            raise ValueError("session candidates have invalid or duplicate identities")
                        identities.add(item["candidate_id"])
                for item in items:
                    normalized = validate_tiling(
                        case["operator"], item["tiling"],
                        output_all_to_allv=case["output_all_to_allv"],
                        m=case["shape"]["m"],
                        n=case["shape"]["n"], k=case["shape"]["k"],
                        world_size=world_size, aic_cores=int(hardware[0]),
                        ub_bytes=int(hardware[1]) if len(hardware) > 1 else 256 * 1024,
                    )
                    if normalized != item["tiling"]:
                        raise ValueError("request tiling must already be normalized")
        except Exception as error:
            local["error"] = "{}: {}".format(type(error).__name__, error)
        gathered = _gather(dist, world_size, local)
        errors = [item for item in gathered if item["error"]]
        if len({item["request_digest"] for item in gathered}) != 1:
            errors.append({"error": "ranks received different cases, candidates, or run plans"})
        if not errors:
            try:
                validate_rank_hardware([item["environment"] for item in gathered])
            except ValueError as error:
                errors.append({"error": str(error)})
        if errors:
            result = {"status": "rejected", "failure_phase": "preflight", "errors": errors}
        else:
            preflight_finished = time.perf_counter()
            progress("preflight_completed")
            environment = {
                "topology_label": request.get("topology_label", ""),
                "ranks": [item["environment"] for item in gathered],
            }
            if request["mode"] == "probe":
                result = {"status": "probed", "environment": environment}
            elif environment != request["environment"]:
                result = {
                    "status": "rejected", "failure_phase": "preflight", "environment": environment,
                    "errors": ["hardware or software environment changed after probe"],
                }
            else:
                udma.init_udma_group(
                    rank=rank, world_size=world_size, local_rank=local_rank,
                    init_method=udma_init_method_from_env(), heap_size=validation_udma_heap_size(),
                )
                initialized_udma = True
                initialized = time.perf_counter()
                progress("shmem_initialized")
                case = request["case"]
                if request["mode"] == "session":
                    result = _run_session(
                        request, output_root, rank, local_rank, world_size,
                        "npu:{}".format(local_rank), torch, torch_npu, dist, udma, progress,
                    )
                    result["environment"] = environment
                    profile = dict(result["phase_timings_s"], rank=rank,
                                   imports=imported - started,
                                   preflight=preflight_finished - imported,
                                   shmem_init=initialized - preflight_finished)
                    result["phase_timings_s"] = _gather(dist, world_size, profile)
                else:
                    samples = _measure(
                        case, request["tiling"], rank, world_size,
                        "npu:{}".format(local_rank), torch, torch_npu, dist, udma,
                    )
                    timing = score_rank_samples(_gather(dist, world_size, samples))
                    result = {
                        "status": "measured", "environment": environment,
                        "tiling": request["tiling"], **timing,
                    }
        result["search_mode"] = "performance_only"
        result["execution_mode"] = EXECUTION_MODE
        result["accuracy_checked"] = False
        result["request_digest"] = digest
        # Every node gets the same aggregate through Gloo. No cross-node file
        # sharing is required; a node's local controller reads its local copy.
        if local_rank == 0:
            _atomic_json(output_root / "result.json", result)
        dist.barrier()
    finally:
        gc.collect()
        if initialized_udma:
            udma.destroy_udma_group()
        if initialized_dist:
            dist.destroy_process_group()
    return 0 if result and result["status"] in ("measured", "probed", "completed") else 1


if __name__ == "__main__":
    sys.exit(main())
