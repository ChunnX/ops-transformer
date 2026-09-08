# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Isolated distributed NPU-event measurement for one tiling, without accuracy checks."""

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
from pathlib import Path


def request_digest(request):
    return hashlib.sha256(
        json.dumps(request, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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
            "tiling_worker.py", "performance_worker.py", "tiling_space.py",
        )),
    ):
        for name in names:
            result[prefix + "/" + name] = _artifact_digest(directory / name)
    return result


def _measure(case, tiling, rank, world_size, device, torch, torch_npu, dist, udma):
    try:
        from .performance_worker import _make_tensor
    except ImportError:
        from performance_worker import _make_tensor

    shape = case["shape"]
    performance = case["performance"]
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

        def launch():
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

        def launch():
            udma.all_to_allv_grouped_mat_mul(
                x, weight, local_table, global_table,
                ep_size=world_size, expert_num=case["expert_num"], out=out,
                received_rows=received_rows, tiling=tiling,
                output_all_to_allv=case["output_all_to_allv"],
                all_to_allv_out=all_to_allv_out,
            )

    for _ in range(performance["warmup"]):
        launch()
    torch_npu.npu.synchronize()
    samples = []
    for _ in range(performance["repeats"]):
        # Host rendezvous is outside the event interval. Primer calls keep the
        # ranks aligned without putting a Gloo collective between timed kernels.
        dist.barrier()
        for _ in range(performance["align_iterations"]):
            launch()
        start = torch_npu.npu.Event(enable_timing=True)
        end = torch_npu.npu.Event(enable_timing=True)
        start.record()
        for _ in range(performance["iterations"]):
            launch()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0 / performance["iterations"])
    torch_npu.npu.synchronize()
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    output_root = Path(args.output_dir).resolve()

    # Keep the module's score helpers importable for CPU-only validation.
    import torch
    import torch.distributed as dist
    import torch_npu
    import ops_transformer_catccos_udma as udma

    try:
        from .common import environment_fingerprint, udma_init_method_from_env, validation_udma_heap_size, write_json
        from .tiling_space import validate_tiling
    except ImportError:
        from common import environment_fingerprint, udma_init_method_from_env, validation_udma_heap_size, write_json
        from tiling_space import validate_tiling

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
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
        local = {"rank": rank, "request_digest": request_digest(request), "error": None}
        try:
            if request.get("search_mode") != "performance_only" or request.get("mode") not in ("probe", "candidate"):
                raise ValueError("tiling search requires a performance_only probe or candidate request")
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
                normalized = validate_tiling(
                    case["operator"], request["tiling"],
                    output_all_to_allv=case["output_all_to_allv"],
                    m=case["shape"]["m"],
                    n=case["shape"]["n"], k=case["shape"]["k"],
                    world_size=world_size, aic_cores=int(hardware[0]),
                    ub_bytes=int(hardware[1]) if len(hardware) > 1 else 256 * 1024,
                )
                if normalized != request["tiling"]:
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
            result = {"status": "rejected", "errors": errors}
        else:
            environment = {
                "topology_label": request.get("topology_label", ""),
                "ranks": [item["environment"] for item in gathered],
            }
            if request["mode"] == "probe":
                result = {"status": "probed", "environment": environment}
            elif environment != request["environment"]:
                result = {
                    "status": "rejected", "environment": environment,
                    "errors": ["hardware or software environment changed after probe"],
                }
            else:
                udma.init_udma_group(
                    rank=rank, world_size=world_size, local_rank=local_rank,
                    init_method=udma_init_method_from_env(), heap_size=validation_udma_heap_size(),
                )
                initialized_udma = True
                case = request["case"]
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
        result["accuracy_checked"] = False
        result["request_digest"] = request_digest(request)
        # Every node gets the same aggregate through Gloo. No cross-node file
        # sharing is required; a node's local controller reads its local copy.
        if local_rank == 0:
            write_json(output_root / "result.json", result)
        dist.barrier()
    finally:
        gc.collect()
        if initialized_udma:
            udma.destroy_udma_group()
        if initialized_dist:
            dist.destroy_process_group()
    return 0 if result and result["status"] in ("measured", "probed") else 1


if __name__ == "__main__":
    sys.exit(main())
