# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Continuously run resident UDMA workloads with optional per-call diagnosis."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from datetime import timedelta
from functools import partial
from pathlib import Path

try:
    from .case_schema import CaseManifestError, read_resolved_cases
    from .common import udma_init_method_from_env, validation_udma_heap_size, write_json
    from .input_generation import make_tensor
except ImportError:
    from case_schema import CaseManifestError, read_resolved_cases
    from common import udma_init_method_from_env, validation_udma_heap_size, write_json
    from input_generation import make_tensor


STRESS_MODES = ("a2avgmm", "gmma2av", "chain")


def positive_iterations(value):
    try:
        result = int(value)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("iterations must be a positive integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("iterations must be a positive integer")
    return result


def positive_timeout_seconds(value):
    try:
        result = float(value)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            "timeout must be a finite positive number of seconds"
        ) from error
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError(
            "timeout must be a finite positive number of seconds"
        )
    return result


def validate_stress_case(case, mode):
    if mode not in STRESS_MODES:
        raise CaseManifestError("unsupported stress mode: {}".format(mode))
    if case["route"] != "balanced":
        raise CaseManifestError("case {}: stress requires balanced routing".format(case["id"]))
    expected = (
        "grouped_mat_mul_all_to_allv" if mode == "gmma2av"
        else "all_to_allv_grouped_mat_mul"
    )
    if case["operator"] != expected:
        raise CaseManifestError(
            "case {}: mode {} requires a case for {}".format(case["id"], mode, expected)
        )
    if mode == "chain" and case["expert_num"] // case["world_size"] > 32:
        raise CaseManifestError("chain requires at most 32 local experts for gmma2av")


def validate_stress_environment(environment):
    # Route diagnostics read device tensors back on every public API call.
    if environment.get("CATCCOS_UDMA_VALIDATE_ROUTES", "0").strip().lower() not in {
        "", "0", "false", "no", "off",
    }:
        raise ValueError("stress requires CATCCOS_UDMA_VALIDATE_ROUTES=0 to avoid D2H synchronization")


def prepare_workload(case, mode, rank, device):
    """Return bound public launches; their tensors remain alive for the entire run."""
    import torch
    import ops_transformer_catccos_udma as udma

    validate_stress_case(case, mode)
    shape = case["shape"]
    world_size = case["world_size"]
    expert_num = case["expert_num"]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(case["seed"] + rank)
    local_table, global_table, received_rows = udma.build_rank_route(
        shape["m"], rank, world_size, expert_num, device
    )
    if received_rows != shape["m"]:
        raise ValueError("balanced route must preserve each rank's row count")
    x = make_tensor((shape["m"], shape["k"]), case["input_distribution"], device, generator)
    weight = make_tensor(
        (expert_num // world_size, shape["k"], shape["n"]),
        case["input_distribution"], device, generator,
    )
    out = torch.empty((received_rows, shape["n"]), dtype=torch.bfloat16, device=device)
    tensors = {"x": x, "weight": weight, "out": out,
               "local_route": local_table, "global_route": global_table}
    common = {"ep_size": world_size, "expert_num": expert_num}
    if mode == "gmma2av":
        launches = [(mode, partial(
            udma.grouped_mat_mul_all_to_allv, x, weight, local_table, global_table,
            out=out, **common,
        ))]
    else:
        export_out = None
        if case["output_all_to_allv"]:
            export_out = torch.empty(
                (received_rows, shape["k"]), dtype=torch.bfloat16, device=device,
            )
            tensors["all_to_allv_out"] = export_out
        launches = [("a2avgmm", partial(
            udma.all_to_allv_grouped_mat_mul, x, weight, local_table, global_table,
            received_rows=received_rows, out=out,
            output_all_to_allv=case["output_all_to_allv"], all_to_allv_out=export_out,
            **common,
        ))]
        if mode == "chain":
            return_weight = make_tensor(
                (expert_num // world_size, shape["n"], shape["k"]),
                case["input_distribution"], device, generator,
            )
            return_out = torch.empty(
                (shape["m"], shape["k"]), dtype=torch.bfloat16, device=device,
            )
            tensors.update(return_weight=return_weight, return_out=return_out)
            # Consume A's GMM output directly. Each iteration still starts from x.
            launches.append(("gmma2av", partial(
                udma.grouped_mat_mul_all_to_allv, out, return_weight, local_table, global_table,
                out=return_out, **common,
            )))
    metadata = {
        "received_rows": received_rows,
        "output_all_to_allv": case["output_all_to_allv"],
        "resident_tensors": {
            name: {"shape": list(tensor.shape), "data_ptr": tensor.data_ptr()}
            for name, tensor in tensors.items()
        },
    }
    return launches, metadata


class DebugRankSynchronizer:
    """Identify the operator call that stalls with one progress line per call."""

    def __init__(self, dist, npu, group, timeout_seconds, rank, report):
        self.dist = dist
        self.npu = npu
        self.group = group
        self.timeout = timedelta(seconds=timeout_seconds)
        self.rank = rank
        self.report = report
        self.sequence = 0

    def _set_event(self, name, iteration, stage):
        self.report["last_debug_event"] = {
            "rank": self.rank, "iteration": iteration, "sequence": self.sequence,
            "operator": name, "stage": stage,
        }

    def before_launch(self, name, iteration):
        self.sequence += 1
        self._set_event(name, iteration, "running")
        # The per-call monitored barrier keeps every rank on the same sequence,
        # so one rank is sufficient to identify the call that stopped advancing.
        if self.rank == 0:
            print(
                "[stress-call] rank=0 iteration={} seq={} op={}".format(
                    iteration, self.sequence, name,
                ),
                flush=True,
            )

    def after_launch(self, name, iteration):
        self.npu.synchronize()
        self.dist.monitored_barrier(group=self.group, timeout=self.timeout)
        self.report["synchronized_calls"][name] += 1
        self._set_event(name, iteration, "completed")


def run_iterations(launches, iterations, report, debug_sync=None):
    """Run the fixed launch sequence, optionally diagnosing every call."""
    for iteration_index in range(iterations):
        iteration = iteration_index + 1
        for name, launch in launches:
            if debug_sync is not None:
                debug_sync.before_launch(name, iteration)
            launch()
            report["submitted_calls"][name] += 1
            if debug_sync is not None:
                debug_sync.after_launch(name, iteration)
        report["submitted_iterations"] += 1


def run_worker(
    case, mode, iterations, output_dir, *, sync_each_call=False,
    sync_timeout_seconds=120.0,
):
    validate_stress_case(case, mode)
    validate_stress_environment(os.environ)
    sync_timeout_seconds = positive_timeout_seconds(sync_timeout_seconds)
    import torch
    import torch.distributed as dist
    import torch_npu
    import ops_transformer_catccos_udma as udma

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if case["world_size"] != world_size:
        raise ValueError("resolved case does not match torchrun world size")
    names = ("a2avgmm", "gmma2av") if mode == "chain" else (mode,)
    report = {
        "case_id": case["id"], "rank": rank, "device": local_rank, "mode": mode,
        "iterations": iterations, "requested_calls": {name: iterations for name in names},
        "submitted_iterations": 0, "submitted_calls": {name: 0 for name in names},
        "completed_iterations": None, "completed_calls": None,
        "sync_each_call": sync_each_call,
        "sync_timeout_seconds": sync_timeout_seconds if sync_each_call else None,
        "synchronized_calls": {name: 0 for name in names} if sync_each_call else None,
        "last_debug_event": None,
        "passed": False, "stage": "initializing", "errors": [],
    }
    report_path = Path(output_dir) / "worker_rank_{}.json".format(rank)
    write_json(report_path, report)
    initialized_dist = False
    initialized_udma = False
    debug_group = None
    launches = None
    try:
        torch_npu.npu.set_device(local_rank)
        if not dist.is_gloo_available():
            raise RuntimeError("stress requires PyTorch Gloo support for initial rank coordination")
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        initialized_dist = True
        if sync_each_call:
            # Keep diagnostics independent from the default process group so a
            # monitored timeout reports peers that did not reach this call.
            debug_group = dist.new_group(
                backend="gloo", timeout=timedelta(seconds=sync_timeout_seconds),
            )
        udma.init_udma_group(
            rank=rank, world_size=world_size, local_rank=local_rank,
            init_method=udma_init_method_from_env(), heap_size=validation_udma_heap_size(),
        )
        initialized_udma = True
        # Keep preparation and every launch on this same stream. Changing UDMA
        # streams would synchronize the previous stream in RecordStream().
        stream = torch_npu.npu.current_stream()
        with torch.no_grad(), torch_npu.npu.stream(stream):
            launches, metadata = prepare_workload(case, mode, rank, "npu:{}".format(local_rank))
            report.update(metadata)
            stream.synchronize()
            dist.barrier()
            report["stage"] = "submitting"
            write_json(report_path, report)
            print(
                "[stress] rank={} mode={} iterations={} sync_each_call={} starting".format(
                    rank, mode, iterations, sync_each_call,
                ),
                flush=True,
            )
            started = time.perf_counter()
            debug_sync = None
            if sync_each_call:
                debug_sync = DebugRankSynchronizer(
                    dist, torch_npu.npu, debug_group, sync_timeout_seconds, rank, report,
                )
            run_iterations(launches, iterations, report, debug_sync)
            report["stage"] = "draining"
            report["submission_seconds"] = time.perf_counter() - started
            write_json(report_path, report)
            if not sync_each_call:
                stream.synchronize()
            report["elapsed_seconds"] = time.perf_counter() - started
            report["completed_iterations"] = iterations
            report["completed_calls"] = dict(report["submitted_calls"])
            report["stage"] = "finalizing"
            write_json(report_path, report)
        # All bound inputs/outputs are ordinary NPU tensors, held until the
        # final drain completes. Runtime teardown may synchronize once more.
        launches = None
    except Exception as error:
        report["errors"].append("{}: {}".format(type(error).__name__, error))
        # Persist partial submission counts before teardown, which itself may
        # wait on a failed peer. Completed counts remain unknown until drained.
        write_json(report_path, report)
        raise
    finally:
        write_json(report_path, report)
        try:
            if initialized_udma:
                udma.destroy_udma_group()
        except Exception as error:
            report["errors"].append("UDMA teardown failed: {}".format(error))
            write_json(report_path, report)
            raise
        finally:
            try:
                if initialized_dist:
                    dist.destroy_process_group()
            except Exception as error:
                report["errors"].append("Gloo teardown failed: {}".format(error))
                write_json(report_path, report)
                raise
    report.update(passed=True, stage="completed")
    write_json(report_path, report)
    print("[stress] rank={} completed {} iterations".format(rank, iterations), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolved-case", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=STRESS_MODES, required=True)
    parser.add_argument("--iterations", type=positive_iterations, required=True)
    parser.add_argument(
        "--sync-each-call", action="store_true",
        help="synchronize NPU work and enter a monitored Gloo barrier after every operator call",
    )
    parser.add_argument(
        "--sync-timeout-seconds", type=positive_timeout_seconds, default=120.0,
        help="monitored Gloo barrier timeout used with --sync-each-call (default: 120)",
    )
    args = parser.parse_args()
    cases = read_resolved_cases(args.resolved_case)
    if len(cases) != 1:
        raise ValueError("stress worker requires exactly one resolved case")
    run_worker(
        cases[0], args.mode, args.iterations, args.output_dir,
        sync_each_call=args.sync_each_call,
        sync_timeout_seconds=args.sync_timeout_seconds,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
