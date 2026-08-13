#!/usr/bin/env python3
"""End-to-end PTA validation for the Ascend950 SHMEM AllToAllV GroupedMatMul V2 API."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import random
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any


DEFAULT_HEAP_BYTES = 1004 * 1024 * 1024
DEFAULT_IP_PORT = "tcp://127.0.0.1:27101"


def build_global_counts(
    case: str,
    world_size: int,
    experts: int,
    tokens_per_route: int,
    seed: int,
) -> list[list[list[int]]]:
    if case not in ("balanced", "uneven"):
        raise ValueError(f"unsupported case: {case}")
    if world_size < 1 or experts < 1 or tokens_per_route < 1:
        raise ValueError("world_size, experts, and tokens_per_route must be positive")

    counts = [
        [[tokens_per_route for _ in range(experts)] for _ in range(world_size)]
        for _ in range(world_size)
    ]
    route_count = world_size * experts
    if case == "uneven" and route_count > 1:
        for source in range(world_size):
            rng = random.Random(seed + source)
            donor = rng.randrange(route_count)
            receiver = rng.randrange(route_count - 1)
            if receiver >= donor:
                receiver += 1
            flat = [value for peer in counts[source] for value in peer]
            flat[receiver] += flat[donor]
            flat[donor] = 0
            counts[source] = [flat[index : index + experts] for index in range(0, route_count, experts)]
    return counts


def rank_send_counts(counts: list[list[list[int]]], rank: int) -> list[int]:
    return [value for destination in counts[rank] for value in destination]


def rank_recv_counts(counts: list[list[list[int]]], rank: int) -> list[int]:
    return [counts[source][rank][expert] for source in range(len(counts)) for expert in range(len(counts[0][0]))]


def build_rank_input(torch_module: Any, rank: int, rows: int, k: int, seed: int) -> Any:
    generator = torch_module.Generator(device="cpu")
    generator.manual_seed(seed + rank * 1009)
    values = torch_module.rand((rows, k), generator=generator, dtype=torch_module.float32) * 2.0 - 1.0
    return values.to(torch_module.bfloat16)


def build_rank_weight(torch_module: Any, rank: int, experts: int, k: int, n: int, seed: int) -> Any:
    generator = torch_module.Generator(device="cpu")
    generator.manual_seed(seed + 100000 + rank * 1013)
    values = torch_module.rand((experts, k, n), generator=generator, dtype=torch_module.float32) * 2.0 - 1.0
    return values.to(torch_module.bfloat16)


def _route_slice(source_input: Any, source_routes: list[list[int]], destination: int, expert: int) -> Any:
    experts = len(source_routes[0])
    flat = [value for peer in source_routes for value in peer]
    route_index = destination * experts + expert
    begin = sum(flat[:route_index])
    end = begin + flat[route_index]
    return source_input[begin:end]


def build_golden(
    torch_module: Any,
    destination: int,
    counts: list[list[list[int]]],
    source_inputs: list[Any],
    local_weight: Any,
    n: int,
) -> Any:
    chunks = []
    for expert in range(local_weight.size(0)):
        for source in range(len(counts)):
            rows = _route_slice(source_inputs[source], counts[source], destination, expert)
            chunks.append(rows.float().matmul(local_weight[expert].float()))
    if not chunks:
        return torch_module.empty((0, n), dtype=torch_module.float32)
    return torch_module.cat(chunks, dim=0)


def precision_metrics(torch_module: Any, actual: Any, golden: Any, k: int) -> dict[str, Any]:
    if tuple(actual.shape) != tuple(golden.shape):
        return {
            "passed": False,
            "error": f"shape mismatch: actual={tuple(actual.shape)}, golden={tuple(golden.shape)}",
        }
    actual_fp32 = actual.float()
    golden_fp32 = golden.float()
    if not bool(torch_module.isfinite(actual_fp32).all().item()):
        return {"passed": False, "error": "actual output contains NaN or Inf"}
    if not bool(torch_module.isfinite(golden_fp32).all().item()):
        return {"passed": False, "error": "golden output contains NaN or Inf"}

    diff = actual_fp32 - golden_fp32
    if diff.numel() == 0:
        return {
            "passed": True,
            "rtol": 2 ** (-7) if k < 2048 else 2 ** (-6),
            "max_abs_error": 0.0,
            "max_relative_error": 0.0,
            "mean_relative_error": 0.0,
            "normalized_mean_bias": 0.0,
        }
    scale = torch_module.maximum(torch_module.ones_like(golden_fp32), golden_fp32.abs())
    rtol = 2 ** (-7) if k < 2048 else 2 ** (-6)
    within = diff.abs() <= rtol * scale
    relative = diff.abs() / torch_module.clamp(golden_fp32.abs(), min=1.0e-12)
    mean_bias = (diff / scale).mean().abs().item()
    passed = bool(within.all().item()) and mean_bias <= 2 ** (-7)
    return {
        "passed": passed,
        "rtol": rtol,
        "max_abs_error": diff.abs().max().item(),
        "max_relative_error": relative.max().item(),
        "mean_relative_error": relative.mean().item(),
        "normalized_mean_bias": mean_bias,
        "mismatched_elements": int((~within).sum().item()),
    }


def _phase(rank: int, case: str, phase: str) -> None:
    print(f"[rank={rank}][case={case}][phase={phase}]", flush=True)


def worker(rank: int, arguments: dict[str, Any], device_ids: list[int], result_queue: Any) -> None:
    args = SimpleNamespace(**arguments)
    initialized = False
    failure = ""
    case_results: list[dict[str, Any]] = []
    precision_failures: list[dict[str, Any]] = []
    try:
        import torch
        import torch_npu

        device_id = device_ids[rank]
        torch_npu.npu.set_device(device_id)
        torch.ops.load_library(args.library)
        _phase(rank, "setup", "init")
        torch.ops.allto_allv_grouped_mat_mul_shmem_pta.init(
            rank, args.world_size, args.heap_bytes, args.ip_port
        )
        initialized = True

        selected_cases = ("balanced", "uneven") if args.case == "all" else (args.case,)
        for case_name in selected_cases:
            case_seed = args.seed + (1000000 if case_name == "uneven" else 0)
            counts = build_global_counts(
                case_name, args.world_size, args.local_experts, args.tokens_per_route, case_seed
            )
            send_counts = rank_send_counts(counts, rank)
            recv_counts = rank_recv_counts(counts, rank)
            local_rows = sum(send_counts)
            _phase(rank, case_name, "build_input")
            x_cpu = build_rank_input(torch, rank, local_rows, args.k, case_seed)
            weight_cpu = build_rank_weight(
                torch, rank, args.local_experts, args.k, args.n, case_seed
            )
            x_npu = x_cpu.to(f"npu:{device_id}")
            weight_npu = weight_cpu.to(f"npu:{device_id}")

            _phase(rank, case_name, "launch")
            y_npu = torch.ops.allto_allv_grouped_mat_mul_shmem_pta.run(
                x_npu, weight_npu, send_counts, recv_counts
            )
            _phase(rank, case_name, "verify")
            source_inputs = [
                build_rank_input(torch, source, sum(rank_send_counts(counts, source)), args.k, case_seed)
                for source in range(args.world_size)
            ]
            golden = build_golden(torch, rank, counts, source_inputs, weight_cpu, args.n)
            actual = y_npu.cpu().float()
            metrics = precision_metrics(torch, actual, golden, args.k)
            metrics.update(
                {
                    "case": case_name,
                    "rank": rank,
                    "device_id": device_id,
                    "output_shape": list(actual.shape),
                    "recv_rows": sum(recv_counts),
                }
            )
            case_results.append(metrics)

            _phase(rank, case_name, "barrier")
            torch.ops.allto_allv_grouped_mat_mul_shmem_pta.barrier()
            if not metrics["passed"]:
                precision_failures.append(metrics)
        if precision_failures:
            raise AssertionError(
                f"precision checks failed after all case barriers: {json.dumps(precision_failures, sort_keys=True)}"
            )
    except BaseException:
        failure = traceback.format_exc()
    finally:
        if initialized:
            try:
                _phase(rank, "setup", "finalize")
                torch.ops.allto_allv_grouped_mat_mul_shmem_pta.finalize()
            except BaseException:
                finalize_failure = traceback.format_exc()
                failure = f"{failure}\n{finalize_failure}" if failure else finalize_failure
        payload = {
            "rank": rank,
            "ok": not failure,
            "cases": case_results,
            "error": failure,
        }
        result_queue.put(payload)
    if failure:
        raise RuntimeError(f"rank {rank} failed\n{failure}")


def _run_self_check(args: argparse.Namespace) -> int:
    selected_cases = ("balanced", "uneven") if args.case == "all" else (args.case,)
    for case_name in selected_cases:
        counts = build_global_counts(
            case_name, args.world_size, args.local_experts, args.tokens_per_route, args.seed
        )
        expected_source_total = args.world_size * args.local_experts * args.tokens_per_route
        for source in range(args.world_size):
            send = rank_send_counts(counts, source)
            if sum(send) != expected_source_total:
                raise AssertionError(f"source {source} total changed in {case_name}: {sum(send)}")
            if case_name == "uneven" and len(send) > 1 and 0 not in send:
                raise AssertionError(f"source {source} has no zero route in uneven case")
        for destination in range(args.world_size):
            recv = rank_recv_counts(counts, destination)
            expected = [
                counts[source][destination][expert]
                for source in range(args.world_size)
                for expert in range(args.local_experts)
            ]
            if recv != expected:
                raise AssertionError(f"recv transpose mismatch on rank {destination}")
        print(f"PASS self-check case={case_name} world_size={args.world_size}")
    return 0


def _parse_device_ids(value: str | None, world_size: int) -> list[int]:
    if value is None:
        return list(range(world_size))
    device_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(device_ids) != world_size:
        raise ValueError(f"--device-ids must contain {world_size} entries, got {device_ids}")
    if len(set(device_ids)) != len(device_ids) or any(device < 0 for device in device_ids):
        raise ValueError(f"--device-ids must be unique nonnegative integers, got {device_ids}")
    return device_ids


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", help="Path to liballto_allv_grouped_mat_mul_shmem_pta.so")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--device-ids", help="Comma-separated physical NPU device IDs")
    parser.add_argument("--case", choices=("balanced", "uneven", "all"), default="all")
    parser.add_argument("--local-experts", type=int, default=2)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--tokens-per-route", type=int, default=2)
    parser.add_argument("--heap-bytes", type=int, default=DEFAULT_HEAP_BYTES)
    parser.add_argument("--ip-port", default=os.environ.get("SHMEM_IP_PORT", DEFAULT_IP_PORT))
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--result-timeout-seconds", type=int, default=540)
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not 1 <= args.world_size <= 8:
        raise ValueError(f"--world-size must be in [1, 8], got {args.world_size}")
    if not 1 <= args.local_experts <= 64:
        raise ValueError(f"--local-experts must be in [1, 64], got {args.local_experts}")
    if min(args.k, args.n, args.tokens_per_route, args.heap_bytes) <= 0:
        raise ValueError("K, N, tokens per route, and heap bytes must be positive")
    if args.self_check:
        return _run_self_check(args)
    if not args.library:
        raise ValueError("--library is required unless --self-check is used")
    library = Path(args.library).expanduser().resolve()
    if not library.is_file():
        raise FileNotFoundError(f"PTA bridge library does not exist: {library}")
    args.library = str(library)
    device_ids = _parse_device_ids(args.device_ids, args.world_size)

    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    arguments = vars(args).copy()
    processes = [
        ctx.Process(target=worker, args=(rank, arguments, device_ids, result_queue), name=f"shmem-pta-rank-{rank}")
        for rank in range(args.world_size)
    ]
    for process in processes:
        process.start()

    results: dict[int, dict[str, Any]] = {}
    deadline = time.monotonic() + args.result_timeout_seconds
    while len(results) < args.world_size and time.monotonic() < deadline:
        try:
            payload = result_queue.get(timeout=min(1.0, max(0.01, deadline - time.monotonic())))
            results[int(payload["rank"])] = payload
        except queue.Empty:
            if all(not process.is_alive() for process in processes):
                break

    for process in processes:
        process.join(timeout=5)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    failed = False
    for rank in range(args.world_size):
        payload = results.get(rank)
        if payload is None:
            print(f"FAIL rank={rank}: no result received", file=sys.stderr)
            failed = True
            continue
        print(json.dumps(payload, sort_keys=True))
        failed = failed or not bool(payload["ok"])
    for process in processes:
        if process.exitcode != 0:
            print(f"FAIL {process.name}: exitcode={process.exitcode}", file=sys.stderr)
            failed = True
    if failed:
        return 1
    print(f"PASS SHMEM PTA world_size={args.world_size} cases={args.case}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
