# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Search tilings by measured performance on the target Ascend machines."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path

try:
    from .case_schema import case_digest, load_case_manifest, select_cases
    from .common import timestamp, write_csv, write_json as _write_json
    from .tiling_space import candidate_id, candidates_for_case, default_tiling
    from .tiling_worker import request_digest, score_rank_samples, validate_rank_hardware
except ImportError:
    from case_schema import case_digest, load_case_manifest, select_cases
    from common import timestamp, write_csv, write_json as _write_json
    from tiling_space import candidate_id, candidates_for_case, default_tiling
    from tiling_worker import request_digest, score_rank_samples, validate_rank_hardware


HERE = Path(__file__).resolve().parent
METRIC = "median_of_batch_max_rank_npu_event_us_per_call"
SEARCH_MODE = "performance_only"


def write_json(path, payload):
    """Keep interruption from leaving a truncated resume record or best file."""
    destination = Path(path)
    temporary = destination.with_name(destination.name + ".tmp")
    _write_json(temporary, payload)
    temporary.replace(destination)


def _positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def launch_command(args, world_size, request_path, output_dir, trial_index):
    if world_size % args.nnodes:
        raise ValueError("case world_size must be divisible by nnodes")
    local_workers = world_size // args.nnodes
    if args.nproc_per_node is not None and args.nproc_per_node != local_workers:
        raise ValueError("nproc-per-node * nnodes must equal every selected case's world_size")
    if (args.master_port - args.udma_master_port) % 2 == 0:
        raise ValueError(
            "torchrun and SHMEM base ports must have opposite parity; each trial advances "
            "both by 2 so their port sets stay disjoint even when nodes advance at different times"
        )
    master_port = args.master_port + 2 * trial_index
    udma_port = args.udma_master_port + 2 * trial_index
    if not (1 <= master_port <= 65535 and 1 <= udma_port <= 65535):
        raise ValueError("trial ports exceed [1, 65535]; lower base ports or candidate count")
    if master_port == udma_port:
        raise ValueError("torchrun and SHMEM bootstrap ports must differ")
    command = [
        sys.executable, "-m", "torch.distributed.run",
        "--nnodes={}".format(args.nnodes),
        "--node-rank={}".format(args.node_rank),
        "--nproc-per-node={}".format(local_workers),
        "--master-addr={}".format(args.master_addr),
        "--master-port={}".format(master_port),
        "--max-restarts=0",
        str(HERE / "tiling_worker.py"),
        "--request", str(request_path), "--output-dir", str(output_dir),
        "--timeout", str(args.timeout),
    ]
    environment = os.environ.copy()
    environment["UDMA_MASTER_ADDR"] = args.master_addr
    environment["UDMA_MASTER_PORT"] = str(udma_port)
    # Worker processes should resolve the installed native extension instead
    # of accidentally importing the source package with a stale/missing DSO.
    return command, environment


def run_isolated(command, environment, output_dir, timeout):
    """Kill the whole local worker process tree on timeout before continuing."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    options = {"start_new_session": True} if os.name != "nt" else {
        "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP,
    }
    with (output_dir / "launch.log").open("w", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(
                command, cwd=str(output_dir), env=environment,
                stdout=log, stderr=subprocess.STDOUT, **options,
            )
        except OSError as error:
            return {"status": "launch_failed", "errors": [str(error)]}
        try:
            returncode = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=log, stderr=subprocess.STDOUT, timeout=10,
                )
            else:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                # Descendants may outlive the torchrun parent after SIGTERM.
                # Always signal its process group once more before reusing NPUs.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait(timeout=10)
            if isinstance(error, KeyboardInterrupt):
                raise
            return {"status": "timeout", "errors": ["candidate exceeded {} seconds".format(timeout)]}
    result_path = output_dir / "result.json"
    if result_path.is_file():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            return {"status": "launch_failed", "errors": ["invalid worker result: {}".format(error)]}
        if not isinstance(result, dict):
            return {"status": "launch_failed", "errors": ["worker result must be a JSON object"]}
        if returncode and result.get("status") in ("passed", "measured", "probed"):
            result.update(status="launch_failed", errors=["worker teardown failed: exit {}".format(returncode)])
        result["returncode"] = returncode
        return result
    return {"status": "launch_failed", "returncode": returncode, "errors": ["worker produced no result"]}


def _require_rank_coverage(items, world_size, label):
    if (
        not isinstance(items, list) or len(items) != world_size
        or any(not isinstance(item, dict) for item in items)
        or any(type(item.get("rank")) is not int for item in items)
        or {item["rank"] for item in items} != set(range(world_size))
    ):
        raise ValueError("{} does not cover every rank exactly once".format(label))


def validate_worker_result(result, request):
    """Require complete worker evidence before a result can become a winner."""
    if not isinstance(result, dict):
        raise ValueError("worker result must be a JSON object")
    if not isinstance(result.get("status"), str):
        raise ValueError("worker result is missing its status")
    if result.get("status") not in ("passed", "measured", "probed"):
        return
    if result.get("request_digest") != request_digest(request):
        raise ValueError("worker result has a missing or mismatched request digest")
    expected_status = "probed" if request["mode"] == "probe" else "measured"
    if result["status"] != expected_status:
        raise ValueError("worker result does not match the requested execution mode")
    world_size = request["world_size"]
    environment = result.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("worker result is missing its environment")
    ranks = environment.get("ranks")
    _require_rank_coverage(ranks, world_size, "hardware probe")
    for rank in ranks:
        hardware = rank.get("hardware")
        if (
            type(rank.get("aic_cores")) is not int or rank["aic_cores"] <= 0
            or not isinstance(hardware, list) or not hardware
            or hardware[0] != rank["aic_cores"]
        ):
            raise ValueError("hardware probe has invalid AIC resource information")
    validate_rank_hardware(ranks)
    if (
        request.get("search_mode") != SEARCH_MODE
        or result.get("search_mode") != SEARCH_MODE
        or result.get("accuracy_checked") is not False
        or "accuracy" in result
    ):
        raise ValueError("worker result does not match the performance-only search protocol")
    if request["mode"] == "probe":
        return
    if environment != request["environment"] or result.get("tiling") != request["tiling"]:
        raise ValueError("worker measured a different environment or tiling")
    case = request["case"]
    samples = result.get("rank_samples_us")
    if (
        not isinstance(samples, list) or len(samples) != world_size
        or any(not isinstance(batch, list) or len(batch) != case["performance"]["repeats"] for batch in samples)
    ):
        raise ValueError("worker timing samples do not cover all ranks and repeats")
    try:
        timing = score_rank_samples(samples)
    except (TypeError, ValueError) as error:
        raise ValueError("worker timing samples are invalid: {}".format(error)) from error
    latency = result.get("latency_us")
    if (
        isinstance(latency, bool) or not isinstance(latency, (int, float))
        or not math.isfinite(latency) or latency <= 0
        or not math.isclose(latency, timing["latency_us"], rel_tol=1e-12)
        or result.get("worst_rank_samples_us") != timing["worst_rank_samples_us"]
        or result.get("metric") != METRIC
    ):
        raise ValueError("worker score does not match its complete rank timing samples")


def validate_resume(previous, current):
    """Prevent reusing results after a workload, search plan, or machine change."""
    if previous != current:
        changed = sorted(key for key in set(previous) | set(current) if previous.get(key) != current.get(key))
        raise ValueError("resume plan/environment differs in {}; use a new output directory".format(
            ", ".join(changed)
        ))


def build_best_results(cases, results, environment):
    best_cases = []
    for case in cases:
        tested = [item for item in results if item["case_id"] == case["id"]]
        eligible = [item for item in tested if (
            item.get("status") == "measured"
            and item.get("search_mode") == SEARCH_MODE
            and item.get("accuracy_checked") is False
            and "accuracy" not in item
            and not isinstance(item.get("latency_us"), bool)
            and isinstance(item.get("latency_us"), (int, float))
            and math.isfinite(item["latency_us"]) and item["latency_us"] > 0
        )]
        if not eligible:
            continue
        best = min(eligible, key=lambda item: (item["latency_us"], item["candidate_id"]))
        baseline = next((item for item in eligible if item.get("is_default")), None)
        entry = {
            key: case[key] for key in (
                "operator", "output_all_to_allv", "shape", "world_size", "expert_num",
            )
        }
        entry.update(
            case_id=case["id"], tiling=best["tiling"], latency_us=best["latency_us"],
            candidate_id=best["candidate_id"], metric=METRIC,
            case_digest=case_digest(case), tested_candidates=len(tested),
            measured_candidates=len(eligible), passed=True, valid=True,
            search_mode=SEARCH_MODE, accuracy_checked=False,
        )
        if baseline:
            entry["default_latency_us"] = baseline["latency_us"]
            entry["speedup_vs_default"] = baseline["latency_us"] / best["latency_us"]
        best_cases.append(entry)
    return {
        "schema_version": 1, "search_mode": SEARCH_MODE, "accuracy_checked": False,
        "environment": environment, "cases": best_cases,
    }


def _report(output_root, cases, results, environment):
    best = build_best_results(cases, results, environment)
    write_json(output_root / "best_tilings.json", best)
    write_json(output_root / "summary.json", {
        "schema_version": 1,
        "passed": len(best["cases"]) == len(cases),
        "search_mode": SEARCH_MODE, "accuracy_checked": False,
        "metric": METRIC, "environment": environment, "results": results,
    })
    write_csv(output_root / "summary.csv", results, [
        "case_id", "candidate_id", "is_default", "status", "accuracy_checked",
        "search_mode", "latency_us", "log_path",
    ])
    return best


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", action="append", help="validation YAML manifest; repeat for both operators")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--include-tag", action="append", default=[])
    parser.add_argument("--space", help="JSON search space (defaults to the supported compiled variants)")
    parser.add_argument("--strategy", choices=("random", "exhaustive", "grid"), default="random")
    parser.add_argument("--max-candidates", type=_positive_int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--aic-cores", type=_positive_int, help="dry-run core count; on hardware must match the queried physical count")
    parser.add_argument("--output-dir")
    parser.add_argument("--dry-run", action="store_true", help="write candidate plan without importing torch or launching NPUs")
    parser.add_argument("--resume", action="store_true", help="resume an identical single-node plan after re-probing hardware")
    parser.add_argument("--retry-failed", action="store_true", help="retry failed candidates with --resume")
    parser.add_argument("--timeout", type=_positive_int, default=600, help="seconds per full probe/candidate process tree")
    parser.add_argument("--topology-label", default="", help="customer topology/configuration identifier recorded with results")
    parser.add_argument("--nnodes", type=_positive_int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--nproc-per-node", type=_positive_int)
    parser.add_argument("--master-addr", default=os.environ.get("MASTER_ADDR", "127.0.0.1"))
    parser.add_argument("--master-port", type=_positive_int, default=29500)
    parser.add_argument("--udma-master-port", type=_positive_int, default=29601)
    parser.add_argument("--run-id", help="identical unique ID required on every node for multi-node runs")
    args = parser.parse_args()
    if not 0 <= args.node_rank < args.nnodes:
        parser.error("node-rank must be in [0, nnodes)")
    if args.nnodes > 1 and not args.run_id:
        parser.error("multi-node search requires the same --run-id on every node")
    if args.nnodes > 1 and args.resume:
        parser.error("--resume currently supports single-node runs only")
    if args.nnodes > 1 and args.master_addr in ("127.0.0.1", "localhost", "::1"):
        parser.error("multi-node search requires a reachable --master-addr")
    if args.resume and (not args.output_dir or args.dry_run):
        parser.error("--resume requires --output-dir and cannot be combined with --dry-run")
    if args.retry_failed and not args.resume:
        parser.error("--retry-failed requires --resume")
    return args


def main():
    args = _parse_args()
    manifests = args.cases or [
        str(HERE / "cases" / "alltoallv_grouped_mat_mul.yaml"),
        str(HERE / "cases" / "grouped_mat_mul_all_to_allv.yaml"),
    ]
    all_cases = [case for path in manifests for case in load_case_manifest(path)]
    if len({case["id"] for case in all_cases}) != len(all_cases):
        raise ValueError("case IDs must be unique across manifests")
    cases = select_cases(
        all_cases, mode="performance", case_ids=args.case_id, include_tags=args.include_tag,
    )
    space = json.loads(Path(args.space).read_text(encoding="utf-8")) if args.space else None
    output_root = Path(args.output_dir or (
        HERE.parent / "validation_results" / timestamp() / "tiling"
    )).resolve()
    if args.nnodes > 1:
        output_root = output_root / "node_{}".format(args.node_rank)
    if output_root.exists() and any(output_root.iterdir()) and not args.resume:
        raise ValueError("output directory is not empty: {}".format(output_root))
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "search_manifest.json"
    previous = None
    if args.resume:
        if not manifest_path.is_file():
            raise ValueError("resume requires an existing search_manifest.json")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = args.run_id or (previous["run_id"] if previous else uuid.uuid4().hex)
    settings = {
        "search_mode": SEARCH_MODE,
        "space": space, "strategy": args.strategy, "max_candidates": args.max_candidates,
        "seed": args.seed, "aic_cores": args.aic_cores, "nnodes": args.nnodes,
        "nproc_per_node": args.nproc_per_node, "master_addr": args.master_addr,
        "master_port": args.master_port, "udma_master_port": args.udma_master_port,
        "timeout": args.timeout, "topology_label": args.topology_label,
    }
    selection_digest = request_digest({"cases": cases, "settings": settings})
    world_sizes = sorted({case["world_size"] for case in cases})
    environments = {}
    trial_index = 0
    for world_size in world_sizes:
        probe_root = output_root / "probe_world_{}".format(world_size)
        request = {
            "schema_version": 1, "mode": "probe", "world_size": world_size,
            "search_mode": SEARCH_MODE,
            "run_id": run_id, "selection_digest": selection_digest,
            "topology_label": args.topology_label,
        }
        request_path = probe_root / "request.json"
        command, process_environment = launch_command(args, world_size, request_path, probe_root, trial_index)
        if not args.dry_run:
            write_json(request_path, request)
            # A previous probe result cannot stand in for a failed current one.
            (probe_root / "result.json").unlink(missing_ok=True)
            print("Probe {} ranks: {}".format(world_size, probe_root / "launch.log"), flush=True)
            result = run_isolated(command, process_environment, probe_root, args.timeout)
            try:
                validate_worker_result(result, request)
            except ValueError as error:
                result = {"status": "rejected", "errors": [str(error)]}
            write_json(probe_root / "report.json", result)
            if result.get("status") != "probed" or result.get("request_digest") != request_digest(request):
                raise RuntimeError("hardware probe failed; inspect {}".format(probe_root / "launch.log"))
            environments[str(world_size)] = result["environment"]
            if args.aic_cores is not None and any(
                rank_environment["aic_cores"] != args.aic_cores
                for rank_environment in result["environment"]["ranks"]
            ):
                raise ValueError("--aic-cores does not match the probed physical AIC count")
        trial_index += 1

    plan = []
    for case in cases:
        rank_environments = environments.get(str(case["world_size"]), {}).get("ranks", [])
        cores = rank_environments[0]["aic_cores"] if rank_environments else args.aic_cores
        ub_bytes = min(
            item["hardware"][1] if len(item["hardware"]) > 1 else 256 * 1024
            for item in rank_environments
        ) if rank_environments else 256 * 1024
        candidates = candidates_for_case(
            case, space, strategy=args.strategy, max_candidates=args.max_candidates,
            seed=args.seed, aic_cores=cores, ub_bytes=ub_bytes,
        )
        if not candidates:
            raise ValueError("no feasible tiling candidates for {}".format(case["id"]))
        baseline = default_tiling(case)
        for tiling in candidates:
            plan.append({
                "case_id": case["id"], "candidate_id": candidate_id(tiling),
                "tiling": tiling, "is_default": tiling == baseline,
                "trial_index": trial_index,
            })
            # Reject port exhaustion while planning, before executing kernels.
            launch_command(args, case["world_size"], "request.json", "output", trial_index)
            trial_index += 1
    environment = {"topology_label": args.topology_label, "world_sizes": environments}
    manifest = {
        "schema_version": 1, "run_id": run_id, "cases": cases,
        "search_mode": SEARCH_MODE,
        "settings": settings, "environment": environment, "plan": plan,
    }
    if previous:
        validate_resume(previous, manifest)
    else:
        write_json(manifest_path, manifest)
    if args.dry_run:
        print("Wrote {} candidate trials to {}".format(len(plan), manifest_path), flush=True)
        return 0
    plan_digest = request_digest(manifest)
    by_id = {case["id"]: case for case in cases}
    results = []
    for index, item in enumerate(plan):
        case = by_id[item["case_id"]]
        candidate_root = output_root / item["case_id"] / item["candidate_id"]
        report_path = candidate_root / "report.json"
        request = {
            "schema_version": 1, "mode": "candidate", "run_id": run_id,
            "search_mode": SEARCH_MODE,
            "world_size": case["world_size"], "case": case, "tiling": item["tiling"],
            "plan_digest": plan_digest, "topology_label": args.topology_label,
            "environment": environments[str(case["world_size"])],
        }
        if args.resume and report_path.is_file():
            cached = json.loads(report_path.read_text(encoding="utf-8"))
            if cached.get("request_digest") != request_digest(request):
                raise ValueError("cached candidate does not match current request: {}".format(report_path))
            validate_worker_result(cached, request)
            if cached.get("status") == "measured" or not args.retry_failed:
                results.append(cached)
                _report(output_root, cases, results, environment)
                continue
        request_path = candidate_root / "request.json"
        write_json(request_path, request)
        (candidate_root / "result.json").unlink(missing_ok=True)
        command, process_environment = launch_command(
            args, case["world_size"], request_path, candidate_root, item["trial_index"],
        )
        print("[{}/{}] {} {}{}".format(
            index + 1, len(plan), item["case_id"], item["candidate_id"],
            " (default)" if item["is_default"] else "",
        ), flush=True)
        result = run_isolated(command, process_environment, candidate_root, args.timeout)
        try:
            validate_worker_result(result, request)
        except ValueError as error:
            result = {"status": "rejected", "errors": [str(error)]}
        result.update(item, request_digest=request_digest(request), log_path=str(candidate_root / "launch.log"))
        write_json(report_path, result)
        results.append(result)
        best = _report(output_root, cases, results, environment)
        print("  {}{}".format(
            result["status"], " {:.3f} us".format(result["latency_us"]) if result.get("status") == "measured" else "",
        ), flush=True)
    best = _report(output_root, cases, results, environment)
    print("Best measured tilings: {}".format(output_root / "best_tilings.json"), flush=True)
    return 0 if len(best["cases"]) == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
