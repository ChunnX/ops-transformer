# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Search tilings by measured performance on the target Ascend machines."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

try:
    from .case_schema import case_digest, load_case_manifest, select_cases
    from .common import timestamp, write_json as _write_json
    from .tiling_space import candidate_id, candidates_for_case, default_tiling
    from .tiling_protocol import (
        EXECUTION_MODE, SEARCH_MODE, batch_path, candidate_request, request_digest,
        validate_batch_payload,
    )
    from .tiling_worker import score_rank_samples, validate_rank_hardware
except ImportError:
    from case_schema import case_digest, load_case_manifest, select_cases
    from common import timestamp, write_json as _write_json
    from tiling_space import candidate_id, candidates_for_case, default_tiling
    from tiling_protocol import (
        EXECUTION_MODE, SEARCH_MODE, batch_path, candidate_request, request_digest,
        validate_batch_payload,
    )
    from tiling_worker import score_rank_samples, validate_rank_hardware


HERE = Path(__file__).resolve().parent
METRIC = "median_of_batch_max_rank_npu_event_us_per_call"


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


def _nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("value must be a nonnegative integer")
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
        raise ValueError("session ports exceed [1, 65535]; lower base ports or restart this search with --resume")
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


def _stop_process_tree(process, log):
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
        # Descendants can outlive torchrun after SIGTERM. Kill the group before
        # another session may reuse the NPUs or bootstrap ports.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=10)


def _read_worker_result(output_dir, returncode):
    result_path = Path(output_dir) / "result.json"
    if result_path.is_file():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            return {"status": "launch_failed", "errors": ["invalid worker result: {}".format(error)]}
        if not isinstance(result, dict):
            return {"status": "launch_failed", "errors": ["worker result must be a JSON object"]}
        if returncode and result.get("status") in ("passed", "measured", "probed", "completed"):
            result.update(status="launch_failed", errors=["worker teardown failed: exit {}".format(returncode)])
        result["returncode"] = returncode
        return result
    return {"status": "launch_failed", "returncode": returncode, "errors": ["worker produced no result"]}


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
            _stop_process_tree(process, log)
            if isinstance(error, KeyboardInterrupt):
                raise
            return {"status": "timeout", "errors": ["candidate exceeded {} seconds".format(timeout)]}
    return _read_worker_result(output_dir, returncode)


def run_persistent(command, environment, output_dir, timeout, request, on_batch):
    """Keep ranks alive; enforce timeout between actual progress milestones.

    Atomic batch files are both streaming results and crash checkpoints. The
    parent only reads each batch once, including the last one before an exit.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    options = {"start_new_session": True} if os.name != "nt" else {
        "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP,
    }
    digest = request_digest(request)
    next_batch = 0
    sequence = -1
    progress_stamp = None
    last_progress = time.monotonic()
    progress_path = output_dir / "progress.json"
    with (output_dir / "launch.log").open("w", encoding="utf-8") as log:
        try:
            process = subprocess.Popen(
                command, cwd=str(output_dir), env=environment,
                stdout=log, stderr=subprocess.STDOUT, **options,
            )
        except OSError as error:
            return {"status": "launch_failed", "errors": [str(error)]}
        try:
            while True:
                while batch_path(output_dir, next_batch).is_file():
                    payload = json.loads(batch_path(output_dir, next_batch).read_text(encoding="utf-8"))
                    on_batch(payload)
                    next_batch += 1
                    last_progress = time.monotonic()
                if progress_path.is_file():
                    stamp = progress_path.stat().st_mtime_ns
                    if stamp != progress_stamp:
                        progress = json.loads(progress_path.read_text(encoding="utf-8"))
                        if (
                            not isinstance(progress, dict)
                            or progress.get("request_digest") != digest
                            or progress.get("execution_mode") != EXECUTION_MODE
                            or type(progress.get("sequence")) is not int
                        ):
                            raise ValueError("worker progress does not match the current session")
                        if progress["sequence"] > sequence:
                            sequence = progress["sequence"]
                            last_progress = time.monotonic()
                        progress_stamp = stamp
                returncode = process.poll()
                if returncode is not None:
                    # The worker can commit a batch between our last file check
                    # and poll(). Drain those files before reporting its exit.
                    while batch_path(output_dir, next_batch).is_file():
                        on_batch(json.loads(batch_path(output_dir, next_batch).read_text(encoding="utf-8")))
                        next_batch += 1
                    return _read_worker_result(output_dir, returncode)
                if time.monotonic() - last_progress > timeout:
                    _stop_process_tree(process, log)
                    return {"status": "timeout", "errors": [
                        "session made no progress for {} seconds after {} batches".format(timeout, next_batch),
                    ]}
                time.sleep(0.2)
        except KeyboardInterrupt:
            _stop_process_tree(process, log)
            raise
        except (OSError, ValueError) as error:
            _stop_process_tree(process, log)
            return {"status": "rejected", "errors": [str(error)]}
        except BaseException:
            # Keep an unexpected callback/programming error from orphaning rank
            # processes even when it cannot be treated as a failed candidate.
            _stop_process_tree(process, log)
            raise


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
    if request.get("execution_mode") is not None and (
        request["execution_mode"] != EXECUTION_MODE
        or result.get("execution_mode") != EXECUTION_MODE
    ):
        raise ValueError("worker result does not match the persistent execution protocol")
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
            search_mode=SEARCH_MODE, execution_mode=EXECUTION_MODE, accuracy_checked=False,
        )
        if baseline:
            entry["default_latency_us"] = baseline["latency_us"]
            entry["speedup_vs_default"] = baseline["latency_us"] / best["latency_us"]
        best_cases.append(entry)
    return {
        "schema_version": 1, "search_mode": SEARCH_MODE, "execution_mode": EXECUTION_MODE,
        "accuracy_checked": False,
        "environment": environment, "cases": best_cases,
    }


class IncrementalReport:
    """Append compact results and update winners without scanning the history."""

    FIELDS = [
        "case_id", "candidate_id", "is_default", "status", "accuracy_checked",
        "search_mode", "execution_mode", "latency_us", "log_path",
    ]

    def __init__(self, output_root, cases, environment):
        self.output_root = Path(output_root)
        self.cases = cases
        self.environment = environment
        self.results = []
        self.keys = set()
        self.state = {
            case["id"]: {"tested": 0, "measured": 0, "best": None, "baseline": None}
            for case in cases
        }
        # Batch checkpoints are authoritative. Rebuilding these two derived
        # streams on resume is a single linear pass and repairs a truncated tail.
        self.jsonl = (self.output_root / "results.jsonl").open("w", encoding="utf-8")
        self.csv_file = (self.output_root / "summary.csv").open("w", newline="", encoding="utf-8")
        self.csv = csv.DictWriter(self.csv_file, fieldnames=self.FIELDS, extrasaction="ignore")
        self.csv.writeheader()

    def record(self, result):
        key = (result["case_id"], result["candidate_id"])
        if key in self.keys:
            return False
        compact = {key: value for key, value in result.items() if key != "environment"}
        self.keys.add(key)
        self.results.append(compact)
        self.jsonl.write(json.dumps(compact, separators=(",", ":")) + "\n")
        self.csv.writerow(compact)
        state = self.state[result["case_id"]]
        state["tested"] += 1
        if result["status"] == "measured":
            state["measured"] += 1
            if state["best"] is None or (
                result["latency_us"], result["candidate_id"]
            ) < (state["best"]["latency_us"], state["best"]["candidate_id"]):
                state["best"] = compact
            if result.get("is_default"):
                state["baseline"] = compact
        return True

    def best(self):
        # At most two records per case, independent of completed candidate count.
        finalists = []
        for state in self.state.values():
            if state["best"]:
                finalists.append(state["best"])
            if state["baseline"] and state["baseline"] is not state["best"]:
                finalists.append(state["baseline"])
        best = build_best_results(self.cases, finalists, self.environment)
        for entry in best["cases"]:
            state = self.state[entry["case_id"]]
            entry["tested_candidates"] = state["tested"]
            entry["measured_candidates"] = state["measured"]
        return best

    def flush_best(self):
        self.jsonl.flush()
        self.csv_file.flush()
        best = self.best()
        write_json(self.output_root / "best_tilings.json", best)
        write_json(self.output_root / "search_progress.json", {
            "schema_version": 1, "execution_mode": EXECUTION_MODE,
            "search_mode": SEARCH_MODE, "accuracy_checked": False,
            "completed_candidates": len(self.results),
            "cases": {key: {
                "tested_candidates": value["tested"], "measured_candidates": value["measured"],
                "best_latency_us": value["best"]["latency_us"] if value["best"] else None,
            } for key, value in self.state.items()},
        })
        return best

    def finalize(self):
        try:
            best = self.flush_best()
            write_json(self.output_root / "summary.json", {
                "schema_version": 1, "passed": len(best["cases"]) == len(self.cases),
                "search_mode": SEARCH_MODE, "execution_mode": EXECUTION_MODE,
                "accuracy_checked": False, "metric": METRIC,
                "environment": self.environment, "results": self.results,
            })
            return best
        finally:
            self.jsonl.close()
            self.csv_file.close()


def _session_request(case, candidates, manifest, plan_digest):
    return {
        "schema_version": 1, "mode": "session", "search_mode": SEARCH_MODE,
        "execution_mode": EXECUTION_MODE, "run_id": manifest["run_id"],
        "world_size": case["world_size"], "case": case, "candidates": candidates,
        "plan_digest": plan_digest, "topology_label": manifest["settings"]["topology_label"],
        "batch_size": manifest["settings"]["batch_size"],
        "environment": manifest["environment"]["world_sizes"][str(case["world_size"])],
    }


def _validated_batch(payload, request, index, session_digest, environment_digest, log_path):
    records = validate_batch_payload(
        payload, request, index, session_digest=session_digest, environment_digest=environment_digest,
    )
    verified = []
    size = request["batch_size"]
    for record, item in zip(records, request["candidates"][index * size:(index + 1) * size]):
        if record.get("status") != "measured":
            raise ValueError("a committed batch must contain only complete measurements")
        full_result = dict(record, environment=request["environment"])
        validate_worker_result(full_result, candidate_request(request, item))
        verified.append(dict(record, log_path=str(log_path)))
    return verified


def _validate_session_result(result, request):
    if result.get("status") != "completed":
        return
    if (
        result.get("request_digest") != request_digest(request)
        or result.get("execution_mode") != EXECUTION_MODE
        or result.get("search_mode") != SEARCH_MODE
        or result.get("accuracy_checked") is not False
        or result.get("environment") != request["environment"]
        or type(result.get("completed_candidates")) is not int
        or result["completed_candidates"] != len(request["candidates"])
    ):
        raise ValueError("worker completion does not match its complete session request")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", action="append", help="validation YAML manifest; repeat for both operators")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--include-tag", action="append", default=[])
    parser.add_argument("--space", help="JSON search space (defaults to the supported compiled variants)")
    parser.add_argument("--strategy", choices=("random", "exhaustive", "grid"), default="random")
    parser.add_argument("--max-candidates", type=_nonnegative_int, default=64,
                        help="candidate budget per case; 0 searches every feasible candidate")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--aic-cores", type=_positive_int, help="dry-run core count; on hardware must match the queried physical count")
    parser.add_argument("--output-dir")
    parser.add_argument("--dry-run", action="store_true", help="write candidate plan without importing torch or launching NPUs")
    parser.add_argument("--resume", action="store_true", help="resume an identical single-node plan after re-probing hardware")
    parser.add_argument("--retry-failed", action="store_true", help="retry failed candidates with --resume")
    parser.add_argument("--timeout", type=_positive_int, default=600,
                        help="seconds without session progress, or per isolated probe/recovery attempt")
    parser.add_argument("--batch-size", type=_positive_int, default=64,
                        help="candidates queued before synchronization and durable checkpointing")
    parser.add_argument("--warmup", type=_nonnegative_int, default=10,
                        help="warmup calls once per case session")
    parser.add_argument("--align-iterations", type=_nonnegative_int, default=3,
                        help="untimed alignment calls per measured repeat")
    parser.add_argument("--iterations", type=_positive_int, default=3,
                        help="timed calls per repeat (overrides case performance settings)")
    parser.add_argument("--repeats", type=_positive_int, default=3,
                        help="repeat samples per candidate (overrides case performance settings)")
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


def _replay_checkpoints(output_root, cases, plan, manifest, plan_digest, report, retry_failed):
    """Read each immutable checkpoint once and reject incompatible evidence."""
    by_case = {case["id"]: {} for case in cases}
    for item in plan:
        by_case[item["case_id"]][item["candidate_id"]] = item
    for case in cases:
        allowed = by_case[case["id"]]
        for session_root in sorted((output_root / case["id"] / "sessions").glob("session_*")):
            request_path = session_root / "request.json"
            if not request_path.is_file():
                continue
            session = json.loads(request_path.read_text(encoding="utf-8"))
            members = session.get("candidates")
            if not isinstance(members, list) or not members:
                raise ValueError("cached session has no candidate list: {}".format(request_path))
            seen = set()
            for item in members:
                if (
                    not isinstance(item, dict) or item.get("candidate_id") not in allowed
                    or item != allowed[item["candidate_id"]] or item["candidate_id"] in seen
                ):
                    raise ValueError("cached session contains an invalid candidate: {}".format(request_path))
                seen.add(item["candidate_id"])
            expected = _session_request(case, members, manifest, plan_digest)
            if session != expected:
                raise ValueError("cached session does not match the current plan: {}".format(request_path))
            digest = request_digest(session)
            environment_digest = request_digest(session["environment"])
            batch_files = sorted(session_root.glob("batch_*.json"))
            for index, path in enumerate(batch_files):
                if path != batch_path(session_root, index):
                    raise ValueError("cached session has noncontiguous checkpoints: {}".format(session_root))
                payload = json.loads(path.read_text(encoding="utf-8"))
                records = _validated_batch(
                    payload, session, index, digest, environment_digest, session_root / "launch.log",
                )
                for record in records:
                    report.record(record)
        # Isolated recovery attempts keep their own atomic report. Failed
        # records can be retried explicitly; completed batch results always win.
        base = _session_request(case, [], manifest, plan_digest)
        for item in allowed.values():
            path = output_root / case["id"] / "isolated" / item["candidate_id"] / "report.json"
            if not path.is_file():
                continue
            cached = json.loads(path.read_text(encoding="utf-8"))
            request = candidate_request(base, item)
            if (
                not isinstance(cached, dict)
                or cached.get("request_digest") != request_digest(request)
                or any(cached.get(key) != value for key, value in item.items())
            ):
                raise ValueError("cached candidate does not match the current request: {}".format(path))
            validate_worker_result(dict(cached, environment=base["environment"]), request)
            if cached.get("status") == "measured" or not retry_failed:
                report.record(cached)
    report.flush_best()


def _run_case_sessions(args, output_root, case, candidates, manifest, plan_digest, report, trial_index):
    pending = [item for item in candidates if (item["case_id"], item["candidate_id"]) not in report.keys]
    sessions_root = output_root / case["id"] / "sessions"
    session_number = 0
    while pending:
        while (sessions_root / "session_{:06d}".format(session_number)).exists():
            session_number += 1
        session_root = sessions_root / "session_{:06d}".format(session_number)
        request = _session_request(case, pending, manifest, plan_digest)
        request_path = session_root / "request.json"
        write_json(request_path, request)
        command, process_environment = launch_command(
            args, case["world_size"], request_path, session_root, trial_index,
        )
        trial_index += 1
        digest = request_digest(request)
        environment_digest = request_digest(request["environment"])
        completed_batches = 0
        print("{}: {} remaining candidates; persistent session {} (batch size {})".format(
            case["id"], len(pending), session_number, args.batch_size,
        ), flush=True)

        def on_batch(payload):
            nonlocal completed_batches
            verified = _validated_batch(
                payload, request, completed_batches, digest, environment_digest, session_root / "launch.log",
            )
            for record in verified:
                report.record(record)
            completed_batches += 1
            report.flush_best()
            state = report.state[case["id"]]
            print("[{}/{}] {} batch {} measured; best {:.3f} us".format(
                state["tested"], len(candidates), case["id"], completed_batches,
                state["best"]["latency_us"],
            ), flush=True)

        result = run_persistent(
            command, process_environment, session_root, args.timeout, request, on_batch,
        )
        try:
            # A last complete checkpoint can arrive just before timeout cleanup.
            while batch_path(session_root, completed_batches).is_file():
                on_batch(json.loads(batch_path(session_root, completed_batches).read_text(encoding="utf-8")))
            _validate_session_result(result, request)
            if result.get("status") == "completed" and completed_batches != math.ceil(len(pending) / args.batch_size):
                raise ValueError("worker completed without committing every candidate batch")
        except (OSError, ValueError) as error:
            result = {"status": "rejected", "errors": [str(error)]}
        write_json(session_root / "report.json", result)
        uncompleted = pending[completed_batches * args.batch_size:]
        if result.get("status") == "completed":
            pending = []
            continue
        print("Session {}: {}; committed batches retained. Log: {}".format(
            session_number, result.get("status"), session_root / "launch.log",
        ), flush=True)
        if (
            result.get("status") == "rejected"
            and result.get("failure_phase") == "preflight"
            and result.get("request_digest") == digest
        ):
            raise RuntimeError("session preflight failed; inspect {}".format(session_root / "launch.log"))
        if args.nnodes > 1:
            raise RuntimeError("multi-node session failed; saved checkpoints retained; stop all nodes before retrying")
        if not uncompleted:
            # A teardown failure after all checkpoints must remain visible even
            # though its already measured candidates are safely recoverable.
            raise RuntimeError("session teardown failed after all measurements; inspect {}".format(session_root / "launch.log"))
        # A bad tiling must not repeatedly kill the same large batch. Isolate
        # only the first incomplete batch; normal candidates stay in one session.
        failed_batch = uncompleted[:args.batch_size]
        for item in failed_batch:
            candidate_root = output_root / case["id"] / "isolated" / item["candidate_id"]
            isolated_request = candidate_request(request, item)
            isolated_path = candidate_root / "request.json"
            write_json(isolated_path, isolated_request)
            (candidate_root / "result.json").unlink(missing_ok=True)
            command, process_environment = launch_command(
                args, case["world_size"], isolated_path, candidate_root, trial_index,
            )
            trial_index += 1
            measured = run_isolated(command, process_environment, candidate_root, args.timeout)
            try:
                validate_worker_result(measured, isolated_request)
            except ValueError as error:
                measured = {"status": "rejected", "errors": [str(error)]}
            measured.update(
                item, request_digest=request_digest(isolated_request),
                execution_mode=EXECUTION_MODE, search_mode=SEARCH_MODE, accuracy_checked=False,
                log_path=str(candidate_root / "launch.log"),
            )
            measured.pop("environment", None)
            write_json(candidate_root / "report.json", measured)
            report.record(measured)
            report.flush_best()
            print("Recovery {} {}: {}{}".format(
                case["id"], item["candidate_id"], measured["status"],
                " {:.3f} us".format(measured["latency_us"]) if measured["status"] == "measured" else "",
            ), flush=True)
        pending = uncompleted[args.batch_size:]
        session_number += 1
    return trial_index


def main():
    args = _parse_args()
    manifests = args.cases or [
        str(HERE / "cases" / "alltoallv_grouped_mat_mul.yaml"),
        str(HERE / "cases" / "grouped_mat_mul_all_to_allv.yaml"),
    ]
    all_cases = [case for path in manifests for case in load_case_manifest(path)]
    if len({case["id"] for case in all_cases}) != len(all_cases):
        raise ValueError("case IDs must be unique across manifests")
    cases = copy.deepcopy(select_cases(
        all_cases, mode="performance", case_ids=args.case_id, include_tags=args.include_tag,
    ))
    sampling = {
        "warmup": args.warmup, "align_iterations": args.align_iterations,
        "iterations": args.iterations, "repeats": args.repeats,
    }
    for case in cases:
        case["performance"].update(sampling)
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
        "search_mode": SEARCH_MODE, "execution_mode": EXECUTION_MODE,
        "sampling": sampling, "batch_size": args.batch_size,
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
            "search_mode": SEARCH_MODE, "execution_mode": EXECUTION_MODE,
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
            case, space, strategy=args.strategy, max_candidates=args.max_candidates or None,
            seed=args.seed, aic_cores=cores, ub_bytes=ub_bytes,
        )
        if not candidates:
            raise ValueError("no feasible tiling candidates for {}".format(case["id"]))
        # Normal runs consume ports per case session, not per candidate. Full
        # grids therefore have no candidate-count-derived TCP port limit.
        launch_command(args, case["world_size"], "request.json", "output", trial_index)
        trial_index += 1
        baseline = default_tiling(case)
        for tiling in candidates:
            plan.append({
                "case_id": case["id"], "candidate_id": candidate_id(tiling),
                "tiling": tiling, "is_default": tiling == baseline,
            })
    environment = {"topology_label": args.topology_label, "world_sizes": environments}
    manifest = {
        "schema_version": 1, "run_id": run_id, "cases": cases,
        "search_mode": SEARCH_MODE, "execution_mode": EXECUTION_MODE,
        "settings": settings, "environment": environment, "plan": plan,
    }
    if previous:
        validate_resume(previous, manifest)
    else:
        write_json(manifest_path, manifest)
    if args.dry_run:
        print("Wrote {} candidates across {} persistent case sessions to {}".format(
            len(plan), len(cases), manifest_path,
        ), flush=True)
        return 0
    plan_digest = request_digest(manifest)
    grouped = {case["id"]: [] for case in cases}
    for item in plan:
        grouped[item["case_id"]].append(item)
    report = IncrementalReport(output_root, cases, environment)
    try:
        if args.resume:
            _replay_checkpoints(output_root, cases, plan, manifest, plan_digest, report, args.retry_failed)
        trial_index = len(world_sizes)
        for case in cases:
            trial_index = _run_case_sessions(
                args, output_root, case, grouped[case["id"]], manifest, plan_digest, report, trial_index,
            )
    finally:
        best = report.finalize()
    print("Best measured tilings: {}".format(output_root / "best_tilings.json"), flush=True)
    return 0 if len(best["cases"]) == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
