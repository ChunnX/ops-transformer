# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Plan, replay and measure candidates in fresh, externally bounded rank groups."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform as platform_module
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .schema import CommConfig, FIELDS, Platform, enumerate_candidates, frozen_geometry

SCHEMA_VERSION = 1


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def case_key(case):
    mode = "export" if case["output_all_to_allv"] else "noexport"
    return f"mte/{case['operator']}/{mode}/{case['id']}/{digest(case)[:12]}"


def make_plan(cases, device, space=None):
    entries = []
    for case in cases:
        if case.get("route", "balanced") != "balanced":
            raise ValueError("mte_comm timing currently requires a balanced route; unbalanced routes remain accuracy/stress cases")
        accepted, rejected = enumerate_candidates(case, device, space)
        if not accepted:
            raise ValueError(f"no legal candidate for {case['id']}")
        # Measure the shipping configuration first when it is in the space.
        baseline = (case["world_size"] - 1, 1, 65536, 2)
        accepted.sort(key=lambda cfg: (cfg.as_tuple() != baseline, cfg.as_tuple()))
        entries.append({"key": case_key(case), "case": case,
                        "frozen_geometry": frozen_geometry(case, device),
                        "candidates": [cfg.to_dict() for cfg in accepted],
                        "pruned": rejected})
    result = {"schema_version": SCHEMA_VERSION, "backend": "mte", "mode": "mte_comm",
              "platform": device.to_dict(), "cases": entries}
    result["plan_digest"] = digest(result)
    return result


def validate_plan(plan):
    if plan.get("schema_version") != SCHEMA_VERSION or plan.get("backend") != "mte" or plan.get("mode") != "mte_comm":
        raise ValueError("only schema 1, backend mte, mode mte_comm is supported")
    if plan.get("plan_digest") != digest({k: v for k, v in plan.items() if k != "plan_digest"}):
        raise ValueError("plan fingerprint mismatch: generate a new plan after editing inputs")
    device = Platform(**plan["platform"])
    if not plan["cases"]:
        raise ValueError("plan has no cases")
    keys = set()
    for entry in plan["cases"]:
        if entry["key"] in keys or entry["key"] != case_key(entry["case"]):
            raise ValueError("duplicate or invalid backend/operator/export/case key")
        keys.add(entry["key"])
        if entry["frozen_geometry"] != frozen_geometry(entry["case"], device):
            raise ValueError("frozen host profile differs from the plan")
        if not entry["candidates"]:
            raise ValueError("case has no candidates")
        for candidate in entry["candidates"]:
            CommConfig.from_dict(candidate).validate(entry["case"], device)
    return device


def environment_fingerprint():
    """Read package/build bytes without importing torch, NPU or the operator DSO."""
    package = Path(__file__).resolve().parents[1]
    sources = {}
    for path in sorted(package.rglob("*")):
        if path.is_file() and (path.suffix in (".py", ".so", ".pyd") or path.name == "build_manifest.json"):
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for part in iter(lambda: stream.read(1 << 20), b""):
                    hasher.update(part)
            sources[str(path.relative_to(package))] = hasher.hexdigest()
    versions = {}
    for name in ("torch", "torch-npu", "ops-transformer-catccos-mte"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    cann_root = os.environ.get("ASCEND_HOME_PATH", os.environ.get("ASCEND_TOOLKIT_HOME", ""))
    cann_versions = {}
    if cann_root:
        for name in ("version.cfg", "compiler/version.info", "aarch64-linux/version.info", "x86_64-linux/version.info"):
            path = Path(cann_root) / name
            if path.is_file():
                cann_versions[name] = path.read_text(encoding="utf-8", errors="replace")
    driver_file = Path("/usr/local/Ascend/driver/version.info")
    driver_version = driver_file.read_text(encoding="utf-8", errors="replace") if driver_file.is_file() else None
    return {"python": sys.version, "platform": platform_module.platform(), "host": platform_module.node(),
            "versions": versions, "package_content": sources, "cann_versions": cann_versions,
            "driver_version": driver_version,
            "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "msprof": shutil.which("msprof"), "cann_root": cann_root}


def _kill_group(process):
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        # The launcher can exit before its child ranks; always kill the group.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=10)


def run_process_group(command, environment, logfile, timeout):
    """Timeout/interrupt terminates the whole msprof/torchrun/rank process tree."""
    if timeout <= 0:
        raise ValueError("external timeout must be positive")
    Path(logfile).parent.mkdir(parents=True, exist_ok=True)
    options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
               else {"start_new_session": True})
    start = time.monotonic()
    with Path(logfile).open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(command) + "\n")
        stream.flush()
        process = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, **options)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(process)
            return {"status": "timeout", "returncode": None, "seconds": time.monotonic() - start}
        except BaseException:
            _kill_group(process)
            raise
    return {"status": "ok" if code == 0 else "process_failure", "returncode": code,
            "seconds": time.monotonic() - start}


def _worker_command(case, stage, payload, output, port):
    return [sys.executable, "-m", "torch.distributed.run", "--max-restarts=0",
            "--master-addr=127.0.0.1", f"--master-port={port}",
            f"--nproc-per-node={case['world_size']}",
            "-m", "ops_transformer_catccos_mte.tuning.worker", "--stage", stage,
            "--payload", str(payload), "--output-dir", str(output)]


def _check_effective(output, case, config):
    for rank in range(case["world_size"]):
        path = output / f"effective_rank_{rank}.json"
        if not path.is_file():
            raise ValueError(f"rank {rank}: no effective configuration record")
        item = read_json(path)
        if item.get("config") != config or item.get("rank") != rank or not item.get("platform_verified"):
            raise ValueError(f"rank {rank}: applied configuration/platform mismatch")


def evaluate(entry, candidate, plan, root, *, timeout, port, task_time="on", stress_repeats=0, factor=1):
    """No process containing SHMEM state is reused between any two phases."""
    case = json.loads(json.dumps(entry["case"]))
    case["performance"]["iterations"] *= factor
    if stress_repeats:
        case["accuracy"]["repeats"] = stress_repeats
    root.mkdir(parents=True, exist_ok=True)
    payload = root / "candidate.json"
    write_json(payload, {"case": case, "config": candidate, "platform": plan["platform"]})
    result = {"case_key": entry["key"], "case_id": case["id"], "config": candidate,
              "candidate_id": digest(candidate)[:16], "status": "pending", "errors": [],
              "artifact_dir": str(root), "phases": {}}
    environment = os.environ.copy()
    environment["MTE_MASTER_ADDR"] = "127.0.0.1"
    environment["MTE_MASTER_PORT"] = str(port + 1)
    for stage in (("accuracy",) if stress_repeats else ("accuracy", "performance")):
        output = root / stage
        output.mkdir(parents=True, exist_ok=True)
        command = _worker_command(case, stage, payload, output, port)
        if stage == "performance":
            msprof = shutil.which("msprof")
            if not msprof:
                result.update(status="profiler_missing", errors=["msprof is not on PATH"])
                return result
            profile = output / "msprof"
            command = [msprof, f"--output={profile}", f"--task-time={task_time}",
                       "--ascendcl=off", "--runtime-api=off", "--aicpu=off", "--ai-core=off"] + command
        try:
            phase = run_process_group(command, environment, output / "process.log", timeout)
        except OSError as error:
            result.update(status=f"{stage}_launch_failure", errors=[str(error)])
            return result
        result["phases"][stage] = phase
        if phase["status"] != "ok":
            result.update(status=f"{stage}_{phase['status']}",
                          errors=[f"{stage} failed: {phase}; see {output / 'process.log'}"])
            return result
        try:
            _check_effective(output, case, candidate)
            if stage == "accuracy":
                # Accuracy main returns failure on any rank, but also require all artifacts.
                for rank in range(case["world_size"]):
                    summary = read_json(output / case["id"] / f"rank_{rank}" / "summary.json")
                    if not summary.get("passed"):
                        raise ValueError(f"rank {rank}: precision/golden validation failed")
            else:
                if not list(profile.rglob("op_summary_*.csv")):
                    exported = run_process_group(
                        [msprof, "--export=on", f"--output={profile}", "--type=text", "--summary-format=csv"],
                        environment, output / "export.log", timeout)
                    if exported["status"] != "ok":
                        raise ValueError(f"msprof export failed: {exported}")
                from ..validation.msprof_parser import parse_msprof_case
                report = parse_msprof_case(profile, case)
                write_json(output / "profile_report.json", report)
                if not report.get("valid") or not report.get("passed"):
                    raise ValueError("; ".join(report.get("errors", ["invalid profiler data"])))
                result["latency_us"] = report["latency_us"]
                result["rank_skew_ratio"] = report["rank_skew_ratio"]
                result["metric_definition"] = report["metric_definition"]
        except (ValueError, OSError, KeyError) as error:
            result.update(status=f"{stage}_validation_failure", errors=[str(error)])
            return result
    result["status"] = "passed"
    return result


def _write_results(root, records):
    write_json(root / "results.json", records)
    names = ["case_key", "candidate_id", "phase", "status", *FIELDS, "p50_us", "p90_us", "errors", "artifact_dir"]
    with (root / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for item in records:
            row = {name: item.get(name, "") for name in names}
            row.update(item["config"])
            row["p50_us"] = item.get("latency_us", {}).get("p50", "")
            row["p90_us"] = item.get("latency_us", {}).get("p90", "")
            row["errors"] = "; ".join(item["errors"])
            writer.writerow(row)


def search(plan, root, *, budget=32, timeout=900, port=29500, task_time="on",
           resume=False, retry_failed=False, topk=3, retest_factor=2, stress_repeats=0):
    validate_plan(plan)
    if budget < 1 or timeout <= 0 or topk < 0 or retest_factor < 1 or stress_repeats < 0 or not 1024 <= port <= 65534:
        raise ValueError("invalid budget/topk/retest/stress/port setting")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    fingerprint = {"plan": plan, "environment": environment_fingerprint(),
                   "measurement": {"timeout": timeout, "task_time": task_time,
                                   "topk": topk, "retest_factor": retest_factor,
                                   "stress_repeats": stress_repeats}}
    fingerprint["digest"] = digest(fingerprint)
    stamp = root / "fingerprint.json"
    if stamp.exists():
        if not resume:
            raise ValueError("output already exists; use --resume or choose an empty output")
        if read_json(stamp) != fingerprint:
            raise ValueError("resume fingerprint mismatch (case/config/platform/build/environment/measurement)")
    elif any(root.iterdir()):
        raise ValueError("output directory is not empty and has no fingerprint")
    else:
        write_json(stamp, fingerprint)
        write_json(root / "plan.json", plan)
    records = read_json(root / "results.json") if (root / "results.json").exists() else []
    used = 0

    def run(entry, candidate, phase, factor=1, stress=0):
        nonlocal used
        ident = digest(candidate)[:16]
        previous = [r for r in records if r["case_key"] == entry["key"] and r["candidate_id"] == ident and r["phase"] == phase]
        if previous and (previous[-1]["status"] == "passed" or not retry_failed):
            return previous[-1]
        if phase == "search" and used >= budget:
            return None
        used += phase == "search"
        # A prior controller can die after a profiler wrote files but before
        # results.json was committed. Reserve a new directory exclusively;
        # orphaned logs remain reviewable and can never contaminate a retry.
        attempt = len(previous)
        while True:
            folder = root / entry["key"] / ident / f"{phase}_{attempt}"
            try:
                folder.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                attempt += 1
            else:
                break
        print(f"[{phase}] {entry['key']} {candidate}", flush=True)
        item = evaluate(entry, candidate, plan, folder, timeout=timeout, port=port,
                        task_time=task_time, factor=factor, stress_repeats=stress)
        item["phase"] = phase
        item["attempt"] = attempt
        records.append(item)
        _write_results(root, records)
        return item

    # Interleave cases so a small search budget covers both model cases/operators.
    width = max(len(entry["candidates"]) for entry in plan["cases"])
    for index in range(width):
        for entry in plan["cases"]:
            if index < len(entry["candidates"]):
                run(entry, entry["candidates"][index], "search")
    winners = []
    for entry in plan["cases"]:
        candidates = {}
        for item in records:
            if item["case_key"] == entry["key"] and item["phase"] == "search":
                candidates[item["candidate_id"]] = item
        ordered = sorted((r for r in candidates.values() if r["status"] == "passed"),
                         key=lambda r: r["latency_us"]["p50"])
        finalists = []
        if topk:
            for item in ordered[:topk]:
                retested = run(entry, item["config"], "retest", factor=retest_factor)
                if retested["status"] == "passed":
                    finalists.append(retested)
        else:
            finalists = ordered
        finalists.sort(key=lambda r: r["latency_us"]["p50"])
        if finalists:
            winner = finalists[0]
            if stress_repeats:
                stress = run(entry, winner["config"], "stress", stress=stress_repeats)
                if stress["status"] != "passed":
                    continue
            winners.append({"key": entry["key"], "case": entry["case"], "config": winner["config"],
                            "latency_us": winner["latency_us"], "retested": bool(topk),
                            "stress_passed": True if stress_repeats else None})
    best = {"schema_version": SCHEMA_VERSION, "backend": "mte", "mode": "mte_comm",
            "platform": plan["platform"], "fingerprint": fingerprint["digest"], "cases": winners,
            "complete": len(winners) == len(plan["cases"]),
            "search_exhaustive": all(any(r["case_key"] == e["key"] and r["candidate_id"] == digest(c)[:16]
                                         and r["phase"] == "search" for r in records)
                                     for e in plan["cases"] for c in e["candidates"])}
    write_json(root / "best.json", best)
    return best


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    planning = sub.add_parser("plan", help="CPU-only legality checking and candidate enumeration")
    planning.add_argument("--cases", action="append", required=True, help="case manifest; repeat for both operators")
    planning.add_argument("--case-id", action="append", default=[])
    planning.add_argument("--space", help="JSON object containing exactly the four communication fields")
    planning.add_argument("--platform", help="JSON Platform declaration; hardware checked during execution")
    planning.add_argument("--output", required=True)
    for command in ("search", "replay"):
        child = sub.add_parser(command)
        child.add_argument("--plan" if command == "search" else "--best", required=True)
        child.add_argument("--output", required=True)
        child.add_argument("--budget", type=int, default=32)
        child.add_argument("--timeout", type=float, default=900)
        child.add_argument("--port", type=int, default=29500)
        child.add_argument("--task-time", choices=("on", "l1"), default="on")
        child.add_argument("--resume", action="store_true")
        child.add_argument("--retry-failed", action="store_true")
        child.add_argument("--topk", type=int, default=3 if command == "search" else 0)
        child.add_argument("--retest-factor", type=int, default=2)
        child.add_argument("--stress-repeats", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "plan":
        from ..validation.case_schema import load_case_manifest, select_cases
        cases = []
        for manifest in args.cases:
            available = load_case_manifest(manifest)
            matched_ids = [case["id"] for case in available if case["id"] in args.case_id]
            if args.case_id and not matched_ids:
                continue
            cases.extend(select_cases(available, mode="performance", case_ids=matched_ids))
        missing = set(args.case_id) - {case["id"] for case in cases}
        if missing:
            raise ValueError(f"unknown/non-performance case IDs: {sorted(missing)}")
        device = Platform(**read_json(args.platform)) if args.platform else Platform()
        plan = make_plan(cases, device, read_json(args.space) if args.space else None)
        validate_plan(plan)
        write_json(args.output, plan)
        print(f"plan: {args.output}; {sum(len(e['candidates']) for e in plan['cases'])} legal candidates")
        return 0
    if args.command == "replay":
        best = read_json(args.best)
        if best.get("backend") != "mte" or best.get("mode") != "mte_comm" or not best.get("cases"):
            raise ValueError("replay requires nonempty mte_comm best.json")
        entries = []
        for winner in best["cases"]:
            entry = make_plan([winner["case"]], Platform(**best["platform"]),
                              {name: [winner["config"][name]] for name in FIELDS})["cases"][0]
            if winner["key"] != entry["key"]:
                raise ValueError("replay case/backend/operator/export mismatch")
            entries.append(entry)
        plan = {"schema_version": SCHEMA_VERSION, "backend": "mte", "mode": "mte_comm",
                "platform": best["platform"], "cases": entries}
        plan["plan_digest"] = digest(plan)
    else:
        plan = read_json(args.plan)
    best = search(plan, args.output, budget=args.budget, timeout=args.timeout, port=args.port,
                  task_time=args.task_time, resume=args.resume, retry_failed=args.retry_failed,
                  topk=args.topk, retest_factor=args.retest_factor, stress_repeats=args.stress_repeats)
    return 0 if best["complete"] else 1
