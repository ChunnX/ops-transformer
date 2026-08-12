#!/usr/bin/env python3
"""Run one A2 debug case and keep an auditable log/dump/compare directory.

The kernel's printf and DumpTensor output is streamed to ``kernel.log``. Only
workflow markers and failures are echoed to the terminal; the runner and
comparator artifacts remain together under one unique directory, so a later
result can be reproduced without flooding the console with tensor payloads.
"""

import argparse
import datetime as _dt
import json
import os
import platform
import re
import subprocess
import sys
import threading
from pathlib import Path


_DUMP_HEADER = re.compile(
    r"DumpTensor\b.*?\bdesc\s*[=:]\s*(\d+).*?\bdump_size\s*[=:]\s*(\d+)",
    re.IGNORECASE,
)
_NUMBER = re.compile(
    r"(?<![A-Za-z0-9_])(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_VISIBLE_LINE = re.compile(
    r"(?:^\[(?:a2-workflow|a2-debug|a2-compare|a2-p-compare|phase2-state)\])"
    r"|(?:Traceback|RuntimeError|AssertionError|FAILED|ERROR|timeout|killed)",
    re.IGNORECASE,
)


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    default_device = os.environ.get(
        "NPU_DEVICE", f"npu:{os.environ.get('ASCEND_DEVICE_ID', '0')}")
    parser.add_argument("--dump-root", default="/tmp/sasa_a2_runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--compare-stage", choices=("none", "final", "rowmax", "rowsum", "opartial"),
        default="final")
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--p-atol", type=float, default=1e-6)
    parser.add_argument(
        "--compare-p", action="store_true",
        help="also compare desc=940 P dump; disabled by default because P has already been checked")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--q-len", type=int, default=4)
    parser.add_argument("--kv-len", type=int, default=512)
    parser.add_argument("--q-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inner-precise", type=int, default=0)
    parser.add_argument("--device", default=default_device)
    return parser.parse_args()


def _write_process_output(output, log, lock=None):
    if not output:
        return
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    visible = "".join(
        line for line in output.splitlines(keepends=True)
        if _VISIBLE_LINE.search(line)
    )
    if lock is None:
        log.write(output)
        log.flush()
        if visible:
            sys.stdout.write(visible)
            sys.stdout.flush()
        return
    with lock:
        log.write(output)
        log.flush()
        if visible:
            sys.stdout.write(visible)
            sys.stdout.flush()


def _run_tee(command, log_path, timeout):
    print(f"[a2-workflow] command={' '.join(command)}")
    print(f"[a2-workflow] log_file={log_path}")
    with log_path.open("w", encoding="utf-8") as log:
        try:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=(os.name != "nt"),
            )
        except OSError as exc:
            log.write(f"[a2-workflow] failed to start: {exc}\n")
            print(f"[a2-workflow] failed to start: {exc}")
            return 127

        stream_errors = []
        log_lock = threading.Lock()

        def _pump_output():
            try:
                for line in proc.stdout:
                    _write_process_output(line, log, log_lock)
            except Exception as exc:  # preserve a useful failure in the log
                stream_errors.append(exc)

        reader = threading.Thread(target=_pump_output, name="a2-log-reader", daemon=True)
        reader.start()
        returncode = 125
        forced_close = False
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                import signal
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
            with log_lock:
                log.write(f"[a2-workflow] timeout after {timeout}s\n")
                log.flush()
            print(f"[a2-workflow] timeout after {timeout}s; process killed")
            proc.wait()
            returncode = 124
        except Exception as exc:  # keep the run directory auditable
            with log_lock:
                log.write(f"[a2-workflow] stream failure: {exc!r}\n")
                log.flush()
            print(f"[a2-workflow] stream failure: {exc!r}")
            if proc.poll() is None:
                if os.name != "nt":
                    import signal
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
            proc.wait()
            returncode = 125
        finally:
            reader.join(timeout=5.0)
            if reader.is_alive():
                # A descendant that inherited stdout can keep the pipe open.
                # Closing our read end lets the daemon reader exit cleanly.
                forced_close = True
                try:
                    proc.stdout.close()
                except (OSError, ValueError):
                    pass
                reader.join(timeout=1.0)
            if stream_errors and returncode == 0 and not forced_close:
                error = stream_errors[0]
                with log_lock:
                    log.write(f"[a2-workflow] stream failure: {error!r}\n")
                    log.flush()
                print(f"[a2-workflow] stream failure: {error!r}")
                returncode = 125
        return returncode


def _write_fallback_result(path, *, stage, dump_dir, returncode, reason, verdict="ERROR"):
    """Leave a comparison result even when the comparator cannot start."""
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and existing.get("verdict"):
                return
        except (OSError, json.JSONDecodeError):
            pass
    path.write_text(json.dumps({
        "stage": stage,
        "dump_dir": str(dump_dir),
        "verdict": verdict,
        "returncode": returncode,
        "reason": reason,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_result_summary(path):
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"verdict": "ERROR", "reason": f"invalid result JSON: {exc}"}
    return result if isinstance(result, dict) else {
        "verdict": "ERROR", "reason": "result JSON is not an object"
    }


def _extract_dump_tensors(log_path, run_dir):
    text = log_path.read_text(encoding="utf-8", errors="replace")
    records = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        match = _DUMP_HEADER.search(lines[index])
        if not match:
            index += 1
            continue
        desc = int(match.group(1))
        expected = int(match.group(2))
        payload = []
        index += 1
        while index < len(lines):
            if payload and _DUMP_HEADER.search(lines[index]):
                break
            payload.append(lines[index])
            if "]" in lines[index] or (
                expected > 0 and len(_NUMBER.findall("\n".join(payload))) >= expected
            ):
                index += 1
                break
            index += 1
        raw = "\n".join(payload)
        values = _NUMBER.findall(raw)
        out_path = run_dir / f"dump_desc{desc}_{len(records)}.txt"
        out_path.write_text(", ".join(values) + "\n", encoding="utf-8")
        records.append({
            "desc": desc,
            "expected_values": expected,
            "actual_values": len(values),
            "file": str(out_path),
        })
    for record in records:
        print(
            f"[a2-workflow] extracted_dump desc={record['desc']} "
            f"values={record['actual_values']}/{record['expected_values']} "
            f"file={record['file']}"
        )
    return records


def main():
    args = _args()
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name = args.run_name or f"run_{stamp}_{os.getpid()}"
    run_dir = (Path(args.dump_root).expanduser().resolve() / name)
    run_dir.mkdir(parents=True, exist_ok=False)
    kernel_log = run_dir / "kernel.log"
    compare_log = run_dir / f"compare_{args.compare_stage}.log"
    compare_result = run_dir / f"compare_{args.compare_stage}.json"
    phase2_state_log = run_dir / "phase2_state.log"
    metadata_path = run_dir / "run_metadata.json"
    workflow_result = run_dir / "workflow_result.json"

    runner = Path(__file__).resolve().parents[1] / "tests" / "run_sparse_attention_score_prefill_a2_debug.py"
    comparator = Path(__file__).resolve().parents[1] / "tests" / "compare_sparse_attention_score_prefill_a2_debug.py"
    p_comparator = Path(__file__).resolve().with_name("compare_a2_p_dump.py")
    runner_cmd = [
        sys.executable, "-u", str(runner), "--dump-dir", str(run_dir),
        "--q-len", str(args.q_len), "--kv-len", str(args.kv_len),
        "--q-heads", str(args.q_heads), "--kv-heads", str(args.kv_heads),
        "--head-dim", str(args.head_dim), "--block-size", str(args.block_size),
        "--top-k", str(args.top_k), "--seed", str(args.seed),
        "--inner-precise", str(args.inner_precise), "--device", args.device,
    ]

    print(f"[a2-workflow] run_dir={run_dir}")
    print(f"[a2-workflow] dump_dir={run_dir}")
    print(f"[a2-workflow] kernel_log={kernel_log}")
    metadata = {
        "run_dir": str(run_dir),
        "dump_dir": str(run_dir),
        "kernel_log": str(kernel_log),
        "compare_log": str(compare_log),
        "compare_result": str(compare_result),
        "phase2_state_log": str(phase2_state_log),
        "workflow_result": str(workflow_result),
        "runner_command": runner_cmd,
        "compare_stage": args.compare_stage,
        "timeout_seconds": args.timeout,
        "python": sys.executable,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "case": vars(args),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    runner_rc = _run_tee(runner_cmd, kernel_log, args.timeout)
    dump_records = _extract_dump_tensors(kernel_log, run_dir)
    metadata["runner_returncode"] = runner_rc
    metadata["device_dumps"] = dump_records

    phase2_state_rc = 0
    phase2_required_desc = {969, 970, 971, 972, 973, 974, 975}
    emitted_desc = {record["desc"] for record in dump_records}
    if runner_rc == 0 and phase2_required_desc.issubset(emitted_desc):
        phase2_analyzer = Path(__file__).resolve().with_name("analyze_phase2_combine_state.py")
        print(f"[a2-workflow] phase2_state_log={phase2_state_log}")
        phase2_state_rc = _run_tee(
            [sys.executable, "-u", str(phase2_analyzer), str(run_dir),
             "--q-heads", str(args.q_heads), "--kv-heads", str(args.kv_heads),
             "--top-k", str(args.top_k), "--head-dim", str(args.head_dim)],
            phase2_state_log, args.timeout)
    elif runner_rc != 0:
        phase2_state_log.write_text(
            "phase2 state comparison skipped because runner failed or timed out\n",
            encoding="utf-8")
        phase2_state_rc = 125
    else:
        missing_desc = sorted(phase2_required_desc - emitted_desc)
        message = f"phase2 state comparison skipped; missing desc={missing_desc}\n"
        phase2_state_log.write_text(message, encoding="utf-8")
        print(f"[a2-workflow] phase2_state=SKIP missing_desc={missing_desc}")
    metadata["phase2_state_returncode"] = phase2_state_rc

    p_compare_rc = 0
    p_dump = next((record for record in dump_records if record["desc"] == 940), None)
    if args.compare_p and p_dump is not None and runner_rc == 0:
        p_compare_log = run_dir / "compare_p_dump.log"
        p_compare_cmd = [
            sys.executable, "-u", str(p_comparator), "--dump-dir", str(run_dir),
            "--dump-file", p_dump["file"], "--atol", str(args.p_atol),
            "--report-file", str(run_dir / "p_elementwise_compare.txt"),
            "--result-file", str(run_dir / "compare_p_dump.json"),
        ]
        print(f"[a2-workflow] p_compare_log={p_compare_log}")
        p_compare_rc = _run_tee(p_compare_cmd, p_compare_log, args.timeout)
    elif runner_rc == 0:
        reason = "disabled" if not args.compare_p else "desc=940 not emitted"
        print(f"[a2-workflow] p_compare=SKIP reason={reason}")
    metadata["p_compare_returncode"] = p_compare_rc

    compare_rc = 0
    if args.compare_stage != "none" and runner_rc == 0:
        compare_cmd = [
            sys.executable, "-u", str(comparator), "--dump-dir", str(run_dir),
            "--stage", args.compare_stage,
            "--result-file", str(compare_result),
        ]
        if args.atol is not None:
            compare_cmd.extend(["--atol", str(args.atol)])
        print(f"[a2-workflow] compare_stage={args.compare_stage}")
        compare_rc = _run_tee(compare_cmd, compare_log, args.timeout)
    elif args.compare_stage != "none":
        compare_rc = 125
        compare_log.write_text("comparison skipped because runner failed\n", encoding="utf-8")
        _write_fallback_result(
            compare_result, stage=args.compare_stage, dump_dir=run_dir,
            returncode=compare_rc, reason="runner failed or timed out")
    else:
        compare_log.write_text("comparison skipped because --compare-stage=none\n", encoding="utf-8")
        _write_fallback_result(
            compare_result, stage="none", dump_dir=run_dir,
            returncode=0, reason="compare-stage-none", verdict="SKIP")
        print("[a2-workflow] compare=SKIP reason=compare-stage-none")

    if args.compare_stage != "none" and not compare_result.exists():
        _write_fallback_result(
            compare_result, stage=args.compare_stage, dump_dir=run_dir,
            returncode=compare_rc, reason="comparator did not produce a result file")

    compare_summary = _load_result_summary(compare_result)
    metadata["compare_returncode"] = compare_rc
    metadata["compare_summary"] = compare_summary
    metadata["files"] = sorted(str(path) for path in run_dir.iterdir())
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verdict = (
        "PASS" if runner_rc == 0 and compare_rc == 0 and p_compare_rc == 0
        and phase2_state_rc == 0 else "FAIL"
    )
    workflow_result.write_text(json.dumps({
        "run_dir": str(run_dir),
        "dump_dir": str(run_dir),
        "kernel_log": str(kernel_log),
        "compare_log": str(compare_log),
        "compare_result": str(compare_result),
        "phase2_state_log": str(phase2_state_log),
        "metadata": str(metadata_path),
        "runner_returncode": runner_rc,
        "compare_returncode": compare_rc,
        "p_compare_returncode": p_compare_rc,
        "phase2_state_returncode": phase2_state_rc,
        "device_dumps": dump_records,
        "compare_summary": compare_summary,
        "verdict": verdict,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata["files"] = sorted(str(path) for path in run_dir.iterdir())
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[a2-workflow] metadata={metadata_path}")
    print(f"[a2-workflow] workflow_result={workflow_result}")
    print(f"[a2-workflow] dump_dir={run_dir}")
    print(f"[a2-workflow] kernel_log={kernel_log}")
    print(f"[a2-workflow] compare_log={compare_log}")
    print(f"[a2-workflow] compare_result={compare_result}")
    print(f"[a2-workflow] phase2_state_log={phase2_state_log}")
    if compare_summary is None:
        print("[a2-workflow] compare_summary=UNAVAILABLE")
    else:
        print(
            f"[a2-workflow] compare_summary verdict={compare_summary.get('verdict')} "
            f"stage={compare_summary.get('stage', args.compare_stage)} "
            f"mismatches={compare_summary.get('mismatches', 'n/a')} "
            f"max_abs={compare_summary.get('max_abs', 'n/a')} "
            f"mean_abs={compare_summary.get('mean_abs', 'n/a')} "
            f"atol={compare_summary.get('atol', 'n/a')}"
        )
    print(
        f"[a2-workflow] runner_rc={runner_rc} compare_rc={compare_rc} "
        f"p_compare_rc={p_compare_rc} phase2_state_rc={phase2_state_rc}")
    print(f"[a2-workflow] verdict={verdict}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
