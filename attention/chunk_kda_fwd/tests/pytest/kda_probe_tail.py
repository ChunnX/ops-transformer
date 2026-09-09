"""在独立进程中复核通用尾块路径，记录超时位置，不将超时当作耗时。"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from kda_rdv_perf import DEFAULT_CASES, lengths_of, read_cases, unsupported_reason


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for case in read_cases(DEFAULT_CASES):
        if (unsupported_reason(case) or (int(case["dk"]), int(case["dv"])) == (128, 128)
                or not any(length % 64 for length in lengths_of(case))):
            continue
        dest = args.output / case["test_name"]
        command = [sys.executable, "-u", str(Path(__file__).with_name("kda_rdv_perf.py")),
                   "--output", str(dest), "--names", case["test_name"]]
        start = time.monotonic()
        with (args.output / (case["test_name"] + ".log")).open("w") as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
            timed_out = False
            try:
                code = proc.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.send_signal(signal.SIGUSR1)
                time.sleep(1)
                proc.terminate()
                try:
                    code = proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    code = proc.wait()
        item = {"test_name": case["test_name"], "status": "TIMEOUT" if timed_out else
                ("PASSED_UNPROFILED" if code == 0 else "FAILED"), "returncode": code,
                "timeout_seconds": args.timeout, "elapsed_wall_seconds": time.monotonic() - start,
                "command": command, "log": str(args.output / (case["test_name"] + ".log"))}
        results.append(item)
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"TAIL_PROBE {case['test_name']} {item['status']}", flush=True)


if __name__ == "__main__":
    main()
