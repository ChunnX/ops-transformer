#!/usr/bin/env python3

import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main():
    plan_dir = Path(sys.argv[1])
    endpoint = sys.argv[2]
    worker_path = Path(sys.argv[3])
    rank_count = int(sys.argv[4])
    processes = []
    stopping = False

    def stop_workers(signum=None, frame=None):
        nonlocal stopping
        if stopping:
            return
        stopping = True
        for process, _ in processes:
            if process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGTERM, stop_workers)
    signal.signal(signal.SIGINT, stop_workers)

    for rank in range(rank_count):
        env = os.environ.copy()
        env["MOE_DEEPEP_CASCADE_PLAN"] = str(plan_dir / f"plan_rank_{rank}.json")
        env["MOE_DEEPEP_CASCADE_RANK"] = str(rank)
        env["MOE_DEEPEP_CASCADE_DEVICE"] = str(rank)
        env["MOE_DEEPEP_CASCADE_ENDPOINT"] = endpoint
        log_path = plan_dir / f"worker_rank_{rank}.log"
        log_file = log_path.open("w", encoding="utf-8", buffering=1)
        processes.append((subprocess.Popen([sys.executable, str(worker_path)], env=env, stdout=log_file, stderr=subprocess.STDOUT), log_file))
    codes = [None] * len(processes)
    deadline = time.monotonic() + int(
        os.environ.get("MOE_DEEPEP_CASCADE_LAUNCH_TIMEOUT_S", 600)
    )
    while any(code is None for code in codes):
        for index, (process, _) in enumerate(processes):
            if codes[index] is None and process.poll() is not None:
                codes[index] = process.returncode
        if any(code not in (None, 0) for code in codes):
            stop_workers()
            deadline = min(deadline, time.monotonic() + 5)
        if time.monotonic() >= deadline:
            stop_workers()
            time.sleep(2)
            for process, _ in processes:
                if process.poll() is None:
                    process.kill()
            break
        time.sleep(0.2)
    for index, (process, _) in enumerate(processes):
        if codes[index] is None:
            codes[index] = process.wait()
    for _, log_file in processes:
        log_file.close()
    if any(code != 0 for code in codes):
        sys.exit(1)


if __name__ == "__main__":
    main()
