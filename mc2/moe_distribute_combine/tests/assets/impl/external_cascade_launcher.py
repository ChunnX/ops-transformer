#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path


def main():
    plan_dir = Path(sys.argv[1])
    endpoint = sys.argv[2]
    worker_path = Path(sys.argv[3])
    rank_count = int(sys.argv[4])
    processes = []
    for rank in range(rank_count):
        env = os.environ.copy()
        env["MC2_EXTERNAL_CASCADE_PLAN"] = str(plan_dir / f"plan_rank_{rank}.json")
        env["MC2_EXTERNAL_CASCADE_RANK"] = str(rank)
        env["MC2_EXTERNAL_CASCADE_ENDPOINT"] = endpoint
        log_file = (plan_dir / f"worker_rank_{rank}.log").open(
            "w", encoding="utf-8", buffering=1
        )
        process = subprocess.Popen(
            [sys.executable, str(worker_path)],
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        processes.append((process, log_file))
    return_codes = []
    pending_log_files = [(rank, log_file) for rank, (_, log_file) in enumerate(processes)]
    for process, log_file in processes:
        process.wait()
        log_file.close()
        return_codes.append(process.returncode)
    if any(return_code != 0 for return_code in return_codes):
        for rank, log_file in pending_log_files:
            log_file.close()
        sys.exit(1)


if __name__ == "__main__":
    main()
