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
        env["MOE_ELASTIC_DISPATCH_PLAN"] = str(plan_dir / f"plan_rank_{rank}.json")
        env["MOE_ELASTIC_DISPATCH_RANK"] = str(rank)
        env["MOE_ELASTIC_DISPATCH_DEVICE"] = str(rank)
        env["MOE_ELASTIC_DISPATCH_ENDPOINT"] = endpoint
        log_path = plan_dir / f"worker_rank_{rank}.log"
        log_file = log_path.open("w", encoding="utf-8", buffering=1)
        processes.append(
            (
                subprocess.Popen(
                    [sys.executable, str(worker_path)],
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                ),
                log_file,
            )
        )
    exit_codes = []
    for rank, (process, log_file) in enumerate(processes):
        process.wait()
        log_file.close()
        exit_codes.append(process.returncode)
        return_code_path = plan_dir / f"worker_rank_{rank}.returncode"
        return_code_path.write_text(f"{process.returncode}\n", encoding="utf-8")
    if any(exit_code != 0 for exit_code in exit_codes):
        sys.exit(1)


if __name__ == "__main__":
    main()
