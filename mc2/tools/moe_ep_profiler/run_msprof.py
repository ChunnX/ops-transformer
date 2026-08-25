from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Sequence

try:
    from .analyze_msprof import analyze
except ImportError:  # direct script execution
    from analyze_msprof import analyze


CHILD_MARKER = "MOE_EP_MSPROF_CHILD"


def run_msprof(
    command: Sequence[str],
    output_root: Path,
    warm_up: int = 0,
    launch_count: int = 1,
    frequency_mhz_override: float | None = None,
) -> int:
    if not command:
        raise ValueError("command must not be empty")
    run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    run_dir = (output_root / run_id).resolve()
    collection_dir = run_dir / "collection"
    analysis_dir = run_dir / "analysis"
    collection_dir.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env[CHILD_MARKER] = "1"
    # Hardware-pipeline collection and cycle instrumentation are separate runs:
    # otherwise GetSystemCycle/local-record instructions perturb msprof counters.
    env["MOE_EP_CYCLE_PROFILER"] = "0"
    msprof_command = ["msprof", "op", f"--output={collection_dir}"]
    if warm_up > 0:
        msprof_command.append(f"--warm-up={warm_up}")
    if launch_count > 1:
        msprof_command.append(f"--launch-count={launch_count}")
    msprof_command.extend(command)
    started = time.time()
    completed = subprocess.run(msprof_command, env=env, check=False)
    manifest = {
        "run_id": run_id,
        "command": list(command),
        "msprof_command": msprof_command,
        "returncode": completed.returncode,
        "started_epoch_s": started,
        "ended_epoch_s": time.time(),
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if completed.returncode == 0:
        analyze(collection_dir, analysis_dir, frequency_mhz_override)
    return completed.returncode


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one command under msprof op and archive MoeEp pipeline data"
    )
    parser.add_argument("--output-root", type=Path, default=Path("moe_ep_msprof_runs"))
    parser.add_argument(
        "--warm-up",
        type=int,
        default=0,
        help="msprof whole-command warmups; internal script warmup is normally sufficient",
    )
    parser.add_argument(
        "--launch-count",
        type=int,
        default=1,
        help="whole-command launches; keep 1 for multi-rank test scripts",
    )
    parser.add_argument(
        "--frequency-mhz",
        type=float,
        help="override frequency in MHz; skips OpBasicInfo/platform_config/fallback",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    return run_msprof(
        command, args.output_root, args.warm_up, args.launch_count, args.frequency_mhz
    )


if __name__ == "__main__":
    raise SystemExit(main())
