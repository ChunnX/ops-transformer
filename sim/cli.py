# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
from datetime import datetime
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from common import ROOT, read, sha, source_hash, validate_config, write


def run_sim(case, command_override=None):
    from validate import validate_case, check_outputs

    validate_case(case)
    cfg = read(case / "config.json")
    command = (
        shlex.split(command_override)
        if command_override
        else cfg["simulation"]["command"]
    )
    if not isinstance(command, list) or not any("{case}" in arg for arg in command):
        raise ValueError('simulation.command must be an argv array containing "{case}"')
    command = [arg.replace("{case}", str(case)) for arg in command]
    # Never mistake outputs left over from an earlier run for new results.
    for r in range(cfg["model"]["ep"]):
        chip = case / f"chip_{r}"
        for entry in read(chip / "mega_task.json").get("ast_output_array", []):
            (chip / entry["name"]).unlink(missing_ok=True)
    log = case / "simulation.log"
    status = {"argv": command, "started": time.time(), "status": "running"}
    write(case / "run_status.json", status)
    print("+ " + shlex.join(command), flush=True)
    process = None
    try:
        with log.open("w") as out:
            process = subprocess.Popen(
                command,
                cwd=case,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=cfg["simulation"]["timeout_seconds"])
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                import signal

                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        status["returncode"] = code
        if code:
            raise RuntimeError(f"npusim exited {code}; see {log}")
        if cfg["simulation"].get("dump_outputs", False):
            status["checks"] = check_outputs(case)
            status["status"] = "outputs_and_guards_checked"
        else:
            status["status"] = "simulator_exited_zero"
            status["task_completion"] = "inspect_debug_simulator_log"
    except BaseException as e:
        status.update(status="failed", error=str(e))
        raise
    finally:
        status["elapsed_seconds"] = time.time() - status["started"]
        write(case / "run_status.json", status)
    print(f"npusim exited zero; status: {case / 'run_status.json'}, log: {log}")


def main():
    p = argparse.ArgumentParser(
        description="MegaMoE arch35 debug npusim workflow (MIX 1c2v, key 730152)"
    )
    p.add_argument(
        "action",
        choices=(
            "build",
            "inputs",
            "attach",
            "case",
            "prepare",
            "all",
            "layout",
            "run",
            "validate",
            "check-output",
            "relocate",
            "test",
        ),
    )
    p.add_argument("--config", type=Path, default=ROOT / "sim/configs/bs72.json")
    p.add_argument("--build-dir", type=Path, default=ROOT / "sim_build")
    p.add_argument(
        "--case",
        type=Path,
        help="New case directory for generation; defaults to a timestamp. Run defaults to sim_cases/latest.",
    )
    p.add_argument(
        "--dependencies",
        type=Path,
        help="Dedicated offline dependency cache (CMake may checkout pinned revisions)",
    )
    p.add_argument(
        "--host-only",
        action="store_true",
        help="build only: build layout helper without device compilation",
    )
    p.add_argument(
        "--inputs-only",
        action="store_true",
        help="validate a prepared input case before attaching the kernel",
    )
    p.add_argument(
        "--npusim-command",
        help='Optional command override, e.g. "/path/to/npusim -c {case}"',
    )
    a = p.parse_args()
    try:
        if a.action == "test":
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    str(ROOT / "sim/tests"),
                    "-v",
                ],
                cwd=ROOT,
            )
            sys.exit(result.returncode)
        cfg = read(a.config)
        validate_config(cfg)
        build_dir = a.build_dir.resolve()
        if a.host_only and a.action != "build":
            raise ValueError("--host-only is only valid with build")
        if a.action in (
            "build",
            "prepare",
            "all",
            "case",
            "inputs",
            "attach",
            "layout",
        ):
            from validate import validate_source_abi

            validate_source_abi()
        if a.action in ("build", "prepare", "all"):
            from build import build

            build(cfg, build_dir, a.dependencies, a.host_only)
        if a.action == "build":
            return
        if a.action in ("prepare", "all", "case", "inputs", "layout"):
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            case = (
                a.case
                or ROOT / "sim_cases" / f"bs{cfg['model']['tokens_per_rank']}_{stamp}"
            ).resolve()
        else:
            case = (a.case or ROOT / "sim_cases/latest").resolve()
        if a.action == "layout":
            from case import layout

            provenance = read(build_dir / "layout_build.json")
            if provenance["source_sha256"] != source_hash() or provenance[
                "helper_sha256"
            ] != sha(build_dir / "tiling_helper"):
                raise ValueError(
                    "Layout helper is stale; run ./mega-sim build --host-only"
                )
            print(layout(cfg, build_dir, case))
            return
        if a.action in ("prepare", "all", "case", "inputs", "attach"):
            from case import generate, generate_inputs, attach

            if a.action == "inputs":
                generate_inputs(cfg, build_dir, case)
            elif a.action == "attach":
                attach(build_dir, case)
            else:
                generate(cfg, build_dir, case)
            latest = ROOT / "sim_cases/latest"
            latest.parent.mkdir(exist_ok=True)
            if latest.is_symlink():
                latest.unlink()
            if not latest.exists():
                latest.symlink_to(case, target_is_directory=True)
        if a.action == "relocate":
            config = read(case / "config.json")
            top_path = case / ("top.pending.json" if a.inputs_only else "top.json")
            top = read(top_path)
            for rank in range(config["model"]["ep"]):
                top[f"chip{rank}"]["die0"]["case_path"] = (
                    str(case / f"chip_{rank}") + "/"
                )
            write(top_path, top)
        if a.action in ("validate", "relocate"):
            from validate import validate_case

            print(validate_case(case, inputs_only=a.inputs_only))
        if a.action == "check-output":
            from validate import check_outputs

            print(check_outputs(case))
        if a.action in ("all", "run"):
            run_sim(case, a.npusim_command)
    except (OSError, ValueError, RuntimeError, KeyError, StopIteration) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
