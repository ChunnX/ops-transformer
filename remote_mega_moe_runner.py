import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict


PROJECT_DIR = Path("/home/developer/ops-transformer")
TEST_FILE = PROJECT_DIR / "test_mega_moe.py"
ASCEND_ENV = "/home/developer/env/ascend-toolkit/set_env.sh"
CONDA_ENV = "/root/miniconda3/etc/profile.d/conda.sh"
CONDA_NAME = "mc2_py310"


def shell(command: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-lc", command],
        check=check,
        text=True,
        capture_output=True,
    )


def paths(world_size: int) -> Dict[str, Path]:
    return {
        "result_dir": Path(f"/tmp/mega_moe_e2e_results_w{world_size}"),
        "log": Path(f"/tmp/mega_moe_e2e_w{world_size}.log"),
        "pid": Path(f"/tmp/mega_moe_e2e_w{world_size}.pid"),
    }


def ensure_world_size(world_size: int) -> None:
    if world_size not in (2, 4):
        raise SystemExit(f"unsupported world size: {world_size}")


def set_world_size(world_size: int) -> None:
    text = TEST_FILE.read_text()
    if "EP_WORLD_SIZE = 4" in text:
        text = text.replace("EP_WORLD_SIZE = 4", f"EP_WORLD_SIZE = {world_size}", 1)
    elif "EP_WORLD_SIZE = 2" in text:
        text = text.replace("EP_WORLD_SIZE = 2", f"EP_WORLD_SIZE = {world_size}", 1)
    else:
        raise SystemExit("cannot find EP_WORLD_SIZE assignment")
    TEST_FILE.write_text(text)


def cleanup(world_size: int) -> None:
    file_paths = paths(world_size)
    shell(
        r"""
        pkill -9 -f 'python.*test_mega_moe.py' || true
        pkill -9 -f 'torchrun.*test_mega_moe.py' || true
        pkill -9 -f 'multiprocessing.forkserver' || true
        pkill -9 -f 'multiprocessing.resource_tracker' || true
        pids=$(npu-smi info | awk -F'|' '/python|torchrun/ {gsub(/ /, "", $3); if ($3 != "") print $3}' | sort -u)
        if [ -n "${pids:-}" ]; then
          echo "$pids" | xargs -r kill -9
        fi
        """,
        check=False,
    )
    shutil.rmtree(file_paths["result_dir"], ignore_errors=True)
    for key in ("log", "pid"):
        try:
            file_paths[key].unlink()
        except FileNotFoundError:
            pass


def start(world_size: int) -> None:
    ensure_world_size(world_size)
    set_world_size(world_size)
    cleanup(world_size)
    file_paths = paths(world_size)
    prewarm = (
        f"export LD_LIBRARY_PATH=\"${{LD_LIBRARY_PATH:-}}\"; "
        f"export PYTHONPATH=\"${{PYTHONPATH:-}}\"; "
        f"export CMAKE_PREFIX_PATH=\"${{CMAKE_PREFIX_PATH:-}}\"; "
        f"source {ASCEND_ENV}; "
        f"source {CONDA_ENV}; "
        f"conda activate {CONDA_NAME}; "
        f"cd {PROJECT_DIR}; "
        "python3 - <<'PY'\n"
        "import torch_npu\n"
        "from npu_ops_transformer.ops.mega_moe import get_symm_buffer_for_mega_moe, mega_moe\n"
        "print('prewarm_ready')\n"
        "PY"
    )
    shell(prewarm)
    command = (
        f"export LD_LIBRARY_PATH=\"${{LD_LIBRARY_PATH:-}}\"; "
        f"export PYTHONPATH=\"${{PYTHONPATH:-}}\"; "
        f"export CMAKE_PREFIX_PATH=\"${{CMAKE_PREFIX_PATH:-}}\"; "
        f"source {ASCEND_ENV}; "
        f"source {CONDA_ENV}; "
        f"conda activate {CONDA_NAME}; "
        f"cd {PROJECT_DIR}; "
        f"nohup python3 -u test_mega_moe.py > {file_paths['log']} 2>&1 < /dev/null & echo $! > {file_paths['pid']}"
    )
    shell(command)
    pid = file_paths["pid"].read_text().strip()
    print(json.dumps({"world_size": world_size, "pid": pid, "log": str(file_paths["log"])}))


def status(world_size: int) -> None:
    ensure_world_size(world_size)
    file_paths = paths(world_size)
    pid = file_paths["pid"].read_text().strip() if file_paths["pid"].exists() else ""
    running = False
    if pid:
        proc = shell(f"ps -p {pid} -o pid=", check=False)
        running = proc.returncode == 0 and proc.stdout.strip() != ""
    log_text = file_paths["log"].read_text(errors="replace") if file_paths["log"].exists() else ""
    files = sorted(p.name for p in file_paths["result_dir"].glob("*")) if file_paths["result_dir"].exists() else []
    print(
        json.dumps(
            {
                "world_size": world_size,
                "pid": pid,
                "running": running,
                "log_exists": file_paths["log"].exists(),
                "log_size": len(log_text.encode("utf-8")),
                "tail": log_text[-4000:],
                "result_dir": str(file_paths["result_dir"]),
                "result_files": files,
            }
        )
    )


def stop(world_size: int) -> None:
    ensure_world_size(world_size)
    file_paths = paths(world_size)
    pid = file_paths["pid"].read_text().strip() if file_paths["pid"].exists() else ""
    if pid:
        shell(f"kill -9 {pid} || true", check=False)
    cleanup(world_size)
    print(json.dumps({"world_size": world_size, "stopped": True}))


def collect(world_size: int) -> None:
    ensure_world_size(world_size)
    file_paths = paths(world_size)
    payload = {"world_size": world_size, "files": {}}
    for file_name in sorted(p.name for p in file_paths["result_dir"].glob("*")):
        payload["files"][file_name] = (file_paths["result_dir"] / file_name).read_text(errors="replace")
    if file_paths["log"].exists():
        payload["log_tail"] = file_paths["log"].read_text(errors="replace")[-12000:]
    print(json.dumps(payload))


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: remote_mega_moe_runner.py <start|status|stop|collect> <2|4>")
    action = sys.argv[1]
    world_size = int(sys.argv[2])
    if action == "start":
        start(world_size)
    elif action == "status":
        status(world_size)
    elif action == "stop":
        stop(world_size)
    elif action == "collect":
        collect(world_size)
    else:
        raise SystemExit(f"unknown action: {action}")


if __name__ == "__main__":
    main()
