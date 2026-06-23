from __future__ import annotations

import subprocess
from pathlib import Path


def prepare_pkg(repo_root: Path, ops: list[str], soc: str, vendor: str, jobs: int = 16, opt_level: str = "3") -> None:
    if not ops:
        raise ValueError("--prepare=pkg requires at least one operator")
    cmd = [
        "bash",
        "build.sh",
        "--pkg",
        f"--soc={soc}",
        f"--ops={','.join(ops)}",
        f"--vendor_name={vendor}",
        f"-j{jobs}",
        f"-O{opt_level}",
    ]
    subprocess.run(cmd, cwd=repo_root, check=True)

    run_files = sorted((repo_root / "build_out").glob(f"cann-ops-transformer-{vendor}_linux-*.run"))
    if not run_files:
        raise FileNotFoundError(f"no package found under {repo_root / 'build_out'} for vendor {vendor}")
    subprocess.run([str(run_files[-1])], cwd=repo_root, check=True)
