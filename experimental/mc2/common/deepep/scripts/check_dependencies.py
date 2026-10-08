# Copyright (c) 2026, edconeone
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check installed SDKs and record the build identity without changing either SDK."""

import argparse
import json
from pathlib import Path
import subprocess

from record_shmem_sdk import check


def validate(cann, shmem, catlass=None, timer=False):
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / "dependencies.lock.json").read_text(encoding="utf-8"))
    version = dict(
        line.split("=", 1)
        for line in (Path(cann) / "compiler/version.info")
        .read_text(encoding="utf-8")
        .splitlines()
        if "=" in line
    )
    if version.get("Version") != lock["cann_version"]:
        raise ValueError(f"This source requires CANN {lock['cann_version']}")
    manifest = json.loads(check(shmem).read_text(encoding="utf-8"))
    megamoe = {"enabled": catlass is not None, "timer": timer}
    if catlass is not None:
        revision = subprocess.check_output(
            ["git", "-C", str(catlass), "rev-parse", "HEAD"], text=True
        ).strip()
        if revision != lock["megamoe"]["catlass_revision"]:
            raise ValueError(
                "MegaMoE CATLASS revision differs from dependencies.lock.json"
            )
        dirty = subprocess.check_output(
            [
                "git",
                "-C",
                str(catlass),
                "status",
                "--porcelain",
                "--untracked-files=normal",
            ],
            text=True,
        ).strip()
        if dirty:
            raise ValueError(
                "MegaMoE CATLASS checkout contains local changes; use a clean pinned checkout"
            )
        if not (Path(catlass) / "include/catlass/catlass.hpp").is_file():
            raise ValueError("CATLASS_ROOT must contain include/catlass/catlass.hpp")
        megamoe["catlass"] = revision
    return {
        "source": lock["source"],
        "cann": version,
        "shmem": manifest,
        "megamoe": megamoe,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cann", required=True)
    parser.add_argument("--shmem", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catlass", type=Path)
    parser.add_argument("--timer", choices=("ON", "OFF"), default="OFF")
    args = parser.parse_args()
    info = validate(args.cann, args.shmem, args.catlass, args.timer == "ON")
    args.output.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
