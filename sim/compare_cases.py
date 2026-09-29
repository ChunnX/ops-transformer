# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Verify that two simulation cases use identical inputs and memory layouts."""

import argparse
from pathlib import Path

from common import read
from validate import validate_case


def compare(left, right):
    left, right = Path(left).resolve(), Path(right).resolve()
    for directory in (left, right):
        validate_case(directory, inputs_only=not (directory / "top.json").exists())
    for name in ("config.json", "layout.json", "memory_map.json", "routing.json"):
        if read(left / name) != read(right / name):
            raise ValueError(f"A/B mismatch: {name}")

    def inputs(directory):
        return {
            name: value
            for name, value in read(directory / "files.json").items()
            if name.endswith(".bin") and Path(name).name not in ("aic.bin", "aiv.bin")
        }

    a, b = inputs(left), inputs(right)
    if a != b:
        changed = sorted(
            name for name in a.keys() | b.keys() if a.get(name) != b.get(name)
        )
        raise ValueError(f"A/B input bytes differ: {changed}")
    return {
        "identical_input_files": len(a),
        "identical_layout": True,
        "device_execution": "not_checked",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("optimized", type=Path)
    args = parser.parse_args()
    print(compare(args.baseline, args.optimized))
