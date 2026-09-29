# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""List/decode stable COND tags; check that kernel updates retain paired mark sites."""

import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "mc2/mega_moe/op_kernel/arch35"
HEADER = KERNEL / "common/mega_moe_sim_marks.h"


def catalog(header=HEADER):
    rows = re.findall(
        r"^\s+(\w+) = (0x[0-9A-Fa-f]+),\s*//\s*(.+)$", header.read_text(), re.M
    )
    result = {}
    names = set()
    for name, value, description in rows:
        ident = int(value, 16)
        if not 0 < ident <= 0xFFFF or ident in result or name in names:
            raise ValueError(f"Duplicate or invalid mark ID: {name}={value}")
        names.add(name)
        result[ident] = {"name": name, "description": description}
    if not result:
        raise ValueError("Missing SimMark catalog")
    return result


def check_sites(kernel=KERNEL):
    entries = catalog(kernel / "common/mega_moe_sim_marks.h")
    sites = {entry["name"]: {"BEGIN": [], "END": []} for entry in entries.values()}
    for path in sorted(kernel.rglob("*")):
        if path.suffix not in (".cpp", ".h") or path.name == "mega_moe_sim_marks.h":
            continue
        # Ignore comments, preserving newlines for useful source locations.
        source = re.sub(
            r"/\*.*?\*/|//[^\n]*",
            lambda m: "\n" * m[0].count("\n"),
            path.read_text(),
            flags=re.S,
        )
        for match in re.finditer(r"MEGA_MOE_MARK_(BEGIN|END)\((\w+)\)", source):
            phase, name = match.groups()
            if name not in sites:
                raise ValueError(f"Unknown mark {name} in {path}")
            line = source[: match.start()].count("\n") + 1
            sites[name][phase].append(f"{path.relative_to(kernel)}:{line}")
    for name, phases in sites.items():
        if not phases["BEGIN"] or len(phases["BEGIN"]) != len(phases["END"]):
            raise ValueError(
                f"Missing/unpaired mark {name}: {phases}; review updated kernel paths"
            )
    return {
        str(ident): {**entry, "sites": sites[entry["name"]]}
        for ident, entry in entries.items()
    }


def decode(value, entries):
    phase, ident = value & 0xFFFF0000, value & 0xFFFF
    if phase not in (0x01000000, 0x02000000) or ident not in entries:
        raise ValueError(f"Unknown MegaMoE tag: {value:#x}")
    entry = entries[ident]
    return f"{value:#010x} {'BEGIN' if phase == 0x01000000 else 'END':5} {entry['name']} — {entry['description']}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tags", nargs="*", help="Hex (0x...) or decimal COND values")
    p.add_argument("--list", action="store_true", help="List BEGIN/END values")
    p.add_argument(
        "--check", action="store_true", help="Check source IDs and paired call sites"
    )
    p.add_argument(
        "--build-json",
        type=Path,
        help="Decode with the case's saved build.json catalog",
    )
    a = p.parse_args()
    if a.check:
        sites = check_sites()
        print(
            f"Checked {len(sites)} IDs: all have paired source sites (not a runtime proof)"
        )
    entries = (
        {int(k): v for k, v in json.loads(a.build_json.read_text())["marks"].items()}
        if a.build_json
        else catalog()
    )
    for tag in a.tags:
        print(
            decode(int(tag, 16) if tag.lower().startswith("0x") else int(tag), entries)
        )
    if a.list or (not a.tags and not a.check):
        for ident, entry in entries.items():
            print(
                f"{0x01000000 | ident:#010x}  {0x02000000 | ident:#010x}  "
                f"{entry['name']:24} {entry['description']}"
            )


if __name__ == "__main__":
    main()
