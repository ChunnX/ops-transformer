# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


Location = Tuple[int, int]
Duplicate = Tuple[str, Location, Location]


def find_duplicates(node: yaml.Node) -> List[Duplicate]:
    seen: Dict[str, Location] = {}
    duplicates: List[Duplicate] = []

    def visit(current: yaml.Node) -> None:
        if isinstance(current, yaml.SequenceNode):
            for item in current.value:
                if isinstance(item, yaml.ScalarNode):
                    location = (item.start_mark.line + 1, item.start_mark.column + 1)
                    first_location = seen.setdefault(item.value, location)
                    if first_location != location:
                        duplicates.append((item.value, first_location, location))
                visit(item)
        elif isinstance(current, yaml.MappingNode):
            for key, value in current.value:
                visit(key)
                visit(value)

    visit(node)
    return duplicates


def check_file(path: Path) -> bool:
    try:
        root = yaml.compose(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        print(f"{path}: failed to read or parse YAML: {error}")
        return False

    if root is None:
        return True

    duplicates = find_duplicates(root)
    for value, first_location, duplicate_location in duplicates:
        first_line, first_column = first_location
        line, column = duplicate_location
        print(f"{path}:{line}:{column}: duplicate item: {value}")
        print(f"{path}:{first_line}:{first_column}: first defined here")
    return not duplicates


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check classify rule list items for duplicates."
    )
    parser.add_argument(
        "files", nargs="*", type=Path, default=[Path("classify_rule.yaml")]
    )
    args = parser.parse_args()
    return 0 if all(check_file(path) for path in args.files) else 1


if __name__ == "__main__":
    raise SystemExit(main())
