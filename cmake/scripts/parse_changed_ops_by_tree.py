#!/usr/bin/env python3
# coding: utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import argparse
import glob
import os
from pathlib import Path

OP_DOMAINS = (
    "attention",
    "ffn",
    "gmm",
    "mamba",
    "mc2",
    "mhc",
    "moe",
    "posembedding",
)
SKIP_EXTS = {".md", ".json", ".ini"}
REPO_ROOT = Path(__file__).resolve().parents[2]


def read_changed_files(changed_file):
    with open(changed_file, "r", encoding="utf-8") as fh:
        for raw in fh:
            path = raw.strip().replace(os.sep, "/")
            if not path or path.startswith("/"):
                continue
            if Path(path).suffix.lower() in SKIP_EXTS or "docs/" in path:
                continue
            yield path


def split_domain_op(path):
    parts = path.split("/")
    if len(parts) < 2 or parts[0] not in OP_DOMAINS:
        return None, None
    return parts[0], parts[1]


def op_dir(domain, op_name):
    return REPO_ROOT / domain / op_name


def domain_op_dirs(domain):
    base = REPO_ROOT / domain
    if not base.is_dir():
        return []
    return [
        child
        for child in base.iterdir()
        if child.is_dir() and child.name not in ("common", "3rd")
    ]


def has_pkg_sources(path):
    return any((path / name).is_dir() for name in ("op_host", "op_kernel", "op_api"))


def has_examples(path):
    example_dir = path / "examples"
    if not example_dir.is_dir():
        return False
    return bool(
        glob.glob(str(example_dir / "test_aclnn_*.cpp"))
        or glob.glob(str(example_dir / "test_geir_*.cpp"))
    )


def has_pr_ut(path):
    ut_dir = path / "tests" / "ut"
    return any((ut_dir / name).is_dir() for name in ("op_host", "op_api"))


def expand_domain(domain, mode):
    if mode == "kernel":
        return sorted(
            child.name for child in domain_op_dirs(domain) if has_pkg_sources(child)
        )
    if mode.startswith("ut"):
        return sorted(
            child.name for child in domain_op_dirs(domain) if has_pr_ut(child)
        )
    return []


def match_mode(path, mode):
    if mode == "kernel":
        return has_pkg_sources(path)
    if mode == "examples":
        return has_examples(path)
    if mode.startswith("ut"):
        return has_pr_ut(path)
    return False


def collect_ops(changed_file, mode):
    ops = set()
    for path in read_changed_files(changed_file):
        domain, op_name = split_domain_op(path)
        if domain is None:
            continue
        if mode == "ut_mc2" and domain != "mc2":
            continue
        if mode == "ut_exclude_mc2" and domain == "mc2":
            continue
        if op_name in ("common", "3rd"):
            ops.update(expand_domain(domain, mode))
            continue
        path_dir = op_dir(domain, op_name)
        if path_dir.is_dir() and match_mode(path_dir, mode):
            ops.add(op_name)
    return sorted(ops)


def main():
    parser = argparse.ArgumentParser(description="Parse changed ops from operator tree")
    parser.add_argument("-f", "--file", required=True, help="changed files list")
    parser.add_argument(
        "--mode",
        required=True,
        choices=("kernel", "examples", "ut_all", "ut_mc2", "ut_exclude_mc2"),
        help="selection mode",
    )
    args = parser.parse_args()
    ops = collect_ops(args.file, args.mode)
    print((";".join(ops) + ";") if ops else "")


if __name__ == "__main__":
    main()
