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

import yaml

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
_CONFIG_CACHE = {}


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


def load_domain_config(domain):
    if domain in _CONFIG_CACHE:
        return _CONFIG_CACHE[domain]
    for candidate in (
        REPO_ROOT / domain / "common" / "ci.yaml",
        REPO_ROOT / domain / "ci.yaml",
    ):
        if candidate.is_file():
            with open(candidate, "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
            _CONFIG_CACHE[domain] = cfg
            return cfg
    _CONFIG_CACHE[domain] = {}
    return {}


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


def match_mode(path, mode):
    if mode == "kernel":
        return has_pkg_sources(path)
    if mode == "examples":
        return has_examples(path)
    if mode.startswith("ut"):
        return has_pr_ut(path)
    return False


def mode_key(mode):
    if mode == "kernel":
        return "kernel"
    if mode == "examples":
        return "examples"
    return "ut"


def op_config(cfg, op_name):
    return (cfg.get("ops") or {}).get(op_name) or {}


def configured_value(cfg, domain, op_name, mode):
    key = mode_key(mode)
    op_cfg = op_config(cfg, op_name)
    if key in op_cfg:
        return op_cfg[key]
    return (cfg.get("defaults") or {}).get(key)


def eligible(cfg, domain, op_name, mode):
    value = configured_value(cfg, domain, op_name, mode)
    if value is False:
        return False
    if value is True:
        return True
    return match_mode(op_dir(domain, op_name), mode)


def add_with_links(ops, cfg, domain, op_name, mode):
    ops.add(op_name)
    for linked in op_config(cfg, op_name).get("links") or []:
        if configured_value(cfg, domain, linked, mode) is not False:
            ops.add(linked)


def iter_domain_ops(cfg, domain):
    configured = cfg.get("ops") or {}
    if configured:
        return [
            op_dir(domain, name) for name in configured if op_dir(domain, name).is_dir()
        ]
    return domain_op_dirs(domain)


def expand_domain(cfg, domain, mode):
    ops = set()
    for child in iter_domain_ops(cfg, domain):
        if eligible(cfg, domain, child.name, mode):
            add_with_links(ops, cfg, domain, child.name, mode)
    return ops


def path_under(path, prefix):
    prefix = prefix.rstrip("/")
    return path == prefix or path.startswith(prefix + "/")


def collect_from_watch(cfg, domain, changed_path, mode):
    ops = set()
    if mode != "examples":
        for prefix in cfg.get("watch") or []:
            if path_under(changed_path, str(prefix)):
                ops.update(expand_domain(cfg, domain, mode))
    for op_name, op_cfg in (cfg.get("ops") or {}).items():
        for prefix in op_cfg.get("watch") or []:
            if path_under(changed_path, str(prefix)) and eligible(
                cfg, domain, op_name, mode
            ):
                add_with_links(ops, cfg, domain, op_name, mode)
    return ops


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
        cfg = load_domain_config(domain)
        if op_name in ("common", "3rd"):
            ops.update(collect_from_watch(cfg, domain, path, mode))
            continue
        path_dir = op_dir(domain, op_name)
        watch_ops = collect_from_watch(cfg, domain, path, mode)
        if watch_ops:
            ops.update(watch_ops)
        if (cfg.get("ops") or {}) and op_name not in (cfg.get("ops") or {}):
            continue
        if path_dir.is_dir() and eligible(cfg, domain, op_name, mode):
            add_with_links(ops, cfg, domain, op_name, mode)
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
