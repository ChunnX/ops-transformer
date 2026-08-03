#!/usr/bin/env python3
# coding: utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prove that preprocessing the trace switch off preserves baseline device tokens."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


DEFAULT_BASELINE = "4b3d39658e1d4f5d1f5b60acedfe2df8201e08b2"
FILES = (
    "mc2/mega_moe/op_kernel/arch35/mega_moe_wave.h",
    "mc2/mega_moe/op_kernel/arch35/mega_moe_impl.h",
    "mc2/mega_moe/op_kernel/arch35/block_epilogue_swiglu_mx_quant.h",
)
TRACE_IF = "#if MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE != 0"
TRACE_INCLUDE = '#include "mega_moe_trace.h"'


class VerificationError(RuntimeError):
    pass


def strip_disabled_trace_blocks(source: str) -> str:
    result: list[str] = []
    dropping_depth = 0
    for line_number, line in enumerate(source.splitlines(keepends=True), 1):
        stripped = line.strip()
        if dropping_depth:
            if stripped.startswith("#if"):
                dropping_depth += 1
            elif stripped.startswith("#endif"):
                dropping_depth -= 1
            elif stripped.startswith("#else") and dropping_depth == 1:
                raise VerificationError(
                    f"trace-only block has an unsupported #else at line {line_number}"
                )
            continue
        if stripped == TRACE_IF:
            dropping_depth = 1
            continue
        if stripped == TRACE_INCLUDE:
            continue
        result.append(line)
    if dropping_depth:
        raise VerificationError("unterminated trace-only preprocessor block")
    return "".join(result)


TOKEN_RE = re.compile(
    r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|'
    r"[A-Za-z_][A-Za-z0-9_]*|0[xX][0-9A-Fa-f]+[A-Za-z]*|"
    r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?[A-Za-z]*|"
    r"::|->|<<=|>>=|==|!=|<=|>=|&&|\|\||\+\+|--|<<|>>|\+=|-=|\*=|/=|%=|"
    r"&=|\|=|\^=|##|\.\.\.|[^\s]"
)


def cpp_tokens(source: str) -> list[str]:
    without_comments = re.sub(r"/\*.*?\*/|//[^\n]*", " ", source, flags=re.DOTALL)
    return TOKEN_RE.findall(without_comments)


def git_show(repo_root: Path, revision: str, path: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{revision}:{path}"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise VerificationError(
            f"cannot read baseline {revision}:{path}: {exc.stderr.strip()}"
        ) from exc
    return result.stdout.lstrip("\ufeff")


def first_difference(left: list[str], right: list[str]) -> str:
    common = min(len(left), len(right))
    index = next(
        (index for index in range(common) if left[index] != right[index]), common
    )
    start = max(0, index - 6)
    end = index + 7
    return (
        f"token {index}: current={left[index : index + 1]!r}, baseline={right[index : index + 1]!r}; "
        f"current context={left[start:end]!r}; baseline context={right[start:end]!r}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare trace-off device-source tokens with the recorded baseline commit."
    )
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[4]
    )
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    failures: list[str] = []
    for relative_path in FILES:
        current_source = (repo_root / relative_path).read_text(encoding="utf-8-sig")
        disabled_source = strip_disabled_trace_blocks(current_source)
        if "CriticalPathTrace" in disabled_source or "TRACE_POINT:" in disabled_source:
            failures.append(
                f"{relative_path}: trace token survives switch-off stripping"
            )
            continue
        baseline_source = git_show(repo_root, args.baseline, relative_path)
        current_tokens = cpp_tokens(disabled_source)
        baseline_tokens = cpp_tokens(baseline_source)
        if current_tokens != baseline_tokens:
            failures.append(
                f"{relative_path}: {first_difference(current_tokens, baseline_tokens)}"
            )
            continue
        print(
            f"[OK] trace-off token identity: {relative_path} ({len(current_tokens)} tokens)"
        )

    trace_header = repo_root / "mc2/mega_moe/op_kernel/arch35/mega_moe_trace.h"
    header_text = trace_header.read_text(encoding="utf-8")
    if "#define MEGA_MOE_ENABLE_CRITICAL_PATH_TRACE 1" not in header_text:
        failures.append(
            "mega_moe_trace.h: diagnostic compile switch does not default to 1"
        )
    if failures:
        raise VerificationError("\n  - " + "\n  - ".join(failures))
    print(
        f"[OK] trace-off production proof passed against baseline {args.baseline}; "
        "the diagnostic branch defaults to one, and an explicit switch value of zero removes "
        "all tracing tokens from modified production device files"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
