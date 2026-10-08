#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Watch the remote log branch and summarize MQFA CI results."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_CASE = "b270_sink_n1_6"
SUMMARY_RE = re.compile(r"^build_verify_logs/[^/]+/SUMMARY\.txt$")


def git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.strip()


def summary_paths(repo: Path, ref: str) -> list[str]:
    paths = git(
        repo, "ls-tree", "-r", "--name-only", ref, "--", "build_verify_logs"
    ).splitlines()
    return sorted(path for path in paths if SUMMARY_RE.match(path))


def blob(repo: Path, ref: str, path: str) -> str:
    return git(repo, "show", f"{ref}:{path}")


def report(
    repo: Path,
    ref: str,
    case: str,
    expected_commit: str | None = None,
    changed_paths: list[str] | None = None,
    strict: bool = False,
) -> bool | None:
    paths = changed_paths if changed_paths is not None else summary_paths(repo, ref)
    for path in reversed(paths):
        summary = blob(repo, ref, path)
        fields = dict(
            line.split("=", 1) for line in summary.splitlines() if "=" in line
        )
        if fields.get("case") != case:
            continue
        tested_commit = fields.get("commit", "unknown")
        if expected_commit and not tested_commit.startswith(expected_commit):
            continue

        print(
            f"CI log commit={git(repo, 'rev-parse', '--short', ref)} "
            f"tested_commit={tested_commit} case={case} "
            f"result={fields.get('result', 'unknown')}"
        )
        parent = str(Path(path).parent).replace("\\", "/")
        available = set(
            git(repo, "ls-tree", "-r", "--name-only", ref, "--", parent).splitlines()
        )
        detail_paths = [
            f"{parent}/{name}" for name in ("build_errors.log", "verify.log")
        ]
        artifacts = [
            artifact
            for artifact in git(
                repo, "ls-tree", "-r", "--name-only", ref, "--", parent
            ).splitlines()
            if artifact.lower().endswith(".pt")
        ]
        print(f"  pt_artifacts={', '.join(artifacts) if artifacts else 'none'}")
        detail = "\n".join(blob(repo, ref, p) for p in detail_paths if p in available)
        compile_result = (
            "FAIL"
            if re.search(
                r"(error:|fatal error:|BUILD FAILED|ninja: build stopped)", detail, re.I
            )
            else "PASS (no build errors recorded)"
        )
        matches = re.findall(r"FailElems\s*:\s*([^\n]+)", detail)
        max_abs = re.findall(r"MaxAbsErr\s*:\s*([^\n]+)", detail)
        precision = (
            f"fail_elements={', '.join(value.strip() for value in matches)}"
            if matches
            else "precision=not reported"
        )
        if max_abs:
            precision += f" max_abs={', '.join(value.strip() for value in max_abs)}"
        print(f"  compile={compile_result}; {precision}")
        for line in detail.splitlines():
            if re.search(
                r"RESULT:|Precision check failed|error:|fatal error:|FailElems\s*:",
                line,
                re.I,
            ):
                print(f"  {line.strip()}")
        if not strict:
            return True

        fail_elements = re.findall(r"FailElems\s*:\s*(\d+)\s*/", detail)
        precision_ok = (
            bool(fail_elements)
            and len(fail_elements) == len(matches)
            and all(int(count) == 0 for count in fail_elements)
        )
        result_ok = fields.get("result", "").strip().upper() == "PASS"
        compile_ok = compile_result.startswith("PASS")
        return result_ok and compile_ok and precision_ok
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="repository root")
    parser.add_argument("--case", default=DEFAULT_CASE, help="test case to report")
    parser.add_argument(
        "--commit",
        help="only report a result whose tested commit starts with this hash",
    )
    parser.add_argument(
        "--interval", type=float, default=20.0, help="poll interval in seconds"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="fetch and report the latest result, then exit",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="require a matching PASS result with zero failed elements; return nonzero otherwise",
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")

    state_path = Path(
        git(repo, "rev-parse", "--git-path", "mqfa-ci-log-monitor/last-seen")
    )
    state_path.parent.mkdir(parents=True, exist_ok=True)

    first_poll = True
    while True:
        git(repo, "fetch", "--quiet", "origin", "log")
        tip = git(repo, "rev-parse", "origin/log")
        # A restarted strict watcher must also inspect already-seen summaries:
        # last-seen tracks the branch, not successful verification of --commit.
        if args.once or (first_poll and args.check):
            status = report(
                repo, "origin/log", args.case, args.commit, strict=args.check
            )
            if status is not None:
                return 0 if status else 1
            if args.once:
                print(
                    f"No CI summary found for case {args.case}"
                    f"{f' and commit {args.commit}' if args.commit else ''} on origin/log."
                )
                return 1 if args.check else 0
        first_poll = False

        last = (
            state_path.read_text(encoding="utf-8").strip()
            if state_path.exists()
            else ""
        )
        if not last:
            status = report(
                repo, "origin/log", args.case, args.commit, strict=args.check
            )
            if args.check and status is True:
                state_path.write_text(tip + "\n", encoding="utf-8")
                return 0
            if args.check and status is False:
                state_path.write_text(tip + "\n", encoding="utf-8")
                return 1
            state_path.write_text(tip + "\n", encoding="utf-8")
        elif last != tip:
            merge_base = git(repo, "merge-base", last, tip, check=False)
            if merge_base == last:
                commits = git(
                    repo, "rev-list", "--reverse", f"{last}..{tip}"
                ).splitlines()
            else:
                commits = [tip]
            for commit in commits:
                changed = git(
                    repo, "diff-tree", "--no-commit-id", "--name-only", "-r", commit
                ).splitlines()
                summaries = [path for path in changed if SUMMARY_RE.match(path)]
                message = git(repo, "show", "-s", "--format=%s", commit)
                print(f"log branch update {commit[:12]}: {message}")
                if summaries:
                    status = report(
                        repo,
                        commit,
                        args.case,
                        args.commit,
                        summaries,
                        strict=args.check,
                    )
                    if args.check and status is True:
                        state_path.write_text(tip + "\n", encoding="utf-8")
                        return 0
                    if args.check and status is False:
                        state_path.write_text(tip + "\n", encoding="utf-8")
                        return 1
            state_path.write_text(tip + "\n", encoding="utf-8")

        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Stopped.", file=sys.stderr)
