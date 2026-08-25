"""Read-only migration lint, not a C++ parser or proof of pipeline correctness.

Checks the six profiling anchors, explicit enum IDs/parser names, lexical marker
pairs, local capacity and final-flush placement. Optional baseline comparison
removes instrumentation and compares remaining source tokens. Dynamic call
counts, conditional paths, asynchronous dependencies and actual compile flags
still require review/build/on-device validation.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
from pathlib import Path
from typing import Sequence

try:
    from .analyze_cycle import DEFAULT_STAGE_NAMES
except ImportError:
    from analyze_cycle import DEFAULT_STAGE_NAMES

OPS = tuple(DEFAULT_STAGE_NAMES)
CURRENT_OPS = ("moe_ep_dispatch_epilogue", "moe_ep_combine")
ENUM = re.compile(r"enum\s+class\s+(\w+ProfileStage)\s*:\s*uint32_t\s*\{(.*?)\};", re.S)
EVENT = re.compile(
    r"MoeEpProfileOp::(\w+)\s*,\s*MoeEpProfileEngine::(\w+)\s*,\s*"
    r"(\w+ProfileStage)::(\w+)\s*,\s*MoeEpProfileBoundary::(BEGIN|END)"
)
MACRO = re.compile(r"\bMOE_EP_PROFILE_(?:MARK_PAYLOAD|MARK|INIT|DECLARE|FLUSH)\s*\(")
BUILD_FILES = (
    "build.sh",
    "CMakeLists.txt",
    "cmake/config.cmake",
    "cmake/scripts/prepare.sh",
    "torch_extension/setup.py",
    "mc2/common/torch_extension/elastic_buffer.py",
)


def without_comments(source: str) -> str:
    # Keep literals intact, including // inside strings.
    pattern = r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*.*?\*/'
    return re.sub(
        pattern,
        lambda m: " " if m.group().startswith(("//", "/*")) else m.group(),
        source,
        flags=re.S,
    )


def macro_spans(source: str):
    """Balanced parentheses suffice for the restricted instrumentation macros."""
    for match in MACRO.finditer(source):
        depth = 1
        end = match.end()
        while end < len(source) and depth:
            depth += (source[end] == "(") - (source[end] == ")")
            end += 1
        if depth:
            raise ValueError("unclosed profiling macro")
        while end < len(source) and source[end].isspace():
            end += 1
        if end < len(source) and source[end] == ";":
            end += 1
        yield match.start(), end, source[match.start() : end]


def business_tokens(source: str) -> list[str]:
    source = without_comments(source)
    source = re.sub(
        r'^\s*#include\s+"[^"\n]*moe_ep_profiler\.h"\s*$', "", source, flags=re.M
    )
    source = ENUM.sub("", source)
    for begin, end, _ in reversed(list(macro_spans(source))):
        source = source[:begin] + " " + source[end:]
    return re.findall(r'"(?:\\.|[^"\\])*"|\w+|[^\w\s]', source)


def inspect_source(op: str, source: str) -> dict:
    clean = without_comments(source)
    errors: list[str] = []
    warnings: list[str] = []
    enums = list(ENUM.finditer(clean))
    if len(enums) != 1:
        return {
            "op": op,
            "errors": ["expected one explicit ProfileStage enum"],
            "warnings": [],
        }
    enum_name = enums[0].group(1)
    stages = {
        name: int(number, 0)
        for name, number in re.findall(
            r"\b(\w+)\s*=\s*(0x[0-9a-fA-F]+|\d+)[uUlL]*", enums[0].group(2)
        )
    }
    if not stages or len(set(stages.values())) != len(stages):
        errors.append("stage IDs missing or duplicated")
    for name, stage_id in stages.items():
        if DEFAULT_STAGE_NAMES[op].get(stage_id) != name:
            errors.append(f"parser mapping mismatch: {name}={stage_id}")
    for anchor in (
        "moe_ep_profiler.h",
        "MOE_EP_PROFILE_DECLARE(",
        "MOE_EP_PROFILE_INIT(",
        "MOE_EP_PROFILE_FLUSH(",
    ):
        if anchor not in clean:
            errors.append(f"missing anchor {anchor}")
    spans = list(macro_spans(clean))
    markers = []
    expected_op = op.removeprefix("moe_ep_").upper()
    for begin, end, body in spans:
        if not body.startswith(
            ("MOE_EP_PROFILE_MARK(", "MOE_EP_PROFILE_MARK_PAYLOAD(")
        ):
            continue
        event = EVENT.search(body)
        if event is None:
            errors.append(
                "marker cannot be decoded; extend the validator for the new syntax"
            )
            continue
        event_op, engine, stage_enum, stage, boundary = event.groups()
        if event_op != expected_op or stage_enum != enum_name or stage not in stages:
            errors.append(f"marker enum/op mismatch: {event.group()}")
        markers.append((begin, end, engine, stage, boundary))
    counts = collections.Counter(
        (engine, stage, boundary) for _, _, engine, stage, boundary in markers
    )
    for engine, stage in {(m[2], m[3]) for m in markers}:
        if counts[engine, stage, "BEGIN"] != counts[engine, stage, "END"]:
            errors.append(f"lexically unmatched BEGIN/END: {engine}/{stage}")
    if not any(m[3:] == ("KERNEL", "BEGIN") for m in markers):
        errors.append("missing KERNEL BEGIN")
    flushes = [
        begin for begin, _, body in spans if body.startswith("MOE_EP_PROFILE_FLUSH(")
    ]
    for flush in flushes:
        previous = [m for m in markers if m[1] <= flush]
        if not previous or previous[-1][3:] != ("KERNEL", "END"):
            errors.append("Flush must follow the KERNEL END marker")
    for begin, end, _, stage, boundary in markers:
        if stage == "KERNEL" and boundary == "BEGIN":
            endings = [
                m[0] for m in markers if m[0] > end and m[3:] == ("KERNEL", "END")
            ]
            if endings and "MOE_EP_PROFILE_INIT(" in clean[end : min(endings)]:
                errors.append("profiler Init moved into measured KERNEL region")
    if len(markers) > 32:
        warnings.append(
            "more than 32 lexical sites: review exclusive branches and per-path marker budget"
        )
    warnings.append(
        "lexical pairing/site counts do not prove runtime paths, loop counts or data dependencies"
    )
    return {
        "op": op,
        "stage_ids": stages,
        "lexical_marker_sites": len(markers),
        "errors": errors,
        "warnings": warnings,
    }


def validate(root: Path, ops: Sequence[str], base_ref: str | None = None) -> dict:
    report = {
        "kernels": [],
        "errors": [],
        "scope": "structural source lint only; no CANN/NPU proof",
    }
    for op in ops:
        relative = f"mc2/{op}/op_kernel/{op}.h"
        source = (root / relative).read_text(encoding="utf-8")
        result = inspect_source(op, source)
        if base_ref:
            old = subprocess.run(
                ["git", "show", f"{base_ref}:{relative}"],
                cwd=root,
                check=True,
                capture_output=True,
                encoding="utf-8",
            ).stdout
            result["business_tokens_equal_baseline"] = business_tokens(
                old
            ) == business_tokens(source)
            if not result["business_tokens_equal_baseline"]:
                result["errors"].append(
                    "non-instrumentation tokens changed: review git diff against baseline"
                )
        report["kernels"].append(result)
        report["errors"].extend(f"{op}: {error}" for error in result["errors"])
    helper = (root / "mc2/common/op_kernel/moe_ep_profiler.h").read_text(
        encoding="utf-8"
    )
    if not re.search(r"MOE_EP_PROFILE_LOCAL_RECORDS\s*=\s*32U?\s*;", helper):
        report["errors"].append(
            "local marker capacity differs from reviewed 32-record ABI"
        )
    for relative in (*BUILD_FILES, *(f"mc2/{op}/op_host/CMakeLists.txt" for op in ops)):
        source = (root / relative).read_text(encoding="utf-8")
        if not re.search(
            r"ENABLE_MOE_EP_PROFILING|CANN_OPS_MOE_EP_PROFILING_BUILD|moe-ep-profiler",
            source,
        ):
            report["errors"].append(
                f"profiling build-switch anchor missing: {relative}"
            )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--ops", nargs="+", choices=OPS, default=list(CURRENT_OPS))
    parser.add_argument(
        "--base-ref",
        help="known business-code baseline; compare after stripping instrumentation",
    )
    args = parser.parse_args(argv)
    try:
        report = validate(args.repo.resolve(), args.ops, args.base_ref)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        report = {"errors": [str(error)]}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
