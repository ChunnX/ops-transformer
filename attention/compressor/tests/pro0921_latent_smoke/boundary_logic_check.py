# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Native CPU logic test for the grouped-decode boundary/window derivation.
#
# The shipped algorithm lives in the AscendC headers, which cannot be compiled
# on a CPU host. This harness extracts the real `BatchMetadata`,
# `CrossesCompressionBoundary` and `DeriveGroupedWindow` definitions from
# compressor_tools.h / quant_compressor_tools.h, compiles only that pure
# arithmetic into a tiny standalone translation unit with g++, and compares the
# result against an independent Python oracle. No AscendC, no bisheng, no NPU.
#
# Modes:
#   --extract-only   print the extracted snippet (no compiler needed)
#   --oracle-only    run the Python oracle only
#   (default)        extract + compile + run + compare

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
HEADERS = {
    "compressor": (HERE / "../../op_kernel/arch35/compressor_tools.h").resolve(),
    "quant_compressor": (
        HERE / "../../../quant_compressor/op_kernel/arch35/quant_compressor_tools.h"
    ).resolve(),
}


def _extract_block(text, marker, with_semicolon=False):
    """Return the marked definition through its matching closing brace.

    Braces are matched by counting so nested blocks inside the function body do
    not truncate the extraction.
    """
    start = text.index(marker)
    brace = text.index("{", start)
    depth = 0
    index = brace
    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index + 1
                if with_semicolon and text[end:end + 1] == ";":
                    end += 1
                return text[start:end]
        index += 1
    raise ValueError("unbalanced braces for marker: %s" % marker)


def extract_snippet(header_path):
    text = header_path.read_text(encoding="utf-8")
    batch = _extract_block(text, "struct BatchMetadata {", with_semicolon=True)
    cross = _extract_block(text, "__aicore__ inline bool CrossesCompressionBoundary(")
    derive = _extract_block(text, "__aicore__ inline void DeriveGroupedWindow(")
    return "\n".join((batch, cross, derive))


def oracle(start_pos, seq_used, ratio, read_size):
    """Independent reference for the peak logic (not the header code)."""
    if ratio == 0:
        return (False, 0, start_pos, 0)
    group_offset = start_pos % ratio
    active = seq_used >= ratio - group_offset
    if not active:
        return (False, 0, start_pos, 0)
    group_end = ((start_pos + seq_used) // ratio) * ratio
    window_start = group_end - read_size
    current = start_pos - window_start if start_pos > window_start else 0
    return (True, read_size, window_start, current)


def case_table():
    cases = []
    for ratio in (4, 128):
        for start_pos in (0, 1, ratio - 1, ratio, ratio + 1, 2 * ratio - 1, 2 * ratio, 5 * ratio + 3):
            for seq_used in (1, 2, ratio):
                if seq_used > ratio:
                    continue
                cases.append((start_pos, seq_used, ratio, ratio))
    return cases


CPP_MAIN = """
int main() {
%(body)s
    return 0;
}
"""


def build_and_compare(snippet, cases, out_dir):
    body_lines = []
    for index, (start_pos, seq_used, ratio, read_size) in enumerate(cases):
        body_lines.append(
            "    { BatchMetadata m; m.rawStartPos = %dU; m.rawSeqUsed = %dU;"
            " DeriveGroupedWindow(m, %dU, %dU);"
            " printf(\"%%u %%u %%u %%u %%u %%u %%u\\n\", m.active ? 1U : 0U, m.logicalSeqUsed,"
            " m.windowStart, m.currentStart, m.rawStartPos, m.rawSeqUsed, %dU); }"
            % (start_pos, seq_used, read_size, ratio, index)
        )
    source = (
        "#include <cstdint>\n#include <cstdio>\n"
        "#define __aicore__\n"
        + snippet
        + CPP_MAIN % {"body": "\n".join(body_lines)}
    )
    src = out_dir / "boundary_logic_tu.cpp"
    exe = out_dir / "boundary_logic_tu"
    src.write_text(source, encoding="utf-8")
    subprocess.run(["g++", "-std=c++17", "-O2", "-o", str(exe), str(src)], check=True)
    output = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout
    failures = []
    for (start_pos, seq_used, ratio, read_size), line in zip(cases, output.strip().splitlines()):
        active, logical, window, current, raw_start, raw_used, index = (int(v) for v in line.split())
        expected = oracle(start_pos, seq_used, ratio, read_size)
        got = (bool(active), logical, window, current)
        if got != expected:
            failures.append((int(index), (start_pos, seq_used, ratio, read_size), got, expected))
    return failures, src, exe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--header", default="compressor")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--oracle-only", action="store_true")
    args = parser.parse_args()

    header_path = HEADERS[args.header]
    if not header_path.exists():
        print("FAIL: header not found: %s" % header_path)
        return 2
    snippet = extract_snippet(header_path)
    if args.extract_only:
        print(snippet)
        return 0

    cases = case_table()
    oracle_failures = []
    for case in cases:
        start_pos, seq_used, ratio, read_size = case
        expected = oracle(start_pos, seq_used, ratio, read_size)
        if expected[1] > 0 and expected[2] % ratio != 0:
            oracle_failures.append((case, expected))
    if oracle_failures:
        print("FAIL: oracle window not group aligned for %r" % oracle_failures[:3])
        return 1
    if args.oracle_only:
        print("ORACLE_OK cases=%d" % len(cases))
        return 0

    with tempfile.TemporaryDirectory(prefix="pro0921_boundary_") as tmp:
        try:
            failures, src, exe = build_and_compare(snippet, cases, Path(tmp))
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print("FAIL: unable to compile/run the extracted logic: %s" % exc)
            return 3
        if failures:
            print("FAIL: %d mismatches" % len(failures))
            for item in failures[:5]:
                print("  case=%r got=%r expected=%r" % item)
            return 1
    print("PASS cases=%d header=%s" % (len(cases), header_path.name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
