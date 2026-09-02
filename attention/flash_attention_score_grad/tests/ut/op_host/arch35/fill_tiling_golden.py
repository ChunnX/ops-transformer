#!/usr/bin/env python3
"""Fill tilingKey / tilingData / workspace goldens for kSkipTilingKey host UTs.

Dump (does not compare goldens):
  OPS_UT_DUMP_GOLDEN=1 <host_ut_bin> --gtest_filter='FlashAttentionScoreGradTiling.*' | tee golden.dump

Apply to skip cases only:
  python3 fill_tiling_golden.py --src test_flash_attention_score_grad_tiling.cpp --dump golden.dump --dry-run
  python3 fill_tiling_golden.py --src test_flash_attention_score_grad_tiling.cpp --dump golden.dump --apply

Default: tilingData with more than 600 int64 tokens uses FNV hash (ExecuteTestCase ... , 0, true),
because 262144-byte caps would dump 32k numbers into the cpp. Pass --plain to force full strings.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SKIP_CALL = (
    "ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, kSkipTilingKey, \"\", {});"
)
TEST_RE = re.compile(
    r"^TEST_F\(FlashAttentionScoreGradTiling, ([A-Za-z0-9_]+)\)\s*$"
)
BEGIN = "### GOLDEN_BEGIN"
END = "### GOLDEN_END"
DEFAULT_HASH_AFTER_TOKENS = 600
WRAP_WIDTH = 118
KNOWN_KEYS = (
    "test",
    "tilingKey",
    "workspace",
    "dataBytes",
    "tilingDataHash",
    "tilingData",
)
# build/gtest logs often prefix: "[2026-09-02 14:20:00] test=..." or "12: ### GOLDEN_BEGIN"
BRACKET_PREFIX_RE = re.compile(r"^(\[[^\]]+\]\s*)+")
CTEST_PREFIX_RE = re.compile(r"^\d+:\s*")


def _content(line: str) -> str:
    line = line.strip()
    line = BRACKET_PREFIX_RE.sub("", line)
    line = CTEST_PREFIX_RE.sub("", line)
    return line.strip()


def parse_dump(text: str) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    in_block = False
    for raw in text.splitlines():
        line = _content(raw.rstrip("\r"))
        if BEGIN in line:
            current = {}
            in_block = True
            continue
        if END in line:
            if current and current.get("test"):
                records[current["test"].strip()] = current
            current = None
            in_block = False
            continue
        if not in_block or current is None:
            continue
        for key in KNOWN_KEYS:
            prefix = key + "="
            if line.startswith(prefix):
                value = line[len(prefix) :]
                current[key] = value.strip() if key != "tilingData" else value
                break
    return records


def wrap_tiling_data(data: str, indent: str = "        ") -> str:
    tokens = data.strip().split()
    if not tokens:
        return f'{indent}"";'
    lines: list[str] = []
    buf: list[str] = []
    width = 0
    budget = WRAP_WIDTH - len(indent) - 3  # quotes + trailing space in quotes
    for tok in tokens:
        add = len(tok) + (1 if buf else 0)
        if buf and width + add > budget:
            lines.append(indent + '"' + " ".join(buf) + ' "')
            buf = [tok]
            width = len(tok)
        else:
            buf.append(tok)
            width += add
    last = " ".join(buf) + " "
    lines.append(indent + '"' + last + '";')
    if len(lines) == 1:
        return f"    std::string expectTilingData =\n{lines[0]}"
    return "    std::string expectTilingData =\n" + "\n".join(lines)


def format_workspaces(ws: str) -> str:
    parts = [p.strip() for p in ws.split(",") if p.strip()]
    if not parts:
        return "    std::vector<size_t> expectWorkspaces = {};"
    return "    std::vector<size_t> expectWorkspaces = {" + ", ".join(parts) + "};"


def render_snippet(rec: dict[str, str], hash_after: int, force_plain: bool) -> str:
    key = rec.get("tilingKey", "0")
    data = rec.get("tilingData", "")
    digest = rec.get("tilingDataHash", "")
    tokens = data.strip().split() if data.strip() else []
    use_hash = (not force_plain) and len(tokens) > hash_after
    lines = [f"    int64_t expectTilingKey = {key};"]
    if use_hash:
        if not digest:
            raise ValueError(f"missing tilingDataHash for {rec.get('test')}")
        lines.append(f'    std::string expectTilingData = "{digest}";')
        lines.append(format_workspaces(rec.get("workspace", "")))
        lines.append(
            "    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, "
            "expectTilingKey, expectTilingData, expectWorkspaces, 0, true);"
        )
    else:
        lines.append(wrap_tiling_data(data if data.endswith(" ") else data + (" " if data else "")))
        lines.append(format_workspaces(rec.get("workspace", "")))
        lines.append(
            "    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, "
            "expectTilingKey, expectTilingData, expectWorkspaces);"
        )
    return "\n".join(lines)


def patch_source(src: str, records: dict[str, dict[str, str]], hash_after: int, force_plain: bool) -> tuple[str, list[str], list[str]]:
    lines = src.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    updated: list[str] = []
    missing: list[str] = []
    current_test: str | None = None
    while i < len(lines):
        line = lines[i]
        m = TEST_RE.match(line.rstrip("\n"))
        if m:
            current_test = m.group(1)
        stripped = line.strip()
        if stripped == SKIP_CALL:
            if current_test is None:
                missing.append("<unknown>")
                out.append(line)
                i += 1
                continue
            rec = records.get(current_test)
            if rec is None:
                missing.append(current_test)
                out.append(line)
                i += 1
                continue
            indent_nl = "\n" if line.endswith("\n") else ""
            snippet = render_snippet(rec, hash_after, force_plain)
            out.append(snippet + indent_nl)
            updated.append(current_test)
            i += 1
            continue
        out.append(line)
        i += 1
    return "".join(out), updated, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="host UT cpp to patch")
    parser.add_argument("--dump", type=Path, required=True, help="gtest stdout captured with OPS_UT_DUMP_GOLDEN=1")
    parser.add_argument("--apply", action="store_true", help="write --src in place (default is dry-run)")
    parser.add_argument("--dry-run", action="store_true", help="print summary only (default if --apply is absent)")
    parser.add_argument(
        "--hash-after-tokens",
        type=int,
        default=DEFAULT_HASH_AFTER_TOKENS,
        help="use FNV hash when tilingData has more int64 tokens than this (default 600)",
    )
    parser.add_argument("--plain", action="store_true", help="always write full tilingData strings")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="with --apply, refuse to write if any skip case has no dump record",
    )
    args = parser.parse_args()

    dump_text = args.dump.read_text(encoding="utf-8", errors="ignore")
    records = parse_dump(dump_text)
    src = args.src.read_text(encoding="utf-8")
    new_src, updated, missing = patch_source(src, records, args.hash_after_tokens, args.plain)

    skip_count = src.count(SKIP_CALL)
    begin_hits = dump_text.count("GOLDEN_BEGIN")
    print(f"GOLDEN_BEGIN in dump file: {begin_hits}")
    print(f"parsed dump records: {len(records)}")
    print(f"skip ExecuteTestCase lines: {skip_count}")
    print(f"would patch: {len(updated)}")
    if updated:
        print("  " + "\n  ".join(updated))
    if missing:
        print(f"skip lines with no dump record: {len(missing)}")
        print("  " + "\n  ".join(missing))
        print(
            "hint: those tests either failed tiling (no dump), were not run, "
            "or golden.dump did not capture ### GOLDEN_BEGIN. "
            "grep -c GOLDEN_BEGIN golden.dump and compare with skip count."
        )

    if args.apply:
        if not records:
            print("error: dump parsed 0 records; not writing", file=sys.stderr)
            return 1
        if args.strict and missing:
            print("error: --strict refuse --apply while dump records are missing", file=sys.stderr)
            return 1
        if new_src == src:
            print("no changes")
            return 0
        args.src.write_text(new_src, encoding="utf-8")
        print(f"wrote {args.src}")
        return 0

    print("dry-run; pass --apply to write the cpp (matched skip cases only)")
    return 0 if not missing else 2


if __name__ == "__main__":
    sys.exit(main())
