#!/usr/bin/env python3
"""Check tiling_dump compatibility across all operators.

Scans every operator's answer directory for:
  - *_tiling.h with TILING_DATA_FIELD_DEF entries
  - *.cpp with SaveToBuffer + SetDataSize

Outputs a structured report to stdout and a JSON file.

Usage:
    python scripts/check_tiling_dump.py
    python scripts/check_tiling_dump.py -o report.json
"""

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RE_SCALAR = re.compile(r"TILING_DATA_FIELD_DEF\(\s*(\w+)\s*,\s*(\w+)\s*\)")
RE_ARRAY = re.compile(r"TILING_DATA_FIELD_DEF_ARR\(\s*(\w+)\s*,\s*(\d+)\s*,\s*(\w+)\s*\)")
RE_CLASS = re.compile(r"BEGIN_TILING_DATA_DEF\(\s*(\w+)\s*\)")
RE_SAVE_BUF = re.compile(r"SaveToBuffer\(")
RE_SET_DATA_SIZE = re.compile(r"SetDataSize\(")

CATEGORY_LABELS = {
    "ok": "支持",
    "no_answer": "无 answer 目录",
    "no_tiling_h": "无 tiling.h",
    "no_fields": "有 tiling.h 但无 FIELD_DEF",
    "no_savebuffer": "有 FIELD_DEF 但无 SaveToBuffer",
    "no_setdatasize": "有 SaveToBuffer 但无 SetDataSize",
}

CATEGORY_ORDER = ["ok", "no_answer", "no_tiling_h", "no_fields", "no_savebuffer", "no_setdatasize"]


def check_operator(op_dir: Path) -> dict:
    """Check a single operator for tiling_dump compatibility."""
    rel = str(op_dir.relative_to(ROOT / "tasks"))
    parts = rel.split(os.sep)
    level = parts[0] if len(parts) > 0 else "?"
    category = parts[1] if len(parts) > 1 else "?"
    op_name = parts[2] if len(parts) > 2 else Path(op_dir).name

    entry = {"path": rel, "level": level, "category": category, "op_name": op_name}

    answer_dir = op_dir / "answer"
    if not answer_dir.is_dir():
        return {**entry, "status": "no_answer", "detail": ""}

    ans_dirs = sorted(d for d in answer_dir.iterdir() if d.is_dir() and d.name.isdigit())
    if not ans_dirs:
        return {**entry, "status": "no_answer", "detail": ""}

    ans = ans_dirs[0]
    tiling_headers = list(ans.glob("op_host/*_tiling.h"))
    if not tiling_headers:
        return {**entry, "status": "no_tiling_h", "detail": ""}

    h = tiling_headers[0]
    content = h.read_text(encoding="utf-8", errors="replace")
    scalars = RE_SCALAR.findall(content)
    arrays = RE_ARRAY.findall(content)
    field_names = [s[1] for s in scalars] + [a[2] for a in arrays]

    if not field_names:
        return {**entry, "status": "no_fields", "detail": f"header: {h.name}"}

    host_cpp_files = list(ans.glob("op_host/*.cpp"))
    savebuf_files = []
    setdatasize_files = []
    for cpp in host_cpp_files:
        txt = cpp.read_text(encoding="utf-8", errors="replace")
        if RE_SAVE_BUF.search(txt):
            savebuf_files.append(cpp.name)
            if RE_SET_DATA_SIZE.search(txt):
                setdatasize_files.append(cpp.name)

    if not savebuf_files:
        return {**entry, "status": "no_savebuffer", "detail": f"fields: {', '.join(field_names[:5])}"}
    if not setdatasize_files:
        return {
            **entry,
            "status": "no_setdatasize",
            "detail": f"SaveToBuffer in {', '.join(savebuf_files)}, no SetDataSize",
        }

    return {
        **entry,
        "status": "ok",
        "detail": f"{len(field_names)} fields ({', '.join(field_names[:5])}{'...' if len(field_names) > 5 else ''})",
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Check tiling_dump compatibility")
    parser.add_argument("-o", "--output", default=None, help="Output report file path (.md)")
    args = parser.parse_args()

    tasks_dir = ROOT / "tasks"
    results = {cat: [] for cat in CATEGORY_ORDER}

    for op_dir in sorted(tasks_dir.rglob("*")):
        if not op_dir.is_dir():
            continue
        if not (op_dir / "validation").is_dir():
            continue
        entry = check_operator(op_dir)
        results.setdefault(entry["status"], []).append(entry)

    # --- Print report ---
    total = sum(len(v) for v in results.values())
    print(f"\n{'=' * 70}")
    print(f"  tiling_dump 兼容性报告")
    print(f"{'=' * 70}")
    print(f"  总算子数: {total}")
    print()

    # Summary table
    print(f"  {'状态':<35} {'数量':>5} {'占比':>7}")
    print(f"  {'-' * 35} {'-' * 5} {'-' * 7}")
    for cat in CATEGORY_ORDER:
        label = CATEGORY_LABELS[cat]
        n = len(results[cat])
        pct = f"{n / total * 100:.1f}%" if total else "0%"
        marker = "  " if cat == "ok" else ">>"
        print(f"{marker}{label:<33} {n:>5} {pct:>7}")
    print()

    # Details for non-ok categories
    for cat in CATEGORY_ORDER:
        if cat == "ok":
            continue
        ops = results[cat]
        if not ops:
            continue
        label = CATEGORY_LABELS[cat]
        print(f"  --- {label} ({len(ops)}) ---")
        for op in ops:
            detail = f"  ({op['detail']})" if op["detail"] else ""
            print(f"    {op['path']}{detail}")
        print()

    # --- Save Markdown report ---
    output_path = args.output or str(ROOT / "tiling_dump_report.md")
    lines = _build_markdown(total, results)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  报告已保存: {output_path}")


def _build_markdown(total: int, results: dict) -> list[str]:
    """Build Markdown report lines."""
    lines = []
    lines.append("# tiling_dump 兼容性报告")
    lines.append("")
    lines.append(f"总算子数: **{total}**")
    lines.append("")

    # Summary table
    lines.append("## 汇总")
    lines.append("")
    lines.append("| 状态 | 数量 | 占比 |")
    lines.append("|------|------|------|")
    for cat in CATEGORY_ORDER:
        label = CATEGORY_LABELS[cat]
        n = len(results[cat])
        pct = f"{n / total * 100:.1f}%" if total else "0%"
        lines.append(f"| {label} | {n} | {pct} |")
    lines.append("")

    # Details for non-ok
    for cat in CATEGORY_ORDER:
        if cat == "ok":
            continue
        ops = results[cat]
        if not ops:
            continue
        label = CATEGORY_LABELS[cat]
        lines.append(f"## {label} ({len(ops)})")
        lines.append("")
        lines.append("| Level | 类别 | 算子 | 详情 |")
        lines.append("|-------|------|------|------|")
        for op in ops:
            lines.append(f"| {op['level']} | {op['category']} | {op['op_name']} | {op['detail']} |")
        lines.append("")

    return lines


if __name__ == "__main__":
    main()
