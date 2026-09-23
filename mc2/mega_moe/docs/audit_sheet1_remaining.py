#!/usr/bin/env python3
"""Count mega_q.xlsx Sheet1 entries that still have duplicate source fragments.

The spreadsheet records a fragment's baseline file and line range, but not its
matching peer. For each entry, take every 10-line code window in the original
fragment (or the whole fragment if it has fewer than 10 code lines). An entry
remains when at least one window still occurs in its recorded file and a
distinct peer location in the current MegaMoe sources. Moving one copy into a
shared implementation resolves the original file-specific entry. Comments,
blank lines, and whitespace are ignored.

Usage: python audit_sheet1_remaining.py /path/to/mega_q.xlsx
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import re
import subprocess

import openpyxl

REPO = pathlib.Path(__file__).resolve().parents[3]
BASELINE = "e66cb2f8e9"
# These three entries have no duplicate peer inside mc2/mega_moe even on the
# untouched baseline (their peer appears outside this scan scope or comments
# leave fewer usable lines). Count them as remaining rather than a false fix.
UNMATCHED_BASELINE_ROWS = {41, 131, 398}


def code_line(line: str) -> str:
    line = re.sub(r"//.*", "", line).strip()
    if not line or line.startswith(("*", "/*", "*/")):
        return ""
    return re.sub(r"\s+", "", line)


def baseline_lines(revision: str, path: str) -> list[str]:
    result = subprocess.run(
        ["git", "show", f"{revision}:{path}"], cwd=REPO, text=True,
        capture_output=True, check=True,
    )
    return result.stdout.splitlines()


def source_files(source_root: pathlib.Path) -> dict[str, list[tuple[int, str]]]:
    root = source_root / "mc2/mega_moe"
    files = {}
    for path in root.rglob("*"):
        if path.suffix not in {".h", ".hpp", ".cpp"} or not path.is_file():
            continue
        rel = str(path.relative_to(source_root))
        files[rel] = [
            (line_no, text) for line_no, line in enumerate(path.read_text(errors="replace").splitlines())
            if (text := code_line(line))
        ]
    return files


def index_windows(files: dict[str, list[tuple[int, str]]], widths: set[int]):
    indexes = {width: collections.defaultdict(list) for width in widths}
    for name, lines in files.items():
        for width, index in indexes.items():
            for start in range(len(lines) - width + 1):
                key = tuple(text for _, text in lines[start:start + width])
                index[key].append((name, lines[start][0]))
    return indexes


def distinct_pair(locations: list[tuple[str, int]], recorded_file: str, span: int) -> bool:
    for name, pos in locations:
        if name != recorded_file:
            continue
        if any(other != name or abs(other_pos - pos) > span + 2
               for other, other_pos in locations):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook")
    parser.add_argument("--baseline-revision", default=BASELINE)
    parser.add_argument("--source-root", type=pathlib.Path, default=REPO)
    parser.add_argument("--list-rows", action="store_true")
    args = parser.parse_args()
    workbook = openpyxl.load_workbook(args.workbook, read_only=True, data_only=True)
    revision = args.baseline_revision
    rows = list(workbook.worksheets[0].values)[1:]
    entries = []
    cached_baseline = {}
    for sheet_row, row in enumerate(rows, 2):
        if not row[0]:
            continue
        path = str(row[0]).split("ops/ops-transformer/")[-1].strip()
        if path not in cached_baseline:
            cached_baseline[path] = baseline_lines(revision, path)
        fragment = [text for line in cached_baseline[path][int(row[2]) - 1:int(row[3])]
                    if (text := code_line(line))]
        if not fragment:
            raise ValueError(f"Sheet1 row {sheet_row} has no baseline code")
        entries.append((sheet_row, path, int(row[1]), fragment, min(10, len(fragment))))

    files = source_files(args.source_root)
    indexes = index_windows(files, {entry[4] for entry in entries})
    remaining = []
    for sheet_row, path, span, fragment, width in entries:
        if sheet_row in UNMATCHED_BASELINE_ROWS:
            remaining.append((sheet_row, path))
            continue
        for start in range(len(fragment) - width + 1):
            key = tuple(fragment[start:start + width])
            if distinct_pair(indexes[width].get(key, []), path, span):
                remaining.append((sheet_row, path))
                break

    total = len(entries)
    print(f"baseline={revision} Sheet1={total} remaining={len(remaining)} "
          f"ratio={len(remaining) / total:.2%} target=<20%")
    by_file = collections.Counter(path for _, path in remaining)
    for path, count in by_file.most_common():
        print(f"{count:3} {path}")
    if args.list_rows:
        print("remaining_sheet_rows=" + ",".join(str(row) for row, _ in remaining))


if __name__ == "__main__":
    main()
