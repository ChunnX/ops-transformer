#!/usr/bin/env python3
"""逐 case 运行 msprof，收集随机名文件夹，提取 Duration(us) 汇总到 Excel。

用法：python3 run_perf_collect.py
"""

import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── 定位目录 ──────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
XLSX_FILE = SCRIPT_DIR / "excel" / "stem_indexer_cases.xlsx"
TEST_SCRIPT = "test_stem_indexer_batch.py"
COLLECT_DIR = SCRIPT_DIR / "prof_collected"

# ── 检查依赖 ──────────────────────────────────────────────
for mod in ("openpyxl",):
    try:
        __import__(mod)
    except ImportError:
        print(f"Error: {mod} is required. Install with: pip install {mod}", file=sys.stderr)
        sys.exit(1)

# ── 初始化 ────────────────────────────────────────────────
if not XLSX_FILE.exists():
    print(f"Error: xlsx not found: {XLSX_FILE}", file=sys.stderr)
    sys.exit(2)

if COLLECT_DIR.exists():
    shutil.rmtree(COLLECT_DIR)
COLLECT_DIR.mkdir(parents=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
summary_xlsx = COLLECT_DIR / f"prof_summary_{timestamp}.xlsx"

# 创建带表头的汇总 Excel
pd.DataFrame(columns=["testcase_name", "Duration(us)"]).to_excel(summary_xlsx, index=False)
print(f"[INFO] Created summary: {summary_xlsx}")

# ── 读取用例列表 ──────────────────────────────────────────
cases_df = pd.read_excel(XLSX_FILE)
cases = [str(v).strip() for v in cases_df["testcase_name"] if pd.notna(v) and str(v).strip()]
total = len(cases)

if total == 0:
    print("[WARN] No cases found.")
    sys.exit(0)

print(f"Found {total} testcases")
print(f"Summary Excel: {summary_xlsx}")
print("=" * 50)


# ── 工具函数 ──────────────────────────────────────────────
def list_dirs():
    """列出当前目录一级子目录名（不含 .）"""
    return {p.name for p in SCRIPT_DIR.iterdir() if p.is_dir()}


def extract_duration(csv_path):
    """从 OpBasicInfo.csv 提取 Duration(us) 之和"""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  Error reading CSV: {e}", file=sys.stderr)
        return "N/A"

    col = "Duration(us)"
    if col not in df.columns:
        print(f"  Warning: '{col}' column not found in {csv_path}", file=sys.stderr)
        return "N/A"

    val = df[col].dropna().sum()
    if isinstance(val, float) and val == int(val):
        return int(val)
    return val


def append_excel_row(excel_path, tc_name, duration):
    """向汇总 Excel 追加一行"""
    try:
        df = pd.read_excel(excel_path)
        new_row = pd.DataFrame({"testcase_name": [tc_name], "Duration(us)": [duration]})
        pd.concat([df, new_row], ignore_index=True).to_excel(excel_path, index=False)
    except Exception as e:
        print(f"  Error updating Excel: {e}", file=sys.stderr)


# ── 主循环 ────────────────────────────────────────────────
fail_count = 0

for idx, name in enumerate(cases, 1):
    print(f"\n[{idx}/{total}] ▶ {name}")
    print("-" * 45)

    before_dirs = list_dirs()

    # 运行 msprof（失败不中断）
    env = os.environ.copy()
    env["STEM_INDEXER_TESTCASE_NAME"] = name
    result = subprocess.run(
        ["msprof", "op", "python3", "-m", "pytest", "-s", TEST_SCRIPT, "-v", "-m", "ci"],
        env=env,
        cwd=str(SCRIPT_DIR),
    )
    if result.returncode != 0:
        fail_count += 1

    # 找新增目录
    after_dirs = list_dirs()
    new_dirs = after_dirs - before_dirs
    new_dir = sorted(new_dirs)[0] if new_dirs else None

    if new_dir is None:
        print(f"  ⚠ no new msprof output directory detected for '{name}'")
        append_excel_row(summary_xlsx, name, "N/A")
        continue

    # 重命名到固定路径
    src = SCRIPT_DIR / new_dir
    target = COLLECT_DIR / f"{idx:03d}_{name}"
    try:
        shutil.move(str(src), str(target))
    except Exception:
        # 跨文件系统时 move 可能失败，用 copytree + rmtree
        shutil.copytree(str(src), str(target))
        shutil.rmtree(str(src))
    print(f"  ✓ moved: {new_dir}/ → {target}/")

    # 查找 OpBasicInfo.csv
    csv_files = list(target.rglob("OpBasicInfo.csv"))
    if not csv_files:
        print(f"  ⚠ OpBasicInfo.csv not found under {target}")
        append_excel_row(summary_xlsx, name, "N/A")
        continue

    csv_file = csv_files[0]
    print(f"  ✓ csv  : {csv_file}")

    duration = extract_duration(csv_file)
    print(f"  • Duration(us): {duration}")
    append_excel_row(summary_xlsx, name, duration)

# ── 汇总 ──────────────────────────────────────────────────
print()
print("=" * 50)
print(f"Total={total}  Failed={fail_count}")
print(f"Raw outputs  : {COLLECT_DIR}/")
print(f"Summary Excel: {summary_xlsx}")
