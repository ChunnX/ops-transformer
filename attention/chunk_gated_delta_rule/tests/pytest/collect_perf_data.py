#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import csv
import glob
import argparse


OP_NAME = "ChunkGatedDeltaRule"


def extract_op_duration(prof_folder, op_name=OP_NAME):
    prof_output = os.path.join(prof_folder, "mindstudio_profiler_output")
    if not os.path.isdir(prof_output):
        return None

    csv_files = glob.glob(os.path.join(prof_output, "op_summary*.csv"))
    if not csv_files:
        return None

    durations = []
    with open(csv_files[0], newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("OP Type") == op_name:
                val = row.get("Task Duration(us)", "")
                try:
                    durations.append(float(val))
                except (ValueError, TypeError):
                    pass
    if not durations:
        return None
    return f"{sum(durations) / len(durations):.4f}"


def find_case_dirs(prof_base_dir):
    result = []
    for name in sorted(os.listdir(prof_base_dir)):
        case_dir = os.path.join(prof_base_dir, name)
        if not os.path.isdir(case_dir):
            continue
        result.append((name, case_dir))
    return result


def find_prof_dir(case_dir):
    for name in sorted(os.listdir(case_dir)):
        if name.startswith("PROF_"):
            return os.path.join(case_dir, name)
    return None


def merge_and_fill(prof_base_dir, output_csv):
    case_entries = find_case_dirs(prof_base_dir)

    print(f"============= 开始合并用例结果与性能数据 =============")
    print(f"用例目录数: {len(case_entries)}")

    all_rows = []
    fieldnames = None

    for i, (case_name, case_dir) in enumerate(case_entries):
        result_csv = os.path.join(case_dir, "result.csv")
        if not os.path.isfile(result_csv):
            print(f"  [{i+1}] {case_name}: 未找到 result.csv, 跳过")
            continue

        with open(result_csv, newline="") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames or []
            for row in reader:
                if "durations" not in row:
                    row["durations"] = ""
                all_rows.append(row)

        prof_dir = find_prof_dir(case_dir)
        if prof_dir is None:
            print(f"  [{i+1}] {case_name}: 未找到 PROF 目录")
            continue

        duration = extract_op_duration(prof_dir)
        if duration is None:
            print(f"  [{i+1}] {case_name}: 未找到 {OP_NAME} 性能数据")
            continue

        if all_rows:
            all_rows[-1]["durations"] = duration
        print(f"  [{i+1}] {case_name}: durations={duration}us")

    if "durations" in fieldnames:
        col_idx = fieldnames.index("durations")
        fieldnames = fieldnames[:col_idx] + fieldnames[col_idx + 1:]
        fieldnames.append("durations")

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    filled = sum(1 for r in all_rows if r.get("durations"))
    print(f"\n结果已保存: {output_csv}")
    print(f"共合并 {len(all_rows)} 条用例, 其中 {filled} 条已填充性能数据")
    print(f"============= 合并完成 =============")


def main():
    parser = argparse.ArgumentParser(description="合并用例结果与 msprof 性能数据")
    parser.add_argument("--prof-dir", type=str, required=True,
                        help="用例目录路径（含 result.csv 和 PROF_ 目录）")
    parser.add_argument("--output", type=str, required=True,
                        help="输出 CSV 路径")
    args = parser.parse_args()

    if not os.path.isdir(args.prof_dir):
        print(f"错误: 目录不存在: {args.prof_dir}")
        exit(1)

    merge_and_fill(args.prof_dir, args.output)


if __name__ == "__main__":
    main()
