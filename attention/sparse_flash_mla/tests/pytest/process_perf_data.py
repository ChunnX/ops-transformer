#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Merge msprof timings into a CSV result for sparse_flash_mla."""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


OP_TYPES = {
    "SparseFlashMlaMetadata",
    "SparseFlashMla",
    "SparseFlashMLAMetadata",
    "SparseFlashMLA",
}


def collect_operator_timings(summary, expected_count):
    type_column = next(
        (name for name in ("OP Type", "Op Type") if name in summary), None
    )
    if type_column is None:
        raise ValueError("op_summary缺少OP Type列")

    summary = summary[summary[type_column].isin(OP_TYPES)]
    if "Task Start Time(us)" in summary:
        summary = summary.sort_values("Task Start Time(us)", kind="stable")

    pairs = []
    metadata_duration = None
    for _, row in summary.iterrows():
        duration = float(row["Task Duration(us)"])
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError(f"无效的算子耗时: {duration}")
        op_type = str(row[type_column])
        if "Metadata" in op_type:
            metadata_duration = duration
        else:
            if metadata_duration is None:
                raise ValueError("attention算子缺少对应的metadata性能数据")
            pairs.append((metadata_duration, duration))
            metadata_duration = None

    if len(pairs) != expected_count:
        raise ValueError(
            f"性能数据与成功用例数不匹配: {len(pairs)}组算子, {expected_count}个用例"
        )
    return pairs


def find_profiler_summary():
    prof_folders = [
        item
        for item in os.listdir()
        if os.path.isdir(item) and item.startswith("PROF")
    ]
    if not prof_folders:
        raise FileNotFoundError("PROF文件夹不存在")
    prof_folder = max(prof_folders, key=os.path.getmtime)
    output_path = Path(prof_folder) / "mindstudio_profiler_output"
    summaries = list(output_path.glob("op_summary*.csv"))
    if not summaries:
        raise FileNotFoundError(f"{output_path}中未找到op_summary*.csv")
    return prof_folder, max(summaries, key=os.path.getmtime)


def process_profiler_data(
    test_result_path, csv_result_path, delete_perf_file=False
):
    result_path = Path(test_result_path)
    if not result_path.exists():
        raise FileNotFoundError(f"测试结果不存在: {result_path}")
    df_result = pd.read_excel(result_path)
    successful = ~df_result["result"].isin(["NPU ERROR", "UNSUPPORTED"])
    if not successful.any():
        raise ValueError("没有成功执行的NPU用例，无法生成性能数据")

    prof_folder, summary_path = find_profiler_summary()
    summary = pd.read_csv(summary_path)
    timings = collect_operator_timings(summary, int(successful.sum()))

    df_result["metadata_duration_us"] = np.nan
    df_result["smla_duration_us"] = np.nan
    row_idx = 0
    for index in range(len(df_result)):
        if successful.iloc[index]:
            metadata_duration, smla_duration = timings[row_idx]
            df_result.at[index, "metadata_duration_us"] = metadata_duration
            df_result.at[index, "smla_duration_us"] = smla_duration
            row_idx += 1
    df_result["perf_result"] = np.where(successful, "Executed", df_result["result"])
    output_path = Path(csv_result_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_result.to_csv(output_path, index=False)
    print(f"性能CSV已保存: {output_path}")

    if delete_perf_file:
        shutil.rmtree(prof_folder, ignore_errors=True)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="处理 sparse_flash_mla 性能数据")
    parser.add_argument("--test_result_path", default="result.xlsx")
    parser.add_argument("--csv_result_path", default="result_perf.csv")
    parser.add_argument("--delete_perf_file", action="store_true")
    args = parser.parse_args()
    process_profiler_data(
        args.test_result_path,
        args.csv_result_path,
        args.delete_perf_file,
    )


if __name__ == "__main__":
    main()
