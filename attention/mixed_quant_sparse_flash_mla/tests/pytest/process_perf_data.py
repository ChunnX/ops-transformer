#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import os
import glob
import pandas as pd
from pathlib import Path
import numpy as np
import argparse
import shutil


def collect_operator_timings(summary, expected_count):
    """Match completed metadata/attention pairs, ignoring orphan metadata calls."""
    type_column = next(
        (name for name in ("OP Type", "Op Type") if name in summary), None
    )
    if type_column is None:
        raise ValueError("op_summary缺少OP Type列，无法匹配算子性能")
    summary = summary[
        summary[type_column].isin(
            ["MixedQuantSparseFlashMlaMetadata", "MixedQuantSparseFlashMla"]
        )
    ]
    if "Task Start Time(us)" in summary:
        summary = summary.sort_values("Task Start Time(us)", kind="stable")
    pairs = []
    metadata_duration = None
    for _, row in summary.iterrows():
        duration = float(row["Task Duration(us)"])
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError(f"无效的算子耗时: {duration}")
        if row[type_column] == "MixedQuantSparseFlashMlaMetadata":
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


def process_profiler_data(args):
    """
    读取性能数据写入test result表格
    """
    print("=============开始处理性能分析数据=============")

    # 检查是否存在PROF开头的文件夹
    prof_folders = [
        f for f in os.listdir() if os.path.isdir(f) and f.startswith("PROF")
    ]

    if not prof_folders:
        print("PROF文件夹不存在")
        return

    prof_folder = max(prof_folders, key=os.path.getmtime)
    print(f"PROF path: {prof_folder}")

    profiler_output_path = os.path.join(prof_folder, "mindstudio_profiler_output")
    if not os.path.exists(profiler_output_path) or not os.path.isdir(
        profiler_output_path
    ):
        print(f"在 {prof_folder} 中不存在mindstudio_profiler_output文件夹")
        return

    # 检查是否存在op_summary开头的表格
    op_summary_files = []
    for ext in [".csv"]:
        pattern = os.path.join(profiler_output_path, f"op_summary*{ext}")
        op_summary_files.extend(glob.glob(pattern))

    if not op_summary_files:
        print(f"在 {profiler_output_path} 中未找到 op_summary 开头的表格文件，直接返回")
        return

    op_summary_file = max(op_summary_files, key=os.path.getmtime)
    print(f"从 {op_summary_file}解析性能数据")

    # 读取表格A (op_summary)
    try:
        if op_summary_file.endswith(".csv"):
            df_a = pd.read_csv(op_summary_file)

    except Exception as e:
        print(f"读取表格A失败: {e}")
        return

    # 读取表格B (result表格)
    try:
        df_b = pd.read_excel(args.test_result_path)

    except Exception as e:
        print(f"读取表格B失败: {e}")
        return

    successful = df_b["result"].ne("NPU ERROR")
    if not successful.any():
        print("没有成功执行的NPU用例，无法生成性能数据。请检查res.log中的NPU ERROR。")
        return
    try:
        timings = collect_operator_timings(df_a, int(successful.sum()))
    except (ValueError, KeyError) as error:
        print(f"无法匹配算子性能: {error}")
        return

    # 读取表格C (duration 基线数据表格)
    if args.is_compare:
        try:
            df_c = pd.read_excel(args.perf_golden_path)

        except Exception as e:
            print(f"读取表格C失败: {e}")
            return

    # 抓取durfation数据存入表格B
    perf_fail_list = []
    perf_threshold = 10
    row_idx = 0
    try:
        df_b["metadata_duration"] = np.nan
        df_b["qsmla_duration"] = np.nan
        df_b["metadata_perf_diff"] = np.nan
        df_b["qsmla_perf_diff"] = np.nan
        df_b["perf_result"] = np.where(successful, "Not compared", "NPU ERROR")

        for i in range(df_b.shape[0]):
            if df_b.iloc[i]["result"] != "NPU ERROR":  # NPU ERROR时跳过
                cur_metadata_perf, cur_qsmla_perf = timings[row_idx]
                df_b.at[i, "metadata_duration"] = cur_metadata_perf
                df_b.at[i, "qsmla_duration"] = cur_qsmla_perf

                if args.is_compare:
                    golden_metadata_perf = df_c.iloc[i]["metadata_duration"]
                    golden_qsmla_perf = df_c.iloc[i]["qsmla_duration"]
                    metadata_perf_diff = cur_metadata_perf - golden_metadata_perf
                    qsmla_perf_diff = cur_qsmla_perf - golden_qsmla_perf
                    df_b.at[i, "metadata_perf_diff"] = metadata_perf_diff
                    df_b.at[i, "qsmla_perf_diff"] = qsmla_perf_diff

                    if abs(qsmla_perf_diff) > perf_threshold:
                        df_b.at[i, "perf_result"] = "Failed"
                        perf_fail_list.append(df_b.iloc[i]["Testcase_Name"])

                    else:
                        df_b.at[i, "perf_result"] = "Pass"

                row_idx += 1

            if i % 10 == 0:
                print(f"已处理 {i}/{df_b.shape[0]} 行")

        print(
            f"数据合并完成，为表格B添加了 {df_b.shape[0]} 行的 metadata_duration 和 qsmla_duration 数据"
        )

    except Exception as e:
        print(f"数据合并过程中出错: {e}")
        import traceback

        traceback.print_exc()
        return

    # 保存处理后的表格B
    new_result_path = args.test_result_path.replace(".xlsx", "_perf.xlsx")
    if os.path.exists(new_result_path):
        try:
            os.remove(new_result_path)
            print("发现历史result_perf文件，删除成功")
        except Exception as e:
            print(f"历史result_perf文件删除失败: {e}")
    try:
        df_b.to_excel(new_result_path, index=False)
        print(f"处理后的表格已保存为: {new_result_path}")

    except Exception as e:
        print(f"保存文件失败: {e}")
        return

    print(
        "=================处理完成！性能失败用例如下=================\n", perf_fail_list
    )

    prof_folder = Path(prof_folder)
    if args.delete_perf_file and prof_folder.exists() and prof_folder.is_dir():
        try:
            shutil.rmtree(prof_folder)
            print(f"成功删除文件夹及其内容: {prof_folder}")
        except OSError as e:
            print("删除PROF文件夹失败：", e)

    return new_result_path


# 命令行接口
def main():
    parser = argparse.ArgumentParser(description="处理性能数据")
    parser.add_argument(
        "--test_result_path",
        type=str,
        default="result.xlsx",
        help="结果表格路径",
        required=False,
    )
    parser.add_argument(
        "--is_compare",
        type=bool,
        default=False,
        help="是否与基线表格对比",
        required=False,
    )
    parser.add_argument(
        "--perf_golden_path",
        type=str,
        default="perf_golden.xlsx",
        help="性能基线数据表格路径",
        required=False,
    )
    parser.add_argument(
        "--delete_perf_file",
        type=bool,
        default=False,
        help="是否删除PROF文件夹内容",
        required=False,
    )
    args = parser.parse_args()

    if process_profiler_data(args) is None:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
