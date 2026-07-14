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
import argparse


METADATA_OP_NAME = "MixedQuantSparseFlashMlaMetadata"
QSMLA_OP_NAME = "MixedQuantSparseFlashMla"


def extract_qsmla_rows(prof_folder):
    prof_output = os.path.join(prof_folder, "mindstudio_profiler_output")
    if not os.path.isdir(prof_output):
        return None

    csv_files = glob.glob(os.path.join(prof_output, "op_summary*.csv"))
    if not csv_files:
        return None

    df = pd.read_csv(csv_files[0])
    metadata_target = df[df['Op Name'] == METADATA_OP_NAME]
    qsmla_target = df[df['Op Name'] == QSMLA_OP_NAME]

    metadata_row = metadata_target.iloc[0].to_dict() if not metadata_target.empty else None
    qsmla_row = qsmla_target.iloc[0].to_dict() if not qsmla_target.empty else None

    if metadata_row is None and qsmla_row is None:
        return None

    return {"metadata": metadata_row, "qsmla": qsmla_row}


def main():
    parser = argparse.ArgumentParser(description='批量收集性能数据')
    parser.add_argument('--test_result_path', type=str, default="result.xlsx")
    parser.add_argument('--is_compare', action='store_true', default=False)
    parser.add_argument('--perf_golden_path', type=str, default="perf_golden.xlsx")
    parser.add_argument('--iter_count', type=int, default=5, help='每条用例的性能采集次数, 取平均值')
    args = parser.parse_args()

    if not os.path.exists(args.test_result_path):
        print(f"结果文件不存在: {args.test_result_path}")
        return

    df_b = pd.read_excel(args.test_result_path)
    valid_mask = df_b["result"] != "NPU ERROR"
    valid_count = valid_mask.sum()

    if valid_count == 0:
        print("没有有效用例,跳过性能数据收集")
        return

    prof_folders = sorted(
        [d for d in os.listdir('.') if os.path.isdir(d) and d.startswith('PROF')],
        key=lambda x: os.path.getmtime(x)
    )

    iter_count = args.iter_count
    expected_prof_count = valid_count * iter_count

    print(f"============= 开始收集性能数据 =============")
    print(f"有效用例数: {valid_count}, 每条用例采集次数: {iter_count}, "
          f"PROF文件夹数: {len(prof_folders)}, 期望: {expected_prof_count}")

    if len(prof_folders) == 0:
        print("未找到PROF文件夹, 跳过性能数据收集")
        return

    if len(prof_folders) != expected_prof_count:
        print(f"警告: PROF文件夹数量({len(prof_folders)})与期望数量({expected_prof_count})不一致")

    task_duration_col = "Task Duration(us)"
    df_b["metadata_duration"] = ""
    df_b["qsmla_duration"] = ""
    df_b["metadata_duration_detail"] = ""
    df_b["qsmla_duration_detail"] = ""
    df_b["metadata_perf_diff"] = ""
    df_b["qsmla_perf_diff"] = ""
    df_b["perf_result"] = ""

    prof_idx = 0
    perf_threshold = 10
    perf_fail_list = []

    if args.is_compare:
        try:
            df_c = pd.read_excel(args.perf_golden_path)
        except Exception as e:
            print(f"读取基线数据失败: {e}")
            args.is_compare = False

    for i in range(df_b.shape[0]):
        if not valid_mask.iloc[i]:
            continue

        if prof_idx + iter_count > len(prof_folders):
            print(f"  [{i}] PROF文件夹不足, 跳过剩余用例")
            break

        case_name = df_b.iloc[i]["Testcase_Name"]
        case_profs = prof_folders[prof_idx:prof_idx + iter_count]
        prof_idx += iter_count

        metadata_durations = []
        qsmla_durations = []

        for prof in case_profs:
            rows_data = extract_qsmla_rows(prof)
            if rows_data is not None:
                if rows_data["metadata"]:
                    metadata_durations.append(float(rows_data["metadata"][task_duration_col]))
                if rows_data["qsmla"]:
                    qsmla_durations.append(float(rows_data["qsmla"][task_duration_col]))

        if metadata_durations or qsmla_durations:
            avg_metadata = sum(metadata_durations) / len(metadata_durations) if metadata_durations else ""
            avg_qsmla = sum(qsmla_durations) / len(qsmla_durations) if qsmla_durations else ""
            df_b.at[i, "metadata_duration"] = avg_metadata
            df_b.at[i, "qsmla_duration"] = avg_qsmla
            df_b.at[i, "metadata_duration_detail"] = str([round(d, 2) for d in metadata_durations])
            df_b.at[i, "qsmla_duration_detail"] = str([round(d, 2) for d in qsmla_durations])

            if args.is_compare and avg_qsmla != "":
                golden_metadata_perf = df_c.iloc[i]["metadata_duration"]
                golden_qsmla_perf = df_c.iloc[i]["qsmla_duration"]
                metadata_perf_diff = float(avg_metadata) - float(golden_metadata_perf)
                qsmla_perf_diff = float(avg_qsmla) - float(golden_qsmla_perf)
                df_b.at[i, "metadata_perf_diff"] = metadata_perf_diff
                df_b.at[i, "qsmla_perf_diff"] = qsmla_perf_diff

                if abs(qsmla_perf_diff) > perf_threshold:
                    df_b.at[i, "perf_result"] = "Failed"
                    perf_fail_list.append(case_name)
                else:
                    df_b.at[i, "perf_result"] = "Pass"

            detail_md = str([round(d, 2) for d in metadata_durations])
            detail_qs = str([round(d, 2) for d in qsmla_durations])
            print(f"  [{i}] {case_name} -> {len(case_profs)}次采集 "
                  f"(metadata均值: {avg_metadata}us, qsmla均值: {avg_qsmla}us)")
            print(f"       metadata详情: {detail_md}")
            print(f"       qsmla详情: {detail_qs}")
        else:
            print(f"  [{i}] {case_name}: 未找到{METADATA_OP_NAME}/{QSMLA_OP_NAME}数据")

    new_path = args.test_result_path.replace(".xlsx", "_perf.xlsx")
    if os.path.exists(new_path):
        os.remove(new_path)
    df_b.to_excel(new_path, index=False)

    print(f"\n性能数据已保存: {new_path}")
    print(f"共收集 {prof_idx // iter_count} 条用例的性能数据 (每条{iter_count}次取平均)")
    if args.is_compare and perf_fail_list:
        print(f"性能不达标用例: {perf_fail_list}")
    print(f"============= 性能数据收集完成 =============")


if __name__ == "__main__":
    main()
