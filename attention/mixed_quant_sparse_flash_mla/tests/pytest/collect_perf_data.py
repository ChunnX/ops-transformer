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

    print(f"============= 开始收集性能数据 =============")
    print(f"有效用例数: {valid_count}, PROF文件夹数: {len(prof_folders)}")

    if len(prof_folders) == 0:
        print("未找到PROF文件夹, 跳过性能数据收集")
        return

    if len(prof_folders) != valid_count:
        print(f"警告: PROF文件夹数量({len(prof_folders)})与有效用例数({valid_count})不一致")

    task_duration_col = "Task Duration(us)"
    df_b["metadata_duration"] = ""
    df_b["qsmla_duration"] = ""
    df_b["metadata_perf_diff"] = ""
    df_b["qsmla_perf_diff"] = ""
    df_b["perf_result"] = ""

    perf_rows = {}
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

        if prof_idx >= len(prof_folders):
            print(f"  [{i}] PROF文件夹不足, 跳过剩余用例")
            break

        prof = prof_folders[prof_idx]
        case_name = df_b.iloc[i]["Testcase_Name"]
        rows_data = extract_qsmla_rows(prof)

        if rows_data is not None:
            metadata_row = rows_data["metadata"]
            qsmla_row = rows_data["qsmla"]

            cur_metadata_perf = metadata_row[task_duration_col] if metadata_row else ""
            cur_qsmla_perf = qsmla_row[task_duration_col] if qsmla_row else ""
            df_b.at[i, "metadata_duration"] = cur_metadata_perf
            df_b.at[i, "qsmla_duration"] = cur_qsmla_perf

            if args.is_compare:
                golden_metadata_perf = df_c.iloc[i]["metadata_duration"]
                golden_qsmla_perf = df_c.iloc[i]["qsmla_duration"]
                metadata_perf_diff = float(cur_metadata_perf) - float(golden_metadata_perf)
                qsmla_perf_diff = float(cur_qsmla_perf) - float(golden_qsmla_perf)
                df_b.at[i, "metadata_perf_diff"] = metadata_perf_diff
                df_b.at[i, "qsmla_perf_diff"] = qsmla_perf_diff

                if abs(qsmla_perf_diff) > perf_threshold:
                    df_b.at[i, "perf_result"] = "Failed"
                    perf_fail_list.append(case_name)
                else:
                    df_b.at[i, "perf_result"] = "Pass"

            if metadata_row and qsmla_row:
                print(f"  [{prof_idx+1}] {case_name} -> {prof} "
                      f"(metadata: {cur_metadata_perf}us, qsmla: {cur_qsmla_perf}us)")
            else:
                print(f"  [{prof_idx+1}] {case_name} -> {prof} (部分op数据缺失)")
        else:
            print(f"  [{prof_idx+1}] {case_name}: 未找到{METADATA_OP_NAME}/{QSMLA_OP_NAME}数据")

        prof_idx += 1

    new_path = args.test_result_path.replace(".xlsx", "_perf.xlsx")
    if os.path.exists(new_path):
        os.remove(new_path)
    df_b.to_excel(new_path, index=False)

    print(f"\n性能数据已保存: {new_path}")
    print(f"共收集 {prof_idx} 条用例的 metadata_duration 和 qsmla_duration 数据")
    if args.is_compare and perf_fail_list:
        print(f"性能不达标用例: {perf_fail_list}")
    print(f"============= 性能数据收集完成 =============")


if __name__ == "__main__":
    main()
