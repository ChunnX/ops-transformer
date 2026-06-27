#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import argparse


def extract_kvquant_row(prof_folder):
    prof_output = os.path.join(prof_folder, "mindstudio_profiler_output")
    if not os.path.isdir(prof_output):
        return None

    csv_files = glob.glob(os.path.join(prof_output, "op_summary*.csv"))
    if not csv_files:
        return None

    df = pd.read_csv(csv_files[0])
    target = df[df['Op Name'] == 'KvQuantSparseAttnSharedkv']
    if target.empty:
        return None

    row = target.iloc[0].to_dict()
    return row


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

    perf_rows = {}
    prof_idx = 0

    for i in range(df_b.shape[0]):
        if not valid_mask.iloc[i]:
            continue

        if prof_idx >= len(prof_folders):
            print(f"  [{i}] PROF文件夹不足, 跳过剩余用例")
            break

        prof = prof_folders[prof_idx]
        case_name = df_b.iloc[i]["case_name"]
        row_data = extract_kvquant_row(prof)

        if row_data is not None:
            perf_rows[i] = row_data
            print(f"  [{prof_idx+1}] {case_name} -> {prof} (Task Duration: {row_data['Task Duration(us)']}us)")
        else:
            print(f"  [{prof_idx+1}] {case_name}: 未找到KvQuantSparseAttnSharedkv数据")

        prof_idx += 1

    if not perf_rows:
        print("未收集到任何性能数据")
        return

    perf_df = pd.DataFrame.from_dict(perf_rows, orient='index')
    overlap = set(df_b.columns) & set(perf_df.columns)
    if overlap:
        rename_map = {col: f"op_{col}" for col in overlap}
        perf_df = perf_df.rename(columns=rename_map)
        print(f"op_summary列名冲突已重命名: {list(rename_map.values())}")

    df_b = pd.concat([df_b, perf_df], axis=1)

    if args.is_compare:
        try:
            df_c = pd.read_excel(args.perf_golden_path)
        except Exception as e:
            print(f"读取基线数据失败: {e}")
            args.is_compare = False

    if args.is_compare:
        perf_threshold = 10
        perf_fail_list = []
        df_b["sas_perf_diff"] = ""
        df_b["perf_result"] = ""

        for i in perf_rows.keys():
            try:
                cur_sas = df_b.at[i, "Task Duration(us)"]
                golden_sas = df_c.iloc[i]["Task Duration(us)"]
                sas_diff = float(cur_sas) - float(golden_sas)
                df_b.at[i, "sas_perf_diff"] = sas_diff

                if abs(sas_diff) > perf_threshold:
                    df_b.at[i, "perf_result"] = "Failed"
                    perf_fail_list.append(df_b.iloc[i]["case_name"])
                else:
                    df_b.at[i, "perf_result"] = "Pass"
            except Exception as e:
                print(f"  基线对比出错 (行{i}): {e}")

        if perf_fail_list:
            print(f"性能不达标用例: {perf_fail_list}")

    new_path = args.test_result_path.replace(".xlsx", "_perf.xlsx")
    if os.path.exists(new_path):
        os.remove(new_path)
    df_b.to_excel(new_path, index=False)

    print(f"\n性能数据已保存: {new_path}")
    print(f"共拼接 {len(perf_rows)} 条用例的 op_summary 全字段数据")
    print(f"============= 性能数据收集完成 =============")


if __name__ == "__main__":
    main()
