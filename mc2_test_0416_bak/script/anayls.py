import os
import pandas as pd

def result_analysis(op_name):
    df_total = pd.DataFrame(columns=["算子名称", "版本信息", "版本标识", "芯片型号", "竞品类型",
                               "总用例数", "总样本量", "执行成功数", "功能失败数", "PASS用例数",
                               "WARNING用例数", "ERROR用例数", "详细信息"])

    df_single_case = pd.DataFrame(columns=["算子类型", "用例", "版本信息", "版本标识", "芯片型号",
                                           "竞品类型", "统计数量", "测试结果", "错误信息", "INF/NAN数量",
                                           "INF/NAN错误数量", "逐点绝对误差通过数", "最大ULP误差比", "平均ULP误差比",
                                           "最大相对误差比", "平均相对误差比", "均方根误差比", "小值域误差比", "逐点绝对误差通过率"])

    df_histogram = pd.DataFrame(columns=["算子名", "版本号", "数据类型",
                                         "MR 0~0.2", "MR 0.2~0.4", "MR 0.4~0.6", "MR 0.6~0.8",
                                         "MR 0.8~1.0", "MR 1.0~1.2", "MR 1.2~1.4", "MR 1.4~1.6",
                                         "MR 1.6~1.8", "MR 1.8~2.0", "MR >2.0",
                                         "ME 0~0.2", "ME 0.2~0.4", "ME 0.4~0.6", "ME 0.6~0.8",
                                         "ME 0.8~1.0", "ME 1.0~1.2", "ME 1.2~1.4", "ME 1.4~1.6",
                                         "ME 1.6~1.8", "ME 1.8~2.0", "ME >2.0",
                                         "RMSE 0~0.2", "RMSE 0.2~0.4", "RMSE 0.4~0.6", "RMSE 0.6~0.8",
                                         "RMSE 0.8~1.0", "RMSE 1.0~1.2", "RMSE 1.2~1.4", "RMSE 1.4~1.6",
                                         "RMSE 1.6~1.8", "RMSE 1.8~2.0", "RMSE >2.0"])

    # 写df_single_case
    inter_xlsx_loc = f"test_result/{op_name}.xlsx"

    col_indices = {
            '算子类型': ('算子\n类型',),
            '用例': ('用例', ),
            '统计数量': ('NPU VS GPU', '统计\n数量'),
            '测试结果': ('测试结果', ),
            '错误信息': ('错误信息', ),
            'INF/NAN数量': ('NPU VS GPU', 'INF/NAN数量'),
            'INF/NAN错误数量': ('NPU VS GPU', 'INF/NAN错误数量'),
            '逐点绝对误差通过数': ('NPU VS GPU', '逐点绝对误差通过数量'),
            '最大ULP误差比': ('NPU VS GPU', '最大abs(ULP)误差比值'),
            '平均ULP误差比': ('NPU VS GPU', 'abs(ULP)平均误差比值'),
            '最大相对误差比': ('NPU VS GPU', '最大相对误差比值'),
            '平均相对误差比': ('NPU VS GPU', '平均相对误差比值'),
            '均方根误差比': ('NPU VS GPU', '均方根误差比值'),
            '小值域误差比': ('NPU VS GPU', '小值域绝对误差比值'),
            '逐点绝对误差通过率': ('NPU VS GPU', '逐点绝对误差通过率')
        }
    case_num = 0
    fun_fail_sum = 0
    pass_count_sum = 0
    warning_count_sum = 0
    error_count_sum = 0
    value_count_sum = 0

    with pd.ExcelFile(inter_xlsx_loc) as xls:
        sheet_names = xls.sheet_names
        for name in sheet_names:
            df = pd.read_excel(inter_xlsx_loc, sheet_name=name, header=[0, 1, 2])

            dtype = name.split("_")[0]

            if not isinstance(df.columns, pd.MultiIndex):
                raise TypeError(f"【{name}】 DataFrame的列索引不是多层索引(MultiIndex)，无法匹配提供的元组索引")

            missing_columns = []
            for col_key, col_tuple in col_indices.items():
                if col_tuple not in df.columns:
                    tuple_str = "('" + "', '".join(col_tuple) + "')"
                    missing_columns.append(f"{col_key} -> {tuple_str}")
            if missing_columns:
                err_msg = f"【{name}】 以下多层列索引不存在于DataFrame中：\n  " + "\n  ".join(missing_columns)
                raise ValueError(err_msg)

            print(f"【{name}】 列名有效性检查: Success.")

            op_type = df[col_indices['算子类型']].squeeze().tolist() if isinstance(
                df[col_indices['算子类型']].squeeze(), pd.Series) else [df[col_indices['算子类型']].squeeze()]
            case_name = df[col_indices['用例']].squeeze().tolist() if isinstance(
                df[col_indices['用例']].squeeze(), pd.Series) else [df[col_indices['用例']].squeeze()]

            value_count = df[col_indices['统计数量']].squeeze().tolist() if isinstance(
                df[col_indices['统计数量']].squeeze(), pd.Series) else [df[col_indices['统计数量']].squeeze()]

            test_result = df[col_indices['测试结果']].squeeze().tolist() if isinstance(
                df[col_indices['测试结果']].squeeze(), pd.Series) else [df[col_indices['测试结果']].squeeze()]

            err_info = df[col_indices['错误信息']].squeeze().tolist() if isinstance(
                df[col_indices['错误信息']].squeeze(), pd.Series) else [df[col_indices['错误信息']].squeeze()]

            infnan_count = df[col_indices['INF/NAN数量']].squeeze().tolist() if isinstance(
                df[col_indices['INF/NAN数量']].squeeze(), pd.Series) else [df[col_indices['INF/NAN数量']].squeeze()]

            infnan_err_count = df[col_indices['INF/NAN错误数量']].squeeze().tolist() if isinstance(
                df[col_indices['INF/NAN错误数量']].squeeze(), pd.Series) else [df[col_indices['INF/NAN错误数量']].squeeze()]

            abs_err_pass = df[col_indices['逐点绝对误差通过数']].squeeze().tolist() if isinstance(
                df[col_indices['逐点绝对误差通过数']].squeeze(), pd.Series) else [df[col_indices['逐点绝对误差通过数']].squeeze()]

            max_ulp = df[col_indices['最大ULP误差比']].squeeze().tolist() if isinstance(
                df[col_indices['最大ULP误差比']].squeeze(), pd.Series) else [df[col_indices['最大ULP误差比']].squeeze()]

            mean_ulp = df[col_indices['平均ULP误差比']].squeeze().tolist() if isinstance(
                df[col_indices['平均ULP误差比']].squeeze(), pd.Series) else [df[col_indices['平均ULP误差比']].squeeze()]

            mr = df[col_indices['最大相对误差比']].squeeze().tolist() if isinstance(
                df[col_indices['最大相对误差比']].squeeze(), pd.Series) else [df[col_indices['最大相对误差比']].squeeze()]

            me = df[col_indices['平均相对误差比']].squeeze().tolist() if isinstance(
                df[col_indices['平均相对误差比']].squeeze(), pd.Series) else [df[col_indices['平均相对误差比']].squeeze()]

            rmse = df[col_indices['均方根误差比']].squeeze().tolist() if isinstance(
                df[col_indices['均方根误差比']].squeeze(), pd.Series) else [df[col_indices['均方根误差比']].squeeze()]

            small_value = df[col_indices['小值域误差比']].squeeze().tolist() if isinstance(
                df[col_indices['小值域误差比']].squeeze(), pd.Series) else [df[col_indices['小值域误差比']].squeeze()]

            abs_err_pass_ratio = df[col_indices['逐点绝对误差通过率']].squeeze().tolist() if isinstance(
                df[col_indices['逐点绝对误差通过率']].squeeze(), pd.Series) else [df[col_indices['逐点绝对误差通过率']].squeeze()]
            value_count_sum += sum(
                int(x) for x in value_count 
                if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))
            )
            case_num += len(test_result)
            fun_fail_sum += sum(
                1 if isinstance(x, (int, float)) and math.isnan(x) else 0
                for x in test_result
            )
            pass_count_sum += sum([1 if x == "PASS" else 0 for x in test_result])
            warning_count_sum += sum([1 if x == "WARNING" else 0 for x in test_result])
            error_count_sum += sum([1 if x == "ERROR" else 0 for x in test_result])

            for i in range(len(op_type)):
                idx = len(df_single_case)
                df_single_case.loc[idx, "算子类型"] = op_name
                df_single_case.loc[idx, "用例"] = case_name[i]
                df_single_case.loc[idx, "统计数量"] = value_count[i]
                df_single_case.loc[idx, "测试结果"] = test_result[i]
                df_single_case.loc[idx, "错误信息"] = err_info[i]
                df_single_case.loc[idx, "INF/NAN数量"] = infnan_count[i]
                df_single_case.loc[idx, "INF/NAN错误数量"] = infnan_err_count[i]
                df_single_case.loc[idx, "逐点绝对误差通过数"] = abs_err_pass[i]
                df_single_case.loc[idx, "最大ULP误差比"] = max_ulp[i]
                df_single_case.loc[idx, "平均ULP误差比"] = mean_ulp[i]
                df_single_case.loc[idx, "最大相对误差比"] = mr[i]
                df_single_case.loc[idx, "平均相对误差比"] = me[i]
                df_single_case.loc[idx, "均方根误差比"] = rmse[i]
                df_single_case.loc[idx, "小值域误差比"] = small_value[i]
                df_single_case.loc[idx, "逐点绝对误差通过率"] = abs_err_pass_ratio[i]

            mr_csv_file = f"test_result/{op_type[0]}/{op_type[0]}_{dtype}_ULP标杆对比_MRE.csv"
            me_csv_file = f"test_result/{op_type[0]}/{op_type[0]}_{dtype}_ULP标杆对比_ME.csv"
            rmse_csv_file = f"test_result/{op_type[0]}/{op_type[0]}_{dtype}_ULP标杆对比_RMSE.csv"

            df_mr = pd.read_csv(mr_csv_file)
            df_me = pd.read_csv(me_csv_file)
            df_rmse = pd.read_csv(rmse_csv_file)

            print(f"{dtype} >>> df_mr", df_mr)
            print(f"{dtype} >>> df_me", df_me)
            print(f"{dtype} >>> df_rmse", df_rmse)

            idx_histogram = len(df_histogram)
            df_histogram.loc[idx_histogram,"算子名"] = op_type[0]
            df_histogram.loc[idx_histogram,"数据类型"] = dtype
            df_histogram.loc[idx_histogram,"MR 0~0.2"] = df_mr["0~0.2"][0]
            df_histogram.loc[idx_histogram,"MR 0.2~0.4"] = df_mr["0.2~0.4"][0]
            df_histogram.loc[idx_histogram,"MR 0.4~0.6"] = df_mr["0.4~0.6"][0]
            df_histogram.loc[idx_histogram,"MR 0.6~0.8"] = df_mr["0.6~0.8"][0]
            df_histogram.loc[idx_histogram,"MR 0.8~1.0"] = df_mr["0.8~1.0"][0]
            df_histogram.loc[idx_histogram,"MR 1.0~1.2"] = df_mr["1.0~1.2"][0]
            df_histogram.loc[idx_histogram,"MR 1.2~1.4"] = df_mr["1.2~1.4"][0]
            df_histogram.loc[idx_histogram,"MR 1.4~1.6"] = df_mr["1.4~1.6"][0]
            df_histogram.loc[idx_histogram,"MR 1.6~1.8"] = df_mr["1.6~1.8"][0]
            df_histogram.loc[idx_histogram,"MR 1.8~2.0"] = df_mr["1.8~2.0"][0]
            df_histogram.loc[idx_histogram,"MR >2.0"] = df_mr[">2.0"][0]
            df_histogram.loc[idx_histogram,"ME 0~0.2"] = df_me["0~0.2"][0]
            df_histogram.loc[idx_histogram,"ME 0.2~0.4"] = df_me["0.2~0.4"][0]
            df_histogram.loc[idx_histogram,"ME 0.4~0.6"] = df_me["0.4~0.6"][0]
            df_histogram.loc[idx_histogram,"ME 0.6~0.8"] = df_me["0.6~0.8"][0]
            df_histogram.loc[idx_histogram,"ME 0.8~1.0"] = df_me["0.8~1.0"][0]
            df_histogram.loc[idx_histogram,"ME 1.0~1.2"] = df_me["1.0~1.2"][0]
            df_histogram.loc[idx_histogram,"ME 1.2~1.4"] = df_me["1.2~1.4"][0]
            df_histogram.loc[idx_histogram,"ME 1.4~1.6"] = df_me["1.4~1.6"][0]
            df_histogram.loc[idx_histogram,"ME 1.6~1.8"] = df_me["1.6~1.8"][0]
            df_histogram.loc[idx_histogram,"ME 1.8~2.0"] = df_me["1.8~2.0"][0]
            df_histogram.loc[idx_histogram,"ME >2.0"] = df_me[">2.0"][0]
            df_histogram.loc[idx_histogram,"RMSE 0~0.2"] = df_rmse["0~0.2"][0]
            df_histogram.loc[idx_histogram,"RMSE 0.2~0.4"] = df_rmse["0.2~0.4"][0]
            df_histogram.loc[idx_histogram,"RMSE 0.4~0.6"] = df_rmse["0.4~0.6"][0]
            df_histogram.loc[idx_histogram,"RMSE 0.6~0.8"] = df_rmse["0.6~0.8"][0]
            df_histogram.loc[idx_histogram,"RMSE 0.8~1.0"] = df_rmse["0.8~1.0"][0]
            df_histogram.loc[idx_histogram,"RMSE 1.0~1.2"] = df_rmse["1.0~1.2"][0]
            df_histogram.loc[idx_histogram,"RMSE 1.2~1.4"] = df_rmse["1.2~1.4"][0]
            df_histogram.loc[idx_histogram,"RMSE 1.4~1.6"] = df_rmse["1.4~1.6"][0]
            df_histogram.loc[idx_histogram,"RMSE 1.6~1.8"] = df_rmse["1.6~1.8"][0]
            df_histogram.loc[idx_histogram,"RMSE 1.8~2.0"] = df_rmse["1.8~2.0"][0]
            df_histogram.loc[idx_histogram,"RMSE >2.0"] = df_rmse[">2.0"][0]

            op_name = op_type[0]
            
        df_total.loc[0, "算子名称"] = op_name
        df_total.loc[0, "芯片型号"] = ""
        df_total.loc[0, "竞品类型"] = ""
        df_total.loc[0, "总用例数"] = case_num
        df_total.loc[0, "总样本量"] = value_count_sum
        df_total.loc[0, "执行成功数"] = case_num - fun_fail_sum
        df_total.loc[0, "功能失败数"] = fun_fail_sum
        df_total.loc[0, "PASS用例数"] = pass_count_sum
        df_total.loc[0, "WARNING用例数"] = warning_count_sum
        df_total.loc[0, "ERROR用例数"] = error_count_sum
        df_total.loc[0, "详细信息"] = ""
   
    os.system("mkdir -p result")
    with pd.ExcelWriter(f'result/{op_name}_result.xlsx', engine='openpyxl') as writer:
        df_total.to_excel(writer, sheet_name="总体情况", index=False)
        df_single_case.to_excel(writer, sheet_name="单用例情况", index=False)
        df_histogram.to_excel(writer, sheet_name="直方图", index=False)
if __name__ == '__main__':
    result_analysis("MC2")