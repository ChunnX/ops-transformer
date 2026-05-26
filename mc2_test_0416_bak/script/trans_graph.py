import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys, getopt

errorVal = 10e-7


def is_valid_numeric(value):
    """判断值是否为有效数值（排除字符串）"""
    if isinstance(value, (int, float, np.number)):
        return True
    if isinstance(value, str):
        try:
            float(value)
            return True
        except ValueError:
            return False
    return False


def calculate_ratio_series(series1, series2, error_val=errorVal):
    """计算两个系列的比值"""
    ratios = []
    for val1, val2 in zip(series1, series2):
        if not is_valid_numeric(val1) or not is_valid_numeric(val2):
            ratios.append(np.nan)
            continue

        try:
            num1 = float(val1)
            num2 = float(val2)
        except (TypeError, ValueError):
            ratios.append(np.nan)
            continue

        if num2 == 0:
            ratios.append((num1 / error_val) if num1 != 0 else 1)
        else:
            ratios.append(num1 / num2)
    return ratios


def plot_hist_left_open_right_closed(values, title, output_path,
                                     bins_intervals=None, round_digits=4):
    """
    使用左开右闭区间绘制无间隙柱状图（Y轴比例显示）
    :param values: 数据数组
    :param title: 图表标题
    :param output_path: 输出文件路径
    :param bins_intervals: 自定义区间列表，必须包含11个区间
    :param round_digits: 数据舍入位数
    """
    # 数据舍入处理
    if round_digits is not None:
        values = np.round(values, round_digits)

    # 默认区间设置
    if bins_intervals is None:
        bins_intervals = [
            (0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8),
            (0.8, 1.0), (1.0, 1.2), (1.2, 1.4), (1.4, 1.6),
            (1.6, 1.8), (1.8, 2.0), (2.0, np.inf)
        ]

    df = pd.DataFrame(columns=["op_name", "dtype", "0~0.2", "0.2~0.4", "0.4~0.6", "0.6~0.8", "0.8~1.0", "1.0~1.2", "1.2~1.4", "1.4~1.6", "1.6~1.8", "1.8~2.0", ">2.0"])

    csv_name = f"{'_'.join(output_path.split('_')[:-1])}.csv"
    file_name_tmp = output_path.split('/')[-1]
    op_name = file_name_tmp.split('_')[0]
    dtype = file_name_tmp.split('_')[1]
    # 验证区间数量
    if len(bins_intervals) != 11:
        raise ValueError("必须提供11个区间")

    # 生成标签
    labels = []
    for i, (lower, upper) in enumerate(bins_intervals):
        if np.isinf(upper):
            labels.append(f"> {lower:.1f}")
        else:
            labels.append(f"{lower:.1f}~{upper:.1f}")

    # 计算计数和比例
    counts = np.zeros(11)
    edges = np.zeros(12)  # 11个区间需要12个边界点

    edges[0] = bins_intervals[0][0]
    for i, (lower, upper) in enumerate(bins_intervals):
        counts[i] = np.sum((values > lower) & (values <= upper))
        edges[i + 1] = upper if not np.isinf(upper) else edges[i] + (edges[i] - edges[i - 1])
    # 计算比例（Y轴0~1)

    df.loc[len(df)] = [
        op_name, dtype,
        int(counts[0]), int(counts[1]), int(counts[2]), int(counts[3]),
        int(counts[4]), int(counts[5]), int(counts[6]), int(counts[7]),
        int(counts[8]), int(counts[9]), int(counts[10])
    ]
    print("------------------------ df: ", df)

    df.to_csv(csv_name, index=False)

    proportions = counts / np.sum(counts) if np.sum(counts) > 0 else counts
    bar_widths = np.diff(edges)  # 每个柱子的实际宽度

    # 绘制柱状图（无间隙）
    plt.figure(figsize=(12, 6))
    bars = plt.bar(edges[:-1], proportions, width=bar_widths, align='edge',
                   edgecolor='black', linewidth=0.5, color='skyblue')

    # 添加百分比标签（只显示非零值）
    for bar, prop in zip(bars, proportions):
        if prop > 0:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01,  # 添加微小偏移
                     f'{prop * 100:.1f}%',
                     ha='center', va='bottom', fontsize=9)

    # X轴设置
    plt.xticks(edges[:-1] + bar_widths / 2, labels, rotation=45, ha='right')
    plt.xlim(edges[0], edges[-1])

    # Y轴设置（比例显示）
    plt.ylim(0, 1)
    plt.yticks(np.linspace(0, 1, 11), [f"{i * 100:.0f}%" for i in np.linspace(0, 1, 11)])
    plt.ylabel("Proportion", fontsize=12)

    # 标题和参考线
    plt.title(title, fontsize=14, pad=20)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.axvline(1.0, color='red', linestyle='--', linewidth=1.5)
    plt.text(1.02, 0.95, 'Baseline: 1.0', color='red')

    # 保存图表
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    # print("---------------------------counts:",counts)
    # print("---------------------------proportions:",proportions)
    print(f"Normalized histogram saved: {output_path}")

def calculate_ratio(file_path):
    all_sheets = pd.read_excel(file_path, sheet_name=None, header=[0,1,2])

    # 遍历处理每个sheet页
    for sheet_name, df in all_sheets.items():
        print(f"\nProcessing sheet: {sheet_name}")

        # 提取Sheet页的Dtype
        Dtype = sheet_name.split('_标杆对比')[0]

        # 列索引配置
        col_indices = {
            'v': ('NPU误差情况', '相对误差分布', '最大值'),  # V列(NPU)
            'av': ('GPU误差情况', '相对误差分布', '最大值'),  # AV列(GPU)
            'm': ('NPU误差情况', '相对误差分布', '均值'),  # M列(NPU均值)
            'am': ('GPU误差情况', '相对误差分布', '均值'),  # AM列(GPU均值)
            'ad': ('NPU误差情况', '均方根\n误差'),  # AD列(NPU均方根误差)
            'bd': ('GPU误差情况', '均方根\n误差')  # BD列(GPU均方根误差)
        }

        # 检查列名是否有效：
        if not isinstance(df.columns, pd.MultiIndex):
            raise TypeError(f"【{sheet_name}】 DataFrame的列索引不是多层索引(MultiIndex)，无法匹配提供的元组索引")
        
        missing_columns = []
        for col_key, col_tuple in col_indices.items():
            if col_tuple not in df.columns:
                tuple_str = "('" + "', '".join(col_tuple) + "')"
                missing_columns.append(f"{col_key} -> {tuple_str}")
        if missing_columns:
            err_msg = f"【{sheet_name}】 以下多层列索引不存在于DataFrame中：\n  " + "\n  ".join(missing_columns)
            raise ValueError(err_msg)

        print(f"【{sheet_name}】 列名有效性检查: Success.")
        # 计算三组比值

        npu_max_err = df[col_indices['v']].tolist()
        gpu_max_err = df[col_indices['av']].tolist()

        npu_mean_err = df[col_indices['m']].tolist()
        gpu_mean_err = df[col_indices['am']].tolist()

        npu_rmse = df[col_indices['ad']].squeeze().tolist() if isinstance(df[col_indices['ad']].squeeze(), pd.Series) else [df[col_indices['ad']].squeeze()]
        gpu_rmse = df[col_indices['bd']].squeeze().tolist() if isinstance(df[col_indices['bd']].squeeze(), pd.Series) else [df[col_indices['bd']].squeeze()]

        # ratios = {
        #     'v_av': calculate_ratio_series(df[col_indices['v']].tolist(), df[col_indices['av']].tolist()),
        #     'm_am': calculate_ratio_series(df[col_indices['m']].tolist(), df[col_indices['am']].tolist()),
        #     'ad_bd': calculate_ratio_series(df[col_indices['ad']].squeeze().tolist(), df[col_indices['bd']].squeeze().tolist())
        # }
        ratios = {
            'v_av': calculate_ratio_series(npu_max_err, gpu_max_err),
            'm_am': calculate_ratio_series(npu_mean_err, gpu_mean_err),
            'ad_bd': calculate_ratio_series(npu_rmse, gpu_rmse)
        }

        # 创建结果DataFrame
        result_df = pd.DataFrame({
            '用例行号': df.index + 4,
            'V列值(NPU)': df[col_indices['v']].tolist(),
            'AV列值(GPU)': df[col_indices['av']].tolist(),
            'NPU/GPU(V/AV)': ratios['v_av'],
            'M列值(NPU均值)': df[col_indices['m']].tolist(),
            'AM列值(GPU均值)': df[col_indices['am']].tolist(),
            'NPU均值/GPU均值(M/AM)': ratios['m_am'],
            'AD列值(NPU均方根误差)': df[col_indices['ad']].squeeze().tolist(),
            'BD列值(GPU均方根误差)': df[col_indices['bd']].squeeze().tolist(),
            'NPU均方根误差/GPU均方根误差(AD/BD)': ratios['ad_bd']
        })

        # 添加用例标识
        if '用例名' in df.columns:
            result_df.insert(0, '用例名', df['用例名'])

        # 保存结果
        output_dir = os.path.dirname(file_path)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        parse_dir = os.path.join(output_dir, f"{file_name}")
        os.makedirs(parse_dir, exist_ok=True)
        output_excel = os.path.join(parse_dir, f"{file_name}_result.xlsx")
        result_df.to_excel(output_excel, index=False, float_format="%.4f")
        print(f"处理完成! 结果已保存至: {output_excel}")
        print("-------- Dump Graph --------")

        #相对误差最大值
        plot_hist_left_open_right_closed(
            np.array(ratios['v_av'])[~np.isnan(ratios['v_av'])],
            f"{file_name} NPU/GPU Maximum Relative Error Ratio distribution",
            os.path.join(parse_dir, f"{file_name}_{Dtype}_MRE_histgram.png")
        )
        #相对误差均值
        plot_hist_left_open_right_closed(
            np.array(ratios['m_am'])[~np.isnan(ratios['m_am'])],
            f"{file_name} NPU/GPU Mean Relative Error Ratio distribution",
            os.path.join(parse_dir, f"{file_name}_{Dtype}_ME_histgram.png")
        )
        #均方根误差
        plot_hist_left_open_right_closed(
            np.array(ratios['ad_bd'])[~np.isnan(ratios['ad_bd'])],
            f"{file_name} NPU/GPU Root Mean Squared Error Ratio distribution",
            os.path.join(parse_dir, f"{file_name}_{Dtype}_RMSE_histgram.png")
        )

    return True


if __name__ == "__main__":

    argv = sys.argv[1:]
    opts, ars = getopt.getopt(argv, "hi:o:", ["case_file="])
    case_file = ""
    for opt, arg in opts:
        if opt in ("--case_file"):
            case_file = arg
    print("case_file=", case_file)
    files = []
    files.append(case_file)
    print("files=", files)

    for file in files:
        if os.path.exists(file):
            print(f"\n正在处理文件: {file}")
            calculate_ratio(file)
        else:
            print(f"文件不存在: {file}")