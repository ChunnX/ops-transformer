import os
import sys
import datetime
import torch
import numpy as np
import pandas as pd
from einops import rearrange

sys.path.append(os.path.dirname(__file__))


FP16 = ["fp16", "float16"]
BF16 = ["bf16", "bfloat16"]
FP32 = ["fp32", "float32", "float"]
FP8_E4M3 = ["fp8_e4m3", "fp8_e4m3fn", "float8_e4m3", "float8_e4m3fn"]
FP8_E5M2 = ["fp8_e5m2", "float8_e5m2", "fp8"]

def log(text, file_name):
    print(text)
    log_file = open(file_name, 'a')
    # log_file.write(str(datetime.datetime.now()) + '  ')
    log_file.write(text)
    log_file.write("\n")
    log_file.close()


def print_log(data=None, file_name='auto_run.txt', level="INFO"):
    """
    print log
    :param data: value to print
    :param level: log level, include DEBUG, INFO(default), WARN and ERROR
    :return:
    """
    # level = "INFO"
    log_info = ("%s - [%s]-%s:%s - %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3], level,
                                          os.path.basename(sys._getframe().f_back.f_code.co_filename),
                                          str(sys._getframe().f_back.f_lineno).zfill(4), data))
    log(log_info, file_name)


def get_col_index(table, col_name):
    col_index = None
    # print table
    for i in range(1, table.max_column + 1):
        if table.cell(row=1, column=i).value == col_name:
            col_index = i
            break
    return col_index


def write_result(result_xlsx_name, args, case_name, str):
    df = pd.read_excel(result_xlsx_name, sheet_name=args.sheet)
    df.loc[df['Testcase_Name'] == case_name, 'running_status'] = str
    df.to_excel(result_xlsx_name, sheet_name=args.sheet, index=False)


def trans_bnsd_to_layout(tensor, input_layout):
    if tensor is None:
        return None
    else:
        if input_layout == "BSH":
            # (B,N,S,D) -> (B,S,N*D)
            return rearrange(tensor.clone(), 'b n s d -> b s (n d)')
        elif input_layout == "SBH":
            # (B,N,S,D) -> (S,B,N*D)
            return rearrange(tensor.clone(), 'b n s d -> s b (n d)')
        elif input_layout == "BSND":
            # (B,N,S,D) -> (B,S,N,D)
            return rearrange(tensor.clone(), 'b n s d -> b s n d')
        elif input_layout == "TND":
            # (1,N,S,D) => (S,N,D)
            return rearrange(tensor.clone(), '1 n s d -> s n d')
        else:
            return tensor.clone()


def checkResult(a, b, name, num_print=16):
    print_log(f"info：开始计算 {name} 精度。")
    ratio_threshold = 0.005
    threshold_diff = 0.005
    rel_error_threshold = 0.08  # 相对误差20%
    print(f"a is {a.shape}")
    print(f"b is {b.shape}")
    if a.shape == b.shape:
        a_flat = a.flatten().detach().cpu().numpy()
        b_flat = b.flatten().detach().cpu().numpy()
        diff_flat = np.abs(a_flat - b_flat)
        rel_error = diff_flat / np.maximum(np.abs(b_flat), 1e-8)

        # 找出相对误差超过20%的索引
        idx_large_error = np.where(rel_error > rel_error_threshold)[0]
        n = min(num_print, len(idx_large_error))
        totalCount = min(num_print, len(a_flat))

        print_log(f"debug: {name} 相对误差超过 {rel_error_threshold*100:.1f}% 的前 {n} 个元素：")
        print_log("        idx |      npu值       |      cpu值       |    abs_diff    |   rel_error(%)")
        print_log("       -----+------------------+------------------+----------------+----------------")
        diff_val = abs(a_flat[0] - b_flat[0]) 
        for i in range(totalCount): # TEST
            print_log(f"       {i:4d} | {a_flat[i]:14.8f} | {b_flat[i]:14.8f} | {diff_flat[i]:12.8f} | {rel_error[i]*100:12.4f}")
        for i in idx_large_error[:n]:
            print_log(f"       {i:4d} | {a_flat[i]:14.8f} | {b_flat[i]:14.8f} | {diff_flat[i]:12.8f} | {rel_error[i]*100:12.4f}")

        # 精度统计逻辑保持不变
        if torch.all(torch.eq(a, b)):
            print_log(f"info：{name} 计算结果与标杆完全相同。")
            ratio_diff = 0
            diff = 0
            return 1, 0, 0
        else:
            diff = torch.abs(a.sub(b))
            min_tensor = torch.empty(a.shape, dtype=torch.float16).to(a.device)
            min_tensor.fill_(0.000025)
            max_val = torch.max(torch.abs(a), torch.abs(b))
            threshold = torch.max(max_val.mul(ratio_threshold), min_tensor)
            mask = diff > threshold
            num_diff = torch.sum(mask)
            ratio_diff = num_diff / torch.numel(a)

            if ratio_diff > threshold_diff:
                print_log(f"error：{name} 计算结果有{num_diff}个元素的偏差超过{ratio_threshold:.2%}，占比为{ratio_diff:.2%}，超过阈值，用例执行失败！！！")
            else:
                print_log(f"info：{name} 计算结果有{num_diff}个元素的偏差超过{ratio_threshold:.2%}，占比为{ratio_diff:.2%}。")

        ratio = (1 - ratio_diff).cpu().detach().numpy()
        max_diff = torch.max(diff).cpu().detach().numpy()
        print_log(f"{name} diff_max : {max_diff:.8f}")
        sum_diff = torch.sum(diff).cpu().detach().numpy()
        print_log(f"{name} diff_sum : {sum_diff:.6f}")
    else:
        print_log(f"error: {name}计算结果错误，shape与标杆不匹配，用例执行失败！！！")
        print_log(f"debug: a {a.shape}")
        print_log(f"debug: b {b.shape}")
        ratio = 0
        max_diff = 999999
        sum_diff = 999999

    return ratio, max_diff, sum_diff

def my_select(cond, x, y):
    x_dtype = x.dtype
    if x_dtype == "float32":
        cond_int32 = cond.astype(np.int32)
        mask_not_nan = - cond_int32
        mask_nan = cond_int32 - 1
        x.dtype = np.int32
        y.dtype = np.int32
        re = np.bitwise_and(x, mask_nan)
        re_mask = np.bitwise_and(y, mask_not_nan)
        re = np.bitwise_or(re, re_mask)
    elif x_dtype == "float64":
        cond_int64 = cond.astype(np.int64)
        mask_not_nan = - cond_int64
        mask_nan = cond_int64 - 1
        x.dtype = np.int64
        y.dtype = np.int64
        re = np.bitwise_and(x, mask_nan)
        re_mask = np.bitwise_and(y, mask_not_nan)
        re = np.bitwise_or(re, re_mask)
    re.dtype = x_dtype

    return re


def _compare(goldens, actual_outputs, rtol=0.005, atol=0.000025):
    different_element_results = np.isclose(actual_outputs, goldens, rtol=rtol, atol=atol, equal_nan=True)
    different_element_indexes = np.where(different_element_results != np.array((True,)))[0]

    ae = abs(goldens - actual_outputs)  # 只有nan的结果不对
    nanfix_mask = np.logical_and(np.isnan(goldens), np.isnan(actual_outputs))
    zero = np.zeros((1), dtype=ae.dtype)
    ae = my_select(nanfix_mask, ae, zero)
    eps = 2.220446049250313e-16
    re = ae / (np.abs(goldens) + eps)
    re = my_select(nanfix_mask, re, zero)
    max_relative_error = np.max(re)
    avg_relative_error = np.mean(re)
    mid_relative_error = np.median(re)
    sum_abs_error = np.sum(ae)
    max_abs_error = np.max(ae)
    index_ae = np.where(ae == max_abs_error)
    index_re = np.where(re == max_relative_error)
    pricision = (float(goldens.size - len(different_element_indexes))) / float(goldens.size)
    pricision = format(pricision, '.8f')
    rmse = np.sqrt((np.mean((ae) ** 2)))
    rme = np.mean(ae)
    absolute_error = actual_outputs - goldens
    positive_count = torch.sum(torch.from_numpy(absolute_error > 0))
    negative_count = torch.sum(torch.from_numpy(absolute_error < 0))
    total_elements = torch.from_numpy(goldens).numel()
    eb = ((positive_count - negative_count) / total_elements).item()
    return {'max_relative_error': max_relative_error, 'avg_relative_error': avg_relative_error,
            'mid_relative_error': mid_relative_error, \
            'sum_abs_error': sum_abs_error, 'max_abs_error': max_abs_error, 'pricision': pricision, 'rmse': rmse,
            'rme': rme, 'eb': eb}


def get_kernel_time_total(file_path):
    # 读取所有kernel时间的和
    if not os.path.exists(file_path):
        print_log(f"profiler error: {file_path} not exists, retuen")
        return 0
    df = pd.read_csv(file_path)
    num_rows = len(df.index)
    total_kernel_time = 0
    for i in range(num_rows):
        kernel_time = df.iloc[i]["Duration(us)"]
        op_type = df["Type"][i]
        if op_type in ["FlashAttentionScore", "FlashAttentionScoreGrad"]:
            total_kernel_time += kernel_time
    return total_kernel_time / 1000

def get_kernel_time_forward(file_path):
    #读取FA正向kernel时间
    if not os.path.exists(file_path):
        print_log(f"profiler error: {file_path} not exists, retuen")
        return 0
    df = pd.read_csv(file_path)
    num_rows = len(df.index)
    forward_kernel_time = 0
    for i in range(num_rows):
        op_type = df["Type"][i]
        if op_type == "FlashAttentionScore":
            forward_kernel_time += df.iloc[i]["Duration(us)"]
    return forward_kernel_time/1000

def get_kernel_time_backward(file_path):
    #读取FA反向kernel时间
    if not os.path.exists(file_path):
        print_log(f"profiler error: {file_path} not exists, retuen")
        return 0
    df = pd.read_csv(file_path)
    num_rows = len(df.index)
    backward_kernel_time = 0
    for i in range(num_rows):
        op_type = df["Type"][i]
        if op_type == "FlashAttentionScoreGrad":
            backward_kernel_time += df.iloc[i]["Duration(us)"]
    return backward_kernel_time / 1000

def get_allocated_memery(file_path):
    # 读取占用内存的最大值
    if not os.path.exists(file_path):
        print_log(f"profiler error: {file_path} not exists, retuen")
        return 0
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        # 跳过第一行
        next(reader)
        # 读取第6列的数据:Allocation Total Allocated(MB)
        col_6 = [float('0' if row[5] == '' else row[5]) for row in reader]
    return max(col_6)


def copy_performance_data_fa(testcase_name="temp"):
    cur_path = os.getcwd()
    if not os.path.exists(f'{cur_path}/daily_result'):
        os.mkdir(f'{cur_path}/daily_result')
    print_log(f"####################### START copy FA core*_summary_log to {cur_path}/daily_result  #######################")
    daily_result_path = f"{cur_path}/daily_result/{testcase_name}"
    if not os.path.exists(daily_result_path):
        print_log(f"---- daily_result_path[{daily_result_path}] not exist, auto create it... ")
        os.mkdir(daily_result_path)
    fa_result_path = f"{cur_path}/daily_result/{testcase_name}/fa"
    if not os.path.exists(fa_result_path):
        print_log(f"---- fa_result_path[{fa_result_path}] not exist, auto create it... ")
        os.mkdir(fa_result_path)
    os.system(f"cp log_ca/*_summary_log '{fa_result_path}'")
    print_log(f"####################### END copy core*_summary_log to {cur_path}/daily_result  #######################")


def collect_time():
    fwd_time = 0
    bkw_time = 0
    for core_no in range(32):
        ticks_array = []
        with open(f"log_ca/core{core_no}_summary_log", 'r') as file:
            lines = file.readlines()
            for line in lines:
                if 'system total ticks : ' in line:
                    ticks = int(line[20:])
                    print_log(f"{core_no} :     {ticks}")
                    ticks_array.append(ticks)
        for i in range(len(ticks_array)):
            if i < 3:
                # 前三次为正向
                if ticks_array[i] > fwd_time:
                    fwd_time = ticks_array[i]
            else:
                # 后三次为反向
                if ticks_array[i] > bkw_time:
                    bkw_time = ticks_array[i]

    print_log(f"fwd_time: {fwd_time} ticks, {fwd_time/1650000:.3f} ms")
    # print_log(f"bkw_time: {bkw_time} ticks, {bkw_time/1650000:.3f} ms")
    return fwd_time, bkw_time
