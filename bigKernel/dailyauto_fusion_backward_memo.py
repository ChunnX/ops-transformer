# FA接口参数介绍：https://wiki.huawei.com/domains/17115/wiki/35808/WIKI202311162363573
import argparse
import os
import multiprocessing
import queue
import time
import datetime
import csv
import torch
import torch_npu
import numpy as np
import xlrd
import pandas as pd
import openpyxl
import math
import sys
import signal
from einops import rearrange
from tools.tools import *
import subprocess

# torch.npu.config.allow_internal_format=False
gtype = torch.float64  # arm host必须选择float64，x86环境选择float32即可，64也行。arm计算很慢，s=8k的场景建议使用x86
catch_saving = True  # 是否保存golden数据缓存，开启后将golden保存在auto_input实现连跑加速。！！！注意此功能将消耗大量磁盘空间！！！
catch_refresh = False  # 强制使用cpu重新运算标杆，不使用golden数据，同时打开catch_saving则会将新数据覆盖旧数据
case_time_out_threshold = 1800

all_case_dict = {
    "default_path":{"golden_path":"/npu/fa_smoke/auto_input"},
    "FlashAttentionScoreCommunity":{"golden_path":"/npu/fa_smoke/auto_input"} ,
    "FlashAttentionScoreFH": {"golden_path":"/npu/fa_smoke/auto_input"},
    "SparseFlashAttentionScore": {"golden_path":"/npu/fa_smoke/auto_input"},
    "FlashAttentionScore": {"golden_path":"/npu/fa_smoke/auto_input"},
    "FlashAttentionScoreX1LongSeq": {"golden_path":"/npu/fa_smoke/auto_input"},
    "FlashAttentionScoreX1LongSeqFuzz":{"golden_path":"/npu/fa_smoke/auto_input"},
    "FlashAttentionRealCase":{"golden_path":"/npu/fa_smoke/auto_input"},
    "FlashAttentionUnequalSparse":{"golden_path":"/npu/fa_smoke/auto_input"},
    "FlashAttentionScoreX1xxxBRealCase":{"golden_path":"/npu/fa_smoke/auto_input"},
    "FlashAttentionExpandPrefixN":{"golden_path":"/npu/fa_smoke/auto_input"},
    "FlashAttentionFP32Dtype":{"golden_path":"/npu/fa_smoke/auto_input"},
    "FlashAttentionOpenSora":{"golden_path":"/npu/fa_smoke/auto_input"}
}

class TimeoutError(Exception):
    pass


def timeout(seconds=100, error_message='Function call timed out'):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator


def log(text):
    print(text)
    log_file = open('auto_run.txt', 'a')
    log_file.write(str(datetime.datetime.now()) + '  ')
    log_file.write(text)
    log_file.write("\n")
    log_file.close()


def print_log(data=None):
    """
    print log
    :param data: value to print
    :param level: log level, include DEBUG, INFO(default), WARN and ERROR
    :return:
    """
    level = "INFO"
    log_info = ("%s - [%s]-%s:%s - %s" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3], level,
                                          os.path.basename(sys._getframe().f_back.f_code.co_filename),
                                          str(sys._getframe().f_back.f_lineno).zfill(4), data))
    log(log_info)


def getColumnIndex(table, columnName):
    columnIndex = None
    # print table
    for i in range(table.ncols):
        # print columnName
        # print table.cell_value(0, i)
        if (table.cell_value(0, i) == columnName):
            columnIndex = i
            # print(i)
            break
    return columnIndex


def checkResult(a, b, name):
    print_log(f"info：开始计算 {name} 精度。")
    ratio_threshold = 0.005
    threshold_diff = 0.005
    if a.shape == b.shape:
        # 两个张量shape相同，开始对比
        if torch.all(torch.eq(a, b)):
            print_log(f"info：{name} 计算结果与标杆完全相同。")
            ratio_diff = 0
            diff = 0
            return 1, 0, 0
        else:
            # 统计两个张量的值差距超过5‰的元素数量和占比
            diff = torch.abs(a.sub(b))
            min = torch.empty(a.shape, dtype=torch.float16).to(a.device)
            min.fill_(0.000025)
            max = torch.max(torch.abs(a), torch.abs(b))
            threshold = torch.max(max.mul(ratio_threshold), min)
            mask = diff > threshold
            num_diff = torch.sum(mask)
            ratio_diff = num_diff / torch.numel(a)
            if ratio_diff > threshold_diff:
                print_log(
                    f"error：{name} 计算结果有{num_diff}个元素的偏差超过{ratio_threshold:.2%}，占比为{ratio_diff:.2%}！！")
            else:
                print_log(
                    f"info：{name} 计算结果有{num_diff}个元素的偏差超过{ratio_threshold:.2%}，占比为{ratio_diff:.2%} 。")
        ratio = (1 - ratio_diff).cpu().detach().numpy()
        # print_log(a.max(), b.max())
        max = torch.max(diff).cpu().detach().numpy()
        print_log(f"{name} diff_max : {max:.8f}")
        sum = torch.sum(diff).cpu().detach().numpy()
        print_log(f"{name} diff_sum : {sum:.6f}")
    else:
        print_log(f"error: {name}计算结果错误，shape与标杆不匹配，用例执行失败！！！")
        print_log(f"debug: a {a.shape}")
        print_log(f"debug: b {b.shape}")
        ratio = 0
        max = 999999
        sum = 999999
    return ratio, max, sum


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
        op_name = df["Name"][i]
        if "aclnnFlashAttentionScore_" in op_name or op_name == "FlashAttentionScore":
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
        op_name = df["Name"][i]
        if "aclnnFlashAttentionScoreGrad" in op_name or op_name == "FlashAttentionScoreGrad":
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


# 由gen_mask获取drop_mask，keep_prob不等于1时，必须使用npu环境获取gen_mask
def get_drop_mask(q,B, N1, S1, S2, seed=2, gen_p=0.2, pttype=torch.float16):
    q = q.to(device)
    print_log(f"starting making gen mask...  gen_p = {gen_p}")
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.manual_seed(seed)
    drop_masks_uint8 = torch_npu._npu_dropout_gen_mask(q,[B, N1, S1, S2], p=gen_p, seed=seed, offset=0, parallel=True,sync=False) # q 用来获取dtype
    print_log(drop_masks_uint8[0:10])
    drop_masks_bit_np = np.unpackbits(drop_masks_uint8.cpu().numpy(), count=B * N1 * S1 * S2, bitorder='little')
    drop_masks_bit = torch.from_numpy(drop_masks_bit_np).reshape([B, N1, S1, S2])
    drop_masks_bit = torch.tensor(drop_masks_bit, dtype=torch.uint8)
    return drop_masks_bit.cpu()


def tsoftmax(x):
    x_max = torch.max(x, dim=-1, keepdims=True)[0]
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdims=True)
    # print("y.shape: ", y.shape)
    # print("x_sum.shape: ", x_sum.shape)
    ans = y.div(x_sum)
    return ans, x_max, x_sum


def tsoftmax_grad(dp, softmax_res):
    muls = dp * softmax_res
    muls_r = muls.sum(dim=-1, keepdims=True)
    sub_r = dp - muls_r
    res = sub_r * softmax_res
    return res


def tforward(q, k, v, drop_mask, atten_mask, pse, scale, keep_prob):
    print_log('tforward running')
    if pse is None or len(pse.shape) == 0:
        qk = torch.matmul(q, k.permute(0, 1, 3, 2)).mul(scale)
    else:
        qk = (torch.matmul(q, k.permute(0, 1, 3, 2)) + pse).mul(scale)
    if atten_mask is None or len(atten_mask.shape) == 0:
        qk = qk
    else:
        qk = qk + atten_mask * (-40000.0)  # -10000
    softmax_res, x_max, x_sum = tsoftmax(qk)
    softmax_res[atten_mask.bool().broadcast_to(softmax_res.shape)] = 0 # 要斟酌一下，部分场景存在计算异常精度问题
    if drop_mask is None or len(drop_mask.shape) == 0:
        drop_res = softmax_res
    else:
        drop_res = softmax_res * drop_mask * (1.0 / (keep_prob))
    y = torch.matmul(drop_res, v)
    return y, softmax_res, x_max, x_sum


def tbackward(dx, q, k, v, softmax_res, drop_mask, pse, scale, keep_prob):
    print_log('tbackward running')
    dp = torch.matmul(dx, v.permute(0, 1, 3, 2))
    if drop_mask is None or len(drop_mask.shape) == 0:
        drop_res = softmax_res.permute(0, 1, 3, 2)
        dp_drop = dp
    else:
        drop_res = softmax_res.mul(drop_mask).mul(1.0 / (keep_prob)).permute(0, 1, 3, 2)
        dp_drop = dp * drop_mask * (1.0 / (keep_prob))
    dv = torch.matmul(drop_res, dx)
    softmax_grad_res = (tsoftmax_grad(dp_drop, softmax_res) * scale)
    dq = torch.matmul(softmax_grad_res, k)
    dk = torch.matmul(softmax_grad_res.permute(0, 1, 3, 2), q)
    return dq, dk, dv


def tforward_n(q, k, v, drop_mask, atten_mask, pse, scale, keep_prob):
    print_log(q.shape)
    print_log(k.shape)
    print_log(v.shape)
    print_log(pse.shape)
    qk = (torch.matmul(q, k.permute(0, 1, 2, 4, 3)) + pse).mul(scale)
    qk = qk + atten_mask.bool() * (-40000.0)
    softmax_res, x_max, x_sum = tsoftmax(qk)
    # softmax_res = torch.softmax(qk, dim=-1)
    # print('softmax_res', softmax_res)
    drop_res = softmax_res * drop_mask * (1.0 / (keep_prob))
    y = torch.matmul(drop_res, v)
    return y, softmax_res


def tbackward_n(dx, q, k, v, softmax_res, drop_mask, pse, scale, keep_prob):
    drop_res = softmax_res.mul(drop_mask).mul(1.0 / (keep_prob)).permute(0, 1, 2, 4, 3)
    dv = torch.matmul(drop_res, dx)
    dp = torch.matmul(dx, v.permute(0, 1, 2, 4, 3))
    dp_drop = dp * drop_mask * (1.0 / (keep_prob))
    softmax_grad_res = (tsoftmax_grad(dp_drop, softmax_res) * scale)
    dq = torch.matmul(softmax_grad_res, k)
    dk = torch.matmul(softmax_grad_res.permute(0, 1, 2, 4, 3), q)
    return dq, dk, dv

def get_padding_mask(padding_mask_type,padding_mask_drop_prob,B,N1,S1,S2):
    if padding_mask_type == 'SS':
        rand_tensor = torch.rand(S1, S2)
    elif padding_mask_type == 'B1SS':
        rand_tensor = torch.rand(B, 1, S1, S2)
    elif padding_mask_type == 'BNSS':
        rand_tensor = torch.rand(B, N1, S1, S2)
    else:
        print_log(f"padding_mask 配置有误，请检查！！！！！！  padding_mask_type = {padding_mask_type}")
        return
    padding_mask = torch.where(rand_tensor < padding_mask_drop_prob, torch.ones_like(rand_tensor),
                                   torch.zeros_like(rand_tensor))
    return padding_mask

def get_prefix(B, S1, S2):
    prefix=[]
    if S1 >= S2:
        if B <= 2:
            for i in range(0, B):
                prefixOffset = np.random.randint(0, S2)
                prefix.append(prefixOffset)
        else:
            prefix = [0]
            for i in range(0, B - 2):
                prefixOffset = np.random.randint(0, S2)
                prefix.append(prefixOffset)
            prefix.append(S2)
    else:
        if B <= 2:
            for i in range(0,B):
                prefixOffset = np.random.randint(S2 - S1, S2)
                prefix.append(prefixOffset)
        else:
            prefix = [S2 - S1]
            for i in range(0, B-2):
                prefixOffset = np.random.randint(S2 - S1, S2)
                prefix.append(prefixOffset)
            prefix.append(S2)
    return prefix

def get_atten_mask(sparse_mode, atten_mask_shape, atten_mask_dtype, B, N1, N2, S1, S2, D, case_pre_tockens,
                   case_next_tockens, dtype, padding_mask_type, padding_mask_drop_prob):
    G = int(N1 / N2)
    prefix = []
    atten_masks = []
    pre_tocken = case_pre_tockens
    next_tocken = case_next_tockens
    if atten_mask_shape == "SS":
        shape = [S1, S2]
    elif atten_mask_shape == "B1SS":
        shape = [B, 1, S1, S2]
    elif atten_mask_shape == "BNSS":
        shape = [B, N1, S1, S2]
    elif atten_mask_shape == "11SS":
        shape = [1,1,S1,S2]
    elif atten_mask_dtype.lower() == 'none' or atten_mask_shape.lower() == 'none':
        if padding_mask_type != "NONE":
            padding_mask = get_padding_mask(padding_mask_type,padding_mask_drop_prob,B,N1,S1,S2)
            atten_masks = torch.zeros([S1, S2]) + padding_mask
            atten_mask = atten_masks.to(dtype)
            prefix = []
            pre_tocken = 65536
            next_tocken = 65536
        else:
            atten_mask = torch.tensor(0)
            prefix = None
            pre_tocken = 65536
            next_tocken = 65536
        atten_mask_npu = atten_mask
        return atten_mask, atten_mask_npu, prefix, pre_tocken, next_tocken
    else:
        raise RuntimeError(
            f"not support shape of atten_mask {atten_mask_shape}, only support [S1,S2], [B, 1, S1, S2], [B, N1, S1, S2] or None")

    if sparse_mode == 0:
        opname = 'sparse_band_flash_attention'
        pre_tocken = case_pre_tockens
        next_tocken = case_next_tockens
        atten_mask_u = torch.triu(torch.ones(shape), diagonal=next_tocken + 1)
        atten_mask_l = torch.tril(torch.ones(shape), diagonal=-pre_tocken - 1)
        atten_mask = (atten_mask_u + atten_mask_l).to(dtype)
        if padding_mask_type != "NONE":
            padding_mask = get_padding_mask(padding_mask_type, padding_mask_drop_prob, B, N1, S1, S2)
            atten_mask = (atten_mask + padding_mask).to(dtype)
        atten_mask_npu = atten_mask

    elif sparse_mode == 1:  # no sparse
        opname = 'flash_attention'
        atten_mask = (torch.zeros(shape)).to(dtype)
        pre_tocken = S1
        next_tocken = S2
        if padding_mask_type != "NONE":
            padding_mask = get_padding_mask(padding_mask_type, padding_mask_drop_prob, B, N1, S1, S2)
            atten_mask = (atten_mask + padding_mask).to(dtype)
        atten_mask_npu = atten_mask
    elif sparse_mode == 2:  # 用于生成golden数据,上三角
        opname = 'sparse_flash_attention'
        atten_mask = (torch.triu(torch.ones(shape), diagonal=1)).to(dtype)
        atten_mask_npu = torch.triu(torch.ones([2048, 2048]), diagonal=1).to(dtype)
        pre_tocken = S1
        next_tocken = 0
    elif sparse_mode == 3:  # 下三角
        opname = 'sparse_flash_attention'
        atten_mask = (torch.triu(torch.ones(shape), diagonal=S2 - S1 + 1)).to(dtype)
        atten_mask_npu = torch.triu(torch.ones([2048, 2048]), diagonal=1).to(dtype)
        pre_tocken = S2
        next_tocken = 0
    elif sparse_mode == 4:  # band
        opname = 'sparse_band_flash_attention'
        pre_tocken = case_pre_tockens
        next_tocken = case_next_tockens
        atten_mask_u = torch.triu(torch.ones(shape), diagonal=next_tocken + 1 + S2 - S1)
        atten_mask_l = torch.tril(torch.ones(shape), diagonal=-pre_tocken - 1 + S2 - S1)
        atten_mask = (atten_mask_u + atten_mask_l).to(dtype)
        atten_mask_npu = torch.triu(torch.ones([2048, 2048]), diagonal=1).to(dtype)

    elif sparse_mode == 5:  # prefix,shape只能是BNSS或者B1SS
        opname = 'sparse_flash_attention'
        if atten_mask_shape == 'SS':
            raise RuntimeError(f"prefix not support shape of atten_mask {atten_mask_shape}, only support BNSS and B1SS")
        atten_masks = torch.triu(torch.ones(shape), diagonal=S2 - S1 + 1)
        prefix=get_prefix(B=B, S1=S1, S2=S2)
        for i in range(0, B):
            atten_masks[i, :, :, 0:prefix[i]] = 0
        atten_mask = atten_masks.to(dtype)
        atten_mask_npu = atten_mask
    elif sparse_mode == 6:
        if atten_mask_shape in ['SS', '11SS']:
            raise RuntimeError(f"prefix not support shape of atten_mask {atten_mask_shape}, only support BNSS and B1SS")
        atten_masks = torch.triu(torch.ones(shape), diagonal=S2 - S1 + 1)
        prefix=get_prefix(B=B, S1=S1, S2=S2)
        for i in range(0, B):
            atten_masks[i, :, :, 0:prefix[i]] = 0
        upper = torch.triu(torch.ones(2048, 2048), diagonal=1)
        lower = torch.cat((torch.zeros(1024, 1024), torch.ones(1024, 1024)), dim=1)
        atten_mask = atten_masks.to(dtype)
        atten_mask_npu = torch.cat((upper, lower), dim=0).to(dtype)
    print_log(f"atten_mask: \n{atten_mask}")
    return atten_mask, atten_mask_npu, prefix, pre_tocken, next_tocken


def get_slopes(n_heads: int):
    n = 2 ** math.floor(math.log2(n_heads))
    m_0 = 2.0 ** (-8.0 / n)
    m = torch.pow(m_0, torch.arange(1, 1 + n))
    if n < n_heads:
        m_hat_0 = 2.0 ** (-4.0 / n)
        m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (n_heads - n), 2))
        m = torch.cat([m, m_hat])
    return m


def alibi_biases_18(q_seq, kv_seq, mask_value):
    # type 18 seq must equal — 向量化实现
    # [0,-inf,-inf]
    # [-1,  0,-inf]
    # [-2, -1,   0]
    y_idx = torch.arange(q_seq).unsqueeze(1)  # [q_seq, 1]
    x_idx = torch.arange(kv_seq).unsqueeze(0)  # [1, kv_seq]
    alibi_biases = torch.where(x_idx > y_idx, torch.tensor(mask_value, dtype=torch.float),
                               torch.where(x_idx < y_idx, (x_idx - y_idx).float(),
                                           torch.zeros(1, dtype=torch.float)))
    return alibi_biases


def alibi_biases_16(q_seq, kv_seq, mask_value):
    # type 16 like bloom — 向量化实现
    alibi_biases = torch.full((q_seq, kv_seq), mask_value)
    if q_seq == 1:
        alibi_biases[0, :] = torch.arange(kv_seq, dtype=torch.float)
    else:
        # 下三角区域填充列号 y
        tril_mask = torch.tril(torch.ones(q_seq, kv_seq), diagonal=0)
        col_idx = torch.arange(kv_seq).unsqueeze(0).expand(q_seq, kv_seq).float()
        alibi_biases = torch.where(tril_mask.bool(), col_idx, alibi_biases)
    return alibi_biases


def get_all_alibi(B, Nq, Sq, Skv, pse_layout, pse_type, pse_dtype, mask_value=-np.inf):
    m = get_slopes(Nq)
    if pse_type == 16:
        if pse_layout.lower() == "bnss":
            alibi_biases = alibi_biases_16(Sq, Skv, mask_value)
            pse_all = torch.zeros(B, Nq, Sq, Skv)
        elif pse_layout.lower() == "bn1s":
            alibi_biases = alibi_biases_16(1, Skv, mask_value)
            pse_all = torch.zeros(B, Nq, 1, Skv)
        elif pse_layout.lower() == "1nss":
            alibi_biases = alibi_biases_16(Sq, Skv, mask_value)
            pse_all = torch.zeros(1, Nq, Sq, Skv)
        else:
            print_log(f"[get_all_alibi] type:{pse_type} lay_out error:{pse_layout}")
            return None, None
        for n in range(Nq):
            pse_all[:, n:n + 1, :, :] = alibi_biases * m[n]
        return pse_all.to(pse_dtype), None
    elif pse_type == 18:
        alibi_biases = alibi_biases_18(Sq, Skv, mask_value)
        if pse_layout.lower() == "bnhs":
            pse_all = torch.zeros(B, Nq, Sq, Skv)
            npu_pse = torch.zeros(B, Nq, 1024, Skv)
        elif pse_layout.lower() == "1nhs":
            pse_all = torch.zeros(1, Nq, Sq, Skv)
            npu_pse = torch.zeros(1, Nq, 1024, Skv)
        elif pse_layout.lower() == "1nss":
            pse_all = torch.zeros(1, Nq, Sq, Skv)
            npu_pse = torch.zeros(1, Nq, Sq, Skv)
        elif pse_layout.lower() == "bnss":
            pse_all = torch.zeros(B, Nq, Sq, Skv)
            npu_pse = torch.zeros(B, Nq, Sq, Skv)
        else:
            print_log(f"[get_all_alibi] type:{pse_type} lay_out error:{pse_layout}")
            return None, None
        for n in range(Nq):
            pse_all[:, n:n + 1, :, :] = alibi_biases * m[n]

        if pse_layout.lower() in ["bnhs","1nhs"]:
            npu_pse = pse_all[:, :, -1024:, :]
            return pse_all.to(pse_dtype), npu_pse.to(pse_dtype)
        else:
            return pse_all.to(pse_dtype), None
    else:
        print_log(f"[get_all_alibi] only support type 16 and 18")
        return None, None

def broadcastKV_sigle(numHeads, numKeyValueHeads, kv_tensor, dtype):
    factor = numHeads // numKeyValueHeads
    kv_shape = kv_tensor.shape
    B = kv_shape[0]
    S = kv_shape[2]
    D = kv_shape[3]
    kv_res = torch.zeros([B, numHeads, S, D]).to(dtype)
    for i in range(numHeads):
        j = i // factor
        kv_res[:, i:i + 1, :, :] = kv_tensor[:, j:j + 1, :, :]
    return kv_res


# @timeout(seconds=3000)
def run(case, golden_path, deterministic, testmod, runmod='all', bin_path=None):
    torch.npu.set_device(int(device_id))
    if deterministic == "true":
        torch.use_deterministic_algorithms(True)
    time10 = time.time()

    # 初始化参数
    testcase_name = case["testcase_name"]
    B = case["B"]
    N1 = case["N1"]
    N2 = case["N2"] if not (case["N2"] is None) else N1
    S1 = case["S1"]
    S2 = case["S2"] if not (case["S2"] is None) else S1
    D = case["D"]
    D2 = case["D2"] if not (case["D2"] is None) else D
    H = N1 * D
    scale = 1 / (D ** 0.5)
    keep_prob = case["keep_prob"]
    padding_mask_drop_prob = 0.05  # padding_mask丢弃概率
    dtype = case["dtype"].lower()  # 'fp16' & 'bf16' & 'fp32'
    sparse_mode = int(case['sparse_mode'] if (case["sparse_mode"] != '') else 0)
    input_layout = case["input_layout"].upper()  # 'BSH' & 'SBH' & 'BNSD' & 'BSND'
    atten_mask_dtype = case["atten_mask_dtype"].upper()  # 'none' & 'bool' & 'qkv'
    atten_mask_shape = case["atten_mask_shape"].upper()  # 'SS' & 'B1SS' & 'BNSS'
    padding_mask_type = case["padding_mask"].upper()  # 'NONE' & 'SS' & 'B1SS' & 'BNSS'
    pse_layout = case["pse_layout"].upper()  # 'NONE' & 'BN1S' & 'BNSS'
    pse_type = int(case["pse_type"] if (case["pse_type"] != '') else 0)
    case_pre_tockens = case["pre_tockens"]
    case_next_tockens = case["next_tockens"]
    inner_precise = case["inner_precise"]

    pttype = torch.float
    if dtype == 'fp16' or dtype == 'FP16':
        pttype = torch.float16
    if dtype == 'bf16' or dtype == 'BF16':
        pttype = torch.bfloat16
    if dtype == 'fp32' or dtype == 'FP32':
        pttype = torch.float

    mask_dtype = torch.bool
    if atten_mask_dtype.lower() == "bool":
        mask_dtype = torch.bool
    if atten_mask_dtype.lower() == "uint8":
        mask_dtype = torch.uint8

    print_log("正在生成标杆数据")
    atten_mask, atten_mask_npu, prefix, pre_tocken, next_tocken = get_atten_mask(sparse_mode, atten_mask_shape,
                                                                                 atten_mask_dtype, B=B,
                                                                                 N1=N1, N2=N2, S1=S1, S2=S2, D=D,
                                                                                 case_pre_tockens=case_pre_tockens,
                                                                                 case_next_tockens=case_next_tockens,
                                                                                 dtype=mask_dtype,
                                                                                 padding_mask_type=padding_mask_type,
                                                                                 padding_mask_drop_prob=padding_mask_drop_prob)
    # 设置qkv
    min_value = -1
    max_value = 1
    q = 2 * (torch.rand([B, N1, S1, D]) - 0.5).to(pttype)
    k = 2 * (torch.rand([B, N2, S2, D]) - 0.5).to(pttype)
    v = 2 * (torch.rand([B, N2, S2, D2]) - 0.5).to(pttype)
    dx = 2 * (torch.rand([B, N1, S1, D2]) - 0.5).to(pttype)
    out_golden1 = 2 * (torch.rand([B, N1, S1, D]) - 0.5).to(pttype)
    x_max = 2 * (torch.rand([B, N1, S1, 8]) - 0.5)
    x_sum = 2 * (torch.rand([B, N1, S1, 8]) - 0.5)
    print_log(q.shape)
    print_log(k.shape)
    print_log(f"sparse, pre_tockens: {pre_tocken}, next_tockens: {next_tocken}")

    # 设置PSE mask
    if pse_type != 0 and pse_layout != "NONE":
        print_log(f"pse type: {pse_type}")
        pse, pse_npu = get_all_alibi(B, N1, S1, S2, pse_layout=pse_layout, pse_type=pse_type, pse_dtype=pttype)
        if pse_layout.lower() in ["bnhs", "1nhs"]:
            pass
        else:
            pse_npu = pse
    elif pse_type == 0 and pse_layout != "NONE":
        if pse_layout == "BN1S":
            shape = [B, N1, 1, S1]
        else:
            shape = [B, N1, S1, S2]
        pse = torch.randn(shape).to(pttype)
        pse_npu = pse
    else:
        pse = None

    # 默认"BNSD"无需处理
    if input_layout == "BSH":
        # (B,N,S,D)=>(B,S,N*D)
        q = rearrange(q, 'b n s d -> b s (n d)').contiguous()
        out_golden1 = rearrange(out_golden1, 'b n s d -> b s (n d)').contiguous()
        k = rearrange(k, 'b n s d -> b s (n d)').contiguous()
        v = rearrange(v, 'b n s d -> b s (n d)').contiguous()
        dx = rearrange(dx, 'b n s d -> b s (n d)').contiguous()
    if input_layout == "SBH":
        # (B,N,S,D)=>(S,B,N*D)
        q = rearrange(q, 'b n s d -> s b (n d)').contiguous()
        out_golden1 = rearrange(out_golden1, 'b n s d -> s b (n d)').contiguous()
        k = rearrange(k, 'b n s d -> s b (n d)').contiguous()
        v = rearrange(v, 'b n s d -> s b (n d)').contiguous()
        dx = rearrange(dx, 'b n s d -> s b (n d)').contiguous()
    if input_layout == "BSND":
        # (B,N,S,D)=>(B,S,N,D)
        q = rearrange(q, 'b n s d -> b s n d').contiguous()
        out_golden1 = rearrange(out_golden1, 'b n s d -> b s n d').contiguous()
        k = rearrange(k, 'b n s d -> b s n d').contiguous()
        v = rearrange(v, 'b n s d -> b s n d').contiguous()
        dx = rearrange(dx, 'b n s d -> b s n d').contiguous()

    if ((atten_mask_dtype.lower() == "none" or atten_mask_shape.lower() == "none") and padding_mask_type.lower() == "none") or atten_mask is None:
        atten_mask = None
    else:
        atten_mask = atten_mask.to(mask_dtype)
        atten_mask = atten_mask.to(device)

    if pse_layout != "NONE":
        pse1 = pse_npu.clone().to(device)
    else:
        pse1 = None
    atten_mask = atten_mask_npu
    q = q.to(device)
    k = k.to(device)
    v = v.to(device)
    dx = dx.to(device)

    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True

    dq, dk, dv, *_ = torch_npu.npu_fusion_attention_grad(
        q, k, v, dx, N1,
        pse=pse1,
        padding_mask=None,
        atten_mask=atten_mask.to(device),
        softmax_max=x_max.float().to(device),
        softmax_sum=x_sum.float().to(device),
        softmax_in=None,
        attention_in=out_golden1.to(pttype).to(device),
        scale_value=scale,
        keep_prob=keep_prob,
        input_layout=input_layout,
        pre_tockens=pre_tocken,
        next_tockens=next_tocken,
        numels=B * N1 * S1 * S2,
        inner_precise=inner_precise,
        prefix=prefix,
        sparse_mode=sparse_mode)

    return 1

def write_result(result_xls_name, case_name, column_name="running_status", str="PASS"):
    df = pd.read_excel(result_xls_name)
    df.loc[df['Testcase_Name'] == case_name, column_name] = str
    df.to_excel(result_xls_name, index=False)

def print_performance_data():
    log(f"---------------------|-------------------start print performance data-------------------|---------------------")
    os.system('cat log_ca/core*_summary_log')
    origin_log_file = open('log_ca/core0_summary_log', 'r')
    con = origin_log_file.readlines()
    for i in con:
        with open('performance_result_log', 'a') as new_log:
            new_log.write(i)
    origin_log_file.close()
    new_log.close()

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Timed out!!! 任务执行超时，子进程退出中... ...")

def run_case(case_list, nrows, deterministic, testmod, runmod='all', bin_path=None):
    while not case_list.empty():
        case = case_list.get(timeout=1)
        # get golden path
        if case["Group"][-1].isdigit():
            golden_group = case["Group"][:-2]
        else:
            golden_group = case["Group"]
        if all_case_dict.get(golden_group):
            golden_path = all_case_dict[golden_group]["golden_path"]
        else:
            golden_path = all_case_dict['default_path']["golden_path"]

        result = 0 # 0:aic err 1：正常执行完成 2：执行超时
        write_result(result_xls_name, case['testcase_name'], str='AIC_ERROR')
        print_log(f"==========================================================Start run case {nrows - 1 - case_list.qsize()}/{nrows - 1},case_name:{case['testcase_name']}==========================================================")
        if case["enable"].lower() == 'enable':
            print_log(str(case))
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(case_time_out_threshold)
            try:
                result = run(case, golden_path, deterministic, testmod, runmod, bin_path)
                signal.alarm(0)
            except TimeoutException as ex:
                print(ex)
                result = 2
                write_result(result_xls_name, case['testcase_name'], str='RTSYNC_TIMEOUT')
                multiprocessing.current_process().kill()
                return result
        else:
            print_log(f"{case['testcase_name']} is not enable, skip")
        write_result(result_xls_name, case['testcase_name'], str='PASS')
    result = 1
    return result

if __name__ == '__main__':
    # 用例使用说明：
    # BNSD：qkv的shape，H=N*D，S>4k时建议使用x86环境，否则执行很慢
    # dtype：枚举型，支持fp32，fp16与bf16
    # sparse：bool，False即测试FA、FAG，True即测试SFA、SFAG
    # input_layout：枚举型，支持BNSD，BSH，SBH
    # bool_attn：将atten_mash设置为bool输入
    # timemax：fa性能阈值，超过则报错性能劣化。timemaxg，timemaxs，timemaxsg同理对应fag，sfa，sfag
    # pse_type: pse输入设置，留空为不使用pse，可选BNSS或BN1S
    print_log(f"任务启动，主进程pid = {os.getpid()}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=int, default=0, help='case_id')
    parser.add_argument('--casefile', type=str, default='NULL', help='casefile')  # 可选
    parser.add_argument('--case_group', type=str, default='FlashAttentionALL', help='case group')
    parser.add_argument('--case_id', type=str, default='all', help='case_id')
    parser.add_argument('--runmod', type=str, default='NULL', help='special run mode')
    parser.add_argument('--version', type=str, default='zhuxian', help='running version')
    parser.add_argument('--deterministic', type=str, default='false', help='deterministic')
    parser.add_argument('--case_level', type=str, default='all', help='running specified level cases')
    #parser.add_argument('--basic_test', type=str, default='false', help='enable basic test') # 只使能测试功能精度，其余不测试
    parser.add_argument('--row_interval', type=str, default='all', help='running specific rows cases') # 执行指定行区间的case,除去表头第一行，输入格式：start_row-end_row，如需执行表格第2行至第30行case，则填入：2-30
    parser.add_argument('--testmod', type=str, default='accu', help='test mode')
    parser.add_argument('--bin_path', type=str, default=None, help='input bin path')

    args = parser.parse_args()
    case_name = args.case_id
    case_level = args.case_level
    row_interval = args.row_interval
    runmod = args.runmod
    print_log(f"device_id: {args.device_id}")
    global device_id
    device_id = args.device_id
    device = torch.device(f'npu:{args.device_id}')
    version = args.version
    casefile = args.casefile
    case_group = args.case_group
    deterministic = args.deterministic.lower()
    testmod = args.testmod
    bin_path = args.bin_path
    soc_version = '910B3'
    # get soc_version
    # soc_version = os.environ.get("SOC_VERSION_DAILY")
    if soc_version in ['910C2', '910B2']:
        soc_version = '910B2'
    elif soc_version in ['910C4','910B4']:
        soc_version = '910B4'
    elif soc_version in ['910B1', '910C1']:
        soc_version = '910B1'
    else:
        print_log('WARRING: the SOC_VERSION_DAILY is not setted, use 910B3 as default')
        soc_version = '910B3'
    print_log(f"info: the soc_version is setted to be {soc_version}")
    path = './'
    # get case_path
    if casefile == 'NULL':
        casefile = "FlashAttentionScoreALL"

    if version == "zhuxian":
        case_path = f"{path}case_xls/FlashAttentionScore/{casefile}_{soc_version}.xls"
    elif version == "shangfen":
        case_path = f"{path}case_xls/FlashAttentionScore/{casefile}_main_{soc_version}.xls"
    else:
        case_path = f"{path}case_xls/FlashAttentionScore/{casefile}_{version.upper()}_main_{soc_version}.xls"

    if os.path.exists(case_path):
        data = xlrd.open_workbook(case_path)
        print_log(f"INFO: using case file {case_path}")
    else:
        print_log(f"ERROR: case file {case_path} is not exist, stopping scripts")
        exit()
    if deterministic == "true":
        date_path = f"{path}/result_file/{datetime.date.today()}/FlashAttentionScoreDeterministic"
    else:
        date_path = f"{path}/result_file/{datetime.date.today()}/FlashAttentionScore"
    if os.path.exists(date_path):
        pass
    else:
        os.system(f"mkdir -p {date_path}")
    result_xls_name = case_path.split("/")[-1][:-4] + f"_Result_{datetime.datetime.now()}.xlsx"
    result_xls_name = result_xls_name[:-4].replace(":", "_").replace("-", "_").replace(" ", "_") + ".xlsx"
    result_xls_name = f'{date_path}/{result_xls_name}'
    os.system(f"cp '{case_path}' '{result_xls_name}'")
    table = data.sheet_by_name("Sheet1")
    if row_interval == 'all':
        start_row = 1
        end_row = table.nrows
    else:
        start_row = int(row_interval.split("-")[0]) - 1
        end_row = int(row_interval.split("-")[1])

    case_list = multiprocessing.Queue()
    Testcase_Name_Col = getColumnIndex(table, 'Testcase_Name')
    B_Col = getColumnIndex(table, 'B')
    N1_Col = getColumnIndex(table, 'N1')
    N2_Col = getColumnIndex(table, 'N2')
    S1_Col = getColumnIndex(table, 'S1')
    S2_Col = getColumnIndex(table, 'S2')
    D_Col = getColumnIndex(table, 'D')
    D2_Col = getColumnIndex(table, 'D2')
    Dtype_Col = getColumnIndex(table, 'Dtype')  # 'fp16' & 'bf16' & 'fp32'
    Sparse_Col = getColumnIndex(table, 'sparse_mode')  # 0,1,2,3,4,5
    Input_layout_Col = getColumnIndex(table, 'Layout')  # 'BSH' & 'SBH' & 'BNSD' & 'BSND'
    Atten_mask_dtype_Col = getColumnIndex(table, 'Atten_mask_Dtype')  # 'None' & 'bool' & 'qkv'
    Atten_mask_shape_Col = getColumnIndex(table, 'Atten_mask_Shape')  # 'SS' & 'B1SS' & 'BNSS'
    Padding_mask_Col = getColumnIndex(table, 'Padding_Mask')  # 'None' & 'SS' & 'B1SS' & 'BNSS'
    PSE_shape_Col = getColumnIndex(table, 'PSE')  # 'None' & 'BN1S' & 'BNSS'
    PSE_type_Col = getColumnIndex(table, 'pse_type')
    Enable_Col = getColumnIndex(table, 'Enable')  # 'enable' & 'disable'
    pre_tockens_Col = getColumnIndex(table, 'pre_tockens')
    next_tockens_Col = getColumnIndex(table, 'next_tockens')
    keep_prob_Col = getColumnIndex(table, 'keep_prob')
    case_level_Col = getColumnIndex(table, 'Level')
    case_group_Col = getColumnIndex(table, 'Group')
    case_inner_precise_Col = getColumnIndex(table, 'inner_precise')

    print_log(Testcase_Name_Col)
    pass_flag = True
    # result = []
    df = pd.read_excel(result_xls_name)
    for i in range(start_row, end_row):
        case = {}
        case["testcase_name"] = table.cell_value(i, Testcase_Name_Col)
        case["B"] = int(table.cell_value(i, B_Col))
        case["N1"] = int(table.cell_value(i, N1_Col))
        case["N2"] = None if table.cell_value(i, N2_Col) == '' else int(table.cell_value(i, N2_Col))
        case["S1"] = int(table.cell_value(i, S1_Col))
        case["S2"] = None if table.cell_value(i, S2_Col) == '' else int(table.cell_value(i, S2_Col))
        case["D"] = int(table.cell_value(i, D_Col))
        case["dtype"] = table.cell_value(i, Dtype_Col)
        case["sparse_mode"] = table.cell_value(i, Sparse_Col)
        case["input_layout"] = table.cell_value(i, Input_layout_Col)
        case["atten_mask_dtype"] = table.cell_value(i, Atten_mask_dtype_Col)
        case["atten_mask_shape"] = table.cell_value(i, Atten_mask_shape_Col)
        case["padding_mask"] = table.cell_value(i, Padding_mask_Col)
        case["pse_layout"] = table.cell_value(i, PSE_shape_Col)
        case["pse_type"] = '' if table.cell_value(i, PSE_type_Col) == '' else table.cell_value(i, PSE_type_Col)
        case["pre_tockens"] = int(table.cell_value(i, pre_tockens_Col)) if table.cell_value(i,
                                                                                            pre_tockens_Col) != '' else 65536
        case["next_tockens"] = int(table.cell_value(i, next_tockens_Col)) if table.cell_value(i,
                                                                                              next_tockens_Col) != '' else 65536
        case["keep_prob"] = float(table.cell_value(i, keep_prob_Col)) if table.cell_value(i, keep_prob_Col) != '' else 1
        case["case_level"] = str(table.cell_value(i, case_level_Col)) if table.cell_value(i, case_level_Col) != '' else "level99"
        case["enable"] = table.cell_value(i, Enable_Col)
        case["Group"] = table.cell_value(i, case_group_Col) if table.cell_value(i, case_group_Col) != '' else "default_path"
        try:
            case["inner_precise"] = 0 if table.cell_value(i, case_inner_precise_Col) == '' else int(
                table.cell_value(i, case_inner_precise_Col))
        except:
            case["inner_precise"] = 0
        try:
            case["D2"] = None if table.cell_value(i, D2_Col) == '' else int(table.cell_value(i, D2_Col))
        except:
            case["D2"] = None
        if ((case_name == 'all' and case_level == 'all' and case_group == 'FlashAttentionALL') or   # 执行所有case，不需要指定case_group
                (case_name == 'all' and case_level == 'all' and case["Group"] == case_group) or  # 执行指定group的case
                (case['case_level'].lower() == case_level.lower() and case_group == 'FlashAttentionALL') or  # 执行全量指定level的case
                (case['case_level'].lower() == case_level.lower() and case["Group"] == case_group) or  # 执行指定group的指定level的case
                case_name == case["testcase_name"]): #执行指定id的case
            if case["enable"].lower() == 'enable':
                case_list.put(case)
            else:
                df.loc[df['Testcase_Name'] == case['testcase_name'], "running_status"] = "PASS"
                df.loc[df['Testcase_Name'] == case['testcase_name'], "Precison_result"] = "Pass"
                df.loc[df['Testcase_Name'] == case['testcase_name'], "Performance_result"] = "Pass"
                df.loc[df['Testcase_Name'] == case['testcase_name'], "Memory_result"] = "Pass"
                print_log(f"{case['testcase_name']} is not enable, skip")

    df.to_excel(result_xls_name, index=False)
    print(case_list)


    while not case_list.empty():
        p = multiprocessing.Process(target=run_case, args=(case_list,end_row, deterministic, testmod, runmod, bin_path))
        p.start()
        print_log(f"拉起子进程，pid = {p.pid}")
        p.join()
        if not case_list.empty():
            print_log(f"ERROR:任务执行异常！正在拉起新进程...")
    if pass_flag:
        print_log("FLASH_ATTENTION TESTING SUCCESS")
    else:
        print_log("FLASH_ATTENTION TESTING FINISH WITH ERROR")
    df = pd.read_excel(result_xls_name, index_col=0)
    result_flag: bool = True
    print(f'result is {result_flag}!')

