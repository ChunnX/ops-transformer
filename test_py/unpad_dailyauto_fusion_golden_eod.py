# FA接口参数介绍：https://wiki.huawei.com/domains/17115/wiki/35808/WIKI202311162363573
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
import argparse
import random

case_time_out_threshold = 120000
"""
unpad执行daily用例中的用例名必须包含unpad字样，否则会调用失败！
"""
all_case_dict = {
    "default_path": {"golden_path": "/npu/fa_smoke/auto_input"},
    "unpadStressTest": {"golden_path": "/npu/fa_smoke/auto_input"},
    "unpadFlashAttentionAlibi": {"golden_path": "/npu/fa_smoke/auto_input"},
    "unpadFlashAttentionSparse": {"golden_path": "/npu/fa_smoke/auto_input"},
    "unpadFlashAttentionExpendPrefixN": {"golden_path": "/npu/fa_smoke/auto_input"},
    "unpadFlashAttentionFP32Dtype": {"golden_path": "/npu/fa_smoke/auto_input"},
    "packedFlashAttention": {"golden_path": "/npu/fa_smoke/auto_input"},
    "unpadFlashAttentionOpenSora": {"golden_path": "/npu/fa_smoke/auto_input"},
    "unpadFlashAttention0Seq": {"golden_path": "/npu/fa_smoke/auto_input"}
}
# /10.155.111.116/FlashAttention_Golden/unpadFlashAttentionSparse/auto_input
csv_name = f"result_Auto_{datetime.datetime.now()}.csv"
stepn = 1
gtype = torch.float64  # arm host必须选择float64，x86环境选择float32即可，64也行。arm计算很慢，s=8k的场景建议使用x86
catch_saving = True  # 是否保存golden数据缓存，开启后将golden保存在auto_input实现连跑加速。！！！注意此功能将消耗大量磁盘空间！！！
catch_refresh = False  # 强制使用cpu重新运算标杆，不使用golden数据，同时打开catch_saving则会将新数据覆盖旧数据


# torch.use_deterministic_algorithms(True)


def load_npy_file(file_path, file):
    file_all_path = file_path + file
    result = np.load(file_all_path)
    os.system(f"touch {file_all_path}")
    return result


def load_torch_file(file_path, file):
    file_all_path = file_path + file
    result = torch.load(file_all_path)
    os.system(f"touch {file_all_path}")
    return result


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
            ratio = 1
            max = 0
            sum = 0
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
            if torch.max(diff) > 0.1:
                print_log(
                    f"error：{name} 计算结果有{num_diff}个元素的偏差超过{ratio_threshold:.2%}，占比为{ratio_diff:.2%} ！！")
            else:
                print_log(
                    f"info：{name} 计算结果有{num_diff}个元素的偏差超过{ratio_threshold:.2%}，占比为{ratio_diff:.2%} 。")
            ratio = (1 - ratio_diff).cpu().detach().numpy()
            # print_log(a.max(), b.max())
            max = torch.max(diff).cpu().detach().numpy()

            sum = torch.sum(diff).cpu().detach().numpy()
        print_log(f"{name} diff_max : {max:.8f}")
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
    # print('===================================')
    # print("goldens_ae:", goldens[index_ae], actual_outputs[index_ae])
    # print("goldens_re:", goldens[index_re], actual_outputs[index_re])
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


# 由gen_mask获取drop_mask，keep_prob不等于1时，必须使用npu环境获取gen_mask
def get_drop_mask(q, length, seed=2, gen_p=0.2, pttype=torch.float16):
    print_log(f"starting making gen mask...  gen_p = {gen_p}")
    q = q.to(device)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.manual_seed(seed)
    # drop_masks_uint8 = torch_npu.npu_dropout_gen_mask([length], p=gen_p, dtype=pttype, device=device)
    drop_masks_uint8 = torch_npu._npu_dropout_gen_mask(q, [length], p=gen_p, seed=seed, offset=0, parallel=True,
                                                       sync=False)
    print_log(drop_masks_uint8[0:10])
    drop_masks_bit_np = np.unpackbits(drop_masks_uint8.cpu().numpy(), count=length, bitorder='little')
    drop_masks_bit = torch.from_numpy(drop_masks_bit_np)
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

    softmax_res[atten_mask.bool().broadcast_to(softmax_res.shape)] = 0  # 要斟酌一下，部分场景存在计算异常精度问题
    if drop_mask is None or len(drop_mask.shape) == 0:
        drop_res = softmax_res
    else:
        drop_res = softmax_res * drop_mask * (1.0 / (keep_prob))
    y = torch.matmul(drop_res, v)
    return y, softmax_res, x_max, x_sum


def tforward_new(q, k, v, drop_mask, atten_mask, pse, scale, keep_prob):
    print_log('tforward running')
    if pse is None or len(pse.shape) == 0:
        qk = torch.matmul(q, k.permute(0, 1, 3, 2)).to(torch.float32).mul(scale)
    else:
        qk = (torch.matmul(q, k.permute(0, 1, 3, 2)).to(torch.float32) + pse).mul(scale)

    if atten_mask is None or len(atten_mask.shape) == 0:
        qk = qk
    else:
        qk = qk + atten_mask * (-40000.0)  # -10000

    softmax_res, x_max, x_sum = tsoftmax(qk)

    softmax_res[atten_mask.bool().broadcast_to(softmax_res.shape)] = 0  # 要斟酌一下，部分场景存在计算异常精度问题
    if drop_mask is None or len(drop_mask.shape) == 0:
        drop_res = softmax_res
    else:
        drop_res = softmax_res * drop_mask * (1.0 / (keep_prob))
    drop_res = drop_res.to(q.dtype)
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


def simple_softmax(src, max, sum):
    dst = torch.exp(src - max) / sum
    return dst


def tbackward_new(dx, q, k, v, x_max, x_sum, atten_in, drop_mask, atten_mask, pse, scale, keep_prob):
    # SubGraphA
    if pse is None or len(pse.shape) == 0:
        qk = torch.matmul(q, k.permute(0, 1, 3, 2)).to(torch.float32).mul(scale)
    else:
        qk = (torch.matmul(q, k.permute(0, 1, 3, 2)).to(torch.float32) + pse).mul(scale)
    qk = qk + atten_mask.bool() * (-4000000000.0)
    p = simple_softmax(qk, x_max, x_sum)
    if drop_mask is not None:
        p_drop = p.mul(drop_mask).mul(1.0 / (keep_prob)).permute(0, 1, 3, 2)
    else:
        p_drop = p.permute(0, 1, 3, 2)
    p_drop = p_drop.to(q.dtype)

    # SubGraphB
    softmax_grad = (dx * atten_in).sum(dim=-1, keepdims=True)
    dyv = torch.matmul(dx, v.permute(0, 1, 3, 2))
    if drop_mask is not None:
        dp = dyv.mul(drop_mask).mul(1.0 / (keep_prob))
    else:
        dp = dyv
    ds = p * (dp - softmax_grad)
    ds = ds.to(q.dtype)

    dv = torch.matmul(p_drop, dx)
    dq = torch.matmul(ds, k).mul(scale)
    dk = torch.matmul(ds.permute(0, 1, 3, 2), q).mul(scale)
    return dq.to(q.dtype), dk.to(q.dtype), dv.to(q.dtype)

def get_padding_mask(padding_mask_type, padding_mask_drop_prob, B, N1, S1, S2):
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


def get_prefix(seqlens_list_q, seqlens_list_k):
    B = len(seqlens_list_q)
    if B <= 2:
        prefix = []
        for i in range(len(seqlens_list_q)):
            S1 = seqlens_list_q[i]
            S2 = seqlens_list_k[i]
            if S1 >= S2:
                prefixOffset = np.random.randint(0, S2)
                prefix.append(prefixOffset)
            else:
                prefixOffset = np.random.randint(S2 - S1, S2)
                prefix.append(prefixOffset)
    else:
        if seqlens_list_q[0] >= seqlens_list_k[0]:
            prefix = [0]
        else:
            prefix = [seqlens_list_k[0] - seqlens_list_q[0]]
        for i in range(1, len(seqlens_list_q) - 1):
            S1 = seqlens_list_q[i]
            S2 = seqlens_list_k[i]
            if S1 >= S2:
                prefixOffset = np.random.randint(0, S2)
                prefix.append(prefixOffset)
            else:
                prefixOffset = np.random.randint(S2 - S1, S2)
                prefix.append(prefixOffset)
        prefix.append(seqlens_list_k[-1])
    return prefix


def get_atten_mask(sparse_mode, atten_mask_shape, atten_mask_dtype, B, N1, N2, seqlens_list_q, seqlens_list_k, D,
                   case_pre_tockens,
                   case_next_tockens, dtype, padding_mask_type, padding_mask_drop_prob):
    S1 = seqlens_list_q.max()
    S2 = seqlens_list_k.max()
    G = int(N1 / N2)
    prefix = []
    pre_tocken = case_pre_tockens
    next_tocken = case_next_tockens
    if atten_mask_shape == "SS":
        shape = [S1, S2]
    elif atten_mask_shape == "11SS":
        shape = [1, 1, S1, S2]
    elif atten_mask_shape == "B1SS":
        shape = [1, 1, S1, S2]
    elif atten_mask_shape == "BNSS":
        shape = [1, N1, S1, S2]
    elif atten_mask_dtype.lower() == 'none' or atten_mask_shape.lower() == 'none':
        if padding_mask_type != "NONE":
            padding_mask = get_padding_mask(padding_mask_type, padding_mask_drop_prob, 1, N1, S1, S2)
            atten_mask = (torch.zeros([S1, S2]) + padding_mask).to(dtype)
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
            padding_mask = get_padding_mask(padding_mask_type, padding_mask_drop_prob, 1, N1, S1, S2)
            atten_mask = (atten_mask + padding_mask).to(dtype)
        atten_mask_npu = atten_mask

    elif sparse_mode == 1:  # no sparse
        opname = 'flash_attention'
        atten_mask = (torch.zeros(shape)).to(dtype)
        pre_tocken = S1
        next_tocken = S2
        if padding_mask_type != "NONE":
            padding_mask = get_padding_mask(padding_mask_type, padding_mask_drop_prob, 1, N1, S1, S2)
            atten_mask = (atten_mask + padding_mask).to(dtype)
        atten_mask_npu = atten_mask
    elif sparse_mode == 2:  # 用于生成golden数据,上三角
        opname = 'sparse_flash_attention'
        atten_mask = (torch.triu(torch.ones(shape), diagonal=1)).to(dtype)
        pre_tocken = S1
        next_tocken = 0
        atten_mask_npu = torch.triu(torch.ones([2048, 2048]), diagonal=1).to(dtype)
    elif sparse_mode == 3:  # 下三角
        opname = 'sparse_flash_attention'
        atten_mask = (torch.triu(torch.ones(shape), diagonal=S2 - S1 + 1)).to(dtype)
        pre_tocken = S1
        next_tocken = 0
        atten_mask_npu = torch.triu(torch.ones([2048, 2048]), diagonal=1).to(dtype)
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
        atten_mask = (torch.triu(torch.ones(shape), diagonal=S2 - S1 + 1)).to(dtype)
        prefix = get_prefix(seqlens_list_q=seqlens_list_q, seqlens_list_k=seqlens_list_k)
        for i in range(0, B):
            if len(shape) > 2:
                atten_mask[:, :, :, 0:prefix[i]] = 0
            else:
                atten_mask[:, 0:prefix[i]] = 0
        atten_mask_npu = atten_mask

    elif sparse_mode == 6:
        atten_mask = torch.triu(torch.ones(shape), diagonal=S2 - S1 + 1)
        prefix = get_prefix(seqlens_list_q=seqlens_list_q, seqlens_list_k=seqlens_list_k)
        upper = torch.triu(torch.ones(2048, 2048), diagonal=1)
        lower = torch.cat((torch.zeros(1024, 1024), torch.ones(1024, 1024)), dim=1)
        atten_mask_npu = torch.cat((upper, lower), dim=0).to(dtype)
    elif sparse_mode == 7 or sparse_mode == 8:
        # 7和8场景下  最大的atten_mask是什么形状？
        # 7是前面B的都是RighDownCasual，最后一个B是Band，8是第一个B是Band，后面的B是LeftUpCasual
        atten_mask = torch.tensor(0)
        atten_mask_npu = torch.triu(torch.ones([2048, 2048]), diagonal=1).to(dtype)
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


def alibi_biases_18(q_seq, kv_seq):
    # type 18 seq must equal
    # [0,-inf,-inf]
    # [-1,  0,-inf]
    # [-2, -1,   0]
    alibi_biases = torch.zeros(q_seq, kv_seq)
    # fill -inf
    for x in range(0, kv_seq):
        for y in range(0, q_seq):
            if x > y:
                alibi_biases[y, x] = -np.inf
            if x < y:
                alibi_biases[y, x] = -y + x
    return alibi_biases


def alibi_biases_16(q_seq, kv_seq):
    # type 16 like boolm
    # [0, 1, 2] q_seq=1
    # or
    # [0,-inf,-inf]
    # [0,  1, -inf]
    # [0,  1,    2]
    alibi_biases = torch.zeros(q_seq, kv_seq)
    # fill -inf
    alibi_biases.fill_(-np.inf)
    if q_seq == 1:
        for i in range(kv_seq):
            alibi_biases[0, i] = i
    else:
        for x in range(q_seq):
            if x + 1 < kv_seq:
                for y in range(x + 1):
                    alibi_biases[x, y] = y
            else:
                for y in range(kv_seq):
                    alibi_biases[x, y] = y
    return alibi_biases


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


def get_all_alibi(B, Nq, Sq, Skv, pse_layout, pse_type, pse_dtype):
    m = get_slopes(Nq)
    if pse_type == 16:
        if pse_layout.lower() == "bnss":
            alibi_biases = alibi_biases_16(Sq, Skv)
            pse_all = torch.zeros(B, Nq, Sq, Skv)
        elif pse_layout.lower() == "bn1s":
            alibi_biases = alibi_biases_16(1, Skv)
            pse_all = torch.zeros(B, Nq, 1, Skv)
        elif pse_layout.lower() == "1nss":
            alibi_biases = alibi_biases_16(Sq, Skv)
            pse_all = torch.zeros(1, Nq, Sq, Skv)
        else:
            print_log(f"[get_all_alibi] type:{pse_type} lay_out error:{pse_layout}")
            return None, None
        for n in range(Nq):
            pse_all[:, n:n + 1, :, :] = alibi_biases * m[n]
        return pse_all.to(pse_dtype), pse_all.to(pse_dtype)
    elif pse_type == 18:
        alibi_biases = alibi_biases_18(Sq, Skv)
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

        if pse_layout.lower() in ["bnhs", "1nhs"]:
            npu_pse = pse_all[:, :, -1024:, :]
        else:
            npu_pse = pse_all
        return pse_all.to(pse_dtype), npu_pse.to(pse_dtype)
    else:
        print_log(f"[get_all_alibi] only support type 16 and 18")
        return None, None


def get_cu_seqlens(seqlens_list):
    cu = torch.zeros(len(seqlens_list) + 1, dtype=torch.int64)
    non_zero_indices = np.nonzero(seqlens_list)[0]
    if len(non_zero_indices) == 0:
        return cu
    for i in range(len(seqlens_list) + 1):
        if i < len(seqlens_list) and seqlens_list[i] == 0 and i > non_zero_indices[0]:
            cu[i] = sum(seqlens_list[:i])
            break
        cu[i] = sum(seqlens_list[:i])
    return cu


def get_unpad_atten_mask(sparse_mode, N1, seqlens_list_q, seqlens_list_k, atten_mask_shape, atten_mask_dtype,
                         padding_mask_type, padding_mask_drop_prob, pre_tocken, next_tocken, mask_dtype, t, prefix):
    band_index = 0
    max_seqlen_q = seqlens_list_q.max()
    max_seqlen_k = seqlens_list_k.max()
    B = len(seqlens_list_q)
    if atten_mask_shape == "SS":
        shape = [seqlens_list_q[t], seqlens_list_k[t]]
    elif atten_mask_shape == "11SS":
        shape = [1, 1, seqlens_list_q[t], seqlens_list_k[t]]
    elif atten_mask_shape == "B1SS":
        shape = [1, 1, seqlens_list_q[t], seqlens_list_k[t]]
    elif atten_mask_shape == "BNSS":
        shape = [1, N1, seqlens_list_q[t], seqlens_list_k[t]]
    elif atten_mask_dtype.lower() == 'none' or atten_mask_shape.lower() == 'none':
        if padding_mask_type != "NONE":
            padding_mask = get_padding_mask(padding_mask_type, padding_mask_drop_prob, 1, N1, max_seqlen_q,
                                            max_seqlen_k)
            atten_mask = (torch.zeros([max_seqlen_q, max_seqlen_k]) + padding_mask).to(mask_dtype)
            prefix = []
            pre_tocken = 65536
            next_tocken = 65536
        else:
            atten_mask = torch.tensor(0)
            prefix = None
            pre_tocken = 65536
            next_tocken = 65536
        return atten_mask
    else:
        raise RuntimeError(
            f"not support shape of atten_mask {atten_mask_shape}, only support [S1,S2], [B, 1, S1, S2], [B, N1, S1, S2] or None")

    if sparse_mode == 5 or sparse_mode == 6:  # prefix,shape只能是BNSS或者B1SS
        opname = 'sparse_flash_attention'
        atten_mask = (torch.triu(torch.ones(shape), diagonal=seqlens_list_k[t] - seqlens_list_q[t] + 1)).to(mask_dtype)
        if len(shape) > 2:
            atten_mask[:, :, :, 0:prefix[t]] = 0
        else:
            atten_mask[:, 0:prefix[t]] = 0
    elif sparse_mode == 7:
        prefix = None
        # 7和8场景下  最大的atten_mask是什么形状？
        # 7是前面B的都是RighDownCasual，最后一个B是Band
        for index in range(B - 1, -1, -1):
            if seqlens_list_q[index] != 0:
                band_index = index
                break
        if t == band_index:
            atten_mask_u = torch.triu(torch.ones(shape),
                                      diagonal=next_tocken + 1 + seqlens_list_k[t] - seqlens_list_q[t])
            atten_mask_l = torch.tril(torch.ones(shape),
                                      diagonal=-pre_tocken - 1 + seqlens_list_k[t] - seqlens_list_q[t])
            atten_mask = (atten_mask_u + atten_mask_l).to(mask_dtype)

        else:
            atten_mask = (torch.triu(torch.ones(shape), diagonal=seqlens_list_k[t] - seqlens_list_q[t] + 1)).to(
                mask_dtype)
    elif sparse_mode == 8:
        prefix = None
        # 是第一个B是Band，后面的B是LeftUpCasual
        for index in range(B):
            if seqlens_list_q[index] != 0:
                band_index = index
                break
        if t == band_index:
            atten_mask_u = torch.triu(torch.ones(shape),
                                      diagonal=next_tocken + 1 + seqlens_list_k[t] - seqlens_list_q[t])
            atten_mask_l = torch.tril(torch.ones(shape),
                                      diagonal=-pre_tocken - 1 + seqlens_list_k[t] - seqlens_list_q[t])
            atten_mask = (atten_mask_u + atten_mask_l).to(mask_dtype)
        else:
            atten_mask = (torch.triu(torch.ones(shape), diagonal=1)).to(mask_dtype)
    return atten_mask

def checkResult_npu(value, golden, name):
    small_value = 1
    small_value_atol = small_value * 4
    debug_switch = False
    golden_inf_switch = False  # 对高精度golden进行inf处理（value类型越界的值设置为inf），根据需要选择
    inf_clip_switch = True  # 将所有inf处理为边界值（inf误差转换为数值误差），推荐溢出类inf场景选择使用，谨慎选择
    nan_toZero_switch = True  # 将所有nan处理为0（nan误差转换为数值误差），推荐溢出类nan场景选择使用，谨慎选择
    if value.shape == golden.shape:
        # 两个张量shape相同，开始对比
        if torch.all(torch.eq(value, golden)):
            print(f"info：{name} 计算结果与标杆完全相同。")
            diff_big_ratio_max = torch.tensor(0.0, dtype=torch.float32)
            diff_big_ratio_avg = torch.tensor(0.0, dtype=torch.float32)
            diff_big_ratio_rmse = torch.tensor(0.0, dtype=torch.float32)
            err_small_num = torch.tensor(0, dtype=torch.long)
            return diff_big_ratio_max, diff_big_ratio_avg, diff_big_ratio_rmse, err_small_num
        else:
            # 对高精度golden进行inf处理（value类型越界的值设置为inf），根据需要选择
            dtype = value.dtype
            if dtype == torch.float16:
                small_err = 2 ** -11
                threshold = 2 ** -16
            elif dtype == torch.float32:
                small_err = 2 ** -14
                threshold = 2 ** -30
            else:
                small_err = 2 ** -8
                threshold = 2 ** -16
            if golden_inf_switch:
                golden[golden > torch.finfo(value.dtype).max] = torch.inf
                golden[golden < torch.finfo(value.dtype).min] = -torch.inf
            # 将所有inf处理为边界值（inf误差转换为数值误差），推荐溢出类inf场景选择使用，谨慎选择
            if inf_clip_switch:
                # golden[torch.isinf(golden)] = torch.finfo(value.dtype).max

                golden[torch.isinf(golden)] = torch.finfo(golden.dtype).max
                # golden[torch.isinf(golden)] = torch.finfo(torch.float32).max

                value[torch.isinf(value)] = torch.finfo(value.dtype).max
            # 将所有nan处理为0（nan误差转换为数值误差），推荐溢出类nan场景选择使用，谨慎选择

            if nan_toZero_switch:
                golden[torch.isnan(golden)] = 0
                value[torch.isnan(value)] = 0

            # inf nan对比

            mask_golden_is_nan = torch.isnan(golden)
            mask_value_is_nan = torch.isnan(value)
            num_total_nan = torch.sum(mask_golden_is_nan)
            err_total_nan = torch.sum(mask_golden_is_nan.logical_xor(mask_value_is_nan))

            mask_golden_is_inf = torch.isinf(golden) & (golden > 0)
            mask_value_is_inf = torch.isinf(value) & (value > 0)
            num_total_inf = torch.sum(mask_golden_is_inf)
            err_total_inf = torch.sum(mask_golden_is_inf.logical_xor(mask_value_is_inf))

            mask_golden_is_ninf = torch.isinf(golden) & (golden < 0)
            mask_value_is_ninf = torch.isinf(value) & (value < 0)
            num_total_ninf = torch.sum(mask_golden_is_ninf)
            err_total_ninf = torch.sum(mask_golden_is_ninf.logical_xor(mask_value_is_ninf))

            if debug_switch:
                print(f" inf/nan总数：{num_total_nan + num_total_inf + num_total_ninf}")
                print(f" inf/nan误差数：{err_total_nan + err_total_inf + err_total_ninf}")

            # 对inf/nan统一赋1，忽略影响
            golden[torch.isinf(golden)] = 1
            value[torch.isinf(value)] = 1
            golden[torch.isnan(golden)] = 1
            value[torch.isnan(value)] = 1

            # 大值对比
            total_big_num = torch.sum(golden.abs() >= small_err)
            total_big_ratio = total_big_num / golden.numel()

            # 对小值统一赋1，忽略影响
            value_big = value.clone()
            value_big[golden.abs() < small_err] = 1
            golden_big = golden.clone()
            golden_big[golden.abs() < small_err] = 1

            diff_big = torch.abs(value_big.sub(golden_big))

            diff_big_ratio = diff_big / (golden_big.abs() + 1e-7)

            diff_big_ratio_max = diff_big_ratio.max()
            diff_big_ratio_avg = diff_big_ratio.sum() / total_big_num
            diff_big_ratio_rmse = torch.sqrt(torch.mean(torch.square(diff_big_ratio)))

            # 小值对比
            total_small_num = torch.sum(golden.abs() < small_err)
            total_small_ratio = total_small_num / golden.numel()

            # 对大值统一赋1，忽略影响
            value_small = value.clone()
            value_small[golden.abs() > small_err] = 1
            golden_small = golden.clone()
            golden_small[golden.abs() > small_err] = 1

            diff_small = torch.abs(value_small.sub(golden_small))

            # 小值域错误数量
            err_small_num = torch.sum((golden_small.abs() < small_err) & (diff_small > threshold))
            err_small_ratio = err_small_num / total_small_num

            return diff_big_ratio_max, diff_big_ratio_avg, diff_big_ratio_rmse, err_small_num
    else:
        print(f"error: {name}计算结果错误，shape与标杆不匹配，用例执行失败！！！")
        print(f"debug: value {value.shape}")
        print(f"debug: golden {golden.shape}")

    return

# @timeout(seconds=3000)
def run(case, deterministic, gpu_ip, gpu_path, bin_path):
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
    dropout_p = 1 - keep_prob
    padding_mask_drop_prob = 0.05  # padding_mask丢弃概率
    dtype = case["dtype"].lower()  # 'fp16' & 'bf16' & 'fp32'
    sparse_mode = int(case['sparse_mode'] if (case["sparse_mode"] != '') else 0)
    input_layout = case["input_layout"].upper()  # 'BSH' & 'SBH' & 'BNSD' & 'BSND'
    atten_mask_dtype = case["atten_mask_dtype"].lower()  # 'none' & 'bool' & 'qkv'
    atten_mask_shape = case["atten_mask_shape"].upper()  # 'SS' & 'B1SS' & 'BNSS'
    padding_mask_type = case["padding_mask"].upper()  # 'NONE' & 'SS' & 'B1SS' & 'BNSS'
    pse_layout = case["pse_layout"].upper() if (case["pse_layout"] != '') else "NONE"  # '1NHS' & 'BNHS' & 'None'
    pse_type = int(case["pse_type"] if (case["pse_type"] != '') else 0)
    case_pre_tockens = case["pre_tockens"]
    case_next_tockens = case["next_tockens"]
    pre_tocken = case_pre_tockens
    next_tocken = case_next_tockens
    # seqlens_list_q = np.array([1000, 2048, 3000, 2048, ])
    seqlens_list_q = np.array(case["seqlens_list_q"])
    seqlens_list_k = np.array(case["seqlens_list_kv"]) if not (case["seqlens_list_kv"] is None) else seqlens_list_q
    B = len(seqlens_list_q)
    mask_dtype = torch.bool
    if atten_mask_dtype.lower() == "bool":
        mask_dtype = torch.bool
    if atten_mask_dtype.lower() == "uint8":
        mask_dtype = torch.uint8
    EOD_pad_num = S1 - sum(seqlens_list_q)
    cu_seqlens_q = get_cu_seqlens(seqlens_list_q)
    cu_seqlens_k = get_cu_seqlens(seqlens_list_k)
    max_seqlen_q = seqlens_list_q.max()
    max_seqlen_k = seqlens_list_k.max()
    # S1 = seqlens_list_q.sum()
    # S2 = seqlens_list_k.sum()
    qk_size = seqlens_list_q * seqlens_list_k
    qk_pointer = get_cu_seqlens(qk_size)
    print(f"seqlens_list_q:{seqlens_list_q}")
    print(f"seqlens_list_k:{seqlens_list_k}")
    print(f"cu_seqlens_q:{cu_seqlens_q}")
    print(f"cu_seqlens_k:{cu_seqlens_k}")
    print(f"max_seqlen_q:{max_seqlen_q}")
    print(f"max_seqlen_k:{max_seqlen_k}")
    print(f"qk_size:{qk_size}")
    print(f"qk_pointer:{qk_pointer}")
    # 判断输入场景是否有相应的gpu场景
    if dtype == "fp32" or D > 256 or keep_prob != 1 or D2 != D:
        golden_type = "cpu"
    else:
        if (atten_mask_shape.lower() == "none" or atten_mask_dtype.lower == "none") or (
                sparse_mode == 0 and case_pre_tockens >= S1 and case_next_tockens >= S2):
            golden_type = "gpu"
        else:
            if (sparse_mode == 3 or (sparse_mode == 4 and case_next_tockens == 0 and case_pre_tockens >= S2)):
                golden_type = "gpu"
            else:
                golden_type = "cpu"
    if N1 == N2 and S1 == S2:
        print_log(f"running case {testcase_name} : BNTD = {B}_{N1}_{S1}_{D}, sparse = {sparse_mode}, dtype = {dtype}")
    else:
        print_log(
            f"running case {testcase_name} : BNTD = {B}_{N1}({N2})_{S1}({S2})_{D}, sparse = {sparse_mode}, dtype = {dtype}")
    if not (N1 % N2 == 0 and N1 >= N2):
        print_log(f"N1与N2不匹配,请检查：N1 = {N1}, N2 = {N2}.")
        return 1
    if bin_path:
        inpu_path = f"{bin_path}/{testcase_name}"
    else:
        inpu_path = f"./auto_input/{testcase_name}"
        bin_path = f"./auto_input"
    print_log(inpu_path)
    # 检查golden数据是否齐全
    ck1 = os.path.exists(inpu_path + '/q.pt')
    ck2 = os.path.exists(inpu_path + '/k.pt')
    ck3 = os.path.exists(inpu_path + '/v.pt')
    ck4 = os.path.exists(inpu_path + '/dx.pt')
    ck5 = True if (atten_mask_shape == 'NONE' or atten_mask_dtype == 'NONE') else (
                os.path.exists(inpu_path + '/atten_mask.pt') or os.path.exists(inpu_path + '/atten_mask_npu.pt'))
    ck6 = os.path.exists(inpu_path + '/out.pt')
    ck7 = os.path.exists(inpu_path + '/dq.pt')
    ck8 = os.path.exists(inpu_path + '/dk.pt')
    ck9 = os.path.exists(inpu_path + '/dv.pt')
    # ck10 = os.path.exists(inpu_path + 'tockens.npy')
    ck11 = True if (pse_layout == 'NONE' and pse_type == 0) else (
                os.path.exists(inpu_path + '/pse.pt') or os.path.exists(inpu_path + '/pse_npu.pt'))
    catch_ok = ck1 and ck2 and ck3 and ck4 and ck5 and ck6 and ck7 and ck8 and ck9 and ck11
    print(
        f"ck1-q:{ck1},ck2-k:{ck2},ck3-v:{ck3},ck4-dx:{ck4},ck5-atten_mask:{ck5},ck6-out:{ck6},ck7-dq:{ck7},ck8-dk:{ck8},ck9-dv:{ck9},ck11-pse:{ck11}")

    # 读取golden缓存
    if catch_refresh:
        print_log("info：catch_refresh置位，强制更新数据")
    elif not catch_ok:
        print_log("info：catch数据不存在或损坏，重新生成数据")
        print(ck1, ck2, ck3, ck4, ck5, ck6, ck7, ck8, ck9, ck11)

    pttype = torch.float
    gtype = torch.float
    if dtype == 'fp16' or dtype == 'FP16':
        pttype = torch.float16
    if dtype == 'bf16' or dtype == 'BF16':
        pttype = torch.bfloat16
    if dtype == 'fp32' or dtype == 'FP32':
        pttype = torch.float
        gtype = torch.float64
    if catch_ok and (not catch_refresh):
        if golden_type == "gpu":
            os.system(f"scp -r {inpu_path}  root@{gpu_ip}:{gpu_path}")
            print(f"scp -r {inpu_path}  root@{gpu_ip}:{gpu_path}")
            return
        print_log(f"catch命中，正在读取标杆数据, path:{inpu_path}")
        q = load_torch_file(inpu_path, '/q.pt')
        k = load_torch_file(inpu_path, '/k.pt')
        v = load_torch_file(inpu_path, '/v.pt')
        dx = load_torch_file(inpu_path, '/dx.pt')
        if dtype == 'fp32' or dtype == 'FP32':
            q = q.to(pttype)
            k = k.to(pttype)
            v = v.to(pttype)
            dx = dx.to(pttype)
        out_golden = load_torch_file(inpu_path, '/out.pt')
        dq_golden = load_torch_file(inpu_path, '/dq.pt')
        dk_golden = load_torch_file(inpu_path, '/dk.pt')
        dv_golden = load_torch_file(inpu_path, '/dv.pt')
        if os.path.exists(inpu_path + "/pse_npu.pt") or os.path.exists(inpu_path + '/pse.pt'):
            if (pse_type == 0 and pse_layout.upper() != "NONE") or pse_layout.lower() not in ["bnhs", "1nhs"]:
                pse = load_torch_file(inpu_path, '/pse_npu.pt') if os.path.exists(
                    inpu_path + "/pse_npu.pt") else load_torch_file(inpu_path, '/pse.pt')
            else:
                pse, pse_npu = get_all_alibi(B, N1, S1, S2, pse_layout=pse_layout, pse_type=pse_type, pse_dtype=pttype)
        else:
            pse = None
        print(f"pse:{pse}")
        if pse_layout.upper() != "NONE" and pse is None:
            print_log(f"read pse file failed")
            # result.put("Failed")
            return 0
        if atten_mask_shape == "NONE" or atten_mask_dtype == "NONE":
            prefix = None
        else:
            if sparse_mode in [5, 6]:
                if os.path.exists(os.path.join(inpu_path, "/prefix.npy")):
                    prefix = load_npy_file(inpu_path, '/prefix.npy').tolist()
                    print_log(f'read prefix.npy :{prefix}')
                else:
                    print_log(f"read prefix file failed")
                    # result.put("Failed")
                    return 0
            else:
                prefix = None
        if os.path.exists(os.path.join(inpu_path, "/atten_mask_npu.pt")):
            atten_mask = load_torch_file(inpu_path, "/atten_mask_npu.pt").to(device)
            print_log('read atten_mask_npu.pt')
        elif os.path.exists(os.path.join(inpu_path, "/atten_mask.pt")):
            atten_mask = load_torch_file(inpu_path, "/atten_mask.pt").to(device)
            print_log('read atten_mask.pt')
        else:
            atten_mask = None
        print(f"atten_mask:{atten_mask}")
        if pre_tocken > S1 and next_tocken > S2:
            opname = 'flash_attention'
        else:
            opname = 'sparse_flash_attention'
            print_log(f"sparse enable, pre_tockens: {pre_tocken}, next_tockens: {next_tocken}")

        print_log("正在生成标杆数据")
        print_log(f"sparse, pre_tockens: {pre_tocken}, next_tockens: {next_tocken}")
        atten_mask, atten_mask_npu, prefix, pre_tocken, next_tocken = get_atten_mask(sparse_mode=sparse_mode,
                                                                                     atten_mask_shape=atten_mask_shape,
                                                                                     atten_mask_dtype=atten_mask_dtype,
                                                                                     B=B, N1=N1, N2=N2,
                                                                                     seqlens_list_q=seqlens_list_q,
                                                                                     seqlens_list_k=seqlens_list_k, D=D,
                                                                                     case_pre_tockens=case_pre_tockens,
                                                                                     case_next_tockens=case_next_tockens,
                                                                                     dtype=mask_dtype,
                                                                                     padding_mask_type=padding_mask_type,
                                                                                     padding_mask_drop_prob=padding_mask_drop_prob)
        print_log(f"atten_mask: {atten_mask.shape}")
        print_log(atten_mask)
        # 设置PSE mask
        if pse_layout.upper() != 'NONE':
            pse_scale = 0.1
            print_log(f"pse type: {pse_layout}")
            pse_s1 = max(1024, max_seqlen_q)
            pse, pse_npu = get_all_alibi(B, N1, pse_s1, max_seqlen_k, pse_layout, pse_type=pse_type, pse_dtype=q.dtype)
            if pse_type in [16, 18]:
                print_log(f"pse shape: {pse.shape}")
            print(pse)

            # pse_npu = pse[:, :, (pse_s - 1024):pse_s, (pse_s - max_seqlen_k):pse_s]
            print_log(f"pse_npu shape: {pse_npu.shape}")
            # print(pse_npu)
        else:
            pse = torch.tensor([0])
            pse_npu = None
        # print('pse', pse)

        # 设置drop_mask
        drop_mask = torch.tensor(1)
        if keep_prob == 1:
            drop_mask = torch.tensor(1)
        else:
            print('get_drop_mask ', qk_pointer[-1] * N1)
            drop_mask = get_drop_mask(q, qk_pointer[-1] * N1, seed=2, gen_p=1 - keep_prob, pttype=q.dtype)
            print_log(f"drop_mask: {drop_mask.shape}")
            print_log(drop_mask)
        # 运算golden
        out = torch.zeros([S1, N1, D2])
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        x_max = torch.empty(0)
        x_sum = torch.empty(0)
        pse = torch.tensor([])
        for i in range(B):
            print_log(f"正在计算正向batch {i}， s_{i}={seqlens_list_q[i]}")
            print_log(cu_seqlens_q[i])
            print_log(cu_seqlens_q[i + 1])
            if seqlens_list_q[i] != 0 and seqlens_list_k[i] != 0:
                qi = q[cu_seqlens_q[i]:cu_seqlens_q[i + 1]]
                ki = k[cu_seqlens_k[i]:cu_seqlens_k[i + 1]]
                vi = v[cu_seqlens_k[i]:cu_seqlens_k[i + 1]]
                print_log(f"q{i} shape: {qi.shape}")
                print_log(f"k{i} shape: {ki.shape}")
                print_log(f"v{i} shape: {vi.shape}")

                qi = rearrange(qi, 's n d -> 1 n s d')
                ki = rearrange(ki, 's n d -> 1 n s d')
                vi = rearrange(vi, 's n d -> 1 n s d')
                print_log(f"q{i} reshape: {qi.shape}")
                print_log(f"k{i} reshape: {ki.shape}")
                print_log(f"v{i} reshape: {vi.shape}")

                # N不等长适配by cdy
                if not (N1 == N2):
                    ki = broadcastKV_sigle(N1, N2, ki, ki.dtype)
                    vi = broadcastKV_sigle(N1, N2, vi, vi.dtype)

                if drop_mask.numel() > 1:
                    drop_maski = drop_mask[(qk_pointer[i] * N1):(qk_pointer[i + 1] * N1)].reshape(N1, seqlens_list_q[i],
                                                                                                  seqlens_list_k[i])
                    print_log(f"drop_mask{i} shape: {drop_maski.shape}")
                else:
                    drop_maski = drop_mask
                if sparse_mode in [5, 6, 7, 8]:
                    atten_maski = get_unpad_atten_mask(sparse_mode=sparse_mode, N1=N1, seqlens_list_q=seqlens_list_q,
                                                       seqlens_list_k=seqlens_list_k,
                                                       atten_mask_shape=atten_mask_shape,
                                                       atten_mask_dtype=atten_mask_dtype,
                                                       padding_mask_type=padding_mask_type,
                                                       padding_mask_drop_prob=padding_mask_drop_prob,
                                                       pre_tocken=pre_tocken,
                                                       next_tocken=next_tocken,
                                                       mask_dtype=mask_dtype, t=i, prefix=prefix)
                else:
                    if atten_mask.numel() > 1:
                        if sparse_mode in [3, 4, 5]:
                            if atten_mask_shape == "SS":
                                atten_maski = atten_mask[max_seqlen_q - seqlens_list_q[i]:,
                                              max_seqlen_k - seqlens_list_k[i]:]
                            else:
                                atten_maski = atten_mask[:, :, max_seqlen_q - seqlens_list_q[i]:,
                                              max_seqlen_k - seqlens_list_k[i]:]

                        else:
                            if atten_mask_shape == "SS":
                                atten_maski = atten_mask[:seqlens_list_q[i], :seqlens_list_k[i]]
                            else:
                                atten_maski = atten_mask[:, :, :seqlens_list_q[i], :seqlens_list_k[i]]
                        print_log(f"atten_mask{i} shape: {atten_maski.shape}")
                    else:
                        atten_maski = atten_mask

                psei = torch.randn([N1, S1_i, S2_i]) * 10
                pse = torch.cat([pse, psei.reshape(-1)])
                gtype = pttype
                # 正向golden运算
                outi, softmax_resi, x_maxi, x_sumi = tforward_new(qi.to(gtype).cpu(), ki.to(gtype).cpu(), vi.to(gtype).cpu(),
                                                                     drop_maski.to(gtype).cpu(), atten_maski.cpu(), psei.to(gtype).cpu(), scale,
                                                                     keep_prob)
                # print('x_max', x_max.shape, x_max)
                print_log(f"out{i} shape: {outi.shape}")
                # 记录out
                print('out_golden[cu_seqlens_q[i]:cu_seqlens_q[i+1]', cu_seqlens_q[i], cu_seqlens_q[i + 1])
                out[cu_seqlens_q[i]:cu_seqlens_q[i + 1]] = rearrange(outi, '1 n s d -> s n d')

                dxi = dx[cu_seqlens_q[i]:cu_seqlens_q[i + 1]]
                print_log(f"dx{i} shape: {dxi.shape}")
                dxi = rearrange(dxi, 's n d -> 1 n s d')
                print_log(f"dx{i} reshape: {dxi.shape}")
                # 反向golden运算
                dqi, dki, dvi = tbackward_new(dxi.cpu().to(gtype), qi.cpu().to(gtype), ki.cpu().to(gtype),
                                            vi.cpu().to(gtype), x_maxi.to(torch.float), x_sumi.to(torch.float), outi.to(gtype),
                                            drop_maski.cpu().to(gtype), atten_maski.cpu(), psei.cpu().to(gtype), scale, keep_prob)

                # N不等长适配by cdy
                if not (N1 == N2):
                    G = int(N1 / N2)
                    s2i = seqlens_list_k[i]
                    dki = torch.sum(dki.reshape(1, N2, G, s2i, D), dim=2, keepdim=True).reshape(1, N2, s2i, D)
                    dvi = torch.sum(dvi.reshape(1, N2, G, s2i, D2), dim=2, keepdim=True).reshape(1, N2, s2i, D2)
                # 记录dqkv
                dq[cu_seqlens_q[i]:cu_seqlens_q[i + 1]] = rearrange(dqi, '1 n s d -> s n d')
                dk[cu_seqlens_k[i]:cu_seqlens_k[i + 1]] = rearrange(dki, '1 n s d -> s n d')
                dv[cu_seqlens_k[i]:cu_seqlens_k[i + 1]] = rearrange(dvi, '1 n s d -> s n d')
    else:
        #return 1
        print_log("正在生成标杆数据")
        if catch_saving and not os.path.exists(inpu_path):
            os.system(f"mkdir -p {inpu_path}")
        # 设置qkv
        min_value = -1
        max_value = 1
        q = 2 * (torch.rand([S1, N1, D]) - 0.5).to(pttype)
        k = 2 * (torch.rand([S2, N2, D]) - 0.5).to(pttype)
        v = 2 * (torch.rand([S2, N2, D2]) - 0.5).to(pttype)
        dx = 2 * (torch.rand([S1, N1, D2]) - 0.5).to(pttype)
        print_log(q.shape)
        print_log(k.shape)
        print_log(f"sparse, pre_tockens: {pre_tocken}, next_tockens: {next_tocken}")
        atten_mask, atten_mask_npu, prefix, pre_tocken, next_tocken = get_atten_mask(sparse_mode=sparse_mode,
                                                                                     atten_mask_shape=atten_mask_shape,
                                                                                     atten_mask_dtype=atten_mask_dtype,
                                                                                     B=B, N1=N1, N2=N2,
                                                                                     seqlens_list_q=seqlens_list_q,
                                                                                     seqlens_list_k=seqlens_list_k, D=D,
                                                                                     case_pre_tockens=case_pre_tockens,
                                                                                     case_next_tockens=case_next_tockens,
                                                                                     dtype=mask_dtype,
                                                                                     padding_mask_type=padding_mask_type,
                                                                                     padding_mask_drop_prob=padding_mask_drop_prob)
        print_log(f"atten_mask: {atten_mask.shape}")
        print_log(atten_mask)

        # 设置PSE mask
        if pse_layout.upper() != 'NONE':
            pse_scale = 0.1
            print_log(f"pse type: {pse_layout}")
            pse_s1 = max(1024, max_seqlen_q)
            pse, pse_npu = get_all_alibi(B, N1, pse_s1, max_seqlen_k, pse_layout, pse_type=pse_type, pse_dtype=q.dtype)
            if pse_type in [16, 18]:
                print_log(f"pse shape: {pse.shape}")
            print(pse)

            # pse_npu = pse[:, :, (pse_s - 1024):pse_s, (pse_s - max_seqlen_k):pse_s]
            print_log(f"pse_npu shape: {pse_npu.shape}")
            # print(pse_npu)
        else:
            pse = torch.tensor([0])
            pse_npu = None
        print('pse', pse)

        # 设置drop_mask
        drop_mask = torch.tensor(1)
        if keep_prob == 1:
            drop_mask = torch.tensor(1)
        else:
            print('get_drop_mask ', max(qk_pointer) * N1)
            drop_mask = get_drop_mask(q, max(qk_pointer) * N1, seed=2, gen_p=1 - keep_prob, pttype=q.dtype)
            print_log(f"drop_mask: {drop_mask.shape}")
            print_log(drop_mask)

        # 运算golden
        out_golden = torch.zeros([S1, N1, D2]).to(gtype)
        dq_golden = torch.zeros_like(q).to(gtype)
        dk_golden = torch.zeros_like(k).to(gtype)
        dv_golden = torch.zeros_like(v).to(gtype)
        x_max = torch.empty(0)
        x_sum = torch.empty(0)
        if pse_layout.upper() == "BNSS":
            pse_npu = torch.tensor([])
        for i in range(B):
            print_log(f"正在计算正向batch {i}， s_{i}={seqlens_list_q[i]}")
            print_log(cu_seqlens_q[i])
            print_log(cu_seqlens_q[i + 1])
            if seqlens_list_q[i] != 0 and seqlens_list_k[i] != 0:
                qi = q[cu_seqlens_q[i]:cu_seqlens_q[i + 1]]
                ki = k[cu_seqlens_k[i]:cu_seqlens_k[i + 1]]
                vi = v[cu_seqlens_k[i]:cu_seqlens_k[i + 1]]

                qi = rearrange(qi, 's n d -> 1 n s d')
                ki = rearrange(ki, 's n d -> 1 n s d')
                vi = rearrange(vi, 's n d -> 1 n s d')

                # N不等长适配by cdy
                if not (N1 == N2):
                    ki = broadcastKV_sigle(N1, N2, ki, ki.dtype)
                    vi = broadcastKV_sigle(N1, N2, vi, vi.dtype)

                if drop_mask.numel() > 1:
                    drop_maski = drop_mask[(qk_pointer[i] * N1):(qk_pointer[i + 1] * N1)].reshape(N1, seqlens_list_q[i],
                                                                                                  seqlens_list_k[i])
                    print_log(f"drop_mask{i} shape: {drop_maski.shape}")
                else:
                    drop_maski = drop_mask
                if sparse_mode in [5, 6, 7, 8]:
                    atten_maski = get_unpad_atten_mask(sparse_mode=sparse_mode, N1=N1, seqlens_list_q=seqlens_list_q,
                                                       seqlens_list_k=seqlens_list_k,
                                                       atten_mask_shape=atten_mask_shape,
                                                       atten_mask_dtype=atten_mask_dtype,
                                                       padding_mask_type=padding_mask_type,
                                                       padding_mask_drop_prob=padding_mask_drop_prob,
                                                       pre_tocken=pre_tocken,
                                                       next_tocken=next_tocken,
                                                       mask_dtype=mask_dtype, t=i, prefix=prefix)
                else:
                    if atten_mask.numel() > 1:
                        if sparse_mode in [3, 4, 5]:
                            if atten_mask_shape == "SS":
                                atten_maski = atten_mask[max_seqlen_q - seqlens_list_q[i]:,
                                              max_seqlen_k - seqlens_list_k[i]:]
                            else:
                                atten_maski = atten_mask[:, :, max_seqlen_q - seqlens_list_q[i]:,
                                              max_seqlen_k - seqlens_list_k[i]:]

                        else:
                            if atten_mask_shape == "SS":
                                atten_maski = atten_mask[:seqlens_list_q[i], :seqlens_list_k[i]]
                            else:
                                atten_maski = atten_mask[:, :, :seqlens_list_q[i], :seqlens_list_k[i]]
                        print_log(f"atten_mask{i} shape: {atten_maski.shape}")
                    else:
                        atten_maski = atten_mask
                if pse.numel() > 1:
                    if pse_layout == "1NHS":
                        psei = pse[:, :, (pse_s1 - seqlens_list_q[i]):pse_s1,
                               (max_seqlen_k - seqlens_list_k[i]):max_seqlen_k]
                    else:
                        psei = pse[i:i + 1, :, (pse_s1 - seqlens_list_q[i]):pse_s1,
                               (max_seqlen_k - seqlens_list_k[i]):max_seqlen_k]
                    print_log(f"pse{i} shape: {psei.shape}")
                    # print(psei[0,0,:,:])
                else:
                    psei = pse
                if pse_layout.upper() == "BNSS":
                    S1_i = cu_seqlens_q[i + 1] - cu_seqlens_q[i]
                    S2_i = cu_seqlens_k[i + 1] - cu_seqlens_k[i]
                    s1s2 = S1_i * S2_i
                    psei = torch.randn([N1, S1_i, S2_i]) * 10
                    pse_npu = torch.cat([pse_npu, psei.reshape(-1)])

                # 正向golden运算
                outi_golden, softmax_resi, x_maxi, x_sumi = tforward(qi.to(gtype), ki.to(gtype), vi.to(gtype),
                                                                     drop_maski, atten_maski, psei.to(gtype), scale,
                                                                     keep_prob)
                # print('x_max', x_max.shape, x_max)
                print_log(f"out{i} shape: {outi_golden.shape}")
                # 记录out
                print('out_golden[cu_seqlens_q[i]:cu_seqlens_q[i+1]', cu_seqlens_q[i], cu_seqlens_q[i + 1])
                out_golden[cu_seqlens_q[i]:cu_seqlens_q[i + 1]] = rearrange(outi_golden, '1 n s d -> s n d')
                # 记录max,sum
                print_log(f"x_maxi shape: {x_maxi.shape}")
                x_maxi = x_maxi.broadcast_to(1, N1, seqlens_list_q[i], 8)
                x_maxi = x_maxi.contiguous().view(-1)
                x_sumi = x_sumi.broadcast_to(1, N1, seqlens_list_q[i], 8)
                x_sumi = x_sumi.contiguous().view(-1)
                print_log(f"x_maxi shape: {x_maxi.shape}")
                x_max = torch.cat([x_max, x_maxi], dim=0)
                x_sum = torch.cat([x_sum, x_sumi], dim=0)
                dxi = dx[cu_seqlens_q[i]:cu_seqlens_q[i + 1]]
                print_log(f"dx{i} shape: {dxi.shape}")
                dxi = rearrange(dxi, 's n d -> 1 n s d')
                print_log(f"dx{i} reshape: {dxi.shape}")
                # 反向golden运算
                dqi_golden, dki_golden, dvi_golden = tbackward(dxi.to(gtype), qi.to(gtype), ki.to(gtype),
                                                               vi.to(gtype), softmax_resi.to(gtype),
                                                               drop_maski.to(gtype), psei.to(gtype), scale, keep_prob)

                # N不等长适配by cdy
                if not (N1 == N2):
                    G = int(N1 / N2)
                    s2i = seqlens_list_k[i]
                    dki_golden = torch.sum(dki_golden.reshape(1, N2, G, s2i, D), dim=2, keepdim=True).reshape(1, N2,
                                                                                                              s2i, D)
                    dvi_golden = torch.sum(dvi_golden.reshape(1, N2, G, s2i, D2), dim=2, keepdim=True).reshape(1, N2,
                                                                                                              s2i, D2)

                # 记录dqkv
                dq_golden[cu_seqlens_q[i]:cu_seqlens_q[i + 1]] = rearrange(dqi_golden, '1 n s d -> s n d')
                dk_golden[cu_seqlens_k[i]:cu_seqlens_k[i + 1]] = rearrange(dki_golden, '1 n s d -> s n d')
                dv_golden[cu_seqlens_k[i]:cu_seqlens_k[i + 1]] = rearrange(dvi_golden, '1 n s d -> s n d')
            # else:
            #     x_maxi = torch.zeros(1, N1, seqlens_list_q[i], 8)
            #     x_maxi = x_maxi.contiguous().view(-1)
            #     x_sumi = torch.zeros(1, N1, seqlens_list_q[i], 8)
            #     x_sumi = x_sumi.contiguous().view(-1)
            #     x_max = torch.cat([x_max, x_maxi], dim=0)
            #     x_sum = torch.cat([x_sum, x_sumi], dim=0)
            #     print_log(f"s_{i}为空，sq={seqlens_list_q[i]}，skv={seqlens_list_k[i]}")

        if EOD_pad_num != 0:
            x_max = torch.cat([x_max, torch.zeros(1, N1, EOD_pad_num, 8).contiguous().view(-1)], dim=0)
            x_sum = torch.cat([x_sum, torch.zeros(1, N1, EOD_pad_num, 8).contiguous().view(-1)], dim=0)
        # 保存标杆数据catch
        if catch_saving:
            if not os.path.exists(bin_path):
                os.mkdir(bin_path)
            if not os.path.exists(inpu_path):
                os.mkdir(inpu_path)
            torch.save(q.detach(), inpu_path + '/q.pt')
            torch.save(k.detach(), inpu_path + '/k.pt')
            torch.save(v.detach(), inpu_path + '/v.pt')
            torch.save(dx.detach(), inpu_path + '/dx.pt')
            torch.save(out_golden.detach(), inpu_path + '/out.pt')
            torch.save(dq_golden.detach(), inpu_path + '/dq.pt')
            torch.save(dk_golden.detach(), inpu_path + '/dk.pt')
            torch.save(dv_golden.detach(), inpu_path + '/dv.pt')
            torch.save(x_max.detach(), inpu_path + '/x_max.pt')
            torch.save(x_sum.detach(), inpu_path + '/x_sum.pt')
            if pse_layout.upper() != "NONE":
                pse_npu = pse_npu.reshape([-1])
                torch.save(pse_npu.detach(), inpu_path + '/pse_npu.pt')
            # 用于计算npu压缩atten_mask场景
            if not ((atten_mask_dtype == "NONE" or atten_mask_shape == "NONE") and padding_mask_type == "NONE"):
                # torch.save(atten_mask.detach(), inpu_path + 'atten_mask.pt')
                torch.save(atten_mask_npu.detach(), inpu_path + '/atten_mask_npu.pt')
            if sparse_mode in [5, 6]:
                np.save(inpu_path + '/prefix.npy', prefix)
            print("保存标杆数据成功")
        if golden_type == "gpu":
            os.system(f"scp -r {inpu_path}  root@{gpu_ip}:{gpu_path}")
            print(f"scp -r {inpu_path}  root@{gpu_ip}:{gpu_path}")
        else:
            out = torch.zeros([S1, N1, D2])
            dq = torch.zeros_like(q)
            dk = torch.zeros_like(k)
            dv = torch.zeros_like(v)
            x_max = torch.empty(0)
            x_sum = torch.empty(0)
            if pse_layout.upper() == "BNSS":
                pse_npu = torch.tensor([])
            for i in range(B):
                print_log(f"正在计算正向batch {i}， s_{i}={seqlens_list_q[i]}")
                print_log(cu_seqlens_q[i])
                print_log(cu_seqlens_q[i + 1])
                if seqlens_list_q[i] != 0 and seqlens_list_k[i] != 0:
                    qi = q[cu_seqlens_q[i]:cu_seqlens_q[i + 1]]
                    ki = k[cu_seqlens_k[i]:cu_seqlens_k[i + 1]]
                    vi = v[cu_seqlens_k[i]:cu_seqlens_k[i + 1]]
                    print_log(f"q{i} shape: {qi.shape}")
                    print_log(f"k{i} shape: {ki.shape}")
                    print_log(f"v{i} shape: {vi.shape}")

                    qi = rearrange(qi, 's n d -> 1 n s d')
                    ki = rearrange(ki, 's n d -> 1 n s d')
                    vi = rearrange(vi, 's n d -> 1 n s d')
                    print_log(f"q{i} reshape: {qi.shape}")
                    print_log(f"k{i} reshape: {ki.shape}")
                    print_log(f"v{i} reshape: {vi.shape}")

                    # N不等长适配by cdy
                    if not (N1 == N2):
                        ki = broadcastKV_sigle(N1, N2, ki, ki.dtype)
                        vi = broadcastKV_sigle(N1, N2, vi, vi.dtype)

                    if drop_mask.numel() > 1:
                        drop_maski = drop_mask[(qk_pointer[i] * N1):(qk_pointer[i + 1] * N1)].reshape(N1,
                                                                                                      seqlens_list_q[i],
                                                                                                      seqlens_list_k[i])
                        print_log(f"drop_mask{i} shape: {drop_maski.shape}")
                    else:
                        drop_maski = drop_mask
                    if sparse_mode in [5, 6, 7, 8]:
                        atten_maski = get_unpad_atten_mask(sparse_mode=sparse_mode, N1=N1,
                                                           seqlens_list_q=seqlens_list_q,
                                                           seqlens_list_k=seqlens_list_k,
                                                           atten_mask_shape=atten_mask_shape,
                                                           atten_mask_dtype=atten_mask_dtype,
                                                           padding_mask_type=padding_mask_type,
                                                           padding_mask_drop_prob=padding_mask_drop_prob,
                                                           pre_tocken=pre_tocken,
                                                           next_tocken=next_tocken,
                                                           mask_dtype=mask_dtype, t=i, prefix=prefix)
                    else:
                        if atten_mask.numel() > 1:
                            if sparse_mode in [3, 4, 5]:
                                if atten_mask_shape == "SS":
                                    atten_maski = atten_mask[max_seqlen_q - seqlens_list_q[i]:,
                                                  max_seqlen_k - seqlens_list_k[i]:]
                                else:
                                    atten_maski = atten_mask[:, :, max_seqlen_q - seqlens_list_q[i]:,
                                                  max_seqlen_k - seqlens_list_k[i]:]

                            else:
                                if atten_mask_shape == "SS":
                                    atten_maski = atten_mask[:seqlens_list_q[i], :seqlens_list_k[i]]
                                else:
                                    atten_maski = atten_mask[:, :, :seqlens_list_q[i], :seqlens_list_k[i]]
                            print_log(f"atten_mask{i} shape: {atten_maski.shape}")
                        else:
                            atten_maski = atten_mask
                    if pse.numel() > 1:
                        if pse_layout == "1NHS":
                            psei = pse[:, :, (pse_s1 - seqlens_list_q[i]):pse_s1,
                                   (max_seqlen_k - seqlens_list_k[i]):max_seqlen_k]
                        else:
                            psei = pse[i:i + 1, :, (pse_s1 - seqlens_list_q[i]):pse_s1,
                                   (max_seqlen_k - seqlens_list_k[i]):max_seqlen_k]
                        print_log(f"pse{i} shape: {psei.shape}")
                        # print(psei[0,0,:,:])
                    else:
                        psei = pse
                    if pse_layout.upper() == "BNSS":
                        S1_i = cu_seqlens_q[i + 1] - cu_seqlens_q[i]
                        S2_i = cu_seqlens_k[i + 1] - cu_seqlens_k[i]
                        s1s2 = S1_i * S2_i
                        psei = torch.randn([N1, S1_i, S2_i]) * 10
                        pse_npu = torch.cat([pse_npu, psei.reshape(-1)])
                    gtype = pttype
                    # 正向golden运算
                    outi, softmax_resi, x_maxi, x_sumi = tforward_new(qi.to(gtype).cpu(), ki.to(gtype).cpu(),
                                                                      vi.to(gtype).cpu(),
                                                                      drop_maski.to(gtype).cpu(), atten_maski.cpu(),
                                                                      psei.to(gtype).cpu(), scale,
                                                                      keep_prob)
                    # print('x_max', x_max.shape, x_max)
                    print_log(f"out{i} shape: {outi.shape}")
                    # 记录out
                    print('out_golden[cu_seqlens_q[i]:cu_seqlens_q[i+1]', cu_seqlens_q[i], cu_seqlens_q[i + 1])
                    out[cu_seqlens_q[i]:cu_seqlens_q[i + 1]] = rearrange(outi, '1 n s d -> s n d')

                    dxi = dx[cu_seqlens_q[i]:cu_seqlens_q[i + 1]]
                    print_log(f"dx{i} shape: {dxi.shape}")
                    dxi = rearrange(dxi, 's n d -> 1 n s d')
                    print_log(f"dx{i} reshape: {dxi.shape}")
                    # 反向golden运算
                    dqi, dki, dvi = tbackward_new(dxi.cpu().to(gtype), qi.cpu().to(gtype), ki.cpu().to(gtype),
                                                  vi.cpu().to(gtype), x_maxi.to(torch.float), x_sumi.to(torch.float),
                                                  outi.to(gtype),
                                                  drop_maski.cpu().to(gtype), atten_maski.cpu(), psei.cpu().to(gtype),
                                                  scale, keep_prob)

                    # N不等长适配by cdy
                    if not (N1 == N2):
                        G = int(N1 / N2)
                        s2i = seqlens_list_k[i]
                        dki = torch.sum(dki.reshape(1, N2, G, s2i, D), dim=2, keepdim=True).reshape(1, N2, s2i, D)
                        dvi = torch.sum(dvi.reshape(1, N2, G, s2i, D2), dim=2, keepdim=True).reshape(1, N2, s2i, D2)
                    # 记录dqkv
                    dq[cu_seqlens_q[i]:cu_seqlens_q[i + 1]] = rearrange(dqi, '1 n s d -> s n d')
                    dk[cu_seqlens_k[i]:cu_seqlens_k[i + 1]] = rearrange(dki, '1 n s d -> s n d')
                    dv[cu_seqlens_k[i]:cu_seqlens_k[i + 1]] = rearrange(dvi, '1 n s d -> s n d')
    if golden_type == "cpu":
        today = datetime.date.today()
        today_str = today.strftime('%Y_%m_%d')
        casename = testcase_name
        df = pd.read_excel(result_xls_name)
        out_diff_big_ratio_max, out_diff_big_ratio_avg, out_diff_big_ratio_rmse, out_err_small_num = checkResult_npu(
            out.detach().cpu().reshape(-1), out_golden.detach().reshape(-1),
            "out")
        dq_diff_big_ratio_max, dq_diff_big_ratio_avg, dq_diff_big_ratio_rmse, dq_err_small_num = checkResult_npu(
            dq.detach().cpu().reshape(-1), dq_golden.detach().reshape(-1),
            "dq")
        dk_diff_big_ratio_max, dk_diff_big_ratio_avg, dk_diff_big_ratio_rmse, dk_err_small_num = checkResult_npu(
            dk.detach().cpu().reshape(-1), dk_golden.detach().reshape(-1),
            "dk")
        dv_diff_big_ratio_max, dv_diff_big_ratio_avg, dv_diff_big_ratio_rmse, dv_err_small_num = checkResult_npu(
            dv.detach().cpu().reshape(-1), dv_golden.detach().reshape(-1),
            "dv")
        df.loc[df['Testcase_Name'] == casename, 'golden_out_diff_big_ratio_max'] = out_diff_big_ratio_max.item()
        df.loc[df['Testcase_Name'] == casename, 'golden_out_diff_big_ratio_avg'] = out_diff_big_ratio_avg.item()
        df.loc[df['Testcase_Name'] == casename, 'golden_out_diff_big_ratio_rmse'] = out_diff_big_ratio_rmse.item()
        df.loc[df['Testcase_Name'] == casename, 'golden_out_err_small_num'] = out_err_small_num.item()
        df.loc[df['Testcase_Name'] == casename, 'golden_dq_diff_big_ratio_max'] = dq_diff_big_ratio_max.item()
        df.loc[df['Testcase_Name'] == casename, 'golden_dq_diff_big_ratio_avg'] = dq_diff_big_ratio_avg.item()
        df.loc[df['Testcase_Name'] == casename, 'golden_dq_diff_big_ratio_rmse'] = dq_diff_big_ratio_rmse.item()
        df.loc[df['Testcase_Name'] == casename, 'golden_dq_err_small_num'] = dq_err_small_num.item()
        df.loc[df['Testcase_Name'] == casename, 'golden_dk_diff_big_ratio_max'] = dk_diff_big_ratio_max.item()
        df.loc[df['Testcase_Name'] == casename, 'golden_dk_diff_big_ratio_avg'] = dk_diff_big_ratio_avg.item()
        df.loc[df['Testcase_Name'] == casename, 'golden_dk_diff_big_ratio_rmse'] = dk_diff_big_ratio_rmse.item()
        df.loc[df['Testcase_Name'] == casename, 'golden_dk_err_small_num'] = dk_err_small_num.item()
        df.loc[df['Testcase_Name'] == casename, 'golden_dv_diff_big_ratio_max'] = dv_diff_big_ratio_max.item()
        df.loc[df['Testcase_Name'] == casename, 'golden_dv_diff_big_ratio_avg'] = dv_diff_big_ratio_avg.item()
        df.loc[df['Testcase_Name'] == casename, 'golden_dv_diff_big_ratio_rmse'] = dv_diff_big_ratio_rmse.item()
        df.loc[df['Testcase_Name'] == casename, 'golden_dv_err_small_num'] = dv_err_small_num.item()
        # 保存修改后的Excel文件
        df.to_excel(result_xls_name, index=False)

        return 1


def write_result(result_xls_name, case_name, str):
    df = pd.read_excel(result_xls_name)
    df.loc[df['Testcase_Name'] == case_name, 'running_status'] = str
    df.to_excel(result_xls_name, index=False)


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Timed out!!! 任务执行超时，子进程退出中... ...")


def run_case(case_list, nrows, deterministic, gpu_ip, gpu_path, bin_path):
    while not case_list.empty():
        case = case_list.get(timeout=1)
        # get golden path

        result = 0  # 0:aic err 1：正常执行完成 2：执行超时
        write_result(result_xls_name, case['testcase_name'], str='AIC_ERROR')
        print_log(
            f"==========================================================Start run case {nrows - 1 - case_list.qsize()}/{nrows - 1},case_name:{case['testcase_name']}==========================================================")
        if case["enable"].lower() == 'enable':
            print_log(str(case))
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(case_time_out_threshold)
            try:
                result = run(case, deterministic, gpu_ip, gpu_path, bin_path)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=int, default=0, help='npu device id')
    parser.add_argument('--casefile', type=str, default='NULL', help='casefile')  # 可选
    parser.add_argument('--case_group', type=str, default='unpadFlashAttentionALL', help='case group')
    parser.add_argument('--case_id', type=str, default='all', help='case_id')
    parser.add_argument('--bin_path', type=str, default=None, help='input bin path')
    parser.add_argument('--version', type=str, default='zhuxian', help='running version')
    parser.add_argument('--deterministic', type=str, default='false', help='deterministic')
    parser.add_argument('--case_level', type=str, default='all', help='running specified level cases')
    parser.add_argument('--gpu_ip', type=str, default='', help='gpu ip')  # 执行gpu标杆的环境
    parser.add_argument('--row_interval', type=str, default='all',
                        help='running specific rows cases')  # 执行指定行区间的case,除去表头第一行，输入格式：start_row-end_row，如需执行表格第2行至第30行case，则填入：2-30
    parser.add_argument('--gpu_path', type=str, default='', help='gpu bin path')  # gpu环境存放真值bin的路径
    args = parser.parse_args()
    print_log(f"device_id: {args.device_id}")
    global device_id
    device_id = args.device_id
    device = torch.device(f'npu:{args.device_id}')
    print("11111device:", device)
    # torch_npu.npu.set_device(args.device_id)
    casefile = args.casefile
    case_name = args.case_id
    version = args.version
    deterministic = args.deterministic.lower()
    case_level = args.case_level
    row_interval = args.row_interval
    case_group = args.case_group
    bin_path = args.bin_path
    gpu_ip = args.gpu_ip
    gpu_path = args.gpu_path

    # get soc_version
    soc_version = os.environ.get("SOC_VERSION_DAILY")
    if soc_version in ['910C2', '910B2']:
        soc_version = '910B2'
    elif soc_version in ['910C4', '910B4']:
        soc_version = '910B4'
    elif soc_version in ['910B1', '910C1']:
        soc_version = '910B1'
    else:
        print_log('WARRING: the SOC_VERSION_DAILY is not setted, use 910B3 as default')
        soc_version = '910B3'
    print_log(f"info: the soc_version is setted to be {soc_version}")

    # get case_path
    if casefile == 'NULL':
        if deterministic == "true":
            casefile = "unpadFlashAttentionScoreDeterministic"
        else:
            casefile = "unpadFlashAttentionScore"
    path = './'

    if version == "zhuxian":
        case_path = f"{path}case_xls/unpadFlashAttentionScore/{casefile}_{soc_version}.xls"
    elif version == "shangfen":
        case_path = f"{path}case_xls/unpadFlashAttentionScore/{casefile}ALL_main_{soc_version}.xls"
    else:
        case_path = f"{path}case_xls/unpadFlashAttentionScore/{casefile}ALL_{version.upper()}_main_{soc_version}.xls"
    if os.path.exists(case_path):
        data = xlrd.open_workbook(case_path)
    else:
        print_log(f"ERROR: case file {case_path} is not exist, stopping scripts")
        exit()
    if deterministic == "true":
        date_path = f"{path}result_file/{datetime.date.today()}/unpadFlashAttentionScoreDeterministic"
    else:
        date_path = f"{path}result_file/{datetime.date.today()}/unpadFlashAttentionScore"
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
    # case_list = []
    case_list = multiprocessing.Queue()
    Testcase_Name_Col = getColumnIndex(table, 'Testcase_Name')
    B_Col = getColumnIndex(table, 'B')
    N1_Col = getColumnIndex(table, 'N1')
    N2_Col = getColumnIndex(table, 'N2')
    S1_Col = getColumnIndex(table, 'T1')
    S2_Col = getColumnIndex(table, 'T2')
    seqlens_list_q_Col = getColumnIndex(table, 'seqlens_list_q')
    seqlens_list_kv_Col = getColumnIndex(table, 'seqlens_list_kv')
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

    print_log(Testcase_Name_Col)
    pass_flag = True
    result = []
    df = pd.read_excel(result_xls_name)
    for i in range(start_row, end_row):
        case = {}
        case["testcase_name"] = table.cell_value(i, Testcase_Name_Col)
        case["B"] = int(table.cell_value(i, B_Col))
        case["N1"] = int(table.cell_value(i, N1_Col))
        case["N2"] = None if table.cell_value(i, N2_Col) == '' else int(table.cell_value(i, N2_Col))
        case["S1"] = int(table.cell_value(i, S1_Col))
        case["S2"] = None if table.cell_value(i, S2_Col) == '' else int(table.cell_value(i, S2_Col))
        case["seqlens_list_q"] = eval(table.cell_value(i, seqlens_list_q_Col))
        case["seqlens_list_kv"] = None if table.cell_value(i, seqlens_list_kv_Col) == '' else eval(
            table.cell_value(i, seqlens_list_kv_Col))
        case["D"] = int(table.cell_value(i, D_Col))
        case["dtype"] = table.cell_value(i, Dtype_Col)
        case["sparse_mode"] = table.cell_value(i, Sparse_Col)
        case["input_layout"] = table.cell_value(i, Input_layout_Col)
        case["atten_mask_dtype"] = table.cell_value(i, Atten_mask_dtype_Col)
        case["atten_mask_shape"] = table.cell_value(i, Atten_mask_shape_Col)
        case["padding_mask"] = 'none' if table.cell_value(i, Padding_mask_Col) == '' else table.cell_value(i,
                                                                                                           Padding_mask_Col)
        case["pse_layout"] = table.cell_value(i, PSE_shape_Col)
        case["pse_type"] = '' if table.cell_value(i, PSE_type_Col) == '' else table.cell_value(i, PSE_type_Col)
        case["pre_tockens"] = int(table.cell_value(i, pre_tockens_Col)) if table.cell_value(i,
                                                                                            pre_tockens_Col) != '' else 65536
        case["next_tockens"] = int(table.cell_value(i, next_tockens_Col)) if table.cell_value(i,
                                                                                              next_tockens_Col) != '' else 65536
        case["keep_prob"] = float(table.cell_value(i, keep_prob_Col)) if table.cell_value(i, keep_prob_Col) != '' else 1
        case["case_level"] = str(table.cell_value(i, case_level_Col)) if table.cell_value(i,
                                                                                          case_level_Col) != '' else "level99"
        case["enable"] = table.cell_value(i, Enable_Col)
        case["Group"] = table.cell_value(i, case_group_Col) if table.cell_value(i,
                                                                                case_group_Col) != '' else "default_path"
        try:
            case["D2"] = None if table.cell_value(i, D2_Col) == '' else int(table.cell_value(i, D2_Col))
        except:
            case["D2"] = None

        if ((
                case_name == 'all' and case_level == 'all' and case_group == 'unpadFlashAttentionALL') or  # 执行所有case，不需要指定case_group
                (case_name == 'all' and case_level == 'all' and case["Group"] == case_group) or  # 执行指定group的case
                (case[
                     'case_level'].lower() == case_level.lower() and case_group == 'unpadFlashAttentionALL') or  # 执行全量指定level的case
                (case['case_level'].lower() == case_level.lower() and case[
                    "Group"] == case_group) or  # 执行指定group的指定level的case
                case_name == case["testcase_name"]):  # 执行指定id的case
            if case["enable"].lower() == 'enable':
                case_list.put(case)
            else:
                df.loc[df['Testcase_Name'] == case['testcase_name'], "running_status"] = "PASS"
                df.loc[df['Testcase_Name'] == case['testcase_name'], "Precison_result"] = "Pass"
                df.loc[df['Testcase_Name'] == case['testcase_name'], "Performance_result"] = "Pass"
                df.loc[df['Testcase_Name'] == case['testcase_name'], "Memory_result"] = "Pass"
                print_log(f"{case['testcase_name']} is not enable, skip")
        else:
            print_log(f"{case['testcase_name']} is not enabled group, skip !!")
    df.to_excel(result_xls_name, index=False)
    print(case_list)

    while not case_list.empty():
        p = multiprocessing.Process(target=run_case, args=(case_list,end_row, deterministic, gpu_ip, gpu_path, bin_path))
        p.start()
        print_log(f"拉起子进程，pid = {p.pid}")
        p.join()
        if not case_list.empty():
            print_log(f"ERROR:任务执行异常！正在拉起新进程...")

    if pass_flag:
        print_log("PACKED_FLASH_ATTENTION TEST pass")
    else:
        print_log("PACKED_FLASH_ATTENTION TEST fail")
    df = pd.read_excel(result_xls_name, index_col=0)
    result_flag: bool = True
    if 'AIC_ERROR' in df['running_status'].values or 'RTSYNC_TIMEOUT' in df['running_status'].values:
        result_flag = False
    print(f'result is {result_flag}!')


