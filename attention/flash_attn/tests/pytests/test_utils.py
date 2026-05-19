#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import torch
import math
import numpy as np
import random
import datetime
import os
import sys
from einops import rearrange

PHILOX_M4_32 = [0xD2511F53, 0xCD9E8D57]
PHIL0X_W_32 = [0x9E3779B9, 0xBB67AE85]

def philox4_32(counter, key, rounds):
    return philox(counter, key, philox4_round, PHILOX_M4_32, philox4_bumpkey, PHIL0X_W_32, 32, 0xffffffff, rounds)


def philox4_round(counter, key, philox_m, len_w, mask_w):
    prod = philox_m[0] * counter[0]
    hi_1 = prod >> len_w
    lo_1 = prod & mask_w
    prod = philox_m[1] * counter[2]
    hi_2 = prod >> len_w
    lo_2 = prod & mask_w
    counter[0] = hi_2 ^ counter[1] ^ key[0]
    counter[1] = lo_2
    counter[2] = hi_1 ^ counter[3] ^ key[1]
    counter[3] = lo_1


def philox4_bumpkey(key, philox_w, mask_w):
    key[0] = (key[0] + philox_w[0]) & mask_w
    key[1] = (key[1] + philox_w[1]) & mask_w


def philox(counter, key, philox_round, philox_m, philox_bumpkey, philox_w, len_w, mask_w, rounds):
    for i in range(rounds -1):
        philox_round(counter, key, philox_m, len_w, mask_w)
        philox_bumpkey(key, philox_w, mask_w)
    philox_round(counter, key, philox_m, len_w, mask_w)
    return counter


def inc_counter(counter):
    for i in range(4):
        counter[i] = (counter[i] + 1) & 0xffffffff
        if counter[i] != 0:
            return counter
    return counter


def philox_random(rounds, counter, key, count):
    ret = list()
    for i in range((count + 255) // 256 * 256 // 4):
        ret.extend(philox4_32(counter[:], key[:], rounds))
        counter = inc_counter(counter[:])
    return np.array(ret)[:count]


def uint32_to_uint8_array(data):
    little_endian = np.array([1], dtype=np.uint32).view(np.uint8)[0] == 1
    uint8_array = np.zeros((4), dtype=np.uint8)
    if little_endian:
        uint8_array[0] = data & 0xff
        uint8_array[1] = (data >> 8) & 0xff
        uint8_array[2] = (data >> 16) & 0xff
        uint8_array[3] = (data >> 24) & 0xff
    else:
        uint32_data = np.array(data, dtype=np.dtype('>u4'))
        uint8_array[0] = uint32_data.view(np.uint8)[3]
        uint8_array[1] = uint32_data.view(np.uint8)[2]
        uint8_array[2] = uint32_data.view(np.uint8)[1]
        uint8_array[3] = uint32_data.view(np.uint8)[0]
    return uint8_array


def gen_dropmask_uint8(rounds, counter, key, uint8_count, keep_prob):
    keep_prob_uint8 = keep_prob * 255
    uint32_count = math.ceil(uint8_count / 4)
    philox_uint32 = philox_random(rounds, counter, key, uint32_count)
    philox_uint8 = []
    golden_uint8 = []
    for each_uint32 in philox_uint32:
        four_uint8 = uint32_to_uint8_array(each_uint32)
        philox_uint8.extend(four_uint8)
    for each_philox_uint8 in philox_uint8:
        compare_res = 1 if each_philox_uint8 <= keep_prob_uint8 else 0
        golden_uint8.append(compare_res)
    return golden_uint8[:uint8_count]


def gen_dropmask(b, n1, s1, s2, keep_prob=0.8, seed=2, offset=0):
    rounds = 7
    seed_high = seed // 0x100000000
    seed_low = seed - seed_high * 0x100000000
    key = [seed_low, seed_high]
    offset_aligned = math.ceil(offset / 16)
    offset_high = offset_aligned // 0x100000000
    offset_low = offset_aligned - offset_high * 0x100000000
    counter = [offset_low, offset_high, 0, 0]
    s2_aligned = math.ceil(s2 / 16) * 16
    count = b * n1 * s1 * s2_aligned
    golden_uint8 = gen_dropmask_uint8(rounds, counter, key, count, keep_prob)
    drop_mask_np = np.array(golden_uint8).reshape(b, n1, s1, s2_aligned)[:, :, :, :s2-s2_aligned].astype(
        np.uint8) if s2_aligned != s2 else np.array(golden_uint8).reshape(b, n1, s1, s2).astype(np.uint8)
    drop_mask = torch.from_numpy(drop_mask_np)
    return drop_mask


def gen_dropmask_tnd(length, keep_prob=0.8, seed=2, offset=0):
    rounds = 7
    seed_high = seed // 0x100000000
    seed_low = seed - seed_high * 0x100000000
    key = [seed_low, seed_high]
    offset_aligned = math.ceil(offset / 16)
    offset_high = offset_aligned // 0x100000000
    offset_low = offset_aligned - offset_high * 0x100000000
    counter = [offset_low, offset_high, 0, 0]
    count = math.ceil(length / 16) * 16
    golden_uint8 = gen_dropmask_uint8(rounds, counter, key, count, keep_prob)
    drop_mask_np = np.array(golden_uint8).astype(np.uint8)
    drop_mask = torch.from_numpy(drop_mask_np)
    return drop_mask


def gen_block_table(b, act_seq_lens_kv, block_size, block_table_shape=[]):
    s2_max = max(act_seq_lens_kv)
    max_block_num_per_batch = (s2_max + block_size - 1) // block_size
    # if block_table_shape provided
    if block_table_shape:
        print(f"generating block_table, the block_table_shape is {block_table_shape}")
        b = block_table_shape[0]
        max_block_num_per_batch = block_table_shape[1]
    block_table = torch.full((b, max_block_num_per_batch), -1, dtype=torch.int32)
    # get block_num
    block_num = 0
    for i in range(b):
        b_seq = act_seq_lens_kv[i] if len(act_seq_lens_kv) > 1 else act_seq_lens_kv[0]
        block_num += (b_seq + block_size - 1) // block_size
    # page cache
    block_id_array = torch.randperm(block_num, dtype=torch.int32)
    index = 0
    for i in range(b):
        b_seq = act_seq_lens_kv[i] if len(act_seq_lens_kv) > 1 else act_seq_lens_kv[0]
        b_block_num = (b_seq + block_size - 1) // block_size
        for j in range(b_block_num):
            block_table[i][j] = block_id_array[index]
            index = index + 1
    return block_table


def page_attn_for_bnsd(bnsd_tensor, b, act_seq_lens_kv, block_table, block_size):
    block_num = int(block_table.max()) + 1
    kv_cache_bnsd_shape = (block_num, bnsd_tensor.shape[1], block_size, bnsd_tensor.shape[3])
    page_cache_tensor = torch.zeros(size=kv_cache_bnsd_shape, dtype=bnsd_tensor.dtype)
    actual_b = block_table.shape[0]
    is_tnd = (b == 1 and actual_b > 1)
    cum_kv_offset = 0
    for i in range(actual_b):
        b_seq = act_seq_lens_kv[i] if len(act_seq_lens_kv) > i else act_seq_lens_kv[0]
        b_block_num = (b_seq + block_size - 1) // block_size
        if is_tnd:
            batch_dim = 0
            seq_offset = cum_kv_offset
        else:
            batch_dim = i
            seq_offset = 0
        for j in range(b_block_num):
            # 最后一个block填不满时，只填充实际数据部分，多余部分保持初始化值
            start_idx = seq_offset + j * block_size
            end_idx = seq_offset + min((j + 1) * block_size, b_seq)
            actual_size = end_idx - start_idx
            slice_data = bnsd_tensor[batch_dim, :, start_idx:end_idx, :]
            page_cache_tensor[block_table[i][j], :, :actual_size, :] = slice_data
        cum_kv_offset += b_seq
    return page_cache_tensor


def dtype_sizeof(data_type):
    if data_type == 'fp16' or data_type == 'bf16':
        return 2
    elif data_type == 'int8' or data_type == 'fp8':
        return 1


def rearrange_by_block_table(bnsd_tensor, block_table, block_size, b, act_seq_lens_kv, kv_storage_mode, kv_dtype):
    page_cache_tensor = page_attn_for_bnsd(bnsd_tensor, b, act_seq_lens_kv, block_table, block_size)
    if kv_storage_mode == "PA_BBND":
        return page_cache_tensor.permute(0, 2, 1, 3) # BNSD->BSND
    elif kv_storage_mode == "PA_BNBD":
        return page_cache_tensor
    elif kv_storage_mode == "PA_NZ":
        blk_elem = 32 // dtype_sizeof(kv_dtype)
        page_cache_tensor = page_cache_tensor.reshape(page_cache_tensor.shape[0],
                                                     page_cache_tensor.shape[1],
                                                     page_cache_tensor.shape[2],
                                                     page_cache_tensor.shape[3] // blk_elem,
                                                     blk_elem).permute(0, 1, 3, 2, 4)
        return page_cache_tensor
    else:
        return None


def trans_bnsd_to_layout(tensor, layout_type, **kwargs):
    b               = kwargs.get("B")
    block_size      = kwargs.get("block_size")
    block_table     = kwargs.get("block_table")
    seqused_kv      = kwargs.get("seqused_kv")
    dtype           = kwargs.get("Dtype", "bf16")

    if tensor is None:
        return None
    else:
        if layout_type == "BSH":
            return rearrange(tensor.clone(), 'b n s d -> b s (n d)')
        elif layout_type == "SBH":
            return rearrange(tensor.clone(), 'b n s d -> s b (n d)')
        elif layout_type == "BSND":
            return rearrange(tensor.clone(), 'b n s d -> b s n d')
        elif layout_type == "TND":
            return rearrange(tensor.clone(), '1 n s d -> s n d')
        elif layout_type == "PA_BBND" or layout_type == "PA_BNBD" or layout_type == "PA_NZ":
            return rearrange_by_block_table(tensor, block_table, block_size, b, seqused_kv, layout_type, dtype)
        return tensor.clone()


def get_seqlen_list(actual_seqlen):
    seq_list = torch.zeros(len(actual_seqlen), dtype=torch.int64)
    seq_list[0] = actual_seqlen[0]
    for i in range(1, len(actual_seqlen)):
        seq_list[i] = actual_seqlen[i] - actual_seqlen[i - 1]
    return seq_list


def generate_pse(b, n1, s1, s2, pse_type, pse_layout, dtype, q_start_idx, kv_start_idx):
    if pse_layout.lower() == "none" or pse_layout is None:
        return None, None
    pse_layout = pse_layout.lower()
    if pse_type in [0, 1]:
        assert pse_layout in ["bnss", "bn1s", "1nss", "bnhs", "1nhs"]
        if pse_layout == "bnss":
            pse = torch.randn(b, n1, s1, s2).to(dtype)
            pse_npu = pse
        elif pse_layout == "bn1s":
            pse = torch.randn(b, n1, 1, s2).to(dtype)
            pse_npu = pse
        elif pse_layout == "1nss":
            pse = torch.randn(1, n1, s1, s2).to(dtype)
            pse_npu = pse
        elif pse_layout == "bnhs":
            pse_bias = 1 / (torch.arange(b) + 1).unsqueeze(1) * 1 / (torch.arange(n1) + 1)
            bn11 = pse_bias.reshape(b, n1, 1, 1)
            alibi_base = -torch.abs(torch.arange(s1).unsqueeze(1) - torch.arange(s2).unsqueeze(0) - (s1 - s2))
            pse = bn11 * alibi_base
            pse_npu = pse[:, :, -1024:, :]
        elif pse_layout == "1nhs":
            pse_bias = 1 / (torch.arange(n1) + 1)
            bn11 = pse_bias.reshape(1, n1, 1, 1)
            alibi_base = -torch.abs(torch.arange(s1).unsqueeze(1) - torch.arange(s2).unsqueeze(0) - (s1 - s2))
            pse = bn11 * alibi_base
            pse_npu = pse[:, :, -1024:, :]
        return pse, pse_npu
    elif pse_type in [2, 3]:
        # alibi
        assert pse_layout in ["bn", "n"]
        slopes_size = n1 if pse_layout == "n" else b * n1
        if pse_layout == "n":
            alibi_slopes = torch.randn(n1).float()
            alibi_bias = alibi_slopes.reshape(1, n1, 1, 1)
        else:
            alibi_slopes = torch.randn(b, n1).float()
            alibi_bias = alibi_slopes.reshape(b, n1, 1, 1)
        if pse_type == 2:
            alibi_base = -torch.abs(torch.arange(s2).unsqueeze(0) - torch.arange(s1).unsqueeze(1) - q_start_idx + kv_start_idx)
        else:
            alibi_base = -(torch.abs(torch.arange(s2).unsqueeze(0) - torch.arange(s1).unsqueeze(1) - q_start_idx + kv_start_idx).sqrt())
        return alibi_bias * alibi_base, alibi_slopes


def generate_npu_mask(b, s1, s2, sparse_mode, pre_tokens, next_tokens, prefix=None):
    if sparse_mode is None:
        return None
    if sparse_mode == 0:
        atten_mask_u = torch.triu(torch.ones(s1, s2), diagonal=next_tokens + 1)
        atten_mask_l = torch.tril(torch.ones(s1, s2), diagonal=-pre_tokens - 1)
        mask = (atten_mask_u + atten_mask_l).bool()
    elif sparse_mode == 1:
        mask = torch.zeros(s1, s2).bool()
    elif sparse_mode in [2, 3, 4, 7, 8]:
        mask = torch.triu(torch.ones(2048, 2048), diagonal=1).bool()
    elif sparse_mode == 5:
        assert prefix is not None
        mask = torch.triu(torch.ones(b, 1, s1, s2), diagonal=s2 - s1 + 1)
        for i in range(0, b):
            mask[i, :, :, :prefix[i]] = 0
        mask = mask.bool()
    elif sparse_mode == 6:
        upper = torch.triu(torch.ones(2048, 2048), diagonal=1)
        lower = torch.cat((torch.zeros(1024, 1024), torch.ones(1024, 1024)), dim=1)
        mask = torch.cat((upper, lower), dim=0).bool()
    return mask


def generate_cpu_mask(b, s1, s2, sparse_mode, pre_tokens, next_tokens, prefix=None, index=None, band_index=None):
    if sparse_mode is None:
        return None
    if sparse_mode == 0:
        atten_mask_u = torch.triu(torch.ones(s1, s2), diagonal=next_tokens + 1)
        atten_mask_l = torch.tril(torch.ones(s1, s2), diagonal=-pre_tokens - 1)
        mask = (atten_mask_u + atten_mask_l).bool()
    elif sparse_mode == 1:
        mask = torch.zeros(s1, s2).bool()
    elif sparse_mode == 2:
        mask = torch.triu(torch.ones(s1, s2), diagonal=1).bool()
    elif sparse_mode == 3:
        mask = torch.triu(torch.ones(s1, s2), diagonal=s2 - s1 + 1).bool()
    elif sparse_mode == 4:
        atten_mask_u = torch.triu(torch.ones(s1, s2), diagonal=next_tokens + 1 + s2 - s1)
        atten_mask_l = torch.tril(torch.ones(s1, s2), diagonal=-pre_tokens - 1 + s2 - s1)
        mask = (atten_mask_u + atten_mask_l).bool()
    elif sparse_mode == 5 or sparse_mode == 6:
        assert prefix is not None
        mask = torch.triu(torch.ones(b, 1, s1, s2), diagonal=s2 - s1 + 1)
        for i in range(0, b):
            mask[i, :, :, :prefix[i]] = 0
        mask = mask.bool()
    elif sparse_mode == 6:
        upper = torch.triu(torch.ones(2048, 2048), diagonal=1)
        lower = torch.cat((torch.zeros(1024, 1024), torch.ones(1024, 1024)), dim=1)
        mask = torch.cat((upper, lower), dim=0).bool()
    elif sparse_mode in [7, 8]:
        if index == band_index:
            atten_mask_u = torch.triu(torch.ones(s1, s2), diagonal=next_tokens + 1 + s2 - s1)
            atten_mask_l = torch.tril(torch.ones(s1, s2), diagonal=-pre_tokens - 1 + s2 - s1)
            mask = (atten_mask_u + atten_mask_l).bool()
        else:
            mask = torch.triu(torch.ones(s1, s2), diagonal=s2 - s1 + 1 if sparse_mode == 7 else 1).bool()
    return mask


def generate_qkv(b, n1, n2, s1, s2, d, d_v, d_rope, input_layout, dtype,
                 q_range=None, k_range=None, v_range=None):
    """生成BNSD排布的 Q/K/V tensor。
    value_range 为 (low, high) 时使用均匀分布随机值；省略则保持原有全10固定值（调试用）。
    """
    def _make_tensor(shape, seed, value_range):
        gen = torch.Generator().manual_seed(seed)
        if value_range is not None:
            lo, hi = float(value_range[0]), float(value_range[1])
            return (torch.rand(shape, generator=gen) * (hi - lo) + lo).to(dtype)
        return torch.randint(10, 11, shape, generator=gen, dtype=torch.int).to(dtype)

    q = _make_tensor((b, n1, s1, d),    42, q_range)
    k = _make_tensor((b, n2, s2, d),    43, k_range)
    v = _make_tensor((b, n2, s2, d_v),  44, v_range)

    # MLA:d=128, d_rope=64
    q_rope = torch.randn(b, n1, s1, d_rope, generator=torch.Generator().manual_seed(45), dtype=dtype)
    k_rope = torch.randn(b, n2, s2, d_rope, generator=torch.Generator().manual_seed(46), dtype=dtype)

    qf = torch.cat((q, q_rope), -1)
    kf = torch.cat((k, k_rope), -1)

    return q, k, v, q_rope, k_rope, qf, kf


# ======================================================================================================================
# 三方对比工具类和函数（迁移自 test_compare.py）
# ======================================================================================================================

def print_log(data=None, level='INFO'):
    print("[%s] [%s]-%s:%s - %s" % (datetime.datetime.now().strftime(
        "%Y/%m/%d %H:%M:%S"), level, os.path.basename(sys._getframe().f_back.f_code.co_filename),
                                    str(sys._getframe().f_back.f_lineno).zfill(4), data))


class ConfigFmk:
    golden_inf_switch = False
    inf_clip_switch = False
    nan_toZero_switch = False


def get_pt_dtype(type_str):
    type_dict = {
        'fp32': torch.float32, 'fp16': torch.float16, 'fp64': torch.float64,
        'int8': torch.int8, 'int16': torch.int16, 'int32': torch.int32, 'int64': torch.int64,
        'uint8': torch.uint8, 'bool': torch.bool, 'complex64': torch.complex64,
        'complex128': torch.complex128, 'bf16': torch.bfloat16, 'uint1': torch.uint8,
        'float8_e4m3fn': torch.float32, 'float4_e2m1': torch.uint8, 'fp8_e8m0': torch.uint8,
        'complex32': torch.complex32
    }
    return type_dict.get(type_str, torch.float32)


class Result:
    def __init__(self, result_name, total_big_num, total_big_ratio, diff_big_max, diff_big_avg, diff_big_sum,
                 err_w1_num, err_w1_ratio, err_k1_num, err_k1_ratio, err_k5_num, err_k5_ratio, err_h1_num, err_h1_ratio,
                 total_small_num, total_small_ratio, err_small_num, err_small_ratio,
                 diff_rmse, rst_eb, diff_eb, bm_cmp_std,
                 num_total_nan=0, err_total_nan=0, num_total_inf=0, err_total_inf=0, num_total_ninf=0,
                 err_total_ninf=0,
                 diff_big_ratio_max=None, diff_big_ratio_avg=None, diff_big_ratio_rmse=None):
        self.result_name = result_name
        self.total_big_num = total_big_num
        self.total_big_ratio = total_big_ratio
        self.diff_big_max = diff_big_max
        self.diff_big_avg = diff_big_avg
        self.diff_big_sum = diff_big_sum
        self.err_w1_num = err_w1_num
        self.err_w1_ratio = err_w1_ratio
        self.err_k1_num = err_k1_num
        self.err_k1_ratio = err_k1_ratio
        self.err_k5_num = err_k5_num
        self.err_k5_ratio = err_k5_ratio
        self.err_h1_num = err_h1_num
        self.err_h1_ratio = err_h1_ratio
        self.total_small_num = total_small_num
        self.total_small_ratio = total_small_ratio
        self.err_small_num = err_small_num
        self.err_small_ratio = err_small_ratio
        self.diff_rmse = diff_rmse
        self.rst_eb = rst_eb
        self.diff_eb = diff_eb
        self.bm_cmp_std = bm_cmp_std
        self.num_total_nan = num_total_nan
        self.err_total_nan = err_total_nan
        self.num_total_inf = num_total_inf
        self.err_total_inf = err_total_inf
        self.num_total_ninf = num_total_ninf
        self.err_total_ninf = err_total_ninf
        self.diff_big_ratio_max = diff_big_ratio_max
        self.diff_big_ratio_avg = diff_big_ratio_avg
        self.diff_big_ratio_rmse = diff_big_ratio_rmse

    def print_result(self):
        print_log(f"正在打印结果：{self.result_name}")
        print_log(f" 大值总数：{self.total_big_num}")
        print_log(f" 大值占比：{self.total_big_ratio:.2%}")
        print_log(f" 大值最大绝对误差：{self.diff_big_max:.8f}")
        print_log(f" 大值平均绝对误差：{self.diff_big_avg:.8f}")
        print_log(f" 大值绝对误差总和：{self.diff_big_sum:.2f}")
        print_log(f" 大值最大相对误差：{self.diff_big_ratio_max:.8f}")
        print_log(f" 大值平均相对误差：{self.diff_big_ratio_avg:.8f}")
        print_log(f" 大值相对误差均方根（RMSE）：{self.diff_big_ratio_rmse:.8f}")
        print_log(f" 大值万分之1误差数：{self.err_w1_num}，占比{self.err_w1_ratio:.2%}")
        print_log(f" 大值千分之1误差数：{self.err_k1_num}，占比{self.err_k1_ratio:.2%}")
        print_log(f" 大值千分之5误差数：{self.err_k5_num}，占比{self.err_k5_ratio:.2%}")
        print_log(f" 大值百分之1误差数：{self.err_h1_num}，占比{self.err_h1_ratio:.2%}")
        print_log(f" 小值总数：{self.total_small_num}")
        print_log(f" 小值占比：{self.total_small_ratio:.2%}")
        print_log(f" 小值错误数：{self.err_small_num}，占比{self.err_small_ratio:.2%}")
        print_log(f" 误差均方根（RMSE）：{self.diff_rmse:.8f}")
        print_log(f" 均衡性偏差计数：{self.rst_eb}")
        print_log(f" 均衡性diff总和：{self.diff_eb:.8f}")
        if (self.num_total_nan + self.num_total_inf + self.num_total_ninf != 0) or \
                (self.err_total_nan + self.err_total_inf + self.err_total_ninf != 0) or True:
            print_log(f" golden nan总数：{self.num_total_nan}")
            print_log(f" nan误差数：{self.err_total_nan}")
            print_log(f" golden inf总数：{self.num_total_inf}")
            print_log(f" inf误差数：{self.err_total_inf}")
            print_log(f" golden -inf总数：{self.num_total_ninf}")
            print_log(f" -inf误差数：{self.err_total_ninf}")

    def check_result_debug(self, benchmark, new=False, output_dtype='fp16'):
        if new:
            lo_bound = 0.0
            if output_dtype == 'fp16':
                lo_bound = 2 ** (-11)
            if output_dtype == 'bf16':
                lo_bound = 2 ** (-8)
            benchmark.diff_big_ratio_max = max(benchmark.diff_big_ratio_max, lo_bound)
            benchmark.diff_big_ratio_avg = max(benchmark.diff_big_ratio_avg, lo_bound)
            benchmark.diff_rmse = max(benchmark.diff_rmse, lo_bound)
        reason_str = ''
        if self.diff_big_ratio_max > benchmark.diff_big_ratio_max * self.bm_cmp_std['max_re_rtol']:
            reason_str += ' diff_big_ratio_max error/'
        elif self.diff_big_ratio_max > benchmark.diff_big_ratio_max:
            reason_str += ' diff_big_ratio_max warning/'
        if self.diff_big_ratio_avg > benchmark.diff_big_ratio_avg * self.bm_cmp_std['avg_re_rtol']:
            reason_str += ' diff_big_ratio_avg error/'
        elif self.diff_big_ratio_avg > benchmark.diff_big_ratio_avg:
            reason_str += ' diff_big_ratio_avg warning/'
        if not new:
            if self.diff_big_sum > benchmark.diff_big_sum * self.bm_cmp_std['avg_re_rtol']:
                reason_str += ' diff_big_sum error/'
            elif self.diff_big_sum > benchmark.diff_big_sum:
                reason_str += ' diff_big_sum warning/'
            if self.diff_big_ratio_rmse > benchmark.diff_big_ratio_rmse * self.bm_cmp_std['rmse_rtol']:
                reason_str += ' diff_big_ratio_rmse error/'
            elif self.diff_big_ratio_rmse > benchmark.diff_big_ratio_rmse:
                reason_str += ' diff_big_ratio_rmse warning/'
        if self.err_small_num > benchmark.err_small_num * self.bm_cmp_std['avg_re_rtol']:
            reason_str += ' err_small_num error/'
        elif self.err_small_num > benchmark.err_small_num:
            reason_str += ' err_small_num warning/'
        if self.diff_rmse > benchmark.diff_rmse * self.bm_cmp_std['rmse_rtol']:
            reason_str += ' diff_rmse error/'
        elif self.diff_rmse > benchmark.diff_rmse:
            reason_str += ' diff_rmse warning/'
        if self.err_total_nan > benchmark.err_total_nan:
            reason_str += ' err_total_nan error/'
        elif self.err_total_nan > 0:
            reason_str += ' err_total_nan warning/'
        if self.err_total_inf > benchmark.err_total_inf or self.err_total_ninf > benchmark.err_total_ninf:
            reason_str += ' err_total_inf error/'
        elif self.err_total_inf > 0 or self.err_total_ninf > 0:
            reason_str += ' err_total_inf warning'
        return reason_str

    def check_result(self, benchmark, new=False, output_dtype='fp16'):
        print(f"comparing result: {self.result_name} VS {benchmark.result_name}")
        reason_str = self.check_result_debug(benchmark, new, output_dtype)
        if 'error' in reason_str:
            print(self.result_name + ' compare result: error')
            return 'Failed', reason_str
        elif 'warning' in reason_str:
            print(self.result_name + ' compare result: warning')
            return 'warning', reason_str
        else:
            print(self.result_name + ' compare result: ok')
        return 'Pass', ''


def checkResultNew(params, value, golden, name, output_dtype):
    cfg_fk = ConfigFmk()
    debug_switch = False
    red = params['red_range'][output_dtype]
    red_list_str = red.split("/")
    red_list = [float(i) for i in red_list_str]
    bm_cmp_std = params['bm_cmp_std'][output_dtype]

    print(f"info：开始计算 {name} 精度。")
    if value.shape == golden.shape:
        if cfg_fk.golden_inf_switch:
            golden[golden > torch.finfo(value.dtype).max] = torch.inf
            golden[golden < torch.finfo(value.dtype).min] = -torch.inf

        if cfg_fk.inf_clip_switch:
            golden[torch.isinf(golden)] = torch.finfo(value.dtype).max
            value[torch.isinf(value)] = torch.finfo(value.dtype).max
        if cfg_fk.nan_toZero_switch:
            golden[torch.isnan(golden)] = 0
            value[torch.isnan(value)] = 0

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

        golden[torch.isinf(golden)] = 1
        value[torch.isinf(value)] = 1
        golden[torch.isnan(golden)] = 1
        value[torch.isnan(value)] = 1

        total_big_num = torch.sum(golden.abs() >= bm_cmp_std['small_value'])
        total_big_ratio = total_big_num / golden.numel()

        value_big = value.clone()
        value_big[golden.abs() < bm_cmp_std['small_value']] = 1
        golden_big = golden.clone()
        golden_big[golden.abs() < bm_cmp_std['small_value']] = 1

        diff_big = torch.abs(value_big.sub(golden_big))
        diff_big_max = diff_big.max()
        diff_big_sum = diff_big.sum()
        diff_big_avg = diff_big_sum / total_big_num
        diff_big_rmse = torch.sqrt(torch.mean(torch.square(diff_big)))

        diff_big_ratio = diff_big / golden_big.abs()
        diff_big_ratio_max = diff_big_ratio.max()
        diff_big_ratio_avg = diff_big_ratio.sum() / total_big_num
        diff_big_ratio_rmse = torch.sqrt(torch.mean(torch.square(diff_big_ratio)))

        err_w1_num = torch.sum(diff_big_ratio > red_list[0])
        err_w1_ratio = err_w1_num / total_big_num
        err_k1_num = torch.sum(diff_big_ratio > red_list[1])
        err_k1_ratio = err_k1_num / total_big_num
        err_k5_num = torch.sum(diff_big_ratio > red_list[2])
        err_k5_ratio = err_k5_num / total_big_num
        err_h1_num = torch.sum(diff_big_ratio > red_list[3])
        err_h1_ratio = err_h1_num / total_big_num

        if debug_switch:
            print_log(f" 大值总数：{total_big_num}")
            print_log(f" 大值占比：{total_big_ratio:.2%}")
            print_log(f" 大值绝对误差均方根（RMSE）：{diff_big_rmse:.8f}")
            print_log(f" 大值平均绝对误差：{diff_big_avg:.8f}")
            print_log(f" 大值绝对误差总和：{diff_big_sum:.2f}")
            print_log(f" 大值最大相对误差：{diff_big_ratio_max:.8f}")
            print_log(f" 大值平均相对误差：{diff_big_ratio_avg:.8f}")
            print_log(f" 大值相对误差均方根（RMSE）：{diff_big_ratio_rmse:.8f}")
            print_log(f" 大值{red_list[0]}误差数：{err_w1_num}，占比{err_w1_ratio:.2%}")
            print_log(f" 大值{red_list[1]}误差数：{err_k1_num}，占比{err_k1_ratio:.2%}")
            print_log(f" 大值{red_list[2]}误差数：{err_k5_num}，占比{err_k5_ratio:.2%}")
            print_log(f" 大值{red_list[3]}误差数：{err_h1_num}，占比{err_h1_ratio:.2%}")

        total_small_num = torch.sum(golden.abs() < bm_cmp_std['small_value'])
        total_small_ratio = total_small_num / golden.numel()

        value_small = value.clone()
        value_small[golden.abs() > bm_cmp_std['small_value']] = 1
        golden_small = golden.clone()
        golden_small[golden.abs() > bm_cmp_std['small_value']] = 1

        diff_small = torch.abs(value_small.sub(golden_small))
        err_small_num = torch.sum(diff_small > bm_cmp_std['small_value_atol'])
        err_small_ratio = err_small_num / total_small_num

        if debug_switch:
            print_log(f" 小值总数：{total_small_num}")
            print_log(f" 小值占比：{total_small_ratio:.2%}")
            print_log(f" 小值错误数：{err_small_num}，占比{err_small_ratio:.2%}")

        diff = torch.abs(value.sub(golden))
        diff_rmse = torch.sqrt(torch.mean(torch.square(diff)))
        if debug_switch:
            print_log(f" 绝对误差均方根（RMSE）：{diff_rmse:.8f}")

        eb_bigger = torch.sum(value > golden)
        eb_smaller = torch.sum(value < golden)
        rst_eb = torch.abs(eb_bigger.sub(eb_smaller))
        diff_eb = torch.sum(value.sub(golden))
        if debug_switch:
            print_log(f" 均衡性偏差计数：{rst_eb}")
            print_log(f" 均衡性diff总和：{diff_eb:.8f}")

        return Result(name,
                      total_big_num.item(), total_big_ratio.item(),
                      diff_big_max.item(), diff_big_avg.item(), diff_big_sum.item(),
                      err_w1_num.item(), err_w1_ratio.item(),
                      err_k1_num.item(), err_k1_ratio.item(),
                      err_k5_num.item(), err_k5_ratio.item(),
                      err_h1_num.item(), err_h1_ratio.item(),
                      total_small_num.item(), total_small_ratio.item(),
                      err_small_num.item(), err_small_ratio.item(),
                      diff_rmse.item(), rst_eb.item(), diff_eb.item(),
                      bm_cmp_std,
                      num_total_nan.item(), err_total_nan.item(),
                      num_total_inf.item(), err_total_inf.item(),
                      num_total_ninf.item(), err_total_ninf.item(),
                      diff_big_ratio_max=diff_big_ratio_max.item(),
                      diff_big_ratio_avg=diff_big_ratio_avg.item(),
                      diff_big_ratio_rmse=diff_big_ratio_rmse.item())
    else:
        print_log(f"error: {name}计算结果错误，shape与标杆不匹配，用例执行失败！！！")
        print_log(f"debug: value {value.shape}")
        print_log(f"debug: golden {golden.shape}")
        raise ValueError(f"Shape mismatch: value {value.shape}, golden {golden.shape}")


def data_compare_benchmark_new(params, npu_output, cpu_output, cpu_golden, output_dtype, i):
    cfg_fk = ConfigFmk()
    real_data = npu_output.flatten()
    data_compe = cpu_output.flatten()
    cpu_golden = cpu_golden.flatten()
    if real_data.size == 0 and real_data.size == data_compe.size and real_data.size == cpu_golden.size:
        print_log('The npu_output is [],and it is same as bm_output/cpu_golden, the result of data_compare is \"Pass\"')
        return "Pass", 100.0, 0
    max_error = ''
    result = "Failed"

    if real_data.size != data_compe.size:
        print_log('Error,the size of npu output[%s] and benchmark[%s] is not equal.' % (real_data.size, data_compe.size))
        return result, '', max_error

    if real_data.size != cpu_golden.size:
        print_log('Error,the size of npu output[%s] and golden[%s] is not equal.' % (real_data.size, cpu_golden.size))
        return result, '', max_error

    def to_target(t, dtype_str):
        if dtype_str in ("float8_e5m2", "float8_e4m3fn", "hifloat8"):
            qDtype = get_pt_dtype(params["dtype_input"][0])
            return t.to(qDtype)
        elif dtype_str == "bfloat16":
            return t.to(torch.bfloat16)
        elif dtype_str == "fp16":
            return t.to(torch.float16)
        else:
            return t

    npu_res = to_target(torch.from_numpy(real_data).detach(), output_dtype)
    benchmark_res = to_target(torch.from_numpy(data_compe).detach(), output_dtype)
    golden = torch.from_numpy(cpu_golden).detach()

    rst_npu = checkResultNew(params, npu_res, golden, params["case_name"], output_dtype)
    rst_npu.print_result()

    golden = torch.from_numpy(cpu_golden).detach()
    rst_gpu = checkResultNew(params, benchmark_res, golden, params["case_name"], output_dtype)
    rst_gpu.print_result()

    str1, str2 = rst_npu.check_result(rst_gpu, True, output_dtype)

    data = [params['op_name'], params['case_name'], f"{rst_npu.total_big_num}", f"{rst_npu.total_big_ratio:.2%}",
            f"{rst_npu.err_w1_ratio:.2%}", f"{rst_npu.err_k1_ratio:.2%}", f"{rst_npu.err_k5_ratio:.2%}",
            f"{rst_npu.err_h1_ratio:.2%}",
            f"{rst_npu.diff_big_max:.8f}", f"{rst_npu.diff_big_avg:.8f}", f"{rst_npu.diff_big_sum:.2f}",
            f"{rst_npu.diff_big_ratio_max:.8f}", f"{rst_npu.diff_big_ratio_avg:.8f}",
            f"{rst_npu.diff_big_ratio_rmse:.8f}",
            f"{rst_npu.total_small_num}", f"{rst_npu.total_small_ratio:.2%}", f"{rst_npu.err_small_num}",
            f"{rst_npu.err_small_ratio:.2%}", f"{rst_npu.num_total_nan:.2f}",
            f"{rst_npu.err_total_nan:.2f}", f"{rst_npu.num_total_inf:.2f}", f"{rst_npu.err_total_inf:.2f}",
            f"{rst_npu.num_total_ninf:.2f}", f"{rst_npu.err_total_ninf:.2f}",
            f"{rst_npu.diff_rmse:.8f}", f"{rst_npu.rst_eb}", f"{rst_npu.diff_eb:.8f}",
            f"{rst_gpu.total_big_num}",
            f"{rst_gpu.total_big_ratio:.2%}",
            f"{rst_gpu.err_w1_ratio:.2%}", f"{rst_gpu.err_k1_ratio:.2%}", f"{rst_gpu.err_k5_ratio:.2%}",
            f"{rst_gpu.err_h1_ratio:.2%}",
            f"{rst_gpu.diff_big_max:.8f}", f"{rst_gpu.diff_big_avg:.8f}", f"{rst_gpu.diff_big_sum:.2f}",
            f"{rst_gpu.diff_big_ratio_max:.8f}", f"{rst_gpu.diff_big_ratio_avg:.8f}",
            f"{rst_gpu.diff_big_ratio_rmse:.8f}",
            f"{rst_gpu.total_small_num}", f"{rst_gpu.total_small_ratio:.2%}", f"{rst_gpu.err_small_num}",
            f"{rst_gpu.err_small_ratio:.2%}", f"{rst_gpu.num_total_nan:.2f}",
            f"{rst_gpu.err_total_nan:.2f}", f"{rst_gpu.num_total_inf:.2f}", f"{rst_gpu.err_total_inf:.2f}",
            f"{rst_gpu.num_total_ninf:.2f}", f"{rst_gpu.err_total_ninf:.2f}", f"{rst_gpu.diff_rmse:.8f}",
            f"{rst_gpu.rst_eb}", f"{rst_gpu.diff_eb:.8f}", f"{str1}", f"{str2}",
            f"output_{i}"]

    return str1, str2, data
