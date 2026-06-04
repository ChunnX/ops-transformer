# -*- coding: utf-8 -*-
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


import os
import numpy as np
import torch


import time
import math
import random
import libs.tools as tools
import libs.training.run_aclnn as training
from libs.tools import set_timeout
import ctypes
import copy
import tensorflow as tf
from libs.generate_hifloat8_data import trans_np_hifuint8_tensor_to_float32
from libs.generate_hifloat8_data import trans_np_float_tensor_to_hifuint8
from ml_dtypes import float8_e5m2
from ml_dtypes import float8_e4m3fn
from ml_dtypes import float8_e8m0fnu
cfg_fk = tools.ConfigFmk()


def find_output_value(input_list, output_value_list):
    temp_list = []
    is_list_or_tuple = False
    for item in input_list:
        if not isinstance(item, list) and not isinstance(item, tuple):
            output_value_list.append(item)
        else:
            for elment in item:
                temp_list.append(elment)
    if len(temp_list):
        is_list_or_tuple = True
    return is_list_or_tuple, temp_list


def get_output_list(is_list_or_tuple, input_list, output_value_list):
    while True:
        if not is_list_or_tuple:
            return output_value_list
        else:
            is_list_or_tuple, input_list = find_output_value(input_list, output_value_list)


def _random_fill_tensor(tensor, shape, random_number, value=0):
    for i in range(0, random_number):
        point = []
        for k in range(0, len(shape)):
            point.append(random.randint(1, shape[k]) - 1)
        tensor[point[0], point[1]] = value
    return tensor


def _t_broadcastKV_sigle(numHeads, numKeyValueHeads, kv_tensor):
    factor = numHeads // numKeyValueHeads
    kv_shape = kv_tensor.shape
    B = kv_shape[0]
    S = kv_shape[2]
    D = kv_shape[3]
    kv_res = torch.zeros([B, numHeads, S, D])
    for i in range(numHeads):
        j = i // factor
        kv_res[:, i:i + 1, :, :] = kv_tensor[:, j:j + 1, :, :]
    return kv_res, kv_res.shape

def _np_broadcastKV_sigle(numHeads, numKeyValueHeads, kv_tensor,dtype):
    factor = numHeads // numKeyValueHeads
    kv_shape = kv_tensor.shape
    B = kv_shape[0]
    S = kv_shape[2]
    D = kv_shape[3]
    kv_res = np.zeros([B, numHeads, S, D], dtype=dtype)
    for i in range(numHeads):
        j = i // factor
        kv_res[:, i:i + 1, :, :] = kv_tensor[:, j:j + 1, :, :]
    return kv_res, kv_res.shape

def broadcast_kv_dequant_tensor(tensor, numKeyValueHeads, numheads, dtype=np.float16):
    factor = numheads // numKeyValueHeads
    shape = tensor.shape
    D = shape[3]
    if 'bfloat16' in str(dtype):
        tensor_new = np.zeros([2, numheads, 1, D], dtype=tf.bfloat16.as_numpy_dtype)
    else:
        tensor_new = np.zeros([2, numheads, 1, D], dtype=dtype)
    for i in range(numheads):
        j = i // factor
        tensor_new[:, i:i + 1, :, :] = tensor[:, j:j + 1, :, :]
    return tensor_new

def broadcast_kv_split_dequant_tensor(tensor, numKeyValueHeads, numheads, dtype=np.float16):
    factor = numheads // numKeyValueHeads

    if hasattr(tensor, 'numpy'):
        tensor = tensor.numpy()

    shape = tensor.shape

    if len(shape) == 1:
        tensor = tensor.reshape(numKeyValueHeads, 1, -1)
    elif len(shape) == 2:
        tensor = tensor.reshape(shape[0], 1, shape[1])
    else:
        pass

    shape = tensor.shape
    D = shape[2]

    if 'bfloat16' in str(dtype):
        np_dtype = tf.bfloat16.as_numpy_dtype
    else:
        np_dtype = dtype

    tensor_new = np.repeat(tensor, factor, axis=0).astype(np_dtype)

    return tensor_new

def broadcast_qkv_split_dequant_tensor(tensor, numKeyValueHeads, numheads):
    # 每个 key/value head 要复制 factor 次
    factor = numheads // numKeyValueHeads
    head_to_repeat = tensor[0:1, :, :]  # shape: (1, 1, D)
    tensor_new = np.tile(tensor, (1, factor, 1, 1))  # (numheads, 1, D)
    return tensor_new

def _create_mask_right_down(m_shape, pre_tokens, next_tokens, actualSeqLengths, actualSeqLengthsKV, actualprefixKV,prefix_kvs, batch, numheads, kvs_list,m_dtype):

    mask_s_q = m_shape[0]
    mask_s_kv = m_shape[1]

    next_tokens_list = []
    re_mask_batch = []
    for i in range(batch):
        if len(actualSeqLengths) == 0:
            S1 = mask_s_q
        else:
            S1 = actualSeqLengths[i]
        if len(actualSeqLengthsKV) != 0:
            S2 = actualSeqLengthsKV[i]+actualprefixKV
        elif len(kvs_list) > 1:
            S2 = kvs_list[i]+actualprefixKV
        else:
            S2 = mask_s_kv-prefix_kvs+actualprefixKV
        next_tokens = S2 - S1
        next_tokens_list.append(next_tokens)
        atten_masks = _create_mask(m_shape, pre_tokens, next_tokens)
        re_mask_batch.append(atten_masks)
    return re_mask_batch, next_tokens_list

def _create_mask(m_shape, pre_tokens, next_tokens):
    next_masks = np.triu(np.ones(m_shape, dtype='uint8'), k=1 + int(next_tokens))  # 生成下三角全是0的矩阵
    pre_mask = np.tril(np.ones(m_shape, dtype='uint8'), k=-1 - int(pre_tokens))  # 生成上三角全是0的矩阵
    atten_masks = pre_mask + next_masks

    return atten_masks
def _create_mask_no_sparse(m_shape, npu_m_shape, pre_tokens, next_tokens, batch, numheads, m_dtype, random_ones=0):
    re_mask_batch = []
    re_mask_npu_batch = []

    pad_flag = False
    npu_mask = None
    if m_shape[0] !=npu_m_shape[0] or m_shape[1]!=npu_m_shape[1]:
        pad_flag = True
        npu_mask = np.ones(npu_m_shape, dtype='uint8')
    cpu_mask = _create_mask(m_shape, pre_tokens, next_tokens)
    if pad_flag:
        if batch == None:
            cpu_mask = _random_fill_tensor(cpu_mask, m_shape, random_ones, 1)
            npu_mask[:cpu_mask.shape[0], :cpu_mask.shape[1]] = cpu_mask
            return cpu_mask, npu_mask
        for i in range(batch):
            re_mask_num = []
            re_mask_npu_num = []
            re_mask = _random_fill_tensor(cpu_mask, m_shape, random_ones, 1)

            npu_mask[:re_mask.shape[0], :re_mask.shape[1]] = re_mask
            if numheads:
                # cpu
                re_mask_num.append(re_mask)
                re_mask_batch.append(re_mask_num)
                # npu

                re_mask_npu_num.append(npu_mask)
                re_mask_npu_batch.append(re_mask_npu_num)
            else:
                re_mask_batch.append(re_mask)
                re_mask_npu_batch.append(npu_mask)
        cpu_mask = np.array(re_mask_batch).astype(m_dtype)
        npu_mask = np.array(re_mask_npu_batch).astype(m_dtype)
        return cpu_mask, npu_mask
    else:
        if batch ==None:
            cpu_mask = _random_fill_tensor(cpu_mask, m_shape, random_ones, 1)
            return cpu_mask,cpu_mask
        for i in range(batch):
            re_mask_num = []
            re_mask = _random_fill_tensor(cpu_mask, m_shape, random_ones, 1)
            if numheads:
                re_mask_num.append(re_mask)
                re_mask_batch.append(re_mask_num)
            else:
                re_mask_batch.append(re_mask)

        cpu_mask = np.array(re_mask_batch).astype(m_dtype)
        return cpu_mask,cpu_mask
def _create_mask_left_up(m_shape, pre_tokens, next_tokens, batch, numheads, m_dtype, random_ones=0):
    re_mask_batch = []
    attentionmask = _create_mask(m_shape, pre_tokens, next_tokens)
    for i in range(batch):
        re_mask_batch.append(attentionmask)
    return re_mask_batch

def _create_mask_band(m_shape, pre_tokens, next_tokens, actualSeqLengths, actualSeqLengthsKV,actualprefixKV, prefix_kvs,batch, numheads, kvs_list, m_dtype):

    mask_s_q = m_shape[0]
    mask_s_kv = m_shape[1]

    pre_tokens_list = []
    next_tokens_list = []
    re_mask_batch = []
    for i in range(batch):
        if len(actualSeqLengths) == 0:
            S1 = mask_s_q
        else:
            S1 = actualSeqLengths[i]
        if len(actualSeqLengthsKV) != 0:
            S2 = actualSeqLengthsKV[i]+actualprefixKV
        elif len(kvs_list) > 1:
            S2 = kvs_list[i]+actualprefixKV
        else:
            S2 = mask_s_kv-prefix_kvs+actualprefixKV
        pre_tokens_new = S1-S2+pre_tokens
        pre_tokens_list.append(pre_tokens_new)

        next_tokens_new = S2-S1+next_tokens
        next_tokens_list.append(next_tokens_new)
        atten_masks = _create_mask(m_shape, pre_tokens_new, next_tokens_new)
        re_mask_batch.append(atten_masks)
    return re_mask_batch, pre_tokens_list, next_tokens_list
def _t_gpu_create_random_mask_by_spars(cpu_m_shape, npu_m_shape, m_dtype, pre_tokens, next_tokens, actualSeqLengths, actualSeqLengthsKV, batch=1, numheads=1, sp_mode=0, random_ones=0):
    # mask shape [sq,skv]  #mshape  npu  fshape cpu
    print(f"[_create_random_mask_by_spars] full_m_shape:{cpu_m_shape} m_shape:{npu_m_shape} datype:{m_dtype} pret:{pre_tokens} nextt:{next_tokens} sp_mode:{sp_mode}")
    cpu_mask = None
    if sp_mode == 0:
        cpu_mask = _create_mask_left_up(cpu_m_shape, pre_tokens, next_tokens, m_dtype, random_ones)
    if sp_mode == 1:
        pre_tokens = 214748647
        next_tokens = 214748647
        cpu_mask = _create_mask_left_up(cpu_m_shape, pre_tokens, next_tokens, m_dtype, random_ones)
    if sp_mode == 2:
        pre_tokens = 214748647
        next_tokens = 0
        print(f"[_create_random_mask_by_spars] sp_mode is 2 npu mask shape:{npu_m_shape}")
        cpu_mask = _create_mask_left_up(cpu_m_shape, pre_tokens, next_tokens, m_dtype)
    if sp_mode == 3:
        pre_tokens = 214748647
        next_tokens = 0
        print(f"[_create_random_mask_by_spars] sp_mode is 2 npu mask shape:{npu_m_shape}")
        cpu_mask, nexttokens = _create_mask_right_down(cpu_m_shape, pre_tokens, next_tokens, actualSeqLengths, actualSeqLengthsKV, batch, numheads, m_dtype)
    if sp_mode ==4:
        cpu_mask = _create_mask_left_up(cpu_m_shape, pre_tokens, next_tokens, m_dtype)
    cpu_mask = torch.from_numpy(cpu_mask)
    return cpu_mask.to(m_dtype)



def _create_random_mask_by_spars(cpu_m_shape, npu_m_shape, m_dtype, pre_tokens, next_tokens, actualSeqLengths, actualSeqLengthsKV, actualprefixKV,prefix_kvs,kvs_list,batch=1, numheads=1, sp_mode=0, random_ones=0):
    # mask shape [sq,skv]  #mshape  npu  fshape cpu
    print(
        f"[_create_random_mask_by_spars] full_m_shape:{cpu_m_shape} m_shape:{npu_m_shape} datype:{m_dtype} pret:{pre_tokens} nextt:{next_tokens} sp_mode:{sp_mode}")
    if sp_mode == 0:
        cpu_mask, npu_mask = _create_mask_no_sparse(cpu_m_shape, npu_m_shape, pre_tokens, next_tokens, batch, numheads,
                                                    m_dtype, random_ones)
        return cpu_mask, npu_mask.astype(m_dtype), pre_tokens, next_tokens
    if sp_mode == 1:
        print(f"[_create_random_mask_by_spars] sp_mode is 1 return all zero mask")
        pre_tokens = 214748647
        next_tokens = 214748647
        cpu_mask, npu_mask = _create_mask_no_sparse(cpu_m_shape, npu_m_shape, pre_tokens, next_tokens, batch, numheads,
                                                    m_dtype, random_ones)
        return cpu_mask, npu_mask.astype(m_dtype), pre_tokens, next_tokens



    if sp_mode == 2:
        pre_tokens = 214748647
        next_tokens = 0
        print(f"[_create_random_mask_by_spars] sp_mode is 2 npu mask shape:{npu_m_shape}")
        npu_mask = np.triu(np.ones(npu_m_shape), k=1)
        cpu_mask = _create_mask_left_up(cpu_m_shape, pre_tokens, next_tokens, batch, numheads, m_dtype)
        return cpu_mask, npu_mask.astype(m_dtype), pre_tokens, next_tokens
    if sp_mode == 3:  # rightdown
        pre_tokens = 214748647
        print(f"[_create_random_mask_by_spars] sp_mode is 3 npu mask shape:{npu_m_shape}")
        npu_mask = np.triu(np.ones(npu_m_shape), k=1)
        cpu_mask, next_tokens_new = _create_mask_right_down(cpu_m_shape, pre_tokens, next_tokens, actualSeqLengths,
                                                            actualSeqLengthsKV, actualprefixKV, prefix_kvs, batch, numheads, kvs_list, m_dtype)
        return cpu_mask, npu_mask.astype(m_dtype), pre_tokens, next_tokens_new
    if sp_mode == 4:
        npu_mask = np.triu(np.ones(npu_m_shape), k=1)
        cpu_mask, pre_tokens_new, next_tokens_new = _create_mask_band(cpu_m_shape, pre_tokens, next_tokens,
                                                                      actualSeqLengths, actualSeqLengthsKV, actualprefixKV, prefix_kvs,batch,
                                                                      numheads, kvs_list, m_dtype)
        return cpu_mask, npu_mask.astype(m_dtype), pre_tokens_new, next_tokens_new
def t_bsh_to_bsnd(tensor, bsh_shape, headnums, actSeqLength, inputLayout="BSH"):
    """
        gpu golden  need  bsnd
    """
    if inputLayout == "SH":  # SH格式
        print("SH")
        if len(actSeqLength) == 0:
            B = 1
            S = bsh_shape[0]
        else:
            B = len(actSeqLength)
            S = sum(actSeqLength)
        H = bsh_shape[1]
        N = headnums
        D = H // N
        if B == 1:
            return tensor.reshape(B, S, N, D), [B, S, N, D]
        else:
            tmp = torch.zeros((B, S, N, D))
            sums = 0
            for i in range(B):
                acts = actSeqLength[i]
                for j in range(acts):
                    print(sums + j)
                    tmp[i:i + 1, j:j + 1] = tensor[sums + j:sums + j + 1, :].reshape(N, D)
                sums += acts
            return tmp, [B, S, N, D]
    elif inputLayout == "BSH":
        print("BSH")
        B = bsh_shape[0]
        S = bsh_shape[1]
        H = bsh_shape[2]
        N = headnums
        D = H // N
        print("BSH", tensor.shape)
        return tensor.reshape(B, S, N, D), [B, S, N, D]
    elif inputLayout == "NSD":
        print("NSD")
        B = 1
        S = bsh_shape[1]
        N = bsh_shape[0]
        D = bsh_shape[2]
        H = D * N
        return tensor.reshape(B, N, S, D).permute(0, 2, 1, 3), [B, S, N, D]
    elif inputLayout == "TND":
        N = bsh_shape[1]
        D = bsh_shape[2]
        B = len(actSeqLength)
        S = max(actSeqLength)
        new_tensor = np.zeros((B, S, N, D), dtype=tensor.dtype)
        t_start = 0
        for b_index in range(B):
            act_s = actSeqLength[b_index]
            t_end = t_start + act_s
            if act_s == 0:
                continue
            for s_index in range(act_s):
                new_tensor[b_index, s_index, 0:N, :] = tensor[t_start:t_end, :, :]
            t_start += act_s
        print(f"[INFO] trans tnd 2 bnsd: {tensor.shape} -> {new_tensor.shape}")
        return new_tensor, [B, S, N, D]
    else:
        print("BNSD", tensor.shape)
        return tensor.permute(0, 2, 1, 3), bsh_shape

def _t_trans_bsh_to_bnsd(tensor, bsh_shape, headnums, actSeqLength, inputLayout="BSH"):
    """
        gpu_small golden  need  bnsd
    """
    if inputLayout == "SH":  # SH格式
        print("SH")
        if len(actSeqLength) == 0:
            B = 1
            S = bsh_shape[0]
        else:
            B = len(actSeqLength)
            S = sum(actSeqLength)
        H = bsh_shape[1]
        N = headnums
        D = H // N
        print(B, N, S, D)
        if B == 1:
            return tensor.reshape(B, S, N, D).permute(0, 2, 1, 3), [B, N, S, D]
        else:
            tmp = torch.zeros((B, S, N, D))
            sums = 0
            for i in range(B):
                acts = actSeqLength[i]
                for j in range(acts):
                    tmp[i:i + 1, j:j + 1] = tensor[sums + j:sums + j + 1, :].reshape(N, D)
                sums += acts
            return tmp.permute(0, 2, 1, 3), [B, N, S, D]
    elif inputLayout == "BSH":
        print("BSH")
        B = bsh_shape[0]
        S = bsh_shape[1]
        H = bsh_shape[2]
        N = headnums
        D = H // N
        print("BSH", tensor.shape)
        return tensor.reshape(B, S, N, D).permute(0, 2, 1, 3), [B, N, S, D]
    elif inputLayout == "NSD":
        print("NSD")
        B = 1
        S = bsh_shape[1]
        N = bsh_shape[0]
        D = bsh_shape[2]
        H = D * N
        return tensor.reshape(B, N, S, D), [B, N, S, D]
    elif inputLayout == "TND":
        B = len(actSeqLength)
        S = max(actSeqLength)
        N = bsh_shape[1]
        D = bsh_shape[2]
        new_tensor = np.zeros((B, N, S, D), dtype=tensor.dtype)
        t_start = 0
        for b_index in range(B):
            act_s = actSeqLength[b_index]
            t_end = t_start + act_s
            if act_s == 0:
                continue
            for n_index in range(N):
                new_tensor[b_index, n_index, 0:act_s, :] = tensor[t_start:t_end, n_index, :]
            t_start += act_s
        print(f"[INFO] trans tnd 2 bnsd: {tensor.shape} -> {new_tensor.shape}")
        return new_tensor, [B, N, S, D]
    else:
        print("BNSD", tensor.shape)
        return tensor, bsh_shape

def trans_bnsd_to_bsh(tensor, shape):
    B = shape[0]
    N = shape[1]
    S = shape[2]
    D = shape[3]
    H = N * D
    return tensor.transpose(0, 2, 1, 3).reshape(B, S, H)


def np_bsh_to_bnsd(tensor, bsh_shape, headnums, actSeqLength, inputLayout="BSH"):
    """
    cpu golden  need  bnsd
    """
    if inputLayout == "SH":  # SH格式
        print("SH")
        if len(actSeqLength) == 0:
            B = 1
            S = bsh_shape[0]
        else:
            B = len(actSeqLength)
            S = sum(actSeqLength)
        H = bsh_shape[1]
        N = headnums
        D = H // N
        print(B, N, S, D)
        if B == 1:
            return tensor.reshape(B, S, N, D).transpose(0, 2, 1, 3), [B, N, S, D]
        else:
            tmp = np.zeros((B, S, N, D), dtype=tensor.dtype)
            sums = 0
            for i in range(B):
                acts = actSeqLength[i]
                for j in range(acts):
                    tmp[i:i + 1, j:j + 1] = tensor[sums + j:sums + j + 1, :].reshape(N, D)
                sums += acts
            return tmp.transpose(0, 2, 1, 3), [B, N, S, D]
    elif inputLayout == "BSH":
        print("BSH")
        B = bsh_shape[0]
        S = bsh_shape[1]
        H = bsh_shape[2]
        N = headnums
        D = H // N
        print("BSH", tensor.shape)
        return tensor.reshape(B, S, N, D).transpose(0, 2, 1, 3), [B, N, S, D]
    elif inputLayout == "NSD":
        print("NSD")
        B = 1
        S = bsh_shape[1]
        N = bsh_shape[0]
        D = bsh_shape[2]
        H = D * N
        return tensor.reshape(B, N, S, D), [B, N, S, D]
    elif inputLayout == "BSND":
        print("BSND")
        B = bsh_shape[0]
        S = bsh_shape[1]
        N = bsh_shape[2]
        D = bsh_shape[3]
        return tensor.transpose(0, 2, 1, 3), [B, N, S, D]
    elif inputLayout == "TND" or inputLayout == "NTD_TND" or inputLayout == "NTD":
        B = len(actSeqLength)
        S = max(actSeqLength)
        N = bsh_shape[1]
        D = bsh_shape[2]
        if inputLayout=="NTD_TND":
            N = bsh_shape[0]
            tensor = tensor.transpose(1, 0, 2)
        new_tensor = np.zeros((B, N, S, D), dtype=tensor.dtype)
        t_start = 0
        for b_index in range(B):
            act_s = actSeqLength[b_index]
            t_end = t_start + act_s
            if act_s == 0:
                continue
            for n_index in range(N):
                new_tensor[b_index, n_index, 0:act_s, :] = tensor[t_start:t_end, n_index, :]
            t_start += act_s
        print(f"[INFO] trans tnd 2 bnsd: {tensor.shape} -> {new_tensor.shape}")
        return new_tensor, [B, N, S, D]
    else:
        print(inputLayout)
        return tensor, bsh_shape


def get_slopes(n_heads):
    n = 2 ** math.floor(math.log2(n_heads))
    m_0 = 2.0 ** (-8.0 / n)
    m = torch.pow(m_0, torch.arange(1, 1 + n))
    if n < n_heads:
        m_hat_0 = 2.0 ** (-4.0 / n)
        m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (n_heads - n), 2))
        m = torch.cat([m, m_hat])
    return m


def alibi_biases(pre_shift_shape):
    # [0,  1, 2]
    # [-1, 0, 1]
    # [-2,-1, 0]
    q_seq = pre_shift_shape[2]
    kv_seq = pre_shift_shape[3]
    alibi_biases = np.zeros([q_seq, kv_seq])
    for x in range(0, kv_seq):
        for y in range(0, q_seq):
            alibi_biases[y, x] = abs(x - y - (kv_seq - q_seq))*(-1)
    return alibi_biases

def get_all_alibi(numHeads,pre_shift_shape,pse_dtype=np.float32): #输入是bnss或者是1nss B, Nq, Sq, Skv, pse_layout, pse_type, pse_dtype,pre_shift_shape
    m = get_slopes(numHeads)  #m  系数
    m = m.numpy()
    alibi_biase = alibi_biases(pre_shift_shape)
    pse_shift = np.zeros(pre_shift_shape,dtype=np.float32)

    for n in range(numHeads):
        pse_shift[:, n:n + 1, :, :] = alibi_biase * m[n]
    return pse_shift.astype(pse_dtype)

def _np_broadcast_pseShift_n(pse_shift_tensor, pse_shift_shape, q_batch):#pre_shift, pre_shift_shape, q_bnsd_shape[0]
    print(f"broadcast_mask_n:mask shape:{pse_shift_shape} q_batch:{q_batch}")
    #1nss or bnss
    B_m = pse_shift_shape[0]
    if B_m == q_batch:
        return pse_shift_tensor
    else:
        B = q_batch
        # pse_res = np.zeros([B, pse_shift_shape[1],pse_shift_shape[2], pse_shift_shape[3]])
        pse_res = np.zeros([B, pse_shift_tensor.shape[1], pse_shift_tensor.shape[2], pse_shift_tensor.shape[3]],dtype=np.float32)
        for i in range(B):
            pse_res[i:i+1] = pse_shift_tensor[0]
        return pse_res
def _t_broadcast_pseShift_n(pse_shift_tensor, pse_shift_shape, q_batch):#pre_shift, pre_shift_shape, q_bnsd_shape[0]
    print(f"broadcast_mask_n:mask shape:{pse_shift_shape} q_batch:{q_batch}")
    #1nss or bnss
    B_m = pse_shift_shape[0]
    if B_m == q_batch:
        return pse_shift_tensor
    else:
        B = q_batch
        pse_res = torch.zeros([B, pse_shift_shape[1],pse_shift_shape[2], pse_shift_shape[3]])
        for i in range(B):
            pse_res[i:i+1] = pse_shift_tensor[0]
        return pse_res
def _np_broadcast_mask_n(m_tensor, m_shape,cpu_m_shape, numheads, q_batch):
    print(f"broadcast_mask_n:mask shape:{m_shape} with numheads:{numheads} q_batch:{q_batch}")
    mask_cur_shape = cpu_m_shape
    if len(m_shape) == 4:
        # b1ss
        B_m = m_shape[0]
        if B_m != 1:
            B = B_m
        else:
            B = q_batch
        m_res = []
        for i in range(B):
            # mask_cur_shape = [m_shape[2], m_shape[3]]
            if B_m == 1:
                mask_cur = m_tensor[:, :, :].reshape(mask_cur_shape)
            else:
                mask_cur = m_tensor[i:i + 1, :, :, :].reshape(mask_cur_shape)
            m_res.append(mask_cur)
        return m_res
    elif len(m_shape) == 3:
        # bss
        B_m = m_shape[0]
        if B_m != 1:
            B = B_m
        else:
            B = q_batch
        m_res = []
        for i in range(B):
            # mask_cur_shape = [m_shape[1], m_shape[2]]
            if B_m == 1:
                mask_cur = m_tensor[:, :, :].reshape(mask_cur_shape)
            else:
                mask_cur = m_tensor[i:i + 1, :, :].reshape(mask_cur_shape)
            m_res.append(mask_cur)
        return m_res
    elif len(m_shape) == 2:
        # ss
        B = q_batch
        m_res = []
        for i in range(B):
            m_res.append(m_tensor)
        return m_res
    else:
        return m_tensor

def _t_broadcast_mask_n(m_tensor, m_shape, numheads, q_batch):
    print(f"broadcast_mask_n:mask shape:{m_shape} with numheads:{numheads} q_batch:{q_batch}")
    if len(m_shape) == 4:
        # b1ss
        B = m_shape[0]

        m_res = torch.zeros([B, numheads, m_shape[2], m_shape[3]])
        for i in range(B):
            mask_cur_shape = [m_shape[2], m_shape[3]]
            mask_cur = m_tensor[i:i + 1, :, :, :].reshape(mask_cur_shape)
            for j in range(numheads):
                m_res[i:i + 1, j:j + 1, :, :] = mask_cur
        return m_res
    elif len(m_shape) == 3:
        # bss
        B_m = m_shape[0]
        if B_m != 1:
            B = B_m
        else:
            B = q_batch
        m_res = torch.zeros([B, numheads, m_shape[1], m_shape[2]])
        for i in range(B):
            mask_cur_shape = [m_shape[1], m_shape[2]]
            if B_m == 1:
                mask_cur = m_tensor[:, :, :].reshape(mask_cur_shape)
            else:
                mask_cur = m_tensor[i:i + 1, :, :].reshape(mask_cur_shape)
            for j in range(numheads):
                m_res[i:i + 1, j:j + 1, :, :] = mask_cur
        return m_res
    elif len(m_shape) == 2:
        # ss
        B = q_batch
        m_res = torch.zeros([B, numheads, m_shape[0], m_shape[1]])
        for i in range(B):
            for j in range(numheads):
                m_res[i:i + 1, j:j + 1, :, :] = m_tensor
        return m_res
    else:
        return m_tensor


def npSoftmax(x):
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    ans = y / x_sum
    return ans, x_max, x_sum

def npSoftmax_new(x, sinks=None):
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    if sinks is not None:
        sinks_reshaped = sinks.reshape(1, -1, 1, 1)
        sinks_sub = sinks_reshaped - x_max
        sink_exp = np.exp(sinks_sub)
        x_sum += sink_exp.sum(axis=-1, keepdims=True)
    ans = y
    return ans, x_max, x_sum


def _t_softmax(x):
    x_max = torch.max(x, dim=-1, keepdims=True)[0]
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdims=True)
    ans = y.div(x_sum)
    return ans, x_max, x_sum


def softmax_flashv2(x, max_front=None, sum_front=None, update=None, is_fp16=False):
    """
    Compute the softmax function for each channel of the input x.
    """
    if update == None:
        if is_fp16:
            x = x.astype(np.float32)
        x_max = np.max(x, axis=-1, keepdims=True)
        # print("#########"*4)
        # print(x_max)
        # print(x)
        x_sub = x - x_max  # -> x
        x_exp = np.exp(x_sub)  # -> x
        x_sum = np.sum(x_exp, axis=-1, keepdims=True)
        out = x_exp
        # out = x_exp / x_sum
        exp_max = None
        if is_fp16:
            out = out.astype(np.float16)
            # x_max = x_max.astype(np.float16)
            # x_sum = x_sum.astype(np.float16)

        return out, x_max, x_sum, exp_max
    else:
        if is_fp16:
            x = x.astype(np.float32)
            max_front = max_front.astype(np.float32)
            sum_front = sum_front.astype(np.float32)
        x_max_tmp = np.max(x, axis=-1, keepdims=True)  # tmp
        x_sub = x - x_max_tmp  # -> x
        x_exp = np.exp(x_sub)  # -> x
        x_sum = np.sum(x_exp, axis=-1, keepdims=True)  # tmp
        x_max = np.max(np.concatenate((max_front, x_max_tmp), axis=-1), axis=-1, keepdims=True)  # ->x_max
        x_exp_new = np.exp(x_max_tmp - x_max)  # -> x_max_tmp
        exp_max = np.exp(max_front - x_max)  # -> exp_max
        # update sum
        exp_max = exp_max * sum_front
        reduce_tmp = x_exp_new * x_sum
        x_sum = exp_max + reduce_tmp  # x_sum

        # exp_max = exp_max/x_sum
        exp_max = np.exp(max_front - x_max)
        out = x_exp * x_exp_new
        # out = x_exp*x_exp_new / x_sum

        # ### softmax ratio
        # softma_ratio = x_sum * x_exp_new / x_sum
        # out_new = out * softma_ratio
        # pround("out_new===================", out_new)
        if is_fp16:
            out = out.astype(np.float16)
            # x_max = x_max.astype(np.float16)
            # x_sum = x_sum.astype(np.float16)
            # exp_max = exp_max.astype(np.float16)

        return out, x_max, x_sum, exp_max


def _np_pfaattention_act_int8(q_tensor, k_tensor, v_tensor, pse_tensor, mask_tensor, scalar, act_seq, act_kv_seq, preTokens, nextTokens,pfa_param,
                             dequant_scale1=None,
                             dequant_scale2=None, quant_scale1=None, quant_scale2=None, quant_offset2=None,
                             out_dtype="<class 'numpy.int8'>"):
    print(
        f"q_shape:{q_tensor.shape} k_shape:{k_tensor.shape} v_shape:{v_tensor.shape} scalar:{scalar} act_seq:{act_seq} act_kv_seq:{act_kv_seq}")
    S = None
    qs_begin = pfa_param['qs_begin']
    qs_end = pfa_param['qs_end']
    kvs_begin = pfa_param['kvs_begin']
    kvs_end = pfa_param['kvs_end']

    q_tensor = q_tensor[:, :, qs_begin:qs_end, :]

    k_tensor = k_tensor[:, :, kvs_begin:kvs_end, :]
    v_tensor = v_tensor[:, :, kvs_begin:kvs_end, :]
    S = kvs_end - kvs_begin

    if mask_tensor is not None:
        print(f"mask_shape:{mask_tensor.shape}")
        if pfa_param['sparse'] == 2 or pfa_param['sparse'] == 3 or pfa_param['sparse'] == 4:
            mask_tensor = mask_tensor[:, :, :(qs_end - qs_begin), :(kvs_end - kvs_begin)]
        else:
            mask_tensor = mask_tensor[:, :, qs_begin:qs_end, kvs_begin:kvs_end]
    # 这里需要加paddingmask
    if pse_tensor is not None:
        pse_tensor = pse_tensor[:, :, qs_begin:qs_end, kvs_begin:kvs_end]

    qs = q_tensor.shape[2]
    qd = q_tensor.shape[3]
    kd = k_tensor.shape[3]
    vd = v_tensor.shape[3]
    one_loop_size = 128
    if qd > 128 and kd == vd:
        one_loop_size = 64
    max_range = (S + one_loop_size - 1) // one_loop_size

    # dequant 算子内部使用float19进行计算
    dequant_scale1.dtype = np.uint32
    dequant_scale1 = np.bitwise_and(dequant_scale1, 0xffffe000)
    dequant_scale1.dtype = np.float32

    dequant_scale2.dtype = np.uint32
    dequant_scale2 = np.bitwise_and(dequant_scale2, 0xffffe000)
    dequant_scale2.dtype = np.float32

    data_length = one_loop_size
    max_front, sum_front, o_front = None, None, None
    for i in range(max_range):
        if i == S // one_loop_size:
            data_length = S % one_loop_size
        data_start = one_loop_size * i
        data_end = one_loop_size * i + data_length
        ki = k_tensor[:, :, data_start:data_end, :]
        vi = v_tensor[:, :, data_start:data_end, :]  # [2, 2, 128, 128]
        qDtype = q_tensor.dtype
        print("start matmul1")
        qk_i = np.matmul(q_tensor.astype(np.int32), ki.transpose(0, 1, 3, 2).astype(np.int32))

        # print("dequant_scale1", dequant_scale1)
        # qk_i = qk_i.astype(np.float16) * dequant_scale1.astype(np.float16)
        qk_i = qk_i.astype(np.float32) * dequant_scale1
        print("end matmul1")

        # 转成fp16防止溢出
        # qk_i = np.where(qk_i > 65504, 65504, qk_i)
        # qk_i = np.where(qk_i < -65504, -65504, qk_i)
        # qk_i = qk_i.astype(np.float16)
        qk_i = qk_i * np.float32(scalar)

        # qk_i = qk_i*scalar
        if pse_tensor is not None:
            pse_tensor_i = pse_tensor[:, :, :, data_start:data_end]
            qk_i += pse_tensor_i

        if mask_tensor is not None:
            atten_mask_i = mask_tensor[:, :, :, data_start:data_end]
            # qk_i += atten_mask_i * (-10000.0)
            qk_i[atten_mask_i.astype(np.bool_)] = -65504
        if i == 0:
            py_s_i, max_i, sum_i, exp_max_i = softmax_flashv2(qk_i, is_fp16=True)
            print("start matmul2")
            py_s_i = py_s_i * quant_scale1.astype(np.float16)
            py_s_i = np.around(py_s_i)

            # 转成int8防止溢出
            py_s_i = np.where(py_s_i > 127, 127, py_s_i)
            py_s_i = np.where(py_s_i < -128, -128, py_s_i)
            py_s_i = py_s_i.astype(np.int8)

            o_i = np.matmul(py_s_i.astype(np.int32), vi.astype(np.int32))
            print("end matmul2")
            # o_i = o_i.astype(np.float16) * dequant_scale2.astype(np.float16)
            o_i = o_i.astype(np.float32) * dequant_scale2
            max_front, sum_front, o_front = max_i, sum_i, o_i
        else:
            py_s_i, max_i, sum_i, exp_max_i = softmax_flashv2(qk_i, max_front, sum_front, update=True, is_fp16=True)
            print("start matmul2")
            py_s_i = py_s_i * quant_scale1.astype(np.float16)
            py_s_i = np.around(py_s_i)

            # 转成int8防止溢出
            py_s_i = np.where(py_s_i > 127, 127, py_s_i)
            py_s_i = np.where(py_s_i < -128, -128, py_s_i)
            py_s_i = py_s_i.astype(np.int8)

            # o_i = np.matmul(py_s_i.astype(np.float32), vi.astype(np.int8).astype(np.float32))
            o_i = np.matmul(py_s_i.astype(np.int32), vi.astype(np.int32))
            o_i = o_i.astype(np.float32) * dequant_scale2
            print("end matmul2")

            # 转成fp16防止溢出
            # o_i = np.where(o_i > 65504, 65504, o_i)
            # o_i = np.where(o_i < -65504, -65504, o_i)
            o_i = o_i.astype(np.float32)

            o_front_update = np.array(o_front * exp_max_i)
            o_i = o_front_update + np.array(o_i)
            max_front, sum_front, o_front = max_i, sum_i, o_i

    lse = None
    if pfa_param['lseflag']:
        lse = np.log(sum_front) + max_front

    o_front = o_front / sum_front.astype(np.float32)
    if mask_tensor is not None:
        for i in range(mask_tensor.shape[2]):
            if mask_tensor[:, :, i, :].all() == 1:
                o_front[:, :, i, :] = 0
                if lse is not None:
                    lse[:, :, i, :] = np.inf

    if str(out_dtype) == "<class 'numpy.int8'>":
        o_front = o_front * quant_scale2.astype(np.float16)
        if quant_offset2 is not None:
            o_front += quant_offset2.astype(np.float16)
        o_front = np.around(o_front)
        # 转成int8防止溢出
        o_front = np.where(o_front > 127, 127, o_front)
        o_front = np.where(o_front < -128, -128, o_front)
        o_front = o_front.astype(np.int8)
    else:
        o_front = o_front.astype(np.float16)

    return o_front, lse

def _np_pfaattention_act_fp8(q_tensor, k_tensor, v_tensor, pse_tensor, mask_tensor, scalar, act_seq, act_kv_seq, preTokens, nextTokens,pfa_param,
                             dequant_scale1=None,
                             dequant_scale2=None, quant_scale1=None, quant_scale2=None, quant_offset2=None,
                             out_dtype="<class 'numpy.int8'>"):
    print(
        f"_np_pfaattention_act_fp8 q_shape:{q_tensor.shape} k_shape:{k_tensor.shape} v_shape:{v_tensor.shape} scalar:{scalar} act_seq:{act_seq} act_kv_seq:{act_kv_seq}")
    S = None
    qs_begin = pfa_param['qs_begin']
    qs_end = pfa_param['qs_end']
    kvs_begin = pfa_param['kvs_begin']
    kvs_end = pfa_param['kvs_end']


    q_tensor = q_tensor[:, :, qs_begin:qs_end, :]

    k_tensor = k_tensor[:, :, kvs_begin:kvs_end, :]
    v_tensor = v_tensor[:, :, kvs_begin:kvs_end, :]
    S = kvs_end-kvs_begin


    if mask_tensor is not None:
        print(f"mask_shape:{mask_tensor.shape}")
        if pfa_param['sparse'] == 2 or pfa_param['sparse'] == 3 or pfa_param['sparse'] == 4:
            mask_tensor = mask_tensor[:, :, :(qs_end - qs_begin), :(kvs_end - kvs_begin)]
        else:
            mask_tensor = mask_tensor[:, :, qs_begin:qs_end, kvs_begin:kvs_end]
    # 这里需要加paddingmask
    if pse_tensor is not None:
        pse_tensor = pse_tensor[:, :, qs_begin:qs_end, kvs_begin:kvs_end]

    qs = q_tensor.shape[2]
    qd = q_tensor.shape[3]
    kd = k_tensor.shape[3]
    vd = v_tensor.shape[3]
    one_loop_size = 128
    if qd > 128 and kd == vd:
        one_loop_size = 64
    max_range = (S + one_loop_size - 1) // one_loop_size    # ceil(S/one_loop_size) 对K和V的S进行切分 计算分块softmax

    # dequant 算子内部使用float19进行计算
    dequant_scale1.dtype = np.uint32
    dequant_scale1 = np.bitwise_and(dequant_scale1, 0xffffe000)
    dequant_scale1.dtype = np.float32

    dequant_scale2.dtype = np.uint32
    dequant_scale2 = np.bitwise_and(dequant_scale2, 0xffffe000)
    dequant_scale2.dtype = np.float32

    data_length = one_loop_size
    max_front, sum_front, o_front = None, None, None
    for i in range(max_range):
        if i == S // one_loop_size:
            data_length = S % one_loop_size
        data_start = one_loop_size * i
        data_end = one_loop_size * i + data_length
        ki = k_tensor[:, :, data_start:data_end, :]
        vi = v_tensor[:, :, data_start:data_end, :]  # [2, 2, 128, 128]
        qDtype = q_tensor.dtype
        print("qDtype:", qDtype)
        print("start matmul1")
        qk_i = np.matmul(q_tensor.astype(np.float32), ki.transpose(0, 1, 3, 2).astype(np.float32))
        # qk_i = qk_i.astype(np.float16) * dequant_scale1.astype(np.float16)
        qk_i = qk_i.astype(np.float32) * dequant_scale1
        # print(qk_i)
        print("end matmul1")

        qk_i = qk_i* np.float32(scalar)

        # qk_i = qk_i*scalar
        if pse_tensor is not None:
            pse_tensor_i = pse_tensor[:, :, :, data_start:data_end]
            qk_i += pse_tensor_i

        if mask_tensor is not None:
            atten_mask_i = mask_tensor[:, :, :, data_start:data_end]
            # qk_i += atten_mask_i * (-10000.0)
            qk_i[atten_mask_i.astype(np.bool_)] = -65504
        if i == 0:
            py_s_i, max_i, sum_i, exp_max_i = softmax_flashv2(qk_i, is_fp16=False)
            # print(py_s_i)
            print("start matmul2")
            py_s_i = py_s_i * quant_scale1.astype(np.float32)

            if qDtype == "float8_e5m2":
                py_s_i = py_s_i.astype(float8_e5m2)
            else:
                py_s_i = py_s_i.astype(float8_e4m3fn)
            # print(py_s_i)

            o_i = np.matmul(py_s_i.astype(np.float32), vi.astype(np.float32))
            print("end matmul2")
            # o_i = o_i.astype(np.float16) * dequant_scale2.astype(np.float16)
            o_i = o_i.astype(np.float32) * dequant_scale2
            max_front, sum_front, o_front = max_i, sum_i, o_i  # 初始前向累计值
        else:
            py_s_i, max_i, sum_i, exp_max_i = softmax_flashv2(qk_i, max_front, sum_front, update=True, is_fp16=False)
            print("start matmul2")
            py_s_i = py_s_i * quant_scale1.astype(np.float32)

            if qDtype == "float8_e5m2":
                py_s_i = py_s_i.astype(float8_e5m2)
            else:
                py_s_i = py_s_i.astype(float8_e4m3fn)

            print(py_s_i.dtype)

            # o_i = np.matmul(py_s_i.astype(np.float32), vi.astype(np.int8).astype(np.float32))
            o_i = np.matmul(py_s_i.astype(np.float32), vi.astype(np.float32))
            o_i = o_i.astype(np.float32) * dequant_scale2
            print("end matmul2")

            o_front_update = np.array(o_front * exp_max_i)
            o_i = o_front_update + np.array(o_i)
            max_front, sum_front, o_front = max_i, sum_i, o_i

    lse = None
    if pfa_param['lseflag']:
        lse = np.log(sum_front) + max_front

    o_front = o_front / sum_front.astype(np.float32)
    if mask_tensor is not None:
        for i in range(mask_tensor.shape[2]):
            if mask_tensor[:, :, i, :].all() == 1:
                o_front[:, :, i, :] = 0
                if lse is not None:
                    lse[:, :, i, :] = np.inf

    if str(out_dtype) == "<class 'numpy.int8'>":
        o_front = o_front * quant_scale2.astype(np.float16)
        if quant_offset2 is not None:
            o_front += quant_offset2.astype(np.float16)
        o_front = np.around(o_front)
        # 转成int8防止溢出
        o_front = np.where(o_front > 127, 127, o_front)
        o_front = np.where(o_front < -128, -128, o_front)
        o_front = o_front.astype(np.int8)
    else:
        o_front = o_front.astype(np.float16)

    return o_front, lse


def _np_pfaattention_act_hifp8(q_tensor_hifp8, k_tensor_hifp8, v_tensor_hifp8, pse_tensor, mask_tensor, scalar, act_seq, act_kv_seq, preTokens, nextTokens,pfa_param,
                             dequant_scale1=None,
                             dequant_scale2=None, quant_scale1=None, quant_scale2=None, quant_offset2=None,
                             out_dtype="<class 'numpy.int8'>"):
    S = None
    q_tensor = trans_np_hifuint8_tensor_to_float32(q_tensor_hifp8).astype(np.float32)
    k_tensor = trans_np_hifuint8_tensor_to_float32(k_tensor_hifp8).astype(np.float32)
    v_tensor = trans_np_hifuint8_tensor_to_float32(v_tensor_hifp8).astype(np.float32)

    print(
        f"q_shape:{q_tensor.shape} k_shape:{k_tensor.shape} v_shape:{v_tensor.shape} scalar:{scalar} act_seq:{act_seq} act_kv_seq:{act_kv_seq}")
    qs_begin = pfa_param['qs_begin']
    qs_end = pfa_param['qs_end']
    kvs_begin = pfa_param['kvs_begin']
    kvs_end = pfa_param['kvs_end']

    q_tensor = q_tensor[:, :, qs_begin:qs_end, :]

    k_tensor = k_tensor[:, :, kvs_begin:kvs_end, :]
    v_tensor = v_tensor[:, :, kvs_begin:kvs_end, :]
    S = kvs_end - kvs_begin

    if mask_tensor is not None:
        print(f"mask_shape:{mask_tensor.shape}")
        if pfa_param['sparse'] == 2 or pfa_param['sparse'] == 3 or pfa_param['sparse'] == 4:
            mask_tensor = mask_tensor[:, :, :(qs_end - qs_begin), :(kvs_end - kvs_begin)]
        else:
            mask_tensor = mask_tensor[:, :, qs_begin:qs_end, kvs_begin:kvs_end]
    # 这里需要加paddingmask
    if pse_tensor is not None:
        pse_tensor = pse_tensor[:, :, qs_begin:qs_end, kvs_begin:kvs_end]

    qs = q_tensor.shape[2]
    qd = q_tensor.shape[3]
    kd = k_tensor.shape[3]
    vd = v_tensor.shape[3]
    one_loop_size = 128
    if qd > 128 and kd == vd:
        one_loop_size = 64
    max_range = (S + one_loop_size - 1) // one_loop_size

    # dequant 算子内部使用float19进行计算
    dequant_scale1.dtype = np.uint32
    dequant_scale1 = np.bitwise_and(dequant_scale1, 0xffffe000)
    dequant_scale1.dtype = np.float32

    dequant_scale2.dtype = np.uint32
    dequant_scale2 = np.bitwise_and(dequant_scale2, 0xffffe000)
    dequant_scale2.dtype = np.float32

    data_length = one_loop_size
    max_front, sum_front, o_front = None, None, None
    for i in range(max_range):
        if i == S // one_loop_size:
            data_length = S % one_loop_size
        data_start = one_loop_size * i
        data_end = one_loop_size * i + data_length
        ki = k_tensor[:, :, data_start:data_end, :]
        vi = v_tensor[:, :, data_start:data_end, :]  # [2, 2, 128, 128]
        qDtype = q_tensor.dtype
        print("start matmul1")
        qk_i = np.matmul(q_tensor.astype(np.float32), ki.transpose(0, 1, 3, 2).astype(np.float32))

        # qk_i = qk_i.astype(np.float16) * dequant_scale1.astype(np.float16)
        qk_i = qk_i.astype(np.float32) * dequant_scale1
        print("end matmul1")

        qk_i = qk_i.astype(np.float32)
        qk_i = qk_i * np.float32(scalar)

        # qk_i = qk_i*scalar
        if pse_tensor is not None:
            pse_tensor_i = pse_tensor[:, :, :, data_start:data_end]
            qk_i += pse_tensor_i

        if mask_tensor is not None:
            atten_mask_i = mask_tensor[:, :, :, data_start:data_end]
            # qk_i += atten_mask_i * (-10000.0)
            qk_i[atten_mask_i.astype(np.bool_)] = -65504
        if i == 0:
            py_s_i, max_i, sum_i, exp_max_i = softmax_flashv2(qk_i, is_fp16=False)
            print("start matmul2")
            py_s_i = py_s_i * quant_scale1.astype(np.float32)

            py_s_i = trans_np_float_tensor_to_hifuint8(in_tensor=py_s_i, round_mode="hybrid",
                                                             over_mode=True)
            py_s_i = trans_np_hifuint8_tensor_to_float32(py_s_i).astype(np.float32)

            o_i = np.matmul(py_s_i.astype(np.float32), vi.astype(np.float32))
            print("end matmul2")
            # o_i = o_i.astype(np.float16) * dequant_scale2.astype(np.float16)
            o_i = o_i.astype(np.float32) * dequant_scale2
            max_front, sum_front, o_front = max_i, sum_i, o_i
        else:
            py_s_i, max_i, sum_i, exp_max_i = softmax_flashv2(qk_i, max_front, sum_front, update=True, is_fp16=False)
            print("start matmul2")
            py_s_i = py_s_i * quant_scale1.astype(np.float32)

            py_s_i = trans_np_float_tensor_to_hifuint8(in_tensor=py_s_i, round_mode="hybrid",
                                                             over_mode=True)
            py_s_i = trans_np_hifuint8_tensor_to_float32(py_s_i).astype(np.float32)

            # o_i = np.matmul(py_s_i.astype(np.float32), vi.astype(np.int8).astype(np.float32))
            o_i = np.matmul(py_s_i.astype(np.float32), vi.astype(np.float32))
            o_i = o_i.astype(np.float32) * dequant_scale2
            print("end matmul2")

            o_front_update = np.array(o_front * exp_max_i)
            o_i = o_front_update + np.array(o_i)
            max_front, sum_front, o_front = max_i, sum_i, o_i

    lse = None
    if pfa_param['lseflag']:
        lse = np.log(sum_front) + max_front

    o_front = o_front / sum_front.astype(np.float32)
    if mask_tensor is not None:
        for i in range(mask_tensor.shape[2]):
            if mask_tensor[:, :, i, :].all() == 1:
                o_front[:, :, i, :] = 0
                if lse is not None:
                    lse[:, :, i, :] = np.inf

    if str(out_dtype) == "<class 'numpy.int8'>":
        o_front = o_front * quant_scale2.astype(np.float16)
        if quant_offset2 is not None:
            o_front += quant_offset2.astype(np.float16)
        o_front = np.around(o_front)
        # 转成int8防止溢出
        o_front = np.where(o_front > 127, 127, o_front)
        o_front = np.where(o_front < -128, -128, o_front)
        o_front = o_front.astype(np.int8)
    else:
        o_front = o_front.astype(np.float16)


    return o_front,lse

def _t_pfaattention_act_innerprecise(q_tensor, k_tensor, v_tensor, pse_tensor, mask_tensor, scalar, act_seq, act_kv_seq, preTokens, nextTokens, dequant_scale1=None,
                        dequant_scale2=None, quant_scale1=None, quant_scale2=None, quant_offset2=None):
    print(
        f"q_shape:{q_tensor.shape} k_shape:{k_tensor.shape} v_shape:{v_tensor.shape} scalar:{scalar} act_seq:{act_seq} act_kv_seq:{act_kv_seq}")
    if act_seq is not None:
        q_tensor = q_tensor[:, :, :act_seq, :]
    if act_kv_seq is not None:
        k_tensor = k_tensor[:, :, :act_kv_seq, :]
        v_tensor = v_tensor[:, :, :act_kv_seq, :]

    q_t = q_tensor.to(torch.float32)
    k_t = k_tensor.to(torch.float32)
    v_t = v_tensor.to(torch.float32)

    q_tensor = q_t.cuda().requires_grad_(False)
    k_tensor = k_t.cuda().requires_grad_(False)
    v_tensor = v_t.cuda().requires_grad_(False)



    print("start matmul1")

    qkBmmRes = torch.matmul(q_tensor, k_tensor.permute([0, 1, 3, 2]))
    print("end matmul1")
    qkEleRes = qkBmmRes * scalar

    # 这里需要加paddingmask
    if pse_tensor is not None:
        if act_seq is not None:
            pse_tensor = pse_tensor[:, :, :act_seq, :]
        if act_kv_seq is not None:
            pse_tensor = pse_tensor[:, :, :act_seq, :act_kv_seq]
        qkEleRes += pse_tensor

    if mask_tensor is not None:
        print(f"mask_shape:{mask_tensor.shape}")
        if act_seq is not None:
            mask_tensor = mask_tensor[:, :, :act_seq, :]
        if act_kv_seq is not None:
            mask_tensor = mask_tensor[:, :, :act_seq, :act_kv_seq]

        # mask_t = torch.from_numpy(mask_tensor)
        # mask_tensor = mask_tensor.to(bool)
        # mask_tensor = mask_tensor.cuda().requires_grad_(False)
        qkEleRes[mask_tensor.to(bool)] = -1.7 * 10 ** 38
        # qkEleRes += mask_tensor * (-1000000.0)
    softmax_res, x_max, x_sum = _t_softmax(qkEleRes)
    if q_tensor.shape[2] > k_tensor.shape[2] + preTokens:
        ss0 = k_tensor.shape[2] + preTokens
        softmax_res[:, :, ss0:, :] = 0
    if nextTokens < 0:
        ss1 = -nextTokens
        softmax_res[:, :, :ss1, :] = 0
    print("start matmul2")
    bmm2Res = torch.matmul(softmax_res.to(torch.float16).to(torch.float32), v_tensor).cpu()
    print("end matmul2")
    print(f"return shape:{bmm2Res.shape}")
    return bmm2Res
def _t_pfaattention_act(q_tensor, k_tensor, v_tensor, pse_tensor, mask_tensor, scalar, act_seq, act_kv_seq, preTokens, nextTokens, dequant_scale1=None,
                        dequant_scale2=None, quant_scale1=None, quant_scale2=None, quant_offset2=None):
    print(
        f"q_shape:{q_tensor.shape} k_shape:{k_tensor.shape} v_shape:{v_tensor.shape} scalar:{scalar} act_seq:{act_seq} act_kv_seq:{act_kv_seq}")
    if act_seq is not None:
        q_tensor = q_tensor[:, :, :act_seq, :]
    if act_kv_seq is not None:
        k_tensor = k_tensor[:, :, :act_kv_seq, :]
        v_tensor = v_tensor[:, :, :act_kv_seq, :]

    q_t = q_tensor.to(torch.float32)
    k_t = k_tensor.to(torch.float32)
    v_t = v_tensor.to(torch.float32)

    q_tensor = q_t.cuda().requires_grad_(False)
    k_tensor = k_t.cuda().requires_grad_(False)
    v_tensor = v_t.cuda().requires_grad_(False)

    print("start matmul1")

    qkBmmRes = torch.matmul(q_tensor, k_tensor.permute([0, 1, 3, 2]))
    print("end matmul1")
    qkEleRes = qkBmmRes * scalar

    # 这里需要加paddingmask
    if pse_tensor is not None:
        if act_seq is not None:
            pse_tensor = pse_tensor[:, :, :act_seq, :]
        if act_kv_seq is not None:
            pse_tensor = pse_tensor[:, :, :act_seq, :act_kv_seq]
        qkEleRes += pse_tensor

    if mask_tensor is not None:
        print(f"mask_shape:{mask_tensor.shape}")
        if act_seq is not None:
            mask_tensor = mask_tensor[:, :, :act_seq, :]
        if act_kv_seq is not None:
            mask_tensor = mask_tensor[:, :, :act_seq, :act_kv_seq]

        # mask_t = torch.from_numpy(mask_tensor)
        # mask_tensor = mask_tensor.to(bool)
        # mask_tensor = mask_tensor.cuda().requires_grad_(False)
        qkEleRes[mask_tensor.to(bool)] = -1.7 * 10 ** 38
        # qkEleRes += mask_tensor * (-1000000.0)
    softmax_res, x_max, x_sum = _t_softmax(qkEleRes)
    if q_tensor.shape[2] > k_tensor.shape[2] + preTokens:
        ss0 = k_tensor.shape[2] + preTokens
        softmax_res[:, :, ss0:, :] = 0
    if nextTokens < 0:
        ss1 = -nextTokens
        softmax_res[:, :, :ss1, :] = 0
    print("start matmul2")
    bmm2Res = torch.matmul(softmax_res.to(torch.bfloat16).to(torch.float32), v_tensor).cpu()
    print("end matmul2")
    print(f"return shape:{bmm2Res.shape}")
    return bmm2Res

def _np_pfaattention_act(q_tensor, k_tensor, v_tensor, pse_tensor, mask_tensor, scalar, act_seq, act_kv_seq, preTokens, nextTokens,pfa_param, quant_scale2=None, quant_offset2=None, antiquant_scale_k=None,
                            antiquant_scale_v=None, antiquant_offset_k=None, antiquant_offset_v=None, out_dtype="<class 'numpy.int8'>", sinks=None):
    print(
        f"q_shape:{q_tensor.shape} k_shape:{k_tensor.shape} v_shape:{v_tensor.shape} scalar:{scalar} act_seq:{act_seq} act_kv_seq:{act_kv_seq} preTokens:{preTokens}nextTokens:{nextTokens}")

    # qs_begin = 0
    # qs_end = q_tensor.shape[2]
    # kvs_begin = 0
    # kvs_end = k_tensor.shape[2]
    qs_begin = pfa_param['qs_begin']
    qs_end = pfa_param['qs_end']
    kvs_begin = pfa_param['kvs_begin']
    kvs_end = pfa_param['kvs_end']

    qdtype = pfa_param['q_dtype']
    q_tensor = q_tensor[:, :, qs_begin:qs_end, :]

    k_tensor = k_tensor[:, :, kvs_begin:kvs_end, :]
    v_tensor = v_tensor[:, :, kvs_begin:kvs_end, :]
    qDtype = q_tensor.dtype

    # KVcahce反量化  这块逻辑需要修改
    if k_tensor.dtype == np.int8:
        k_tensor = k_tensor.astype(np.float16)
        v_tensor = v_tensor.astype(np.float16)
        if antiquant_offset_k is not None:
            if pfa_param['anti_or_kvanti'] == 2 and pfa_param['k_antiquant_mode'] == 1:
                antiquant_offset_k = np.expand_dims(antiquant_offset_k[:, kvs_begin:kvs_end], axis=-1)
                antiquant_offset_v = np.expand_dims(antiquant_offset_v[:, kvs_begin:kvs_end], axis=-1)
            k_tensor += antiquant_offset_k
            v_tensor += antiquant_offset_v
        if pfa_param['anti_or_kvanti'] == 2 and pfa_param['k_antiquant_mode'] == 1:
            antiquant_scale_k = np.expand_dims(antiquant_scale_k[:, kvs_begin:kvs_end], axis=-1)
            antiquant_scale_v = np.expand_dims(antiquant_scale_v[:, kvs_begin:kvs_end], axis=-1)
        k_tensor *= antiquant_scale_k
        v_tensor *= antiquant_scale_v
    print("start matmul1")
    qkBmmRes = np.matmul(q_tensor, k_tensor.transpose([0, 1, 3, 2]))
    print("end matmul1")
    print("qkBmmRes.shape", qkBmmRes.shape)
    print("scalar", scalar)
    qkEleRes = qkBmmRes * scalar  #

    # 这里需要加paddingmask
    print("start pse")
    if pse_tensor is not None:
        pse_tensor = pse_tensor[:, :, qs_begin:qs_end, kvs_begin:kvs_end]
        print("pse_tensor", pse_tensor.shape)
        qkEleRes += pse_tensor.astype(np.float32)

    print("end pse")

    print("start mask")
    if mask_tensor is not None:
        print(f"mask_shape:{mask_tensor.shape}")
        if pfa_param['sparse'] == 2 or pfa_param['sparse'] == 3 or pfa_param['sparse'] == 4:
            # actul_prefix = 0
            # if 'actualprefixKV' in pfa_param:
            #     actul_prefix =  pfa_param['actualprefixKV']
            mask_tensor = mask_tensor[:, :, :(qs_end - qs_begin), :(kvs_end - kvs_begin)]
        else:
            mask_tensor = mask_tensor[:, :, qs_begin:qs_end, kvs_begin:kvs_end]
        qkEleRes[mask_tensor.astype(np.bool_)] = -1.7 * 10 ** 38
        # qkEleRes += mask_tensor * (-1000000.0)
    print("end mask")
    softmax_res, x_max, x_sum = npSoftmax_new(qkEleRes, sinks)

    print(
        f"softmax_res.shape:{softmax_res.shape} x_max.shape:{x_max.shape} x_sum.shape:{x_sum.shape}")



    print("start lse")
    lse = None
    if 'lseflag' in pfa_param and pfa_param['lseflag']:
        lse = np.log(x_sum) + x_max
    print("end lse")

    print("start matmul2")
    if qdtype == np.float16:
        bmm2Res = np.matmul(softmax_res.astype(np.float16).astype(np.float32), v_tensor)
    elif qdtype == np.float32:
        bmm2Res = np.matmul(softmax_res, v_tensor)
    else:
        bmm2Res = np.matmul(softmax_res.astype(tf.bfloat16.as_numpy_dtype).astype(np.float32), v_tensor)
    print("end matmul2")

    # /sum
    bmm2Res = bmm2Res / x_sum

    # lqj 2
    if str(out_dtype) == "<class 'numpy.int8'>":
        print(bmm2Res.shape)
        print(quant_scale2.shape)
        bmm2Res = bmm2Res * quant_scale2.astype(np.float32)
        if quant_offset2 is not None:
            bmm2Res += quant_offset2.astype(np.float32)
        bmm2Res = np.around(bmm2Res)
        # 转成int8防止溢出
        bmm2Res = np.where(bmm2Res > 127, 127, bmm2Res)
        bmm2Res = np.where(bmm2Res < -128, -128, bmm2Res)
        bmm2Res = bmm2Res.astype(np.int8)
    print("########str(out_dtype)",str(out_dtype))
    # if str(out_dtype) == "<class 'ml_dtypes.float8_e4m3fn'>":
    position = str(out_dtype).find("float8_e4m3fn")
    if position != -1:
        print("float8_e4m3fn post quant")
        print(bmm2Res.shape)
        print(quant_scale2.shape)
        # for index, item in enumerate(bmm2Res.tolist()[0][0]):
        #     print(f"=={index}=src=, 元素: {item}")
        #     print("-----------")
        bmm2Res = bmm2Res * quant_scale2.astype(np.float32)
        # for index, item in enumerate(quant_scale2.tolist()):
        #     print(f"=={index}=quant_scale2=, 元素: {item}")
        # for index, item in enumerate(bmm2Res.tolist()[0][0]):
        #     print(f"=={index}=res=, 元素: {item}")
        #     print("-----------")
        if quant_offset2 is not None:
            bmm2Res += quant_offset2.astype(np.float32)
        # 转成float8_e4m3fn防止溢出
        # bmm2Res = np.where(bmm2Res > 127, 127, bmm2Res)
        # bmm2Res = np.where(bmm2Res < -128, -128, bmm2Res)
        bmm2Res = bmm2Res.astype(float8_e4m3fn)
    print("########str(out_dtype)", str(out_dtype))
    position = str(out_dtype).find("float8_e5m2")
    if position != -1:
        print("float8_e5m2 post quant")
        print(bmm2Res.shape)
        print(quant_scale2.shape)
        bmm2Res = bmm2Res * quant_scale2.astype(np.float32)
        if quant_offset2 is not None:
            bmm2Res += quant_offset2.astype(np.float32)
        # 转成float8_e5m2防止溢出
        # bmm2Res = np.where(bmm2Res > 127, 127, bmm2Res)
        # bmm2Res = np.where(bmm2Res < -128, -128, bmm2Res)
        bmm2Res = bmm2Res.astype(float8_e5m2)

    if str(out_dtype) == "<class 'numpy.uint8'>":
        print("hifloat8 post quant")
        print(bmm2Res.shape)
        print(quant_scale2.shape)
        bmm2Res = bmm2Res * quant_scale2.astype(np.float32)
        if quant_offset2 is not None:
            bmm2Res += quant_offset2.astype(np.float32)
        # 转成hifuint8防止溢出
        # bmm2Res = np.where(bmm2Res > 127, 127, bmm2Res)
        # bmm2Res = np.where(bmm2Res < -128, -128, bmm2Res)

    if str(out_dtype) == "<class 'numpy.float32'>" and quant_scale2 is not None:
        print("gold post quant")
        print(bmm2Res.shape)
        print(quant_scale2.shape)
        bmm2Res = bmm2Res * quant_scale2.astype(np.float32)
        if quant_offset2 is not None:
            bmm2Res += quant_offset2.astype(np.float32)

    #20240521新修改
    if mask_tensor is not None:
        for i in range(mask_tensor.shape[2]):
            if mask_tensor[:, :, i, :].all() == 1:
                bmm2Res[:, :, i, :] = 0
                if lse is not None:
                    lse[:,:,i,:] = np.inf


    # if q_tensor.shape[2] > k_tensor.shape[2] + preTokens:
    #     ss0 = k_tensor.shape[2] + preTokens
    #     bmm2Res[:, :, ss0:, :] = 0
    #
    # if nextTokens < 0:
    #     ss1 = -nextTokens
    #     bmm2Res[:, :, :ss1, :] = 0


    print(f"return shape:{bmm2Res.shape}")
    return bmm2Res, lse


def _t_promtattention_bnsd(q_tensor, q_shape, k_tensor, k_shape, v_tensor, v_shape,  pse_tensor,mask_tensor, scale, actseqlens,
                           actkvseqlens, preTokens, nextTokens, dequant_scale1=None, dequant_scale2=None, quant_scale1=None, quant_scale2=None,
                           quant_offset2=None, out_dtype="<class 'numpy.int8'>", innerprecise=1):
    batch_value = q_shape[0]
    numhead_value = q_shape[1]
    actseqlens_size = len(actseqlens)
    print(f"B:{batch_value}  actseqlens:{actseqlens_size}")
    if 0 in q_shape:
        print(f"q shape:{q_shape} is empty,skip")
        return
    y = torch.zeros(q_shape, dtype=torch.float32)
    for b_index in range(batch_value):
        if len(actseqlens) == 0:
            act_seq = None
        else:
            act_seq = int(actseqlens[b_index])

        if len(actkvseqlens) == 0:
            act_kv_seq = None
        else:
            act_kv_seq = int(actkvseqlens[b_index])
        if act_seq == 0 or act_kv_seq == 0 or 0 in k_shape or 0 in v_shape:
            continue
        # bnsd
        for n_index in range(numhead_value):
            if len(q_shape) == 4 and len(k_shape) == 4 and len(v_shape) == 4:
                q_tensor_cur = q_tensor[b_index:(b_index + 1), n_index:(n_index + 1), :, :]
                k_tensor_cur = k_tensor[b_index:(b_index + 1), n_index:(n_index + 1), :, :]
                v_tensor_cur = v_tensor[b_index:(b_index + 1), n_index:(n_index + 1), :, :]
                if mask_tensor is None:
                    mask_cur = None
                else:
                    mask_cur = mask_tensor[b_index:(b_index + 1), n_index:(n_index + 1), :, :]

                if pse_tensor is None:
                    pse_cur = None
                else:
                    pse_cur = pse_tensor[b_index:(b_index + 1), n_index:(n_index + 1), :, :]

                if act_seq is None:
                    if innerprecise==0:
                            y[b_index:(b_index + 1), n_index:(n_index + 1), :, :] = _t_pfaattention_act_innerprecise(q_tensor_cur,k_tensor_cur,v_tensor_cur, pse_cur,
                                                                                                        mask_cur, scale,
                                                                                                        act_seq,
                                                                                                        act_kv_seq,
                                                                                                        preTokens,
                                                                                                        nextTokens,
                                                                                                        dequant_scale1,
                                                                                                        dequant_scale2,
                                                                                                        quant_scale1,
                                                                                                        quant_scale2,
                                                                                                        quant_offset2)
                    else:
                            y[b_index:(b_index + 1), n_index:(n_index + 1), :, :] = _t_pfaattention_act(q_tensor_cur,
                                                                                                    k_tensor_cur,
                                                                                                    v_tensor_cur,
                                                                                                    pse_cur,
                                                                                                    mask_cur, scale,
                                                                                                    act_seq, act_kv_seq,
                                                                                                    preTokens,
                                                                                                    nextTokens,
                                                                                                    dequant_scale1,
                                                                                                    dequant_scale2,
                                                                                                    quant_scale1,
                                                                                                    quant_scale2,
                                                                                                    quant_offset2)
                else:
                    if innerprecise==0:
                        y[b_index:(b_index + 1), n_index:(n_index + 1), :act_seq, :] = _t_pfaattention_act_innerprecise(
                                q_tensor_cur,k_tensor_cur,v_tensor_cur, pse_cur,mask_cur, scale,act_seq,act_kv_seq, preTokens, nextTokens,
                                dequant_scale1,dequant_scale2,quant_scale1,quant_scale2, quant_offset2)
                    else:
                        y[b_index:(b_index + 1), n_index:(n_index + 1), :act_seq, :] = _t_pfaattention_act(q_tensor_cur,
                                                                                                           k_tensor_cur,
                                                                                                           v_tensor_cur,
                                                                                                           pse_cur,
                                                                                                           mask_cur,
                                                                                                           scale,
                                                                                                           act_seq,
                                                                                                           act_kv_seq,
                                                                                                           preTokens,
                                                                                                           nextTokens,
                                                                                                           dequant_scale1,
                                                                                                           dequant_scale2,
                                                                                                           quant_scale1,
                                                                                                           quant_scale2,
                                                                                                           quant_offset2)

    return y

def _np_promtattention_bnsd(q_tensor, q_shape, k_tensor_list, k_shape_list, v_tensor_list, v_shape_list,
                            pse_bnsd_tensor, mask_tensor, scale, actseqlens,
                            actkvseqlens, slopes, preTokens_list, nextTokens_list, sparse, pfa_param, dequant_scale1=None,
                            dequant_scale2=None, quant_scale1=None, dequantScale_query=None,  k_antiquant_scale=None, v_antiquant_scale=None,
                            quant_scale2=None, quant_offset2=None, antiquant_scale=None, antiquant_offset=None, psetype=0, qstartidx=0, kvstartidx=0,
                            out_dtype="<class 'numpy.int8'>", q_dtype_origin="origin", sinks=None):


    batch_value = q_shape[0]
    numhead_value = q_shape[1]
    actseqlens_size = len(actseqlens)
    print(f" _np_promtattention_bnsd B:{batch_value}  actseqlens:{actseqlens_size}")
    # y = np.zeros(q_shape, dtype=np.float32)
    outShape = copy.deepcopy(q_shape)
    outShape[-1] = v_shape_list[0][-1]
    #y = np.zeros(outShape, dtype=out_dtype)
    y = np.zeros(outShape, dtype=np.float32)
    lse = np.full(pfa_param['lseshape'], np.inf)
    preTokens = preTokens_list
    nextTokens = nextTokens_list
    # print(mask_tensor[0][0])
    tensor_list_flag = len(k_tensor_list)
    if mask_tensor is not None:
        mask_cur = np.zeros([1, 1, q_shape[2], len(mask_tensor[0][0])], dtype='uint8')
    if tensor_list_flag==1:

        k_tensor = k_tensor_list[0]
        k_shape = k_shape_list[0]
        v_tensor = v_tensor_list[0]
        v_shape = v_shape_list[0]
    for b_index in range(batch_value):  # 遍历b
        if tensor_list_flag!=1:
            k_tensor = k_tensor_list[b_index]
            k_shape = k_shape_list[b_index]
            v_tensor = v_tensor_list[b_index]
            v_shape = v_shape_list[b_index]



        if len(actseqlens) == 0:
            act_seq = None
        else:
            act_seq = int(actseqlens[b_index])

        if len(actkvseqlens) == 0:
            act_kv_seq = None
        else:
            if len(actkvseqlens) == 1:
                act_kv_seq = int(actkvseqlens[0])
            else:
                act_kv_seq = int(actkvseqlens[b_index])
        if act_seq == 0 or act_kv_seq == 0 or 0 in k_shape or 0 in v_shape:
            continue

        qs_begin = 0
        qs_end = q_tensor.shape[2]  # s1
        # print("q_tensor.dtype is", q_tensor.dtype)
        kvs_begin = 0
        kvs_end = k_tensor.shape[2]  # s2
        if 'queryPaddingSize' in pfa_param:
            print(q_tensor.shape[2])
            print(act_seq)
            print(pfa_param['queryPaddingSize'][0])
            qs_begin = int(q_tensor.shape[2] - act_seq - pfa_param['queryPaddingSize'][0])
            qs_end = int(q_tensor.shape[2] - pfa_param['queryPaddingSize'][0])
            print(f"query_left padding--- s_begin:{qs_begin}, s_end:{qs_end}")

        else:
            if act_seq is not None:
                qs_end = act_seq
        if 'kvPaddingSize' in pfa_param:
            kvs_begin = int(k_tensor.shape[2] - act_kv_seq - pfa_param['kvPaddingSize'][0])
            kvs_end = int(k_tensor.shape[2] - pfa_param['kvPaddingSize'][0])
            print(f"kv_left padding--- s_begin:{kvs_begin}, s_end:{kvs_end}")
        else:
            if act_kv_seq is not None:
                kvs_end = act_kv_seq
        pfa_param['qs_begin'] = qs_begin
        pfa_param['qs_end'] = qs_end
        pfa_param['kvs_begin'] = kvs_begin
        pfa_param['kvs_end'] = kvs_end
        if 'actualprefixKV' in pfa_param:
            if pfa_param['actualprefixKV']==0:
                pfa_param['kvs_end']+=pfa_param["shared_prefix_k"].shape[2]
            else:
                pfa_param['kvs_end']+=pfa_param['actualprefixKV']
        pfa_param['sparse'] = sparse


        # bnsd
        if sparse == 4:
            preTokens = preTokens_list[b_index]
        if sparse == 3 or sparse == 4:
            nextTokens = nextTokens_list[b_index]

        for n_index in range(numhead_value):   # 遍历n
            alibi_pse = None
            if len(qstartidx) == 0:
                qstartidx = [0]
            if len(kvstartidx) == 0:
                kvstartidx = [0]
            if psetype == 2 or psetype == 3:
                qs = q_shape[2]
                slope = -slopes[n_index]
                alibi_offset = kvstartidx[0] - qstartidx[0]
                alibi_pse = np.zeros((qs, qs), dtype=np.float32)
                for i in range(qs):
                    alibi_pse[i, :] = np.arange(alibi_offset - i, alibi_offset - i + qs, dtype=np.float32)
                print("alibi_pse:", alibi_pse)
                alibi_pse = np.abs(alibi_pse)
                if psetype == 3:
                    alibi_pse = np.sqrt(alibi_pse)
                alibi_pse = alibi_pse * slope
                alibi_pse = alibi_pse.reshape(1,1,qs,qs)
                # alibi_pse.dtype = np.float32



            if len(q_shape) == 4 and len(k_shape) == 4 and len(v_shape) == 4:

                # print(
        # f"k_tensor:{k_tensor.shape} v_tensor:{v_tensor.shape, v_tensor.dtype}")
                if (dequantScale_query is not None and  k_antiquant_scale is not None and v_antiquant_scale is not None):
                    dequantScale_query_cur = dequantScale_query[b_index:(b_index + 1), n_index:(n_index + 1), : , : ]
                    k_antiquant_scale_cur = k_antiquant_scale[b_index:(b_index + 1), n_index:(n_index + 1), : , :]
                    v_antiquant_scale_cur =v_antiquant_scale[b_index:(b_index + 1), n_index:(n_index + 1), : , :]

                q_tensor_cur = q_tensor[b_index:(b_index + 1), n_index:(n_index + 1), :, :]
                # print("q_tensor_cur.dtype is", q_tensor_cur.dtype)
                if tensor_list_flag==1:
                    k_tensor_cur = k_tensor[b_index:(b_index + 1), n_index:(n_index + 1), :, :]
                    v_tensor_cur = v_tensor[b_index:(b_index + 1), n_index:(n_index + 1), :, :]
                else:
                    k_tensor_cur = k_tensor[:, n_index:(n_index + 1), :, :]
                    v_tensor_cur = v_tensor[:, n_index:(n_index + 1), :, :]

                if "shared_prefix_k" in pfa_param:
                    actual_prefix_len = pfa_param['actualprefixKV']
                    if actual_prefix_len==0:
                        shared_prefix_k = pfa_param["shared_prefix_k"][:,n_index:(n_index + 1),:,:]
                        shared_prefix_v = pfa_param["shared_prefix_v"][:,n_index:(n_index + 1),:,:]
                    else:
                        shared_prefix_k = pfa_param["shared_prefix_k"][:, n_index:(n_index + 1), :actual_prefix_len, :]
                        shared_prefix_v = pfa_param["shared_prefix_v"][:, n_index:(n_index + 1), :actual_prefix_len, :]
                    k_tensor_cur = np.append(shared_prefix_k, k_tensor_cur, axis=2)
                    v_tensor_cur = np.append(shared_prefix_v, v_tensor_cur, axis=2)

                if mask_tensor is None:
                    mask_cur = None
                else:
                    mask_cur[0, 0, :, :] = mask_tensor[b_index]
                    # mask_cur = mask_tensor[b_index:(b_index + 1), n_index:(n_index + 1), :, :]

                if pse_bnsd_tensor is None:
                    pse_cur = None
                else:
                    pse_cur = pse_bnsd_tensor[b_index:(b_index + 1), n_index:(n_index + 1), :, :]

                if psetype == 2 or psetype == 3:
                    pse_cur = alibi_pse


                quant_scale2_cur = None
                quant_offset2_cur = None
                if quant_scale2 is not None:
                    if len(quant_scale2.shape) == 1:
                        quant_scale2_cur = quant_scale2
                    else:
                        quant_scale2_cur = quant_scale2[:, n_index:(n_index + 1), :, :]
                    if quant_offset2 is not None:
                        if len(quant_offset2.shape) == 1:
                            quant_offset2_cur = quant_offset2
                        else:
                            quant_offset2_cur = quant_offset2[:, n_index:(n_index + 1), :, :]

                antiquant_scale_k = None
                antiquant_scale_v = None
                antiquant_offset_k = None
                antiquant_offset_v = None
                if k_tensor_cur.dtype==np.int8 and q_tensor_cur.dtype!=np.int8:
                    if pfa_param['anti_or_kvanti']==1:
                        if len(antiquant_scale.shape) == 1:
                            antiquant_scale_k = antiquant_scale[0]
                            antiquant_scale_v = antiquant_scale[1]
                        else:
                            antiquant_scale_k = antiquant_scale[0,n_index:(n_index + 1),:,:]
                            antiquant_scale_v = antiquant_scale[1,n_index:(n_index + 1),:,:]
                        if antiquant_offset is not None:
                            if len(antiquant_offset.shape) ==1:
                                antiquant_offset_k = antiquant_offset[0]
                                antiquant_offset_v = antiquant_offset[1]
                            else:
                                antiquant_offset_k = antiquant_offset[0,n_index:(n_index + 1),:,:]
                                antiquant_offset_v = antiquant_offset[1,n_index:(n_index + 1),:,:]
                    if pfa_param['anti_or_kvanti']==2:#分离量化参数
                        if pfa_param['k_antiquant_mode'] ==0 :
                            if len(pfa_param['k_antiquant_scale'].shape) == 1:
                                antiquant_scale_k = pfa_param['k_antiquant_scale']
                                antiquant_scale_v = pfa_param['v_antiquant_scale']
                            else:
                                antiquant_scale_k = pfa_param['k_antiquant_scale'][n_index:(n_index + 1),:,:]
                                antiquant_scale_v = pfa_param['v_antiquant_scale'][n_index:(n_index + 1),:,:]
                            if pfa_param['k_antiquant_offset'] is not None:
                                if len(pfa_param['k_antiquant_offset'].shape) ==1:#petensor
                                    antiquant_offset_k = pfa_param['k_antiquant_offset']
                                    antiquant_offset_v = pfa_param['v_antiquant_offset']
                                else:
                                    antiquant_offset_k = pfa_param['k_antiquant_offset'][n_index:(n_index + 1),:,:]
                                    antiquant_offset_v = pfa_param['v_antiquant_offset'][n_index:(n_index + 1),:,:]
                        if pfa_param['k_antiquant_mode'] == 1:
                            antiquant_scale_k = pfa_param['k_antiquant_scale'][b_index:(b_index + 1),  :]
                            antiquant_scale_v = pfa_param['v_antiquant_scale'][b_index:(b_index + 1),  :]
                            if pfa_param['k_antiquant_offset'] is not None:
                                antiquant_offset_k = pfa_param['k_antiquant_offset'][b_index:(b_index + 1), :]
                                antiquant_offset_v = pfa_param['v_antiquant_offset'][b_index:(b_index + 1), :]

                sinks_cur = sinks[n_index:(n_index + 1)] if sinks is not None else None

                if q_tensor.dtype == "int8":
                    y[b_index:(b_index + 1), n_index:(n_index + 1), qs_begin:qs_end, :],lse[b_index:(b_index + 1), n_index:(n_index + 1), qs_begin:qs_end, :]= _np_pfaattention_act_int8(q_tensor_cur,
                                                                                                      k_tensor_cur,
                                                                                                      v_tensor_cur,
                                                                                                      pse_cur,
                                                                                                      mask_cur,
                                                                                                      scale,
                                                                                                      act_seq,
                                                                                                      act_kv_seq,
                                                                                                      preTokens,
                                                                                                      nextTokens,
                                                                                                      pfa_param,
                                                                                                      dequant_scale1,
                                                                                                      dequant_scale2,
                                                                                                      quant_scale1,
                                                                                                      quant_scale2_cur,
                                                                                                      quant_offset2_cur,
                                                                                                      out_dtype)
                elif q_tensor.dtype == "float8_e5m2" or q_tensor.dtype == "float8_e4m3fn" or q_tensor.dtype == torch.float8_e5m2 or q_tensor.dtype == torch.float8_e4m3fn:
                    if pfa_param['k_antiquant_mode'] != 7:
                        if (pfa_param['k_antiquant_mode']) == 6 and (pfa_param['query_quant_mode']) == 6 and (pfa_param['v_antiquant_mode']) == 8:
                            y,lse = None, None
                        else:
                            y[b_index:(b_index + 1), n_index:(n_index + 1), qs_begin:qs_end, :],lse[b_index:(b_index + 1), n_index:(n_index + 1), qs_begin:qs_end, :]= _np_pfaattention_act_fp8(q_tensor_cur,
                                                                                                      k_tensor_cur,
                                                                                                      v_tensor_cur,
                                                                                                      pse_cur,
                                                                                                      mask_cur,
                                                                                                      scale,
                                                                                                      act_seq,
                                                                                                      act_kv_seq,
                                                                                                      preTokens,
                                                                                                      nextTokens,
                                                                                                      pfa_param,
                                                                                                      dequant_scale1,
                                                                                                      dequant_scale2,
                                                                                                      quant_scale1,
                                                                                                      quant_scale2_cur,
                                                                                                      quant_offset2_cur,
                                                                                                      out_dtype)
                    elif pfa_param['k_antiquant_mode'] == 7:
                        y,lse = None, None
                elif q_dtype_origin == "hifloat8":
                    if pfa_param['k_antiquant_mode'] != 7:
                        y[b_index:(b_index + 1), n_index:(n_index + 1), qs_begin:qs_end, :],lse[b_index:(b_index + 1), n_index:(n_index + 1), qs_begin:qs_end, :]= _np_pfaattention_act_hifp8(q_tensor_cur,
                                                                                                        k_tensor_cur,
                                                                                                        v_tensor_cur,
                                                                                                        pse_cur,
                                                                                                        mask_cur,
                                                                                                        scale,
                                                                                                        act_seq,
                                                                                                        act_kv_seq,
                                                                                                        preTokens,
                                                                                                        nextTokens,
                                                                                                        pfa_param,
                                                                                                        dequant_scale1,
                                                                                                        dequant_scale2,
                                                                                                        quant_scale1,
                                                                                                        quant_scale2_cur,
                                                                                                        quant_offset2_cur,
                                                                                                        out_dtype)
                    elif pfa_param['k_antiquant_mode'] == 7:
                        y,lse = None, None
                else:
                    y[b_index:(b_index + 1), n_index:(n_index + 1), qs_begin:qs_end, :], lse[b_index:(b_index + 1), n_index:(n_index + 1), qs_begin:qs_end, :]= _np_pfaattention_act(q_tensor_cur,
                                                                                                 k_tensor_cur,
                                                                                                 v_tensor_cur,
                                                                                                 pse_cur,
                                                                                                 mask_cur, scale,
                                                                                                 act_seq,
                                                                                                 act_kv_seq,
                                                                                                 preTokens,
                                                                                                 nextTokens,
                                                                                                 pfa_param,
                                                                                                 quant_scale2_cur,
                                                                                                 quant_offset2_cur,
                                                                                                 antiquant_scale_k,
                                                                                                 antiquant_scale_v,
                                                                                                 antiquant_offset_k,
                                                                                                 antiquant_offset_v,
                                                                                                 out_dtype,
                                                                                                 sinks_cur)


    # print(f"11 {y.dtype}")
    return y,lse


def conv_float_to_u32(data_f):
    fp = ctypes.pointer(ctypes.c_float(data_f))
    cp = ctypes.cast(fp, ctypes.POINTER(ctypes.c_uint))
    data_hex = cp.contents.value

    result = (data_hex // 8192) * 8192

    return result


def trans_19bit(deqscale):
    res_19bit = np.zeros(deqscale.shape[0], dtype=np.uint64)

    for idx, scale in enumerate(deqscale):
        val = np.bitwise_and(((conv_float_to_u32(scale) >> 13) << 13), 0xffffffff)  # [31:13], M
        val = val.astype(np.uint64)
        res_19bit[idx] = val

    return res_19bit


def qtensor_seqlength(q_shape, inputLayout):
    if inputLayout == "SH":  # SH格式
        return q_shape[0]
    elif inputLayout == "BSH":
        return q_shape[1]
    elif inputLayout == "NSD":
        return q_shape[1]
    elif inputLayout == "BSND":
        return q_shape[1]
    else:  # BNSD
        return q_shape[2]


def aclnnPromptFlashAttention_NPU_unification(torch_tensor_list, params):
    q_tensor = torch_tensor_list[0]
    k_tensor = torch_tensor_list[1]
    v_tensor = torch_tensor_list[2]

    m_tensor = None

    num_heads = None
    input_layout = None
    scale_value = None
    actual_seq_lengths = None
    num_key_value_heads = None
    pre_tokens = None
    next_tokens = None
    actual_seq_lengths_kv = None
    sparse_mode = None

    num_heads = params['numheads']
    flagList = params['flaglist']
    if flagList[3]:
        p_tensor = torch_tensor_list[3]
    if flagList[4]:
        m_tensor = torch_tensor_list[4]

    if flagList[5]:
        actual_seq_lengths = params["actualseqlengths"]
    if flagList[6]:
        actual_seq_lengths_kv = params["actualseqlengthskv"]
    # if flagList[13]:
    #     scale_value = params["scalevalue"]
    # if flagList[14]:
    #     pre_tokens = params["pretokens"]
    # if flagList[15]:
    #     next_tokens = params["nexttokens"]
    # if flagList[16]:
    #     input_layout = params["inputlayout"]
    # if flagList[16]:
    #     num_key_value_heads = params["numkeyvalueheads"]
    # if flagList[17]:
    #     sparse_mode = params['sparsemode']

    scale_value = params['scalevalue'] if 'scalevalue' in params else 0
    pre_tokens = params['pretokens'] if 'pretokens' in params else 0
    next_tokens = params['nexttokens'] if 'nexttokens' in params else 0
    input_layout = params['inputlayout'] if 'inputlayout' in params else 0
    num_key_value_heads = params['numkeyvalueheads'] if 'numkeyvalueheads' in params else 0
    sparse_mode = params['sparsemode'] if 'sparsemode' in params else 0

    # torch_npu.npu_prompt_flash_attention
    # torch.ops.npu.npu_prompt_flash_attention
    print(q_tensor.dtype)
    import torch_npu
    npu_out = torch_npu.npu_prompt_flash_attention(q_tensor, k_tensor, v_tensor, atten_mask=m_tensor,
                                                   actual_seq_lengths=actual_seq_lengths, num_heads=num_heads,
                                                   scale_value=scale_value, pre_tokens=pre_tokens,
                                                   next_tokens=next_tokens, input_layout=input_layout,
                                                   num_key_value_heads=num_key_value_heads,
                                                   actual_seq_lengths_kv=actual_seq_lengths_kv,
                                                   sparse_mode=sparse_mode).cpu()
    return npu_out


def kvantiquantmode(flagList, torch_tensor_list, params, v_end_index, numKeyValueHeads):
    antiquant_scale = None
    antiquant_offset = None

    k_antiquant_scale = None
    v_antiquant_scale = None
    dequantScale_query = None

    k_antiquant_offset = None
    v_antiquant_offset = None

    anti_or_kvanti = 0
    # if len(flagList) > 12:
    # 获取kv量化参数
    flagtensor = 0
    if flagList[12]:
        anti_or_kvanti = 1
        antiquantscale_shape = params['shape_input'][v_end_index + 8]
        antiquantscale_tensor = torch_tensor_list[v_end_index + 8]
        if len(antiquantscale_shape) == 1:  # pertensor
            print("antiquantscale_pertensor")
            antiquant_scale = antiquantscale_tensor
        else:  # perchanel
            # 将kv量化参数转换为2n1d(匹配bnsd)
            antiquantscale_2n1d_tensor = _t_trans_2h_to_2n1d(antiquantscale_shape, antiquantscale_tensor,
                                                           numKeyValueHeads)

            antiquant_scale = antiquantscale_2n1d_tensor.permute(0, 2, 1, 3)
    if flagList[13]:
        antiquantoffset_shape = params['shape_input'][v_end_index + 9]
        antiquantoffset_tensor = torch_tensor_list[v_end_index + 9]
        if len(antiquantoffset_shape) == 1:
            print("antiquantoffset_pertensor")
            antiquant_offset = antiquantoffset_tensor
        else:
            antiquantoffset_2n1d_tensor = _t_trans_2h_to_2n1d(antiquantoffset_shape, antiquantoffset_tensor,
                                                            numKeyValueHeads)

            antiquant_offset = antiquantoffset_2n1d_tensor.permute(0, 2, 1, 3)
    if flagList[12]:
        return antiquant_scale[:1], antiquant_scale[1:2], antiquant_offset[:1], antiquant_offset[1:2]
    # 判断kv分离的伪量化
    if flagList[17] or flagList[19]:
        anti_or_kvanti = 2
        q_dtype_np = tools.get_np_dtype(params['dtype_input'][0])
        k_antiquantscale_shape = params['shape_input'][v_end_index + 13]
        k_antiquantscale_tensor = torch_tensor_list[v_end_index + 13]
        v_antiquantscale_shape = params['shape_input'][v_end_index + 15]
        v_antiquantscale_tensor = torch_tensor_list[v_end_index + 15]
        if v_end_index + 25 < len(params['shape_input']):
            dequantScale_query_shape = params['shape_input'][v_end_index + 25]
        else:
            dequantScale_query_shape = None
        if v_end_index + 25 < len(torch_tensor_list):
            dequantScale_query_tensor = torch_tensor_list[v_end_index + 25]
        else:
            dequantScale_query_tensor = None

        if params['k_antiquant_mode'] == 0:
            if len(k_antiquantscale_shape) == 1 and k_antiquantscale_shape[0] == 1:  # pertensor
                print("antiquantscale_pertensor")
                k_antiquant_scale = k_antiquantscale_tensor
                v_antiquant_scale = v_antiquantscale_tensor
            else:  # perchanel
                # 将kv量化参数转换为n1d(匹配bnsd)
                k_antiquantscale_n1d_tensor = _t_trans_h_to_n1d(k_antiquantscale_shape, k_antiquantscale_tensor,
                                                              numKeyValueHeads)
                v_antiquantscale_n1d_tensor = _t_trans_h_to_n1d(v_antiquantscale_shape, v_antiquantscale_tensor,
                                                              numKeyValueHeads)


                k_antiquant_scale = k_antiquantscale_n1d_tensor.permute(0, 2, 1, 3)
                v_antiquant_scale = v_antiquantscale_n1d_tensor.permute(0, 2, 1, 3)
        if params['k_antiquant_mode'] == 1:  # pertoken
            # pertoken -> BS
            k_antiquant_scale = k_antiquantscale_tensor.reshape(k_antiquantscale_tensor.shape[0],k_antiquantscale_tensor.shape[1],1,1)
            v_antiquant_scale = v_antiquantscale_tensor.reshape(v_antiquantscale_tensor.shape[0],v_antiquantscale_tensor.shape[1],1,1)

    if flagList[18] or flagList[20]:
        q_dtype_np = tools.get_np_dtype(params['dtype_input'][0])
        k_antiquantoffset_shape = params['shape_input'][v_end_index + 14]
        k_antiquantoffset_tensor = torch_tensor_list[v_end_index + 14]
        v_antiquantoffset_shape = params['shape_input'][v_end_index + 16]
        v_antiquantoffset_tensor = torch_tensor_list[v_end_index + 16]

        if params['k_antiquant_mode'] == 0:
            if len(k_antiquantoffset_shape) == 1 and k_antiquantoffset_shape[0] == 1:  # pertensor
                print("antiquantscale_pertensor")
                k_antiquant_offset = k_antiquantoffset_tensor
                v_antiquant_offset = v_antiquantoffset_tensor
            else:  # perchanel
                # 将kv量化参数转换为2n1d(匹配bnsd)
                k_antiquantoffset_n1d_tensor = _t_trans_h_to_n1d(k_antiquantoffset_shape,
                                                               k_antiquantoffset_tensor, numKeyValueHeads)
                v_antiquantoffset_n1d_tensor = _t_trans_h_to_n1d(v_antiquantoffset_shape,
                                                               v_antiquantoffset_tensor, numKeyValueHeads)


                k_antiquant_offset = k_antiquantoffset_n1d_tensor.permute(0, 2, 1, 3)
                v_antiquant_offset = v_antiquantoffset_n1d_tensor.permute(0, 2, 1, 3)
        if params['k_antiquant_mode'] == 1:  # pertoken
            # pertoken -> BS
            k_antiquant_offset = k_antiquantoffset_tensor.reshape(k_antiquantoffset_tensor.shape[0],k_antiquantoffset_tensor.shape[1],1,1)
            v_antiquant_offset = v_antiquantoffset_tensor.reshape(v_antiquantoffset_tensor.shape[0],v_antiquantoffset_tensor.shape[1],1,1)
    if flagList[17]:
        return k_antiquant_scale, v_antiquant_scale, k_antiquant_offset, v_antiquant_offset
    return k_antiquant_scale, v_antiquant_scale, dequantScale_query, k_antiquant_offset, v_antiquant_offset


def aclnnPromptFlashAttention_GPU_unification(torch_tensor_list, params):
    enable_gpu = params['enablegpu']
    print("#" * 20, "enable_gpu")
    print(enable_gpu)
    innerprecise = params['innerprecise']
    if enable_gpu == "True" or enable_gpu == "TRUE":
        print("#" * 20)
        print("gpu biggggggg")
        from flash_attn.flash_attn_interface import flash_attn_with_kvcache
        from flash_attn.flash_attn_interface import flash_attn_func
        # 设置默认值
        # scaleValue = None
        # numKeyValueHeads = 0
        # sp_mode = 0
        flagList = params['flaglist']

        scaleValue = params['scalevalue'] if 'scalevalue' in params else 0
        preTokens = params['pretokens'] if 'pretokens' in params else 0
        nextTokens = params['nexttokens'] if 'nexttokens' in params else 0
        inputLayout = params['inputlayout'] if 'inputlayout' in params else 0
        numKeyValueHeads = params['numkeyvalueheads'] if 'numkeyvalueheads' in params else 0
        sp_mode = params['sparsemode'] if 'sparsemode' in params else 0

        q_shape = params["shape_input"][0]
        k_shape = params["shape_input"][1]
        # inputLayout = params["inputlayout"]
        numHeads = params["numheads"]

        actualSeqLengthsKV = []
        actualSeqLengths = params['actualseqlengths']
        if numKeyValueHeads == 0:
            numKeyValueHeads = numHeads
        if inputLayout == "SH":
            actualSeqLengthsKV = actualSeqLengths
        else:
            actualSeqLengthsKV = params['actualseqlengthskv']

        if inputLayout == "TND" or inputLayout == "NTD_TND":
            # 将tnd格式下的act seq转成普通的act seq
            actualSeqLengths = trans_tnd_actseq(actualSeqLengths)
            actualSeqLengthsKV = trans_tnd_actseq(actualSeqLengthsKV)

        # k和v的shape_list只能通过抛去已知tensor个数，除2计算
        num_input = len(params['shape_input'])
        tensornum = 23
        # if len(flagList) > 23:
        #     tensornum = 14
        # if len(flagList) > 25:
        #     tensornum = 16
        # if len(flagList) > 28:
        #     tensornum = 20
        k_shape_num = v_shape_num = int((num_input - tensornum) / 2)
        print(f"k_shape_num:{k_shape_num}")

        k_start_index = 1
        k_end_index = int(k_shape_num)
        v_start_index = int(k_shape_num) + 1
        v_end_index = int(k_shape_num + v_shape_num)

        # 转成gpu需要的bsnd
        q_tensor, q_bnsd_shape = t_bsh_to_bsnd(torch_tensor_list[0], q_shape, numHeads, actualSeqLengths, inputLayout)

        k_tensor, k_bnsd_shape = t_bsh_to_bsnd(torch_tensor_list[1], k_shape, numKeyValueHeads, actualSeqLengthsKV,
                                               inputLayout)
        v_tensor, v_bnsd_shape = t_bsh_to_bsnd(torch_tensor_list[2], k_shape, numKeyValueHeads, actualSeqLengthsKV,
                                               inputLayout)

        # if len(flagList) > 23:
        if flagList[15]:  # queryPaddingSize
            pass
            # fanzhuanq
        if flagList[16]:  # kvPaddingSize
            pass
            # 翻转kv
            # pfa_param['kvPaddingSize'] = torch_tensor_list[v_end_index + 14]

        #判断是否伪量化
        k_antiquant_scale,v_antiquant_scale,dequantScale_query,k_antiquant_offset,v_antiquant_offset = kvantiquantmode(flagList,torch_tensor_list,params,v_end_index,numKeyValueHeads)

        # KVcahce反量化  这块逻辑需要修改
        if q_tensor.dtype != torch.int8 and k_tensor.dtype == torch.int8:
            k_tensor = k_tensor.to(torch.float16)
            v_tensor = v_tensor.to(torch.float16)
            if k_antiquant_offset is not None:
                k_tensor += k_antiquant_offset
                v_tensor += v_antiquant_offset
            k_tensor *= k_antiquant_scale
            v_tensor *= v_antiquant_scale
            k_tensor = k_tensor.to(torch.bfloat16)
            v_tensor = v_tensor.to(torch.bfloat16)
        pse = None
        if flagList[3]:
            pse = get_slopes(numHeads).cuda().requires_grad_(False)
        k_cache = k_tensor
        v_cache = v_tensor
        q_tensor = q_tensor.cuda().requires_grad_(False)
        k_cache = k_cache.cuda().requires_grad_(False)
        v_cache = v_cache.cuda().requires_grad_(False)

        if len(actualSeqLengthsKV) == 0:
            actualSeqLengthsKV = None
        else:
            actualSeqLengthsKV = torch.from_numpy(np.array(actualSeqLengthsKV)).to(torch.int32)
            actualSeqLengthsKV = actualSeqLengthsKV.cuda().requires_grad_(False)

        causal_switch = False
        window_size0 = -1
        window_size1 = -1
        qs = q_bnsd_shape[1]
        kvs = k_bnsd_shape[1]
        if sp_mode == 0:
            if actualSeqLengthsKV is not None:
                window_size0 = preTokens + actualSeqLengthsKV[0] - qs
                window_size1 = nextTokens - actualSeqLengthsKV[0] + qs
            else:
                window_size0 = preTokens + kvs - qs  # 65535+4096-512
                window_size1 = nextTokens - kvs + qs  # 0-4096+512
        if sp_mode == 2:
            if actualSeqLengthsKV is not None:
                window_size1 = 0 - actualSeqLengthsKV[0] + qs
            else:
                window_size1 = 0 - kvs + qs
        if sp_mode == 3:
            causal_switch = True
        if sp_mode == 4:
            window_size0 = preTokens
            window_size1 = nextTokens

        if ("softmaxlseflag" in params and params["softmaxlseflag"] == True) or (
                "softmax_lse_flag" in params and params["softmax_lse_flag"] == True):
            out, lse, smask = flash_attn_func(q=q_tensor, k=k_cache, v=v_cache,
                                              softmax_scale=scaleValue, causal=causal_switch,
                                              window_size=(window_size0, window_size1), alibi_slopes=pse,
                                              return_attn_probs=True)
            lse = lse.cpu()
        else:
            out = flash_attn_with_kvcache(q=q_tensor, k_cache=k_cache, v_cache=v_cache, k=None, v=None,
                                          cache_seqlens=actualSeqLengthsKV,
                                          softmax_scale=scaleValue, causal=causal_switch, alibi_slopes=pse,
                                          window_size=(window_size0, window_size1), num_splits=0)

        y_all = out.cpu()

        # 增加quant
        if flagList[10]:
            quant_scale2 = torch_tensor_list[6]
            quant_scale2_shape = params['shape_input'][6]
            per_channel = True
            if len(quant_scale2_shape) == 1:
                if quant_scale2_shape[0] == 1:
                    per_channel = False
            if per_channel:
                quant_scale2 = _t_trans_1h_to_1n1d(quant_scale2_shape, quant_scale2,
                                                   numHeads, inputLayout)
            y_all = y_all * quant_scale2.cpu().to(torch.float16)
        if flagList[11]:
            quant_offset2 = torch_tensor_list[7]
            quant_offset2_shape = params['shape_input'][7]
            per_channel = True
            if len(quant_offset2_shape) == 1:
                if quant_offset2_shape[0] == 1:
                    per_channel = False
            if per_channel:
                quant_offset2 = _t_trans_1h_to_1n1d(quant_offset2_shape, quant_offset2,
                                                    numHeads, inputLayout)
            y_all += quant_offset2.to(torch.float16)
        if flagList[10]:
            y_all = torch.round(y_all)
            # 转成int8防止溢出
            y_all = torch.where(y_all > 127, 127, y_all)
            y_all = torch.where(y_all < -128, -128, y_all)
            y_all = y_all.to(torch.int8)

        out_shape = q_shape
        if inputLayout == "BSH":
            y_all = y_all.reshape(out_shape)
        elif inputLayout == "NSD":
            y_all = y_all.permute(0, 2, 1, 3).reshape(out_shape)
        elif inputLayout == "TND":
            T = sum(actualSeqLengths)
            B = len(actualSeqLengths)
            N = out_shape[1]
            D = out_shape[2]
            output = np.zeros((T, N, D), dtype=y_all.dtype)
            t_start = 0
            for b_index in range(B):
                act_s = actualSeqLengths[b_index]
                t_end = t_start + act_s
                if act_s == 0:
                    continue
                for n_index in range(N):
                    output[t_start:t_end, n_index, :] = y_all[b_index, n_index, :act_s, :]
                t_start += act_s
            y_all = output
        elif len(out_shape) == 2:
            if q_bnsd_shape[0] == 1:
                y_all = y_all.reshape(out_shape).astype(np.float32)
            else:
                # print(y_all)
                sums = 0
                B = q_bnsd_shape[0]
                N = q_bnsd_shape[1]
                S = q_bnsd_shape[2]
                D = q_bnsd_shape[3]
                # y_all = y_all.transpose(0, 2, 1, 3)
                yNewAll = torch.zeros((S, N * D))
                for i in range(B):
                    for j in range(actualSeqLengths[i]):
                        yNewAll[sums + j, :] = y_all[i:i + 1, j].reshape(1, N * D)
                    sums += actualSeqLengths[i]
                y_all = yNewAll.astype(np.float32)
        elif inputLayout == "BNSD":
            y_all = y_all.permute([0, 2, 1, 3])
        else:
            y_all = y_all
        if ("softmaxlseflag" in params and params["softmaxlseflag"] == True) or (
                "softmax_lse_flag" in params and params["softmax_lse_flag"] == True):
            return y_all, lse
        else:
            return y_all
    elif enable_gpu == "SMALL" or enable_gpu == "False" or enable_gpu == "FALSE":  # 走小算子
        print("#" * 20)
        print("small")
        flagList = params['flaglist']
        actualSeqLengths = params['actualseqlengths']
        actualSeqLengthsKV = None
        numHeads = params['numheads']
        # 设置参数默认值
        scaleValue = 1
        preTokens = 214748647
        nextTokens = 0
        inputLayout = "BSH"
        numKeyValueHeads = 0
        sp_mode = 0
        # if flagList[13]:
        #     scaleValue = params['scalevalue']
        # if flagList[14]:
        #     preTokens = params['pretokens']
        # if flagList[15]:
        #     nextTokens = params['nexttokens']
        # if flagList[16]:
        #     inputLayout = params['inputlayout']
        # if flagList[17]:
        #     numKeyValueHeads = params['numkeyvalueheads']
        # if flagList[18]:
        #     sp_mode = params['sparsemode']
        scaleValue = params['scalevalue'] if 'scalevalue' in params else 0
        preTokens = params['pretokens'] if 'pretokens' in params else 0
        nextTokens = params['nexttokens'] if 'nexttokens' in params else 0
        inputLayout = params['inputlayout'] if 'inputlayout' in params else 0
        numKeyValueHeads = params['numkeyvalueheads'] if 'numkeyvalueheads' in params else 0
        sp_mode = params['sparsemode'] if 'sparsemode' in params else 0

        if numKeyValueHeads == 0:
            numKeyValueHeads = numHeads

        q_shape = params['shape_input'][0]
        q_dtype = tools.get_np_dtype(params['dtype_input'][0])  # numpy类型
        out_dtype = tools.get_np_dtype(params['dtype_output'][0])  # numpy类型

        if inputLayout == "SH":
            actualSeqLengthsKV = actualSeqLengths
        else:
            actualSeqLengthsKV = params['actualseqlengthskv']
        # q_tensor,q_bnsd_shape = _t_trans_bsh_to_bnsd(torch_tensor_list[0],q_shape,numHeads)
        q_tensor, q_bnsd_shape = _t_trans_bsh_to_bnsd(torch_tensor_list[0], q_shape, numHeads, actualSeqLengths,
                                                      inputLayout)
        kv_empty_flag = False
        k_shape = params['shape_input'][1]

        if 0 in k_shape or len(k_shape) == 0:
            kv_empty_flag = True
        else:
            # k_tensor,k_bnsd_shape = _t_trans_bsh_to_bnsd(torch_tensor_list[1],k_shape,numKeyValueHeads)
            k_tensor, k_bnsd_shape = _t_trans_bsh_to_bnsd(torch_tensor_list[1], k_shape, numKeyValueHeads,
                                                          actualSeqLengthsKV,
                                                          inputLayout)
            if numKeyValueHeads != numHeads:
                print("broadcast k")
                k_dtype = tools.get_pt_dtype(params['dtype_input'][1])
                print(k_dtype)
                k_tensor, k_bnsd_shape = _t_broadcastKV_sigle(numHeads, numKeyValueHeads, k_tensor)
                k_tensor = k_tensor.to(k_dtype)
        v_shape = params['shape_input'][2]
        if 0 in v_shape or len(v_shape) == 0 or kv_empty_flag:
            kv_empty_flag = True
        else:
            # v_tensor,v_bnsd_shape = _t_trans_bsh_to_bnsd(torch_tensor_list[2],k_shape,numKeyValueHeads)
            v_tensor, v_bnsd_shape = _t_trans_bsh_to_bnsd(torch_tensor_list[2], k_shape, numKeyValueHeads,
                                                          actualSeqLengthsKV,
                                                          inputLayout)
            if numKeyValueHeads != numHeads:
                print("broadcast v")
                v_dtype = tools.get_pt_dtype(params['dtype_input'][3])
                v_tensor, v_bnsd_shape = _t_broadcastKV_sigle(numHeads, numKeyValueHeads, v_tensor)
                v_tensor = v_tensor.to(k_dtype)
        pse_bnsd_tensor = None
        pse_shift_shape = params['shape_input'][3]
        if flagList[3] == 0 or 0 in pse_shift_shape:
            pse_bnsd_tensor = None
        else:
            pse_shift = torch_tensor_list[3]

            pse_bnsd_tensor = _t_broadcast_pseShift_n(pse_shift, pse_shift_shape, q_bnsd_shape[0])  # to bnsd

        m_bnsd_tensor = None
        npu_m_shape = params['shape_input'][4]
        qs = q_bnsd_shape[2]
        kvs = k_bnsd_shape[2]
        m_dtype = tools.get_pt_dtype(params['dtype_input'][4])
        if flagList[4] == 0 or 0 in npu_m_shape:
            m_bnsd_tensor = None
        else:
            batch = q_bnsd_shape[0]
            numheads = q_bnsd_shape[1]
            if sp_mode == 0 or sp_mode == 1:
                cpu_m_shape = npu_m_shape
            else:  # sparse 为leftUpCausal2  rightDownCausal3  band4
                cpu_m_shape = [qs, kvs]  # cpu
                if sp_mode != 4:
                    preTokens = 214748647

            cpu_m_tensor = _t_gpu_create_random_mask_by_spars(cpu_m_shape, npu_m_shape, m_dtype, preTokens,
                                                              nextTokens,
                                                              actualSeqLengths, actualSeqLengthsKV, batch,
                                                              numheads, sp_mode, random_ones=0)
            if sp_mode != 3:
                m_bnsd_tensor = _t_broadcast_mask_n(cpu_m_tensor, cpu_m_shape, numHeads, q_bnsd_shape[0])
            else:
                m_bnsd_tensor = cpu_m_tensor

        dequant_scale1, dequant_scale2, quant_scale1, quant_scale2, quant_offset2 = None, None, None, None, None

        out_shape = q_shape
        if kv_empty_flag:
            return torch.zeros(out_shape)
        else:
            y_all = _t_promtattention_bnsd(q_tensor, q_bnsd_shape, k_tensor, k_bnsd_shape, v_tensor, v_bnsd_shape,
                                           pse_bnsd_tensor,
                                           m_bnsd_tensor, scaleValue, actualSeqLengths, actualSeqLengthsKV, preTokens,
                                           nextTokens,
                                           dequant_scale1,
                                           dequant_scale2, quant_scale1, quant_scale2, quant_offset2, out_dtype,
                                           innerprecise)
            if inputLayout == "BSH":
                y_all = y_all.permute(0, 2, 1, 3).reshape(out_shape)
            elif inputLayout == "NSD":
                y_all = y_all.reshape(out_shape)
            elif inputLayout == "TND":
                T = sum(actualSeqLengths)
                B = len(actualSeqLengths)
                N = out_shape[1]
                D = out_shape[2]
                output = np.zeros((T, N, D), dtype=y_all.dtype)
                t_start = 0
                for b_index in range(B):
                    act_s = actualSeqLengths[b_index]
                    t_end = t_start + act_s
                    if act_s == 0:
                        continue
                    for n_index in range(N):
                        output[t_start:t_end, n_index, :] = y_all[b_index, n_index, :act_s, :]
                    t_start += act_s
                y_all = output
            elif len(out_shape) == 2:
                if q_bnsd_shape[0] == 1:
                    y_all = y_all.permute(0, 2, 1, 3).reshape(out_shape).to(torch.float32)
                else:
                    # print(y_all)
                    sums = 0
                    B = q_bnsd_shape[0]
                    N = q_bnsd_shape[1]
                    S = q_bnsd_shape[2]
                    D = q_bnsd_shape[3]
                    y_all = y_all.permute(0, 2, 1, 3)
                    yNewAll = torch.zeros((S, N * D))
                    for i in range(B):
                        for j in range(actualSeqLengths[i]):
                            yNewAll[sums + j, :] = y_all[i:i + 1, j].reshape(1, N * D)
                        sums += actualSeqLengths[i]
                    y_all = yNewAll.to(torch.float32)
            else:
                y_all = y_all
            # y_all = torch.from_numpy(y_all)
            print(y_all)
            return y_all


def get_attention_mask_batch_num(npu_m_shape, q_bnsd_shape):
    batch, numhead = None, None
    if len(npu_m_shape) == 2:
        s1 = npu_m_shape[0]
        s2 = npu_m_shape[1]
        return batch, numhead, s1, s2
    if len(npu_m_shape) == 3:
        batch = npu_m_shape[0]
        s1 = npu_m_shape[1]
        s2 = npu_m_shape[2]
        return batch, numhead, s1, s2
    if len(npu_m_shape) == 4:
        batch = npu_m_shape[0]
        numhead = npu_m_shape[1]
        s1 = npu_m_shape[2]
        s2 = npu_m_shape[3]
        return batch, numhead, s1, s2


def _trans_2h_to_2n1d(shape, tensor, numKeyValueHeads):
    if len(shape) == 4:  # 2N1D
        print("2N1D")
        return tensor
    elif len(shape) == 2:  # 2H
        d_num = shape[1] // numKeyValueHeads
        print(f"[INFO]_n_trans_2h_to_2n1d : h={shape[1]}, n={numKeyValueHeads} ,d(h/n)={d_num}")
        new_tensor = tensor.reshape(2, 1, numKeyValueHeads, d_num).transpose(0, 2, 1, 3)
        return new_tensor
    elif len(shape) == 3:  # 2ND
        print("2ND")
        N = shape[1]
        D = shape[2]
        new_tensor = tensor.reshape(2, 1, N, D).transpose(0, 2, 1, 3)
        return new_tensor
    else:
        print(f"[ERROR]_n_trans_2h_to_2n1d : len(shape):{len(shape)}")
        return tensor


def _t_trans_2h_to_2n1d(shape, tensor, numKeyValueHeads):
    if len(shape) == 4:  # 2N1D
        print("2N1D")
        return tensor
    elif len(shape) == 2:  # 2H
        d_num = shape[1] // numKeyValueHeads
        print(f"[INFO]_n_trans_2h_to_2n1d : h={shape[1]}, n={numKeyValueHeads} ,d(h/n)={d_num}")
        new_tensor = tensor.reshape(2, 1, numKeyValueHeads, d_num).permute(0, 2, 1, 3)
        return new_tensor
    elif len(shape) == 3:  # 2ND
        print("2ND")
        N = shape[1]
        D = shape[2]
        new_tensor = tensor.reshape(2, 1, N, D).permute(0, 2, 1, 3)
        return new_tensor
    else:
        print(f"[ERROR]_n_trans_2h_to_2n1d : len(shape):{len(shape)}")
        return tensor


def _trans_h_to_n1d(shape, tensor, numKeyValueHeads):
    if len(shape) == 3:  # N1D
        print("N1D")
        return tensor
    elif len(shape) == 1:  # H
        d_num = shape[0] // numKeyValueHeads
        print(f"[INFO]_n_trans_h_to_n1d : h={shape[0]}, n={numKeyValueHeads} ,d(h/n)={d_num}")
        new_tensor = tensor.reshape(1, numKeyValueHeads, d_num).transpose(1, 0, 2)
        return new_tensor
    elif len(shape) == 2:  # ND
        print("ND")
        N = shape[0]
        D = shape[1]
        new_tensor = tensor.reshape(1, N, D).transpose(1, 0, 2)
        return new_tensor
    else:
        print(f"[ERROR]_n_trans_1h_to_1n1d : len(shape):{len(shape)}")
        return tensor


def _t_trans_h_to_n1d(shape, tensor, numKeyValueHeads):
    if len(shape) == 3:  # N1D
        print("N1D")
        return tensor
    elif len(shape) == 1:  # H
        d_num = shape[0] // numKeyValueHeads
        print(f"[INFO]_n_trans_h_to_n1d : h={shape[1]}, n={numKeyValueHeads} ,d(h/n)={d_num}")
        new_tensor = tensor.reshape(1, numKeyValueHeads, d_num).permute(1, 0, 2)
        return new_tensor
    elif len(shape) == 2:  # ND
        print("ND")
        N = shape[0]
        D = shape[1]
        new_tensor = tensor.reshape(1, N, D).permute(1, 0, 2)
        return new_tensor
    else:
        print(f"[ERROR]_n_trans_1h_to_1n1d : len(shape):{len(shape)}")
        return tensor


def _t_trans_1h_to_1n1d(shape, tensor, numHeads, inputLayout):
    if len(shape) == 4:
        # 1n1d->1n1d
        if inputLayout == "BSND":  # 1,1,n,d
            return tensor
        else:  # BNSD 1N1D
            return tensor.transpose(1, 2)
    elif len(shape) == 3:
        if inputLayout == "BSND":  # 1,n,d ->1,n,1,d
            n = shape[1]
            d = shape[2]
            return tensor.reshape(1, 1, n, d)
        elif inputLayout == "BNSD" or inputLayout == "NSD":  # n,1,d -> 1,n,1,d
            n = shape[0]
            d = shape[2]
            return tensor.reshape(1, n, 1, d).transpose(1, 2)
        else:  # bsh 11h->1n1d
            h = shape[2]
            d = h // numHeads
            return tensor.reshape(1, 1, numHeads, d)
    elif len(shape) == 2:
        if inputLayout == "BSH":  # 1,H->1,n,1,d
            d = shape[1] // numHeads
            return tensor.reshape(1, 1, numHeads, d)
        else:  # BSND ND ->1N1D
            n = shape[0]
            d = shape[1]
            return tensor.reshape(1, 1, n, d)
    elif len(shape) == 1:
        # h->1n1d
        h = shape[0]
        d = h // numHeads
        return tensor.reshape(1, 1, numHeads, d)
    else:
        print("[ERROR]trans_to_1n1d: Unknown input shape!")
        exit(1)


def _trans_1h_to_1n1d(shape, tensor, numHeads, inputLayout):
    if len(shape) == 4:
        # 1n1d->1n1d
        if inputLayout == "BSND":  # 1,1,n,d
            return tensor.transpose(0, 2, 1, 3)
        else:  # BNSD 1N1D
            return tensor
    elif len(shape) == 3:
        if inputLayout == "BSND":  # 1,n,d ->1,n,1,d
            n = shape[1]
            d = shape[2]
            return tensor.reshape(1, 1, n, d).transpose(0, 2, 1, 3)
        elif inputLayout == "BNSD" or inputLayout == "NSD" or inputLayout == "BNSD_BSND" or inputLayout == "TND":  # n,1,d -> 1,n,1,d
            n = shape[0]
            d = shape[2]
            return tensor.reshape(1, n, 1, d)
        else:  # bsh 11h->1n1d
            h = shape[2]
            d = h // numHeads
            return tensor.reshape(1, 1, numHeads, d).transpose(0, 2, 1, 3)
    elif len(shape) == 2:
        if inputLayout == "BSH":  # 1,H->1,n,1,d
            d = shape[1] // numHeads
            return tensor.reshape(1, 1, numHeads, d).transpose(0, 2, 1, 3)
        else:  # BSND ND ->1N1D
            n = shape[0]
            d = shape[1]
            return tensor.reshape(1, 1, n, d).transpose(0, 2, 1, 3)
    elif len(shape) == 1:
        # h->1n1d
        h = shape[0]
        d = h // numHeads
        return tensor.reshape(1, 1, numHeads, d).transpose(0, 2, 1, 3)
    else:
        print("[ERROR]trans_to_1n1d: Unknown input shape!")
        exit(1)


def gen_outshape(layout, qshape, vshape, numKeyValueHeads, numHeads):
    outshape = copy.deepcopy(qshape)
    if layout == "BSH":
        vd = int(vshape[-1] / numKeyValueHeads)
        outshape[-1] = vd * numHeads
        return outshape
    else:
        outshape[-1] = vshape[-1]
        return outshape


def concat_tensor(tensor1, shape1, tensor2, shape2, n, tnd_flag=False):
    if len(shape1) != len(shape2):
        print(f"[ERROR]concat_tensor: 相加的两个tensor 维数不同! shape1 = {shape1}, shape2 =  {shape2}")
        return None, None
    elif len(shape1) == 2:
        # 量化参数1h
        if shape1[0] != shape2[0]:
            print(f"[ERROR]concat_tensor: 相加的两个tensor 维度非法! shape1 = {shape1}, shape2 =  {shape2}")
            return None, None
        d1 = int(shape1[1] / n)
        d2 = int(shape2[1] / n)
        tensor1_1nd = tensor1.reshape(shape1[0], n, d1)
        tensor2_1nd = tensor2.reshape(shape2[0], n, d2)
        concatenated_tensor = np.concatenate((tensor1_1nd, tensor2_1nd), axis=2)
        concatenated_shape = [shape1[0], shape1[1] + shape2[1]]
        concatenated_tensor = concatenated_tensor.reshape(concatenated_shape)
    elif len(shape1) == 3:
        if shape1[0] != shape2[0] or shape1[1] != shape2[1]:
            print(f"[ERROR]concat_tensor: 相加的两个tensor 维度非法! shape1 = {shape1}, shape2 =  {shape2}")
            return None, None
        if tnd_flag:
            concatenated_tensor = np.concatenate((tensor1, tensor2), axis=2)
            concatenated_shape = [shape1[0], shape1[1], shape1[2] + shape2[2]]
        else:
            d1 = int(shape1[2] / n)
            d2 = int(shape2[2] / n)
            tensor1_bsnd = tensor1.reshape(shape1[0], shape1[1], n, d1)
            tensor2_bsnd = tensor2.reshape(shape2[0], shape2[1], n, d2)
            concatenated_tensor = np.concatenate((tensor1_bsnd, tensor2_bsnd), axis=3)
            concatenated_shape = [shape1[0], shape1[1], shape1[2] + shape2[2]]
            concatenated_tensor = concatenated_tensor.reshape(concatenated_shape)
    elif len(shape1) == 4:
        if shape1[0] != shape2[0] or shape1[1] != shape2[1] or shape1[2] != shape2[2]:
            print(f"[ERROR]concat_tensor: 相加的两个tensor 维度非法! shape1 = {shape1}, shape2 =  {shape2}")
            return None, None
        concatenated_tensor = np.concatenate((tensor1, tensor2), axis=3)
        concatenated_shape = [shape1[0], shape1[1], shape1[2], shape1[3] + shape2[3]]
    else:
        print(f"[ERROR]concat_tensor: shape1 维度非法! shape1 = {shape1}")
        return None, None
    return concatenated_tensor, concatenated_shape

def mxfp8_pa_preprocessing(pfa_param, params, tensor_bnsd, index, is_vscale = False):
    """
    MXFP8 PA 预处理：将 BNSD 张量转为 PagedAttention 块格式
    输入：BNSD = [Batch, N(Head), Seq, D]
    输出：
        dim=4 → [BlockNum, N, BlockSize, D] (BNBD)
        dim=5 → [BlockNum, N, BlockSize, D2, 2]
    特殊：is_vscale=True 时，拆尾轴2是在seq维度，不是D维度，因此 seq = 原始seq / 32 / 2 = 原始seq / 64
    """
    dim = len(params['shape_input'][index])
    numKeyValueHeads = params['numkeyvalueheads']
    actualSeqLengthsKV = pfa_param['actualSeqLengthsKV']
    blockSize = params['blocksize']
    block_table = pfa_param['block_table']

    KV_layout = "none"
    KV_cache = params['shape_input'][21]
    if KV_cache[1] == blockSize and len(params['shape_input'][21]) == 4:
        KV_layout = "BnBsND"
        print(f"++++++++++++++++KV_layout====={KV_layout}")
    elif KV_cache[2] == blockSize and len(params['shape_input'][21]) == 4:
        KV_layout = "BnNBsD"
        print(f"++++++++++++++++KV_layout====={KV_layout}")
    elif len(params['shape_input'][21]) == 5:
        KV_layout = "PA_NZ"
        print(f"++++++++++++++++KV_layout====={KV_layout}")
    else:
        raise ValueError(f"不支持的 KV_cache_layout = {params['shape_input'][21]}")

    # blockNumPerBlock 通常是未压缩场景下的块数列表
    B, N, S, D = tensor_bnsd.shape
    block_num = []

    print("pJJ mxfp8_pa_preprocessing")
    print("B, N, S, D", {B, N, S, D})

    if is_vscale:
        compress_ratio = 64
        for actual_seq in actualSeqLengthsKV:
            # 目标序列长度
            target_seq_len = int(math.ceil(actual_seq / compress_ratio))
            blockSize = math.ceil(params['blocksize'] / compress_ratio)
            # 所需块数
            block_num.append(int(math.ceil(target_seq_len / blockSize)))
    else:
        if 'blockNumPerBlock' in pfa_param:
            block_num = pfa_param['blockNumPerBlock']
        else:
            for actual_seq in actualSeqLengthsKV:
                block_num.append(int(math.ceil(actual_seq / blockSize)))
    total_block_num = sum(block_num)
    print("total_block_num", {total_block_num})

    # 2. 数据预处理与维度变换
    # 使用 reshape 保证尾轴拆分后的数据排序与原始内存布局一致

    # KV cache 处理
    if index == 21 or index == 22 or index == 25:
        tensor_processed = tensor_bnsd
        out_shape = (total_block_num, N, blockSize, D)

        print("out_shape", {out_shape})

    elif index == 15 or index == 17:    # KV Scale 处理
        if is_vscale:
            # Vscale: 拆分 S 维度
            # 输入 tensor_bnsd 形状: S_in = actual_seq / 32
            # 目标: 在 S 维度拆尾轴 2 -> S_out = S_in / 2 = actual_seq / 64
            # 输出: [B, N, S_in//2, D, 2]
            # 使用 convert_v_scale_bnsd_to_layout 将 S 维度拆分: (s0,s1,s2,s3) ->  (s1,s3), (s0,s2)...
            B, N, S, D = tensor_bnsd.shape
            new_shape = (B, N, S * 32, D)# 进入convert处理会//64，但是这里的shape已经是除过32的对齐长度了
            tensor_processed = torch.from_numpy(convert_v_scale_bnsd_to_layout(tensor_bnsd, actualSeqLengthsKV, "BNSD", new_shape, params))
            # 缓存形状: [BlockNum, N, BlockSize, D, 2]
            # 注意 BlockSize 对应的是压缩后的序列长度 (S/2) 上的块大小
            out_shape = (total_block_num, N, blockSize, D, 2)
        else:
            # BNBD2: 拆分 D 维度
            # 输入: [B, N, S, D] -> 输出: [B, N, S, D//2, 2]
            # 注意：reshape 会将连续的 D 维度拆成 (D//2, 2)，保持 d0,d1 紧挨着
            if D % 2 != 0:
                raise ValueError("Head dimension D must be even to split into D2, 2")
            tensor_processed = tensor_bnsd.reshape(B, N, S, D // 2, 2)
            # 缓存形状: [BlockNum, N, BlockSize, D//2, 2]
            out_shape = (total_block_num, N, blockSize, D // 2, 2)

    # 1. 处理 BlockNum 不匹配问题

    if params['shape_input'][index][0] != total_block_num:
        print(params['shape_input'][index])
        print(params['shape_input'][21])
        print("xxxxxxx")
        out_shape = list(out_shape)
        out_shape[0] = params['shape_input'][index][0]
        out_shape = tuple(out_shape)

    # 初始化输出缓存
    out_cache = torch.zeros(out_shape, dtype=tensor_bnsd.dtype)

    # 3. 数据填充 (PagedAttention Block 填充)
    current_block_idx = 0
    print("Pjjjj, B=", {B})
    for b in range(B):
        block_per_b = block_num[b]
        print("Pjjjj, block_per_b=", block_per_b) #### 2
        # 获取当前 batch 的物理块 ID 列表
        # 注意：block_table 通常是 [Batch, MaxBlocks] 的形状
        bid_table = block_table[b]
        print("Pjjjj, bid_table=", bid_table) #### 1 0
        # 确定源数据在序列维度上的有效长度（针对最后一个块）
        # 这里的 seq_len 是指 tensor_processed 的序列长度
        if is_vscale:
            # 对于 vscale，tensor_processed 的 seq len 是 actual_seq / 64
            seq_len = math.ceil(actualSeqLengthsKV[b] / 64)
        else:
            seq_len = actualSeqLengthsKV[b]

        for blockid_index in range(block_per_b):
            # 获取物理块 ID
            blockid = bid_table[blockid_index]
            # 计算逻辑偏移
            block_offset = blockid_index * blockSize
            print("Pjjjj, block_offset=", block_offset)
            # 拷贝数据
            # 处理最后一个块可能不满的情况
            if block_offset + blockSize > seq_len:
                valid_len = seq_len - block_offset
                # 源数据切片
                src_slice = tensor_processed[b, :, block_offset : block_offset + valid_len]
                # 目标切片
                out_cache[blockid, :, 0:valid_len] = src_slice
            else:
                # 满块拷贝
                print("Pjjjj, blockid=", blockid)
                print("Pjjjj, out_cache=", out_cache.shape)
                src_slice = tensor_processed[b, :, block_offset : block_offset + blockSize]
                print("Pjjjj, src_slice=", src_slice.shape)
                out_cache[blockid, :, :] = src_slice

    # 4. 数据转换、返回结果
    if index == 25 or KV_layout == "BnNBsD":
        return out_cache
    elif KV_layout == "BnBsND":
        return out_cache.transpose(1,2)
    elif KV_layout == "PA_NZ":
        if index == 21 or index == 22:
            Bn, KV_N, Bs, KV_D = out_cache.shape
            return out_cache.reshape(Bn, KV_N, Bs, (KV_D + 31) // 32, 32).transpose(2, 3)
        elif index == 15:
            # Bn, KV_N, Bs, D_scale, two = out_cache.shape
            # outer = (D_scale + 16 - 1) // 16
            # padded_D = outer * 16
            # scale_padded = torch.full((Bn, KV_N, Bs, padded_D, 2), 0.0000000000000000000000000000000000000000001, dtype=out_cache.dtype)
            # scale_padded[:, :, :, :D_scale, :] = out_cache
            # scale_6d = scale_padded.reshape(Bn, KV_N, Bs, outer, 16, 2)
            # return scale_6d.permute(0, 1, 3, 2, 4, 5).contiguous()
            Bn, KV_N, Bs, KV_D, _ = out_cache.shape
            return out_cache.reshape(Bn, KV_N, Bs // 16, 16, KV_D, _).transpose(3, 4)
        elif index == 17:
            Bn, KV_N, Bs, KV_D, _ = out_cache.shape
            return out_cache.reshape(Bn, KV_N, Bs, KV_D // 16, 16, _).transpose(2, 3)

def deepseek_ds_pa_preprocessing(pfa_param, params,torch_tensor_list):
    v_end_index = pfa_param['v_end_index']
    k_cache = torch_tensor_list[v_end_index + 19]
    v_cache = torch_tensor_list[v_end_index + 20]
    k_rope_cache = torch_tensor_list[v_end_index + 23]

    numKeyValueHeads = params['numkeyvalueheads']
    numHeads = params['numheads']
    if numKeyValueHeads == 0:
        numKeyValueHeads = numHeads

    actualSeqLengthsKV = pfa_param['actualSeqLengthsKV']
    kv_inputlayout = pfa_param['kv_inputlayout']
    k_tensor_bnsd_list = []
    k_shape_bnsd_raw_list = []

    v_tensor_bnsd_list = []
    v_shape_bnsd_raw_list = []
    k_shape_num = pfa_param['k_num']
    for i in range(k_shape_num):
        k_bnsd_tensor, k_bnsd_shape = np_bsh_to_bnsd(torch_tensor_list[1], pfa_param['k_shape_list_raw'][0], numKeyValueHeads,
                                                actualSeqLengthsKV,
                                                kv_inputlayout)
        k_tensor_bnsd_list.append(k_bnsd_tensor) #转成bnsd原始
        k_shape_bnsd_raw_list.append(k_bnsd_shape)

    for i in range(k_shape_num):
        v_shape = pfa_param['v_shape_list_raw'][i]
        v_bnsd_tensor, v_bnsd_shape = np_bsh_to_bnsd(torch_tensor_list[2], v_shape, numKeyValueHeads,
                                                    actualSeqLengthsKV,
                                                    kv_inputlayout)
        v_tensor_bnsd_list.append(v_bnsd_tensor)  # 转成bnsd原始
        v_shape_bnsd_raw_list.append(v_bnsd_shape)

    k_rope_tensor = torch_tensor_list[v_end_index + 22]
    k_rope_shape = params['shape_input'][v_end_index + 22]
    k_rope_bnsd_tensor, k_rope_bnsd_shape = np_bsh_to_bnsd(k_rope_tensor, k_rope_shape, numKeyValueHeads,
                                                           actualSeqLengthsKV,
                                                           kv_inputlayout)

    blockNumPerBlock = pfa_param['blockNumPerBlock']
    block_table = pfa_param['block_table']
    blockSize = params['blocksize']

    k_tensor_bsh_raw = k_tensor_bnsd_list[0]
    v_tensor_bsh_raw = v_tensor_bnsd_list[0]
    k_rope_bsh_raw = k_rope_bnsd_tensor

    if k_cache.ndim == 3:
        k_tensor_bsh_raw = trans_bnsd_to_bsh(k_tensor_bnsd_list[0], k_shape_bnsd_raw_list[0])
        v_tensor_bsh_raw = trans_bnsd_to_bsh(v_tensor_bnsd_list[0], v_shape_bnsd_raw_list[0])
        k_rope_bsh_raw = trans_bnsd_to_bsh(k_rope_bnsd_tensor, k_rope_bnsd_shape)
        for b, block_per_b in enumerate(blockNumPerBlock):
            for blockid_index in range(block_per_b):
                blockid = block_table[b][blockid_index]
                block_offset = blockid_index * blockSize
                if blockid_index == block_per_b - 1:
                    blocksize_left = actualSeqLengthsKV[b] - block_offset
                    k_cache[blockid, 0:blocksize_left, :] = k_tensor_bsh_raw[b,
                                                            block_offset:(block_offset + blocksize_left), :]
                    v_cache[blockid, 0:blocksize_left, :] = v_tensor_bsh_raw[b,
                                                            block_offset:(block_offset + blocksize_left), :]
                    k_rope_cache[blockid, 0:blocksize_left, :] = k_rope_bsh_raw[b,
                                                                 block_offset:(block_offset + blocksize_left), :]
                else:
                    k_cache[blockid, 0:blockSize, :] = k_tensor_bsh_raw[b,
                                                       block_offset:(block_offset + blockSize), :]
                    v_cache[blockid, 0:blockSize, :] = v_tensor_bsh_raw[b,
                                                       block_offset:(block_offset + blockSize), :]
                    k_rope_cache[blockid, 0:blockSize, :] = k_rope_bsh_raw[b,
                                                       block_offset:(block_offset + blockSize), :]
    if k_cache.ndim == 4:
        # gen kv cache
        for b, block_per_b in enumerate(blockNumPerBlock):
            for blockid_index in range(block_per_b):
                blockid = block_table[b][blockid_index]
                block_offset = blockid_index * blockSize
                if blockid_index == block_per_b - 1:
                    blocksize_left = actualSeqLengthsKV[b] - block_offset
                    k_cache[blockid, :, 0:blocksize_left, :] = k_tensor_bsh_raw[b, :,
                                                               block_offset:(block_offset + blocksize_left),
                                                               :]
                    v_cache[blockid, :, 0:blocksize_left, :] = v_tensor_bsh_raw[b, :,
                                                               block_offset:(block_offset + blocksize_left),
                                                               :]
                    k_rope_cache[blockid, :, 0:blocksize_left, :] = k_rope_bsh_raw[b, :,
                                                               block_offset:(block_offset + blocksize_left),
                                                               :]
                else:
                    k_cache[blockid, :, 0:blockSize, :] = k_tensor_bsh_raw[b, :,
                                                          block_offset:(block_offset + blockSize), :]
                    v_cache[blockid, :, 0:blockSize, :] = v_tensor_bsh_raw[b, :,
                                                          block_offset:(block_offset + blockSize), :]
                    k_rope_cache[blockid, :, 0:blockSize, :] = k_rope_bsh_raw[b, :,
                                                          block_offset:(block_offset + blockSize), :]

    if k_cache.ndim == 5:
        kd = k_tensor_bsh_raw.shape[3]
        vd = v_tensor_bsh_raw.shape[3]
        numheadkv = k_tensor_bsh_raw.shape[1]
        k_cache_bnsd = np.zeros((k_cache.shape[0], numheadkv, blockSize, kd))
        v_cache_bnsd = np.zeros((k_cache.shape[0], numheadkv, blockSize, vd))
        k_rope_cache_bnsd = np.zeros((k_rope_cache.shape[0], numheadkv, blockSize, 64))
        # gen kv cache
        for b, block_per_b in enumerate(blockNumPerBlock):
            for blockid_index in range(block_per_b):
                blockid = block_table[b][blockid_index]
                block_offset = blockid_index * blockSize
                if blockid_index == block_per_b - 1:
                    blocksize_left = actualSeqLengthsKV[b] - block_offset
                    k_cache_bnsd[blockid, :, 0:blocksize_left, :] = k_tensor_bsh_raw[b, :,
                                                                    block_offset:(
                                                                            block_offset + blocksize_left),
                                                                    :]
                    v_cache_bnsd[blockid, :, 0:blocksize_left, :] = v_tensor_bsh_raw[b, :,
                                                                    block_offset:(
                                                                            block_offset + blocksize_left),
                                                                    :]
                    k_rope_cache_bnsd[blockid, :, 0:blocksize_left, :] = k_rope_bsh_raw[b, :,
                                                                    block_offset:(
                                                                            block_offset + blocksize_left),
                                                                    :]
                else:
                    k_cache_bnsd[blockid, :, 0:blockSize, :] = k_tensor_bsh_raw[b, :,
                                                               block_offset:(block_offset + blockSize), :]
                    v_cache_bnsd[blockid, :, 0:blockSize, :] = v_tensor_bsh_raw[b, :,
                                                               block_offset:(block_offset + blockSize), :]
                    k_rope_cache_bnsd[blockid, :, 0:blockSize, :] = k_rope_bsh_raw[b, :,
                                                               block_offset:(block_offset + blockSize), :]

            base = 16
            k_cache = k_cache_bnsd.reshape(k_cache_bnsd.shape[0],
                                           k_cache_bnsd.shape[1],
                                           k_cache_bnsd.shape[2],
                                           k_cache_bnsd.shape[3] // base,
                                           base).transpose(0, 1, 3, 2, 4)
            v_cache = v_cache_bnsd.reshape(v_cache_bnsd.shape[0],
                                           v_cache_bnsd.shape[1],
                                           v_cache_bnsd.shape[2],
                                           v_cache_bnsd.shape[3] // base,
                                           base).transpose(0, 1, 3, 2, 4)
            k_rope_cache = k_rope_cache_bnsd.reshape(k_rope_cache_bnsd.shape[0],
                                           k_rope_cache_bnsd.shape[1],
                                           k_rope_cache_bnsd.shape[2],
                                           k_rope_cache_bnsd.shape[3] // base,
                                           base).transpose(0, 1, 3, 2, 4)

    kv_dtype = tools.get_np_dtype(params['dtype_input'][1])

    if str(kv_dtype) == "<class 'bfloat16'>":
        k_cache = torch.from_numpy(k_cache).to(torch.bfloat16)
        v_cache = torch.from_numpy(v_cache).to(torch.bfloat16)
        k_rope_cache = torch.from_numpy(k_rope_cache).to(torch.bfloat16)
    elif str(kv_dtype) != "<class 'float8_e5m2'>" and params['dtype_input'][1] != "hifloat8" and \
            params['dtype_input'][1] != "float8_e4m3fn":
        k_cache = torch.from_numpy(k_cache.astype(kv_dtype))
        v_cache = torch.from_numpy(v_cache.astype(kv_dtype))
        k_rope_cache = torch.from_numpy(k_rope_cache.astype(kv_dtype))

    tools.modify_alcnn_input_file(ids=21, origin_index=[21], type='tensor_list',
                                  mode='rewrite',
                                  tensors=[k_cache],
                                  params=params,
                                  data_dtype=params['dtype_input'][1])
    tools.modify_alcnn_input_file(ids=22, origin_index=[22], type='tensor_list',
                                  mode='rewrite',
                                  tensors=[v_cache],
                                  params=params,
                                  data_dtype=params['dtype_input'][2])
    tools.modify_alcnn_input_file(ids=25, origin_index=[25], type='tensor',
                                  mode='rewrite',
                                  tensors=k_rope_cache,
                                  params=params,
                                  data_dtype=params['dtype_input'][1])

def deepseek_preprocessing(params, pfa_param, torch_tensor_list, numHeads,numKeyValueHeads, tnd_flag, k_shape_num):
    v_end_index = pfa_param['v_end_index']
    # >> q rope info
    q_rope_tensor_shape = params['shape_input'][v_end_index + 21]
    q_rope_dtype = params['dtype_input'][v_end_index + 21]
    q_rope_tensor = torch_tensor_list[v_end_index + 21]

    # >> k rope info
    k_rope_shape = params['shape_input'][v_end_index + 22]
    k_rope_dtype = params['dtype_input'][v_end_index + 22]
    k_rope_tensor = torch_tensor_list[v_end_index + 22]

    # 非全量化场景1.将Q与QROPE拼接2.将K与KROPE拼接3.伪量化场景，k_antiscale与k_rope_antiscale拼接
    # if not ifa_param['in_quant_flag']:
    q_new_tensor, q_new_shape = concat_tensor(torch_tensor_list[0], params['shape_input'][0],
                                              q_rope_tensor,
                                              q_rope_tensor_shape, numHeads, tnd_flag)
    k_new_tensor, k_new_shape = concat_tensor(torch_tensor_list[1], params['shape_input'][1],
                                              k_rope_tensor,
                                              k_rope_shape, numKeyValueHeads, tnd_flag)

    pfa_param['q_tensor'] = q_new_tensor
    pfa_param['q_shape'] = q_new_shape
    pfa_param['k_tensor_list'][0] = k_new_tensor
    pfa_param['k_shape_list'][0] = k_new_shape

    return pfa_param


def get_param_fus(torch_tensor_list, params, qdim):
    pfa_param = {}
    # ===参数获取===
    flag_list = params['flaglist']
    inputLayout = params['inputlayout']
    actualSeqLengths = params['actualseqlengths']
    q_shape = params['shape_input'][0]

    tnd_flag = False
    if inputLayout in ["TND", "TND_NTD", "NTD_TND", "NTD"]:
        tnd_flag = True
        batch = len(actualSeqLengths)
    else:
        batch = q_shape[0]
    pfa_param['batch'] = batch

    pfa_param['q_tensor'] = torch_tensor_list[0]
    pfa_param['q_shape'] = params['shape_input'][0]

    # >> kv info
    # k和v的位置通过b计算
    k1_shape = params['shape_input'][1]
    kb1 = k1_shape[0]
    # 如果第一个K的B=1,则默认kv列表长度为b;如果第一个K的B!=1,则默认kv列表长度为1
    if kb1 != 1 or inputLayout in ["TND", "NTD_TND", "NTD"]:
        k_shape_num = v_shape_num = 1
    else:
        k_shape_num = v_shape_num = batch
    pfa_param['v_end_index'] = v_end_index = int(k_shape_num + v_shape_num)
    pfa_param['tnd_flag'] = tnd_flag
    pfa_param['k_num'] = k_shape_num
    pfa_param['k_start_index'] = k_start_index = 1
    pfa_param['k_end_index'] = k_end_index = int(k_shape_num)
    pfa_param['v_start_index'] = v_start_index = int(k_shape_num) + 1
    print(
        f"k_start_index:{k_start_index} k_end_index:{k_end_index} v_start_index:{v_start_index} v_end_index:{v_end_index}")

    pfa_param['k_shape_list'] = pfa_param['k_shape_list_raw'] = params['shape_input'][k_start_index:k_end_index + 1]
    pfa_param['k_tensor_list'] = torch_tensor_list[k_start_index:k_end_index + 1]

    if len(flag_list) >= 27 and flag_list[26] and qdim == 512:
        pfa_param['v_shape_list'] = params['shape_input'][k_start_index:k_end_index + 1]
        pfa_param['v_tensor_list'] = torch_tensor_list[k_start_index:k_end_index + 1]
    else:
        pfa_param['v_shape_list'] = params['shape_input'][v_start_index:v_end_index + 1]
        pfa_param['v_tensor_list'] = torch_tensor_list[v_start_index:v_end_index + 1]

    pfa_param['k_shape_list_raw'] = params['shape_input'][k_start_index:k_end_index + 1]
    pfa_param['k_tensor_list_raw']=  torch_tensor_list[k_start_index:k_end_index + 1]
    if len(flag_list) >= 27 and flag_list[26] and qdim == 512:
        pfa_param['v_shape_list_raw'] = params['shape_input'][k_start_index:k_end_index + 1]
        pfa_param['v_tensor_list_raw'] = torch_tensor_list[k_start_index:k_end_index + 1]
    else:
        pfa_param['v_shape_list_raw'] =  params['shape_input'][v_start_index:v_end_index + 1]
        pfa_param['v_tensor_list_raw'] = torch_tensor_list[v_start_index:v_end_index + 1]
    return pfa_param
def qtensor_dim(q_shape, inputLayout, numhead):
    if inputLayout == "SH":  # SH格式
        return q_shape[0]
    elif inputLayout == "BSH" or inputLayout == "BSH_NBSD":
        return int(q_shape[2] / numhead)
    else:
        return q_shape[-1]


def generate_alibi_slope(n):
    # 计算步长
    step = 8 / n
    # 生成从 8/n 开始，共有 n 个数，步长为 8/n 的数组
    arr = np.arange(step, step * n + step, step, dtype=np.float32)
    # 以2为底进行指数运算
    result = 2 ** arr
    result = 1 / result
    return result


def aclnnPromptFlashAttention_unification(torch_tensor_list, params):   # main
    # import traceback
    # traceback.print_stack()
    # torch_tensor_list ---tensor  params--attr参数
    pfa_param = {}

    action_type = params["action_type"]
    gold = ["bm_output", "bm_output_gold", "bm_gold"]

    flagList = params['flaglist']
    actualSeqLengths = params['actualseqlengths']
    actualSeqLengthsKV = params['actualseqlengthskv']
    numHeads = int(params['numheads'])

    if flagList[25] == 0:
        inputLayout = 'BSH'
        # 所有的参数的默认值都给写在这里
        numKeyValueHeads = 0
        scaleValue = 1.0
        preTokens = 214748647
        nextTokens = 214748647
        sp_mode = 0
        block_size = 0
    else:
        scaleValue = params['scalevalue'] if 'scalevalue' in params else 1
        preTokens = params['pretokens'] if 'pretokens' in params else 214748647
        nextTokens = params['nexttokens'] if 'nexttokens' in params else 214748647
        inputLayout = params['inputlayout'] if 'inputlayout' in params else "BSH"
        numKeyValueHeads = params['numkeyvalueheads'] if 'numkeyvalueheads' in params else 0
        sp_mode = params['sparsemode'] if 'sparsemode' in params else 0

    if numKeyValueHeads == 0:
        numKeyValueHeads = numHeads
    q_shape = params['shape_input'][0]
    qdim = qtensor_dim(q_shape, inputLayout, numHeads)
    pfa_param = get_param_fus(torch_tensor_list, params, qdim)
    q_shape = pfa_param['q_shape']
    q_dtype = tools.get_np_dtype(params['dtype_input'][0])  # numpy类型
    pfa_param['q_dtype'] = q_dtype
    batch = pfa_param["batch"]
    tnd_flag = pfa_param["tnd_flag"]
    # 设置参数默认值
    psetype = int(params.get('psetype', params.get('p_dtype', 0)))
    params['psetype'] = psetype
    qstartidx = params.get('qstartidx', [0])
    kvstartidx = params.get('kvstartidx', [0])

    # alibi pse slopes
    slopes = None
    if psetype == 2 or psetype == 3:
        slopes = generate_alibi_slope(numHeads)
        # print("LLLLLLLLLLLLLLLLLLLLLLLLLL", numheads)
        # print("ppppppppppppppppppppppp:slopes:", slopes)
        input_file = os.path.join(
            params['case_path'], params['case_name'] + '_input_' + str(pfa_param['v_end_index'] + 1) + '.bin')
        tools.dump_file(slopes, input_file, params['dtype_input'][pfa_param['v_end_index'] + 1], params)

    kv_inputlayout = inputLayout
    if (inputLayout == "TND" or inputLayout == "NTD_TND" or inputLayout == "NTD") and flagList[14]:
        kv_shape = pfa_param['k_shape_list'][0] if pfa_param.get('k_shape_list') else params['shape_input'][1]
        if len(kv_shape) == 4 and kv_shape[1] == numKeyValueHeads:
            kv_inputlayout = "BNSD"
        elif len(kv_shape) == 4 and kv_shape[2] == numKeyValueHeads:
            kv_inputlayout = "BSND"
        else:
            kv_inputlayout = "BNSD"

    if inputLayout == "SH":
        actualSeqLengthsKV = actualSeqLengths
    else:
        actualSeqLengthsKV = params['actualseqlengthskv']
    if inputLayout == "TND" or inputLayout == "NTD_TND" or inputLayout == "NTD":
        # 将tnd格式下的act seq转成普通的act seq
        actualSeqLengths = trans_tnd_actseq(actualSeqLengths)
    if (inputLayout == "TND" or inputLayout == "NTD_TND" or inputLayout == "NTD") and flagList[14] == 0:  # TND非PA场景需要转成普通actseq
        actualSeqLengthsKV = trans_tnd_actseq(actualSeqLengthsKV)
    # >> actualSeqLengths预处理：actualSeqLengths为单值场景，如果长度为1且b不为1，则将actualSeqLengths扩展为b个单值的列表
    if inputLayout != "TND" and inputLayout != "NTD_TND" and inputLayout != "NTD":
        if actualSeqLengths != None:
            if len(actualSeqLengths) == 1 and len(actualSeqLengths) != q_shape[0]:
                actualSeqLengths_item = actualSeqLengths[0]
                for b_count in range(batch - 1):
                    actualSeqLengths.append(actualSeqLengths_item)
            # >> actualSeqLengths预处理：actualSeqLengths长度超过
            if len(actualSeqLengths) > batch:
                actualSeqLengths = actualSeqLengths[:batch]
        if actualSeqLengthsKV != None:
            if len(actualSeqLengthsKV) == 1 and len(actualSeqLengthsKV) != q_shape[0]:
                actualSeqLengthsKV_item = actualSeqLengthsKV[0]
                for b_count in range(batch - 1):
                    actualSeqLengthsKV.append(actualSeqLengthsKV_item)
            # >> actualSeqLengths预处理：actualSeqLengths长度超过
            if len(actualSeqLengthsKV) > batch:
                actualSeqLengthsKV = actualSeqLengthsKV[:batch]
    pfa_param['actualSeqLengthsKV'] = actualSeqLengthsKV
    k_shape_num = pfa_param['k_num']
    v_shape = params['shape_input'][1 + k_shape_num]
    out_dtype = tools.get_np_dtype(params['dtype_output'][0])  # numpy类型
    out_shape = gen_outshape(inputLayout, q_shape, v_shape, numKeyValueHeads, numHeads)
    print(f"[INFO] out_shape is {out_shape}")

    if 0 in q_shape or len(q_shape) == 0:
        if params["softmax_lse_flag"]:
            return torch.from_numpy(torch_tensor_list[0]), torch.from_numpy(np.array([]))
        return torch.from_numpy(torch_tensor_list[0])
    k_shape = pfa_param['k_shape_list'][0]
    if 0 in k_shape or len(k_shape) == 0:
        if params["softmax_lse_flag"]:
            return torch.zeros(out_shape), torch.zeros(pfa_param['lseshape'])
        return torch.zeros(out_shape)
    pfa_param['q_tensor'] = torch_tensor_list[0]
    if len(flagList) >= 27 and flagList[26]: #### rope
        if params['query_quant_mode'] == 6 and params['k_antiquant_mode'] == 6 and params['v_antiquant_mode'] == 8:
            pass
        else:
            pfa_param = deepseek_preprocessing(params, pfa_param, torch_tensor_list, numHeads, numKeyValueHeads, tnd_flag, k_shape_num)
    q_tensor, q_bnsd_shape = np_bsh_to_bnsd(pfa_param['q_tensor'], pfa_param['q_shape'], numHeads, actualSeqLengths, inputLayout)

    print(f"[pjjjj] q_tensor shape is {q_tensor.shape, q_bnsd_shape}")  ### (1, 4, 128) -> (1, 4, 1, 128)

    if inputLayout == "TND" or inputLayout == "NTD_TND" or inputLayout == "NTD":
        pfa_param['lseshape'] = [q_shape[0], q_shape[1], 1]
    else:
        pfa_param['lseshape'] = [q_bnsd_shape[0], q_bnsd_shape[1], q_bnsd_shape[2], 1]

    # >> kv info
    k_tensor_list = []
    v_tensor_list = []
    k_shape_bnsd_list = []
    k_shape_bnsd_origin_list = []
    v_shape_bnsd_list = []
    v_shape_bnsd_origin_list = []
    k_tensor_bnsd_list = []
    v_tensor_bnsd_list = []
    k_shape_bnsd_raw_list = []
    v_shape_bnsd_raw_list = []
    k_start_index = pfa_param['k_start_index']
    k_end_index = pfa_param['k_end_index']
    v_start_index = pfa_param['v_start_index']
    v_end_index = pfa_param['v_end_index']
    kvs_max = 0
    kvs_list = []
    k_dtype = params['dtype_input'][1]
    print(f"[INFO] k_start_index:{k_start_index}, k_end_index:{k_end_index}")

    for i in range(k_shape_num):
        k_shape = pfa_param['k_shape_list'][i]
        if 0 in k_shape or len(k_shape) == 0:
            if params["softmax_lse_flag"]:
                return torch.zeros(out_shape), torch.zeros(pfa_param['lseshape'])
            return torch.zeros(out_shape)
        k_tensor, k_bnsd_shape = np_bsh_to_bnsd(pfa_param['k_tensor_list'][i], k_shape, numKeyValueHeads, actualSeqLengthsKV,
                                                kv_inputlayout)
        # 扩展前原始的BNSDshape
        k_shape_bnsd_origin_list.append(k_bnsd_shape)

        k_tensor_bnsd_list.append(k_tensor)
        k_shape_bnsd_raw_list.append(k_bnsd_shape)
        kvs_list.append(k_bnsd_shape[2])
        kvs_max = max(kvs_max, k_bnsd_shape[2])
        if numKeyValueHeads != numHeads:
            k_tensor, k_bnsd_shape = _np_broadcastKV_sigle(numHeads, numKeyValueHeads, k_tensor, k_tensor.dtype)
            print(f"[INFO] broadcast k to {k_tensor.shape}")
        k_tensor_list.append(k_tensor)
        k_shape_bnsd_list.append(k_bnsd_shape)
    for i in range(k_shape_num):
        v_shape = pfa_param['v_shape_list'][i]
        v_tensor, v_bnsd_shape = np_bsh_to_bnsd(pfa_param['v_tensor_list'][i], v_shape, numKeyValueHeads, actualSeqLengthsKV,
                                                kv_inputlayout)
        # 扩展前原始的BNSDshape
        v_shape_bnsd_origin_list.append(v_bnsd_shape)

        v_tensor_bnsd_list.append(v_tensor)
        v_shape_bnsd_raw_list.append(v_bnsd_shape)
        if numKeyValueHeads != numHeads:
            v_tensor, v_bnsd_shape = _np_broadcastKV_sigle(numHeads, numKeyValueHeads, v_tensor, v_tensor.dtype)   # v_tensor是numpy
            print(f"[INFO] broadcast v to {v_tensor.shape}")
        v_tensor_list.append(v_tensor)
        v_shape_bnsd_list.append(v_bnsd_shape)
    kvs = kvs_max
    actualprefixKV = 0
    prefix_kvs = 0
    if flagList[21]:
        prefix_k_shape = params['shape_input'][v_end_index + 17]
        prefix_v_shape = params['shape_input'][v_end_index + 18]
        if len(params['prefix_act_lens']) != 0:
            actualprefixKV = params['prefix_act_lens'][0]

        prefix_k_tensor,prefix_k_bnsd_shape = np_bsh_to_bnsd(torch_tensor_list[v_end_index+17], prefix_k_shape, numKeyValueHeads, actualprefixKV,
                                                inputLayout)

        prefix_v_tensor,prefix_v_bnsd_shape = np_bsh_to_bnsd(torch_tensor_list[v_end_index + 18], prefix_v_shape, numKeyValueHeads,
                                         actualprefixKV,
                                         inputLayout)
        if numKeyValueHeads != numHeads:
            prefix_k_tensor, prefix_k_bnsd_shape = _np_broadcastKV_sigle(numHeads, numKeyValueHeads, prefix_k_tensor, prefix_k_tensor.dtype)
            prefix_v_tensor, prefix_v_bnsd_shape = _np_broadcastKV_sigle(numHeads, numKeyValueHeads, prefix_v_tensor, prefix_v_tensor.dtype)
        pfa_param ["actualprefixKV"] = actualprefixKV
        pfa_param ["shared_prefix_k"] = prefix_k_tensor
        pfa_param ["shared_prefix_v"] = prefix_v_tensor
        prefix_kvs = prefix_k_bnsd_shape[2]
        if actualprefixKV == 0:
            actualprefixKV = prefix_k_bnsd_shape[2]
        kvs += prefix_kvs

    pse_bnsd_tensor = None
    pse_shift_shape = params['shape_input'][v_end_index+1]  ##1nss or bnss
    qs = q_bnsd_shape[2]

    if psetype == 0:
        if flagList[3] == 0 or 0 in pse_shift_shape:
            pse_bnsd_tensor = None
        else:
            pse_dtype_torch = tools.get_pt_dtype(params['dtype_input'][v_end_index + 1])
            pse_shift = None
            pse_shift_random_flag = True
            if psetype == 0:
                if action_type in gold:
                    pse_shift = torch_tensor_list[v_end_index + 1]
                else:
                    if 'prandom' in params:
                        if params['prandom'] != 0:
                            pse_shift = torch_tensor_list[v_end_index + 1]
                        else:
                            pse_shift_random_flag = False
                    else:
                        pse_shift_random_flag = False
                    if not pse_shift_random_flag:
                        print("pse")
                        pse_shift = get_all_alibi(numHeads, pse_shift_shape)
                        pse_shift[:, :, qs:, :] = 1
                        pse_shift[:, :, :, kvs:] = 1
                        npu_pse_shift = torch.from_numpy(pse_shift).to(pse_dtype_torch)
                        pse_shift = npu_pse_shift.to(torch.float32).numpy().astype(np.float32)
                        tools.modify_alcnn_input_file(ids=3, origin_index=[3], type='tensor', mode='rewrite',
                                                    tensors=npu_pse_shift,
                                                    params=params)
                cpu_pse_shift = pse_shift[:, :, :qs, :kvs]

                pse_bnsd_tensor = _np_broadcast_pseShift_n(cpu_pse_shift, pse_shift_shape, q_bnsd_shape[0])  # to bnsd

    m_bnsd_tensor = None
    npu_m_shape = params['shape_input'][v_end_index + 2]
    m_dtype = tools.get_np_dtype(params['dtype_input'][v_end_index + 2])

    randoms = 0
    mrandom_type = "NORMAL"
    if 'mrandomtype' in params:
        mrandom_type = params['mrandomtype']
        if mrandom_type == 'ones':
            randoms = int(params['mrandom'])

    if flagList[4] == 0 or 0 in npu_m_shape:
        preTokens = 214748647
        nextTokens = 214748647
    else:
        batch = q_bnsd_shape[0]
        numheads = q_bnsd_shape[1]
        npu_m_shape_s = npu_m_shape
        if sp_mode == 0 or sp_mode == 1:
            batch, numheads, ns1, ns2 = get_attention_mask_batch_num(npu_m_shape,q_bnsd_shape)  # 获取输入attentionmask的batch 和numhead
            npu_m_shape_s = [ns1, ns2]
        cpu_m_shape = [qs, kvs]  # cpu

        cpu_m_tensor, npu_m_tensor, preTokens, nextTokens = _create_random_mask_by_spars(cpu_m_shape, npu_m_shape_s,
                                                                                         m_dtype, preTokens, nextTokens,
                                                                                         actualSeqLengths,
                                                                                         actualSeqLengthsKV, actualprefixKV,prefix_kvs,kvs_list, batch,
                                                                                         numheads, sp_mode,
                                                                                         random_ones=randoms)
        if mrandom_type == 'invalid' or mrandom_type == 'invaild':
            randoms = int(params['mrandom'])
            cpu_m_tensor[..., :randoms] = 1
            npu_m_tensor[..., :randoms] = 1

        npu_m_tensor = torch.from_numpy(npu_m_tensor)
        tools.modify_alcnn_input_file(ids=4, origin_index=[4], type='tensor', mode='rewrite', tensors=npu_m_tensor,
                                      params=params)

        if sp_mode == 0 or sp_mode == 1:
            m_bnsd_tensor = _np_broadcast_mask_n(cpu_m_tensor, npu_m_shape, cpu_m_shape, numHeads, q_bnsd_shape[0])
        else:
            m_bnsd_tensor = cpu_m_tensor

    dequant_scale1, dequant_scale2, quant_scale1, quant_scale2, quant_offset2 = None, None, None, None, None

    if flagList[7]:
        dequant_scale1 = torch_tensor_list[v_end_index + 3]
        dequant_scale1 = np.frombuffer(dequant_scale1, dtype=np.float32)
        dequant_scale1 = dequant_scale1[: 1]

    if flagList[8]:
        quant_scale1 = torch_tensor_list[v_end_index + 4]
        print(f"[INFO] quant_scale1 is {quant_scale1}")
    if flagList[9]:
        dequant_scale2 = torch_tensor_list[v_end_index + 5]
        dequant_scale2 = np.frombuffer(dequant_scale2, dtype=np.float32)
        dequant_scale2 = dequant_scale2[: 1]
    if flagList[10]:
        quant_scale2 = torch_tensor_list[v_end_index + 6]
        quant_scale2_shape = params['shape_input'][v_end_index + 6]
        per_channel = True
        if len(quant_scale2_shape) == 1:
            if quant_scale2_shape[0] == 1:
                per_channel = False
        if per_channel:
            quant_scale2 = _trans_1h_to_1n1d(quant_scale2_shape, quant_scale2,
                                             numHeads, inputLayout)
    if flagList[11]:
        quant_offset2 = torch_tensor_list[v_end_index + 7]
        quant_offset2_shape = params['shape_input'][v_end_index + 7]
        per_channel = True
        if len(quant_offset2_shape) == 1:
            if quant_offset2_shape[0] == 1:
                per_channel = False
        if per_channel:
            quant_offset2 = _trans_1h_to_1n1d(quant_offset2_shape, quant_offset2,
                                              numHeads, inputLayout)

    antiquant_scale = None
    antiquant_offset = None

    k_antiquant_scale = None
    v_antiquant_scale = None
    dequantScale_query = None
    p_qscale = torch.tensor((1.0)).to(torch.float32)

    k_antiquant_offset = None
    v_antiquant_offset = None


    anti_or_kvanti = 0
    if len(flagList) > 20:
        # 获取kv量化参数
        if flagList[12]:
            anti_or_kvanti = 1
            antiquantscale_shape = params['shape_input'][v_end_index + 8]
            antiquantscale_tensor = torch_tensor_list[v_end_index + 8]
            if len(antiquantscale_shape) == 1:  # pertensor
                print("antiquantscale_pertensor")
                antiquant_scale = antiquantscale_tensor
            else:  # perchanel
                # 将kv量化参数转换为2n1d(匹配bnsd)
                antiquantscale_2n1d_tensor = _trans_2h_to_2n1d(antiquantscale_shape, antiquantscale_tensor, numKeyValueHeads)

                # GQA场景，扩展kv反量化参数
                if numKeyValueHeads != numHeads:
                    print("broadcast kvanti")
                    antiquant_scale = broadcast_kv_dequant_tensor(antiquantscale_2n1d_tensor, numKeyValueHeads, numHeads)
                else:
                    antiquant_scale = antiquantscale_2n1d_tensor
        if flagList[13]:
            antiquantoffset_shape = params['shape_input'][v_end_index + 9]
            antiquantoffset_tensor = torch_tensor_list[v_end_index + 9]
            if len(antiquantoffset_shape) == 1:
                print("antiquantoffset_pertensor")
                antiquant_offset = antiquantoffset_tensor
            else:
                antiquantoffset_2n1d_tensor = _trans_2h_to_2n1d(antiquantoffset_shape, antiquantoffset_tensor,numKeyValueHeads)

                # GQA场景，扩展kv反量化参数
                if numKeyValueHeads != numHeads:
                    print("broadcast kvanti")
                    antiquant_offset = broadcast_kv_dequant_tensor(antiquantoffset_2n1d_tensor, numKeyValueHeads, numHeads)
                else:
                    antiquant_offset = antiquantoffset_2n1d_tensor
        # 判断kv分离的伪量化

        if flagList[17] or flagList[19]:

            print("pjjjjjjj 17-19")

            anti_or_kvanti = 2
            q_dtype_np = tools.get_np_dtype(params['dtype_input'][0])
            print(f"[INFO] q_dtype_np is {q_dtype_np}")
            k_antiquantscale_shape = params['shape_input'][v_end_index + 13]
            k_antiquantscale_tensor = torch_tensor_list[v_end_index + 13]
            v_antiquantscale_shape = params['shape_input'][v_end_index + 15]
            v_antiquantscale_tensor = torch_tensor_list[v_end_index + 15]
            dequantScale_query_shape = params['shape_input'][v_end_index +25]
            dequantScale_query_tensor = torch_tensor_list[v_end_index + 25]


            if params['k_antiquant_mode'] == 0:
                if len(k_antiquantscale_shape) == 1 and k_antiquantscale_shape[0] == 1:  # pertensor
                    print("antiquantscale_pertensor")
                    k_antiquant_scale = k_antiquantscale_tensor
                    v_antiquant_scale = v_antiquantscale_tensor
                else:  # perchanel
                    # 将kv量化参数转换为n1d(匹配bnsd)
                    k_antiquantscale_n1d_tensor = _trans_h_to_n1d(k_antiquantscale_shape, k_antiquantscale_tensor,numKeyValueHeads)
                    v_antiquantscale_n1d_tensor = _trans_h_to_n1d(v_antiquantscale_shape, v_antiquantscale_tensor,numKeyValueHeads)
                    # GQA场景，扩展kv反量化参数DoOpTiling()IterateBmm1MxFp8
                    if numKeyValueHeads != numHeads:
                        print("broadcast kvanti")
                        k_antiquant_scale = broadcast_kv_split_dequant_tensor(k_antiquantscale_n1d_tensor, numKeyValueHeads,numHeads,q_dtype_np)
                        v_antiquant_scale = broadcast_kv_split_dequant_tensor(v_antiquantscale_n1d_tensor,numKeyValueHeads,numHeads,q_dtype_np)
                        print(v_antiquant_scale.shape)
                    else:
                        k_antiquant_scale = k_antiquantscale_n1d_tensor
                        v_antiquant_scale = v_antiquantscale_n1d_tensor
            if params['k_antiquant_mode'] == 1:  # pertoken
                # pertoken -> BS
                k_antiquant_scale = k_antiquantscale_tensor
                v_antiquant_scale = v_antiquantscale_tensor
            if params['query_quant_mode'] == 6 and params['k_antiquant_mode'] == 6 and params['v_antiquant_mode'] == 8:  # mxfp8
                print("====================生成MXFP8 QKV及量化参数====================")
                print(f"------Q/K:per-token-group量化, V:per-channel-group量化------")
                # 非对齐处理
                v_shape_fp16 = v_shape_bnsd_origin_list[0]
                k_shape_fp16 = k_shape_bnsd_origin_list[0]

                q_bnsd_shape_org = q_bnsd_shape.copy()
                k_bnsd_shape_org = k_shape_fp16.copy()
                v_bnsd_shape_org = v_shape_fp16.copy()
                print(f"+++++Sq trans from {q_bnsd_shape_org[2]} to {q_bnsd_shape[2]}")
                if flagList[14]:    # PA场景对齐 512
                    blockSize = params['blocksize']
                    q_bnsd_shape[2] = (q_bnsd_shape[2] + 64 - 1) // 64 * 64
                    k_shape_fp16[2] = (k_shape_fp16[2] + blockSize - 1) // blockSize * blockSize
                    v_shape_fp16[2] = (v_shape_fp16[2] + blockSize - 1) // blockSize * blockSize
                    print(f"------PA场景下对齐blocksize------")
                else:  # 非对齐场景对齐64 = 32 *2
                    q_bnsd_shape[2] = (q_bnsd_shape[2] + 64 - 1) // 64 * 64
                    k_shape_fp16[2] = (k_shape_fp16[2] + 64 - 1) // 64 * 64
                    v_shape_fp16[2] = (v_shape_fp16[2] + 64 - 1) // 64 * 64
                    print(f"------非PA场景下对齐量化group-----")
                print(f"Sq trans from {q_bnsd_shape_org[2]} to {q_bnsd_shape[2]}")
                print(f"Skv trans from {k_bnsd_shape_org[2]} to {k_shape_fp16[2]}")
                # 生成原始FP16的QKV  bnsd  和表格一致的shape
                np.random.seed(54)
                q_datarange = params['range_input'][0]
                q_bnsd_fp16 = torch.from_numpy(np.zeros( (q_bnsd_shape[0], q_bnsd_shape[1], q_bnsd_shape[2], q_bnsd_shape[3]) )).to(torch.float16)
                # q_bnsd_shape_org = torch.from_numpy(np.random.uniform(low=q_datarange[0], high=q_datarange[1], size=[q_bnsd_shape_org[0], q_bnsd_shape_org[1], q_bnsd_shape_org[2], q_bnsd_shape_org[3]])).to(torch.float16)
                q_bnsd_fp16_org = torch.from_numpy(np.random.randn(q_bnsd_shape_org[0], q_bnsd_shape_org[1], q_bnsd_shape_org[2], q_bnsd_shape_org[3])).to(torch.float16)
                q_bnsd_fp16[:q_bnsd_shape_org[0],:q_bnsd_shape_org[1],:q_bnsd_shape_org[2],:q_bnsd_shape_org[3]] = q_bnsd_fp16_org
                # print(f"qqqqqqqqqqqqqqqqqqqqqqq{q_bnsd_fp16[:]}")
                # q_bnsd_fp16 = torch.from_numpy(np.ones(shape=[q_bnsd_shape[0], q_bnsd_shape[1], q_bnsd_shape[2], q_bnsd_shape[3]],dtype=np.float16)).to(torch.float16)
                print(f"[INFO] q_bnsd_fp16.shape is {q_bnsd_fp16.shape}")
                np.random.seed(3)
                k_datarange = params['range_input'][1]
                k_bnsd_fp16 = torch.from_numpy(np.zeros( (k_shape_fp16[0], k_shape_fp16[1], k_shape_fp16[2], k_shape_fp16[3]) )).to(torch.float16)
                k_bnsd_fp16_org = torch.from_numpy(np.random.uniform(low=k_datarange[0], high=k_datarange[1], size=[k_bnsd_shape_org[0], k_bnsd_shape_org[1], k_bnsd_shape_org[2], k_bnsd_shape_org[3]])).to(torch.float16)
                # k_bnsd_fp16_org = torch.from_numpy(np.ones(shape=[k_bnsd_shape_org[0], k_bnsd_shape_org[1], k_bnsd_shape_org[2], k_bnsd_shape_org[3]],dtype=np.float16)).to(torch.float16)
                k_bnsd_fp16[:k_bnsd_shape_org[0],:k_bnsd_shape_org[1],:k_bnsd_shape_org[2],:k_bnsd_shape_org[3]] = k_bnsd_fp16_org

                print(f"[INFO] k_bnsd_fp16.shape is {k_bnsd_fp16.shape}")
                np.random.seed(4)
                v_datarange = params['range_input'][2]
                v_bnsd_fp16 = torch.from_numpy(np.zeros((v_shape_fp16[0], v_shape_fp16[1], v_shape_fp16[2], v_shape_fp16[3]))).to(torch.float16)
                v_bnsd_fp16_org = torch.from_numpy(np.random.uniform(low=v_datarange[0], high=v_datarange[1], size=[v_bnsd_shape_org[0], v_bnsd_shape_org[1], v_bnsd_shape_org[2], v_bnsd_shape_org[3]])).to(torch.float16)
                # v_bnsd_fp16_org = torch.from_numpy(np.ones(shape=[v_bnsd_shape_org[0], v_bnsd_shape_org[1], v_bnsd_shape_org[2], v_bnsd_shape_org[3]],dtype=np.float16)).to(torch.float16)
                v_bnsd_fp16[:v_bnsd_shape_org[0],:v_bnsd_shape_org[1],:v_bnsd_shape_org[2],:v_bnsd_shape_org[3]] = v_bnsd_fp16_org
                print(f"[INFO] v_bnsd_fp16.shape is {v_bnsd_fp16.shape}")

                if len(flagList) >= 27 and flagList[26]:
                    print("----------分离传入场景 额外生成MXFP8QK rope参数----------")
                    # 生成原始FP16的QKrope  bnsd  和表格一致的shape
                    v_end_index = pfa_param['v_end_index']
                    # >> q rope info
                    q_rope_shape = params['shape_input'][v_end_index + 21]
                    q_rope_dtype = params['dtype_input'][v_end_index + 21]
                    q_rope_datarange = params['range_input'][v_end_index + 21]
                    # >> k rope info
                    k_rope_shape = params['shape_input'][v_end_index + 22]
                    k_rope_dtype = params['dtype_input'][v_end_index + 22]
                    k_rope_datarange = params['range_input'][v_end_index + 22]
                    print("Pjjjjj, v_shape", params['shape_input'][v_end_index])
                    print("Pjjjjj, q_rope_shape", q_rope_shape)


                    np.random.seed(8)
                    qr_bnsd_fp16 = torch.from_numpy(np.zeros( (q_bnsd_shape[0], q_bnsd_shape[1], q_bnsd_shape[2], q_rope_shape[-1]) )).to(torch.float16)
                    # qr_bnsd_shape_org = torch.from_numpy(np.random.uniform(low=q_rope_datarange[0], high=q_rope_datarange[1], size=[q_bnsd_shape_org[0], q_bnsd_shape_org[1], q_bnsd_shape_org[2], q_rope_shape[-1]])).to(torch.float16)
                    qr_bnsd_fp16_org = torch.from_numpy(np.random.randn(q_bnsd_shape_org[0], q_bnsd_shape_org[1], q_bnsd_shape_org[2], q_rope_shape[-1])).to(torch.float16)
                    qr_bnsd_fp16[:q_bnsd_shape_org[0],:q_bnsd_shape_org[1],:q_bnsd_shape_org[2],:q_rope_shape[-1]] = qr_bnsd_fp16_org
                    print(f"[INFO] qr_bnsd_fp16.shape is {qr_bnsd_fp16.shape}")
                    np.random.seed(9)
                    kr_bnsd_fp16 = torch.from_numpy(np.zeros( (k_shape_fp16[0], k_shape_fp16[1], k_shape_fp16[2], k_rope_shape[-1]) )).to(torch.float16)
                    # kr_bnsd_shape_org = torch.from_numpy(np.random.uniform(low=k_rope_datarange[0], high=k_rope_datarange[1], size=[k_shape_fp16[0], k_shape_fp16[1], k_shape_fp16[2], k_rope_shape[-1]])).to(torch.float16)
                    kr_bnsd_fp16_org = torch.from_numpy(np.random.randn(k_shape_fp16[0], k_shape_fp16[1], k_shape_fp16[2], k_rope_shape[-1])).to(torch.float16)
                    kr_bnsd_fp16[:k_shape_fp16[0],:k_shape_fp16[1],:k_shape_fp16[2],:k_rope_shape[-1]] = kr_bnsd_fp16_org
                    print(f"[INFO] kr_bnsd_fp16.shape is {kr_bnsd_fp16.shape}")

                fp8_scale_dtype = torch.float32
                # FP8类型 float8_e5m2 float8_e4m3fn
                fp8_dtype = torch.float8_e5m2
                if q_dtype_np == float8_e4m3fn:
                    fp8_dtype = torch.float8_e4m3fn
                if q_dtype_np == np.uint8:
                    fp8_dtype = torch.uint8


                # 获取量化scale
                scale_q_bnsd = get_mxfp8_per_token_group_quant_scale(q_bnsd_fp16, fp8_dtype, 32).to(fp8_scale_dtype)
                scale_k_bnsd = get_mxfp8_per_token_group_quant_scale(k_bnsd_fp16, fp8_dtype, 32).to(fp8_scale_dtype)
                scale_v_bnsd = get_mxfp8_per_channel_group_quant_scale(v_bnsd_fp16, fp8_dtype, 32).to(fp8_scale_dtype)
                # 反量化因子
                dequantScale_query = 1 / scale_q_bnsd
                print("[INFO] dequantScale_query shape is:", dequantScale_query.shape)
                k_antiquant_scale = 1 / scale_k_bnsd
                print("[INFO] k_antiquant_scale shape is:", k_antiquant_scale.shape)
                v_antiquant_scale = 1 / scale_v_bnsd
                print("[INFO] v_antiquant_scale shape is:", v_antiquant_scale.shape)


                deq_scale_q_tmp = dequantScale_query.numpy()
                deq_scale_k_tmp = k_antiquant_scale.numpy()
                deq_scale_v_tmp = v_antiquant_scale.numpy()

                deq_scale_q_np = convert_q_scale_bnsd_to_layout(deq_scale_q_tmp, actualSeqLengths, inputLayout, q_bnsd_shape_org)
                deq_scale_k_np = convert_k_scale_bnsd_to_layout(deq_scale_k_tmp, actualSeqLengthsKV, inputLayout, k_bnsd_shape_org, params)
                deq_scale_v_np = convert_v_scale_bnsd_to_layout(deq_scale_v_tmp, actualSeqLengthsKV, inputLayout, v_bnsd_shape_org, params)

                print("------------生成Q/K/V反量化因子bin数据------------")
                print("[INFO] deq_scale_q_np in bin shape is:", deq_scale_q_np.shape)
                print("[INFO] deq_scale_k_np in bin shape is:", deq_scale_k_np.shape)
                print("[INFO] deq_scale_v_np in bin shape is:", deq_scale_v_np.shape)
                deq_scale_q_np = convert_q_scale_tnd_to_n2gtd_layout(deq_scale_q_np, numKeyValueHeads)

                #  覆盖原有的bin文件
                input_file_deq = os.path.join(params['case_path'], params['case_name'] + '_input_' + str(27) + '.bin') # 文件名
                input_file_dek = os.path.join(params['case_path'], params['case_name'] + '_input_' + str(15) + '.bin') # 文件名
                input_file_dev = os.path.join(params['case_path'], params['case_name'] + '_input_' + str(17) + '.bin') # 文件名
                tools.dump_file(deq_scale_q_np.astype(float8_e8m0fnu), input_file_deq, "float8_e8m0fnu", params)
                tools.dump_file(deq_scale_k_np.astype(float8_e8m0fnu), input_file_dek, "float8_e8m0fnu", params)
                tools.dump_file(deq_scale_v_np.astype(float8_e8m0fnu), input_file_dev, "float8_e8m0fnu", params)

                # 将原始FP16数据量化成FP8
                q_bnsd_fp8 = mxfp8_per_token_group_quant_or_dequant_single(q_bnsd_fp16, scale_q_bnsd, 32).to(fp8_dtype)
                k_bnsd_fp8 = mxfp8_per_token_group_quant_or_dequant_single(k_bnsd_fp16, scale_k_bnsd, 32).to(fp8_dtype)
                v_bnsd_fp8 = mxfp8_per_channel_group_quant_or_dequant_single(v_bnsd_fp16, scale_v_bnsd, 32).to(fp8_dtype)
                q = q_bnsd_fp8.to(torch.float32)
                k = k_bnsd_fp8.to(torch.float32)
                v = v_bnsd_fp8.to(torch.float32)
                q_np_tmp = q.to(torch.float).numpy()
                k_np_tmp = k.to(torch.float).numpy()
                v_np_tmp = v.to(torch.float).numpy()

                q_np, q_np_tmp = convert_q_bnsd_to_layout(q_np_tmp, actualSeqLengths, inputLayout, q_bnsd_shape_org)
                k_np, k_np_tmp = convert_kv_bnsd_to_layout(k_np_tmp, actualSeqLengthsKV, inputLayout, k_bnsd_shape_org, params)
                v_np, v_np_tmp = convert_kv_bnsd_to_layout(v_np_tmp, actualSeqLengthsKV, inputLayout, v_bnsd_shape_org, params)

                print("------------生成Q/K/V FP8 bin数据------------")
                print("[INFO] q_np shape in bin is", q_np.shape)
                print("[INFO] k_np shape in bin is", k_np.shape)
                print("[INFO] v_np shape in bin is", v_np.shape)
                # 覆盖原有的bin文件
                input_file_q = os.path.join(params['case_path'], params['case_name'] + '_input_' + str(0) + '.bin') # 文件名
                input_file_k = os.path.join(params['case_path'], params['case_name'] + '_input_' + str(1) + '.bin') # 文件名
                input_file_v = os.path.join(params['case_path'], params['case_name'] + '_input_' + str(2) + '.bin') # 文件名
                tools.dump_file(q_np.astype(q_dtype_np), input_file_q, str(q_dtype_np), params)
                tools.dump_file(k_np.astype(q_dtype_np), input_file_k, str(q_dtype_np), params)
                tools.dump_file(v_np.astype(q_dtype_np), input_file_v, str(q_dtype_np), params)

                if len(flagList) >= 27 and flagList[26]:
                    qr_np_tmp = qr_bnsd_fp16.to(torch.float).numpy()
                    kr_np_tmp = kr_bnsd_fp16.to(torch.float).numpy()
                    qr_np, _  = convert_q_bnsd_to_layout(qr_np_tmp, actualSeqLengths, inputLayout, q_bnsd_shape_org)
                    kr_np, _ = convert_kv_bnsd_to_layout(kr_np_tmp, actualSeqLengthsKV, inputLayout, k_bnsd_shape_org, params)
                    print("----------生成Q/K rope FP16 bin数据----------")
                    print("[INFO] qr_np shape in bin is", qr_np.shape)
                    print("[INFO] kr_np shape in bin is", kr_np.shape)
                    # 覆盖原有的bin文件
                    input_file_qr = os.path.join(params['case_path'], params['case_name'] + '_input_' + str(22) + '.bin') # 文件名
                    input_file_kr = os.path.join(params['case_path'], params['case_name'] + '_input_' + str(23) + '.bin') # 文件名
                    tools.dump_file(qr_np.astype(np.float16), input_file_qr, "float16", params)
                    tools.dump_file(kr_np.astype(np.float16), input_file_kr, "float16", params)


                if flagList[8]:
                    # s_tmp = torch.matmul(q_bnsd_fp16.to(torch.float32), k_bnsd_fp16.to(torch.float32).permute(0, 1, 3, 2))
                    # m_tmp, _ = torch.max(s_tmp, dim=-1, keepdims = True)
                    # p_tmp = torch.exp(s_tmp - m_tmp)
                    # p_qscale = get_tensor_quant_scale(p_tmp, fp8_dtype).to(fp8_scale_dtype)
                    # p_qscale_np = p_qscale.numpy()
                    # np.random.seed(228)
                    # p_qscale = np.random.uniform(1, 448, size=1).astype(np.float32)
                    p_qscale = np.ones(1).astype(np.float32)
                    print("pjjjjjjj, p_qscale = ", p_qscale)
                    p_qscale_np = torch.from_numpy(p_qscale)
                    input_file_deq = os.path.join(params['case_path'], params['case_name'] + '_input_' + str(6) + '.bin') # 文件名
                    print(input_file_deq)
                    tools.dump_file(p_qscale.astype(np.float32), input_file_deq, "float32", params)
                    print("[INFO] p_qscale_np shape in bin is ", p_qscale_np)

                q_tensor = torch.from_numpy(q_np_tmp).to(fp8_dtype)
                k_tensor = torch.from_numpy(k_np_tmp).to(fp8_dtype)
                v_tensor = torch.from_numpy(v_np_tmp).to(fp8_dtype)

                # k_tensor_list.clear()
                # v_tensor_list.clear()
                # k_tensor_list.append(k_bnsd_fp8)
                # v_tensor_list.append(v_bnsd_fp8)
                print("=============================================================")

            if params['k_antiquant_mode'] == 7:  # FP8 perblock 全量化
                # k_antiquant_scale = k_antiquantscale_tensor   # bnsd
                # print("k_antiquant_scale.shape:", k_antiquant_scale.shape)
                # v_antiquant_scale = v_antiquantscale_tensor   # bnsd
                # print("v_antiquant_scale.shape:", v_antiquant_scale.shape)
                # dequantScale_query = dequantScale_query_tensor # bnsd
                # print("dequantScale_query.shape:", dequantScale_query.shape)

                # 生成原始FP16的QKV  bnsd  和表格一致的shape
                np.random.seed(2)
                q_datarange = params['range_input'][0]
                q_bnsd_fp16 = torch.from_numpy(np.random.uniform(low=q_datarange[0], high=q_datarange[1], size=[q_bnsd_shape[0], q_bnsd_shape[1], q_bnsd_shape[2], q_bnsd_shape[3]])).to(torch.float16)
                # print("q_bnsd_fp16.shape is", q_bnsd_fp16.shape)
                np.random.seed(3)
                k_datarange = params['range_input'][1]
                k_shape_fp16 = k_shape_bnsd_origin_list[0]
                k_bnsd_fp16 = torch.from_numpy(np.random.uniform(low=k_datarange[0], high=k_datarange[1], size=[k_shape_fp16[0], k_shape_fp16[1], k_shape_fp16[2], k_shape_fp16[3]])).to(torch.float16)
                # print("k_bnsd_fp16.shape is", k_bnsd_fp16.shape)
                np.random.seed(4)
                v_datarange = params['range_input'][2]
                v_shape_fp16 = v_shape_bnsd_origin_list[0]
                v_bnsd_fp16 = torch.from_numpy(np.random.uniform(low=v_datarange[0], high=v_datarange[1], size=[v_shape_fp16[0], v_shape_fp16[1], v_shape_fp16[2], v_shape_fp16[3]])).to(torch.float16)
                # print("v_bnsd_fp16.shape is", v_bnsd_fp16.shape)

                fp8_scale_dtype = torch.float32
                # FP8类型 float8_e5m2 float8_e4m3fn
                fp8_dtype = torch.float8_e5m2
                if q_dtype_np == float8_e4m3fn:
                    fp8_dtype = torch.float8_e4m3fn
                if q_dtype_np == np.uint8:
                    fp8_dtype = torch.uint8
                # 获取量化scale
                scale_q_bnsd = get_block_quant_scale(q_bnsd_fp16, fp8_dtype, False, False, 128).to(fp8_scale_dtype)
                scale_k_bnsd = get_block_quant_scale(k_bnsd_fp16, fp8_dtype, False, False, 256).to(fp8_scale_dtype)
                scale_v_bnsd = get_block_quant_scale(v_bnsd_fp16, fp8_dtype, False, False, 256).to(fp8_scale_dtype)
                # 反量化因子
                dequantScale_query = 1 / scale_q_bnsd
                # print("dequantScale_query.shape:", dequantScale_query.shape)
                k_antiquant_scale = 1 / scale_k_bnsd
                # print("k_antiquant_scale.shape:", k_antiquant_scale.shape)
                v_antiquant_scale = 1 / scale_v_bnsd
                # print("v_antiquant_scale.shape:", v_antiquant_scale.shape)

                deq_scale_q_np = dequantScale_query.numpy()
                deq_scale_k_np = k_antiquant_scale.numpy()
                deq_scale_v_np = v_antiquant_scale.numpy()
                #  覆盖原有的bin文件
                input_file_deq = os.path.join(params['case_path'], params['case_name'] + '_input_' + str(27) + '.bin') # 文件名
                input_file_dek = os.path.join(params['case_path'], params['case_name'] + '_input_' + str(15) + '.bin') # 文件名
                input_file_dev = os.path.join(params['case_path'], params['case_name'] + '_input_' + str(17) + '.bin') # 文件名
                tools.dump_file(deq_scale_q_np.astype(np.float32), input_file_deq, "float32", params)
                tools.dump_file(deq_scale_k_np.astype(np.float32), input_file_dek, "float32", params)
                tools.dump_file(deq_scale_v_np.astype(np.float32), input_file_dev, "float32", params)

                # deq_scale_q_np.astype(np.float32).tofile(f"./fp8_input/d_scale_q_np.bin")
                # deq_scale_k_np.astype(np.float32).tofile(f"./fp8_input/d_scale_k_np.bin")
                # deq_scale_v_np.astype(np.float32).tofile(f"./fp8_input/d_scale_v_np.bin")
                # 将原始FP16数据量化成FP8
                if q_dtype != np.uint8:
                    q_bnsd_fp8 = block_quant_or_dequant_single(q_bnsd_fp16, scale_q_bnsd, False, 128).to(fp8_dtype)
                    k_bnsd_fp8 = block_quant_or_dequant_single(k_bnsd_fp16, scale_k_bnsd, False, 256).to(fp8_dtype)
                    v_bnsd_fp8 = block_quant_or_dequant_single(v_bnsd_fp16, scale_v_bnsd, False, 256).to(fp8_dtype)
                    q = q_bnsd_fp8.to(torch.float32)
                    k = k_bnsd_fp8.to(torch.float32)
                    v = v_bnsd_fp8.to(torch.float32)
                    q_np = q.to(torch.float).numpy()
                    k_np = k.to(torch.float).numpy()
                    v_np = v.to(torch.float).numpy()
                else:
                    q_bnsd = block_quant_or_dequant_single(q_bnsd_fp16, scale_q_bnsd, False, 128).to(torch.float32).numpy()
                    k_bnsd = block_quant_or_dequant_single(k_bnsd_fp16, scale_k_bnsd, False, 256).to(torch.float32).numpy()
                    v_bnsd = block_quant_or_dequant_single(v_bnsd_fp16, scale_v_bnsd, False, 256).to(torch.float32).numpy()
                    q_np = trans_np_float_tensor_to_hifuint8(q_bnsd)
                    k_np = trans_np_float_tensor_to_hifuint8(k_bnsd)
                    v_np = trans_np_float_tensor_to_hifuint8(v_bnsd)
                    q_bnsd_fp8 = torch.from_numpy(q_np)
                    k_bnsd_fp8 = torch.from_numpy(k_np)
                    v_bnsd_fp8 = torch.from_numpy(v_np)
                # 产生保存npu相关数据
                B, N1, S1, D = q_np.shape[0], q_np.shape[1], q_np.shape[2], q_np.shape[3]
                # print(B, N1, S1, D)
                B, N2, S2, DV = k_np.shape[0], k_np.shape[1], k_np.shape[2], k_np.shape[3]
                # print(B, N2, S2, DV)
                # 转化回原始的inputLayout
                if inputLayout == "BSND":
                    # (B,N,S,D) =>(B,S,N,D)
                    q_np = q_np.transpose(0, 2, 1, 3)
                    k_np = k_np.transpose(0, 2, 1, 3)
                    v_np = v_np.transpose(0, 2, 1, 3)
                elif inputLayout == "BSH":
                    # (B,N,S,D) =>(B,S,H)
                    q_np = q_np.transpose(0, 2, 1, 3).reshape(B, S1, N1 * D)
                    k_np = k_np.transpose(0, 2, 1, 3).reshape(B, S2, N2 * DV)
                    v_np = v_np.transpose(0, 2, 1, 3).reshape(B, S2, N2 * DV)
                # print("q_np.shape is", q_np.shape)
                # print("k_np.shape is", k_np.shape)
                # q_bnsd_fp8.to(torch.float16).numpy().astype(q_dtype_np).tofile(f"./fp8_input/q_fp8.bin")
                # k_bnsd_fp8.to(torch.float16).numpy().astype(q_dtype_np).tofile(f"./fp8_input/k_fp8.bin")
                # v_bnsd_fp8.to(torch.float16).numpy().astype(q_dtype_np).tofile(f"./fp8_input/v_fp8.bin")
                # 覆盖原有的bin文件
                input_file_q = os.path.join(params['case_path'], params['case_name'] + '_input_' + str(0) + '.bin') # 文件名
                input_file_k = os.path.join(params['case_path'], params['case_name'] + '_input_' + str(1) + '.bin') # 文件名
                input_file_v = os.path.join(params['case_path'], params['case_name'] + '_input_' + str(2) + '.bin') # 文件名
                tools.dump_file(q_np.astype(q_dtype_np), input_file_q, str(q_dtype_np), params)
                tools.dump_file(k_np.astype(q_dtype_np), input_file_k, str(q_dtype_np), params)
                tools.dump_file(v_np.astype(q_dtype_np), input_file_v, str(q_dtype_np), params)
                # q_np.astype(q_dtype_np).tofile(f"./fp8_input/q_np.bin")
                # k_np.astype(q_dtype_np).tofile(f"./fp8_input/k_np.bin")
                # v_np.astype(q_dtype_np).tofile(f"./fp8_input/v_np.bin")

                # 产生CPU计算相关数据
                if numKeyValueHeads != numHeads:
                #    k_antiquant_scale = broadcast_qkv_split_dequant_tensor(k_antiquant_scale, numKeyValueHeads,numHeads)
                #    v_antiquant_scale = broadcast_qkv_split_dequant_tensor(v_antiquant_scale, numKeyValueHeads,numHeads)
                   k_bnsd_fp8 = broadcastKV_sigle(numHeads, numKeyValueHeads, k_bnsd_fp8, k_bnsd_fp8.dtype)
                   v_bnsd_fp8 = broadcastKV_sigle(numHeads, numKeyValueHeads, v_bnsd_fp8, v_bnsd_fp8.dtype)

                q_tensor = q_bnsd_fp8
                k_tensor_list.clear()
                v_tensor_list.clear()
                k_tensor_list.append(k_bnsd_fp8)
                v_tensor_list.append(v_bnsd_fp8)

                # #GQA场景，扩展KV反量化参数
                # k_antiquantscale_n1d_tensor = _trans_h_to_n1d(k_antiquantscale_shape, k_antiquantscale_tensor,numKeyValueHeads)
                # v_antiquantscale_n1d_tensor = _trans_h_to_n1d(v_antiquantscale_shape, v_antiquantscale_tensor,numKeyValueHeads)
                # if numKeyValueHeads != numHeads:
                #     print("broad dequant")
                #     k_antiquant_scale = broadcast_qkv_split_dequant_tensor(k_antiquantscale_n1d_tensor, numKeyValueHeads,numHeads)
                #     v_antiquant_scale = broadcast_qkv_split_dequant_tensor(v_antiquantscale_n1d_tensor, numKeyValueHeads,numHeads)
                #     print("v_antiquant_scale.shape:", v_antiquant_scale.shape)
        pfa_param['k_antiquant_scale'] = k_antiquant_scale
        pfa_param['v_antiquant_scale'] = v_antiquant_scale
        pfa_param['dequantScale_query'] = dequantScale_query

        if flagList[18] or flagList[20]:
            q_dtype_np = tools.get_np_dtype(params['dtype_input'][0])
            k_antiquantoffset_shape = params['shape_input'][v_end_index + 14]
            k_antiquantoffset_tensor = torch_tensor_list[v_end_index + 14]
            v_antiquantoffset_shape = params['shape_input'][v_end_index + 16]
            v_antiquantoffset_tensor = torch_tensor_list[v_end_index + 16]

            if params['k_antiquant_mode'] == 0:
                if len(k_antiquantoffset_shape) == 1 and k_antiquantoffset_shape[0] == 1:  # pertensor
                    print("antiquantscale_pertensor")
                    k_antiquant_offset = k_antiquantoffset_tensor
                    v_antiquant_offset = v_antiquantoffset_tensor
                else:  # perchanel
                    # 将kv量化参数转换为2n1d(匹配bnsd)
                    k_antiquantoffset_n1d_tensor = _trans_h_to_n1d(k_antiquantoffset_shape, k_antiquantoffset_tensor,numKeyValueHeads)
                    v_antiquantoffset_n1d_tensor = _trans_h_to_n1d(v_antiquantoffset_shape, v_antiquantoffset_tensor,numKeyValueHeads)

                    # GQA场景，扩展kv反量化参数
                    if numKeyValueHeads != numHeads:
                        print("broadcast kvanti")
                        k_antiquant_offset = broadcast_kv_split_dequant_tensor(k_antiquantoffset_n1d_tensor, numKeyValueHeads,numHeads,q_dtype_np)
                        v_antiquant_offset = broadcast_kv_split_dequant_tensor(v_antiquantoffset_n1d_tensor,numKeyValueHeads,numHeads,q_dtype_np)
                    else:
                        k_antiquant_offset = k_antiquantoffset_n1d_tensor
                        v_antiquant_offset = v_antiquantoffset_n1d_tensor
            if params['k_antiquant_mode'] == 1:  # pertoken
                # pertoken -> BS
                k_antiquant_offset = k_antiquantoffset_tensor
                v_antiquant_offset = v_antiquantoffset_tensor

        pfa_param['k_antiquant_offset'] = k_antiquant_offset
        pfa_param['v_antiquant_offset'] = v_antiquant_offset

        pfa_param['anti_or_kvanti'] = anti_or_kvanti
        pfa_param['k_antiquant_mode'] = params['k_antiquant_mode']
        pfa_param['v_antiquant_mode'] = params['v_antiquant_mode']
        pfa_param['query_quant_mode'] = params['query_quant_mode']

        if flagList[14] and "output" not in params["action_type"]:
            # shape_input取tensorshape
            blockSize = params['blocksize']
            btShape = params['shape_input'][v_end_index + 10]
            kvcacheShape = params['shape_input'][v_end_index + 19]
            if 0 in btShape:
                print('[WARNING]:block_table为空场景，输出空tensor！')
                return torch.zeros(out_shape)
            if 0 in kvcacheShape:
                print('[WARNING]:PA场景下kvcache为空tensor，输出空tensor！')
                return torch.zeros(out_shape)
            block_idx_list = np.arange(kvcacheShape[0])
            block_idx_list = np.random.permutation(block_idx_list)
            block_idx = 0
            block_table = np.full(btShape, -1).astype(np.int32)
            # print(f"++++++++++++++++++++block_table {block_table}")
            # print(f"++++++++++++++++++++btShape {btShape}")
            print(f"++++++++++++++++++++params['shape_input'][21] {params['shape_input'][21]}")
            blockNumPerBlock = []
            for actual_seq in actualSeqLengthsKV:
                blockNumPerBlock.append(math.ceil(actual_seq / blockSize))
                # print(f"++++++++++++++++++++blockNumPerBlock {blockNumPerBlock}")
            for idx, block_per_b in enumerate(blockNumPerBlock):
                for j in range(block_per_b):
                    block_table[idx][j] = (block_idx_list[block_idx])
                    block_idx += 1

            input_file_block_table = os.path.join(params['case_path'], params['case_name'] + '_input_' + str(12) + '.bin') # 文件名
            tools.dump_file(block_table.astype(np.int32), input_file_block_table, str(np.int32), params)

            # tools.modify_alcnn_input_file(ids=12, origin_index=[12], type='tensor', mode='rewrite',
            #                               tensors=torch.from_numpy(block_table),
            #                               params=params)
            pfa_param['blockNumPerBlock'] = blockNumPerBlock
            pfa_param['block_table'] = block_table
            # print(f"++++++++++++++++++++block_table {block_table}")
            #### flagList[26] qrope
            if  len(flagList) >= 27 and flagList[26]:
                if params['query_quant_mode'] == 6 and params['k_antiquant_mode'] == 6 and params['v_antiquant_mode'] == 8:
                    print('[INFO]:PA + MXFP8 + rope分离传入！')
                    print('--------重新转换bin的K/V、K/V scale以及K rope的shape--------')
                    # print(f"++++++++++++++++++++block_table {block_table}")
                    # print(f"++++++++++++++++++++blockSize {blockSize}")

                    print("pjjjj, k_bnsd_fp8 shape", k_bnsd_fp8.shape)
                    print("pjjjj, kr_bnsd_fp16 shape", kr_bnsd_fp16.shape)

                    k_bnsd_fp8_reshape = mxfp8_pa_preprocessing(pfa_param, params, k_bnsd_fp8, 21)
                    v_bnsd_fp8_reshape = mxfp8_pa_preprocessing(pfa_param, params, v_bnsd_fp8, 22)
                    kr_bnsd_fp16_reshape = mxfp8_pa_preprocessing(pfa_param, params, kr_bnsd_fp16, 25)
                    deq_scale_k_reshape = mxfp8_pa_preprocessing(pfa_param, params, k_antiquant_scale, 15)
                    deq_scale_v_reshape = mxfp8_pa_preprocessing(pfa_param, params, v_antiquant_scale, 17, is_vscale = True)

                    k_bnsd_fp8_reshape_uint8 = k_bnsd_fp8_reshape.view(torch.uint8)
                    v_bnsd_fp8_reshape_uint8 = v_bnsd_fp8_reshape.view(torch.uint8)
                    k_cache_np = k_bnsd_fp8_reshape_uint8.numpy()
                    v_cache_np = v_bnsd_fp8_reshape_uint8.numpy()
                    k_cache_reshape = torch.from_numpy(k_cache_np).view(torch.float8_e4m3fn)
                    v_cache_reshape = torch.from_numpy(v_cache_np).view(torch.float8_e4m3fn)

                    print("pjjjj, kr_reshape_np", kr_bnsd_fp16_reshape.shape)

                    kr_reshape_np = kr_bnsd_fp16_reshape.numpy()
                    deq_scale_k_reshape_np = deq_scale_k_reshape.numpy()
                    deq_scale_v_reshape_np = deq_scale_v_reshape.numpy()
                    k_cache_reshape_np = k_cache_reshape.to(torch.float).numpy()
                    v_cache_reshape_np = v_cache_reshape.to(torch.float).numpy()


                    print(f"[INFO] k_cache shape in bin {k_np.shape} trans to {k_cache_reshape.shape}")
                    print(f"[INFO] v_cache shape in bin {v_np.shape} trans to {v_cache_reshape.shape}")
                    print(f"[INFO] k rope shape in bin {kr_np.shape} trans to {kr_bnsd_fp16_reshape.shape}")
                    print(f"[INFO] deq_scale_k shape in bin {deq_scale_k_np.shape} trans to {deq_scale_k_reshape.shape}")
                    print(f"[INFO] deq_scale_v shape in bin {deq_scale_v_np.shape} trans to {deq_scale_v_reshape.shape}")

                    # tools.modify_alcnn_input_file(ids=21, origin_index=[21], type='tensor_list', mode='rewrite', tensors=[k_cache_reshape],
                    #                             params=params, data_dtype="float8_e4m3fn")
                    # tools.modify_alcnn_input_file(ids=22, origin_index=[22], type='tensor_list', mode='rewrite', tensors=[v_cache_reshape],
                    #                             params=params, data_dtype="float8_e4m3fn")

                    #  覆盖原有的bin文件
                    input_file_dek = os.path.join(params['case_path'], params['case_name'] + '_input_' + str(15) + '.bin') # 文件名
                    input_file_dev = os.path.join(params['case_path'], params['case_name'] + '_input_' + str(17) + '.bin') # 文件名
                    input_file_kr = os.path.join(params['case_path'], params['case_name'] + '_input_' + str(25) + '.bin') # 文件名
                    input_file_kcache = os.path.join(params['case_path'], params['case_name'] + '_input_' + str(21) + '.bin') # 文件名
                    input_file_vcache = os.path.join(params['case_path'], params['case_name'] + '_input_' + str(22) + '.bin') # 文件名

                    tools.dump_file(deq_scale_k_reshape_np.astype(float8_e8m0fnu), input_file_dek, "float8_e8m0fnu", params)
                    tools.dump_file(deq_scale_v_reshape_np.astype(float8_e8m0fnu), input_file_dev, "float8_e8m0fnu", params)
                    tools.dump_file(kr_reshape_np.astype(np.float16), input_file_kr, "float16", params)
                    tools.dump_file(k_cache_reshape_np.astype(float8_e4m3fn), input_file_kcache, "float8_e4m3fn", params)
                    tools.dump_file(v_cache_reshape_np.astype(float8_e4m3fn), input_file_vcache, "float8_e4m3fn", params)
                    # pass
                else:
                    print('[INFO]:PA + Deepseek！')
                    deepseek_ds_pa_preprocessing(pfa_param, params,torch_tensor_list)
            else:
                if params['query_quant_mode'] == 6 and params['k_antiquant_mode'] == 6 and params['v_antiquant_mode'] == 8:
                    print('[INFO]:PA + MXFP8 + rope不分离传入！')
                    print('----------重新转换bin的K/V以及K/V scale的shape----------')
                    # print(f"++++++++++++++++++++block_table {block_table}")
                    # print(f"++++++++++++++++++++blockSize {blockSize}")
                    k_bnsd_fp8_reshape = mxfp8_pa_preprocessing(pfa_param, params, k_bnsd_fp8, 21)
                    v_bnsd_fp8_reshape = mxfp8_pa_preprocessing(pfa_param, params, v_bnsd_fp8, 22)
                    deq_scale_k_reshape = mxfp8_pa_preprocessing(pfa_param, params, k_antiquant_scale, 15)
                    deq_scale_v_reshape = mxfp8_pa_preprocessing(pfa_param, params, v_antiquant_scale, 17, is_vscale = True)

                    k_bnsd_fp8_reshape_uint8 = k_bnsd_fp8_reshape.view(torch.uint8)
                    v_bnsd_fp8_reshape_uint8 = v_bnsd_fp8_reshape.view(torch.uint8)
                    k_cache_np = k_bnsd_fp8_reshape_uint8.numpy()
                    v_cache_np = v_bnsd_fp8_reshape_uint8.numpy()
                    k_cache_reshape = torch.from_numpy(k_cache_np).view(torch.float8_e4m3fn)
                    v_cache_reshape = torch.from_numpy(v_cache_np).view(torch.float8_e4m3fn)

                    deq_scale_k_reshape_np = deq_scale_k_reshape.numpy()
                    deq_scale_v_reshape_np = deq_scale_v_reshape.numpy()
                    k_cache_reshape_np = k_cache_reshape.to(torch.float).numpy()
                    v_cache_reshape_np = v_cache_reshape.to(torch.float).numpy()

                    print(f"[INFO] k_cache shape in bin {k_np.shape} trans to {k_cache_reshape.shape}")
                    print(f"[INFO] v_cache shape in bin {v_np.shape} trans to {v_cache_reshape.shape}")
                    print(f"[INFO] deq_scale_k shape in bin {deq_scale_k_np.shape} trans to {deq_scale_k_reshape.shape}")
                    print(f"[INFO] deq_scale_v shape in bin {deq_scale_v_np.shape} trans to {deq_scale_v_reshape.shape}")

                    # tools.modify_alcnn_input_file(ids=21, origin_index=[21], type='tensor_list', mode='rewrite', tensors=[k_cache_reshape],
                    #                             params=params, data_dtype="float8_e4m3fn")
                    # tools.modify_alcnn_input_file(ids=22, origin_index=[22], type='tensor_list', mode='rewrite', tensors=[v_cache_reshape],
                    #                             params=params, data_dtype="float8_e4m3fn")

                    #  覆盖原有的bin文件
                    input_file_dek = os.path.join(params['case_path'], params['case_name'] + '_input_' + str(15) + '.bin') # 文件名
                    input_file_dev = os.path.join(params['case_path'], params['case_name'] + '_input_' + str(17) + '.bin') # 文件名
                    input_file_kcache = os.path.join(params['case_path'], params['case_name'] + '_input_' + str(21) + '.bin') # 文件名
                    input_file_vcache = os.path.join(params['case_path'], params['case_name'] + '_input_' + str(22) + '.bin') # 文件名

                    tools.dump_file(deq_scale_k_reshape_np.astype(float8_e8m0fnu), input_file_dek, "float8_e8m0fnu", params)
                    tools.dump_file(deq_scale_v_reshape_np.astype(float8_e8m0fnu), input_file_dev, "float8_e8m0fnu", params)
                    tools.dump_file(k_cache_reshape_np.astype(float8_e4m3fn), input_file_kcache, "float8_e4m3fn", params)
                    tools.dump_file(v_cache_reshape_np.astype(float8_e4m3fn), input_file_vcache, "float8_e4m3fn", params)
                else:
                    print('[INFO]:PA + no Deepseek！')
                    # trans kv to bsh(此处使用的tensor, 没有经过n的扩展)
                    k_cache = torch_tensor_list[v_end_index + 19]
                    v_cache = torch_tensor_list[v_end_index + 20]
                    print(k_cache.shape)
                    print(v_cache.shape)
                    if k_cache.ndim == 3:
                        k_tensor_bsh_raw = trans_bnsd_to_bsh(k_tensor_bnsd_list[0], k_shape_bnsd_raw_list[0])
                        v_tensor_bsh_raw = trans_bnsd_to_bsh(v_tensor_bnsd_list[0], v_shape_bnsd_raw_list[0])
                        for b, block_per_b in enumerate(blockNumPerBlock):
                            for blockid_index in range(block_per_b):
                                blockid = block_table[b][blockid_index]
                                block_offset = blockid_index * blockSize
                                if blockid_index == block_per_b - 1:
                                    blocksize_left = actualSeqLengthsKV[b] - block_offset
                                    k_cache[blockid, 0:blocksize_left, :] = k_tensor_bsh_raw[b,
                                                                            block_offset:(block_offset + blocksize_left), :]
                                    v_cache[blockid, 0:blocksize_left, :] = v_tensor_bsh_raw[b,
                                                                            block_offset:(block_offset + blocksize_left), :]
                                else:
                                    k_cache[blockid, 0:blockSize, :] = k_tensor_bsh_raw[b,
                                                                    block_offset:(block_offset + blockSize), :]
                                    v_cache[blockid, 0:blockSize, :] = v_tensor_bsh_raw[b,
                                                                    block_offset:(block_offset + blockSize), :]
                    if k_cache.ndim == 4:
                        k_tensor_bsh_raw = k_tensor_bnsd_list[0]
                        v_tensor_bsh_raw = v_tensor_bnsd_list[0]

                        # gen kv cache
                        for b, block_per_b in enumerate(blockNumPerBlock):
                            for blockid_index in range(block_per_b):
                                blockid = block_table[b][blockid_index]
                                block_offset = blockid_index * blockSize
                                if blockid_index == block_per_b - 1:
                                    blocksize_left = actualSeqLengthsKV[b] - block_offset
                                    k_cache[blockid, :, 0:blocksize_left, :] = k_tensor_bsh_raw[b, :,
                                                                            block_offset:(block_offset + blocksize_left),
                                                                            :]
                                    v_cache[blockid, :, 0:blocksize_left, :] = v_tensor_bsh_raw[b, :,
                                                                            block_offset:(block_offset + blocksize_left),
                                                                            :]
                                else:
                                    k_cache[blockid, :, 0:blockSize, :] = k_tensor_bsh_raw[b, :,
                                                                        block_offset:(block_offset + blockSize), :]
                                    v_cache[blockid, :, 0:blockSize, :] = v_tensor_bsh_raw[b, :,
                                                                        block_offset:(block_offset + blockSize), :]

                    if k_cache.ndim == 5:
                        k_tensor_bsh_raw = k_tensor_bnsd_list[0]
                        v_tensor_bsh_raw = v_tensor_bnsd_list[0]
                        kd = k_tensor_bsh_raw.shape[3]
                        vd = v_tensor_bsh_raw.shape[3]
                        k_cache_bnsd = np.zeros((k_cache.shape[0], k_cache.shape[1], blockSize, kd))
                        v_cache_bnsd = np.zeros((k_cache.shape[0], k_cache.shape[1], blockSize, vd))
                        # gen kv cache
                        for b, block_per_b in enumerate(blockNumPerBlock):
                            for blockid_index in range(block_per_b):
                                blockid = block_table[b][blockid_index]
                                block_offset = blockid_index * blockSize
                                if blockid_index == block_per_b - 1:
                                    blocksize_left = actualSeqLengthsKV[b] - block_offset
                                    k_cache_bnsd[blockid, :, 0:blocksize_left, :] = k_tensor_bsh_raw[b, :,
                                                                                    block_offset:(
                                                                                                block_offset + blocksize_left),
                                                                                    :]
                                    v_cache_bnsd[blockid, :, 0:blocksize_left, :] = v_tensor_bsh_raw[b, :,
                                                                                    block_offset:(
                                                                                                block_offset + blocksize_left),
                                                                                    :]
                                else:
                                    k_cache_bnsd[blockid, :, 0:blockSize, :] = k_tensor_bsh_raw[b, :,
                                                                            block_offset:(block_offset + blockSize), :]
                                    v_cache_bnsd[blockid, :, 0:blockSize, :] = v_tensor_bsh_raw[b, :,
                                                                            block_offset:(block_offset + blockSize), :]

                            if k_dtype == "int8":
                                base = 32
                            else:
                                base = 16
                            k_cache = k_cache_bnsd.reshape(k_cache_bnsd.shape[0],
                                                        k_cache_bnsd.shape[1],
                                                        k_cache_bnsd.shape[2],
                                                        k_cache_bnsd.shape[3] // base,
                                                        base).transpose(0, 1, 3, 2, 4)
                            v_cache = v_cache_bnsd.reshape(v_cache_bnsd.shape[0],
                                                        v_cache_bnsd.shape[1],
                                                        v_cache_bnsd.shape[2],
                                                        v_cache_bnsd.shape[3] // base,
                                                        base).transpose(0, 1, 3, 2, 4)
                            print(f"[INFO]k_cache NZ: {k_cache.shape}")
                            print(f"[INFO]v_cache NZ: {v_cache.shape}")

                    kv_dtype = tools.get_np_dtype(params['dtype_input'][1])

                    if str(kv_dtype) == "<class 'ml_dtypes.bfloat16'>":
                        k_cache = torch.from_numpy(k_cache).to(torch.bfloat16)
                        v_cache = torch.from_numpy(v_cache).to(torch.bfloat16)
                    elif str(kv_dtype) != "<class 'float8_e5m2'>" and params['dtype_input'][1] != "hifloat8" and \
                            params['dtype_input'][1] != "float8_e4m3fn":
                        k_cache = torch.from_numpy(k_cache.astype(kv_dtype))
                        v_cache = torch.from_numpy(v_cache.astype(kv_dtype))

                    if str(kv_dtype) != "<class 'float8_e5m2'>" and params['dtype_input'][1] != "hifloat8" and params['dtype_input'][1] != "float8_e4m3fn":
                        k_cache_np = k_cache.numpy()
                        v_cache_np = v_cache.numpy()
                        input_file_kcache = os.path.join(params['case_path'], params['case_name'] + '_input_' + str(21) + '.bin') # 文件名
                        input_file_vcache = os.path.join(params['case_path'], params['case_name'] + '_input_' + str(22) + '.bin') # 文件名
                        tools.dump_file(k_cache_np.astype(kv_dtype), input_file_kcache, str(kv_dtype), params)
                        tools.dump_file(v_cache_np.astype(kv_dtype), input_file_vcache, str(kv_dtype), params)

                        # tools.modify_alcnn_input_file(ids=21, origin_index=[21], type='tensor_list', mode='rewrite', tensors=[k_cache],
                        #                             params=params)
                        # tools.modify_alcnn_input_file(ids=22, origin_index=[22], type='tensor_list', mode='rewrite', tensors=[v_cache],
                        #                             params=params)
                    else:
                        k_cache_np = k_cache.numpy()
                        v_cache_np = v_cache.numpy()
                        input_file_kcache = os.path.join(params['case_path'], params['case_name'] + '_input_' + str(21) + '.bin') # 文件名
                        input_file_vcache = os.path.join(params['case_path'], params['case_name'] + '_input_' + str(22) + '.bin') # 文件名
                        tools.dump_file(k_cache_np.astype(kv_dtype), input_file_kcache, str(kv_dtype), params)
                        tools.dump_file(v_cache_np.astype(kv_dtype), input_file_vcache, str(kv_dtype), params)

                        # tools.modify_alcnn_input_file(ids=21, origin_index=[21], type='tensor_list',
                        #                             mode='rewrite',
                        #                             tensors=[k_cache],
                        #                             params=params,
                        #                             data_dtype=params['dtype_input'][1])
                        # tools.modify_alcnn_input_file(ids=22, origin_index=[22], type='tensor_list',
                        #                             mode='rewrite',
                        #                             tensors=[v_cache],
                        #                             params=params,
                        #                             data_dtype=params['dtype_input'][2])

    if flagList[15]:
        pfa_param['queryPaddingSize'] = torch_tensor_list[v_end_index+11]
    if flagList[16]:
        pfa_param['kvPaddingSize'] = torch_tensor_list[v_end_index+12]
    pfa_param['lseflag'] = params["softmax_lse_flag"]
    pfa_param['lseshape'] = [q_bnsd_shape[0],q_bnsd_shape[1],q_bnsd_shape[2],1]
    if len(flagList) > 32 and flagList[32]:
        sinks = torch_tensor_list[28]
    else:
        sinks = None

    y_all, lse = _np_promtattention_bnsd(q_tensor, q_bnsd_shape_org, k_tensor_list, k_shape_bnsd_list, v_tensor_list,
                                         v_shape_bnsd_list,
                                         pse_bnsd_tensor,
                                         m_bnsd_tensor, scaleValue, actualSeqLengths, actualSeqLengthsKV, slopes, preTokens,
                                         nextTokens, sp_mode, pfa_param, dequant_scale1,
                                         dequant_scale2, quant_scale1, dequantScale_query, k_antiquant_scale, v_antiquant_scale,
                                         quant_scale2, quant_offset2, antiquant_scale,
                                         antiquant_offset, psetype, qstartidx, kvstartidx, out_dtype, params['dtype_input'][0], sinks)

    if q_tensor.dtype == "float8_e5m2" or q_tensor.dtype == "float8_e4m3fn" or q_tensor.dtype == torch.float8_e5m2 or q_tensor.dtype == torch.float8_e4m3fn or q_tensor.dtype == torch.uint8:
        if (pfa_param['k_antiquant_mode']) == 7:
            ## FP8 perblock全量化golden
            k_tensor = k_tensor_list[0]
            v_tensor = v_tensor_list[0]
            qDtype = q_tensor.dtype
            print("out_dtype is", out_dtype)
            print('running flash_attention_fp8')
            EPSILON = 1e-20
            Q_BLOCK_SIZE = 128
            KV_BLOCK_SIZE = 256  # 计算的基本块
            KV_SCALE_SIZE = 256   # scale生成的单位

            deq_scale_q = dequantScale_query
            deq_scale_k = k_antiquant_scale
            deq_scale_v = v_antiquant_scale
            if qDtype == torch.float8_e5m2 or qDtype == torch.float8_e4m3fn:
                q_tensor = q_tensor.to(torch.float32)
                k_tensor = k_tensor.to(torch.float32)
                v_tensor = v_tensor.to(torch.float32)
            else:
                q_tensor = torch.from_numpy(trans_np_hifuint8_tensor_to_float32(q_tensor.numpy()))
                k_tensor = torch.from_numpy(trans_np_hifuint8_tensor_to_float32(k_tensor.numpy()))
                v_tensor = torch.from_numpy(trans_np_hifuint8_tensor_to_float32(v_tensor.numpy()))
            b, n, s, d = q_tensor.shape
            dv = v_tensor.shape[-1]
            out = torch.zeros([b, n, s, dv], dtype=torch.float32)  # out
            o_sum = torch.zeros(q_tensor.shape[:-1])[..., None]  # sum
            o_max = torch.ones(q_tensor.shape[:-1])[..., None] * torch.finfo(torch.float).min  # max
            Sq = q_tensor.shape[2]
            Skv = k_tensor.shape[2]
            Tq = (Sq + Q_BLOCK_SIZE - 1) // Q_BLOCK_SIZE
            Tkv = (Skv + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE

            Q_BLOCKS = list(torch.split(q_tensor, Q_BLOCK_SIZE, dim=2))
            K_BLOCKS = list(torch.split(k_tensor, KV_BLOCK_SIZE, dim=2))
            V_BLOCKS = list(torch.split(v_tensor, KV_BLOCK_SIZE, dim=2))
            o_BLOCKS = list(torch.split(out, Q_BLOCK_SIZE, dim=2))
            s_BLOCKS = list(torch.split(o_sum, Q_BLOCK_SIZE, dim=2))
            m_BLOCKS = list(torch.split(o_max, Q_BLOCK_SIZE, dim=2))
            if deq_scale_q.size(1) != deq_scale_k.size(1):
                deq_scale_k = broadcastKV_sigle(q_tensor.size(1), deq_scale_k.size(1), deq_scale_k, deq_scale_k.dtype)
                deq_scale_v = broadcastKV_sigle(q_tensor.size(1), deq_scale_v.size(1), deq_scale_v, deq_scale_v.dtype)
            for j in range(Tkv):
                Kj = K_BLOCKS[j]
                Vj = V_BLOCKS[j]
                Skv_start = j * KV_BLOCK_SIZE
                Skv_end = min(Skv_start + KV_BLOCK_SIZE, Skv)
                for i in range(Tq):
                    Qi = Q_BLOCKS[i]    # Q分块
                    oi = o_BLOCKS[i]    # out
                    si = s_BLOCKS[i]    # sum
                    mi = m_BLOCKS[i]    # max
                    Sq_start = i * Q_BLOCK_SIZE
                    Sq_end = min(Sq_start + Q_BLOCK_SIZE,Sq)
                    # print("start matmul1")
                    S_ij = torch.matmul(Qi.to(torch.float32), Kj.to(torch.float32).permute(0, 1, 3, 2))
                    # print("end matmul1")
                    # 对bmm1结果反量化
                    S_ij = S_ij * deq_scale_q[:,:,i:i+1,:] * deq_scale_k[:,:,j * KV_BLOCK_SIZE // KV_SCALE_SIZE:j * KV_BLOCK_SIZE // KV_SCALE_SIZE+1,:]
                    S_ij = S_ij * 1 / math.sqrt(d)

                    # softmax计算
                    m_block_ij, _ = torch.max(S_ij, dim=-1, keepdims=True)
                    mi_new = torch.maximum(m_block_ij, mi)
                    P_ij = torch.exp(S_ij - mi_new)
                    torch.set_printoptions(profile="full", precision=4)
                    s_block_ij = torch.sum(P_ij, dim=-1, keepdims=True)
                    P_ij_drop = P_ij.clone()


                    if qDtype == "float8_e5m2" or qDtype == torch.float8_e5m2:
                        P_ij_drop = P_ij_drop.to(torch.float8_e5m2)    # softmax计算后cast成FP8
                        P_ij_drop = P_ij_drop.to(torch.float32)
                    elif qDtype == torch.uint8:
                        P_ij_drop_hifp8 = trans_np_float_tensor_to_hifuint8(P_ij_drop.numpy())
                        P_ij_drop_np32 = trans_np_hifuint8_tensor_to_float32(P_ij_drop_hifp8)
                        P_ij_drop = torch.from_numpy(P_ij_drop_np32)
                    else:
                        P_ij_drop = P_ij_drop.to(torch.float8_e4m3fn)
                        P_ij_drop = P_ij_drop.to(torch.float32)

                    # print("start matmul2")
                    P_ij_Vj = torch.matmul(P_ij_drop,Vj.to(torch.float32))
                    # print("end matmul2")
                    # 对bmm2结果反量化
                    P_ij_Vj = P_ij_Vj * deq_scale_v[:,:,j * KV_BLOCK_SIZE // KV_SCALE_SIZE:j * KV_BLOCK_SIZE // KV_SCALE_SIZE+1,:]
                    torch.set_printoptions(sci_mode=False)
                    si_new = torch.exp(mi - mi_new) * si +  s_block_ij
                    o_BLOCKS[i] = (si * torch.exp(mi - mi_new) * oi +  P_ij_Vj)/si_new
                    s_BLOCKS[i] = si_new
                    m_BLOCKS[i] = mi_new
            out = torch.cat(o_BLOCKS, dim=2)
            o_sum = torch.cat(s_BLOCKS, dim=2)
            o_max = torch.cat(m_BLOCKS, dim=2)
            y_all = out.numpy()

        if (pfa_param['k_antiquant_mode']) == 6 and (pfa_param['query_quant_mode']) == 6 and (pfa_param['v_antiquant_mode']) == 8:
            ## MXFP8 全量化golden  Q/K:per-channel-group V:per-token-group
            print("====================开始MXFP8 GQA golden计算====================")
            # k_tensor = k_tensor_list[0]
            # v_tensor = v_tensor_list[0]
            deq_scale_q = dequantScale_query
            deq_scale_k = k_antiquant_scale
            deq_scale_v = v_antiquant_scale

            qDtype = q_tensor.dtype
            print("out_dtype is", out_dtype)
            print('running flash_attention_mxfp8')
            # torch.set_printoptions(profile="full", precision=4)
            torch.set_printoptions(sci_mode=False)

            if numKeyValueHeads != numHeads:
                print("------------------GQA场景, 广播KV的N------------------")
                k_tensor = broadcastKV_sigle(numHeads, numKeyValueHeads, k_tensor, k_tensor.dtype)
                v_tensor = broadcastKV_sigle(numHeads, numKeyValueHeads, v_tensor, v_tensor.dtype)
                deq_scale_k = broadcastKV_sigle(numHeads, numKeyValueHeads, deq_scale_k, deq_scale_k.dtype)
                deq_scale_v = broadcastKV_sigle(numHeads, numKeyValueHeads, deq_scale_v, deq_scale_v.dtype)
                print(f"[INFO] q_tensor shape {q_tensor.shape} and dtype {q_tensor.dtype}")
                print(f"[INFO] k_tensor shape {k_tensor.shape} and dtype {k_tensor.dtype}")
                print(f"[INFO] v_tensor shape {v_tensor.shape} and dtype {v_tensor.dtype}")
                print(f"[INFO] deq_scale_q shape {deq_scale_q.shape} and dtype {deq_scale_q.dtype}")
                print(f"[INFO] deq_scale_k shape {deq_scale_k.shape} and dtype {deq_scale_k.dtype}")
                print(f"[INFO] deq_scale_v shape {deq_scale_v.shape} and dtype {deq_scale_v.dtype}")
                if len(flagList) >= 27 and flagList[26]:
                    kr_bnsd_fp16 = broadcastKV_sigle(numHeads, numKeyValueHeads, kr_bnsd_fp16, kr_bnsd_fp16.dtype)
                    print(f"[INFO] qr_bnsd_fp16.shape {qr_bnsd_fp16.shape} and dtype {qr_bnsd_fp16.dtype}")
                    print(f"[INFO] kr_bnsd_fp16.shape {kr_bnsd_fp16.shape} and dtype {kr_bnsd_fp16.dtype}")

            EPSILON = 1e-20
            Q_BLOCK_SIZE = 128
            KV_BLOCK_SIZE = 256  # 计算的基本块,C1V1C1V1C2V2
            QK_SCALE_SIZE = 32   # QK scale生成的单位 per-token-group
            V_SCALE_SIZE = 32   # V scale生成的单位 per-channel-group
            K_BLOCK_SIZE = int(KV_BLOCK_SIZE/2)
            V_BLOCK_SIZE = KV_BLOCK_SIZE

            if qDtype == torch.float8_e5m2 or qDtype == torch.float8_e4m3fn:
                q_tensor = q_tensor.to(torch.float32)
                k_tensor = k_tensor.to(torch.float32)
                v_tensor = v_tensor.to(torch.float32)
            else:
                q_tensor = torch.from_numpy(trans_np_hifuint8_tensor_to_float32(q_tensor.numpy()))
                k_tensor = torch.from_numpy(trans_np_hifuint8_tensor_to_float32(k_tensor.numpy()))
                v_tensor = torch.from_numpy(trans_np_hifuint8_tensor_to_float32(v_tensor.numpy()))
            b, n, s, d = q_tensor.shape
            if flagList[26] and flagList[27]: # nope rope分离传入
                qr_tensor = qr_bnsd_fp16
                kr_tensor = kr_bnsd_fp16
                dr = qr_tensor.shape[-1]
                d = d + dr
            dv = v_tensor.shape[-1]
            out = torch.zeros([b, n, s, dv], dtype=torch.float32)  # out
            o_sum = torch.zeros(q_tensor.shape[:-1])[..., None]  # sum
            o_max = torch.ones(q_tensor.shape[:-1])[..., None] * torch.finfo(torch.float).min  # max
            Sq = q_tensor.shape[2]
            Skv = k_tensor.shape[2]
            Tq = (Sq + Q_BLOCK_SIZE - 1) // Q_BLOCK_SIZE
            Tkv = (Skv + K_BLOCK_SIZE - 1) // K_BLOCK_SIZE

            if inputLayout == "TND":
                q_lens_t = torch.tensor(actualSeqLengths, dtype=torch.int32)
                k_lens_t = torch.tensor(actualSeqLengthsKV, dtype=torch.int32)
                q_lens_acl = q_lens_t.view(b, 1, 1, 1)  # [B, 1, 1, 1]
                k_lens_acl = k_lens_t.view(b, 1, 1, 1)  # [B, 1, 1, 1]
            else :
                q_seq_len = q_bnsd_shape_org[2]
                k_seq_len = k_bnsd_shape_org[2]
                q_lens_t = torch.full((b,), q_seq_len, dtype=torch.int32)
                k_lens_t = torch.full((b,), k_seq_len, dtype=torch.int32)
                if actualSeqLengths != 0 :
                    q_lens_t = torch.tensor(actualSeqLengths, dtype=torch.int32)
                if actualSeqLengthsKV != 0 :
                    k_lens_t = torch.tensor(actualSeqLengthsKV, dtype=torch.int32)
                q_lens_acl = q_lens_t.view(b, 1, 1, 1)
                k_lens_acl = k_lens_t.view(b, 1, 1, 1)
            # print(f"q_lens_acl{q_lens_acl}")
            # print(f"k_lens_acl{k_lens_acl}")
            q_range = torch.arange(Sq).view(1, 1, -1, 1)  # (Sq, 1)
            k_range = torch.arange(Skv).view(1, 1, 1, -1) # (1, Skv)
            q_padding_mask = q_range >= q_lens_acl
            k_padding_mask = k_range >= k_lens_acl
            # Causal 条件: k_idx > q_idx
            delta = k_lens_acl - q_lens_acl
            causal_mask = k_range > (q_range + delta)  # [1, 1, Sq, Skv]
            if pfa_param['sparse' ]== 3:
                # causal_mask = k_range > q_range  # (Sq, Skv)
                mask_global = causal_mask | q_padding_mask | k_padding_mask
            else:
                # Sparse Mode 0: 仅应用 Padding Mask
                mask_global = q_padding_mask | k_padding_mask
            # print(f"[INFO] causal_mask_j shape: {mask_global.shape} \n{mask_global[:,:,:,:]}")

            Q_BLOCKS = list(torch.split(q_tensor, Q_BLOCK_SIZE, dim=2))
            K_BLOCKS = list(torch.split(k_tensor, K_BLOCK_SIZE, dim=2))
            V_BLOCKS = list(torch.split(v_tensor, V_BLOCK_SIZE, dim=2))
            o_BLOCKS = list(torch.split(out, Q_BLOCK_SIZE, dim=2))
            s_BLOCKS = list(torch.split(o_sum, Q_BLOCK_SIZE, dim=2))
            m_BLOCKS = list(torch.split(o_max, Q_BLOCK_SIZE, dim=2))
            if len(flagList) >= 27 and flagList[26]:
                Qr_BLOCKS = list(torch.split(qr_tensor, Q_BLOCK_SIZE, dim=2))
                Kr_BLOCKS = list(torch.split(kr_tensor, K_BLOCK_SIZE, dim=2))

            for i in range(Tq):
                Qi = Q_BLOCKS[i]    # Q分块
                Sq_start = i * Q_BLOCK_SIZE
                Sq_end = min(Sq_start + Q_BLOCK_SIZE, Sq)
                deq_scale_q_i = deq_scale_q[:, :, Sq_start:Sq_end, :]  # (B, H, Q_BLOCK_SIZE, HEAD_DIM/32)
                # 扩展量化因子维度以匹配Q的最后一维（HEAD_DIM）
                # 方法：将(HEAD_DIM/32)维度重复32次，扩展到HEAD_DIM
                deq_scale_q_i_expanded = deq_scale_q_i.repeat_interleave(QK_SCALE_SIZE, dim=-1)  # (B, H, Q_BLOCK_SIZE, HEAD_DIM)
                ln_p_scale = torch.log(p_qscale_np)
                for j in range(0, Tkv, 2):
                    oi = o_BLOCKS[i]    # out
                    si = s_BLOCKS[i]    # sum
                    mi = m_BLOCKS[i]    # max

                    # 1. 获取当前 K 块并计算 S_ij
                    Kj = K_BLOCKS[j]
                    Sk_start = j * K_BLOCK_SIZE
                    Sk_end = min(Sk_start + K_BLOCK_SIZE, Skv)
                    # 截取当前K分块对应的量化因子
                    deq_scale_k_j = deq_scale_k[:, :, Sk_start:Sk_end, :]
                    deq_scale_k_j_expanded = deq_scale_k_j.repeat_interleave(QK_SCALE_SIZE, dim=-1)  # 扩展到HEAD_DIM
                    # print("start matmul1")
                    # print(f"[INFO] Qi {Qi.shape} deq_scale_q_i_expanded {deq_scale_q_i_expanded.shape} Kj {Kj.shape} deq_scale_k_j_expanded {deq_scale_k_j_expanded.shape}")
                    S_ij = torch.matmul(Qi.to(torch.float32)* deq_scale_q_i_expanded, (Kj.to(torch.float32)*deq_scale_k_j_expanded).permute(0, 1, 3, 2))
                    if len(flagList) >= 27 and flagList[26]:
                        Qri = Qr_BLOCKS[i]    # Q分块
                        Krj = Kr_BLOCKS[j]
                        S_ij += torch.matmul(Qri.to(torch.float32), Krj.to(torch.float32).permute(0, 1, 3, 2))
                    # print("end matmul1")
                    # print(f"=========C1-1: {S_ij}")

                    S_ij = S_ij * 1 / math.sqrt(d)
                    # if pfa_param['sparse'] == 3:
                    causal_mask_j = mask_global[:,:,Sq_start:Sq_end, Sk_start:Sk_end]
                        # print(f"[INFO] causal_mask_j shape: {causal_mask_j.shape} \n{causal_mask_j[:,:,:,:]}")

                    S_ij = S_ij.masked_fill(causal_mask_j, float('-inf'))
                        # print(f"[INFO] S_ij shape: {S_ij.shape} \n{S_ij[:,:,:,:]}")

                    # softmax计算（局部 max，暂不更新全局状态）
                    m_block_j, _ = torch.max(S_ij, dim=-1, keepdims=True)
                    m_block_j = m_block_j - ln_p_scale
                    m_block_j = torch.max(mi, m_block_j)
                    P_ij_raw = torch.exp(S_ij - m_block_j)
                    # torch.set_printoptions(profile="full", precision=4)
                    s_block_j = torch.sum(P_ij_raw, dim=-1, keepdims=True)
                    # print(f"[INFO] v1-1 max slice: \n{m_block_j}") # [11.3137]
                    # print(f"[INFO] v1-1 exp slice: \n{P_ij_raw}") # 1
                    # print(f"[INFO] v1-1 sum slice: \n{s_block_j}") # 256

                    if qDtype == "float8_e5m2" or qDtype == torch.float8_e5m2:
                        P_ij_drop = P_ij_raw.to(torch.float8_e5m2)    # softmax计算后cast成FP8
                        P_ij_drop = P_ij_raw.to(torch.float32)
                    elif qDtype == torch.uint8:
                        P_ij_drop_hifp8 = trans_np_float_tensor_to_hifuint8(P_ij_raw.numpy())
                        P_ij_drop_np32 = trans_np_hifuint8_tensor_to_float32(P_ij_drop_hifp8)
                        P_ij_drop = torch.from_numpy(P_ij_drop_np32)
                    else:
                        P_ij_drop = P_ij_raw.to(torch.float8_e4m3fn)
                        P_ij_drop = P_ij_drop.to(torch.float32)

                    if j + 1 < Tkv:
                        # --- C1V1 round 2: KV block j+1 ---
                        Kj1 = K_BLOCKS[j + 1]
                        # print("[INFO][SWL] ==============Kj1:", Kj1.shape, Kj1)
                        Sk1_start = (j + 1) * K_BLOCK_SIZE
                        Sk1_end = min(Sk1_start + K_BLOCK_SIZE, Skv)
                        # 截取当前K分块对应的量化因子
                        deq_scale_k_j1 = deq_scale_k[:, :, Sk1_start:Sk1_end, :]
                        deq_scale_k_j1_expanded = deq_scale_k_j1.repeat_interleave(QK_SCALE_SIZE, dim=-1)  # 扩展到HEAD_DIM
                        # print("start matmul1")
                        S_ij = torch.matmul(Qi.to(torch.float32)* deq_scale_q_i_expanded, (Kj1.to(torch.float32)*deq_scale_k_j1_expanded).permute(0, 1, 3, 2))
                        if len(flagList) >= 27 and flagList[26]:
                            Qri = Qr_BLOCKS[i]    # Q分块
                            Krj1 = Kr_BLOCKS[j+1]
                            S_ij += torch.matmul(Qri.to(torch.float32), Krj1.to(torch.float32).permute(0, 1, 3, 2))
                        # print(f"=========C1-2: {S_ij}")
                        # print("end matmul1")
                        S_ij1 = S_ij * 1 / math.sqrt(d)
                        # if pfa_param['sparse'] == 3:
                        causal_mask_j1 = mask_global[:,:,Sq_start:Sq_end, Sk1_start:Sk1_end]
                            # print(f"[INFO] causal_mask_j1 shape: {causal_mask_j1.shape} \n{causal_mask_j1[:,:,:,:]}")
                            # tensor_to_txt(S_ij1, "./fp8_input/S_ij1_bf.txt")
                        S_ij1 = S_ij1.masked_fill(causal_mask_j1, float('-inf'))
                            # tensor_to_txt(S_ij1, "./fp8_input/S_ij1_af.txt")
                            # print(f"[INFO] S_ij1 shape: {S_ij1.shape} \n{S_ij1[:,:,:,:]}")

                        # softmax计算（局部 max）
                        m_block_j1, _ = torch.max(S_ij1, dim=-1, keepdims=True)
                        m_block_j1 = m_block_j1 - ln_p_scale
                        m_block_j1 = torch.max(m_block_j, m_block_j1)
                        P_ij1_raw = torch.exp(S_ij1 - m_block_j1)
                        s_block_j1 = torch.sum(P_ij1_raw, dim=-1, keepdims=True)
                        update_mul_j = torch.exp(m_block_j - m_block_j1)
                        update_mul_si = torch.exp(mi - m_block_j1)
                        # print(f"[INFO] m_block_j shape: {m_block_j.shape}\n{m_block_j}")
                        # print(f"[INFO] m_block_j1 shape: {m_block_j1.shape}\n{m_block_j1}")
                        # print(f"[INFO] v1-2 max slice: \n{m_block_j1}") # [11.3137]
                        # print(f"[INFO] v1-2 exp slice: \n{P_ij1_raw}") # 1
                        # print(f"[INFO] v1-2 sum slice: \n{s_block_j1}") # 256

                        if qDtype == "float8_e5m2" or qDtype == torch.float8_e5m2:
                            P_ij1_drop = P_ij1_raw.to(torch.float8_e5m2)    # softmax计算后cast成FP8
                            P_ij1_drop = P_ij1_raw.to(torch.float32)
                        elif qDtype == torch.uint8:
                            P_ij1_drop_hifp8 = trans_np_float_tensor_to_hifuint8(P_ij1_raw.numpy())
                            P_ij1_drop_np32 = trans_np_hifuint8_tensor_to_float32(P_ij1_drop_hifp8)
                            P_ij1_drop = torch.from_numpy(P_ij1_drop_np32)
                        else:
                            P_ij1_drop = P_ij1_raw.to(torch.float8_e4m3fn)
                            P_ij1_drop = P_ij1_drop.to(torch.float32)

                            # --- C2V2 combined: 两块合并一次更新 ---
                        # 反量化V然后C2
                        Vj  = V_BLOCKS[int(j/2)]
                        # print("[INFO][SWL] ==============Vj: ", Vj.shape)
                        Sv_start = int(j/2)* V_BLOCK_SIZE
                        Sv_end = min(Sv_start + V_BLOCK_SIZE, Skv)
                        v_scale_start = Sv_start // V_SCALE_SIZE
                        v_scale_end = (Sv_end - 1) // V_SCALE_SIZE + 1
                        deq_scale_v_j = deq_scale_v[:, :, v_scale_start:v_scale_end, :]
                        deq_scale_v_j_tmp = deq_scale_v_j.repeat_interleave(V_SCALE_SIZE, dim=2)  # 扩展到V_BLOCK_SIZE
                        deq_scale_v_j_expanded = deq_scale_v_j_tmp[:, :, :Vj.shape[2], :]
                        Vj_dequant = Vj.to(torch.float32) * deq_scale_v_j_expanded

                        # print(f"[INFO] C2 Vj \n{Vj}")
                        # print(f"[INFO] C2 deq_scale_v_j_expanded \n{deq_scale_v_j_expanded}")
                        # print(f"[INFO] C2 Vj_dequant \n{Vj_dequant}")

                        split_k = True
                        if not split_k:
                            P_ij_drop_concact = torch.cat((P_ij_drop, P_ij1_drop), dim=-1)
                            P_ij_Vj   = torch.matmul(P_ij_drop_concact,  Vj_dequant)
                        else: # 切k矩阵乘，方便定位
                            V_part1 = Vj_dequant[:, :, :Kj.shape[2], :]
                            V_part2 = Vj_dequant[:, :, Kj.shape[2] : Kj.shape[2] + Kj1.shape[2], :]
                            # print(f"[INFO] V_part1[:,:,256:,:] shape: {V_part1.shape} \n{V_part1}")
                            # print(f"[INFO] V_part2[:,:,256:,:] shape: {V_part2.shape} \n{V_part2}")
                            P_V_part1 = torch.matmul(P_ij_drop, V_part1)
                            # print(f"[INFO] P_V_part1 shape: {P_V_part1.shape} \n{P_V_part1[:,:,:,:]}")
                            P_V_part2 = torch.matmul(P_ij1_drop, V_part2)
                            # print(f"[INFO] P_V_part2 shape: {P_V_part2.shape} \n{P_V_part2[:,:,:,:]}")
                            P_ij_Vj = P_V_part1 + P_V_part2

                        # print(f"P_ij_Vj shape: {P_ij_Vj.shape} \n{P_ij_Vj}")

                        #  V2
                        si_new = update_mul_si * si + s_block_j * update_mul_j + s_block_j1
                        # print(f"[INFO] update_mul_si: \n{update_mul_si}") # 0
                        # print(f"[INFO] si: \n{si}") # 0
                        # print(f"[INFO] s_block_j: \n{s_block_j}") # 256
                        # print(f"[INFO] update_mul_j: \n{update_mul_j}") # 1
                        # print(f"[INFO] s_block_j1: \n{s_block_j1}") # 256
                        # print(f"[INFO] row sum slice: \n{si_new}") # 512
                        o_BLOCKS[i] = (update_mul_si * oi * si + P_ij_Vj) / (si_new+ EPSILON)
                        s_BLOCKS[i] = si_new
                        m_BLOCKS[i] = m_block_j1

                    else:
                        # Tkv 为奇数时最后一块，正常处理

                        Vj = V_BLOCKS[int(j/2)]
                        Sv_start = j* K_BLOCK_SIZE
                        Sv_end = min(Sv_start + K_BLOCK_SIZE, Skv)

                        v_scale_start = Sv_start // V_SCALE_SIZE
                        v_scale_end = (Sv_end - 1) // V_SCALE_SIZE + 1
                        # print(f"[INFO] v_scale_start {v_scale_start} v_scale_end {v_scale_end} Sv_start {Sv_start} Sv_end {Sv_end}")

                        deq_scale_v_j = deq_scale_v[:, :, v_scale_start:v_scale_end, :]
                        deq_scale_v_j_tmp = deq_scale_v_j.repeat_interleave(V_SCALE_SIZE, dim=2)  # 扩展到V_BLOCK_SIZE
                        deq_scale_v_j_expanded = deq_scale_v_j_tmp[:, :, :Kj.shape[2], :]
                        Vj_expanded = Vj[:, :, :Kj.shape[2], :]
                        Vj_dequant = Vj_expanded.to(torch.float32) * deq_scale_v_j_expanded
                        # print("start matmul2")
                        P_ij_Vj = torch.matmul(P_ij_drop, Vj_dequant)
                        # print("end matmul2")
                        # 对bmm2结果反量化
                        # torch.set_printoptions(sci_mode=False)
                        update_mul_si = torch.exp(mi - m_block_j)
                        si_new = update_mul_si * si + s_block_j
                        o_BLOCKS[i] = (update_mul_si * oi * si + P_ij_Vj) / (si_new+ EPSILON)
                        s_BLOCKS[i] = si_new
                        m_BLOCKS[i] = m_block_j
            out = torch.cat(o_BLOCKS, dim=2)
            o_sum = torch.cat(s_BLOCKS, dim=2)
            o_max = torch.cat(m_BLOCKS, dim=2)
            y_all = out.numpy()
            # print(f"y_all slice : \n{y_all}")

    if inputLayout == "BSH":
        y_all = y_all.transpose(0, 2, 1, 3).reshape(out_shape)
    elif inputLayout == "NSD":
        y_all = y_all.reshape(out_shape)
    elif inputLayout == "BSND" or inputLayout == "BNSD_BSND":
        y_all = y_all.transpose(0, 2, 1, 3)
    elif inputLayout == "TND" or inputLayout == "NTD_TND" or inputLayout == "NTD":
        T = sum(actualSeqLengths)
        B = len(actualSeqLengths)
        if inputLayout == "TND":
            N = out_shape[1]
        else:
            N = out_shape[0]
        D = out_shape[2]
        if inputLayout == "TND" or inputLayout == "NTD_TND":
            output = np.zeros((T, N, D), dtype=y_all.dtype)
        else:
            output = np.zeros((N, T, D), dtype=y_all.dtype)
        t_start = 0
        for b_index in range(B):
            act_s = actualSeqLengths[b_index]
            t_end = t_start + act_s
            if act_s == 0:
                continue
            for n_index in range(N):
                if inputLayout == "TND" or inputLayout == "NTD_TND":
                    output[t_start:t_end, n_index, :] = y_all[b_index, n_index, :act_s, :]
                else:
                    output[n_index, t_start:t_end, :] = y_all[b_index, n_index, :act_s, :]
            t_start += act_s
        y_all = output
    elif len(out_shape) == 2:
        if q_bnsd_shape[0] == 1:
            y_all = y_all.transpose(0, 2, 1, 3).reshape(out_shape).astype(np.float32)
        else:
            sums = 0
            B = q_bnsd_shape[0]
            N = q_bnsd_shape[1]
            S = q_bnsd_shape[2]
            D = q_bnsd_shape[3]
            y_all = y_all.transpose(0, 2, 1, 3)
            yNewAll = np.zeros((S, N * D))
            for i in range(B):
                for j in range(actualSeqLengths[i]):
                    yNewAll[sums + j, :] = y_all[i:i + 1, j].reshape(1, N * D)
                sums += actualSeqLengths[i]
            y_all = yNewAll.astype(np.float32)
    else:
        y_all = y_all

    y_all = torch.from_numpy(y_all)

    if params["softmax_lse_flag"] == True:
        if inputLayout == "TND" or inputLayout == "NTD_TND" or inputLayout == "NTD":
            T = sum(actualSeqLengths)
            B = len(actualSeqLengths)
            if inputLayout == "TND":
                N = out_shape[1]
            else:
                N = out_shape[0]
            lse_output = np.zeros((T, N, 1), dtype=lse.dtype)
            t_start = 0
            for b_index in range(B):
                act_s = actualSeqLengths[b_index]
                t_end = t_start + act_s
                if act_s == 0:
                    continue
                for n_index in range(N):
                    lse_output[t_start:t_end, n_index, :] = lse[b_index, n_index, :act_s, :]
                t_start += act_s
            lse = lse_output
        lse = torch.from_numpy(lse.astype(np.float32))
        return y_all, lse

    # lse = np.zeros([1],dtype=np.float32)
    # # lse = np.zeros(pfa_param['lseshape'], dtype=np.float32)
    # lse = torch.from_numpy(lse.astype(np.float32))
    # return y_all,lse
    return y_all


def trans_tnd_actseq(list):
    list_len = len(list)
    list_new = []
    list_new.append(list[0])
    for i in range(list_len - 1):
        new_item = list[i + 1] - list[i]
        if new_item >= 0:
            list_new.append(new_item)
        else:
            print(f"[ERROR]trans_tnd_actseq: Wrong input actseq:{list}, in loop {i}, item {new_item} < 0")
    print(f"[INFO] trans_tnd_actseq before trans: {list} -> {list_new}")
    # print(f"[INFO]before trans: {list}")
    # print(f"[INFO]after trans: {list_new}")
    return list_new


def get_mxfp8_per_token_group_quant_scale(tensor, data_type, GROUP_SIZE=32) -> torch.Tensor:
    """
    MXFP8量化逻辑：scale = 2^shared_exp → shared_exp = floor(log2(scale)) - EMAX_ELEM(8)
    如果一组内所有元素都是 0 → scale = 1
    """
    GTYPE = torch.float32
    if data_type == torch.float8_e5m2 or data_type == torch.float8_e4m3fn or data_type == torch.uint8:
        EMAX_ELEM = 8
    elif data_type == torch.float16 or data_type == torch.bfloat16:
        EMAX_ELEM = 16
    elif data_type == torch.float32:
        EMAX_ELEM = 32
    else:
        EMAX_ELEM = -1
        print(f"{data_type} Not support mxfp8 quant")
        return None

    dim1, dim2, dim3, dim4 = tensor.shape
    scale_for_tensor = torch.ones([dim1, dim2, dim3, math.ceil(dim4 / GROUP_SIZE)]).to(GTYPE)

    for b in range(dim1):
        for n in range(dim2):
            for s in range(dim3):
                for d in range(0, dim4, GROUP_SIZE):
                    d_start = d
                    d_end = min(d + GROUP_SIZE, dim4)
                    chunk = tensor[b, n, s, d_start:d_end]
                    if torch.all(chunk == 0):
                        scale_for_tensor[b, n, s, d // GROUP_SIZE] = 1.0
                        continue
                    max_val = torch.max(torch.abs(chunk)).clamp(min=1e-12)
                    shared_exp = torch.floor(torch.log2(max_val)) - EMAX_ELEM
                    scale_for_tensor[b, n, s, d // GROUP_SIZE] = 2**shared_exp

    return scale_for_tensor


def get_mxfp8_per_channel_group_quant_scale(tensor, data_type, GROUP_SIZE=32) -> torch.Tensor:
    """
    MXFP8量化逻辑：scale = 2^shared_exp → shared_exp = floor(log2(scale)) - EMAX_ELEM(8)
    如果一组内所有元素都是 0 → scale = 1
    """
    GTYPE = torch.float32
    if data_type == torch.float8_e5m2 or data_type == torch.float8_e4m3fn or data_type == torch.uint8:
        EMAX_ELEM = 8
    elif data_type == torch.float16 or data_type == torch.bfloat16:
        EMAX_ELEM = 16
    elif data_type == torch.float32:
        EMAX_ELEM = 32
    else:
        EMAX_ELEM = -1
        print(f"{data_type} Not support mxfp8 quant")
        return None

    dim1, dim2, dim3, dim4 = tensor.shape
    scale_for_tensor = torch.ones([dim1, dim2, math.ceil(dim3 / GROUP_SIZE), dim4]).to(GTYPE)

    for b in range(dim1):
        for n in range(dim2):
            for d in range(dim4):
                for s in range(0, dim3, GROUP_SIZE):
                    s_start = s
                    s_end = min(s + GROUP_SIZE, dim3)
                    chunk = tensor[b, n, s_start:s_end, d]
                    if torch.all(chunk == 0):
                        scale_for_tensor[b, n, s // GROUP_SIZE, d] = 1.0
                        continue
                    max_val = torch.max(torch.abs(chunk)).clamp(min=1e-12)
                    shared_exp = torch.floor(torch.log2(max_val)) - EMAX_ELEM
                    scale_for_tensor[b, n, s // GROUP_SIZE, d] = 2**shared_exp

    return scale_for_tensor
def get_tensor_quant_scale(tensor, data_type):
    if data_type == torch.float8_e5m2:
        FP8_MAX = 57344.0
    elif data_type == torch.float8_e4m3fn:
        FP8_MAX = 448.0
    elif data_type == torch.uint8:
        FP8_MAX = 32768
    else:
        FP8_MAX = -1
        return None
    max_val = torch.max(torch.abs(tensor))
    epsilon = 1e-8
    tensor_quant_scale = FP8_MAX / (max_val + epsilon)
    return tensor_quant_scale

def get_block_quant_scale(tensor, data_type, is_scale_ds=False, is_vcore_split=False, NEW_PER_BLOCK_SIZE=128):
    PER_VCORE_BLOCK_SIZE = 64
    GTYPE = torch.float32
    if data_type == torch.float8_e5m2:
        FP8_MAX = 57344.0
    elif data_type == torch.float8_e4m3fn:
        FP8_MAX = 448.0
    elif data_type == torch.uint8:
        FP8_MAX = 32768
    else:
        FP8_MAX = -1
        print(f"{data_type} Not support block quant")
        return None

    dim1 = tensor.shape[0]  # b
    dim2 = tensor.shape[1]  # n2
    dim4 = tensor.shape[2]  # s1
    dim5 = tensor.shape[3]  # d or s2
    if not is_scale_ds:
        print("fff quant not is_scale_ds")
        scale_for_tensor = torch.ones([dim1, dim2, math.ceil(dim4 / NEW_PER_BLOCK_SIZE), 1]).to(GTYPE)
        for b in range(dim1):
            for n in range(dim2):
                for s in range(0, dim4, NEW_PER_BLOCK_SIZE):
                    # 取出128*D个数
                    s_start = s // NEW_PER_BLOCK_SIZE * NEW_PER_BLOCK_SIZE
                    s_end = min(s // NEW_PER_BLOCK_SIZE * NEW_PER_BLOCK_SIZE + NEW_PER_BLOCK_SIZE, dim4)
                    chunk = tensor[b, n, s_start:s_end, :]
                    max_val = torch.max(torch.abs(chunk))
                    epsilon = 1e-8
                    scale_for_tensor[b, n, s // NEW_PER_BLOCK_SIZE, 0] = FP8_MAX / (max_val + epsilon)
    else:
        if not is_vcore_split:
            if dim4 % PER_VCORE_BLOCK_SIZE == 0:
                scale_for_tensor = torch.ones(
                    [dim1, dim2, math.ceil(dim4 / PER_VCORE_BLOCK_SIZE), math.ceil(dim5 / NEW_PER_BLOCK_SIZE)]).to(GTYPE)
            else:
                scale_for_tensor = torch.ones(
                    [dim1, dim2, math.ceil(dim4 / PER_VCORE_BLOCK_SIZE) + 1, math.ceil(dim5 / NEW_PER_BLOCK_SIZE)]).to(
                    GTYPE)
            s1_tail = NEW_PER_BLOCK_SIZE if dim4 % NEW_PER_BLOCK_SIZE == 0 else dim4 % NEW_PER_BLOCK_SIZE
            for b in range(dim1):
                for n in range(dim2):
                    for s2 in range(0, dim5, NEW_PER_BLOCK_SIZE):
                        s2_start = s2 // NEW_PER_BLOCK_SIZE * NEW_PER_BLOCK_SIZE
                        s2_end = min(s2 // NEW_PER_BLOCK_SIZE * NEW_PER_BLOCK_SIZE + NEW_PER_BLOCK_SIZE, dim5)
                        for s1 in range(0, dim4 - s1_tail, PER_VCORE_BLOCK_SIZE):
                            s1_start = s1 // PER_VCORE_BLOCK_SIZE * PER_VCORE_BLOCK_SIZE
                            s1_end = min(s1_start + PER_VCORE_BLOCK_SIZE, dim4)
                            chunk = tensor[b, n, s1_start:s1_end, s2_start:s2_end]
                            max_val = torch.max(torch.abs(chunk))
                            scale_for_tensor[b, n, s1 // PER_VCORE_BLOCK_SIZE, s2 // NEW_PER_BLOCK_SIZE] = FP8_MAX / max_val

                        # 处理尾块 v0
                        s1_start = dim4 - s1_tail
                        s1_end = s1_start + math.ceil(s1_tail / 2)
                        chunk = tensor[b, n, s1_start:s1_end, s2_start:s2_end]
                        max_val = torch.max(torch.abs(chunk))
                        scale_for_tensor[
                            b, n, s1_start // PER_VCORE_BLOCK_SIZE, s2 // NEW_PER_BLOCK_SIZE] = FP8_MAX / max_val
                        v0_s1_start_idx = s1_start // PER_VCORE_BLOCK_SIZE
                        # 处理尾块 v1
                        s1_start = s1_end
                        s1_end = dim4
                        chunk = tensor[b, n, s1_start:s1_end, s2_start:s2_end]
                        max_val = torch.max(torch.abs(chunk))
                        scale_for_tensor[b, n, v0_s1_start_idx + 1, s2 // NEW_PER_BLOCK_SIZE] = FP8_MAX / max_val
        else:
            # S1方向区分V0与V1，shape=S1*D
            if dim4 % PER_VCORE_BLOCK_SIZE == 0:
                scale_for_tensor = torch.ones([dim1, dim2, math.ceil(dim4 / PER_VCORE_BLOCK_SIZE), 1]).to(GTYPE)
            else:
                scale_for_tensor = torch.ones([dim1, dim2, math.ceil(dim4 / PER_VCORE_BLOCK_SIZE) + 1, 1]).to(GTYPE)
            for b in range(dim1):
                for n in range(dim2):
                    for s in range(0, dim4, PER_VCORE_BLOCK_SIZE):
                        # 取出64*D个数
                        s_start = s // PER_VCORE_BLOCK_SIZE * PER_VCORE_BLOCK_SIZE
                        s_end = min(s_start + PER_VCORE_BLOCK_SIZE, dim4)
                        if s_end - start == PER_VCORE_BLOCK_SIZE:
                            chunk = tensor[b, n, s_start:s_end, :]
                            max_val = torch.max(torch.abs(chunk))
                            scale_for_tensor[b, n, s // PER_VCORE_BLOCK_SIZE, 0] = FP8_MAX / max_val
                        else:
                            s_end = s_start + math.ceil(s_end - s_start) / 2
                            chunk = tensor[b, n, s_start:s_end, :]
                            max_val = torch.max(torch.abs(chunk))
                            scale_for_tensor[b, n, s // PER_VCORE_BLOCK_SIZE, 0] = FP8_MAX / max_val
                            s_start = s_end
                            s_end = dim4
                            chunk = tensor[b, n, s_start:s_end, :]
                            max_val = torch.max(torch.abs(chunk))
                            scale_for_tensor[b, n, s // PER_VCORE_BLOCK_SIZE + 1, 0] = FP8_MAX / max_val
    return scale_for_tensor

def mxfp8_per_token_group_quant_or_dequant_single(tensor, scale, GROUP_SIZE=32):
    GTYPE = torch.float32
    dim1 = tensor.shape[0]  # b
    dim2 = tensor.shape[1]  # n2
    dim3 = tensor.shape[2]  # s1
    dim4 = tensor.shape[3]  # d or s2
    quanted_tensor = torch.zeros([dim1,dim2,dim3,dim4]).to(GTYPE)

    for b in range(dim1):
        for n in range(dim2):
            for s in range(dim3):
                for d in range(0, dim4, GROUP_SIZE):
                    d_start = d
                    d_end = min(d + GROUP_SIZE, dim4)
                    quanted_tensor[b, n, s ,d_start: d_end] = tensor[b, n, s ,d_start: d_end] * scale[
                        b, n, s, d // GROUP_SIZE]

    return quanted_tensor

def mxfp8_per_channel_group_quant_or_dequant_single(tensor, scale, GROUP_SIZE=32):
    GTYPE = torch.float32
    dim1 = tensor.shape[0]  # b
    dim2 = tensor.shape[1]  # n2
    dim3 = tensor.shape[2]  # s1
    dim4 = tensor.shape[3]  # d or s2
    quanted_tensor = torch.zeros([dim1,dim2,dim3,dim4]).to(GTYPE)

    for b in range(dim1):
        for n in range(dim2):
            for d in range(dim4):
                for s in range(0, dim3, GROUP_SIZE):
                    s_start = s
                    s_end = min(s + GROUP_SIZE, dim3)
                    quanted_tensor[b, n, s_start: s_end, d] = tensor[b, n, s_start: s_end, d] * scale[
                        b, n, s // GROUP_SIZE, d]

    return quanted_tensor
def block_quant_or_dequant_single(tensor, scale, is_vcore_split=False, NEW_PER_BLOCK_SIZE=128):
    PER_VCORE_BLOCK_SIZE = 64
    GTYPE = torch.float32
    dim1 = tensor.shape[0]  # b
    dim2 = tensor.shape[1]  # n2
    dim3 = tensor.shape[2]  # s1
    dim4 = tensor.shape[3]  # d or s2
    quanted_tensor = torch.zeros([dim1, dim2, dim3, dim4]).to(GTYPE)
    if not is_vcore_split:
        for b in range(dim1):
            for n in range(dim2):
                for s in range(0, dim3, NEW_PER_BLOCK_SIZE):
                    # 取出128*D个数
                    s_start = s // NEW_PER_BLOCK_SIZE * NEW_PER_BLOCK_SIZE
                    s_end = min(s // NEW_PER_BLOCK_SIZE * NEW_PER_BLOCK_SIZE + NEW_PER_BLOCK_SIZE, dim3)

                    quanted_tensor[b, n, s_start: s_end, :] = tensor[b, n, s_start: s_end, :] * scale[
                        b, n, s // NEW_PER_BLOCK_SIZE, 0]
    else:
        s1_tail = NEW_PER_BLOCK_SIZE if dim3 % NEW_PER_BLOCK_SIZE == 0 else dim3 % NEW_PER_BLOCK_SIZE
        for b in range(dim1):
            for n in range(dim2):
                for s2 in range(0, dim4, NEW_PER_BLOCK_SIZE):
                    s2_start = s2 // NEW_PER_BLOCK_SIZE * NEW_PER_BLOCK_SIZE
                    s2_end = min(s2_start + NEW_PER_BLOCK_SIZE, dim4)
                    for s1 in range(0, dim3 - s1_tail, PER_VCORE_BLOCK_SIZE):
                        # 取出64*128个数
                        s1_start = s1 // PER_VCORE_BLOCK_SIZE * PER_VCORE_BLOCK_SIZE
                        s1_end = min(s1_start + PER_VCORE_BLOCK_SIZE, dim3)
                        quanted_tensor[b, n, s1_start: s1_end, s2_start: s2_end] = tensor[b, n, s1_start: s1_end,
                                                                                   s2_start: s2_end] * scale[
                                                                                       b, n, s1 // PER_VCORE_BLOCK_SIZE, s2 // NEW_PER_BLOCK_SIZE]
                    # 处理尾块 v0
                    s1_start = dim3 - s1_tail
                    s1_end = s1_start + math.ceil(s1_tail / 2)
                    quanted_tensor[b, n, s1_start: s1_end, s2_start: s2_end] = tensor[b, n, s1_start: s1_end,
                                                                               s2_start: s2_end] * scale[
                                                                                   b, n, s1_start // PER_VCORE_BLOCK_SIZE, s2 // NEW_PER_BLOCK_SIZE]
                    v0_s1_start_idx = s1_start // PER_VCORE_BLOCK_SIZE
                    # 处理尾块 v1
                    s1_start = s1_end
                    s1_end = dim3
                    quanted_tensor[b, n, s1_start: s1_end, s2_start: s2_end] = tensor[b, n, s1_start: s1_end,
                                                                               s2_start: s2_end] * scale[
                                                                                   b, n, v0_s1_start_idx + 1, s2 // NEW_PER_BLOCK_SIZE]

    return quanted_tensor

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


def convert_q_bnsd_to_layout(tensor_bnsd, seq_lens, inputLayout, originShape):
    B, N, S, D = tensor_bnsd.shape
    orgS = int(originShape[2])
    tensor_np = tensor_bnsd[:, :, :orgS, :]

    if inputLayout == "BNSD":
        return tensor_np, tensor_bnsd
    elif inputLayout == "BSND":
        return tensor_np.transpose(0, 2, 1, 3), tensor_bnsd
    elif inputLayout == "BSH":
        return tensor_np.transpose(0, 2, 1, 3).reshape(B, orgS, N * D), tensor_bnsd
    elif inputLayout == "TND":
        T = sum(seq_lens)
        tensor_np = np.zeros((T, N, D), dtype=np.float32)
        t_start = 0
        for b_index in range(len(seq_lens)):
            act_s = seq_lens[b_index]
            t_end = t_start + act_s
            if act_s <= 0:
                continue
            for n_index in range(N):
                tensor_np[t_start:t_end, n_index, :] = tensor_bnsd[b_index, n_index, :act_s, :]
                tensor_bnsd[b_index, n_index, act_s:, :] = 0
            t_start = t_end
        return tensor_np, tensor_bnsd
    else:
        raise ValueError(f"Unsupported inputLayout: {inputLayout}")

def convert_kv_bnsd_to_layout(tensor_bnsd, seq_lens, inputLayout, originShape, params):
    B, N, S, D = tensor_bnsd.shape
    orgS = int(originShape[2])
    print("orgS++++++++++++++++++",orgS)
    tensor_np = tensor_bnsd[:, :, :orgS, :]
    print("tensor_np++++++++++++++++++",tensor_np.shape)
    flagList = params['flaglist']
    print("seq_lens++++++++++++++++++",seq_lens)
    if inputLayout == "BNSD":
        return tensor_np, tensor_bnsd
    elif inputLayout == "BSND" or flagList[14] != 0:
        return tensor_np.transpose(0, 2, 1, 3), tensor_bnsd
    elif inputLayout == "BSH":
        return tensor_np.transpose(0, 2, 1, 3).reshape(B, orgS, N * D), tensor_bnsd
    elif inputLayout == "TND":
        T = sum(seq_lens)
        tensor_np = np.zeros((T, N, D), dtype=np.float32)
        t_start = 0
        for b_index in range(len(seq_lens)):
            act_s = seq_lens[b_index]
            t_end = t_start + act_s
            if act_s <= 0:
                continue
            for n_index in range(N):
                tensor_np[t_start:t_end, n_index, :] = tensor_bnsd[b_index, n_index, :act_s, :]
                tensor_bnsd[b_index, n_index, act_s:, :] = 0
            t_start = t_end
        return tensor_np, tensor_bnsd
    else:
        raise ValueError(f"Unsupported inputLayout: {inputLayout}")

def convert_q_scale_tnd_to_n2gtd_layout(tensor_tnd, numKeyValueHeads):
    T, N, D, _ = tensor_tnd.shape
    tensor_reshape = tensor_tnd.reshape(T, int(numKeyValueHeads), int( N / numKeyValueHeads), D, 2)
    return tensor_reshape.transpose(1, 0, 2, 3, 4)

def convert_q_scale_bnsd_to_layout(tensor_bnsd, act_seq_lens, inputLayout, originShape):
    B, N, S, D = tensor_bnsd.shape
    orgS = int(originShape[2])
    if inputLayout == "BNSD":
        # [B, N, S, D] -> [B, N, S, D//2, 2]
        return tensor_bnsd[:, :, :orgS, :].reshape(B, N, orgS, D // 2, 2).astype(np.float32)
    elif inputLayout == "BSND":
        # [B, N, S, D] -> [B, S, N, D//2, 2]
        transposed = tensor_bnsd[:, :, :orgS, :].transpose(0, 2, 1, 3)
        return transposed.reshape(B, orgS, N, D // 2, 2).astype(np.float32)
    elif inputLayout == "BSH":
        # [B, N, S, D] -> [B, S, N*D//2, 2]
        transposed = tensor_bnsd[:, :, :orgS, :].transpose(0, 2, 1, 3).reshape(B, S, N * D)
        return transposed.reshape(B, orgS, (N * D) // 2, 2).astype(np.float32)
    elif inputLayout == "TND":
        T = sum(act_seq_lens)
        tensor_np = np.zeros((T, N, D // 2, 2), dtype=np.float32)
        output = np.zeros((T, N, D), dtype=np.float32)
        t_start = 0
        for b_index in range(B):
            act_s = act_seq_lens[b_index]
            t_end = t_start + act_s
            if act_s == 0:
                continue
            for n_index in range(N):
                output[t_start:t_end, n_index, :] = tensor_bnsd[b_index, n_index, :act_s, :]
            t_start += act_s
        tensor_np = output.reshape(T, N, D // 2, 2)
        return tensor_np
    else:
        raise ValueError(f"Unsupported layout: {inputLayout}")


def convert_k_scale_bnsd_to_layout(tensor_bnsd, act_seq_lens, inputLayout, originShape, params):
    B, N, S, D = tensor_bnsd.shape
    flagList = params['flaglist']
    orgS = int(originShape[2])
    if inputLayout == "BNSD" or flagList[14] != 0:
        # [B, N, S, D] -> [B, N, S, D//2, 2]
        return tensor_bnsd[:, :, :orgS, :].reshape(B, N, orgS, D // 2, 2).astype(np.float32)
    elif inputLayout == "BSND":
        # [B, N, S, D] -> [B, S, N, D//2, 2]
        transposed = tensor_bnsd[:, :, :orgS, :].transpose(0, 2, 1, 3)
        return transposed.reshape(B, orgS, N, D // 2, 2).astype(np.float32)
    elif inputLayout == "BSH":
        # [B, N, S, D] -> [B, S, N*D//2, 2]
        transposed = tensor_bnsd[:, :, :orgS, :].transpose(0, 2, 1, 3).reshape(B, orgS, N * D)
        return transposed.reshape(B, orgS, (N * D) // 2, 2).astype(np.float32)
    elif inputLayout == "TND":
        T = sum(act_seq_lens)
        tensor_np = np.zeros((T, N, D // 2, 2), dtype=np.float32)
        output = np.zeros((T, N, D), dtype=np.float32)
        t_start = 0
        for b_index in range(B):
            act_s = act_seq_lens[b_index]
            t_end = t_start + act_s
            if act_s == 0:
                continue
            for n_index in range(N):
                output[t_start:t_end, n_index, :] = tensor_bnsd[b_index, n_index, :act_s, :]
            t_start += act_s
        tensor_np = output.reshape(T, N, D // 2, 2)
        return tensor_np

    else:
        raise ValueError(f"Unsupported layout: {inputLayout}")

def convert_v_scale_bnsd_to_layout(tensor_bnsd, act_seq_lens, inputLayout, originShape, params, group_size=32):
    B, N, S, DV = tensor_bnsd.shape
    flagList = params['flaglist']
    orgS = int(originShape[2])
    
    if inputLayout == "BNSD" or flagList[14] != 0:
        transposed = tensor_bnsd[:, :, :math.ceil(orgS / (group_size)), :]
        Sg = math.ceil(orgS / group_size)  # 压缩后的行数
        
        if Sg % 2 != 0:
            pad = np.zeros((B, N, 1, DV), dtype=np.float32)
            transposed = np.concatenate([transposed, pad], axis=2)
            Sg += 1
        
        S_out = Sg // 2  # 现在 Sg 是偶数，S_out 正确
        tensor_np = np.zeros((B, N, S_out, DV, 2), dtype=np.float32)
        tensor_np[..., 0] = transposed[..., ::2, :]   # 偶数位
        tensor_np[..., 1] = transposed[..., 1::2, :]  # 奇数位
        return tensor_np
        
    elif inputLayout == "BSND":
        transposed = tensor_bnsd[:, :, :math.ceil(orgS / (group_size)), :].transpose(0, 2, 1, 3)
        Sg = math.ceil(orgS / group_size)
        
        if Sg % 2 != 0:
            pad = np.zeros((B, 1, N, DV), dtype=np.float32)
            transposed = np.concatenate([transposed, pad], axis=1)
            Sg += 1
        
        S_out = Sg // 2
        tensor_np = np.zeros((B, S_out, N, DV, 2), dtype=np.float32)
        tensor_np[..., 0] = transposed[:, ::2, :, :]
        tensor_np[..., 1] = transposed[:, 1::2, :, :]
        return tensor_np
        
    elif inputLayout == "BSH":
        transposed = tensor_bnsd[:, :, :math.ceil(orgS / (group_size)), :].transpose(0, 2, 1, 3).reshape(B, S, N * DV)
        Sg = math.ceil(orgS / group_size)
        
        if Sg % 2 != 0:
            pad = np.zeros((B, 1, N * DV), dtype=np.float32)
            transposed = np.concatenate([transposed, pad], axis=1)
            Sg += 1
        
        S_out = Sg // 2
        tensor_np = np.zeros((B, S_out, N * DV, 2), dtype=np.float32)
        tensor_np[..., 0] = transposed[:, ::2, :]
        tensor_np[..., 1] = transposed[:, 1::2, :]
        return tensor_np
        
    elif inputLayout == "TND":

        # 计算 V scale 的总长度
        Tv = sum(math.ceil(seq_len / (group_size * 2)) for seq_len in act_seq_lens)
        tensor_np = np.zeros((Tv, N, DV, 2), dtype=np.float32)
        t_start = 0
        for b in range(B):
            orig_seq = act_seq_lens[b]
            src_len = math.ceil(orig_seq / group_size)
            # =====  奇数行 pad 后再计算 act_s =====
            if src_len % 2 != 0:
                src_len += 1  # pad 后行数
            
            act_s = src_len // 2  # 正确的目标长度
            t_end = t_start + act_s
            if act_s <= 0:
                continue
            for n in range(N):
                src = tensor_bnsd[b, n, :math.ceil(orig_seq / group_size), :]  # 取原始数据
                
                # ===== src 奇数行 pad 到偶数 =====
                actual_src_len = src.shape[0]
                if actual_src_len % 2 != 0:
                    pad = np.ones((1, DV), dtype=np.float32)
                    src = np.concatenate([src, pad], axis=0)
                
                out_len = src.shape[0] // 2  # pad 后的正确长度
                tensor_np[t_start : t_start + out_len, n, :, 0] = src[::2, :]    # 偶数索引
                tensor_np[t_start : t_start + out_len, n, :, 1] = src[1::2, :]   # 奇数索引
            t_start = t_end
        return tensor_np
        
    else:
        raise ValueError(f"Unsupported layout: {inputLayout}")