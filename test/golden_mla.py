# -*- coding: utf-8 -*-
# Copyright 2020 Huawei Technologies Co., Ltd
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
import math
import libs.tools as tools
import tensorflow as tf
import copy
from libs.generate_float_data import new_trans_np_fp4_e1m2_tensor_to_bfloat16, new_trans_np_fp4_e2m1_tensor_to_bfloat16
from libs.generate_hifloat8_data import trans_np_hifuint8_tensor_to_float32

cfg_fk = tools.ConfigFmk()


def _create_mask(m_shape, pre_tokens, next_tokens):
    next_masks = np.triu(np.ones(m_shape, dtype='uint8'), k=1 + int(next_tokens))  # 生成下三角全是0的矩阵
    pre_mask = np.tril(np.ones(m_shape, dtype='uint8'), k=-1 - int(pre_tokens))  # 生成上三角全是0的矩阵
    atten_masks = pre_mask + next_masks

    return atten_masks


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


def _create_mask_right_down(m_shape, pre_tokens, next_tokens, actualSeqLengths, actualSeqLengthsKV, actualprefixKV,
                            prefix_kvs, batch, numheads, kvs_list, m_dtype):
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
            S2 = actualSeqLengthsKV[i] + actualprefixKV
        elif len(kvs_list) > 1:
            S2 = kvs_list[i] + actualprefixKV
        else:
            S2 = mask_s_kv - prefix_kvs + actualprefixKV
        next_tokens = S2 - S1
        next_tokens_list.append(next_tokens)
        atten_masks = _create_mask(m_shape, pre_tokens, next_tokens)
        re_mask_batch.append(atten_masks)
    return re_mask_batch, next_tokens_list


def _create_random_mask_by_spars(cpu_m_shape, npu_m_shape, m_dtype, pre_tokens, next_tokens, actualSeqLengths,
                                 actualSeqLengthsKV, actualprefixKV, prefix_kvs, kvs_list, batch=1, numheads=1,
                                 sp_mode=0, random_ones=0):
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
                                                            actualSeqLengthsKV, actualprefixKV, prefix_kvs, batch,
                                                            numheads, kvs_list, m_dtype)
        return cpu_mask, npu_mask, pre_tokens, next_tokens_new
    if sp_mode == 4:
        npu_mask = np.triu(np.ones(npu_m_shape), k=1)
        cpu_mask, pre_tokens_new, next_tokens_new = _create_mask_band(cpu_m_shape, pre_tokens, next_tokens,
                                                                      actualSeqLengths, actualSeqLengthsKV,
                                                                      actualprefixKV, prefix_kvs, batch,
                                                                      numheads, kvs_list, m_dtype)
        return cpu_mask, npu_mask.astype(m_dtype), pre_tokens_new, next_tokens_new


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


def dequant(x, deqscale, relu_weight):
    deqscale = np.uint64(deqscale)
    deqscale = np.frombuffer(deqscale, dtype=np.float32)
    deqscale = deqscale[: 1][0]
    # print(f"deqscale:{deqscale}")
    if relu_weight is None:
        relu_weight = deqscale
    else:
        relu_weight = np.full(deqscale.shape, relu_weight)
    res_cal = np.zeros(x.shape, np.float16)
    for n in range(x.shape[0]):
        for c in range(x.shape[1]):
            for h in range(x.shape[2]):
                for w in range(x.shape[3]):
                    x[n, c, h, w] = x[n, c, h, w].astype(np.float32)
                    if x[n, c, h, w] >= 0:
                        res_cal[n, c, h, w] = x[n, c, h, w] * deqscale
                    else:
                        res_cal[n, c, h, w] = x[n, c, h, w] * relu_weight
    return res_cal


def s8_saturation(inputdata):
    if inputdata > 127:
        inputdata = 127
    elif inputdata < -128:
        inputdata = -128
    outdata = np.int8(inputdata)
    return outdata


def s9_saturation(inputdata):
    if inputdata > 255:
        inputdata = 255
    elif inputdata < -256:
        inputdata = -256
    return inputdata


def quant(x, qscale, qoffset):
    s8_res_cal = np.zeros(x.shape, np.int8)
    for n in range(x.shape[0]):
        for c in range(x.shape[1]):
            for h in range(x.shape[2]):
                for w in range(x.shape[3]):
                    s8_res_cal[n, c, h, w] = s8_saturation(
                        np.round(s9_saturation(np.half(x[n, c, h, w]) * np.half(qscale)) + np.half(qoffset)))
    return s8_res_cal


def quant_pc(x, qscale, qoffset):
    print(f"qscale:{qscale.shape}")
    print(f"x:{x.shape}")
    s8_res_cal = np.zeros(x.shape, np.int8)
    for n in range(x.shape[0]):
        for c in range(x.shape[1]):
            for h in range(x.shape[2]):
                for w in range(x.shape[3]):
                    s8_res_cal[n, c, h, w] = s8_saturation(
                        np.round(s9_saturation(np.half(x[n, c, h, w]) * np.half(qscale[0, c, 0, w])) + np.half(
                            qoffset[0, c, 0, w])))
    return s8_res_cal


def ieee_754_conversion(sign, exponent_raw, mantissa, exp_len=8, mant_len=7):
    """ Convert binary data into the floating point value """
    sign_mult = -1 if sign == 1 else 1
    exponent = exponent_raw - (2 ** (exp_len - 1) - 1)
    mant_mult = 1
    for b in range(mant_len - 1, -1, -1):
        if mantissa & (2 ** b):
            mant_mult += 1 / (2 ** (mant_len - b))

    return sign_mult * (2 ** exponent) * mant_mult


def trans_tensor_fp8_e8m0_to_bf16(tensor):
    # import pdb;pdb.set_trace()
    from ml_dtypes import bfloat16
    # 创建一个与输入tensor形状相同的新tensor
    new_tensor = np.zeros_like(tensor).astype(bfloat16)
    # 使用nditer遍历tensor中的每个元素
    with np.nditer(tensor, flags=['multi_index'], op_flags=['readwrite']) as it:
        for x in it:
            # 将元素传递给trans函数进行处理
            # new_value = new_value.astype(bfloat16)

            # x = float(x)
            # new_value = 2 ** (x - 127)
            new_value = ieee_754_conversion(0, int(x), 0)
            # 将处理后的结果回填到新tensor中
            new_tensor[it.multi_index] = new_value
    return new_tensor


def antiquant(k, v, scale, offset, pto_flag, lowprecision_flag, kv_dtype="int8", q_dtype="float16"):
    N = k.shape[1]
    S = v.shape[2]
    if kv_dtype in ["fp4_e1m2", "fp4_e2m1", "float4_e1m2", "float4_e2m1"]:
        if lowprecision_flag:
            if q_dtype == "float16":
                scale_bf16 = trans_tensor_fp8_e8m0_to_bf16(scale)
                scale = scale_bf16.astype(np.float16)
            elif q_dtype == "bfloat16":
                scale_bf16 = trans_tensor_fp8_e8m0_to_bf16(scale)
                scale = scale_bf16
            else:
                print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                exit(1)
        else:
            scale_bf16 = trans_tensor_fp8_e8m0_to_bf16(scale)
            scale = scale_bf16.astype(np.float16)
        scale1 = scale[0]
        scale2 = scale[1]
        if kv_dtype in ["fp4_e1m2", "float4_e1m2"]:
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = new_trans_np_fp4_e1m2_tensor_to_bfloat16(k).astype(np.float16)
                    value = new_trans_np_fp4_e1m2_tensor_to_bfloat16(v).astype(np.float16)
                elif q_dtype == "bfloat16":
                    key = new_trans_np_fp4_e1m2_tensor_to_bfloat16(k)
                    value = new_trans_np_fp4_e1m2_tensor_to_bfloat16(v)
                else:
                    print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                    exit(1)
            else:
                key = new_trans_np_fp4_e1m2_tensor_to_bfloat16(k).astype(np.float16)
                value = new_trans_np_fp4_e1m2_tensor_to_bfloat16(v).astype(np.float16)
        else:
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = new_trans_np_fp4_e2m1_tensor_to_bfloat16(k).astype(np.float16)
                    value = new_trans_np_fp4_e2m1_tensor_to_bfloat16(v).astype(np.float16)
                elif q_dtype == "bfloat16":
                    key = new_trans_np_fp4_e2m1_tensor_to_bfloat16(k)
                    value = new_trans_np_fp4_e2m1_tensor_to_bfloat16(v)
                else:
                    print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                    exit(1)
            else:
                key = new_trans_np_fp4_e2m1_tensor_to_bfloat16(k).astype(np.float16)
                value = new_trans_np_fp4_e2m1_tensor_to_bfloat16(v).astype(np.float16)
        g = scale.shape[4]
        # print(scale.shape)
        # print(g)
        grp_size = 32
        for nIdx in range(N):
            for sIdx in range(S):
                for gIdx in range(g):
                    # copy from
                    key_cur = key[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1), gIdx * grp_size:(gIdx + 1) * grp_size]
                    # broardcast
                    scale_k = scale1[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1), gIdx:(gIdx + 1)]
                    scale_k = np.full((1, grp_size), scale_k)
                    # mul
                    key_cur = key_cur * scale_k
                    # copy to
                    key[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1),
                    gIdx * grp_size:(gIdx + 1) * grp_size] = key_cur

                for gIdx in range(g):
                    # copy from
                    value_cur = value[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1), gIdx * grp_size:(gIdx + 1) * grp_size]
                    # broardcast
                    scale_v = scale2[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1), gIdx:(gIdx + 1)]
                    scale_v = np.full((1, grp_size), scale_v)
                    # mul
                    value_cur = value_cur * scale_v
                    # copy to
                    value[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1),
                    gIdx * grp_size:(gIdx + 1) * grp_size] = value_cur
    else:
        if lowprecision_flag:
            if q_dtype == "float16":
                scale = scale.astype(np.float16)
                offset = offset.astype(np.float16)
            elif q_dtype == "bfloat16":
                scale = scale.astype(tf.bfloat16.as_numpy_dtype)
                offset = offset.astype(tf.bfloat16.as_numpy_dtype)
            else:
                print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                exit(1)
        else:
            scale = scale.astype(np.float32)
            offset = offset.astype(np.float32)
        if kv_dtype == "int8":
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = k.astype(np.float16)
                    value = v.astype(np.float16)
                elif q_dtype == "bfloat16":
                    key = k.astype(tf.bfloat16.as_numpy_dtype)
                    value = v.astype(tf.bfloat16.as_numpy_dtype)
                else:
                    print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                    exit(1)
            else:
                key = k.astype(np.float32)
                value = v.astype(np.float32)
        elif kv_dtype in ["int4", "int32"]:
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = k.astype(np.float16)
                    value = v.astype(np.float16)
                elif q_dtype == "bfloat16":
                    key = k.astype(np.float16).astype(tf.bfloat16.as_numpy_dtype)
                    value = v.astype(np.float16).astype(tf.bfloat16.as_numpy_dtype)
                else:
                    print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                    exit(1)
            else:
                scale = scale.astype(np.float16)
                offset = offset.astype(np.float16)
                key = k.astype(np.float16)
                value = v.astype(np.float16)
        elif kv_dtype == "hifloat8":
            # uint8->fp16，精度转换
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = trans_np_hifuint8_tensor_to_float32(k).astype(np.float16)
                    value = trans_np_hifuint8_tensor_to_float32(v).astype(np.float16)
                elif q_dtype == "bfloat16":
                    key = trans_np_hifuint8_tensor_to_float32(k).astype(tf.bfloat16.as_numpy_dtype)
                    value = trans_np_hifuint8_tensor_to_float32(v).astype(tf.bfloat16.as_numpy_dtype)
                else:
                    print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                    exit(1)
            else:
                key = trans_np_hifuint8_tensor_to_float32(k).astype(np.float16)
                value = trans_np_hifuint8_tensor_to_float32(v).astype(np.float16)
        elif kv_dtype in ["float8_e5m2", "float8_e4m3fn"]:
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = k.astype(np.float16)
                    value = v.astype(np.float16)
                elif q_dtype == "bfloat16":
                    key = k.astype(np.float16).astype(tf.bfloat16.as_numpy_dtype)
                    value = v.astype(np.float16).astype(tf.bfloat16.as_numpy_dtype)
                else:
                    print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                    exit(1)
            else:
                key = k.astype(np.float16)
                value = v.astype(np.float16)
        else:
            print(f"[ERROR]Invalid input kv dtype: {kv_dtype}")
            exit(1)

        if pto_flag:
            scale1 = scale[0:1, :, :]
            scale2 = scale[1:2, :, :]
            offset1 = offset[0:1, :, :]
            offset2 = offset[1:2, :, :]
            for nIdx in range(N):
                key_cur = key[:, nIdx:(nIdx + 1), :, :]
                scale1_cur = np.expand_dims(scale1, axis=-1)
                offset1_cur = np.expand_dims(offset1, axis=-1)
                key_cur = np.add(key_cur, offset1_cur)
                key_cur = np.multiply(key_cur, scale1_cur)
                key[:, nIdx:(nIdx + 1), :, :] = key_cur

                value_cur = value[:, nIdx:(nIdx + 1), :, :]
                scale2_cur = np.expand_dims(scale2, axis=-1)
                offset2_cur = np.expand_dims(offset2, axis=-1)
                value_cur = np.add(value_cur, offset2_cur)
                value_cur = np.multiply(value_cur, scale2_cur)
                value[:, nIdx:(nIdx + 1), :, :] = value_cur
        else:
            # 2,n,1,d -> 1,n,1,d
            scale1 = scale[0:1, :, :, :]
            scale2 = scale[1:2, :, :, :]
            offset1 = offset[0:1, :, :, :]
            offset2 = offset[1:2, :, :, :]
            for nIdx in range(N):
                for sIdx in range(S):
                    key_cur = key[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1), :]
                    key_cur = key_cur + offset1[:, nIdx:(nIdx + 1), :, :]
                    key_cur = key_cur * scale1[:, nIdx:(nIdx + 1), :, :]
                    key[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1), :] = key_cur

                    value_cur = value[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1), :]
                    value_cur = value_cur + offset2[:, nIdx:(nIdx + 1), :, :]
                    value_cur = value_cur * scale2[:, nIdx:(nIdx + 1), :, :]
                    value[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1), :] = value_cur
    return key, value

def kv_split_antiquant_tensor(k, v, k_scale, k_offset, v_scale, v_offset, k_anti_mode, v_anti_mode, kv_dtype="int8",q_dtype="float16"):
    lowprecision_flag = False if q_dtype == "float32" else True
    if kv_dtype in ["fp4_e1m2", "fp4_e2m1", "float4_e1m2", "float4_e2m1"]:
        raise NotImplementedError

    else:
        if lowprecision_flag:
            if q_dtype == "bfloat16":
                k_scale = k_scale.type(torch.bfloat16)
                k_offset = k_offset.type(torch.bfloat16)
                v_scale = v_scale.type(torch.bfloat16)
                v_offset = v_offset.type(torch.bfloat16)
            elif q_dtype == "float16":
                k_scale = k_scale.type(torch.float16)
                k_offset = k_offset.type(torch.float16)
                v_scale = v_scale.type(torch.float16)
                v_offset = v_offset.type(torch.float16)
            else:
                print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                exit(1)
        else:
            k_scale = k_scale.type(torch.float32)
            k_offset = k_offset.type(torch.float32)
            v_scale = v_scale.type(torch.float32)
            v_offset = v_offset.type(torch.float32)
        if kv_dtype == "int8":
            if lowprecision_flag:
                if q_dtype == "bfloat16":
                    key = k.type(torch.bfloat16)
                    value = v.type(torch.bfloat16)
                elif q_dtype == "float16":
                    key = k.type(torch.float16)
                    value = v.type(torch.float16)
                else:
                    print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                    exit(1)
            else:
                key = k.type(torch.float32)
                value = v.type(torch.float32)
        else:
            raise NotImplementedError

    if k_anti_mode != '0' or v_anti_mode != '0':
        raise NotImplementedError
    key = (key + k_offset) * k_scale
    value = (value + v_offset) * v_scale
    if q_dtype == "float16":
        key = key.type(torch.float16)
        value = value.type(torch.float16)
    elif q_dtype == "bfloat16":
        key = key.type(torch.bfloat16)
        value = value.type(torch.bfloat16)
    elif q_dtype == "float32":
        key = key.type(torch.float32)
        value = value.type(torch.float32)
    else:
        raise NotImplementedError
    return key, value


def kv_split_antiquant(k, v, k_scale, k_offset, v_scale, v_offset, k_anti_mode, v_anti_mode, lowprecision_flag,kv_dtype="int8", q_dtype="float16"):
    N = k.shape[1]
    S = k.shape[2]
    if kv_dtype in ["fp4_e1m2", "fp4_e2m1", "float4_e1m2", "float4_e2m1"]:
        k_scale_bf16 = trans_tensor_fp8_e8m0_to_bf16(k_scale)
        v_scale_bf16 = trans_tensor_fp8_e8m0_to_bf16(v_scale)
        if lowprecision_flag:
            if q_dtype == "float16":
                k_scale = k_scale_bf16.astype(np.float16)
                v_scale = v_scale_bf16.astype(np.float16)
            elif q_dtype == "bfloat16":
                k_scale = k_scale_bf16
                v_scale = v_scale_bf16
            else:
                print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                exit(1)
        else:
            k_scale = k_scale_bf16.astype(np.float16)
            v_scale = v_scale_bf16.astype(np.float16)

        if kv_dtype in ["fp4_e1m2", "float4_e1m2"]:
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = new_trans_np_fp4_e1m2_tensor_to_bfloat16(k).astype(np.float16)
                    value = new_trans_np_fp4_e1m2_tensor_to_bfloat16(v).astype(np.float16)
                elif q_dtype == "bfloat16":
                    key = new_trans_np_fp4_e1m2_tensor_to_bfloat16(k)
                    value = new_trans_np_fp4_e1m2_tensor_to_bfloat16(v)
                else:
                    print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                    exit(1)
            else:
                key = new_trans_np_fp4_e1m2_tensor_to_bfloat16(k).astype(np.float16)
                value = new_trans_np_fp4_e1m2_tensor_to_bfloat16(v).astype(np.float16)
        else:
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = new_trans_np_fp4_e2m1_tensor_to_bfloat16(k).astype(np.float16)
                    value = new_trans_np_fp4_e2m1_tensor_to_bfloat16(v).astype(np.float16)
                elif q_dtype == "bfloat16":
                    key = new_trans_np_fp4_e2m1_tensor_to_bfloat16(k)
                    value = new_trans_np_fp4_e2m1_tensor_to_bfloat16(v)
                else:
                    print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                    exit(1)
            else:
                key = new_trans_np_fp4_e2m1_tensor_to_bfloat16(k).astype(np.float16)
                value = new_trans_np_fp4_e2m1_tensor_to_bfloat16(v).astype(np.float16)

    else:
        if lowprecision_flag:
            if q_dtype == "float16":
                k_scale = k_scale.astype(np.float16)
                k_offset = k_offset.astype(np.float16)
                v_scale = v_scale.astype(np.float16)
                v_offset = v_offset.astype(np.float16)
            elif q_dtype == "bfloat16":
                k_scale = k_scale.astype(tf.bfloat16.as_numpy_dtype)
                k_offset = k_offset.astype(tf.bfloat16.as_numpy_dtype)
                v_scale = v_scale.astype(tf.bfloat16.as_numpy_dtype)
                v_offset = v_offset.astype(tf.bfloat16.as_numpy_dtype)
            else:
                print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                exit(1)
        else:
            k_scale = k_scale.astype(np.float32)
            k_offset = k_offset.astype(np.float32)
            v_scale = v_scale.astype(np.float32)
            v_offset = v_offset.astype(np.float32)
        if kv_dtype == "int8":
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = k.astype(np.float16)
                    value = v.astype(np.float16)
                elif q_dtype == "bfloat16":
                    key = k.astype(tf.bfloat16.as_numpy_dtype)
                    value = v.astype(tf.bfloat16.as_numpy_dtype)
                else:
                    print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                    exit(1)
            else:
                key = k.astype(np.float32)
                value = v.astype(np.float32)
        elif kv_dtype in ["int4", "int32"]:
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = k.astype(np.float16)
                    value = v.astype(np.float16)
                elif q_dtype == "bfloat16":
                    key = k.astype(np.float16).astype(tf.bfloat16.as_numpy_dtype)
                    value = v.astype(np.float16).astype(tf.bfloat16.as_numpy_dtype)
                else:
                    print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                    exit(1)
            else:
                k_scale = k_scale.astype(np.float16)
                k_offset = k_offset.astype(np.float16)
                v_scale = v_scale.astype(np.float16)
                v_offset = v_offset.astype(np.float16)
                key = k.astype(np.float16)
                value = v.astype(np.float16)
        elif kv_dtype == "hifloat8":
            # uint8->fp16，精度转换
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = trans_np_hifuint8_tensor_to_float32(k).astype(np.float16)
                    value = trans_np_hifuint8_tensor_to_float32(v).astype(np.float16)
                elif q_dtype == "bfloat16":
                    key = trans_np_hifuint8_tensor_to_float32(k).astype(tf.bfloat16.as_numpy_dtype)
                    value = trans_np_hifuint8_tensor_to_float32(v).astype(tf.bfloat16.as_numpy_dtype)
                else:
                    print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                    exit(1)
            else:
                key = trans_np_hifuint8_tensor_to_float32(k).astype(np.float16)
                value = trans_np_hifuint8_tensor_to_float32(v).astype(np.float16)
        elif kv_dtype in ["float8_e5m2", "float8_e4m3fn"]:
            if lowprecision_flag:
                if q_dtype == "float16":
                    key = k.astype(np.float16)
                    value = v.astype(np.float16)
                elif q_dtype == "bfloat16":
                    key = k.astype(np.float16).astype(tf.bfloat16.as_numpy_dtype)
                    value = v.astype(np.float16).astype(tf.bfloat16.as_numpy_dtype)
                else:
                    print(f"[ERROR]Invalid input q dtype: {q_dtype}")
                    exit(1)
            else:
                key = k.astype(np.float16)
                value = v.astype(np.float16)
        else:
            print(f"[ERROR]Invalid input kv dtype: {kv_dtype}")
            exit(1)

    if k_anti_mode in ['1', '4']:
        scale1 = k_scale[0:1, :, :]  # 11S
        offset1 = k_offset[0:1, :, :]
        for nIdx in range(N):
            key_cur = key[:, nIdx:(nIdx + 1), :, :]
            # scale: 11s -> 11s1
            scale1_cur = np.expand_dims(scale1, axis=-1)
            offset1_cur = np.expand_dims(offset1, axis=-1)
            key_cur = np.add(key_cur, offset1_cur)
            key_cur = np.multiply(key_cur, scale1_cur)
            key[:, nIdx:(nIdx + 1), :, :] = key_cur
    elif k_anti_mode in ['0', '2']:
        # 2,n,1,d -> 1,n,1,d
        scale1 = k_scale[0:1, :, :, :]
        offset1 = k_offset[0:1, :, :, :]
        for nIdx in range(N):
            for sIdx in range(S):
                key_cur = key[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1), :]
                key_cur = key_cur + offset1[:, nIdx:(nIdx + 1), :, :]
                key_cur = key_cur * scale1[:, nIdx:(nIdx + 1), :, :]
                key[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1), :] = key_cur
    elif k_anti_mode in ['3', '5']:
        # 计算, scale为1NS, key为1nsd
        scale1 = k_scale[0:1, :, :]
        offset1 = k_offset[0:1, :, :]
        for nIdx in range(N):
            # key_cur：11sd
            key_cur = key[:, nIdx:(nIdx + 1), :, :]
            # scale/offset：11s
            scale1_cur = scale1[:, nIdx:(nIdx + 1), :]
            offset1_cur = offset1[:, nIdx:(nIdx + 1), :]
            # scale/offset：11s1
            scale1_cur = np.expand_dims(scale1_cur, axis=-1)
            offset1_cur = np.expand_dims(offset1_cur, axis=-1)
            # (11sd + 11s1) * 11s1
            key_cur = np.add(key_cur, offset1_cur)
            key_cur = np.multiply(key_cur, scale1_cur)
            key[:, nIdx:(nIdx + 1), :, :] = key_cur
    elif k_anti_mode == '6':
        g = k_scale.shape[4]
        k_scale = k_scale[0]
        grp_size = 32
        for nIdx in range(N):
            for sIdx in range(S):
                for gIdx in range(g):
                    # copy from
                    key_cur = key[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1), gIdx * grp_size:(gIdx + 1) * grp_size]
                    # broardcast
                    scale_k = k_scale[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1), gIdx:(gIdx + 1)]
                    scale_k = np.full((1, grp_size), scale_k)
                    # mul
                    key_cur = key_cur * scale_k
                    # copy to
                    key[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1),
                    gIdx * grp_size:(gIdx + 1) * grp_size] = key_cur
    if v_anti_mode in ['1', '4']:
        scale2 = v_scale[0:1, :, :]
        offset2 = v_offset[0:1, :, :]
        for nIdx in range(N):
            value_cur = value[:, nIdx:(nIdx + 1), :, :]
            scale2_cur = np.expand_dims(scale2, axis=-1)
            offset2_cur = np.expand_dims(offset2, axis=-1)
            value_cur = np.add(value_cur, offset2_cur)
            value_cur = np.multiply(value_cur, scale2_cur)
            value[:, nIdx:(nIdx + 1), :, :] = value_cur
    elif v_anti_mode in ['0', '2']:
        # 2,n,1,d -> 1,n,1,d
        scale2 = v_scale[0:1, :, :, :]
        offset2 = v_offset[0:1, :, :, :]
        for nIdx in range(N):
            for sIdx in range(S):
                value_cur = value[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1), :]
                value_cur = value_cur + offset2[:, nIdx:(nIdx + 1), :, :]
                value_cur = value_cur * scale2[:, nIdx:(nIdx + 1), :, :]
                value[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1), :] = value_cur
    elif v_anti_mode in ['3', '5']:
        # 计算, scale为1SN, value为bnsd
        scale1 = v_scale[0:1, :, :]
        offset1 = v_offset[0:1, :, :]
        for nIdx in range(N):
            # value_cur：11sd
            value_cur = value[:, nIdx:(nIdx + 1), :, :]
            # scale/offset：11s
            scale1_cur = scale1[:, nIdx:(nIdx + 1), :]
            offset1_cur = offset1[:, nIdx:(nIdx + 1), :]
            # scale/offset：11s1
            scale1_cur = np.expand_dims(scale1_cur, axis=-1)
            offset1_cur = np.expand_dims(offset1_cur, axis=-1)
            # (11sd + 11s1) * 11s1
            value_cur = np.add(value_cur, offset1_cur)
            value_cur = np.multiply(value_cur, scale1_cur)
            value[:, nIdx:(nIdx + 1), :, :] = value_cur
    elif v_anti_mode == '6':
        g = v_scale.shape[4]
        v_scale = v_scale[0]
        grp_size = 32
        for nIdx in range(N):
            for sIdx in range(S):
                for gIdx in range(g):
                    # copy from
                    value_cur = value[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1), gIdx * grp_size:(gIdx + 1) * grp_size]
                    # broardcast
                    scale_v = v_scale[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1), gIdx:(gIdx + 1)]
                    scale_v = np.full((1, grp_size), scale_v)
                    # mul
                    value_cur = value_cur * scale_v
                    # copy to
                    value[:, nIdx:(nIdx + 1), sIdx:(sIdx + 1),
                    gIdx * grp_size:(gIdx + 1) * grp_size] = value_cur
    return key, value



def _trans_antiparam_to_1bs(tensor):
    if len(tensor.shape) == 2:
        tensor_new = tensor.reshape(1, tensor.shape[0], tensor.shape[1])
    else:
        tensor_new = tensor

    return tensor_new


def softmaxv1(x):
    x = x.astype(np.float32)
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    ans = y / x_sum
    return ans, x_max, x_sum


def softmax(x):
    # this func is only used by quant_dequant
    x = x.astype(np.float32)
    x_max = x.max(axis=-1, keepdims=True)
    x_sub = x - x_max
    y = np.exp(x_sub)
    x_sum = y.sum(axis=-1, keepdims=True)
    ans = y
    # ans = ans.astype(np.float16)
    return ans, x_sum, x_max


def softmax_flash(x, max_front=None, sum_front=None, axis=None, update=False):
    # print("===>update:", update)
    if update == False:
        # print("===M  x1.shape", x.shape)
        x = x.astype(np.float32)
        x_max = x.max(axis=-1, keepdims=True)
        x_sub = x - x_max
        y = np.exp(x_sub)
        x_sum = y.sum(axis=-1, keepdims=True)
        ans = y / x_sum
        ans = ans.astype(np.float16)
        x_max = x_max.astype(np.float16)
        x_sum = x_sum.astype(np.float16)
        exp_max = None
        # print("++++++x flash 000:", x)
        # print("++++++ans flash 000:", ans)
        # print("++++++x_sum flash 111:", x_sum)
        return x_max, x_sum, ans, exp_max
    else:
        x = x.astype(np.float16)
        max_front = max_front.astype(np.float16)
        sum_front = sum_front.astype(np.float16)
        x_max = np.max(np.concatenate((max_front, x), axis=-1), axis=-1, keepdims=True)
        x_exp = np.exp(x - x_max)
        x_sum = np.sum(x_exp, axis=-1, keepdims=True)
        exp_max = np.exp(max_front - x_max)
        x_sum = exp_max * sum_front + x_sum
        exp_max = exp_max * sum_front / x_sum
        out = x_exp / x_sum
        out = out.astype(np.float16)
        exp_max = exp_max.astype(np.float16)
        return x_max, x_sum, out, exp_max


def sum_ceil(lst, a):
    res = 0
    for num in lst:
        res += math.ceil(num / a)
    return res


def get_h(shape, layout):
    if layout == "BNSD":
        h = shape[1] * shape[3]
    elif layout == "BSND":
        h = shape[2] * shape[3]
    elif layout == "BSH":
        h = shape[2]
    elif layout == "TND":
        h = shape[1] * shape[2]
    else:
        h = 0
    return h


def get_sinner(ifa_param):
    from tbe.common.utils.op_tiling import do_op_tiling
    from tbe.common.platform import set_current_compile_soc_info
    set_current_compile_soc_info("Ascend910B1")
    # print("===>ifa_param:", ifa_param)
    print("=============>ASCEND_OPP_PATH:", os.environ.get('ASCEND_OPP_PATH'))
    q_shape = ifa_param['q_shape']
    q_dtype = trans_input_dtype(ifa_param['q_dtype'])
    m_shape = ifa_param['m_shape']
    m_dtype = ifa_param['m_dtype']
    p_shape = ifa_param['p_shape']
    p_dtype = ifa_param['p_dtype']
    k_shape = ifa_param['k_shape_list'][0]
    v_shape = ifa_param['v_shape_list'][0]
    k_shape[0] = v_shape[0] = q_shape[0]
    out_shape = ifa_param['out_shape']
    out_dtype = ifa_param['out_dtype']

    op_type = "FusedInferAttentionScore"
    compile_info = {}
    # qkv
    inputs = [{'shape': q_shape, 'ori_shape': q_shape, 'format': 'ND', 'ori_format': 'ND', 'dtype': q_dtype,
               'range': [(0, None)], 'param_name': 'query'},
              {'shape': k_shape, 'ori_shape': k_shape, 'format': 'ND', 'ori_format': 'ND',
               'dtype': 'int8', 'range': [(0, None)], 'param_name': 'key'},
              {'shape': v_shape, 'ori_shape': v_shape, 'format': 'ND', 'ori_format': 'ND',
               'dtype': 'int8', 'range': [(0, None)], 'param_name': 'value'}]
    # pse_shift
    if ifa_param['p_flag']:
        inputs.extend([{'shape': p_shape, 'ori_shape': p_shape, 'format': 'ND', 'ori_format': 'ND', 'dtype': p_dtype,
                        'range': [(0, None)], 'param_name': 'pse_shift'}])
    else:
        inputs.extend([None])
    # atten_mask
    if ifa_param['m_flag']:
        inputs.extend([{'shape': m_shape, 'ori_shape': m_shape, 'format': 'ND', 'ori_format': 'ND',
                        'dtype': m_dtype, 'range': [(0, None)], 'param_name': 'atten_mask'}])
    else:
        inputs.extend([None])
    # actual_seq_lengths
    inputs.extend([None])
    # actual_seq_lengths_kv
    if ifa_param['as_flag']:
        act_seq = ifa_param['actualSeqLengths_raw']
        inputs.extend([{'shape': [len(act_seq)], 'ori_shape': [len(act_seq)], 'format': 'ND', 'ori_format': 'ND',
                        'range': [(0, None)], 'const_value': act_seq, 'dtype': 'int64',
                        'param_name': 'actual_seq_lengths_kv'}])
    else:
        inputs.extend([None])
    # dequant_scale1 quant_scale1 dequant_scale2
    if ifa_param['in_quant_flag']:
        inputs.extend([{'shape': [1], 'ori_shape': [1], 'format': 'ND',
                        'ori_format': 'ND', 'dtype': 'uint64', 'range': [(0, None)], 'param_name': 'dequant_scale_1'},
                       {'shape': [1], 'ori_shape': [1], 'format': 'ND',
                        'ori_format': 'ND', 'dtype': 'float32', 'range': [(0, None)], 'param_name': 'quant_scale_1'},
                       {'shape': [1], 'ori_shape': [1], 'format': 'ND',
                        'ori_format': 'ND', 'dtype': 'uint64', 'range': [(0, None)], 'param_name': 'dequant_scale_2'}])
    else:
        inputs.extend([None, None, None])
    # quant_scale2 quant_offset2
    if ifa_param['out_quant_flag']:
        quantScale2_shape = ifa_param['quantScale2_shape']
        quantOffset2_shape = ifa_param['quantOffset2_shape']
        quantScale2_dtype = ifa_param['quantScale2_dtype']
        quantOffset2_dtype = ifa_param['quantOffset2_dtype']
        inputs.extend([{'shape': quantScale2_shape, 'ori_shape': quantScale2_shape, 'format': 'ND',
                        'ori_format': 'ND', 'dtype': quantScale2_dtype, 'range': [(0, None)],
                        'param_name': 'quant_scale_2'}])
        if ifa_param['out_quant_offset_flag']:
            inputs.extend([{'shape': quantOffset2_shape, 'ori_shape': quantOffset2_shape, 'format': 'ND',
                            'ori_format': 'ND', 'dtype': quantOffset2_dtype, 'range': [(0, None)],
                            'param_name': 'quant_offset_2'}])
        else:
            inputs.extend([None])
    else:
        inputs.extend([None, None])
    # antiquant_scale antiquant_offset
    if ifa_param['kv_quant_flag']:
        anti_quant_shape = ifa_param['antiquantscale_shape_raw']
        anti_quant_dtype = ifa_param['antiquantscale_dtype']
        inputs.extend([{'shape': anti_quant_shape, 'ori_shape': anti_quant_shape, 'format': 'ND',
                        'ori_format': 'ND', 'dtype': anti_quant_dtype, 'range': [(0, None)],
                        'param_name': 'antiquant_scale'}])
        if ifa_param['kv_quant_offset_flag']:
            inputs.extend([{'shape': anti_quant_shape, 'ori_shape': anti_quant_shape, 'format': 'ND',
                            'ori_format': 'ND', 'dtype': anti_quant_dtype, 'range': [(0, None)],
                            'param_name': 'antiquant_offset'}])
        else:
            inputs.extend([None])
    else:
        inputs.extend([None, None])

    # block_table
    if ifa_param['pa_flag']:
        bt_shape = ifa_param['bt_shape']
        bt_dtype = ifa_param['bt_dtype']
        inputs.extend([{'shape': bt_shape, 'ori_shape': bt_shape, 'format': 'ND',
                        'ori_format': 'ND', 'dtype': bt_dtype, 'range': [(0, None)], 'param_name': 'block_table'}])
        bs = int(ifa_param['blocksize'])
        act_seq = ifa_param['actualSeqLengths']
        kvcache_shape_1 = sum_ceil(act_seq, bs)
        kvcache_shape_2 = bs
        kvcache_shape_3 = get_h(k_shape, ifa_param['inputLayout'])
        inputs[1]['shape'] = inputs[1]['ori_shape'] = [kvcache_shape_1, kvcache_shape_2, kvcache_shape_3]
        inputs[2]['shape'] = inputs[2]['ori_shape'] = [kvcache_shape_1, kvcache_shape_2, kvcache_shape_3]
    else:
        inputs.extend([None])
    # query_padding_size
    inputs.extend([None])
    # kv_padding_size
    if ifa_param['padding_flag']:
        inputs.extend([{'shape': [1], 'ori_shape': [1], 'format': 'ND',
                        'ori_format': 'ND', 'dtype': 'int64', 'range': [(0, None)], 'param_name': 'kv_padding_size'}])
    else:
        inputs.extend([None])

    outputs = [{'shape': out_shape, 'ori_shape': out_shape, 'format': 'ND', 'ori_format': 'ND', 'dtype': out_dtype,
                'range': [(0, None)], 'param_name': 'attention_out'}]

    attrs = [{'name': 'num_heads', 'dtype': 'int', 'value': ifa_param['numHeads']},
             {'name': 'scale_value', 'dtype': 'float', 'value': ifa_param['scaleValue']},
             {'name': 'pre_tokens', 'dtype': 'int', 'value': ifa_param['pre_tokens']},
             {'name': 'next_tokens', 'dtype': 'int', 'value': ifa_param['next_tokens']},
             {'name': 'input_layout', 'dtype': 'str', 'value': ifa_param['inputLayout']},
             {'name': 'num_key_value_heads', 'dtype': 'int', 'value': ifa_param['numKeyValueHeads']},
             {'name': 'sparse_mode', 'dtype': 'int', 'value': ifa_param['sparse_mode']},
             {'name': 'inner_precise', 'dtype': 'int', 'value': ifa_param['innerprecise']},
             {'name': 'block_size', 'dtype': 'int', 'value': ifa_param['blocksize']},
             {'name': 'antiquant_mode', 'dtype': 'int', 'value': int(ifa_param['antiquant_mode'])},
             {'name': 'softmax_lse_flag', 'dtype': 'bool', 'value': False}]

    print(f"inputs:{inputs}")
    print(f"outputs:{outputs}")
    print(f"attrs:{attrs}")
    ret = do_op_tiling(op_type, compile_info, inputs, outputs, attrs=attrs)
    tiling_key = ret.get("tiling_key")
    cache_tiling = ret.get("tiling_data")
    tiling_data_pr = np.frombuffer(cache_tiling, dtype=np.int32)
    # 20240118前:245
    idx_singleProcessSInnerSize = 254
    singleProcessSInnerSize = tiling_data_pr[idx_singleProcessSInnerSize]
    # print("============>cache_tiling:", (tiling_data_pr))
    print(f"[INFO]get_sinner result --> tiling_key:{tiling_key}, singleProcessSInnerSize:{singleProcessSInnerSize}")
    return singleProcessSInnerSize


def _t_broadcastKV_sigle(numHeads, numKeyValueHeads, kv_tensor, input_dtype):
    factor = numHeads // numKeyValueHeads
    kv_shape = kv_tensor.shape
    B = kv_shape[0]
    S = kv_shape[2]
    D = kv_shape[3]
    kv_res = np.zeros([B, numHeads, S, D], dtype=kv_tensor.dtype)
    for i in range(numHeads):
        j = i // factor
        kv_res[:, i:i + 1, :, :] = kv_tensor[:, j:j + 1, :, :]
    return kv_res, kv_res.shape


def trans_uint64_2_float32(input):
    output = np.uint64(input)
    output = np.frombuffer(output, dtype=np.float32)
    output = output[: 1][0]
    print(f"uint64:{input}, float32:{output}")
    return output


def torch_to_numpy(tensor):
    n_tensor = None
    print(tensor.dtype)
    if type(tensor) == "<class 'torch.Tensor'>":
        if tensor.dtype in ['torch.float16', 'torch.bool']:
            n_tensor = tensor.numpy()
        elif tensor.dtype in ['torch.bfloat16']:
            n_tensor = tensor.numpy()
    return n_tensor


def _n_trans_shape_to_bnsd(tensor, shape, layout, headnums=None, act_seq=None):
    if layout in ["BSH", "BSH_NBSD"]:
        if headnums is None:
            return tensor, shape
        B = shape[0]
        S = shape[1]
        H = shape[2]
        N = headnums
        D = H // N
        tensor = tensor.reshape(B, S, N, D).transpose(0, 2, 1, 3)
        return tensor, [B, N, S, D]
    elif layout in ["BSND", "BSND_NBSD"]:
        B = shape[0]
        S = shape[1]
        N = shape[2]
        D = shape[3]
        tensor = tensor.transpose(0, 2, 1, 3)
        return tensor, [B, N, S, D]
    elif layout in ["TND", "TND_NTD"]:
        T = shape[0]
        N = shape[1]
        D = shape[2]
        B = len(act_seq)
        S = max(act_seq)
        new_tensor = np.zeros((B, N, S, D), dtype=tensor.dtype)
        t_start = 0
        for b_index in range(B):
            act_s = act_seq[b_index]
            t_end = t_start + act_s
            # print(f"[TEMP]b_index:{b_index} act_s:{act_s}, t_start:{t_start}, t_end:{t_end}")
            if act_s == 0:
                continue
            for n_index in range(N):
                new_tensor[b_index, n_index, 0:act_s, :] = tensor[t_start:t_end, n_index, :]
            t_start += act_s
        print(f"[TEMP]trans tnd 2 bnsd: {tensor.shape} -> {new_tensor.shape}")
        # print(f"[TEMP]new_tensor:{new_tensor}")
        return new_tensor, [B, N, S, D]
    else:
        return tensor, shape


def _t_bsh_to_bnsd(tensor, bsh_shape, headnums, batch=1):
    if len(bsh_shape) == 2:
        B = batch
        S = bsh_shape[0] // B
        H = bsh_shape[1]
        N = headnums
        D = H // N
        return tensor.reshape(B, S, N, D).transpose(0, 2, 1, 3), [B, N, S, D]
    elif len(bsh_shape) == 3:
        B = bsh_shape[0]
        S = bsh_shape[1]
        H = bsh_shape[2]
        N = headnums
        D = H // N
        return tensor.reshape(B, S, N, D).transpose(0, 2, 1, 3), [B, N, S, D]
    else:
        return tensor, bsh_shape


def trans_bnsd_to_layout(tensor, shape, layout, act_q=None):
    if layout == "BSH":
        output = tensor.permute(0, 2, 1, 3).contiguous().view(shape)
        return output
    elif layout == "BSND":
        output = tensor.permute(0, 2, 1, 3).contiguous()
        return output
    elif layout in ["BSND_NBSD", "BNSD_NBSD", "BSH_NBSD"]:
        output = tensor.permute(1, 0, 2, 3).contiguous()
        return output
    elif layout in ["TND", "TND_NTD"]:
        T = sum(act_q)
        B = tensor.shape[0]
        N = tensor.shape[1]
        D = tensor.shape[3]
        output = torch.zeros(size=(T, N, D), dtype=tensor.dtype)
        t_start = 0
        for b_index in range(B):
            act_s = act_q[b_index]
            t_end = t_start + act_s
            if act_s == 0:
                continue
            for n_index in range(N):
                output[t_start:t_end, n_index, :] = tensor[b_index, n_index, :act_s, :]
            t_start += act_s
        if layout == "TND_NTD":
            output = output.permute(1, 0, 2).contiguous()
        return output
    else:
        return tensor


def trans_bnsd_to_bsh(tensor, shape):
    if len(shape) == 4:
        B = shape[0]
        N = shape[1]
        S = shape[2]
        D = shape[3]
        H = N * D
        return tensor.transpose(0, 2, 1, 3).reshape(B, S, H)
    if len(shape) == 3:
        B = shape[0]
        S = shape[1]
        H = shape[2]
        return tensor.transpose(0, 2, 1, 3).reshape(B, S, H)
    else:
        return tensor


def _t_broadcast_mask_n(m_tensor, m_shape, N, B):
    print(f"broadcast_mask_n:mask shape:{m_shape}")
    if len(m_shape) == 4:
        # b11s
        m_res = torch.zeros([B, N, m_shape[2], m_shape[3]])
        for i in range(B):
            for j in range(N):
                m_res[i:i + 1, j:j + 1, :, :] = m_tensor[i:i + 1, :, :, :]
        print(f"after broadcast_mask_n:mask shape:{m_res.shape}")
        return m_res
    elif len(m_shape) == 3:
        # b1s
        m_res = torch.zeros([B, N, m_shape[1], m_shape[2]])
        for i in range(B):
            for j in range(N):
                m_res[i:i + 1, j:j + 1, :, :] = m_tensor[i:i + 1, :, :]
        print(f"after broadcast_mask_n:mask shape:{m_res.shape}")
        return m_res
    elif len(m_shape) == 2:
        # bs
        m_res = torch.zeros([B, N, 1, m_shape[1]])
        for i in range(B):
            for j in range(N):
                m_res[i:i + 1, j:j + 1, 0, :] = m_tensor[i:i + 1, :]
        print(f"after broadcast_mask_n:mask shape:{m_res.shape}")
        return m_res
    else:
        print("[ERROR] m_tensor shape is invalid!")
        exit(1)


def _trans_antiparam_to_2bnsg(shape, tensor, layout, numKeyValueHeads):
    # bsh: (2, B, S, H/32) -> (2, B, N, S, D/32)
    if layout in ["BSH", "BSH_NBSD"]:
        B = shape[1]
        S = shape[2]
        G = shape[3]
        N = numKeyValueHeads
        G2 = int(G / N)
        print(f"[INFO]_trans_antiparam_to_2bnsg: layout={layout}, b={B}, s={S} ,n={numKeyValueHeads},g(h/32)={G}")
        new_tensor = tensor.reshape(2, B, S, N, G2).transpose(0, 1, 3, 2, 4)
        return new_tensor
    # bsh: (2, B, S, N, D/32) -> (2, B, N, S, D/32)
    elif layout in ["BSND", "BSND_NBSD"]:
        print(f"[INFO]_trans_antiparam_to_2bnsg: layout={layout}")
        new_tensor = tensor.transpose(0, 1, 3, 2, 4)
        return new_tensor
    else:
        print(f"[INFO]_trans_antiparam_to_2bnsg: layout={layout}")
        return tensor


def _trans_antiparam_to_1bnsg(shape, tensor, layout, numKeyValueHeads):
    # bsh: (1, B, S, H/32) -> (1, B, N, S, D/32)
    if layout in ["BSH", "BSH_NBSD"]:
        B = shape[1]
        S = shape[2]
        G = shape[3]
        N = numKeyValueHeads
        G2 = int(G / N)
        print(f"[INFO]_trans_antiparam_to_1bnsg: layout={layout}, b={B}, s={S} ,n={numKeyValueHeads},g(h/32)={G}")
        new_tensor = tensor.reshape(1, B, S, N, G2).transpose(0, 1, 3, 2, 4)
        return new_tensor
    # bsh: (1, B, S, N, D/32) -> (1, B, N, S, D/32)
    elif layout in ["BSND", "BSND_NBSD"]:
        print(f"[INFO]_trans_antiparam_to_1bnsg: layout={layout}")
        new_tensor = tensor.transpose(0, 1, 3, 2, 4)
        return new_tensor
    else:
        print(f"[INFO]_trans_antiparam_to_1bnsg: layout={layout}")
        return tensor


def broadcast_kv_dequant_tensor_fp8e8m0(tensor, numKeyValueHeads, numheads):
    B = tensor.shape[1]
    S = tensor.shape[3]
    G = tensor.shape[4]
    if numKeyValueHeads == numheads:
        tensor.astype(np.uint8)
        return tensor
    else:
        factor = numheads // numKeyValueHeads
        shape = tensor.shape
        D = shape[3]
        tensor_new = np.zeros([2, B, numheads, S, G], dtype=np.uint8)
        for i in range(numheads):
            j = i // factor
            tensor_new[:, :, i:i + 1, :, :] = tensor[:, :, j:j + 1, :, :]
        return tensor_new


def broadcast_kv_split_antiquant_tensor_fp8e8m0(tensor, numKeyValueHeads, numheads):
    B = tensor.shape[1]
    S = tensor.shape[3]
    G = tensor.shape[4]
    if numKeyValueHeads == numheads:
        tensor.astype(np.uint8)
        return tensor
    else:
        factor = numheads // numKeyValueHeads
        shape = tensor.shape
        D = shape[3]
        tensor_new = np.zeros([1, B, numheads, S, G], dtype=np.uint8)
        for i in range(numheads):
            j = i // factor
            tensor_new[:, :, i:i + 1, :, :] = tensor[:, :, j:j + 1, :, :]
        return tensor_new


def _t_softmax(x):
    x_max = torch.max(x, dim=-1, keepdims=True)[0]
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdims=True)
    ans = y.div(x_sum)
    return ans, x_max, x_sum


def AS_fp32(int_vals):
    arr = np.array(int_vals, dtype=np.int32)
    uint32 = arr.view(np.uint32)
    s = (uint32 >> 31) & 0x1
    e = (uint32 >> 23) & 0xFF
    m = uint32 & 0x007FFFFF
    fp32_bits = (s << 31) | (e << 23) | m
    return fp32_bits.view(np.float32).tolist()


def _t_indequant_ds_act(ifa_param):
    batch_index = ifa_param['batch_id']
    q_tensor = ifa_param['q_tensor_cur']
    k_tensor = ifa_param['k_sub_tensor']
    v_tensor = ifa_param['v_sub_tensor']
    q_rope_tensor = ifa_param['q_rope_tensor_cur']
    k_rope_tensor = ifa_param['k_rope_tensor_cur']
    scalar = ifa_param['scaleValue']
    act_seq = ifa_param['act_seq']
    act_seq_q = ifa_param['act_seq_q']
    lse_flag = ifa_param['softmax_lse_flag']
    print(f"q_shape:{q_tensor.shape} k_shape:{k_tensor.shape} v_shape:{v_tensor.shape} scalar:{scalar} act:{act_seq} act_q:{act_seq_q}")

    dequantScale1 = ifa_param['dequantScale1_cur'] # scaleK * scaleQ
    dequantScale2 = ifa_param['dequantScale2'] # scaleV
    print(f'dequantScale2: {dequantScale2.shape}')

    if lse_flag:
        lse = np.zeros((q_tensor.shape[0], q_tensor.shape[1], q_tensor.shape[2], 1))
    else:
        lse = None

    if ifa_param['m_flag']:
        mask_tensor = ifa_param['mask_cur'].numpy()

    kv_smax = k_tensor.shape[2]
    if act_seq != None and act_seq != -1:
        kv_smax = act_seq

    s2Inner = np.minimum(128, kv_smax) # 修改：David s2inner 128
    loop = np.ceil(kv_smax / s2Inner).astype(np.int32)
    tail = kv_smax - (loop - 1) * s2Inner
    print(f"s2Inner:{s2Inner} kv_smax:{kv_smax} loop:{loop} tail:{tail}")

    rowSum = np.zeros(shape=(q_tensor.shape[0], q_tensor.shape[1], q_tensor.shape[2], 1), dtype=np.float32)
    rowMax = np.full(shape=(q_tensor.shape[0], q_tensor.shape[1], q_tensor.shape[2], 1), fill_value=-np.inf,
                     dtype=np.float32)
    O_flash = np.zeros(shape=(q_tensor.shape[0], q_tensor.shape[1], q_tensor.shape[2], q_tensor.shape[3]),
                       dtype=np.float32) # 修改：这里需要该数据类型
    pse_cur = None
    mask_cur = None
    for i in range(loop): # 循环所有基本块
        kv_start_pos = i * s2Inner
        kv_end_pos = (i + 1) * s2Inner
        if i == (loop - 1): # 最后一个块
            kv_end_pos = i * s2Inner + tail
        k_cur = k_tensor[:, :, kv_start_pos: kv_end_pos, :]
        v_cur = v_tensor[:, :, kv_start_pos: kv_end_pos, :]
        k_rope_cur = k_rope_tensor[:, :, kv_start_pos: kv_end_pos, :]
        if ifa_param['p_flag']:
            pse_cur = ifa_param['pse_cur'][:, :, :, kv_start_pos: kv_end_pos]
        if ifa_param['m_flag']:
            mask_cur = mask_tensor[:, :, :, kv_start_pos: kv_end_pos]

        qkNopeRes = np.matmul(q_tensor.astype(np.float32), k_cur.transpose(0, 1, 3, 2).astype(np.float32))
        # qkNopeRes = qkNopeRes + bias # 修改：删除偏置

        # 直接fp32计算
        # bias2 = np.zeros_like(qkNopeRes, dtype=np.float32) - 2 ** 21 - 2 ** 23
        # qkNopeRes_fp32 = (AS_fp32(qkNopeRes) + bias2).astype(np.float32)


        qkRopeRes = np.matmul(q_rope_tensor.astype(np.float32), k_rope_cur.transpose(0, 1, 3, 2).astype(np.float32))

        qkBmmRes = qkNopeRes + qkRopeRes # 修改：qkNopeRes

        # V1部分计算
        qkBmmRes = qkBmmRes * dequantScale1 * scalar

        if pse_cur is not None:
            qkBmmRes += pse_cur
        if mask_cur is not None:
            qkBmmRes[mask_cur.astype(np.bool_)] = -1000000000000

        rowmax_tmp = qkBmmRes.max(axis=3, keepdims=True) # D 这个维度上的最大值
        replace_idx = np.where(rowMax < rowmax_tmp) # 找到rowMax中 “当前最大值小于新计算的最大值（rowmax_tmp）” 的位置索引。
        rowMax_old = copy.deepcopy(rowMax) # 深拷贝当前的rowMax，保存 “更新前的旧最大值”。
        rowMax[replace_idx] = rowmax_tmp[replace_idx] # 将rowMax中需要更新的位置（replace_idx）替换为新计算的最大值（rowmax_tmp对应位置的值）
        update_mul = np.exp(rowMax_old - rowMax) # 计算 “旧最大值与新最大值的差值的指数”，用于后续的累加修正。

        softmaxres = np.exp(qkBmmRes - rowMax).astype(np.float32) # softmax分子
        rowSum_A = softmaxres.sum(axis=3, keepdims=True).astype(np.float32)
        rowMax_A = (np.exp(rowmax_tmp - rowMax) + np.finfo(float).eps).astype(np.float32) # 由于rowMax已更新为max(旧rowMax, rowmax_tmp)，因此当rowmax_tmp是新最大值时，该差值为 0（exp(0)=1）；当rowMax未更新时，该差值为负数（exp(负数)是小于 1 的值）。
        rowSum = update_mul * rowSum + rowSum_A # 更新全局的 “softmax 分母累加和”，整合历史结果与当前批次结果。

        quantScale = 448 / rowMax_A # 修改：127改为448

        if i == loop - 1:
            lse = np.log(rowSum) + rowMax

        softmaxres = (softmaxres * quantScale).astype(np.float32)

        # 修改：case to fp8e4m3，再转换成fp32
        softmaxres = softmaxres.astype(numpy_float8_e4m3fn()).astype(np.float32)

        # if ifa_param['q_dtype'] != "float32":
        #     softmaxres = np.round(softmaxres.astype(np.float16)).astype(np.int8)
        # else:
        #     softmaxres = np.round(softmaxres).astype(np.int8)

        bmm2Res = np.matmul(softmaxres.astype(np.float32), v_cur.astype(np.float32)) # 修改：改为与v_cur相乘，数据类型fp32
        bmm2Res = bmm2Res.astype(np.float32) * rowMax_A.astype(np.float32) # 修改：直接使用fp32

        O_flash = O_flash.astype(np.float32) * update_mul.astype(np.float32) + bmm2Res # 修改：直接使用fp32

    O = O_flash.astype(np.float32) / rowSum / np.float32(448) * np.float32(dequantScale2) # 修改：127改为448

    # 这里处理行无效,要使用整个kv_s的mask,即mask_tensor
    if ifa_param['m_flag']:
        if mask_tensor is not None:
            if ifa_param['sparse_mode'] == 3 or act_seq_q > 1:
                for i in range(mask_tensor.shape[2]):
                    if mask_tensor[:, :, i, :].all() == 1:
                        print(f'mask i: {i} all is 1')
                        O[:, :, i, :] = 0
                        if lse is not None:
                            lse[:, :, i, :] = np.inf

    return O, lse

def _t_indequant_ds_act_golden(ifa_param):
    print("def _t_indequant_ds_act_golden(ifa_param): enter")

    q_tensor = ifa_param['q_tensor_cur'].astype(np.float64)
    k_tensor = ifa_param['k_sub_tensor'].astype(np.float64)
    v_tensor = ifa_param['v_sub_tensor'].astype(np.float64)
    q_rope_tensor = ifa_param['q_rope_tensor_cur'].astype(np.float64)
    k_rope_tensor = ifa_param['k_rope_tensor_cur'].astype(np.float64)
    scalar = np.array(ifa_param['scaleValue']).astype(np.float64)
    act_seq = ifa_param['act_seq']
    act_seq_q = ifa_param['act_seq_q']
    lse_flag = ifa_param['softmax_lse_flag']
    print(
        f"q_shape:{q_tensor.shape} k_shape:{k_tensor.shape} v_shape:{v_tensor.shape} scalar:{scalar} act:{act_seq} act_q:{act_seq_q}")

    dequantScale1 = ifa_param['dequantScale1_cur'].astype(np.float64)  # [1, 128, 1, 1]
    dequantScale2 = ifa_param['dequantScale2'].astype(np.float64)  # float
    scaleQ = dequantScale1 / dequantScale2
    scaleK = dequantScale2

    q_tensor = q_tensor * scaleQ
    k_tensor = k_tensor * scaleK
    v_tensor = v_tensor * scaleK

    #q_rope_tensor = q_rope_tensor * scaleQ * scaleK

    q_tensor_all = np.concatenate([q_tensor, q_rope_tensor], axis=3)
    k_tensor_all = np.concatenate([k_tensor, k_rope_tensor], axis=3)

    # dequantScale1 = ScaleQ * ScaleK
    # dequantScale2 = ScaleK

    if lse_flag:
        lse = np.zeros((q_tensor.shape[0], q_tensor.shape[1], q_tensor.shape[2], 1))
    else:
        lse = None

    kv_smax = k_tensor.shape[2]
    if act_seq != None and act_seq != -1:
        kv_smax = act_seq

    s2Inner = np.minimum(512, kv_smax)
    loop = np.ceil(kv_smax / s2Inner).astype(np.int32)
    tail = kv_smax - (loop - 1) * s2Inner
    print(f"s2Inner:{s2Inner} kv_smax:{kv_smax} loop:{loop} tail:{tail}")

    rowSum = np.zeros(shape=(q_tensor.shape[0], q_tensor.shape[1], q_tensor.shape[2], 1), dtype=np.float64)
    rowMax = np.full(shape=(q_tensor.shape[0], q_tensor.shape[1], q_tensor.shape[2], 1), fill_value=-np.inf,
                     dtype=np.float64)
    O_flash = np.zeros(shape=(q_tensor.shape[0], q_tensor.shape[1], q_tensor.shape[2], q_tensor.shape[3]),
                       dtype=np.float64)
    pse_cur = None
    mask_cur = None

    for i in range(loop):
        if i < loop - 1:
            seq_start = i * s2Inner
            seq_end = (i + 1) * s2Inner
        else:
            seq_start = i * s2Inner
            seq_end = i * s2Inner + tail
        k_cur = k_tensor_all[:, :, seq_start:seq_end, :]
        v_cur = v_tensor[:, :, seq_start:seq_end, :]
        if ifa_param['p_flag']:
            pse_cur = ifa_param['pse_cur'][:, :, :, seq_start:seq_end]
        if ifa_param['m_flag']:
            mask_tensor = ifa_param['mask_cur'].numpy()
            mask_cur = mask_tensor[:, :, :, seq_start:seq_end]

        qkBmmRes = np.matmul(q_tensor_all, k_cur.transpose(0, 1, 3, 2)).astype(np.float64)
        qkBmmRes = (qkBmmRes * scalar).astype(np.float64)

        if pse_cur is not None:
            qkBmmRes += pse_cur
        if mask_cur is not None:
            qkBmmRes[mask_cur.astype(np.bool_)] = -1000000000000

        rowmax_tmp = qkBmmRes.max(axis=3, keepdims=True)
        replace_idx = np.where(rowMax < rowmax_tmp)
        rowMax_old = copy.deepcopy(rowMax)
        rowMax[replace_idx] = rowmax_tmp[replace_idx]
        update_mul = np.exp(rowMax_old - rowMax)

        softmaxres = np.exp(qkBmmRes - rowMax).astype(np.float64)
        rowSum_A = softmaxres.sum(axis=3, keepdims=True).astype(np.float64)
        rowMax_A = (np.exp(rowmax_tmp - rowMax) + np.finfo(float).eps).astype(np.float64)
        rowSum = update_mul * rowSum + rowSum_A

        quantScale = 127 / rowMax_A

        softmaxres = (softmaxres * quantScale).astype(np.float64)
        softmaxres = np.round(softmaxres).astype(np.int8)

        bmm2Res = np.matmul(softmaxres.astype(np.float64), v_cur.astype(np.float64))
        bmm2Res = bmm2Res.astype(np.float64) * rowMax_A.astype(np.float64)

        O_flash = O_flash * update_mul + bmm2Res

    O = O_flash / rowSum / 127

    if ifa_param['m_flag']:
        if mask_tensor is not None:
            if ifa_param['sparse_mode'] == 3 or act_seq_q > 1:
                for i in range(mask_tensor.shape[2]):
                    if mask_tensor[:, :, i, :].all() == 1:
                        O[:, :, i, :] = 0
                        if lse is not None:
                            lse[:, :, i, :] = np.inf

    return O, lse


def _t_indequant_ds_act_gpu_base(ifa_param):
    print("def _t_indequant_ds_act_gpu_base(ifa_param): enter")
    q_tensor = torch.from_numpy(ifa_param['q_tensor_cur']).to(torch.int32).cuda().requires_grad_(False)
    k_tensor = torch.from_numpy(ifa_param['k_sub_tensor']).to(torch.int32).cuda().requires_grad_(False)
    v_tensor = torch.from_numpy(ifa_param['v_sub_tensor']).to(torch.int32).cuda().requires_grad_(False)
    q_rope_tensor = torch.from_numpy(ifa_param['q_rope_tensor_cur']).to(torch.bfloat16).cuda().requires_grad_(False)
    k_rope_tensor = torch.from_numpy(ifa_param['k_rope_tensor_cur']).cuda().requires_grad_(False)

    scalar = np.float32(ifa_param['scaleValue'])
    act_seq = ifa_param['act_seq']
    act_seq_q = ifa_param['act_seq_q']
    lse_flag = ifa_param['softmax_lse_flag']
    print(
        f"q_shape:{q_tensor.shape} k_shape:{k_tensor.shape} v_shape:{v_tensor.shape} scalar:{scalar} act:{act_seq} act_q:{act_seq_q}")

    dequantScale1 = torch.from_numpy(np.array(ifa_param['dequantScale1_cur'])).to(torch.float32).cuda().requires_grad_(
        False)  # [1, 128, 1, 1]
    dequantScale2 = torch.from_numpy(ifa_param['dequantScale2']).to(torch.float32).cuda().requires_grad_(False)  # float

    scaleQ = dequantScale1 / dequantScale2
    scaleK = dequantScale2

    #q_rope_tensor = q_rope_tensor.to(torch.float32) * dequantScale1

    if lse_flag:
        lse = np.zeros((q_tensor.shape[0], q_tensor.shape[1], q_tensor.shape[2], 1))
    else:
        lse = None

    kv_smax = k_tensor.shape[2]
    if act_seq != None and act_seq != -1:
        kv_smax = act_seq

    s2Inner = np.minimum(512, kv_smax)
    loop = np.ceil(kv_smax / s2Inner).astype(np.int32)
    tail = kv_smax - (loop - 1) * s2Inner
    print(f"s2Inner:{s2Inner} kv_smax:{kv_smax} loop:{loop} tail:{tail}")

    rowSum = torch.from_numpy(np.zeros(shape=(q_tensor.shape[0], q_tensor.shape[1], q_tensor.shape[2], 1),
                                       dtype=np.float32)).to(torch.float32).cuda().requires_grad_(False)
    rowMax = torch.from_numpy(
        np.full(shape=(q_tensor.shape[0], q_tensor.shape[1], q_tensor.shape[2], 1), fill_value=-np.inf,
                dtype=np.float32)).to(torch.float32).cuda().requires_grad_(False)
    # O_flash = torch.from_numpy(np.zeros(shape=(q_tensor.shape[0], q_tensor.shape[1], q_tensor.shape[2], q_tensor.shape[3]),
    #                    dtype=np.float32)).to(torch.float16).cuda().requires_grad_(False)
    O_flash = torch.from_numpy(
        np.zeros(shape=(q_tensor.shape[0], q_tensor.shape[1], q_tensor.shape[2], q_tensor.shape[3]),
                 dtype=np.float32)).to(torch.float32).cuda().requires_grad_(False)
    pse_cur = None
    mask_cur = None

    for i in range(loop):
        if i < loop - 1:
            seq_start = i * s2Inner
            seq_end = (i + 1) * s2Inner
        else:
            seq_start = i * s2Inner
            seq_end = i * s2Inner + tail
        k_cur = k_tensor[:, :, seq_start:seq_end, :]
        k_rope_cur = k_rope_tensor[:, :, seq_start:seq_end, :]
        v_cur = v_tensor[:, :, seq_start:seq_end, :]
        if ifa_param['p_flag']:
            pse_cur = torch.from_numpy(ifa_param['pse_cur'][:, :, :, seq_start:seq_end]).to(
                torch.float32).cuda().requires_grad_(False)
        if ifa_param['m_flag']:
            mask_tensor = torch.from_numpy(ifa_param['mask_cur'].numpy()).to(torch.float32).cuda().requires_grad_(False)
            mask_cur = mask_tensor[:, :, :, seq_start:seq_end]

        qkBmmRes_nope = torch.matmul(q_tensor.to(torch.float32), k_cur.permute(0, 1, 3, 2).to(torch.float32)).to(
            torch.int32)
        qkBmmRes_rope = torch.matmul(q_rope_tensor.to(torch.float32),
                                     k_rope_cur.permute(0, 1, 3, 2).to(torch.float32)).to(torch.float32)
        # qkBmmRes_rope = torch.matmul(q_rope_tensor.to(torch.bfloat16), k_rope_cur.permute(0, 1, 3, 2).to(torch.bfloat16)).to(torch.float32)

        qkBmmRes = qkBmmRes_nope.to(torch.float32) * scaleK.to(torch.float32) * scaleQ.to(torch.float32) + qkBmmRes_rope
        qkBmmRes = (qkBmmRes * scalar).to(torch.float32)

        if pse_cur is not None:
            qkBmmRes += pse_cur
        if mask_cur is not None:
            qkBmmRes[mask_cur.to(torch.bool)] = -1000000000000

        rowmax_tmp = torch.max(qkBmmRes, dim=3, keepdim=True)[0]
        replace_idx = np.where((rowMax < rowmax_tmp).cpu().numpy())
        rowMax_old = copy.deepcopy(rowMax)
        rowMax[replace_idx] = rowmax_tmp[replace_idx]
        update_mul = torch.exp(rowMax_old - rowMax)

        softmaxres = torch.exp(qkBmmRes - rowMax).to(torch.float32)
        rowSum_A = torch.sum(softmaxres, dim=3, keepdim=True).to(torch.float32)
        rowMax_A = (torch.exp(rowmax_tmp - rowMax) + np.finfo(np.float32).eps).to(torch.float32)
        rowSum = update_mul * rowSum + rowSum_A

        quantScale = 127 / rowMax_A

        softmaxres = (softmaxres * quantScale).to(torch.float32)
        # softmaxres = torch.round(softmaxres).to(torch.int8)
        softmaxres = torch.round(softmaxres.to(torch.float16)).to(torch.int8)

        bmm2Res = torch.matmul(softmaxres.to(torch.float32), v_cur.to(torch.float32)).to(torch.int32)
        bmm2Res = bmm2Res.to(torch.float32) * rowMax_A.to(torch.float32)

        O_flash = O_flash * update_mul + bmm2Res
        # bmm2Res = bmm2Res.to(torch.float16) * rowMax_A.to(torch.float16)

        # O_flash = O_flash.to(torch.float16) * update_mul.to(torch.float16) + bmm2Res.to(torch.float16)

    O = O_flash / rowSum / 127 * scaleK
    O = O.cpu()

    if ifa_param['m_flag']:
        if mask_tensor is not None:
            if ifa_param['sparse_mode'] == 3 or act_seq_q > 1:
                for i in range(mask_tensor.shape[2]):
                    if mask_tensor[:, :, i, :].all() == 1:
                        O[:, :, i, :] = 0
                        if lse is not None:
                            lse[:, :, i, :] = np.inf

    return O, lse


def _t_ifaattention_act(ifa_param, device="cpu"):
    print("def _t_ifaattention_act(ifa_param):")
    # 低精度标记位，默认走高精度流程，lowprecision_flag = False
    lowprecision_flag = False

    # 如果跑one命令或者innerprecise=1且跑bm命令时，改为走低精度流程，lowprecision_flag = True
    if ifa_param['action_type'] == "one" or (ifa_param['action_type'] == "bm" and ifa_param['innerprecise'] == 1):
        lowprecision_flag = True
        print("低精度模式...")
    q_tensor = ifa_param['q_tensor_cur']
    k_tensor = ifa_param['k_sub_tensor']
    v_tensor = ifa_param['v_sub_tensor']
    m_tensor = ifa_param['mask_cur']
    m_dtype = ifa_param['m_dtype']
    p_tensor = ifa_param['pse_cur']
    scalar = ifa_param['scaleValue']
    act_seq = ifa_param['act_seq']
    act_seq_q = ifa_param['act_seq_q']
    sinner = ifa_param['sinner']
    lse_flag = ifa_param['softmax_lse_flag']
    if ifa_param['padding_flag']:
        s_begin = k_tensor.shape[2] - act_seq - ifa_param['padding_size']
        s_end = k_tensor.shape[2] - ifa_param['padding_size']
        print(f"left padding--- s_begin:{s_begin}, s_end:{s_end}")

    print(
        f"q_shape:{q_tensor.shape} k_shape:{k_tensor.shape} v_shape:{v_tensor.shape} scalar:{scalar} act:{act_seq} act_q:{act_seq_q}")
    # matmul_dtype 赋值
    matmul_dtype = np.float32
    if ifa_param['in_quant_flag']:
        # 输入int8 场景，mm使用int32
        matmul_dtype = np.float32
    # elif ifa_param['out_quant_flag']:
    # 输出int8场景，mm使用float16
    #    matmul_dtype = np.float16
    else:
        # 非量化场景，mm使用float32
        matmul_dtype = np.float32

    kv_smax = k_tensor.shape[2]
    if act_seq is None or act_seq == -1:
        k_cur = k_tensor
        v_cur = v_tensor
    else:
        kv_smax = act_seq
        if ifa_param['padding_flag']:
            k_cur = k_tensor[:, :, s_begin:s_end, :]
            v_cur = v_tensor[:, :, s_begin:s_end, :]
            if ifa_param['kv_quant_pto_flag']:
                ifa_param['antiquantScale_cur'] = ifa_param['antiquantScale_cur'][:, :, s_begin:s_end]
                ifa_param['antiquantOffset_cur'] = ifa_param['antiquantOffset_cur'][:, :, s_begin:s_end]
            if ifa_param['k_antiquant_pto_flag'] or ifa_param['k_antiquant_ptopa_flag']:
                ifa_param['k_antiquantScale_cur'] = ifa_param['k_antiquantScale_cur'][:, :, s_begin:s_end]
                ifa_param['k_antiquantOffset_cur'] = ifa_param['k_antiquantOffset_cur'][:, :, s_begin:s_end]
            if ifa_param['v_antiquant_pto_flag'] or ifa_param['v_antiquant_ptopa_flag']:
                ifa_param['v_antiquantScale_cur'] = ifa_param['v_antiquantScale_cur'][:, :, s_begin:s_end]
                ifa_param['v_antiquantOffset_cur'] = ifa_param['v_antiquantOffset_cur'][:, :, s_begin:s_end]

            if ifa_param['k_antiquant_ptoh_flag'] or ifa_param['k_antiquant_ptohpa_flag']:
                ifa_param['k_antiquantScale_cur'] = ifa_param['k_antiquantScale_cur'][:, :, s_begin:s_end]
                ifa_param['k_antiquantOffset_cur'] = ifa_param['k_antiquantOffset_cur'][:, :, s_begin:s_end]
            if ifa_param['v_antiquant_ptoh_flag'] or ifa_param['v_antiquant_ptohpa_flag']:
                ifa_param['v_antiquantScale_cur'] = ifa_param['v_antiquantScale_cur'][:, :, s_begin:s_end]
                ifa_param['v_antiquantOffset_cur'] = ifa_param['v_antiquantOffset_cur'][:, :, s_begin:s_end]

            if ifa_param['kv_quant_ptog_flag']:
                ifa_param['antiquantScale_cur'] = ifa_param['antiquantScale_cur'][:, :, :, s_begin:s_end, :]
                ifa_param['antiquantOffset_cur'] = ifa_param['antiquantOffset_cur'][:, :, :, s_begin:s_end, :]
            if ifa_param['k_antiquant_ptog_flag']:
                ifa_param['k_antiquantScale_cur'] = ifa_param['k_antiquantScale_cur'][:, :, :, s_begin:s_end, :]
                ifa_param['k_antiquantOffset_cur'] = ifa_param['k_antiquantOffset_cur'][:, :, :, s_begin:s_end, :]
            if ifa_param['v_antiquant_ptog_flag']:
                ifa_param['v_antiquantScale_cur'] = ifa_param['v_antiquantScale_cur'][:, :, :, s_begin:s_end, :]
                ifa_param['v_antiquantOffset_cur'] = ifa_param['v_antiquantOffset_cur'][:, :, :, s_begin:s_end, :]
        else:
            k_cur = k_tensor[:, :, :act_seq, :]
            v_cur = v_tensor[:, :, :act_seq, :]
            if ifa_param['kv_quant_pto_flag']:
                ifa_param['antiquantScale_cur'] = ifa_param['antiquantScale_cur'][:, :, :act_seq]
                ifa_param['antiquantOffset_cur'] = ifa_param['antiquantOffset_cur'][:, :, :act_seq]
            if ifa_param['k_antiquant_pto_flag'] or ifa_param['k_antiquant_ptopa_flag']:
                ifa_param['k_antiquantScale_cur'] = ifa_param['k_antiquantScale_cur'][:, :, :act_seq]
                ifa_param['k_antiquantOffset_cur'] = ifa_param['k_antiquantOffset_cur'][:, :, :act_seq]
            if ifa_param['v_antiquant_pto_flag'] or ifa_param['v_antiquant_ptopa_flag']:
                ifa_param['v_antiquantScale_cur'] = ifa_param['v_antiquantScale_cur'][:, :, :act_seq]
                ifa_param['v_antiquantOffset_cur'] = ifa_param['v_antiquantOffset_cur'][:, :, :act_seq]

            if ifa_param['k_antiquant_ptoh_flag'] or ifa_param['k_antiquant_ptohpa_flag']:
                ifa_param['k_antiquantScale_cur'] = ifa_param['k_antiquantScale_cur'][:, :, :act_seq]
                ifa_param['k_antiquantOffset_cur'] = ifa_param['k_antiquantOffset_cur'][:, :, :act_seq]
            if ifa_param['v_antiquant_ptoh_flag'] or ifa_param['v_antiquant_ptohpa_flag']:
                ifa_param['v_antiquantScale_cur'] = ifa_param['v_antiquantScale_cur'][:, :, :act_seq]
                ifa_param['v_antiquantOffset_cur'] = ifa_param['v_antiquantOffset_cur'][:, :, :act_seq]

            if ifa_param['kv_quant_ptog_flag']:
                ifa_param['antiquantScale_cur'] = ifa_param['antiquantScale_cur'][:, :, :, :act_seq, :]
                ifa_param['antiquantOffset_cur'] = ifa_param['antiquantOffset_cur'][:, :, :, :act_seq, :]
            if ifa_param['k_antiquant_ptog_flag']:
                ifa_param['k_antiquantScale_cur'] = ifa_param['k_antiquantScale_cur'][:, :, :, :act_seq, :]
                ifa_param['k_antiquantOffset_cur'] = ifa_param['k_antiquantOffset_cur'][:, :, :, :act_seq, :]
            if ifa_param['v_antiquant_ptog_flag']:
                ifa_param['v_antiquantScale_cur'] = ifa_param['v_antiquantScale_cur'][:, :, :, :act_seq, :]
                ifa_param['v_antiquantOffset_cur'] = ifa_param['v_antiquantOffset_cur'][:, :, :, :act_seq, :]

    # print(f"k_cur_shape:{k_cur.shape} v_cur_shape:{v_cur.shape} ")
    # print("start matmul1")
    if ifa_param['in_quant_flag'] and ifa_param['deep_seek_flag']:
        if device == "gpu":
            # 走base流程，生成GPU base数据。
            print("bmm2Res, lse = _t_indequant_ds_act_gpu_base(ifa_param)")
            bmm2Res, lse = _t_indequant_ds_act_gpu_base(ifa_param)
        elif ifa_param['action_type'] == 'bm_output_gold':
            # 走golden流程，生成高精度cpu golden数据。
            print("bmm2Res, lse = _t_indequant_ds_act_golden(ifa_param)")
            bmm2Res, lse = _t_indequant_ds_act_golden(ifa_param)
        else:
            # 走DaTo流程，生成cpu仿npu计算的golden数据。
            print("bmm2Res, lse = _t_indequant_ds_act(ifa_param)")
            bmm2Res, lse = _t_indequant_ds_act(ifa_param)
    else:
        if ifa_param["k_antiquant_flag"] and ifa_param["v_antiquant_flag"]:
            k_cur, v_cur = kv_split_antiquant(k_cur, v_cur, ifa_param['k_antiquantScale_cur'],
                                              ifa_param['k_antiquantOffset_cur'],
                                              ifa_param['v_antiquantScale_cur'], ifa_param['v_antiquantOffset_cur'],
                                              ifa_param['k_antiquant_mode'], ifa_param['v_antiquant_mode'],
                                              lowprecision_flag, ifa_param['k_dtype'], ifa_param['q_dtype'])

        elif ifa_param['kv_quant_flag']:
            # print("[INFO]begin kv_quant")
            k_cur, v_cur = antiquant(k_cur, v_cur, ifa_param['antiquantScale_cur'], ifa_param['antiquantOffset_cur'],
                                     ifa_param['kv_quant_pto_flag'], lowprecision_flag, ifa_param['k_dtype'],ifa_param['q_dtype'])

        qkBmmRes = np.matmul(q_tensor, k_cur.transpose(0, 1, 3, 2), dtype=matmul_dtype)
        # print(f"==========================q_tensor:{q_tensor}===============================")
        # print(f"==========================k_cur:{k_cur}===============================")
        # print(f"==========================qkBmmRes:{qkBmmRes}===============================")
        if ifa_param['in_quant_flag'] and not ifa_param['deep_seek_flag']:
            # print("start dequant after matmul1")
            qkBmmRes = dequant(qkBmmRes, ifa_param['dequantScale1'], None)
            # print(f"==========================qkBmmRes after dequant:{qkBmmRes}===============================")
        # print("end matmul1")
        qkEleRes = qkBmmRes * scalar

        if ifa_param['p_flag']:
            # print ("begin add pse")
            if act_seq is None or act_seq == -1:
                pse_cur = ifa_param['pse_cur']
            else:
                if ifa_param['padding_flag']:
                    pse_cur = ifa_param['pse_cur'][:, :, :, s_begin:s_end]
                else:
                    pse_cur = ifa_param['pse_cur'][:, :, :, : act_seq]
            qkEleRes = qkEleRes + pse_cur
        mask_cur = None
        if ifa_param['m_flag']:
            # print("begin handle mask")
            mask_tensor = ifa_param['mask_cur'].numpy()
            if act_seq is None:
                mask_cur = mask_tensor
            else:
                if ifa_param['padding_flag']:
                    mask_cur = mask_tensor[:, :, :, s_begin:s_end]
                else:
                    mask_cur = mask_tensor[:, :, :, : act_seq]
            # print(f"============mask:{mask_cur.shape}==================")
            # print(f"============qkEleRes:{qkEleRes.shape}===============================")

            if m_dtype == 'float16':
                qkEleRes = qkEleRes + mask_cur * -10000
            else:
                # qkEleRes = qkEleRes + mask_cur * -65504
                qkEleRes[mask_cur.astype(np.bool_)] = -1000000000000
            # print(f"==========================qkEleRes after mask:{qkEleRes}===============================")
        # 处理输入量化（带s切分）
        if ifa_param['in_quant_flag'] and not ifa_param['deep_seek_flag']:
            bmm2Res = np.zeros(q_tensor.shape, dtype=np.float16)  # B, S, N, D
            s_loop_times = int((kv_smax + sinner - 1) / sinner)
            # =================================
            # print("[INFO]Total s_loop_times:", s_loop_times)
            if s_loop_times > 1:
                N = q_tensor.shape[1]
                for nIdx in range(N):
                    # print(f"[INFO]输入量化+sinner切分！")
                    for sinner_loop_times in range(s_loop_times):
                        # print(f"[INFO]Current n:{nIdx}, sinner_loop_times:{sinner_loop_times}")
                        sinner_tail = sinner
                        # 最后一次loop
                        if sinner_loop_times == (kv_smax - 1) // sinner:
                            if kv_smax == sinner:
                                sinner_tail = sinner
                            else:
                                sinner_tail = kv_smax % sinner
                        qk_cur_loop = qkEleRes[:, nIdx:(nIdx + 1), :, (sinner_loop_times * sinner):(
                                (sinner_loop_times) * sinner + sinner_tail)]  # 1, 1, 1, Sinner
                        v_cur_loop = v_cur[:, nIdx:(nIdx + 1),
                                     (sinner_loop_times * sinner):((sinner_loop_times) * sinner + sinner_tail),
                                     :]  # 1, 1, Sinner, 1
                        # 处理第一次loop
                        if sinner_loop_times == 0:
                            softmax_max, softmax_sum, softmax_res, exp_max_res = softmax_flash(qk_cur_loop)
                            # print(f"==========================softmax_res:{softmax_res}===============================")
                            # print("start dequant before matmul2")
                            softmax_res = quant(softmax_res, ifa_param['quantScale1'], 0)
                            # print(f"==========================softmax_res after quant:{softmax_res}===============================")
                            # print("start matmul2")
                            bmm2Res_1 = np.matmul(softmax_res, v_cur_loop, dtype=matmul_dtype)
                            # print(f"==========================bmm2Res:{bmm2Res}===============================")
                            # print("start dequant after matmul2")
                            bmm2Res_1 = dequant(bmm2Res_1, ifa_param['dequantScale2'], None)
                            # print(f"==========================bmm2Res after dequant:{bmm2Res}===============================")
                            max_front = softmax_max
                            sum_front = softmax_sum
                            o_front = bmm2Res_1
                        # 处理后续场景
                        else:
                            softmax_max, softmax_sum, softmax_res, exp_max_res = softmax_flash(qk_cur_loop,
                                                                                               max_front=max_front,
                                                                                               sum_front=sum_front,
                                                                                               update=True)
                            softmax_res = quant(softmax_res, ifa_param['quantScale1'], 0)
                            o_front_update = o_front.astype(np.float32) * exp_max_res.astype(np.float32)  # Mul
                            o_front_update = o_front_update.astype(np.float16)
                            bmm2Res_1 = np.matmul(softmax_res, v_cur_loop, dtype=matmul_dtype)
                            bmm2Res_1 = dequant(bmm2Res_1, ifa_param['dequantScale2'], None)
                            bmm2Res_1 = o_front_update.astype(np.float32) + bmm2Res_1.astype(np.float32)
                            max_front = softmax_max
                            sum_front = softmax_sum
                            o_front = bmm2Res_1.astype(np.float16)
                    # 按n拼接
                    bmm2Res[:, nIdx:(nIdx + 1), :, :] = o_front
            # 量化非切分场景
            else:
                print(f"[INFO]输入量化+无sinner切分！")
                softmax_res = softmax(qkEleRes)
                softmax_res = quant(softmax_res, ifa_param['quantScale1'], 0)
                # print(f"==========================softmax_res:{softmax_res}===============================")
                bmm2Res = np.matmul(softmax_res, v_cur, dtype=matmul_dtype)
                bmm2Res = dequant(bmm2Res, ifa_param['dequantScale2'], None)
        # 非输入量化场景
        else:
            if kv_smax <= 512 and ifa_param["softmaxV1_flag"] == True:
                softmax_res, softmax_sum, softmax_max = softmaxv1(qkEleRes)
                # print(f"==========================softmax_res:{softmax_res}===============================")
                if ifa_param['q_dtype'] == "float16":
                    bmm2Res = np.matmul(softmax_res.astype(np.float16).astype(np.float32), v_cur,
                                        dtype=matmul_dtype)
                elif ifa_param['q_dtype'] == "bfloat16":
                    bmm2Res = np.matmul(softmax_res.astype(tf.bfloat16.as_numpy_dtype).astype(np.float32), v_cur,
                                        dtype=matmul_dtype)
                else:
                    bmm2Res = np.matmul(softmax_res, v_cur, dtype=matmul_dtype)
            else:
                softmax_res, softmax_sum, softmax_max = softmax(qkEleRes)
                if ifa_param['q_dtype'] == "float16":
                    bmm2Res = np.matmul(softmax_res.astype(np.float16).astype(np.float32), v_cur,
                                        dtype=matmul_dtype) / softmax_sum
                elif ifa_param['q_dtype'] == "bfloat16":
                    bmm2Res = np.matmul(softmax_res.astype(tf.bfloat16.as_numpy_dtype).astype(np.float32), v_cur,
                                        dtype=matmul_dtype) / softmax_sum
                else:
                    bmm2Res = np.matmul(softmax_res, v_cur, dtype=matmul_dtype) / softmax_sum
            if lse_flag:
                lse = np.log(softmax_sum) + softmax_max
            else:
                lse = None
            if mask_cur is not None:
                if ifa_param['sparse_mode'] == 3 or act_seq_q > 1:
                    for i in range(mask_cur.shape[2]):
                        if mask_cur[:, :, i, :].all() == 1:
                            bmm2Res[:, :, i, :] = 0
                for i in range(mask_cur.shape[2]):
                    if mask_cur[:, :, i, :].all() == 1:
                        if lse is not None:
                            lse[:, :, i, :] = np.inf
    # print("end matmul2")
    # print(f"return shape:{bmm2Res.shape}")
    return bmm2Res, lse


def _t_ifaattention_act_gpu(ifa_param):
    q_tensor = torch.from_numpy(ifa_param['q_tensor_cur']).cuda().requires_grad_(False)
    k_tensor = torch.from_numpy(ifa_param['k_sub_tensor']).cuda().requires_grad_(False)
    v_tensor = torch.from_numpy(ifa_param['v_sub_tensor']).cuda().requires_grad_(False)
    if ifa_param['q_dtype'] == 'bfloat16':
        q_tensor = q_tensor.to(torch.bfloat16)
    else:
        q_tensor = q_tensor.to(torch.float32)
    if ifa_param['k_dtype'] == 'bfloat16':
        k_tensor = k_tensor.to(torch.bfloat16)
        v_tensor = v_tensor.to(torch.bfloat16)
    else:
        k_tensor = k_tensor.to(torch.float32)
        v_tensor = v_tensor.to(torch.float32)

    scalar = ifa_param['scaleValue']
    act_seq = ifa_param['act_seq']
    act_seq_q = ifa_param['act_seq_q']
    lse_flag = ifa_param['softmax_lse_flag']
    if ifa_param['padding_flag']:
        s_begin = k_tensor.shape[2] - act_seq - ifa_param['padding_size']
        s_end = k_tensor.shape[2] - ifa_param['padding_size']
        print(f"left padding--- s_begin:{s_begin}, s_end:{s_end}")

    print(
        f"q_shape:{q_tensor.shape} k_shape:{k_tensor.shape} v_shape:{v_tensor.shape} scalar:{scalar} act:{act_seq} act_q:{act_seq_q}")

    if act_seq is None or act_seq == -1:
        k_cur = k_tensor
        v_cur = v_tensor
    else:
        if ifa_param['padding_flag']:
            k_cur = k_tensor[:, :, s_begin:s_end, :]
            v_cur = v_tensor[:, :, s_begin:s_end, :]
        else:
            k_cur = k_tensor[:, :, :act_seq, :]
            v_cur = v_tensor[:, :, :act_seq, :]

    # print(f"k_cur_shape:{k_cur.shape} v_cur_shape:{v_cur.shape} ")
    k_cur = k_cur.permute([0, 1, 3, 2])
    print(f"start matmul1.q:{q_tensor.shape}|{q_tensor.dtype} k:{k_cur.shape}|{k_cur.dtype} ")
    qkBmmRes = torch.matmul(q_tensor.to(torch.float32), k_cur.to(torch.float32))
    print(f"matmul1 end. qkBmmRes:{qkBmmRes.shape}|{qkBmmRes.dtype}")
    # print("end matmul1")
    qkEleRes = qkBmmRes * scalar
    if ifa_param['p_flag']:
        print("begin add pse")
        p_tensor = torch.from_numpy(ifa_param['pse_cur']).cuda().requires_grad_(False)
        if act_seq is None or act_seq == -1:
            pse_cur = p_tensor
        else:
            if ifa_param['padding_flag']:
                pse_cur = p_tensor[:, :, :, s_begin:s_end]
            else:
                pse_cur = p_tensor[:, :, :, : act_seq]
        qkEleRes = qkEleRes + pse_cur
    if ifa_param['m_flag']:
        print("begin handle mask")
        m_tensor = ifa_param['mask_cur'].cuda().requires_grad_(False)
        m_dtype = ifa_param['m_dtype']
        if act_seq is None:
            mask_cur = m_tensor
        else:
            if ifa_param['padding_flag']:
                mask_cur = m_tensor[:, :, :, s_begin:s_end]
            else:
                mask_cur = m_tensor[:, :, :, : act_seq]
        print(f"============mask:{mask_cur.shape}==================")
        # print(f"============qkEleRes:{qkEleRes.shape}===============================")
        if m_dtype == 'float16':
            qkEleRes = qkEleRes + mask_cur * -10000
        else:
            # qkEleRes = qkEleRes + mask_cur * -65504
            # qkEleRes[mask_cur.to(bool)] = -1000000000000
            qkEleRes[mask_cur.to(bool)] = -1000000000000
        # print(f"==========================qkEleRes after mask:{qkEleRes}===============================")

    print(f"start softmax. qkEleRes:{qkEleRes.shape}|{qkEleRes.dtype}")
    softmax_res, softmax_sum, softmax_max = _t_softmax(qkEleRes)
    softmax_res = softmax_res.to(q_tensor.dtype)
    print(f"start matmul2.softmax_res:{softmax_res.shape}|{softmax_res.dtype} v:{v_cur.shape}|{v_cur.dtype} ")
    bmm2Res = torch.matmul(softmax_res.to(torch.float32), v_cur.to(torch.float32)).cpu()
    print(f"matmul2 end. bmm2Res:{bmm2Res.shape}|{bmm2Res.dtype}")
    return bmm2Res


def str_to_bool_list(s: str):
    bool_list = []
    for item in s:
        if item == 0:
            bool_list.append(False)
        else:
            bool_list.append(True)
    return bool_list


def trans_input_dtype(input_dtype):
    if input_dtype == 'fp16':
        return 'float16'
    elif input_dtype == 'int8':
        return 'int8'
    elif input_dtype == 'uint8':
        return 'uint8'
    elif input_dtype == 'bf16':
        return 'bfloat16'
    elif input_dtype == 'bool':
        return 'bool'
    elif input_dtype == 'int32':
        return 'int32'
    elif input_dtype == 'fp32':
        return 'float32'
    elif input_dtype == 'int4':
        return 'int4'
    else:
        return input_dtype


def split_tensor_shape_by_b(input_list):
    # [[3,N,S,D]]-->[[1,N,S,D],[1,N,S,D],[1,N,S,D]]
    list_len = input_list[0][0]
    list_new = []
    for i in range(0, list_len):
        list_new_item = []
        list_new_item.append(1)
        list_new_item.append(input_list[0][1])
        list_new_item.append(input_list[0][2])
        list_new_item.append(input_list[0][3])
        list_new.append(list_new_item)
    # print(f"list_new:{list_new}")
    return list_new


def split_tensor_by_b(input_list):
    # tensor:[[3,N,S,D]]-->[[1,N,S,D],[1,N,S,D],[1,N,S,D]]
    input_tensor = input_list[0]
    split_data = np.split(input_tensor, input_tensor.shape[0])
    # print(f"split_data.type:{type(split_data)}")
    return split_data


def _trans_antiparam_to_2n1d(shape, tensor, layout, numKeyValueHeads, d):
    # pertensor
    if shape == [2]:
        new_tensor = np.zeros((2, numKeyValueHeads, 1, d))
        new_tensor[0, :, :, :] = tensor[0]
        new_tensor[1, :, :, :] = tensor[1]
    # per channel
    elif len(shape) == 2:
        d_num = int(shape[1] / numKeyValueHeads)
        print(f"[INFO]_trans_2h_to_2n1d: layout={layout}, h={shape[1]}, n={numKeyValueHeads} ,d(h/n)={d_num}")
        new_tensor = tensor.reshape(2, 1, numKeyValueHeads, d_num).transpose(0, 2, 1, 3)
    elif len(shape) == 3:
        print(f"[INFO]_trans_2nd_to_2n1d : layout={layout}, n={shape[1]} ,d={shape[2]}")
        new_tensor = tensor.reshape(2, 1, shape[1], shape[2]).transpose(0, 2, 1, 3)
    elif len(shape) == 4:
        if shape[1] == 1:
            print(f"[INFO]_trans_21nd_to_2n1d : layout={layout}, n={shape[2]} ,d={shape[3]}")
            new_tensor = tensor.transpose(0, 2, 1, 3)
        else:
            new_tensor = tensor
    else:
        print(f"[ERROR]_trans_antiparam_to_2n1d : len(shape)({len(shape)}) > 4")
        new_tensor = tensor
    return new_tensor


def _trans_antiparam_to_2n1d_for_gpu(shape, tensor, layout, numKeyValueHeads, d):
    # pertensor
    if shape == [2]:
        new_tensor = np.zeros((2, numKeyValueHeads, 1, d))
        new_tensor[0, :, :, :] = tensor[0]
        new_tensor[1, :, :, :] = tensor[1]
    # per channel
    elif len(shape) == 2:
        d_num = int(shape[1] / numKeyValueHeads)
        print(f"[INFO]_trans_2h_to_2n1d: layout={layout}, h={shape[1]}, n={numKeyValueHeads} ,d(h/n)={d_num}")
        new_tensor = tensor.reshape([2, 1, numKeyValueHeads, d_num]).transpose(1, 2)
    elif len(shape) == 3:
        n = shape[1]
        d = shape[2]
        print(f"[INFO]_trans_2nd_to_2n1d : layout={layout}, n={shape[1]} ,d={shape[2]}")
        new_tensor = tensor.reshape([2, 1, n, d]).transpose(1, 2)
    elif len(shape) == 4:
        if shape[1] == 1:
            print(f"[INFO]_trans_21nd_to_2n1d : layout={layout}, n={shape[2]} ,d={shape[3]}")
            new_tensor = tensor.transpose(1, 2)
        else:
            new_tensor = tensor
    else:
        print(f"[ERROR]_trans_antiparam_to_2n1d : len(shape)({len(shape)}) > 4")
        new_tensor = tensor
    return new_tensor


def _trans_antiparam_to_1n1d(shape, tensor, layout, numKeyValueHeads, d, mode):
    if mode == '0':
        # per tensor
        if shape == [1]:
            new_tensor = np.zeros((1, numKeyValueHeads, 1, d))
            new_tensor[0, :, :, :] = tensor[0]
        # per channel
        elif len(shape) == 1:
            h = shape[0]
            d_num = int(h / numKeyValueHeads)
            print(f"[INFO]_trans_h_to_1n1d: layout={layout}, h={h}, n={numKeyValueHeads} ,d(h/n)={d_num}")
            new_tensor = tensor.reshape(1, 1, numKeyValueHeads, d_num).transpose(0, 2, 1, 3)
        elif len(shape) == 2:
            if shape[0] == 1:
                h = shape[1]
                d_num = int(h / numKeyValueHeads)
                print(f"[INFO]_trans_1h_to_1n1d: layout={layout}, h={h}, n={numKeyValueHeads} ,d(h/n)={d_num}")
                new_tensor = tensor.reshape(1, 1, numKeyValueHeads, d_num).transpose(0, 2, 1, 3)
            else:
                print(f"[INFO]_trans_nd_to_1n1d : layout={layout}, n={shape[0]} ,d={shape[1]}")
                new_tensor = tensor.reshape(1, 1, shape[0], shape[1]).transpose(0, 2, 1, 3)
        elif len(shape) == 3:
            if shape[0] == 1:
                print(f"[INFO]_trans_1nd_to_1n1d : layout={layout}, n={shape[1]} ,d={shape[2]}")
                new_tensor = tensor.reshape(1, 1, shape[1], shape[2]).transpose(0, 2, 1, 3)
            else:
                print(f"[INFO]_trans_n1d_to_1n1d : layout={layout}, n={shape[0]} ,d={shape[2]}")
                new_tensor = tensor.reshape(1, 1, shape[0], shape[2]).transpose(0, 2, 1, 3)
        elif len(shape) == 4:
            if shape[1] == 1:
                print(f"[INFO]_trans_11nd_to_1n1d : layout={layout}, n={shape[2]} ,d={shape[3]}")
                new_tensor = tensor.transpose(0, 2, 1, 3)
            else:
                new_tensor = tensor
        else:
            print(f"[ERROR]_trans_antiparam_to_1n1d : len(shape)({len(shape)}) > 4")
            new_tensor = tensor
        return new_tensor
    elif mode == '2':
        # pertensor-head
        print(f"shape is n:{numKeyValueHeads}")
        new_tensor = np.zeros((1, numKeyValueHeads, 1, d))
        for i in range(numKeyValueHeads):
            new_tensor[:, i, :, :] = tensor[i]
        return new_tensor
    else:
        print(f"[INFO]_trans_antiparam_to_1n1d : pass！！")
        return tensor


def broadcast_kv_dequant_tensor(tensor, numKeyValueHeads, numheads, dtype=np.float16):
    if numKeyValueHeads == numheads:
        return tensor
    else:
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
    if numKeyValueHeads == numheads:
        return tensor
    else:
        factor = numheads // numKeyValueHeads
        shape = tensor.shape
        D = shape[3]
        if 'bfloat16' in str(dtype):
            tensor_new = np.zeros([1, numheads, 1, D], dtype=tf.bfloat16.as_numpy_dtype)
        else:
            tensor_new = np.zeros([1, numheads, 1, D], dtype=dtype)
        for i in range(numheads):
            j = i // factor
            tensor_new[:, i:i + 1, :, :] = tensor[:, j:j + 1, :, :]
        return tensor_new


# trans bns(kvn) -> bns(qn)
def broadcast_kv_split_dequant_tensor_ptoh(tensor, numKeyValueHeads, numheads, dtype=np.float32):
    if numKeyValueHeads == numheads:
        return tensor
    else:
        factor = numheads // numKeyValueHeads
        shape = tensor.shape
        B = shape[0]
        G = shape[1]
        S = shape[2]
        tensor_new = np.zeros([B, numheads, S], dtype=dtype)
        for i in range(numheads):
            j = i // factor
            tensor_new[:, i:i + 1, :] = tensor[:, j:j + 1, :]
        return tensor_new


def _t_increattention_bnsd(ifa_param):
    batch_value = ifa_param['q_shape_bnsd'][0]
    n = ifa_param['q_shape_bnsd'][1]
    qs = ifa_param['q_shape_bnsd'][2]
    k_tensor_list = ifa_param['k_tensor_list_bnsd']
    v_tensor_list = ifa_param['v_tensor_list_bnsd']
    k_shape_list = ifa_param['k_shape_list_bnsd']
    v_shape_list = ifa_param['v_shape_list_bnsd']
    actseqlens_q = ifa_param['actualSeqLengths_q']
    actseqlens = ifa_param['actualSeqLengths']
    actseqlens_size = len(actseqlens)
    print(
        f"B:{batch_value} k tensor number:{len(k_tensor_list)} v tensor number:{len(v_tensor_list)}  actseqlens_q:{actseqlens_q} actseqlens:{actseqlens}")
    # deepseek场景下，output的d轴需要复原
    out_shape = ifa_param['q_shape_bnsd']
    if ifa_param['deep_seek_flag']:
        if ifa_param['inputLayout'] in ["BNSD", "BSND", "BSND_NBSD", "BNSD_NBSD"]:
            out_shape[3] = ifa_param['out_shape'][3]
        elif ifa_param['inputLayout'] in ["TND", "TND_NTD"]:
            out_shape[3] = ifa_param['out_shape'][2]
        else:
            out_shape[3] = int(ifa_param['out_shape'][2] / ifa_param['numHeads'])
    y = np.zeros(out_shape, dtype=np.float32)
    lse = np.full([batch_value, n, qs, 1], np.inf)
    # 连续场景预处理
    if ifa_param['continue_flag']:
        # 处理连续场景：将单个tensor shape依据B值拆成列表
        k_shape_list = split_tensor_shape_by_b(k_shape_list)
        v_shape_list = split_tensor_shape_by_b(v_shape_list)
        # 处理连续场景：将单个tensor依据B值拆成列表
        k_tensor_list = split_tensor_by_b(k_tensor_list)
        v_tensor_list = split_tensor_by_b(v_tensor_list)

    for b_index in range(batch_value):
        prefix_len = 0
        ifa_param['batch_id'] = b_index
        if ifa_param['prefix_flag']:
            prefix_len = ifa_param['prefix_act_lens']
        # 处理实际参与计算的kv s
        if ifa_param['as_flag']:
            ifa_param['act_seq'] = act_seq = actseqlens[b_index] + prefix_len
        else:
            ifa_param['act_seq'] = act_seq = k_shape_list[b_index][2] + prefix_len

        if ifa_param['as_q_flag']:
            ifa_param['act_seq_q'] = act_seq_q = actseqlens_q[b_index]
        else:
            ifa_param['act_seq_q'] = act_seq_q = qs

        ifa_param['k_sub_shape'] = k_sub_shape = k_shape_list[b_index]
        ifa_param['v_sub_shape'] = v_sub_shape = v_shape_list[b_index]
        # print(f"==========================k_sub_shape:{k_sub_shape}===============================")
        if act_seq == 0 or act_seq_q == 0 or 0 in k_sub_shape or 0 in v_sub_shape:
            continue
        ifa_param['k_sub_tensor'] = k_tensor_list[b_index]
        ifa_param['v_sub_tensor'] = v_tensor_list[b_index]
        ifa_param['q_tensor_cur'] = ifa_param['q_tensor_bnsd'][b_index:(b_index + 1), :, :act_seq_q, :]
        if ifa_param['in_quant_flag']:
            ifa_param['dequantScale1_cur'] = ifa_param['dequantScale1'][b_index:(b_index + 1), :, :act_seq_q, :]
            ifa_param['q_rope_tensor_cur'] = ifa_param['q_rope_bnsd_tensor'][b_index:(b_index + 1), :, :act_seq_q, :]
            ifa_param['k_rope_tensor_cur'] = ifa_param['k_rope_bnsd_tensor'][b_index:(b_index + 1), :, :, :]

        # prefix拼接
        if ifa_param['prefix_flag']:
            k_sub_tensor_temp = np.concatenate((ifa_param['k_prefix_tensor'], ifa_param['k_sub_tensor']), axis=2)
            v_sub_tensor_temp = np.concatenate((ifa_param['v_prefix_tensor'], ifa_param['v_sub_tensor']), axis=2)
            ifa_param['k_sub_tensor'] = k_sub_tensor_temp
            ifa_param['v_sub_tensor'] = v_sub_tensor_temp
            print(ifa_param['k_sub_tensor'].shape)

        # 判断attenmask是否为空
        if not ifa_param['m_flag']:
            ifa_param['mask_cur'] = None
        else:
            ifa_param['mask_cur'] = ifa_param['m_tensor'][b_index:(b_index + 1), :, :act_seq_q, :]
        # 判断pse是否为空,如果非空,检查pse第一维是否为1：如果格式为1n1s,则直接传入下层计算;如果格式为bn1s,则按B拆分后进入下层。
        if not ifa_param['p_flag']:
            pse_cur = None
        elif ifa_param['p_tensor'].shape[0] == 1:
            pse_cur = ifa_param['p_tensor']
        else:
            pse_cur = ifa_param['p_tensor'][b_index:(b_index + 1), :, :, :]
        ifa_param['pse_cur'] = pse_cur
        # 伪量化Per token处理
        if ifa_param['kv_quant_pto_flag']:
            ifa_param['antiquantScale_cur'] = ifa_param['antiquantScale'][:, b_index:(b_index + 1), :]
            ifa_param['antiquantOffset_cur'] = ifa_param['antiquantOffset'][:, b_index:(b_index + 1), :]
        # ptog伪量化参数切分
        elif ifa_param['kv_quant_ptog_flag']:
            ifa_param['antiquantScale_cur'] = ifa_param['antiquantScale'][:, b_index:(b_index + 1), :, :, :]
            ifa_param['antiquantOffset_cur'] = ifa_param['antiquantOffset'][:, b_index:(b_index + 1), :, :, :]
        elif ifa_param['kv_quant_flag']:
            ifa_param['antiquantScale_cur'] = ifa_param['antiquantScale']
            ifa_param['antiquantOffset_cur'] = ifa_param['antiquantOffset']
        else:
            pass
        # kv分离伪量化处理
        if ifa_param['k_antiquant_pto_flag'] or ifa_param['k_antiquant_ptopa_flag']:
            ifa_param['k_antiquantScale_cur'] = ifa_param['k_antiquantScale'][:, b_index:(b_index + 1), :]
            ifa_param['k_antiquantOffset_cur'] = ifa_param['k_antiquantOffset'][:, b_index:(b_index + 1), :]
        elif ifa_param['k_antiquant_ptoh_flag'] or ifa_param['k_antiquant_ptohpa_flag']:
            ifa_param['k_antiquantScale_cur'] = ifa_param['k_antiquantScale'][b_index:(b_index + 1), :, :]
            ifa_param['k_antiquantOffset_cur'] = ifa_param['k_antiquantOffset'][b_index:(b_index + 1), :, :]
        elif ifa_param['k_antiquant_ptog_flag']:
            ifa_param['k_antiquantScale_cur'] = ifa_param['k_antiquantScale'][:, b_index:(b_index + 1), :, :, :]
            ifa_param['k_antiquantOffset_cur'] = ifa_param['k_antiquantOffset'][:, b_index:(b_index + 1), :, :, :]
        elif ifa_param['k_antiquant_flag']:
            ifa_param['k_antiquantScale_cur'] = ifa_param['k_antiquantScale']
            ifa_param['k_antiquantOffset_cur'] = ifa_param['k_antiquantOffset']
        else:
            pass
        if ifa_param['v_antiquant_pto_flag'] or ifa_param['v_antiquant_ptopa_flag']:
            ifa_param['v_antiquantScale_cur'] = ifa_param['v_antiquantScale'][:, b_index:(b_index + 1), :]
            ifa_param['v_antiquantOffset_cur'] = ifa_param['v_antiquantOffset'][:, b_index:(b_index + 1), :]
        elif ifa_param['v_antiquant_ptoh_flag'] or ifa_param['v_antiquant_ptohpa_flag']:
            ifa_param['v_antiquantScale_cur'] = ifa_param['v_antiquantScale'][b_index:(b_index + 1), :, :]
            ifa_param['v_antiquantOffset_cur'] = ifa_param['v_antiquantOffset'][b_index:(b_index + 1), :, :]
        elif ifa_param['v_antiquant_ptog_flag']:
            ifa_param['v_antiquantScale_cur'] = ifa_param['v_antiquantScale'][:, b_index:(b_index + 1), :, :, :]
            ifa_param['v_antiquantOffset_cur'] = ifa_param['v_antiquantOffset'][:, b_index:(b_index + 1), :, :, :]
        elif ifa_param['v_antiquant_flag']:
            ifa_param['v_antiquantScale_cur'] = ifa_param['v_antiquantScale']
            ifa_param['v_antiquantOffset_cur'] = ifa_param['v_antiquantOffset']
        else:
            pass

        if cfg_fk.device_type == 'gpu':
            if ifa_param['in_quant_flag'] and ifa_param['deep_seek_flag']:
                y[b_index:(b_index + 1), :, :act_seq_q, :], _ = _t_ifaattention_act(ifa_param, device="gpu")
            else:
                y[b_index:(b_index + 1), :, :act_seq_q, :] = _t_ifaattention_act_gpu(ifa_param)

        else:
            if ifa_param['as_q_flag']:
                y[b_index:(b_index + 1), :, :act_seq_q, :], lse[b_index:(b_index + 1), :, :act_seq_q, :] = _t_ifaattention_act(
                ifa_param)
            else:
                y[b_index:(b_index + 1), :, :act_seq_q, :], lse[b_index:(b_index + 1), :, :, :] = _t_ifaattention_act(
                    ifa_param)


    if ifa_param['out_quant_flag']:
        # print(f"==========================y_before_quant:{y}===============================")
        if ifa_param['out_quant_pc_flag']:
            y = quant_pc(y, ifa_param['quantScale2'], ifa_param['quantOffset2'])
        else:
            y = quant(y, ifa_param['quantScale2'][0], ifa_param['quantOffset2'][0])
        # print(f"==========================y_after_quant:{y}===============================")

    # print(lse)
    return y, lse


def _bsh_bnsd_to_bsnd_for_gpu(tensor, inputlayout, shape, numheads):
    B = N = S = H = D = None
    # print("input shape: ",shape)
    if inputlayout in ["BNSD", "BNSD_NBSD"]:
        B = int(shape[0])
        N = int(shape[1])
        S = int(shape[2])
        D = int(shape[3])
        return tensor.transpose(1, 2)
    elif inputlayout in ["BSH", "BSH_NBSD"]:
        B = int(shape[0])
        S = int(shape[1])
        H = int(shape[2])
        N = int(numheads)
        D = H // N
        out_tensor = tensor.reshape([B, S, N, D])
        return out_tensor
    elif inputlayout in ["BSND", "BSND_NBSD"]:
        return tensor
    elif inputlayout in ["TND", "TND_NTD"]:
        B = int(shape[0])
        N = int(shape[1])
        S = int(shape[2])
        D = int(shape[3])
        return tensor.transpose(1, 2)
    else:
        print("Unknown layout!")


def get_slopes(n_heads):
    n = 2 ** math.floor(math.log2(n_heads))
    m_0 = 2.0 ** (-8.0 / n)
    m = torch.pow(m_0, torch.arange(1, 1 + n))
    if n < n_heads:
        m_hat_0 = 2.0 ** (-4.0 / n)
        m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (n_heads - n), 2))
        m = torch.cat([m, m_hat])
    return m


def is_0_tensor(tensor_list):
    for i in tensor_list:
        if 0 in i:
            return True
    return False


def trans_to_1n1d(tensor, shape, d):
    if len(shape) == 4:
        # 1n1d->1n1d
        return tensor
    elif len(shape) == 2:
        # nd->1n1d
        n = shape[0]
        d = shape[1]
        new_tensor = tensor.reshape(1, n, 1, d)
        return new_tensor
    elif len(shape) == 1:
        # h->1n1d
        h = shape[0]
        if h % d != 0:
            print(f"[ERROR]trans_to_1n1d: h={h}, d={d}. Wrong input !")
            return None
        n = int(h / d)
        new_tensor = tensor.reshape(1, n, 1, d)
        return new_tensor
    else:
        print("[ERROR]trans_to_1n1d: Unknown input shape!")
        return None


def get_kvs_list(shape_list, layout):
    kvs_list = []
    for shape in shape_list:
        if layout == "BNSD":
            kvs = shape[2]
        else:
            kvs = shape[1]
        kvs_list.append(kvs)
    return kvs_list


def cut_padding_size(tensor, list, padding_size):
    shape = tensor.shape
    ms = shape[-1]
    for i in range(len(list)):
        cut_len = ms - padding_size - list[i]
        tensor[i:(i + 1), ..., :cut_len] = 1
    return tensor


# 融合表格参数获取函数
def get_param_fus(torch_tensor_list, params):
    ifa_param = {}
    ifa_param['normal_flag'] = True
    # ===参数获取===
    ifa_param['flag_list'] = str_to_bool_list(params['flaglist'])

    # >> attr info
    ifa_param['actualSeqLengths_q_raw'] = actualSeqLengths_q = params['actualseqlengths']
    ifa_param['actualSeqLengths_raw'] = actualSeqLengths = params['actualseqlengthskv']
    ifa_param['numHeads'] = numHeads = params['numheads']
    ifa_param['scaleValue'] = scaleValue = params['scalevalue']
    ifa_param['inputLayout'] = inputLayout = params['inputlayout']

    # >> q info
    ifa_param['q_shape'] = q_shape = params['shape_input'][0]
    ifa_param['q_dtype'] = q_dtype = trans_input_dtype(params['dtype_input'][0])
    ifa_param['q_tensor'] = torch_tensor_list[0]

    if inputLayout in ["TND", "TND_NTD"]:
        ifa_param['b'] = len(actualSeqLengths_q)
    else:
        ifa_param['b'] = ifa_param['q_shape'][0]

    # >> kv info
    # k和v的位置通过b计算
    k1_shape = params['shape_input'][1]
    kb1 = k1_shape[0]
    # 如果第一个K的B=1,则默认kv列表长度为b;如果第一个K的B!=1,则默认kv列表长度为1
    if kb1 == 1:
        k_shape_num = v_shape_num = ifa_param['b']
    else:
        k_shape_num = v_shape_num = 1

    ifa_param['k_num'] = k_shape_num
    ifa_param['k_start_index'] = k_start_index = 1
    ifa_param['k_end_index'] = k_end_index = int(k_shape_num)
    ifa_param['v_start_index'] = v_start_index = int(k_shape_num) + 1
    ifa_param['v_end_index'] = v_end_index = int(k_shape_num + v_shape_num)

    print(
        f"k_start_index:{k_start_index} k_end_index:{k_end_index} v_start_index:{v_start_index} v_end_index:{v_end_index}")

    ifa_param['k_shape_list'] = k_ori_shape_list = params['shape_input'][k_start_index:k_end_index + 1]
    ifa_param['v_shape_list'] = v_ori_shape_list = params['shape_input'][v_start_index:v_end_index + 1]
    ifa_param['k_dtype'] = k_dtype = trans_input_dtype(params['dtype_input'][1])
    ifa_param['v_dtype'] = v_dtype = trans_input_dtype(params['dtype_input'][v_start_index])
    ifa_param['k_tensor_list'] = torch_tensor_list[k_start_index:k_end_index + 1]
    ifa_param['v_tensor_list'] = torch_tensor_list[v_start_index:v_end_index + 1]

    # >> out info
    ifa_param['out_shape'] = q_shape
    ifa_param['out_dtype'] = trans_input_dtype(params['dtype_output'][0])

    # >> p info
    ifa_param['p_shape'] = p_shape = params['shape_input'][v_end_index + 1]
    ifa_param['p_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 1])
    ifa_param['p_tensor'] = p_tensor = torch_tensor_list[v_end_index + 1]

    # >> m info
    ifa_param['m_shape'] = m_shape = params['shape_input'][v_end_index + 2]
    ifa_param['m_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 2])
    ifa_param['m_tensor'] = m_tensor = torch_tensor_list[v_end_index + 2]

    # kv prefix
    ifa_param['k_prefix_shape'] = k_prefix_shape = params['shape_input'][v_end_index + 17]
    ifa_param['k_prefix_dtype'] = k_prefix_dtype = trans_input_dtype(params['dtype_input'][v_end_index + 17])
    ifa_param['k_prefix_tensor'] = k_prefix_tensor = torch_tensor_list[v_end_index + 17]

    ifa_param['v_prefix_shape'] = v_prefix_shape = params['shape_input'][v_end_index + 18]
    ifa_param['v_prefix_dtype'] = v_prefix_dtype = trans_input_dtype(params['dtype_input'][v_end_index + 18])

    ifa_param['v_prefix_tensor'] = v_prefix_tensor = torch_tensor_list[v_end_index + 18]

    # >> q rope info
    ifa_param['q_rope_shape'] = params['shape_input'][v_end_index + 21]
    ifa_param['q_rope_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 21])
    ifa_param['q_rope_tensor'] = torch_tensor_list[v_end_index + 21]

    # >> k rope info
    ifa_param['k_rope_shape'] = params['shape_input'][v_end_index + 22]
    ifa_param['k_rope_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 22])
    ifa_param['k_rope_tensor'] = torch_tensor_list[v_end_index + 22]
    ifa_param['k_rope_cache_shape'] = params['shape_input'][v_end_index + 23]

    # 当numKeyValueHeads传入0时，处理为与numHeads相等
    numKeyValueHeads = params['numkeyvalueheads'] if params['numkeyvalueheads'] != 0 else numHeads
    ifa_param['numKeyValueHeads'] = numKeyValueHeads
    ifa_param['blocksize'] = params['blocksize']
    ifa_param['innerprecise'] = params['innerprecise']
    ifa_param['antiquant_mode'] = str(params['antiquant_mode'])
    # kv antiquant分离
    ifa_param['k_antiquant_mode'] = str(params['k_antiquant_mode'])
    ifa_param['v_antiquant_mode'] = str(params['v_antiquant_mode'])

    ifa_param['pre_tokens'] = params['pretokens']
    ifa_param['next_tokens'] = params['nexttokens']
    ifa_param['sparse_mode'] = params['sparsemode']
    ifa_param['softmax_lse_flag'] = params['softmax_lse_flag']

    # >> quant info
    ifa_param['dequantScale1'] = params['range_input'][v_end_index + 3][0]
    ifa_param['quantScale1'] = params['range_input'][v_end_index + 4][0]
    ifa_param['dequantScale2'] = params['range_input'][v_end_index + 5][0]

    # V4
    ifa_param['query_quant_mode'] = str(params['query_quant_mode'])
    ifa_param['dequant_scale_query'] = params['range_input'][v_end_index + 24][0]

    ifa_param['quantScale2_shape'] = params['shape_input'][v_end_index + 6]
    ifa_param['quantScale2'] = torch_tensor_list[v_end_index + 6]

    ifa_param['quantOffset2_shape'] = params['shape_input'][v_end_index + 7]
    ifa_param['quantOffset2'] = torch_tensor_list[v_end_index + 7]

    # >> kv_s list
    ifa_param['kv_s_list'] = get_kvs_list(ifa_param['k_shape_list'], ifa_param['inputLayout'])

    # ===flag_list判断===
    ifa_param['p_flag'] = False
    ifa_param['m_flag'] = False
    ifa_param['as_flag'] = False
    ifa_param['as_q_flag'] = False
    ifa_param['pa_flag'] = False
    ifa_param['in_quant_flag'] = ifa_param['out_quant_flag'] = ifa_param['kv_quant_flag'] = False
    ifa_param['out_quant_offset_flag'] = ifa_param['kv_quant_offset_flag'] = ifa_param['kv_quant_pto_flag'] = False
    ifa_param['k_antiquant_flag'] = ifa_param['v_antiquant_flag'] = False
    ifa_param['k_antiquant_offset_flag'] = ifa_param['v_antiquant_offset_flag'] = False
    ifa_param['k_antiquant_pc_flag'] = ifa_param['v_antiquant_pc_flag'] = False
    ifa_param['k_antiquant_pto_flag'] = ifa_param['v_antiquant_pto_flag'] = False
    ifa_param['k_antiquant_pth_flag'] = ifa_param['v_antiquant_pth_flag'] = False
    ifa_param['k_antiquant_ptoh_flag'] = ifa_param['v_antiquant_ptoh_flag'] = False
    ifa_param['k_antiquant_ptopa_flag'] = ifa_param['v_antiquant_ptopa_flag'] = False
    ifa_param['k_antiquant_ptohpa_flag'] = ifa_param['v_antiquant_ptohpa_flag'] = False
    ifa_param['k_antiquant_ptog_flag'] = ifa_param['v_antiquant_ptog_flag'] = False
    ifa_param['kv_quant_ptog_flag'] = False
    ifa_param['continue_flag'] = False
    ifa_param['padding_flag'] = False
    ifa_param['out_quant_pc_flag'] = False
    ifa_param['deep_seek_flag'] = False
    ifa_param['prefix_flag'] = ifa_param['prefix_act_flag'] = False
    ifa_param['tnd_flag'] = False
    if inputLayout in ["TND", "TND_NTD"]:
        ifa_param['tnd_flag'] = True
    flag_list = str_to_bool_list(params['flaglist'])
    kv_batch_value = ifa_param['k_shape_list'][0][0]
    if ifa_param['k_dtype'] in ['fp4_e2m1', 'fp4_e1m2', 'float4_e1m2', 'float4_e2m1']:
        ifa_param['kv_quant_ptog_flag'] = True
    # 判断是否是连续场景
    if kv_batch_value >= 1 and len(ifa_param['k_shape_list']) == 1:
        ifa_param['continue_flag'] = True
    if not flag_list[0] or not flag_list[1] or not flag_list[2] or not flag_list[24]:
        ifa_param['normal_flag'] = False
        print('[WARNING]:异常-->  q/k/v/out不输入场景！')
    if 0 in q_shape:
        ifa_param['normal_flag'] = False
        print('[WARNING]:异常-->  q为空场景！')
    if 0 in k_ori_shape_list or len(k_ori_shape_list) == 0 or is_0_tensor(k_ori_shape_list):
        ifa_param['normal_flag'] = False
        print('[WARNING]:异常-->  k为空场景！')
    if 0 in v_ori_shape_list or len(v_ori_shape_list) == 0 or is_0_tensor(v_ori_shape_list):
        ifa_param['normal_flag'] = False
        print('[WARNING]:异常-->  v为空场景！')
    if flag_list[3]:
        ifa_param['p_flag'] = True
    if flag_list[4]:
        ifa_param['m_flag'] = True
    if flag_list[5]:
        ifa_param['as_q_flag'] = True
    if flag_list[6]:
        ifa_param['as_flag'] = True
    if flag_list[7] and flag_list[8] and flag_list[9]:
        ifa_param['in_quant_flag'] = True
    if flag_list[17] and flag_list[19] and flag_list[29] and flag_list[26]:
        ifa_param['in_quant_flag'] = True  # 等价于MLA 全量化 17: key scale 19:value scale 29: query scale 26: query_rope
    if flag_list[10]:
        ifa_param['out_quant_flag'] = True
        if flag_list[11]:
            ifa_param['out_quant_offset_flag'] = True
        else:
            # 当不传入quantOffset2时，也需要生成一个和scale一样大小的全0 Offset，用于后续计算
            ifa_param['quantOffset2'] = np.zeros(ifa_param['quantScale2_shape'], np.float32)
            ifa_param['quantOffset2_shape'] = ifa_param['quantScale2_shape']
        if ifa_param['quantScale2_shape'] != [1]:
            ifa_param['out_quant_pc_flag'] = True

    # 判断kv 分离的antiquant是否存在
    if flag_list[17]:
        ifa_param["k_antiquant_flag"] = True
        if flag_list[18]:
            ifa_param['k_antiquant_offset_flag'] = True
        if ifa_param['k_antiquant_mode'] == '0':
            ifa_param['k_antiquant_pc_flag'] = True
        if ifa_param['k_antiquant_mode'] == '1':
            ifa_param['k_antiquant_pto_flag'] = True
        if ifa_param['k_antiquant_mode'] == '2':
            ifa_param['k_antiquant_pth_flag'] = True
        if ifa_param['k_antiquant_mode'] == '3':
            ifa_param['k_antiquant_ptoh_flag'] = True
        if ifa_param['k_antiquant_mode'] == '4':
            ifa_param['k_antiquant_ptopa_flag'] = True
        if ifa_param['k_antiquant_mode'] == '5':
            ifa_param['k_antiquant_ptohpa_flag'] = True
        if ifa_param['k_antiquant_mode'] == '6':
            ifa_param['k_antiquant_ptog_flag'] = True
            ifa_param['kv_quant_ptog_flag'] = False
    if flag_list[19]:
        ifa_param["v_antiquant_flag"] = True
        if flag_list[20]:
            ifa_param['v_antiquant_offset_flag'] = True
        if ifa_param['v_antiquant_mode'] == '0':
            ifa_param['v_antiquant_pc_flag'] = True
        if ifa_param['v_antiquant_mode'] == '1':
            ifa_param['v_antiquant_pto_flag'] = True
        if ifa_param['v_antiquant_mode'] == '2':
            ifa_param['v_antiquant_pth_flag'] = True
        if ifa_param['v_antiquant_mode'] == '3':
            ifa_param['v_antiquant_ptoh_flag'] = True
        if ifa_param['v_antiquant_mode'] == '4':
            ifa_param['v_antiquant_ptopa_flag'] = True
        if ifa_param['v_antiquant_mode'] == '5':
            ifa_param['v_antiquant_ptohpa_flag'] = True
        if ifa_param['v_antiquant_mode'] == '6':
            ifa_param['v_antiquant_ptog_flag'] = True
            ifa_param['kv_quant_ptog_flag'] = False
    if flag_list[12] and not flag_list[17] and not flag_list[19]:
        ifa_param['kv_quant_flag'] = True
        if flag_list[13]:
            ifa_param['kv_quant_offset_flag'] = True
        if ifa_param['antiquant_mode'] == '1':
            ifa_param['kv_quant_pto_flag'] = True
    if flag_list[14]:
        ifa_param['pa_flag'] = True
        ifa_param['bt_shape'] = params['shape_input'][12]
        ifa_param['bt_dtype'] = trans_input_dtype(params['dtype_input'][12])
        ifa_param['block_table'] = torch_tensor_list[12]
        ifa_param['k_cache_shape'] = params['shape_input'][21]
        ifa_param['v_cache_shape'] = params['shape_input'][22]

    if flag_list[16] and ifa_param['as_flag'] and ifa_param['continue_flag'] and not ifa_param['pa_flag']:
        ifa_param['padding_flag'] = True
        ifa_param['padding_size'] = params['range_input'][v_end_index + 12][0]
        if ifa_param['padding_size'] < 0:
            ifa_param['padding_size'] = 0
    if flag_list[21] and flag_list[22]:
        ifa_param['prefix_flag'] = True
    elif not flag_list[21] and not flag_list[22]:
        ifa_param['prefix_flag'] = False
    else:
        ifa_param['normal_flag'] = False
        print('[WARNING]:异常--> prefix未成对出现！')
    if flag_list[23]:
        ifa_param['prefix_act_flag'] = True
        ifa_param['prefix_act_lens'] = params['prefix_act_lens'][0]

    if flag_list[26]:
        ifa_param['deep_seek_flag'] = True
        ifa_param['v_shape_list'] = v_ori_shape_list = params['shape_input'][k_start_index:k_end_index + 1]
        ifa_param['v_dtype'] = v_dtype = trans_input_dtype(params['dtype_input'][1])
        ifa_param['v_tensor_list'] = torch_tensor_list[k_start_index:k_end_index + 1]

    # 获取kv量化参数
    if ifa_param['kv_quant_flag']:
        ifa_param['antiquantscale_shape_raw'] = antiquantscale_shape_raw = params['shape_input'][v_end_index + 8]
        if 0 in ifa_param['antiquantscale_shape_raw']:
            ifa_param['normal_flag'] = False
            print('[WARNING]:异常-->  antiquantscale为空场景！')
        ifa_param['antiquantscale_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 8])
        ifa_param['antiquantscale_tensor_raw'] = torch_tensor_list[v_end_index + 8]
        if ifa_param['kv_quant_offset_flag']:
            ifa_param['antiquantoffset_shape_raw'] = antiquantoffset_shape_raw = params['shape_input'][v_end_index + 9]
            if 0 in antiquantoffset_shape_raw:
                ifa_param['normal_flag'] = False
                print('[WARNING]:异常-->  antiquantoffset为空场景！')
            ifa_param['antiquantoffset_tensor_raw'] = torch_tensor_list[v_end_index + 9]
        else:
            ifa_param['antiquantoffset_shape_raw'] = antiquantscale_shape_raw
            ifa_param['antiquantoffset_tensor_raw'] = torch.zeros(antiquantscale_shape_raw).numpy()
    # 获取kv分离的量化参数
    ifa_param['k_rope_antiquantscale_shape_raw'] = None
    ifa_param['k_rope_antiquantscale_dtype_raw'] = None
    ifa_param['k_rope_antiquantscale_tensor_raw'] = None
    if ifa_param['k_antiquant_flag']:
        ifa_param['k_antiquantscale_shape_raw'] = k_antiquantscale_shape_raw = params['shape_input'][v_end_index + 13]
        if 0 in ifa_param['k_antiquantscale_shape_raw']:
            ifa_param['normal_flag'] = False
            print('[WARNING]:异常-->  k_antiquantscale为空场景！')
        ifa_param['k_antiquantscale_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 13])
        ifa_param['k_antiquantscale_tensor_raw'] = torch_tensor_list[v_end_index + 13]
        if ifa_param['k_antiquant_offset_flag']:
            ifa_param['k_antiquantoffset_shape_raw'] = k_antiquantoffset_shape_raw = params['shape_input'][
                v_end_index + 14]
            ifa_param['k_antiquantoffset_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 14])
            if 0 in k_antiquantoffset_shape_raw:
                ifa_param['normal_flag'] = False
                print('[WARNING]:异常-->  k_antiquantoffset为空场景！')
            ifa_param['k_antiquantoffset_tensor_raw'] = torch_tensor_list[v_end_index + 14]
        else:
            ifa_param['k_antiquantoffset_shape_raw'] = k_antiquantscale_shape_raw
            ifa_param['k_antiquantoffset_dtype'] = ifa_param['k_antiquantscale_dtype']
            ifa_param['k_antiquantoffset_tensor_raw'] = torch.zeros(k_antiquantscale_shape_raw).numpy()
        # rope quant scale
        if len(ifa_param['flag_list']) > 28:
            ifa_param['k_rope_antiquantscale_shape_raw'] = params['shape_input'][v_end_index + 24]
            ifa_param['k_rope_antiquantscale_dtype_raw'] = trans_input_dtype(params['dtype_input'][v_end_index + 24])
            ifa_param['k_rope_antiquantscale_tensor_raw'] = torch_tensor_list[v_end_index + 24]
            # 创建一个空的rope offset
            ifa_param['k_rope_antiquantoffset_shape_raw'] = params['shape_input'][v_end_index + 24]
            ifa_param['k_rope_antiquantoffset_dtype_raw'] = trans_input_dtype(params['dtype_input'][v_end_index + 24])
            ifa_param['k_rope_antiquantoffset_tensor_raw'] = torch.zeros(ifa_param['k_rope_antiquantoffset_shape_raw']).numpy()

    if ifa_param['v_antiquant_flag']:
        if ifa_param['flag_list'][26]:
            ifa_param['v_antiquantscale_shape_raw'] = copy.deepcopy(ifa_param['k_antiquantscale_shape_raw'])
            ifa_param['v_antiquantscale_dtype'] = copy.deepcopy(ifa_param['k_antiquantoffset_dtype'])
            ifa_param['v_antiquantscale_tensor_raw'] = copy.deepcopy(ifa_param['k_antiquantscale_tensor_raw'])

            ifa_param['v_antiquantoffset_shape_raw'] = copy.deepcopy(ifa_param['k_antiquantoffset_shape_raw'])
            ifa_param['v_antiquantoffset_dtype'] = copy.deepcopy(ifa_param['k_antiquantoffset_dtype'])
            ifa_param['v_antiquantoffset_tensor_raw'] = copy.deepcopy(ifa_param['k_antiquantoffset_tensor_raw'])
        else:
            ifa_param['v_antiquantscale_shape_raw'] = v_antiquantscale_shape_raw = params['shape_input'][v_end_index + 15]
            if 0 in ifa_param['v_antiquantscale_shape_raw']:
                ifa_param['normal_flag'] = False
                print('[WARNING]:异常-->  v_antiquantscale为空场景！')
            ifa_param['v_antiquantscale_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 15])
            ifa_param['v_antiquantscale_tensor_raw'] = torch_tensor_list[v_end_index + 15]
            if ifa_param['v_antiquant_offset_flag']:
                ifa_param['v_antiquantoffset_shape_raw'] = v_antiquantoffset_shape_raw = params['shape_input'][v_end_index + 16]
                ifa_param['v_antiquantoffset_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 16])
                if 0 in v_antiquantoffset_shape_raw:
                    ifa_param['normal_flag'] = False
                    print('[WARNING]:异常-->  v_antiquantoffset为空场景！')
                ifa_param['v_antiquantoffset_tensor_raw'] = torch_tensor_list[v_end_index + 16]
            else:
                ifa_param['v_antiquantoffset_shape_raw'] = v_antiquantscale_shape_raw
                ifa_param['v_antiquantoffset_dtype'] = ifa_param['v_antiquantscale_dtype']
                ifa_param['v_antiquantoffset_tensor_raw'] = torch.zeros(v_antiquantscale_shape_raw).numpy()

    # ND+rope： 当kvs<=512用softmaxV1 ;transpose/nz+rope ：golden计算都用softmaxV2
    ifa_param["softmaxV1_flag"] = False
    # if ifa_param['deep_seek_flag'] and ifa_param['pa_flag']:
    #     if inputLayout in ["TND", "BSH", "BNSD", "BSND"]:
    #         if len(ifa_param['k_cache_shape']) != 5:
    #             ifa_param["softmaxV1_flag"] = True
    return ifa_param


# GPU分支：tensor shape = B,S,N,D
def cut_kvs_by_mask(tensor, mask):
    new_tensor = tensor.clone()
    b = tensor.shape[0]
    smax = tensor.shape[1]
    new_slist = []
    mask = mask.squeeze()
    for maskBatch in range(b):
        s = 0
        m_tensor_batch = mask[maskBatch:maskBatch + 1, :]
        for maskIndex in range(smax):
            maskOfsindex = m_tensor_batch[0][maskIndex]
            if not maskOfsindex:
                new_tensor[maskBatch:maskBatch + 1, s:(s + 1), :, :] = tensor[maskBatch:maskBatch + 1,
                                                                       maskIndex:(maskIndex + 1), :, :]
                s += 1
        new_slist.append(s)
    return new_tensor, new_slist


# GPU分支：比较两个list，每个值都取较小的那个，输出新list
def get_min_list(list1, list2):
    new_list = [min(x, y) for x, y in zip(list1, list2)]
    return new_list


def left_padding_handler(tensor, act_list, padding_size):
    new_tensor = tensor.clone()
    b = tensor.shape[0]
    smax = tensor.shape[1]
    for bindex in range(b):
        act_s = act_list[bindex]
        start_index = smax - padding_size - act_s
        end_index = smax - padding_size
        new_tensor[bindex:bindex + 1, 0:act_s, :, :] = tensor[bindex:bindex + 1, start_index:end_index, :, :]
    return new_tensor


def left_padding_handler_mask(tensor, act_list, padding_size):
    new_tensor = tensor.clone()
    b = tensor.shape[0]
    smax = tensor.shape[1]
    for bindex in range(b):
        act_s = act_list[bindex]
        start_index = smax - padding_size - act_s
        end_index = smax - padding_size
        new_tensor[bindex:bindex + 1, 0:act_s] = tensor[bindex:bindex + 1, start_index:end_index]
        new_tensor[bindex:bindex + 1, act_s:] = 1
    return new_tensor


# 将int32类型的数据,拆成Int4,存进int8
def trans_int32_2_int4(input_int32):
    # 将Int32类型的数据按bit位平均拆成8份，每份长度为4bit
    parts = [(input_int32 >> i) & 0xf for i in range(0, 32, 4)]
    output_int4 = []
    for part in parts:
        # 将每份数据构造成一个Int8类型的数据
        # 符号位为第一个bit的值
        sign = (part >> 3) & 0x1
        # 剩余的值作为int8最后3个bit的值
        value = part & 0x7
        if sign == 1:
            # 如果符号位为1，则需要将value转换成负数
            value = -((value ^ 0x7) + 1)
        output_int4.append(value)
    return output_int4


# 将int32的tensor拆分，存入Int8类型的tensor
def trans_int32_2_int4_tensor_bnsd(tensor_int32, shape_int32):
    shape_int4 = [shape_int32[0], shape_int32[1], shape_int32[2], shape_int32[3] * 8]
    tensor_int4 = np.zeros(shape_int4).astype(np.int8)
    for Bid in range(shape_int32[0]):
        for Nid in range(shape_int32[1]):
            for Sid in range(shape_int32[2]):
                for Did in range(shape_int32[3]):
                    int4_data_list = trans_int32_2_int4(
                        int(tensor_int32[Bid:Bid + 1, Nid:Nid + 1, Sid:Sid + 1, Did:Did + 1]))
                    for i in range(8):
                        tensor_int4[Bid:Bid + 1, Nid:Nid + 1, Sid:Sid + 1, 8 * Did + i:8 * Did + i + 1] = \
                            int4_data_list[i]
    return tensor_int4, shape_int4


# pertoken_pa 场景，将scale/offset从BB转为1BS
def trans_bb_2_1bs(input_tensor, tables, batch):
    block_size = input_tensor.shape[1]
    max_seq = tables.shape[1] * block_size
    max_block_per_seq = math.ceil(max_seq / block_size)
    output_tensor = np.zeros((1, batch, max_seq), dtype=np.float32)
    for bindex in range(batch):
        table = tables[bindex]
        for block_base_index in range(max_block_per_seq):
            block_index = table[block_base_index]
            if block_index == -1:
                break
            if (block_base_index + 1) * block_size > max_seq:
                lens = max_seq - block_base_index * block_size
                output_tensor[0, bindex, block_base_index * block_size:max_seq] = input_tensor[block_index][0:lens]
            else:
                output_tensor[0, bindex, block_base_index * block_size:(block_base_index + 1) * block_size] = \
                    input_tensor[block_index]
    return output_tensor


# perheadtoken_pa 场景，将scale/offset从BnB转为1BnS
def trans_bb_2_1bns(input_tensor, tables, batch):
    num_kv_head = input_tensor.shape[1]
    block_size = input_tensor.shape[2]
    max_seq = tables.shape[1] * block_size
    max_block_per_seq = math.ceil(max_seq / block_size)
    output_tensor = np.zeros((batch, num_kv_head, max_seq), dtype=np.float32)

    for bindex in range(batch):
        table = tables[bindex]
        for block_base_index in range(max_block_per_seq):
            block_index = table[block_base_index]
            if block_index == -1:
                break
            if (block_base_index + 1) * block_size > max_seq:
                lens = max_seq - block_base_index * block_size
                output_tensor[bindex, :, block_base_index * block_size:max_seq] = input_tensor[block_index][:][0:lens]
            else:
                output_tensor[bindex, :, block_base_index * block_size:(block_base_index + 1) * block_size] = \
                    input_tensor[block_index]
    return output_tensor


def concat_tensor_gpu(tensor1, shape1, tensor2, shape2, n, tnd_flag=False):
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
        concatenated_tensor = torch.cat((tensor1_1nd, tensor2_1nd), dim=2)
        concatenated_shape = [shape1[0], shape1[1] + shape2[1]]
        concatenated_tensor = concatenated_tensor.reshape(concatenated_shape)
    elif len(shape1) == 3:
        if shape1[0] != shape2[0] or shape1[1] != shape2[1]:
            print(f"[ERROR]concat_tensor: 相加的两个tensor 维度非法! shape1 = {shape1}, shape2 =  {shape2}")
            return None, None
        if tnd_flag:
            concatenated_tensor = torch.cat((tensor1, tensor2), dim=2)
            concatenated_shape = [shape1[0], shape1[1], shape1[2] + shape2[2]]
        else:
            d1 = int(shape1[2] / n)
            d2 = int(shape2[2] / n)
            tensor1_bsnd = tensor1.reshape(shape1[0], shape1[1], n, d1)
            tensor2_bsnd = tensor2.reshape(shape2[0], shape2[1], n, d2)
            concatenated_tensor = torch.cat((tensor1_bsnd, tensor2_bsnd), dim=3)
            concatenated_shape = [shape1[0], shape1[1], shape1[2] + shape2[2]]
            concatenated_tensor = concatenated_tensor.reshape(concatenated_shape)
    elif len(shape1) == 4:
        if shape1[0] != shape2[0] or shape1[1] != shape2[1] or shape1[2] != shape2[2]:
            print(f"[ERROR]concat_tensor: 相加的两个tensor 维度非法! shape1 = {shape1}, shape2 =  {shape2}")
            return None, None
        concatenated_tensor = torch.cat((tensor1, tensor2), dim=3)
        concatenated_shape = [shape1[0], shape1[1], shape1[2], shape1[3] + shape2[3]]
    else:
        print(f"[ERROR]concat_tensor: shape1 维度非法! shape1 = {shape1}")
        return None, None
    return concatenated_tensor, concatenated_shape


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
    print(f"[INFO]before trans: {list}")
    print(f"[INFO]after trans: {list_new}")
    return list_new


def deepseek_preprocessing(ifa_param):
    numHeads = ifa_param['numHeads']
    numKeyValueHeads = ifa_param['numKeyValueHeads']
    # 备份拼接前的信息
    ifa_param['q_tensor_old'] = copy.deepcopy(ifa_param['q_tensor'])
    ifa_param['q_shape_old'] = copy.deepcopy(ifa_param['q_shape'])
    ifa_param['k_tensor_list_old'] = copy.deepcopy(ifa_param['k_tensor_list'])
    ifa_param['k_shape_list_old'] = copy.deepcopy(ifa_param['k_shape_list'])
    # 非全量化场景1.将Q与QROPE拼接2.将K与KROPE拼接3.伪量化场景，k_antiscale与k_rope_antiscale拼接
    if not ifa_param['in_quant_flag']:
        q_new_tensor, q_new_shape = concat_tensor(ifa_param['q_tensor'], ifa_param['q_shape'],
                                                  ifa_param['q_rope_tensor'],
                                                  ifa_param['q_rope_shape'], numHeads, ifa_param['tnd_flag'])
        if q_new_tensor is not None:
            ifa_param['q_tensor'] = q_new_tensor
            ifa_param['q_shape'] = q_new_shape
        else:
            print("[ERROR]q tensor的deepseek预处理异常，输出空tensor！")
            # return torch.zeros(out_shape)
        if ifa_param['k_num'] == 1:
            k_new_tensor, k_new_shape = concat_tensor(ifa_param['k_tensor_list'][0], ifa_param['k_shape_list'][0],
                                                      ifa_param['k_rope_tensor'],
                                                      ifa_param['k_rope_shape'], numKeyValueHeads,
                                                      ifa_param['tnd_flag'])
            if k_new_tensor is not None:
                ifa_param['k_tensor_list'][0] = k_new_tensor
                ifa_param['k_shape_list'][0] = k_new_shape
            else:
                print("[ERROR]k tensor的deepseek预处理异常，输出空tensor！")
                # return torch.zeros(out_shape)
        else:
            print("[ERROR]k tensor 长度不为1, deepseek预处理异常，输出空tensor！")
            # return torch.zeros(out_shape)
        # rope scale处理
        if ifa_param['k_rope_antiquantscale_shape_raw'] is not None:
            k_antiquantscale_shape_raw0 = ifa_param['k_antiquantscale_shape_raw']
            k_antiquantscale_tensor_raw0 = ifa_param['k_antiquantscale_tensor_raw']
            k_rope_antiquantscale_shape_raw = ifa_param['k_rope_antiquantscale_shape_raw']
            k_rope_antiquantscale_tensor_raw = ifa_param['k_rope_antiquantscale_tensor_raw']
            k_scale_new_tensor, k_scale_new_shape = concat_tensor(k_antiquantscale_tensor_raw0,
                                                                  k_antiquantscale_shape_raw0,
                                                                  k_rope_antiquantscale_tensor_raw,
                                                                  k_rope_antiquantscale_shape_raw, numKeyValueHeads,
                                                                  ifa_param['tnd_flag'])

            # offset是空的，但是也要拼接成大小相同的tensor
            k_antiquantoffset_shape_raw0 = ifa_param['k_antiquantoffset_shape_raw']
            k_antiquantoffset_tensor_raw0 = ifa_param['k_antiquantoffset_tensor_raw']
            k_rope_antiquantoffset_shape_raw = ifa_param['k_rope_antiquantoffset_shape_raw']
            k_rope_antiquantoffset_tensor_raw = ifa_param['k_rope_antiquantoffset_tensor_raw']
            k_offset_new_tensor, k_offset_new_shape = concat_tensor(k_antiquantoffset_tensor_raw0,
                                                                    k_antiquantoffset_shape_raw0,
                                                                    k_rope_antiquantoffset_tensor_raw,
                                                                    k_rope_antiquantoffset_shape_raw, numKeyValueHeads,
                                                                    ifa_param['tnd_flag'])
            if k_scale_new_tensor is not None:
                ifa_param['k_antiquantscale_tensor_raw'] = k_scale_new_tensor
                ifa_param['k_antiquantscale_shape_raw'] = k_scale_new_shape
                ifa_param['k_antiquantoffset_tensor_raw'] = k_offset_new_tensor
                ifa_param['k_antiquantoffset_shape_raw'] = k_offset_new_shape
            else:
                print("[ERROR]k antiquant scale tensor的deepseek预处理异常，输出空tensor！")
                # return torch.zeros(out_shape)
    return ifa_param


def trans_bnsd_to_dequant_layout(tensor, shape, layout, act_q=None):
    B = tensor.shape[0]
    N = tensor.shape[1]
    if layout in ["TND", "TND_NTD"]:
        T = sum(act_q)
        output = torch.zeros(size=(T, N), dtype=tensor.dtype)
        t_start = 0
        for b_index in range(B):
            act_s = act_q[b_index]
            t_end = t_start + act_s
            if act_s == 0:
                continue
            for n_index in range(N):
                output[t_start:t_end, n_index:n_index + 1] = tensor[b_index, n_index, :act_s]
            t_start += act_s
        return output
    else:
        S = shape[2]
        if layout in ["BSND", "BSH"]:
            return tensor.permute(0, 2, 1, 3).reshape(B, S, N)
        else:
            # BNSD
            return tensor.reshape(B, N, S)


def trans_dequant_layout_to_bnsd(tensor, shape, layout, act_seq=None):
    print(f"def trans_dequant_layout_to_bsnd(tensor, shape, layout({layout}), act_q=None):")
    print("tensor.shape : ", tensor.shape)
    if layout in ["BSND", 'BSND_NBSD',"BSH", 'BSH_NBSD']:
        B = tensor.shape[0]
        S = tensor.shape[1]
        N = tensor.shape[2]
        return tensor.reshape([B, S, N, 1]).transpose(0, 2, 1, 3)
    elif layout in ["TND", "TND_NTD"]:
        T = shape[0]
        N = shape[1]
        B = len(act_seq)
        S = max(act_seq)
        new_tensor = np.zeros((B, N, S), dtype=tensor.dtype)
        t_start = 0
        for b_index in range(B):
            act_s = act_seq[b_index]
            t_end = t_start + act_s
            # print(f"[TEMP]b_index:{b_index} act_s:{act_s}, t_start:{t_start}, t_end:{t_end}")
            if act_s == 0:
                continue
            for n_index in range(N):
                new_tensor[b_index, n_index, 0:act_s] = tensor[t_start:t_end, n_index]
            t_start += act_s
        # print(f"[TEMP]trans tnd 2 bnsd: {tensor.shape} -> {new_tensor.shape}")
        # print(f"[TEMP]new_tensor:{new_tensor}")
        return new_tensor.reshape([B, N, S, 1])
    else:
        print("error inputlayout")


def trans_bnsd_to_input_layout(tensor, shape, layout, act_q=None):
    if layout in ["BSH", "BSH_NBSD"]:
        output = tensor.permute(0, 2, 1, 3).contiguous().view(shape)
        return output
    elif layout in ["BSND", "BSND_NBSD"]:
        output = tensor.permute(0, 2, 1, 3).contiguous()
        return output
    elif layout in ["TND", "TND_NTD"]:
        T = sum(act_q)
        B = tensor.shape[0]
        N = tensor.shape[1]
        D = tensor.shape[3]
        # output = torch.zeros(size=(T, N, D), dtype=tensor.dtype)
        output = torch.zeros(size=(T, N, D), dtype=torch.float32)
        output = output.to(tensor.dtype)
        t_start = 0
        for b_index in range(B):
            act_s = act_q[b_index]
            t_end = t_start + act_s
            if act_s == 0:
                continue
            for n_index in range(N):
                output[t_start:t_end, n_index, :] = tensor[b_index, n_index, :act_s, :]
            t_start += act_s
        return output
    else:
        return tensor


def deepseek_ds_pa_preprocessing(ifa_param, params):
    k_np_dtype = ifa_param['k_np_dtype']
    v_np_dtype = ifa_param['v_np_dtype']
    k_ori_shape_list = ifa_param['k_shape_list_old']
    numKeyValueHeads = ifa_param['numKeyValueHeads']
    numHeads = ifa_param['numHeads']
    kv_layout = ifa_param['kv_layout']
    actualSeqLengths = ifa_param['actualSeqLengths']
    actualSeqLengths_q = ifa_param['actualSeqLengths_q']
    # k_shape_bnsd_list = ifa_param['k_shape_bnsd_list']
    blockSize = ifa_param['blocksize']
    blockTableShape = ifa_param['bt_shape']
    block_table = ifa_param['block_table']
    inputLayout = ifa_param['inputLayout']
    k_cache = np.zeros(ifa_param['k_cache_shape']).astype(k_np_dtype)
    v_cache = np.zeros(ifa_param['v_cache_shape']).astype(v_np_dtype)
    k_rope_cache = np.zeros(ifa_param['k_rope_cache_shape']).astype(k_np_dtype)
    k_dtype = ifa_param['k_dtype']
    # 对备份的拼接前信息进行预处理
    x = 0
    k_tensor_bnsd_list = []
    for i in range(0, ifa_param['k_num']):
        k_tensor_bnsd_list = []
        k_tensor, k_bnsd_shape = _n_trans_shape_to_bnsd(ifa_param['k_tensor_list_old'][i], k_ori_shape_list[x],
                                                        kv_layout,
                                                        numKeyValueHeads, actualSeqLengths)
        x = x + 1
        k_tensor_bnsd_list.append(k_tensor)
        # k_shape_bnsd_list.append(k_bnsd_shape)
        # k int32预处理
        if ifa_param['k_dtype'] == "int32":
            k_tensor, k_bnsd_shape = trans_int32_2_int4_tensor_bnsd(k_tensor, k_bnsd_shape)

        if k_bnsd_shape[2] > ifa_param['ks_max']:
            ifa_param['ks_max'] = k_bnsd_shape[2]
    k_rope_bnsd_tensor, k_rope_bnsd_shape = _n_trans_shape_to_bnsd(ifa_param['k_rope_tensor'],
                                                                   ifa_param['k_rope_shape'],
                                                                   kv_layout,
                                                                   numKeyValueHeads, actualSeqLengths)
    if ifa_param['k_dtype'] == "int32":
        k_rope_bnsd_tensor, k_rope_bnsd_shape = trans_int32_2_int4_tensor_bnsd(k_rope_bnsd_tensor,
                                                                               k_rope_bnsd_shape)
    # gen k cache
    # 获取qd
    q_bnsd_tensor, q_bnsd_shape = _n_trans_shape_to_bnsd(ifa_param['q_tensor_old'], ifa_param['q_shape_old'],
                                                         inputLayout,
                                                         numHeads, actualSeqLengths_q)
    if len(ifa_param['k_cache_shape']) != len(ifa_param['k_rope_cache_shape']):
        print('[WARNING]:k_cache_shape与k_rope_cache_shape 维数不符，输出空tensor！')
        # return torch.zeros(out_shape)
    if len(ifa_param['k_cache_shape']) == 3:
        # trans kv to bsh(此处使用的tensor, 没有经过n的扩展)
        k_tensor_bsh_raw = trans_bnsd_to_bsh(k_tensor_bnsd_list[0], k_tensor_bnsd_list[0].shape)
        # kv paddIng
        kvh = numKeyValueHeads * q_bnsd_shape[3]
        k_tensor_bsh = np.zeros((q_bnsd_shape[0], blockTableShape[1] * blockSize, kvh)).astype(k_np_dtype)
        print(f"[INFO]k_tensor_bbh: {k_tensor_bsh.shape}")
        k_tensor_bsh[:, :k_tensor_bsh_raw.shape[1], :] = k_tensor_bsh_raw[:, :, :]
        for b in range(q_bnsd_shape[0]):
            for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                block_offset = block_i * blockSize
                if kv_cache_blk_id == -1:
                    continue
                else:
                    k_cache[kv_cache_blk_id, 0:blockSize, :] = k_tensor_bsh[b,
                                                               block_offset:(block_offset + blockSize), :]

        k_rope_tensor_bsh_raw = trans_bnsd_to_bsh(k_rope_bnsd_tensor, k_rope_bnsd_shape)

        kropeh = numKeyValueHeads * k_rope_bnsd_shape[3]
        k_rope_tensor_bsh = np.zeros((q_bnsd_shape[0], blockTableShape[1] * blockSize, kropeh)).astype(
            k_np_dtype)
        print(f"[INFO]k_rope_tensor_bbh: {k_rope_tensor_bsh.shape}")
        k_rope_tensor_bsh[:, :k_rope_tensor_bsh_raw.shape[1], :] = k_rope_tensor_bsh_raw[:, :, :]
        for b in range(q_bnsd_shape[0]):
            for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                block_offset = block_i * blockSize
                if kv_cache_blk_id == -1:
                    continue
                else:
                    k_rope_cache[kv_cache_blk_id, 0:blockSize, :] = k_rope_tensor_bsh[b,
                                                                    block_offset:(block_offset + blockSize), :]
    elif len(ifa_param['k_cache_shape']) == 4:
        kvn = k_tensor_bnsd_list[0].shape[1]
        kvs = k_tensor_bnsd_list[0].shape[2]
        kvd = k_tensor_bnsd_list[0].shape[3]
        k_tensor_bnsd = np.zeros((q_bnsd_shape[0], kvn, blockTableShape[1] * blockSize, kvd)).astype(k_np_dtype)
        print(f"[INFO]k_tensor_bnbd: {k_tensor_bnsd.shape}")
        k_tensor_bnsd[:, :, :kvs, :] = k_tensor_bnsd_list[0][:, :, :, :]

        for b in range(q_bnsd_shape[0]):
            for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                block_offset = block_i * blockSize
                if kv_cache_blk_id == -1:
                    continue
                else:
                    for n in range(q_bnsd_shape[1]):
                        k_cache[kv_cache_blk_id, n:n + 1, 0:blockSize, :] = k_tensor_bnsd[b, n:n + 1,
                                                                            block_offset:(
                                                                                    block_offset + blockSize),
                                                                            :]
        kropen = k_rope_bnsd_shape[1]
        kropes = k_rope_bnsd_shape[2]
        kroped = k_rope_bnsd_shape[3]
        k_rope_tensor_bnsd = np.zeros((q_bnsd_shape[0], kropen, blockTableShape[1] * blockSize, kroped)).astype(
            k_np_dtype)
        print(f"[INFO]k_rope_tensor_bnbd: {k_rope_tensor_bnsd.shape}")
        k_rope_tensor_bnsd[:, :, :kropes, :] = k_rope_bnsd_tensor[:, :, :, :]

        for b in range(q_bnsd_shape[0]):
            for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                block_offset = block_i * blockSize
                if kv_cache_blk_id == -1:
                    continue
                else:
                    for n in range(q_bnsd_shape[1]):
                        k_rope_cache[kv_cache_blk_id, n:n + 1, 0:blockSize, :] = k_rope_tensor_bnsd[b, n:n + 1,
                                                                                 block_offset:(
                                                                                         block_offset + blockSize),
                                                                                 :]
    elif len(ifa_param['k_cache_shape']) == 5:
        kvn = k_tensor_bnsd_list[0].shape[1]
        kvs = k_tensor_bnsd_list[0].shape[2]
        kvd = k_tensor_bnsd_list[0].shape[3]
        k_tensor_bnsd = np.zeros((q_bnsd_shape[0], kvn, blockTableShape[1] * blockSize, kvd)).astype(k_np_dtype)
        print(f"[INFO]k_tensor_bnbd: {k_tensor_bnsd.shape}")
        k_tensor_bnsd[:, :, :kvs, :] = k_tensor_bnsd_list[0][:, :, :, :]
        k_cache_tensor_bnbd = np.zeros((k_cache.shape[0], kvn, blockSize, kvd)).astype(
            k_np_dtype)

        for b in range(q_bnsd_shape[0]):
            for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                block_offset = block_i * blockSize
                if kv_cache_blk_id == -1:
                    continue
                else:
                    for n in range(q_bnsd_shape[1]):
                        k_cache_tensor_bnbd[kv_cache_blk_id, n:n + 1, 0:blockSize, :] = k_tensor_bnsd[b,
                                                                                        n:n + 1,
                                                                                        block_offset:(
                                                                                                block_offset + blockSize),
                                                                                        :]
        # [block_num, N, block_size, D] -> [block_num, N, D/16, block_size, 16]
        print(f"[INFO]k_cache_tensor_bnbd BNBD: {k_cache_tensor_bnbd.shape}")
        if k_dtype == "int8":
            base = 32
        else:
            base = 16
        k_cache = k_cache_tensor_bnbd.reshape(k_cache_tensor_bnbd.shape[0], k_cache_tensor_bnbd.shape[1],
                                              k_cache_tensor_bnbd.shape[2],
                                              k_cache_tensor_bnbd.shape[3] // base,
                                              base).transpose(0, 1, 3, 2, 4)
        print(f"[INFO]k_cache NZ: {k_cache.shape}")

        kropen = k_rope_bnsd_shape[1]
        kropes = k_rope_bnsd_shape[2]
        kroped = k_rope_bnsd_shape[3]
        k_rope_tensor_bnsd = np.zeros((q_bnsd_shape[0], kropen, blockTableShape[1] * blockSize, kroped)).astype(
            k_np_dtype)
        print(f"[INFO]k_rope_tensor_bnbd: {k_rope_tensor_bnsd.shape}")
        k_rope_tensor_bnsd[:, :, :kropes, :] = k_rope_bnsd_tensor[:, :, :, :]
        k_rope_cache_tensor_bnbd = np.zeros(
            (k_rope_cache.shape[0], kropen, blockSize, kroped)).astype(k_np_dtype)
        print(f"[INFO]k_rope_cache_tensor_bnbd: {k_rope_cache_tensor_bnbd.shape}")

        for b in range(q_bnsd_shape[0]):
            for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                block_offset = block_i * blockSize
                if kv_cache_blk_id == -1:
                    continue
                else:
                    for n in range(q_bnsd_shape[1]):
                        k_rope_cache_tensor_bnbd[kv_cache_blk_id, n:n + 1, 0:blockSize, :] = k_rope_tensor_bnsd[
                                                                                             b, n:n + 1,
                                                                                             block_offset:(
                                                                                                     block_offset + blockSize),
                                                                                             :]
        # [block_num, N, block_size, ROPE_D] -> [block_num, N, ROPE_D/16, block_size, 16]
        print(f"[INFO]k_rope_cache_tensor_bnbd BNBD: {k_rope_cache_tensor_bnbd.shape}")
        k_rope_cache = k_rope_cache_tensor_bnbd.reshape(k_rope_cache_tensor_bnbd.shape[0],
                                                        k_rope_cache_tensor_bnbd.shape[1],
                                                        k_rope_cache_tensor_bnbd.shape[2],
                                                        k_rope_cache_tensor_bnbd.shape[3] // base, base).transpose(0, 1,
                                                                                                                   3, 2,
                                                                                                                   4)
        print(f"[INFO]k_rope_cache NZ: {k_rope_cache.shape}")
    else:
        print(f"[ERROR]Wrong input k_cache_shape: {ifa_param['k_cache_shape']}")
        # return torch.zeros(out_shape)
    # 将kv cache 生成新的bin文件
    k_cache_index = 21
    v_cache_index = 22
    k_rope_cache_index = 25
    tools.modify_alcnn_input_file(ids=k_cache_index, origin_index=[k_cache_index], type='tensor_list',
                                  mode='rewrite',
                                  tensors=[k_cache],
                                  params=params,
                                  data_dtype=params['dtype_input'][1])
    tools.modify_alcnn_input_file(ids=v_cache_index, origin_index=[v_cache_index], type='tensor_list',
                                  mode='rewrite',
                                  tensors=[k_cache],
                                  params=params,
                                  data_dtype=params['dtype_input'][2])
    tools.modify_alcnn_input_file(ids=k_rope_cache_index, origin_index=[k_rope_cache_index], type='tensor',
                                  mode='rewrite',
                                  tensors=k_rope_cache,
                                  params=params,
                                  data_dtype=params['dtype_input'][1])

def numpy_float8_e4m3fn():
    try:
        # noinspection PyUnresolvedReferences
        from ml_dtypes import float8_e4m3fn
        return float8_e4m3fn
    except ModuleNotFoundError:
        raise RuntimeError("ml_dtypes is needed to support float8_e4m3fn dtype!!! "
                           "Please install with `pip3 install ml-dtypes`")

def deepseek_inquant_ds_pa_preprocessing(ifa_param, params):
    # 生成blocktable
    k_np_dtype = numpy_float8_e4m3fn()
    v_np_dtype = numpy_float8_e4m3fn()
    k_rope_dtype = tools.get_np_dtype(params['dtype_input'][24])
    k_tensor_bnsd_list = ifa_param['k_tensor_list_bnsd']
    numKeyValueHeads = ifa_param['numKeyValueHeads']
    q_bnsd_shape = ifa_param['q_bnsd_shape']
    blockSize = ifa_param['blocksize']
    blockTableShape = ifa_param['bt_shape']
    block_table = ifa_param['block_table']
    k_rope_bnsd_tensor = ifa_param['k_rope_bnsd_tensor']
    k_rope_bnsd_shape = ifa_param['k_rope_bnsd_shape']
    print(
        f"[PageAtten]Input Kdtype:{params['dtype_input'][1]} Vdtype:{params['dtype_input'][2]} KRopeType:{params['dtype_input'][24]}")
    k_cache = np.zeros(ifa_param['k_cache_shape']).astype(k_np_dtype)
    v_cache = np.zeros(ifa_param['v_cache_shape']).astype(v_np_dtype)
    print('[INFO]:PA + Deepseek！')
    k_rope_cache = np.zeros(ifa_param['k_rope_cache_shape']).astype(k_rope_dtype)
    if len(ifa_param['k_cache_shape']) != len(ifa_param['k_rope_cache_shape']):
        print('[WARNING]:k_cache_shape与k_rope_cache_shape 维数不符，输出空tensor！')
    if len(ifa_param['k_cache_shape']) == 3:
        # trans kv to bsh(此处使用的tensor, 没有经过n的扩展)
        k_tensor_bsh_raw = trans_bnsd_to_bsh(k_tensor_bnsd_list[0], k_tensor_bnsd_list[0].shape)
        # kv paddIng
        kvh = numKeyValueHeads * q_bnsd_shape[3]
        k_tensor_bsh = np.zeros((q_bnsd_shape[0], blockTableShape[1] * blockSize, kvh)).astype(k_np_dtype)
        print(f"[INFO]k_tensor_bbh: {k_tensor_bsh.shape}")
        k_tensor_bsh[:, :k_tensor_bsh_raw.shape[1], :] = k_tensor_bsh_raw[:, :, :]
        for b in range(q_bnsd_shape[0]):
            for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                block_offset = block_i * blockSize
                if kv_cache_blk_id == -1:
                    continue
                else:
                    k_cache[kv_cache_blk_id, 0:blockSize, :] = k_tensor_bsh[b,
                                                               block_offset:(block_offset + blockSize), :]

        k_rope_tensor_bsh_raw = trans_bnsd_to_bsh(k_rope_bnsd_tensor, k_rope_bnsd_shape)

        kropeh = numKeyValueHeads * k_rope_bnsd_shape[3]
        k_rope_tensor_bsh = np.zeros((q_bnsd_shape[0], blockTableShape[1] * blockSize, kropeh)).astype(
            k_rope_dtype)
        print(f"[INFO]k_rope_tensor_bbh: {k_rope_tensor_bsh.shape}")
        k_rope_tensor_bsh[:, :k_rope_tensor_bsh_raw.shape[1], :] = k_rope_tensor_bsh_raw[:, :, :]
        for b in range(q_bnsd_shape[0]):
            for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                block_offset = block_i * blockSize
                if kv_cache_blk_id == -1:
                    continue
                else:
                    k_rope_cache[kv_cache_blk_id, 0:blockSize, :] = k_rope_tensor_bsh[b,
                                                                    block_offset:(block_offset + blockSize), :]
    elif len(ifa_param['k_cache_shape']) == 4:
        kvn = k_tensor_bnsd_list[0].shape[1]
        kvs = k_tensor_bnsd_list[0].shape[2]
        kvd = k_tensor_bnsd_list[0].shape[3]
        k_tensor_bnsd = np.zeros((q_bnsd_shape[0], kvn, blockTableShape[1] * blockSize, kvd)).astype(k_np_dtype)
        print(f"[INFO]k_tensor_bnbd: {k_tensor_bnsd.shape}")
        k_tensor_bnsd[:, :, :kvs, :] = k_tensor_bnsd_list[0][:, :, :, :]

        for b in range(q_bnsd_shape[0]):
            for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                block_offset = block_i * blockSize
                if kv_cache_blk_id == -1:
                    continue
                else:
                    for n in range(q_bnsd_shape[1]):
                        k_cache[kv_cache_blk_id, n:n + 1, 0:blockSize, :] = k_tensor_bnsd[b, n:n + 1,
                                                                            block_offset:(
                                                                                    block_offset + blockSize),
                                                                            :]
        kropen = k_rope_bnsd_shape[1]
        kropes = k_rope_bnsd_shape[2]
        kroped = k_rope_bnsd_shape[3]
        k_rope_tensor_bnsd = np.zeros((q_bnsd_shape[0], kropen, blockTableShape[1] * blockSize, kroped)).astype(
            k_rope_dtype)
        print(f"[INFO]k_rope_tensor_bnbd: {k_rope_tensor_bnsd.shape}")
        k_rope_tensor_bnsd[:, :, :kropes, :] = k_rope_bnsd_tensor[:, :, :, :]

        for b in range(q_bnsd_shape[0]):
            for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                block_offset = block_i * blockSize
                if kv_cache_blk_id == -1:
                    continue
                else:
                    for n in range(q_bnsd_shape[1]):
                        k_rope_cache[kv_cache_blk_id, n:n + 1, 0:blockSize, :] = k_rope_tensor_bnsd[b, n:n + 1,
                                                                                 block_offset:(
                                                                                         block_offset + blockSize),
                                                                                 :]
    elif len(ifa_param['k_cache_shape']) == 5:
        kvn = k_tensor_bnsd_list[0].shape[1]
        kvs = k_tensor_bnsd_list[0].shape[2]
        kvd = k_tensor_bnsd_list[0].shape[3]
        k_tensor_bnsd = np.zeros((q_bnsd_shape[0], kvn, blockTableShape[1] * blockSize, kvd),dtype=k_np_dtype)
        k_tensor_bnsd[:, :, :kvs, :] = k_tensor_bnsd_list[0][:, :, :, :]
        k_cache_tensor_bnbd = np.zeros((k_cache.shape[0], kvn, blockSize, kvd)).astype(
            k_np_dtype)

        for b in range(q_bnsd_shape[0]):
            for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                block_offset = block_i * blockSize
                if kv_cache_blk_id == -1:
                    continue
                else:
                    for n in range(q_bnsd_shape[1]):
                        k_cache_tensor_bnbd[kv_cache_blk_id, n:n + 1, 0:blockSize, :] = k_tensor_bnsd[b,
                                                                                        n:n + 1, block_offset:(
                                    block_offset + blockSize), :]
        basek = 32
        basekrope = 16
        print(f"k_cache_tensor_bnbd.shape = {k_cache_tensor_bnbd.shape}")
        k_cache = k_cache_tensor_bnbd.reshape(k_cache_tensor_bnbd.shape[0], k_cache_tensor_bnbd.shape[1],
                                              k_cache_tensor_bnbd.shape[2], k_cache_tensor_bnbd.shape[3] // basek,
                                              basek).transpose(0, 1, 3, 2, 4)
        print(f"[INFO]k_cache NZ: {k_cache.shape}")
        kropen = k_rope_bnsd_shape[1]
        kropes = k_rope_bnsd_shape[2]
        kroped = k_rope_bnsd_shape[3]
        k_rope_tensor_bnsd = np.zeros((q_bnsd_shape[0], kropen, blockTableShape[1] * blockSize, kroped)).astype(
            k_rope_dtype)
        print(f"[INFO]k_rope_tensor_bnbd: {k_rope_tensor_bnsd.shape} {kropes} {k_rope_bnsd_tensor.shape}")
        k_rope_tensor_bnsd[:, :, :kropes, :] = k_rope_bnsd_tensor[:, :, :, :]
        k_rope_cache_tensor_bnbd = np.zeros(
            (k_rope_cache.shape[0], kropen, blockSize, kroped)).astype(k_rope_dtype)
        print(f"[INFO]k_rope_cache_tensor_bnbd: {k_rope_cache_tensor_bnbd.shape}")

        for b in range(q_bnsd_shape[0]):
            for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                block_offset = block_i * blockSize
                if kv_cache_blk_id == -1:
                    continue
                else:
                    for n in range(q_bnsd_shape[1]):
                        k_rope_cache_tensor_bnbd[kv_cache_blk_id, n:n + 1, 0:blockSize, :] = k_rope_tensor_bnsd[
                                                                                             b, n:n + 1,
                                                                                             block_offset:(
                                                                                                     block_offset + blockSize),
                                                                                             :]
        k_rope_cache = k_rope_cache_tensor_bnbd.reshape(
            k_rope_cache_tensor_bnbd.shape[0], k_rope_cache_tensor_bnbd.shape[1],
            k_rope_cache_tensor_bnbd.shape[2], k_rope_cache_tensor_bnbd.shape[3] // basekrope, basekrope).transpose(0,1,3,2,4)
        print(f"[INFO]k_rope_cache NZ: {k_rope_cache.shape}")
    else:
        print(f"[ERROR]Wrong input k_cache_shape: {ifa_param['k_cache_shape']}")
        # return torch.zeros(out_shape)
    # 将kv cache 生成新的bin文件
    k_cache_index = 21
    v_cache_index = 22
    k_rope_cache_index = 25
    tools.modify_alcnn_input_file(ids=k_cache_index, origin_index=[k_cache_index], type='tensor_list',
                                  mode='rewrite',
                                  tensors=[k_cache],
                                  params=params,
                                  data_dtype="float8_e4m3fn")
    tools.modify_alcnn_input_file(ids=v_cache_index, origin_index=[v_cache_index], type='tensor_list',
                                  mode='rewrite',
                                  tensors=[k_cache],
                                  params=params,
                                  data_dtype="float8_e4m3fn")
    tools.modify_alcnn_input_file(ids=k_rope_cache_index, origin_index=[k_rope_cache_index], type='tensor',
                                  mode='rewrite',
                                  tensors=k_rope_cache,
                                  params=params,
                                  data_dtype=params['dtype_input'][24])


def deepseek_inquant_preprocessing(ifa_param, params):
    numHeads = ifa_param['numHeads']
    numKeyValueHeads = ifa_param['numKeyValueHeads']
    actualSeqLengths_q = ifa_param['actualSeqLengths_q']
    actualSeqLengths = ifa_param['actualSeqLengths']
    inputLayout = ifa_param['inputLayout']
    kv_layout = ifa_param['kv_layout']
    q_rope_bnsd_tensor, q_rope_bnsd_shape = _n_trans_shape_to_bnsd(ifa_param['q_rope_tensor'],ifa_param['q_rope_shape'],
                                                                   inputLayout, numHeads, actualSeqLengths_q)
    ifa_param['q_rope_bnsd_tensor'] = q_rope_bnsd_tensor
    ifa_param['q_rope_shape_bnsd'] = q_rope_bnsd_shape

    k_rope_bnsd_tensor, k_rope_bnsd_shape = _n_trans_shape_to_bnsd(ifa_param['k_rope_tensor'],ifa_param['k_rope_shape'],
                                                                   kv_layout, numKeyValueHeads, actualSeqLengths)
    ifa_param['k_rope_bnsd_tensor'] = k_rope_bnsd_tensor
    ifa_param['k_rope_bnsd_shape'] = k_rope_bnsd_shape
    q_bnsd_tensor = ifa_param['q_tensor_bnsd']
    q_bnsd_shape = ifa_param['q_shape_bnsd']
    k_tensor_list = ifa_param['k_tensor_list_bnsd']

    v_tensor_list = ifa_param['v_tensor_list_bnsd']

    rowMax_Qn = np.abs(q_bnsd_tensor).max(axis=(3), keepdims=True).astype(np.float32)
    tenMax_Kn = np.abs(k_tensor_list[0]).max(keepdims=False).astype(np.float32)
    kv_dtype_max = 448
    scaleQ = rowMax_Qn / kv_dtype_max
    scaleK = tenMax_Kn / kv_dtype_max

    dequantScale1 = scaleQ * scaleK

    q_int8_bnsd_tensor = np.round(q_bnsd_tensor / scaleQ).astype(numpy_float8_e4m3fn())
    k_int8_bnsd_tensor = np.round(k_tensor_list[0] / scaleK).astype(numpy_float8_e4m3fn())
    k_tensor_list[0] = k_int8_bnsd_tensor
    v_tensor_list[0] = k_int8_bnsd_tensor
    q_bnsd_tensor = q_int8_bnsd_tensor
    qmin_datarange = int(q_int8_bnsd_tensor.min())
    qmax_datarange = int(q_int8_bnsd_tensor.max())
    kmin_datarange = int(k_int8_bnsd_tensor.min())
    kmax_datarange = int(k_int8_bnsd_tensor.max())

    # q_int8_bnsd_tensor = torch.from_numpy(q_int8_bnsd_tensor).to(torch.float8_e4m3fn)
    # k_int8_bnsd_tensor = torch.from_numpy(k_int8_bnsd_tensor).to(torch.float8_e4m3fn)

    q_int8_bnsd_tensor_uint8 = q_int8_bnsd_tensor.view(np.uint8)
    q_int8_bnsd_tensor = torch.from_numpy(q_int8_bnsd_tensor_uint8).view(torch.float8_e4m3fn)

    k_int8_bnsd_tensor_uint8 = k_int8_bnsd_tensor.view(np.uint8)
    k_int8_bnsd_tensor = torch.from_numpy(k_int8_bnsd_tensor_uint8).view(torch.float8_e4m3fn)

    ifa_param['q_tensor_bnsd'] = q_bnsd_tensor
    print("ifa_param['q_tensor_bnsd'] = q_bnsd_tensor, q_bnsd_tensor : ", q_bnsd_tensor.shape, q_bnsd_tensor.dtype)

    ifa_param['dequantScale1'] = dequantScale1
    print("ifa_param['dequantScale1'] = dequantScale1, dequantScale1 : ", dequantScale1.shape, dequantScale1.dtype)
    ifa_param['dequantScale2'] = np.array([scaleK])
    print("ifa_param['dequantScale2'] = np.array([scaleK]), np.array([scaleK]) : ", np.array([scaleK]),np.array([scaleK]).dtype)

    dequantScale1 = torch.from_numpy(dequantScale1).to(torch.float32)
    dequantScale2 = torch.from_numpy(np.array([scaleK])).to(torch.float32)

    dequantScale1 = trans_bnsd_to_dequant_layout(dequantScale1, dequantScale1.shape, inputLayout, actualSeqLengths_q)
    q_shape = ifa_param['q_shape']
    k_shape = ifa_param['k_shape_list'][0]
    q_int8_npu = trans_bnsd_to_input_layout(q_int8_bnsd_tensor, q_shape, inputLayout,
                                            actualSeqLengths_q)
    k_int8_npu = trans_bnsd_to_input_layout(k_int8_bnsd_tensor, k_shape, kv_layout, actualSeqLengths)

    tools.modify_alcnn_input_file(ids=0, origin_index=[0], type='tensor', mode='rewrite', tensors=q_int8_npu,
                                  params=params, data_range=[qmin_datarange, qmax_datarange], data_dtype="float8_e4m3fn")
    tools.modify_alcnn_input_file(ids=1, origin_index=[1], type='tensor_list', mode='rewrite',
                                  tensors=[k_int8_npu], params=params, data_range=[[kmin_datarange, kmax_datarange]],
                                  data_dtype="float8_e4m3fn")
    tools.modify_alcnn_input_file(ids=2, origin_index=[2], type='tensor_list', mode='rewrite',
                                  tensors=[k_int8_npu], params=params, data_range=[[kmin_datarange, kmax_datarange]],
                                  data_dtype="float8_e4m3fn")
    dequantScaleQ = torch.from_numpy(scaleQ).to(torch.float32)
    dequantScaleQ = trans_bnsd_to_dequant_layout(dequantScaleQ, dequantScaleQ.shape, inputLayout, actualSeqLengths_q)
    tools.modify_alcnn_input_file(ids=27, origin_index=[27], type='tensor', mode='rewrite', tensors=dequantScaleQ,
                                  params=params)
    tools.modify_alcnn_input_file(ids=15, origin_index=[15], type='tensor', mode='rewrite', tensors=dequantScale2,
                                  params=params)
    tools.modify_alcnn_input_file(ids=17, origin_index=[17], type='tensor', mode='rewrite', tensors=dequantScale2,
                                  params=params)

    ifa_param['k_tensor_list_bnsd'] = k_tensor_list
    ifa_param['v_tensor_list_bnsd'] = v_tensor_list

    print("k_tensor_list : ", k_tensor_list[0].shape, k_tensor_list[0].dtype)
    print("v_tensor_list : ", v_tensor_list[0].shape, v_tensor_list[0].dtype)

    return ifa_param


def deepseek_inquant_preprocessing_for_recovery(ifa_param, torch_tensor_list, params):
    print("def deepseek_inquant_preprocessing_for_recovery(ifa_param, torch_tensor_list, params): enter")

    numHeads = ifa_param['numHeads']
    numKeyValueHeads = ifa_param['numKeyValueHeads']
    actualSeqLengths_q = ifa_param['actualSeqLengths_q']
    actualSeqLengths = ifa_param['actualSeqLengths']
    inputLayout = ifa_param['inputLayout']
    kv_layout = ifa_param['kv_layout']
    dequantScale1 = torch_tensor_list[27] * torch_tensor_list[17]
    dequantScale1_bnsd = trans_dequant_layout_to_bnsd(dequantScale1,
                                                      dequantScale1.shape, ifa_param['inputLayout'],
                                                      ifa_param['actualSeqLengths_q'])

    index = 28
    rope_shape = ifa_param['q_rope_shape']
    if len(params['shape_input'])==index:
        from libs.generate_tensor_data import gen_tensor_data

        params["shape_input"].append(params["shape_input"][23])
        params["dtype_input"].append("fp32")
        params["range_input"].append(params["range_input"][23])

        input_data_i = gen_tensor_data(params, index)

        q_rope_bnsd_tensor, q_rope_bnsd_shape = _n_trans_shape_to_bnsd(input_data_i, ifa_param['q_rope_shape'],
                                                                       inputLayout, numHeads, actualSeqLengths_q)

        q_rope_bnsd_tensor_quant = np.nan_to_num(q_rope_bnsd_tensor.astype(np.float32) / dequantScale1_bnsd, nan=0.0)

        gpu_q_rope = torch.from_numpy(q_rope_bnsd_tensor.astype(np.float32)).to(torch.float32)
        npu_q_rope = torch.from_numpy(q_rope_bnsd_tensor_quant).to(torch.bfloat16)


        gpu_q_rope = trans_bnsd_to_input_layout(gpu_q_rope, rope_shape, inputLayout, actualSeqLengths_q)
        npu_q_rope = trans_bnsd_to_input_layout(npu_q_rope, rope_shape, inputLayout, actualSeqLengths_q)

        tools.modify_alcnn_input_file(ids=28, origin_index=[28], type='tensor', mode='add', tensors=gpu_q_rope,
                                      params=params)
        tools.modify_alcnn_input_file(ids=23, origin_index=[23], type='tensor', mode='rewrite', tensors=npu_q_rope,
                                      params=params)
    else:
        q_rope_bnsd_tensor, q_rope_bnsd_shape = _n_trans_shape_to_bnsd(torch_tensor_list[28], rope_shape,
                                                                       inputLayout, numHeads, actualSeqLengths_q)
    ifa_param['q_rope_bnsd_tensor'] = q_rope_bnsd_tensor
    ifa_param['q_rope_shape_bnsd'] = q_rope_bnsd_shape

    k_rope_bnsd_tensor, k_rope_bnsd_shape = _n_trans_shape_to_bnsd(ifa_param['k_rope_tensor'],
                                                                   ifa_param['k_rope_shape'],
                                                                   kv_layout, numKeyValueHeads, actualSeqLengths)
    ifa_param['k_rope_bnsd_tensor'] = k_rope_bnsd_tensor
    ifa_param['k_rope_bnsd_shape'] = k_rope_bnsd_shape
    ifa_param['dequantScale1'] = dequantScale1_bnsd
    ifa_param['dequantScale2'] = torch_tensor_list[17]

    return ifa_param


def aclnn_op_func_ifa_cpu(torch_tensor_list, params):

    print(f"aclnn_op_func: {type(torch_tensor_list[0])}")
    print(params)
    print("cpu执行中...")
    ifa_param = get_param_fus(torch_tensor_list, params)
    # print("ifa_param :", ifa_param)

    if not ifa_param['normal_flag']:
        return torch.zeros(ifa_param['out_shape'])
    ifa_param["action_type"] = params["action_type"]

    # ===参数预处理===
    actualSeqLengths_q = ifa_param['actualSeqLengths_q_raw']
    actualSeqLengths = ifa_param['actualSeqLengths_raw']
    inputLayout = ifa_param['inputLayout']
    numHeads = ifa_param['numHeads']
    numKeyValueHeads = ifa_param['numKeyValueHeads']
    p_shape = ifa_param['p_shape']
    k_start_index = ifa_param['k_start_index']
    k_end_index = ifa_param['k_end_index']
    v_start_index = ifa_param['v_start_index']
    v_end_index = ifa_param['v_end_index']
    out_shape = ifa_param['out_shape']
    k_ori_shape_list = copy.deepcopy(ifa_param['k_shape_list'])
    v_ori_shape_list = copy.deepcopy(ifa_param['v_shape_list'])
    k_dtype = ifa_param['k_dtype']
    v_dtype = ifa_param['v_dtype']
    m_tensor = ifa_param['m_tensor']
    m_shape = ifa_param['m_shape']
    sparse_mode = ifa_param['sparse_mode']
    if ifa_param['tnd_flag'] and ifa_param['pa_flag']:
        kv_layout = "BNSD"
    else:
        kv_layout = inputLayout
    ifa_param['kv_layout'] = kv_layout

    # >> actualSeqLengths预处理：actualSeqLengths为单值场景，如果长度为1且b不为1，则将actualSeqLengths扩展为b个单值的列表
    # >> 仅非tnd场景，或tnd+pa场景，需要处理
    if not ifa_param['tnd_flag'] or (ifa_param['tnd_flag'] and ifa_param['pa_flag']):
        # actq
        if len(actualSeqLengths_q) == 1 and len(actualSeqLengths_q) != ifa_param['b']:
            actualSeqLengthsq_item = actualSeqLengths_q[0]
            for b_count in range(ifa_param['b'] - 1):
                actualSeqLengths_q.append(actualSeqLengthsq_item)
        if len(actualSeqLengths_q) > ifa_param['b']:
            actualSeqLengths_q = actualSeqLengths_q[:ifa_param['b']]
        # actkv
        if len(actualSeqLengths) == 1 and len(actualSeqLengths) != ifa_param['b']:
            actualSeqLengths_item = actualSeqLengths[0]
            for b_count in range(ifa_param['b'] - 1):
                actualSeqLengths.append(actualSeqLengths_item)
        # >> actualSeqLengths预处理：actualSeqLengths长度超过b
        if len(actualSeqLengths) > ifa_param['b']:
            actualSeqLengths = actualSeqLengths[:ifa_param['b']]
    if ifa_param['tnd_flag']:
        # 将tnd格式下的act seq转成普通的act seq
        actualSeqLengths_q = trans_tnd_actseq(actualSeqLengths_q)
        if not ifa_param['pa_flag']:
            actualSeqLengths = trans_tnd_actseq(actualSeqLengths)
        print(f"[INFO]trans_tnd_actseq end.")

    ifa_param['actualSeqLengths'] = actualSeqLengths
    ifa_param['actualSeqLengths_q'] = actualSeqLengths_q
    # deepseek 预处理： 1.将Q与QROPE拼接 2.将K与KROPE拼接 3.伪量化场景，k_antiscale与k_rope_antiscale拼接
    if ifa_param['deep_seek_flag']:
        ifa_param = deepseek_preprocessing(ifa_param)
    # >> q 预处理：将q转换为bnsd
    q_bnsd_tensor, q_bnsd_shape = _n_trans_shape_to_bnsd(ifa_param['q_tensor'], ifa_param['q_shape'], inputLayout,
                                                         numHeads, actualSeqLengths_q)
    ifa_param['q_bnsd_shape'] = q_bnsd_shape

    qs = q_bnsd_shape[2]
    ifa_param['q_tensor_bnsd'] = q_bnsd_tensor
    ifa_param['q_shape_bnsd'] = q_bnsd_shape
    ifa_param['q_d'] = q_bnsd_shape[3]

    # >> kv预处理：1、将kv list 转换为bnsd 2、GQA场景，将kvn扩展为qn
    k_tensor_list = []
    v_tensor_list = []
    k_shape_list = []
    v_shape_list = []
    k_tensor_bnsd_list = []
    v_tensor_bnsd_list = []
    k_shape_bnsd_list = []
    v_shape_bnsd_list = []
    ifa_param['ks_max'] = 0

    for i in range(0, ifa_param['k_num']):
        k_tensor, k_bnsd_shape = _n_trans_shape_to_bnsd(ifa_param['k_tensor_list'][i], ifa_param['k_shape_list'][i],
                                                        kv_layout,
                                                        numKeyValueHeads, actualSeqLengths)
        k_tensor_bnsd_list.append(k_tensor)
        k_shape_bnsd_list.append(k_bnsd_shape)
        # k int32预处理
        if ifa_param['k_dtype'] == "int32":
            k_tensor, k_bnsd_shape = trans_int32_2_int4_tensor_bnsd(k_tensor, k_bnsd_shape)

        if numKeyValueHeads != numHeads:
            if ifa_param['deep_seek_flag'] and ifa_param['in_quant_flag']:
                pass
            else:
                k_tensor, k_bnsd_shape = _t_broadcastKV_sigle(numHeads, numKeyValueHeads, k_tensor, k_dtype)
        if k_bnsd_shape[2] > ifa_param['ks_max']:
            ifa_param['ks_max'] = k_bnsd_shape[2]
        k_tensor_list.append(k_tensor)
        k_shape_list.append(k_bnsd_shape)

    x = 0
    for i in range(0, ifa_param['k_num']):
        v_tensor, v_bnsd_shape = _n_trans_shape_to_bnsd(ifa_param['v_tensor_list'][i], v_ori_shape_list[x], kv_layout,
                                                        numKeyValueHeads, actualSeqLengths)
        x = x + 1
        v_tensor_bnsd_list.append(v_tensor)
        v_shape_bnsd_list.append(v_bnsd_shape)
        # v int32预处理
        if ifa_param['v_dtype'] == "int32":
            v_tensor, v_bnsd_shape = trans_int32_2_int4_tensor_bnsd(v_tensor, v_bnsd_shape)

        if numKeyValueHeads != numHeads:
            if ifa_param['deep_seek_flag'] and ifa_param['in_quant_flag']:
                pass
            else:
                v_tensor, v_bnsd_shape = _t_broadcastKV_sigle(numHeads, numKeyValueHeads, v_tensor, v_dtype)
        v_tensor_list.append(v_tensor)
        v_shape_list.append(v_bnsd_shape)

    ifa_param['k_tensor_list_bnsd'] = k_tensor_list
    ifa_param['k_shape_list_bnsd'] = k_shape_list
    ifa_param['v_tensor_list_bnsd'] = v_tensor_list
    ifa_param['v_shape_list_bnsd'] = v_shape_list
    # 处理deepseek全量化：1.将qkv从FP16转成INT8 2.生成dequantscale1,dequantscale2
    if ifa_param['deep_seek_flag'] and ifa_param['in_quant_flag']:
        if ifa_param['action_type'] in ['bm_output_gold', 'bm_output']:
            ifa_param = deepseek_inquant_preprocessing_for_recovery(ifa_param, torch_tensor_list, params)
        else:
            ifa_param = deepseek_inquant_preprocessing(ifa_param, params)

    # >> left padding 异常场景处理：起点/终点异常，返回全0
    if ifa_param['padding_flag']:
        max_act_seq = max(actualSeqLengths)
        kv_s = ifa_param['ks_max']
        if kv_s - ifa_param['padding_size'] - max_act_seq < 0:
            print('[WARNING]:paddingsize 溢出，输出空tensor！')
            return torch.zeros(out_shape)
    action_type = params["action_type"]
    # >> pse 处理
    if ifa_param['p_flag']:
        maya = get_slopes(p_shape[1])
        maya = maya.numpy()
        pse_shift = np.zeros(p_shape)
        for n in range(p_shape[1]):
            alibi_biases = np.zeros([1, p_shape[-1]])
            for x in range(0, p_shape[-1]):
                alibi_biases[0, x] = -1 * x
            pse_shift[:, n:n + 1, :, :] = alibi_biases * maya[n]
        pse_shift = pse_shift.astype(np.float32)

        # 覆写
        if action_type in ["bm_output_gold", "bm_output"]:
            pass
        else:
            if params['dtype_input'][v_end_index + 1] == "fp16":
                pse_shift = pse_shift.astype(np.float16)
                pse_shift[pse_shift == -math.inf] = -65504
                ifa_param['p_tensor'] = pse_shift
                p_tensor = torch.tensor(pse_shift, dtype=torch.float16)
            elif params['dtype_input'][v_end_index + 1] == "bf16":
                pse_shift = pse_shift.astype(tf.bfloat16.as_numpy_dtype)
                pse_shift[pse_shift == -math.inf] = -65504
                ifa_param['p_tensor'] = pse_shift
                p_tensor = torch.tensor(pse_shift.astype(np.float32), dtype=torch.bfloat16)
            else:
                print('[ERROR]:wrong p_tensor dtype!')
                return torch.zeros(out_shape)
            tools.modify_alcnn_input_file(ids=3, origin_index=[3], type='tensor', mode='rewrite', tensors=p_tensor,params=params)

    # >> m预处理：1、将m扩展为BN1S  2、padding场景下，偏移部分设置为1 3、针对FP16格式，将tensor转成0/1
    m_dtype = tools.get_pt_dtype(params['dtype_input'][v_end_index + 2])
    m_tensor = torch.tensor(m_tensor, dtype=m_dtype)

    # qs大于1
    if sparse_mode == 3:
        batch = q_bnsd_shape[0]
        numheads = q_bnsd_shape[1]
        npu_m_shape_s = m_shape
        if sparse_mode == 0 or sparse_mode == 1:
            batch, numheads, ns1, ns2 = get_attention_mask_batch_num(m_shape,q_bnsd_shape)  # 获取输入attentionmask的batch 和numhead
            npu_m_shape_s = [ns1, ns2]
        kvs = k_shape_list[0][2]
        cpu_m_shape = [qs, kvs]  # cpu
        preTokens = ifa_param['pre_tokens']
        nextTokens = ifa_param['next_tokens']
        actualprefixKV = 0
        prefix_kvs = 0
        kvs_list = [kvs]
        cpu_m_tensor, npu_m_tensor, preTokens, nextTokens = _create_random_mask_by_spars(cpu_m_shape, npu_m_shape_s,
                                                                                         m_dtype, preTokens, nextTokens,
                                                                                         ifa_param[
                                                                                             'actualSeqLengths_q'],
                                                                                         actualSeqLengths,
                                                                                         actualprefixKV,
                                                                                         prefix_kvs, kvs_list, batch,
                                                                                         numheads, sparse_mode)

        npu_m_tensor = torch.from_numpy(npu_m_tensor).to(m_dtype)

        if action_type in ["bm_output", "bm_output_gold"]:
            pass
        else:
            tools.modify_alcnn_input_file(ids=4, origin_index=[4], type='tensor', mode='rewrite', tensors=npu_m_tensor,
                                          params=params)

        if sparse_mode == 0 or sparse_mode == 1:
            # m_bnsd_tensor = _np_broadcast_mask_n(cpu_m_tensor, npu_m_shape, cpu_m_shape, numHeads, q_bnsd_shape[0])
            pass
        else:
            cpu_m_tensor = np.array(cpu_m_tensor)
            cpu_m_tensor = torch.from_numpy(cpu_m_tensor)
            ifa_param['m_tensor'] = _t_broadcast_mask_n(cpu_m_tensor, cpu_m_tensor.shape, numHeads, q_bnsd_shape[0])
    else:
        # print(f"m_tensor_raw===={m_tensor}")
        if ifa_param['m_flag']:
            m_rewrite_flag = False
            if ifa_param['padding_flag']:
                m_tensor = cut_padding_size(m_tensor, actualSeqLengths, ifa_param['padding_size'])
                m_rewrite_flag = True
            if ifa_param['m_dtype'] == "float16":
                m_tensor = torch.where(m_tensor < 0.8, torch.tensor(0, dtype=torch.float16),
                                       torch.tensor(1, dtype=torch.float16))
                m_rewrite_flag = True
            if m_rewrite_flag:
                print(f"[INFO]rewrite attenmask")
                tools.modify_alcnn_input_file(ids=4, origin_index=[4], type='tensor', mode='rewrite',
                                              tensors=m_tensor, params=params, data_range=[0, 2])
            ifa_param['m_tensor'] = _t_broadcast_mask_n(m_tensor, m_shape, numHeads, m_shape[0])

    # >> 后量化参数处理：1、给dtype赋值 2、如果是perchannel模式，要将后量化参数统一转换成1n1d格式
    if ifa_param['out_quant_flag']:
        ifa_param['quantScale2_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 6])
        ifa_param['quantOffset2_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 7])
    if ifa_param['out_quant_pc_flag']:
        ifa_param['quantScale2'] = trans_to_1n1d(ifa_param['quantScale2'], ifa_param['quantScale2_shape'],
                                                 ifa_param['q_shape_bnsd'][3])
        if ifa_param['quantScale2'] is None:
            return torch.zeros(out_shape)
        ifa_param['quantOffset2'] = trans_to_1n1d(ifa_param['quantOffset2'], ifa_param['quantOffset2_shape'],
                                                  ifa_param['q_shape_bnsd'][3])
        if ifa_param['quantOffset2'] is None:
            return torch.zeros(out_shape)
        ifa_param['quantScale2_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 6])
        ifa_param['quantOffset2_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 7])

        # >> 处理pageatten场景（以下处理不涉及cpu、gpu等真值计算，仅为npu生成输入）：
        # 1、生成随机的block_table，并覆写原有bin文件
        # 2、将kv shape 统一转换成bsh
        # 3、生成kv cache
        # 4、将kv cache dump成新的bin文件，供aclnn接口调用
    if ifa_param['pa_flag'] and "output" not in params["action_type"]:
        # 生成blocktable
        blockSize = ifa_param['blocksize']
        blockTableShape = ifa_param['bt_shape']
        blockNum = ifa_param['k_cache_shape'][0]
        if 0 in ifa_param['bt_shape']:
            print('[WARNING]:block_table为空场景，输出空tensor！')
            return torch.zeros(out_shape)
        blockNumPerBlock = []
        block_num_min = 0
        for actual_seq in actualSeqLengths:
            blockNumPerBlock.append(math.ceil(actual_seq / blockSize))
            block_num_min += math.ceil(actual_seq / blockSize)
        if block_num_min > blockNum:
            print(f"[ERROR]Wrong input k_cache_shape: get blockNum = {blockNum}, but expect blockNum > {block_num_min}")
            exit(1)
        block_idx_list = np.arange(0, blockNum, 1)
        block_idx_list = np.random.permutation(block_idx_list).astype(np.int32)
        block_idx = 0
        block_table = [-1] * blockTableShape[1]
        block_table = np.tile(block_table, (blockTableShape[0], 1)).astype(np.int32)
        block_table_batch_idx = 0
        for idx in blockNumPerBlock:
            for j in range(idx):
                block_table[block_table_batch_idx][j] = (block_idx_list[block_idx])
                block_idx += 1
            block_table_batch_idx += 1
        ifa_param['block_table'] = block_table
        # 将block_table 覆盖原有的input
        # print("===>end gen block_table", block_table)
        tools.modify_alcnn_input_file(ids=12, origin_index=[12], type='tensor', mode='rewrite',
                                      tensors=torch.tensor(block_table, dtype=torch.int32),
                                      params=params)
        k_np_dtype = tools.get_np_dtype(params['dtype_input'][1])
        v_np_dtype = tools.get_np_dtype(params['dtype_input'][2])
        ifa_param['k_np_dtype'] = k_np_dtype
        ifa_param['v_np_dtype'] = v_np_dtype
        print(f"[PageAtten]Input Kdtype:{params['dtype_input'][1]} Vdtype:{params['dtype_input'][2]}")
        k_cache = np.zeros(ifa_param['k_cache_shape']).astype(k_np_dtype)
        v_cache = np.zeros(ifa_param['v_cache_shape']).astype(v_np_dtype)
        if not ifa_param['deep_seek_flag']:
            print('[INFO]:PA + no Deepseek！')
            # gen kv cache
            if len(ifa_param['k_cache_shape']) == 3:
                # trans kv to bsh(此处使用的tensor, 没有经过n的扩展)
                k_tensor_bsh_raw = trans_bnsd_to_bsh(k_tensor_bnsd_list[0], k_shape_bnsd_list[0])
                v_tensor_bsh_raw = trans_bnsd_to_bsh(v_tensor_bnsd_list[0], v_shape_bnsd_list[0])

                # kv paddIng
                kvh = numKeyValueHeads * q_bnsd_shape[3]
                if ifa_param['k_dtype'] == "int32":
                    kvh = int(kvh / 8)
                k_tensor_bsh = np.zeros((q_bnsd_shape[0], blockTableShape[1] * blockSize, kvh)).astype(k_np_dtype)
                v_tensor_bsh = np.zeros((q_bnsd_shape[0], blockTableShape[1] * blockSize, kvh)).astype(v_np_dtype)
                k_tensor_bsh[:, :k_tensor_bsh_raw.shape[1], :] = k_tensor_bsh_raw[:, :, :]
                v_tensor_bsh[:, :v_tensor_bsh_raw.shape[1], :] = v_tensor_bsh_raw[:, :, :]
                for b in range(q_bnsd_shape[0]):
                    for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                        block_offset = block_i * blockSize
                        if kv_cache_blk_id == -1:
                            continue
                        else:
                            k_cache[kv_cache_blk_id, 0:blockSize, :] = k_tensor_bsh[b,
                                                                       block_offset:(block_offset + blockSize), :]
                            v_cache[kv_cache_blk_id, 0:blockSize, :] = v_tensor_bsh[b,
                                                                       block_offset:(block_offset + blockSize), :]
            elif len(ifa_param['k_cache_shape']) == 4:
                kvn = k_tensor_bnsd_list[0].shape[1]
                kvs = k_tensor_bnsd_list[0].shape[2]
                kvd = k_tensor_bnsd_list[0].shape[3]
                k_tensor_bnsd = np.zeros((q_bnsd_shape[0], kvn, blockTableShape[1] * blockSize, kvd)).astype(k_np_dtype)
                v_tensor_bnsd = np.zeros((q_bnsd_shape[0], kvn, blockTableShape[1] * blockSize, kvd)).astype(v_np_dtype)
                k_tensor_bnsd[:, :, :kvs, :] = k_tensor_bnsd_list[0][:, :, :, :]
                v_tensor_bnsd[:, :, :kvs, :] = v_tensor_bnsd_list[0][:, :, :, :]

                for b in range(q_bnsd_shape[0]):
                    for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                        block_offset = block_i * blockSize
                        if kv_cache_blk_id == -1:
                            continue
                        else:
                            for n in range(q_bnsd_shape[1]):
                                k_cache[kv_cache_blk_id, n:n + 1, 0:blockSize, :] = k_tensor_bnsd[b, n:n + 1,
                                                                                    block_offset:(
                                                                                            block_offset + blockSize),
                                                                                    :]
                                v_cache[kv_cache_blk_id, n:n + 1, 0:blockSize, :] = v_tensor_bnsd[b, n:n + 1,
                                                                                    block_offset:(
                                                                                            block_offset + blockSize), :]
            elif len(ifa_param['k_cache_shape']) == 5:  # NZ
                kvn = k_tensor_bnsd_list[0].shape[1]
                kvs = k_tensor_bnsd_list[0].shape[2]
                kvd = k_tensor_bnsd_list[0].shape[3]
                k_tensor_bnsd = np.zeros((q_bnsd_shape[0], kvn, blockTableShape[1] * blockSize, kvd)).astype(k_np_dtype)
                v_tensor_bnsd = np.zeros((q_bnsd_shape[0], kvn, blockTableShape[1] * blockSize, kvd)).astype(v_np_dtype)
                k_tensor_bnsd[:, :, :kvs, :] = k_tensor_bnsd_list[0][:, :, :, :]
                v_tensor_bnsd[:, :, :kvs, :] = v_tensor_bnsd_list[0][:, :, :, :]
                k_cache_tensor_bnbd = np.zeros((k_cache.shape[0], kvn, blockSize, kvd)).astype(k_np_dtype)
                v_cache_tensor_bnbd = np.zeros((k_cache.shape[0], kvn, blockSize, kvd)).astype(k_np_dtype)
                for b in range(q_bnsd_shape[0]):
                    for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                        block_offset = block_i * blockSize
                        if kv_cache_blk_id == -1:
                            continue
                        else:
                            for n in range(q_bnsd_shape[1]):
                                k_cache_tensor_bnbd[kv_cache_blk_id, n:n + 1, 0:blockSize, :] = k_tensor_bnsd[b,
                                                                                                n:n + 1,
                                                                                                block_offset:(block_offset + blockSize),:]
                                v_cache_tensor_bnbd[kv_cache_blk_id, n:n + 1, 0:blockSize, :] = v_tensor_bnsd[b,
                                                                                                n:n + 1,
                                                                                                block_offset:(block_offset + blockSize),:]
                if k_dtype == "int8":
                    base = 32
                else:
                    base = 16
                k_cache = k_cache_tensor_bnbd.reshape(k_cache_tensor_bnbd.shape[0], k_cache_tensor_bnbd.shape[1],
                                                      k_cache_tensor_bnbd.shape[2],
                                                      k_cache_tensor_bnbd.shape[3] // base, base).transpose(0, 1, 3, 2,
                                                                                                            4)
                v_cache = v_cache_tensor_bnbd.reshape(k_cache_tensor_bnbd.shape[0], k_cache_tensor_bnbd.shape[1],
                                                      k_cache_tensor_bnbd.shape[2],
                                                      k_cache_tensor_bnbd.shape[3] // base, base).transpose(0, 1, 3, 2,
                                                                                                            4)
            else:
                print(f"[ERROR]Wrong input kv_cache_shape: {ifa_param['k_cache_shape']}")
                return torch.zeros(out_shape)
            # 将kv cache 生成新的bin文件

            k_cache_index = 21
            v_cache_index = 22
            tools.modify_alcnn_input_file(ids=k_cache_index, origin_index=[k_cache_index], type='tensor_list',
                                          mode='rewrite',
                                          tensors=[k_cache],
                                          params=params,
                                          data_dtype=params['dtype_input'][1])
            tools.modify_alcnn_input_file(ids=v_cache_index, origin_index=[v_cache_index], type='tensor_list',
                                          mode='rewrite',
                                          tensors=[v_cache],
                                          params=params,
                                          data_dtype=params['dtype_input'][2])
        else:
            print('[INFO]:PA + Deepseek！')
            if ifa_param['in_quant_flag']:
                deepseek_inquant_ds_pa_preprocessing(ifa_param, params)
            else:
                deepseek_ds_pa_preprocessing(ifa_param, params)
    # >> kv 伪量化预处理
    if ifa_param['kv_quant_flag']:
        antiquantscale_shape_raw = ifa_param['antiquantscale_shape_raw']
        antiquantscale_tensor_raw = ifa_param['antiquantscale_tensor_raw']
        antiquantoffset_shape_raw = ifa_param['antiquantoffset_shape_raw']
        antiquantoffset_tensor_raw = ifa_param['antiquantoffset_tensor_raw']

        # >> [TEMP]pertoken_group：1、将kv量化参数统一转换成2,B,N,S,D/32 2、GQA场景下，扩展kv量化参数
        if ifa_param['kv_quant_ptog_flag']:
            antiquantscale_tensor = _trans_antiparam_to_2bnsg(antiquantscale_shape_raw, antiquantscale_tensor_raw,
                                                              inputLayout, numKeyValueHeads)
            antiquantoffset_tensor = _trans_antiparam_to_2bnsg(antiquantoffset_shape_raw, antiquantoffset_tensor_raw,
                                                               inputLayout, numKeyValueHeads)
            antiquantscale_tensor = broadcast_kv_dequant_tensor_fp8e8m0(antiquantscale_tensor, numKeyValueHeads,
                                                                        numHeads)
            antiquantoffset_tensor = broadcast_kv_dequant_tensor_fp8e8m0(antiquantoffset_tensor, numKeyValueHeads,
                                                                         numHeads)

            ifa_param['antiquantScale'] = antiquantscale_tensor
            ifa_param['antiquantOffset'] = antiquantoffset_tensor
        # pertensor/perchannel ->1、将kv量化参数统一转换成2n1d 2、GQA场景下，扩展kv量化参数
        elif not ifa_param['kv_quant_pto_flag']:
            # 将kv量化参数转换为2n1d(匹配bnsd)
            antiquantscale_tensor = _trans_antiparam_to_2n1d(antiquantscale_shape_raw, antiquantscale_tensor_raw,
                                                             inputLayout, numKeyValueHeads, ifa_param['q_d'])
            antiquantoffset_tensor = _trans_antiparam_to_2n1d(antiquantoffset_shape_raw, antiquantoffset_tensor_raw,
                                                              inputLayout, numKeyValueHeads, ifa_param['q_d'])
            # GQA场景，扩展kv反量化参数
            q_dtype_np = tools.get_np_dtype(params['dtype_input'][0])
            antiquantscale_tensor = broadcast_kv_dequant_tensor(antiquantscale_tensor, numKeyValueHeads, numHeads,
                                                                q_dtype_np)
            antiquantoffset_tensor = broadcast_kv_dequant_tensor(antiquantoffset_tensor, numKeyValueHeads, numHeads,
                                                                 q_dtype_np)

            ifa_param['antiquantScale'] = antiquantscale_tensor
            ifa_param['antiquantOffset'] = antiquantoffset_tensor
        else:
            ifa_param['antiquantScale'] = antiquantscale_tensor_raw
            ifa_param['antiquantOffset'] = antiquantoffset_tensor_raw
    # 判断kv分离的伪量化
    if ifa_param["k_antiquant_flag"] and not ifa_param["v_antiquant_flag"]:
        print("[ERROR] k_antiquant exit, but v_antiquant not exit")
        exit(-1)
    elif not ifa_param["k_antiquant_flag"] and ifa_param["v_antiquant_flag"]:
        print("[ERROR] v_antiquant exit, but k_antiquant not exit")
        exit(-1)
    else:
        if ifa_param["k_antiquant_flag"]:
            k_antiquantscale_shape_raw = ifa_param['k_antiquantscale_shape_raw']
            k_antiquantscale_tensor_raw = ifa_param['k_antiquantscale_tensor_raw']
            k_antiquantoffset_shape_raw = ifa_param['k_antiquantoffset_shape_raw']
            k_antiquantoffset_tensor_raw = ifa_param['k_antiquantoffset_tensor_raw']
            print(f"k_antiquant_pto_flag: {ifa_param['k_antiquant_pto_flag']}")
            print(f"k_antiquant_ptoh_flag: {ifa_param['k_antiquant_ptoh_flag']}")
            print(f"k_antiquant_ptopa_flag: {ifa_param['k_antiquant_ptopa_flag']}")
            print(f"k_antiquant_ptohpa_flag: {ifa_param['k_antiquant_ptohpa_flag']}")
            # pertoken_head
            if ifa_param['k_antiquant_ptoh_flag']:
                # GQA场景，扩展kv反量化参数
                k_antiquantscale_tensor = broadcast_kv_split_dequant_tensor_ptoh(k_antiquantscale_tensor_raw,
                                                                                 numKeyValueHeads, numHeads)
                k_antiquantoffset_tensor = broadcast_kv_split_dequant_tensor_ptoh(k_antiquantoffset_tensor_raw,
                                                                                  numKeyValueHeads, numHeads)
                ifa_param['k_antiquantScale'] = k_antiquantscale_tensor
                ifa_param['k_antiquantOffset'] = k_antiquantoffset_tensor
            # pertensor/perchannel ->1、将kv量化参数统一转换成1n1d 2、GQA场景下，扩展kv量化参数
            elif ifa_param['k_antiquant_pc_flag'] or ifa_param['k_antiquant_pth_flag']:
                k_antiquantscale_tensor = _trans_antiparam_to_1n1d(k_antiquantscale_shape_raw,
                                                                   k_antiquantscale_tensor_raw,
                                                                   inputLayout, numKeyValueHeads, ifa_param['q_d'],
                                                                   ifa_param['k_antiquant_mode'])
                k_antiquantoffset_tensor = _trans_antiparam_to_1n1d(k_antiquantoffset_shape_raw,
                                                                    k_antiquantoffset_tensor_raw,
                                                                    inputLayout, numKeyValueHeads, ifa_param['q_d'],
                                                                    ifa_param['k_antiquant_mode'])
                # GQA场景，扩展kv反量化参数
                q_dtype_np = tools.get_np_dtype(params['dtype_input'][0])
                k_antiquantscale_tensor = broadcast_kv_split_dequant_tensor(k_antiquantscale_tensor, numKeyValueHeads,
                                                                            numHeads, q_dtype_np)
                k_antiquantoffset_tensor = broadcast_kv_split_dequant_tensor(k_antiquantoffset_tensor, numKeyValueHeads,
                                                                             numHeads, q_dtype_np)
                ifa_param['k_antiquantScale'] = k_antiquantscale_tensor
                ifa_param['k_antiquantOffset'] = k_antiquantoffset_tensor
            # pertoken_pa -> 1、将bb转成1bs
            elif ifa_param['k_antiquant_ptopa_flag']:
                if ifa_param['pa_flag']:
                    ifa_param['k_antiquantScale'] = trans_bb_2_1bs(k_antiquantscale_tensor_raw,
                                                                   ifa_param['block_table'], ifa_param['b'])
                    ifa_param['k_antiquantOffset'] = trans_bb_2_1bs(k_antiquantoffset_tensor_raw,
                                                                    ifa_param['block_table'], ifa_param['b'])
                else:
                    print("[ERROR]Got antiquant mode == 4,but not PA!")
                    exit(1)
            # pertoken_head_pa -> 1、将bb转成1bns
            elif ifa_param['k_antiquant_ptohpa_flag']:
                if ifa_param['pa_flag']:
                    k_antiquantscale_tensor = trans_bb_2_1bns(k_antiquantscale_tensor_raw, ifa_param['block_table'],
                                                              ifa_param['b'])
                    k_antiquantoffset_tensor = trans_bb_2_1bns(k_antiquantoffset_tensor_raw, ifa_param['block_table'],
                                                               ifa_param['b'])
                    k_antiquantscale_tensor = broadcast_kv_split_dequant_tensor_ptoh(k_antiquantscale_tensor,
                                                                                     numKeyValueHeads, numHeads)
                    k_antiquantoffset_tensor = broadcast_kv_split_dequant_tensor_ptoh(k_antiquantoffset_tensor,
                                                                                      numKeyValueHeads, numHeads)
                    ifa_param['k_antiquantScale'] = k_antiquantscale_tensor
                    ifa_param['k_antiquantOffset'] = k_antiquantoffset_tensor
                else:
                    print("[ERROR]Got antiquant mode == 5,but not PA!")
                    exit(1)

            elif ifa_param['k_antiquant_ptog_flag']:
                k_antiquantscale_tensor = broadcast_kv_split_antiquant_tensor_fp8e8m0(k_antiquantscale_tensor_raw,
                                                                                      numKeyValueHeads, numHeads)
                k_antiquantoffset_tensor = broadcast_kv_split_antiquant_tensor_fp8e8m0(k_antiquantoffset_tensor_raw,
                                                                                       numKeyValueHeads, numHeads)

                ifa_param['k_antiquantScale'] = k_antiquantscale_tensor
                ifa_param['k_antiquantOffset'] = k_antiquantoffset_tensor

            # pertoken -> 如果是BS,转成1BS
            else:
                ifa_param['k_antiquantScale'] = _trans_antiparam_to_1bs(k_antiquantscale_tensor_raw)
                ifa_param['k_antiquantOffset'] = _trans_antiparam_to_1bs(k_antiquantoffset_tensor_raw)

        if ifa_param["v_antiquant_flag"]:
            v_antiquantscale_shape_raw = ifa_param['v_antiquantscale_shape_raw']
            v_antiquantscale_tensor_raw = ifa_param['v_antiquantscale_tensor_raw']
            v_antiquantoffset_shape_raw = ifa_param['v_antiquantoffset_shape_raw']
            v_antiquantoffset_tensor_raw = ifa_param['v_antiquantoffset_tensor_raw']
            # pertoken_head
            if ifa_param['v_antiquant_ptoh_flag']:
                # GQA场景，扩展kv反量化参数
                v_antiquantscale_tensor = broadcast_kv_split_dequant_tensor_ptoh(v_antiquantscale_tensor_raw,
                                                                                 numKeyValueHeads, numHeads)
                v_antiquantoffset_tensor = broadcast_kv_split_dequant_tensor_ptoh(v_antiquantoffset_tensor_raw,
                                                                                  numKeyValueHeads, numHeads)
                ifa_param['v_antiquantScale'] = v_antiquantscale_tensor
                ifa_param['v_antiquantOffset'] = v_antiquantoffset_tensor
            # pertensor/perchannel ->1、将kv量化参数统一转换成1n1d 2、GQA场景下，扩展kv量化参数
            elif ifa_param['v_antiquant_pc_flag'] or ifa_param['v_antiquant_pth_flag']:
                v_antiquantscale_tensor = _trans_antiparam_to_1n1d(v_antiquantscale_shape_raw,
                                                                   v_antiquantscale_tensor_raw,
                                                                   inputLayout, numKeyValueHeads, ifa_param['q_d'],
                                                                   ifa_param['v_antiquant_mode'])
                v_antiquantoffset_tensor = _trans_antiparam_to_1n1d(v_antiquantoffset_shape_raw,
                                                                    v_antiquantoffset_tensor_raw,
                                                                    inputLayout, numKeyValueHeads, ifa_param['q_d'],
                                                                    ifa_param['v_antiquant_mode'])
                # GQA场景，扩展kv反量化参数
                q_dtype_np = tools.get_np_dtype(params['dtype_input'][0])
                v_antiquantscale_tensor = broadcast_kv_split_dequant_tensor(v_antiquantscale_tensor, numKeyValueHeads,
                                                                            numHeads, q_dtype_np)
                v_antiquantoffset_tensor = broadcast_kv_split_dequant_tensor(v_antiquantoffset_tensor, numKeyValueHeads,
                                                                             numHeads, q_dtype_np)
                ifa_param['v_antiquantScale'] = v_antiquantscale_tensor
                ifa_param['v_antiquantOffset'] = v_antiquantoffset_tensor
            # pertoken_pa -> 1、将bb转成1bs
            elif ifa_param['v_antiquant_ptopa_flag']:
                if ifa_param['pa_flag']:
                    ifa_param['v_antiquantScale'] = trans_bb_2_1bs(v_antiquantscale_tensor_raw,
                                                                   ifa_param['block_table'], ifa_param['b'])
                    ifa_param['v_antiquantOffset'] = trans_bb_2_1bs(v_antiquantoffset_tensor_raw,
                                                                    ifa_param['block_table'], ifa_param['b'])
                else:
                    print("[ERROR]Got antiquant mode == 4,but not PA!")
                    exit(1)

            # per_token_head_pa -> 1、将bnb转成1bns
            elif ifa_param['v_antiquant_ptohpa_flag']:
                print(v_antiquantscale_tensor_raw.shape)
                if ifa_param['pa_flag']:
                    v_antiquantscale_tensor = trans_bb_2_1bns(v_antiquantscale_tensor_raw, ifa_param['block_table'],
                                                              ifa_param['b'])
                    v_antiquantoffset_tensor = trans_bb_2_1bns(v_antiquantoffset_tensor_raw, ifa_param['block_table'],
                                                               ifa_param['b'])
                    v_antiquantscale_tensor = broadcast_kv_split_dequant_tensor_ptoh(v_antiquantscale_tensor,
                                                                                     numKeyValueHeads, numHeads)
                    v_antiquantoffset_tensor = broadcast_kv_split_dequant_tensor_ptoh(v_antiquantoffset_tensor,
                                                                                      numKeyValueHeads, numHeads)
                    ifa_param['v_antiquantScale'] = v_antiquantscale_tensor
                    ifa_param['v_antiquantOffset'] = v_antiquantoffset_tensor
                else:
                    print("[ERROR]Got antiquant mode == 5,but not PA!")
                    exit(1)

            # per_token_group -> 1、将bb转成1bs
            elif ifa_param['v_antiquant_ptog_flag']:
                v_antiquantscale_tensor = broadcast_kv_split_antiquant_tensor_fp8e8m0(v_antiquantscale_tensor_raw,
                                                                                      numKeyValueHeads,
                                                                                      numHeads)
                v_antiquantoffset_tensor = broadcast_kv_split_antiquant_tensor_fp8e8m0(v_antiquantoffset_tensor_raw,
                                                                                       numKeyValueHeads,
                                                                                       numHeads)

                ifa_param['v_antiquantScale'] = v_antiquantscale_tensor
                ifa_param['v_antiquantOffset'] = v_antiquantoffset_tensor
            # pertoken -> 如果是BS,转成1BS
            else:
                ifa_param['v_antiquantScale'] = _trans_antiparam_to_1bs(v_antiquantscale_tensor_raw)
                ifa_param['v_antiquantOffset'] = _trans_antiparam_to_1bs(v_antiquantoffset_tensor_raw)

    # >> 输入量化预处理：获取切分使用的sinner
    ifa_param['sinner'] = 0
    if ifa_param['in_quant_flag'] and not ifa_param['deep_seek_flag']:
        sinner = get_sinner(ifa_param)
        ifa_param['sinner'] = sinner

    # psefix 预处理：1.转成1nsd 2.GQA场景扩展N; 3.按act_prefix裁剪 4.获取perfix_act_lens
    if ifa_param['prefix_flag']:
        k_prefix_tensor = ifa_param['k_prefix_tensor']
        k_prefix_shape = ifa_param['k_prefix_shape']
        v_prefix_tensor = ifa_param['v_prefix_tensor']
        v_prefix_shape = ifa_param['v_prefix_shape']
        k_prefix_tensor, k_prefix_shape = _n_trans_shape_to_bnsd(k_prefix_tensor, k_prefix_shape, inputLayout,
                                                                 numKeyValueHeads)
        v_prefix_tensor, v_prefix_shape = _n_trans_shape_to_bnsd(v_prefix_tensor, v_prefix_shape, inputLayout,
                                                                 numKeyValueHeads)

        # kv prefix int32预处理
        if ifa_param['k_prefix_dtype'] == "int32":
            k_prefix_tensor, k_prefix_shape = trans_int32_2_int4_tensor_bnsd(k_prefix_tensor, k_prefix_shape)
        if ifa_param['v_prefix_dtype'] == "int32":
            v_prefix_tensor, v_prefix_shape = trans_int32_2_int4_tensor_bnsd(v_prefix_tensor, v_prefix_shape)

        if numKeyValueHeads != numHeads:
            k_prefix_tensor, k_prefix_shape = _t_broadcastKV_sigle(numHeads, numKeyValueHeads, k_prefix_tensor,
                                                                   ifa_param['k_prefix_dtype'])
            v_prefix_tensor, v_prefix_shape = _t_broadcastKV_sigle(numHeads, numKeyValueHeads, v_prefix_tensor,
                                                                   ifa_param['v_prefix_dtype'])

        if ifa_param['prefix_act_flag']:
            k_prefix_tensor = k_prefix_tensor[:, :, :ifa_param['prefix_act_lens'], :]
            v_prefix_tensor = v_prefix_tensor[:, :, :ifa_param['prefix_act_lens'], :]
        else:
            ifa_param['prefix_act_lens'] = k_prefix_shape[2]

        ifa_param['k_prefix_tensor'] = k_prefix_tensor
        ifa_param['v_prefix_tensor'] = v_prefix_tensor

        print(f"prefix_act_lens:{str(ifa_param['prefix_act_lens'])}")

    # ===CPU计算入口===
    y_all, lse = _t_increattention_bnsd(ifa_param)
    y_all = torch.from_numpy(y_all)
    y_all = trans_bnsd_to_layout(y_all, out_shape, inputLayout, actualSeqLengths_q)

    print(f"out_shape:{out_shape}")
    if ifa_param["softmax_lse_flag"]:
        print("softmax_lse_flag:", ifa_param["softmax_lse_flag"])
        print("inputLayout:", inputLayout)
        if inputLayout != "TND":
            lse = lse.astype('float64')
            lse = torch.from_numpy(lse)
        elif inputLayout == "TND":
            B = ifa_param['q_shape_bnsd'][0]
            T = ifa_param['q_shape'][0]
            N = ifa_param['q_shape'][1]
            lse_output = np.zeros((T, N, 1), dtype=lse.dtype)
            t_start = 0
            for b_index in range(B):
                act_s = actualSeqLengths_q[b_index]
                t_end = t_start + act_s
                if act_s == 0:
                    continue
                for n_index in range(N):
                    lse_output[t_start:t_end, n_index, :] = lse[b_index, n_index, :act_s, :]
                t_start += act_s
            lse = lse_output
            lse = torch.from_numpy(lse.astype(np.float32))
        return y_all, lse
    else:
        return y_all

def ifa_gpu_flash_attn_with_kvcache(torch_tensor_list, params, ifa_param):
    from flash_attn.flash_attn_interface import flash_attn_with_kvcache
    print("flash_attn_with_kvcache...")
    sparse_mode = ifa_param['sparse_mode']
    causal = True if sparse_mode == 3 else False
    q_shape = ifa_param['q_shape']
    q_tensor = ifa_param['q_tensor']
    q_dtype = ifa_param['q_dtype']
    act_seq_list = ifa_param['actualSeqLengths_raw']
    inputLayout = ifa_param['inputLayout']
    numHeads = ifa_param['numHeads']
    numKeyValueHeads = ifa_param['numKeyValueHeads']
    scaleValue = ifa_param['scaleValue']
    out_shape = ifa_param['out_shape']
    k_ori_shape_list = ifa_param['k_shape_list']
    v_ori_shape_list = ifa_param['v_shape_list']
    k_ori_tensor_list = ifa_param['k_tensor_list']
    v_ori_tensor_list = ifa_param['v_tensor_list']
    flag_list = ifa_param['flag_list']

    # 参数处理
    k_tensor = None
    v_tensor = None
    q_tensor = _bsh_bnsd_to_bsnd_for_gpu(q_tensor, inputLayout, q_shape, numHeads)
    q_d = q_tensor.shape[-1]
    kvs_list = ifa_param['kv_s_list']
    kvs_max = max(kvs_list)
    # 判断是否非连续且s不等长场景，如果是，则需要打开act_seq
    bdc_flag = False
    if not ifa_param['continue_flag']:
        if all(x == kvs_list[0] for x in kvs_list):
            bdc_flag = False
            print("[INFO]>>非连续等长场景<<")
        else:
            bdc_flag = True
            print("[INFO]>>非连续不等长场景<<")

    # kv 预处理
    if ifa_param['pa_flag'] and len(ifa_param['k_cache_shape'])!=5:
        print(f"-----------PA--------------")
        blockSize = int(ifa_param['blocksize'])
        blockNum = int(ifa_param['k_cache_shape'][0])
        kvn = int(ifa_param['numKeyValueHeads'])
        k_tensor = torch_tensor_list[21]
        v_tensor = torch_tensor_list[22]
        if len(ifa_param['k_cache_shape']) == 3:
            D = int(int(ifa_param['k_cache_shape'][2]) / kvn)
        else:
            D = int(ifa_param['k_cache_shape'][3])
        k_tensor = k_tensor.reshape([blockNum, blockSize, kvn, D])
        v_tensor = v_tensor.reshape([blockNum, blockSize, kvn, D])
        bt = torch_tensor_list[12]
    else:
        print(f"-----------NOT PA--------------")
        bt = None
        for i in range(len(k_ori_shape_list)):
            k_sub_tensor = _bsh_bnsd_to_bsnd_for_gpu(k_ori_tensor_list[i], inputLayout, k_ori_shape_list[i],
                                                     numKeyValueHeads)
            # 不等长场景padding
            if k_sub_tensor.shape[1] < kvs_max:
                pad_length = kvs_max - k_sub_tensor.shape[1]
                k_sub_tensor = torch.nn.functional.pad(k_sub_tensor, (0, 0, 0, 0, 0, pad_length, 0, 0),
                                                       mode='constant', value=0)
            # 拼接
            if k_tensor is None:
                if k_sub_tensor is not None:
                    k_tensor = k_sub_tensor
            else:
                if k_sub_tensor is not None:
                    k_tensor = torch.cat([k_tensor, k_sub_tensor], dim=0)

        for j in range(len(v_ori_shape_list)):
            v_sub_tensor = _bsh_bnsd_to_bsnd_for_gpu(v_ori_tensor_list[j], inputLayout, v_ori_shape_list[j],
                                                     numKeyValueHeads)
            # 不等长场景padding
            if v_sub_tensor.shape[1] < kvs_max:
                pad_length = kvs_max - v_sub_tensor.shape[1]
                v_sub_tensor = torch.nn.functional.pad(v_sub_tensor, (0, 0, 0, 0, 0, pad_length, 0, 0),
                                                       mode='constant', value=0)
            # 拼接
            if v_tensor is None:
                if v_sub_tensor is not None:
                    v_tensor = v_sub_tensor
            else:
                if v_sub_tensor is not None:
                    v_tensor = torch.cat([v_tensor, v_sub_tensor], dim=0)
    # kv分离
    if ifa_param["k_antiquant_flag"] and ifa_param["v_antiquant_flag"]:
        k_antiquant_scale = ifa_param['k_antiquantscale_tensor_raw'].reshape(1, 1, numKeyValueHeads, -1)
        # k_antiquant_scale = ifa_param['k_antiquantscale_tensor_raw'].reshape(1,1,int(numKeyValueHeads), k_ori_shape_list[0][2] // int(numKeyValueHeads))
        k_antiquant_offset = ifa_param['k_antiquantoffset_tensor_raw'].reshape(1, 1, numKeyValueHeads, -1)
        # k_antiquant_offset = ifa_param['k_antiquantoffset_tensor_raw'].reshape(1,1,int(numKeyValueHeads), k_ori_shape_list[0][2] // int(numKeyValueHeads))

        v_antiquant_scale = ifa_param['v_antiquantscale_tensor_raw'].reshape(1, 1, numKeyValueHeads, -1)
        # v_antiquant_scale = ifa_param['v_antiquantscale_tensor_raw'].reshape(1,1,int(numKeyValueHeads), v_ori_shape_list[0][2] // int(numKeyValueHeads))
        v_antiquant_offset = ifa_param['k_antiquantoffset_tensor_raw'].reshape(1, 1, numKeyValueHeads, -1)
        # v_antiquant_offset = ifa_param['v_antiquantoffset_tensor_raw'].reshape(1,1,int(numKeyValueHeads), v_ori_shape_list[0][2] // int(numKeyValueHeads))
        k_new_tensor, v_new_tensor = kv_split_antiquant_tensor(k_tensor, v_tensor,
                                                               torch.tensor(k_antiquant_scale).cuda(),
                                                               torch.tensor(k_antiquant_offset).cuda(),
                                                               torch.tensor(v_antiquant_scale).cuda(),
                                                               torch.tensor(v_antiquant_offset).cuda(),
                                                               ifa_param['k_antiquant_mode'],
                                                               ifa_param['v_antiquant_mode'], ifa_param['k_dtype'],
                                                               ifa_param['q_dtype'])
        k_tensor = torch.tensor(k_new_tensor).cuda()
        v_tensor = torch.tensor(v_new_tensor).cuda()
        if str(q_dtype) == "bfloat16":
            k_tensor = k_tensor.cuda().bfloat16()
            v_tensor = v_tensor.cuda().bfloat16()
        else:
            k_tensor = k_tensor.cuda().half()
            v_tensor = v_tensor.cuda().half()
    # 伪量化预处理
    elif ifa_param['kv_quant_flag']:
        antiquantscale_shape_raw = ifa_param['antiquantscale_shape_raw']
        antiquantscale_tensor_raw = ifa_param['antiquantscale_tensor_raw']
        antiquantoffset_shape_raw = ifa_param['antiquantoffset_shape_raw']
        antiquantoffset_tensor_raw = ifa_param['antiquantoffset_tensor_raw']
        antiquantscale_tensor_raw = antiquantscale_tensor_raw.float().cpu().numpy()
        if type(antiquantoffset_tensor_raw) == torch.Tensor:
            antiquantoffset_tensor_raw = antiquantoffset_tensor_raw.float().cpu().numpy()
        # pertensor/perchannel ->1、将kv量化参数统一转换成2n1d
        if not ifa_param['kv_quant_pto_flag']:
            print("[INFO]>>伪量化pertensor/perchannel场景<<")
            # 将kv量化参数转换为2n1d(匹配bnsd)
            antiquantscale_tensor = _trans_antiparam_to_2n1d(antiquantscale_shape_raw, antiquantscale_tensor_raw,
                                                             inputLayout, numKeyValueHeads, q_d)
            antiquantoffset_tensor = _trans_antiparam_to_2n1d(antiquantoffset_shape_raw, antiquantoffset_tensor_raw,
                                                              inputLayout, numKeyValueHeads, q_d)
        else:
            print("[INFO]>>伪量化pertoken场景<<")
            antiquantscale_tensor = antiquantscale_tensor_raw
            antiquantoffset_tensor = antiquantoffset_tensor_raw

        # 伪量化
        k_tensor = k_tensor.cpu().numpy()
        v_tensor = v_tensor.cpu().numpy()
        k_tensor_out = np.zeros(k_tensor.shape, dtype=np.float32)
        v_tensor_out = np.zeros(v_tensor.shape, dtype=np.float32)
        for b_index in range(ifa_param['b']):
            if ifa_param['kv_quant_pto_flag']:
                antiquantScale_cur = antiquantscale_tensor[:, b_index:(b_index + 1), :]
                antiquantOffset_cur = antiquantoffset_tensor[:, b_index:(b_index + 1), :]
            else:
                antiquantScale_cur = antiquantscale_tensor
                antiquantOffset_cur = antiquantoffset_tensor
            k_cur = k_tensor[b_index:(b_index + 1), :, :, :].transpose([0, 2, 1, 3])
            v_cur = v_tensor[b_index:(b_index + 1), :, :, :].transpose([0, 2, 1, 3])
            k_cur, v_cur = antiquant(k_cur, v_cur, antiquantScale_cur, antiquantOffset_cur,
                                     ifa_param['kv_quant_pto_flag'])
            k_tensor_out[b_index:(b_index + 1), :, :, :] = k_cur.transpose([0, 2, 1, 3])
            v_tensor_out[b_index:(b_index + 1), :, :, :] = v_cur.transpose([0, 2, 1, 3])
        k_tensor = torch.from_numpy(k_tensor_out)
        v_tensor = torch.from_numpy(v_tensor_out)
        if str(q_dtype) == "bfloat16":
            k_tensor = k_tensor.cuda().bfloat16()
            v_tensor = v_tensor.cuda().bfloat16()
        else:
            k_tensor = k_tensor.cuda().half()
            v_tensor = v_tensor.cuda().half()
    # act_seq获取
    if not ifa_param['as_flag']:
        if bdc_flag:
            act_seq_list = ifa_param['kv_s_list']
        else:
            act_seq_list = None
    else:
        if len(act_seq_list) == 1 and q_shape[0] != 1:
            actualSeqLengths_item = act_seq_list[0]
            for b_count in range(q_shape[0] - 1):
                act_seq_list.append(actualSeqLengths_item)

    # 左padding处理
    if ifa_param['padding_flag']:
        print("[INFO]>>left padding场景<<")
        k_tensor = left_padding_handler(k_tensor, act_seq_list, ifa_param['padding_size'])
        v_tensor = left_padding_handler(v_tensor, act_seq_list, ifa_param['padding_size'])
        if ifa_param['m_flag']:
            m_tensor = ifa_param['m_tensor'].squeeze()
            print(m_tensor)
            m_tensor = left_padding_handler_mask(m_tensor, act_seq_list, ifa_param['padding_size'])
            print(m_tensor)

    # mask预处理
    if ifa_param['m_flag']:
        print("[INFO]>>mask场景<<")
        m_tensor = ifa_param['m_tensor']
        k_tensor, new_ks_list = cut_kvs_by_mask(k_tensor, m_tensor)
        v_tensor, new_vs_list = cut_kvs_by_mask(v_tensor, m_tensor)
        print(k_tensor.shape)

        if act_seq_list:
            act_seq_list = get_min_list(act_seq_list, new_ks_list)
        else:
            act_seq_list = new_ks_list

    # 参数转换
    if act_seq_list:
        print(f"[INFO]>>act_seq_list:{act_seq_list}<<")
        actualseqlengths = torch.from_numpy(np.array(act_seq_list)).to(torch.int32)
        actualseqlengths = actualseqlengths.cuda().requires_grad_(False)
    else:
        print("[INFO]>>act_seq_list:None<<")
        actualseqlengths = None

    q_tensor = q_tensor.cuda().requires_grad_(False)
    k_cache = k_tensor.cuda().requires_grad_(False)
    v_cache = v_tensor.cuda().requires_grad_(False)

    pse = None
    if flag_list[3]:
        pse = get_slopes(numHeads).cuda().requires_grad_(False)
    print(k_cache.shape)
    gpu_out = flash_attn_with_kvcache(q=q_tensor, k_cache=k_cache, v_cache=v_cache, k=None, v=None,
                                      cache_seqlens=actualseqlengths,
                                      alibi_slopes=pse, causal=causal, block_table=bt)

    gpu_out = gpu_out.cpu()
    if inputLayout == "BSH":
        gpu_out = gpu_out.reshape(out_shape)
    elif inputLayout == "BNSD":
        gpu_out = gpu_out.permute([0, 2, 1, 3])
    elif inputLayout in ["BNSD_NBSD", "BSND_NBSD"]:
        gpu_out = gpu_out.permute([0, 2, 1, 3]).permute([1, 0, 2, 3])
    else:
        pass

    return gpu_out


def _tnd_to_bsnd_for_gpu(tensor, shape, actualseqlengths, pad_value=0):
    T = int(shape[0])  # S == T
    N = int(shape[1])
    D = int(shape[2])

    diff_list = [actualseqlengths[0]]
    for idx in range(len(actualseqlengths) - 1):
        diff_list.append(actualseqlengths[idx + 1] - actualseqlengths[idx])
    S = max(diff_list)

    tensor_list = []
    tensor_start_idx = 0
    for seq_length in actualseqlengths:
        temp_tensor = torch.full((1, S, N, D), pad_value, dtype=tensor.dtype)
        cur_tensor = tensor[tensor_start_idx:seq_length]
        temp_tensor[0][:seq_length - tensor_start_idx] = cur_tensor
        tensor_list.append(temp_tensor)
        tensor_start_idx = seq_length

    result = torch.cat(tensor_list, dim=0).to(tensor.device)
    return result


def ifa_gpu_flash_mla_with_kvcache(torch_tensor_list, params, ifa_param):
    # ===参数对应关系===
    # cache_seqlens:actseq_k
    # s_q: qs
    # h_q: qn
    # h_kv: kn
    # q:q + qrope
    # blocked_k:kcache + kropecache (需要重新生成)
    # block_table: blocktable(需要重新生成)
    # d:qd+qrope_d
    # dv:qd
    # causal: sparse3时为true, sparse0时为false
    # =================
    print("flash_mla_with_kvcache...")
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata
    import triton
    import torch.nn.functional as F
    # 获取参数
    q_shape = ifa_param['q_shape']
    q_tensor = ifa_param['q_tensor']
    q_dtype = ifa_param['q_dtype']
    q_rope_shape = ifa_param['q_rope_shape']
    q_rope_tensor = ifa_param['q_rope_tensor']
    k_shape = ifa_param['k_shape_list'][0]
    print(f"{k_shape=}")
    k_tensor = ifa_param['k_tensor_list'][0]
    print(f"k_tensor1111：{k_tensor.shape}")
    k_dtype = ifa_param['k_dtype']
    k_rope_shape = ifa_param['k_rope_shape']
    k_rope_tensor = ifa_param['k_rope_tensor']
    out_shape = ifa_param['out_shape']
    act_seq_list = ifa_param['actualSeqLengths_raw']
    inputLayout = ifa_param['inputLayout']
    numHeads = ifa_param['numHeads']
    numKeyValueHeads = ifa_param['numKeyValueHeads']
    sparse_mode = ifa_param['sparse_mode']
    causal = True if sparse_mode == 3 else False
    q_actualseqlengths = ifa_param['actualSeqLengths_q_raw']

    # act_seq预处理
    if len(act_seq_list) == 1 and q_shape[0] != 1:
        actualSeqLengths_item = act_seq_list[0]
        for b_count in range(q_shape[0] - 1):
            act_seq_list.append(actualSeqLengths_item)
    if inputLayout not in ["TND", "TND_NTD"]:
        act_seq_list = act_seq_list[:q_shape[0]]
    print(f"act_seq_list:{act_seq_list}")
    # 输入shape转换->bsnd
    if inputLayout in ["TND", "TND_NTD"]:
        q_tensor = _tnd_to_bsnd_for_gpu(q_tensor, q_shape, q_actualseqlengths)
    else:
        q_tensor = _bsh_bnsd_to_bsnd_for_gpu(q_tensor, inputLayout, q_shape, numHeads)
    b = q_tensor.shape[0]
    s_q = q_tensor.shape[1]
    h_q = q_tensor.shape[2]
    dv = q_tensor.shape[3]
    if inputLayout in ["TND", "TND_NTD"]:
        q_rope_tensor = _tnd_to_bsnd_for_gpu(q_rope_tensor, q_rope_shape, q_actualseqlengths)
    else:
        q_rope_tensor = _bsh_bnsd_to_bsnd_for_gpu(q_rope_tensor, inputLayout, q_rope_shape, numHeads)
    d = q_rope_tensor.shape[3] + dv
    k_tensor = _bsh_bnsd_to_bsnd_for_gpu(k_tensor, inputLayout, k_shape, numKeyValueHeads)
    k_s = max(act_seq_list)

    h_kv = k_tensor.shape[2]
    print(f"{k_tensor.shape=}")
    k_rope_tensor = _bsh_bnsd_to_bsnd_for_gpu(k_rope_tensor, inputLayout, k_rope_shape, numKeyValueHeads)

    # 拼接q/k rope
    q_new_tensor, q_new_shape = concat_tensor_gpu(q_tensor, q_tensor.shape,
                                                  q_rope_tensor,
                                                  q_rope_tensor.shape, numHeads, ifa_param['tnd_flag'])
    if q_new_tensor is None:
        print("[ERROR]q tensor的deepseek预处理异常，输出空tensor！")
        return torch.zeros(out_shape)

    k_new_tensor, k_new_shape = concat_tensor_gpu(k_tensor, k_tensor.shape,
                                                  k_rope_tensor,
                                                  k_rope_tensor.shape, numKeyValueHeads,
                                                  ifa_param['tnd_flag'])
    if k_new_tensor is None:
        print("[ERROR]k tensor的deepseek预处理异常，输出空tensor！")
        return torch.zeros(out_shape)

    if ifa_param["k_antiquant_flag"] and ifa_param["v_antiquant_flag"]:
        k_antiquant_scale = ifa_param['k_antiquantscale_tensor_raw']
        k_antiquant_offset = ifa_param['k_antiquantoffset_tensor_raw']
        k_rope_antiquantscale = ifa_param['k_rope_antiquantscale_tensor_raw']
        k_rope_antiquantoffset = ifa_param['k_rope_antiquantoffset_tensor_raw']
        v_antiquant_scale = ifa_param['v_antiquantscale_tensor_raw']
        v_antiquant_offset = ifa_param['v_antiquantoffset_tensor_raw']
        k_antiquant_scale_new, k_antiquant_scale_new_shape = concat_tensor_gpu(torch.tensor(k_antiquant_scale),
                                                                               k_antiquant_scale.shape,
                                                                               torch.tensor(k_rope_antiquantscale),
                                                                               k_rope_antiquantscale.shape,
                                                                               numKeyValueHeads,
                                                                               ifa_param['tnd_flag'])
        k_antiquant_offset_new, k_antiquant_offset_new_shape = concat_tensor_gpu(torch.tensor(k_antiquant_offset),
                                                                                 k_antiquant_offset.shape,
                                                                                 torch.tensor(k_rope_antiquantoffset),
                                                                                 k_rope_antiquantoffset.shape,
                                                                                 numKeyValueHeads,
                                                                                 ifa_param['tnd_flag'])
        k_new_tensor, v_new_tensor = kv_split_antiquant_tensor(k_new_tensor, k_tensor, k_antiquant_scale_new.cuda(),
                                                               k_antiquant_offset_new.cuda(),
                                                               v_antiquant_scale.cuda(),
                                                               torch.tensor(v_antiquant_offset).cuda(),
                                                               ifa_param['k_antiquant_mode'],
                                                               ifa_param['v_antiquant_mode'], ifa_param['k_dtype'],
                                                               ifa_param['q_dtype'])
        k_new_tensor = torch.tensor(k_new_tensor).cuda()

    # pa场景生成blocktable
    block_size = 64
    cache_seqlens = torch.tensor(act_seq_list, dtype=torch.int32).cuda()
    max_seqlen = cache_seqlens.max().item()
    max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256
    padding_length = max_seqlen_pad - max_seqlen
    # 裁剪ks
    k_new_tensor = k_new_tensor[:, :k_s, :, :]
    print(f"k_cache_before_padding:{k_new_tensor.shape}")
    k_new_tensor = F.pad(k_new_tensor, (0, 0, 0, 0, 0, padding_length), "constant", 0)
    print(f"k_cache_after_padding:{k_new_tensor.shape}")
    block_table = torch.arange(
        b * max_seqlen_pad // block_size, dtype=torch.int32
    ).view(b, max_seqlen_pad // block_size)
    block_table = block_table.cuda()
    k_new_tensor = k_new_tensor.transpose(1, 2)
    blocked_k = k_new_tensor.reshape(block_table.numel(), block_size, h_kv, d)
    for i in range(b):
        if blocked_k.dtype == torch.int8:
            blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = (0)
        else:
            blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = (
                float("nan")
            )
    blocked_k = blocked_k.cuda()

    # flash_mla_with_kvcache接口调用
    print(f"{b=}, {s_q=}, {h_q=}, {h_kv=}, {d=}, {dv=}, {causal=}")
    print(f"block_table:{block_table.shape} blocked_k:{blocked_k.shape}")
    q = q_new_tensor
    tile_scheduler_metadata, num_splits = get_mla_metadata(
        cache_seqlens, s_q * h_q // h_kv, h_kv
    )

    def flash_mla():
        return flash_mla_with_kvcache(
            q,
            blocked_k,
            block_table,
            cache_seqlens,
            dv,
            tile_scheduler_metadata,
            num_splits,
            causal=causal,
        )

    out_flash, lse_flash = flash_mla()

    # 输出转换
    gpu_out = out_flash.cpu()
    if inputLayout == "BSH":
        gpu_out = gpu_out.reshape(out_shape)
    elif inputLayout == "BNSD":
        gpu_out = gpu_out.permute([0, 2, 1, 3])
    elif inputLayout in ["BNSD_NBSD", "BSND_NBSD", "BSH_NBSD"]:
        gpu_out = gpu_out.permute([0, 2, 1, 3]).permute([1, 0, 2, 3])
    elif inputLayout in ["TND", "TND_NTD"]:
        diff_list = [q_actualseqlengths[0]]
        for idx in range(len(q_actualseqlengths) - 1):
            diff_list.append(q_actualseqlengths[idx + 1] - q_actualseqlengths[idx])
        new_gpu_out = []
        for idx in range(len(diff_list)):
            if diff_list[idx] != 0:
                sub_gpu_out = gpu_out[idx][:diff_list[idx]]
                new_gpu_out.append(sub_gpu_out)
        gpu_out_result = torch.cat(new_gpu_out, dim=0).to(gpu_out.device)
        if inputLayout == "TND_NTD":
            gpu_out_result = gpu_out_result.permute([1, 0, 2])
        return gpu_out_result
    else:
        pass

    return gpu_out


def aclnn_op_func_ifa_gpu(torch_tensor_list, params):
    print("def aclnn_op_func_ifa_gpu(torch_tensor_list, params): enter")
    print("gpu执行中...")

    # 获取参数
    print(params)
    ifa_param = get_param_fus(torch_tensor_list, params)

    if not ifa_param['normal_flag']:
        return torch.zeros(ifa_param['out_shape'])

    if params['enablegpu'] == 'small':
        print("if params['enablegpu'] == 'small': enter")
        out = aclnn_op_func_ifa_gpu_small(torch_tensor_list, params)
    elif ifa_param['deep_seek_flag'] and ifa_param['pa_flag']:
        out = ifa_gpu_flash_mla_with_kvcache(torch_tensor_list, params, ifa_param)
    else:
        out = ifa_gpu_flash_attn_with_kvcache(torch_tensor_list, params, ifa_param)
    return out


def convert_tensor_to_numpy(obj):
    """
    递归地将字典中的所有 torch.tensor 对象转换为 numpy 数组。
    """
    if isinstance(obj, dict):
        # 如果是字典，则递归处理每个值
        return {k: convert_tensor_to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # 如果是列表，则递归处理每个元素
        return [convert_tensor_to_numpy(item) for item in obj]
    elif isinstance(obj, torch.Tensor):
        # 如果是 torch.Tensor，则转换为 numpy 数组
        if obj.dtype == torch.bfloat16:
            obj = obj.to(torch.float32)
        return obj.cpu().numpy()
    else:
        # 其他类型的对象直接返回
        return obj


def aclnn_op_func_ifa_gpu_small(torch_tensor_list, params):
    print("gpu小算子执行中...")
    print(params)
    ifa_param = get_param_fus(torch_tensor_list, params)
    ifa_param = convert_tensor_to_numpy(ifa_param)
    if not ifa_param['normal_flag']:
        return torch.zeros(ifa_param['out_shape'])
    ifa_param["action_type"] = params["action_type"]

    # ===参数预处理===
    actualSeqLengths_q = ifa_param['actualSeqLengths_q_raw']
    actualSeqLengths = ifa_param['actualSeqLengths_raw']
    inputLayout = ifa_param['inputLayout']
    numHeads = ifa_param['numHeads']
    numKeyValueHeads = ifa_param['numKeyValueHeads']
    p_shape = ifa_param['p_shape']
    k_start_index = ifa_param['k_start_index']
    k_end_index = ifa_param['k_end_index']
    v_start_index = ifa_param['v_start_index']
    v_end_index = ifa_param['v_end_index']
    out_shape = ifa_param['out_shape']
    k_ori_shape_list = copy.deepcopy(ifa_param['k_shape_list'])
    v_ori_shape_list = copy.deepcopy(ifa_param['v_shape_list'])
    k_dtype = ifa_param['k_dtype']
    v_dtype = ifa_param['v_dtype']
    m_tensor = ifa_param['m_tensor']
    m_shape = ifa_param['m_shape']
    sparse_mode = ifa_param['sparse_mode']
    if ifa_param['tnd_flag'] and ifa_param['pa_flag']:
        kv_layout = "BNSD"
    else:
        kv_layout = inputLayout
    ifa_param['kv_layout'] = kv_layout

    # >> actualSeqLengths预处理：actualSeqLengths为单值场景，如果长度为1且b不为1，则将actualSeqLengths扩展为b个单值的列表
    # >> 仅非tnd场景，或tnd+pa场景，需要处理
    if not ifa_param['tnd_flag'] or (ifa_param['tnd_flag'] and ifa_param['pa_flag']):
        # actq
        if len(actualSeqLengths_q) == 1 and len(actualSeqLengths_q) != ifa_param['b']:
            actualSeqLengthsq_item = actualSeqLengths_q[0]
            for b_count in range(ifa_param['b'] - 1):
                actualSeqLengths_q.append(actualSeqLengthsq_item)
        if len(actualSeqLengths_q) > ifa_param['b']:
            actualSeqLengths_q = actualSeqLengths_q[:ifa_param['b']]
        # actkv
        if len(actualSeqLengths) == 1 and len(actualSeqLengths) != ifa_param['b']:
            actualSeqLengths_item = actualSeqLengths[0]
            for b_count in range(ifa_param['b'] - 1):
                actualSeqLengths.append(actualSeqLengths_item)
        # >> actualSeqLengths预处理：actualSeqLengths长度超过b
        if len(actualSeqLengths) > ifa_param['b']:
            actualSeqLengths = actualSeqLengths[:ifa_param['b']]
    if ifa_param['tnd_flag']:
        # 将tnd格式下的act seq转成普通的act seq
        actualSeqLengths_q = trans_tnd_actseq(actualSeqLengths_q)
        if not ifa_param['pa_flag']:
            actualSeqLengths = trans_tnd_actseq(actualSeqLengths)
        print(f"[INFO]trans_tnd_actseq end.")

    ifa_param['actualSeqLengths'] = actualSeqLengths
    ifa_param['actualSeqLengths_q'] = actualSeqLengths_q

    # deepseek 预处理： 1.将Q与QROPE拼接 2.将K与KROPE拼接
    if ifa_param['deep_seek_flag']:
        ifa_param = deepseek_preprocessing(ifa_param)
    # >> q 预处理：将q转换为bnsd
    q_bnsd_tensor, q_bnsd_shape = _n_trans_shape_to_bnsd(ifa_param['q_tensor'], ifa_param['q_shape'], inputLayout,
                                                         numHeads, actualSeqLengths_q)
    ifa_param['q_bnsd_shape'] = q_bnsd_shape

    qs = q_bnsd_shape[2]
    ifa_param['q_tensor_bnsd'] = q_bnsd_tensor
    ifa_param['q_shape_bnsd'] = q_bnsd_shape
    ifa_param['q_d'] = q_bnsd_shape[3]

    # >> kv预处理：1、将kv list 转换为bnsd 2、GQA场景，将kvn扩展为qn
    k_tensor_list = []
    v_tensor_list = []
    k_shape_list = []
    v_shape_list = []
    k_tensor_bnsd_list = []
    v_tensor_bnsd_list = []
    k_shape_bnsd_list = []
    v_shape_bnsd_list = []
    ifa_param['ks_max'] = 0

    for i in range(0, ifa_param['k_num']):
        k_tensor, k_bnsd_shape = _n_trans_shape_to_bnsd(ifa_param['k_tensor_list'][i], ifa_param['k_shape_list'][i],
                                                        kv_layout,
                                                        numKeyValueHeads, actualSeqLengths)
        k_tensor_bnsd_list.append(k_tensor)
        k_shape_bnsd_list.append(k_bnsd_shape)
        # k int32预处理
        if ifa_param['k_dtype'] == "int32":
            k_tensor, k_bnsd_shape = trans_int32_2_int4_tensor_bnsd(k_tensor, k_bnsd_shape)

        if numKeyValueHeads != numHeads:
            if ifa_param['deep_seek_flag'] and ifa_param['in_quant_flag']:
                pass
            else:
                k_tensor, k_bnsd_shape = _t_broadcastKV_sigle(numHeads, numKeyValueHeads, k_tensor, k_dtype)
        if k_bnsd_shape[2] > ifa_param['ks_max']:
            ifa_param['ks_max'] = k_bnsd_shape[2]
        k_tensor_list.append(k_tensor)
        k_shape_list.append(k_bnsd_shape)

    x = 0
    for i in range(0, ifa_param['k_num']):
        v_tensor, v_bnsd_shape = _n_trans_shape_to_bnsd(ifa_param['v_tensor_list'][i], v_ori_shape_list[x], kv_layout,
                                                        numKeyValueHeads, actualSeqLengths)
        x = x + 1
        v_tensor_bnsd_list.append(v_tensor)
        v_shape_bnsd_list.append(v_bnsd_shape)
        # v int32预处理
        if ifa_param['v_dtype'] == "int32":
            v_tensor, v_bnsd_shape = trans_int32_2_int4_tensor_bnsd(v_tensor, v_bnsd_shape)

        if numKeyValueHeads != numHeads:
            if ifa_param['deep_seek_flag'] and ifa_param['in_quant_flag']:
                pass
            else:
                v_tensor, v_bnsd_shape = _t_broadcastKV_sigle(numHeads, numKeyValueHeads, v_tensor, v_dtype)
        v_tensor_list.append(v_tensor)
        v_shape_list.append(v_bnsd_shape)

    ifa_param['k_tensor_list_bnsd'] = k_tensor_list
    ifa_param['k_shape_list_bnsd'] = k_shape_list
    ifa_param['v_tensor_list_bnsd'] = v_tensor_list
    ifa_param['v_shape_list_bnsd'] = v_shape_list

    # 处理deepseek全量化：1.将qkv从FP16转成INT8 2.生成dequantscale1,dequantscale2
    if ifa_param['deep_seek_flag'] and ifa_param['in_quant_flag']:
        if ifa_param['action_type'] in ['bm_output_gold', 'bm_output']:
            print("if ifa_param['action_type'] in ['bm_output_gold', 'bm_output']: enter")
            print("ifa_param = deepseek_inquant_preprocessing_for_recovery(ifa_param, torch_tensor_list, params)")
            numpy_tensor_list = convert_tensor_to_numpy(torch_tensor_list)
            ifa_param = deepseek_inquant_preprocessing_for_recovery(ifa_param, numpy_tensor_list, params)
        else:
            ifa_param = deepseek_inquant_preprocessing(ifa_param, params)

    # >> left padding 异常场景处理：起点/终点异常，返回全0
    if ifa_param['padding_flag']:
        max_act_seq = max(actualSeqLengths)
        kv_s = ifa_param['ks_max']
        if kv_s - ifa_param['padding_size'] - max_act_seq < 0:
            print('[WARNING]:paddingsize 溢出，输出空tensor！')
            return torch.zeros(out_shape)

    # >> pse 处理
    if ifa_param['p_flag']:
        maya = get_slopes(p_shape[1])
        maya = maya.numpy()
        pse_shift = np.zeros(p_shape)
        for n in range(p_shape[1]):
            alibi_biases = np.zeros([1, p_shape[-1]])
            for x in range(0, p_shape[-1]):
                alibi_biases[0, x] = -1 * x
            pse_shift[:, n:n + 1, :, :] = alibi_biases * maya[n]
        pse_shift = pse_shift.astype(np.float32)

        # 覆写
        action_type = params["action_type"]
        if action_type in ["bm_output_gold", "bm_output"]:
            pass
        else:
            if params['dtype_input'][v_end_index + 1] == "fp16":
                pse_shift = pse_shift.astype(np.float16)
                pse_shift[pse_shift == -math.inf] = -65504
                ifa_param['p_tensor'] = pse_shift
                p_tensor = torch.tensor(pse_shift, dtype=torch.float16)
            elif params['dtype_input'][v_end_index + 1] == "bf16":
                pse_shift = pse_shift.astype(tf.bfloat16.as_numpy_dtype)
                pse_shift[pse_shift == -math.inf] = -65504
                ifa_param['p_tensor'] = pse_shift
                p_tensor = torch.tensor(pse_shift.astype(np.float32), dtype=torch.bfloat16)
            else:
                print('[ERROR]:wrong p_tensor dtype!')
                return torch.zeros(out_shape)
            tools.modify_alcnn_input_file(ids=3, origin_index=[3], type='tensor', mode='rewrite', tensors=p_tensor,
                                          params=params)

    # >> m预处理：1、将m扩展为BN1S  2、padding场景下，偏移部分设置为1 3、针对FP16格式，将tensor转成0/1
    m_dtype = tools.get_pt_dtype(params['dtype_input'][v_end_index + 2])
    m_tensor = torch.tensor(m_tensor, dtype=m_dtype)
    if sparse_mode == 3:
        batch = q_bnsd_shape[0]
        numheads = q_bnsd_shape[1]
        npu_m_shape_s = m_shape
        if sparse_mode == 0 or sparse_mode == 1:
            batch, numheads, ns1, ns2 = get_attention_mask_batch_num(m_shape,
                                                                     q_bnsd_shape)  # 获取输入attentionmask的batch 和numhead
            npu_m_shape_s = [ns1, ns2]
        kvs = k_shape_list[0][2]
        cpu_m_shape = [qs, kvs]  # cpu
        preTokens = ifa_param['pre_tokens']
        nextTokens = ifa_param['next_tokens']
        actualprefixKV = 0
        prefix_kvs = 0
        kvs_list = [kvs]
        cpu_m_tensor, npu_m_tensor, preTokens, nextTokens = _create_random_mask_by_spars(cpu_m_shape, npu_m_shape_s,
                                                                                         m_dtype, preTokens, nextTokens,
                                                                                         ifa_param[
                                                                                             'actualSeqLengths_q'],
                                                                                         actualSeqLengths,
                                                                                         actualprefixKV,
                                                                                         prefix_kvs, kvs_list, batch,
                                                                                         numheads, sparse_mode)

        npu_m_tensor = torch.from_numpy(npu_m_tensor).to(m_dtype)

        if ifa_param["action_type"] in ["bm_output", "bm_output_gold"]:
            pass
        else:
            tools.modify_alcnn_input_file(ids=4, origin_index=[4], type='tensor', mode='rewrite', tensors=npu_m_tensor,
                                          params=params)

        if sparse_mode == 0 or sparse_mode == 1:
            # m_bnsd_tensor = _np_broadcast_mask_n(cpu_m_tensor, npu_m_shape, cpu_m_shape, numHeads, q_bnsd_shape[0])
            pass
        else:
            cpu_m_tensor = np.array(cpu_m_tensor)
            cpu_m_tensor = torch.from_numpy(cpu_m_tensor)
            ifa_param['m_tensor'] = _t_broadcast_mask_n(cpu_m_tensor, cpu_m_tensor.shape, numHeads, q_bnsd_shape[0])
    else:
        # >> m预处理：1、将m扩展为BN1S
        # print(f"m_tensor_raw===={m_tensor}")
        if ifa_param['m_flag']:
            m_rewrite_flag = False
            if ifa_param['padding_flag']:
                m_tensor = cut_padding_size(m_tensor, actualSeqLengths, ifa_param['padding_size'])
                m_rewrite_flag = True
            if ifa_param['m_dtype'] == "float16":
                m_tensor = torch.where(m_tensor < 0.8, torch.tensor(0, dtype=torch.float16),
                                       torch.tensor(1, dtype=torch.float16))
                m_rewrite_flag = True
            if m_rewrite_flag:
                print(f"[INFO]rewrite attenmask")
                tools.modify_alcnn_input_file(ids=4, origin_index=[4], type='tensor', mode='rewrite',
                                              tensors=m_tensor, params=params, data_range=[0, 2])
            ifa_param['m_tensor'] = _t_broadcast_mask_n(m_tensor, m_shape, numHeads, m_shape[0])

    # >> 后量化参数处理：1、给dtype赋值 2、如果是perchannel模式，要将后量化参数统一转换成1n1d格式
    if ifa_param['out_quant_flag']:
        ifa_param['quantScale2_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 6])
        ifa_param['quantOffset2_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 7])
    if ifa_param['out_quant_pc_flag']:
        ifa_param['quantScale2'] = trans_to_1n1d(ifa_param['quantScale2'], ifa_param['quantScale2_shape'],
                                                 ifa_param['q_shape_bnsd'][3])
        if ifa_param['quantScale2'] is None:
            return torch.zeros(out_shape)
        ifa_param['quantOffset2'] = trans_to_1n1d(ifa_param['quantOffset2'], ifa_param['quantOffset2_shape'],
                                                  ifa_param['q_shape_bnsd'][3])
        if ifa_param['quantOffset2'] is None:
            return torch.zeros(out_shape)
        ifa_param['quantScale2_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 6])
        ifa_param['quantOffset2_dtype'] = trans_input_dtype(params['dtype_input'][v_end_index + 7])

        # >> 处理pageatten场景（以下处理不涉及cpu、gpu等真值计算，仅为npu生成输入）：
        # 1、生成随机的block_table，并覆写原有bin文件
        # 2、将kv shape 统一转换成bsh
        # 3、生成kv cache
        # 4、将kv cache dump成新的bin文件，供aclnn接口调用
    if ifa_param['pa_flag'] and "output" not in params["action_type"]:
        # 生成blocktable
        blockSize = ifa_param['blocksize']
        blockTableShape = ifa_param['bt_shape']
        blockNum = ifa_param['k_cache_shape'][0]
        if 0 in ifa_param['bt_shape']:
            print('[WARNING]:block_table为空场景，输出空tensor！')
            return torch.zeros(out_shape)
        blockNumPerBlock = []
        block_num_min = 0
        for actual_seq in actualSeqLengths:
            blockNumPerBlock.append(math.ceil(actual_seq / blockSize))
            block_num_min += math.ceil(actual_seq / blockSize)
        if block_num_min > blockNum:
            print(f"[ERROR]Wrong input k_cache_shape: get blockNum = {blockNum}, but expect blockNum > {block_num_min}")
            exit(1)
        block_idx_list = np.arange(0, blockNum, 1)
        block_idx_list = np.random.permutation(block_idx_list).astype(np.int32)
        block_idx = 0
        block_table = [-1] * blockTableShape[1]
        block_table = np.tile(block_table, (blockTableShape[0], 1)).astype(np.int32)
        block_table_batch_idx = 0
        for idx in blockNumPerBlock:
            for j in range(idx):
                block_table[block_table_batch_idx][j] = (block_idx_list[block_idx])
                block_idx += 1
            block_table_batch_idx += 1
        ifa_param['block_table'] = block_table
        # 将block_table 覆盖原有的input
        # print("===>end gen block_table", block_table)
        tools.modify_alcnn_input_file(ids=12, origin_index=[12], type='tensor', mode='rewrite',
                                      tensors=torch.tensor(block_table, dtype=torch.int32),
                                      params=params)
        k_np_dtype = tools.get_np_dtype(params['dtype_input'][1])
        v_np_dtype = tools.get_np_dtype(params['dtype_input'][2])
        ifa_param['k_np_dtype'] = k_np_dtype
        ifa_param['v_np_dtype'] = v_np_dtype
        print(f"[PageAtten]Input Kdtype:{params['dtype_input'][1]} Vdtype:{params['dtype_input'][2]}")
        k_cache = np.zeros(ifa_param['k_cache_shape']).astype(k_np_dtype)
        v_cache = np.zeros(ifa_param['v_cache_shape']).astype(v_np_dtype)
        if not ifa_param['deep_seek_flag']:
            print('[INFO]:PA + no Deepseek！')
            # gen kv cache
            if len(ifa_param['k_cache_shape']) == 3:
                # trans kv to bsh(此处使用的tensor, 没有经过n的扩展)
                k_tensor_bsh_raw = trans_bnsd_to_bsh(k_tensor_bnsd_list[0], k_shape_bnsd_list[0])
                v_tensor_bsh_raw = trans_bnsd_to_bsh(v_tensor_bnsd_list[0], v_shape_bnsd_list[0])

                # kv paddIng
                kvh = numKeyValueHeads * q_bnsd_shape[3]
                if ifa_param['k_dtype'] == "int32":
                    kvh = int(kvh / 8)
                k_tensor_bsh = np.zeros((q_bnsd_shape[0], blockTableShape[1] * blockSize, kvh)).astype(k_np_dtype)
                v_tensor_bsh = np.zeros((q_bnsd_shape[0], blockTableShape[1] * blockSize, kvh)).astype(v_np_dtype)
                k_tensor_bsh[:, :k_tensor_bsh_raw.shape[1], :] = k_tensor_bsh_raw[:, :, :]
                v_tensor_bsh[:, :v_tensor_bsh_raw.shape[1], :] = v_tensor_bsh_raw[:, :, :]
                for b in range(q_bnsd_shape[0]):
                    for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                        block_offset = block_i * blockSize
                        if kv_cache_blk_id == -1:
                            continue
                        else:
                            k_cache[kv_cache_blk_id, 0:blockSize, :] = k_tensor_bsh[b,
                                                                       block_offset:(block_offset + blockSize), :]
                            v_cache[kv_cache_blk_id, 0:blockSize, :] = v_tensor_bsh[b,
                                                                       block_offset:(block_offset + blockSize), :]
            elif len(ifa_param['k_cache_shape']) == 4:
                kvn = k_tensor_bnsd_list[0].shape[1]
                kvs = k_tensor_bnsd_list[0].shape[2]
                kvd = k_tensor_bnsd_list[0].shape[3]
                k_tensor_bnsd = np.zeros((q_bnsd_shape[0], kvn, blockTableShape[1] * blockSize, kvd)).astype(k_np_dtype)
                v_tensor_bnsd = np.zeros((q_bnsd_shape[0], kvn, blockTableShape[1] * blockSize, kvd)).astype(v_np_dtype)
                k_tensor_bnsd[:, :, :kvs, :] = k_tensor_bnsd_list[0][:, :, :, :]
                v_tensor_bnsd[:, :, :kvs, :] = v_tensor_bnsd_list[0][:, :, :, :]

                for b in range(q_bnsd_shape[0]):
                    for block_i, kv_cache_blk_id in enumerate(block_table[b]):
                        block_offset = block_i * blockSize
                        if kv_cache_blk_id == -1:
                            continue
                        else:
                            for n in range(q_bnsd_shape[1]):
                                k_cache[kv_cache_blk_id, n:n + 1, 0:blockSize, :] = k_tensor_bnsd[b, n:n + 1,
                                                                                    block_offset:(
                                                                                            block_offset + blockSize),
                                                                                    :]
                                v_cache[kv_cache_blk_id, n:n + 1, 0:blockSize, :] = v_tensor_bnsd[b, n:n + 1,
                                                                                    block_offset:(
                                                                                            block_offset + blockSize),
                                                                                    :]
            else:
                print(f"[ERROR]Wrong input kv_cache_shape: {ifa_param['k_cache_shape']}")
                return torch.zeros(out_shape)
            # 将kv cache 生成新的bin文件

            k_cache_index = 21
            v_cache_index = 22
            tools.modify_alcnn_input_file(ids=k_cache_index, origin_index=[k_cache_index], type='tensor_list',
                                          mode='rewrite',
                                          tensors=[k_cache],
                                          params=params,
                                          data_dtype=params['dtype_input'][1])
            tools.modify_alcnn_input_file(ids=v_cache_index, origin_index=[v_cache_index], type='tensor_list',
                                          mode='rewrite',
                                          tensors=[v_cache],
                                          params=params,
                                          data_dtype=params['dtype_input'][2])
        else:
            print('[INFO]:PA + Deepseek！')
            if ifa_param['in_quant_flag']:
                deepseek_inquant_ds_pa_preprocessing(ifa_param, params)
            else:
                deepseek_ds_pa_preprocessing(ifa_param, params)

    # >> kv 伪量化预处理
    if ifa_param['kv_quant_flag']:
        antiquantscale_shape_raw = ifa_param['antiquantscale_shape_raw']
        antiquantscale_tensor_raw = ifa_param['antiquantscale_tensor_raw']
        antiquantoffset_shape_raw = ifa_param['antiquantoffset_shape_raw']
        antiquantoffset_tensor_raw = ifa_param['antiquantoffset_tensor_raw']

        # >> [TEMP]pertoken_group：1、将kv量化参数统一转换成2,B,N,S,D/32 2、GQA场景下，扩展kv量化参数
        if ifa_param['kv_quant_ptog_flag']:
            antiquantscale_tensor = _trans_antiparam_to_2bnsg(antiquantscale_shape_raw, antiquantscale_tensor_raw,
                                                              inputLayout, numKeyValueHeads)
            antiquantoffset_tensor = _trans_antiparam_to_2bnsg(antiquantoffset_shape_raw, antiquantoffset_tensor_raw,
                                                               inputLayout, numKeyValueHeads)
            antiquantscale_tensor = broadcast_kv_dequant_tensor_fp8e8m0(antiquantscale_tensor, numKeyValueHeads,
                                                                        numHeads)
            antiquantoffset_tensor = broadcast_kv_dequant_tensor_fp8e8m0(antiquantoffset_tensor, numKeyValueHeads,
                                                                         numHeads)

            ifa_param['antiquantScale'] = antiquantscale_tensor
            ifa_param['antiquantOffset'] = antiquantoffset_tensor
        # pertensor/perchannel ->1、将kv量化参数统一转换成2n1d 2、GQA场景下，扩展kv量化参数
        elif not ifa_param['kv_quant_pto_flag']:
            # 将kv量化参数转换为2n1d(匹配bnsd)
            antiquantscale_tensor = _trans_antiparam_to_2n1d(antiquantscale_shape_raw, antiquantscale_tensor_raw,
                                                             inputLayout, numKeyValueHeads, ifa_param['q_d'])
            antiquantoffset_tensor = _trans_antiparam_to_2n1d(antiquantoffset_shape_raw, antiquantoffset_tensor_raw,
                                                              inputLayout, numKeyValueHeads, ifa_param['q_d'])
            # GQA场景，扩展kv反量化参数
            q_dtype_np = tools.get_np_dtype(params['dtype_input'][0])
            antiquantscale_tensor = broadcast_kv_dequant_tensor(antiquantscale_tensor, numKeyValueHeads, numHeads,
                                                                q_dtype_np)
            antiquantoffset_tensor = broadcast_kv_dequant_tensor(antiquantoffset_tensor, numKeyValueHeads, numHeads,
                                                                 q_dtype_np)

            ifa_param['antiquantScale'] = antiquantscale_tensor
            ifa_param['antiquantOffset'] = antiquantoffset_tensor
        else:
            ifa_param['antiquantScale'] = antiquantscale_tensor_raw
            ifa_param['antiquantOffset'] = antiquantoffset_tensor_raw
    # 判断kv分离的伪量化
    if ifa_param["k_antiquant_flag"] and not ifa_param["v_antiquant_flag"]:
        print("[ERROR] k_antiquant exit, but v_antiquant not exit")
        exit(-1)
    elif not ifa_param["k_antiquant_flag"] and ifa_param["v_antiquant_flag"]:
        print("[ERROR] v_antiquant exit, but k_antiquant not exit")
        exit(-1)
    else:
        if ifa_param["k_antiquant_flag"]:
            k_antiquantscale_shape_raw = ifa_param['k_antiquantscale_shape_raw']
            k_antiquantscale_tensor_raw = ifa_param['k_antiquantscale_tensor_raw']
            k_antiquantoffset_shape_raw = ifa_param['k_antiquantoffset_shape_raw']
            k_antiquantoffset_tensor_raw = ifa_param['k_antiquantoffset_tensor_raw']
            print(f"k_antiquant_pto_flag: {ifa_param['k_antiquant_pto_flag']}")
            print(f"k_antiquant_ptoh_flag: {ifa_param['k_antiquant_ptoh_flag']}")
            print(f"k_antiquant_ptopa_flag: {ifa_param['k_antiquant_ptopa_flag']}")
            print(f"k_antiquant_ptohpa_flag: {ifa_param['k_antiquant_ptohpa_flag']}")
            # pertoken_head
            if ifa_param['k_antiquant_ptoh_flag']:
                # GQA场景，扩展kv反量化参数
                k_antiquantscale_tensor = broadcast_kv_split_dequant_tensor_ptoh(k_antiquantscale_tensor_raw,
                                                                                 numKeyValueHeads, numHeads)
                k_antiquantoffset_tensor = broadcast_kv_split_dequant_tensor_ptoh(k_antiquantoffset_tensor_raw,
                                                                                  numKeyValueHeads, numHeads)
                ifa_param['k_antiquantScale'] = k_antiquantscale_tensor
                ifa_param['k_antiquantOffset'] = k_antiquantoffset_tensor
            # pertensor/perchannel ->1、将kv量化参数统一转换成1n1d 2、GQA场景下，扩展kv量化参数
            elif ifa_param['k_antiquant_pc_flag'] or ifa_param['k_antiquant_pth_flag']:
                k_antiquantscale_tensor = _trans_antiparam_to_1n1d(k_antiquantscale_shape_raw,
                                                                   k_antiquantscale_tensor_raw,
                                                                   inputLayout, numKeyValueHeads, ifa_param['q_d'],
                                                                   ifa_param['k_antiquant_mode'])
                k_antiquantoffset_tensor = _trans_antiparam_to_1n1d(k_antiquantoffset_shape_raw,
                                                                    k_antiquantoffset_tensor_raw,
                                                                    inputLayout, numKeyValueHeads, ifa_param['q_d'],
                                                                    ifa_param['k_antiquant_mode'])
                # GQA场景，扩展kv反量化参数
                q_dtype_np = tools.get_np_dtype(params['dtype_input'][0])
                k_antiquantscale_tensor = broadcast_kv_split_dequant_tensor(k_antiquantscale_tensor, numKeyValueHeads,
                                                                            numHeads, q_dtype_np)
                k_antiquantoffset_tensor = broadcast_kv_split_dequant_tensor(k_antiquantoffset_tensor, numKeyValueHeads,
                                                                             numHeads, q_dtype_np)
                ifa_param['k_antiquantScale'] = k_antiquantscale_tensor
                ifa_param['k_antiquantOffset'] = k_antiquantoffset_tensor
            # pertoken_pa -> 1、将bb转成1bs
            elif ifa_param['k_antiquant_ptopa_flag']:
                if ifa_param['pa_flag']:
                    ifa_param['k_antiquantScale'] = trans_bb_2_1bs(k_antiquantscale_tensor_raw,
                                                                   ifa_param['block_table'], ifa_param['b'])
                    ifa_param['k_antiquantOffset'] = trans_bb_2_1bs(k_antiquantoffset_tensor_raw,
                                                                    ifa_param['block_table'], ifa_param['b'])
                else:
                    print("[ERROR]Got antiquant mode == 4,but not PA!")
                    exit(1)
            # pertoken_head_pa -> 1、将bb转成1bns
            elif ifa_param['k_antiquant_ptohpa_flag']:
                if ifa_param['pa_flag']:
                    k_antiquantscale_tensor = trans_bb_2_1bns(k_antiquantscale_tensor_raw, ifa_param['block_table'],
                                                              ifa_param['b'])
                    k_antiquantoffset_tensor = trans_bb_2_1bns(k_antiquantoffset_tensor_raw, ifa_param['block_table'],
                                                               ifa_param['b'])
                    k_antiquantscale_tensor = broadcast_kv_split_dequant_tensor_ptoh(k_antiquantscale_tensor,
                                                                                     numKeyValueHeads, numHeads)
                    k_antiquantoffset_tensor = broadcast_kv_split_dequant_tensor_ptoh(k_antiquantoffset_tensor,
                                                                                      numKeyValueHeads, numHeads)
                    ifa_param['k_antiquantScale'] = k_antiquantscale_tensor
                    ifa_param['k_antiquantOffset'] = k_antiquantoffset_tensor
                else:
                    print("[ERROR]Got antiquant mode == 5,but not PA!")
                    exit(1)

            elif ifa_param['k_antiquant_ptog_flag']:
                k_antiquantscale_tensor = broadcast_kv_split_antiquant_tensor_fp8e8m0(k_antiquantscale_tensor_raw,
                                                                                      numKeyValueHeads, numHeads)
                k_antiquantoffset_tensor = broadcast_kv_split_antiquant_tensor_fp8e8m0(k_antiquantoffset_tensor_raw,
                                                                                       numKeyValueHeads, numHeads)

                ifa_param['k_antiquantScale'] = k_antiquantscale_tensor
                ifa_param['k_antiquantOffset'] = k_antiquantoffset_tensor

            # pertoken -> 如果是BS,转成1BS
            else:
                ifa_param['k_antiquantScale'] = _trans_antiparam_to_1bs(k_antiquantscale_tensor_raw)
                ifa_param['k_antiquantOffset'] = _trans_antiparam_to_1bs(k_antiquantoffset_tensor_raw)

        if ifa_param["v_antiquant_flag"]:
            v_antiquantscale_shape_raw = ifa_param['v_antiquantscale_shape_raw']
            v_antiquantscale_tensor_raw = ifa_param['v_antiquantscale_tensor_raw']
            v_antiquantoffset_shape_raw = ifa_param['v_antiquantoffset_shape_raw']
            v_antiquantoffset_tensor_raw = ifa_param['v_antiquantoffset_tensor_raw']
            # pertoken_head
            if ifa_param['v_antiquant_ptoh_flag']:
                # GQA场景，扩展kv反量化参数
                v_antiquantscale_tensor = broadcast_kv_split_dequant_tensor_ptoh(v_antiquantscale_tensor_raw,
                                                                                 numKeyValueHeads, numHeads)
                v_antiquantoffset_tensor = broadcast_kv_split_dequant_tensor_ptoh(v_antiquantoffset_tensor_raw,
                                                                                  numKeyValueHeads, numHeads)
                ifa_param['v_antiquantScale'] = v_antiquantscale_tensor
                ifa_param['v_antiquantOffset'] = v_antiquantoffset_tensor
            # pertensor/perchannel ->1、将kv量化参数统一转换成1n1d 2、GQA场景下，扩展kv量化参数
            elif ifa_param['v_antiquant_pc_flag'] or ifa_param['v_antiquant_pth_flag']:
                v_antiquantscale_tensor = _trans_antiparam_to_1n1d(v_antiquantscale_shape_raw,
                                                                   v_antiquantscale_tensor_raw,
                                                                   inputLayout, numKeyValueHeads, ifa_param['q_d'],
                                                                   ifa_param['v_antiquant_mode'])
                v_antiquantoffset_tensor = _trans_antiparam_to_1n1d(v_antiquantoffset_shape_raw,
                                                                    v_antiquantoffset_tensor_raw,
                                                                    inputLayout, numKeyValueHeads, ifa_param['q_d'],
                                                                    ifa_param['v_antiquant_mode'])
                # GQA场景，扩展kv反量化参数
                q_dtype_np = tools.get_np_dtype(params['dtype_input'][0])
                v_antiquantscale_tensor = broadcast_kv_split_dequant_tensor(v_antiquantscale_tensor, numKeyValueHeads,
                                                                            numHeads, q_dtype_np)
                v_antiquantoffset_tensor = broadcast_kv_split_dequant_tensor(v_antiquantoffset_tensor, numKeyValueHeads,
                                                                             numHeads, q_dtype_np)
                ifa_param['v_antiquantScale'] = v_antiquantscale_tensor
                ifa_param['v_antiquantOffset'] = v_antiquantoffset_tensor
            # pertoken_pa -> 1、将bb转成1bs
            elif ifa_param['v_antiquant_ptopa_flag']:
                if ifa_param['pa_flag']:
                    ifa_param['v_antiquantScale'] = trans_bb_2_1bs(v_antiquantscale_tensor_raw,
                                                                   ifa_param['block_table'], ifa_param['b'])
                    ifa_param['v_antiquantOffset'] = trans_bb_2_1bs(v_antiquantoffset_tensor_raw,
                                                                    ifa_param['block_table'], ifa_param['b'])
                else:
                    print("[ERROR]Got antiquant mode == 4,but not PA!")
                    exit(1)

            # per_token_head_pa -> 1、将bnb转成1bns
            elif ifa_param['v_antiquant_ptohpa_flag']:
                print(v_antiquantscale_tensor_raw.shape)
                if ifa_param['pa_flag']:
                    v_antiquantscale_tensor = trans_bb_2_1bns(v_antiquantscale_tensor_raw, ifa_param['block_table'],
                                                              ifa_param['b'])
                    v_antiquantoffset_tensor = trans_bb_2_1bns(v_antiquantoffset_tensor_raw, ifa_param['block_table'],
                                                               ifa_param['b'])
                    v_antiquantscale_tensor = broadcast_kv_split_dequant_tensor_ptoh(v_antiquantscale_tensor,
                                                                                     numKeyValueHeads, numHeads)
                    v_antiquantoffset_tensor = broadcast_kv_split_dequant_tensor_ptoh(v_antiquantoffset_tensor,
                                                                                      numKeyValueHeads, numHeads)
                    ifa_param['v_antiquantScale'] = v_antiquantscale_tensor
                    ifa_param['v_antiquantOffset'] = v_antiquantoffset_tensor
                else:
                    print("[ERROR]Got antiquant mode == 5,but not PA!")
                    exit(1)

            # per_token_group -> 1、将bb转成1bs
            elif ifa_param['v_antiquant_ptog_flag']:
                v_antiquantscale_tensor = broadcast_kv_split_antiquant_tensor_fp8e8m0(v_antiquantscale_tensor_raw,
                                                                                      numKeyValueHeads,
                                                                                      numHeads)
                v_antiquantoffset_tensor = broadcast_kv_split_antiquant_tensor_fp8e8m0(v_antiquantoffset_tensor_raw,
                                                                                       numKeyValueHeads,
                                                                                       numHeads)

                ifa_param['v_antiquantScale'] = v_antiquantscale_tensor
                ifa_param['v_antiquantOffset'] = v_antiquantoffset_tensor
            # pertoken -> 如果是BS,转成1BS
            else:
                ifa_param['v_antiquantScale'] = _trans_antiparam_to_1bs(v_antiquantscale_tensor_raw)
                ifa_param['v_antiquantOffset'] = _trans_antiparam_to_1bs(v_antiquantoffset_tensor_raw)

    # >> 输入量化预处理：获取切分使用的sinner
    ifa_param['sinner'] = 0
    if ifa_param['in_quant_flag'] and not ifa_param['deep_seek_flag']:
        sinner = get_sinner(ifa_param)
        ifa_param['sinner'] = sinner

    # psefix 预处理：1.转成1nsd 2.GQA场景扩展N; 3.按act_prefix裁剪 4.获取perfix_act_lens
    if ifa_param['prefix_flag']:
        k_prefix_tensor = ifa_param['k_prefix_tensor']
        k_prefix_shape = ifa_param['k_prefix_shape']
        v_prefix_tensor = ifa_param['v_prefix_tensor']
        v_prefix_shape = ifa_param['v_prefix_shape']
        k_prefix_tensor, k_prefix_shape = _n_trans_shape_to_bnsd(k_prefix_tensor, k_prefix_shape, inputLayout,
                                                                 numKeyValueHeads)
        v_prefix_tensor, v_prefix_shape = _n_trans_shape_to_bnsd(v_prefix_tensor, v_prefix_shape, inputLayout,
                                                                 numKeyValueHeads)

        # kv prefix int32预处理
        if ifa_param['k_prefix_dtype'] == "int32":
            k_prefix_tensor, k_prefix_shape = trans_int32_2_int4_tensor_bnsd(k_prefix_tensor, k_prefix_shape)
        if ifa_param['v_prefix_dtype'] == "int32":
            v_prefix_tensor, v_prefix_shape = trans_int32_2_int4_tensor_bnsd(v_prefix_tensor, v_prefix_shape)

        if numKeyValueHeads != numHeads:
            k_prefix_tensor, k_prefix_shape = _t_broadcastKV_sigle(numHeads, numKeyValueHeads, k_prefix_tensor,
                                                                   ifa_param['k_prefix_dtype'])
            v_prefix_tensor, v_prefix_shape = _t_broadcastKV_sigle(numHeads, numKeyValueHeads, v_prefix_tensor,
                                                                   ifa_param['v_prefix_dtype'])

        if ifa_param['prefix_act_flag']:
            k_prefix_tensor = k_prefix_tensor[:, :, :ifa_param['prefix_act_lens'], :]
            v_prefix_tensor = v_prefix_tensor[:, :, :ifa_param['prefix_act_lens'], :]
        else:
            ifa_param['prefix_act_lens'] = k_prefix_shape[2]

        ifa_param['k_prefix_tensor'] = k_prefix_tensor
        ifa_param['v_prefix_tensor'] = v_prefix_tensor

        print(f"prefix_act_lens:{str(ifa_param['prefix_act_lens'])}")
    ##############################################################

    # # ===参数预处理===
    # actualSeqLengths_q = ifa_param['actualSeqLengths_q_raw']
    # actualSeqLengths = ifa_param['actualSeqLengths_raw']
    # inputLayout = ifa_param['inputLayout']
    # numHeads = ifa_param['numHeads']
    # numKeyValueHeads = ifa_param['numKeyValueHeads']
    # p_shape = ifa_param['p_shape']
    # k_start_index = ifa_param['k_start_index']
    # k_end_index = ifa_param['k_end_index']
    # v_start_index = ifa_param['v_start_index']
    # v_end_index = ifa_param['v_end_index']
    # out_shape = ifa_param['out_shape']
    # k_ori_shape_list = copy.deepcopy(ifa_param['k_shape_list'])
    # v_ori_shape_list = copy.deepcopy(ifa_param['v_shape_list'])
    # k_dtype = ifa_param['k_dtype']
    # v_dtype = ifa_param['v_dtype']
    # m_tensor = ifa_param['m_tensor']
    # m_shape = ifa_param['m_shape']

    # # >> actualSeqLengths预处理：actualSeqLengths为单值场景，如果长度为1且b不为1，则将actualSeqLengths扩展为b个单值的列表
    # # >> 仅非tnd场景，或tnd+pa场景，需要处理
    # if not ifa_param['tnd_flag'] or (ifa_param['tnd_flag'] and ifa_param['pa_flag']):
    #     if len(actualSeqLengths) == 1 and len(actualSeqLengths) != ifa_param['b']:
    #         actualSeqLengths_item = actualSeqLengths[0]
    #         for b_count in range(ifa_param['b'] - 1):
    #             actualSeqLengths.append(actualSeqLengths_item)
    #     # >> actualSeqLengths预处理：actualSeqLengths长度超过b
    #     if len(actualSeqLengths) > ifa_param['b']:
    #         actualSeqLengths = actualSeqLengths[:ifa_param['b']]
    #     if len(actualSeqLengths_q) > ifa_param['b']:
    #         actualSeqLengths_q = actualSeqLengths_q[:ifa_param['b']]
    # if ifa_param['tnd_flag']:
    #     # 将tnd格式下的act seq转成普通的act seq
    #     actualSeqLengths_q = trans_tnd_actseq(actualSeqLengths_q)
    #     if not ifa_param['pa_flag']:
    #         actualSeqLengths = trans_tnd_actseq(actualSeqLengths)
    #     print(f"[INFO]trans_tnd_actseq end.")

    # ifa_param['actualSeqLengths'] = actualSeqLengths
    # ifa_param['actualSeqLengths_q'] = actualSeqLengths_q

    # # deepseek 预处理： 1.将Q与QROPE拼接 2.将K与KROPE拼接
    # if ifa_param['deep_seek_flag']:
    #     # 备份拼接前的信息
    #     ifa_param['q_tensor_old'] = copy.deepcopy(ifa_param['q_tensor'])
    #     ifa_param['q_shape_old'] = copy.deepcopy(ifa_param['q_shape'])
    #     ifa_param['k_tensor_list_old'] = copy.deepcopy(ifa_param['k_tensor_list'])
    #     ifa_param['k_shape_list_old'] = copy.deepcopy(ifa_param['k_shape_list'])

    #     q_new_tensor, q_new_shape = concat_tensor(ifa_param['q_tensor'], ifa_param['q_shape'],
    #                                               ifa_param['q_rope_tensor'],
    #                                               ifa_param['q_rope_shape'], numHeads, ifa_param['tnd_flag'])
    #     if q_new_tensor is not None:
    #         ifa_param['q_tensor'] = q_new_tensor
    #         ifa_param['q_shape'] = q_new_shape
    #     else:
    #         print("[ERROR]q tensor的deepseek预处理异常，输出空tensor！")
    #         return torch.zeros(out_shape)

    #     if ifa_param['k_num'] == 1:
    #         k_new_tensor, k_new_shape = concat_tensor(ifa_param['k_tensor_list'][0], ifa_param['k_shape_list'][0],
    #                                                   ifa_param['k_rope_tensor'],
    #                                                   ifa_param['k_rope_shape'], numKeyValueHeads,
    #                                                   ifa_param['tnd_flag'])
    #         if k_new_tensor is not None:
    #             ifa_param['k_tensor_list'][0] = k_new_tensor
    #             ifa_param['k_shape_list'][0] = k_new_shape
    #         else:
    #             print("[ERROR]k tensor的deepseek预处理异常，输出空tensor！")
    #             return torch.zeros(out_shape)

    #     else:
    #         print("[ERROR]k tensor 长度不为1, deepseek预处理异常，输出空tensor！")
    #         return torch.zeros(out_shape)

    # # >> q 预处理：将q转换为bnsd
    # q_bnsd_tensor, q_bnsd_shape = _n_trans_shape_to_bnsd(ifa_param['q_tensor'], ifa_param['q_shape'], inputLayout,
    #                                                      numHeads, actualSeqLengths_q)
    # ifa_param['q_tensor_bnsd'] = q_bnsd_tensor
    # ifa_param['q_shape_bnsd'] = q_bnsd_shape
    # ifa_param['q_d'] = q_bnsd_shape[3]
    # ifa_param['q_b'] = q_bnsd_shape[0]

    # # >> kv预处理：1、将kv list 转换为bnsd 2、GQA场景，将kvn扩展为qn
    # k_tensor_list = []
    # v_tensor_list = []
    # k_shape_list = []
    # v_shape_list = []
    # k_tensor_bnsd_list = []
    # v_tensor_bnsd_list = []
    # k_shape_bnsd_list = []
    # v_shape_bnsd_list = []
    # ifa_param['ks_max'] = 0

    # if ifa_param['tnd_flag'] and ifa_param['pa_flag']:
    #     kv_layout = "BNSD"
    # else:
    #     kv_layout = inputLayout

    # for i in range(0, ifa_param['k_num']):
    #     k_tensor, k_bnsd_shape = _n_trans_shape_to_bnsd(ifa_param['k_tensor_list'][i], ifa_param['k_shape_list'][i],
    #                                                     kv_layout,
    #                                                     numKeyValueHeads, actualSeqLengths)
    #     k_tensor_bnsd_list.append(k_tensor)
    #     k_shape_bnsd_list.append(k_bnsd_shape)
    #     # k int32预处理
    #     if ifa_param['k_dtype'] == "int32":
    #         k_tensor, k_bnsd_shape = trans_int32_2_int4_tensor_bnsd(k_tensor, k_bnsd_shape)

    #     if numKeyValueHeads != numHeads:
    #         k_tensor, k_bnsd_shape = _t_broadcastKV_sigle(numHeads, numKeyValueHeads, k_tensor, k_dtype)
    #     if k_bnsd_shape[2] > ifa_param['ks_max']:
    #         ifa_param['ks_max'] = k_bnsd_shape[2]
    #     k_tensor_list.append(k_tensor)
    #     k_shape_list.append(k_bnsd_shape)

    # x = 0
    # for i in range(0, ifa_param['k_num']):
    #     v_tensor, v_bnsd_shape = _n_trans_shape_to_bnsd(ifa_param['v_tensor_list'][i], v_ori_shape_list[x], kv_layout,
    #                                                     numKeyValueHeads, actualSeqLengths)
    #     x = x + 1
    #     v_tensor_bnsd_list.append(v_tensor)
    #     v_shape_bnsd_list.append(v_bnsd_shape)
    #     # v int32预处理
    #     if ifa_param['v_dtype'] == "int32":
    #         v_tensor, v_bnsd_shape = trans_int32_2_int4_tensor_bnsd(v_tensor, v_bnsd_shape)

    #     if numKeyValueHeads != numHeads:
    #         # print("broadcast v")
    #         v_tensor, v_bnsd_shape = _t_broadcastKV_sigle(numHeads, numKeyValueHeads, v_tensor, v_dtype)
    #     v_tensor_list.append(v_tensor)
    #     v_shape_list.append(v_bnsd_shape)

    # ifa_param['k_tensor_list_bnsd'] = k_tensor_list
    # ifa_param['k_shape_list_bnsd'] = k_shape_list
    # ifa_param['v_tensor_list_bnsd'] = v_tensor_list
    # ifa_param['v_shape_list_bnsd'] = v_shape_list

    # # >> left padding 异常场景处理：起点/终点异常，返回全0
    # if ifa_param['padding_flag']:
    #     max_act_seq = max(actualSeqLengths)
    #     kv_s = ifa_param['ks_max']
    #     if kv_s - ifa_param['padding_size'] - max_act_seq < 0:
    #         print('[WARNING]:paddingsize 溢出，输出空tensor！')
    #         return torch.zeros(out_shape)
    # # qs大于1
    # qs = q_bnsd_shape[2]
    # sparse_mode = ifa_param['sparse_mode']
    # m_dtype = tools.get_pt_dtype(params['dtype_input'][v_end_index + 2])
    # m_tensor = torch.tensor(m_tensor, dtype=m_dtype)
    # if sparse_mode == 3:
    #     batch = q_bnsd_shape[0]
    #     numheads = q_bnsd_shape[1]
    #     npu_m_shape_s = m_shape
    #     if sparse_mode == 0 or sparse_mode == 1:
    #         batch, numheads, ns1, ns2 = get_attention_mask_batch_num(m_shape,
    #                                                                  q_bnsd_shape)  # 获取输入attentionmask的batch 和numhead
    #         npu_m_shape_s = [ns1, ns2]
    #     kvs = k_shape_list[0][2]
    #     cpu_m_shape = [qs, kvs]  # cpu
    #     preTokens = ifa_param['pre_tokens']
    #     nextTokens = ifa_param['next_tokens']
    #     actualprefixKV = 0
    #     prefix_kvs = 0
    #     kvs_list = [kvs]
    #     cpu_m_tensor, npu_m_tensor, preTokens, nextTokens = _create_random_mask_by_spars(cpu_m_shape, npu_m_shape_s,
    #                                                                                      m_dtype, preTokens,
    #                                                                                      nextTokens,
    #                                                                                      ifa_param[
    #                                                                                          'actualSeqLengths_q'],
    #                                                                                      actualSeqLengths,
    #                                                                                      actualprefixKV,
    #                                                                                      prefix_kvs, kvs_list,
    #                                                                                      batch,
    #                                                                                      numheads, sparse_mode)
    #     if sparse_mode == 0 or sparse_mode == 1:
    #         m_bnsd_tensor = _np_broadcast_mask_n(cpu_m_tensor, npu_m_shape, cpu_m_shape, numHeads, q_bnsd_shape[0])
    #     else:
    #         cpu_m_tensor = np.array(cpu_m_tensor)
    #         cpu_m_tensor = torch.from_numpy(cpu_m_tensor)
    #         ifa_param['m_tensor'] = _t_broadcast_mask_n(cpu_m_tensor, cpu_m_tensor.shape, numHeads, q_bnsd_shape[0])
    # else:
    #     # >> m预处理：1、将m扩展为BN1S
    #     # print(f"m_tensor_raw===={m_tensor}")
    #     if ifa_param['m_flag']:
    #         ifa_param['m_tensor'] = _t_broadcast_mask_n(m_tensor, m_shape, numHeads, m_shape[0])

    # ===计算入口===
    y_all, lse = _t_increattention_bnsd(ifa_param)
    y_all = torch.from_numpy(y_all)
    y_all = trans_bnsd_to_layout(y_all, out_shape, inputLayout, actualSeqLengths_q)
    print(f"out_shape:{out_shape}")
    return y_all

