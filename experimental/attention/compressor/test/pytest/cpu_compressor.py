# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import random
import torch
import torch_npu
import torchair
import custom_ops
import numpy as np
import torch.nn as nn

np.random.seed(21)  # 固定随机种子
np.set_printoptions(suppress=True)

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))

def get_seq_used_by_batch(batch_idx, S, seqused, cu_seqlens):
    if seqused is not None:
        return seqused[batch_idx]
    else:
        if cu_seqlens is not None:
            return cu_seqlens[batch_idx + 1] - cu_seqlens[batch_idx]
        else:
            return S

def softmax_columns(z):
    """
    对输入矩阵 z 按列计算 Softmax。
    参数:
    z -- 一个二维NumPy数组（矩阵），形状为 (行, 列)
    返回:
    一个与 z 形状相同的数组，其中每一列都是一个概率分布（该列元素之和为1）。
    """
    # 仅支持float32计算
    # 1. 数值稳定技巧：减去每列的最大值，防止指数计算溢出
    z_max = np.max(z, axis=0, keepdims=True) # 计算每列最大值，保持二维维度
    z_stable = z - z_max
    
    # 2. 计算指数
    exp_z = np.exp(z_stable)
    
    # 3. 按列求和（axis=0），并进行归一化
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def rms_norm(x, weight, eps):
    # 仅支持float32计算
    var = np.mean(np.square(x), axis=-1, keepdims=True)
    x = x * np.reciprocal(np.sqrt(var + eps))
    return weight * x

def rotary_emb(x, rope_sin, rope_cos, rotary_mode):
    """
    参数:
    x -- shape is (sc, rope_head_dim)
    rope_sin -- shape is (sc, rope_head_dim)
    rope_cos -- shape is (sc, rope_head_dim)
    rotary_mode -- 1:half 2:interleave
    """
    sc = x.shape[0]
    rope_head_dim = x.shape[-1]
    rope_sin = rope_sin.reshape(sc, rope_head_dim)
    rope_cos = rope_cos.reshape(sc, rope_head_dim)
    y = np.zeros(shape=x.shape, dtype=x.dtype)
    group = rope_head_dim // 2
    for s in range(sc):
        for i in range(group):
            if rotary_mode == 1:
                a = x[s][i]
                b = x[s][i + group]
                y[s][i] = a * rope_cos[s][i] - b * rope_sin[s][i] # y_a
                y[s][i + group] = a * rope_sin[s][i + group] + b * rope_cos[s][i + group] # y_b
            if rotary_mode == 2:
                idx = 2 * i
                a = x[s][idx]
                b = x[s][idx + 1]
                y[s][idx] = a * rope_cos[s][idx] - b * rope_sin[s][idx] # y_a
                y[s][idx + 1] = a * rope_sin[s][idx + 1] + b * rope_cos[s][idx + 1] # y_b
    return y

# state.shape is (block_num, block_size, coff * head_dim), is numpy type
# new_state.shape is (s, coff * head_dim), data writed to state
# block_table.shape is (B, (s_max + block_size - 1) // block_size)
def write_state_page_cache(state, sc_new_state, b_idx, start_seq_idx, end_seq_idx, block_table):
    block_size = state.shape[1] ## 应该从state，不是从block_table
    seq_cnt = end_seq_idx - start_seq_idx
    finish_cnt = 0
    while finish_cnt < seq_cnt:
        cur_seq_id = start_seq_idx + finish_cnt
        block_id = block_table[b_idx][cur_seq_id // block_size]
        block_start_seq_id = cur_seq_id % block_size
        can_write_seq_cnt = block_size - block_start_seq_id
        if can_write_seq_cnt > seq_cnt - finish_cnt:
            can_write_seq_cnt = seq_cnt - finish_cnt
        # block_id为0表示block无效,不写数据
        if block_id != 0:
            state[block_id:(block_id+1), block_start_seq_id:(block_start_seq_id + can_write_seq_cnt), :] = sc_new_state[finish_cnt:(finish_cnt + can_write_seq_cnt), :]
        finish_cnt = finish_cnt + can_write_seq_cnt

# state.shape is (block_num, block_size, coff * head_dim)
# block_table.shape is (B, (s_max + block_size - 1) // block_size)
# when coff_id = 0, read pre data; when coff_id = 1, read cur data
# out.shape is (end_seq_idx - start_seq_idx, head_dim)
def read_state_page_cache(state, b_idx, start_seq_idx, end_seq_idx, block_table, d_start, d_end):
    result = np.zeros(shape=(end_seq_idx - start_seq_idx, d_end - d_start), dtype=np.float32)
    block_size = state.shape[1] ## 应该从state，不是从block_table
    seq_cnt = end_seq_idx - start_seq_idx
    finish_cnt = 0
    while finish_cnt < seq_cnt:
        cur_seq_id = start_seq_idx + finish_cnt
        block_id = block_table[b_idx][cur_seq_id // block_size]
        block_start_seq_id = cur_seq_id % block_size
        can_read_seq_cnt = block_size - block_start_seq_id
        if can_read_seq_cnt > seq_cnt - finish_cnt:
            can_read_seq_cnt = seq_cnt - finish_cnt
        # 如果block_id为0, 输出错误日志, 但是依然读数据
        if block_id == 0:
            print(f"Error Info: [read_state_page_cache] block_id of block_table is 0, b_idx={b_idx} cur_seq_id={cur_seq_id} block_size={block_size}")
        result[finish_cnt:(finish_cnt + can_read_seq_cnt), :] = state[
            block_id:(block_id+1), block_start_seq_id:(block_start_seq_id + can_read_seq_cnt), d_start:d_end]
        finish_cnt = finish_cnt + can_read_seq_cnt
    return result

def cpu_compressor(
    x, wkv, wgate, kv_state, score_state, ape, norm_weight, rope_sin, rope_cos,
    block_table=None, cu_seqlens=None, seqused=None, start_pos=None,
    rope_head_dim=64, cmp_ratio=4, coff=1, norm_eps=1e-6, rotary_mode=1):
    x_dtype = x.dtype
    x = x.to(torch.float32).numpy()
    wkv = wkv.to(torch.float32).numpy()
    wgate = wgate.to(torch.float32).numpy()
    kv_state = kv_state.numpy()
    score_state = score_state.numpy()
    ape = ape.numpy()
    norm_weight = norm_weight.to(torch.float32).numpy()
    rope_sin = rope_sin.to(torch.float32).numpy()
    rope_cos = rope_cos.to(torch.float32).numpy()
    matmul_dtype = np.float32
    for i in range(wkv.shape[1] // 128):
        leftH = i *128
        rightH = (i + 1) * 128
    new_kv_state = np.matmul(x, wkv.T, dtype=matmul_dtype)
    new_score_state = np.matmul(x, wgate.T, dtype=matmul_dtype)

    B = len(start_pos)
    head_dim = wkv.shape[0] // coff
    bs_combine_flag = False
    if cu_seqlens is not None:
        bs_combine_flag = True

    # 将B和S合轴, 便于后面取索引
    if bs_combine_flag == False:
        S = x.shape[1]
        new_kv_state = new_kv_state.reshape(B * S, new_kv_state.shape[-1])
        new_score_state = new_score_state.reshape(B * S, new_score_state.shape[-1])
        cmp_kv = np.zeros(shape=(B, (S + cmp_ratio - 1) // cmp_ratio, head_dim), dtype=matmul_dtype)
    else:
        cmp_kv = np.zeros(shape=(min(x.shape[0], x.shape[0] // cmp_ratio + B), head_dim), dtype=matmul_dtype)
    cmp_kv_mask = np.zeros_like(cmp_kv, dtype=bool)
    out_cu_seqlen = [0] * (B + 1)
    out_seqused = [0] * B

    out_sum_sc_cnt = 0
    for b_idx in range(B):
        batch_out_sc_id = 0
        batch_start_pos = start_pos[b_idx]
        # get_seq_used_by_batch
        if seqused is not None:
            batch_seq_used = seqused[b_idx]
        else:
            if bs_combine_flag == False:
                batch_seq_used = x.shape[1] # x的shape为3维, BSh
            else:
                batch_seq_used = cu_seqlens[b_idx + 1] - cu_seqlens[b_idx]# x的shape为2维, Th
        # 小于compress_seq_id的seq_id为需要压缩的token
        compress_seq_id = (batch_start_pos + batch_seq_used) // cmp_ratio * cmp_ratio

        batch_seq_idx = 0
        while batch_seq_idx < batch_seq_used:
            # 获取要处理的S方向起始和结束点, 已加start_pos
            start_seq_idx = batch_start_pos + batch_seq_idx
            end_seq_idx = start_seq_idx // cmp_ratio * cmp_ratio + cmp_ratio
            if end_seq_idx > (batch_start_pos + batch_seq_used):
                end_seq_idx = batch_start_pos + batch_seq_used

            # calc offset T of one sc, for get data from new state
            base_offset = cu_seqlens[b_idx] if bs_combine_flag else b_idx * x.shape[1]
            start_offset = base_offset + (start_seq_idx - batch_start_pos)
            end_offset = base_offset + (end_seq_idx - batch_start_pos)

            # add ape
            start_seq_id_in_sc = start_seq_idx % cmp_ratio
            end_seq_idx_in_sc = start_seq_id_in_sc + (end_seq_idx - start_seq_idx)
            new_score_state[start_offset:end_offset, :] = np.add(new_score_state[start_offset:end_offset, :], ape[start_seq_id_in_sc : end_seq_idx_in_sc, :])

            # 1.判断块是否需要存储到state
            # 2.判断块是否需要压缩
            save_flag = True if start_seq_idx >= (compress_seq_id - (coff - 1) * cmp_ratio) else False
            compress_flag = True if start_seq_idx < compress_seq_id else False

            if save_flag:
                tmp_kv_state = new_kv_state[start_offset:end_offset, :]
                tmp_score_state = new_score_state[start_offset:end_offset, :]
                write_state_page_cache(kv_state, tmp_kv_state, b_idx, start_seq_idx, end_seq_idx, block_table)
                write_state_page_cache(score_state, tmp_score_state, b_idx, start_seq_idx, end_seq_idx, block_table)

            if compress_flag:
                # init value and shape of one sc state
                sc_kv_state = np.zeros(shape=(coff, cmp_ratio, head_dim), dtype=matmul_dtype)
                sc_score_state = np.full(shape=(coff, cmp_ratio, head_dim), fill_value=-float('inf'), dtype=matmul_dtype)

                # fill cur data
                coff_id = coff - 1
                d_start = coff_id * head_dim
                d_end = (coff_id + 1) * head_dim
                cnt_from_state = 0
                if batch_start_pos == start_seq_idx:
                    # 第一块
                    cnt_from_state = batch_start_pos % cmp_ratio
                    if cnt_from_state > 0:
                        copy_start_seq_id = batch_start_pos - cnt_from_state
                        copy_end_seq_id = batch_start_pos
                        sc_kv_state[coff_id, 0:cnt_from_state, :] = read_state_page_cache(kv_state, b_idx, copy_start_seq_id, copy_end_seq_id, block_table, d_start, d_end)
                        sc_score_state[coff_id, 0:cnt_from_state, :] = read_state_page_cache(score_state, b_idx, copy_start_seq_id, copy_end_seq_id, block_table, d_start, d_end)
                sc_kv_state[coff_id, cnt_from_state:cmp_ratio, :] = new_kv_state[start_offset:end_offset, d_start:d_end]
                sc_score_state[coff_id, cnt_from_state:cmp_ratio, :] = new_score_state[start_offset:end_offset, d_start:d_end]

                # fill pre data
                if coff == 2:
                    coff_id = 0
                    d_start = coff_id * head_dim
                    d_end = (coff_id + 1) * head_dim
                    cnt_from_state = 0
                    if batch_start_pos == start_seq_idx:
                        # 第一块
                        # when batch_start_pos < cmp_ratio, no exist pre data, no need copy;
                        # others, all pre data need copy from state
                        cnt_from_state = cmp_ratio
                        if batch_start_pos >= cmp_ratio:
                            copy_start_seq_id = batch_start_pos - batch_start_pos % cmp_ratio - cmp_ratio
                            copy_end_seq_id = copy_start_seq_id + cnt_from_state
                            sc_kv_state[coff_id, 0:cnt_from_state, :] = read_state_page_cache(kv_state, b_idx, copy_start_seq_id, copy_end_seq_id, block_table, d_start, d_end)
                            sc_score_state[coff_id, 0:cnt_from_state, :] = read_state_page_cache(score_state, b_idx, copy_start_seq_id, copy_end_seq_id, block_table, d_start, d_end)
                    elif start_seq_idx - cmp_ratio < batch_start_pos:
                        # 第二块, pre数据部分在state中
                        cnt_from_state = batch_start_pos % cmp_ratio
                        if cnt_from_state > 0: # 不可能为0
                            copy_start_seq_id = batch_start_pos - batch_start_pos % cmp_ratio
                            copy_end_seq_id = batch_start_pos
                            sc_kv_state[coff_id, 0:cnt_from_state, :] = read_state_page_cache(kv_state, b_idx, copy_start_seq_id, copy_end_seq_id, block_table, d_start, d_end)
                            sc_score_state[coff_id, 0:cnt_from_state,:] = read_state_page_cache(score_state, b_idx, copy_start_seq_id, copy_end_seq_id, block_table, d_start, d_end)
                    if cnt_from_state < cmp_ratio:
                        # 需要拷贝的数据就在start_offset的前面
                        pre_start_offset = start_offset - (cmp_ratio - cnt_from_state)
                        pre_end_offset = start_offset
                        sc_kv_state[coff_id, cnt_from_state:cmp_ratio, :] = new_kv_state[pre_start_offset:pre_end_offset, d_start:d_end]
                        sc_score_state[coff_id, cnt_from_state:cmp_ratio, :] = new_score_state[pre_start_offset:pre_end_offset, d_start:d_end]

                # softmax_columns for score state
                sc_kv_state = sc_kv_state.reshape(coff * cmp_ratio, head_dim)
                sc_score_state = sc_score_state.reshape(coff * cmp_ratio, head_dim)
                sc_score_state = softmax_columns(sc_score_state)
                # kv * score
                sc_data = sc_kv_state * sc_score_state
                # reduce sum
                sc_cmp_kv = np.sum(sc_data, axis=0, keepdims=True)
                # RmsNorm
                sc_cmp_kv = rms_norm(sc_cmp_kv, norm_weight, norm_eps)
                # inplace rotary_emb
                
                if bs_combine_flag == False:
                    sc_cmp_kv[:, -rope_head_dim:] = rotary_emb(sc_cmp_kv[:, -rope_head_dim:], rope_sin[b_idx, batch_out_sc_id, :], rope_cos[b_idx, batch_out_sc_id, :, :], rotary_mode)
                    cmp_kv[b_idx, batch_out_sc_id, :] = sc_cmp_kv
                    cmp_kv_mask[b_idx, batch_out_sc_id, :] = 1
                else:
                    sc_cmp_kv[:, -rope_head_dim:] = rotary_emb(sc_cmp_kv[:, -rope_head_dim:], rope_sin[out_sum_sc_cnt, :], rope_cos[out_sum_sc_cnt, :], rotary_mode)
                    cmp_kv[out_sum_sc_cnt, :] = sc_cmp_kv
                    cmp_kv_mask[out_sum_sc_cnt, :] = 1
                batch_out_sc_id = batch_out_sc_id + 1
                out_sum_sc_cnt = out_sum_sc_cnt + 1

            # update loop idx
            batch_seq_idx = end_seq_idx - batch_start_pos
        out_cu_seqlen[b_idx + 1] = out_sum_sc_cnt
        out_seqused[b_idx] = batch_out_sc_id

    print(f"out_cu_seqlen: {out_cu_seqlen}")
    print(f"out_seqused: {out_seqused}")
    print(f"cmp_kv.shape: {cmp_kv.shape}")
    cmp_kv_torch = torch.tensor(cmp_kv).to(x_dtype)
    return cmp_kv_torch, cmp_kv_mask