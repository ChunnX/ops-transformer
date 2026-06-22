#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MaskKComp Self-Test: Data Generation Script

Generates test input data according to aclnnMaskKComp interface constraints.
Saves data as .bin files for NPU execution.

Constraints:
  - query/key: FP16, BNSD layout [B, N, Sq, D]
  - headDim = 128 (fixed)
  - blockShapeX/Y: must be multiples of 64
  - sparsity: in range (0, 1)
  - numKeyHeads == numHeads (MHA only)

Outputs:
  - input_query.bin:   FP16, shape [B, N, Sq, D]
  - input_key.bin:     FP16, shape [B, N, Skv, D]
  - input_params.bin:  INT64, [B, N, Sq, Skv, D, blockX, blockY, sparsity*1000]
"""

import os
import argparse
import torch

seed=42
torch.manual_seed(seed)
import random
random.seed(seed)

# torch.npu.empty_cache()
# device = "npu:7"
# torch.npu.set_device(device)

DTYPE = torch.float16
#     const aclTensor *query,
#     const aclTensor *key,
#     const aclIntArray *blockShapeOptional,
#     const aclIntArray *postBlockShapeOptional,
#     const aclIntArray *actualSeqLengthsOptional,
#     const aclIntArray *actualSeqLengthsKvOptional,
#     const aclIntArray *actualBlockLenQueryOptional,
#     const aclIntArray *actualBlockLenKeyOptional,
#     char *qInputLayout,
#     char *kvInputLayout,
#     int64_t numKeyValueHeads,
#     double scaleValue,
#     double sparsity,
#     aclTensor *qCompOut,
#     aclTensor *kCompOut,
#     aclTensor *blockSparseMaskOut,
#     uint64_t *workspaceSize,
#     aclOpExecutor **executor);

def save_data(inputs):
    

def generate_seqlen_tnd(
    max_seqlen_q,           # q 的最大序列长度
    max_seqlen_kv,          # kv 的最大序列长度
    batch_size,            # q kv 的 batch size
):
    """
    根据 max_seqlen 和 batch size accumulate_seq_length
    每个样本的序列长度会在 [1, max_seqlen] 范围内随机生成
    """
    random.seed(seed)
    torch.manual_seed(seed)
    
    # 1. 随机生成每个样本的序列长度 (范围: 1 到 max_seqlen)
    seqlen_q_list = [random.randint(1, max_seqlen_q) for _ in range(batch_size)]
    seqlen_kv_list = [random.randint(1, max_seqlen_kv) for _ in range(batch_size)]

    # 将最大值设置为max_seqlen
    max_idx = torch.argmax(seqlen_q_list)
    seqlen_q_list[max_idx] = max_seqlen_q
    max_idx = torch.argmax(seqlen_kv_list)
    seqlen_kv_list[max_idx] = max_seqlen_kv

    # 2. 计算总序列长度
    total_seqlen_q = sum(seqlen_q_list)
    total_seqlen_kv = sum(seqlen_kv_list)
    
    # 3. 生成 accumulate_seq_length 索引列表
    # accumulate_seq_length = [0]
    # for length in seqlen_q_list:
    #     accumulate_seq_length.append(accumulate_seq_length[-1] + length)
    
    # accumulate_seq_lengths_kv = [0]
    # for length in seqlen_kv_list:
    #     accumulate_seq_lengths_kv.append(accumulate_seq_lengths_kv[-1] + length)
    
    return (
        # accumulate_seq_length,
        # accumulate_seq_lengths_kv,
        seqlen_q_list,
        seqlen_kv_list,
        total_seqlen_q,
        total_seqlen_kv,
    )
    
def generate_data(params):
    random.seed(seed)
    torch.manual_seed(seed)
    layout = params["layout"]
    B, N, S1, S2, D, bx, by = params["B"], params["N"], params["S1"], params["S2"], params["D"], params["bx"], params["by"]
    q = None
    k = None
    q_seqlen_list = []
    k_seqlen_list = []
    q_block_len_list = []
    k_block_len_list = []

    block_x_nums = (S1 + bx - 1) // bx
    block_y_nums = (S1 + by - 1) // by

    if layout == "BNSD":
        q = torch.rand((B, N, S1, D), dtype=DTYPE)
        K = torch.rand((B, N, S2, D), dtype=DTYPE)
        if params["use_seqlen"]:
            new_S1 = random.randint(1, S1)
            q_seqlen_list = B * [new_S1]
            new_s2 = random.randint(1, S2)
            q_seqlen_list = B * [S1]
            k_seqlen_list = B * [new_s2]
        else:
            q_seqlen_list = None
            k_seqlen_list = None

        if params["use_q_block_len"]:
            q_block_len_list = [[random.randint(0, bx) for _ in range(block_x_nums)] for _ in range(B)]
        else:
            q_block_len_list = None

        if params["use_k_block_len"]:
            k_block_len_list = [[random.randint(0, by) for _ in range(block_y_nums)] for _ in range(B)]
        else:
            k_block_len_list = None
    else:
        (
        seqlen_q_list,
        seqlen_kv_list,
        total_seqlen_q,
        total_seqlen_kv,
        ) = generate_seqlen_tnd(S1, S2, B)
        q_seqlen_list = seqlen_q_list
        k_seqlen_list = seqlen_kv_list
        q = torch.rand((total_seqlen_q, N, D), dtype=DTYPE)
        K = torch.rand((total_seqlen_kv, N, D), dtype=DTYPE)
        if params["use_q_block_len"]:
            # todo 疑问 按最大bloc_nums生成吗？
            q_block_len_list = [random.randint(0, bx) for _ in range(block_x_nums * B)]
        else:
            q_block_len_list = None

        if params["use_k_block_len"]:
            k_block_len_list = [random.randint(0, by) for _ in range(block_y_nums * B)]
        else:
            k_block_len_list = None


    return (
        q,
        k,
        q_seqlen_list,
        k_seqlen_list,
        q_block_len_list,
        k_block_len_list,
    )


if __name__ == "__main__":
    # B N S1 S2 D bx by layout sparsity use_seqlen use_q_block_len use_k_block_len
    # 参数名列表（按注释顺序）
    keys = ["B", "N", "S1", "S2", "D", "bx", "by", "layout", "sparsity", "use_seqlen", "use_q_block_len", "use_k_block_len"]
    config_params = [False, False, False]
    params = [1, 1, 128, 128, 128, 128, 128, "BNSD", 0.5, config_params]

    # 手动构建值列表：取 params 的前6个数字 + bx（第7个元素） + by（这里用 bx 的值） + layout + sparsity + 展开 config_params
    values = (
        params[:9]                     # B, N, S1, S2, D, bx, by, "layout", "sparsity"
        + config_params                # use_seqlen, use_q_block_len, use_k_block_len
    )
    # 转换为字典
    params_dict = dict(zip(keys, values))
    print(params_dict)

    (
        q,
        k,
        q_seqlen_list,
        k_seqlen_list,
        q_block_len_list,
        k_block_len_list,
    ) = generate_data(params_dict)
    
    block_shape = [params_dict["bx"], params_dict["by"]]
    input_keys = ["q", "k", "block_shape", "q_seqlen_list", "k_seqlen_list", "q_block_len_list", "k_block_len_list", "layout", "sparsity"]
    inputs = [q, k, block_shape, q_seqlen_list, k_seqlen_list, q_block_len_list, k_block_len_list]





