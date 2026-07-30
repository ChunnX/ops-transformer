# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import torch_npu
# import torchair
# import custom_ops
import numpy as np
import torch.nn as nn
import math
import random
# from precision_compare import data_compare

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))

np.set_printoptions(
    precision=4,           # 保留小数点后4位
    linewidth=1536 / 2,    # 每行显示128个数（约1536字符）
    threshold=np.inf,      # 完整打印，不截断
    suppress=False         # 不使用科学计数法（可选）
)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(42)

def check_result(golden_output, output, name=None):
    rtol = 5*1e-03
    atol = 5*1e-03
    diff = golden_output - output
    abs_diff = torch.abs(diff)
    print("[INFO] LIG {} max diff is {}, mean diff is {}".format(name, torch.max(abs_diff), torch.mean(abs_diff)))
    
    result_check = torch.allclose(golden_output, output, rtol=rtol, atol=atol)
    if not result_check:
        tolerance = atol + rtol * torch.abs(golden_output)
        mismatched_positions = torch.nonzero(abs_diff > tolerance)

        if golden_output.dim() == 1:
            mismatched_indices = torch.nonzero(mismatched_positions).flatten()
        else:
            mismatched_indices = torch.nonzero(mismatched_positions)

        # print(mismatched_positions.cpu().numpy())
        mismatched_golden = golden_output[mismatched_positions.split(1, dim=1)]
        mismatched_actual = output[mismatched_positions.split(1, dim=1)]
        mismatched_diffs = diff[mismatched_positions.split(1, dim=1)]
        
        if mismatched_diffs.numel() > (golden_output.numel() * 0.001):
            flatten_m_golden = torch.flatten(mismatched_golden)
            flatten_m_actual = torch.flatten(mismatched_actual)
            flatten_m_diff = torch.flatten(mismatched_diffs)
            print_n = min(100, mismatched_diffs.numel())
            for i in range(print_n):
                print("index: {}, expect: {}, actual: {}, diff is:{}".format(i, flatten_m_golden[i], flatten_m_actual[i], flatten_m_diff[i]))
            print("[INFO] FlashAttention BWD {} Check Acuraccy, diff num is: {}, percentage is: {}".format(
                    name,
                    mismatched_diffs.numel(), \
                    mismatched_diffs.numel() * 1.0 / output.numel()))
        else:
            print("[INFO] LIG {} Check Acuraccy Pass".format(name))
    else:
        print("[INFO] LIG {} Check Acuraccy Pass".format(name))

class LightningIndicesKLLoss:
    def __init__(self, q, k, query_index, key_index, weight, topk_index, q_rope, k_rope, actual_seq_q_len, actual_seq_kv_len, scale, layout, block_size=1):
        self.q = torch.cat((q, q_rope), -1).to(torch.float)
        self.k = torch.cat((k, k_rope), -1).to(torch.float)
        self.q_index = query_index.to(torch.float)
        self.k_index = key_index.to(torch.float)
        self.weight = weight.to(torch.float)
        self.topk_index = topk_index
        self.actual_seq_q_len = actual_seq_q_len
        self.actual_seq_kv_len = actual_seq_kv_len
        self.scale = scale
        self.layout = layout
        self.block_size = block_size

        self.B = len(actual_seq_q_len)
        self.N1 = self.q.shape[1]
        self.N2 = self.k.shape[1]
        self.Nqi = self.q_index.shape[1]
        self.Nki = self.k_index.shape[1]
        self.G = self.N1 // self.N2
        self.G_index = self.Nqi // self.Nki
        self.T1 = self.actual_seq_q_len[-1]
        self.T2 = self.actual_seq_kv_len[-1]
        self.D1 = self.q.shape[2]
        self.D2 = self.k_index.shape[2]

    def getCaseInfo(self, q, k, q_index, k_index, sparse_indices):
        D = q.shape[3]
        B = q.shape[0]
        S1 = q.shape[1]
        S2 = k.shape[1]
        N1 = q.shape[2]
        N2 = k.shape[2]
        G = N1 // N2
        Nqi = q_index.shape[2]
        Nki = k_index.shape[2]
        G_index = Nqi // Nki
        D_index = q_index.shape[3]
        topKSize = sparse_indices.shape[3]
        return B, S1, S2, N1, N2, G, Nqi, Nki, G_index, D, D_index, topKSize

    def getTNDCaseInfo(self, q, k, q_index, k_index, sparse_indices, actual_seq_lengths_query, actual_seq_lengths_key):
        B = len(actual_seq_lengths_query)
        T1 = actual_seq_lengths_query[-1]
        T2 = actual_seq_lengths_key[-1]
        NQuery = q.shape[1]
        N2 = k.shape[1]
        NQueryIndex = q_index.shape[1]
        D1 = q.shape[2]
        D2 = q_index.shape[2]
        return B, T1, T2, NQuery, N2, NQueryIndex, D1, D2

    def get_spoint_by_bIdx(self, bIdx):
        Sq, Skv = 0, 0
        if bIdx == 0:
            Sq = self.actual_seq_q_len[0]
            Skv = self.actual_seq_kv_len[0]
        elif bIdx > 0:
            Sq = self.actual_seq_q_len[bIdx]
            Skv = self.actual_seq_kv_len[bIdx]
        return Sq, Skv

    def run_unpad(self):
        if self.layout != "TND":
            raise ValueError(f"[error] case layout is not invalid!, case layout is {self.layout}, please check correct interface!")
        # 初始化zeros tensor 接收每个B的输出
        B, T1, T2, NQuery, N2, NQueryIndex, DQuery, DQuerIndex = self.getTNDCaseInfo(self.q, self.k, self.q_index, self.k_index, self.topk_index, self.actual_seq_q_len, self.actual_seq_kv_len)
        d_query_index = torch.zeros(T1,NQueryIndex,DQuerIndex)
        d_key_index = torch.zeros(T2,N2,DQuerIndex)
        d_weight = torch.zeros(T1,NQueryIndex)
        loss = torch.zeros(1)
        p_max = None
        p_sum = None

        for bIdx in range(B):
            q_start, kv_start = self.get_spoint_by_bIdx(bIdx-1)
            q_end, kv_end = self.get_spoint_by_bIdx(bIdx)
            if q_start == q_end:
                continue
            # transpose to BSND
            q_i = self.q[q_start:q_end,:,:].unsqueeze(0) # (T1, N1, DQuery) -> (1, S1, N1, DQuery)
            k_i = self.k[kv_start:kv_end,:,:].unsqueeze(0) # (T1, N2, DQuery) -> (1, S2, N2, DQuery)
            q_index_i = self.q_index[q_start:q_end,:,:].unsqueeze(0)
            k_index_i = self.k_index[kv_start:kv_end,:,:].unsqueeze(0)
            weight = self.weight[q_start:q_end, :].unsqueeze(0)
            sparse_indices = self.topk_index[q_start:q_end, :, : ].unsqueeze(0) # [S1, N2, K] -> [1, S1, N2, K]

            d_query_index_i, d_key_index_i, d_weight_i, loss_i, p_max_i, p_sum_i, key_topk, bmm2_res, bmm1_res, reluRes, sReduceGRes, sSoftmaxRes, pReduceGRes = \
                self.golden(q_i, k_i, q_index_i, k_index_i, self.scale, weight, sparse_indices)

            d_query_index[q_start:q_end,:,:] = d_query_index_i
            d_key_index[kv_start:kv_end,:,:] = d_key_index_i
            d_weight[q_start:q_end,:] = d_weight_i
            loss[0] += loss_i

            if p_max is None:
                p_max = p_max_i
                p_sum = p_sum_i
            else:
                p_max = torch.cat((p_max, p_max_i), dim=0)
                p_sum = torch.cat((p_sum, p_sum_i), dim=0)
        p_max.unsqueeze(0)
        p_sum.unsqueeze(0)
        return d_query_index, d_key_index, d_weight, loss, p_max, p_sum

    def tsoftmax(self, x):
        x_max = torch.max(x, dim=-1, keepdims=True)[0]
        x_sub = x.sub(x_max)
        y = torch.exp(x_sub)
        x_sum = y.sum(dim=-1, keepdims=True)
        ans = y.div(x_sum)
        return ans, x_max, x_sum

    def safe_gather(self, key, topk_index, S1):
        S2 = key.size(1)
        mask = (topk_index < 0) | (topk_index >= S2)  # 标记非法索引
        # 先安全限制索引范围
        safe_index = topk_index.clamp(min=0, max=S2 - 1) #(1 S1 TOPK D)
        # 安全 gather
        key_expanded = key.unsqueeze(1).expand(-1, S1, -1, -1) # (1 S1 S2 D)
        key_topk = torch.gather(key_expanded, 2, safe_index)
        # 把非法部分置为 0（或其他你想要的默认值）
        key_topk = key_topk.masked_fill(mask, 0.0)
        return key_topk

    def safe_scatter_add(self, beforeScatterAdd, sparse_indices, d_key_index):
        B, S1, S2, D_index = d_key_index.shape
        # 1️⃣ 生成 mask，标记哪些索引是有效的
        mask = sparse_indices != -1   # True 表示合法，False 表示 -1
        # 2️⃣ 将 -1 替换成 0（以防 scatter 越界）
        safe_indices = sparse_indices.clone()
        safe_indices[~mask] = 0
        # 3️⃣ 扩展索引维度以匹配 scatter_add 需求
        index_expanded = safe_indices.unsqueeze(-1).expand(-1, -1, -1, D_index)  # [B, S1, K, D]
        # 4️⃣ 对应无效位置的 beforeScatterAdd 清零（跳过聚合）
        masked_beforeScatterAdd = beforeScatterAdd * mask.unsqueeze(-1)
        # 5️⃣ scatter_add 聚合
        d_key_index.scatter_add_(2, index_expanded, masked_beforeScatterAdd)
        return d_key_index

    def deal_invalid_token(self, B, S1, S2, topKSize, src_tensor):
        dst_tensor = src_tensor
        for b_idx in range(B):
            for s1_idx in range(S1):
                invalid_len = 0
                s2_realsize = (int)((S2 - S1) + s1_idx + 1)
                if s2_realsize <= 0:
                    s2_realsize = S2
                if s2_realsize > topKSize:
                    block_remain = s2_realsize % self.block_size
                    if (self.block_size != 1 and block_remain !=0):
                        s2_realsize = topKSize - (self.block_size - block_remain)
                        dst_tensor[b_idx, s1_idx, :, s2_realsize : ] = -float('inf')
                else:
                    invalid_len = topKSize - s2_realsize
                    if invalid_len > 0:
                        dst_tensor[b_idx, s1_idx, :, s2_realsize :] = -float('inf')
        return dst_tensor


    def golden(self, q, k, q_index, k_index, scale, weight, sparse_indices):
        B, S1, S2, N1, N2, G, Nqi, Nki, G_index, D, D_index, topKSize = self.getCaseInfo(q, k, q_index, k_index, sparse_indices)
        # print("==================计算 P==================")
        query = q.to(torch.float)      # (B, S1, N1, D)
        key = k.squeeze(2).to(torch.float) #N2轴压缩
        sparse_indices = sparse_indices.squeeze(2).to(torch.int64) # (1 T1 TOPK)
        topk_index = sparse_indices.unsqueeze(-1).expand(-1, -1, -1, D) # (1 T1 TOPK D)

        key_topk = self.safe_gather(key, topk_index, S1) # B,S1,K,D
        p = torch.matmul(query, key_topk.permute(0, 1, 3, 2).contiguous())# (B, S1, N1, topK)
        p = p * scale
        p = self.deal_invalid_token(B, S1, S2, topKSize, p)
        pSoftmaxRes, p_max, p_sum = self.tsoftmax(p)
        # simplesoftmax
        reduceGShape = [B, S1, N2, G, topKSize]
        pSoftmaxRes = pSoftmaxRes.reshape(reduceGShape)
        pReduceGRes = torch.sum(pSoftmaxRes, dim=3) # (B, S1, 1, topK)
        pReduceGRes = pReduceGRes / G

        # calc S', Y
        # print("==================计算 SY==================")
        query_index = q_index.to(torch.float) # (B, S1, N1, D)
        key_index = k_index.squeeze(2).to(torch.float) # -> [B,S2,D]
        topk_index = sparse_indices.unsqueeze(-1).expand(-1, -1, -1, D_index) # -> [B,S1,K,D]
        key_index_topk = self.safe_gather(key_index, topk_index, S1)
        s = torch.matmul(query_index, key_index_topk.permute(0, 1, 3, 2).contiguous()) # (B, S1, N1, topK)
        # relu res 复用
        reluRes = torch.relu(s)
        weight = weight.unsqueeze(-1).to(torch.float) # (B,S1,N1,1)
        sReluRes = reluRes * weight
        reduceGShape = [B, S1, Nki, G_index, topKSize]
        hideGShape = [B, S1, Nki, topKSize]
        sReduceGRes = torch.sum(sReluRes.reshape(reduceGShape), dim=3).reshape(hideGShape)
        # S2无效行场景 kSize 选择的行大于S2实际的行 末尾赋值-1 把矩阵末尾的数据修改成-inf
        sReduceGRes = self.deal_invalid_token(B, S1, S2, topKSize, sReduceGRes)
        softmax_func = nn.Softmax(dim=3)
        sSoftmaxRes = softmax_func(sReduceGRes) # (B, S1, N2, topK)
        # print("==================计算 KLLoss==================")
        min_value = torch.tensor([1e-8])
        pReduceGResClip = torch.max(pReduceGRes, min_value)
        sSoftmaxResClip = torch.max(sSoftmaxRes, min_value)
        log_p = torch.log(pReduceGResClip)
        log_s = torch.log(sSoftmaxResClip)
        sub_res = log_p - log_s
        mul_res = sub_res * pReduceGRes
        loss = torch.sum(mul_res)

        # print("==================计算 dW, dQ, dK==================")
        q_sub_p = sSoftmaxRes - pReduceGRes # (B, S1, 1, topK)
        # bmm3,4 is simplified to matmul
        d_weight = torch.matmul(q_sub_p, reluRes.permute(0,1,3,2).contiguous()).squeeze(2) # (B,S1,1,K) * (B,S1,N,K)
        # reluGrad (x > 0) = 1; (x <= 0) = 0
        relu_grad = q_sub_p * weight * (reluRes > 0).to(torch.float16).to(torch.float) # (B,S1,1,topK) * (B,S1,N1,1) = (B,S1,N1,topK)
        
        d_query_index = torch.matmul(relu_grad, key_index_topk) # (B, S1, N1, topK) * (B, S1, K, D) = [B S1 N1 D]
        beforeScatterAdd = torch.matmul(relu_grad.permute(0,1,3,2).contiguous(), query_index) # [B,S1,K,D]

        # for B:
        #     for S1:
        #         取K index
        #         add->d_key_index
        # index扩展到[...D]
        # 创建目标张量
        d_key_index_before = torch.zeros(B, S1, S2, D_index, dtype=beforeScatterAdd.dtype, device=beforeScatterAdd.device)
        d_key_index = self.safe_scatter_add(beforeScatterAdd, sparse_indices, d_key_index_before)
        # 如果目标K轴要变成1，则聚合掉K轴
        d_key_index = d_key_index.sum(1, keepdim=True).squeeze(1).unsqueeze(2) # [B, S1, S2, D] -> [B, 1, S2, D] -> [B, S2, 1, D]
        p_max = p_max.squeeze(0)
        p_sum = p_sum.squeeze(0)
        p_max = p_max.squeeze(-1)
        p_sum = p_sum.squeeze(-1)
        s = s.squeeze(0)
        p = p.squeeze(0)
        d_query_index = d_query_index.squeeze(0)
        d_key_index = d_key_index.squeeze(0)
        key_index_topk = key_index_topk.squeeze(0)
        sSoftmaxRes = sSoftmaxRes.squeeze(0)
        sReduceGRes = sReduceGRes.squeeze(0)
        return d_query_index, d_key_index, d_weight, loss, p_max, p_sum, key_index_topk, s, p, reluRes, sReduceGRes, sSoftmaxRes, pReduceGRes



def gen_seq_len(B, T1):
    random_numbers = np.random.rand(B)
    normalized_numbers = random_numbers / random_numbers.sum()
    q_len_tmp = [int(T1 * normalized_numbers[i]) for i in range(B)]
    q_len = [1 if q_len_tmp[i] == 0 else q_len_tmp[i] for i in range(B)]
    # kv_len = [q_len[i] + np.random.randint(1, 8192) for i in range(B)]
    kv_len = [q_len[i] + np.random.randint(0, 1) for i in range(B)]
    return q_len, kv_len

def print_diff(name, tensor_gpu, tensor_npu):
    diff = torch.abs(tensor_npu.float() - tensor_gpu.float())
    max_diff = torch.max(diff)
    mean_diff = torch.mean(diff)
    print(f'{name} max diff: %.9f' % max_diff)
    print(f'{name} mean diff: %.9f' % mean_diff)
    lhs = tensor_npu.float()
    rhs = tensor_gpu.float()
    d = lhs.dim()
    # if d == 2:
    #     count = 0
    #     for i in range(lhs.shape[0]):
    #         for j in range(lhs.shape[1]):
    #             if torch.abs(tensor_npu[i][j] - tensor_gpu[i][j]) / torch.abs(tensor_npu[i][j]) > 0.005:
    #                 count += 1
    #     print('relative diff > 0.005 占比： ', count / (lhs.shape[0] * lhs.shape[1]))
    max_diff_flat_idx = torch.argmax(diff)
    max_diff_idx = torch.unravel_index(max_diff_flat_idx, diff.shape)
    print(f'{name} max diff pos tensor value: %.9f, %.9f' % (tensor_npu[max_diff_idx], tensor_gpu[max_diff_idx]))

    return max_diff, mean_diff

def test(blockCount, blockSize):
    device = torch.device(f'npu:{0}')
    torch.npu.set_device(device)
    dtype = torch.float16

    headNum = 16
    headNumK = 1
    headNumIdx = 8
    headNumKKIdx = 1
    headDimP = 512
    headDimSY = 128
    scale = 1 / (192 ** 0.5)
    layout = "TND"

    # inputs
    path = '/workspace/dsa_debug/dump_content_layer17_rank191_klloss65231468.00.pt'
    tmp = torch.load(path)
    
    if isinstance(tmp, dict):
        print("\nDictionary keys:")
        for key in tmp.keys():
            value = tmp[key]
            print(f"  - '{key}': {type(value).__name__}", end="")
            if torch.is_tensor(value):
                print(f", shape: {value.shape}, dtype: {value.dtype}")
            elif isinstance(value, np.ndarray):
                print(f", shape: {value.shape}, dtype: {value.dtype}")
            else:
                print(f", value: {str(value)[:50]}...")
        
        # 查看具体张量信息
        print("\nTensor details:")
        for key, value in tmp.items():
            if torch.is_tensor(value):
                print(f"{key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
    else:
        # 如果不是字典，直接查看
        print(f"Value: {tmp}")
        if hasattr(tmp, 'shape'):
            print(f"Shape: {tmp.shape}")

    actual_seq_lengths_query = tmp["actual_seq_lengths_query"].numpy()
    actual_seq_lengths_key = tmp["actual_seq_lengths_kv"].numpy()
    print('actual_seq_lengths_query', actual_seq_lengths_query)
    B = len(actual_seq_lengths_query)
    T1 = np.sum(actual_seq_lengths_query)
    T2 = np.sum(actual_seq_lengths_key)
    cu_actual_seq_lengths_query = list(np.cumsum(actual_seq_lengths_query))
    cu_actual_seq_lengths_key = list(np.cumsum(actual_seq_lengths_key))
    print('B, T1, T2, cu_actual_seq_lengths_query, blockCount: ', B, T1, T2, cu_actual_seq_lengths_query, blockCount)

    q = tmp['q_absorb']
    k = tmp['ag_ct_kv']
    topk_index = tmp['topk_indices']
    q_rope = tmp['query_rope']
    k_rope = tmp['key_rope']
    input_max = tmp['softmax_max']
    input_sum = tmp['softmax_sum']

    query_index = tmp['li_index_q']
    key_index = tmp['ag_index_k']
    weights = tmp['li_chunk_w']

    org_loss = tmp['origin_loss']
    org_dq = tmp['dindex_q']
    org_dk = tmp['_dag_index_k']
    org_dw = tmp['dweights']

    op = LightningIndicesKLLoss(q,
                                k,
                                query_index,
                                key_index,
                                weights,
                                topk_index,
                                q_rope,
                                k_rope,
                                cu_actual_seq_lengths_query,
                                cu_actual_seq_lengths_key,
                                scale,
                                layout,
                                blockSize)

    cpu_d_query_index, cpu_d_key_index, cpu_d_weights, cpu_loss, softmax_max, softmax_sum = op.run_unpad()

    mlp_q = q.to(device).to(dtype).contiguous()
    mlp_q_rope = q_rope.to(device).to(dtype).contiguous()
    mlp_k = k.to(device).to(dtype).contiguous()
    mlp_k_rope = k_rope.to(device).to(dtype).contiguous()
    # mlp_topk_index = topk_index.to(device).contiguous()
    mlp_query_index = query_index.to(device).to(dtype).contiguous()
    mlp_key_index = key_index.to(device).to(dtype).contiguous()
    mlp_weights = weights.to(device).to(dtype).contiguous()
    mlp_max = input_max.to(device).contiguous()
    mlp_sum = input_sum.to(device).contiguous()
    mlp_cu_q = torch.tensor(cu_actual_seq_lengths_query).to(torch.int32).to(device).contiguous()
    mlp_cu_k = torch.tensor(cu_actual_seq_lengths_key).to(torch.int32).to(device).contiguous()
    for _ in range(1):
        # topk_index, _ = torch_npu.mlp_lightning_indexer(
        #     mlp_query_index, 
        #     mlp_key_index, 
        #     mlp_weights, 
        #     actual_seq_lengths_query=mlp_cu_q, 
        #     actual_seq_lengths_key=mlp_cu_k, 
        #     block_table=None,
        #     layout_query="TND", 
        #     layout_key="TND", 
        #     sparse_count=2560,
        #     kv_block_len=1, 
        #     q_block_len=1, 
        #     init_num=0, 
        #     local_num=0, 
        #     sparse_mode=3, 
        #     pre_tokens=512*1024,
        #     next_tokens=512*1024, 
        #     return_value=False)

        mlp_topk_index = topk_index.to(device).contiguous()

        real_dq, real_dk, real_dw, real_loss = torch_npu.mlp_sparse_lightning_indexer_grad_kl_loss(
            mlp_q,
            mlp_k,
            mlp_query_index,
            mlp_key_index,
            mlp_weights,
            mlp_topk_index,
            mlp_max,
            mlp_sum,
            mlp_q_rope,
            mlp_k_rope,
            cu_actual_seq_lengths_query,
            cu_actual_seq_lengths_key,
            scale,
            layout,
            3, # sparse_mode
            512*1024,
            512*1024,
            blockSize,
            False
        )
        torch.npu.synchronize()

        print_diff('dq', cpu_d_query_index, real_dq.cpu())
        print('cpu_d_query_index', cpu_d_query_index)
        check_result(cpu_d_query_index.float(), real_dq.float().cpu(), 'dq')
        print('real_dq', real_dq)

        print_diff('dk', cpu_d_key_index, real_dk.cpu())
        check_result(cpu_d_key_index.float(), real_dk.float().cpu(), 'dk')

        # print('dk value', real_dk)
        print_diff('dw', cpu_d_weights.to(dtype), real_dw.cpu())
        check_result(cpu_d_weights.float(), real_dw.float().cpu(), 'dw')

        print_diff('loss', cpu_loss, real_loss.cpu())
        print('loss value', cpu_loss, real_loss)
        
        print('loss value', real_loss)
        print('org loss', org_loss)

        # print_diff('org dq', cpu_d_query_index, org_dq.cpu())
        # # print('cpu_d_query_index', cpu_d_query_index)
        # check_result(cpu_d_query_index.float(), org_dq.float().cpu(), 'dq')
        # # print('real_dq', real_dq)
        # print_diff('org dk', cpu_d_key_index, org_dk.cpu())
        # check_result(cpu_d_key_index.float(), org_dk.float().cpu(), 'dk')

        # # print('dk value', real_dk)
        # print_diff('org dw', cpu_d_weights.to(dtype), org_dw.cpu())
        # check_result(cpu_d_weights.float(), org_dw.float().cpu(), 'dw')

        if (real_loss > 500):
            print("check fail")

if __name__ == "__main__":
    test(2560, 1)
