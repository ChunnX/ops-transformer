import torch
import torch_npu
import math
import numpy as np
from einops import rearrange, repeat
import numpy as np
from array import array
import os
import sys
import random
torch.set_printoptions(precision=6)

seed=128
torch.manual_seed(seed)
device = "npu:7"
DTYPE = torch.float16
torch.npu.empty_cache()

def compera(golden, out):
    golden = golden.cpu().flatten().to(torch.float32)
    # golden = torch.mean(golden, dim=-1)
    out  = out.cpu().flatten().to(torch.float32)
    # out = torch.mean(out, dim=-2)
    # q_t = q_t.flatten().to(torch.float32)
    res = torch.allclose(golden, out, rtol=1e-03, atol=1e-03)
    print(res)

    if res != True:
        abs_diff = torch.abs(out - golden)
        rel_diff = abs_diff / (torch.abs(golden) + 1e-7)

        rel_diff_max = rel_diff.max()
        rel_diff_max_id = rel_diff.argmax().item()

        abs_diff_max = abs_diff.max()
        diff_max_id = abs_diff.argmax().item()

        print("abs diff max id= ", diff_max_id) 
        print("abs diff max = ", abs_diff_max) 
        print("abs diff max = ", abs_diff[diff_max_id]) 
        print("abs diff max : out = ", out[diff_max_id])
        print("abs diff max : golden = ", golden[diff_max_id])

        print("rel diff max id= ", rel_diff_max_id) 
        print("rel diff max = ", rel_diff_max)
        print("rel diff max = ", rel_diff[rel_diff_max_id])
        print("rel diff max : out = ", out[rel_diff_max_id])
        print("rel diff max : golden = ", golden[rel_diff_max_id])

        equal_mask = (torch.isclose(golden, out, rtol=1e-3, atol=1e-3))
        unequal_count = torch.sum(~equal_mask)

        print("diff count:", unequal_count)

        unequal_indices = torch.where(~equal_mask)[0]
        print("diff idxs:", unequal_indices)

def compare_golden(cpu_dq, cpu_dk, cpu_dv, d_query, d_key, d_value):
    print("dq")
    compera(cpu_dq, d_query)

    print("dk")
    compera(cpu_dk, d_key)

    print("dv")
    compera(cpu_dv, d_value)


def attention_ori(query, key, value, scale_value):
    """
    (b, n, s,  d)
    """
    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)

    curr_device = query.device
    batch_size, head_num, seq_size, head_dim = query.shape

    bmm1_res = torch.matmul(query, torch.transpose(key, 2, 3)).to(torch.float32)
    bmm1_res = bmm1_res * scale_value

    softmax_max, _ = torch.max(bmm1_res, dim=-1, keepdim=True)
    softmax_sub = bmm1_res - softmax_max

    # print("attention_ori")
    # print(f"softmax_sub shape = {softmax_sub.shape}")
    softmax_exp = torch.exp(softmax_sub)
    softmax_sum = torch.from_numpy(np.sum(softmax_exp.cpu().numpy(), -1, keepdims=True)).to(curr_device)
    # print(f"softmax_sum shape = {softmax_sum.shape}")
    softmax_out = softmax_exp / softmax_sum
    # print(f"softmax_out shape = {softmax_out.shape}")
    # softmax_out = softmax_out.to(torch.float)
    attention_out = torch.matmul(softmax_out, value)
    # print(f"attention_out shape = {attention_out.shape}")
    softmax_log_max_sum = softmax_max + torch.log(softmax_sum)

    return softmax_log_max_sum, attention_out
    # return softmax_max, softmax_sum, softmax_log_max_sum, attention_out

def attention_tnd(query, key, value, scale_value, cu_seqlens_q, cu_seqlens_k):
    """
    (T, N, D) 格式的 MHA 注意力前向实现
    T: 所有序列拼接后的总 token 数
    N: num_heads
    D: head_dim

    Args:
        query: (T_q, N, D)
        key: (T_k, N, D)  或 (T_k, N_k, D) 对于 GQA/MQA
        value: (T_k, N, D) 或 (T_k, N_k, D)
        scale_value: float, 缩放因子
        cu_seqlens_q: (B+1,) cumulative sequence lengths for query
        cu_seqlens_k: (B+1,) cumulative sequence lengths for key/value

    Returns:
        softmax_log_max_sum: (T_q, N, 1) log-sum-exp 的 log(max) + log(sum) 用于反向
        attention_out: (T_q, N, D) 注意力输出
    """
    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)

    curr_device = query.device
    t_q, num_heads, head_dim = query.shape
    t_k, num_heads_k, _ = key.shape

    batch_size = len(cu_seqlens_q) - 1
    assert len(cu_seqlens_k) - 1 == batch_size, "cu_seqlens_k batch size mismatch"

    attention_out = torch.zeros_like(query)
    softmax_log_max_sum = torch.zeros((t_q, num_heads, 1), dtype=torch.float32)

    for b in range(batch_size):
        q_start, q_end = cu_seqlens_q[b], cu_seqlens_q[b + 1]
        k_start, k_end = cu_seqlens_k[b], cu_seqlens_k[b + 1]

        q_len = q_end - q_start
        k_len = k_end - k_start

        q_b = query[q_start:q_end]
        k_b = key[k_start:k_end]
        v_b = value[k_start:k_end]

        q_b = q_b.permute(1, 0, 2)
        k_b = k_b.permute(1, 0, 2)
        v_b = v_b.permute(1, 0, 2)

        bmm1_res = torch.matmul(q_b, torch.transpose(k_b, 1, 2)).to(torch.float32)
        bmm1_res = bmm1_res * scale_value

        softmax_max, _ = torch.max(bmm1_res, dim=-1, keepdim=True)
        softmax_sub = bmm1_res - softmax_max

        softmax_exp = torch.exp(softmax_sub)
        softmax_sum = torch.from_numpy(np.sum(softmax_exp.cpu().numpy(), -1, keepdims=True))
        softmax_out = softmax_exp / softmax_sum

        attn_out_b = torch.matmul(softmax_out, v_b)

        lse_b = softmax_max + torch.log(softmax_sum)

        attn_out_b = attn_out_b.permute(1, 0, 2)
        lse_b = lse_b.permute(1, 0, 2)

        attention_out[q_start:q_end] = attn_out_b
        softmax_log_max_sum[q_start:q_end] = lse_b

    return softmax_log_max_sum, attention_out.to(DTYPE)

def generate_base_sparsity_mask(max_seqlen_q, max_seqlen_k, m_block_dim, n_block_dim, batch_size, num_blocksparse_heads, sparsity_list, causal=False, device="cuda"):
    assert len(sparsity_list) == num_blocksparse_heads
    def round_to_multiple(x, base):
        return ((x + base - 1) // base)
    
    nrow, ncol = round_to_multiple(max_seqlen_q, m_block_dim), round_to_multiple(max_seqlen_k, n_block_dim)
    base_mask = torch.zeros(batch_size, num_blocksparse_heads, nrow, ncol, dtype=torch.uint8)
    # print("generate_base_sparsity_mask ========")
    # print(f"base_mask shape = {base_mask.shape}")
    # print(f"sparsity_list = {sparsity_list}")
    torch.manual_seed(seed) 
    for batch in range(batch_size):
        for head_rank in range(num_blocksparse_heads):
            sparsity = sparsity_list[head_rank]
            if not sparsity == 0.0 and not sparsity == 1.0:
                for i in range(nrow):
                    idx = nrow - i - 1
                    if causal:
                        available_col_num = max(0, ncol - i)
                        num_one = max(1, int(sparsity * available_col_num))
                        base_mask[batch][head_rank][idx, torch.randperm(available_col_num)[:num_one]] = True
                    else:
                        available_col_num = ncol
                        num_one = max(1, int(sparsity * available_col_num))
                        # print(f"num_one = {num_one}")
                        # print(f"available_col_num = {available_col_num}")
                        # print(f"idx = {idx}")
                        base_mask[batch][head_rank][idx, torch.randperm(available_col_num)[:num_one]] = True
            elif sparsity == 1.0:
                # print("sparsity == 1.0")
                base_mask[batch][head_rank] = torch.ones_like(base_mask[batch][head_rank])
                
    return base_mask

def tailor_mixedmask_for_test(spanded_base_mixedmask, seqlen_q, seqlen_k):
    batch_size = spanded_base_mixedmask.shape[0]
    nheads = spanded_base_mixedmask.shape[1]
    spanded_base_mixedmask = spanded_base_mixedmask[:, :, :seqlen_q, :seqlen_k]
    pad_blockmask = torch.zeros(batch_size, nheads, seqlen_q, seqlen_k, dtype=torch.bool, device = spanded_base_mixedmask.device)
    pad_blockmask[:, :, :spanded_base_mixedmask.shape[2], :spanded_base_mixedmask.shape[3]] = spanded_base_mixedmask
    spanded_base_mixedmask = pad_blockmask
    spanded_base_mixedmask = spanded_base_mixedmask.contiguous()
    return spanded_base_mixedmask

def q_k_mad_bnsd(query, key, blockSparseMask, blockShape, batch_idx=-1, headnum_idx=-1):
    """
    batch_idx=-1  
    headnum_idx=-1 
    (b, n, s,  d)
    """
    # print("q_k_mad_bnsd =====================")
    # print("ker shape: ", key.shape)
    # print("query shape: ", query.shape)
    key_t = torch.transpose(key, 2, 3).contiguous()
    # print("key_t shape: ", query.shape)
    # print("blockSparseMask shape: ", blockSparseMask.shape)
    # print("blockShape: ", blockShape)
    batch_size, num_heads, actual_seqlen_q, _ = query.shape
    actual_seqlen_k = key.shape[2]

    # 获取划分的基本块shape 目前设定为（128,128）
    q_block, kv_block = blockShape[0], blockShape[1]

    # 划分之后， Q 和 k 的块数已经尾块size
    # q_row, k_col = blockSparseMask.shape[-2], blockSparseMask.shape[-1]
    # q_tail_block_size = max_seq_q - (q_row - 1 ) * q_block if q_row * q_block != max_seq_q else 0
    # k_tail_block_size = max_seq_kv - (k_col - 1 ) * kv_block if k_col * kv_block != max_seq_q else 0

    # S = torch.zeros((batch_size, head_num, max_seq_q, max_seq_kv), dtype=query.dtype)
    S = torch.einsum("bnqd, bndk->bnqk", query, key_t)
    # print(f"S.shape: {S.shape}")
    full_mask = repeat(blockSparseMask, "b n s_m s_n -> b n (s_m d_m) (s_n d_n)", d_m=q_block, d_n=kv_block)
    # print(f"full_mask.shape = {full_mask.shape}")
    full_mask = tailor_mixedmask_for_test(full_mask, actual_seqlen_q, actual_seqlen_k)
    # print(f"full_mask.shape = {full_mask.shape}")
    neg_full_mask = ~full_mask # ?
    # print(f"neg_full_mask.shape: {neg_full_mask.shape}")
    # print(f"batch_idx.shape: {batch_idx}")
    # print(f"batch_idx.shape: {headnum_idx}")
    if batch_idx != -1:
        neg_full_mask = neg_full_mask[batch_idx:batch_idx+1, ...]
    if headnum_idx != -1:
        neg_full_mask = neg_full_mask[:, headnum_idx:headnum_idx+1, ...]
    # print(f"neg_full_mask.shape: {neg_full_mask.shape}")
    # S = S.mul(full_mask)
    S = S.masked_fill(neg_full_mask, float('-inf'))
    # print(f"S.shape: {S.shape}")
    # print("q_k_mad_bnsd =====================")
    # print("q_k_mad_bnsd end =====================")
    return S

def torch_batch_dot(a, b):
    batch, n, h, _ = a.shape
    _, _, _, w = b.shape
    a = a.to(torch.float)
    b = b.to(torch.float)
    res = torch.zeros((batch, n, h, w), dtype=torch.float)
    for i in range(batch):
        for j in range(n):
            res[i, j] = torch.matmul(a[i, j], b[i, j])
    return res
    
def convert_sqlen_idx_to_num(sqlen_idx):
    '''
    example : max_sqlen = 117
    sqlen_idx = [  0,  95, 189, 299, 408, 508]
    sqlen_mum = [ 95, 94, 110, 109, 100]
    '''
    batch = len(sqlen_idx) -1
    sqlen_mum = [0] * batch  
    # print(sqlen_mum)
    for i in range(batch - 1, -1, -1):
        sqlen_mum[i] = sqlen_idx[i + 1] - sqlen_idx[i]
    # print(sqlen_mum)
    return sqlen_mum

def bsa_mha_tnd_cpu(
    dout, query, key, value, out, softmaxLse, blockSparseMask, blockShape,
    accumulate_seq_length, accumulate_seq_lengths_kv, scaleValue, max_seqlen_q, max_seqlen_k,
) :
    """
    (T. N,  D)
    dout: (Tq, N, d) attention_in
    query : (Tq, N, d)
    key : (Tkv, N, d)
    value : ((Tkv, N, d)
    out : (Tq, N, d)   dy
    softmaxLse : (Tq, N, d)
    blockShape : (q_block_szie, k_blcok_size) 暂定(128, 128)
    blockSparseMask : (b, n, [q_s / q_block_szie], [kv_s / k_blcok_size] ) 向上取整
    accumulate_seq_length : list 序号
    accumulate_seq_lengths_kv : list 序号
    """
    def round_to_multiple(x, base):
        return ((x + base - 1) // base) * base

    ori_dtype = query.dtype
    t_q, head_num, head_dim = query.shape
    t_kv, _, _ = key.shape
    batch_size = len(accumulate_seq_length) - 1
    # query = query.to(torch.float)
    # key = key.to(torch.float)
    # value = value.to(torch.float)
    # dout = dout.to(torch.float) # O
    # out = out.to(torch.float) # dO
    # softmaxLse = softmaxLse.to(torch.float)

    # print("bsa_mha_bnsd_cpu =====================")
    # print(f"query shape : {query.shape}")
    # print(f"key shape : {key.shape}")
    # print(f"value shape : {value.shape}")
    # print(f"dout shape : {dout.shape}")
    # print(f"out shape : {out.shape}")
    # print(f"softmaxLse shape : {softmaxLse.shape}")
    # print(f"blockSparseMask shape : {blockSparseMask.shape}")
    # print(f"blockShape : {blockShape}")
    # print(f"batch_size : {batch_size}")
    # print(f"head_num : {head_num}")
    # print(f"accumulate_seq_length : {accumulate_seq_length}")
    # print(f"accumulate_seq_lengths_kv : {accumulate_seq_lengths_kv}")
    # print(f"max_seqlen_q : {max_seqlen_q}")
    # print(f"max_seqlen_k : {max_seqlen_k}")

    q_len_nums = convert_sqlen_idx_to_num(accumulate_seq_length)
    kv_len_nums = convert_sqlen_idx_to_num(accumulate_seq_lengths_kv)

    def tnd_to_1nsd(tensor, idx_front, idx_back, nums, head_num, head_dim):
        '''
        针对不等长场景
        针对batch为1的格式转换
        idx_front: s 序号首 切片用
        idx_back: s 序号尾 切片用
        nums: s value
        '''
        return tensor[idx_front:idx_back, ...].reshape(1, nums, head_num, head_dim).permute(0, 2, 1, 3).contiguous()

    def bsa_mha_bnsd_cpu_inner(
        dout, query, key, value, out, softmaxLse, blockSparseMask, blockShape, scaleValue, batch_idx
    ) :                                            
        '''
        batch 1
        '''
        ori_dtype = query.dtype
        query = query.to(torch.float)
        key = key.to(torch.float)
        value = value.to(torch.float)
        dout = dout.to(torch.float) # O
        out = out.to(torch.float) # dO
        softmaxLse = softmaxLse.to(torch.float)

        batch_size, head_num, seq_q, head_dim = query.shape
        _, _, max_seq_kv, _ = key.shape

        # L (b n q_s, 1)
        # print("softmaxLse shape: ", softmaxLse.shape)
        # print("query.shape shape: ", query.shape)
        softmax_log_max_sum = softmaxLse.view((batch_size, head_num, seq_q, 1)).to(torch.float)
        # print(f"L shape : {softmax_log_max_sum.shape}")

        # D (b n q_s, 1)
        softmax_out_sum = (dout.detach().to(torch.float) * out.detach().to(torch.float)).sum(-1, keepdim=True).reshape(
            (batch_size, head_num, seq_q, 1))

        # softmax_out_sum =  torch.full((batch_size, head_num, seq_q, 1), 1.0, dtype=torch.float32)
        # print("dout")
        # for j in range(head_num):
        #     print("j: ", j)
        #     for i in range(10):
        #         print(dout[:, j:j+1, i:i+1, :8])

        # print("out")
        # for j in range(head_num):
        #     print("j: ", j)
        #     for i in range(10):
        #         print(out[:, j:j+1, i:i+1, :8])

        # print("softmax_out_sum")
        # print("softmax_out_sum shape:", softmax_out_sum.shape)
        # for j in range(head_num):
        #     print("j: ", j)
        #     for i in range(10):
        #         print(softmax_out_sum[:, j:j+1, i:i+1, :])
        # print(f"softmax_out_sum D shape : {softmax_out_sum.shape}")
        # print(softmax_out_sum.flatten()[:10])


        # S (b, n, q_s, k_s)
        S = q_k_mad_bnsd(query, key, blockSparseMask, blockShape, batch_idx)

        # print("S")
        # print("s shape:", S.shape)
        # print(S[:, :, :, 0:128])
        # print(S[:, :, :, 128:256])
        # print(S[:, :, :, 256:384])
        # print(S[:, :, :, 384:512])
        # for i in range(10):
        #     print(S[:, :, i:i+1, :])
        S = S * scaleValue

        # print("S * scale")
        # for j in range(head_num):
        #     print("j: ", j)
        #     for i in range(10):
        #         print(S[:, j:j+1, i:i+1, :])

        # dP (b n q_s, v_s)
        # dP = torch.einsum("bnqd, bndv->bnqv", out, torch.transpose(value, 2, 3))
        dP = torch_batch_dot(dout, torch.transpose(value, 2, 3)).to(torch.float)
        # print("dP")
        # for j in range(head_num):
        #     print("j: ", j)
        #     for i in range(10):
        #         print(dP[:, j:j+1, i:i+1, :])
        # print("softmax_log_max_sum")
        # print(softmax_log_max_sum)

        P = torch.exp(S - softmax_log_max_sum).to(ori_dtype)
        t1 = S - softmax_log_max_sum

        # print("L")
        # for j in range(head_num):
        #     print("j: ", j)
        #     for i in range(10):
        #         print(softmax_log_max_sum[:, j:j+1, i:i+1, :])

        # print("S - L")
        # for j in range(head_num):
        #     print("j: ", j)
        #     for i in range(10):
        #         print(t1[:, j:j+1, i:i+1, :])

        # print("P")
        # for j in range(head_num):
        #     print("j: ", j)
        #     for i in range(10):
        #         print(P[:, j:j+1, i:i+1, :])

        # print("P T")
        # for j in range(head_num):
        #     print("j: ", j)
        #     for i in range(10):
        #         print(P.transpose(-2, -1)[:, j:j+1, i:i+1, :])

        # dS (b n q_s, kv_s)
        dS = ((dP - softmax_out_sum) * P.to(torch.float32)).to(ori_dtype)
        # t1 = dP - softmax_out_sum
        # print("dP - softmax_out_sum")
        # for j in range(head_num):
        #     print("j: ", j)
        #     for i in range(10):
        #         print(t1[:, j:j+1, i:i+1, :])

        # print("dS")
        # for j in range(head_num):
        #     print("j: ", j)
        #     for i in range(10):
        #         print(dS[:, j:j+1, i:i+1, :])
        # print(dS.flatten()[:10])
        # (b n k v)
        # value_grad = torch.einsum("bnkq, bnqd->bnkd", torch.transpose(P, 2, 3), out)
        # key_grad = torch.einsum("bnkq, bnqd->bnkd", torch.transpose(dS, 2, 3), out) * scaleValue
        # query_grad = torch.einsum("bnqk, bnkd->bnqd", dS, key)
        # print(f"P shape : {P.shape}")
        # print(f"dout shape : {dout.shape}")
        # print("dout")
        # print(dout[:, :, :, :40])
        value_grad = torch_batch_dot(torch.transpose(P, 2, 3).to(torch.float), dout)

        key_grad = torch_batch_dot(torch.transpose(dS, 2, 3).to(torch.float), query) * scale_value
        query_grad = torch_batch_dot(dS.to(torch.float), key) * scale_value

        # print("value_grad")
        # print("cpu_dv shape: ", value_grad.shape)
        # print("n = 0")
        # for i in range(10):
        #     print("i: ", i)
        #     print(value_grad[:, 0:1, i:i+1, :40])
        # print("n = 1")
        # for i in range(10):
        #     print("i: ", i)
        #     print(value_grad[:, 1:2, i:i+1, :40])

        # print("query_grad")
        # print("cpu_dq shape: ", query_grad.shape)
        # print("n = 0")
        # for i in range(10):
        #     print("i: ", i)
        #     print(query_grad[:, 0:1, i:i+1, :40])
        # print("n = 1")
        # for i in range(10):
        #     print("i: ", i)
        #     print(query_grad[:, 1:2, i:i+1, :40])

        return query_grad.to(ori_dtype), key_grad.to(ori_dtype), value_grad.to(ori_dtype)

    # tnd
    dq = torch.zeros_like(query)
    dk = torch.zeros_like(key)
    dv = torch.zeros_like(value)


    # print("dout tnd")
    # print(dout[:, 0:1, :40])

    # print(dout[:, 1:2, :40])

    # print("dq dtype: ", dq.dtype)
    # print("for tnd")
    for i in range(batch_size):
        # print("batch: ", i)
        q_s_num = q_len_nums[i]
        kv_s_num = kv_len_nums[i]
        q_temp = tnd_to_1nsd(query, accumulate_seq_length[i], accumulate_seq_length[i+1], q_s_num, head_num, head_dim)
        dout_temp = tnd_to_1nsd(dout, accumulate_seq_length[i], accumulate_seq_length[i+1], q_s_num, head_num, head_dim)
        out_temp = tnd_to_1nsd(out, accumulate_seq_length[i], accumulate_seq_length[i+1], q_s_num, head_num, head_dim)
        k_temp = tnd_to_1nsd(key, accumulate_seq_lengths_kv[i], accumulate_seq_lengths_kv[i+1], kv_s_num, head_num, head_dim)
        v_temp = tnd_to_1nsd(value, accumulate_seq_lengths_kv[i], accumulate_seq_lengths_kv[i+1], kv_s_num, head_num, head_dim)
        # lse_tmp = softmaxLse[i, :, 0 : q_s_num] # (t_q, n, 1)
        lse_tmp = tnd_to_1nsd(softmaxLse, accumulate_seq_length[i], accumulate_seq_length[i+1], q_s_num, head_num, 1)

        # bnsd b为1
        dq_t, dk_t, dv_t = bsa_mha_bnsd_cpu_inner(
            dout_temp, q_temp, k_temp, v_temp, out_temp, lse_tmp, blockSparseMask, blockShape, scaleValue, i)

        # print(f"dq_t shape : {dq_t.shape}")
        # print(f"dk_t shape : {dk_t.shape}")
        # print(f"dv_t shape : {dv_t.shape}")
        # bsnd b为1
        dq_t = dq_t.permute(0, 2, 1, 3)
        dk_t = dk_t.permute(0, 2, 1, 3)
        dv_t = dv_t.permute(0, 2, 1, 3)
        # print("transpose after")
        # print(f"dq_t shape : {dq_t.shape}")
        # print(f"dk_t shape : {dk_t.shape}")
        # print(f"dv_t shape : {dv_t.shape}")

        # print(f"dq_t dtype : {dq_t.dtype}")
        # print(f"dk_t dtype : {dk_t.dtype}")
        # print(f"dv_t dtype : {dv_t.dtype}")

        # print(f"dq shape : {dq.shape}")
        # print(f"dk shape : {dk.shape}")
        # print(f"dv shape : {dv.shape}")   
        dq[accumulate_seq_length[i]:accumulate_seq_length[i+1], ...] = dq_t[0:1, ...]
        dk[accumulate_seq_lengths_kv[i]:accumulate_seq_lengths_kv[i+1], ...] = dk_t[0:1, ...]
        dv[accumulate_seq_lengths_kv[i]:accumulate_seq_lengths_kv[i+1], ...] = dv_t[0:1, ...]
    return dq, dk, dv

def gen_golden(
    query,
    key,
    value,
    d_out,
    block_sparse_mask,
    attention_out_cpu,
    softmax_lse_cpu,
    block_shape,
    scale_value,
    S1_max,
    S2_max,
    accumulate_seq_length,
    accumulate_seq_length_kv,
):
    cpu_dq, cpu_dk, cpu_dv = bsa_mha_tnd_cpu(
        d_out,
        query,
        key,
        value,
        attention_out_cpu,
        softmax_lse_cpu,
        block_sparse_mask,
        block_shape,
        accumulate_seq_length,
        accumulate_seq_length_kv,
        scale_value,
        S1_max,
        S2_max,
    )

    return  cpu_dq.to(query.dtype), cpu_dk.to(query.dtype), cpu_dv.to(query.dtype)

def generate_seqlen(
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
    # seqlen_q_list = [random.randint(1, max_seqlen_q) for _ in range(batch_size)]
    # # seqlen_q_list = [121 for _ in range(batch_size)]
    # seqlen_kv_list = [random.randint(1, max_seqlen_kv) for _ in range(batch_size)]
    # seqlen_kv_list = [255 for _ in range(batch_size)]

    # seqlen_q_list = [121, 109, 31, 52, 66, 47, 18, 62, 4, 20]
    # seqlen_kv_list = [164, 129, 154, 35, 58, 16, 149, 191, 63, 226]

    # seqlen_q_list = [121, 109, 31, 52, 66, 47, 18, 62, 4, 20]
    seqlen_q_list = [121, 109, 31, 52, 66, 47, 18, 62, 4, 20]
    seqlen_kv_list = [164, 129, 154, 35, 58, 16, 149, 191, 226, 63]

    actual_q_seq_max = max(seqlen_q_list)
    actual_kv_seq_max = max(seqlen_kv_list)
    # seqlen_q_list = [10 for _ in range(batch_size)]
    # seqlen_kv_list = [10 for _ in range(batch_size)]
    
    # 2. 计算总序列长度
    total_seqlen_q = sum(seqlen_q_list)
    total_seqlen_kv = sum(seqlen_kv_list)
    
    # 3. 生成 accumulate_seq_length 索引列表
    accumulate_seq_length = [0]
    for length in seqlen_q_list:
        accumulate_seq_length.append(accumulate_seq_length[-1] + length)
    
    accumulate_seq_lengths_kv = [0]
    for length in seqlen_kv_list:
        accumulate_seq_lengths_kv.append(accumulate_seq_lengths_kv[-1] + length)
    
    return (
        accumulate_seq_length,
        accumulate_seq_lengths_kv,
        seqlen_q_list,
        seqlen_kv_list,
        total_seqlen_q,
        total_seqlen_kv,
        actual_q_seq_max,
        actual_kv_seq_max
    )

def gen_data(params, block_shape):
    """
    Arguments:
        params: [B, N1, S1_max, S2_max, D, num_kv_heads sparsity mha_type format_type]
        block_shape: (block_shape_x, block_shape_y)
    """
    # B, N1, S1_max, S2_max, D，num_kv_heads sparsity mha_type format_type block shape
    B, N, S1, S2, D = params[0], params[1], params[2], params[3], params[4]
    scale_value = 1.0 / math.sqrt(D)
    # scale_value = 1.0
    sparsity = params[6]
    mha_type = params[7]
    format_type = params[8]
    num_kv_heads = N if mha_type == "mha" else params[5]
    assert N % num_kv_heads == 0

    ceil_q = (S1 + block_shape[0] - 1) // block_shape[0]
    ceil_kv = (S2 + block_shape[1] - 1) // block_shape[1]

    (
        accumulate_seq_length,
        accumulate_seq_length_kv,
        seqlen_q_list,
        seqlen_kv_list,
        total_seqlen_q,
        total_seqlen_kv,
        actual_q_seq_max,
        actual_kv_seq_max
    ) = generate_seqlen(S1, S2, B)
    params.append(total_seqlen_q)
    params.append(total_seqlen_kv)
    params.append(actual_q_seq_max)
    params.append(actual_kv_seq_max)

    # 生成 Q, K, V dout (TND 格式)
    query = torch.randn(total_seqlen_q, N, D, dtype=DTYPE)
    key = torch.randn(total_seqlen_kv, num_kv_heads, D,  dtype=DTYPE)
    value = torch.randn(total_seqlen_kv, num_kv_heads, D, dtype=DTYPE)
    d_out = torch.randn(total_seqlen_q, N, D, dtype=DTYPE)

    # query = torch.ones(total_seqlen_q, N, D, dtype=DTYPE)
    # key = torch.ones(total_seqlen_kv, num_kv_heads, D,  dtype=DTYPE)
    # key = torch.full((total_seqlen_kv, num_kv_heads, D), 2.0, dtype=DTYPE)
    # value = torch.ones(total_seqlen_kv, num_kv_heads, D, dtype=DTYPE)
    # d_out = torch.full((total_seqlen_q, N, D), 0.1, dtype=DTYPE)
    # d_out = torch.tril(torch.ones((B, N, S1, D), dtype=DTYPE)).permute(0, 2, 1, 3).reshape(total_seqlen_q, N, D).contiguous()
    # print(d_out)

    # 生成mask
    num_blocksparse_heads = N
    sparsity_list = [sparsity] * num_blocksparse_heads
    block_sparse_mask = generate_base_sparsity_mask(actual_q_seq_max, actual_kv_seq_max, block_shape[0], block_shape[1], B, num_blocksparse_heads, sparsity_list)
    softmax_lse_cpu, attention_out_cpu = attention_tnd(query, key, value, scale_value, accumulate_seq_length, accumulate_seq_length_kv)

    return (
        query,
        key,
        value,
        d_out,
        block_sparse_mask,
        attention_out_cpu,
        softmax_lse_cpu,
        block_shape,
        num_kv_heads,
        scale_value,
        S1,
        S2,
        accumulate_seq_length,
        accumulate_seq_length_kv,
        seqlen_q_list,
        seqlen_kv_list,
        actual_q_seq_max, 
        actual_kv_seq_max
    )


def excute_bsag(
    query,
    key,
    value,
    d_out,
    block_sparse_mask,
    attention_out_cpu,
    softmax_lse_cpu,
    block_shape,
    num_kv_heads,
    scale_value,
    S1_max,
    S2_max,
    seqlen_q_list,
    seqlen_kv_list,  
):
    query = query.to(device)
    key = key.to(device)
    value = value.to(device)
    d_out = d_out.to(device)
    block_sparse_mask = block_sparse_mask.to(device)
    attention_out = attention_out_cpu.to(device)
    softmax_lse_out = softmax_lse_cpu.to(device)
    # print("query dtye: ", query.dtype)
    # print("key dtye: ", key.dtype)
    # print("value dtye: ", value.dtype)
    # print("attention_out dtye: ", attention_out.dtype)
    # print("d_out dtye: ", d_out.dtype)
    # print("block_sparse_mask dtye: ", block_sparse_mask.dtype)
    # print("softmax_lse_out dtye: ", softmax_lse_out.dtype)
    # print("query dtye: ", query.dtype)
    # print("key dtye: ", key.dtype)
    # print("value dtye: ", value.dtype)

    d_query, d_key, d_value = torch_npu.npu_block_sparse_attention_backward(
        d_out, query, key, value,
        attention_out, softmax_lse_out, block_sparse_mask,
        block_shape=block_shape,
        actual_seq_lengths=seqlen_q_list, actual_seq_lengths_kv=seqlen_kv_list,
        q_input_layout="TND", kv_input_layout="TND",
        num_key_value_heads=num_kv_heads, scale_value=scale_value,
    )

    return d_query, d_key, d_value


if __name__ == "__main__":
    # B, N1, S1_max, S2_max, D，num_kv_heads sparsity mha_type format_type block shape
    blokc_shape = [(128, 128)]
    sparsity = [0.5]
    mha_type = "mha"
    format_type = "tnd"
    params = [
            [10, 15, 126, 256, 128, 1, 0.5, "mha", "tnd"]
    ]
    for si in range(len(sparsity)):
        # for i in range(len(params)):
        #     params[i][-3] = sparsity[si]
        for k in range(len(blokc_shape)):
            for i in range(len(params)):
                (
                    query,
                    key,
                    value,
                    d_out,
                    block_sparse_mask,
                    attention_out_cpu,
                    softmax_lse_cpu,
                    block_shape,
                    num_kv_heads,
                    scale_value,
                    S1_max,
                    S2_max,
                    accumulate_seq_length,
                    accumulate_seq_length_kv,
                    seqlen_q_list,
                    seqlen_kv_list,       
                    actual_q_seq_max, 
                    actual_kv_seq_max
                ) = gen_data(params[i], blokc_shape[k])
                print("B, N1, S1_max, S2_max, D, num_kv_heads sparsity mha_type format_type T_Q, T_KV, actual_q_seq_max, actual_kv_seq_max, block shape")
                print("s1 kv seqlen")
                print(seqlen_q_list)
                print(accumulate_seq_length)
                print("s2 lv sqelen")
                print(seqlen_kv_list)
                print(accumulate_seq_length_kv)
                print("mask shapp: ", block_sparse_mask.shape)
                # print(block_sparse_mask)
                print("=================================")
                print(params[i], blokc_shape[k])
                cpu_dq, cpu_dk, cpu_dv = gen_golden(
                    query,
                    key,
                    value,
                    d_out,
                    block_sparse_mask,
                    attention_out_cpu,
                    softmax_lse_cpu,
                    block_shape,
                    scale_value,
                    S1_max,
                    S2_max,
                    accumulate_seq_length,
                    accumulate_seq_length_kv,
                )
                torch.npu.synchronize()
                torch.npu.empty_cache()
                for j in range(10):
                    print("遍历次数 :::::::::::::::::::::::::::: ", j)
                    d_query, d_key, d_value = excute_bsag(
                        query,
                        key,
                        value,
                        d_out,
                        block_sparse_mask,
                        attention_out_cpu,
                        softmax_lse_cpu,
                        block_shape,
                        num_kv_heads,
                        scale_value,
                        S1_max,
                        S2_max,
                        seqlen_q_list,
                        seqlen_kv_list,  
                    )
                    torch.npu.synchronize()
                    torch.npu.empty_cache()
                    # b, n, s1, s2 = params[0][0], params[0][1], params[0][2], params[0][3]
                    # d_query =  d_query.reshape(b, n, s1, 128).contiguous()
                    # d_key =  d_key.reshape(b, n, s2, 128).contiguous()
                    # d_value =  d_value.reshape(b, n, s2, 128).contiguous()

                    # cpu_dq =  cpu_dq.reshape(b, s1, n, 128).permute(0, 2, 1, 3).contiguous()
                    # cpu_dk =  cpu_dk.reshape(b, s2, n, 128).permute(0, 2, 1, 3).contiguous()
                    # cpu_dv =  cpu_dv.reshape(b, s2, n, 128).permute(0, 2, 1, 3).contiguous()
                    
                    # print(cpu_dq[0, 0, :10])
                    # print(d_query[0, 0, :10])
                    print("compare ressssssssssssss:")
                    compare_golden(cpu_dq, cpu_dk, cpu_dv, d_query, d_key, d_value)