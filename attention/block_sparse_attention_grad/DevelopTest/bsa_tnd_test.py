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

seed=42
torch.manual_seed(seed)
device = "npu:7"
DTYPE = torch.float16
torch.npu.empty_cache()

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
    softmax_log_max_sum = torch.zeros((t_q, num_heads, 1), dtype=torch.float32, device=curr_device)

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
        softmax_sum = torch.from_numpy(np.sum(softmax_exp.cpu().numpy(), -1, keepdims=True)).to(curr_device)
        softmax_out = softmax_exp / softmax_sum

        attn_out_b = torch.matmul(softmax_out, v_b)

        lse_b = softmax_max + torch.log(softmax_sum)

        attn_out_b = attn_out_b.permute(1, 0, 2)
        lse_b = lse_b.permute(1, 0, 2)

        attention_out[q_start:q_end] = attn_out_b
        softmax_log_max_sum[q_start:q_end] = lse_b

    return softmax_log_max_sum, attention_out

def generate_base_sparsity_mask(max_seqlen_q, max_seqlen_k, m_block_dim, n_block_dim, batch_size, num_blocksparse_heads, sparsity_list, causal=False, device="cuda"):
    assert len(sparsity_list) == num_blocksparse_heads
    def round_to_multiple(x, base):
        return ((x + base - 1) // base)
    
    nrow, ncol = round_to_multiple(max_seqlen_q, m_block_dim), round_to_multiple(max_seqlen_k, n_block_dim)
    base_mask = torch.zeros(batch_size, num_blocksparse_heads, nrow, ncol, dtype=torch.uint8)
    # print("generate_base_sparsity_mask ========")
    # print(f"base_mask shape = {base_mask.shape}")
    # print(f"sparsity_list = {sparsity_list}")
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
    print(sqlen_mum)
    for i in range(batch - 1, -1, -1):
        sqlen_mum[i] = sqlen_idx[i + 1] - sqlen_idx[i]
    print(sqlen_mum)
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

    t_q, head_num, head_dim = query.shape
    t_kv, _, _ = key.shape
    batch_size = len(accumulate_seq_length) - 1
    query = query.to(torch.float)
    key = key.to(torch.float)
    value = value.to(torch.float)
    dout = dout.to(torch.float) # O
    out = out.to(torch.float) # dO
    # softmaxLse = softmaxLse.to(torch.float)

    print("bsa_mha_bnsd_cpu =====================")
    print(f"query shape : {query.shape}")
    print(f"key shape : {key.shape}")
    print(f"value shape : {value.shape}")
    print(f"dout shape : {dout.shape}")
    print(f"out shape : {out.shape}")
    print(f"softmaxLse shape : {softmaxLse.shape}")
    print(f"blockSparseMask shape : {blockSparseMask.shape}")
    print(f"blockShape : {blockShape}")
    print(f"batch_size : {batch_size}")
    print(f"head_num : {head_num}")
    print(f"accumulate_seq_length : {accumulate_seq_length}")
    print(f"accumulate_seq_lengths_kv : {accumulate_seq_lengths_kv}")
    print(f"max_seqlen_q : {max_seqlen_q}")
    print(f"max_seqlen_k : {max_seqlen_k}")

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
        batch_size, head_num, seq_q, head_dim = query.shape
        _, _, max_seq_kv, _ = key.shape

        # L (b n q_s, 1)
        print("softmaxLse shape: ", softmaxLse.shape)
        print("query.shape shape: ", query.shape)
        softmax_log_max_sum = softmaxLse.view((batch_size, head_num, seq_q, 1)).to(torch.float)
        print(f"L shape : {softmax_log_max_sum.shape}")

        # D (b n q_s, 1)
        softmax_out_sum = (dout.detach().to(torch.float) * out.detach().to(torch.float)).sum(-1, keepdim=True).reshape(
            (batch_size, head_num, seq_q, 1))
        print(f"softmax_out_sum D shape : {softmax_out_sum.shape}")
        # print(softmax_out_sum.flatten()[:10])


        # S (b, n, q_s, k_s)
        S = q_k_mad_bnsd(query, key, blockSparseMask, blockShape, batch_idx)
        S = S * scaleValue
        print(f"S shape : {S.shape}")
        # print(S.flatten()[:10])

        # dP (b n q_s, v_s)
        # dP = torch.einsum("bnqd, bndv->bnqv", out, torch.transpose(value, 2, 3))
        dP = torch_batch_dot(dout, torch.transpose(value, 2, 3)).to(torch.float)
        print(f"dP shape : {dP.shape}")
        # print(dP.flatten()[:10])
        # p (b, n, q_s, k_s)
        P = torch.exp(S - softmax_log_max_sum).to(torch.float)
        print("P")
        # print(P.flatten()[:10])
        # dS (b n q_s, kv_s)
        ds = (dP - softmax_out_sum) * P * scaleValue
        print("ds")
        # print(ds.flatten()[:10])
        # (b n k v)
        # value_grad = torch.einsum("bnkq, bnqd->bnkd", torch.transpose(P, 2, 3), out)
        # key_grad = torch.einsum("bnkq, bnqd->bnkd", torch.transpose(ds, 2, 3), out) * scaleValue
        # query_grad = torch.einsum("bnqk, bnkd->bnqd", ds, key)
        print(f"P shape : {P.shape}")
        print(f"dout shape : {dout.shape}")
        value_grad = torch_batch_dot(torch.transpose(P, 2, 3).to(torch.float), dout)
        key_grad = torch_batch_dot(torch.transpose(ds, 2, 3).to(torch.float), query)
        query_grad = torch_batch_dot(ds.to(torch.float), key)

        return query_grad, key_grad, value_grad

    # tnd
    dq = torch.zeros_like(query)
    dk = torch.zeros_like(key)
    dv = torch.zeros_like(value)


    print("for tnd")
    for i in range(batch_size):
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

        print(f"dq_t shape : {dq_t.shape}")
        print(f"dk_t shape : {dk_t.shape}")
        print(f"dv_t shape : {dv_t.shape}")
        # bsnd b为1
        dq_t = dq_t.permute(0, 2, 1, 3)
        dk_t = dk_t.permute(0, 2, 1, 3)
        dv_t = dv_t.permute(0, 2, 1, 3)
        print("transpose after")
        print(f"dq_t shape : {dq_t.shape}")
        print(f"dk_t shape : {dk_t.shape}")
        print(f"dv_t shape : {dv_t.shape}")

        print(f"dq shape : {dq.shape}")
        print(f"dk shape : {dk.shape}")
        print(f"dv shape : {dv.shape}")   
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
    seqlen_q_list = [random.randint(1, max_seqlen_q) for _ in range(batch_size)]
    seqlen_kv_list = [random.randint(1, max_seqlen_kv) for _ in range(batch_size)]
    
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
    ) = generate_seqlen(S1, S2, B)
    params.append(total_seqlen_q)
    params.append(total_seqlen_kv)

    # 生成 Q, K, V dout (TND 格式)
    query = torch.randn(total_seqlen_q, N, D, dtype=DTYPE)
    key = torch.randn(total_seqlen_kv, num_kv_heads, D,  dtype=DTYPE)
    value = torch.randn(total_seqlen_kv, num_kv_heads, D, dtype=DTYPE)
    d_out = torch.randn(total_seqlen_q, N, D, dtype=DTYPE)
    
    # 生成mask
    num_blocksparse_heads = N
    sparsity_list = [sparsity] * num_blocksparse_heads
    block_sparse_mask = generate_base_sparsity_mask(S1, S2, block_shape[0], block_shape[1], B, num_blocksparse_heads, sparsity_list)
    #softmax_lse_cpu, attention_out_cpu = attention(query, key, value, scale_value, accumulate_seq_length, accumulate_seq_length_kv)
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
    )


if __name__ == "__main__":
    # B, N1, S1_max, S2_max, D，num_kv_heads sparsity mha_type format_type block shape
    blokc_shape = [(128, 128)]
    sparsity = [1]
    mha_type = "mha"
    format_type = "tnd"
    params = [
            [25, 6, 24, 4, 128, 6, 1, "mha", "tnd"]
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
                ) = gen_data(params[i], blokc_shape[k])
                print("B, N1, S1_max, S2_max, D, num_kv_heads sparsity mha_type format_type T_Q, T_KV, block shape")
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
                # torch.npu.synchronize()
                # for j in range(10):
                #     print("遍历次数 : ", j)
                #     d_query, d_key, d_value = test_pre(
                #         query,
                #         key,
                #         value,
                #         d_out,
                #         block_sparse_mask,
                #         attention_out_cpu,
                #         softmax_lse_cpu,
                #         block_shape,
                #         num_kv_heads,
                #         scale_value,
                #         S1,
                #         S2,
                #     )
                #     torch.npu.synchronize()
                #     compare_golden(cpu_dq, cpu_dk, cpu_dv, d_query, d_key, d_value)