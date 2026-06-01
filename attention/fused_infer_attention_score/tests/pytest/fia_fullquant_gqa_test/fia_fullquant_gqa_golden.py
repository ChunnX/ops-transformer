import numpy as np
import torch
import math
import torch_npu

# ==============================================================================
# 配置区
# ==============================================================================
GRAPH_PATH = 0
DEVICE_ID = 0

B = 1
N_q = 2
N_kv = 2
D = 128

ACTUAL_SEQ_Q = [1]
ACTUAL_SEQ_KV = [254]

# Layout 选择
INPUT_LAYOUT = "TND"  # 支持: TND
Q_SCALE_LAYOUT = "TN"

# PA KV Cache Layout
# BnNBsD: [BlockNum, N, BlockSize, D]
# PA_NZ: fp8=[Bn,N,D//32,Bs,32], Kscale=[Bn,N,Bs//16,D//64,16,2], Vscale=[Bn,N,D//16,Bs//64,16,2]
KV_CACHE_LAYOUT = "BnNBsD"

# Date Range
Q_DATA_RANGE = (-1.0, 1.0)
K_DATA_RANGE = (-5.0, 5.0)
V_DATA_RANGE = (-5.0, 5.0)

ENABLE_PA = True
BLOCK_SIZE = 128
SPARSE_MODE = 0
SCALE_VALUE = None  # None 时使用 1 / sqrt(D)


# Seed
SEED_Q = 54
SEED_K = 3
SEED_V = 4
SEED_BLOCK_TABLE = 1234
FP8_DTYPE = torch.float8_e4m3fn
P_SCALE = 1.0
EPSILON = 1e-20

Q_BLOCK_SIZE = 128
KV_BLOCK_SIZE = 256

# ==============================================================================
# 数据生成
# ==============================================================================
def get_fp8_per_token_head_quant_scale(tensor):
    """
    用于生成 query/key quant scale
    per-token-head quant scale: shape (B, N, S, 1)
    """
    B, N, S, D = tensor.shape
    fp8_e4m3_max = 448.0
    row_max = torch.abs(tensor).max(dim=3, keepdim=True).values  # (B, N, S, D)
    scale = fp8_e4m3_max / row_max
    return scale.view(B, N, S, 1).float()

def get_fp8_per_head_quant_scale(tensor):
    """
    用于生成 value quant scale
    per-head quant scale: shape (1, N, 1, 1)
    """
    B, N, S, D = tensor.shape
    fp8_e4m3_max = 448.0
    head_max = torch.abs(tensor).amax(dim=(0, 2, 3))  # shape (N,)
    scale = fp8_e4m3_max / head_max
    return scale.view(1, N, 1, 1).float()

def quant_fp16_to_fp8(tensor, scale):
    """将 fp16 数据量化为 fp8_e4m3"""
    return tensor.float() * scale

def create_block_table(actual_seq_kv, block_size, seed=SEED_BLOCK_TABLE):
    block_num_per_batch = [math.ceil(int(seq_len) / block_size) for seq_len in actual_seq_kv]
    total_blocks = sum(block_num_per_batch)
    max_blocks = max(block_num_per_batch)
    block_idx_list = np.random.default_rng(seed).permutation(np.arange(total_blocks, dtype=np.int32))
    block_table = np.full((len(actual_seq_kv), max_blocks), -1, dtype=np.int32)
    idx = 0

    for b_index, block_num in enumerate(block_num_per_batch):
        block_table[b_index, :block_num] = block_idx_list[idx:idx + block_num]
        idx += block_num
    return block_table

def bnsd_to_k_cache(k_fp8_bnsd, seq_lens, block_size, block_table, k_scale_fp32_bnsd=None):
    B_dim, N_dim, S_dim, D_dim = k_fp8_bnsd.shape

    # K data to PA BnNBsD
    block_num_per_seq = [math.ceil(s / block_size) for s in seq_lens]
    total_blocks = sum(block_num_per_seq)

    k_pa = torch.zeros((total_blocks, N_dim, block_size + 4, D_dim), dtype=FP8_DTYPE, device=k_fp8_bnsd.device)
    for b in range(B_dim):
        bid_table = block_table[b]
        for blk_idx in range(block_num_per_seq[b]):
            blockid = int(bid_table[blk_idx])
            start_s = blk_idx * block_size
            end_s = min(start_s + block_size, seq_lens[b])
            valid = end_s - start_s
            if valid <= 0:
                continue
            k_pa[blockid, :, :valid, :] = k_fp8_bnsd[b, :, start_s:end_s, :]
    return k_pa

def bns1_to_k_scale_cache_fp32(k_scale_fp32_bnsd, seq_lens, block_size, block_table):
    B_dim, N_dim, S_dim, _ = k_scale_fp32_bnsd.shape
    D_dim = block_size  # 实际 head dim 是 128，这里用于计算 D/4 = 32
    # block_size 应为原始值，不要改变
    block_num_per_seq = [math.ceil(s / block_size) for s in seq_lens]
    total_blocks = sum(block_num_per_seq)

    scale_rows = 4  # 最后4行存放 scale
    # 创建 fp32 缓存，形状 [Bn, N, Bs + 4, D/4]
    cache = torch.zeros((total_blocks, N_dim, block_size + scale_rows, D_dim // 4),
                        dtype=torch.float32, device=k_scale_fp32_bnsd.device)

    for b in range(B_dim):
        for blk_idx in range(block_num_per_seq[b]):
            block_id = int(block_table[b, blk_idx])
            start = blk_idx * block_size
            end = min(start + block_size, seq_lens[b])
            valid = end - start
            if valid <= 0:
                continue
            for head in range(N_dim):
                # 取出该 block 该 head 的有效 scale (长度 valid)
                scales_slice = k_scale_fp32_bnsd[b, head, start:end, 0]  # (valid,)
                # 将 scales_slice 填入最后 4 行的对应位置（按行优先）
                # 首先将最后 4 行展平为一维数组 (4 * (D_dim//4) = D_dim = 128)
                flat_scale = cache[block_id, head, block_size:block_size+scale_rows, :].reshape(-1)
                # 将有效 scale 写入前 valid 个位置
                flat_scale[:valid] = scales_slice
                # 其余位置保持 0（已经是 0）
    return cache

def bnsd_to_v_cache(tensor_bnsd, seq_lens, block_size, block_table):
    device = tensor_bnsd.device
    batch, heads, S, dim = tensor_bnsd.shape
    block_num_per_batch = [math.ceil(int(s) / block_size) for s in seq_lens]
    total_blocks = sum(block_num_per_batch)

    out_cache = torch.zeros((total_blocks, heads, block_size + 4, dim),
                            dtype=torch.uint8, device=device)

    for b in range(batch):
        for blk_idx in range(block_num_per_batch[b]):
            block_id = int(block_table[b, blk_idx].item())
            block_offset = blk_idx * block_size
            valid_len = min(block_size, seq_lens[b] - block_offset)
            if valid_len <= 0:
                continue
            out_cache[block_id, :, :valid_len, :] = tensor_bnsd[b, :, block_offset:block_offset + valid_len, :]
    return out_cache

def generate_data():
    """生成 BNSD FP16 Q/K/V 并做 FP8 量化，使用原始序列长度"""
    max_sq = max(ACTUAL_SEQ_Q)
    max_skv = max(ACTUAL_SEQ_KV)
    print(f"[INFO] max_sq={max_sq}, max_skv={max_skv}")

    # np.random.seed(SEED_Q)
    # q_fp16 = torch.from_numpy(
    #     np.random.uniform(
    #         low=Q_DATA_RANGE[0],
    #         high=Q_DATA_RANGE[1],
    #         size=(B, N_q, max_sq, D),
    #     ).astype(np.float16)
    # )
    # np.random.seed(SEED_K)
    # k_fp16 = torch.from_numpy(
    #     np.random.uniform(
    #         low=K_DATA_RANGE[0],
    #         high=K_DATA_RANGE[1],
    #         size=(B, N_kv, max_skv, D),
    #     ).astype(np.float16)
    # )
    # np.random.seed(SEED_V)
    # v_fp16 = torch.from_numpy(
    #     np.random.uniform(
    #         low=V_DATA_RANGE[0],
    #         high=V_DATA_RANGE[1],
    #         size=(B, N_kv, max_skv, D),
    #     ).astype(np.float16)
    # )

    q_fp16 = torch.ones((B, N_q, max_sq, D), dtype=torch.float16)
    data_np = np.zeros((B, N_kv, max_skv, D), dtype=np.float16)
    for b in range(B):
        for n in range(N_kv):
            for s in range(ACTUAL_SEQ_KV[b]):
                data_np[b, n, s, :] = np.float16(s + 1)
    k_fp16 = torch.from_numpy(data_np)
    v_fp16 = torch.from_numpy(data_np)

    print(f"[INFO] q_fp16={q_fp16.shape}, k_fp16={k_fp16.shape}, v_fp16={v_fp16.shape}")
    np.set_printoptions(threshold=np.inf)
    print("k_fp16 : ", k_fp16)

    quant_scale_q = get_fp8_per_token_head_quant_scale(q_fp16)
    quant_scale_k = get_fp8_per_token_head_quant_scale(k_fp16)
    quant_scale_v = get_fp8_per_head_quant_scale(v_fp16)
    dequant_scale_q = 1 / quant_scale_q
    dequant_scale_k = 1 / quant_scale_k
    dequant_scale_v = 1 / quant_scale_v

    q_fp8 = quant_fp16_to_fp8(q_fp16, quant_scale_q).to(FP8_DTYPE)
    k_fp8 = quant_fp16_to_fp8(k_fp16, quant_scale_k).to(FP8_DTYPE)
    v_fp8 = quant_fp16_to_fp8(v_fp16, quant_scale_v.view(1, -1, 1, 1)).to(FP8_DTYPE)
    print("k_fp8 : ", k_fp8)
    print("dequant_scale_k : ", dequant_scale_k)

    p_scale = torch.tensor([P_SCALE], dtype=torch.float32)

    return (q_fp8, k_fp8, v_fp8, dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale)

# ==============================================================================
# CPU 调用
# ==============================================================================
def get_softmax_scale(scale_value, head_dim):
    if scale_value is not None:
        return float(scale_value)
    return 1.0 / math.sqrt(head_dim)

def torch_broadcast_kv(num_heads, num_kv_heads, tensor):
    if num_heads == num_kv_heads:
        return tensor
    factor = num_heads // num_kv_heads
    return tensor.repeat_interleave(factor, dim=1)

def cpu_fp8_perhead_golden(q_fp8, k_fp8, v_fp8, 
                            deq_q, deq_k, deq_v, p_scale,
                            actual_seq_q, actual_seq_kv, softmax_scale):
    """per-head CPU golden。"""
    q_tensor = q_fp8.to(torch.float32)
    k_tensor = k_fp8.to(torch.float32)
    v_tensor = v_fp8.to(torch.float32)

    if N_q != N_kv:
        k_tensor = torch_broadcast_kv(N_q, N_kv, k_tensor)
        v_tensor = torch_broadcast_kv(N_q, N_kv, v_tensor)
        deq_k = torch_broadcast_kv(N_q, N_kv, deq_k)
        deq_v = torch_broadcast_kv(N_q, N_kv, deq_v)

    batch, heads, q_seq, _ = q_tensor.shape
    v_dim = v_tensor.shape[-1]
    out = torch.zeros((batch, heads, q_seq, v_dim), dtype=torch.float32)
    o_sum = torch.zeros(q_tensor.shape[:-1], dtype=torch.float32)[..., None]
    o_max = torch.ones(q_tensor.shape[:-1], dtype=torch.float32)[..., None] * torch.finfo(torch.float32).min

    q_lens_t = torch.tensor(actual_seq_q, dtype=torch.int32)
    k_lens_t = torch.tensor(actual_seq_kv, dtype=torch.int32)
    q_lens_acl = q_lens_t.view(batch, 1, 1, 1)
    k_lens_acl = k_lens_t.view(batch, 1, 1, 1)

    Sq, Skv = q_tensor.shape[2], k_tensor.shape[2]
    q_range = torch.arange(Sq).view(1, 1, -1, 1)
    k_range = torch.arange(Skv).view(1, 1, 1, -1)
    q_padding_mask = q_range >= q_lens_acl
    k_padding_mask = k_range >= k_lens_acl

    if SPARSE_MODE == 3:
        delta = k_lens_acl - q_lens_acl
        causal_mask = k_range > (q_range + delta)
        mask_global = causal_mask | q_padding_mask | k_padding_mask
    else:
        mask_global = q_padding_mask | k_padding_mask
    mask_q_blocks = list(torch.split(mask_global, Q_BLOCK_SIZE, dim=2))
    mask_blocks = []
    for mask_q_block in mask_q_blocks:
        mask_blocks.append(list(torch.split(mask_q_block, KV_BLOCK_SIZE, dim=3)))

    q_blocks = list(torch.split(q_tensor, Q_BLOCK_SIZE, dim=2))
    k_blocks = list(torch.split(k_tensor, KV_BLOCK_SIZE, dim=2))
    v_blocks = list(torch.split(v_tensor, KV_BLOCK_SIZE, dim=2))
    o_blocks = list(torch.split(out, Q_BLOCK_SIZE, dim=2))
    s_blocks = list(torch.split(o_sum, Q_BLOCK_SIZE, dim=2))
    m_blocks = list(torch.split(o_max, Q_BLOCK_SIZE, dim=2))

    ln_p_scale = torch.tensor([math.log(p_scale)], dtype=torch.float32)
    
    for j, (kj, vj) in enumerate(zip(k_blocks, v_blocks)):
        scale_j = j
        for i, qi in enumerate(q_blocks):
            oi = o_blocks[i]
            si = s_blocks[i]
            mi = m_blocks[i]
            sij = torch.matmul(qi.to(torch.float32), kj.to(torch.float32).permute(0, 1, 3, 2))
            sij = sij * deq_q * deq_k.transpose(-1, -2)
            sij = sij * softmax_scale

            causal_mask = mask_blocks[i][j]
            sij = sij.masked_fill(causal_mask, float('-inf'))
            
            m_block, _ = torch.max(sij, dim=-1, keepdims=True)
            m_block = m_block - ln_p_scale
            mi_new = torch.maximum(m_block, mi)
            pij = torch.exp(sij - mi_new)
            s_block = torch.sum(pij, dim=-1, keepdims=True)
            pij_drop = pij.to(FP8_DTYPE).to(torch.float32)
            pij_v = torch.matmul(pij_drop, vj.to(torch.float32))
            pij_v = pij_v * deq_v
            si_new = torch.exp(mi - mi_new) * si + s_block
            o_blocks[i] = (si * torch.exp(mi - mi_new) * oi + pij_v) / (si_new + EPSILON)
            s_blocks[i] = si_new
            m_blocks[i] = mi_new

    return torch.cat(o_blocks, dim=2)

# ==============================================================================
# Layout 转换
# ==============================================================================
def convert_q_bnsd_to_layout(tensor_bnsd, seq_lens, layout):
    """Q/K/V BNSD → 各种 layout，支持 fp8 tensor"""
    tensor = tensor_bnsd if isinstance(tensor_bnsd, torch.Tensor) else torch.as_tensor(tensor_bnsd)
    B, N, _, D = tensor.shape
    max_org_s = max(seq_lens)
    
    if layout == "BNSD":
        return tensor[:, :, :max_org_s, :].contiguous()
    elif layout == "BSND":
        return tensor[:, :, :max_org_s, :].permute(0, 2, 1, 3).contiguous()
    elif layout == "BSH":
        return tensor[:, :, :max_org_s, :].permute(0, 2, 1, 3).reshape(B, max_org_s, N * D).contiguous()
    elif layout == "TND":
        T = sum(seq_lens)
        result = torch.zeros((T, N, D), dtype=tensor.dtype, device=tensor.device)
        t = 0
        for b in range(B):
            act_s = seq_lens[b]
            for n in range(N):
                result[t:t + act_s, n, :] = tensor[b, n, :act_s, :]
            t += act_s
        return result.contiguous()
    else:
        raise ValueError(f"Unsupported layout: {layout}")

def convert_scale_to_layout(tensor, seq_lens, scale_type):
    if scale_type == "deq_q":
        B, N, _, _ = tensor.shape
        T = sum(seq_lens)
        result = torch.zeros((T, N, 1), dtype=tensor.dtype)
        t = 0
        for b in range(B):
            act_s = seq_lens[b]
            # D 维度上的值都相同，取第一个即可
            for n in range(N):
                result[t:t + act_s, n, 0] = tensor[b, n, :act_s, 0]
            t += act_s
    elif scale_type == "deq_v":
        result = tensor
    return result.contiguous()

def make_accum_seq(seq_lens):
    result = []
    acc = 0
    for s in seq_lens:
        acc += s
        result.append(acc)
    return result
# ==============================================================================
# NPU 调用
# ==============================================================================
def fa_run_npu(q, k, v, mask, actual_seq_q, actual_seq_kv,
                dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale,
                block_table, block_size, q_n, kv_n, softmax_scale, layout, out_dtype):
    device_id = DEVICE_ID
    torch_npu.npu.set_device(int(device_id))
    if GRAPH_PATH == 0:
        print("[NPU] 调用单算子模式...")
        atten_out, lse_out = torch_npu.npu_fused_infer_attention_score_v2(
            q, k, v,
            atten_mask=mask,
            actual_seq_qlen=actual_seq_q,
            actual_seq_kvlen=actual_seq_kv,
            dequant_scale_query=dequant_scale_q,
            dequant_scale_key=dequant_scale_k,
            dequant_scale_value=dequant_scale_v,
            block_table=block_table,
            block_size=block_size,
            num_query_heads=q_n,
            num_key_value_heads=kv_n,
            softmax_scale=softmax_scale,
            input_layout=layout,
            sparse_mode=SPARSE_MODE,
            query_quant_mode=3,
            key_quant_mode=3,
            value_quant_mode=2,
            query_dtype=FP8_DTYPE,
            key_dtype=FP8_DTYPE,
            value_dtype=FP8_DTYPE,
            dequant_scale_query_dtype=torch_npu.float32,
            dequant_scale_key_dtype=torch_npu.float32,
            dequant_scale_value_dtype=torch_npu.float32,
            quant_scale_p=p_scale,
            out_dtype=out_dtype,
        )
        return atten_out, lse_out 

    npu_mode = Network().to("npu:%s" % device_id)
    config = CompilerConfig()
    with torch.no_grad():
        npu_backend = tng.get_npu_backend(compiler_config=config)
        if GRAPH_PATH == 3:
            print("[NPU] 调用静态图...")
            torch._dynamo.reset()
            npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=False)
            atten_out, lse_out = npu_mode(q, k, v, q_rope, k_rope, mask, actual_seq_q, actual_seq_kv,
                dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale, block_table,
                q_n, kv_n, softmax_scale, layout, block_size, sparse_mode, out_dtype)

        elif GRAPH_PATH == 5:
            print("[NPU] 调用动态图...")
            torch._dynamo.reset()
            npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=True)
            atten_out, lse_out = npu_mode(q, k, v, q_rope, k_rope, mask, actual_seq_q, actual_seq_kv,
                dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale, block_table,
                q_n, kv_n, softmax_scale, layout, block_size, sparse_mode, out_dtype)
    
        elif GRAPH_PATH == 6 or GRAPH_PATH == 7:
            if GRAPH_PATH == 7:
                print("[NPU] 调用aclgraph...")
                config.debug.aclgraph.disable_reinplace_inplaceable_ops_pass = True
                config.mode = "reduce-overhead"
            else:
                print("[NPU] 调用tiling下沉...")
                config.experimental_config.tiling_schedule_optimize = True
            npu_mode = torch.compile(npu_mode, fullgraph=True, backend=npu_backend, dynamic=True)
            if q is not None:
                torch._dynamo.mark_static(q)
            if k is not None:
                torch._dynamo.mark_static(k)
            if v is not None:
                torch._dynamo.mark_static(v)
            if q_rope is not None:
                torch._dynamo.mark_static(q_rope)
            if k_rope is not None:
                torch._dynamo.mark_static(k_rope)
            if mask is not None:
                torch._dynamo.mark_static(mask)
            if dequant_scale_q is not None:
                torch._dynamo.mark_static(dequant_scale_q)
            if dequant_scale_k is not None:
                torch._dynamo.mark_static(dequant_scale_k)
            if dequant_scale_v is not None:
                torch._dynamo.mark_static(dequant_scale_v)
            if p_scale is not None:
                torch._dynamo.mark_static(p_scale)
            if block_table is not None:
                torch._dynamo.mark_static(block_table)

            atten_out, lse_out = npu_mode(q, k, v, q_rope, k_rope, mask, actual_seq_q, actual_seq_kv,
                dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale, block_table,
                q_n, kv_n, softmax_scale, layout, block_size, sparse_mode, out_dtype)

        atten_out = atten_out.cpu().detach()
        lse_out = lse_out.cpu().detach()
        return atten_out, lse_out 


def npu_fp8_full_quant(q_fp8, k_fp8, v_fp8,
                 dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale,
                 actual_seq_q, actual_seq_kv):
    softmax_scale = 1.0 / math.sqrt(D)
    accum_seq_q = make_accum_seq(actual_seq_q)

    npu_input_layout = "TND" if ENABLE_PA else INPUT_LAYOUT
    q_npu = convert_q_bnsd_to_layout(q_fp8, actual_seq_q, npu_input_layout).contiguous().view(FP8_DTYPE).npu()
    print(f"[NPU {npu_input_layout}] q={q_npu.shape}")

    deq_q_npu = convert_scale_to_layout(dequant_scale_q, ACTUAL_SEQ_Q, 'deq_q').npu()
    deq_v_npu = convert_scale_to_layout(dequant_scale_v, ACTUAL_SEQ_KV, 'deq_v').npu()
    p_scale_npu = p_scale.npu()

    out_dtype = torch.float16

    if ENABLE_PA:
        block_table = create_block_table(ACTUAL_SEQ_KV, BLOCK_SIZE)

        k_pa = bnsd_to_k_cache(k_fp8, ACTUAL_SEQ_KV, BLOCK_SIZE, block_table, dequant_scale_k)
        v_pa = bnsd_to_v_cache(v_fp8, ACTUAL_SEQ_KV, BLOCK_SIZE, block_table)
        k_npu = k_pa.contiguous().view(FP8_DTYPE).npu()
        v_npu = v_pa.contiguous().view(FP8_DTYPE).npu()
        
        deq_k_npu = bns1_to_k_scale_cache_fp32(dequant_scale_k, actual_seq_kv, BLOCK_SIZE, block_table).npu()
        block_table_npu = block_table.npu() if isinstance(block_table, torch.Tensor) else torch.as_tensor(block_table, dtype=torch.int32).npu()

        print(f"[NPU PA] kv_layout={KV_CACHE_LAYOUT}")
        print(f"[NPU PA] k_npu shape: {k_npu.shape}, v_npu shape: {v_npu.shape}")
        print(f"[INFO] deq_q_npu shape: {deq_q_npu.shape}, deq_k_npu shape: {deq_k_npu.shape}, deq_v_npu shape: {deq_v_npu.shape}")
        torch.set_printoptions(threshold=np.inf)
        print("k_npu: ", k_npu)
        print("deq_k_npu: ", deq_k_npu)
         # SPARSE_MODE=0时不传mask，其他模式传 causal mask
        if SPARSE_MODE == 0:
            mask_arg = None
        else:
            mask_arg = torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1).npu()

        print("[NPU] 调用 PA 模式...")
        output = fa_run_npu(q_npu, k_npu, v_npu, mask_arg, accum_seq_q, actual_seq_kv,
            deq_q_npu, deq_k_npu, deq_v_npu, p_scale_npu, block_table_npu, BLOCK_SIZE, N_q, N_kv, softmax_scale, npu_input_layout, out_dtype)
    else:
        raise NotImplementedError("当前仅支持 PA 模式")

    atten_out = output[0]
    T_actual = sum(actual_seq_q)
    if atten_out.shape[0] > T_actual:
        atten_out = atten_out[:T_actual]
    print(f"[NPU] atten_out = {atten_out.shape}")
    return output


if __name__ == '__main__':
    print("=" * 60)
    print("FIA Full Quant GQA GOLDEN")
    print("输出: 逐元素表格 + 统计汇总 (PctRlt 通过率)")
    print("=" * 60)
    print(f"场景: {'PA' if ENABLE_PA else 'TND'}")
    print(f"INPUT_LAYOUT={INPUT_LAYOUT}, Q_SCALE_LAYOUT={Q_SCALE_LAYOUT}")
    print(f"KV_CACHE_LAYOUT={KV_CACHE_LAYOUT}")
    print(f"B={B}, N_q={N_q}, N_kv={N_kv}, D={D}")
    print(f"ACTUAL_SEQ_Q={ACTUAL_SEQ_Q}, ACTUAL_SEQ_KV={ACTUAL_SEQ_KV}")

    print("\n[Step 1] 数据生成")
    (q_fp8, k_fp8, v_fp8, dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale) = generate_data()

    print("\n[Step 2] CPU Golden")
    cpu_out = cpu_fp8_perhead_golden(q_fp8, k_fp8, v_fp8,
                                     dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale,
                                     ACTUAL_SEQ_Q, ACTUAL_SEQ_KV, get_softmax_scale(SCALE_VALUE, D))
    print("cpu_out: ", cpu_out)
    print("cpu_out shape", cpu_out.shape)

    print("\n[Step 3] NPU 调用")
    npu_out = npu_fp8_full_quant(q_fp8, k_fp8, v_fp8,
                           dequant_scale_q, dequant_scale_k, dequant_scale_v, p_scale,
                           ACTUAL_SEQ_Q, ACTUAL_SEQ_KV)
    # print("npu_out: ", npu_out[0])

    # print("\n[Step 4] 精度对比")
    # cpu_tnd_torch = convert_q_bnsd_to_layout(cpu_out, ACTUAL_SEQ_Q, "TND")
    
    print("#" * 60)

