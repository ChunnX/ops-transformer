import torch
import torch_npu
import numpy as np
import custom_ops
import torch.nn as nn
import math

def get_seq_used_by_batch(batch_idx, S, seqused, cu_seqlens):
    if seqused is not None:
        return seqused[batch_idx]
    else:
        if cu_seqlens is not None:
            return cu_seqlens[batch_idx + 1] - cu_seqlens[batch_idx]
        else:
            return S

data_type = torch.bfloat16
hidden_size = 4096
rope_head_dim = 64
norm_eps = 1e-6
coff = 1 # 1:no overlap 2:overlap
cmp_ratio = 128
rotary_mode = 2
cache_mode = 1
head_dim = 512
cu_seqlens = [0, 1]
# ------------- 
B = 1
S = 1
S_max = 0
block_size = 128
start_pos = [8191] * B # (B,)
start_p=8191
seqused = None # (B,), None时cu_seqlens的数据全部参与计算，否则按传参实际值计算

# BS是否合轴
bs_combine_flag = True
update_flag = 1
save_state_seqlens = None
if seqused is not None:
    seqused = torch.tensor(seqused).to(torch.int32)
if start_pos is not None:
    start_pos = torch.tensor(start_pos).to(torch.int32)
else:
    start_pos = torch.full((B,), start_p, dtype=torch.int32)

if bs_combine_flag:
    if cu_seqlens is None:
        T = B * S
        if T !=0:
            cu_seqlens = torch.arange(0, T + 1, S, dtype=torch.int32)
        else:
            cu_seqlens = torch.zeros((B+1), dtype=torch.int32)
    else:
        cu_seqlens = torch.tensor(cu_seqlens).to(torch.int32)
    for i in range(B):
        if start_pos[i] + cu_seqlens[i + 1] - cu_seqlens[i] > S_max:
            S_max = start_pos[i] + cu_seqlens[i + 1] - cu_seqlens[i] 
else:
    cu_seqlens = None
    S_max = max(start_pos) + S
### ======================== gen input data start =============================
# page state
if cache_mode == 1:
    max_block_num_per_batch = (S_max + block_size - 1) // block_size
    block_num = B * max_block_num_per_batch
    next_block_id = 1
    print(f"max_block_num_per_batch: {max_block_num_per_batch}")
    block_table = torch.zeros(size=(B, max_block_num_per_batch), dtype=torch.int32)
    for i in range(B):
        # 需要读取state的范围
        cur_start = start_pos[i] // cmp_ratio * cmp_ratio - cmp_ratio
        cur_end = start_pos[i] // cmp_ratio * cmp_ratio + cmp_ratio
        if start_pos[i] % cmp_ratio == 0:
            cur_end = start_pos[i]
        cur_end = min(cur_end, start_pos[i] + S)
        cur_start_block_id = (cur_start // block_size) if cur_start >= 0 else 0
        cur_end_block_id = (cur_end - 1) // block_size
        for j in range(cur_start_block_id, cur_end_block_id + 1):
            block_table[i][j] = next_block_id
            next_block_id = next_block_id + 1
        # 需要写入state的范围
        end_pos = get_seq_used_by_batch(i, S, seqused, cu_seqlens)
        if save_state_seqlens is not None:
            next_start = start_pos[i] + end_pos - save_state_seqlens[i]
            next_end = start_pos[i] + end_pos
        else:
            next_start = (start_pos[i] + end_pos) // cmp_ratio * cmp_ratio - cmp_ratio
            next_end = (start_pos[i] + end_pos) // cmp_ratio * cmp_ratio + cmp_ratio
            if (start_pos[i] + end_pos) % cmp_ratio == 0:
                next_end = start_pos[i] + end_pos
        next_end = min(next_end, start_pos[i] + end_pos)
        next_start_block_id = (next_start // block_size) if next_start >= 0 else 0
        next_end_block_id = (next_end - 1) // block_size
        for j in range(next_start_block_id, next_end_block_id + 1):
            if block_table[i][j] == 0:
                block_table[i][j] = next_block_id
                next_block_id = next_block_id + 1

    if B==0:
        kv_state = torch.tensor(np.random.uniform(-10, 10, (0, block_size, coff * head_dim))).to(torch.float32)
        score_state = torch.tensor(np.random.uniform(-10, 10, (0, block_size, coff * head_dim))).to(torch.float32)
    else:
        kv_state = torch.tensor(np.random.uniform(-10, 10, (torch.max(block_table) + 1, block_size, coff * head_dim))).to(torch.float32)
        score_state = torch.tensor(np.random.uniform(-10, 10, (torch.max(block_table) + 1, block_size, coff * head_dim))).to(torch.float32)

# other input
if bs_combine_flag:
    x_shape = (cu_seqlens[-1], hidden_size)
    rope_sin_shape = (min(x_shape[0], x_shape[0] // cmp_ratio + B), rope_head_dim)
    rope_cos_shape = rope_sin_shape
else:
    x_shape = (B, S, hidden_size)
    rope_sin_shape = (B, (S + cmp_ratio - 1) // cmp_ratio, rope_head_dim)
    rope_cos_shape = rope_sin_shape

x = torch.tensor(np.random.uniform(-10.0, 10.0, x_shape)).to(data_type).npu()
wkv = torch.tensor(np.random.uniform(-10, 10, (coff * head_dim, hidden_size))).to(data_type).npu()
wgate = torch.tensor(np.random.uniform(-10, 10, (coff * head_dim, hidden_size))).to(data_type).npu()
ape = torch.tensor(np.random.uniform(-10, 10, (cmp_ratio, coff * head_dim))).to(torch.float32).npu()
norm_weight = torch.tensor(np.random.uniform(-10, 10, (head_dim))).to(data_type).npu()
rope_sin = torch.tensor(np.random.uniform(-1, 1, rope_sin_shape)).to(data_type).npu()
rope_cos = torch.tensor(np.random.uniform(-1, 1, rope_cos_shape)).to(data_type).npu()
if cache_mode == 1:  # 连续buffer
    state_cache = torch.zeros((kv_state.shape[0], kv_state.shape[1], 2*kv_state.shape[2]))
    state_cache = state_cache.npu()
    state_cache[:, :, :state_cache.shape[2]//2] = kv_state.clone()
    state_cache[:, :, state_cache.shape[2]//2:] = score_state.clone()
block_table = block_table.npu()
start_pos = torch.tensor(start_pos).to(torch.int32).npu()
if cu_seqlens is not None:
    cu_seqlens = torch.tensor(cu_seqlens).to(torch.int32).npu()
if seqused is not None:
    seqused = torch.tensor(seqused).to(torch.int32).npu()

cmp_kv = (
    torch.ops.custom.compressor(
        x,
        wkv,
        wgate,
        state_cache,
        ape,
        norm_weight, 
        rope_sin,
        rope_cos,
        state_block_table = block_table,
        cu_seqlens = cu_seqlens,
        seqused = seqused,
        start_pos = start_pos,
        rope_head_dim = rope_head_dim,
        cmp_ratio = cmp_ratio,
        coff = coff,
        norm_eps = norm_eps,
        rotary_mode = rotary_mode,
        cache_mode = cache_mode
    )
)