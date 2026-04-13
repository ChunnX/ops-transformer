import torch
import kernel_gen_ops
import numpy as np

def adapter_capacity(sorted_row_idx, sorted_expert_idx, capacity):
    count = 0
    last = sorted_expert_idx[0]
    for i, val in enumerate(sorted_expert_idx):
        if last != val:
            count = 1
            last = val
        else:
            count += 1
            if count > capacity:
                sorted_expert_idx[i] = -1
                sorted_row_idx[i] = -1

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x, expert_idx, scale_optional, offset_optional, active_num, expert_capacity, expert_num, drop_pad_mode, expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag, quant_mode):
        input_x = x.to(torch.float32).numpy()
        expert_idx = expert_idx.numpy()
        scale = scale_optional.numpy()
        offset_t = offset_optional.numpy()

        num_rows = input_x.shape[0]
        hidden_size = input_x.shape[-1]
        k = expert_idx.shape[-1]
        # keyValueSort(flatten(expertIdx)): 按专家 ID 排序；sorted_row_idx[i] 为排序后第 i 个在原 flatten 中的下标
        flat_expert = expert_idx.reshape(-1)
        sorted_row_idx = np.argsort(flat_expert, kind="stable")
        sorted_expert_idx = flat_expert[sorted_row_idx].astype(np.int32, copy=True)
        if drop_pad_mode == 1 and expert_num <= 0:
            return
        expert_tokens_count_or_cumsum = None
        expert_tokens_before_capacity = None
        # 每个专家 token 计数（离散 expert id，避免 histogram 的 range/bins 与整数 bin 边界歧义）
        expert_idx_hist = np.bincount(sorted_expert_idx, minlength=expert_num).astype(np.int64)
        expert_token_idx = np.cumsum(expert_idx_hist)
        if drop_pad_mode == 1 and expert_tokens_before_capacity_flag:
            expert_tokens_before_capacity = expert_idx_hist.astype("int32")
        if drop_pad_mode == 0 and expert_tokens_count_or_cumsum_flag == 1:
            expert_tokens_count_or_cumsum = expert_token_idx.astype("int32")
        elif drop_pad_mode == 0 and expert_tokens_count_or_cumsum_flag == 2:
            expert_tokens_count_or_cumsum = expert_idx_hist.astype("int32")
        
        # Drop/Pad 下仅部分槽位写入 token；与 NPU 一致，未使用槽位保持 0，不对 padding 做 quant
        slot_filled = None
        if drop_pad_mode == 0:
            expanded_row_idx = np.zeros(sorted_row_idx.shape, dtype=np.int32)
            expanded_row_idx[sorted_row_idx] = np.arange(sorted_row_idx.shape[-1], dtype=np.int32)

            if active_num == 0:
                active_num = num_rows * k
            else:
                active_num = min(active_num, num_rows * k)
            expanded_x = input_x[sorted_row_idx[:active_num] // k, :]
        else:
            adapter_capacity(sorted_row_idx, sorted_expert_idx, expert_capacity)
            sort_row_tmp = np.full((expert_num * expert_capacity), -1, dtype=int)
            offset = 0
            lastExpertId = 0
            for i, val in enumerate(sorted_row_idx):
                if val != -1:
                    if lastExpertId != sorted_expert_idx[i]:
                        offset = 0
                        lastExpertId = sorted_expert_idx[i]
                    sort_row_tmp[sorted_expert_idx[i] * expert_capacity + offset] = sorted_row_idx[i]
                    offset += 1
            
            expanded_row_idx = np.full(sorted_row_idx.shape, -1)
            for i, val in enumerate(sort_row_tmp):
                if val != -1:
                    expanded_row_idx[val] = i
            
            expanded_x = np.full((expert_num * expert_capacity, hidden_size), 0, dtype=input_x.dtype)
            slot_filled_flat = np.zeros(expert_num * expert_capacity, dtype=bool)
            for i, val in enumerate(sort_row_tmp):
                if val != -1:
                    expanded_x[i] = input_x[val // k]
                    slot_filled_flat[i] = True
            expanded_x = expanded_x.reshape(expert_num, expert_capacity, hidden_size)
            slot_filled = slot_filled_flat.reshape(expert_num, expert_capacity)
        
        if expert_tokens_count_or_cumsum is None:
            expert_tokens_count_or_cumsum = torch.tensor([])
        else:
            expert_tokens_count_or_cumsum = torch.from_numpy(expert_tokens_count_or_cumsum)
        ds = torch.tensor([])
        if quant_mode == 0:
            expanded_x = np.clip(expanded_x, np.finfo(np.float16).min, np.finfo(np.float16).max)
            expanded_x = expanded_x.astype(np.float16)
            scale_v = np.clip(scale[0], np.finfo(np.float16).min, np.finfo(np.float16).max)
            offset_v = offset_t.astype("float16")
            rr = expanded_x * scale_v + offset_v
            roundd = np.rint(rr)
            roundd = np.clip(roundd, -128, 127)
            roundd = roundd.astype("int8")
            if slot_filled is not None:
                mask = slot_filled[:, :, np.newaxis]
                expanded_x = np.where(mask, roundd, np.int8(0))
            else:
                expanded_x = roundd
        else:
            xf = expanded_x.astype("float32")
            xa = np.abs(xf)
            xm = np.max(xa, axis=-1, keepdims=True)
            ds_arr = xm / 127.0
            q = np.round(np.where(xm > 0, xf / np.maximum(ds_arr, 1e-30), 0.0)).astype("int8")
            if slot_filled is not None:
                mask = slot_filled[:, :, np.newaxis]
                expanded_x = np.where(mask, q, np.int8(0))
            else:
                expanded_x = q
            ds = torch.from_numpy(ds_arr).reshape(-1)
        t_expanded_x = torch.from_numpy(expanded_x)
        if expert_tokens_before_capacity is None:
            expert_tokens_before_capacity = torch.tensor([])
        else:
            expert_tokens_before_capacity = torch.from_numpy(expert_tokens_before_capacity)

        return t_expanded_x, torch.from_numpy(expanded_row_idx.astype("int32")), expert_tokens_count_or_cumsum, expert_tokens_before_capacity, ds

class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x, expert_idx, scale_optional, offset_optional, active_num, expert_capacity, expert_num, drop_pad_mode, expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag, quant_mode):
        # NPU 实现调用
        expanded_x_out, expanded_row_idx_out, expert_tokens_count_or_cumsum, expert_tokens_before_capacity, dynamic_quant_scale_out = kernel_gen_ops.moe_init_routing_quant_v2(
            x, 
            expert_idx, 
            scale_optional, 
            offset_optional, 
            active_num, 
            expert_capacity, 
            expert_num, 
            drop_pad_mode, 
            expert_tokens_count_or_cumsum_flag, 
            expert_tokens_before_capacity_flag, 
            quant_mode
        )
        return expanded_x_out, expanded_row_idx_out, expert_tokens_count_or_cumsum, expert_tokens_before_capacity, dynamic_quant_scale_out