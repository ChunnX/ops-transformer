from typing import Any


import torch
import numpy
import kernel_gen_ops

def adapter_capacity(sorted_row_idx, sorted_expert_idx, capacity):
    count = 0
    last = sorted_expert_idx[0]
    for i, val in enumerate(sorted_expert_idx):
        if val != last:
            count = 1
            last = val
        else:
            count += 1
            if count > capacity:
                sorted_expert_idx[i] = -1
                sorted_row_idx[i] = -1

# CPU 参考实现
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x, expert_idx, active_num, expert_capacity, expert_num, drop_pad_mode,
                expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag):
        x_dtype = x.dtype
        x = x.to(torch.float32).numpy()
        expert_idx = expert_idx.numpy()
        num_rows = x.shape[0]
        hidden_size = x.shape[-1]
        k = expert_idx.shape[-1]
        sorted_row_idx = numpy.argsort(expert_idx.reshape((-1,)), axis=-1, kind="stable")
        sorted_expert_idx = numpy.sort(expert_idx.reshape((-1,)), axis=-1)
        if drop_pad_mode == 1 and expert_num <= 0:
            return

        expert_tokens_count_or_cumsum = None
        expert_tokens_before_capacity = None
        expert_idx_hist, bins = numpy.histogram(sorted_expert_idx, bins=expert_num, range=(0, expert_num - 1))
        expert_token_idx = numpy.cumsum(expert_idx_hist)
        if drop_pad_mode == 1 and expert_tokens_before_capacity_flag:
            expert_tokens_before_capacity = expert_idx_hist.astype("int32")
        if drop_pad_mode == 0 and expert_tokens_count_or_cumsum_flag == 1:
            expert_tokens_count_or_cumsum = expert_token_idx.astype("int32")
        elif drop_pad_mode == 0 and expert_tokens_count_or_cumsum_flag == 2:
            expert_tokens_count_or_cumsum = expert_idx_hist.astype("int32")
        
        if drop_pad_mode == 0:
            expanded_row_idx = numpy.zeros(sorted_row_idx.shape, dtype=numpy.int32)
            expanded_row_idx[sorted_row_idx] = numpy.arange(sorted_row_idx.shape[-1], dtype=numpy.int32)

            if active_num == 0:
                active_num = num_rows * k
            else:
                active_num = min(active_num, num_rows * k)
            expanded_x = x[sorted_row_idx[:active_num] // k, :]
        else:
            adapter_capacity(sorted_row_idx, sorted_expert_idx, expert_capacity)
            sorted_row_tmp = numpy.full((expert_num * expert_capacity), -1, dtype=int)
            offset = 0
            lastExpertId = 0
            for i, val in enumerate(sorted_row_idx):
                if val != -1:
                    if lastExpertId != sorted_expert_idx[i]:
                        offset = 0
                        lastExpertId = sorted_expert_idx[i]
                    sorted_row_tmp[sorted_expert_idx[i] * expert_capacity + offset] = sorted_row_idx[i]
                    offset = offset + 1

            expanded_row_idx = numpy.full(sorted_row_idx.shape, -1)
            for i, val in enumerate(sorted_row_tmp):
                if val != -1:
                    expanded_row_idx[val] = i
            
            expanded_x = numpy.full((expert_num * expert_capacity, hidden_size), 0, dtype=x.dtype)
            for i, val in enumerate(sorted_row_tmp):
                if val != -1:
                    expanded_x[i] = x[val // k]
            expanded_x = expanded_x.reshape(expert_num, expert_capacity, hidden_size)
        
        if expert_tokens_count_or_cumsum is not None:
            expert_tokens_count_or_cumsum = torch.from_numpy(expert_tokens_count_or_cumsum)
        
        return torch.from_numpy(expanded_x).to(x_dtype), torch.from_numpy(expanded_row_idx.astype("int32")), expert_tokens_count_or_cumsum, torch.from_numpy(expert_tokens_before_capacity)


# NPU 调用实现
class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x, expert_idx, active_num, expert_capacity, expert_num, drop_pad_mode,
                expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag):
        output = kernel_gen_ops.moe_init_routing_v2(
            x, expert_idx, active_num, expert_capacity, expert_num, drop_pad_mode,
            expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag
        )
        return output