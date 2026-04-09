import torch
import kernel_gen_ops
import numpy as np


def moe_init_routing(x, row_idx, expert_idx, active_num):
    num_rows = x.shape[0]
    hidden_size = x.shape[-1]
    k = expert_idx.shape[-1]
    sort_expert_for_source_row = np.argsort(expert_idx.reshape((-1,)), axis=-1, kind="stable")
    expanded_expert_idx = np.sort(
        expert_idx.reshape((-1,)), axis=-1)
    
    expanded_dst_to_src_row = np.take_along_axis(
        row_idx.reshape((-1,)), sort_expert_for_source_row, axis=-1)
    expanded_row_idx = np.zeros(expanded_dst_to_src_row.shape).astype(np.int32)
    expanded_row_idx[expanded_dst_to_src_row] = np.arange(expanded_dst_to_src_row.shape[-1])
    active_num = min(active_num, num_rows) * k
    expanded_x = x[expanded_dst_to_src_row[:active_num] % num_rows, :]
    return expanded_x, expanded_row_idx, expanded_expert_idx

def quant_a(x, scale, offset):
    sr = x.astype(np.float16) * scale + offset
    roundd = np.rint(sr)
    roundd = np.clip(roundd, -128, 127)
    output = roundd.astype(np.int8)
    return output

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x, row_idx, expert_idx, active_num, scale, offset):
        expanded_x, expanded_row_idx, expanded_expert_idx = moe_init_routing(x.to(torch.float32).numpy(), row_idx.numpy(), expert_idx.numpy(), active_num)
        expanded_x = quant_a(expanded_x, scale, offset)
        return torch.from_numpy(expanded_x), torch.from_numpy(expanded_row_idx.astype(np.int32)), torch.from_numpy(expanded_expert_idx)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x, row_idx, expert_idx, active_num, scale, offset):
        # NPU 实现调用
        expanded_x_out, expanded_row_idx_out, expanded_expert_idx_out = kernel_gen_ops.moe_init_routing_quant(
            x, 
            row_idx, 
            expert_idx, 
            active_num, 
            scale, 
            offset
        )
        return expanded_x_out, expanded_row_idx_out, expanded_expert_idx_out