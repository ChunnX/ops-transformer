import torch
import kernel_gen_ops
import numpy as np

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x, row_idx, expert_idx, active_num):
        x_np = x.to(torch.float32).numpy()
        expert_idx = expert_idx.numpy()
        row_idx = row_idx.numpy()
        num_rows = x_np.shape[0]
        k = expert_idx.shape[-1]
        sort_expert_idx = np.argsort(expert_idx.reshape((-1,)), axis=-1, kind="stable")
        expanded_expert_idx = np.sort(expert_idx.reshape((-1,)), axis=-1, kind="stable")

        expanded_dst_to_src_row = np.take_along_axis(row_idx.reshape((-1,)), sort_expert_idx, axis=-1)
        expanded_row_idx = np.zeros(expanded_dst_to_src_row.shape, dtype=np.int32)
        expanded_row_idx[expanded_dst_to_src_row] = np.arange(expanded_dst_to_src_row.shape[-1], dtype=np.int32)

        active_num = min(active_num, num_rows) * k
        expanded_x = x_np[expanded_dst_to_src_row[:active_num] % num_rows, :]
        return torch.from_numpy(expanded_x).to(x.dtype), torch.from_numpy(expanded_row_idx.astype(np.int32)), torch.from_numpy(expanded_expert_idx)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x, row_idx, expert_idx, active_num):
        # NPU 实现调用
        expanded_x_out, expanded_row_idx_out, expanded_expert_idx_out = kernel_gen_ops.moe_init_routing(
            x, 
            row_idx, 
            expert_idx, 
            active_num
        )
        return expanded_x_out, expanded_row_idx_out, expanded_expert_idx_out