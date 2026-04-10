import torch
import numpy as np
# import tensorflow as tf
import kernel_gen_ops

def moe_init_routing_v2_grad(grad_expanded_x, expanded_row_idx, top_k, drop_pad_mode, active_num):
    # num_rows = grad_expanded_x.shape[0] // top_k
    # hidden_size = grad_expanded_x.shape[-1]

    # grad_x = np.zeros((num_rows, hidden_size), dtype=np.float32)
    # for i in range(num_rows):
    #     for j in range(i * top_k, i * top_k + top_k, 1):
    #         expanded_x_idx = expanded_row_idx[j]
    #         if drop_pad_mode == 1:
    #             if expended_x_idx == -1:
    #                 continue
    #         elif active_num > 0:
    #             if expanded_x_idx >= active_num:
    #                 continue
    #         grad_x[i] = np.add(grad_x[i], grad_expanded_x[expanded_x_idx].astype(np.float32))
        
    # return grad_x

    num_rows = grad_expanded_x.shape[0] // top_k
    hidden_size = grad_expanded_x.shape[-1]

    grad_x = torch.zeros((num_rows, hidden_size), dtype=torch.float32)
    for i in range(num_rows):
        for j in range(i * top_k, i * top_k + top_k, 1):
            expanded_x_idx = expanded_row_idx[j]
            if drop_pad_mode == 1:
                if expended_x_idx == -1:
                    continue
            elif active_num > 0:
                if expanded_x_idx >= active_num:
                    continue
            grad_x[i] = torch.add(grad_x[i], grad_expanded_x[expanded_x_idx].to(torch.float32))
        
    return grad_x


class Model(torch.nn.Module):
    def __init__(self, ):
        super(Model, self).__init__()
    
    def forward(self, grad_expanded_x, expanded_row_idx, top_k, drop_pad_mode, active_num):
        if grad_expanded_x.dtype == torch.bfloat16:
            grad_expanded_x = grad_expanded_x.float()
        #     grad_expanded_x = grad_expanded_x.float().numpy()
        #     grad_expanded_x = grad_expanded_x.astype(tf.bfloat16.as_numpy_dtype)
        # else:
        #     grad_expanded_x = grad_expanded_x.numpy()
        # expanded_row_idx = expanded_row_idx.numpy()
        expanded_row_idx = expanded_row_idx
        output = moe_init_routing_v2_grad(grad_expanded_x, expanded_row_idx, top_k, drop_pad_mode, active_num)
        # return torch.from_numpy(output).to(grad_expanded_x.dtype)
        return output.to(grad_expanded_x.dtype)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, grad_expanded_x, expanded_row_idx, top_k, drop_pad_mode, active_num):
        # NPU 实现调用
        grad_x = kernel_gen_ops.moe_init_routing_v2_grad(
            grad_expanded_x, 
            expanded_row_idx, 
            top_k, 
            drop_pad_mode, 
            active_num
        )
        return grad_x