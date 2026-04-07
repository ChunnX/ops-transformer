import torch
import numpy as np
import json

def get_init_inputs(row, device=None):
    # 设置随机种子
    np.random.seed(2023)
    torch.manual_seed(2023)
    
    # 从row中提取参数
    case_id = row['case_id']
    dtype = row['dtype']
    unpermuted_tokens_grad_shape = eval(row['unpermuted_tokens_grad_shape'])
    sorted_indices_shape = eval(row['sorted_indices_shape'])
    permuted_probs_output_grad_shape = eval(row['permuted_probs_output_grad_shape'])
    topk_num = row['topk_num']
    range_vals = eval(row['range'])
    padded_mode = row['padded_mode'] == 1
    
    # 将数据类型字符串转换为torch数据类型
    if dtype == "float32":
        torch_dtype = torch.float32
        np_dtype = np.float32
    elif dtype == "float16":
        torch_dtype = torch.float16
        np_dtype = np.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
        np_dtype = np.float32  # numpy不支持bfloat16，使用float32代替
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    # 生成unpermuted_tokens_grad
    unpermuted_tokens_grad = torch.tensor(np.random.uniform(-1.0, 1.0, unpermuted_tokens_grad_shape), dtype=torch_dtype)
    
    # 生成sorted_indices
    # sorted_indices应该是一个1D张量，包含0到num_tokens*topk_num-1的整数，经过排序
    num_tokens = sorted_indices_shape[0]
    sorted_indices = torch.randint(0, num_tokens, sorted_indices_shape, dtype=torch.int32)
    
    # 生成permuted_probs_output_grad (可选)
    if permuted_probs_output_grad_shape is not None and permuted_probs_output_grad_shape != []:
        permuted_probs_output_grad = torch.tensor(np.random.uniform(-1.0, 1.0, permuted_probs_output_grad_shape), dtype=torch_dtype)
    else:
        permuted_probs_output_grad = None
    
    # 将张量移动到指定设备
    if device is not None:
        unpermuted_tokens_grad = unpermuted_tokens_grad.to(device)
        sorted_indices = sorted_indices.to(device)
        if permuted_probs_output_grad is not None:
            permuted_probs_output_grad = permuted_probs_output_grad.to(device)
    
    return [
        unpermuted_tokens_grad,
        sorted_indices,
        permuted_probs_output_grad,
        topk_num,
        range_vals,
        padded_mode
    ]

def get_inputs(row, device=None):
    # 调用get_init_inputs函数，因为两个函数的功能相同
    return get_init_inputs(row, device=device)