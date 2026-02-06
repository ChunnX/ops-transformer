import torch
import numpy as np


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    grad_shape = eval(param.get('grad_shape', '[10, 8]'))
    num_weights = int(param.get('num_weights', 100))
    dtype_str = param.get('dtype', 'float32')
    padding_idx = int(param.get('padding_idx', -1))
    
    dtype = getattr(torch, dtype_str)
    
    # 生成梯度张量
    grad = torch.randn(grad_shape, device=device, dtype=dtype)
    
    # 生成索引（flatten后的大小）
    total_elements = int(np.prod(grad_shape[:-1]))
    
    # 生成随机索引（范围在 [0, num_weights)）
    indices = torch.randint(0, num_weights, (total_elements,), device=device, dtype=torch.int32)
    
    # 对索引排序，获得 sort_indices 和 pos_idx
    sort_indices, pos_idx = torch.sort(indices)
    # pos_idx 需要记录原始位置
    _, pos_idx = torch.sort(torch.argsort(indices))
    
    return (grad, sort_indices, pos_idx.to(torch.int32))


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    
    Args:
        param (dict): Parameters from a pandas DataFrame row
        device: torch device
        
    Returns:
        list: Initialization parameters [num_weights, padding_idx, scale_grad_by_freq]
    """
    num_weights = int(param.get('num_weights', 100))
    padding_idx = int(param.get('padding_idx', -1))
    scale_grad_by_freq = param.get('scale_grad_by_freq', 'False').lower() == 'true'
    
    return (num_weights, padding_idx, scale_grad_by_freq)


def custom_check_precision(param, outputs, outputs_new):
    dtype_str = param.get('dtype', 'float32')
    dtype = getattr(torch, dtype_str)
    
    # 根据数据类型设置容差
    if dtype == torch.float16:
        RTOL_GENERAL = 1 / 512
    elif dtype == torch.bfloat16:
        RTOL_GENERAL = 1 / 256 + 1 / 16384
    elif dtype == torch.float32:
        RTOL_GENERAL = 1 / 2048 + 1 / 16384
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
    rtol = RTOL_GENERAL
    outputs = [outputs] if not isinstance(outputs, list) else outputs
    outputs_new = [outputs_new] if not isinstance(outputs_new, list) else outputs_new
    
    all_abs_diff, all_rel_diff = [], []
    is_pass = 1
    
    for out, out_new in zip(outputs, outputs_new):
        abs_diff = torch.abs(out - out_new)
        rel_diff = abs_diff / (torch.abs(out) + 1e-7)
        
        tolerance = rtol * torch.maximum(torch.tensor(1.0, device=out.device), torch.abs(out))
        if torch.any(abs_diff > tolerance):
            is_pass = 0
        
        abs_diff = abs_diff.view(-1)
        rel_diff = rel_diff.view(-1)
        
        all_abs_diff.append(abs_diff)
        all_rel_diff.append(rel_diff)
    
    all_abs_diff = torch.cat(all_abs_diff)
    all_rel_diff = torch.cat(all_rel_diff)
    return is_pass, all_abs_diff, all_rel_diff