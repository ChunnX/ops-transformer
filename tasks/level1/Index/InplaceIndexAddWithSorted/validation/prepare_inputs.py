import torch
import numpy as np


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    var_shape = eval(param.get('var_shape', '[10, 8]'))
    value_shape = eval(param.get('value_shape', '[5, 8]'))
    indices_num = int(param.get('indices_num', 5))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    axis = int(param.get('axis', 0))
    enable_alpha = int(param.get('enable_alpha', 0))
    
    # 生成var张量
    if dtype in [torch.int32, torch.int16]:
        var = torch.randint(-10, 10, var_shape, device=device, dtype=dtype)
    else:
        var = torch.randn(var_shape, device=device, dtype=dtype)
    
    # 生成value张量
    if dtype in [torch.int32, torch.int16]:
        value = torch.randint(-10, 10, value_shape, device=device, dtype=dtype)
    else:
        value = torch.randn(value_shape, device=device, dtype=dtype)
    
    # 生成sorted_indices (必须是排序的)
    # 从var的axis维度范围内随机选择索引,然后排序
    max_idx = var_shape[axis]
    sorted_indices = torch.randint(0, max_idx, (indices_num,), device=device, dtype=torch.int32)
    sorted_indices, _ = torch.sort(sorted_indices)
    
    # 生成pos索引 (从value的axis维度范围内选择)
    max_pos = value_shape[axis]
    pos = torch.randint(0, max_pos, (indices_num,), device=device, dtype=torch.int32)
    
    # 生成alpha (可选)
    if enable_alpha == 1:
        alpha = torch.tensor([2.0], device=device, dtype=torch.float32)
    else:
        alpha = None
    
    return var, value, sorted_indices, pos, axis, alpha


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for InplaceIndexAddWithSorted.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
    """
    return []  # No special initialization inputs needed


def custom_check_precision(param, outputs, outputs_new):
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    
    # 根据数据类型设置容差
    if dtype in [torch.int16, torch.int32]:
        RTOL_GENERAL = 0  # 整数类型要求精确匹配
    elif dtype == torch.float16:
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
        
        # 整数类型:检查绝对差值是否为0
        if dtype in [torch.int16, torch.int32]:
            if torch.any(abs_diff > 0):
                is_pass = 0
        # 浮点数类型:使用相对容差
        else:
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