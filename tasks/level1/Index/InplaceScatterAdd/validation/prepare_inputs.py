import torch
import numpy as np


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    M = int(param.get('M', 10))
    N = int(param.get('N', 8))
    K = int(param.get('K', 5))
    
    var_dtype_str = param.get('var_dtype', 'float32')
    var_dtype = getattr(torch, var_dtype_str)
    
    indices_dtype_str = param.get('indices_dtype', 'int32')
    indices_dtype = getattr(torch, indices_dtype_str)
    
    # 生成var张量
    if var_dtype == torch.int32:
        var = torch.randint(-100, 100, (M, N), device=device, dtype=var_dtype)
    else:
        var = torch.randn((M, N), device=device, dtype=var_dtype)
    
    # 生成indices张量，确保索引在[0, M)范围内
    indices = torch.randint(0, M, (K,), device=device, dtype=indices_dtype)
    
    # 生成updates张量，与var类型一致
    if var_dtype == torch.int32:
        updates = torch.randint(-50, 50, (K, N), device=device, dtype=var_dtype)
    else:
        updates = torch.randn((K, N), device=device, dtype=var_dtype)

    return var, indices, updates


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for InplaceScatterAdd.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
    """
    return []  # No special initialization inputs needed


def custom_check_precision(param, outputs, outputs_new):
    var_dtype_str = param.get('var_dtype', 'float32')
    var_dtype = getattr(torch, var_dtype_str)
    
    # 根据数据类型设置 RTOL_GENERAL
    if var_dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool]:
        RTOL_GENERAL = 0  # 整数类型要求精确匹配
    elif var_dtype == torch.float16:
        RTOL_GENERAL = 1 / 512
    elif var_dtype == torch.bfloat16:
        RTOL_GENERAL = 1 / 256 + 1 / 16384
    elif var_dtype == torch.float32:
        RTOL_GENERAL = 1 / 2048 + 1 / 16384
    else:
        raise ValueError(f"Unsupported dtype: {var_dtype}")

    rtol = RTOL_GENERAL
    outputs = [outputs] if not isinstance(outputs, list) else outputs
    outputs_new = [outputs_new] if not isinstance(outputs_new, list) else outputs_new

    all_abs_diff, all_rel_diff = [], []
    is_pass = 1

    for out, out_new in zip(outputs, outputs_new):
        abs_diff = torch.abs(out - out_new)
        rel_diff = abs_diff / (torch.abs(out) + 1e-7)
        
        # 整数类型：检查绝对差值是否为0
        if var_dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool]:
            if torch.any(abs_diff > 0):
                is_pass = 0
        # 浮点数类型：使用相对容差
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