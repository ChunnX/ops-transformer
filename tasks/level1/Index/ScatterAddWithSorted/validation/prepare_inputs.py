import torch
import numpy as np


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    var_shape = eval(param.get('var_shape', '[10, 5]'))
    value_shape = eval(param.get('value_shape', '[5, 5]'))
    dtype_str = param.get('dtype', 'float32')
    dtype = getattr(torch, dtype_str)
    use_pos = param.get('use_pos', 'False') == 'True'
    
    # 生成var张量
    if dtype in [torch.int32, torch.int8, torch.uint8]:
        var = torch.randint(-50, 50, var_shape, device=device, dtype=dtype)
    else:
        var = torch.randn(var_shape, device=device, dtype=dtype)
    
    # 生成value张量
    if dtype in [torch.int32, torch.int8, torch.uint8]:
        value = torch.randint(-10, 10, value_shape, device=device, dtype=dtype)
    else:
        value = torch.randn(value_shape, device=device, dtype=dtype)
    
    # 生成sorted_index张量（必须是排序的且在有效范围内）
    num_indices = value_shape[0]
    max_index = var_shape[0]
    
    # 生成随机索引并排序
    sorted_index = torch.randint(0, max_index, (num_indices,), device=device, dtype=torch.int32)
    sorted_index, _ = torch.sort(sorted_index)
    
    # 生成pos张量（可选）
    if use_pos:
        # pos表示每个索引在value中的位置
        pos = torch.arange(num_indices, device=device, dtype=torch.int32)
    else:
        pos = None

    return var, value, sorted_index, pos


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for ScatterAddWithSorted.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
    """
    return []


def custom_check_precision(param, outputs, outputs_new):
    dtype_str = param.get('dtype', 'float32')
    dtype = getattr(torch, dtype_str)
    
    # 根据数据类型设置精度容差
    if dtype in [torch.int8, torch.int32, torch.uint8]:
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
        
        # 整数类型：检查绝对差值是否为0
        if dtype in [torch.int8, torch.int32, torch.uint8]:
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