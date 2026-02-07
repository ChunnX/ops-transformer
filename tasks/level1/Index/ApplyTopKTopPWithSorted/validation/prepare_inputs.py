import torch
import numpy as np


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    batch_size = int(param.get('batch_size', 2))
    vocab_size = int(param.get('vocab_size', 100))
    dtype_str = param.get('dtype', 'float32')
    dtype = getattr(torch, dtype_str)
    
    mode = param.get('mode', 'topk_topp')  # 'topk_topp', 'topk', 'topp'
    
    # 生成随机logits并排序
    logits = torch.randn(batch_size, vocab_size, device=device, dtype=torch.float32)
    sorted_value, sorted_indices = torch.sort(logits, dim=-1, descending=False)
    
    # 转换为目标数据类型
    sorted_value = sorted_value.to(dtype)
    sorted_indices = sorted_indices.to(torch.int32)
    
    # 根据模式生成p和k
    p = None
    k = None
    
    if mode == 'topk_topp':
        # 同时使用k和p
        p_value = float(param.get('p_value', 0.9))
        k_value = int(param.get('k_value', 10))
        p = torch.full((batch_size,), p_value, device=device, dtype=dtype)
        k = torch.full((batch_size,), k_value, device=device, dtype=torch.int32)
    elif mode == 'topk':
        # 仅使用k
        k_value = int(param.get('k_value', 10))
        k = torch.full((batch_size,), k_value, device=device, dtype=torch.int32)
    elif mode == 'topp':
        # 仅使用p
        p_value = float(param.get('p_value', 0.9))
        p = torch.full((batch_size,), p_value, device=device, dtype=dtype)
    
    return sorted_value, sorted_indices, p, k


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for ApplyTopKTopPWithSorted.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
    """
    return []


def custom_check_precision(param, outputs, outputs_new):
    dtype_str = param.get('dtype', 'float32')
    dtype = getattr(torch, dtype_str)
    
    # 根据数据类型设置容差
    if dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool]:
        RTOL_GENERAL = 0
    elif dtype == torch.float16:
        RTOL_GENERAL = 1 / 512
    elif dtype == torch.bfloat16:
        RTOL_GENERAL = 1 / 256 + 1 / 16384
    elif dtype == torch.float32:
        RTOL_GENERAL = 1 / 2048 + 1 / 16384
    else:
        RTOL_GENERAL = 1e-5

    rtol = RTOL_GENERAL
    outputs = [outputs] if not isinstance(outputs, list) else outputs
    outputs_new = [outputs_new] if not isinstance(outputs_new, list) else outputs_new

    all_abs_diff, all_rel_diff = [], []
    is_pass = 1

    for out, out_new in zip(outputs, outputs_new):
        # 处理-inf：直接比较是否相等
        is_inf_out = torch.isinf(out) & (out < 0)
        is_inf_out_new = torch.isinf(out_new) & (out_new < 0)
        
        # 检查-inf位置是否一致
        if not torch.equal(is_inf_out, is_inf_out_new):
            is_pass = 0
        
        # 对非-inf的值进行精度检查
        valid_mask = ~is_inf_out & ~is_inf_out_new
        if valid_mask.any():
            out_valid = out[valid_mask]
            out_new_valid = out_new[valid_mask]
            
            abs_diff = torch.abs(out_valid - out_new_valid)
            rel_diff = abs_diff / (torch.abs(out_valid) + 1e-7)
            
            tolerance = rtol * torch.maximum(torch.tensor(1.0, device=out.device), torch.abs(out_valid))
            if torch.any(abs_diff > tolerance):
                is_pass = 0
            
            all_abs_diff.append(abs_diff.view(-1))
            all_rel_diff.append(rel_diff.view(-1))
        else:
            # 如果全是-inf，添加零向量
            all_abs_diff.append(torch.zeros(1, device=out.device))
            all_rel_diff.append(torch.zeros(1, device=out.device))

    all_abs_diff = torch.cat(all_abs_diff) if all_abs_diff else torch.zeros(1)
    all_rel_diff = torch.cat(all_rel_diff) if all_rel_diff else torch.zeros(1)
    
    return is_pass, all_abs_diff, all_rel_diff