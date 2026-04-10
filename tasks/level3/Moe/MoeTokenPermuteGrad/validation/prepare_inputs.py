import torch
import numpy as np
from framework.utils import check_precision


def permute(tokens, indices, num_out_tokens=None, padded_mode=False):
    if padded_mode:
        return tokens.index_select(dim=0, index=indices.view(-1)), indices
    
    if indices.dim() == 1:
        topk = 1
    else:
        topk = indices.size(1)
    flatten_indices = indices.view(-1)

    sorted_indices = torch.argsort(flatten_indices, stable=True)
    sorted_indices1 = torch.argsort(sorted_indices, stable=True)

    if num_out_tokens is not None and num_out_tokens != 0:
        sorted_indices = sorted_indices[:num_out_tokens]
    s_k = sorted_indices // topk
    permuted_tokens = tokens.index_select(0, s_k)
    return permuted_tokens, sorted_indices1


def get_inputs(param, device=None):
    tokens_shape = eval(param.get('tokens_shape', [1,7]))
    indices_shape = eval(param.get('indices_shape', [1]))
    num_out_tokens = int(param.get('num_out_tokens', 1548263338))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    padded_mode = bool(param.get('padded_mode', 0))

    total_tokens = indices_shape[0]
    if len(indices_shape) == 2:
        total_tokens = total_tokens * indices_shape[1]

    tokens = torch.rand(tokens_shape, device="cpu", dtype=dtype)
    indices = torch.randint(low=0, high=total_tokens, size=indices_shape).to(torch.int32).to("cpu")
    tmp, sorted_indices = permute(tokens, indices, num_out_tokens, padded_mode)
    permuted_output_grad = torch.rand(tmp.shape, device="cpu", dtype=dtype)


    return (tokens.to(device), permuted_output_grad.to(device), indices.to(torch.int32).to(device), num_out_tokens, padded_mode)


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for sinh.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
    """
    return []  # No special initialization inputs needed


def custom_check_precision(param, outputs, outputs_new):
    """
    自定义精度检查函数
    """
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    check_outputs = []
    check_outputs_new = []
    # 检查每个输出
    for i in range(len(outputs)):
        output = outputs[i]
        output_new = outputs_new[i]
        
        # 跳过None值
        if output is None or output_new is None:
            continue
        check_outputs.append(output)
        check_outputs_new.append(output_new)
    # 根据数据类型设置不同的精度阈值
    if dtype == torch.float32:
        return check_precision(check_outputs, check_outputs_new, max_abs_error=0.0001, max_rel_error=0.0001)
    elif dtype == torch.float16:
        return check_precision(check_outputs, check_outputs_new, max_abs_error=0.001, max_rel_error=0.001)
    elif dtype == torch.bfloat16:
        return check_precision(check_outputs, check_outputs_new, max_abs_error=0.004, max_rel_error=0.004)
    else:
        return check_precision(check_outputs, check_outputs_new, max_abs_error=0.0001, max_rel_error=0.0001)
