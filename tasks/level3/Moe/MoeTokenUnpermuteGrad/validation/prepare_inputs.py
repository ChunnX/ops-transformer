import torch
import numpy as np
import random
from framework.utils import check_precision


def get_inputs(param, device=None):
    """
    生成输入张量，用于 forward 方法。

    Args:
        param (dict): 包含 'tokens_num', 'topk', 'hidden_size', 'dtype', 'with_probs' 的字典
        device (torch.device): 设备，如 'cpu' 或 'npu'

    Returns:
        tuple: (permuted_tokens, unpermuted_tokens_grad, sorted_indices, probs)
    """
    tokens_num = int(param.get('tokens_num', 4))
    topk = int(param.get('topk', 2))
    hidden_size = int(param.get('hidden_size', 16))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    with_probs = bool(param.get('with_probs', 1))
    if not with_probs:
        topk = 1
    padded_mode = bool(param.get('padded_mode', 0))
    # permutedTokens shape: (tokens_num * topk, hidden_size)
    permuted_shape = (tokens_num * topk, hidden_size)
    permuted_tokens = torch.rand(permuted_shape, device=device, dtype=dtype)

    # unpermutedTokensGrad shape: (tokens_num, hidden_size)
    unpermuted_shape = (tokens_num, hidden_size)
    unpermuted_tokens_grad = torch.rand(unpermuted_shape, device=device, dtype=dtype)

    # sortedIndices shape: (tokens_num * topk,)
    # 取值范围是[0, tokens_num * topK_num - 1]，且没有重复索引
    total_tokens = tokens_num * topk
    sorted_indices_after = torch.randint(low=0, high=total_tokens, size=(tokens_num, topk)).to(torch.int32).to(device)
    def mPermute(tokens, indices, num_out_tokens=None, padded_mode=False, with_probs=True):
        if (indices.dim() == 1) or not with_probs:
            topk = 1
        else:
            topk = indices.size(1)
        flatten_indices = indices.view(-1)
        sorted_indices = torch.argsort(flatten_indices, stable=True)
        oringin_sorted_indices = sorted_indices
        sorted_indices = torch.argsort(sorted_indices, stable=True)
        if num_out_tokens is not None:
            sorted_indices = sorted_indices[:num_out_tokens]
        sorted_indices = sorted_indices
        permuted_tokens = tokens.index_select(0, sorted_indices // topk)
        return permuted_tokens, sorted_indices, oringin_sorted_indices
    
    permuted_tokens, sorted_indices_tmp, oringin_sorted_indices = mPermute(permuted_tokens, sorted_indices_after)

    # probsOptional shape: (tokens_num, topk)
    if with_probs:
        probs_shape = (tokens_num, topk)
        probs = torch.rand(probs_shape, device=device, dtype=dtype)
    else:
        probs = None

    return (permuted_tokens, unpermuted_tokens_grad, oringin_sorted_indices.to(torch.int32), probs, padded_mode)


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
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    for i in range(len(outputs)):
        output = outputs[i]
        output_new = outputs_new[i]
        if dtype == torch.float32:
            return check_precision(output, output_new, max_abs_error=0.0001, max_rel_error=0.0001)
        elif dtype == torch.float16:
            return check_precision(output, output_new, max_abs_error=0.001, max_rel_error=0.001)
        elif dtype == torch.bfloat16:
            return check_precision(output, output_new, max_abs_error=0.004, max_rel_error=0.004)
        else:
            return check_precision(output, output_new, max_abs_error=0.0001, max_rel_error=0.0001)
