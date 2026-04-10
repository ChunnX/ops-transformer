import torch
import numpy as np
from framework.utils import check_precision


def get_inputs(param, device=None):
    permuted_tokens_shape = eval(param.get('permuted_tokens_shape', [1,512]))
    sorted_indices_shape = eval(param.get('sorted_indices_shape', [1]))
    probs_shape = eval(param.get('probs_shape', [1,1]))
    restore_shape = eval(param.get('restore_shape', [16,16]))
    padded_mode = bool(param.get('padded_mode', 0))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    def mPermute(tokens, indices, num_out_tokens=None, padded_mode=False):
        if indices.dim() == 1:
            topk = 1
        else:
            topk = indices.size(1)
        flatten_indices = indices.view(-1)
        sorted_indices = torch.argsort(flatten_indices, stable=True)
        oringin_sorted_indices = sorted_indices
        sorted_indices = torch.argsort(sorted_indices, stable=True)
        if num_out_tokens is not None:
            sorted_indices = sorted_indices[:num_out_tokens]
        permuted_tokens = tokens.index_select(0, sorted_indices // topk)
        return permuted_tokens, sorted_indices, oringin_sorted_indices
    
    permuted_tokens = torch.rand(permuted_tokens_shape, dtype=dtype)
    probs = torch.rand(probs_shape, dtype=dtype)
    sorted_indices = torch.randint(low=0, high=1, size=sorted_indices_shape).to(torch.int32)
    sorted_indices_after = torch.arange(sorted_indices_shape[0]).reshape(sorted_indices_shape).to(torch.int32)
    idx = torch.randperm(sorted_indices_after.nelement())
    sorted_indices_after = sorted_indices_after.view(-1)[idx].view(sorted_indices_after.size())
    permuted_tokens, sorted_indices_tmp, origin_sorted_indices = mPermute(permuted_tokens, sorted_indices_after)
    
    return (permuted_tokens.to(device), origin_sorted_indices.to(torch.int32).to(device), probs.to(device), padded_mode)


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
        print(output.shape)
        print(output_new.shape)
        if dtype == torch.float32:
            return check_precision(output, output_new, max_abs_error=0.0001, max_rel_error=0.0001)
        elif dtype == torch.float16:
            return check_precision(output, output_new, max_abs_error=0.001, max_rel_error=0.001)
        elif dtype == torch.bfloat16:
            return check_precision(output, output_new, max_abs_error=0.004, max_rel_error=0.004)
        else:
            return check_precision(output, output_new, max_abs_error=0.0001, max_rel_error=0.0001)

