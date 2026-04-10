import torch
import numpy as np
from framework.utils import check_precision


def get_inputs(param, device=None):
    grad_expanded_x_shape = eval(param.get('grad_expanded_x_shape', [32,89]))
    expanded_row_idx_shape = eval(param.get('expanded_row_idx_shape', [32]))
    top_k = int(param.get('top_k', 1))
    drop_pad_mode = int(param.get('drop_pad_mode', 0))
    active_num = int(param.get('active_num', 0))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    grad_expanded_x = torch.randn(grad_expanded_x_shape, dtype=dtype).to(device)
    expanded_row_idx = torch.randint(0, 5, size=expanded_row_idx_shape, dtype=torch.int32).to(device)
    
    return (grad_expanded_x, expanded_row_idx, top_k, drop_pad_mode, active_num)

def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for MoeInitRoutingV3.

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
