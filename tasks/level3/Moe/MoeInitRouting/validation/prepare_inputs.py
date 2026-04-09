import torch
import numpy as np
from framework.utils import check_precision

def get_inputs(param, device=None):
    x_shape = eval(param.get('x_shape', [1,9]))
    row_idx_shape = eval(param.get('row_idx_shape', [1,1]))
    expert_idx_shape = eval(param.get('expert_idx_shape', [1,1]))
    active_num = int(param.get('active_num', 209))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    x = torch.randn(x_shape, dtype=dtype).to(device)
    row_idx = torch.arange(expert_idx_shape[0] * expert_idx_shape[1]).reshape(expert_idx_shape).to(torch.int32).to(device)
    expert_idx = torch.randint(1, 2, size=expert_idx_shape, dtype=torch.int32).to(device)
    return (x, row_idx, expert_idx, active_num)


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
    print(f"check_outputs: {check_outputs}", flush=True)
    print(f"check_outputs_new: {check_outputs_new}", flush=True)
    # 根据数据类型设置不同的精度阈值
    if dtype == torch.float32:
        return check_precision(check_outputs, check_outputs_new, max_abs_error=0.0001, max_rel_error=0.0001)
    elif dtype == torch.float16:
        return check_precision(check_outputs, check_outputs_new, max_abs_error=0.001, max_rel_error=0.001)
    elif dtype == torch.bfloat16:
        return check_precision(check_outputs, check_outputs_new, max_abs_error=0.004, max_rel_error=0.004)
    else:
        return check_precision(check_outputs, check_outputs_new, max_abs_error=0.0001, max_rel_error=0.0001)
