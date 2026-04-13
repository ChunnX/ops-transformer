import torch
import numpy as np
from framework.utils import check_precision


def get_inputs(param, device=None):
    x_shape = eval(param.get('x_shape', [36,32]))
    exprt_idx_shape = eval(param.get('exprt_idx_shape', [36,1]))
    active_num = int(param.get('active_num', 166))
    expert_capacity = int(param.get('expert_capacity', 4))
    expert_num = int(param.get('expert_num', 361))
    drop_pad_mode = int(param.get('drop_pad_mode', 1))
    expert_tokens_count_or_cumsum_flag = int(param.get('expert_tokens_count_or_cumsum_flag', 2))
    expert_tokens_before_capacity_flag = int(param.get('expert_tokens_before_capacity_flag', 0))
    quant_mode = int(param.get('quant_mode', 0))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    x = torch.randn(x_shape, dtype=dtype).to(device)
    expert_idx = torch.randint(1, 2, size=exprt_idx_shape, dtype=torch.int32).to(device)
    
    # 生成 scale_optional 和 offset_optional
    scale_optional = None
    offset_optional = None
    if quant_mode == 0:
        # 静态量化场景，必须输入 scale 和 offset
        scale_optional = torch.tensor([0.6], dtype=torch.float32).to(device)
        offset_optional = torch.tensor([0.6], dtype=torch.float32).to(device)
    elif quant_mode == 1:
        # 动态量化场景，scale 为可选输入
        # 这里生成一个可选的 scale
        if np.random.random() > 0.5:
            scale_optional = torch.randn([1, x_shape[-1]], dtype=torch.float32).to(device)
    
    return (x, expert_idx, scale_optional, offset_optional, active_num, expert_capacity, expert_num, drop_pad_mode, expert_tokens_count_or_cumsum_flag, expert_tokens_before_capacity_flag, quant_mode)


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
