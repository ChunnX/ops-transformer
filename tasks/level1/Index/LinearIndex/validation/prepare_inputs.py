import torch


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    indices_shape = eval(param.get('indices_shape', '[10]'))
    shape_values = eval(param.get('shape_values', '[20]'))
    dtype_str = param.get('dtype', 'int32')
    dtype = getattr(torch, dtype_str)
    
    # 根据shape_values确定有效索引范围
    axis = int(param.get('axis', -1))
    if axis < 0:
        axis += len(shape_values)
    
    target = shape_values[axis] if axis < len(shape_values) else 100
    
    # 生成索引，确保在有效范围内（支持负数索引）
    # 生成范围：[-target, target-1]
    if dtype == torch.int32:
        indices = torch.randint(-target, target, indices_shape, device=device, dtype=dtype)
    else:  # int64
        indices = torch.randint(-target, target, indices_shape, device=device, dtype=dtype)
    
    # shape张量（固定为int32）
    shape = torch.tensor(shape_values, device=device, dtype=torch.int32)
    
    return indices, shape


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Initialization parameters [axis, combine]
    """
    axis = int(param.get('axis', -1))
    combine = param.get('combine', 'False') == 'True'
    
    return [axis, combine]


def custom_check_precision(param, outputs, outputs_new):
    """
    自定义精度检查函数，LinearIndex输出为int32，要求精确匹配。
    """
    dtype = torch.int32
    RTOL_GENERAL = 0  # 整数类型要求精确匹配
    
    rtol = RTOL_GENERAL
    outputs = [outputs] if not isinstance(outputs, list) else outputs
    outputs_new = [outputs_new] if not isinstance(outputs_new, list) else outputs_new

    all_abs_diff, all_rel_diff = [], []
    is_pass = 1

    for out, out_new in zip(outputs, outputs_new):
        abs_diff = torch.abs(out - out_new)
        rel_diff = abs_diff / (torch.abs(out) + 1e-7)
        
        # 整数类型：检查绝对差值是否为0
        if torch.any(abs_diff > 0):
            is_pass = 0
        
        abs_diff = abs_diff.view(-1)
        rel_diff = rel_diff.view(-1)
        
        all_abs_diff.append(abs_diff)
        all_rel_diff.append(rel_diff)

    all_abs_diff = torch.cat(all_abs_diff)
    all_rel_diff = torch.cat(all_rel_diff)
    return is_pass, all_abs_diff, all_rel_diff