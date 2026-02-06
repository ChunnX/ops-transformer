import torch


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    # 解析参数
    num_dims = int(param.get('num_dims', 2))
    index_shape_str = param.get('index_shape', '[10]')
    index_shape = eval(index_shape_str)
    dtype_str = param.get('dtype', 'int32')
    dtype = getattr(torch, dtype_str)
    
    # 解析stride和value_size
    stride_list = eval(param.get('stride', '[1]'))
    value_size_list = eval(param.get('value_size', '[10]'))
    
    # 生成indices_list
    indices_list = []
    for i in range(num_dims):
        max_val = value_size_list[i] * 2  # 生成一些超出value_size的值来测试取模
        if dtype in [torch.int32, torch.int64]:
            indices = torch.randint(0, max_val, index_shape, device=device, dtype=dtype)
        else:
            indices = torch.randint(0, max_val, index_shape, device=device, dtype=torch.int32).to(dtype)
        indices_list.append(indices)
    
    # 生成stride和value_size张量
    stride = torch.tensor(stride_list, device=device, dtype=torch.int32)
    value_size = torch.tensor(value_size_list, device=device, dtype=torch.int32)
    
    return (indices_list, stride, value_size)


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for LinearIndexV2.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
    """
    return []  # No special initialization inputs needed


def custom_check_precision(param, outputs, outputs_new):
    """
    自定义精度检查函数 - LinearIndexV2是整数运算,要求精确匹配
    """
    dtype_str = param.get('dtype', 'int32')
    dtype = getattr(torch, dtype_str)
    
    # 整数类型要求精确匹配
    RTOL_GENERAL = 0
    rtol = RTOL_GENERAL
    
    outputs = [outputs] if not isinstance(outputs, list) else outputs
    outputs_new = [outputs_new] if not isinstance(outputs_new, list) else outputs_new

    all_abs_diff, all_rel_diff = [], []
    is_pass = 1

    for out, out_new in zip(outputs, outputs_new):
        abs_diff = torch.abs(out - out_new)
        rel_diff = abs_diff / (torch.abs(out) + 1e-7)
        
        # 整数类型检查绝对差值是否为0
        if torch.any(abs_diff > 0):
            is_pass = 0
        
        abs_diff = abs_diff.view(-1)
        rel_diff = rel_diff.view(-1)
        
        all_abs_diff.append(abs_diff)
        all_rel_diff.append(rel_diff)

    all_abs_diff = torch.cat(all_abs_diff)
    all_rel_diff = torch.cat(all_rel_diff)
    return is_pass, all_abs_diff, all_rel_diff