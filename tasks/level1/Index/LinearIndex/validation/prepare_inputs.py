import torch

def get_inputs(param, device=None):
    """
    构造输入数据。
    """
    # 1. 解析 indices 的形状和类型
    input_shape = eval(param.get('input_shape', '[32, 64]'))
    dtype_str = param.get('dtype', 'int32')
    dtype = getattr(torch, dtype_str)
    
    # 2. 解析 shape tensor (作为输入传入)
    shape_val = eval(param.get('shape_tensor', '[100, 100, 100]'))
    shape_tensor = torch.tensor(shape_val, dtype=torch.int32, device=device)
    
    # 3. 获取 axis 以确定生成数据的范围
    axis = int(param.get('axis', -1))
    rank = len(shape_val)
    if axis < 0:
        axis += rank
    
    # 确保 axis 有效
    if axis >= rank:
        axis = rank - 1
        
    dim_size = shape_val[axis]
    
    # 4. 生成随机索引，包含负数以测试 normalization
    # 范围在 [-dim_size, dim_size - 1] 之间
    indices = torch.randint(-dim_size, dim_size, input_shape, dtype=dtype, device=device)
    
    return indices, shape_tensor

def get_init_inputs(param, device=None):
    """
    获取模型初始化参数 (axis, combine)。
    """
    axis = int(param.get('axis', -1))
    combine = param.get('combine', 'False') == 'True'
    return [axis, combine]

def custom_check_precision(param, outputs, outputs_new):
    """
    自定义精度检查，因为输出是整数索引，要求精确匹配。
    """
    outputs = [outputs] if not isinstance(outputs, list) else outputs
    outputs_new = [outputs_new] if not isinstance(outputs_new, list) else outputs_new
    
    is_pass = 1
    all_abs_diff = []
    all_rel_diff = []
    
    for out, out_new in zip(outputs, outputs_new):
        # 转换为 float 进行比较
        out_f = out.float()
        out_new_f = out_new.float()
        
        abs_diff = torch.abs(out_f - out_new_f)
        rel_diff = abs_diff / (torch.abs(out_f) + 1e-7)
        
        # 整数必须完全相等
        if torch.any(abs_diff > 0):
            is_pass = 0
            
        all_abs_diff.append(abs_diff.view(-1))
        all_rel_diff.append(rel_diff.view(-1))
        
    all_abs_diff = torch.cat(all_abs_diff)
    all_rel_diff = torch.cat(all_rel_diff)
    return is_pass, all_abs_diff, all_rel_diff