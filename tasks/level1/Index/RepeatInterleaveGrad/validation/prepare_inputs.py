import torch

def get_inputs(param, device=None):
    """
    构造 RepeatInterleaveGrad 的输入。
    注意：为了确保测试数据的合法性，这里的 input_shape 参数指的是
    **原始张量 X (即 x_grad) 的形状**。
    然后根据 repeats 配置生成 Forward 后的 y_grad，作为本算子的输入。
    """
    x_shape = eval(param.get('input_shape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    axis = int(param.get('axis', -1))
    
    if axis < 0: 
        axis_idx = axis + len(x_shape)
    else:
        axis_idx = axis

    repeats_type = param.get('repeats_type', 'scalar') 
    repeats_dtype_str = param.get('repeats_dtype', 'int32')
    repeats_dtype = getattr(torch, repeats_dtype_str)
    
    if repeats_type == 'scalar':
        r_val = int(param.get('repeats_val', 2))
        repeats = torch.tensor([r_val], dtype=repeats_dtype, device=device)
        y_shape = list(x_shape)
        y_shape[axis_idx] *= r_val
    else:
        dim_size = x_shape[axis_idx]
        r_list = torch.randint(1, 5, (dim_size,), dtype=repeats_dtype, device=device)
        repeats = r_list
        total_len = r_list.sum().item()
        y_shape = list(x_shape)
        y_shape[axis_idx] = total_len

    if dtype_str == "float32":
        y_grad = torch.randn(y_shape, dtype=dtype, device=device)
    else:
        y_grad = torch.randn(y_shape, dtype=torch.float32, device=device).to(dtype)

    return y_grad, repeats, axis

def get_init_inputs(param, device=None):
    return []

def custom_check_precision(param, outputs, outputs_new):
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    
    if dtype == torch.float16:
        rtol = 1e-3
        atol = 1e-3
    elif dtype == torch.bfloat16:
        rtol = 4e-3 
        atol = 4e-3
    else:
        rtol = 1e-4
        atol = 1e-4

    outputs = [outputs] if not isinstance(outputs, list) else outputs
    outputs_new = [outputs_new] if not isinstance(outputs_new, list) else outputs_new

    all_abs_diff, all_rel_diff = [], []
    is_pass = 1

    for out, out_new in zip(outputs, outputs_new):
        abs_diff = torch.abs(out - out_new)
        # 防止除零
        rel_diff = abs_diff / (torch.abs(out) + 1e-7)
        
        tolerance = rtol * torch.maximum(torch.tensor(1.0, device=out.device), torch.abs(out)) + atol
        if torch.any(abs_diff > tolerance):
            is_pass = 0
            
        all_abs_diff.append(abs_diff.view(-1))
        all_rel_diff.append(rel_diff.view(-1))

    all_abs_diff = torch.cat(all_abs_diff)
    all_rel_diff = torch.cat(all_rel_diff)
    return is_pass, all_abs_diff, all_rel_diff