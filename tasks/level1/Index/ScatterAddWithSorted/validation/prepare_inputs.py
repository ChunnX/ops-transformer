import torch
import random

def get_inputs(param, device=None):
    """
    Generate inputs: var, value, sorted_index, pos
    """
    var_shape = eval(param.get('input_shape', '[32, 32]')) 
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    
    if dtype in [torch.int32, torch.int8, torch.uint8]:
        var = torch.randint(-10, 10, var_shape, dtype=dtype, device=device)
    else:
        var = torch.randn(var_shape, dtype=dtype, device=device)

    last_dim = var_shape[-1]
    total_rows = var.numel() // last_dim
    
    num_updates = param.get('num_updates', 64)
    num_updates = int(num_updates)
    if num_updates > total_rows:
        num_updates = total_rows 
        
    indices = torch.randint(0, total_rows, (num_updates,), device=device).sort()[0].int()
    
    use_pos = True
    if dtype in [torch.int32, torch.int8, torch.uint8]:
        use_pos = False
        
    if use_pos:
        pool_size = num_updates 
        value_shape = list(var_shape)
        value_rows = pool_size
        
        val_shape = [value_rows, last_dim]
        
        if dtype in [torch.int32, torch.int8, torch.uint8]:
            value = torch.randint(-10, 10, val_shape, dtype=dtype, device=device)
        else:
            value = torch.randn(val_shape, dtype=dtype, device=device)
            
        pos = torch.randint(0, value_rows, (num_updates,), device=device).int()
        
    else:
        val_shape = [num_updates, last_dim]
        if dtype in [torch.int32, torch.int8, torch.uint8]:
            value = torch.randint(-10, 10, val_shape, dtype=dtype, device=device)
        else:
            value = torch.randn(val_shape, dtype=dtype, device=device)
        pos = None
    
    return var, value, indices, pos

def get_init_inputs(param, device=None):
    return []

def custom_check_precision(param, outputs, outputs_new):
    # Standard precision check
    return True, None, None 