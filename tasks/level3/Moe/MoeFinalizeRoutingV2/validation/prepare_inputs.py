import torch
import random

# Fix random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    batch_size = int(param.get('batch_size', 16))
    num_experts = int(param.get('num_experts', 8))
    tokens_per_expert = int(param.get('tokens_per_expert', 16))
    hidden_size = int(param.get('hidden_size', 256))
    k = int(param.get('k', 2))
    use_bias = bool(param.get('use_bias', True))
    
    # Calculate expandedX shape: (batch_size * k, hidden_size)
    expanded_x_shape = (batch_size * k, hidden_size)
    expandedX = torch.randn(expanded_x_shape, device=device, dtype=dtype)
    
    # expandedRowIdx: (batch_size * k,)
    expandedRowIdx = torch.randint(0, batch_size, (batch_size * k,), device=device, dtype=torch.int32)
    
    # x1Optional: (batch_size, hidden_size)
    x1Optional = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    
    # x2Optional: (batch_size, hidden_size)
    x2Optional = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
    
    # biasOptional: (num_experts, hidden_size) or None
    if use_bias:
        biasOptional = torch.randn(num_experts, hidden_size, device=device, dtype=dtype)
    else:
        biasOptional = torch.zeros((0,), device=device, dtype=dtype)  # Empty tensor indicates no bias
    
    # scalesOptional: (batch_size, k)
    scalesOptional = torch.randn(batch_size, k, device=device, dtype=dtype)
    
    # expertIdxOptional: (batch_size, k)
    expertIdxOptional = torch.randint(0, num_experts, (batch_size, k), device=device, dtype=torch.int32)
    
    # dropPadMode: 0 or 1
    dropPadMode = int(param.get('dropPadMode', 0))
    
    return (expandedX, expandedRowIdx, x1Optional, x2Optional, biasOptional, scalesOptional, expertIdxOptional, dropPadMode)


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for MoeFinalizeRoutingV2.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
    """
    return []  # No special initialization inputs needed