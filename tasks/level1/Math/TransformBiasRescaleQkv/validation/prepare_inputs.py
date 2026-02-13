import torch


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    qkv: [batch, token, 3*dim], qkv_bias: [3*dim], num_heads: int.
    """
    batch = int(param.get('batch', 2))
    token = int(param.get('token', 128))
    dim = int(param.get('dim', 256))
    num_heads = int(param.get('num_heads', 8))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    triple_dim = 3 * dim
    qkv = torch.randn(batch, token, triple_dim, device=device, dtype=dtype)
    qkv_bias = torch.randn(triple_dim, device=device, dtype=dtype)
    return (qkv, qkv_bias, num_heads)


def get_init_inputs(param, device=None):
    return []
