import torch

def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    # 1. 解析输入
    input_shape = eval(param.get("input_shape", "[3,4]"))
    dtype_str = param.get("dtype", "float32")
    eps_str = param.get("eps", "null")
    dtype = getattr(torch, dtype_str)
    eps = float(eps_str) if eps_str != "null" else None

    # 2. 用 torch 生成 [0.1, 0.9] 范围的随机数据
    torch.manual_seed(1234)
    input_tensor = torch.rand(input_shape, dtype=dtype, device=device) * 0.8 + 0.1

    # 3. 返回
    return [input_tensor, eps]


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Initialization parameters for the MSE Loss model
    """
    # 从参数中获取reduction值

    return []