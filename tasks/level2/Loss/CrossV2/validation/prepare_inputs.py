import torch


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    # 1. 解析输入
    shape = eval(param.get("shape", "[3,4,5]"))
    dtype_str = param.get("dtype", "float32")
    dim_str = param.get("dim", "2")
    dtype = getattr(torch, dtype_str)
    dim = int(dim_str)

    # 2. 使用 PyTorch 生成随机数据（范围 [-1, 1]）
    torch.manual_seed(1234)
    x1 = torch.rand(shape, dtype=dtype, device=device) * 2 - 1
    x2 = torch.rand(shape, dtype=dtype, device=device) * 2 - 1

    # 3. 返回
    return [x1, x2, dim]

def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Initialization parameters for the MSE Loss model
    """

    return []