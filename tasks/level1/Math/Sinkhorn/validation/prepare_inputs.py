import torch
import numpy as np


def get_inputs(param, device=None):
    """
    生成输入张量，用于 forward 方法。

    Args:
        param (dict): 包含 'input_shape', 'dtype' 的字典
        device (torch.device): 设备，如 'cpu' 或 'npu'

    Returns:
        tuple: (cost,)
    """
    input_shape = eval(param.get('input_shape', '[1, 1]'))
    dtype_str = param.get('dtype', 'float32')
    dtype = getattr(torch, dtype_str)

    cost = torch.rand(input_shape, device=device, dtype=dtype)
    return (cost,)


def get_init_inputs(param, device=None):
    """
    提取初始化参数，用于模型构造。

    Args:
        param (dict): 来自 DataFrame 的一行

    Returns:
        tuple: (cost, tol)
    """
    # 获取 input_shape 和 dtype
    input_shape = eval(param.get('input_shape', '[1, 1]'))
    dtype_str = param.get('dtype', 'float32')
    dtype = getattr(torch, dtype_str)

    # 生成 cost
    cost = torch.rand(input_shape, device=device, dtype=dtype)

    # 获取 tol，若无则使用默认值
    tol = param.get('tol', 0.0001)

    return (cost, float(tol))