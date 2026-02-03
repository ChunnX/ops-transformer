import torch
import numpy as np
import random

def get_inputs(param, device=None):
    """
    生成输入张量，用于 forward 方法。

    Args:
        param (dict): 包含 'input_shape', 'dtype', 'with_group_idx', 'num_groups' 的字典
        device (torch.device): 设备，如 'cpu' 或 'npu'

    Returns:
        tuple: (grad_y, group_idx) 或 (grad_y, None)
    """
    input_shape = eval(param.get('input_shape', '[1, 1]'))
    dtype_str = param.get('dtype', 'float32')
    dtype = getattr(torch, dtype_str)

    with_group_idx = param.get('with_group_idx', 'true') == 1

    if with_group_idx:
        # 有 group_idx：grad_y 为 2D，shape=[GB, H]
        grad_y = torch.rand(input_shape, device=device, dtype=dtype)
        num_groups = int(param.get('num_groups', 4))

        # 1. 从 [1, input_shape[0]) 中随机选择 num_groups - 1 个数
        indices = torch.randint(1, input_shape[0]//2, (num_groups - 1,), dtype=torch.int32, device=device)

        # 2. 排序，确保递增
        indices = torch.sort(indices).values

        # 3. 添加结束点
        group_idx = torch.cat([indices, torch.tensor([input_shape[0]], dtype=torch.int32, device=device)])

        return (grad_y, group_idx)
    else:
        # 无 group_idx：grad_y 为 3D，shape=[G, C, H]
        G = int(input_shape[0])
        C = int(input_shape[1])
        H = int(input_shape[2])
        grad_y = torch.rand((G, C, H), device=device, dtype=dtype)
        # 生成默认 group_idx: [C]
        C_total = G * C
        group_idx = torch.tensor([C_total], dtype=torch.int32, device=device)
        return (grad_y, None)

def get_init_inputs(param, device=None):
    """
    提取初始化参数，用于模型构造。

    Args:
        param (dict): 来自 DataFrame 的一行

    Returns:
        tuple: (grad_y, group_idx)
    """
    input_shape = eval(param.get('input_shape', '[1, 1]'))
    dtype_str = param.get('dtype', 'float32')
    dtype = getattr(torch, dtype_str)

    with_group_idx = param.get('with_group_idx', 'true') == 1

    if with_group_idx:
        grad_y = torch.rand(input_shape, device=device, dtype=dtype)
        num_groups = int(param.get('num_groups', 4))

        # 1. 从 [1, input_shape[0]) 中随机选择 num_groups - 1 个数
        indices = torch.randint(1, input_shape[0]//2, (num_groups - 1,), dtype=torch.int32, device=device)

        # 2. 排序，确保递增
        indices = torch.sort(indices).values

        # 3. 添加结束点
        group_idx = torch.cat([indices, torch.tensor([input_shape[0]], dtype=torch.int32, device=device)])

        return (grad_y, group_idx)
    else:
        G = int(input_shape[0])
        C = int(input_shape[1])
        H = int(input_shape[2])
        grad_y = torch.rand((G, C, H), device=device, dtype=dtype)
        C_total = G * C
        group_idx = torch.tensor([C_total], dtype=torch.int32, device=device)
        return (grad_y, None)