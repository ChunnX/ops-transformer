import torch
from typing import Tuple, Dict, Any

def get_inputs(param, device=None):
    """
    生成输入数据，用于 forward 方法。

    Args:
        param (dict): 包含 'input_shape', 'dtype', 'with_group_idx', 'num_groups' 的字典
        device (torch.device): 设备，如 'cpu' 或 'npu'

    Returns:
        tuple: (input_tensor, pdf_ref, statistic, reshuff)
    """
    # 1. 输入 shape
    input_shape = eval(param.get("input_shape", '[4, 1024, 64, 64]'))
    dtype_str = param.get("dtype", "float32")
    dtype = getattr(torch, dtype_str)
    assert dtype in [torch.float32, torch.float16, torch.bfloat16], "dtype must be float32/float16/bfloat16"

    # 2. input_tensor
    input_tensor = torch.randn(input_shape, dtype=dtype, device=device)
    assert input_tensor.numel() % 64 == 0, "input_tensor numel must be multiple of 64"
    assert input_tensor.numel() >= 32768, "input_tensor numel must >= 32768"

    # 3. pdf_ref
    input_tensor = input_tensor.contiguous()
    pdf_ref = torch.zeros(1, 256, dtype=torch.int32, device=device)

    # 4. 控制参数
    statistic = True if param.get("statistic", True) == 1 else False
    reshuff = True if param.get("reshuff", False) == 1 else False

    return (input_tensor, pdf_ref, statistic, reshuff)

def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for sinh.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
    """
    return []  # No special initialization inputs needed
