import torch
import numpy as np 


def get_inputs(param, device=None):
    # 1. 解析输入
    xyz1_shape = eval(param.get("xyz1_shape", "[2,2,2]"))
    xyz2_shape = eval(param.get("xyz2_shape", "[2,2,2]"))
    idx1_shape = eval(param.get("idx1_shape", "[2,2]"))
    idx2_shape = eval(param.get("idx2_shape", "[2,2]"))
    gradDist1_shape = eval(param.get("gradDist1_shape", "[2,2]"))
    gradDist2_shape = eval(param.get("gradDist2_shape", "[2,2]"))
    dtype_str = param.get("dtype", "float32")
    dtype = getattr(torch, dtype_str)
    assert dtype in [torch.float32], "dtype must be float32"


    # 2. 生成随机数据
    np.random.seed(1234)
    torch.manual_seed(1234)
    xyz1 = torch.rand(xyz1_shape, dtype=dtype, device=device)
    xyz2 = torch.rand(xyz2_shape, dtype=dtype, device=device)
    idx1 = np.random.randint(0, xyz2_shape[1], size=idx1_shape, dtype=np.int32)
    idx2 = np.random.randint(0, xyz1_shape[1], size=idx2_shape, dtype=np.int32)
    gradDist1 = torch.rand(gradDist1_shape, dtype=dtype, device=device)
    gradDist2 = torch.rand(gradDist2_shape, dtype=dtype, device=device)

    # 3. 转为 tensor（用于 golden）
    idx1_t = torch.from_numpy(idx1).to(device).to(torch.int32)
    idx2_t = torch.from_numpy(idx2).to(device).to(torch.int32)

    # 4. 返回
    return [xyz1, xyz2, idx1_t, idx2_t, gradDist1, gradDist2]

def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Initialization parameters for the MSE Loss model
    """

    return []