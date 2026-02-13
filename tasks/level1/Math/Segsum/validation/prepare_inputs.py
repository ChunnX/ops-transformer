import torch


def get_inputs(param, device=None):
    """
    根据 test_cases.csv 的一行生成 segsum 的输入。
    segsum 输入 x: 3D 或 4D，最后一维为 N，支持 float16/float32/bfloat16。
    """
    shape = eval(param.get('shape', '[1, 4]'))
    dtype_str = param.get('dtype', 'float32')
    dtype = getattr(torch, dtype_str)

    x = torch.randn(shape, device=device, dtype=dtype)
    return (x,)


def get_init_inputs(param, device=None):
    return []
