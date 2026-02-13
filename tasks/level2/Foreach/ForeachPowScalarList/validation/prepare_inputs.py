import torch


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    shape_list = eval(param.get('input_shape', '[[1]]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    scalar_dtype = dtype
    if dtype == torch.int32:
        scalar_dtype = torch.int64
    elif dtype == torch.bfloat16:
        scalar_dtype = torch.float32

    # 创建随机张量列表
    x_list = []
    scalars_list = []
    for shape in shape_list:
        if dtype == torch.int32:
            # 整数类型使用randint
            x = torch.randint(-100, 100, shape, device=device, dtype=dtype)
            scalars = torch.randint(-100, 100, [1], device=device, dtype=scalar_dtype)
        else:
            # 浮点类型使用randn
            x = torch.randn(shape, device=device, dtype=dtype)
            scalars = torch.randn([1], device=device, dtype=scalar_dtype)

        x_list.append(x)
        scalars_list.append(scalars)

    return (x_list, scalars_list)


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
