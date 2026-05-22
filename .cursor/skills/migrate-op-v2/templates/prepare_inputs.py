import torch


def get_inputs(param, device=None):
    """
    根据 test_cases.csv 的一行（param dict）生成输入张量。

    Args:
        param: dict，包含 test_cases.csv 中的字段（如 shape, dtype 等）
        device: str，目标设备（"cpu" 或 "npu"）

    Returns:
        tuple: 与 Model.forward() 签名匹配的参数列表
    """
    shape = eval(param['shape'])  # CSV 中 shape 为字符串如 "[2, 4]"
    dtype_name = param.get('dtype', 'float32')
    dtype = getattr(torch, dtype_name)
    device = device or "cpu"

    # TODO: 根据算子语义生成输入张量
    # 确保生成的数据不违反算子约束（如 sqrt 不接受负数）
    x = torch.randn(shape, dtype=dtype, device=device)
    return (x,)


def get_init_inputs(param, device=None):
    """
    返回 Model(*init_args) 和 ModelNew(*init_args) 的初始化参数。
    大多数算子不需要 init 参数，返回空列表。
    """
    return []


def custom_check_precision(param, ref_output, test_output):
    """
    自定义精度检查函数（可选）。

    Args:
        param: dict，测试用例参数
        ref_output: torch.Tensor，Model.forward() 的 CPU 输出
        test_output: torch.Tensor，ModelNew.forward() 的 NPU 输出（已转为 CPU）

    Returns:
        tuple: (is_pass: bool, abs_diff: torch.Tensor, rel_diff: torch.Tensor)
    """
    abs_diff = torch.abs(ref_output - test_output)
    rel_diff = abs_diff / (torch.abs(ref_output) + 1e-7)

    dtype = getattr(torch, param.get('dtype', 'float32'))
    if dtype in (torch.int32, torch.int64, torch.int8, torch.uint8, torch.bool):
        # 整型：必须精确匹配
        is_pass = torch.all(ref_output == test_output).item()
    else:
        # 浮点型：在标准容差内
        if dtype == torch.float16:
            rtol = 1.0 / 512
        elif dtype == torch.bfloat16:
            rtol = 1.0 / 256 + 1.0 / 16384
        else:  # float32, complex64
            rtol = 1.0 / 2048 + 1.0 / 16384
        max_abs_error = 1e-5
        is_pass = bool(
            torch.all(torch.isfinite(ref_output)).item() and
            torch.all(torch.isfinite(test_output)).item() and
            torch.all((abs_diff <= max_abs_error) | (rel_diff <= rtol)).item()
        )

    return is_pass, abs_diff.reshape(-1), rel_diff.reshape(-1)
