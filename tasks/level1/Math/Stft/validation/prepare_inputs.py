# 参考 Rfft1D：只构造 x、window 和参数，plan 在 pybind.cpp 内生成
import torch


def _to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return str(v).lower() == "true"


def get_inputs(param, device=None):
    shape = eval(param.get("shape", "[64]"))
    dtype_str = param.get("dtype", "float32")
    dtype = getattr(torch, dtype_str)
    n_fft = int(param.get("n_fft", "8"))
    hop_length = int(param.get("hop_length", "4"))
    win_length = int(param.get("win_length", "8"))
    normalized = _to_bool(param.get("normalized", False))
    onesided = _to_bool(param.get("onesided", True))
    return_complex = _to_bool(param.get("return_complex", True))

    x = torch.randn(shape, device=device, dtype=dtype)
    window = torch.ones(win_length, device=device, dtype=dtype)
    return (x, window, n_fft, hop_length, win_length, normalized, onesided, return_complex)


def get_init_inputs(param, device=None):
    return []
