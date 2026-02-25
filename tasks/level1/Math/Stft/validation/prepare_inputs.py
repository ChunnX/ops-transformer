# 参考 Rfft1D：只构造 x、window 和参数，plan 在 pybind.cpp 内生成
import torch
from framework.utils import check_precision
from framework.kernel_gen_config import config


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

    # CANN STFT 约束：n_fft <= L，win_length <= n_fft（见 CANN/ops-math/math/stft/README.md）
    L = shape[-1] if len(shape) >= 2 else shape[0]
    if n_fft > L:
        raise ValueError(f"STFT constraint violated: n_fft({n_fft}) must be <= L({L}), shape={shape}")
    if win_length > n_fft:
        raise ValueError(f"STFT constraint violated: win_length({win_length}) must be <= n_fft({n_fft})")
    if hop_length <= 0 or win_length <= 0 or n_fft <= 0:
        raise ValueError(f"STFT requires n_fft, hop_length, win_length > 0, got n_fft={n_fft}, hop_length={hop_length}, win_length={win_length}")

    x = torch.randn(shape, device=device, dtype=dtype)
    window = torch.ones(win_length, device=device, dtype=dtype)  # shape [win_length]，与 README 一致
    return (x, window, n_fft, hop_length, win_length, normalized, onesided, return_complex)


def get_init_inputs(param, device=None):
    return []


def _to_float32(t):
    """统一转为 float32 或 complex64，避免 Half/Double 混用导致 'expected scalar type Double but found Half'。"""
    if isinstance(t, (list, tuple)):
        return type(t)(_to_float32(x) for x in t)
    if not isinstance(t, torch.Tensor):
        return t
    if t.is_complex():
        return t.to(torch.complex64)
    return t.float()


def custom_check_precision(row, ref_output, test_output):
    """比较前将 ref/test 统一为 float32/complex64，再调用框架 check_precision，避免 Half/Double 混用报错。"""
    ref = ref_output if isinstance(ref_output, (list, tuple)) else [ref_output]
    test = test_output if isinstance(test_output, (list, tuple)) else [test_output]
    ref = [_to_float32(t) for t in ref]
    test = [_to_float32(t) for t in test]
    ref = ref[0] if len(ref) == 1 else ref
    test = test[0] if len(test) == 1 else test
    return check_precision(ref, test, config.eval.max_abs_error, config.eval.max_rel_error)
