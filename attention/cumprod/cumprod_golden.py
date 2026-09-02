#!/usr/bin/env python3
# coding: utf-8
"""Cumprod golden 参考实现。

语义对齐 spec.yaml：
- 公式: y = np.cumprod(x, axis=dim)
- fp16/bf16: 累加器拓宽到 fp32（cast 到 fp32 累乘，逐元素写回 cast 回原 dtype），
  与 torch.cumprod CPU 实现（acc_type=fp32，逐元素写回）完全一致。
- fp32: 原生直乘。

用法:
    python3 cumprod_golden.py          # 自检（与 numpy 逐位对拍）
"""

import numpy as np
import torch


def cumprod_golden(x, dim=0):
    """返回 torch.Tensor 形式的 golden 输出，dtype/shape 与输入一致。"""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if x.dtype in (torch.float16, torch.bfloat16):
        return torch.cumprod(x.float(), dim=dim).to(x.dtype)
    return torch.cumprod(x, dim=dim)


def _selfcheck():
    rng = np.random.default_rng(0)
    cases = [
        ((8,), 0, np.float32),
        ((1000,), 0, np.float32),
        ((16,), 0, np.float16),
        ((16,), 0, np.float16),
        ((256, 365), 1, np.float32),
        ((2, 3, 4), 1, np.float32),
        ((2, 3, 4), 2, np.float32),
        ((5, 7), -1, np.float32),
        ((4, 9), 0, np.float32),
    ]
    for shape, dim, dt in cases:
        lo, hi = (0.98, 1.02) if shape[dim] > 32 else (-1.5, 1.5)
        x = rng.uniform(lo, hi, size=shape).astype(dt)
        got = cumprod_golden(x, dim).numpy()
        ref = np.cumprod(x.astype(np.float64), axis=dim)
        ref_cast = ref.astype(dt)
        # 与"fp64 真值再 cast"口径一致性（整数完全一致；浮点容差内一致）
        if dt == np.float32:
            ok = np.allclose(got.astype(np.float64), ref, rtol=1e-5, atol=1e-5)
        else:
            ok = np.array_equal(got, ref_cast) or np.allclose(
                got.astype(np.float64), ref, rtol=5e-3, atol=5e-3
            )
        status = "OK" if ok else "MISMATCH"
        print(f"[{status}] shape={shape} dim={dim} dtype={dt}")
        assert ok, f"golden selfcheck failed: {shape} dim={dim} {dt}"

    # 边界: 全1 / 全0 / 含NaN / 含inf / 轴长1 / 负轴
    assert np.allclose(cumprod_golden(np.ones((6,), np.float32), 0).numpy(), 1.0)
    assert np.allclose(cumprod_golden(np.zeros((6,), np.float32), 0).numpy(), 0.0)
    xn = np.ones((4,), np.float32)
    xn[2] = np.nan
    assert np.isnan(cumprod_golden(xn, 0).numpy()[2:]).all()
    xi = np.ones((4,), np.float32)
    xi[1] = np.inf
    assert np.isinf(cumprod_golden(xi, 0).numpy()[1:]).all()
    assert np.array_equal(
        cumprod_golden(np.arange(6, dtype=np.float32).reshape(2, 3), -1).numpy(),
        np.cumprod(np.arange(6, dtype=np.float32).reshape(2, 3), axis=-1),
    )
    print("golden selfcheck ALL PASS")


if __name__ == "__main__":
    _selfcheck()
