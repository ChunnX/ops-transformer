#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""kimik3 正反向串联 autograd 绑定正确性验证（NPU, bf16）。

验证内容：
  1. 前向算子（causal_conv1d_fn）经 CausalConv1dFnAutoGrad 包装后 y.requires_grad=True
  2. autograd 反向三路输出（dx / dweight / dbias）与 CPU golden.py 对比：
     matched_ratio >= 0.99（rtol=atol=4e-3，bf16 混合容差）
  3. 反向算子直调（不经 autograd）同样对比 golden，交叉验证绑定无损
  4. loss.backward() 完整链路冒烟：x.grad / weight.grad / bias.grad 有限且非零

用法：
  ASCEND_CUSTOM_OPP_PATH=<...>/custom_opp/vendors/causal_conv1d_grad_custom \
  python3 test_autograd_bf16_correctness.py
"""
import os
import sys
from pathlib import Path

# ---- 环境变量必须在 import torch / torch_npu 之前设置 ----
_THIS_DIR = Path(__file__).resolve().parent
_CONV1D_DIR = _THIS_DIR.parent.parent                                # mamba/causal_conv1d
_TORCH_EXT_DIR = _CONV1D_DIR / "torch_extension"
_SRC_ROOT = _CONV1D_DIR.parents[1]                                   # ops-transformer 根
_GRAD_REPO = Path(
    "/workspace/huangqiang/hq_claude/code/cannbot-skills/plugins-official/"
    "ops-registry-invoke/operators/causal_conv1d_grad"
)
_CUSTOM_OPP_VENDOR = _GRAD_REPO / "custom_opp" / "vendors" / "causal_conv1d_grad_custom"

os.environ.setdefault("ASCEND_CUSTOM_OPP_PATH", str(_CUSTOM_OPP_VENDOR))
if not (_CUSTOM_OPP_VENDOR / "op_api" / "lib" / "libcust_opapi.so").exists():
    raise FileNotFoundError(f"libcust_opapi.so not found under {_CUSTOM_OPP_VENDOR}")

# 源码仓 cann_ops_transformer（op_builder/__init__.py 完整）必须先于安装副本命中，
# 否则 torch_extension/__init__.py -> causal_conv1d_fn.py 的
# `from cann_ops_transformer.op_builder import OpBuilder` 会失败。
sys.path.insert(0, str(_SRC_ROOT / "torch_extension"))
sys.path.insert(0, str(_CONV1D_DIR))          # 使 torch_extension 可作为包导入
sys.path.insert(0, str(_GRAD_REPO / "tests" / "st"))  # golden.py

import numpy as np
import torch
import torch_npu  # noqa: F401

from torch_extension.causal_conv1d_grad_ext import (
    causal_conv1d_grad,
    causal_conv1d_fn_autograd,
)
from golden import causal_conv1d_grad_golden

# ---- 验证参数（kimik3 小 shape 冒烟） ----
B, T, D, KW = 2, 16, 12288, 4
DTYPE = torch.bfloat16
RTOL = 4e-3
ATOL = 4e-3
THRESHOLD = 0.99
DEVICE = "npu:0"


def compare_tensors(npu_out: np.ndarray, golden_out: np.ndarray, name: str):
    """与 ST（test_minimal.py）口径一致的 matched_ratio 比对。"""
    assert npu_out.shape == golden_out.shape, \
        f"{name} shape mismatch: {npu_out.shape} vs {golden_out.shape}"
    abs_err = np.abs(npu_out.astype(np.float32) - golden_out.astype(np.float32))
    rel_err = abs_err / (np.abs(golden_out.astype(np.float32)) + 1e-8)
    match = (abs_err <= ATOL) | (rel_err <= RTOL)
    matched_ratio = float(np.mean(match))
    status = "PASS" if matched_ratio >= THRESHOLD else "FAIL"
    print(f"  [{name}] matched_ratio={matched_ratio:.4f} (threshold>={THRESHOLD}) [{status}] "
          f"abs_err max={abs_err.max():.3e} mean={abs_err.mean():.3e}")
    return matched_ratio


def main():
    torch.npu.set_device(0)
    print("=" * 78)
    print("autograd 绑定正确性验证 (NPU, bf16)")
    print("=" * 78)
    print(f"B={B}, T={T}, D={D}, kW={KW}, activation=silu")
    print(f"rtol={RTOL}, atol={ATOL}, matched_ratio threshold={THRESHOLD}")
    print(f"ASCEND_CUSTOM_OPP_PATH={os.environ['ASCEND_CUSTOM_OPP_PATH']}")

    # ---- 固定种子构造输入（与 golden 自测同分布） ----
    gen = torch.Generator(device="cpu").manual_seed(42)
    x = (torch.randn(B, T, D, generator=gen, dtype=torch.float32) * 0.5).to(DTYPE).to(DEVICE)
    weight = (torch.randn(KW, D, generator=gen, dtype=torch.float32) * 0.1).to(DTYPE).to(DEVICE)
    bias = (torch.randn(D, generator=gen, dtype=torch.float32) * 0.01).to(DTYPE).to(DEVICE)
    grad_y = (torch.randn(B, T, D, generator=gen, dtype=torch.float32) * 0.1).to(DTYPE).to(DEVICE)

    conv_states = torch.zeros(B, KW - 1, D, dtype=DTYPE, device=DEVICE)
    query_start_loc = torch.arange(B + 1, dtype=torch.int32, device=DEVICE) * T

    # ---- golden（CPU, fp32 中间计算） ----
    dx_ref, dw_ref, db_ref = causal_conv1d_grad_golden(
        grad_y.float().cpu().numpy(), x.float().cpu().numpy(),
        weight.float().cpu().numpy(), bias.float().cpu().numpy(),
    )
    # golden 返回 fp32；比对时以 NPU bf16 输出为准做混合容差

    all_pass = True

    # ---- 路线 1：反向算子直调（不经 autograd），交叉验证 ----
    print("\n[路线 1] causal_conv1d_grad 直调 vs golden")
    dx0, dw0, db0 = causal_conv1d_grad(grad_y, x, weight, bias)
    torch.npu.synchronize()
    r1 = [
        compare_tensors(dx0.float().cpu().numpy(), dx_ref, "dx"),
        compare_tensors(dw0.float().cpu().numpy(), dw_ref, "dweight"),
        compare_tensors(db0.float().cpu().numpy(), db_ref, "dbias"),
    ]
    all_pass = all_pass and all(r >= THRESHOLD for r in r1)

    # ---- 路线 2：autograd 绑定，torch.autograd.grad 固定 grad_y ----
    print("\n[路线 2] CausalConv1dFnAutoGrad + torch.autograd.grad vs golden")
    x_leaf = x.detach().clone().requires_grad_(True)
    w_leaf = weight.detach().clone().requires_grad_(True)
    b_leaf = bias.detach().clone().requires_grad_(True)
    y = causal_conv1d_fn_autograd(x_leaf, w_leaf, b_leaf, conv_states, query_start_loc,
                                  activation="silu")
    torch.npu.synchronize()
    print(f"  y.requires_grad = {y.requires_grad} (期望 True)")
    all_pass = all_pass and y.requires_grad

    dx, dw, db = torch.autograd.grad(y, (x_leaf, w_leaf, b_leaf), grad_outputs=grad_y)
    torch.npu.synchronize()
    r2 = [
        compare_tensors(dx.float().cpu().numpy(), dx_ref, "dx"),
        compare_tensors(dw.float().cpu().numpy(), dw_ref, "dweight"),
        compare_tensors(db.float().cpu().numpy(), db_ref, "dbias"),
    ]
    all_pass = all_pass and all(r >= THRESHOLD for r in r2)

    # ---- 路线 3：loss.backward() 完整链路冒烟 ----
    print("\n[路线 3] loss.backward() 链路冒烟")
    x2 = x.detach().clone().requires_grad_(True)
    w2 = weight.detach().clone().requires_grad_(True)
    b2 = bias.detach().clone().requires_grad_(True)
    y2 = causal_conv1d_fn_autograd(x2, w2, b2, conv_states, query_start_loc, activation="silu")
    loss = (y2 * grad_y).sum()  # dloss/dy == grad_y，与路线 2 数值一致
    loss.backward()
    torch.npu.synchronize()
    for name_, t_ in (("dx", x2.grad), ("dweight", w2.grad), ("dbias", b2.grad)):
        finite = bool(torch.isfinite(t_.float()).all())
        nonzero = bool((t_.float().abs() > 0).any())
        print(f"  {name_}: defined={t_ is not None}, finite={finite}, nonzero={nonzero}")
        all_pass = all_pass and finite and nonzero

    # 路线 3 与路线 2 数值一致性（同为 grad_y 反传，应当 bit 级一致）
    dx_diff = (x2.grad.float() - dx.float()).abs().max().item()
    print(f"  路线3 vs 路线2 dx max diff = {dx_diff:.3e} (期望 0)")
    all_pass = all_pass and dx_diff == 0.0

    print("\n" + "=" * 78)
    print(f"总体结论: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 78)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
