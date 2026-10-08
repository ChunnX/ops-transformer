#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""A 路线(conv1d 拼接)误差根因定位：逐段对比 golden。

层次：
  1. CPU fp32  A 前向 y vs golden 前向 y     → 验证算法/布局
  2. NPU fp32  A 前向 y vs golden 前向 y     → 验证 NPU conv1d 数值(fp32)
  3. NPU bf16  A 前向 y vs golden 前向 y     → 定位 forward 是否已错
  4. NPU bf16  A grad_z（y*gy 对 z 的梯度）vs golden silu'*gy → 定位 backward 哪段错
"""
import os
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_CONV1D_DIR = _THIS_DIR.parent.parent
_SRC_ROOT = _CONV1D_DIR.parents[1]
_GRAD_REPO = Path(
    "/workspace/huangqiang/hq_claude/code/cannbot-skills/plugins-official/"
    "ops-registry-invoke/operators/causal_conv1d_grad"
)
sys.path.insert(0, str(_GRAD_REPO / "tests" / "st"))

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

DIM, KW = 512, 4  # 缩小 D 便于诊断
B, T = 2, 16

gen = torch.Generator(device="cpu").manual_seed(0)
x = torch.randn(B, T, DIM, generator=gen)
weight = torch.randn(KW, DIM, generator=gen) * 0.1
bias = torch.randn(DIM, generator=gen) * 0.01
grad_y = torch.randn(B, T, DIM, generator=gen) * 0.1


def a_forward_np(x_, w_, b_):
    """numpy 直算 golden 前向（fp32）。"""
    B_, T_, D_ = x_.shape
    xp = np.concatenate([np.zeros((B_, KW - 1, D_), np.float32), x_], axis=1)
    z = (w_[0] * xp[:, 3:T_ + 3, :] + w_[1] * xp[:, 2:T_ + 2, :]
         + w_[2] * xp[:, 1:T_ + 1, :] + w_[3] * xp[:, 0:T_, :] + b_)
    return z, 1.0 / (1.0 + np.exp(-z)) * z  # silu


def a_forward_torch(x_, w_, b_):
    x_c = x_.transpose(1, 2)
    w_c = w_.flip(0).t().reshape(DIM, 1, KW)
    z = F.conv1d(F.pad(x_c, (KW - 1, 0)), w_c, b_, groups=DIM)
    y = F.silu(z)
    return z.transpose(1, 2), y.transpose(1, 2)


def cmp(name, a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    err = np.abs(a - b)
    rel = err / (np.abs(b) + 1e-8)
    print(f"  {name:<28} max_abs={err.max():.3e} mean_abs={err.mean():.3e} "
          f"match(4e-3)={np.mean((err <= 4e-3) | (rel <= 4e-3)):.4f}")


# ---- 1. CPU fp32 ----
z_ref, y_ref = a_forward_np(x.numpy(), weight.numpy(), bias.numpy())
z_cpu, y_cpu = a_forward_torch(x, weight, bias)
print("[1] CPU fp32 A forward vs golden forward")
cmp("z", z_cpu.numpy(), z_ref)
cmp("y", z_cpu.numpy(), y_ref)

# ---- 2/3. NPU fp32 / bf16 前向 ----
xn, wn, bn, gn = (t.to("npu") for t in (x, weight, bias, grad_y))
z_n32, y_n32 = a_forward_torch(xn.float(), wn.float(), bn.float())
torch.npu.synchronize()
print("[2] NPU fp32 A forward vs golden forward")
cmp("z", z_n32.float().cpu().numpy(), z_ref)
cmp("y", y_n32.float().cpu().numpy(), y_ref)

z_nb, y_nb = a_forward_torch(xn.bfloat16(), wn.bfloat16(), bn.bfloat16())
torch.npu.synchronize()
print("[3] NPU bf16 A forward vs golden forward (cast fp32 比对)")
cmp("z", z_nb.float().cpu().numpy(), z_ref)
cmp("y", y_nb.float().cpu().numpy(), y_ref)

# ---- 4. NPU bf16 全链路梯度 vs CPU fp32 全链路梯度（同 layout 的 A 路线参考） ----
# CPU fp32 参考梯度（A 路线结构，即 conv1d 的 autograd）
xl = x.clone().requires_grad_(True)
wl = weight.clone().requires_grad_(True)
bl = bias.clone().requires_grad_(True)
_, yl = a_forward_torch(xl, wl, bl)
loss = (yl * grad_y).sum()
dx_c, dw_c, db_c = torch.autograd.grad(loss, (xl, wl, bl))

xln = xn.bfloat16().detach().clone().requires_grad_(True)
wln = wn.bfloat16().detach().clone().requires_grad_(True)
bln = bn.bfloat16().detach().clone().requires_grad_(True)
_, yln = a_forward_torch(xln, wln, bln)
lossn = (yln * gn.bfloat16()).sum()
dx_n, dw_n, db_n = torch.autograd.grad(lossn, (xln, wln, bln))
torch.npu.synchronize()
print("[4] NPU bf16 A autograd 梯度 vs CPU fp32 A autograd 梯度")
cmp("dx", dx_n.float().cpu().numpy(), dx_c.numpy())
cmp("dweight", dw_n.float().cpu().numpy(), dw_c.numpy())
cmp("dbias", db_n.float().cpu().numpy(), db_c.numpy())
