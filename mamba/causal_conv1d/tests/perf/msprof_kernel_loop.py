# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------
"""CausalConv1dGrad 反向 kernel 的 msprof 采集循环（最小化，聚焦 kernel 本身）。

不走完整 autograd，直接调用 torch_extension.causal_conv1d_grad_ext.causal_conv1d_grad
（aclnnCausalConv1dGrad 两段式接口），构造合法 bf16 输入后循环 ITERS 次。

用法：python3 msprof_kernel_loop.py <B> <T> <ITERS>
固定参数：D=12288, kW=4, dtype=bf16, activation=silu（kimik3 场景）。

运行前置条件（同 profile_bwd_compare_composite_vs_kernel.py）：
  ASCEND_CUSTOM_OPP_PATH 指向 custom_opp/vendors/causal_conv1d_grad_custom（脚本内 setdefault）。
"""
import argparse
import os
import sys
import time
from pathlib import Path

# ---- 环境与路径（必须在 import torch / torch_npu 之前） ----
_THIS_DIR = Path(__file__).resolve().parent
_CONV1D_DIR = _THIS_DIR.parent.parent                                # mamba/causal_conv1d
_SRC_ROOT = _CONV1D_DIR.parents[1]                                   # ops-transformer 根
_GRAD_REPO = Path(
    "/workspace/huangqiang/hq_claude/code/cannbot-skills/plugins-official/"
    "ops-registry-invoke/operators/causal_conv1d_grad"
)
_CUSTOM_OPP_VENDOR = _GRAD_REPO / "custom_opp" / "vendors" / "causal_conv1d_grad_custom"

os.environ.setdefault("ASCEND_CUSTOM_OPP_PATH", str(_CUSTOM_OPP_VENDOR))
sys.path.insert(0, str(_SRC_ROOT / "torch_extension"))  # 源码仓 cann_ops_transformer 优先
sys.path.insert(0, str(_CONV1D_DIR))                    # torch_extension 包可导入

import torch  # noqa: E402
import torch_npu  # noqa: E402

from torch_extension.causal_conv1d_grad_ext import causal_conv1d_grad  # noqa: E402

# ---- kimik3 场景固定参数 ----
DIM = 12288
KW = 4
DTYPE = torch.bfloat16
DEVICE = "npu:0"


def make_inputs(B, T, seed=1234):
    """与 profile_bwd_compare_composite_vs_kernel.py 同分布：反向 kernel 只需 4 个输入。"""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(B, T, DIM, generator=gen, dtype=torch.float32)).to(DTYPE).to(DEVICE)
    weight = (torch.randn(KW, DIM, generator=gen, dtype=torch.float32) * 0.1).to(DTYPE).to(DEVICE)
    bias = (torch.randn(DIM, generator=gen, dtype=torch.float32) * 0.01).to(DTYPE).to(DEVICE)
    grad_y = (torch.randn(B, T, DIM, generator=gen, dtype=torch.float32) * 0.1).to(DTYPE).to(DEVICE)
    return grad_y, x, weight, bias


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("B", type=int, help="batch size")
    parser.add_argument("T", type=int, help="sequence length")
    parser.add_argument("ITERS", type=int, help="kernel 调用次数（不含 warmup）")
    parser.add_argument("--warmup", type=int, default=3, help="kernel 预热次数")
    args = parser.parse_args()

    B, T, ITERS = args.B, args.T, args.ITERS
    torch.npu.set_device(0)
    print(f"ASCEND_CUSTOM_OPP_PATH={os.environ.get('ASCEND_CUSTOM_OPP_PATH')}")
    print(f"[msprof_kernel_loop] B={B} T={T} D={DIM} kW={KW} dtype=bf16 "
          f"warmup={args.warmup} iters={ITERS}")

    grad_y, x, weight, bias = make_inputs(B, T)

    # 正确性冒烟：一次调用确认 shape 合法（不做数值比对，本脚本只服务 perf 采集）
    dx, dweight, dbias = causal_conv1d_grad(grad_y, x, weight, bias)
    torch.npu.synchronize()
    assert dx.shape == (B, T, DIM) and dweight.shape == (KW, DIM) and dbias.shape == (DIM,)
    print(f"[msprof_kernel_loop] smoke ok: dx{tuple(dx.shape)} dw{tuple(dweight.shape)} "
          f"db{tuple(dbias.shape)}")
    del dx, dweight, dbias

    for _ in range(args.warmup):
        causal_conv1d_grad(grad_y, x, weight, bias)
    torch.npu.synchronize()

    start = torch_npu.npu.Event(enable_timing=True)
    end = torch_npu.npu.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        causal_conv1d_grad(grad_y, x, weight, bias)
    end.record()
    torch.npu.synchronize()
    avg_us = start.elapsed_time(end) * 1000.0 / ITERS
    print(f"[msprof_kernel_loop] bwd kernel avg: {avg_us:.2f} us over {ITERS} iters "
          f"(B={B}, T={T}, tokens={B * T})")

    del grad_y, x, weight, bias
    torch.npu.empty_cache()


if __name__ == "__main__":
    main()
