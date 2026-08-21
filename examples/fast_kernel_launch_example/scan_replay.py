#!/usr/bin/env python3
# capture 版: NPUGraph capture N 次 add/grouped_matmul
#
# 环境变量控制:
#   OP_TYPE=add|gemm   选择算子（默认 add）
#   OP_ITERS=101       capture 迭代次数（默认 101）
#
# 核数说明: 核数由安装的 wheel 决定（run_head_overhead.sh 通过 sed 改源码 + rebuild）
#   本脚本不控制核数，只负责 capture
import os
import sys
import torch
import torch_npu
import ascend_ops

op_type = os.environ.get("OP_TYPE", "add")
op_iters = int(os.environ.get("OP_ITERS", "101"))

# === 数据准备 + run 函数 ===
if op_type == "add":
    # add: fp16, 2 args (核数由 wheel 决定)
    x = torch.randn(1000, 1000, dtype=torch.float16).npu()
    y = torch.randn(1000, 1000, dtype=torch.float16).npu()
    def run():
        torch.ops.ascend_ops.add(x, y)

elif op_type == "gemm":
    # grouped_matmul: bf16 (Cube 核, 核数由 wheel 决定)
    # 与 scan_eager.py 一致: E=4, M=128, K=32, N=32 (小矩阵, 讋开计算开销)
    E, M, K, N, PERGROUP = 4, 16, 16, 16, 16
    x_cpu = torch.rand(M, K, dtype=torch.bfloat16)
    weight_cpu = torch.rand(K, N, dtype=torch.bfloat16)
    x = [x_cpu.npu()]
    weight = [weight_cpu.npu()]
    gl = torch.zeros(E, dtype=torch.int64)
    total = PERGROUP
    for i in range(E):
        if total <= M:
            gl[i] = PERGROUP
            total += PERGROUP
        else:
            gl[i] = PERGROUP - (total - M)
            break
    group_list = gl.npu()
    def run():
        torch.ops.ascend_ops.grouped_matmul(
            x, weight, None, None, None, None, None,
            group_list, None, 0, 0, 0, 0, None
        )
else:
    raise ValueError(f"Unknown OP_TYPE: {op_type}")

print(f"capture {op_iters} iters ({op_type}, no warmup)...")
g = torch.npu.NPUGraph()
with torch.npu.graph(g):
    for _ in range(op_iters):
        run()

# gemm sync segfault 已修复 (grouped_matmul.cpp 悬垂引用修复), 统一走完整 replay 路径
torch.npu.synchronize()
print(f"replay once ({op_type}, {op_iters} iters)...")
g.replay()
torch.npu.synchronize()
print(f"replay done")
