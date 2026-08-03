#!/usr/bin/env python3
"""
验证 moe_init_routing active_num SymInt 改造效果。

原理：
  dynamic=True 时，x.size(0) 为 SymInt。
  - int schema:    active_num 被 dispatcher concretize 成常量 → 不同值触发重编译
  - SymInt schema: active_num 保持符号传播 → 不同值复用编译缓存

注意：必须用 dynamic=True，否则 dynamo 对 shape 做 specialization，
x.size(0) 变成具体 int，active_num 自然也是 int，无法验证 SymInt 效果。

运行：python3 verify_symint_active_num.py
"""
import os
os.environ.setdefault('TORCH_LOGS', 'recompiles')

import time
import torch
import torch.nn as nn
import torch_npu
import torchair as tng
from torchair.configs.compiler_config import CompilerConfig
from cann_ops_transformer.ops import moe_init_routing

import torch._dynamo
torch._dynamo.reset()


def get_graph_count():
    try:
        return torch._dynamo.utils.counters["stats"]["unique_graphs"]
    except Exception:
        return -1


config = CompilerConfig()
npu_backend = tng.get_npu_backend(compiler_config=config)

h = 2
k = 4
expert_num = 256
active_expert_range = [0, 255]


class MoeInitRoutingModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, expert_idx, *, topk_weight=None, scale=None, offset=None,
                active_num=-1, expert_capacity=-1, expert_num=-1, drop_pad_mode=0,
                expert_tokens_num_type=0, expert_tokens_num_flag=False, quant_mode=-1,
                active_expert_range=None, row_idx_type=0):
        return moe_init_routing(x, expert_idx, topk_weight=topk_weight,
                                scale=scale, offset=offset, active_num=active_num,
                                expert_capacity=expert_capacity, expert_num=expert_num,
                                drop_pad_mode=drop_pad_mode,
                                expert_tokens_num_type=expert_tokens_num_type,
                                expert_tokens_num_flag=expert_tokens_num_flag,
                                quant_mode=quant_mode,
                                active_expert_range=active_expert_range,
                                row_idx_type=row_idx_type)


model = MoeInitRoutingModel().npu()
compiled = torch.compile(model, backend=npu_backend, dynamic=True)


def run(n, label=""):
    x = torch.randn((n, h), dtype=torch.float32).npu()
    expert_idx = torch.randint(0, expert_num, (n, k), dtype=torch.int32).npu()
    topk_weight = torch.randn((n, k), dtype=torch.float32).npu()
    scale = torch.randn((1,), dtype=torch.float32).npu()
    offset = torch.randn((1,), dtype=torch.float32).npu()

    cnt_before = get_graph_count()
    t0 = time.time()
    expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale, expanded_topk_weight = \
        compiled(x, expert_idx, topk_weight=topk_weight, scale=scale, offset=offset,
                 active_num=x.size(0) * k, expert_capacity=-1, expert_num=expert_num,
                 drop_pad_mode=0, expert_tokens_num_type=1,
                 expert_tokens_num_flag=True, quant_mode=0,
                 active_expert_range=active_expert_range, row_idx_type=1)
    torch.npu.synchronize()
    t1 = time.time()
    cnt_after = get_graph_count()
    delta = cnt_after - cnt_before

    print(f"  [{label}] n={n:<4d}  耗时={t1-t0:.3f}s  "
          f"graphs={cnt_after}(+{delta})")
    print(f"    input shape: {topk_weight.shape}")
    print(f"    output shape: {expanded_topk_weight.shape}")
    print(f"    output dtype: {expanded_topk_weight.dtype}")


print("=" * 70)
print("验证 SymInt schema：dynamic=True 下不同 n 能否复用编译缓存")
print("  active_num = x.size(0) * k  (图模式下为 SymInt)")
print("=" * 70)

print("\n[1] n=3 — 首次编译")
run(3, "首次编译")

print("\n[2] n=12 — 不同 n，验证缓存复用")
run(12, "缓存验证")

print("\n[3] n=24 — 再换一个 n")
run(24, "再次验证")

print("\n" + "=" * 70)
print("判断标准：")
print("  ✓ SymInt 生效: [2][3] graphs 不增加 + 无 recompiling 日志 + 耗时显著低于[1]")
print("  ✗ int 回退   : [2][3] graphs 增加   + 出现 recompiling 日志 + 耗时接近[1]")
print("=" * 70)
