"""
复现: torch.compile + npugraphex + dynamic=True — bs 被 MoE 算子 guard 固化
cachecompile 版本（基于 torchair.inference.cache_compile）

针对 cann_ops_transformer.moe_init_routing（V4, SymInt active_num）验证。

用法: python demo_simple_cache2.py <bs> [bs2]

  python demo_simple_cache2.py 3        # 编译 bs=3
  python demo_simple_cache2.py 3 12     # 编译 bs=3，再跑 bs=12 看是否泛化

Run: python demo_simple_cache2.py 3
"""

import sys, os, types, torch, torch.nn as nn
import torch_npu
import torchair
import logging
from cann_ops_transformer.ops import moe_init_routing

torch._logging.set_logs(dynamo=logging.DEBUG, aot=logging.DEBUG, output_code=True,
                        graph_code=True, recompiles=True)

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".torchair_cache2")

bs = int(sys.argv[1])

h = 2
k = 4
expert_num = 256
active_expert_range = [0, 255]


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, expert_idx, topk_weight, scale, offset):
        token_count, _ = x.shape
        expanded_x, expanded_row_idx, expert_tokens_count, expanded_scale, expanded_topk_weight = \
            moe_init_routing(x, expert_idx, topk_weight=topk_weight, scale=scale, offset=offset,
                             active_num=token_count * k,
                             expert_capacity=-1, expert_num=expert_num,
                             drop_pad_mode=0, expert_tokens_num_type=1,
                             expert_tokens_num_flag=True, quant_mode=0,
                             active_expert_range=active_expert_range, row_idx_type=1)
        return expanded_x, expanded_topk_weight


model = Model().npu()

# 使用 torchair.inference.cache_compile 包装 forward
cached_forward = torch.npu.npugraph_ex.inference.cache_compile(
    model.forward,
    dynamic=True,
    cache_dir=CACHE_DIR,
)

print(f"运行 bs={bs}")
x = torch.randn(bs, h, dtype=torch.float32).npu()
expert_idx = torch.randint(0, expert_num, (bs, k), dtype=torch.int32).npu()
topk_weight = torch.randn(bs, k, dtype=torch.float32).npu()
scale = torch.randn(1, dtype=torch.float32).npu()
offset = torch.randn(1, dtype=torch.float32).npu()

# 显式标记 batch dim 为动态，防止被 MoE 算子 guard 固化
torch._dynamo.mark_dynamic(x, 0)
torch._dynamo.mark_dynamic(expert_idx, 0)
torch._dynamo.mark_dynamic(topk_weight, 0)

expanded_x, expanded_topk_weight = cached_forward(x, expert_idx, topk_weight, scale, offset)

print(f"input shape: {topk_weight.shape}")
print(f"output shape: {expanded_topk_weight.shape}")
print(f"output dtype: {expanded_topk_weight.dtype}")

# 可选：第二个 bs 测试泛化
if len(sys.argv) > 2:
    bs2 = int(sys.argv[2])
    print(f"\n测试 bs={bs2}")
    try:
        x2 = torch.randn(bs2, h, dtype=torch.float32).npu()
        expert_idx2 = torch.randint(0, expert_num, (bs2, k), dtype=torch.int32).npu()
        topk_weight2 = torch.randn(bs2, k, dtype=torch.float32).npu()
        scale2 = torch.randn(1, dtype=torch.float32).npu()
        offset2 = torch.randn(1, dtype=torch.float32).npu()
        torch._dynamo.mark_dynamic(x2, 0)
        torch._dynamo.mark_dynamic(expert_idx2, 0)
        torch._dynamo.mark_dynamic(topk_weight2, 0)
        expanded_x2, expanded_topk_weight2 = cached_forward(x2, expert_idx2, topk_weight2, scale2, offset2)
        print(f"input shape: {topk_weight2.shape}")
        print(f"output shape: {expanded_topk_weight2.shape}")
        print(f"output dtype: {expanded_topk_weight2.dtype}")
    except Exception as e:
        print(f"泛化失败: {type(e).__name__}: {e}")


# python  demo_simple_cache2.py 3 12

# 1、查看torch打屏日志，可有看到经过该算子是否新增guard校验s0为固定值
# 2、也可以开启export TORCH_COMPILE_DEBUG=1，查看torair缓存里的bs是否为s0还是固定值
# 3、跑完一遍想再重新跑之前记得删掉脚本里指定路径下的缓存
