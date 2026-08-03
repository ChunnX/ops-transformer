"""
复现: torch.compile + npugraphex + dynamic=True — bs 被 MoE 算子 guard 固化
cachecompile 版本（基于 torchair.inference.cache_compile）

针对 cann_ops_transformer.moe_init_routing（V4, SymInt active_num）验证。

用法: python demo_simple_cache2.py <bs> [bs2]

  python demo_simple_cache2.py 6        # 编译 bs=6
  python demo_simple_cache2.py 6 4      # 编译 bs=6，再跑 bs=4 看是否泛化

Run: python demo_simple_cache2.py 6
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


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 128)

    def forward(self, hidden_states, topk_idx):
        token_count, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        expanded_x, _, _, _, _ = moe_init_routing(
            hidden_states, topk_idx,
            active_num=token_count * 8,
            # active_num=-1,
            expert_num=256,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[0, 256],
            quant_mode=-1,
        )
        return self.linear(expanded_x)


model = Model().npu()

# 使用 torchair.inference.cache_compile 包装 forward
cached_forward = torch.npu.npugraph_ex.inference.cache_compile(
    model.forward,
    dynamic=True,
    cache_dir=CACHE_DIR,
)

print(f"运行 bs={bs}")
hidden_states = torch.randn(bs, 128).npu()
topk_idx = torch.randint(0, 256, (bs, 8), dtype=torch.int32).npu()

# 显式标记 batch dim 为动态，防止被 MoE 算子 guard 固化
#torch._dynamo.mark_dynamic(hidden_states, 0)
#torch._dynamo.mark_dynamic(topk_idx, 0)

y = cached_forward(hidden_states, topk_idx)

print(f"output=============: {list(y.shape)}")

# 可选：第二个 bs 测试泛化
if len(sys.argv) > 2:
    bs2 = int(sys.argv[2])
    print(f"\n测试========== bs={bs2}")
    try:
        hidden_states2 = torch.randn(bs2, 128).npu()
        topk_idx2 = torch.randint(0, 256, (bs2, 8), dtype=torch.int32).npu()
        torch._dynamo.mark_dynamic(hidden_states2, 0)
        torch._dynamo.mark_dynamic(topk_idx2, 0)
        y2 = cached_forward(hidden_states2, topk_idx2)
        print(f"output========: {list(y2.shape)}")
    except Exception as e:
        print(f"泛化失败: {type(e).__name__}: {e}")


# python  demo_simple_cache2.py 6 4

# 1、查看torch打屏日志，可有看到经过该算子是否新增guard校验s0为固定值
# 2、也可以开启export TORCH_COMPILE_DEBUG=1，查看torair缓存里的bs是否为s0还是固定值
# 3、跑完一遍想再重新跑之前记得删掉脚本里指定路径下的缓存
