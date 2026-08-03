# moe_init_routing 算子 active_num SymInt 改造设计说明

## 1. 背景

`moe_init_routing`（V4）的 `active_num` 参数原为 `int` 类型。在 `torch.compile(dynamic=True)` 图模式下，PyTorch dispatcher 会将 `SymInt` 强制 concretize 成具体整数，导致 dynamo guard 把 `active_num` 烧死成常量值。不同 batch_size 触发重编译，编译缓存无法复用，影响动态 shape 场景的编译效率。

### 问题复现

```python
# 图模式下 active_num=x.size(0)*k 为 SymInt
model = torch.compile(model, backend=npu_backend, dynamic=True)
model(x, expert_idx, active_num=x.size(0) * k)
```

改造前 guard 日志：
```
- active_num == 12  # guard 烧死成具体值
```
不同 bs（如 3、12、24）各编译一次，缓存无法复用。

## 2. 改造目标

将 `active_num` 从 `int` 改为 `SymInt`，使其在图模式下保持符号传播，guard 只记录符号关系（如 `s0 * 4`）而非具体值，从而实现不同 batch_size 复用同一份编译图。

## 3. 改造方案

### 3.1 改动文件清单

| 文件 | 改动点 |
|---|---|
| `ops/moe_init_routing.py` | schema `int`→`SymInt`；公开 API 标注 `int`→`Union[int, torch.SymInt]`；docstring 同步 |
| `ops/graph_convert/graph_convert_moe_init_routing.py` | converter 标注 `int`→`Union[int, Tensor]`；`ge.Const`→`dtype_promote` |
| `docs/zh/moe_init_routing.md` | 函数原型与参数说明同步类型变更 |

### 3.2 各层改动详解

#### 3.2.1 schema 层（`moe_init_routing.py:267`）

```diff
- "int active_num=-1, int expert_capacity=-1, ..."
+ "SymInt active_num=-1, int expert_capacity=-1, ..."
```

**作用**：让 dispatcher 在图模式下允许 `active_num` 作为符号整数传播，不 concretize。

**为何不改 C++**（`ops/csrc/moe_init_routing.cpp`）：
- C++ pybind 函数只在 eager 调用，eager 下 `active_num` 恒为具体 `int`（`SymInt` 在非 tracing 上下文无法构造）
- 图模式走 fx2ge converter，不经过 C++ pybind
- Meta 里 `active_num` 不参与输出 shape 推导（输出用 `bs*k`），不受影响

#### 3.2.2 公开 API 层（`moe_init_routing.py:342`）

```diff
- active_num: int = -1,
+ active_num: Union[int, torch.SymInt] = -1,
```

**作用**：静态类型检查与 IDE 提示对齐，标注 `active_num` 可接受符号值。运行时无影响（转发到 `torch.ops` 走 schema）。

#### 3.2.3 图模式 converter 层（`graph_convert_moe_init_routing.py`）

```diff
- from typing import Optional
+ from typing import Optional, Union
+ from torchair._ge_concrete_graph.utils import dtype_promote

  active_num: int = -1,
+ active_num: Union[int, Tensor] = -1,

- active_num_tensor = ge.Const(active_num, dtype=DataType.DT_INT64)
+ active_num_tensor = dtype_promote(active_num, target_dtype=DataType.DT_INT64)
```

**作用**：
- `Union[int, Tensor]` 标注：对齐 torchair 约定，fx2ge 把 SymInt 以 `int`（具体）或 `Tensor`（符号）两种形式传给 converter
- `dtype_promote` 替换 `ge.Const`：`ge.Const` 对 `Tensor` 入参会抛错，无法承载符号值；`dtype_promote` 对 int→Const、Tensor→Cast/直传 统一处理，是 torchair 处理 SymInt 标量的标准方式（参考 `hcom_allgather` converter）

## 4. 两条执行路径

| 路径 | active_num 类型 | 调用链 |
|---|---|---|
| eager | `int`（SymInt 已 concretize） | `_moe_init_routing` → `_op_module.moe_init_routing`（C++ `int64_t`）→ `ACLNN_CMD` |
| graph | `SymInt`（符号） | `convert_moe_init_routing` → `dtype_promote` 转 GE tensor → `MoeInitRoutingV4` 算子 |

两条路径互斥：eager 不经 converter，graph 不经 C++ pybind。

## 5. 对上层调用脚本的影响

**无破坏性变更**，既有用法完全兼容：

- **eager 模式**：`active_num=128`，传 int，行为与改前一致
- **图模式**：`active_num=x.size(0) * k`，`dynamic=True` 下 `x.size(0)` 返回 `SymInt`，自动走符号传播

上层调用脚本无需感知类型差异，同一份代码在两种模式下均可正常运行。

## 6. 验证

改造后 guard 日志变为符号形式：
```
# guard 不再出现 active_num == 12 这类值比较
# 不同 bs 复用同一份编译图，无 recompiles 日志
```

### 验证脚本

#### 脚本1：verify_symint_active_num.py

基于 `torch.compile` + `torchair.get_npu_backend`，对比不同 n 下的 graphs 计数与耗时：

```python
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
```

**判断标准**：
- 生效：`[2][3]` graphs 不增加 + 无 recompiling 日志 + 耗时显著低于 `[1]`
- 未生效：`[2][3]` graphs 增加 + 出现 recompiling 日志 + 耗时接近 `[1]`

#### 脚本2：demo_simple_cache2.py

基于 `torch.npu.npugraph_ex.inference.cache_compile`，配合 `cache_dir` 持久化编译图：

```python
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
```

**运行方式**：
```bash
rm -rf .torchair_cache2
python demo_simple_cache2.py 3 12
```

**判断标准**：
- 生效：`测试 bs=12` 段无 `__recompiles` 日志 + guard 为符号形式（`s0 * 4`）+ 耗时显著低于首次编译
- 未生效：`测试 bs=12` 段出现 `__recompiles` 日志 + guard 为值比较（`active_num == 12`）+ 耗时接近首次编译
