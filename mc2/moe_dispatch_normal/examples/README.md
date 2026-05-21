# MoE Prefill 示例

本目录包含 MC2 (Moe Communication Collective 2) 算子在 MoE prefill 场景下的测试示例。

## 目录结构

```
mc2/
├── moe_dispatch_normal/examples/    # 主prefill示例和共享工具
│   ├── utils.py                     # 通用辅助函数 (bench, init_dist 等)
│   ├── moe_prefill_sample.py        # 完整prefill workflow测试
│   ├── moe_dispatch_prefill_sample.py  # 引用 moe_prefill_sample.py
│   └── README.md                    # 本文件
│
├── notify_dispatch/examples/
│   └── notify_dispatch_sample.py    # Notify dispatch 独立测试
│
├── dispatch_layout/examples/
│   ├── dispatch_layout_sample.py        # 独立布局测试
│   ├── get_dispatch_layout_sample.py    # 引用 moe_prefill_sample.py
│   └── README.md
│
└── moe_combine_normal/examples/
    └── moe_combine_prefill_sample.py    # 引用 moe_prefill_sample.py
```

## Workflow

完整的 MoE prefill workflow 包含三个集成的算子：

1. **get_dispatch_layout**: 从 top-k expert indices 计算分发布局
   - 输入: `topkIdx`, `numExperts`, `numRanks`
   - 输出: `numTokensPerExpert`, `sendTokenIdxSmall`

2. **moe_dispatch_prefill**: 将 token 分发到 expert (组合 notify_dispatch + moe_dispatch_normal)
   - 输入: `x`, `topkIdx`, `topkWeights`, `numTokensPerExpert`, `sendTokenIdxSmall`, `groupEp`, `rank`, `numRanks`
   - 输出: `expandxOut`, `dynamicScalesOut`, `expandIdxOut`, `recvCount`, `recvTokensPerExpert`

3. **moe_combine_prefill**: 合并 expert 输出 (封装 moe_combine_normal)
   - 输入: `x`, `topkIdx`, `topkWeights`, `srcIdx`, `sendHead`, `groupEp`, `rank`, `numRanks`
   - 输出: `combinedX`

## 测试分类

### 可独立测试（无workflow依赖）

| 算子 | 原因 | 测试命令 |
|------|------|----------|
| `dispatch_layout` | 仅计算布局，不涉及实际数据通信 | `python dispatch_layout_sample.py --test-type dispatch_layout` |
| `get_dispatch_layout` | 仅计算布局，简化版本 | `python dispatch_layout_sample.py --test-type get_dispatch_layout` |
| `notify_dispatch` | 通信协调算子，可独立验证通信功能 | `python notify_dispatch_sample.py` |

### 必须在完整workflow中测试（有依赖）

| 算子 | 依赖链 | 原因 |
|------|--------|------|
| `moe_dispatch_normal` | notify_dispatch → moe_dispatch_normal | 需要 notify_dispatch 的输出 (recvOffset, recvCount) |
| `moe_combine_normal` | moe_dispatch_normal → moe_combine_normal | 需要 dispatch 的输出 (expandIdxOut, recv_x) |
| `moe_dispatch_prefill` | 内部调用: notify_dispatch + moe_dispatch_normal | 组合算子，自动处理依赖 |
| `moe_combine_prefill` | 需要 dispatch_prefill 的输出 | 组合算子 |

### Workflow依赖图

```
get_dispatch_layout (可独立测试)
        ↓
        numTokensPerExpert, sendTokenIdxSmall
        ↓
notify_dispatch (可独立测试)
        ↓
        recvOffset, recvCount, maxBs
        ↓
moe_dispatch_normal ──→ moe_combine_normal
        ↑                      ↑
        └── moe_dispatch_prefill ──→ moe_combine_prefill ──┘
              (组合算子，自动处理依赖)
```

## 运行测试

### 推荐：完整Workflow测试

测试从布局 → 分发 → 合并的完整流程：

```bash
cd mc2/moe_dispatch_normal/examples
python moe_prefill_sample.py --num-processes 16 --num-tokens 1024 --hidden 7168 --num-topk 8 --num-experts 16
```

### 独立布局测试（轻量）

```bash
# 测试 get_dispatch_layout (简化版本)
cd mc2/dispatch_layout/examples
python dispatch_layout_sample.py --test-type get_dispatch_layout --num-processes 16

# 测试 dispatch_layout (原始版本)
python dispatch_layout_sample.py --test-type dispatch_layout --num-processes 16

# 测试两者
python dispatch_layout_sample.py --num-processes 16
```

### 独立通信测试（中等）

```bash
cd mc2/notify_dispatch/examples
python notify_dispatch_sample.py --num-processes 16 --num-tokens 1024
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|---------|------|
| `--num-processes` | 16 | 启动的进程数量 |
| `--num-tokens` | 1024 | token 数量 |
| `--hidden` | 7168 | 隐藏层维度大小 |
| `--num-topk` | 8 | top-k expert 数量 |
| `--num-experts` | 16 | expert 总数 |
| `--active-ranks` | "" | 逗号分隔的活跃rank列表 (如 "0,1,3") |
| `--test-type` | both | 测试哪个算子 (dispatch_layout/get_dispatch_layout/both) |

## 依赖

- torch
- torch_npu
- torch.distributed (hccl backend)
- npu_ops_transformer (torch_extension)

## 注意事项

1. 测试需要 NPU 硬件和 HCCL 通信支持
2. Prefill workflow 测试端到端正确性，验证 dispatch + combine 返回原始 token 按 expert 选择加权
3. 独立测试 (dispatch_layout, notify_dispatch) 用于底层验证
4. 生产使用建议采用组合算子 (`moe_dispatch_prefill`, `moe_combine_prefill`)，自动处理所有依赖
5. 各算子examples目录下的简化测试脚本通过 sys.path 引用 moe_dispatch_normal/examples 下的共享工具