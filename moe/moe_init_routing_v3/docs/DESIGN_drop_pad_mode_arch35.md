# MoeInitRoutingV3 算子 A5 版本非量化模板 DropPad 模式需求设计文档

## 文档信息

| 项目名称 | MoeInitRoutingV3 算子  DropPad 模式支持 |
|---------|-----------------------------------------------------|
| 文档版本 | V1.0                                                |
| 创建日期 | 2026-05-14                                          |
| 作者     | CANNBot                                             |
| 适用范围 | Ascend 950PR/Ascend 950DT (arch35)                  |

---

## 1. 需求概述

### 1.1 需求背景

当前 MoeInitRoutingV3 算子在 arch35 版本（Ascend 950PR/Ascend 950DT）仅支持 `drop_pad_mode=0`（Dropless 场景）。实际 MoE 模型推理场景中存在 DropPad 模式需求：需要对每个专家处理的 Token 数量进行 capacity 对齐，超过的丢弃，不足的零填充。

### 1.2 需求描述

**功能需求**：MoeInitRoutingV3 算子在 A5 版本支持 `drop_pad_mode=1` 场景。

**约束条件**：
1. 仅限 arch35 硬件版本（Ascend 950PR/Ascend 950DT）
2. 第一优先级：支持非量化模式（`quantMode=-1`），第二优先级：支持非量化模式MXFP8透传、MXFP8量化、MXFP4透传、MXFP4量化。第三优先级：（全载模板优先级待确认）其他
3. 需校验 `expertNum` 和 `expertCapacity` 参数有效性
4. `expandedXOut` 输出为 3D Tensor：`[expertNum, expertCapacity, h]`。需要对每个专家处理的 Token 数量进行 capacity 对齐，超过的丢弃，不足的零填充。
5. `expandedRowIdxOut` shape 保持 `[numRows * k]`，但内容需反映 DropPad 映射关系
6. `expandedScaleOut` 输出为 1D Tensor：`[expertNum * expertCapacity]`。需要对每个专家处理的 Token 对应的scale数量进行 capacity 对齐，超过的丢弃，不足的零填充。
7. `expertTokensCountOrCumsumOut`: Drop 前的原始 token 数量
8. 仅支持 `rowIdxType=0`（gather 索引），不支持 `rowIdxType=1`（scatter 索引）
9. activeExpertRange必须为 [0, expertNum]

### 1.3 参考实现

- **moe_init_routing_v3** 非 arch35 版本已有 DropPad 实现（非量化、动态量化、静态量化）

---

## 2. 功能分析

### 2.1 DropPad 模式计算流程

```
输入：
  x: [numRows, h]           - Token 特征输入
  expertIdx: [numRows, k]   - 每个 Token 对应的 k 个专家索引

处理流程：
1. 排序阶段：对 expertIdx 排序，得到 sortedExpertIdx 和 sortedRowIdx
2. 统计阶段：统计每个专家的 Token 数量（expertTokensCount）不受dropPad影响
3. DropPad 处理：
   - 对每个专家，保留前 expertCapacity 个 Token
   - 超过 capacity 的 Token 被丢弃
   - 不足 capacity 的位置用零填充
4. 受DropPad输出：
   - expandedXOut: [expertNum, expertCapacity, h]
   - expandedRowIdxOut: [numRows * k] - 映射关系
   - expertTokensCountOrCumsumOut: Drop 前的原始 token 数量（类似 V2 的 expertTokensBeforeCapacityOut）
   - expandedScaleOut：[expertNum * expertCapacity] 与expandedXOut类似的处理，保证expandedXOut一一对应

输出：
  expandedXOut: [expertNum, expertCapacity, h]
  expandedRowIdxOut: [numRows * k]
  expertTokensCountOrCumsumOut
  expandedScaleOut：[expertNum * expertCapacity]
```

### 2.2 与 Dropless 模式的差异对比

| 特性 | Dropless (drop_pad_mode=0) | DropPad (drop_pad_mode=1) |
|------|----------------------------|---------------------------|
| expandedXOut shape | 2D `[n*k, h]` | 3D `[expertNum, expertCapacity, h]` |
| expandedScaleOut shape | 1D `[n*k]` 或不输出 | **2D `[expertNum, expertCapacity]`**（与 expandedXOut 一一对应） |
| 输出大小 | 取决于 activeNum 或 n*k | 固定为 expertNum × expertCapacity |
| Token 处理 | 全部保留 | 按 capacity 对齐，Drop 或 Pad |
| 零填充 | 无 | 不足 capacity 时零填充 |
| expertCapacity 参数 | 不校验 | 必须校验且 ∈ (0, numRows] |
| activeExpertRange | 可指定范围 | **必须为 [0, expertNum]** |
| rowIdxType | 0 或 1 | **仅支持 0（gather）** |
| expertTokensCountOrCumsumOut | Cumsum/Count/KeyValue | Cumsum/Count/KeyValue （Drop 前原始数量）** |

### 2.3 参数约束

| 参数 | Dropless 模式约束 | DropPad 模式约束 |
|------|-------------------|------------------|
| expertCapacity | 不校验 | > 0 且 ≤ numRows |
| expertNum | > 0 且 ≤ 10240（或 5120） | > 0 且 ≤ 10240 |
| activeExpertRange | [expertStart, expertEnd) ∈ [0, expertNum] | **必须为 [0, expertNum]** |
| rowIdxType | 0 或 1 | **必须为 0（gather）** |
| quantMode | -1, 0, 1, 2, 3, 6-9 | 第一优先级：-1（非量化）；第二优先级：2,3,9（MXFP8/MXFP4）；第三优先级：待确认 |
| expertTokensNumType | 0, 1, 2 | **是否都支持？？？（COUNT）** |

---

## 3. 技术方案

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    MoeInitRoutingV3                         │
│                    (arch35 DropPad 模式)                    │
├─────────────────────────────────────────────────────────────┤
│  Tiling 层                                                   │
│  ├─ CheckSetAttrs() - 参数校验                               │
│  ├─ CheckOutputs() - 输出校验                                │
│  ├─ Tiling4SrcToDstDropPadCompute() - DropPad Tiling        │
│  └─ GetTilingKey() - TilingKey 生成                         │
├─────────────────────────────────────────────────────────────┤
│  Kernel 层                                                   │
│  ├─ moe_v3_full_load_unquantized_droppad.h - 全载模板 (暂不处理)      │
│  ├─ MoeRowIdxGatherOutDroppad - RowIdx 处理                 │
│  ├─ MoeGatherOutDroppad - expandedXOut & expandedScaleOut输出                 │
│  └─ ExpertTokensCount - expertTokensCountOrCumsumOut 输出                   │
└─────────────────────────────────────────────────────────────┘
```
具体修改文件列表和每个文件的修改逻辑
```
moe/moe_init_routing_v3/
├── op_host/
│   ├── moe_init_routing_v3_tiling_arch35.cpp    # 参考moe/moe_init_routing_v3/moe_init_routing_v3_tiling.cpp 中droppad实现
│   └── moe_init_routing_v3_infershape.cpp       # 无需修改（已支持）
├── op_kernel/
│   ├── arch35/
│   │   ├── moe_init_routing_v3_arch35_tiling_def.h  # 新增 TilingData
│   │   ├── 新增文件moe_v3_row_idx_gather_droppad.h  # 在moe\moe_init_routing_v3\op_kernel\arch35\moe_v3_row_idx_gather.h基础上新增droppad实现，可参考moe\moe_init_routing_v3\op_kernel\moe_v3_row_idx_gather_droppad.h中的droppad实现
│   │   ├── moe_v3_expert_tokens_count.h  # 在moe\moe_init_routing_v3\op_kernel\arch35\moe_v3_expert_tokens_count.h基础上新增droppad实现，可参考moe\moe_init_routing_v3\op_kernel\moe_v3_expert_tokens_count.h中的droppad实现
│   │   └── 新增文件moe_v3_gather_out_droppad.h   # 在moe\moe_init_routing_v3\op_kernel\arch35\moe_v3_gather_out.h基础上新增droppad实现，可参考moe\moe_init_routing_v3\op_kernel\moe_v3_gather_out_droppad.h中的droppad实现
│   └── moe_init_routing_v3_apt.cpp           # 新增droppad实现，可参考moe\moe_init_routing_v3\op_kernel\moe_init_routing_v3中的droppad实现
├── docs/
│   ├── aclnnMoeInitRoutingV3.md                 # 更新文档
└── tests/
    └── ut/
        ├── op_host/
        │   ├── test_moe_init_routing_v3_arch35_tiling.cpp  # 新增测试
        │   └── test_moe_init_routing_v3_infershape.cpp     # 补充 DropPad 测试
        └── op_kernel/
            └── test_moe_init_routing_v3.cpp                # 新增 Kernel 测试
```

