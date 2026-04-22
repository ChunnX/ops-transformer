# 3rd_moe_gather_v2 算子全量分析报告

## 1. 算子概述

**算子名称**: `moe_gather_v2`
**算子类**: 第三方算子（3rd）— MoE GatherV2
**功能描述**: 在 MoE 场景下实现 GatherV2 操作，即根据索引张量从输入张量中按指定轴收集数据。该算子是 PyTorch `torch.gather` 操作的 NPU 适配版本，支持批处理维度和负索引。无自定义 kernel，复用 CANN 内置 GatherV2。

**支持平台**: ascend910b, ascend910_93, ascend950（通过 AiCore/AiCPU 自动分流）
**支持数据类型**: FP16/BF16/FP32/INT8/INT16/INT32/INT64/UINT8/UINT16/UINT32/UINT64/BOOL（输入），INT32/INT64（索引）

---

## 2. 目录结构与文件职责

```
moe/3rd/moe_gather_v2/
├── CMakeLists.txt                              # 顶层构建（标记 aclnn_exclude）
└── op_host/
    ├── CMakeLists.txt                          # 子目录遍历配置
    └── op_api/
        ├── moe_gather_v2.h                     # L0 op 接口声明（19行）
        └── moe_gather_v2.cpp                   # AiCore/AiCPU 双路径 + Shape 推导（121行）
```

注：本算子无 `op_kernel/` 目录、无 `*_tiling.cpp`、无 `*_def.cpp`、无 `*_infershape.cpp`。复用 CANN 内置 GatherV2 kernel。

---

## 3. Host 侧分析

### 3.1 算子定义

该算子**没有**标准的 `_def.cpp` 文件。算子定义通过 `OP_TYPE_REGISTER(GatherV2)` 复用 CANN 内置的 GatherV2 算子注册信息。

**输入输出接口**:

| 方向 | 名称 | 参数类型 | 数据类型 | 形状 |
|------|------|----------|----------|------|
| 输入 | self | REQUIRED | 11+ 种（见下） | 任意 |
| 参数 | axis | 标量 | INT64 | - |
| 输入 | indices | REQUIRED | INT32/INT64 | 任意 |
| 输出 | 输出 | 自动推导 | 同 self | 由 Shape 推导决定 |

**属性**（通过内部参数传递）:

| 属性名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| batchDims | int | 0 | 批处理维度数 |
| negativeIndexSupport | bool | false | 是否支持负索引 |

### 3.2 Tiling 策略

无自定义 Tiling。复用 CANN 内置 GatherV2 的 Tiling 逻辑。

### 3.3 Shape 推导 (moe_gather_v2.cpp, 第 88-120 行)

```
输入: self.shape = [d0, d1, ..., dn], indices.shape = [i0, i1, ..., im]
      axis = k, batchDims = b

输出 shape:
  outShape[0..k-1]  = self.shape[0..k-1]        (axis 之前的维度)
  outShape[k..]     = indices.shape[b..]         (indices 去掉 batchDims 的维度)
  outShape[最后部分] = self.shape[k+1..]          (axis 之后的维度)

特殊: self 为零维 tensor 时, out 始终为零维 tensor
```

### 3.4 ACLNN API

本算子作为 L0 op（`ACLNNTYPE aclnn_exclude`），不暴露独立 ACLNN 接口，被其他算子内部调用。

---

## 4. Kernel 侧分析

### 4.1 Kernel 入口与初始化

无自定义 kernel。通过 `ADD_TO_LAUNCHER_LIST_AICORE(GatherV2, ...)` 或 `ADD_TO_LAUNCHER_LIST_AICPU(GatherV2, ...)` 调用 CANN 内置 GatherV2。

### 4.2 核心计算流程

**执行路径选择**:
```
MoeGatherV2 入口
  ├─ IsAiCoreSupport() == true → AiCore 路径
  │   (类型检查 + 平台检查通过)
  │   输入: self, indices, axis(转为 tensor)
  │   输出: gatherV2Out
  │
  └─ AiCore 不支持 → AiCPU 路径
      属性: batch_dims, negative_index_support
```

### 4.3 数据结构与内存布局

无自定义数据结构。

### 4.4 多核并行策略

由 CANN 内置 GatherV2 kernel 决定。

### 4.5 通信模式

由 CANN 内置 GatherV2 kernel 决定。

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构定义

无自定义 Tiling 数据结构。

### 5.2 Workspace 布局

由 CANN 内置 GatherV2 管理，无额外算子级 workspace。

---

## 6. 完整数据流

```
  self [d0, d1, ..., dn]   indices [i0, i1, ..., im]   axis (int64)
       │                         │                          │
       └─────────────────────────┼──────────────────────────┘
                                   │
                         Shape 推导
                         axis 转为 Tensor(DT_INT64, 1元素)
                                   │
                    ┌──────────────┴──────────────┐
                    │                              │
             IsAiCoreSupport?                 不支持
                    │                              │
             AiCore 路径                     AiCPU 路径
             GatherV2 kernel                 GatherV2 CPU
                    │                              │
                    └──────────────┬──────────────┘
                                   │
                            输出 tensor
                            (由 Shape 推导决定 shape)
```

---

## 7. 性能相关设计

1. **AiCore/AiCPU 自动分流**: 通过 `IsAiCoreSupport()` 检查数据类型和平台，自动选择 NPU kernel 或 CPU 回退路径，最大化利用硬件加速。

2. **L0 op 设计**: 作为 Level-0 操作注册，不生成独立 ACLNN 接口，减少构建开销，仅被上层算子内部调用。

3. **平台差异化数据类型**: 非 arch3510 平台支持 11 种数据类型，arch3510 平台额外支持 BF16/DOUBLE/COMPLEX64，共 12+ 种。

---

## 8. 依赖关系

```
moe/3rd/moe_gather_v2/
└── op_host/op_api/
    ├── moe_gather_v2.h           ← L0 op 接口声明
    │   └── aclnn_launcher.h
    └── moe_gather_v2.cpp         ← 实现
        ├── aclnn_launcher.h      (ADD_TO_LAUNCHER_LIST_AICORE/AICPU)
        ├── OP_TYPE_REGISTER(GatherV2)  ← CANN 内置算子注册
        └── IsAiCoreSupport()     ← 平台/类型检查
```

---

## 9. 关键约束与注意事项

| 约束 | 说明 |
|------|------|
| 算子定义 | 无 _def.cpp，复用 CANN 内置 GatherV2 |
| Tiling | 无自定义 Tiling |
| Kernel | 无自定义 kernel，复用 CANN 内置 |
| BF16 | 仅在 ascend910b 和 ascend910_93 上支持 |
| indices 类型 | 仅 INT32/INT64 |
| 构建标记 | CMakeLists.txt 中 `ACLNNTYPE aclnn_exclude` |
| 负索引 | 默认不支持（negativeIndexSupport=false） |

---

## 10. 与同族变体的核心差异

| 维度 | moe_gather_v2（本算子） | 标准 GatherV2 |
|------|------------------------|---------------|
| 位置 | moe/3rd/ 目录下 | CANN 内置 |
| 自定义 kernel | 无（复用内置） | CANN 内置 |
| 自定义 Tiling | 无 | CANN 内置 |
| 构建标记 | aclnn_exclude | 标准 |
| 调用方式 | L0 op，被其他算子调用 | 独立算子 |
| 文件数量 | 4 个文件 | N/A |
