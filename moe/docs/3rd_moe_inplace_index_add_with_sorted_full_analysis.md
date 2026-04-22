# 3rd_moe_inplace_index_add_with_sorted 算子全量分析报告

## 1. 算子概述

**算子名称**: `MoeInplaceIndexAddWithSorted`
**算子类**: 第三方算子（3rd），预排序索引原地加法
**功能描述**: 基于**预排序**索引的原地索引加法。与 `moe_inplace_index_add` 的区别在于接收已排序的 `sorted_indices` 和 `pos` 位置信息，避免 kernel 内排序操作。核心语义: `var[sorted_indices[pos[i]]] += alpha * updates[pos[i]]`

**两种 Kernel 模式**:
- **Avg 模式 (TilingKey=1/4/5)**: 使用硬件原子加（`SetAtomicAdd`），各核按索引平均分配
- **Fix 模式 (TilingKey=2/3/6)**: 相同索引在同一核处理，避免 FP16/BF16 不支持原子加的问题

**支持平台**: ascend910b, ascend910_93
**支持数据类型**: FP32/FP16/BF16/INT32/INT16

---

## 2. 目录结构与文件职责

```
3rd/moe_inplace_index_add_with_sorted/
├── CMakeLists.txt
├── op_host/
│   ├── moe_inplace_index_add_with_sorted_def.cpp           # 算子定义
│   ├── moe_inplace_index_add_with_sorted_tiling.h          # Tiling数据结构
│   ├── moe_inplace_index_add_with_sorted_tiling.cpp        # Tiling策略
│   └── op_api/
│       ├── moe_inplace_index_add_with_sorted.h             # L0 接口声明
│       └── moe_inplace_index_add_with_sorted.cpp           # AiCore launcher
└── op_kernel/
    ├── moe_inplace_index_add_with_sorted.cpp               # Kernel入口
    ├── moe_inplace_index_add_with_sorted_base.h            # 公共宏
    ├── moe_inplace_index_add_with_sorted_fix.h             # Fix模式kernel
    └── moe_inplace_index_add_with_sorted_avg.h             # Avg模式kernel
```

---

## 3. Host 侧分析

### 3.1 算子定义 (moe_inplace_index_add_with_sorted_def.cpp)

**输入输出接口**:

| 方向 | 名称 | 参数类型 | 数据类型 | 说明 |
|------|------|----------|----------|------|
| 输入 | var | REQUIRED（同时为输出） | FP32/FP16/BF16/INT32/INT16 | 目标张量 |
| 输入 | value | REQUIRED | 与 var 相同 | 更新值 |
| 输入 | sorted_indices | REQUIRED | INT32 | 已排序索引 |
| 输入 | pos | REQUIRED | INT32 | 位置信息 |
| 输入 | alpha | OPTIONAL | FP32/INT32/INT16 | 缩放因子 |
| 输出 | var_out | REQUIRED | 与 var 相同 | 输出（原地更新） |

**属性**:

| 属性名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| axis | Int | REQUIRED | 目标轴（**当前仅支持 axis=0**） |

### 3.2 Tiling 策略 (moe_inplace_index_add_with_sorted_tiling.cpp)

**TilingData 结构**（tiling.h 第21-44行）:

```
MoeInplaceIndexAddWithSortedTilingData
├── usedCoreNum        : int32_t   # 实际使用核数
├── enableAlpha        : int32_t   # 是否启用 alpha
├── eachIndexCount     : int64_t   # 每核索引数（整核）
├── lastIndexCount     : int64_t   # 尾核索引数
├── batchCount         : int64_t   # 批次数
├── eachBatchNum       : int64_t   # 每核批次数
├── lastBatchNum       : int64_t   # 尾核批次数
├── inputCount         : int64_t   # 输入总元素数
├── indicesCount       : int64_t   # 索引总数
├── updatesCount       : int64_t   # 更新值总元素数
├── updatesOneTime     : int64_t   # 每次更新元素数
├── maxSize            : int64_t   # UB 单次最大处理数
├── eachNum            : int64_t   # 每循环处理元素数
├── eachLoop           : int64_t   # 数据搬运循环次数
├── eachTail           : int64_t   # 尾循环元素数
├── varDimNum          : int64_t   # var 在 axis 维的大小
├── eachUBIndexRound   : int64_t   # 整核 UB 索引搬运轮次
├── eachUBIndexCount   : int64_t   # 每轮索引数（整核）
├── eachUBIndexTail    : int64_t   # 整核尾轮索引数
├── lastUBIndexRound   : int64_t   # 尾核 UB 索引搬运轮次
├── lastUBIndexCount   : int64_t   # 尾核每轮索引数
└── lastUBIndexTail    : int64_t   # 尾核尾轮索引数
```

**分核策略**（tiling.cpp 第241-274行）:
```
usedCoreNum = min(indicesCount, coreNum)
eachIndexCount = ceil(indicesCount, usedCoreNum)
UB索引分段: UB_INDEX_NUM = 1536
eachUBIndexRound = ceil(idxCount, 1536)
```

**UB 缓冲数量映射**:
| TilingKey | bufCnt | 说明 |
|-----------|--------|------|
| 1 (FP32) | 2 | inputBuf + valueBuf |
| 2 (FP16) | 7 | + outBuf + castSelf + castValue + calcBuf*2 |
| 3 (BF16) | 7 | 同 FP16 |
| 4 (INT16) | 2 | |
| 5 (INT32) | 2 | |
| 6 (FIX_FP32) | 3 | inputBuf + valueBuf + outBuf |

**TilingKey 选择**（tiling.cpp 第193-239行）:
```
deterministic==1 && DT_FLOAT → 6
DT_FLOAT → 1, DT_FLOAT16 → 2, DT_BF16 → 3
DT_INT16 → 4, DT_INT32 → 5
```

### 3.3 Shape 推导

无独立 infershape 文件。`var_out` shape 与 `var` 相同。

### 3.4 ACLNN API

L0 接口声明在 `op_api/moe_inplace_index_add_with_sorted.h`，AiCore launcher 在 `.cpp`。无独立 ACLNN 两段式 API 文档。

---

## 4. Kernel 侧分析

### 4.1 Kernel 入口 (moe_inplace_index_add_with_sorted.cpp)

**入口函数**（第19-63行）: 接收 5 个 GM_ADDR + workspace + tiling。

**TilingKey 分发**:

| Key | 模板类 | 模式 |
|-----|--------|------|
| 1 | `MoeInplaceIndexAddWithSortedAvg<FP32>` | Avg（原子加） |
| 2 | `MoeInplaceIndexAddWithSortedFix<FP16>` | Fix（Cast FP32） |
| 3 | `MoeInplaceIndexAddWithSortedFix<BF16>` | Fix（Cast FP32） |
| 4 | `MoeInplaceIndexAddWithSortedAvg<INT16>` | Avg |
| 5 | `MoeInplaceIndexAddWithSortedAvg<INT32>` | Avg |
| 6 | `MoeInplaceIndexAddWithSortedFix<FP32>` | Fix（确定性） |

### 4.2 核心计算流程

**Avg 模式**（avg.h）:
```
for eachLoop 次数据搬运:
  AvgProcess(loopIdx):
    for 每轮UB索引:
      搬入 sorted_indices 和 pos 到 UB
      for 每个 index:
        if 索引变化: CopyOut 上一个索引的结果
        CopyValueIn(pos[i]) → out += alpha * value
    CopyOut(最终索引)
```

**Fix 模式**（fix.h）:
```
AvgProcess(i):           -- 处理自己的索引段
  for 每轮UB索引:
    if 核id!=0 且 index==lastCoreIndex: 跳过
    CopySelfIn → CopyValueIn → self += alpha * value

SameIndexProcess(i):     -- 处理后面核中相同索引
  while lastIndex == nextIndex:
    从 GM 读取 → Compute
  CopyOut(lastIndex)
```

### 4.3 数据结构与内存布局

Avg 模式使用 `SetAtomicAdd` 保证多核写正确性。Fix 模式通过 `SameIndexProcess` 跨核处理共享索引边界，确保同一 var 位置只由一个核写入。

FP16/BF16 路径在 Fix 模式中使用 Cast: `input→FP32→累加→Cast回原类型`。

### 4.4 多核并行策略

按 indicesCount 均匀分配核。`eachIndexCount` 为每核索引数，`lastIndexCount` 为尾核索引数。

### 4.5 通信模式

**Avg 模式**: 使用硬件原子加，无需核间同步。
**Fix 模式**: 通过 `SameIndexProcess` 处理跨核共享索引，无 SyncAll。每核在 `AvgProcess` 中跳过与其他核共享的首个索引，在 `SameIndexProcess` 中处理后续核的共享数据。

---

## 5. Host-Kernel 接口

### 5.1 Tiling 数据结构定义

见 3.2 节 TilingData 结构（22 个字段）。

**CompileInfo**（tiling.h 第47-51行）:
```
MoeInplaceIndexAddWithSortedCompileInfo
├── totalCoreNum    : int32_t  (默认 40)
├── ubSizePlatForm  : uint64_t
└── workspaceSize   : uint64_t
```

### 5.2 Workspace 布局

无额外算子级 workspace（tiling.cpp 使用系统 workspace）。

---

## 6. 完整数据流

### 6.1 Avg 模式 (FP32/INT16/INT32)

```
  var [N0, H]    sorted_indices [M]    pos [M]    updates [M, H]
       |                |                  |             |
       v                v                  v             v
  +---------------------------------------------------------------+
  | Kernel (Avg, 多核原子加)                                      |
  |                                                               |
  | per core:                                                     |
  |   for pos[i]:                                                 |
  |     idx = sorted_indices[pos[i]]                              |
  |     value = updates[pos[i]]                                   |
  |     SetAtomicAdd → var[idx] += alpha * value                  |
  |     SetAtomicNone                                             |
  +---------------------------------------------------------------+
       |
       v
  var_out [N0, H] (原地更新)
```

### 6.2 Fix 模式 (FP16/BF16/确定性FP32)

```
  var [N0, H]    sorted_indices [M]    pos [M]    updates [M, H]
       |                |                  |             |
       v                v                  v             v
  +---------------------------------------------------------------+
  | Kernel (Fix, 同索引同核)                                      |
  |                                                               |
  | per core:                                                     |
  |   AvgProcess: 处理自己的索引段 (跳过共享边界)                  |
  |     CopySelfIn(var[idx]) → Cast→FP32 → += alpha*value         |
  |   SameIndexProcess: 处理后面核中相同索引                       |
  |     从GM读取 → += alpha*value                                 |
  |   CopyOut: Cast回原类型 → 写回 var[idx]                       |
  +---------------------------------------------------------------+
       |
       v
  var_out [N0, H] (原地更新)
```

---

## 7. 性能相关设计

1. **排序索引优化**: 前置排序使相同 index 连续排列，减少 cache miss，避免 kernel 内排序开销。

2. **双模式设计**: FP32/INT16/INT32 使用硬件原子加（简单高效），FP16/BF16 使用 Fix 模式避免不支持原子加的问题。

3. **Fix 模式跨核处理**: `SameIndexProcess` 处理跨核共享索引，无需 SyncAll，减少同步开销。

4. **UB 索引分段**: 固定预留 1536 个索引空间，当每核索引超过 1536 时自动分轮搬运，避免 UB 溢出。

5. **FP16/BF16 精度提升**: Fix 模式中 Cast 到 FP32 计算，减少累加精度损失。

---

## 8. 依赖关系

```
op_host/moe_inplace_index_add_with_sorted_tiling.cpp
└── moe_inplace_index_add_with_sorted_tiling.h
    ├── register/op_impl_registry.h
    └── tiling_base/tiling_base.h

op_host/op_api/moe_inplace_index_add_with_sorted.cpp
├── moe_inplace_index_add_with_sorted.h
└── external/aclnn_kernels/aclnn_platform.h

op_kernel/moe_inplace_index_add_with_sorted.cpp
├── moe_inplace_index_add_with_sorted_avg.h
│   └── moe_inplace_index_add_with_sorted_base.h
│       └── kernel_operator.h
├── moe_inplace_index_add_with_sorted_fix.h
│   └── moe_inplace_index_add_with_sorted_base.h
└── kernel_operator.h
```

---

## 9. 关键约束与注意事项

| 约束 | 说明 |
|------|------|
| axis 限制 | 当前仅支持 axis=0（tiling.cpp 第152-155行） |
| sorted_indices | 必须已排序，否则结果不正确 |
| 确定性模式 | 仅 FP32 (TilingKey=6) |
| alpha 可选 | enableAlpha 由 Tiling 判断 |
| 原子加限制 | FP16/BF16 不支持原子加，必须走 Fix 模式 |
