# 代码检视报告：dispatch_v2 A5 前缀和 CumSum tmp 空间不足修复

**提交**: `1ddf3ab3f` `[MOE] 修复dispatch_v2 A5前缀和CumSum tmp空间不足问题`
**分支**: cumsum
**文件**: 3 个，+27 / -2

---

## 1. 改动概述

### 背景

`SetExpertPrefixSumA5` 中 `statusBuf_` 兼作 CumSum API 的 `sharedTmpBuffer`。原代码在 `epWorldSize_ >= 8` 时触发 CumSum，但 `statusBuf_` 大小为 `moeExpertNumPerRank_ * epWorldSize_ * 32` 字节，CumSum 最小 tmp 要求为 `1024 * Ceil(localExpertNum, 8)` 字节。当 `8 <= epWorldSize_ < 32` 时，tmp 空间不足，导致越界。

### 修复方案

- **tiling 侧**：调 `GetCumSumMaxMinTmpSize` 计算 CumSum 最小 tmp 大小，写入 tiling 字段 `cumsumTmpMinSize`
- **kernel BufferInit**：`statusBufSize = max(原始, cumsumTmpMinSize_)`，超 `totalUbSize_` 则回退原始大小
- **kernel SetExpertPrefixSumA5**：判断 `statusBuf_` 实际大小是否 >= `cumsumTmpMinSize_`，够则走 CumSum，不够退化标量循环

---

## 2. 逐文件检视

### 2.1 moe_distribute_dispatch_v2_tiling.h

```diff
-    bool reserved1;
+    uint32_t cumsumTmpMinSize;    // cumsum min tmp size, 0 means no cumsum
```

| 检查项 | 结论 |
|--------|------|
| 字段替换 | `reserved1` 为保留字段，替换无兼容性问题 |
| 布局影响 | bool(1B) → uint32_t(4B)，后续字段偏移变化。host/kernel 同次编译，无二进制兼容问题 |
| 注释风格 | 与原有英文短注释风格一致 |
| 默认值 | uint32_t 零初始化等价于"不需要 CumSum"，语义正确 |

### 2.2 moe_distribute_dispatch_v2_tiling.cpp

```cpp
// A5 CumSum 最小 tmp 大小
if (tilingData->moeDistributeDispatchV2Info.expertTokenNumsType != 0) {
    tilingData->moeDistributeDispatchV2Info.cumsumTmpMinSize = 0;
} else {
    uint32_t localExpertNumAlign = (localMoeExpertNum + 7U) / 8U * 8U;
    std::vector<int64_t> cumsumShapeDims = {1, static_cast<int64_t>(localExpertNumAlign)};
    auto cumsumSrcShape = ge::Shape(cumsumShapeDims);
    uint32_t cumsumMaxTmp = 0;
    uint32_t cumsumMinTmp = 0;
    GetCumSumMaxMinTmpSize(cumsumSrcShape, sizeof(float), true, false, cumsumMaxTmp, cumsumMinTmp);
    tilingData->moeDistributeDispatchV2Info.cumsumTmpMinSize = cumsumMinTmp;
    OP_LOGD(nodeName, "MTE cumsumTmpMinSize=%u", cumsumMinTmp);
}
```

| 检查项 | 结论 |
|--------|------|
| `localMoeExpertNum` 可用性 | L1939 声明，L1969 `CheckInputParam` 赋值，此处已有效 |
| `expertTokenNumsType != 0` 分支 | 输出原始计数不需前缀和，设 0 正确 |
| `localExpertNumAlign` 计算 | `(x + 7) / 8 * 8` = Ceil(x, 8)，与 kernel `Ceil(localExpertNum * 4, 32) * 8` 等价 |
| `GetCumSumMaxMinTmpSize` 参数 | shape={1, align}, typeSize=4(float), isLastAxis=true, isReuseSource=false，与 kernel `CumSumInfo{1, cumsumInner}` + `SET_EXPERT_TOKEN_NUMS_CUMSUM_CONFIG{true, false, ...}` 一致 |
| 头文件引用 | 删除了 `#include "adv_api/math/cumsum_tiling.h"`，声明通过 `tiling/tiling_api.h`（`moe_distribute_dispatch_tiling_v2.h` 间接引入）可见，编译通过 |
| `cumsumMaxTmp` 未使用 | 调用需要该出参，未使用无问题 |

### 2.3 moe_distribute_dispatch_v2.h

**成员变量 + Init 读取**:
```cpp
uint32_t cumsumTmpMinSize_ = 0;
// Init():
cumsumTmpMinSize_ = tilingData->moeDistributeDispatchV2Info.cumsumTmpMinSize;
```

| 检查项 | 结论 |
|--------|------|
| 位置 | 紧跟 `isPerformanceFlag_`，合理 |
| 类型 | uint32_t，与 tiling 字段一致 |

**BufferInit 扩展 statusBuf**:
```cpp
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    uint32_t statusBufOrigin = moeExpertNumPerRank_ * epWorldSize_ * UB_ALIGN;
    uint32_t statusBufSize = (statusBufOrigin > cumsumTmpMinSize_) ? statusBufOrigin : cumsumTmpMinSize_;
    if (totalUsedUB_ + statusBufSize > totalUbSize_) {
        statusBufSize = statusBufOrigin;
    }
#else
    uint32_t statusBufSize = moeExpertNumPerRank_ * epWorldSize_ * UB_ALIGN;
#endif
```

| 检查项 | 结论 |
|--------|------|
| arch 条件 | 仅 3510(A5) 扩展，非 3510(A3) 保持原逻辑 |
| `totalUsedUB_` 语义 | `BufferInit` 开头 `tpipe_->Reset()` 清空后重新累计，此时值为本阶段已分配 buffer 总和 |
| `totalUbSize_` 语义 | 来自 `GetCoreMemSize(UB)`，是物理 UB 总量。Reset 后本阶段可用全部 UB，比较正确 |
| 溢出判断 | `totalUsedUB_`(uint32) + `statusBufSize`(uint32) 与 `totalUbSize_`(uint32) 比较，UB 不超 256KB，无溢出风险 |
| 回退逻辑 | 超 UB 时回退 `statusBufOrigin`，`SetExpertPrefixSumA5` 会据此退化标量，正确 |

**SetExpertPrefixSumA5 判断**:
```cpp
bool useCumSum = (cumsumTmpMinSize_ > 0U) && (statusBuf_.Get<uint8_t>().GetSize() >= cumsumTmpMinSize_);
```

| 检查项 | 结论 |
|--------|------|
| `cumsumTmpMinSize_ > 0` | 过滤 `expertTokenNumsType != 0` 场景（tiling 设 0） |
| `statusBuf_.GetSize() >= cumsumTmpMinSize_` | 反映 BufferInit 实际分配大小（可能因 UB 不足未扩展），判断正确 |
| 原条件移除 | 原 `epWorldSize_ >= UB_ALIGN / sizeof(float)` 被替换，新条件更准确 |

---

## 3. 路径覆盖分析

| 路径 | tiling 入口 | 是否设 cumsumTmpMinSize | kernel 入口 | 是否用 cumsumTmpMinSize_ | 结论 |
|------|------------|----------------------|-------------|------------------------|------|
| 3510 + CCU | arch35 `MoeDistributeDispatchTilingImpl` | 否（默认 0） | `host_kfc.h` | 否（不走此模板） | 不影响 |
| 3510 + fullmesh | `v2_tiling.cpp` | 是 | `moe_distribute_dispatch_v2.h` (3510 分支) | 是 | **已覆盖** |
| non-3510 + fullmesh | `v2_tiling.cpp` | 是 | `moe_distribute_dispatch_v2.h` (else 分支) | 否（不扩展，不调 SetExpertPrefixSumA5） | 不影响 |

---

## 4. 问题清单

### 4.1 [建议] 共享专家 rank 场景 cumsumTmpMinSize_ 偏大

tiling 中 `localMoeExpertNum` 是非共享专家数，但共享专家 rank 的 kernel 用 `localExpertNum=1`。`cumsumTmpMinSize_` 基于非共享专家数计算，值偏大。

- **影响**：共享专家 rank 可能因 `cumsumTmpMinSize_` 过大导致 statusBuf 扩展失败，退化标量。但 `localExpertNum=1` 时标量循环仅 1 次，性能影响可忽略
- **功能正确性**：不受影响
- **建议**：可接受，无需修改

### 4.2 [确认] `GetCumSumMaxMinTmpSize` 声明可见性

删除 `#include "adv_api/math/cumsum_tiling.h"` 后编译通过。声明通过 `moe_distribute_dispatch_tiling_v2.h` → `tiling/tiling_api.h` 间接引入。依赖间接 include，若上游头文件变更可能断裂。

- **建议**：可接受，`tiling/tiling_api.h` 是稳定公共头

### 4.3 [正确] tiling 与 kernel CumSum 参数一致性

| 参数 | tiling | kernel |
|------|--------|--------|
| shape | {1, Ceil(localExpertNum, 8)} | CumSumInfo{1, cumsumInner} |
| typeSize | sizeof(float) = 4 | CumSum<float> |
| isLastAxis | true | config.isLastAxis = true |
| isReuseSource | false | config 第2字段 = false |

全部一致。

---

## 5. 总结

| 维度 | 评价 |
|------|------|
| 功能正确性 | 通过。修复了 `8 <= epWorldSize_ < 32` 时 CumSum tmp 越界问题，UB 不足时正确退化 |
| 路径覆盖 | 通过。3510+fullmesh 路径已覆盖，CCU 和 non-3510 路径不受影响 |
| 参数一致性 | 通过。tiling 与 kernel 的 CumSum 调用参数完全对齐 |
| UB 安全 | 通过。扩展前判断 `totalUsedUB_ + statusBufSize > totalUbSize_`，超限回退 |
| 退化安全 | 通过。statusBuf 不足时 `useCumSum=false`，走标量循环，功能不受影响 |
| 代码风格 | 通过。注释风格与原文件一致 |

**结论：通过，无阻塞问题。**
