# 代码检视报告

**项目名称**：ops-transformer
**检视 PR**：[#9449](https://gitcode.com/cann/ops-transformer/pull/9449) `[WIP]修复dispatch_v2 A5前缀和CumSum tmp空间不足问题`
**检视模块**：
- `mc2/moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_dispatch_v2_tiling.cpp`
- `mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2.h`
- `mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2_tiling.h`

**作者**：zhong-zixin
**检视人**：CANNBot
**检视日期**：2026-08-03
**检视版本**：commit 839941a76（最新）

---

## 变更概述

本 PR 修复 A5 平台（`__NPU_ARCH__ == 3510`）下 `moe_distribute_dispatch_v2` 算子 CumSum 前缀和计算时 tmp 空间不足的问题。

**改动内容**：
1. **Host 侧（tiling.cpp）**：新增 CumSum 最小 tmp 大小计算，通过 `GetCumSumMaxMinTmpSize` 获取 `cumsumTmpMinSize` 并写入 TilingData，附带 `OP_LOGI` 降级说明。
2. **Kernel 侧（kernel.h）**：新增成员变量 `cumsumTmpMinSize_`，Buffer 分配时扩展 `statusBuf_` 兼作 CumSum tmpBuffer（超 UB 则回退），`useCumSum` 判定从 `epWorldSize_ >= 8` 改为基于实际 tmp 空间判断；`SetExpertPrefixSumA5` 改为接收 `useCumSum` 参数避免冗余查询。
3. **TilingData 结构体（tiling.h）**：在结构体末尾新增 `uint32_t cumsumTmpMinSize` 字段。

---

## 检视概览

| 统计项 | 数值 |
| ---- | ---- |
| 发现问题总数 | 1 个 |
| 严重级（CRITICAL）问题 | 0 个 |
| 中等级（MEDIUM）问题 | 0 个 |
| 轻微级（LOW）问题 | 1 个 |

**核心结论**：本 PR 整体质量良好，修复逻辑正确，整数运算安全，UB 容量有保护，流水同步完整，结构体字段遵循末尾添加原则，ABI 兼容，无阻塞级问题。1 处轻微存疑项需确认。

---

## 正面评价

| 序号 | 评价项 | 证据 |
|------|--------|------|
| 1 | **整数运算安全** | `localMoeExpertNum` 为 `uint32_t`，在 `CheckAttrs`（tiling.cpp:962）已校验 `> 0` 且 `* epWorldSize <= 2048`。`(localMoeExpertNum + 7U) / 8U * 8U` 最大值 2055，远不溢出 `uint32_t`。 |
| 2 | **UB 容量保护** | `BufferCalculation` 中扩展 `statusBuf` 时有 `totalUsedUB_ + statusBufSize > totalUbSize_` 回退保护（kernel.h:1156），确保不超 UB。 |
| 3 | **判定逻辑一致性** | `SetExpertPrefixSumA5`（kernel.h:1506，接收参数）和 `SetExpertTokenNumsA5`（kernel.h:1559，计算并传递）两处 `useCumSum` 判定条件完全同步。 |
| 4 | **流水同步正确** | CumSum 前后均有 `PipeBarrier<PIPE_V>`（kernel.h:1528,1531），Cast 后 `SyncFunc<V_MTE3>`（kernel.h:1534），DataCopy 前同步完整。 |
| 5 | **无别名冲突** | CumSum 的 4 个 tensor（`expertTokenNumCumsumTensor`/`cumsumLastRowTensor`/`expertTokenNumTensor_`/`cumsumTmpTensor`）分别来自 `gatherMaskOutBuf_`/`scalarBuf_`/`tokenNumBuf_`/`statusBuf_`，无别名冲突。 |
| 6 | **成员变量有初始化** | `cumsumTmpMinSize_ = 0`（kernel.h:271），符合规范 3.1。 |
| 7 | **LOG API 类型匹配** | `OP_LOGI(nodeName, "MTE cumsumTmpMinSize=%u. ...", cumsumMinTmp)` — `cumsumMinTmp` 为 `uint32_t`，`%u` 匹配，符合规范 11.3。 |
| 8 | **条件编译隔离** | A5 专属逻辑用 `#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)` 隔离，不影响 A3/A2 路径。 |
| 9 | **字段时序安全** | `expertTokenNumsType` 在 `GetAttrAndSetTilingData`（tiling.cpp:1943）中通过 `SetAttrParams`（行 839）设置，早于修改点（行 1988），无未初始化读取风险。 |
| 10 | **安全降级设计** | 当 tmp 空间不足时 `useCumSum = false`，降级到标量循环前缀和（kernel.h:1510-1520），功能正确性不受影响。 |
| 11 | **结构体字段末尾添加** | `cumsumTmpMinSize` 添加在 `MoeDistributeDispatchV2Info` 结构体末尾（tiling.h:55），`reserved1` 保留不变，符合规范 10.4，ABI 兼容。 |
| 12 | **出参预初始化兜底** | `cumsumMaxTmp`/`cumsumMinTmp` 预初始化为 0（tiling.cpp:1994-1995），即使 `GetCumSumMaxMinTmpSize` 异常未设置出参，`cumsumTmpMinSize=0` 也会触发 Kernel 侧安全降级。 |
| 13 | **降级可观测性** | Host 侧 `OP_LOGI` 说明降级可能性（tiling.cpp:1998-1999），便于排查。 |
| 14 | **避免冗余查询** | `useCumSum` 在 `SetExpertTokenNumsA5` 中计算后传递给 `SetExpertPrefixSumA5`，`statusBuf_.Get<uint8_t>().GetSize()` 仅调用一次（kernel.h:1559）。 |

---

## 问题详情及修改建议

### 问题ID：ISSUE-001 | 严重级别：LOW（存疑）

#### 假设检验过程

**代码段**：`MoeDistributeDispatchA3TilingFuncImplPublic` 函数中 CumSum tmp 大小计算（tiling.cpp:1996）

**假设**：H0: 该代码段是安全的

| 证据序号 | 证据类型 | 规范ID | 证据描述 | 分值增量 | 累计自信值 |
|---------|---------|--------|---------|---------|-----------|
| 1 | 函数调用链风险 | 5.1 | `GetCumSumMaxMinTmpSize` 调用未接收/检查返回值。本地仓库及 CANN SDK 头文件均未找到该 API 声明，无法确认是否为 void 返回。 | +25% | 25% |
| 2 | 上下文防御存在 | — | 出参 `cumsumMaxTmp`/`cumsumMinTmp` 预初始化为 0（行 1994-1995）。即使 API 未设置出参，`cumsumTmpMinSize = 0` 时 Kernel 侧 `useCumSum` 判定为 false，安全降级到标量循环。 | -15% | 10% |

**结论**：自信值 **10%** < 60%，**不推翻原假设 H0**。风险极低，标记为"存疑"。

---

**关联规范条款**：cpp-secure.md 5.1（资源申请后判断是否成功）— `[适用: Tiling]`
**代码路径**：`mc2/moe_distribute_dispatch_v2/op_host/op_tiling/moe_distribute_dispatch_v2_tiling.cpp:1996`
**问题类型**：API 返回值未检查（存疑）
**问题描述**：`GetCumSumMaxMinTmpSize` 通过引用出参返回 `cumsumMaxTmp`/`cumsumMinTmp`，调用处未接收返回值。该 API 在本地仓库和 CANN SDK 头文件中均未找到声明，无法确认其返回类型。若 API 返回错误码且可能失败，则违反规范 5.1。但从调用方式（引用出参 + 未接收返回值）和同类 API 命名惯例推断，大概率是 void 返回的工具函数；且出参预初始化为 0，即使异常也有安全降级保护。

#### 修改建议

**修改说明**：建议确认 `GetCumSumMaxMinTmpSize` 的 API 签名。若该 API 有返回值，添加返回值检查；若确认为 void 返回，可忽略本条。当前代码已有预初始化兜底，不阻塞合入。

---

## 规范覆盖检查清单

### C++ 安全编码规范（cpp-secure.md）

| 规范编号 | 规范名称 | 检查结果 | 说明 |
|---------|---------|---------|------|
| 2.1 | 有符号整数运算不溢出 | PASS | 新增代码无有符号整数运算 |
| 2.2 | 无符号整数运算不回绕 | PASS | `localMoeExpertNum` 已校验 <= 2048，`(x+7)/8*8` 最大 2055 不回绕；`moeExpertNumPerRank_ * epWorldSize_ * UB_ALIGN` 业务值远小于 UINT32_MAX |
| 2.3 | 除法/余数除零保护 | PASS | 无新增除法/求余（`/ 8U` 除数为常量非零） |
| 3.1 | 禁止未初始化变量 | PASS | `cumsumTmpMinSize_ = 0` 有初始化；`cumsumMaxTmp`/`cumsumMinTmp` 预初始化为 0 |
| 3.3 | 数组索引校验 | PASS | 无新增数组索引访问 |
| 4.1 | 外部输入合法性校验 | PASS | `expertTokenNumsType` 在 arch35 tiling 中已校验为 0 或 1 |
| 5.1 | 资源申请后判断是否成功 | **存疑** | 见 ISSUE-001 |
| 10.4 | 结构体字段末尾添加 | PASS | `cumsumTmpMinSize` 添加在结构体末尾 |
| 10.5 | 接口变更兼容性 | PASS | 末尾添加字段，旧版忽略尾部字段，兼容 |
| 11.3 | LOG API 类型匹配 | PASS | `%u` 对应 `uint32_t` |

### Ascend C API 规范（ascendc-api.md）

| 规范编号 | 规范名称 | 检查结果 | 说明 |
|---------|---------|---------|------|
| API-1 | 禁止 GlobalTensor::SetValue/GetValue | PASS | `!useCumSum` 分支的 `GetValue`/`SetValue` 为 LocalTensor 操作且为已有代码，非本 PR 引入 |
| API-3 | DataCopy/DataCopyPad 对齐 | PASS | 无新增 DataCopy，已有 DataCopyPad 使用 `DataCopyExtParams`（blockLen 为字节） |
| API-7 | 禁止动态内存分配 | PASS | `std::vector` 在 Host 侧使用，Kernel 侧无动态分配 |

### 高性能编程规范（ascendc-perf.md）

| 规范编号 | 规范名称 | 检查结果 | 说明 |
|---------|---------|---------|------|
| PREC-1 | 流水线同步正确性 | PASS | CumSum 前后 PipeBarrier，Cast 后 SyncFunc<V_MTE3>，DataCopy 前同步完整 |
| TIL-2 | 片上缓存容量不溢出 | PASS | statusBuf 扩展有 UB 超限回退保护 |
| PERF-1 | 循环内禁止逐元素操作 | PASS | `!useCumSum` 分支的标量循环为已有代码；本 PR 改进了进入 CumSum 分支的条件，减少标量循环触发 |

---

## 报告生成时间
2026-08-03

## 报告状态
已完成检视。**建议合入**（无阻塞级问题，1 处轻微存疑项需确认 API 签名，已有预初始化兜底）。
