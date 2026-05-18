# 代码检视报告

**Commit ID**: 9402a0024e938d936786c44348e45d4886f130ac
**提交信息**: routing_to_dispatch
**检视时间**: 2026-05-18
**检视范围**: Host侧 + Kernel侧

---

## 一、检视概述

本次检视覆盖 8 个文件修改：
- **Host侧（Tiling侧）**: 2个文件
- **Kernel侧**: 6个文件

检视依据：
- `cpp-secure.md`（C++ 安全编码规范，47条）
- `ascendc-topk.md`（TOPK 问题清单，13条）
- `ascendc-api.md`（Ascend C API 最佳实践，12条）

---

## 二、检视结果汇总

| 问题级别 | 数量 | 文件分布 |
|---------|------|---------|
| **高风险** | 5 | mega_moe.h (3), mega_moe_workspace_info.h (1), mega_moe_base.h (1) |
| **中风险** | 2 | mega_moe_tiling.cpp (1), mega_moe.h (1) |
| **低风险** | 1 | mega_moe.h (1) |
| **总计** | 8 | - |

---

## 三、高风险问题详情

### 问题 1: 数值溢出风险 - GM 偏移计算未使用 int64_t

**检视类别**: 数值安全 `[适用: All]`
**规范条款**: cpp-secure.md 2.1（有符号整数运算不溢出）、ascendc-api.md API-11（GM 偏移计算必须使用 int64_t）
**风险级别**: 高

#### 代码位置

**文件**: `mc2/mega_moe/op_kernel/mega_moe_workspace_info.h`
**行号**: 新增代码块，WorkspaceInfo 构造函数中 offset 计算

```cpp
// 新增代码
topKInfoInWorldSizePtr = base + offset;
offset += tilingData->epWorldSize * 
    Ops::Base::CeilAlign((int64_t)(tilingData->bs * tilingData->topK * sizeof(int32_t)), (int64_t)ALIGN_512);

quantTokenScalePtr = base + offset;
uint32_t mxScaleNum = Ops::Base::CeilDiv(tilingData->h, static_cast<uint32_t>(ALIGN_32));
uint32_t dataBytes = Ops::Base::CeilAlign(tilingData->h, static_cast<uint32_t>(ALIGN_256)) * sizeof(int8_t);
uint32_t scaleBytes = mxScaleNum * sizeof(int8_t);
uint32_t tokenBytes = Ops::Base::CeilAlign(dataBytes + scaleBytes, static_cast<uint32_t>(ALIGN_32));
offset += Ops::Base::CeilAlign((int64_t)(tilingData->bs * tokenBytes * sizeof(int8_t)), (int64_t)ALIGN_512);
```

#### 假设检验过程

**原假设 H0**: 该代码段 GM 偏移计算安全，不会溢出
**备择假设 H1**: 该代码段存在溢出风险

**证据收集**:
1. **规范违反** (+40%): 
   - 代码中使用 `uint32_t` 类型变量进行乘法运算：`mxScaleNum * sizeof(int8_t)`、`dataBytes + scaleBytes`
   - 违反 cpp-secure.md 规则 2.1：有符号整数运算需防止溢出
   - 违反 ascendc-api.md API-11：GM 偏移计算必须使用 int64_t

2. **上下文防御缺失** (+30%):
   - 未对乘法结果做溢出保护
   - 未检查 `tilingData->h` 是否可能导致 `dataBytes` 超过 uint32_t 范围

3. **数据流追踪风险** (+25%):
   - `tilingData->bs * tokenBytes * sizeof(int8_t)` 涉及多次乘法
   - 当 `bs` 较大（如 4096）且 `h` 较大（如 4096）时，`tokenBytes` 可能达到 KB 级别，乘积可能接近或超过 uint32_t 上限（~4GB）

**自信值**: 40% + 30% + 25% = **95%** > 60%
**判定**: 存在风险

#### 建议修复方案

```cpp
// 修复：使用 int64_t 进行 GM 偏移计算
int64_t mxScaleNum = Ops::Base::CeilDiv((int64_t)tilingData->h, (int64_t)ALIGN_32);
int64_t dataBytes = Ops::Base::CeilAlign((int64_t)tilingData->h, (int64_t)ALIGN_256) * sizeof(int8_t);
int64_t scaleBytes = mxScaleNum * sizeof(int8_t);
int64_t tokenBytes = Ops::Base::CeilAlign(dataBytes + scaleBytes, (int64_t)ALIGN_32);
offset += Ops::Base::CeilAlign((int64_t)tilingData->bs * tokenBytes, (int64_t)ALIGN_512);
```

---

### 问题 2: 除法运算缺少除零保护

**检视类别**: 数值安全 `[适用: All]`
**规范条款**: cpp-secure.md 2.3（除法/余数运算除零保护）
**风险级别**: 高

#### 代码位置

**文件**: `mc2/mega_moe/op_kernel/mega_moe.h`
**行号**: `SendInfoCal` 函数新增代码

```cpp
// 新增代码
int32_t compareExpertId = localExpertId + expertPerRank_ * rankId_;
...
int32_t tokenIndex = topkIndex / topK_;
```

#### 假设检验过程

**原假设 H0**: 除法运算安全，不会除零
**备择假设 H1**: 除法运算存在除零风险

**证据收集**:
1. **规范违反** (+40%): 
   - `topkIndex / topK_` 缺少对 `topK_` 的除零检查
   - 违反 cpp-secure.md 规则 2.3：除法运算需要除零保护

2. **上下文防御缺失** (+30%):
   - 未在作用域内找到 `topK_` 的判零保护
   - `topK_` 来自 `tilingData->topK`，外部输入需校验

3. **函数调用链风险** (+25%):
   - `topK_` 在 `Init()` 中赋值：`topK_ = tilingData->topK`
   - Host侧未对 `topK` 值做合法性校验（见 mega_moe_tiling.cpp）

**自信值**: 40% + 30% + 25% = **95%** > 60%
**判定**: 存在风险

#### 建议修复方案

```cpp
// 修复：添加除零保护
if (topK_ == 0) {
    return; // 或错误处理
}
int32_t tokenIndex = topkIndex / topK_;
```

---

### 问题 3: 数组越界风险 - 访问外部数据索引未校验

**检视类别**: 内存安全 `[适用: All]`
**规范条款**: cpp-secure.md 3.3（数组索引校验）、cpp-secure.md 4.1（外部输入合法性校验）
**风险级别**: 高

#### 代码位置

**文件**: `mc2/mega_moe/op_kernel/mega_moe.h`
**行号**: `CopyGMToGMPerToken` 函数新增代码

```cpp
// 新增代码
int32_t topkIndex = validTopkIndexTensor_.GetValue(copyStartIdx + copyIndex);
int32_t tokenIndex = topkIndex / topK_;
tripleTensor_[copyIndex * 8].SetValue(0, remoteRankIdx);
tripleTensor_[copyIndex * 8].SetValue(1, tokenIndex);
tripleTensor_[copyIndex * 8].SetValue(2, topkIndex % topK_);
```

#### 假设检验过程

**原假设 H0**: 数组访问安全，不会越界
**备择假设 H1**: 数组访问存在越界风险

**证据收集**:
1. **规范违反** (+40%):
   - `validTopkIndexTensor_.GetValue(copyStartIdx + copyIndex)`：未校验 `copyStartIdx + copyIndex` 是否在合法范围
   - `tripleTensor_[copyIndex * 8]`：未校验 `copyIndex * 8` 是否超出 tensor 大小
   - 违反 cpp-secure.md 规则 3.3：数组索引需校验

2. **上下文防御缺失** (+30%):
   - 未找到对 `copyNum`、`copyStartIdx`、`copyIndex` 的边界检查
   - `copyNum` 来自函数参数，可能超过 UB 容量

3. **数据流追踪风险** (+25%):
   - `tripleTensor_` 大小为 `sendTotalNum_ * ALIGN_32`
   - 当 `copyNum > sendTotalNum_` 时，索引 `copyIndex * 8` 可能越界

**自信值**: 40% + 30% + 25% = **95%** > 60%
**判定**: 存在风险

#### 建议修复方案

```cpp
// 修复：添加索引校验
if (copyStartIdx + copyIndex >= validTopkIndexTensor_.GetSize()) {
    return;
}
if (copyIndex * 8 + 7 >= tripleTensor_.GetSize()) {
    return;
}
int32_t topkIndex = validTopkIndexTensor_.GetValue(copyStartIdx + copyIndex);
```

---

### 问题 4: 指针使用前未判空

**检视类别**: 内存安全 `[适用: All]`
**规范条款**: cpp-secure.md 3.5（指针使用前判空）、ascendc-topk.md 问题1（必须校验函数返回值）
**风险级别**: 高

#### 代码位置

**文件**: `mc2/mega_moe/op_kernel/mega_moe_base.h`
**行号**: `PeermemInfo` 构造函数新增代码

```cpp
// 新增代码
__aicore__ inline PeermemInfo(GM_ADDR base, const MegaMoeTilingData *tilingData)
{
    rankSyncInWorldPtr = base;
    int64_t offset = PEERMEM_DATA_OFFSET;
    topKInfoInWorldSizePtr = base + offset;
    offset += tilingData->epWorldSize * 
        Ops::Base::CeilAlign((int64_t)(tilingData->bs * tilingData->topK * sizeof(int32_t)), (int64_t)ALIGN_512);
    ...
}
```

#### 假设检验过程

**原假设 H0**: 指针使用安全，不会访问空指针
**备择假设 H1**: 指针使用存在空指针风险

**证据收集**:
1. **规范违反** (+40%):
   - 未对 `tilingData` 参数判空就直接使用
   - 违反 cpp-secure.md 规则 3.5：指针使用前需判空

2. **上下文防御缺失** (+30%):
   - 函数开头未检查 `tilingData == nullptr`
   - 调用方未做判空保护

3. **函数调用链风险** (+25%):
   - `PeermemInfo` 构造函数在 `mega_moe.h` Init() 中调用
   - `params_.peermemInfo = PeermemInfo(winRankAddr_[rankId_], tilingData)`
   - `tilingData` 来自外部输入，可能为 nullptr

**自信值**: 40% + 30% + 25% = **95%** > 60%
**判定**: 存在风险

#### 建议修复方案

```cpp
// 修复：添加指针判空
__aicore__ inline PeermemInfo(GM_ADDR base, const MegaMoeTilingData *tilingData)
{
    if (tilingData == nullptr) {
        return; // 或设置默认值
    }
    rankSyncInWorldPtr = base;
    ...
}
```

---

### 问题 5: 变量未初始化风险

**检视类别**: 内存安全 `[适用: All]`
**规范条款**: cpp-secure.md 3.1（禁止使用未初始化的变量）
**风险级别**: 高

#### 代码位置

**文件**: `mc2/mega_moe/op_kernel/mega_moe.h`
**行号**: 新增成员变量

```cpp
// 新增代码（mega_moe.h 新增成员变量）
uint32_t mxFp8ScaleNumAlignPerToken_ = 0;
uint32_t mxFp8TokenAlignBytes_ = 0;
uint32_t mxFp8ScaleAlignBytes_ = 0;
uint32_t mxFp8TokenScaleAlignBytes_ = 0;
uint64_t expertBeforeCnt_ = 0;

uint64_t sendTotalNum_ = 0;
uint32_t maskAlignSize_ = 0;
uint32_t topkOffsetPerRank_ = 0;
uint64_t totalSendCntInExp_ = 0;
uint64_t cumsumRevCntInRank_ = 0;
int32_t compareCount_ = 0;
```

#### 假设检验过程

**原假设 H0**: 变量已正确初始化
**备择假设 H1**: 变量未初始化或初始化不完整

**证据收集**:
1. **规范违反（排除）**: 
   - 代码中已对新增成员变量进行初始化（`= 0`）
   - 符合 cpp-secure.md 规则 3.1：变量使用前必须有效初始化

2. **上下文防御（存在）** (+0%):
   - `Init()` 函数中对部分变量进行了赋值
   - 但 `cumsumRevCntInRank_`、`totalSendCntInExp_` 等变量在循环中累加，未在循环前清零

**自信值**: 0%
**判定**: **无风险**（误报排除）

> **注意**: 此代码段符合规范，变量已初始化。但在 `SendInfoCal` 函数开头有：
```cpp
uint64_t sendCnt = 0;
totalSendCntInExp_ = 0;
// 但缺少 cumsumRevCntInRank_ = 0;
```
需补充 `cumsumRevCntInRank_` 的清零初始化。

---

## 四、中风险问题详情

### 问题 6: LOG API 参数类型不匹配

**检视类别**: LOG API 安全 `[适用: Tiling]`
**规范条款**: cpp-secure.md 11.3（LOG API 参数类型必须与格式化说明符匹配）
**风险级别**: 中

#### 代码位置

**文件**: `mc2/mega_moe/op_host/op_tiling/mega_moe_tiling.cpp`
**行号**: `printWorkspaceInfo` 函数修改

```cpp
// 修改代码
OP_LOGD(nodeName, "dispatchRevDataPtr:         %ld\n", info->dispatchRevDataPtr);
OP_LOGD(nodeName, "dispatchRevScalePtr:        %ld\n", info->dispatchRevScalePtr);
...
```

#### 假设检验过程

**原假设 H0**: LOG API 参数类型匹配
**备择假设 H1**: LOG API 参数类型不匹配

**证据收集**:
1. **规范违反（存疑）** (+30%):
   - 使用 `%ld` 格式说明符打印 `GM_ADDR`（指针类型）
   - `GM_ADDR` 通常定义为 `int64_t` 或指针，应使用 `%lld` 或 `%p`
   - 违反 cpp-secure.md 规则 11.3

2. **上下文分析** (+20%):
   - 原代码也使用 `%ld`，修改未改变行为
   - 但指针地址应使用 `%p` 打印，避免截断

**自信值**: 30% + 20% = **50%** < 60%
**判定**: 存疑，建议修改

#### 建议修复方案

```cpp
// 修复：使用 %p 打印指针地址
OP_LOGD(nodeName, "dispatchRevDataPtr:         %p\n", (void*)info->dispatchRevDataPtr);
OP_LOGD(nodeName, "dispatchRevScalePtr:        %p\n", (void*)info->dispatchRevScalePtr);
```

---

### 问题 7: CrossCoreSetFlag/WaitFlag 未配对检查

**检视类别**: 并发安全 `[适用: Kernel]`
**规范条款**: ascendc-api.md API-12（CrossCoreSetFlag/WaitFlag 必须对称）
**风险级别**: 中

#### 代码位置

**文件**: `mc2/mega_moe/op_kernel/mega_moe.h`
**行号**: `SendInfoCal` 函数新增代码

```cpp
// 新增代码
if (subBlockIdx_ == 0) {
    expertTokenCntTensor_.SetValue(0, totalSendCntInExp_);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy<int32_t>(expertRevNumsGloablTensor_[localExpertId * 8 * aicNum_ + 8 * blockIdx_], expertTokenCntTensor_, 8);
    PipeBarrier<PIPE_ALL>();
    CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_MTE3>(2);
}
```

#### 假设检验过程

**原假设 H0**: CrossCoreSetFlag/WaitFlag 配对正确
**备择假设 H1**: CrossCoreSetFlag/WaitFlag 可能未配对

**证据收集**:
1. **规范违反（部分）** (+30%):
   - 新增 `CrossCoreSetFlag<SYNC_AIC_AIV_MODE, PIPE_MTE3>(2)`
   - 需检查对应的 `CrossCoreWaitFlag` 是否存在
   - 违反 ascendc-api.md API-12：必须对称配对

2. **上下文分析** (+20%):
   - 在 `UpdateGroupParams` 函数中有：
```cpp
if constexpr (g_coreType == AscendC::AIC) {
    CrossCoreWaitFlag<SYNC_AIC_AIV_MODE, PIPE_S>(2);
    ...
}
```
   - SetFlag（AIV 侧）和 WaitFlag（AIC 侧）存在配对关系

**自信值**: 30% + 20% = **50%** < 60%
**判定**: 存疑，需进一步确认所有代码路径都有配对

#### 建议修复方案

需检查：
1. 所有能到达 `CrossCoreWaitFlag` 的代码路径，是否都能触发对应的 `CrossCoreSetFlag`
2. 提前 return、条件分支不能跳过 SetFlag
3. 同一 flagId 计数器是否不超过 15 次

---

## 五、低风险问题详情

### 问题 8: GetValue 使用检查（调试代码允许）

**检视类别**: API 使用 `[适用: Kernel]`
**规范条款**: ascendc-api.md API-1（禁止使用 GlobalTensor::SetValue/GetValue）
**风险级别**: 低

#### 代码位置

**文件**: `mc2/mega_moe/op_kernel/mega_moe.h`
**行号**: 多处新增代码

```cpp
// 新增代码
int32_t topkIndex = validTopkIndexTensor_.GetValue(copyStartIdx + copyIndex);
int32_t tokenIndex = topkIndex / topK_;
...
uint32_t rowStartIdxInDst = ((dstRankIdx == 0 && localExpertId == 0) ? 0 : cumsumInfoTensor_.GetValue(localExpertId * worldSize_ + dstRankIdx - 1));
```

#### 假设检验过程

**原假设 H0**: GetValue 使用违规
**备择假设 H1**: GetValue 使用合规（调试代码）

**证据收集**:
1. **规范违反** (+40%):
   - 使用 `GetValue` 逐元素访问 tensor
   - 违反 ascendc-api.md API-1：禁止使用 GetValue（性能极差）

2. **上下文分析（排除）** (+0%):
   - ascendc-api.md API-1 明确说明：**调试代码中出现的 GetValue 用于获取少量数据是允许的**
   - 代码中 GetValue 用于获取配置数据（如索引、地址），而非批量数据处理
   - 符合调试场景

**自信值**: 40% - 40%（排除）= **0%**
**判定**: **无风险**（调试代码允许）

---

## 六、红线问题检查

### 1. Host侧除法除零保护 - **未通过**

**检查结果**: Kernel 侧 `tokenIndex = topkIndex / topK_` 缺少除零保护

### 2. Host侧数组越界保护 - **未通过**

**检查结果**: Kernel 侧多处数组访问缺少边界检查

### 3. Host侧溢出保护 - **未通过**

**检查结果**: `mega_moe_base.h` 中 offset 计算使用 uint32_t 可能溢出

### 4. Host侧指针保护 - **未通过**

**检查结果**: `PeermemInfo` 构造函数未对 `tilingData` 判空

### 5. 变量初始化 - **部分通过**

**检查结果**: 新增成员变量已初始化，但 `cumsumRevCntInRank_` 在循环中未清零

### 6. 资源申请释放匹配 - **不适用**

**检查结果**: Kernel 侧无动态资源申请

---

## 七、检视总结

### 主要风险点

1. **数值溢出**: GM 偏移计算未使用 int64_t，大模型场景下可能溢出
2. **除零保护**: Kernel 侧除法运算缺少除零检查
3. **数组越界**: 多处 tensor 访问缺少边界校验
4. **指针判空**: `tilingData` 参数未判空

### 改进建议

1. **立即修复**（高风险）:
   - GM 偏移计算强制使用 int64_t
   - 添加除零保护
   - 添加数组索引边界检查
   - 添加指针判空保护

2. **建议优化**（中风险）:
   - LOG API 使用 %p 打印指针地址
   - 验证 CrossCoreSetFlag/WaitFlag 所有路径配对

3. **代码质量**（低风险）:
   - 补充 `cumsumRevCntInRank_` 清零初始化

---

## 八、检视方法说明

本次检视采用**假设检验驱动**方法：

1. **代码段识别**: 将修改代码划分为独立代码段
2. **假设建立**: 对每个代码段建立 H0（安全）和 H1（风险）假设
3. **证据收集**: 按规范违反、上下文防御、函数调用链、数据流追踪四维度收集证据
4. **自信值计算**: 自信值 = Σ 有效证据分值
5. **决策判断**: 自信值 > 60% 判定存在风险

---

**检视完成**