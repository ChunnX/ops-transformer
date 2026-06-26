# SMLA S2 Split FD 适配方案

## 1. 目标与边界

本文描述 `sparse_flash_mla`（以下简称 SMLA）接入 S2 Split Flash Decode（以下简称 FD）的适配方案。目标是在一个逻辑 `(B, M)` 行被多个 core 按 S2 维拆分时，SMLA 可以复用公共 FD 能力完成 partial O 规约，并得到与未拆分 S2 等价的最终输出。

SMLA 已经有独立的 metadata 算子：

```text
attention/sparse_flash_mla_metadata
```

并且 `attention/sparse_flash_mla/op_kernel/sparse_flash_mla_metadata.h` 已定义 FA/FD 两段 metadata ABI。本文不是要求重新设计一套 SMLA metadata，而是要求基于现有 `sparse_flash_mla_metadata` 生成结果，在 `sparse_flash_mla` kernel 侧补齐 FD staging、FD reducer 和最终输出适配。

复用的公共文件：

```text
attention/sparse_flash_mla/op_kernel/arch35/flash_decode.h
```

SMLA 侧应复用以下 data-plane 接口：

1. `StageVec1Lse`：Vec1 最终 S2 loop 将 `max/sum` 暂存到 GM；
2. `StageVec2PartialO`：Vec2 最终 S2 loop 将归一化 `partial O` 暂存到 GM；
3. `Reduce`：FD 从 GM 读取多个 split 的 `partial O/max/sum`，计算权重并规约。

### 1.1 QSMLA 对齐优先原则

本次 SMLA FD 适配要求以 QSMLA 的 S2 split FD 实现为代码母版。除 SMLA 自身 metadata ABI、字段命名、shape/layout、输出地址和模板接口确实不同的地方外，kernel 侧控制流、函数拆分、event 顺序、staging 调用位置和 FD reduce 调度应尽量与 QSMLA 保持逐行等价。

实施规则：

- 每个阶段编码前先定位 QSMLA 对应实现，并在改动说明中写明来源函数；
- 能整段搬运的代码优先整段搬运，只做 SMLA 必需的局部替换；
- 局部替换必须保持原 QSMLA 语句顺序、条件判断、event wait/set 顺序和 DMA 参数含义；
- 不为了“重新设计 SMLA 风格”而改写 QSMLA 已验证过的 FD 控制流；
- 若 SMLA 必须偏离 QSMLA，必须在 `SMLA_FD_WORK_PROGRESS.md` 中记录偏离点、原因和验证方式。

当前对齐检查结论：

- `ProcessMainLoop()` 中 FA metadata parse、`s2SplitIdxCounter`、`firstFdDataWorkspaceIdx` 传递，已基本按 QSMLA `mixed_quant_sparse_flash_mla_scfa_kernel.h` 搬运；差异仅保留 SMLA 自身 `ComputeParamBatch()`、`ComputeS2LoopInfo()`、`ProcessVec0()` 参数形式；
- `ApplyS2MetadataRange()` 算法应保持 QSMLA 原样，仅将 QSMLA 的 `s2LineStartIdx/s2LineOriEndIdx` 字段替换为 SMLA 的 `s2OriLineStartIdx/s2OriLineEndIdx`；
- `ParseFdRunInfo()` 应保持 QSMLA 原样，使用公共 `AttentionCommon::FdRunInfo`，metadata index 仍使用 SMLA 已有 `GetAttrAbsIndex(aivIdx, *, true)`；
- SMLA 当前尚未搬运 QSMLA 的 `SyncAll() -> ProcessFlashDecode()` FD 阶段、Vec1/Vec2 staging helper、`InitFDBuffers()` 和 final copy-out 适配，这些属于后续阶段必须按 QSMLA 结构补齐的 gap。

不能作为外部依赖直接复用的 QSMLA 私有内容：

- QSMLA 的 `RunInfo`、`ConstInfo`；
- QSMLA metadata 字段编号；
- QSMLA 的 BSND/TND 输出地址公式；
- QSMLA 的 Split-G core 映射细节，除非 SMLA 具有完全相同的 head 分配语义。

这里的“不直接复用”指不共享 QSMLA 私有 ABI 或地址公式；不表示重新实现控制流。SMLA 应复制 QSMLA 已验证的 FD 代码结构，再用 SMLA 字段和 helper 做最小替换。

SMLA 只接入公共 FD 的 data-plane，control-plane 继续由 SMLA 既有 metadata 与 kernel 调度实现：

```text
已有 sparse_flash_mla_metadata -> sparse_flash_mla kernel 解析 FA/FD metadata
-> SMLA 计算 FA writer slot/M offset -> common Vec1/Vec2 staging
-> SMLA 全核 SyncAll -> SMLA FD metadata 转 common FD run info -> common Reduce
-> SMLA 输出地址与 CopyOut
```

普通 S2 分块但每个逻辑 M 行仍由单 core 完整负责的场景，不启用 FD，继续走 SMLA 现有直接输出路径。

## 2. 数值与内存协议

### 2.1 数值协议

对同一个逻辑 M 行，第 `i` 个 S2 split 产生：

```text
m_i = local max
s_i = sum(exp(score - m_i))
p_i = sum(exp(score - m_i) * V) / s_i
```

FD 规约计算：

```text
M   = max_i(m_i)
G   = sum_i(exp(m_i - M) * s_i)
w_i = exp(m_i - M) * s_i / G
O   = sum_i(w_i * p_i)
```

因此 SMLA 必须保证：

- Vec1 staging 的是最终 online-softmax 状态下的 `max/sum`；
- Vec2 staging 的是已经完成最终除法的归一化 `partial O = p_i`；
- one-loop S2 split 也必须走最终除法路径；
- 不能把 numerator 暂存后再期望 FD 额外除一次。

这是最关键的数值契约。如果此处不一致，通常会出现稳定但全局数值错误。

### 2.2 GM staging ABI

SMLA 必须采用和 common FD 一致的 workspace layout：

```text
[partial O slots][max slots][sum slots]
```

每个 slot 尺寸：

```text
partial O: stagingM * dAlign * sizeof(float)
max      : stagingM * 8 * sizeof(float)
sum      : stagingM * 8 * sizeof(float)
```

要求：

- `partial O` 使用 FP32；
- `max/sum` 每行是 8 个 FP32 广播值，不是单个 scalar；
- `dAlign` 必须与 SMLA Vec2/FD 实际使用的 D 对齐宽度一致；
- `stagingM` 由 SMLA 的 head/M 映射规则决定，不能硬套 QSMLA 的固定值；
- host workspace 计算、FA writer、FD reader 必须使用同一套公式。

slot 协议固定为：

```text
FA writer slot = firstFdDataWorkspaceIdx + s2SplitIdx
FD reader slot = workspaceIdx + splitIdx, splitIdx in [0, workspaceNum)
```

当前 SMLA 适配沿用 QSMLA/common FD 的 `maxSplits = 2` 协议，并依赖现有 `sparse_flash_mla_metadata` 保证：

```text
1 <= workspaceNum <= 2
```

超过 2 split 要么 host 拒绝，要么整体扩展 metadata、workspace、kernel loop 和测试，不允许 kernel 静默截断。
本轮 SMLA kernel 不额外添加 `workspaceNum > 2` 的防御性限制。

### 2.3 同步协议

SMLA kernel 调度必须分成两个阶段：

```text
FA producer 阶段
所有启动 core SyncAll()
FD consumer 阶段
```

要求：

- 有 FA work 的 core 写完 staging 后进入 `SyncAll()`；
- 没有 FA work 的 core 也必须进入同一个 `SyncAll()`；
- `coreEnable == false` 的 FD core 也不能提前绕过会导致其他 core 等待的 barrier；
- FD reducer 只能在 barrier 后读取 staging。

common FD 内部仍需要遵守 UB event 生命周期：

- Vec1 max/sum staging source：`V -> MTE3`；
- Vec2 partial O staging source：`V -> MTE3`；
- FD max/sum UB：`MTE2 -> V`；
- FD partial O UB：`MTE2 -> V -> MTE2`。

全核 phase barrier 属于 SMLA control-plane，不应移入 common。

## 3. SMLA 需要适配的模块

### 3.1 既有 metadata ABI 复用

SMLA 的 metadata 不在 `sparse_flash_mla` host tiling 中重新生成，而是由 `sparse_flash_mla_metadata` 算子生成。当前已有 ABI：

```text
attention/sparse_flash_mla/op_kernel/sparse_flash_mla_metadata.h
```

其中已经定义：

```text
FA_METADATA_SIZE = 9
FD_METADATA_SIZE = 8

FA:
  FA_CORE_ENABLE_INDEX
  FA_BN2_START_INDEX
  FA_M_START_INDEX
  FA_S2_START_INDEX
  FA_BN2_END_INDEX
  FA_M_END_INDEX
  FA_S2_END_INDEX
  FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX
  FA_S2_MAX_NUM

FD:
  FD_CORE_ENABLE_INDEX
  FD_BN2_IDX_INDEX
  FD_M_IDX_INDEX
  FD_WORKSPACE_IDX_INDEX
  FD_WORKSPACE_NUM_INDEX
  FD_M_START_INDEX
  FD_M_NUM_INDEX
```

`sparse_flash_mla_metadata/op_kernel_aicpu/sparse_flash_mla_metadata_aicpu.cpp` 中已经有 FD 任务记录和生成逻辑：

```text
RecordFDInfo()
SplitFD()
GenMetadata()
```

因此 SMLA FD 适配的 metadata 侧主要工作不是新增字段，而是核对并消费已有字段。

FA task 至少需要表达：

```text
coreEnable
BN / batch index
M index or M block index
S2Start
S2End
firstFdDataWorkspaceIdx
s2SplitIdx
isS2Split
```

FD task 至少需要表达：

```text
coreEnable
BN / batch index
M index or M block index
workspaceIdx
workspaceNum
mStartIdx
mNum
```

核心要求：

- `S2Start/S2End` 统一使用 block 还是 token 单位，kernel 内只能转换一次；
- 内部 S2 区间统一为左闭右开 `[start, end)`；
- `S2_END == 0` 的 terminal sentinel 只能在定义好的末尾语义下转换为真实 S2 长度；
- `isS2Split` 应由当前 M 行被裁剪后的 S2 范围判断，不能只靠“是否第一个/最后一个 task”推断；
- FD task 的 `workspaceIdx/workspaceNum` 必须能覆盖同一个 `(B, M)` 的所有 split slot。
- `sparse_flash_mla_metadata` 生成的 `firstFdDataWorkspaceIdx`、`fdWorkspaceIdx`、`fdS2SplitNum`、`fdMStart`、`fdMNum` 必须和 kernel staging layout 一致。

### 3.2 Kernel FA/FD run info 适配

SMLA kernel 当前已有 `RunInfo`，FA metadata 也已经被 arch35 kernel 解析。FD 适配时应在 `sparse_flash_mla` kernel 内解析公共 `AttentionCommon::FdRunInfo`，而不是复用 QSMLA 的 `RunInfo` / `ConstInfo`。

建议形成两个逻辑视图：

```cpp
struct SmlaFaRunInfo {
    bool isS2Split;
    uint32_t firstFdDataWorkspaceIdx;
    uint32_t s2SplitIdx;
    uint32_t s2Start;
    uint32_t s2End;
    // SMLA 原有 FA 必需字段
};

struct FdRunInfo {
    bool coreEnable;
    int64_t bn2Idx;
    int64_t mIdx;
    int64_t workspaceIdx;
    int64_t workspaceNum;
    int64_t mStartIdx;
    int64_t mNum;
};
```

传给 common FD 的字段来自同一个公共结构：

```cpp
AttentionCommon::FdRunInfo {
    coreEnable,
    bn2Idx,
    mIdx,
    workspaceIdx,
    workspaceNum,
    mStartIdx,
    mNum
};
```

`bn2Idx/mIdx/coreEnable` 由 common `FdRunInfo` 承载，各算子仍自行解释输出地址语义。

当前 `sparse_flash_mla_metadata.h` 的 `GetAttrAbsIndex(coreIdx, metaIdx, isAIV)` 已经支持：

```text
isAIV = false -> FA metadata
isAIV = true  -> FD metadata
```

所以 kernel 侧应新增的是 FD metadata parse 和 FD 阶段调度，不是新增 metadata 存储布局。

### 3.3 Vec1 max/sum staging

SMLA Vec1 在最终 S2 loop 完成 online-softmax 后，如果当前 task 是 S2 split，则调用：

```text
AttentionCommon::StageVec1Lse
```

SMLA 需要自己计算并传入：

```text
workspaceIdx = firstFdDataWorkspaceIdx + s2SplitIdx
stagingMOffset
validRows
max UB
sum UB
temporary broadcast UB
layout
staging GM base
event id
```

检查项：

- `validRows` 必须是当前 producer 实际负责的 M/head 行数；
- `stagingMOffset` 必须和 Vec2 partial O 的 offset 完全一致；
- max/sum 必须在最终 S2 loop 后写；
- 无工作 core 不得写陈旧 staging。

### 3.4 Vec2 partial O staging

SMLA Vec2 在最终 S2 loop 得到最终 partial O 后：

- 非 S2 split：继续走原 SMLA 直接输出路径；
- S2 split：调用 common `StageVec2PartialO` 写 GM staging。

要求：

```text
partial O = normalized p_i
GM row width = dValid
UB row width = dAlign
```

如果 SMLA 的 Vec2 UB 是 compact D，而 FD 按 `dAlign` 读取，需要用 common 接口中的 padding copy 路径，不能直接按 compact stride 写。

### 3.5 FD reducer 与输出适配

SMLA 的 FD 阶段：

1. parse `AttentionCommon::FdRunInfo`；
2. 如果 `coreEnable == true`，使用公共 FD run info；
3. 调用 common `Reduce` 得到 FP32 accumulated O；
4. 使用 SMLA 自己的 output helper 写回最终输出。

offset 责任边界必须明确：

```text
common Reduce:
  owns staging read offsets for partialO/max/sum
  inputs: layout + workspaceIdx + workspaceNum + mStartIdx + mNum

SMLA:
  owns FA staging writer slot/M offset
  owns final attentionOut write offset
```

也就是说，FD 阶段从 GM staging 读取 `partialO/max/sum` 的 offset 不应在 SMLA 中重复实现；SMLA 只需要把已有 FD metadata 解析成 common `FdRunInfo` 并传入 layout。最终 `attentionOut` 的写回地址仍由 SMLA 自己计算。

输出地址必须从 SMLA layout 重新推导：

```text
outputBase =
    batch/token base
  + M block base
  + mStartIdx / row-in-task base
  + D/head offset
```

不要复用 QSMLA 的输出公式。特别是：

- BSND、TND、PA layout 的 batch/token base 可能不同；
- TND 必须从 `cu_seqlens_q` 推导 token base；
- `mStartIdx` 是 FD task 内需要写回的逻辑 M 行起点；
- Split-G 或多 head 模式下，head/M 维 stride 必须按 SMLA 自身 layout 计算。

### 3.6 QSMLA 源函数映射

后续实现必须按下表核对 SMLA 与 QSMLA 的一致性。除“允许的 SMLA 替换”列外，其余代码应尽量保持 QSMLA 原结构。

| SMLA 适配点 | QSMLA 来源 | 搬运要求 | 允许的 SMLA 替换 |
| --- | --- | --- | --- |
| FA/FD phase 调度 | `mixed_quant_sparse_flash_mla_scfa_kernel.h::ProcessMainLoop` | 保持 `ParseFdRunInfo -> FA producer -> SyncAll -> ProcessFlashDecode` 的顺序 | SMLA `ComputeParamBatch/ComputeS2LoopInfo/ProcessVec0` 的参数形式 |
| S2 metadata 裁剪 | `ApplyS2MetadataRange` | 算法、sentinel 处理、range clamp、loop end 重算保持逐行等价 | `s2LineStartIdx/s2LineOriEndIdx` 替换为 `s2OriLineStartIdx/s2OriLineEndIdx` |
| FD metadata 解析 | `ParseFdRunInfo` | 读取顺序和 early return 保持一致 | 使用公共 `AttentionCommon::FdRunInfo`，字段类型按 metadata ABI |
| FA staging slot | `SCFABlockVec::GetFaStagingWorkspaceIdx` | 函数放在 SMLA Vec block 内，公式保持 `firstFdDataWorkspaceIdx + s2SplitIdx` | 函数名可加 SMLA 前缀但不要改公式 |
| FA staging M offset | `SCFABlockVec::GetFaStagingMOffset` | 优先搬运 QSMLA subBlock/split-G offset 结构 | 若 SMLA head 分配不同，只替换 offset 计算项 |
| staging layout | `SCFABlockVec::GetS2SplitFdStagingLayout` | 保持调用 common layout 的字段顺序 | `stagingM`、slot count 按 SMLA host workspace 公式 |
| Vec1 LSE staging | `SCFABlockVec::ProcessVec1` 中 `isS2Split && final loop` 分支 | 在原 Vec1 final-update 后插入，调用 `StageVec1Lse` 的参数顺序保持一致 | tmp UB、event id 使用 SMLA Vec1 已有资源 |
| Vec2 partial O staging | `SCFABlockVec::ProcessVec2` 中 `if (runInfo.isS2Split)` 分支 | 保持 final division 之后 split 写 staging、非 split copy out 的结构 | SMLA copy-out helper 和 `dSizeV/dAlign` 来源 |
| FD UB 初始化 | `AttentionCommon::InitFDBuffers` | 保持 `tPipe->Reset()` 和按 `workspaceNum/mNum/chunkRows` 申请的结构 | 各算子持有 `AttentionCommon::FdBuffers<TBuf<>>` 实例 |
| FD reduce | `SCFABlockVec::ProcessFlashDecode` | 保留输出 offset 前置计算、LocalTensor 获取、`AttentionCommon::Reduce`、最终 copy out 的结构 | 输出 offset 必须按 SMLA layout 重推，不能复制 QSMLA 公式 |
| no-work core barrier | `ProcessMainLoop` 的 `hasLoad == 0` 分支 | 保持 no-FA core 也 `SyncAll()` 后按 `fdRunInfo.coreEnable` 做 FD | SMLA 当前 Phase 2 尚未接 FD 时可暂不启用，Phase 5 必须补齐 |

实现完成后，`SMLA_FD_WORK_PROGRESS.md` 必须记录每个映射项的状态：`copied as-is`、`copied with SMLA field rename`、`SMLA-specific replacement` 或 `not implemented yet`。

## 4. 分阶段实施与验证

每个阶段应独立提交。进入下一阶段前，必须满足本阶段验收标准；失败时回退当前阶段，不叠加后续重构。

### 阶段 0：SMLA 既有 metadata 与 FD ABI 核对

**改动范围**

- 不改运行时代码；
- 梳理 `sparse_flash_mla_metadata` 生成的 FA/FD metadata；
- 梳理 `sparse_flash_mla` 当前 arch35 kernel 对 FA metadata 的解析、Vec1、Vec2、output copy；
- 明确以下参数来源：

```text
s2BaseSize
dValid
dAlign
gSize / head group size
stagingM
slotCount
maxSplits
workspaceIdx
workspaceNum
mStartIdx
mNum
```

**验收标准**

- 写清楚 `sparse_flash_mla_metadata` 中 `firstFdDataWorkspaceIdx/fdWorkspaceIdx/fdS2SplitNum/fdMStart/fdMNum` 的含义；
- 写清楚 `sparse_flash_mla` host workspace 公式；
- 写清楚 kernel 访问 GM staging 的 offset 公式；
- 确认二者最后访问字节不越界；
- 确认当前 SMLA 是否存在 Split-G 或类似 head 分半模式。

### 阶段 1：复用 existing metadata，补齐 workspace 对齐

**改动范围**

- 不新增一套 metadata 区；
- 复用 `sparse_flash_mla_metadata` 已有 FA/FD 字段；
- 核对 metadata 算子中 FD task 与 FA writer slot 的对应关系；
- 在 `sparse_flash_mla` host tiling/workspace 中按 common layout 预留 FD staging 空间。

**检查项**

```text
FA writer slot = firstFdDataWorkspaceIdx + s2SplitIdx
FD reader base = workspaceIdx
FD reader count = workspaceNum
workspaceNum <= MAX_S2_SPLIT_NUM
```

**验收标准**

- `sparse_flash_mla_metadata` trace 能打印：

```text
FA: (BN, M, S2Start, S2End, isS2Split, firstFdDataWorkspaceIdx, s2SplitIdx)
FD: (coreEnable, BN, M, workspaceIdx, workspaceNum, mStartIdx, mNum)
```

- 每个 FD task 能找到对应 FA writer；
- 没有 writer slot 被两个 FA task 重复写。
- `sparse_flash_mla` host workspace 字节数覆盖最后一个 FD staging 访问上界。

### 阶段 2：Kernel FD metadata parse 与 S2 裁剪对齐

**改动范围**

- 保留并核对当前 SMLA kernel FA metadata parse；
- 新增或补齐 SMLA kernel FD metadata parse；
- FA 侧根据已有 metadata 裁剪 S2 loop；
- 计算 `isS2Split`、`s2SplitIdx`、`firstFdDataWorkspaceIdx`；
- 不接入 common FD 写 staging。

**验收标准**

- 原非 split 用例结果不变；
- S2 split trace 中 `[S2Start, S2End)` 与 host 一致；
- terminal `S2_END == 0` 只在最后 task 被解释为真实 end；
- 空 Q / 空 KV / no-work core 不遗留旧 split 状态。
- FD metadata parse 结果与 `sparse_flash_mla_metadata` 的 `GenMetadata()` 输出一致。

### 阶段 3：接入 Vec1 max/sum staging

**改动范围**

- 在 SMLA Vec1 最终 S2 loop 后调用 `StageVec1Lse`；
- 保持非 split 路径不变；
- 增加 SMLA 自己的 `GetFaStagingWorkspaceIdx()` 和 `GetFaStagingMOffset()` helper。

**验收标准**

- max/sum GM 每行 8 个 FP32 值一致；
- `(slot, M row)` 至多一个 writer；
- `stagingMOffset + validRows <= stagingM`；
- 空任务不写 staging。

### 阶段 4：接入 Vec2 partial O staging

**改动范围**

- 在 SMLA Vec2 中增加 S2 split 分支；
- split 分支写 common staging，不直接 CopyOut；
- 非 split 分支继续直接输出；
- 确保 final division 在写 staging 前完成。

**验收标准**

- one-loop split 和 multi-loop split 都写 normalized partial O；
- `partial O` 的 slot 和 M offset 与 Vec1 max/sum 完全一致；
- 非 split 用例结果不变；
- split 用例如果临时禁用 FD，最终输出应只缺规约，不应发生越界或死锁。

### 阶段 5：接入 FD reducer 与 SMLA output copy

**改动范围**

- FA 阶段结束后插入所有启动 core 的 `SyncAll()`；
- no-work core 也进入 barrier；
- barrier 后，`coreEnable` 的 core 调用 common `Reduce`；
- Reduce 得到 accumulated O 后，用 SMLA output helper 写回。

**验收标准**

- FD 输出覆盖 `[mStartIdx, mStartIdx + mNum)`；
- 不重叠、不漏行；
- BSND/TND/PA 等 layout 的 output offset 分别验证；
- 同一 S2 split 用例连续跑 20 次结果稳定。

### 阶段 6：迁移 QSMLA paramset 中的 FD 覆盖用例

**改动范围**

- 从 `attention/mixed_quant_sparse_flash_mla/tests/pytest/mixed_quant_sparse_flash_mla_paramset.py` 所有用例；
- 按 `attention/sparse_flash_mla/tests/pytest/sparse_flash_mla_paramset.py` 的 schema 改写成 SMLA paramset；
- 不直接机械复制所有字段，必须按 SMLA 的输入、metadata 算子、layout、topk、sparse indices、golden 参数重新映射；
- 将迁移后的用例纳入 SMLA 单测或 batch 参数集，作为 FD 适配的第一批回归集合。

建议优先迁移以下 QSMLA 用例意图：

| QSMLA paramset | 迁移到 SMLA 后的覆盖目标 |
| --- | --- |
| `single_batch_2_cores` | 最小二 split FD 主路径 |
| `single_batch_3_cores` | 三核分配下的 split slot 与 FD reader |
| `N32_single_batch_3_cores` | 小 head/group 场景，验证非固定 64 行 stride |
| `single_batch_full_cores` | 满核分配与 workspace 上界 |
| `single_batch_full_cores_with_blocks` | 满核 + 多 block 分配 |
| `multi_batch_full_cores` | 多 batch output base |
| `1_core_has_2_reduce` | 一个 AIV 承担多个 FD task |
| `1_core_has_2_reduce_with_no_reduce_insight` | FD task 与无 FD/无 reduce task 交错 |
| `with_qs_0` | 空 Q / zero-row 路径与 barrier |
| `with_kvs_0` | 空 KV / no-work 路径与 barrier |
| `tnd_padding_kv`、`tnd_padding_1` | TND padding 与 `cu_seqlens` output base |
| `N128` | Split-G 或 SMLA 等价 head 分组路径；若 SMLA 不支持等价模式，则标记为不可迁移并说明原因 |
| `residual`、`with_tail` | residual / tail S2 loop 的 final-loop staging |
| `K_1024` | 大 K / 大 sparse workload 下的 workspace 与资源压力 |

**迁移规则**

- 用例名称可以保留 QSMLA 原名后缀，例如 `smla_fd_single_batch_2_cores`，便于追踪来源；
- shape 只能作为起点，必须以 metadata trace 确认迁移后仍触发相同 FD 语义；
- QSMLA 的 `N/gSize/splitG` 语义不能直接等同于 SMLA 的 head/group 语义，必须重新确认 `stagingM` 和 `stagingMOffset`；
- QSMLA 的 mixed-quant 输入字段不能直接复制到 SMLA，用 SMLA golden 需要的 `ori/cmp kv`、topk、sparse indices、residual、sinks 字段替代；
- 对不能在 SMLA 中表达的 QSMLA case，记录“不迁移原因”，不要做一个形似但不覆盖 FD 语义的假用例。

**验收标准**

- 每个迁移用例都有对应 metadata trace，证明实际触发了预期的 FA split 和 FD task；
- 至少覆盖二 split、三核分配、小 head/group、多 batch、空 Q、空 KV、TND padding、tail/residual；
- 迁移用例全部通过 SMLA CPU golden；
- S2 split 迁移用例连续执行 20 次结果稳定；
- 迁移后的 paramset 不影响 SMLA 原有非 FD 用例执行。

### 阶段 7：收口与回归

**改动范围**

- 删除临时 trace、device printf、临时断言；
- 保留 SMLA 私有 adapter helper；
- 不把 SMLA metadata parse 或 output 逻辑塞进 common；
- 补充设计注释。

**验收标准**

- SMLA 全量已启用参数集通过；
- 迁移自 QSMLA 的 FD paramset 全部通过；
- 非 FD 用例结果和性能无回退；
- FD split 用例重复 20 次稳定；
- UB/resource report 不超过目标 SoC 限制。

## 5. 测试用例设计与覆盖率验证

建议按独立轴设计，不要只靠 case 名称判断覆盖。

第一批 FD 回归用例优先从 QSMLA 已验证 paramset 迁移，来源为：

```text
attention/mixed_quant_sparse_flash_mla/tests/pytest/mixed_quant_sparse_flash_mla_paramset.py
```

目标落点为：

```text
attention/sparse_flash_mla/tests/pytest/sparse_flash_mla_paramset.py
```

迁移不是逐字段复制。必须用 metadata trace 证明迁移后仍覆盖相同 S2 split / FD 语义；否则该 case 只能作为 shape 参考，不能计入 FD 覆盖。

| 覆盖轴 | 必测值 | 验收点 |
| --- | --- | --- |
| S2 ownership | unsplit、二 split、terminal sentinel、非末尾 M 行 split | metadata clipping 和 slot 对齐 |
| FD workload | `mNum=1`、`mNum=16`、`mNum>16`、no FA but FD | tail、full chunk、多 chunk |
| head/M 模式 | 小 head、正常 head、Split-G/类 Split-G | `stagingM` 和 M offset 正确 |
| S2 loop | one-loop、multi-loop、tail、sparse/residual | final division 和 staging 时机正确 |
| layout | BSND、TND、PA/paged KV | output base 正确 |
| batch | 单 batch、多 batch、变长 batch | batch/token base 正确 |
| control flow | 空 Q、空 KV、active core 少于 launched core | 无死锁、无陈旧 staging |
| negative | 超过 `MAX_S2_SPLIT_NUM` | host 拒绝或完整扩展 |

每个 split 用例建议保留 metadata trace：

```text
FA: (BN, M, S2Start, S2End, isS2Split, firstFdDataWorkspaceIdx, s2SplitIdx)
FD: (coreEnable, BN, M, workspaceIdx, workspaceNum, mStartIdx, mNum)
```

并校验：

- 每个 staged `(slot, M row)` 恰好有一个 writer；
- 每个 FD reader 读取的 slot 都来自同一个 `(BN, M)`；
- 所有 row offset 小于 `stagingM`；
- FD 输出行集合无重叠、无空洞；
- split case 连续执行 20 次结果一致。

最终结果必须与 CPU golden reference 对齐。重复执行中 wrong element 位置变化，优先排查同步；wrong element 稳定但数值整体偏移，优先排查 partial O 归一化和 max/sum scale。

## 6. 同步与 UB 生命周期检查

SMLA 接入前应建立 ownership 表，避免在迁移中靠经验插 event。

| Data/buffer | Producer | Consumer | Required ordering |
| --- | --- | --- |
| Vec1 max/sum staging source | V | MTE3 | `V_MTE3` before GM write；MTE3 完成后释放 |
| Vec2 partial O staging source | V | MTE3 | `V_MTE3` before GM write；MTE3 完成后释放 |
| FD max/sum UB | MTE2 | V | copy all splits 后 `MTE2_V`，再算 scale |
| FD partial O UB | MTE2 | V | 等待 V 释放后覆盖，`MTE2_V` 后规约 |
| FD scale/max/sum UB between chunks | V | MTE2 | 下一 chunk 复用前等待 V 完成 |

single `partialO` buffer 的基本生命周期：

```text
initialization: SetFlag(V_MTE2, partial-buffer-free)

for each split:
    WaitFlag(V_MTE2, partial-buffer-free)
    MTE2 copy staged partial O into partialO
    SetFlag(MTE2_V, partial-ready)
    WaitFlag(MTE2_V, partial-ready)
    V reduce partialO into accumulatedO
    PipeBarrier(PIPE_V)
    SetFlag(V_MTE2, partial-buffer-free)
```

检查项：

- 每个 `WaitFlag` 都有初始 `SetFlag` 或前序 producer `SetFlag`；
- zero-row / no-work 路径也不能漏掉必要 release；
- `SyncAll()` 之前所有 staging 写入必须完成；
- `SyncAll()` 之后 FD 才能读 staging；
- 不用宽泛 `PIPE_ALL` 替代明确的 buffer ownership 证明。

## 7. 内存申请与复用方法

### 7.1 GM workspace

GM staging 必须按逻辑 FD slot 分配和索引：

```text
partialO bytes/slot = stagingM * dAlign * sizeof(float)
max bytes/slot      = stagingM * 8 * sizeof(float)
sum bytes/slot      = stagingM * 8 * sizeof(float)
```

整体 workspace：

```text
slotCount = SMLA 根据 core/head 映射计算的逻辑 slot 数
workspace += slotCount * maxSplits * (
    stagingM * dAlign * sizeof(float)
  + 2 * stagingM * 8 * sizeof(float)
)
```

如果 SMLA 存在类似 Split-G 的两个 producer 共同构成一个逻辑 output row group 的模式，应让两个 producer 写同一个逻辑 slot 的不同 M/head offset，而不是各自新开 slot。

### 7.2 UB lifecycle

FA 与 FD 的 UB 生命周期应分离。FA 阶段结束并完成全核 barrier 后，FD 可以重新申请或复用已释放 UB。

FD UB 建议按运行时任务尺寸申请：

```text
accumulatedO = mNum * dAlign * sizeof(float)
partialO     = dealRows * dAlign * sizeof(float)
max/sum      = workspaceNum * dealRows * 8 * sizeof(float)
lseExp       = dealRows * 8 * sizeof(float)
```

当前 `workspaceNum <= 2` 的场景优先使用 single `partialO` buffer。只有在 UB 资源充足且 profile 证明 DMA overlap 有收益时，再引入 ping-pong。

## 8. 重点风险与排查入口

| 风险 / 现象 | 优先排查项 |
| --- | --- |
| 结果稳定但整体数值错误 | `partial O` 是否归一化；FD scale 公式是否匹配 |
| max/sum 每 8 个值不一致 | broadcast 写入、`DataCopyExtParams` 字段顺序 |
| 连续输出行 zero 或 shift | output M stride、`mStartIdx` 地址项 |
| 多次运行错误位置变化 | 缺少 `SyncAll()`、event release 或 `PIPE_V` barrier |
| 稀疏/空 batch 死锁 | no-work core 是否进入 phase barrier |
| UB overflow/alignment fault | compact D 与 `dAlign` 混用；FD UB 未按 runtime 申请 |
| N/head 特定场景错误 | `stagingM` 和 `stagingMOffset` 是否硬套 QSMLA 规则 |
| terminal split 输出缺失 | `S2_END == 0` 是否被误当 token 0 |

小规模 dump 只保留用于验证协议的最小信息：

```text
slot index
M offset
one max/sum broadcast row
one partial O row
FD output address
```

定位后删除所有 device `printf` 和临时 dump。

## 9. 提交拆分建议

建议按以下顺序拆分提交，便于 bisect 和回滚：

1. `feat: align SMLA FD workspace with existing metadata ABI`
2. `feat: parse SMLA FD run info from sparse_flash_mla_metadata`
3. `feat: stage SMLA Vec1 LSE for S2 split FD`
4. `feat: stage SMLA Vec2 partial O for S2 split FD`
5. `feat: enable SMLA S2 split FD reduction`
6. `test: port QSMLA FD paramsets to SMLA`
7. `test: add SMLA S2 split FD stress coverage`

每个提交说明里至少附：

```text
- 覆盖的 metadata trace 摘要
- 运行的参数集
- 如果是迁移用例，注明来源 QSMLA paramset 名称
- 是否覆盖 one-loop split / multi-loop split
- 是否覆盖 no-work core barrier
- 是否连续重复 20 次稳定
```

## 10. 完成标准

声明 SMLA FD 适配完成前，必须同时满足：

- host allocation、metadata slot numbering、FA writes、FD reads 使用同一 staging ABI；
- 数值公式、dtype、归一化语义完全一致；
- event ownership 覆盖 normal、zero-row、zero-FA-load、split、chunk-reuse 路径；
- 输出地址按 SMLA layout 重新推导并通过多 layout 验证；
- 覆盖矩阵有实际执行 trace 和 CPU golden 结果；
- 迁移自 QSMLA 的 FD paramset 已在 SMLA schema 下执行通过；
- S2 split stress case 连续至少 20 次结果稳定；
- target build 与资源报告通过，无 UB overflow 或 alignment fault。
