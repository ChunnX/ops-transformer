# S2 Split Flash Decode 公共化实施方案

## 1. 目标与边界

本文将提交 `77fc33eb3fc3be10bfdb96914977419c045c4717` 中 MixedQuantSparseFlashMla 的 S2 Split Flash Decode（以下简称 FD）路径抽取到公共目录。目标新文件为：

```text
attention/sparse_flash_mla/op_kernel/arch35/flash_decode.h
```

公共文件只负责以下三个可复用阶段：

1. Vec1 最终 S2 loop 将 `max/sum` 暂存到 GM；
2. Vec2 最终 S2 loop 将归一化的 partial O 暂存到 GM；
3. FD 从 GM 读取所有 split 的数据、计算权重并规约 partial O。

不纳入公共文件的内容：metadata 生成与解析、S2 区间裁剪、算子特有的 `RunInfo/ConstInfo`、BSND/TND 输出地址计算、最终输出 copy、以及全核阶段屏障。它们属于算子调度和布局语义，继续保留在 `mixed_quant_sparse_flash_mla`。

本方案不修改 `tests/pytest/mixed_quant_sparse_flash_mla_paramset.py`；每个阶段直接使用其中现有用例验证。

### 1.1 实施原则：以代码剪切为主，适配为辅

本次公共化不是重写 FD。优先将提交 `77fc33eb` 中已经验证过的函数体原样剪切到公共头文件，再以极少量参数替换原先的 QSMLA 成员访问。除非编译或接口隔离必需，不做下列改动：

- 不改变 `Brcb`、`DataCopyPad`、`ComputeScaleValue_VF`、`ReduceFinalRes_const_VF` 的调用顺序、模板参数或 DMA 参数。
- 不改变现有 `V_MTE3/MTE3_V`、`V_MTE2/MTE2_V` 的 event 使用顺序。
- 不改变 chunk 大小、split 遍历次序、partial O 的写入格式和 max/sum 的 8-float 广播格式。
- 不顺带重构 `RunInfo`、`ConstInfo`、metadata、workspace 分配或输出 copy。

公共文件应先提供可直接容纳现有函数主体的轻量上下文（layout、GM 基址、UB、event id），而不是抽象出新的复杂框架。QSMLA 侧先保留同名私有包装函数，包装函数只组装参数并转调公共实现。这样每个阶段的 diff 可局限在一个被剪切的函数、一个公共实现和一个调用点。

## 2. 数值与内存协议

### 2.1 数值协议

对同一逻辑 M 行的第 `i` 个 S2 split，Vec1/Vec2 产生：

```text
m_i = local max
s_i = sum(exp(score - m_i))
p_i = sum(exp(score - m_i) * V) / s_i
```

FD 规约应计算：

```text
M   = max_i(m_i)
G   = sum_i(exp(m_i - M) * s_i)
w_i = exp(m_i - M) * s_i / G
O   = sum_i(w_i * p_i)
```

因此 Vec2 写入 staging 的内容必须是已经完成最终除法的 `p_i`。公共 reducer 使用现有 `ComputeScaleValue_VF` 与 `ReduceFinalRes_const_VF` 时，不得写入未除以 `s_i` 的 numerator。

### 2.2 GM staging ABI

workspace 使用连续三段：

```text
[partial O slots][max slots][sum slots]
```

每个 slot 的尺寸：

```text
partial O: stagingM * dAlign * sizeof(float)
max      : stagingM * 8 * sizeof(float)
sum      : stagingM * 8 * sizeof(float)
```

其中：

- `dAlign` 是 Vec2/FD 使用的 FP32 对齐 D 宽度；当前实现为 `dTemplateAlign64`。
- `stagingM` 在非 Split-G 场景为运行时 `gSize`；Split-G（当前 N=128）场景为 128。
- `max/sum` 的一行是 8 个相同 FP32 值的广播块，不能替换为单个 FP32 标量。
- `maxSplits` 当前为 2。metadata 生成端必须保证 `1 <= workspaceNum <= maxSplits`；不允许 kernel 静默截断第三个 split。

slot 索引协议固定如下：

```text
FA writer slot = firstFdDataWorkspaceIdx + s2SplitIdx
FD reader slot = fdRunInfo.workspaceIdx + splitIdx, splitIdx in [0, workspaceNum)
```

host workspace 分配、FA writer 和 FD reader 必须从同一 layout 规则推导 offset。

### 2.3 同步边界

- Vec1 写 max/sum 和 Vec2 写 partial O 均使用 `V_MTE3 -> MTE3_V`，确保 UB 数据被 DMA 消费后再复用。
- FD 中复用 single `partialO` UB 时，每个 split 执行 `V_MTE2 -> MTE2_V -> V_MTE2`；首轮必须有初始 `V_MTE2` set。
- 所有参与启动的核都必须进入 FA 完成后的 `SyncAll()`，即使该核 `hasLoad == 0`。FD reducer 只能在这个屏障之后开始读取 staging。
- `SyncAll()` 由算子 kernel 保留，不移入公共文件。

## 3. 目标接口与文件职责

公共文件建议定义 `AttentionCommon` 命名空间，并只接收基础数据结构和 Local/GlobalTensor，不依赖 QSMLA 的 `RunInfo`、`ConstInfo`。接口设计以承接原函数局部变量为目标，不追求一次性泛化。

```cpp
struct S2SplitFdLayout {
    uint32_t stagingM;
    uint32_t dAlign;
    uint32_t dValid;
    uint32_t slotCount;       // 已包含 maxSplits
    uint32_t maxSplits;       // 当前为 2
    uint32_t broadcastElems;  // 8
    uint32_t chunkRows;       // 16
};

struct S2SplitFdTask {
    uint32_t workspaceIdx;
    uint32_t workspaceNum;
    uint32_t mStart;
    uint32_t mNum;
};
```

公共文件提供三类接口。第一版参数可保留现有的 `workspaceIdx`、`stagingMOffset`、`halfMRealSize`、`dTemplateAlign64` 等标量名称，避免为了命名统一而改写函数主体：

```cpp
StageVec1Lse(...);       // max/sum -> staging
StageVec2PartialO(...);  // normalized partial O -> staging
Reduce(...);             // staging -> accumulatedO UB
```

接口的 event id 作为参数传入，公共文件不得自行分配或释放 TPipe event。buffer 的申请仍由 QSMLA 侧完成；公共 reducer 只使用传入的 `accumulatedO`、`partialO`、`blockMax`、`blockSum`、`lseExp` UB。

### 3.1 RunInfo/FdRunInfo 的实际依赖与拆分边界

`RunInfo` 与 `ConstInfo` 定义在 QSMLA 私有的 `op_kernel/arch35/util_regbase.h`。它们不应进入 common 接口；但在第一次剪切时，无需重构其定义，只需将公共代码实际读取的字段转为参数。

#### Vec1 staging 对 RunInfo 的依赖

`CopyMaxSumToStaging` 及其调用条件实际使用的 `RunInfo` 字段如下：

| 用途 | 字段 |
| --- | --- |
| 是否到最终 S2 loop | `isS2Split`、`s2LoopCount`、`s2LoopLimit` |
| 计算 staging slot | `firstFdDataWorkspaceIdx`、`s2SplitIdx` |
| 计算 staging M 行偏移 | `firstHalfMRealSize`、`goIdx`，以及 `ConstInfo::subBlockIdx` |
| 实际写入 M 行数 | `halfMRealSize` |

公共 `StageVec1Lse` 不传 `RunInfo`，而传入已经计算好的：`workspaceIdx`、`stagingMOffset`、`validRows`、`maxUb`、`sumUb`、临时广播 UB、layout、GM 基址和 event id。`isS2Split/s2LoopCount/s2LoopLimit` 的判断继续留在 QSMLA 的 `ProcessVec1` 原调用点。

#### Vec2 staging 对 RunInfo 的依赖

`ProcessVec2` 中 S2 split 的 staging 分支实际使用：

| 用途 | 字段 |
| --- | --- |
| 是否写 staging、是否为最终 loop | `isS2Split`、`s2LoopCount`、`s2LoopLimit` |
| staging slot/M 行偏移 | `firstFdDataWorkspaceIdx`、`s2SplitIdx`、`firstHalfMRealSize`、`goIdx` |
| 实际写入行数 | `vec2MRealSize` |

公共 `StageVec2PartialO` 只接收 `workspaceIdx`、`stagingMOffset`、`validRows`、`dValid`、`dAlign`、`vec2ResUb`、GM 基址和 event id。`FlashUpdateLastNew`、one-loop `LastDivNew`、`isS2Split` 分支和普通 `CopyOutAttentionOut` 继续留在 QSMLA 的 `ProcessVec2`。

#### FD reducer 对 FdRunInfo 的依赖

`ProcessFlashDecode` 的规约循环不依赖 `RunInfo`，使用的是 `FdRunInfo` 中的：

| 用途 | 字段 |
| --- | --- |
| staging 读起始 slot 与 split 数 | `workspaceIdx`、`workspaceNum` |
| FD 规约 M 行范围 | `mStartIdx`、`mNum` |
| QSMLA 最终输出地址 | `bn2Idx`、`mIdx` |
| 是否调度 FD | `coreEnable` |

其中前四个规约字段可以形成 common 的最小结构：

```cpp
namespace AttentionCommon {
struct S2SplitFdRunInfo {
    uint32_t workspaceIdx;
    uint32_t workspaceNum;
    uint32_t mStartIdx;
    uint32_t mNum;
};
}
```

`coreEnable`、`bn2Idx`、`mIdx` 保留在 QSMLA：前者是 QSMLA metadata 调度语义，后两者用于 BSND/TND 输出寻址。当前 `ProcessFlashDecode` 末尾构造临时 `RunInfo`，仅为了复用 `CopyOutAttentionOut` 所需的 `vec2MRealSize` 和 `attentionOutOffset`；这不属于 FD reducer 算法，仍留在 QSMLA 侧。

### 3.2 面向 SMLA 的复用边界

本公共文件定位为 S2 split FD 的 data-plane，使 SparseFlashMla（SMLA）也能复用：

- SMLA 可直接调用 `StageVec1Lse`，前提是提供 max/sum UB、staging GM 基址、slot、M 行偏移和 event id。
- SMLA 可直接调用 `StageVec2PartialO`，前提是最终 S2 loop 已经得到归一化 FP32 partial O，且使用相同 staging ABI。
- SMLA 可直接调用 `Reduce`，前提是自身补齐 FD metadata 到 `S2SplitFdRunInfo` 的转换、FD UB、所有核 phase barrier 和最终输出适配。
- SMLA 不应复用 QSMLA 的 `FdRunInfo` 原结构、`RunInfo`、`ConstInfo`、metadata 解析、S2 区间裁剪或 BSND/TND 输出地址计算。

因此 common 只承载 data-plane；QSMLA 和 SMLA 分别保留 control-plane：

```text
各自 metadata -> 各自计算 slot/M offset -> common Vec1/Vec2 staging
各自全核 SyncAll -> 各自 metadata 转 common S2SplitFdRunInfo -> common Reduce
-> 各自输出地址与 CopyOut
```

## 4. 分阶段实施与验证

每个阶段应独立提交。进入下一阶段前，必须满足本阶段的验收标准；失败时直接回退当前阶段，不叠加后续重构。每阶段遵循相同机械步骤：复制原函数到公共文件、只替换不可见成员为入参、将原位置改为同名薄包装、运行对应参数集、确认 diff 没有无关格式化或算法变化。

### 阶段 0：基线与 ABI 固化

**改动范围**

- 不改变任何运行时代码。
- 新建公共头文件的空壳、头文件保护和只满足现有函数编译的必要 include；此阶段不接入调用点。
- 在 host 和 kernel 侧梳理并注释以下值的来源：`stagingM`、`dAlign`、`slotCount`、`maxSplits`、`workspaceIdx`、`workspaceNum`、`mStart`、`mNum`。

**检查项**

- 确认 host workspace 中 `partial O / max / sum` 三段的 byte 公式与 kernel 完全一致；host 不应仅依赖当前恰好等于 512 的 D 常量，应以同一 `dAlign` 规则计算。
- 确认 `FdRunInfo` 字段与 metadata 字段一一对应。
- 确认 metadata 端对超过两个 split 的请求有拒绝或不可达保证。

**验证用例**

- `single_batch_2_cores`
- `single_batch_3_cores`
- `N32_single_batch_3_cores`
- `N128`

**验收标准**

- 不改变结果与性能路径；编译通过，以上用例全部通过。
- 记录一份 metadata trace，至少包含 FA 的 `firstFdDataWorkspaceIdx/s2SplitIdx/isS2Split` 和 FD 的 `workspaceIdx/workspaceNum/mStart/mNum`，用于后续阶段比对。

### 阶段 1：抽取公共 layout 与地址计算

**改动范围**

- 仅将现有 `StagingAttenOutBytes`、`StagingMaxSumBytes` 和三段 region 地址计算剪切为公共 inline helper；函数公式、返回类型和局部变量的语义保持不变。
- QSMLA 原 helper 保留为同名包装，内部只转调公共 helper。先不删除、不合并 `StagingMSize`、`GetFaStagingWorkspaceIdx`、`GetFaStagingMOffset`，以避免影响 Split-G 与 metadata 逻辑。
- 不在本阶段统一 host/kernel 的实现形式；只增加必要注释和上界比对。host 公式收敛留到抽取稳定后的独立小改动。

**验证用例**

- `N32_single_batch_3_cores`：非 Split-G 的动态 `gSize`。
- `single_batch_full_cores`、`single_batch_full_cores_with_blocks`：多核 slot 数和 workspace 上界。
- `N128`：Split-G 的 128 行 staging 与配对核 slot 数。

**验收标准**

- 输出与阶段 0 一致。
- host 计算的 workspace 字节数等于 kernel 访问的最后一个字节上界加一；可通过临时断言或 host trace 验证。
- Split-G 下两个 64-head producer 写入同一个逻辑 128 行 slot 的不同 M 偏移。

### 阶段 2：抽取 Vec1 的 max/sum 暂存

**改动范围**

- 将 `CopyMaxSumToStaging` 的函数体整体剪切到公共 `StageVec1Lse`，保持语句、局部变量与 DMA 参数顺序不变。
- 原 `CopyMaxSumToStaging` 保留函数签名，函数体缩减为：计算现有 `workspaceIdx/stagingMOffset`，将成员 `s2SplitStagingBase`、`dequantScaleBuff`、event id 作为参数传给 `StageVec1Lse`。
- Vec1 调用点保持不动：仍只在 `isS2Split && s2LoopCount == s2LoopLimit` 触发原包装函数。

**实现要点**

- 使用 `Brcb` 将每行 `max/sum` 扩为 8-float 广播块。
- `DataCopyExtParams` 的 `blockCount` 为有效 M 行数，`blockLen` 为 `8 * sizeof(float)`。
- 临时 UB 必须能同时容纳 max 与 sum 两个广播区；不在公共代码中假设 QSMLA 的 `dequantScaleBuff` 名称。

**验证用例**

- `single_batch_2_cores`：最小二 split 规约。
- `single_batch_3_cores`：非末尾 split 与末尾 split。
- `with_kvs_0`：无有效 S2 的 batch 不得产生陈旧 staging。
- `with_qs_0`：无有效 Q 的核仍需满足全局阶段屏障。

**验收标准**

- 结果与阶段 1 一致。
- 对一个 split M 行导出或调试读取 max/sum，8 个元素必须完全相同。
- 每个 `(slot, M row)` 至多一个 writer；无工作任务不写 staging。

### 阶段 3：抽取 Vec2 的 partial O 暂存

**改动范围**

- 只剪切 `ProcessVec2` 内 `if (runInfo.isS2Split)` 的 GM 写出分支到公共 `StageVec2PartialO`；`ProcessVec2` 的其它控制流保持原样。
- 原位置用一次公共函数调用替换，普通非 split 的 `CopyOutAttentionOut` 分支不动。
- 调用公共接口前的 `FlashUpdateLastNew`、`LastDivNew` 保持在原位置，避免数值路径重排。

**实现要点**

- staging 中 partial O 使用 FP32，GM 行宽为 `dValid`，UB 行宽为 `dAlign`，通过 `DataCopyPad` 处理 padding。
- `partial O` 的 M 偏移与 Vec1 的 M 偏移必须相同。
- Vec2 在 `vec2MBaseSize == 0` 的早退路径仍需保持跨核 buffer 的 release，不能导致上游或下游等待死锁。

**验证用例**

- `single_batch_2_cores`：二 split 的数值主路径。
- `N32_single_batch_3_cores`：N 小于 64 时不能错误使用固定 64 行 stride。
- `N128`：Split-G 的第二个 64-head 半区必须写入 M 偏移 64 后的位置。
- `with_tail`、`residual`：非对齐 KV 尾块和 residual 情况。

**验收标准**

- 用例精度与阶段 2 一致。
- 在 one-loop S2 split 用例上，partial O 已经过 final division；不得发生 FD 后的额外除法。
- 非 split 用例仍走直接输出路径，结果无变化。

### 阶段 4：抽取 FD chunk 规约

**改动范围**

- `InitFDBuffers` 保持在 QSMLA，不抽取；它依赖该算子的 TPipe 生命周期与成员 buffer。
- 仅将 `ProcessFlashDecode` 中从 staging region 计算开始，到 chunk 规约结束的连续代码段整体剪切为公共 `Reduce`；保持 chunk 循环、DataCopy、Wait/SetFlag 和 VF 调用逐行等价。
- `ProcessFlashDecode` 保留函数签名、`InitFDBuffers`、BSND/TND 的输出 offset、LocalTensor 获取和最终 `CopyOutAttentionOut`；它只将已有局部变量传入 `Reduce`。

**实现要点**

- chunk 宽度保持 16 行；最后一个 chunk 使用真实 `dealRowCount`。
- 先拷入全部 split 的 max/sum，等待 `MTE2_V` 后再计算 scale。
- `partialO` 使用单 buffer 时，严格维持每轮的 `V_MTE2 -> MTE2_V -> Reduce -> V_MTE2` 生命周期。
- 不把 `workspaceNum` 当作固定 2；以 task 字段循环，但入参检查不得超过 `maxSplits`。

**验证用例**

- `1_core_has_2_reduce`：一个核承担多个 FD task。
- `1_core_has_2_reduce_with_no_reduce_insight`：FD 与无规约工作交错。
- `multi_batch_full_cores`：多 batch 输出偏移。
- `single_batch_full_cores`、`single_batch_full_cores_with_blocks`：满核与跨 block 分配。
- `N128`：Split-G reducer 读取 128 行逻辑 slot。

**验收标准**

- FD 输出与阶段 3 一致，且每个 FD task 的输出行集合恰为 `[mStart, mStart + mNum)`，没有重叠或漏行。
- 多次执行同一组 split 用例结果稳定；建议连续执行 20 次。
- FD 使用的 UB 总量不超过目标 SoC 的可用 UB，编译资源报告通过。

### 阶段 5：收口、回归与删除冗余

**改动范围**

- 保留 QSMLA 中的同名薄包装和算子专用地址 helper；只删除确实重复且不再被调用的函数体片段。不要为了“整洁”继续合并 helper 或改变成员组织。
- 清理临时 trace、设备 `printf`、仅用于迁移的断言。
- 补充公共头文件的接口注释，写明数值、GM layout、同步和 `workspaceNum` 限制。

**回归用例**

执行 `tests/pytest/mixed_quant_sparse_flash_mla_paramset.py` 中全部已启用参数集，重点覆盖：

| 覆盖目标 | 已有参数集 |
| --- | --- |
| 两/三核 S2 split | `single_batch_2_cores`、`single_batch_3_cores` |
| N 小于 64 的动态 stagingM | `N32_single_batch_3_cores`、`N2` |
| 满核与跨 block | `single_batch_full_cores`、`single_batch_full_cores_with_blocks` |
| 单核多个 FD task | `1_core_has_2_reduce`、`1_core_has_2_reduce_with_no_reduce_insight` |
| 空 Q/空 KV 调度 | `with_qs_0`、`with_kvs_0` |
| TND、padding、尾块、residual | `tnd_padding_kv`、`tnd_padding_1_bak`、`tnd_padding_1`、`with_tail`、`residual` |
| Split-G | `N128` |
| 稀疏大 K | `K_1024`、`SCFA_33` |

**验收标准**

- 全量参数集通过；同一 split 用例重复 20 次没有不稳定结果。
- 非 FD 用例不发生性能或数值回退。
- `git diff` 中公共文件只包含可复用算法与基础数据协议；QSMLA 文件只保留算子适配与调度。

## 5. 需要重点审查的风险

1. **host/kernel workspace 尺寸不一致**：host 当前按固定 `D_SIZE=512` 计算，kernel 使用 `dTemplateAlign64`。公共化时必须统一为 `dAlign`。
2. **partial O 的归一化位置错误**：如果 Vec2 写入 numerator，现有 FD scale/reduce 公式会产生稳定但错误的结果。
3. **Split-G 行偏移错误**：N=128 场景第二个 64-head producer 必须从逻辑 M 行 64 开始写，不能新开 slot。
4. **metadata 越界或 slot 混用**：writer 和 reader 必须共享 slot 定义，禁止按本地核临时编号推导 slot。
5. **同步缺失**：任何 `hasLoad == 0` 核跳过 `SyncAll()` 都可能使其他核死锁；任何 UB 复用缺少初始 event set 都可能首轮死锁。
6. **输出地址混用**：FD 输出的 batch/token 基址、M block 基址、M 行基址必须分别推导。TND 必须使用 `cu_seqlens_q`，不能复用 BSND 的 `batch * s1Size`。

## 6. 提交建议

建议按以下顺序拆分提交，便于 bisect 和回滚。每个提交应以“剪切现有已验证代码”为描述主线，避免混入重命名、格式化或接口大改：

1. `refactor: add S2 split FD common layout`（阶段 0、1）
2. `refactor: extract S2 split Vec1 LSE staging`（阶段 2）
3. `refactor: extract S2 split Vec2 partial-O staging`（阶段 3）
4. `refactor: extract S2 split FD reduction`（阶段 4）
5. `refactor: clean QSMLA FD adapters and run full regression`（阶段 5）

每个提交说明中应附上对应参数集名称、metadata trace 摘要和编译资源结果。
