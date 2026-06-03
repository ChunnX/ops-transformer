# chunk_gated_delta_rule 流水分析

> 依据当前 checkout 的 `op_kernel/chunk_gated_delta_rule.h`、`stage1.h`、`stage2.h`、`stage3.h` 做的静态分析。

## 结论先说

1. `stage1 -> stage2 -> stage3` 这三段在当前实现里是**按 chunk group 纯串行**执行的。
2. 串行的含义是：`RunStage1(cg)` 结束后，才会 `SyncAll<false>()` 进入 `RunStage2(cg)`，再同步后进入 `RunStage3(cg)`。
3. 但每个 stage 内部都不是“一条线跑到底”，而是有自己的 **AIV / AIC 子流水**，靠 `CrossCoreWaitFlag` / `CrossCoreSetFlag` 做阶段交接。
4. 你说的 `cv`，按代码里实际命名，我这里理解为 `cvOffset_ / vBetaWsGm_` 这条支路，也就是 `v_beta = value * beta` 这一路。

## 外层调度

入口在 `CGDR::Process()`：

```cpp
RunStage1(cg);
SyncAll<false>();
RunStage2(cg, curState, finalState);
SyncAll<false>();
RunStage3(cg);
SyncAll<false>();
```

所以：

- `stage1` 完整结束后，才会开始 `stage2`
- `stage2` 完整结束后，才会开始 `stage3`
- 这里没有做 stage 间重叠调度

换句话说，**stage123 之间不是 pipeline overlap，而是 stage 级串行**。

## Stage1 的内部流水

Stage1 不是单核单路计算，而是分成两条角色明确的子路径：

- **AIV 侧**：做输入预处理、`g/beta` 相关的标量/向量计算、`k_cumdecay` / `v_beta` / `q_prime` 的生成
- **AIC 侧**：做矩阵乘主计算，主要是 `key@keyT`、`query@keyT`、逆矩阵块补完、`attn @ k_cumdecay`、`attn @ v_beta`

### Stage1 里 `TASK_RATIO=2`

`stage1` 里有这几个关键量：

- `halfChunkSize_ = chunkSize_ / TASK_RATIO`
- `subOffset_ = subBlockIdx_ * halfChunkSize_`
- `cvOffset_ = chunkSize_ * dv_`

这意味着 stage1 的 chunk 内部会按 `TASK_RATIO=2` 切成两半处理。对 `v_beta` 这条支路来说，workspace 是整块 `chunkSize × Dv`，但实际写入时会按 `subOffset_` 落到前半或后半：

- `subBlockIdx_ = 0` 写前半
- `subBlockIdx_ = 1` 写后半

所以 `cv` 不是独立 stage，而是 stage1 里的一个**chunk 内部半块写回路径**。

## Stage1 的实际时序

### 1. AIV 先做 Q/K 预处理

AIV 先把 query/key 搬到连续 workspace：

- `QKPreProcess(query...)`
- `QKPreProcess(key...)`

完成后发 `0x9`：

```cpp
CrossCoreSetFlag<0x2, PIPE_MTE3>(0x9);
```

这个信号告诉 AIC：

- `q/k` 连续化已经完成
- 可以开始 `key @ keyT`

### 2. AIC 做 `key @ keyT`

AIC 等 `0x9` 后开始：

```cpp
keyConGm @ keyConGm -> kkWsGm
```

做完后发 `0x8`，通知 AIV：

- `kkWsGm` 已可用
- 可以继续做 `KKBetaCompute` 和后续 inverse 相关计算

### 3. AIV 并行做 `g/beta` 支路准备

在等 `0x8` 之前，AIV 这边会先把不依赖 `kkWsGm` 的事情做掉：

- `GCumExpCompute`：算 `g.cumsum().exp()`
- `GammaCompute`：算 mask/gamma 矩阵
- `BetaCopyInWithStride`：把 `beta` 搬到 local

这段就是 stage1 内部最明显的 overlap：

- AIC 在算 `key @ keyT`
- AIV 在做 `g/beta` 相关准备

### 4. AIV 等 `0x8` 后做 inverse 前半

AIC 完成 `key @ keyT` 以后发 `0x8`，AIV 才进入：

- `KKBetaCompute`
- `InverseCompute`

这里的依赖关系很直接：

- `KKBetaCompute` 需要 `kkWsGm`
- `InverseCompute` 需要 `gamma` / `attnWsGm` 这条中间态

做完后 AIV 发 `0x7`，通知 AIC 可以接着补完逆矩阵相关计算。

### 5. AIC 做 inverse 补完和两个后续 matmul

AIC 等 `0x7` 后：

- `AttnInverseMMCompute`
- 等 AIV 的 `k_cumdecay` 就绪后做 `attn @ k_cumdecay`
- 再等 `0x5` 后做 `attn @ v_beta`

所以 AIC 侧的后半段也不是一次性完成，而是被 AIV 产物分段唤醒。

### 6. AIV 继续生成 `k_cumdecay` / `v_beta` / `q_prime`

AIV 在 `0x7` 之后发起：

- `GBKCompute`：生成 `k_cumdecay`，`gOptional` 打开时还会生成 `kg`
- `VBetaCompute`：生成 `v_beta`
- `QPrimeCompute`：生成 `q_prime`

然后分别发：

- `0x6`：`k_cumdecay` 可用
- `0x5`：`v_beta` 可用

### Stage1 事件链

按依赖顺序可以简化成：

```text
AIV: QKPreProcess
  -> 0x9
AIC: key@keyT
  -> 0x8
AIV: KKBeta + Inverse
  -> 0x7
AIC: AttnInverseMMCompute
  -> 0x6
AIC: attn @ k_cumdecay
  -> 0x5
AIC: attn @ v_beta
```

这里的关键点是：

- **不是 stage1 整体串行**
- 而是 **AIV / AIC 两条子流水串接**
- 每一步都被前一步的中间张量和跨核 flag 限住

## 各 stage 的结果落点

| Stage | 结果 | 存放位置 | 备注 |
|---|---|---|---|
| Stage1 | `gCum` | `gCum_` | 只有 `gOptional` 打开时才会写，且由 `subBlockIdx_ == 0` 的 AIV 子块落 GM |
| Stage1 | `k_cumdecay` | `kCumDecay_` | AIC 在 `attn @ gBKWs` 后写回，是 Stage2 的输入 |
| Stage1 | `vInner` | `vInner_` | AIC 在 `attn @ v_beta` 后写回，是 Stage2/Stage3 的输入 |
| Stage1 | `qPrime` | `qPrime_` | AIV 直接写回，是 Stage2 的输入 |
| Stage1 | `kg` | `kg_` | 仅 `gOptional` 打开时写回，是 Stage2 的输入 |
| Stage1 | `qkt` | `qkt_` | AIV 写回，供 Stage3 使用 |
| Stage2 | `finalState` | `finalState_` | 递归状态输出；AIV 先 seed/刷新，AIC 再用 `CalStateNew` 覆盖成新状态 |
| Stage2 | `qPrime @ state^T` 中间态 | `out_` / `attnOut` 这块 GM 域 | 当前代码直接写到最终输出域里，属于 Stage2 的中间结果，不是独立对外结果 |
| Stage3 | 最终 attention output | `out_` / `attnOut` | 最终输出，和 Stage2 的中间态复用同一块 GM 域，Stage3 会覆盖这里 |

补一句：`CGDR` 里虽然预留了 `attnInter_` 这块 workspace 成员，但当前 `RunStage2()` 没有把它接进来，Stage2 直接写的是 `out_` 这块输出域。

## Buffer 申请和生命周期

先区分两类 buffer：

- **GM workspace**：host 侧通过 `GetWorkspaceSize()` 一次性申请，kernel 里 `GetUserWorkspace(workspaceGM)` 后按 offset 切分。它不是每个 stage 重新申请，只是不同 stage 重新解释/复用其中的某些区域。
- **TPipe 本地 buffer**：`InitBuffer()` 在 kernel 内给 TQue/TBuf 切 UB 资源。当前实现里 `InitMask()` 后会 `pipe_->Reset()`，每个 `RunStage1/2/3()` 末尾也都会 `pipe_->Reset()`，所以 stage 内的 TPipe 本地 buffer 是**每个 chunk group、每个 stage 重新 InitBuffer，再在 stage 结束释放/失效**。

本文只统计当前算子显式写出来的 `InitBuffer()`、`GetWithOffset()` 和 `user workspace` 切分；`MatmulImpl` 内部的 L0/L1/L0C 资源由 matmul 库按 tiling 管理，这里不展开。

符号：

- `C = chunkSize`，当前 tiling 固定为 64
- `H = halfChunkSize = C / TASK_RATIO`，`TASK_RATIO = 2`
- `P = stageOneParaNum`，当前 host tiling 固定为 4
- `Core = aiCoreNum`
- `S = maxGroupLength`
- `DkA = ceil(Dk / 16) * 16`
- `DvA = ceil(Dv / 16) * 16`
- `M = max(DvA / 2, DkA / 2, C)`

### Kernel 外层生命周期

```text
op.Init()
  InitMatmul()
  InitMask()   -> InitBuffer(tmpBuff_) -> pipe_->Reset()

for each chunk group:
  RunStage1()  -> Stage1 InitBuffer(...) -> Process -> pipe_->Reset()
  RunStage2()  -> Stage2 InitBuffer(...) -> Process -> pipe_->Reset()
  RunStage3()  -> Stage3 InitBuffer(...) -> Process -> pipe_->Reset()
```

所以回答“每个阶段会不会重新申请和分配 buffer”：

- **UB/TQue/TBuf 会按 stage 重新初始化**。
- **GM workspace 不会按 stage 重新申请**，是一次申请后切成 inter workspace、mask workspace、stage temp workspace 复用。

### 整体 GM workspace 切分

`CGDR::Init()` 里从 `user` 开始顺序切分：

| 名称 | Kernel 视角大小 | 生命周期 | 说明 |
|---|---:|---|---|
| `gCum_` | `4 * Nv * S` | 当前 chunk group 内 Stage1 -> Stage2/Stage3 | `float`，`gOptional` 打开时有效 |
| `kCumDecay_` | `2 * Nv * S * Dk` | 当前 chunk group 内 Stage1 -> Stage2 | `bf16` |
| `vInner_` | `2 * Nv * S * Dv` | Stage1 写入，Stage2 会更新/复用，Stage3 读取 | `bf16` |
| `qPrime_` | `2 * Nv * S * Dk` | 当前 chunk group 内 Stage1 -> Stage2 | `bf16` |
| `attnInter_` | `2 * Nv * S * Dv` | 当前实现预留但未接入 | `RunStage2()` 当前直接写 `out_` |
| `kg_` | `2 * Nv * S * Dk` | 当前 chunk group 内 Stage1 -> Stage2 | `bf16`，`gOptional` 打开时由 Stage1 写 |
| `qkt_` | `2 * Nv * S * C` | 当前 chunk group 内 Stage1 -> Stage3 | `bf16` |
| `highState_` | `4 * B * Nv * Dv * Dk` | 当前实现预留但未使用 | `float` |
| `stageOneMask_` | `4 * C * C * Core * 2` | 整个 kernel call | `InitMask()` 生成，Stage1 读取 |
| `stageThreeMask_` | `4 * C * C * Core * 2` | 整个 kernel call | `InitMask()` 生成，Stage3 读取 |
| `stageWsAddr_` | host 侧按 `4 * C * (2C + 3Dk + Dv) * P * Core` 预留 | 每个 stage 临时复用 | Stage1/Stage3 都会用这段 GM temp |

注意：host 侧 `interWorkspaceSz` 计算时很多中间量都按 `float` 大小预留，kernel 里 `lowType` 实际按 `bf16` 切分，因此 workspace 整体是偏保守的。

### InitMask 的本地 buffer

只在 `op.Init()` 期间执行一次，且只在 AIV 分支执行：

| Buffer | 数量/深度 | 大小 | 生命周期 |
|---|---:|---:|---|
| `tmpBuff_` | 1 个 TBuf | `C * C * sizeof(float)` | `InitMask()` 内有效，结束后 `pipe_->Reset()` |
| `cCFloat_` | `tmpBuff_` 内 1 个 LocalTensor | `C * C * sizeof(float)` | 构造 `stageOneMask_` / `stageThreeMask_` 时复用 |

`InitMask()` 的 GM 结果是 `stageOneMask_` 和 `stageThreeMask_`，它们会保留到整个 kernel call 结束。

### Stage1 本地 UB buffer

Stage1 的本地 buffer 只在 AIV 分支执行 `InitLocalBuffers()`。AIC 分支主要调用 matmul，不走这套 UB buffer 初始化。

| Buffer | 数量/深度 | 大小 | 生命周期 |
|---|---:|---:|---|
| `inQueue_` | `VECIN` depth 1 | `C * M * sizeof(float)` | 每次 `RunStage1()` 内有效 |
| `outQueue_` | `VECOUT` depth 1 | `C * M * sizeof(float)` | 每次 `RunStage1()` 内有效 |
| `gOutQueue_` | `VECOUT` depth 1 | `C * sizeof(float)` | 仅 `gOptional` 打开时初始化 |
| `tmpBuff_` | 1 个 TBuf | `140 * 1024` bytes | Stage1 内所有 LocalTensor 从这里切分 |

`tmpBuff_` 内部切分：

| LocalTensor | 大小 | 复用/别名关系 | 用途 |
|---|---:|---|---|
| `betaUbBfloat16_` | `H * sizeof(bfloat16_t)` | 独立 | beta bf16 gather 暂存 |
| `gCumUbFloat_` | `C * sizeof(float)` | 独立 | g 输入/前缀和源 |
| `gBUbFloat_` | `H * sizeof(float)` | 独立 | `-beta * exp(g)` 中间量 |
| `gEndBroadUbFloat_` | `H * sizeof(float)` | 独立 | `kg` 的末端 g broadcast |
| `betaUbFloat_` | `H * P * sizeof(float)` | 独立 | 多个并行 chunk 的 beta float |
| `gBroadUbFloat_` | `C * M * P * sizeof(float)` | `gammaUbFloat_` / `valueUbFloat_` / `qUbFloat_` 共用 | 大块向量计算复用区 |
| `gTransBroadUbFloat_` | `C * M * sizeof(float)` | `attnUbFloat_` / `gCumExpBroadUbFloat_` 共用 | gamma 转置广播、attn 中间、q_prime broadcast |
| `qUbFloatCon_` | `H * DkA * sizeof(float)` | `kUbFloatCon_` 共用 | 连续 Q/K 的 float 暂存 |
| `inputGatherBuffer_` | `C * sizeof(uint32_t)` | 独立 | gather index |
| `gCumSumUbFloat_` | `C * P * sizeof(float)` | 独立 | 每个并行 chunk 的 g cumsum |
| `gCumExpUbFloat_` | `C * P * sizeof(float)` | 独立 | 每个并行 chunk 的 exp(g cumsum) |
| `inverseBuffer_` | `H * H * 5 * sizeof(float)` | `qPrimeUbFloat_` / `vBetaUbFloat_` / `gBKUbFloat_` / `kgUbFloat_` / `kkLocal_` 共用 | inverse 和多条 RHS 的临时复用区 |
| `inverseGatherBuffer_` | `32 * sizeof(uint32_t)` | 独立 | inverse gather index |
| `inverseRes_` | `C * H * sizeof(float)` | 独立 | inverse 结果临时区 |

以当前常见上限 `C=64, H=32, P=4, DkA<=128, DvA<=128, M=64` 估算，`tmpBuff_` 实际切分约 `130496` bytes，小于固定申请的 `140KB`。

### Stage1 的 GM 临时 buffer

Stage1 还会在 `stageWsAddr_` 里按 core 和 `P` 切 GM temp。它们不是 UB buffer，生命周期是当前 `RunStage1()`，stage 结束后逻辑失效，后续 stage 可以复用 `stageWsAddr_`。

| GM temp | 每 core 大小 | 生产者 | 消费者 |
|---|---:|---|---|
| `gBKWsGm_` | `P * C * Dk * sizeof(bfloat16_t)` | AIV `GBKCompute` | AIC `attn @ k_cumdecay` |
| `queryConGm_` | `P * C * Dk * sizeof(bfloat16_t)` | AIV `QKPreProcess(query)` | AIC `query @ keyT` / AIV `QPrimeCompute` |
| `keyConGm_` | `P * C * Dk * sizeof(bfloat16_t)` | AIV `QKPreProcess(key)` | AIC `key @ keyT` / AIV `GBKCompute` |
| `kkWsGm_` | `P * C * C * sizeof(bfloat16_t)` | AIC `key @ keyT` | AIV `KKBetaCompute` |
| `attnWsGm_` | `P * C * C * sizeof(bfloat16_t)` | AIV inverse / AIC inverse 补完 | AIC `attn @ k_cumdecay` 和 `attn @ v_beta` |
| `vBetaWsGm_` | `P * C * Dv * sizeof(bfloat16_t)` | AIV `VBetaCompute` | AIC `attn @ v_beta` |

Stage1 这几块 GM temp 每 core 总量约为：

```text
P * C * (3 * Dk + 2 * C + Dv) * sizeof(bfloat16_t)
```

跨全部 core 再乘 `Core`。

### Stage2 本地 UB buffer

Stage2 的 `InitLocalBuffers()` 一进来就判断 `if ASCEND_IS_AIC return;`，所以这套本地 buffer 只属于 AIV 分支。AIC 分支只使用 matmul 和 GM 输入输出。

| Buffer | 数量/深度 | 大小 | 生命周期 |
|---|---:|---:|---|
| `inQueue_` | `VECIN` depth 1 | `max(C, Dv) * DkA * sizeof(bfloat16_t)` | 每次 `RunStage2()` 内有效 |
| `outQueue_` | `VECOUT` depth 1 | `max(C * C * sizeof(bfloat16_t), Dv * DkA * sizeof(bfloat16_t))` | 每次 `RunStage2()` 内有效 |
| `tmpBuff_` | 1 个 TBuf | `(max(C, Dv) * DkA + 8) * sizeof(float)` | Stage2 内有效 |

`tmpBuff_` 内部切分：

| LocalTensor | 大小 | 用途 |
|---|---:|---|
| `tmpBuffer1_` | `Dv * DkA * sizeof(float)`，offset 按 32B 对齐推进 | state cast/muls 临时区 |
| `lastGCum_` | `1 * sizeof(float)` | 当前 chunk 末尾 g 累积量的 exp 值 |

Stage2 没有显式使用 `stageWsAddr_` 里的 GM temp。它读取 Stage1 留在 inter workspace 的 `qPrime_`、`vInner_`、`gCum_`、`kCumDecay_`、`kg_`，写 `finalState_`，并把 `qPrime @ state^T` 写进 `out_` 这块输出域作为 Stage2 中间态。

### Stage3 本地 UB buffer

Stage3 的 `Init()` 中 AIC 分支直接返回；AIV 分支里 `GetSubBlockIdx() == 1` 也直接返回。因此真正初始化 UB buffer 的只有 **AIV subBlock 0**。

| Buffer | 数量/深度 | 大小 | 生命周期 |
|---|---:|---:|---|
| `inQueue_` | `VECIN` depth 1 | `C * max(C, DvA) * sizeof(bfloat16_t)` | 每次 `RunStage3()` 内有效 |
| `outQueue_` | `VECOUT` depth 1 | `C > DvA ? C * C * sizeof(float) : C * DvA * sizeof(bfloat16_t)` | 每次 `RunStage3()` 内有效 |
| `tmpBuff_` | 1 个 TBuf | `4 * C * C * sizeof(float)` | Stage3 内有效 |

`tmpBuff_` 内部切分：

| LocalTensor | 大小 | 用途 |
|---|---:|---|
| `tmpBuffer1_` | `C * C * sizeof(float)` | gamma/mask 中间 |
| `tmpBuffer2_` | `C * C * sizeof(float)` | qkt cast/scale/mul 中间 |
| `maskBuffer_` | `C * C * sizeof(float)` | 从 `stageThreeMask_` 搬入的 mask |
| 未命名保留区 | `C * C * sizeof(float)` | `STAGE3_BUFFER_COUNT=4`，当前只显式切了 3 块 |

Stage3 还会用 `stageWsAddr_` 里的 `tmpGM_` 存 masked qkt：

| GM temp | 大小 | 生命周期 | 用途 |
|---|---:|---|---|
| `tmpGM_` | 每 core `C * C * sizeof(bfloat16_t)` | Stage3 当前 chunk 内有效，逐 chunk 覆盖 | AIV 写 masked qkt，AIC 读后做 `masked_qkt @ vInner` |

`tmpGM_` 的 base 是 `stageWsAddr_ + Core * C * C * sizeof(float)`，说明 Stage3 也是复用 `stageWsAddr_` 这块 stage temp GM，不是重新申请一块新的 workspace。

## 你问的 `cv` 是什么

按当前代码，`cv` 对应的就是这条路径：

- `cvOffset_ = chunkSize_ * dv_`
- `vBetaWsGm_`
- `VBetaCompute(...)`

它的语义是：

```text
value[chunk] × beta[chunk] -> v_beta[chunk, Dv]
```

然后 AIC 再用：

```text
attn[chunk, chunk] @ v_beta[chunk, Dv] -> vInner
```

所以 `cv` 不是“copy+vector”的泛称，而是 stage1 里 **value / beta 这一路的中间结果 workspace**。

## stage2 / stage3 简述

### Stage2

Stage2 也是 `AIV + AIC` 两角色协作，但它关注的是 recurrent state 更新：

- AIV：处理 `curState`、`gCumExp`、`CopyOutState`
- AIC：算 `k_cumdecay @ state^T`、`q_prime @ state^T`、`v_inner^T @ kg`

它的内部也有 `0x2 / 0x5 / 0x4` 这类交接，但 stage2 自己仍然是在 `RunStage2()` 里单独串行执行。

### Stage3

Stage3 也是 stage 级串行，但内部是：

- AIV：先做 masked `QKT`
- AIC：再做 `masked_qkt @ vInner`

它和 stage1 类似，也是一个明确的 AIV/AIC 两段式。

## 一句话总结

这个算子外层是：

```text
Stage1 -> Stage2 -> Stage3
```

严格串行。

但 stage1 内部不是直线，而是：

```text
AIV 预处理 / g-beta 支路
   和
AIC 主 matmul / inverse / 乘法支路
```

靠事件号把中间结果一段一段接起来。
