# CausalConv1d 算子设计文档

## 1. 算子功能概述

CausalConv1d 实现因果一维卷积，用于序列建模（如 Mamba/SSM 架构）。因果性意味着每个时间步的输出仅依赖于当前及历史输入，看不到未来信息。

卷积核宽度 $W \in \{2, 3, 4\}$，支持 fp16 / bf16，可选偏置 $b$ 与 SiLU 激活。算子同时维护卷积状态缓存 `conv_states`，用于增量推理时高效处理单 token 输入。

---

## 2. 公式计算

### 2.1 基础公式

对二维修剪后的输入 $x \in \mathbb{R}^{T \times D}$（$T$ 为序列长度，$D$ 为特征维度），权重 $w \in \mathbb{R}^{W \times D}$，偏置 $b \in \mathbb{R}^{D}$（可选）：

$$y_t = \text{Act}\left(\sum_{j=0}^{W-1} w_j \odot x_{t-j} + b\right), \quad t = 0, 1, \dots, T-1$$

其中 $\odot$ 表示逐元素乘，$\text{Act} \in \{\text{Identity}, \text{SiLU}\}$。

展开为逐元素形式（对第 $d$ 个特征维度）：

$$y_{t,d} = \text{Act}\left(\sum_{j=0}^{W-1} w_{j,d} \cdot x_{t-j,\ d} + b_d\right)$$

### 2.2 各输入对公式的影响

| 输入 | shape | 公式中的角色 | 缺失时的行为 |
|------|-------|-------------|-------------|
| `x` | `[T, D]` 或 `[B, S, D]` | 当前及历史输入 $x_{t-j}$ | 必须提供 |
| `weight` | `[W, D]` | 卷积权重 $w_j$，$W$ 由 weight.shape[0] 确定 | 必须提供 |
| `bias` | `[D]` | 偏置项 $b$ | 不提供则 $b=0$，此时公式退化为纯加权和 |
| `conv_states` | `[L, S_len, D]` | 缓存历史输入，用于初始化 ring buffer 中的 $x_{-1}, x_{-2}, \dots$ | 必须提供；若无历史则内容应填 0 |
| `query_start_loc` | `[batch+1]` | 变长序列模式下，`x[ qsl[i] : qsl[i+1] ]` 为第 i 个序列 | 2D 变长模式必须提供；batch 模式可选 |
| `cache_indices` | `[batch]` | 第 i 个序列映射到 `conv_states[cache_indices[i]]` | 不提供时按 identity 映射（seq i → cache slot i） |
| `initial_state_mode` | `[batch]` | 0: 无初始状态(历史填零)；1: 从 conv_states 加载历史 | 不提供时默认全部无初始状态 |
| `num_accepted_tokens` | `[batch]` | 投机解码时，每个序列实际接受的 token 数，控制状态偏移 | 不提供时无投机解码逻辑 |
| `activation_mode` (attr) | — | 0: $y_t = \sum w_j \cdot x_{t-j} + b$; 1: $y_t = \text{SiLU}(\sum w_j \cdot x_{t-j} + b)$ | 默认 0 |
| `run_mode` (attr) | — | 0: 前向模式；1: 增量更新模式 | 默认 0 |

### 2.3 两种运行模式的计算差异

**run_mode = 0（前向/fn）**：输入完整序列 $x[0:T]$，输出完整 $y[0:T]$。Ring buffer 从 `conv_states` 加载 $x_{-W+1}, \dots, x_{-1}$ 作为历史 padding，然后遍历 $t=0..T-1$ 依次计算。

**run_mode = 1（更新/update）**：输入新 token（每序列 1 个或投机解码时多个）。Ring buffer 从 `conv_states` 加载历史，计算卷积输出后，将序列末尾的 $(W-1)$ 个 token 写回 `conv_states` 供下次调用使用。

$$t = T_{\text{new}}-1 \text{ 时: } \text{conv\_states}[\text{pos}] \leftarrow x'_{T_{\text{new}} - W + 1 + \text{pos}}, \quad \text{pos}=0,\dots,W-2$$

---

## 3. Proto 定义

```cpp
REG_OP(CausalConv1d)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(weight, TensorType({DT_BF16, DT_FLOAT16}))
    .OPTIONAL_INPUT(bias, TensorType({DT_BF16, DT_FLOAT16}))
    .INPUT(conv_states, TensorType({DT_BF16, DT_FLOAT16}))
    .OPTIONAL_INPUT(query_start_loc, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(cache_indices, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(initial_state_mode, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(num_accepted_tokens, TensorType({DT_INT64}))
    .ATTR(activation_mode, Int, 0)
    .ATTR(pad_slot_id, Int, -1)
    .ATTR(run_mode, Int, 0)
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16}))
    .OP_END_FACTORY_REG(CausalConv1d)
```

**输入 shape 约束：**

| 模式 | x shape | weight shape | conv_states shape |
|------|---------|-------------|-------------------|
| fn 变长 | `[cu_seqlen, D]` | `[W, D]` (W=2/3/4) | `[≥batch, state_len, D]` |
| fn 定长 | `[B, S, D]` | `[W, D]` | `[≥B, state_len, D]` |
| update 变长 | `[cu_seqlen, D]` | `[W, D]` | `[≥batch, state_len, D]` |
| update 固定batch | `[B, 1, D]` | `[W, D]` | `[≥B, state_len, D]` |

- $D$ 必须是 16 的整数倍
- conv_states.shape[1] $\ge$ W-1（update 模式需更大）
- `query_start_loc[0] == 0`，`query_start_loc[last] == cu_seqlen`

---

## 4. aclnn 接口定义

两段式接口：

```c
// 第一段：获取 workspace 大小并创建 executor
aclnnStatus aclnnCausalConv1dGetWorkspaceSize(
    const aclTensor *x, const aclTensor *weight, const aclTensor *bias,
    aclTensor *convStates,
    const aclTensor *queryStartLoc, const aclTensor *cacheIndices,
    const aclTensor *initialStateMode, const aclTensor *numAcceptedTokens,
    int64_t activationMode, int64_t padSlotId, int64_t runMode,
    aclTensor *y, uint64_t *workspaceSize, aclOpExecutor **executor);

// 第二段：执行计算
aclnnStatus aclnnCausalConv1d(void *workspace, uint64_t workspaceSize,
                               aclOpExecutor *executor, aclrtStream stream);
```

内部调用链：`aclnnCausalConv1dGetWorkspaceSize` → `CausalConv1dCommonProcess`（处理 contiguous、CreateView）→ `l0op::CausalConv1d` → `ADD_TO_LAUNCHER_LIST_AICORE` → 框架调度 Davinci kernel。

---

## 5. Tiling 切分方案

### 5.1 总体流程

```
GetPlatformInfo(ubSize, coreNum)
     │
     ├─ run_mode=0 (fn)
     │    └─ ChooseFnHostPlan → 确定 caseKind + executionPlan + baseDimChoice
     │         ├─ dim ≤ 4096 → TOKEN_FIRST  + CUTBS
     │         └─ dim > 4096 → TOKEN_DIM_CO_SPLIT + CUTBSD
     │       │
     │       ├─ ChooseUnifiedFnTokenBlockPlan → tokenBlockSize/Cnt/Grid
     │       ├─ BuildFnTokenCoreMappingChoice → blockDim
     │       └─ BuildFnTokenSeqRangePlan → 变长序列 token→seq 映射
     │
     └─ run_mode=1 (update)
          └─ ChooseCanonicalUpdateBaseDimChoice
               从 {4096,2048,1024,512,384,192} 选 baseDim
               grid = batch × baseDimCnt ≥ coreNum 为硬约束
     │
     ▼
SetBlockDim / SetWorkspace / SetTilingKey → 回填 CausalConv1dTilingData
```

### 5.2 切分轴

**Update 模式** — 沿 `batch × baseDimCnt` 二维网格切分：

```
grid[i, j] → 负责 seq[i] 的 dim[j*baseDim : (j+1)*baseDim] 通道块
每个 core 通过 blockIdx + k×blockNum 遍历整个网格
```

**Fn 模式** — 三层切分层次：

```
TokenTile(k) × baseDimIdx(j) → 负责 token 范围 [k·tokenBlockSize, (k+1)·tokenBlockSize)
                              中所有 seq 的 dim[j·baseDim : (j+1)·baseDim] 通道
```

### 5.3 baseDim 选择策略

**`ChooseCanonicalUpdateBaseDimChoice`**（update 模式）：
1. 从候选列表 `{4096, 2048, 1024, 512, 384, 192}` 遍历
2. 优先恰好整除 dim 的候选，选 grid ≥ coreNum 且 gap 最小的
3. 若无≥coreNum 的候选，选 grid 最大的 under-utilized 方案
4. 第一轮全是 under 时，放宽整除约束再搜一轮

**`ChooseFnTokenDimCoSplitBaseDimChoice`**（fn 模式 dim>4096）：
1. 由 UB 大小约束计算 `ubLimitedBaseDim`
2. 公式：`ubLimitedBaseDim = (ubSize - 512) / (5×2 + 2×2 + 8×4)`，向下对齐到 16
3. 调整 baseDimCnt 使其能整除 coreNum（整除性保证负载均衡）
4. 若调整后 baseDim 超出 UB 限制或 4096，回退到 UB 约束值

### 5.4 Token Tiling（fn 模式特有）

将 `cuSeqlen` 个 token 按 tokenCoreBudget 均分：

```
tokenCoreBudget = CUTBS ? coreNum : coreNum / baseDimCnt
idealBlockSize = ceil(cuSeqlen / tokenCoreBudget)
tokenBlockCnt  = ceil(cuSeqlen / idealBlockSize)
```

对于变长序列，预计算 `TokenSeqRangePlan`：每个 token tile 映射到它跨越的 seq 范围 `[startSeq, endSeq]`，容量上限 128。kernel 端可直接查表而无需二分搜索。

### 5.5 Workspace 管理

仅在 fn 模式 + `hasInitialStateMode != 0` 时需要 workspace：

```
workspace = AscendC 保留区(16MB) + numCacheLines × stateLen × D × sizeof(T)
```

用于 `PrefetchInitStatesToWorkspace`：所有 core 的 channel block 各自 dump `conv_states` 到 workspace，SyncAll 后再从 workspace 读回 ring buffer。避免多 core 并行写 `conv_states` 时的数据竞争。

### 5.6 chip 差异化

Tiling 层不区分芯片型号，仅通过 `ubSize` 和 `coreNum` 差异间接影响切分结果。config 层（binary.json、simplified_key.ini）针对 ascend950 配置，`opFile.value = "causal_conv1d_apt"`。

---

## 6. Kernel 实现方案

### 6.1 文件组织

```
op_kernel/
  causal_conv1d_apt.cpp       # kernel 入口 (__global__)
  causal_conv1d.h              # 基类：UB 管理、计算原语、调度逻辑
  causal_conv1d_common.h       # Ring Buffer 工具、常量、CalcBuf 内存布局
  causal_conv1d_fn.h           # fn 模式子类
  causal_conv1d_update.h       # update 模式子类
  causal_conv1d_fn_tasks.h     # fn TokenTiling 调度逻辑
  causal_conv1d_tiling_data.h  # Tiling 数据结构体
  causal_conv1d_tiling_key.h   # 模板参数声明 + 实例化表
  arch35/causal_conv1d_regbase.h  # arch35 平台寄存器级 SIMD 优化
```

### 6.2 模板实例化

通过 3 个模板参数产生 7 组编译期特化：

| runModeKey | widthKey | fnPlanKey | 场景 |
|-----------|----------|-----------|------|
| UPDATE | RUNTIME | INVALID | update 模式（W 运行时确定） |
| FN | W=2 | CUTBS | fn + W=2 + dim≤4096 |
| FN | W=3 | CUTBS | fn + W=3 + dim≤4096 |
| FN | W=4 | CUTBS | fn + W=4 + dim≤4096 |
| FN | W=2 | CUTBSD | fn + W=2 + dim>4096 |
| FN | W=3 | CUTBSD | fn + W=3 + dim>4096 |
| FN | W=4 | CUTBSD | fn + W=4 + dim>4096 |

TilingKey 格式：`(runModeKey << 0) | (widthKey << 1) | (fnPlanKey << 3)`，由 tiling 层计算后传入 `context->SetTilingKey()`。

### 6.3 UB 内存布局

```
calcBuf (VECCALC): (MAX_WIDTH + 4) × 4096 × sizeof(float)
  ├─ [0, 4×4096):    weightF[4]    卷积权重（W<4 时高位补零）
  ├─ [4×4096, 5×4096):  biasF      偏置（无 bias 时填零）
  ├─ [5×4096, 6×4096):  accF       累加器
  ├─ [6×4096, 7×4096):  tmpF       临时/激活输出
  └─ [7×4096, 8×4096):  currF      当前输入 token

inBuf (VECIN): 5 × 4096 × sizeof(T)   Ring Buffer
  5 个 slot 循环复用，slotCurr(t) = (t+3) % 5
  布局: [padding...][hist0][hist1][hist2][curr]
        前 MAX_WIDTH-W 个填零，后 W 个为有效窗口

outBuf (VECOUT): 2 × 4096 × sizeof(T)  Ping-Pong 输出缓冲
```

### 6.4 核心计算原语

#### LoadWeightAndBias
```
1. weight 前 (4-W) 行填 0
2. weight[0:W] GM → UB, bf16→f32 Cast
3. bias GM → UB (有则加载，无则填0)
```

#### InitRing
```
1. ring 前 (4-W) 个 slot 填 0
2. 若有初始状态: 从 convStates[cacheIdx, stateTokenOffset+i, :] 加载 W-1 个历史 token
3. 否则: 历史 slot 全部填 0
4. x[start] → ring[slotCurr(0)], set inputMte2ToVEvent_[slot0]
5. len>1: set inputVToMte2Event_（允许流水预取）
```

#### RunSeq（通用路径）
```
for t in [0, len):
  Wait inputMte2ToVEvent_[slotCurr(t)]       // 等 x[t] 就绪
  Prefetch x[t+1] → ring[slotNext]           // 异步搬下一步输入
  acc = 0
  for j in [0, W):                           // O(W) 累加
    Cast tmpF = ring[SlotHist(t, tap)]       // 取历史输入
    acc = MulAdd(acc, tmpF, weightF[j])      // 累加 weightF[j] × x[t-tap]
  Act(acc)                                   // SiLU (可选)
  Cast outBuf[ping] = result                 // Ping-Pong 写输出
  MTE3: outBuf[pong] → yGm                   // 异步写回
```

#### RunSeqFnRolling（快速路径）
```
启用条件: FN模式 && 编译期已知W && 无bias && 非投机解码

初始化 RestoreFnLocalPartials:
  用 ring 中 W-1 个历史 token + weight 预算 state0/1/2
  例 W=4:
    state0 = x[-3]×w0 + x[-2]×w1 + x[-1]×w2
    state1 = x[-2]×w0 + x[-1]×w1
    state2 = x[-1]×w0

循环 for t in [0, len):
  ComputeFnRollingOutput:
    state0 += x[t] × w0                     // 一步累加，O(1)
    output = Act(state0)
  AdvanceFnLocalPartials:
    state0 = state1 + x[t] × w[W-1]         // 递推，移位+新项
    state1 = state2 + x[t] × w[W-2]
    state2 =          x[t] × w0
```

优势：每步 O(1) 而非 O(W)，历史状态常驻寄存器而不访问 ring 不同 slot。

#### arch35 寄存器级优化

`__CCE_AICORE__ == 310` 时启用 `causal_conv1d_regbase.h`：

- `ComputeFnRollingOutputRegbase`：用 `__ubuf__` 直接寻址 + RegTensor + VF_CALL SIMD
- `AdvanceFnLocalPartialsRegbase`：按 W 特化 WidthTwo/Three/Four，每条 VF_CALL 处理 V_LENGTH=16 个 float，循环 `ceil(baseDim, 16)` 次

### 6.5 状态写回

#### WriteBackState（通用）
```
取 ring 中最后 W-1 个 slot 的值写回 convStates[cacheIdx, 0:W-1, :]
```

#### WriteBackStateSpec（投机解码 + W=4）
```
1. 将 convStates 中旧的 state[1],state[2] 前移到 state[0],state[1]
2. 将当前 x[t] 写入 state[2], state[3], ...（len 个新 token）
等价于: convStates[:] ← [old[1], old[2], x[t=0], x[t=1], ...]
降级条件: W≠4 或 stateLen 不足时回退到 WriteBackState
```

### 6.6 流水线同步

25 个 HardEvent 覆盖 5 类数据流：

| 数据流 | Events | 模式 |
|--------|--------|------|
| 权重加载 | `weightBiasMte2ToVEvent_` | MTE2→V |
| 状态初始化 | `stateMte2ToVEvent_` | MTE2→V |
| 输入加载 | `inputMte2ToVEvent_[5]`, `inputVToMte2Event_` | MTE2↔V, 双缓冲流水 |
| 输出写回 | `outMte3ToVEvent_[2]`, `outVToMte3Event_[2]` | V↔MTE3, Ping-Pong |
| 状态写回 | `stateWriteback*`, `stateShift*`, `specWriteback*[2]` | MTE2↔MTE3, Drain 同步 |
| 初始快照 | `initSnapshot*`, `initSync*` | MTE2↔MTE3 快照+SyncAll |

流水示意（输入-计算-输出三级重叠）：

```
MTE2:  [搬 x[0]] [搬 x[1]] [搬 x[2]] [搬 x[3]]
V:              [算 t=0]  [算 t=1]  [算 t=2]
MTE3:                     [写 y[0]] [写 y[1]] [写 y[2]]
```

### 6.7 Fn 模式调度：ProcessVarlenTokenTiled

```
ResolveFnDirectBlockTask(blockIdx) → 本 core 负责的 tokenTile+channel
  │
  ├─ hasInitStateWorkspace ∧ tokenTileId==0
  │    → PrefetchInitStatesToWorkspace (dump convStates→workspace)
  │    → SyncAll() 全局同步
  │
  ├─ 确定起始 seq:
  │   变长模式: ResolveExplicitTokenTileSeqRange (tiling 预置查表)
  │             或 FindVarlenSeqByToken (二分搜索)
  │   定长模式: tokenStart / seqLen
  │
  └─ while cursor < tokenEnd:
       ├─ ResolveSeqTaskWindow(seq) → [start, len)
       ├─ 跳过 padSlotId、跳过无效窗口
       ├─ ProcessFnChunk:
       │    LoadWeightAndBias → InitRingSeqSplit → RunSeq
       │    → MaybeWriteBackSeqSplitTailChunk
       └─ cursor += tileLen; seq++
```

`InitRingSeqSplit` 处理 token tile 跨序列边界：历史 token 可能来自同序列前部 (xGm)、前一序列的 state (convStatesGm/workspace)、或填零。

### 6.8 Update 模式调度：ProcessDefault

```
for task in [blockIdx, blockIdx+blockNum, ...]:  // 遍历 batch × baseDimCnt 网格
  seq       = task / baseDimCnt
  channelStart = (task % baseDimCnt) × baseDim
  ResolveSeqTaskWindow(seq) → [start, len)
  跳过 padSlotId
  投机解码: stateTokenOffset = numAcceptedTokens[seq] - 1
  LoadWeightAndBias → InitRing → RunSeq
  WriteBackState / WriteBackStateSpec → convStates
  DrainTaskMte3
```

### 6.9 输入模式（inputMode）枚举

| inputMode | 含义 | window 解析方式 |
|-----------|------|----------------|
| 0 | 2D 变长序列 `[cuSeqlen, D]` | `queryStartLoc[seq]` ~ `queryStartLoc[seq+1]` |
| 1 | 3D 定长 batch `[B, S, D]` | `seq × S` 起始，长度 S |
| 2 | 2D decode 模式 `[B, D]` | `seq` 起始，长度 1 |

---

## 7. 数据流总览

```
Host 端:
  aclnnCausalConv1dGetWorkspaceSize
    → l0op::CausalConv1d
      → ADD_TO_LAUNCHER_LIST_AICORE
        → Tiling (op_host/causal_conv1d_tiling.cpp)
          → tilingData → tilingKey

Device 端:
  causal_conv1d_apt (__global__)
    → GET_TILING_DATA
    → RunCausalConv1d<T, runModeKey, widthKey, fnPlanKey>
      ├─ FN   → CausalConv1dFn::Process → ProcessVarlenTokenTiled
      └─ UPDATE → CausalConv1dUpdate::Process → ProcessDefault
           └─ LoadWeightAndBias → InitRing → RunSeq/RunSeqFnRolling
              → WriteBackState → DrainTaskMte3
```
