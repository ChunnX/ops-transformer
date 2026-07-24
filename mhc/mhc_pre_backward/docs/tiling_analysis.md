# mhc_pre_backward 算子分核策略分析

## 一、核间策略（Inter-Core）

### 1.1 整体架构：AIV + AIC 异构协同

```cpp
// tiling L307
context->SetBlockDim(aicNum);   // BlockDim = AIC核数 (A2: 24)
```

- **BlockDim = aicNum = 24**，每个 block 包含 1个AIC + 2个AIV（A2典型配比），总核数 = 24 AIC + 48 AIV
- AIV 和 AIC 通过 `CrossCoreSetFlag/CrossCoreWaitFlag<0x2>` 进行流水线同步

### 1.2 AIV 侧数据切分

```cpp
// kernel L1137-1138
for (taskOffset = blkIdx_ * tileCoreBS_; taskOffset < totalTasksAligned_;
     taskOffset += aivNum_ * tileCoreBS_)
```

| 参数 | 值 | 说明 |
|------|------|------|
| `totalTasks_` | B × S | 总任务数 = Batch × SeqLen |
| `tileCoreBS_` | 32 (硬编码) | 每核每次处理 32 个 BS 任务 |
| `aivNum_` | 48 | AIV 核数 |
| `totalTasksAligned_` | CeilDiv(totalTasks, 48×32) × 48×32 | 对齐到 1536 的倍数 |

- 每轮：48个AIV核各处理32个BS，共处理 48×32 = **1536** 个BS
- 每核单轮工作量：`tileTaskCount = min(32, totalTasks - taskOffset)`
- 末尾tile自动处理尾块（`tileTaskCount < 32`）

### 1.3 AIC 侧数据切分

```cpp
// kernel L1238-1239
for (taskOffset = blkIdx_ * 2 * tileCoreBS_; taskOffset < totalTasksAligned_;
     taskOffset += aicNum_ * 2 * tileCoreBS_)
```

- 每轮：24个AIC核各处理 `2×32 = 64` 个BS任务
- **AIC处理量是AIV的2倍**（`2 * tileCoreBS_`），因为AIC需要做两个matmul（mm1和mm2），计算量更大

### 1.4 AIV↔AIC 同同步流程

```
AIV: ComputeGradPre → ComputeGradHHat2 → CrossCoreSetFlag<0x2, PIPE_MTE3>(8)
                                              ↓
AIC: CrossCoreWaitFlag<0x2>(8) → ProcessMatmul1 → ProcessMatmul2 → CrossCoreSetFlag<0x2, PIPE_FIX>(9)
                                              ↓
AIV: CrossCoreWaitFlag<0x2>(9) → ComputeGradX1
```

| Flag | 方向 | 含义 |
|------|------|------|
| Flag 8 | AIV → AIC | workspace 数据就绪（gradH2Global_, xWorkspaceGlobal_） |
| Flag 9 | AIC → AIV | matmul 结果就绪（gradXCubeGlobal_） |

### 1.5 确定性模式多核归约

```cpp
// kernel L1261-1276
if DETERMINISTIC:
    SyncAll<false>()
    useCoreNum = min(CeilDiv(totalTasks_, tileCoreBS_), aivNum_)
    ComputeDeterministic(gradHcBaseWSGlobal_, gradHcBaseGlobal_, useCoreNum, fusionSize_)
    ComputeDeterministic(gradHcScaleWSGlobal_, gradHcScaleGlobal_, useCoreNum, 3)
    ComputeDeterministic(gradWeightWSGlobal_, gradWeightGlobal_, useCoreNum, fusionSize_ * (n_ * c_))
```

- **非确定性**：mm2直接写 `gradWeightGlobal_`（多核原子累加，结果不确定）
- **确定性**：mm2写per-core workspace `gradWeightWSGlobal_`，再由AIV多核归约

### 1.6 Matmul 核间切分

```cpp
// tiling L371, L385
mm1Tiling.SetDim(aivNum);   // mm1: 48核切分
mm2Tiling.SetDim(aivNum);   // mm2: 48核切分
```

- 两个matmul都使用 `MultiCoreMatmulTiling`，按 `aivNum=48` 核切分
- `SetSingleShape(M, N, K)` = `SetShape(M, N, K)`，不分块，整个shape作为single shape

| Matmul | M | N | K | 公式 | 说明 |
|--------|------|------|------|------|------|
| mm1 | tile×2=64 | N×C=2048 | fusionSize | x_rs_grad = H_mix_grad @ phi | K轴=fusionSize |
| mm2 | fusionSize | N×C=2048 | B×S | phi_grad = G^T @ X | K轴=B×S, A转置 |

### 1.7 Workspace 布局

```cpp
// tiling L428-430
currentWorkspace[0] = systemWorkspaceSize                      // 系统workspace
                   + usrWorkSpaceSize                            // 用户workspace
                   + aicNum * fusionSize * (n * c) * sizeof(float)     // 确定性: per-core mm2 workspace
                   + aivNum * (fusionSize + ELEMENTS_SIZE_PER_BLOCK) * sizeof(float);  // 确定性: bias/scale归约workspace
```

其中 `usrWorkSpaceSize`：

```
xCastWorkspace     = B × S × N × D × sizeof(float) × 2   // x和x×grad_y的fp32副本
gradHat2Workspace  = B × S × fusionSize × sizeof(float)    // H_mix_grad
```

---

## 二、核内策略（Intra-Core）

### 2.1 UB Buffer 规划

| Buffer | 大小 | 双缓冲 | 用途 |
|--------|------|--------|------|
| inputXInQueue | 2×tile×N×c0×1B | 2 | 加载x |
| OutQueue | 2×tile×N×c0×0.5B | 2 | 输出grad_x |
| inputGradQueue | 1×tile×N×8×4B | 1 | 加载grad/pre |
| fusedGradHPre2AndGradHPost2Buf_ | tile×N×2×4B | - | H_pre/post 梯度 |
| gradRsqrtBuf_ | tile×fusionSize×4B×2 | - | inv_rms 梯度 |
| gradBiasBuf_ | 4×tile×fusionSize×4B | - | bias/scale 梯度（复用） |
| onesBuf_ | tile×N×2×4B | - | 全1向量 |
| hcBaseBuf_ | fusionSize×4B | - | bias常量 |
| ScaleBuf_ | 64B | - | alpha scale |
| tempBuf_ | ~41KB | - | dhat/rsqrt/hat2Scale等临时变量 |

### 2.2 核内循环：INNER_SPILT_NUM=8

```cpp
// kernel L73
constexpr int32_t INNER_SPILT_NUM = 8;
// kernel L329
onceTask_ = tileCoreBS_ / INNER_SPILT_NUM;  // 32 / 8 = 4
```

**AIV 内层循环**：

```cpp
// kernel L1150-1156
for (taskOffsetInner = 0; taskOffsetInner < tileTaskCount; taskOffsetInner += onceTask_) {
    ComputeGradPre(taskOffset + taskOffsetInner, tileTaskCountInner, taskOffsetInner);
}
```

- 32个BS任务分成**8轮**（每轮4个），逐轮处理 `ComputeGradPre`
- **原因**：UB放不下 `tileCoreBS_ × N × c0` 的全量数据，需要分批

### 2.3 三阶段流水线

每个外层tile（32个BS）执行三阶段：

#### 阶段1 - ComputeGradPre（核内分8轮）

- 加载 `x[4, N, c0]`、`grad_hin[4, D]`、`h_pre[4, N]`
- 计算 `grad_pre = sum(grad_y × x, dim=D)` → kahan累加到 dPrePostTempLocal_
- 保存 `x × grad_y` 到 `xWorkspaceGlobal_`（供AIC mm2使用）

#### 阶段2 - ComputeGradHHat2（核内一次性处理32个BS）

- grad_h_res → dhatLocal_（res blocks 填充）
- sigmoid门控反向 + 残差反向 + scale乘法
- WholeReduceSum 计算 inv_rms_grad
- 输出 `gradH2Global_`（H_mix_grad，供AIC mm1使用）

#### 阶段3 - ComputeGradX1（核内分8轮）

- 等待AIC mm1结果 `gradXCubeGlobal_`
- 广播 h_pre 和 inv_rms_grad 到 D 维度
- 组装 `grad_x = vec3(H_pre×H_in_grad) + vec1(inv_rms_grad) + mm(gradH2@phi)`

### 2.4 Ping-Pong 双缓冲

```cpp
// kernel L304-306
pipe_->InitBuffer(inputXInQueue, 2, tileCoreBS_ * n_ * c0_ * sizeof(float) / 4);  // 双缓冲
pipe_->InitBuffer(inputGradQueue, 1, tileCoreBS_ * n_ * ELEMENTS_SIZE_PER_BLOCK * sizeof(float));  // 单缓冲
pipe_->InitBuffer(OutQueue, 2, tileCoreBS_ * n_ * c0_ * sizeof(float) / 8);  // 双缓冲
```

| 流水 | 机制 | 说明 |
|------|------|------|
| MTE2→V | `inputXInQueue` EnQue/DeQue | GM→UB 搬运与计算流水 |
| V→MTE3 | `OutQueue` EnQue/DeQue | 计算与UB→GM 回写流水 |

### 2.5 事件同步

```cpp
// kernel L315-317
eventIdVToMTE3XCast = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();   // V→MTE3
eventIdMTE2ToV = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();        // MTE2→V
eventIdVToMTE2 = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();        // V→MTE2
```

| 事件 | 方向 | 用途 |
|------|------|------|
| V→MTE3 | V计算→MTE3回写 | V计算完成后通知MTE3写回GM |
| MTE2→V | MTE2搬运→V计算 | 数据搬运完成后通知V计算 |
| V→MTE2 | V使用→MTE2加载 | V使用完成后通知MTE2加载下一批 |

### 2.6 tempBuf_ 复用策略

tempBuf_ 内部通过 offset 分配多个子缓冲区，分阶段复用：

**ComputeGradPre 阶段**：

| 子缓冲 | 大小 | 用途 |
|--------|------|------|
| preBrcbLocal_ | CeilDiv(tile×N, 8)×64×4B | h_pre广播到D维度 |
| dRsqrtBrcbLocal_ | CeilDiv(tile, 8)×64×4B | inv_rms广播 |
| gradYCastLocal_ | onceTask×c0×4B | grad_y类型转换 |
| gradXCastLocal_ | onceTask×N×c0×4B | grad_x中间结果 |
| xCastLocal_ | onceTask×N×c0×4B | x类型转换 |

**ComputeGradHHat2 阶段**（offset重置为0，复用同一UB区域）：

| 子缓冲 | 大小 | 用途 |
|--------|------|------|
| dhatLocal_ | tile×fusionSize×2×4B | dhat双倍空间 |
| rsqrtbrcbLocal_ | CeilDiv(tile,8)×8×fusionSize×4B | rsqrt广播 |
| hat2Scale | tile×fusionSize×4B | hat2 scale中间值 |
| dhatBeforeNormLocal | tile×fusionSize×4B | norm前dhat |
| hatLocal | tile×fusionSize×4B | hat中间值 |
| rsqrtTempLocal_ | tile×fusionSize×4B | rsqrt临时值 |

---

## 三、总体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    核间策略 (Inter-Core)                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  BlockDim = 24 (AIC数), 每block = 1 AIC + 2 AIV             │
│  AIV: 48核, 每核tile=32 BS, 步进 aivNum×32=1536             │
│  AIC: 24核, 每核tile=64 BS, 步进 aicNum×64=1536             │
│                                                             │
│  ┌─────────┐  Flag 8   ┌─────────┐  Flag 9   ┌─────────┐   │
│  │  AIV    │ ────────→ │  AIC    │ ────────→ │  AIV    │   │
│  │ GradPre │           │ mm1     │           │ GradX1  │   │
│  │ GradHH2 │           │ mm2     │           │         │   │
│  └─────────┘           └─────────┘           └─────────┘   │
│                                                             │
│  确定性: SyncAll → AIV多核归约 mm2/bias/scale               │
│  非确定性: mm2原子累加, 无归约                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    核内策略 (Intra-Core)                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  内层切分: onceTask=4 (32/8), 分8轮处理                     │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  阶段1: ComputeGradPre (分8轮, 每轮4个BS)             │  │
│  │    x → grad_pre = sum(grad_y × x) → kahan累加         │  │
│  │    x × grad_y → xWorkspaceGlobal_                     │  │
│  └───────────────────────┬───────────────────────────────┘  │
│                          ↓ Flag 8                           │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  阶段2: ComputeGradHHat2 (全量32个BS)                 │  │
│  │    dhat ← grad_h_res + sigmoid_grad + scale           │  │
│  │    WholeReduceSum → inv_rms_grad                      │  │
│  │    → gradH2Global_                                    │  │
│  └───────────────────────┬───────────────────────────────┘  │
│  ┌───────────────────────↓ Flag 9 ───────────────────────┐  │
│  │  阶段3: ComputeGradX1 (分8轮, 每轮4个BS)              │  │
│  │    grad_x = vec3(H_pre×H_in) + vec1(inv_rms) + mm     │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  双缓冲: inputXInQueue(2), OutQueue(2)                      │
│  事件同步: MTE2↔V, V↔MTE3 三级流水                           │
│  tempBuf_复用: dhat/rsqrt/hat2Scale 交替使用同一UB区域       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 四、关键参数汇总

| 参数 | 值 | 来源 | 说明 |
|------|------|------|------|
| BlockDim | 24 | tiling L307 | = aicNum |
| aivNum | 48 | platform | A2平台 |
| aicNum | 24 | platform | A2平台 |
| tile (tileCoreBS_) | 32 | tiling L350 | 硬编码, 每核BS任务数 |
| INNER_SPILT_NUM | 8 | kernel L73 | 核内分8轮 |
| onceTask_ | 4 | kernel L329 | = tile / INNER_SPILT_NUM |
| c0 | 256 | tiling L346 | = Vector Length (VL) |
| fusionSize | 32 或 24 | phi shape | N!+2N 或 N²+2N |
| mm1 [M,N,K] | [64, 2048, fusionSize] | tiling L357-359 | x_rs_grad = H_mix_grad @ phi |
| mm2 [M,N,K] | [fusionSize, 2048, B×S] | tiling L361-363 | phi_grad = G^T @ X |
| UB Size | 192 KB | platform | A2每核UB |
