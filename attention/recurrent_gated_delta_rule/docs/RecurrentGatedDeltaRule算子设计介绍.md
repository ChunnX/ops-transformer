# RecurrentGatedDeltaRule 算子设计介绍

## 1 概述

RecurrentGatedDeltaRule（循环门控 Delta Rule，简称 RGDR）是一种应用于循环神经网络 / 线性注意力机制的算子。在每个时间步 $t$，根据当前输入 $q_t$、$k_t$、$v_t$ 和上一隐藏状态 $S_{t-1}$，计算当前输出 $o_t$ 和新隐藏状态 $S_t$。

$$
S_t := \alpha_t \cdot \text{Diag}(\alpha_{kt}) \cdot S_{t-1} + \beta_t \cdot (v_t - \alpha_t \cdot \text{Diag}(\alpha_{kt}) \cdot S_{t-1} \cdot k_t) \cdot k_t^T
$$

$$
o_t := \frac{S_t \cdot q_t}{\sqrt{d_k}}
$$

其中 $S \in \mathbb{R}^{d_v \times d_k}$，$q, k \in \mathbb{R}^{d_k}$，$v \in \mathbb{R}^{d_v}$，$\alpha = e^g$，$\alpha_k = e^{g_k}$，$\beta \in \mathbb{R}$，$o \in \mathbb{R}^{d_v}$。

### 支持产品

| 产品 | 架构代号 | Kernel 入口文件 |
|:-----|:---------|:----------------|
| Ascend 950PR / Ascend 950DT | arch35（apt） | `recurrent_gated_delta_rule_apt.cpp` |
| Atlas A3 训练 / 推理系列产品 | arch35（apt） | `recurrent_gated_delta_rule_apt.cpp` |
| Atlas A2 训练 / 推理系列产品 | arch22 | `recurrent_gated_delta_rule.cpp` |

## 2 目录结构

```
recurrent_gated_delta_rule/
├── op_api/                          # L0 接口实现（aclnn 两段式）
│   ├── recurrent_gated_delta_rule.cpp   # 算子图构造 + Infershape + Launcher
│   └── aclnn_recurrent_gated_delta_rule.cpp  # aclnn GetWorkspaceSize / aclnn 入口
├── op_host/                         # Host 侧 Tiling / Infershape / OpDef
│   ├── recurrent_gated_delta_rule_def.cpp        # 算子定义（输入输出/属性/AICore 配置）
│   ├── recurrent_gated_delta_rule_infershape.cpp # Shape 与 DataType 推导
│   ├── recurrent_gated_delta_rule_tiling.cpp     # Tiling 核心逻辑
│   └── recurrent_gated_delta_rule_tiling.h       # Tiling 类声明
├── op_kernel/                       # Device 侧 Kernel
│   ├── recurrent_gated_delta_rule.cpp             # arch22 入口
│   ├── recurrent_gated_delta_rule_apt.cpp         # arch35 入口
│   ├── recurrent_gated_delta_rule_tiling_data.h   # TilingData 结构体
│   ├── recurrent_gated_delta_rule_tiling_key.h    # TilingKey 定义
│   ├── arch22/recurrent_gated_delta_rule.h        # A2 Kernel 类 RGDR
│   └── arch35/
│       ├── recurrent_gated_delta_rule.h           # 950/A3 Kernel 类 RGDR
│       ├── vf_vec_mul_mat.h                       # VF 向量乘矩阵 intrinsic
│       └── vf_outer_add.h                         # VF 外积加 intrinsic
├── docs/                            # 文档
├── examples/                        # 调用示例
└── tests/                           # UT / pytest
```

## 3 计算流程

### 3.1 整体流程

算子为 **AIV-only**（纯 Vector）算子，不使用 Cube 矩阵乘单元。所有矩阵-向量乘法通过 Vector 侧的逐元素乘 + ReduceSum 实现。

```
Process()
│
├── ComputeAvgload()          # 计算每个核的平均负载，用于任务均衡
│
└── for batch_i in [0, B):    # 遍历每个 batch
      seq0, seq1 = 当前 batch 的 token 范围 [seq0, seq1)
      │
      └── for head_i in [0, NV):   # 遍历每个 value head
            │
            ├── IsCurrentBlock()   # 判断当前 head 是否由本核处理（负载均衡）
            │
            ├── (首次) CopyInGamaBeta()  # 拷入 g/beta，计算 Exp
            │
            └── ProcessHead(seq0, seq1, head_i, stateOffset)
```

### 3.2 ProcessHead 单头处理流程

```
ProcessHead(seq0, seq1, head_i, stateOffset)
│
├── CopyInQKV()                    # 从 GM 拷入 q/k/v，Cast 到 fp32，q 乘 scale
│   └── (可选) CopyIn gk，Exp(gk)
│
└── for v_i in [0, Dv, vStep):     # 沿 Dv 方向分块（vStep 为分块大小）
      curSingleV = min(vStep, Dv - v_i)
      │
      ├── CopyInState()            # 从 GM 拷入状态矩阵 S 的一个 vStep×Dk 分块
      │
      └── for seq_i in [seq0, seq1):  # 遍历序列中每个 token（时间步）
            │
            ├── 获取当前 token 的 gama/beta 标量
            │
            ├── Compute()          # 核心 Delta Rule 计算（见 3.3）
            │
            ├── CopyOutAttn()      # 输出 o_t 到 GM
            │
            └── CopyOutState()     # 输出更新后的 S_t 到 GM（原地更新 state）
```

> **注意**：状态矩阵 $S$ 在序列内是**递推更新**的——每个 token 计算后更新的 $S_t$ 立即写回 GM，下一个 token 读取最新的 $S_t$ 继续计算。`ssmStateIndices` 提供 token 到 state 槽位的映射，支持变步长（MTP）场景。

### 3.3 Compute 核心计算步骤

以 arch22 为例，arch35 逻辑等价但使用 VF intrinsic：

| 步骤 | 操作 | 公式 |
|:-----|:-----|:-----|
| 1 | (可选) `Muls(state, gama)` | $S \leftarrow \alpha \cdot S$ |
| 2 | (可选) `MatVecMul(state, gk)` | $S \leftarrow S \cdot \text{Diag}(\alpha_k)$（逐行乘 gk 向量） |
| 3 | `MatVecMul(state, k) → tmp` | $\text{tmp} \leftarrow S \cdot k$（逐元素乘） |
| 4 | `ReduceSum(tmp → delta)` | $\delta \leftarrow \sum_j S_{ij} k_j$（行求和） |
| 5 | `delta = v - delta` | $\delta \leftarrow v - S \cdot k$ |
| 6 | `Muls(delta, beta)` | $\delta \leftarrow \beta \cdot \delta$ |
| 7 | `Broadcast(tmp, delta)` | 将 $\delta$ 广播为 $d_v \times d_k$ 矩阵 |
| 8 | `MatVecMul(tmp, k, state, isAdd)` | $S \leftarrow S + \delta \cdot k^T$（外积加） |
| 9 | `MatVecMul(state, q) → tmp` | $\text{tmp} \leftarrow S \cdot q$ |
| 10 | `ReduceSum(tmp → attn)` | $o \leftarrow \sum_j S_{ij} q_j$ |
| 11 | `Cast(stateOut, state)` / `Cast(attnOut, attn)` | 输出转回 bf16 |

其中 `MatVecMul` 在 arch22 中通过 `Mul`/`MulAddDst` 指令逐 repeat 实现（`REPEAT_LENTH=64`），在 arch35 中替换为 `VecMulMatVF` MicroAPI intrinsic，利用寄存器级 LoadAlign/Mul/StoreAlign 提升效率。

## 4 Tiling 策略

### 4.1 核间切分

- **任务粒度**：以 `(batch, head)` 为最小任务单元，总任务数 = $B \times N_v$。
- **核数分配**：`vectorCoreNum = min(aivNum, B * NV)`，当任务数少于核数时降级使用核数。
- **负载均衡**：`ComputeAvgload()` 计算 `avgload = Ceil(realT * NV, blockDim)`，`IsCurrentBlock()` 采用累积负载轮转策略，将 `(batch, head)` 任务均匀分配到各核，避免长序列集中导致核间不均。

### 4.2 核内切分 — vStep

沿 $D_v$ 方向分块，分块大小 `vStep` 是 Tiling 的核心参数：

- **arch22**：由 Host 侧 `CalUbSize()` 根据剩余 UB 空间动态计算（详见第 5 节）。
- **arch35**：在 Kernel 构造函数中固定计算：`vStep = (8192 / alignDK)`，限制不超过 255，对齐到 16。其中 8192 = 32KB(stateBuff) / 4(sizeof fp32)。

### 4.3 TilingKey

| TilingKey | 含值 | 说明 |
|:----------|:-----|:-----|
| `TILING_KEY_RGDR_BF16_STATE` | 0 | state 数据类型为 BF16 |
| `TILING_KEY_RGDR_FP32_STATE` | 1 | state 数据类型为 FP32 |

Kernel 入口根据 TilingKey 选择模板实例化 `RGDR<bf16, bf16, bf16>` 或 `RGDR<bf16, bf16, float>`。

### 4.4 TilingData 字段

| 字段 | 类型 | 说明 |
|:-----|:-----|:-----|
| `vectorCoreNum` | uint32 | 实际使用的 Vector 核数 |
| `ubCalSize` | uint32 | UB 总大小（来自编译信息） |
| `ubRestBytes` | uint32 | 分配给 tmpBuff 的剩余 UB 字节数 |
| `t` | uint32 | 累积序列长度 T |
| `nk` / `dk` | uint32 | key 的头数 / 维度 |
| `nv` / `dv` | uint32 | value 的头数 / 维度 |
| `sBlockNum` | uint32 | state 的第 0 维大小（BlockNum） |
| `b` | uint32 | batch size |
| `vStep` | uint32 | Dv 方向分块大小 |
| `scale` | float | query 缩放因子 1/sqrt(dk) |
| `hasGama` / `hasGamaK` / `hasAcceptedTokens` | uint32 | 可选输入存在标志 |
| `stateStride0` / `stateStride1` | uint64 | state 非连续 stride（轴 0/1） |

## 5 UB 分配策略

### 5.1 数据对齐规则

所有 UB 分配遵循 32 字节对齐：
- BF16：16 个元素为一个 32B block（`BF16_NUM_PER_BLOCK = 16`）
- FP32：8 个元素为一个 32B block（`FP32_NUM_PER_BLOCK = 8`）
- 对齐后维度：`alignK = Ceil(dk, 16) * 16`，`alignV = Ceil(dv, 16) * 16`，`alignNv = Ceil(nv, 16) * 16`

### 5.2 arch22（Atlas A2）UB 分配

arch22 采用**动态 UB 分配**策略：Host 侧根据输入 shape 计算各 buffer 大小，推导出最优 `vStep`，再将剩余空间传递给 Kernel。

#### 5.2.1 分配顺序与空间布局

UB 空间分为两大类：**TQue（队列 buffer）** 和 **TBuf（工作 buffer）**。

```
┌──────────────────────────── UB ─────────────────────────────┐
│                                                              │
│  ◆ TQue（VECIN/VECOUT，通过 InitBuffer 分配）                │
│  ┌──────────────┬──────────────┬──────────────┐             │
│  │ qInQueue_    │ kInQueue_    │ vInQueue_    │  bf16       │
│  │ 8*alignK*2B  │ 8*alignK*2B  │ 8*alignV*2B  │             │
│  ├──────────────┼──────────────┼──────────────┤             │
│  │ stateInQueue_│ stateOutQueue_│ attnOutQueue_│            │
│  │ alignK*vStep │ alignK*vStep  │ vStep*2B     │ stateType  │
│  │  *stateSize  │  *stateSize   │              │            │
│  ├──────────────┼──────────────┤                            │
│  │ betaInQueue_ │(gamaInQueue_)│  bf16/fp32                 │
│  │ 8*alignNv*2B │8*alignNv*4B  │                            │
│  ├──────────────┤                                           │
│  │(gamaKInQueue_)│  fp32                                    │
│  │ 8*alignK*4B  │                                           │
│  └──────────────┴──────────────┴──────────────┘             │
│                                                              │
│  ◆ TBuf tmpBuff（剩余空间，通过 GetWithOffset 切分）          │
│  ┌────────────┬────────────┬──────────────────┐             │
│  │ deltaInUb  │ attnInUb   │ vInUb            │  fp32       │
│  │ vStep*4B   │ vStep*4B   │ 8*alignV*4B      │             │
│  ├────────────┼────────────┼──────────────────┤             │
│  │ qInUb      │ kInUb      │ stateInUb        │  fp32       │
│  │8*alignK*4B │8*alignK*4B │ alignK*vStep*4B  │             │
│  ├────────────┼────────────┼──────────────────┤             │
│  │broadTmpInUb│ betaInUb   │                  │  fp32       │
│  │alignK*vStep│ betaUbSize │                  │             │
│  │  *4B       │            │                  │             │
│  └────────────┴────────────┴──────────────────┘             │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

> 其中 `8 = MAX_MTP`（单次最多处理 8 个 token），`stateSize = sizeof(stateType)`。

#### 5.2.2 Host 侧 vStep 计算（`CalUbSize()`）

```
第 1 步：计算固定 TQue 的 UB 占用（不含 state/attnOut 队列）
  usedUbBytes  = 8 * (4*alignK + 2*alignV)        // q + k 队列(bf16→4B/elem) + v 队列(bf16→2B/elem)
  usedUbBytes += 128                               // 预留
  usedUbBytes += hasGamaK ? 8 * 4 * alignK : 0    // gk 队列(fp32)
  usedUbBytes += hasGama  ? 8 * 4 * alignNv : 0   // g  队列(fp32)
  usedUbBytes += 8 * 2 * alignNv                   // beta 队列(bf16)
  ubRestBytes  = ubSize - usedUbBytes              // → tmpBuff 初始可用空间

第 2 步：加上 tmpBuff 中固定部分（不随 vStep 变化）
  usedUbBytes += 8 * (8*alignK + 4*alignV + 4*alignNv)
                 // qInUb(4) + kInUb(4) + vInUb(4) + betaInUb(4)，均为 fp32

第 3 步：计算每增加 1 行 vStep 的边际 UB 成本
  stateInElemBytes  = (stateDtype == FP32) ? 4 : 2
  stateOutElemBytes = stateInElemBytes
  coeff  = (stateInElemBytes + stateOutElemBytes) * alignK + 4
           // stateInQueue_ 每行 + stateOutQueue_ 每行 + attnOutQueue_ 每行
  coeff += (4 + 4) * alignK + 4 + 4
           // stateInUb 每行(fp32) + broadTmpInUb 每行(fp32) + deltaInUb 每行 + attnInUb 每行

第 4 步：计算 vStep
  vStep = (ubSize - usedUbBytes) / coeff / 8 * 8    // 对齐到 8（32B / sizeof(fp32)）
  约束：vStep >= 8，否则报错 "shape is too big"

第 5 步：微调 vStep（均匀切分 Dv，减少尾块）
  rptime = CeilDiv(dv, vStep)
  vStep  = CeilAlign(CeilDiv(dv, rptime), 8)

第 6 步：更新 ubRestBytes（扣除 state/attnOut 队列实际占用）
  ubRestBytes -= ((stateInElemBytes + stateOutElemBytes) * alignK + 4) * vStep
```

**核心思想**：先分配不随 vStep 变化的固定 buffer，再将剩余空间按 `coeff`（每行 vStep 的边际成本）均分给 state 相关 buffer，使得 vStep 尽可能大，减少 Dv 方向的循环次数。最后通过 `rptime` 微调使尾块尽量均匀。

#### 5.2.3 Kernel 侧 tmpBuff 切分

Kernel `InitLocalBuffers()` 使用 `GetWithOffset<float>()` 从 `tmpBuff` 中顺序切分各工作 buffer，每个 buffer 通过偏移量紧密排列，无内部碎片：

| 工作 Buffer | 元素数 | 用途 |
|:------------|:-------|:-----|
| `deltaInUb` | vStep | 存储 $\delta = v - S \cdot k$ |
| `attnInUb` | vStep | 存储 $o = S \cdot q$ 的 fp32 中间结果 |
| `vInUb` | 8 × alignV | v 的 fp32 副本（所有 token） |
| `qInUb` | 8 × alignK | q 的 fp32 副本（已乘 scale） |
| `kInUb` | 8 × alignK | k 的 fp32 副本 |
| `stateInUb` | alignK × vStep | state 的 fp32 工作副本 |
| `broadTmpInUb` | alignK × vStep | 广播 / 矩阵乘中间结果 |
| `betaInUb` | 8 × alignNv（对齐） | beta 的 fp32 副本 |

### 5.3 arch35（Ascend 950 / A3）UB 分配

arch35 采用**固定大小 UB 分配**策略，各 buffer 使用预定义常量大小，简化 Tiling 逻辑并利用更大的 UB 容量：

#### 5.3.1 Buffer 分配

| Buffer | 类型 | 大小 | 说明 |
|:-------|:-----|:-----|:-----|
| `inputQueue1_` | TQue (VECIN, **Double Buffer**) | 32K (fp32 state) / 16K (bf16 state) | 复用队列：依次拷入 gk / q / k / v / state |
| `stateOutQueue_` | TQue (VECOUT) | 32K / 16K | state 输出队列 |
| `attnOutQueue_` | TQue (VECOUT) | 16K | attention 输出队列 |
| `gamaBuff_` | TBuf | 16K | gk 的 fp32 副本 |
| `stateBuff_` | TBuf | 32K | state 的 fp32 工作副本 |
| `queryBuff_` | TBuf | 16K | q 的 fp32 副本 |
| `keyBuff_` | TBuf | 16K | k 的 fp32 副本 |
| `valueBuff_` | TBuf | 16K | v 的 fp32 副本 |
| `tmpBuff2_` | TBuf | 32K | 矩阵乘中间结果 |
| `tmpBuff1_` | TBuf | 2K | ReduceSum 结果 |

> **Double Buffer**：`inputQueue1_` 使用 `DOUBLE_BUFFER=2`，实现 MTE2 搬运与 V 计算的流水并行。

#### 5.3.2 vStep 计算

arch35 的 vStep 在 Kernel 构造函数中计算，不依赖 Host Tiling：

```
vStep = 8192 / alignDK          // 8192 = 32KB(stateBuff) / 4B(sizeof fp32)
if (vStep > 255) vStep = 255    // 硬件 repeat 次数上限
vStep = vStep / 16 * 16         // 对齐到 16（64B，2 个 32B block）
```

#### 5.3.3 arch35 性能优化要点

1. **VF intrinsic**：使用 MicroAPI 自定义 `VecMulMatVF`（向量乘矩阵）和 `OuterAddVF`（外积加），通过寄存器级 `LoadAlign`/`Mul`/`MulAddDst`/`StoreAlign` 替代 arch22 的高阶 API，减少指令开销。
2. **Broadcast 优化**：`OuterAddVF` 使用 `DIST_BRC_B32` 加载模式，将 delta 首元素广播到整个寄存器，省去显式 Broadcast 操作。
3. **流水交错**：`Compute()` 中在 `dvIdx==0 && bSeqIdx==0` 时触发 `CopyInQK()` / `CopyInV()`，将首个 token 的数据搬运与后续 token 的计算重叠。
4. **队列复用**：`inputQueue1_` 复用于 gk/q/k/v/state 的 GM→UB 搬运，减少 UB 占用。

### 5.4 两代架构 UB 策略对比

| 维度 | arch22（A2） | arch35（950/A3） |
|:-----|:------------|:-----------------|
| 分配方式 | Host 动态计算，shape 自适应 | Kernel 固定常量大小 |
| vStep 来源 | Host `CalUbSize()` 推导 | Kernel 构造函数计算 |
| 工作 Buffer | 单一 tmpBuff + GetWithOffset 切分 | 独立 TBuf 分配 |
| Double Buffer | 否（BUFFER_NUM=1） | 是（inputQueue1_ Double Buffer） |
| 矩阵乘实现 | Mul/MulAddDst 高阶 API | VecMulMatVF/OuterAddVF MicroAPI |
| 队列复用 | 每 input 独立 TQue | inputQueue1_ 复用 |
| UB 利用率 | 高（按 shape 精确分配） | 中（固定大小，大 shape 时可能浪费） |

## 6 Infershape 设计

- **out** shape = value shape `(T, NV, DV)`，dtype = BF16
- **stateOut** shape = state shape `(BlockNum, NV, DV, DK)`，dtype = state dtype（BF16 或 FP32）
- 输出 dtype 不依赖输入数值，仅依赖 state 输入 dtype

## 7 约束与注意事项

1. **序列长度**：$0 < L_i \le 8$（单 batch 最多 8 个 token，即 MTP ≤ 8）
2. **头数约束**：$0 < N_k \le 256$，$N_k \le N_v \le 256$，$N_v \bmod N_k = 0$
3. **维度约束**：$0 < D_k \le 512$，$0 < D_v \le 512$
4. **State 约束**：`BlockNum >= T`；仅支持轴 0、轴 1 非连续（最内两维必须连续）
5. **数值约束**（用户保证，算子不校验）：$|q|, |k| \le 1$，$g < 0$，$g_k < 0$，$0 < \beta < 1$
6. **确定性**：默认确定性计算

## 8 下一代芯片适配分析（UB=384KB, AIV=1）

假设下一代芯片将 UB 总空间调整为 384KB（当前 A2/950 典型为 192KB），且 AIV 数量为 1。以下分析 UB 分配是否需要调整，以及对计算流程的影响。

### 8.1 对 arch22（动态分配）UB 的影响

arch22 的 `CalUbSize()` 从平台信息动态读取 `ubSize`，按 `vStep = (ubSize - usedUbBytes) / coeff / 8 * 8` 自适应推导。**UB 变大后无需修改代码即可自动适配**，vStep 会按比例增大。

以最坏场景（$D_k=512, D_v=512, N_v=256$，FP32 state，含 g/gk）为例，量化对比：

| UB 大小 | 固定占用 | coeff（每行边际） | vStep（原始） | vStep（微调后） | $D_v$ 迭代次数 |
|:--------|:---------|:-----------------|:--------------|:----------------|:--------------|
| 192KB | ~108KB | 8204 B | 8 | 8 | 64 |
| 384KB | ~108KB | 8204 B | 32 | 32 | 16 |

> vStep 从 8 提升到 32，$D_v$ 方向迭代次数从 64 降至 16，减少 75% 的 state GM 读写往返。

以典型场景（$D_k=128, D_v=128, N_v=8$，FP32 state，含 g/gk）为例：

| UB 大小 | vStep（原始） | vStep（微调后） | $D_v$ 迭代次数 |
|:--------|:--------------|:----------------|:--------------|
| 192KB | 80 | 64 | 2 |
| 384KB | 176 | 128 | 1 |

> 典型场景下 vStep 达到 $D_v=128$，单头仅需 1 次 $D_v$ 迭代即可完成，state 只需一次性 Load/Store。

**结论**：arch22 无需修改代码。vStep 增大后，state 相关 TQue（`stateInQueue_`/`stateOutQueue_`）和 TBuf（`stateInUb`/`broadTmpInUb`）的容量按 `vStep` 同比例增长，`CalUbSize()` 公式保证总占用不超过 UB 上限。

**注意点**：
1. 当 vStep 增大至 $\ge D_v$ 时，`rptime = CeilDiv(Dv, vStep) = 1`，vStep 被 `CeilAlign(Dv, 8)` 截断为 $D_v$（对齐后），**超出部分 UB 空间闲置**。此时可考虑将闲置空间用于 Double Buffer 或增大 `MAX_MTP`（但 `MAX_MTP` 受 $L_i \le 8$ 约束，不宜调整）。
2. arch22 的 `MatVecMul` 内部以 `MAX_REPEAT_TIME=255` 分段循环，无 vStep 上限硬约束。`ReduceSum` 高阶 API 内部自行 tiling，也不受 vStep 大小限制。
3. TilingData 中 `ubCalSize`/`ubRestBytes` 为 `uint32_t`，384KB = 393216 远未溢出。

### 8.2 对 arch35（固定分配）UB 的影响

arch35 的 buffer 大小为**硬编码常量**（`BUFFER_SIZE_BYTE_16K`/`32K` 等），vStep 在 Kernel 构造函数中按 `vStep = 8192 / alignDK` 计算（8192 = 32KB / 4B，绑定 `stateBuff_` 的 32KB 固定大小）。**UB 变大后不会自动适配，额外空间被完全浪费**。

当前 arch35 固定 UB 占用（FP32 state）：

| Buffer | 大小 | 说明 |
|:-------|:-----|:-----|
| `inputQueue1_`（Double Buffer） | 64KB | 32K × 2 |
| `stateOutQueue_` | 32KB | |
| `stateBuff_` | 32KB | vStep 基准 |
| `tmpBuff2_` | 32KB | VecMulMat 输出 |
| `attnOutQueue_` / `gamaBuff_` / `queryBuff_` / `keyBuff_` / `valueBuff_` | 16KB × 5 = 80KB | |
| `tmpBuff1_` | 2KB | ReduceSum 结果 |
| **合计** | **242KB** | |

| UB 大小 | 占用 | 剩余 | 剩余利用率 |
|:--------|:-----|:-----|:-----------|
| 192KB | 242KB | **-50KB（超限）** | FP32 state + 大 shape 在 192KB 下已无法运行 |
| 384KB | 242KB | 142KB | **36.9% 浪费** |

> **关键发现**：arch35 当前固定分配 242KB 已超过 192KB，说明 950 实际 UB 大于 192KB（推测为 256KB+）。在 384KB 下有 142KB 闲置。

**需调整的 buffer 及 vStep 提升效果**：

`stateBuff_` 和 `tmpBuff2_` 是与 vStep 直接相关的两个 buffer（分别存储 state 工作副本和 VecMulMat 输出，大小均为 `alignDK × vStep × 4B`）。增大这两个 buffer 即可提升 vStep：

| `stateBuff_`/`tmpBuff2_` 大小 | vStep（alignDK=512） | vStep（alignDK=128） | vStep（alignDK=64） | 新总占用 | 384KB 内余量 |
|:------------------------------|:---------------------|:---------------------|:---------------------|:---------|:-------------|
| 32KB（当前） | 16 | 64 | 128 | 242KB | 142KB |
| 64KB | 32 | 128 | 240 | 306KB | 78KB |
| 96KB | 48 | 192 | 240 | 370KB | 14KB |
| 128KB | 64 | 256→255 | 255 | 434KB | **超限** |

> - **推荐调整为 96KB**：在 384KB 内可容纳（余 14KB），vStep 提升显著（alignDK=512 时从 16→48，提升 3 倍）。
> - 128KB 方案总占用 434KB 超 384KB 上限，不可行。
> - vStep 受 `if (vStep > 255) vStep = 255` 硬件 repeat 上限约束，alignDK=64 时 96KB 方案已达上限 240（对齐 16 后）。

### 8.3 对 arch35 的推荐调整方案

若下一代芯片采用 arch35 路线，建议新增 arch 配置（如 arch36），做以下调整：

```cpp
// 推荐调整（基于 384KB UB）
constexpr uint32_t BUFFER_SIZE_BYTE_96K = 98304;
// stateBuff_ 和 tmpBuff2_ 从 32K 增至 96K
pipe_->InitBuffer(stateBuff_, BUFFER_SIZE_BYTE_96K);    // 原 32K
pipe_->InitBuffer(tmpBuff2_, BUFFER_SIZE_BYTE_96K);     // 原 32K
// vStep 公式同步调整
vStep_ = (96 * 1024 / 4) / alignDK_;   // = 24576 / alignDK_
if (vStep_ > 255) vStep_ = 255;
vStep_ = vStep_ / BF16_NUM_PER_BLOCK * BF16_NUM_PER_BLOCK;
```

其余 buffer（`inputQueue1_`/`queryBuff_`/`keyBuff_` 等）保持不变——它们存储的是 `MAX_MTP × alignDK` 级别的数据，与 vStep 无关，且 16KB 已足够覆盖最大 shape（$8 \times 512 \times 4B = 16KB$）。

### 8.4 对计算流程的影响（AIV=1）

#### 8.4.1 任务调度

AIV=1 时 `vectorCoreNum = min(1, B × NV) = 1`，`SetBlockDim(1)`，全部 $B \times N_v$ 个任务在单核上串行执行：

| 调度逻辑 | AIV > 1（当前） | AIV = 1 |
|:---------|:----------------|:--------|
| `ComputeAvgload()` | `Ceil(realT × NV, aivNum)` 均分到多核 | `Ceil(realT × NV, 1) = realT × NV`，全部归一核 |
| `IsCurrentBlock()` | 轮转分配，`usedblk` 递增切换核 | `usedblk` 始终为 0，所有任务命中 `blockIdx==0` |
| 并行度 | $B \times N_v$ 任务分散到多核 | 全串行，无并行 |

**调度逻辑无需修改**，`IsCurrentBlock` 在 `blockIdx=0, usedblk=0` 时对所有任务返回 `true`，行为正确。

#### 8.4.2 状态更新

state 在序列内递推更新（每 token 写回 GM）。多核场景下不同核处理不同 `(batch, head)`，state 槽位通过 `ssmStateIndices` 映射，天然无冲突。AIV=1 时全部串行，**反而简化了状态一致性**——无需担心核间竞争，state 写回顺序确定。

#### 8.4.3 性能影响

| 因素 | 影响 |
|:-----|:-----|
| 核间并行丧失 | $B \times N_v$ 任务全串行，吞吐量降为 1/aivNum（相比多核） |
| vStep 增大（384KB） | 减少 $D_v$ 方向迭代次数，降低 GM 读写往返，部分弥补串行损失 |
| 单核流水线 | arch35 Double Buffer 的 MTE2/V 流水仍在单核内有效，不受 AIV 数量影响 |
| 数据搬运 | 单核独占带宽，无多核竞争 GM 带宽 |

> **净效果**：AIV=1 导致性能大幅下降（串行化），但 384KB UB 通过增大 vStep 可减少 GM 访问次数，对单核吞吐有一定补偿。对于 $D_v$ 较大的 shape（如 512），vStep 从 8→32 可减少 75% 的 state Load/Store 次数，缓解串行瓶颈。

#### 8.4.4 Workspace

`workspaceSize_` 固定为 16MB（系统 workspace），与 AIV 数量和 UB 大小无关，无需调整。

### 8.5 总结

| 维度 | arch22（动态分配） | arch35（固定分配） |
|:-----|:-------------------|:-------------------|
| **UB 适配** | 自动适配，无需改码 | 需手动调整 `stateBuff_`/`tmpBuff2_` 大小及 vStep 公式 |
| **vStep 变化** | 192K→384K：8→32（最坏），64→128（典型） | 32K→96K：16→48（alignDK=512） |
| **代码改动** | 无 | 新增 arch 或修改常量（推荐 96K 方案） |
| **计算流程** | 不变 | 不变 |
| **AIV=1 调度** | 不需修改，`IsCurrentBlock` 自动收敛 | 同左 |
| **性能** | vStep 增大补偿部分串行损失 | 同左，但需先完成 UB 调整才能获益 |
| **风险** | vStep≥Dv 时 UB 闲置（低风险） | 96K 方案余量仅 14KB，需验证各 shape 组合 |
