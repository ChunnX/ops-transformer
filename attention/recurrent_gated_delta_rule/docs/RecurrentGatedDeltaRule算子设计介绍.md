# RecurrentGatedDeltaRule 算子设计介绍

## 1 概述

### 1.1 算子背景与功能

RecurrentGatedDeltaRule（循环门控 Delta Rule，简称 RGDR）是一种应用于循环神经网络 / 线性注意力机制的算子。它源自 DeltaNet 的 Delta Rule 状态更新机制，并引入两组衰减门控（全局衰减 $\alpha_t$ 和维度衰减 $\alpha_{kt}$），使状态矩阵能够在长序列中实现选择性遗忘与更新。

在每个时间步 $t$，根据当前输入 $q_t$、$k_t$、$v_t$ 和上一隐藏状态 $S_{t-1}$，计算当前输出 $o_t$ 和新隐藏状态 $S_t$。

### 1.2 原始 Delta Rule

DeltaNet 的核心思想是维护一个关联矩阵 $S \in \mathbb{R}^{d_v \times d_k}$，将 key 向量 $k$ 映射到 value 向量 $v$。在每个时间步，通过"预测-纠错"机制更新状态：

1. **预测**：用当前 key $k_t$ 查询状态，得到预测值 $\hat{v}_t = S_{t-1} \cdot k_t$
2. **纠差（Delta）**：计算预测误差 $\delta_t = v_t - \hat{v}_t = v_t - S_{t-1} \cdot k_t$（此即"Delta"名称的由来——它是实际值与预测值之差）
3. **更新**：将误差沿 $k_t$ 方向写回状态 $S_t = S_{t-1} + \beta_t \cdot \delta_t \cdot k_t^T$

其中 $\beta_t$ 为写入门控，控制新信息的写入强度。原始 Delta Rule 的完整公式为：

$$S_t = S_{t-1}(I - \beta_t k_t k_t^T) + \beta_t v_t k_t^T$$

展开可验证其等价于上述三步：

$$S_{t-1} - \beta_t S_{t-1} k_t k_t^T + \beta_t v_t k_t^T = S_{t-1} + \beta_t(v_t - S_{t-1}k_t)k_t^T$$

### 1.3 门控扩展（Gated Delta Rule）

RGDR 在 Delta Rule 基础上引入两组衰减门控，使状态能够随时间主动遗忘：

- **全局衰减门** $\alpha_t = e^{g_t} \in (0, 1)$（标量，约束 $g_t < 0$）：控制整个状态矩阵的遗忘程度。每个 token、每个 value head 独立设定。
- **维度衰减门** $\alpha_{kt} = e^{g_{kt}} \in (0, 1)^{d_k}$（向量，约束 $g_{kt} < 0$）：沿 key 维度进行选择性衰减，不同维度可以以不同速率遗忘。每个 token、每个 value head、每个 key 维度独立设定。

衰减门作用于状态矩阵的 **key 维度（列方向）**，实现为 $S_{t-1} \cdot \text{Diag}(\alpha_{kt})$（右乘对角矩阵），对每一列 $j$ 独立缩放 $\alpha_{kt}[j]$。全局衰减 $\alpha_t$ 再对整个矩阵做标量缩放。

> **注意**：$\text{Diag}(\alpha_{kt})$ 为右乘而非左乘。左乘 $\text{Diag}(\alpha_{kt}) \cdot S_{t-1}$ 作用于行方向（$d_v$ 维），当 $d_v \neq d_k$ 时维度不兼容。右乘 $S_{t-1} \cdot \text{Diag}(\alpha_{kt})$ 作用于列方向（$d_k$ 维），与 key 维度对齐。

门控 Delta Rule 的完整公式（第一形式，紧凑型）：

$$S_t = \alpha_t \cdot S_{t-1} \cdot \text{Diag}(\alpha_{kt}) \cdot (I - \beta_t k_t k_t^T) + \beta_t v_t k_t^T$$

展开 $(I - \beta_t k_t k_t^T)$ 并合并项，得到等价的第二形式（计算友好型）：

$$\boxed{S_t = \alpha_t \cdot S_{t-1} \cdot \text{Diag}(\alpha_{kt}) + \beta_t \cdot (v_t - \alpha_t \cdot S_{t-1} \cdot \text{Diag}(\alpha_{kt}) \cdot k_t) \cdot k_t^T}$$

**等价推导**：

$$\begin{aligned}
& \alpha_t S_{t-1} \text{Diag}(\alpha_{kt})(I - \beta_t k_t k_t^T) + \beta_t v_t k_t^T \\
=& \alpha_t S_{t-1} \text{Diag}(\alpha_{kt}) - \alpha_t \beta_t S_{t-1} \text{Diag}(\alpha_{kt}) k_t k_t^T + \beta_t v_t k_t^T \\
=& \alpha_t S_{t-1} \text{Diag}(\alpha_{kt}) + \beta_t (v_t - \alpha_t S_{t-1} \text{Diag}(\alpha_{kt}) k_t) k_t^T
\end{aligned}$$

第二形式的优势：**先衰减 → 再预测 → 最后纠差**，计算路径清晰，每个中间量有明确物理含义，且避免了显式构造 $d_k \times d_k$ 矩阵 $(I - \beta_t k_t k_t^T)$。算子实现采用此形式。

### 1.4 输出公式

$$o_t = \frac{S_t \cdot q_t}{\sqrt{d_k}}$$

输出 $o_t$ 是用 query $q_t$ 查询更新后状态 $S_t$ 的结果，缩放因子 $\frac{1}{\sqrt{d_k}}$ 与标准 Attention 一致，防止内积值随维度增大而发散。

从线性注意力视角看，状态 $S$ 是一个"外积记忆矩阵"：它递推地累积 $(k, v)$ 关联对，输出 $o = Sq / \sqrt{d_k}$ 即线性注意力的 query 阶段。门控机制使其成为带选择性遗忘的线性注意力，优于传统线性注意力的无界状态增长问题。

### 1.5 变量与维度

| 符号 | 维度 | 含义 |
|:-----|:-----|:-----|
| $S_{t-1}, S_t$ | $\mathbb{R}^{d_v \times d_k}$ | 隐藏状态矩阵（递推更新） |
| $q_t$ | $\mathbb{R}^{d_k}$ | Query 向量 |
| $k_t$ | $\mathbb{R}^{d_k}$ | Key 向量 |
| $v_t$ | $\mathbb{R}^{d_v}$ | Value 向量 |
| $\alpha_t = e^{g_t}$ | $\mathbb{R}$（标量） | 全局衰减门，$g_t < 0 \Rightarrow \alpha_t \in (0, 1)$ |
| $\alpha_{kt} = e^{g_{kt}}$ | $\mathbb{R}^{d_k}$（向量） | 维度衰减门，$g_{kt} < 0 \Rightarrow \alpha_{kt}[j] \in (0, 1)$ |
| $\beta_t$ | $\mathbb{R}$（标量） | 写入门，$0 < \beta_t < 1$ |
| $o_t$ | $\mathbb{R}^{d_v}$ | 输出向量 |

> **门控取值约束**：$g < 0$ 和 $g_k < 0$ 保证衰减因子 $e^g \in (0, 1)$，确保状态随时间衰减而非爆炸；$0 < \beta < 1$ 限制写入强度，保证数值稳定性。这些约束由用户保证，算子不校验具体数值。
>
> **可选门控**：$g$（对应参数 `g`）和 $g_k$（对应参数 `gk`）为可选输入。传入 `nullptr` 时分别等价于 $\alpha_t = 1$（不衰减）和 $\alpha_{kt} = \mathbf{1}$（不衰减），退化为原始 Delta Rule。

### 1.6 变步长与推测解码支持

RGDR 支持变步长（Variable Step Length）场景，适配 Multi-Token Prediction（MTP）和推测解码（Speculative Decoding）：

- **`actualSeqLengths`**（shape `(B,)`）：每个 batch 的实际序列长度 $L_i$（$0 < L_i \le 8$），允许不同 batch 处理不同数量的 token。所有 batch 的序列长度累加 $T = \sum_i L_i$。
- **`ssmStateIndices`**（shape `(T,)`）：将每个 token 映射到 state 矩阵的槽位索引。`state[ssmStateIndices[i]]` 表示第 $i$ 个 token 对应的状态矩阵。同一序列内的 token 通过递推共享/更新同一状态槽位。
- **`numAcceptedTokens`**（shape `(B,)`，可选）：推测解码场景下，每个 batch 实际接受的 token 数。状态矩阵的初始读取位置从 `ssmStateIndices[seq0 + numAcceptedTokens[i] - 1]` 开始，跳过被拒绝的 token 对应的状态，仅从最后一个被接受 token 的状态继续递推。

> **状态递推**：状态矩阵 $S$ 在序列内是**递推更新**的——每个 token 计算后更新的 $S_t$ 立即写回 GM，下一个 token 读取最新的 $S_t$ 继续计算。这种设计使得单个序列内的 token 必须串行处理，但不同 `(batch, head)` 之间可以并行。

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

`Compute()` 函数实现了第二形式公式的完整计算流程。以 arch22 为例（arch35 逻辑等价但使用 VF intrinsic），共分三个阶段：**状态衰减**、**Delta 纠差更新**、**Query 查询输出**。

#### 阶段一：状态衰减（对应公式 $\tilde{S} = \alpha_t \cdot S_{t-1} \cdot \text{Diag}(\alpha_{kt})$）

| 步骤 | 操作 | 公式 | 说明 |
|:-----|:-----|:-----|:-----|
| 1 | (可选) `Muls(state, gama)` | $S \leftarrow \alpha_t \cdot S$ | 全局衰减，标量乘整个状态矩阵 |
| 2 | (可选) `MatVecMul(state, gk)` | $S \leftarrow S \cdot \text{Diag}(\alpha_{kt})$ | 维度衰减，每列 $j$ 乘 $\alpha_{kt}[j]$（实现上逐行乘 gk 向量） |

> 步骤 1-2 合并后得到衰减后的状态 $\tilde{S} = \alpha_t \cdot S_{t-1} \cdot \text{Diag}(\alpha_{kt})$。若 $g$ / $g_k$ 未传入（`nullptr`），对应步骤跳过，$\alpha_t = 1$ 或 $\alpha_{kt} = \mathbf{1}$。

#### 阶段二：Delta 纠差与状态更新（对应公式 $\delta = \beta_t(v_t - \tilde{S}k_t)$，$S_t = \tilde{S} + \delta k_t^T$）

| 步骤 | 操作 | 公式 | 说明 |
|:-----|:-----|:-----|:-----|
| 3 | `MatVecMul(state, k) → tmp` | $\text{tmp}_{ij} \leftarrow S_{ij} \cdot k_j$ | 逐元素乘，准备矩阵-向量乘 |
| 4 | `ReduceSum(tmp → delta)` | $\delta_i \leftarrow \sum_j S_{ij} k_j$ | 行求和，得到预测值 $\hat{v} = \tilde{S} \cdot k_t$ |
| 5 | `delta = v - delta` | $\delta \leftarrow v_t - \tilde{S} \cdot k_t$ | **Delta（预测误差）**：实际值减预测值 |
| 6 | `Muls(delta, beta)` | $\delta \leftarrow \beta_t \cdot \delta$ | 写入门控，控制误差写入强度 |
| 7 | `Broadcast(tmp, delta)` | $\text{tmp}_{ij} \leftarrow \delta_i$ | 将 $\delta \in \mathbb{R}^{d_v}$ 广播为 $d_v \times d_k$ 矩阵 |
| 8 | `MatVecMul(tmp, k, state, isAdd)` | $S_{ij} \leftarrow S_{ij} + \delta_i \cdot k_j$ | **外积加**：$S_t = \tilde{S} + \delta \cdot k_t^T$ |

> 步骤 3-4 组合实现矩阵-向量乘 $\hat{v} = \tilde{S} \cdot k_t$（因 AIV-only 无 Cube 单元，通过逐元素乘 + ReduceSum 实现）。步骤 5-8 实现 Delta Rule 的核心纠差更新。

#### 阶段三：Query 查询与输出（对应公式 $o_t = S_t \cdot q_t / \sqrt{d_k}$）

| 步骤 | 操作 | 公式 | 说明 |
|:-----|:-----|:-----|:-----|
| 9 | `MatVecMul(state, q) → tmp` | $\text{tmp}_{ij} \leftarrow S_{ij} \cdot q_j$ | 逐元素乘，q 已在 CopyInQKV 阶段乘 scale |
| 10 | `ReduceSum(tmp → attn)` | $o_i \leftarrow \sum_j S_{ij} q_j$ | 行求和，得到输出 $o_t = S_t \cdot q_t$ |
| 11 | `Cast(stateOut, state)` / `Cast(attnOut, attn)` | — | 输出转回 bf16（若 state 为 fp32 则直接 DataCopy） |

> 步骤 9-10 组合实现 $o_t = S_t \cdot q_t$。query 的缩放因子 $\frac{1}{\sqrt{d_k}}$ 在数据搬入阶段（`CopyInQKV` 中 `Muls(qInUb, qInUb, scale_)`）已预先乘到 q 上，避免在计算阶段额外操作。

#### MatVecMul 实现差异

`MatVecMul`（矩阵-向量乘，通过逐元素乘 + 预留行求和位置实现）在两代架构中的实现方式不同：

| 架构 | 实现方式 | 关键参数 |
|:-----|:---------|:---------|
| arch22 | `Mul` / `MulAddDst` 高阶 API，逐 repeat 循环 | `REPEAT_LENTH=64`（单次处理 64 个 fp32 元素），`MAX_REPEAT_TIME=255`（单次最大 repeat 数，超过则分段循环） |
| arch35 | `VecMulMatVF` MicroAPI intrinsic | 寄存器级 `LoadAlign` / `Mul` / `MulAddDst` / `StoreAlign`，无 repeat 分段限制 |

arch35 的 `OuterAddVF` 还使用 `DIST_BRC_B32` 加载模式替代步骤 7 的显式 `Broadcast` 操作，将 delta 首元素广播到整个寄存器，减少一条指令。

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

## 8 下一代芯片适配分析（AIV=1, UB=384KB）

下一代芯片基于 arch35 路线演进，两个关键变化：**AIV 数量从多核改为 1**（单 Vector 核），**UB 总空间扩展至 384KB**。以下基于 arch35 现有实现分析适配方案。

### 8.1 当前 arch35 UB 占用现状

arch35 采用**硬编码常量**分配各 buffer，不依赖 Host 侧动态推导。当前各 buffer 占用如下：

| Buffer | 分配方式 | 大小 | FP32 state 合计 | BF16 state 合计 |
|:-------|:---------|:-----|:----------------|:----------------|
| `inputQueue1_` | TQue Double Buffer | 32K×2 / 16K×2 | 64KB | 32KB |
| `stateOutQueue_` | TQue Single | 32K / 16K | 32KB | 16KB |
| `stateBuff_` | TBuf | 32K | 32KB | 32KB |
| `tmpBuff2_` | TBuf | 32K | 32KB | 32KB |
| `attnOutQueue_` | TQue Single | 16K | 16KB | 16KB |
| `gamaBuff_` | TBuf | 16K | 16KB | 16KB |
| `queryBuff_` | TBuf | 16K | 16KB | 16KB |
| `keyBuff_` | TBuf | 16K | 16KB | 16KB |
| `valueBuff_` | TBuf | 16K | 16KB | 16KB |
| `tmpBuff1_` | TBuf | 2K | 2KB | 2KB |
| **总计** | | | **242KB** | **194KB** |

> 当前 FP32 state 场景占用 242KB，说明 950 实际 UB 不小于 256KB。下一代芯片 UB 扩展至 384KB 后，FP32 state 场景有 142KB 闲置，BF16 state 场景有 190KB 闲置。

### 8.2 vStep 提升方案

vStep 由 `stateBuff_` 和 `tmpBuff2_` 两个 buffer 的大小决定（分别存储 state 的 fp32 工作副本和 VecMulMat 输出，容量均为 `alignDK × vStep × 4B`）。当前公式：

```
vStep = 8192 / alignDK          // 8192 = 32KB(stateBuff_) / 4B(sizeof fp32)
if (vStep > 256) vStep = 255    // 硬件 repeat 上限
vStep = vStep / 16 * 16         // 对齐到 16
```

增大这两个 buffer 即可线性提升 vStep。以下分析各方案（以 FP32 state 为基准）：

| `stateBuff_`/`tmpBuff2_` 大小 | vStep（alignDK=512） | vStep（alignDK=128） | vStep（alignDK=64） | 新总占用 | 384KB 内余量 |
|:------------------------------|:---------------------|:---------------------|:---------------------|:---------|:-------------|
| 32KB（当前） | 16 | 64 | 128 | 242KB | 142KB |
| 64KB | 32 | 128 | 240 | 306KB | 78KB |
| 96KB | 48 | 192 | 240 | 370KB | 14KB |
| 128KB | 64 | 256 | 240 | 434KB | **超限** |

> - vStep 受 `if (vStep > 256) vStep = 255` + 对齐 16 约束，实际有效上限为 240（vStep > 256 时被截断为 255，再对齐到 240）
> - **推荐 96KB 方案**：384KB 内可容纳（余 14KB），vStep 提升显著（alignDK=512 时 16→48，提升 3 倍）
> - 128KB 方案总占用 434KB 超 384KB 上限，不可行
> - 其余 buffer（`inputQueue1_`/`queryBuff_`/`keyBuff_` 等）存储 `MAX_MTP × alignDK` 级数据，与 vStep 无关，16KB 已覆盖最大 shape（$8 \times 512 \times 4B = 16KB$），无需调整

**vStep 提升对 $D_v$ 迭代次数的影响**：

| 场景 | vStep（32KB） | $D_v$ 迭代次数 | vStep（96KB） | $D_v$ 迭代次数 | 减少 |
|:-----|:--------------|:--------------|:--------------|:--------------|:-----|
| $D_k=512, D_v=512$ | 16 | 32 | 48 | 11 | 66% |
| $D_k=128, D_v=128$ | 64 | 2 | 192 | 1 | 50% |
| $D_k=64, D_v=512$ | 128 | 4 | 240 | 3 | 25% |

> 每次 $D_v$ 迭代涉及一次 state 的 GM→UB Load 和 UB→GM Store。vStep 增大直接减少 GM 读写往返次数，是单核场景下的关键优化手段。

### 8.3 推荐代码调整

建议新增 arch 配置（如 arch36），做以下调整：

```cpp
constexpr uint32_t BUFFER_SIZE_BYTE_96K = 98304;

pipe_->InitBuffer(stateBuff_, BUFFER_SIZE_BYTE_96K);    // 原 32K
pipe_->InitBuffer(tmpBuff2_, BUFFER_SIZE_BYTE_96K);     // 原 32K

vStep_ = (96 * 1024 / 4) / alignDK_;   // = 24576 / alignDK_
if (vStep_ > 256) {
    vStep_ = 255;
}
vStep_ = vStep_ / BF16_NUM_PER_BLOCK * BF16_NUM_PER_BLOCK;
```

其余 buffer 和计算流程保持不变。TilingData 中无需新增字段（arch35 的 vStep 在 Kernel 构造函数中计算，不依赖 Host 传递）。

### 8.4 AIV=1 对任务调度的影响

AIV=1 时 `vectorCoreNum = min(1, B × NV) = 1`，`SetBlockDim(1)`，全部 $B \times N_v$ 个任务在单核上串行执行。

| 调度逻辑 | AIV > 1（当前） | AIV = 1（下一代） |
|:---------|:----------------|:------------------|
| `ComputeAvgload()` | `Ceil(realT × NV, aivNum)` 均分到多核 | `Ceil(realT × NV, 1) = realT × NV`，全部归一核 |
| `IsCurrentBlock()` | 轮转分配，`usedblk` 递增切换核 | `usedblk` 始终为 0，所有任务命中 `blockIdx==0` |
| 并行度 | $B \times N_v$ 任务分散到多核 | 全串行，无并行 |

**调度逻辑无需修改**。`IsCurrentBlock` 在 `blockIdx=0, usedblk=0` 时对所有任务返回 `true`，行为正确。`ComputeAvgload` 中 `maxSeqLen > avgload` 的保护逻辑也仍然有效（单核时 `avgload = max(realT × NV, maxSeqLen)`）。

### 8.5 AIV=1 对状态更新的影响

state 在序列内递推更新（每 token 写回 GM）。多核场景下不同核处理不同 `(batch, head)`，state 槽位通过 `ssmStateIndices` 映射，天然无冲突。AIV=1 时全部串行，**反而简化了状态一致性**——无需担心核间竞争，state 写回顺序确定。

### 8.6 AIV=1 对性能的影响

| 因素 | 影响 |
|:-----|:-----|
| 核间并行丧失 | $B \times N_v$ 任务全串行，吞吐量降为 1/aivNum（相比多核） |
| vStep 增大（96KB 方案） | 减少 $D_v$ 方向迭代次数，降低 GM 读写往返，部分弥补串行损失 |
| Double Buffer | `inputQueue1_` 的 MTE2/V 流水在单核内仍然有效，不受 AIV 数量影响 |
| GM 带宽 | 单核独占带宽，无多核竞争，DataCopy 效率不降 |
| 计算流水 | arch35 的 VecMulMatVF / OuterAddVF MicroAPI 流水不受核数影响 |

> **净效果**：AIV=1 导致性能大幅下降（串行化），这是下一代芯片的主要性能瓶颈。384KB UB 通过将 `stateBuff_`/`tmpBuff2_` 从 32KB 扩至 96KB，使 vStep 提升 3 倍（alignDK=512 场景），减少 66% 的 state GM 读写往返，对单核吞吐有一定补偿。但对于 $B \times N_v$ 较大的场景，串行化损失难以仅靠 vStep 增大完全弥补。

### 8.7 Workspace

`workspaceSize` 固定为 16MB（系统 workspace），与 AIV 数量和 UB 大小无关，无需调整。

### 8.8 总结

| 维度 | 分析结论 |
|:-----|:---------|
| **UB 适配** | 需手动调整 `stateBuff_`/`tmpBuff2_` 从 32KB 至 96KB，并同步修改 vStep 公式 |
| **vStep 变化** | alignDK=512：16→48；alignDK=128：64→192；$D_v$ 迭代次数显著减少 |
| **代码改动** | 仅 2 个 buffer 大小常量 + vStep 公式分母，其余不变 |
| **AIV=1 调度** | 不需修改，`IsCurrentBlock` / `ComputeAvgload` 自动收敛到单核 |
| **AIV=1 状态更新** | 串行执行，无核间竞争，状态一致性更简单 |
| **性能** | 串行化是主要瓶颈，vStep 增大部分补偿（减少 GM 访问），Double Buffer 流水仍有效 |
| **风险** | 96KB 方案余量仅 14KB，需验证各 shape + dtype 组合不超限 |
