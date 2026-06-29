# flash_attn_example 算子详解文档

## 一、算子设计

### 1.1 功能与架构

本算子基于华为昇腾 NPU 的 AscendC 框架，采用 **Host（CPU）做 Tiling + Device（NPU）跑 Kernel** 的分离式架构，实现标准自注意力计算。接口参数与约束详见 [README.md](./README.md)。

完整调用链路：**Python** `torch.ops.ascend_ops.flash_attn_example(q,k,v,mask,scale,True)` → **C++ 入口** `FlashAttnNpu()` → **Host Tiling** `FaTiling::DoTiling()` → **TilingData 拷贝到 Device** → **Kernel 调度** `FaKernelInterface → FlashAttentionKernel`。

### 1.2 Tiling 基本块

Host Tiling 侧通过 `AdjustSinnerAndSouter()` 固定设定：

| 参数 | 值 | 含义 |
|------|-----|------|
| `sOuterFactor_` | 64 | S1 维度的基本单位（AIV 视角） |
| `sInnerFactor_` | 128 | S2 维度的基本单位 |
| `mBaseSize`（split 用） | `sOuterFactor_ × CV_RATIO = 128` | AIC 视角的 M 维度 tile 大小 |
| `s2BaseSize`（split 用） | 128 | S2 维度 tile 大小 |

Kernel 侧同样以 M=128、S2=128、D=128 为基本 tile（`Aligned128`，`fa_kernel_public.h`）。

### 1.3 多核切分策略

算子将 `(B, N2, S1G, S2)` 四维迭代空间切分到各 AIC core 上并行执行，目标是各 core 负载尽量均衡。最小调度单位为 (M, S2) tile。

**Cost 模型**：每个 tile 的计算 cost 通过经验公式估算 `CalcCost(m, s2) = 6 × ceil(M/16) + 10 × ceil(S2/64)`。normal tile（M=128, S2=128）cost = 68；tail tile 按实际尾块大小折算。

**核心数选择**：对每个 batch 遍历所有 S1G 行，在 causal 约束下计算有效 S2 block 总数 `totalBlockNum` 和 `totalCost`。候选核心数上界为 `min(aicNum, totalBlockNum)`，下界为 `floor(sqrt(totalBlockNum) + 0.5)`，`+0.5` 为了向上取整。在候选范围内遍历，选取使各 core 最大 cost 最小的核心数。

**贪心分配**：确定核心数后，按三层粒度依次尝试为每个 core 分配工作量：
1. **按 Batch 整取**：若整个 (B×N2) 组的 cost 不超过 avgCost，整体拿走
2. **按行整取**：若一个 S1G 行的全部 S2 block 不超过 avgCost（含 tolerance），整行拿走
3. **按 Block 切分**：行内逐 S2 block 分配，实现 streamK 并行
4. **保底**：若什么都未分配到，强制至少取 1 个 block

**Combine 触发**：同一 S1G 行被多核切分时，该行 partial 结果需后续 Combine 阶段归约。Tiling 阶段记录行内切分信息，生成 Combine 任务元数据。

### 1.4 内存划分

NPU 上存在清晰的多级存储层次，从大到小依次为 GM → L1 → L0A/L0B/L0C → UB。

#### GM（Global Memory）

GM 是 Device 端主存（HBM，GB 级）。S2 多核切分时每个 S1G 行需 workspace 暂存 partial 结果（仅 `isCombine=true` 时分配，`fa_tiling.cpp:167-181`）。Workspace 为 ping/pong 双缓冲，包含三部分：

| 区域 | 大小计算 | 用途 |
|------|---------|------|
| **accumOut** | `numBlocks × 2 × mSize × dSize × sizeof(float)` | P×V partial 累加器 |
| **LSE sum** | `numBlocks × 2 × mSize × 8` | Log-Sum-Exp 的 sum |
| **LSE max** | `numBlocks × 2 × mSize × 8` | Log-Sum-Exp 的 max |

其中 `mSize = 128`，`dSize = 128`。

#### L1（AIC 侧暂存）

L1 是 AIC core 的片上 SRAM，总池 512KB。Q/K/V 三个 tile 分别做 double-buffer，P 矩阵做 triple-buffer（AIV 写、AIC 读）：

| Buffer | 大小 | 缓冲策略 |
|--------|------|---------|
| `l1QBuffers`（Q tile） | 32KB × 2 = 64KB | double-buffer（同一 S1G 行内 Q 复用） |
| `l1KBuffers`（K tile） | 32KB × 2 = 64KB | double-buffer |
| `l1VBuffers`（V tile） | 32KB × 2 = 64KB | double-buffer |
| `l1PBuffers`（P matrix） | 32KB × 3 = 96KB | triple-buffer（AIV 产出、AIC 消费） |
| **L1 总计** | **288KB**（从 512KB 总池分配） | |

#### L0（AIC Cube 专用）

L0 是 Cube 引擎的本地高速缓存，分 **L0A**（左矩阵输入）、**L0B**（右矩阵输入）、**L0C**（结果输出）三个独立空间，仅 AIC core 可访问（`fa_block_cube.h:118-141`）：

| 空间 | 总容量 | 本算子分配 | 缓冲策略 |
|------|--------|----------|---------|
| **L0A** | 64KB | 32KB × 2 = 64KB | double-buffer，存放 Q（BMM1）或 P（BMM2） |
| **L0B** | 64KB | 32KB × 2 = 64KB | double-buffer，存放 K^T（BMM1）或 V（BMM2） |
| **L0C** | 256KB | 64KB × 4 = 256KB | **quad-buffer** |

当前 D=128 满足 `128×128×4=64KB` 条件，使用 quad-buffer（不满足时降级 double-buffer），可同时保有 4 份结果以隐藏 Fixpipe 延迟。

#### UB（AIV Vector 专用 + AIC Fixpipe 写入）

UB 主要被 AIV Vector Core 使用，同时 AIC 通过 Fixpipe（L0C→UB）将矩阵乘结果写入 UB 供 AIV 消费。以下是本算子 UB 中各 buffer 的使用情况：

**UB 各 Buffer 汇总**：

| Buffer | 大小 | 块数 | 说明 |
|--------|------|:---:|------|
| `bmm1Buffers` | 32KB（64×128×fp32） | 2（ping/pong） | BMM1 结果，AIC Cube→UB 写入，AIV Vector 消费 |
| `bmm2Buffers` | 32KB（64×128×fp32） | 2（ping/pong） | BMM2 结果，AIC Cube→UB 写入，AIV Vector 消费 |
| `softmaxSumBuf` | 256B | 3 | Softmax 分母 sum（64×fp32） |
| `softmaxMaxBuf` | 256B | 3 | Softmax 分子 max（64×fp32） |
| `softmaxExpBuf` | 256B | 3 | Flash Update 指数比例因子（64×fp32） |
| `stage1OutQue` | 16.5KB（128×128×bf16） | 2（ping/pong） | AIV cast bf16 后搬至 L1，作为 BMM2 P 矩阵输入 |
| `stage2OutBuf` | 32KB（64×128×fp32） | 1 | Flash Update 结果 |
| `attnMaskInQue` | 8KB（64×128×uint8） | 2（ping/pong） | GM→UB bool mask 拷贝（hasAttn=true 时） |
| `maxBrdcst` | 2KB（64×8×fp32） | 1 | Softmax max Broadcast 暂存 |
| `sumBrdcst` | 2KB（64×8×fp32） | 1 | Softmax sum Broadcast 暂存 |
| `commonTBuf` | 512B | 1 | 通用临时空间 |
| **总计** | | | **~215.75KB** |

每个 AIV core 启动 2 个 sub-block，各处理 M/2 行（BMM buffer 中 `mBaseSize/CV_RATIO=64` 的来源）。

### 1.5 PRELOAD 流水设计

#### 流水线概览

AIC 和 AIV 是**不同的物理 core**，各自运行同一份 kernel 二进制但走不同分支。一次 tile 的完整生命周期包含 4 个步骤：**C1（BMM1）→ V1（Softmax）→ C2（BMM2）→ V2（FlashUpdate+输出）**。C1 与 V1 之间、V1 与 C2 之间、C2 与 V2 之间均通过 cross-core sync（`SetCrossCore`/`WaitCrossCore`）实现生产者-消费者同步。

为隐藏 AIC 与 AIV 之间的等待延迟，设计了 **PRELOAD_N=2 的软件流水线**——每个 core（无论 AIC 还是 AIV）在同一轮迭代中，同时处理"当前 tile 的 Phase1"和"2 轮前 tile 的 Phase2"：

```
        AIC Core                              AIV Core
        ────────                              ────────
loop 0: C1(T0)                                V1(T0)
loop 1: C1(T1)                                V1(T1)
loop 2: C1(T2) + C2(T0)                       V1(T2) + V2(T0)
loop 3: C1(T3) + C2(T1)                       V1(T3) + V2(T1)
loop 4: C1(T4) + C2(T2)                       V1(T4) + V2(T2)
...
```

> 说明：**设计PRELOAD两轮** —— V1（Softmax计算 + P矩阵写入 L1 ）耗时较长，PRELOAD_N = 2 保证首轮 C2 启动时，对应的 V1 一定已经完成 P 矩阵的拷贝，避免跨核同步等待。

核心思想：**AIC/AIV 各自并行地处理当前 tile 的 Phase1 与 PRELOAD_N 轮前 tile 的 Phase2**。从 loop=2 开始进入稳态，每个 core 每轮同时发射 Phase1 和 Phase2。

> 具体代码见 `fa_kernel.h:250-295`。

---

## 二、代码结构说明

### 2.1 整体架构概览

```
torch_interface.cpp          ← PyTorch 算子注册 + 入口函数
  ├─ op_host/fa_tiling.h/.cpp     ← Host Tiling：计算分核元数据
  │    ├─ fa_tiling_public.h      ← 公共类型定义（ContextParamsForTiling 等）
  │    ├─ split_core.h/.cpp       ← 多核负载划分算法
  │    └─ fia_tiling_data.h       ← Host→Device 传递的 TilingData 结构体
  └─ op_kernel/fa_kernel_interface.h  ← Device Kernel 入口（模板分发 AIC/AIV）
       └─ op_kernel/kernel/fa_kernel.h ← Kernel层（主循环 + 流水控制）
            ├─ fa_block_cube.h    ← AIC Cube：Q/K/V 拷贝 + BMM1/BMM2
            ├─ fa_block_vec.h     ← AIV Vector：Softmax + Flash Update + 输出
            └─ fa_block_vec_combine.h ← AIV Combine：多核 partial 结果合并
```

附属功能模块按职责划分：

| 目录 | 功能 |
|------|------|
| `op_kernel/memcopy/` | GM↔L1↔UB 数据搬运：`offset_calculator.h`（多维索引计算，支持 BSND/BSNGD 等 24 种 layout）+ `memory_copy.h`（CopyQueryGmToL1、CopyKvGmToL1、CopyAttnOutUbToGm） |
| `op_kernel/matmul/` | Cube 矩阵乘：`MatmulFull` / `MatmulK`（D 维度切分）/ `MatmulN`（S2 维度切分），`MatmulBase` 根据 M×K×N 自动选择 |
| `op_kernel/vector/` | Vector 操作：Mask 拷贝（含 G 维度广播 + causal 跳过优化）、无效行检测与置零 |
| `op_kernel/vector/vf/` | VF 指令加速的 Flash Softmax（6 个实现分别处理 aligned128/unaligned64/unaligned128 × update/no_update）、Flash Update Rescale、Combine LSE 计算 |
| `op_kernel/buffer_manager/` | Buffer 抽象（`Buffer<T>` 封装内存+同步原语）、线性分配器（`BufferManager`）、多缓冲策略（DB/3buff/4buff） |

### 2.2 Torch接口：`torch_interface.cpp` 与 `fia_tiling_data.h`

`torch_interface.cpp` 是算子唯一入口：接收 PyTorch tensor → 校验 → `FaTiling::DoTiling()` 生成 **TilingData** → `aclrtMemcpy` 发送到 Device → 启动 Kernel。

TilingData（~2.3KB，定义在 `fia_tiling_data.h`）包含三类元数据：`FaBaseParams`（B/S1/S2/G/N2/D/scale/coreNum）、`FaAttnMaskParams`（sparseMode/preTokens/nextTokens）、`FaMetaData`（per-core 起止区间 + workspace 索引 + combine 任务）。

### 2.3 Host Tiling：`op_host/`

`FaTiling::DoTiling()` 流程：解析 input shape + 平台信息 → 设置 causal 参数（`preToken=S1, nextToken=S2-S1`）→ **分核**（`SplitPolicy` 调用 `split_core::SplitCore()`，详见 2.3.1）→ 填充 base/mask params → 计算 workspace。

#### 2.3.1 分核算法：`op_host/split_core.cpp`

分核算法解决"给定 N 个 AIC core，如何将总工作量尽量均匀地分给各个 core"的问题。

**Cost 模型**：每处理一个 (M, S2) tile 的 cost 由经验公式估算：`CalcCost(m, s2) = 6×ceil(M/16) + 10×ceil(S2/64)`。对于 normal tile（M=128, S2=128），cost = 6×8 + 10×2 = **68**；tail tile 按实际尾块大小按此公式折算。

**分核流程**（`SplitCore`，`split_core.cpp:616-651`）：

1. **CalcSplitInfo**：计算每个 batch 的 S1G 块数 = `ceil(S1×G/mBaseSize)`、S2 块数 = `ceil(S2/s2BaseSize)`、尾块大小
2. **CalcCostInfo**：对每个 batch，遍历所有 S1G 行，通过 `CalcS2Range` 计算 causal mask 约束下的有效 S2 范围，对每个有效 S2 block 累加 cost。最终得到 `totalCost` 和 `totalBlockNum`
3. **搜索最优核心数**：`minCore = floor(sqrt(totalBlockNum) + 0.5)`（最少要 sqrt 块数，`+0.5` 为了向上取整），`maxCore = min(aicNum, totalBlockNum)`。遍历候选 coreNum，对每个调用 `CalcSplitPlan`，选取 `maxCost` 最小的方案
4. **CalcSplitPlan 贪心分配**（`split_core.cpp:522-589`）：对每个 core，按三层粒度尝试：
   - **AssignByBatch**：若整个 B×N2 组的 cost 能放入 avgCost，整组拿走
   - **AssignByRow**：若整行 S1G 的所有 S2 block 能放入（含 tolerance = lastBlockCost/2），整行拿走
   - **AssignByBlock**（streamK=true）：对单个 S2 block 进行分配（行内切分）
   - 若什么都没分配到，**ForceAssign** 强制至少拿走 1 个 block
5. **记录 per-core 区间**：`bN2End[i]` / `mEnd[i]` / `s2End[i]` 分别标记 core i 处理到哪个 batch-head / S1G 行 / S2 块
6. **Combine 标记**：当同一 S1G 行被切分到多个 core 时（`curKvSplitPart > 1`），记录 Combine 任务。后续 `SplitCombine` 将 combine 工作进一步分配到 AIV cores

### 2.4 Device Kernel：`op_kernel/`

#### 2.4.1 Attention 计算

**入口与模板分发**（`fa_kernel_interface.h`）：每一个 NPU 物理 core 都执行同一个 Kernel 函数，通过 `g_coreType` 判断自己的角色，用 `std::conditional` 选择实现——AIC 启用 `FABlockCube`，AIV 启用 `FABlockVec` + `FaBlockVecCombine`。

**公共定义**（`fa_kernel_public.h`）：定义运行时信息结构 `RunInfoX`（单次 tile 的坐标、实际 M/S2 大小、first/last S2 loop 标志等）和常量信息 `CommonConstInfo`（从 TilingData 解析的维度参数和 per-core 起止范围）。

**Kernel层**（`fa_kernel.h`，详见 1.5 节）：`FlashAttentionKernel::Process()` 的 Init 阶段读取 per-core 元数据、初始化 GM tensor 指针、AIV 清零输出、分配 UB/L1 buffer。FlashAttention 主循环以 PRELOAD_N=2 流水线运行，dispatch 阶段按 `TASK_DEAL_MODE` 枚举判断每个 `(bN2, gS1, s2)` 位置的处理方式（CREATE_TASK / SKIP / DEAL_ZERO / NOT_START / S2_END），execute 阶段同步推进当前 tile 的 Phase1 与 2 轮前 tile 的 Phase2。

**Cube 操作**（`fa_block_cube.h`）：

1. **CopyQueryTile**：GM（BSNGD）→ L1（NZ），128×128 bf16 = 32KB。首次 S2 loop 时拷贝，后续复用（Q 在同一 S1G 行内不变）
2. **CopyKeyTile / CopyValueTile**：GM（BSND）→ L1（NZ），每次 S2 loop 重新拷贝
3. **IterateBmm1**：Q×K^T 矩阵乘，D 维度 K-split，结果经 `FixpipeMm1` 从 L0C→UB，dual-destination 拆分为两个 64×128 fp32 给两个 AIV sub-core
4. **IterateBmm2**：从 L1 读取 P → P×V 矩阵乘（`MatmulFull`，支持 realM tail rows）→ Fixpipe L0C→UB

**Vector 操作**（`fa_block_vec.h`）：

1. **ProcessVec1**（Softmax）：从 BMM1 UB 获取 QK^T → AttnMaskCopyIn（GM [2048,2048] → UB，GS1 layout 广播 G）→ ProcessVec1Vf（按 S2 大小和对齐性 dispatch 到 6 种 VF 实现之一）→ DataCopy P(bf16) 到 L1 → UpdateExpSumAndExpMax（非首次 S2 loop 时合并统计量）→ 若为 S2 分核场景且最后 S2 loop，ComputeLogSumExpAndCopyToGm 将 LSE 写入 GM
2. **ProcessVec2**（Flash Update）：从 BMM2 UB 获取 P×V → 根据 first/last S2 loop 选择 `FlashUpdateNew`（中间）或 `FlashUpdateLastNew`（最后）+ `LastDivNew`（单块）→ RowInvalid（无效行置零）→ DivCast（fp32→bf16）→ CopyOutAttentionOut 写出
3. **InitOutputSingleCore**：初始化阶段将本 core 负责的输出区清零
4. **DealZeroActSeqLen**：零长序列时，将该 (B, N2) 组输出清零

#### 2.4.2 Combine 规约

**Combine 操作**（`fa_block_vec_combine.h`）：当 S2 被多核切分时，所有 core 的主循环结束后 AIV core 执行 Combine。从 workspace GM 读取各 KV 分片的 LSE sum/max 到 UB → ComputeScaleValue（找全局 max，计算归一化 softmax scale）→ ReduceFinalRes（`output += scale × partial_output`，double-buffer 搬运）→ DealInvalidRows + CopyFinalResOut（fp32→bf16 cast → 写回 GM）。

---

## 三、典型 Case 代入走读

以下分析基于 **32 Cube Core + 64 Vector Core**（CV_RATIO=2）、Tiling 块大小 M=128 / S2=128 / D=128。

### Case 1：B=1, N1=32, N2=1, S1=2, S2=8192, D=128

#### 3.1.1 Host Tiling — 分核

**基础数据**：
- G = N1/N2 = 32（GQA ratio）
- GS1 空间 = S1 × G = 2 × 32 = **64** 个 GS1 token
- S1G 块数 = ceil(64 / 128) = **1** 块，**tail size = 64**（不满 128）
- S2 块数 = ceil(8192 / 128) = **64** 块，全部 normal
- causal 参数：preToken=2，nextToken=8192-2=**8190**

> S1=2、S2=8192 是典型的 decode 阶段场景（KV cache 已长，每次只输入少量新 token）。两个 Q token 可 attend 到 KV 中几乎全部 token。

**Causal 约束**：唯一 S1G 块的 S1 token 范围为 [0, 1] → s2FirstToken = 0，s2LastToken = 8191 → 有效 S2 范围 = **全部 64 个块**。

**Cost 与核心分配**：
- tail tile cost = `CalcCost(64, 128) = 6×4 + 10×2 = 44`
- 总 cost = 64 × 44 × 1(N2) × 1(B) = **2816**
- 总 block 数 = 64
- maxCore = min(32, 64) = **32**，minCore = √64+0.5 ≈ 8.5 → 遍历 8→32，选取 **32 个 AIC core**

avgCost = 2816/32 = **88**（恰好 2 个 tail block 的 cost）。每个 core 各分配 2 个 S2 block——同一 S1G 行被 32 个 core 切分，**必须进入 Combine 路径**（combineNum = 1，拆为多 AIV core 并行）。

#### 3.1.2 Device Kernel — 单 core 计算流程

以 core 0（处理 S2 block 0~1 = KV tokens 0~255）为例：
- `actMSize=64`（tail M）、`actVecMSize=32`（sub-core 处理 32 rows）
- `isS2SplitCore=true`，每个 core 运行 2 个 S2 loop

**Step A — BMM1**

1. **Copy Q**：Q[b=0, n2=0, gS1_start:gS1_start+64, d=0:128] → L1 NZ（64×128 bf16 = 16KB）
2. **Copy K**：K[b=0, n2=0, s2*128:s2*128+128, d=0:128] → L1 NZ（128×128 bf16 = 32KB）
3. **Matmul**：Q(64×D) × K^T(D×128)，K-split matmul。L1→L0A/L0B（double-buffer）→ Cube MM → L0C
4. **Fixpipe** L0C→UB：dual-destination 拆为 2×32×128 fp32 → `SetCrossCore` 通知 AIV

Q 在同一 S1G 行内不变，loop 1 时仅重新拷贝 K（block 1 的 KV），复用 Q。

**Step B — Vec1**

> 每个 AIC core 对应 2 个 AIV sub-core，各处理 M/2=32 rows。

**Loop 0**（`isFirstS2Loop=true`, `isLastS2Loop=false`）：

1. **AttnMaskCopyIn**：从 GM（2048×2048 bool）拷贝 [0:32, 0:128] 区域到 UB，按 GS1 layout 广播 G=32
2. **ProcessVec1Vf**（no_update, S2=128, aligned128）：逐行 `QK^T × scale → mask(-inf) → row_max → exp(x-max) → sum → norm → cast bf16`
3. **DataCopy** → L1 `l1PBuffers`（32×128 bf16）+ `SetCrossCore` 通知 AIC
4. 首次 loop，跳过 `UpdateExpSumAndExpMax`
5. `isS2SplitCore=true` 且是首次 loop → `ComputeLogSumExpAndCopyToGm` 将 LSE max/sum 写入 workspace GM

**Loop 1**（`isFirstS2Loop=false`, `isLastS2Loop=true`）：

1. AttnMaskCopyIn → `[0:32, 128:256]` 区域
2. ProcessVec1Vf（**update**, S2=128, aligned128）：与 loop 0 的统计量合并，更新 row_max/row_sum
3. DataCopy → L1 → `SetCrossCore`
4. 最后 loop → 更新 LSE 到 workspace GM

**Step C — BMM2**

1. 从 L1 读取 P（等待 cross-core sync），64×128 bf16
2. **Copy V** → L1 NZ，128×128 bf16
3. **MatmulFull** P(64×S2) × V(S2×128)，realM=64
4. Fixpipe L0C→UB dual-dest 2×32×128 fp32 → `SetCrossCore`

**Step D — Vec2**

**Loop 0**：首次 loop → `isS2SplitCore=true`，通过 `Bmm2ResForCombineCopyOut` 将 64×128 fp32 partial 累加器写入 workspace GM

**Loop 1**：`FlashUpdateNew` 与 loop 0 的 partial 合并 → 更新 workspace partial 累加器

**流水推进**：`UpdateAxisInfo → s2Cur++`（loop 0→1）→ s2Cur 越界 → 当前 core 完成。

其他 31 个 core 流程相同，仅 KV 分片不同（S2 blocks 2~63，对应 KV tokens 256~8191）。

#### 3.1.3 Combine 阶段

所有 32 个 core 完成后，AIV core 执行 Combine（`fa_block_vec_combine.h`）：

1. **读取 LSE**：从 workspace GM 读取 32 个 KV 分片的 LSE max 和 sum（各 64 行 × fp32）
2. **ComputeScaleValue**：找全局 max → `scale[i] = exp(lse_max[i] - global_max)` → 归一化
3. **ReduceFinalRes**：逐行 merge → `output += scale[i] × partial_output[i]`（double-buffer 流水搬运）
4. **DealInvalidRows + CopyFinalResOut**：causal 无效行置零 → fp32→bf16 cast → 写回 GM

---

### Case 2：B=1, N1=32, N2=1, S1=8192, S2=8192, D=128

#### 3.2.1 Host Tiling — 分核

**基础数据**：
- G = N1/N2 = 32（GQA ratio）
- GS1 空间 = S1 × G = 8192 × 32 = **262144** 个 GS1 token
- S1G 块数 = ceil(262144 / 128) = **2048** 块，全部为 normal（262144 恰被 128 整除）
- S2 块数 = ceil(8192 / 128) = **64** 块，全部为 normal（8192 恰被 128 整除）
- causal 参数：preToken=8192，nextToken=0（S1=S2，无"未来"token）

**Causal 约束下的有效 S2 范围**（按 S1G 块分组）：

每 S1G 块对应 128 个 GS1 token = 4 个 S1 token。S2 有效范围为 [0, s1_last]（causal，preToken=8192 意味着 S2 只能看到位置 ≤ S1 的 token）。每 32 个连续 S1G 块共享同一 S2 块数，共 64 档：

| S1G 块区间 | S1 token 范围 | S2 块数 / 行 |
|-----------|--------------|:--------:|
| 0 ~ 31    | 0 ~ 127      | 1 |
| 32 ~ 63   | 128 ~ 255    | 2 |
| 64 ~ 95   | 256 ~ 383    | 3 |
| ... | ... | ... |
| 2016 ~ 2047 | 8064 ~ 8191   | 64 |

**Cost 与核心分配**：
- 单 normal tile cost = `CalcCost(128, 128) = 6×8 + 10×2 = 68`
- 总 S2 block 数 = 32×(1+2+...+64) = **66560** 块
- 总 cost = 66560 × 68 = **4,526,080**
- maxCore = min(32, 66560) = **32**
- minCore = √(66560) + 0.5 ≈ 258.5 → min(258, 32) = **32**
- 因此固定使用 **32 个 AIC core**（对应 64 个 AIV core）

以 32 核为例，avgCost per core ≈ 4,526,080/32 = **141,440**（≈2080 个 normal block 的 cost）。分配结果：
- **前半 core**（~前 16 个）：处理较多 S1G 行（80~120 行），每行 S2 块数少（1~16 块），每次恰好到达 row 边界
- **后半 core**（~后 16 个）：处理较少 S1G 行（20~40 行），每行 S2 块数多（49~64 块）。由于 avgCost=141,440 不能总是被每行 cost 整除，部分 core 会在行内切分 S2 块

例如：在第 9 个 core 处（处理第 ~576 行，该行有 18 个 S2 block），贪心策略放入若干整行后，剩余 cost 限额不足以装入完整一行，恰好切分该行的部分 S2 block——该行被分到两个 core。此时记录分核标记，**触发 Combine 路径**。

因此 Case 2 在使用 32 核时 **会触发 combine**（combineNum > 0），但只有少数 S1G 行被切分（边界 core 处），大多数行被单个 core 完整处理。

#### 3.2.2 Device Kernel — 单 tile 计算流程

以某个 core 处理一个 normal tile（M=128, S2=128，该行只有 1 个 S2 块）为例，走完 1.5 节描述的 C1→V1→C2→V2 四步流水：

**Step A — BMM1（AIC Cube）**

1. **Copy Q**：Q[b=0, n2=0, gS1_start:gS1_start+128, d=0:128] → L1 NZ（128×128 bf16 = 32KB）
2. **Copy K**：K[b=0, n2=0, s2*128:s2*128+128, d=0:128] → L1 NZ（128×128 bf16 = 32KB）
3. **Matmul**：Q(128×D) × K^T(D×128)，K-split matmul（D 维度以 128 切分，此处一次即完成）。L1→L0A/L0B（double-buffer）→ Cube MM → L0C
4. **Fixpipe** L0C→UB：NZ→ND 格式转换 + dual-destination 拆为 2×64×128 fp32 → `SetCrossCore` 通知 AIV

**Step B — Vec1（AIV Vector）**

> 每个 AIC core 对应 2 个 AIV sub-core，各处理 M/2=64 rows。

1. **AttnMaskCopyIn**：从 GM（2048×2048 bool）拷贝 [0:64, s2*128:s2*128+128] 区域到 UB → 按 GS1 layout 广播 G=32
2. **ProcessVec1Vf**（no_update, S2=128, aligned128）：逐行 `QK^T × scale → mask(-inf) → row_max → exp(x-max) → sum → norm → cast bf16`
3. **DataCopy** → L1 `l1PBuffers`（64×128 bf16）+ `SetCrossCore` 通知 AIC

**Step C — BMM2（AIC Cube）**

1. 从 L1 读取 P（等待 cross-core sync），128×128 bf16
2. **Copy V** → L1 NZ，128×128 bf16
3. **MatmulFull** P(128×S2) × V(S2×128)，realM=128
4. Fixpipe L0C→UB dual-dest 2×64×128 fp32 → `SetCrossCore`

**Step D — Vec2（AIV Vector）**

1. 读取 BMM2 结果 → 因单块（isFirstS2Loop && isLastS2Loop），除以 softmax_sum 归一化
2. `RowInvalid`（causal 导致的无效行置零）→ `DivCast`（fp32→bf16）→ `CopyOutAttentionOut` 写入 GM

对于有多 S2 块的 S1G 行（后半段 core），首个 S2 块触发 no_update Softmax → 中间块触发 FlashUpdateNew 合并累加器 → 最后块触发 FlashUpdateLastNew + LastDivNew + 输出（对应 `fa_block_vec.h` 的 `ProcessVec2`）。

## 四、性能测试

以下场景在950DT平台上完成测试。

| B | Q_N | KV_N | Q_S | KV_S | D | Duration | MFU |
|---|-----|------|-----|------|---|----------|-----|
| 1 | 32 | 1 | 1024 | 16384 | 128 | 769.26 | 80.87% |
| 1 | 32 | 1 | 8192 | 16384 | 128 | 4592.747 | 83.17% |
| 4 | 32 | 1 | 1024 | 16384 | 128 | 2974.165 | 83.02% |
| 4 | 32 | 1 | 8192 | 16384 | 128 | 18310.22 | 83.33% |
