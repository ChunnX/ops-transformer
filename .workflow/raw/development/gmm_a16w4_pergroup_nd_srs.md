# GMM A16W4 PerGroup ND — SRS 算子实现方案

> 本文档从设计和逻辑链路上说明 GMM A16W4 weight ND pergroup 数据流的实现方案。
> 面向读者：理解算子需求和实现方案，非编码级详细设计。

---

## §1. 需求概述

### 1.1 数据流说明

**A16W4 INT4 weight ND pergroup 伪量化**：激活（x）为 FP16/BF16（16-bit），权重（weight）为 INT4，通过 pergroup 反量化后参与矩阵乘计算。weight 数据格式为 ND（非 NZ），支持转置和非转置。

### 1.2 计算公式

$$
y_i = x_i \times (weight_i + antiquant\_offset_i) \times antiquant\_scale_i + bias_i
$$

对称量化（antiquantOffset 为 null）时退化为：

$$
y_i = x_i \times weight_i \times antiquant\_scale_i + bias_i
$$

### 1.3 PerGroup 反量化模型

K 轴按 `groupSize` 分段，每段内的 weight 元素共享同一个 scale（和 offset）：

```
K 轴: |← gs →|← gs →|← gs →|...|← gs →|
       group0  group1  group2      group_{G-1}

scale shape: (G, N)  其中 G = K / gs
weight_dequant[k, n] = (weight_int4[k, n] + offset[k/gs, n]) × scale[k/gs, n]
```

### 1.4 与 A2/A3 已有场景的关系

本场景是对 A2/A3 已有 A16W4 伪量化数据流的**兼容性扩展**：
- A2/A3 已支持 A16W4 perchannel（整个 K 轴共用一组 scale）
- Ascend 950 新增 pergroup 模式（K 轴按 gs 分段，每段独立 scale）
- 所有修改在 arch35 路径下进行，A2/A3 的 arch30 路径完全不受影响

### 1.5 关键约束

| 约束项 | 条件 |
|--------|------|
| group_size | ∈ {32, 64, 128, 256} |
| K % group_size | == 0 |
| weight 格式 | ND（非 NZ） |
| weight 最后一维 | 偶数 |
| x 转置 | 不支持（固定不转置） |
| 输入模式 | 单单单（groupType=0）/ 多多多（groupType=-1） |

---

## §2. Tiling 切分策略

![tiling切分策略](svg/tiling_split_strategy.svg)

### 2.1 切分常量

| 常量 | 值 | 说明 |
|------|:--:|------|
| baseM | 256（无 bias）/ 240（有 bias） | M 方向基本块大小 |
| baseN | 256 | N 方向基本块大小 |
| baseK | 64 | K 方向基本块大小 |
| baseN_min | 128 | N 方向最小块（动态调整下限） |
| stepK | 4（默认）/ 3（groupSize=192） | K 方向步进系数 |
| depth | 8 | 流水线深度 |
| bufferNum | 2 | 双缓冲 |

### 2.2 Host 侧 SetTiling() 流程

```
SetTiling(context):
  ① AnalyzeAttr(context)       — 解析属性、校验输入、设置 shapeList
  ② CalcResplitTiling(context) — 计算尾块重切分参数
  ③ SetBaseTiling()            — 填充 GMMWeightQuantParam 结构
  ④ SetMatMulTiling()          — 填充 TCubeTiling 结构（baseM/N/K 等）
  ⑤ SetTilingKey(context)      — 生成 tiling key → 选择 kernel 模板
  ⑥ SetCustomParam(context)    — 设置 blockDim=coreNum，拷贝 tiling 数据到 GM
```

### 2.3 Kernel 侧分核算法

每个 AI Core 上执行 Controller `Process()`，核心逻辑：

**① 动态调整 baseM**（根据 M 大小选择切分粒度）：

| 条件 | actualBaseM | 含义 |
|------|:-----------:|------|
| M ≤ baseM 或 M ≥ 4×baseM | 256 | 不调整 |
| M ≤ 2×baseM | ⌈M/2⌉ | 二分 |
| 2×baseM < M < 4×baseM | ⌈M/4⌉ | 四分 |

**② 动态调整 baseN**（根据 N 是否够分满核）：

| 条件 | actualBaseN | 含义 |
|------|:-----------:|------|
| N ≥ baseN × coreNum | 256 | N 足够大，标准块 |
| N < baseN × coreNum | 128 | N 较小，缩小块以分满核 |

**③ M/N 二维切分 + Round-Robin 分核**：

```
mBlockNum = ⌈M / actualBaseM⌉
nBlockNum = ⌈N / actualBaseN⌉
totalBlocks = mBlockNum × nBlockNum

// 每个核按步长 coreNum 领取 (mBlock, nBlock) 任务
curBlock = myCoreIdx
while curBlock < totalBlocks:
    mOffset = (curBlock / nBlockNum) × actualBaseM
    nOffset = (curBlock % nBlockNum) × actualBaseN
    ComputeBasicBlock(mOffset, nOffset)   // K 方向在内部按 kbL1Size 步进
    curBlock += coreNum
```

**关键结论**：M 和 N 方向都会切分，K 方向不切分（在 BasicBlock 内部按 kbL1Size 步进遍历）。各核以 round-robin 方式领取任务，保证负载均衡。

### 2.3.1 kL1=256 的选择原因

kL1（即 kbL1Size，L1 层 K 方向 tile 大小）选择 256 的原因是 group_size 需要支持 32/64/128/256，kL1=256 可以保证在这几种 group_size 下，每个 tile 的 K 方向都不会跨越 group 边界，从而简化代码实现。具体来说：

| group_size | 一个 tile 包含的完整 group 数 | 计算 |
|:----------:|:---------------------------:|:----:|
| 32 | 8 | 256 / 32 = 8 |
| 64 | 4 | 256 / 64 = 4 |
| 128 | 2 | 256 / 128 = 2 |
| 256 | 1 | 256 / 256 = 1 |

kL1 = baseK × stepKb × depth = 64 × 4 / depth × depth = 256。每个 L1 tile 的 K 方向恰好包含整数个 group，避免了反量化时跨 group 边界的复杂处理逻辑。

### 2.4 尾块重切分

**使能条件**：

```
EnableTailResplit():
  多多多 → 不使能
  单单单 + weight ND + B 非转置 → 不使能
  其余 → 使能
```

**A16W4 pergroup ND 非转置场景**：不使能尾块重切分，走 Basic Controller。

**使能时的三分支**：

| 分支 | 条件 | 策略 |
|------|------|------|
| 1 | N % (coreNum × 256) == 0 | mainBlockSize=256, mainBlockCount=N/(coreNum×256) |
| 2 | N > coreNum × 128 | 可分满核，调整 mainBlockSize |
| 3 | N ≤ coreNum × 128 | 可能无法分满核，缩小块大小 |

### 2.5 单单单 vs 多多多差异

| 维度 | 单单单（groupType=0） | 多多多（groupType=-1） |
|------|:---------------------:|:---------------------:|
| tensor 数量 | 各 1 个 | 各多个 |
| M 来源 | groupList 差值 | mList[i] |
| K 来源 | kList[0]（共享） | kList[i]（独立） |
| N 来源 | nList[0]（共享） | nList[i]（独立） |
| groupNum | weight.shape[0] | tensor 个数 |
| 尾块重切分 | 可使能（NZ/转置时） | 不使能 |

### 2.6 A16W4 PerGroup 适配点

**用户确认的策略**：x 和 weight 都按 256×256 切分。多多多每个核分配一个完整 expert；单单单 K 和 N 方向不切分，M 方向切分成多个块。

**当前 A16W8 代码行为**：M 和 N 方向都会切分（动态 baseM/baseN），多多多也是 round-robin 分块。

**⚠️ 差异**：用户描述的"K 和 N 方向不切分"与当前代码的 N 方向动态切分存在差异。A16W4 pergroup 可能需要修改 Controller 分核逻辑。

### 2.7 Tiling Key 选择逻辑

Tiling key 由以下参数组合决定，用于在编译期选择对应的 kernel 模板：

| 参数 | 取值 | 说明 |
|------|------|------|
| W_TYPE | WQGMM_ND | weight ND 格式 |
| W_QUANT_TYPE | **WQGMM_PER_GROUP** | pergroup 量化（区别于 PER_CHANNEL） |
| OFFSET_OR_BIAS_EXIT | 4 种组合 | 有/无 offset × 有/无 bias |
| WQ_B_TRANS | NO_TRANS / TRANS | weight 非转置/转置 |

共 8 种 tiling key 组合（4 种 offset/bias × 2 种转置）。

### 2.8 PerGroup groupSize 传递方式

```
Tiling 阶段:  antiquantScale shape → 推导 groupSize = K / G
              ↓ 写入 GMMWeightQuantParam.groupSize
Kernel 阶段:  gmmBaseTiling_->groupSize → wqmmBasicBlock_.Init(groupSize)
              ↓ 传递到 VF 函数
VF 函数:      按 groupSize 运行时分支（32/64/128/256）执行对应反量化逻辑
```

---

## §3. Buffer 空间分配

![buffer空间分配](svg/buffer_allocation.svg)

### 3.1 L1 Buffer 分配（512KB，Cube Core 使用）

L1 总大小 512KB，由 Cube Core 使用，存储反量化后的 weight（FP16）、A 矩阵和 Bias。所有区域均为**双缓冲**（P0/P1 交替），通过 `(loopIdx & 1) × dbOffset` 切换。

**计算公式**：

```
weightL1Space = baseN × stepKb × baseK × sizeof(xType)    // 单块 weight
biasL1Space   = 4KB（2048 half 元素）                       // 单块 bias（有 bias 时）
aL1Space      = (512KB - 2×(weightL1Space + biasL1Space)) / 2   // 有 bias
aL1Space      = 256KB - weightL1Space                            // 无 bias
```

**kL1=256, baseN=256, baseK=64, stepKb=4, xType=half 时的具体数值**：

| 区域 | 有 Bias | 无 Bias | 说明 |
|------|:-------:|:-------:|------|
| WeightL1_P0 | 128KB（65536 half） | 128KB | 反量化后 FP16 weight |
| Bias_P0 | 4KB（2048 half） | — | Bias 数据 |
| AL1_P0 | 124KB（63488 half） | 128KB（65536 half） | A 矩阵 |
| AL1_P1 | 124KB | 128KB | A 矩阵双缓冲 |
| Bias_P1 | 4KB | — | Bias 双缓冲 |
| WeightL1_P1 | 128KB | 128KB | Weight 双缓冲 |
| **总计** | **512KB** | **512KB** | |

**双缓冲机制**：P0 计算时 P1 预取下一块数据，隐藏搬运延迟。Weight/Bias/A 均为双缓冲。

**⚠️ 遗留问题：有 bias 场景 L1 空间不足**

当 kL1=256, nL1=256, mL1=256 时：
- WeightL1 双缓冲 = 128KB × 2 = 256KB
- AL1 双缓冲（无 bias）= 128KB × 2 = 256KB
- 总计 = 512KB，恰好满载

有 bias 时需要额外 4KB × 2 = 8KB 给 BiasL1，导致 AL1 每缓冲缩减到 124KB，但 A 矩阵（NZ 格式）需要 128KB，**差 4KB**。

这是 L1 物理空间 512KB 的硬限制，暂记为遗留问题，后续评估解决方案（如减小有 bias 时的 kL1、调整 BIAS_L1_SIZE、或 A 矩阵采用单缓冲等）。

### 3.2 UB Buffer 分配（248KB 可用，Vector Core 使用）

UB 总大小 256KB，实际可用约 248KB。由 Vector Core 使用，存储 INT4 原始 weight、反量化中间结果和 scale/offset。**UB buffer 大小是静态固定的**，不随 baseM/baseN/baseK 变化。

**UB Buffer 配置（kL1=256）**：

| Buffer | 总大小 | 缓冲数 | 单缓冲大小 | 用途 |
|--------|:------:|:------:|:---------:|------|
| weightInputLowBit | 174KB | 2 | 87KB | INT4 原始 weight 数据（实际每次 MTE2 仅用 32KB，INT4 packed） |
| highBitData | 66KB | 2 | 33KB | VF 分批输出反量化后 FP16 weight（每次 32KB） |
| antiQuantScale | 4KB | 2 | 2KB | pergroup scale（gs=32 时恰好够） |
| antiQuantOffset | 4KB | 2 | 2KB | pergroup offset |
| **总计** | **248KB** | | | |

**双缓冲机制**：MTE2 输入双缓冲（ubMte2BufferNum=2），MTE3 输出 ND 双缓冲/NZ 四缓冲。AIV-AIC 跨核通过 SetFlag/WaitFlag 同步。

**说明**：weightInputLowBit 总分配 174KB，但 INT4 packed 后每次 MTE2 实际仅搬运 32KB（nL1=256, kL1=256, INT4 = 256×256/2 = 32KB）。highBitData 为 VF 分批输出，每次输出约 32KB。

### 3.3 Buffer 与基本块大小的关系

| Buffer | 大小决定方式 | 与 tiling 参数的关系 |
|--------|-------------|---------------------|
| WeightL1 单块 | baseN × stepKb × baseK × sizeof(xType) | 与 baseN、baseK 成正比 |
| AL1 单块 | L1 剩余空间 / 2 | 与 baseN、baseK 成**反比**（WeightL1 越大，AL1 越小） |
| UB weightInputLowBit | 固定 174KB | 实际使用量 = nL1Size × mte2RealK × sizeof(wType) |
| UB highBitData | 固定 66KB | 实际使用量 = vfNLen × vfKLen × sizeof(xType) |

tiling 策略需确保每次搬运/计算的数据量不超过 buffer 容量。

### 3.4 PerChannel vs PerGroup 的 antiQuantScale 差异

**PerChannel**：每次搬运 1 行 `[1, nL1Size]`，4KB 足够。

**PerGroup**：每次搬运 `ceil(ubMte2KSize / groupSize)` 行，空间需求与 groupSize 和搬运粒度相关：

| groupSize | ubMte2KSize | 行数 | 单缓冲 | 双缓冲总 | 4KB 够? |
|:---------:|:-----------:|:----:|:------:|:-------:|:-------:|
| 128 | 64 (kbL1Size) | 1 | 256B | 512B | ✓ |
| 64 | 64 | 1 | 256B | 512B | ✓ |
| 32 | 64 | 2 | 512B | 1KB | ✓ |
| 128 | 512 (NdNk) | 4 | 1KB | 2KB | ✓ |
| 64 | 512 | 8 | 2KB | 4KB | ✓ 刚好 |
| 32 | 512 | 16 | 4KB | 8KB | ✗ 不够 |

**结论**：NdKn 场景（ubMte2KSize = kbL1Size = baseK × stepKb）下 4KB 通常够用。NdNk 场景（ubMte2KSize = 512）下 groupSize ≤ 32 时 4KB 不够，需增大到 8KB。

### 3.5 A16W4 PerGroup 的 Buffer 适配

现有 `GetNdBufferInfo()` 仅支持 perchannel。A16W4 pergroup 需新建 `GetNdPerGroupBufferInfo()`，主要调整：

1. **antiQuantScale**：kL1=256, gs=32 时，scale 元素数 = ⌈256/32⌉ × 256 = 2048 half = 4KB，恰好达到当前 4KB 上限。若后续需更大余量可增大到 8KB（同时缩减 highBitData 到 62KB）
2. **antiQuantOffset**：与 scale 对称，4KB 在 kL1=256 时恰好够
3. **weightInputLowBit**：INT4 数据量是 INT8 的一半，174KB 分配充足（实际每次 MTE2 仅用 32KB）
4. **highBitData**：保持不变（反量化后 FP16 数据量与量化类型无关）

---

## §4. 计算流程

### 4.1 主循环伪代码

```
Controller.Process():
  for groupIdx in 0..groupNum:
    ① SetMKN(groupIdx)          // 从 mList/kList/nList 取当前 group 的 M/K/N
    ② SetGmAddr(groupIdx)       // 计算各 tensor 的 GM 地址偏移
                                // ★ pergroup: scale 偏移 += (kSize/gs) × nSize
                                // ★ bias 偏移独立: += nSize
    ③ 动态调整 baseM/baseN      // 根据实际 M/N 大小调整切分块

    for each BasicBlock (mL1, nL1):
      ┌──────── AIV (Vector Core) ────────┐  ┌──── AIC (Cube Core) ────┐
      │                                    │  │                          │
      │ for kBlock in K 方向:              │  │ for kBlock:              │
      │   ④ MTE2: weight(INT4) GM → UB    │  │   ⑥ Wait AIV            │
      │   ④ MTE2: scale/offset GM → UB    │  │   ⑦ MTE1: A/Bias GM→L1  │
      │   ⑤ VF: INT4 unpack → FP16        │  │   ⑧ Matmul: C += A × W  │
      │      + offset (可选)               │  │                          │
      │      × scale (pergroup)            │  │   ⑨ GetTensorC: C → GM  │
      │   ⑤ MTE3: W_dequant UB → L1       │  │                          │
      │   SetAivToAic() ──────────────────→│  │   SetAicToAiv()         │
      └────────────────────────────────────┘  └──────────────────────────┘

  End() — 写回所有 C 到 GM
```

### 4.2 PerGroup 反量化核心计算

**NK 排布**（B 转置，weight shape = (K, N)，N 为内轴）：

```
FOR groupIdx = 0 TO groupNum-1:
  FOR nBubIdx = 0 TO ubLoopN-1:
    scale = LoadBroadcast(scaleBase[groupIdx + nBubIdx × stride])  // 广播加载
    offset = LoadBroadcast(offsetBase[groupIdx + nBubIdx × stride])
    weightB4 = LoadUnpack4(weightInBase[...])                       // INT4 unpack
    weightB16 = Cast(weightB4)                                      // INT4 → FP16
    IF hasOffset: weightB16 += offset
    weightB16 *= scale                                              // 反量化
    StoreBlockCopy(weightOutBase, weightB16)
```

按 groupSize 运行时分支（32/64/128/256），每种 gs 对应专用 VF 逻辑。

**KN 排布**（B 非转置，weight shape = (N, K)，K 为内轴）：

```
FOR nIdx = 0 TO nLoop-1:
  FOR gIdx = 0 TO groupNum-1:
    scale = LoadNorm(scaleBase[nIdx × 128 + gIdx × stride])        // 直接加载
    FOR kIdx = 0 TO groupSize-1:                                    // group 内逐 K 行
      weightIn = LoadUnpack4(weightInBase[...])
      weightOut = Cast(weightIn)
      IF hasOffset: weightOut += offset
      weightOut *= scale
      StoreBlockCopy(weightOutBase, weightOut)
```

KN 排布统一处理所有 groupSize，无需按 gs 值分支。

### 4.3 A16W4 VF 反量化架构与 A16W8 的关键差异

A16W4 的 VF 反量化实现参考 wqbmmv2（`weight_quant_batch_matmul_v2_reg_base_common.h`），而非 A16W8 的 `weight_quant_vec_compute.h`。两者的架构差异如下：

| 维度 | A16W8（当前 GMM 实现） | A16W4（参考 wqbmmv2） |
|------|----------------------|---------------------|
| 参考代码 | `weight_quant_vec_compute.h` | `weight_quant_batch_matmul_v2_reg_base_common.h` |
| 反量化函数 | 内嵌在 GMM 的 VF 框架中 | `AntiquantComputeKNPerGroup` / `AntiquantComputeW4NKPerGroup` |
| 处理方式 | 先按 vfNStandardLen×vfKStandardLen 切分，再对每个子块反量化 | 一次处理整个 bubKLen×bubNLen 数据块 |
| VF_CALL 次数 | 每个 bubble 多次 | 每个 bubble 仅 1 次 |
| 切分责任 | 调用者 + VF 函数内部双重切分 | 仅调用者按 bubble 切分 |

**迁移策略**：保留 GMM 的 VF 分批框架（vfNStandardLen × vfKStandardLen 循环），在循环体内替换为 wqbmmv2 的 pergroup 反量化函数。每次 VF_CALL 处理一个 vfNStandardLen × vfKStandardLen 的子块。

**wqbmmv2 反量化函数内部循环结构**：
- **KN 排布**（`AntiquantPerGroupKN`）：nLoop → groupNum → groupSize（N→group→K 三层循环）
- **NK 排布**（`AntiquantComputeW4NKPerGroup`）：按 groupSize 分支，gs=128 时为 groupNum → bubNLen（group→N 两层循环）

---

## §5. 软件算法到硬件映射

### 5.1 Ascend 950 硬件架构概述

- **32 个 AI Core**，每个 AI Core 包含：
  - **1 个 Cube Core**：负责矩阵乘计算，local memory 包括 L1（512KB）、BiasTable、L0A/L0B、L0C
  - **2 个 Vector Core**：负责向量计算（反量化等），local memory 为 UB（256KB，实际可用约 248KB）

### 5.2 数据流在硬件上的执行过程

![dataflow](svg/dataflow.svg)

### 5.3 软件操作与硬件单元映射

| 软件操作 | 硬件单元 | Local Memory | 数据搬运 |
|----------|----------|--------------|----------|
| ④ MTE2: weight/scale/offset GM→UB | Vector Core | UB | MTE2 |
| ⑤ VF: INT4 unpack + 反量化 | Vector Core | UB | VF 指令 |
| ⑤ MTE3: 反量化后 weight UB→L1 | Vector Core → Cube Core | L1 | MTE3 |
| ⑦ MTE1: A/Bias GM→L1 | Cube Core | L1 / BiasTable | MTE1 |
| ⑧ Matmul: A × W_dequant | Cube Core | L0A × L0B → L0C | Cube 指令 |
| BiasAdd: C + bias | Cube Core (Fixpipe) | L0C + BiasTable | Fixpipe |
| ⑨ GetTensorC: C L1→GM | Cube Core | L1 → GM | MTE1 |

---

## §6. 地址偏移方案

### 6.1 PerChannel vs PerGroup 偏移对比

| Tensor | PerChannel 偏移 | PerGroup 偏移 | 差异 |
|--------|:---------------:|:-------------:|:----:|
| x | `+= m_i × k_i` | 同 perchannel | 无 |
| weight | `+= (n_i × k_i) >> 1` | 同 perchannel | 无 |
| y | `+= m_i × n_i` | 同 perchannel | 无 |
| bias | `+= n_i` | `+= n_i`（**独立偏移变量**） | ★ 分离 |
| antiquantScale | `+= n_i` | **`+= (k_i/gs) × n_i`** | ★ 核心差异 |
| antiquantOffset | `+= n_i` | **`+= (k_i/gs) × n_i`** | ★ 核心差异 |

### 6.2 核心差异说明

**PerChannel**：scale shape = (E, N)，每个 group 占用 N 个元素。

**PerGroup**：scale shape = (E, G, N)，其中 G = K/gs，每个 group 占用 G × N = (K/gs) × N 个元素。

```
示例：K=512, gs=128, N=256
  PerChannel: 每 group scale 元素数 = 256
  PerGroup:   每 group scale 元素数 = (512/128) × 256 = 4 × 256 = 1024（4 倍）
```

### 6.3 Bias 偏移分离设计

现有代码中 bias 地址与 antiquantScale 共用同一个偏移变量 `antiquantParamsBaseOffset`。在 pergroup 场景下两者偏移量不同（bias 为 N，scale 为 (K/gs)×N），**需要为 bias 新增独立的 `biasBaseOffset` 变量**。

### 6.4 INT4 weight 的 >>1 字节偏移

INT4 每字节存储 2 个元素，因此 weight 字节偏移 = 元素数 >> 1。此处理在 perchannel 和 pergroup 下一致，无需修改。

---

## §7. A2/A3 → Ascend 950 兼容性关键修改点

### 7.1 关键修改点汇总

| 修改点 | A2/A3 实现 | Ascend 950 适配 | 修改文件 |
|--------|-----------|----------------|----------|
| 判断函数 | 无 pergroup ND 判断 | 新增 `IsA16W4NDPergroup()` | tiling.cpp |
| groupSize 校验 | 仅 S8S4 校验 | 新增 A16W4 gs∈{32,64,128,256} 校验 | tiling.cpp |
| antiquantScale 维度 | 2 维 (E,N) | 允许 3 维 (E,G,N) 或 2 维 (G_i,N_i) | tiling.cpp |
| tiling key antiquantType | PER_CHANNEL | 新增 PER_GROUP 组合（8 个） | tiling_key.h |
| kernel 入口分支 | 无 pergroup ND 分支 | 新增 8 个 if constexpr 分支 | apt.cpp |
| WqmmConfig | 无 ND pergroup 配置 | 新增 4 个实例 | config.h |
| Buffer 配置 | scale UB=4KB | 新增 `GetNdPerGroupBufferInfo()` scale=8KB | config.h |
| VF 反量化函数 | 仅 perchannel ND | 新增 `AntiQuantInt4NdNk/KnPerGroup` | vf_nd.h |
| SetGmAddr 偏移 | `+= nSize` | pergroup: `+= (kSize/gs) × nSize` | controller.h |
| bias 偏移 | 与 scale 共用变量 | 新增独立 `biasBaseOffset` | controller.h |
| 算子定义 config950 | 无 INT4 ND | 新增 weight=INT4+ND 组合 | def.cpp |

### 7.2 代码分支隔离策略

**编译期 `if constexpr`**（首选）：
- `WqmmConfig.antiQuantType` 为编译期常量
- 未使用分支在编译期被消除，零运行时开销
- 适用：Controller 地址偏移、VF 函数分发、buffer 配置选择

**运行时分支**（辅助）：
- `groupSize` 为运行时值（32/64/128/256）
- 适用：NK 排布 VF 函数内部按 groupSize 分发

### 7.3 不影响已有功能的关键保护措施

| 保护点 | 措施 |
|--------|------|
| perchannel 路径 | `if constexpr` 分支隔离，perchannel 代码路径完全不变 |
| tiling key | 新增组合不影响现有序号 |
| A2/A3 平台 | 所有修改在 arch35 路径下，arch30 不受影响 |
| Cube 计算 | 接收反量化后 FP16 数据，与反量化类型无关，完全复用 |
| 主循环框架 | Process()/SetMKN()/同步机制 均通用，无需修改 |

---

## §8. 交付件概览

### 8.1 需要修改的文件

| 序号 | 文件 | 修改量级 | 说明 |
|:----:|------|:--------:|------|
| 1 | `grouped_weight_quant_batch_matmul_tiling.h` | 小 | 新增辅助函数声明 |
| 2 | `grouped_weight_quant_batch_matmul_tiling.cpp` | 中 | pergroup 校验/计算分支 |
| 3 | `weight_quant_tiling_key.h` | 小 | 新增 8 个 tiling key 组合 |
| 4 | `grouped_matmul_def.cpp` | 小~中 | config950 新增 INT4 ND 组合 |
| 5 | `grouped_matmul_apt.cpp` | 中 | 新增 8 个 if constexpr 分支 |
| 6 | `basic_block_config.h` | 小 | 新增 WqmmConfig 实例 + buffer 配置 |
| 7 | `basic_block_vf_nd.h` | **大** | 新增 pergroup VF 函数（核心新增代码） |
| 8 | `weight_quant_vec_compute.h` | 中 | AntiQuantProcessNd pergroup 分发 |
| 9 | `grouped_matmul_weight_quant_basic_controller.h` | 小 | SetGmAddr pergroup 偏移 |
| 10 | tiling UT 文件 | 中 | 新增 tiling UT 用例 |
| 11 | kernel UT 文件 | 中 | 新增 kernel UT 用例 |

### 8.2 无需修改的文件（确认可复用）

| 文件 | 说明 |
|------|------|
| `grouped_matmul_tiling_data_apt.h` | GMMWeightQuantParam 已有 groupSize 字段 |
| `weight_quant_basic_block.h` | BasicBlock 分发逻辑通用 |
| `weight_quant_cube_compute.h` | Cube 计算流程通用 |
| `grouped_matmul_weight_quant_resplit_controller.h` | Resplit 路径不涉及 |
