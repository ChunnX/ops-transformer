# 02 与硬件无关的实现方案 — GMM A16W4 PerGroup ND

> 本文档描述 A16W4 pergroup ND 数据流的总体设计，不依赖具体硬件实现细节。
> 相关文档：[01 交付件清单](./01_deliverables.md)、[03 Tiling 方案](./03_tiling_design.md)、[04 Kernel 方案](./04_kernel_design.md)

---

## 1. 数据流总体设计

### 1.1 计算公式

A16W4 pergroup 伪量化数据流的完整计算公式：

$$
y_i = x_i \times (weight_i + antiquant\_offset_i) \times antiquant\_scale_i + bias_i
$$

其中：
- $x_i$：FLOAT16/BFLOAT16 激活矩阵，shape 为 $(M, K)$ 或 $(m_i, k_i)$
- $weight_i$：INT4 权重矩阵，反量化前 shape 为 $(N, K)$ 或 $(K, N)$（转置）
- $antiquant\_scale_i$：pergroup 缩放因子，shape 为 $(K/gs, N)$
- $antiquant\_offset_i$：pergroup 偏移量（可选），shape 同 scale
- $bias_i$：偏置（可选），shape 为 $(N,)$
- $gs$：group_size，取值 ∈ {32, 64, 128, 256}

对称量化（$antiquant\_offset$ 为 null）时退化为：

$$
y_i = x_i \times weight_i \times antiquant\_scale_i + bias_i
$$

### 1.2 反量化模型

**PerGroup 反量化**：K 轴按 `groupSize` 分段，每段内的 weight 元素共享同一个 scale（和 offset）。

```
K 轴: |← gs →|← gs →|← gs →|...|← gs →|
       group0  group1  group2      group_{G-1}

scale shape: (G, N)  其中 G = K / gs
每个 scale[g, n] 对应 weight 中 K 方向第 g 个 group、N 方向第 n 列的所有 weight 元素
```

**反量化过程**（逐元素）：
```
weight_dequant[k, n] = (weight_int4[k, n] + offset[k/gs, n]) × scale[k/gs, n]
```

### 1.3 与 PerChannel 量化的本质差异

| 维度 | PerChannel（A16W8 现有） | PerGroup（A16W4 新增） |
|------|------------------------|----------------------|
| **反量化粒度** | 整个 K 轴共用一组 scale | K 轴按 gs 分段，每段独立 scale |
| **scale shape** | $(N,)$ — 1 维 | $(K/gs, N)$ — 2 维 |
| **scale 数据量** | $N$ 个元素 | $(K/gs) \times N$ 个元素 |
| **GM→UB 搬运** | 1 行 × N 列 | $K/gs$ 行 × N 列（2D 搬运） |
| **VF 加载方式** | `scale[nOffset]` | `scale[kOffset/gs × N + nOffset]` |
| **UB 空间需求** | 小（仅 1 行 scale） | 大（需容纳 $\lceil kL1/gs \rceil \times N$ 个 scale） |

**核心差异总结**：pergroup 在 K 方向引入了额外的分组维度，导致 scale/offset 的数据量从 $O(N)$ 增长到 $O(K/gs \times N)$，进而影响地址偏移计算、UB 空间划分和 VF 函数实现。

---

## 2. 地址偏移公式推导

### 2.1 多多多场景（groupType=-1）

多多多场景下，每个 group 的 tensor 独立存储，地址通过 `GetTensorAddr(groupIdx)` 获取，**无需计算偏移**。

```
x[groupIdx]           → GetTensorAddr<xType>(groupIdx, xGm_)
weight[groupIdx]      → GetTensorAddr<wType>(groupIdx, weightGm_)
antiquantScale[groupIdx] → GetTensorAddr<xType>(groupIdx, antiquantScaleGm_)
antiquantOffset[groupIdx] → GetTensorAddr<xType>(groupIdx, antiquantOffsetGm_)
bias[groupIdx]        → GetTensorAddr<biasType>(groupIdx, biasGm_)
y[groupIdx]           → GetTensorAddr<yType>(groupIdx, yGm_)
```

**结论**：多多多场景下，pergroup 与 perchannel 的地址获取方式完全一致，无需修改。

### 2.2 单单单场景（groupType=0）

单单单场景下，所有 group 共享同一个 tensor，通过基址 + 偏移量定位每个 group 的数据。偏移量在遍历 group 时逐步累加。

#### 2.2.1 x 偏移

x shape 为 $(M, K)$，每个 group 占用 $m_i \times k_i$ 个元素。

$$
xBaseOffset_{i+1} = xBaseOffset_i + m_i \times k_i
$$

**与 perchannel 一致，无需修改。**

#### 2.2.2 weight 偏移

weight shape 为 $(E, N, K)$（非转置）或 $(E, K, N)$（转置），每个 group 占用 $N \times K$ 个 INT4 元素。

**INT4 字节偏移特殊处理**：INT4 每字节存储 2 个元素，因此字节数为元素数的一半。

非转置（weight shape = $(E, N, K)$）：
$$
weightBaseOffset_{i+1} = weightBaseOffset_i + (N_i \times K_i) \gg 1
$$

转置（weight shape = $(E, K, N)$）：
$$
weightBaseOffset_{i+1} = weightBaseOffset_i + (K_i \times N_i) \gg 1
$$

**推导验证**：
- 元素数 = $N \times K$
- 字节数 = $N \times K / 2$（因为 INT4 每字节 2 个元素）
- 右移 1 位等价于除以 2

**与 perchannel 一致**（`>> 1` 处理已有），无需修改。

#### 2.2.3 antiquantScale/antiquantOffset 偏移（★ 核心差异）

**PerChannel（A16W8 现有）**：
- antiquantScale shape = $(E, N)$
- 每个 group 占用 $N_i$ 个元素
- 偏移公式：$antiquantParamsBaseOffset_{i+1} = antiquantParamsBaseOffset_i + N_i$

**PerGroup（A16W4 新增）**：
- antiquantScale shape = $(E, G, N)$，其中 $G = K / gs$
- 每个 group 占用 $G_i \times N_i = (K_i / gs) \times N_i$ 个元素
- 偏移公式：

$$
antiquantParamsBaseOffset_{i+1} = antiquantParamsBaseOffset_i + \frac{K_i}{gs} \times N_i
$$

**推导验证**：

以 group $i$ 为例，antiquantScale 的第 $i$ 个 group 的 shape 为 $(G_i, N_i)$：
- $G_i = K_i / gs$（K 方向 group 数）
- 元素总数 = $G_i \times N_i = (K_i / gs) \times N_i$
- dtype 为 FLOAT16/BFLOAT16（2 字节/元素），但偏移以元素数计量（`GetTensorAddr<xType>` 内部处理字节转换）

**数值验证**：
- 假设 $K=512, gs=128, N=256$
- $G = 512/128 = 4$
- 每个 group 的 scale 元素数 = $4 \times 256 = 1024$
- 对比 perchannel：每个 group 的 scale 元素数 = $256$
- pergroup 是 perchannel 的 $K/gs = 4$ 倍

**antiquantOffset 偏移公式与 antiquantScale 完全一致**（shape 相同）。

#### 2.2.4 bias 偏移

bias shape 为 $(E, N)$（单单单），每个 group 占用 $N_i$ 个元素。

$$
biasBaseOffset_{i+1} = biasBaseOffset_i + N_i
$$

**与 perchannel 一致，无需修改。** 注意：bias 的偏移与 antiquantScale 的偏移使用不同的累加变量，在现有代码中 bias 偏移复用了 `antiquantParamsBaseOffset`，需要确认 pergroup 场景下 bias 偏移是否仍与 antiquantScale 偏移绑定。

**⚠️ 关键发现**：现有代码中 `SetGmAddr()` 的 bias 地址计算为：
```cpp
biasGm = GetTensorAddr<biasType>(0, biasGm_) + antiquantParamsBaseOffset;
```
这意味着 bias 偏移与 antiquantScale 偏移共用同一个累加变量。在 pergroup 场景下，bias shape 为 $(E, N)$ 而 antiquantScale shape 为 $(E, G, N)$，两者偏移量不同，**需要分离 bias 偏移的计算**。

**修正方案**：为 bias 新增独立的偏移变量 `biasBaseOffset`：
$$
biasBaseOffset_{i+1} = biasBaseOffset_i + N_i
$$

#### 2.2.5 y 偏移

y shape 为 $(M, N)$，每个 group 占用 $m_i \times n_i$ 个元素。

$$
yBaseOffset_{i+1} = yBaseOffset_i + m_i \times n_i
$$

**与 perchannel 一致，无需修改。**

### 2.3 地址偏移公式汇总

| Tensor | 多多多（singleWeight=0） | 单单单 PerChannel | 单单单 PerGroup（★ 新增） |
|--------|:------------------------:|:-----------------:|:-------------------------:|
| x | `GetTensorAddr(i)` | `base + Σ(m_i × k_i)` | 同 perchannel |
| weight | `GetTensorAddr(i)` | `base + Σ(n_i × k_i) >> 1` | 同 perchannel |
| y | `GetTensorAddr(i)` | `base + Σ(m_i × n_i)` | 同 perchannel |
| bias | `GetTensorAddr(i)` | `base + Σ(n_i)` | `base + Σ(n_i)`（独立偏移） |
| antiquantScale | `GetTensorAddr(i)` | `base + Σ(n_i)` | **`base + Σ(k_i/gs × n_i)`** |
| antiquantOffset | `GetTensorAddr(i)` | `base + Σ(n_i)` | **`base + Σ(k_i/gs × n_i)`** |

---

## 3. 兼容性策略

### 3.1 不破坏已有 A2/A3 A16W4 功能

**隔离机制**：A2/A3 和 Ascend 950 使用不同的 tiling 类和 kernel 入口路径。

| 平台 | Tiling 类 | Kernel 入口 |
|------|-----------|-------------|
| A2/A3 | `GroupedWeightQuantBatchMatmulTiling`（arch30 路径） | arch30 kernel |
| Ascend 950 | `GroupedWeightQuantBatchMatmulTiling`（arch35 路径） | arch35 kernel（`GMMWeightQuantBasicController`） |

**策略**：
- 所有 pergroup 修改均在 arch35 路径下进行
- A2/A3 的 arch30 路径完全不受影响
- `grouped_matmul_def.cpp` 中 config950 的新增条目不影响 A2/A3 的 config

### 3.2 不破坏已有 Ascend 950 A16W8 PerChannel 功能

**隔离机制**：通过编译期 `if constexpr` 分支隔离 perchannel 和 pergroup 路径。

```cpp
if constexpr (wqmmConfig.antiQuantType == QuantType::PER_GROUP) {
    // pergroup 路径（新增）
} else {
    // perchannel 路径（现有，不修改）
}
```

**关键保护点**：

| 修改点 | 保护措施 |
|--------|----------|
| `SetGmAddr()` 偏移公式 | `if constexpr` 分支，perchannel 路径不变 |
| `SetTilingKey()` antiquantType | perchannel 仍设为 `PER_CHANNEL`，pergroup 设为 `PER_GROUP` |
| `GetBufferConfig()` | perchannel 仍走 `GetNdBufferInfo()`，pergroup 走新增 `GetNdPerGroupBufferInfo()` |
| VF 函数 | perchannel 仍走 `AntiQuantInt4NdNk/Kn`，pergroup 走新增 `AntiQuantInt4NdNkPerGroup/KnPerGroup` |
| tiling key | 新增组合不影响现有组合 |

### 3.3 代码分支隔离策略

**编译期 `if constexpr`**（首选）：
- `WqmmConfig.antiQuantType` 为编译期常量，`if constexpr` 可在编译期消除未使用分支
- 优势：零运行时开销，编译器可优化死代码
- 适用场景：Controller 层地址偏移、VF 函数分发、buffer 配置选择

**运行时分支**（辅助）：
- `groupSize` 为运行时值（32/64/128/256），VF 函数内部按 groupSize 分支
- 适用场景：NK 排布的 pergroup VF 函数内部（参照 wqbmmv2 的实现）

---

## 4. 测试方案

### 4.1 Tiling UT 用例设计

#### 4.1.1 正常场景

| 用例编号 | 场景 | groupType | groupSize | 转置 | 验证点 |
|:--------:|------|:---------:|:---------:|:----:|--------|
| T-01 | 多多多 + gs=32 | -1 | 32 | 否 | tiling 参数正确性 |
| T-02 | 多多多 + gs=64 | -1 | 64 | 否 | tiling 参数正确性 |
| T-03 | 多多多 + gs=128 | -1 | 128 | 否 | tiling 参数正确性 |
| T-04 | 多多多 + gs=256 | -1 | 256 | 否 | tiling 参数正确性 |
| T-05 | 单单单 + gs=128 | 0 | 128 | 否 | tiling 参数正确性 |
| T-06 | 单单单 + gs=128 + 转置 | 0 | 128 | 是 | tiling 参数正确性 |
| T-07 | 多多多 + 转置 | -1 | 128 | 是 | tiling 参数正确性 |
| T-08 | 有 bias | -1 | 128 | 否 | baseM 调整为 240 |
| T-09 | 有 antiquantOffset | -1 | 128 | 否 | offset 参数传递 |
| T-10 | groupListType=0 | 0 | 128 | 否 | cumsum 模式 |
| T-11 | groupListType=1 | 0 | 128 | 否 | count 模式 |

#### 4.1.2 异常拦截场景

| 用例编号 | 异常类型 | 输入 | 预期结果 |
|:--------:|----------|------|----------|
| E-01 | groupSize 非法 | gs=48 | 拦截报错 |
| E-02 | K % groupSize != 0 | K=100, gs=32 | 拦截报错 |
| E-03 | antiquantScale shape 不匹配 | scale shape=(E, G+1, N) | 拦截报错 |
| E-04 | 多多多 pergroup 长度不一致 | s_1=128, s_2=64 | 拦截报错 |
| E-05 | pergroup 转置 s_i 非偶数 | s_i=33, 转置 | 拦截报错 |
| E-06 | 非对称量化 + pergroup | antiquantOffset 非 null + pergroup | 拦截报错（本场景仅支持对称量化 pergroup） |
| E-07 | weight 最后一维非偶数 | N=3 | 拦截报错 |
| E-08 | antiquantScale dtype 不匹配 | x=FP16, scale=BF16 | 拦截报错 |

### 4.2 Kernel UT 用例设计

#### 4.2.1 功能验证

| 用例编号 | x dtype | groupSize | groupType | 转置 | bias | offset | shape |
|:--------:|:-------:|:---------:|:---------:|:----:|:----:|:------:|-------|
| K-01 | FP16 | 128 | -1 | 否 | 否 | 否 | M=256, K=512, N=256 |
| K-02 | FP16 | 128 | -1 | 否 | 是 | 否 | M=256, K=512, N=256 |
| K-03 | BF16 | 128 | -1 | 否 | 否 | 否 | M=256, K=512, N=256 |
| K-04 | FP16 | 32 | -1 | 否 | 否 | 否 | M=256, K=512, N=256 |
| K-05 | FP16 | 64 | -1 | 否 | 否 | 否 | M=256, K=512, N=256 |
| K-06 | FP16 | 256 | -1 | 否 | 否 | 否 | M=256, K=512, N=256 |
| K-07 | FP16 | 128 | 0 | 否 | 否 | 否 | M=256, K=512, N=256, E=4 |
| K-08 | FP16 | 128 | -1 | 是 | 否 | 否 | M=256, K=512, N=256 |
| K-09 | FP16 | 128 | 0 | 是 | 否 | 否 | M=256, K=512, N=256, E=4 |

#### 4.2.2 边界值验证

| 用例编号 | 场景 | shape | 验证点 |
|:--------:|------|-------|--------|
| KB-01 | 极小 M | M=1, K=128, N=256 | 尾块处理 |
| KB-02 | 极小 N | M=256, K=128, N=1 | 尾块处理 |
| KB-03 | 最小 K | M=256, K=gs, N=256 | 单 group |
| KB-04 | 单 expert | E=1 | 单单单退化 |
| KB-05 | 最大 expert | E=1024 | 单单单极限 |
| KB-06 | 多 expert 不等 M | m_1=1, m_2=256 | 多多多不均 |

### 4.3 兼容性验证方案

| 验证项 | 方法 | 通过标准 |
|--------|------|----------|
| A2/A3 A16W4 回归 | 运行 A2/A3 已有 UT | 全部通过 |
| Ascend 950 A16W8 perchannel 回归 | 运行 950 已有 UT | 全部通过 |
| Ascend 950 MX 回归 | 运行 MX 已有 UT | 全部通过 |
| Ascend 950 S8S4 回归 | 运行 S8S4 已有 UT | 全部通过 |
| 新增 A16W4 pergroup 功能 | 运行新增 UT | 全部通过 |
