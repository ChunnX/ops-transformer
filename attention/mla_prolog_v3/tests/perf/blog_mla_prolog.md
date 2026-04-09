# 从0到1构建 MlaProlog 融合算子

本文以 Ascend NPU 上的 MlaProlog 算子为例，完整讲解一个高性能融合算子从数学公式到工程实现的全过程。读者将学到：算子的数学原理、融合的动机与收益、AIC/AIV 双流水线设计、Host-Kernel 完整调用链路、TilingData 传递机制、量化策略的设计决策，以及各组件代码的实现方法。

---

## 1. 引言：为什么需要 MlaProlog

DeepSeek-V2 提出了 **MLA（Multi-head Latent Attention）**机制，其核心思想是用低秩压缩替代传统 MHA 中庞大的 KV Cache。在推理时，MLA 需要先将输入隐藏状态 $x$ 变换为 Query、QueryRoPE、KVCache、KRCache 四组输出——这个前处理过程就是 **MlaProlog**。

如果用 PyTorch 直接实现，这个前处理需要：
- 4 次矩阵乘法（MatMul）
- 2 次 RmsNorm 归一化
- 2 次 RoPE 位置编码
- 2 次 ScatterCache 缓存更新
- 可能还有量化/反量化操作

合计约 **12 次独立算子调用**。每次调用意味着一次 Kernel Launch 开销（~5us）和中间结果在 HBM 上的读写。对于推理延迟敏感的场景，这些开销不可忽视。

**MlaProlog 融合算子**将这 12 个操作合并为一次 Kernel 调用，中间结果全部在片上缓存（L1/UB）内流转，从根本上消除了冗余的 HBM 访问和 Launch 开销。

---

## 2. 公式解析：MlaProlog 计算了什么

MlaProlog 的计算可以分解为 **4 条路径**，最终产出 4 组输出。

### 2.1 路径总览

$$q^N = \alpha_q \cdot \mathrm{RmsNorm}(x \cdot W^{DQ}) \cdot W^{UQ} \cdot W^{UK} \tag{a}$$

$$q^R = \mathrm{RoPE}(\alpha_q \cdot \mathrm{RmsNorm}(x \cdot W^{DQ}) \cdot W^{QR}) \tag{b}$$

$$k^R = \mathrm{RoPE}(x \cdot W^{KR}) \tag{c}$$

$$c^{KV} = \alpha_{kv} \cdot \mathrm{RmsNorm}(x \cdot W^{DKV}) \tag{d}$$

其中 $\alpha_q$ 和 $\alpha_{kv}$ 是 V3 版本引入的尺度矫正因子（来自 [Meituan 论文](https://arxiv.org/abs/2509.01322)），分别对应代码中的 `qcQrScale` 和 `kcScale`。

### 2.2 分步计算单元

将 4 条路径拆解为 9 个基本计算单元（橙色 = AIC Cube，紫色 = AIV Vector）：

```mermaid
graph LR
    X["输入 x<br/>[T, He]"]

    subgraph Path_AB ["路径 a+b: Query"]
        MM1["① MM1: MatmulCq<br/>x · W_DQ → cQ<br/><i>AIC</i>"]
        RN1["② RmsNormCq<br/>α_q · RmsNorm(cQ)<br/><i>AIV</i>"]
        MM3["⑦ MM3: MatmulQcQr<br/>cQ_norm · [W_UQ|W_QR]<br/><i>AIC</i>"]
        RoPE_Qr["⑧ RopeQr<br/>RoPE(qR)<br/><i>AIV</i>"]
        MM4["⑨ MM4: MatmulQn<br/>qC · W_UK<br/><i>AIC</i>"]
        MM1 --> RN1 --> MM3
        MM3 -->|"qR"| RoPE_Qr
        MM3 -->|"qC"| MM4
    end

    subgraph Path_CD ["路径 c+d: KV Cache"]
        MM2["③ MM2: MatmulCkvKr<br/>x · [W_DKV|W_KR]<br/><i>AIC</i>"]
        RN2["④ RmsNormCkv<br/>α_kv · RmsNorm(cKV)<br/><i>AIV</i>"]
        RoPE_Kr["⑤ RopeKr<br/>RoPE(kR)<br/><i>AIV</i>"]
        SC["⑥ ScatterCkvKr<br/>写入 Cache<br/><i>AIV</i>"]
        MM2 -->|"cKV"| RN2 --> SC
        MM2 -->|"kR"| RoPE_Kr --> SC
    end

    X --> MM1
    X --> MM2

    OUT_QN["输出: qN"]
    OUT_QR["输出: qR"]
    OUT_KV["输出: KVCache"]
    OUT_KR["输出: KRCache"]

    MM4 --> OUT_QN
    RoPE_Qr --> OUT_QR
    SC --> OUT_KV
    SC --> OUT_KR

    style MM1 fill:#e17055,color:#fff
    style MM2 fill:#e17055,color:#fff
    style MM3 fill:#e17055,color:#fff
    style MM4 fill:#e17055,color:#fff
    style RN1 fill:#6c5ce7,color:#fff
    style RN2 fill:#6c5ce7,color:#fff
    style RoPE_Qr fill:#6c5ce7,color:#fff
    style RoPE_Kr fill:#6c5ce7,color:#fff
    style SC fill:#6c5ce7,color:#fff
    style X fill:#2d3436,color:#fff
    style OUT_QN fill:#00b894,color:#fff
    style OUT_QR fill:#00b894,color:#fff
    style OUT_KV fill:#00b894,color:#fff
    style OUT_KR fill:#00b894,color:#fff
```

> [在线编辑此图](https://l.mermaid.ai/qMzHtq)

对应的计算公式与管线分配：

| 步骤 | 计算 | 所在管线 |
|------|------|---------|
| ① MatmulCq | $c^Q = x \cdot W^{DQ}$ | AIC (Cube) |
| ② RmsNormCq | $c^Q_{norm} = \alpha_q \cdot \mathrm{RmsNorm}(c^Q)$ | AIV (Vector) |
| ③ MatmulCkvKr | $[c^{KV}, k^R] = x \cdot [W^{DKV} \| W^{KR}]$ | AIC |
| ④ RmsNormCkv | $c^{KV}_{norm} = \alpha_{kv} \cdot \mathrm{RmsNorm}(c^{KV})$ | AIV |
| ⑤ RopeKr | $k^R = \mathrm{RoPE}(k^R)$ | AIV |
| ⑥ ScatterCkvKr | 写入 KVCache 和 KRCache | AIV |
| ⑦ MatmulQcQr | $[q^C, q^R] = c^Q_{norm} \cdot [W^{UQ} \| W^{QR}]$ | AIC |
| ⑧ RopeQr | $q^R = \mathrm{RoPE}(q^R)$ | AIV |
| ⑨ MatmulQn | $q^N = q^C \cdot W^{UK}$ | AIC |

### 2.3 关键优化：权重矩阵拼接

步骤③和⑦利用了矩阵乘法的性质，将两个权重矩阵横向拼接成一个：

$$x \cdot [W^{DKV} | W^{KR}] = [x \cdot W^{DKV} | x \cdot W^{KR}] \tag{7}$$

这样只需一次 MatMul 调用就能同时计算 $c^{KV}$ 和 $k^R$，然后按列 Split 即可。

### 2.4 RmsNorm 公式

$$\mathrm{RmsNorm}(x) = \gamma \cdot \frac{x_i}{\mathrm{RMS}(x)}, \quad \mathrm{RMS}(x) = \sqrt{\frac{1}{N}\sum_{i=1}^{N}x_i^2 + \epsilon} \tag{3-4}$$

其中 $\gamma$ 是可学习的缩放参数（对应 `rmsnormGammaCq`/`rmsnormGammaCkv`），$\epsilon$ 是数值稳定性参数。

### 2.5 RoPE 公式与奇偶重排优化

RoPE 的标准公式将输入 $x$ 与旋转矩阵 $R_{\Theta,m}^d$ 相乘。展开后为逐元素运算（公式 14）。算子内部进一步优化为**奇偶重排**形式（公式 15），将 $x$ 的偶数位和奇数位分别聚集：

$$\mathrm{RoPE}(x) = [x_0, x_2, \ldots, x_{d-2}, x_1, x_3, \ldots, x_{d-1}]^T \otimes [\cos\theta, \cos\theta]^T + [-x_1, -x_3, \ldots, x_0, x_2, \ldots]^T \otimes [\sin\theta, \sin\theta]^T$$

**为什么可以重排？** 因为后续 $q^R \cdot k^R$ 矩阵乘的 Reduce 操作是对所有元素求和，元素位置的重排不影响最终结果。重排后 $\cos/\sin$ 数据变为连续存储，节省一半的存储空间。

---

## 3. 融合动机：为什么 12 个算子合成 1 个

以 DeepSeek-V3 的典型参数（He=7168, Hcq=1536, N=128, D=128, Dr=64）为例，分析非融合方案的开销。

### 3.1 非融合方案的 HBM 读写量（per token）

| 步骤 | 写出中间结果 | 下一步读回 |
|------|------------|-----------|
| MatmulCq → RmsNormCq | cQ: 1536×2B = 3KB | 3KB |
| RmsNormCq → MatmulQcQr | cQ_norm: 1536×2B = 3KB | 3KB |
| MatmulQcQr → RopeQr | QcQr: 24576×2B = 48KB | 48KB |
| MatmulCkvKr → RmsNormCkv | CkvKr: 576×2B = 1.1KB | 1.1KB |
| ... | ... | ... |
| **合计** | **~110KB/token 写** | **~90KB/token 读** |

12 次 Kernel Launch 开销 ≈ 60us。

### 3.2 融合后

- 中间结果在 **L1 (512KB)** 和 **UB (256KB)** 内流转，不写回 HBM
- 仅 1 次 Kernel Launch
- HBM 访问仅剩：读输入 x（14KB/token）+ 读权重（一次性）+ 写最终输出

**收益：节省 ~80% HBM 带宽，消除 ~60us Launch 开销。**

---

## 4. 算子架构：CANN 标准目录结构

MlaProlog 遵循 CANN 框架的标准算子目录结构：

```
mla_prolog_v3/
├── op_host/                              ← ① Host 侧（CPU 执行）
│   ├── mla_prolog_v3_def.cpp             # 算子注册
│   ├── mla_prolog_v3_tiling.h            # Tiling 策略
│   ├── mla_prolog_v3_infershape.h/cpp    # 输出 Shape 推导
│   └── op_api/aclnn_mla_prolog_v3.h/cpp  # ACLNN C 接口
│
├── op_kernel/                            ← ② Device 侧（NPU 执行）
│   ├── mla_prolog_v3.cpp                 # Kernel 入口 + 模板实例化
│   ├── kernel_mla_prolog_split_n.h       # 核心计算模板类
│   ├── mla_prolog_tiling_data.h          # TilingData 结构体
│   ├── mla_prolog_template_tiling_key.h  # TilingKey 位域 + 模板选择
│   └── service_*.h                       # 各计算模块（Matmul, RmsNorm, RoPE, Scatter）
│
└── tests/pytest/                         ← ③ 测试
    ├── test.py                           # pytest 主入口
    └── mla_prolog_v3_cpu_ref.py          # CPU 参考实现
```

**组件间的连接关系：**

```mermaid
graph LR
    DEF["def.cpp<br/>算子注册"] --> GE["GE 框架调度"]
    GE --> TILING["tiling.h<br/>计算切分方案"]
    TILING --> TD["TilingData 结构体"]
    TD -->|"二进制序列化<br/>随 kernel launch 传递"| ENTRY["mla_prolog_v3.cpp<br/>反序列化"]
    ENTRY --> TMPL["模板实例化<br/>(if constexpr)"]
    TMPL --> KERNEL["kernel_mla_prolog_split_n.h<br/>核心计算类"]
    KERNEL --> SVC["service_*.h<br/>Matmul / RmsNorm / RoPE / Scatter"]

    style DEF fill:#4a90d9,color:#fff
    style GE fill:#636e72,color:#fff
    style TILING fill:#4a90d9,color:#fff
    style TD fill:#fdcb6e,color:#2d3436
    style ENTRY fill:#e17055,color:#fff
    style TMPL fill:#e17055,color:#fff
    style KERNEL fill:#e17055,color:#fff
    style SVC fill:#6c5ce7,color:#fff
```

---

## 5. 完整调用链路：从 Python 到 NPU 核心

这是本文最核心的章节，追踪一次算子调用的完整路径。

### 5.1 用户调用

```python
# Python 侧（通过 torch_npu 间接调用 ACLNN API）
output = torch_npu.npu_mla_prolog_v3(tokenX, weightDq, weightUqQr, weightUk, ...)
```

底层调用 `aclnnMlaPrologV3GetWorkspaceSize()` 计算所需 workspace 大小，然后 `aclnnMlaPrologV3()` 提交执行。ACLNN API 接受 **37 个参数**（21 个输入 tensor + 12 个属性 + 5 个输出 tensor）。

### 5.2 算子注册与 Dispatch

`mla_prolog_v3_def.cpp` 中通过 `REG_OP(MlaPrologV3)` 向 GE 框架注册算子，声明所有输入/输出 tensor 及其类型组合。GE 根据输入 tensor 的 dtype 自动匹配到正确的类型组合。

最后一行 `OP_ADD(MlaPrologV3, optiling::MlaPrologCompileInfo)` 将 **Tiling 函数绑定到算子**——这是 Host 侧和 Device 侧的桥梁。

### 5.3 Tiling 计算（Host CPU）

当 GE 调度到 MlaProlog 算子时，Host 上的 Tiling 函数 `RunBigKernelTiling()` 被调用，执行以下步骤：

```
GetNpuInfo()         → 读硬件信息（AIC=32, AIV=64, L1=512KB, ...）
SetShapeInfo()       → 从输入 tensor 提取维度（B, S, He, Hcq, ...）
FillMatmul1Tiling()  → 计算 MM1 切分: mm1BlockNum=24, mm1SingleCoreN=64
FillMatmul2Tiling()  → 计算 MM2 切分: mm2BlockNum=9, mm2SingleCoreN=64
FillMatmul3Tiling()  → 计算 MM3 切分: mm3BlockNum=32, mm3SingleCoreN=768
FillMatmul4Tiling()  → 计算 MM4 切分: mm4BlockNum=32
FillTiling()         → 填充 MlaPrologBaseParams 结构体
```

产出的 `MlaPrologTilingData` 包含所有切分参数。

### 5.4 TilingKey 生成

`GenTilingKey()` 将运行时参数编码为 **64-bit 整数**，每个二进制位域对应一个模板参数：

| 位域 | 变量 | 含义 | 示例值 |
|------|------|------|--------|
| [0-3] | CACHE_MODE | KVCache 格式 | 1 (PA_BSND) |
| [4-5] | SCENARIO | 量化场景 | 1 (BF16) |
| [6-9] | QUANT_MODE | 量化模式 | 0 (NO_QUANT) |
| [10] | DEQUANT_OPT | 反量化优化 | 0 |
| [11] | GROUP_COMPUTE_OPT | 分组计算优化 | 0 |
| [12-13] | EMPTY_TENSOR_MODE | 空 tensor | 0 |
| [14-15] | ACTUAL_SEQ_LEN_MODE | 变长序列 | 0 |
| [16-17] | SPLIT_M_MODE | 切 M 模式 | 0 (切 N) |
| [18-25] | CV_MODE | AIC:AIV 比例 | 7 (1:2 模式) |

**TilingKey 的作用**：在编译期，每个合法的 TilingKey 对应一个唯一的 kernel 函数实例（C++ 模板特化）。运行时根据 TilingKey 直接跳转到对应的 kernel，避免运行时分支开销。

### 5.5 TilingData 如何传递到 Kernel

```mermaid
graph LR
    subgraph HOST ["Host CPU"]
        TD2["MlaPrologTilingData<br/>{baseParams:<br/>  batchSize,<br/>  mm1Block, ...}"]
    end

    subgraph DEVICE ["Device NPU"]
        GM["__gm__ uint8_t *tiling<br/>GET_TILING_DATA_WITH_STRUCT(...)"]
    end

    TD2 -->|"二进制序列化<br/>GE 框架管理<br/>随 kernel launch 参数传递"| GM

    style HOST fill:#4a90d9,color:#fff
    style DEVICE fill:#e17055,color:#fff
```

GE 框架将 `MlaPrologTilingData` 结构体**二进制序列化**，作为 kernel launch 的最后一个参数传递。在 Device 侧，`GET_TILING_DATA_WITH_STRUCT` 宏通过 **reinterpret_cast 零拷贝反序列化**——直接将 `__gm__ uint8_t*` 指针转换为 `MlaPrologTilingData*`，无需任何数据拷贝。

### 5.6 Kernel 入口

`mla_prolog_v3.cpp` 中的 `__global__` 函数是 NPU 侧的入口：

```cpp
// mla_prolog_v3.cpp (简化)
template<uint8_t CacheMode, uint8_t Scenario, uint8_t QuantMode, ...>
__global__ __aicore__ void mla_prolog_v3(
    __gm__ uint8_t *tokenX, ...,  // 28 个 GM 指针
    __gm__ uint8_t *tiling)        // TilingData
{
    // 1. 反序列化 TilingData
    GET_TILING_DATA_WITH_STRUCT(MlaPrologTilingData, tilingData, tiling);

    // 2. 根据编译期模板参数选择类型组合，实例化计算类
    TPipe pipe;
    if constexpr (Scenario == NO_QUANT) {
        MlaPrologVecS1CubS2<MLAPType<bf16, bf16, bf16, ...>> op(&pipe, ...);
        op.Init(tokenX, weightDq, ...);  // 绑定 GM tensor
        op.Process();                     // 执行计算
    } else if constexpr (QuantMode == FULL_QUANT) {
        MlaPrologVecS1CubS2<MLAPType<int8, int8, bf16, ...>> op(&pipe, ...);
        op.Init(...); op.Process();
    }
    // ... 9 种量化模式的 if constexpr 分支
}
```

关键点：
- `if constexpr` 在**编译期**求值，运行时无分支开销
- `MLAPType<...>` 是类型参数包，将量化模式映射为具体数据类型
- `MlaPrologVecS1CubS2` 是核心计算模板类，定义在 `kernel_mla_prolog_split_n.h`

---

## 6. Tiling 数据切分：把数据分给 32 个核

### 6.1 为什么需要 Tiling

以 MM3（MatmulQcQr）为例：B 矩阵 $W^{UQ}|W^{QR}$ 的大小为 1536 × 24576 × 2B = **72 MB**，而 L1 仅有 512 KB。必须将矩阵分成小块，逐块加载到片上缓存中计算。

### 6.2 TilingData 结构体

```cpp
// mla_prolog_tiling_data.h
struct MlaPrologBaseParams {
    uint32_t batchSize;           // 批大小
    uint32_t stepBatchSize;       // 每步处理的 token 数（≤ 128）
    uint32_t headSizeX;           // 输入隐藏维度 He=7168
    uint32_t headSizeCq;          // 潜在 Query 维度 Hcq=1536
    uint32_t headSizeCkv;         // 潜在 KV 维度 Hckv=512
    uint32_t numHeadSize;         // 注意力头数 N=128
    uint32_t dimHeadSizeQc;       // 单头 Query 维度 D=128
    uint32_t dimHeadRope;         // 单头 RoPE 维度 Dr=64

    uint32_t mm1BlockNum;         // MM1 使用的 Cube 核数
    uint32_t mm1SingleCoreN;      // MM1 每核处理的 N 轴大小
    uint32_t mm2BlockNum, mm2SingleCoreN;  // MM2 切分
    uint32_t mm3BlockNum, mm3SingleCoreN;  // MM3 切分
    uint32_t mm4BlockNum;                   // MM4 切分
    uint32_t vectorBlockNum;               // Vector 核数

    float qcQrScale, kcScale;    // V3 尺度矫正因子
    uint32_t kvQuantMode;        // KVCache 量化模式
    // ... 更多参数
};
```

### 6.3 4 个 Matmul 的核间切分

| Matmul | M 轴 | N 轴 | K 轴 | 核间切分方式 | 设计理由 |
|--------|------|------|------|-------------|---------|
| MM1 (Cq) | T | Hcq=1536 | He=7168 | N 轴按 24 核均分 | 切 N 使每核 B 矩阵更小，利于 L1 缓存 |
| MM2 (CkvKr) | T | Hckv+Dr=576 | He=7168 | N 轴固定分 9 块 | N 轴小，固定分块简化控制逻辑 |
| MM3 (QcQr) | T | N*(D+Dr)=24576 | Hcq=1536 | N 轴按 32 核均分 | N 轴最大（24576），切分后每核仅 768 列 |
| MM4 (Qn) | T | Hckv=512 | D=128 | **不切 N** | 矩阵很小（128×512），全核同步后执行 |

### 6.4 stepBatchSize：M 轴分步

当 T（token 总数）过大时，无法一次处理所有 token。Tiling 计算出 `stepBatchSize`（通常为 128），主循环按步处理：

```cpp
for (i = 0; i < ceil(T / stepBatchSize); i++) {
    当前步处理 min(stepBatchSize, T - i * stepBatchSize) 个 token
}
```

---

## 7. 流水设计：AIC 和 AIV 如何并行

### 7.1 双流水线架构

Ascend NPU 的每个 AI Core 包含两个**物理独立**的计算单元：

- **AIC (AI Core / Cube)**：矩阵乘法单元，执行 MMAD 运算
- **AIV (AI Vector)**：向量计算单元，执行逐元素操作

两者可以**真正并行执行**，但当一方需要另一方的计算结果时，必须插入同步。

### 7.2 时序图

```mermaid
sequenceDiagram
    participant AIC as AIC (Cube)
    participant AIV as AIV (Vector)

    Note over AIV: CopyIn + GetSinCos

    rect rgb(225, 112, 85, 0.3)
    AIC->>AIC: ① MM1 MatmulCq (17.89 us)
    end
    AIC-->>AIV: SYNC_MMCQ ✉️

    rect rgb(108, 92, 231, 0.3)
    Note over AIV: 全核同步 (V→V)
    AIV->>AIV: ② RmsNormCq (0.015 us)
    end

    par AIC 做 MM2 & AIV 做 RmsNormCq
        rect rgb(225, 112, 85, 0.3)
        AIC->>AIC: ③ MM2 MatmulCkvKr (17.89 us)
        end
    end

    AIV-->>AIC: SYNC_RMSNORM_CQ ✉️
    AIC-->>AIV: SYNC_MMCKVKR ✉️

    rect rgb(108, 92, 231, 0.3)
    AIV->>AIV: ④ RmsNormCkv (0.005 us)
    AIV->>AIV: ⑤ RopeKr (0.001 us)
    AIV->>AIV: ⑥ ScatterCkvKr (0.002 us)
    end

    rect rgb(225, 112, 85, 0.3)
    AIC->>AIC: ⑦ MM3 MatmulQcQr (47.19 us)
    end
    AIC-->>AIV: SYNC_MMQCQR ✉️

    Note over AIC,AIV: SYNC_ALL_CUBE (全核同步)

    par AIC 做 MM4 & AIV 做 RopeQr
        rect rgb(225, 112, 85, 0.3)
        AIC->>AIC: ⑨ MM4 MatmulQn (0.66 us)
        end
        rect rgb(108, 92, 231, 0.3)
        AIV->>AIV: ⑧ RopeQr (0.105 us)
        end
    end
```

> [在线编辑此图](https://l.mermaid.ai/y7XF2T)

### 7.3 同步点详解

以下是设计文档中 `Process()` 函数的核心伪代码（简化），逐同步点解释：

```cpp
if ASCEND_IS_AIC {
    // ① AIC: 执行 MatmulCq
    MatmulCq(tokenX, weightDq, cqRes);
    CrossCoreSetFlag(SYNC_MMCQ);           // → 通知 AIV: "Cq 结果已就绪"

    // ② AIC: 执行 MatmulCkvKr（与 AIV 做 RmsNormCq 并行）
    MatmulCkvKr(tokenX, weightDkvKr, ckvKrRes);
    CrossCoreSetFlag(SYNC_MMCKVKR);        // → 通知 AIV: "CkvKr 结果已就绪"

    // ③ AIC: 等待 AIV 完成 RmsNormCq 后才能开始 MatmulQcQr
    CrossCoreWaitFlag(SYNC_RMSNORM_CQ);    // ← 等 AIV: "RmsNorm 结果已就绪"
    MatmulQcQr(weightUqQr, qcQrRes);
    CrossCoreSetFlag(SYNC_MMQCQR);         // → 通知 AIV: "QcQr 结果已就绪"

    // ④ AIC: 全核同步（MM3 的切分策略与 MM4 不同）
    CrossCoreSetFlag(SYNC_ALL_CUBE);
    CrossCoreWaitFlag(SYNC_ALL_CUBE);      // 等所有 Cube 核完成 MM3
    MatmulQn(qc, weightUk, qnRes);
}

if ASCEND_IS_AIV {
    GetSinCos(tokenIndex);                 // 预加载 sin/cos 表

    CrossCoreWaitFlag(SYNC_MMCQ);          // ← 等 AIC: MM1 完成
    // 全核同步确保数据搬运完成
    CrossCoreSetFlag(SYNC_ALL_VECTOR);
    CrossCoreWaitFlag(SYNC_ALL_VECTOR);

    RmsNormCq(cqRes);                      // AIV: RmsNorm（与 AIC 做 MM2 并行！）

    CrossCoreSetFlag(SYNC_RMSNORM_CQ);     // → 通知 AIC: "可以开始 MM3 了"
    CrossCoreWaitFlag(SYNC_MMCKVKR);       // ← 等 AIC: MM2 完成

    RmsNormCkv(ckvKrRes);                  // AIV: RmsNorm + RoPE + Scatter
    RopeKr(krRes);
    ScatterCkvKr(kvCache, krCache);

    CrossCoreWaitFlag(SYNC_MMQCQR);        // ← 等 AIC: MM3 完成
    RopeQr(qrRes);                         // AIV: RoPE Qr
}
```

**为什么需要这些同步？**

| 同步点 | 方向 | 原因 |
|--------|------|------|
| SYNC_MMCQ | AIC→AIV | MM1 对 N 轴做了 split，每核只有部分列；RmsNorm 需要完整行 |
| SYNC_RMSNORM_CQ | AIV→AIC | RmsNormCq 按行切分到不同 AIV 核；MM3 需要所有行的结果 |
| SYNC_MMCKVKR | AIC→AIV | 同 SYNC_MMCQ，MM2 split-N 后需全核数据 |
| SYNC_MMQCQR | AIC→AIV | MM3 的 split-N 与 RopeQr 的 split 策略不同 |
| SYNC_ALL_CUBE | 全核 | MM4 依赖 MM3 全部输出，但两者切分策略不同 |

### 7.4 CV 模式

- **1:2 模式**（默认）：1 个 AIC 核配 2 个 AIV 核，AIV 算力翻倍
- **1:1 模式**：1:1 配对，同步更简单
- 通过 TilingKey 的 `CV_MODE` 位域选择，对应模板参数 `CvMode`

---

## 8. 量化设计：哪些 Matmul 能量化，为什么

### 8.1 量化总览

| Matmul | BF16 权重 | INT8 权重 | FP8 权重 | 不能量化的原因 |
|--------|:--------:|:--------:|:-------:|-------------|
| MM1 (Cq) | ✓ | ✓ | ✓ | — |
| MM2 (CkvKr) | ✓ | ✓ | ✓ | — |
| MM3 (QcQr) | ✓ | ✓ | ✓ | — |
| MM4 (Qn) | ✓ | ✗ | ✗ | 矩阵小(128×512) + 直接产出 Query，精度敏感 |

**MM4 为什么不量化？** 两个原因：
1. W^UK 仅 128×512 = 128KB，量化节省的带宽非常有限
2. MM4 直接输出 Query，被下游 Attention Score 使用，INT8 误差会被放大

### 8.2 10 种量化模式

`mla_prolog_v3.cpp` 中通过 `if constexpr` 链实现了 10 种模板实例化。`MLAPType<A, B, C, ...>` 的 3 个核心类型参数含义：

| QuantMode | MLAPType<A, B, C> | 含义 |
|-----------|-------------------|------|
| 0 NO_QUANT | `<bf16, bf16, bf16>` | 全 BF16 |
| 1 PARTIAL_KV_NO | `<bf16, int8, bf16>` | MM3 权重 INT8 |
| 2 PARTIAL_KV_QUANT | `<bf16, int8, int8>` | + KVCache INT8 |
| 3 FULL_KV_NO | `<int8, int8, bf16>` | MM1/2/3 权重都 INT8 |
| 4 FULL_KV_QUANT | `<int8, int8, int8>` | + KVCache INT8 |
| 5 PARTIAL_PERTILE | `<bf16, int8, int8>` + isPertile | KVCache per-tile 量化 |
| 6 FULL_PERTILE | `<int8, int8, int8>` + isPertile | 全量化 + per-tile |
| 7 MXFP8_KV_NO | `<FP8E4M3, FP8E4M3, bf16>` | MXFP8 量化 |
| 8 MXFP8_KV_QUANT | `<FP8E4M3, FP8E4M3, FP8E4M3>` | MXFP8 + KVCache FP8 |
| 9 MXFP8_PERTILE | `<FP8E4M3, FP8E4M3, FP8E4M3>` + isPertile | MXFP8 + per-tile |

### 8.3 Dequant 在 Pipeline 中的位置

当 MM1/MM2 使用 INT8 权重时，输出为 INT32（整数累加结果）。需要在 AIV 侧做 **Dequant + RmsNorm** 一体化计算：

```
AIC: MM1(INT8×INT8→INT32) → signal → ...
AIV: ← wait → Dequant(INT32→BF16) + RmsNorm(BF16→BF16) → signal → ...
```

Dequant 公式：`x_float = x_int32 * dequant_scale`。对应 `service_dequant.h` 中的 `DequantPerChannel`（按列）或 `DequantPerToken`（按行）。

---

## 9. 内核代码解析：Service 模块

### 9.1 核心计算类

`kernel_mla_prolog_split_n.h` 定义了模板类 `MlaPrologVecS1CubS2<MLAPT>`：

```cpp
template <typename MLAPT>
class MlaPrologVecS1CubS2 {
    // 类型别名从 MLAPT 提取
    using mmInputType = typename MLAPT::mmInputType;      // bf16/int8/fp8
    using mmCqOutputType = typename MLAPT::mmCqOutputType; // bf16/int32
    using kvCacheType = typename MLAPT::kvCacheType;       // bf16/int8/fp8

    // 4 个 Matmul 的参数
    MMParams mmCqParam_, mmCkvKrParam_, mmQcQrParam_, mmQnParam_;

    // GM tensor 句柄
    GlobalTensor<mmInputType> tokenXGm_, weightDqGm_, ...;

    // 生命周期
    void Init(__gm__ uint8_t *tokenX, ...);  // 绑定 GM 指针
    void Process();                           // 主循环（含 AIC/AIV 分支和同步）
};
```

`Init()` 将原始 `uint8_t*` 指针转换为强类型 `GlobalTensor`，并从 TilingData 读取切分参数初始化 `MMParams`。

### 9.2 Matmul 模块 (service_matmul.h)

使用 AscendC 的 **Matmul 高阶 API**，封装了 L1→L0 搬运、MMAD 计算、FixPipe 写回的完整流程。核心方法是 `MatmulSplitN()`：

```cpp
template <typename T, typename O>
void MatmulSplitN(
    GlobalTensor<O> &tensorRes,    // 输出
    GlobalTensor<T> &tensorA,      // A 矩阵（激活）
    GlobalTensor<T> &tensorB,      // B 矩阵（权重）
    MMParams &param);              // M, N, K, splitN 等参数
```

内部通过双缓冲管理 L0A/L0B，实现搬运与计算的流水重叠。

### 9.3 RmsNorm 模块 (service_rms_norm.h)

两个核心函数：

- `RmsNormNormal()`：标准 RmsNorm。输入可以是 BF16 或 INT32（量化模式）。如果输入是 INT32，先做 Dequant 再做 Norm。
- `RmsNormDynamicQuant()`：RmsNorm + 动态量化融合。输出 INT8 + per-token scale。

### 9.4 RoPE 模块 (service_rotary_position_embedding.h)

`RotaryPosEmbPerTensor()` 实现公式 (15) 的奇偶重排 RoPE：
1. 从 `service_gather_sin_cos.h` 查表获取对应 position 的 sin/cos
2. 将输入按奇偶拆分：`x_even = x[0::2], x_odd = x[1::2]`
3. 计算：`out_even = x_even * cos - x_odd * sin`，`out_odd = x_odd * cos + x_even * sin`

### 9.5 ScatterCache 模块 (service_scatter_cache.h)

支持 4 种 KVCache 格式（PA_BSND, PA_NZ, PA_BLK_BSND, PA_BLK_NZ），通过 `cacheIndex` 间接寻址：

```cpp
cache[blockIdx][posInBlock][head] = computedKV[token]
// 其中 blockIdx, posInBlock 由 cacheIndex 计算得到
```

NZ 格式需要特殊的 stride 处理：将完整 H 维度按 32B 小块分块后，stride 设为 `32B × BlockSize`。

---

## 10. 测试与性能分析

### 10.1 CPU 参考实现

`mla_prolog_v3_cpu_ref.py` 用 PyTorch 实现了完整的计算图，核心函数 `cal_mlaprolog()` 按顺序执行 MM1→RmsNorm→MM3→RoPE→Scatter 等步骤。测试时将 NPU 输出与 CPU 参考逐 tensor 比较精度。

### 10.2 运行测试

```bash
cd attention/mla_prolog_v3/tests/pytest/
pytest test.py -v                                    # 全量测试
pytest test.py -m ci                                 # CI 测试
MLA_PROLOG_V3_ENABLE_FUZZ=1 pytest test.py -v        # Fuzz 测试
```

### 10.3 性能建模

`tests/perf/` 目录包含理论性能分析工具：

```bash
# 查看各阶段瓶颈
python perf_analyzer.py --chip 950 --batch-size 1 --head-num 128 --mode bound

# Pipeline 时序分析
python perf_analyzer.py --chip 950 --batch-size 1 --head-num 128 --mode pipeline

# Prefill 场景 K-outer 优化分析
python perf_analyzer.py --chip 950 --batch-size 256 --seq-len 4 --head-num 128 --mode prefill
```

---

## 总结

构建 MlaProlog 融合算子的完整路径：

1. **理解数学** → 4 条路径、9 个计算单元、权重拼接优化
2. **设计融合策略** → 消除中间 HBM 读写，单 kernel 完成
3. **设计 Tiling** → 核间切 N、stepBatchSize 分步、TilingKey 位域编码
4. **设计流水** → AIC/AIV 并行、6 个同步点、1:2 CV 模式
5. **实现 Host 侧** → def.cpp 注册、tiling 计算、ACLNN API
6. **实现 Kernel** → 模板实例化（10 种量化模式）→ 计算类 → Service 模块
7. **测试验证** → CPU 参考 + pytest + 性能建模

整个过程体现了**算法-硬件协同优化**的核心思想：理解硬件的存储层次和计算单元特性，才能设计出真正高效的融合算子。
