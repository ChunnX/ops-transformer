# KvQuantSparseFlashAttentionPioneer 算子代码梳理

## 1. 算子概述

### 1.1 功能定位

`KvQuantSparseFlashAttentionPioneer`（以下简称 QSFA-Pioneer）是面向 Sparse Attention 场景的高性能 FlashAttention 算子，运行于 Ascend 950 平台（arch35/C310 架构）。该算子在 `SparseFlashAttention` 基础上引入了以下核心特性：

- **Per-Token-Head-Tile-128 量化输入**：KV Cache 以 FP8（float8_e4m3fn / hifloat8）存储，配合 per-tile（tile_size=128）的 FP32 反量化系数，在 Kernel 内完成在线反量化
- **MLA-Absorb 模式**：支持 Multi-Latent Attention 架构，Q 的 D 维度包含 nope（512）+ rope（64）= 576，KV 共享同一份数据（KV_N=1）
- **离散稀疏访存优化**：通过 `sparse_indices` 索引从 KV Cache 中离散取出重要 token，并在搬运层面进行指令缩减与聚合优化
- **Param Sink 支持**：可选地将 sink token（固定 128 个）拼接到 KV 序列首部参与 Attention 计算
- **AIC/AIV 异构三级流水**：Cube 核负责 BMM1/BMM2 矩阵乘，Vector 核负责 KV 搬运反量化（Vec0）、Softmax（Vec1）、Flash Update 输出（Vec2），三级流水并行

### 1.2 计算公式

$$
\tilde{K} = \text{Gather}(Key; SparseIndices)
$$

$$
\tilde{K}_{\text{Nope}} = \text{DeQuant}_{8 \rightarrow 16}(\tilde{K}[..., :512]; key\_scale)
$$

$$
\tilde{K}_{\text{Rope}} = \tilde{K}[..., 512:576]
$$

$$
\tilde{K} = \text{concat}[\tilde{K}_{\text{Nope}}, \tilde{K}_{\text{Rope}}]
$$

$$
\tilde{K} = \text{concat}[key\_sink, \tilde{K}] \quad (\text{if hasSink})
$$

$$
Attention = \text{softmax}\left(\frac{Q \cdot \tilde{K}^T}{\sqrt{d_k}}\right) \cdot \tilde{V}
$$

其中：
- $Q$：query，shape `[B,S1,Q_N,576]`（nope 512 + rope 64）
- $\tilde{K}, \tilde{V}$：经稀疏索引选取并反量化后的 Key/Value
- $d_k = 576$，`scale_value = 1/sqrt(576) ≈ 0.0417`
- $\text{DeQuant}$：FP8 → FP32 → 乘 scale → Cast 到 FP16/BF16

### 1.3 支持平台

| SoC | 架构 | 说明 |
|-----|------|------|
| Ascend 950PR / 950DT | arch35 (C310) | 唯一支持平台，AIC/AIV 异构 |

---

## 2. 目录结构

```
kv_quant_sparse_flash_attention_pioneer/
├── CMakeLists.txt                                          # 顶层构建入口，遍历子目录
├── README.md                                               # 算子功能说明、参数表、约束
│
├── docs/
│   └── aclnnKvQuantSparseFlashAttentionPioneer.md          # aclnn 接口文档（函数原型、参数详表）
│
├── examples/
│   └── test_kv_quant_sparse_flash_attention_pioneer.cpp    # aclnn 调用示例
│
├── op_host/                                                # ═══ Host 侧（CPU 端）═══
│   ├── CMakeLists.txt                                      # Host 构建配置，编译选项
│   ├── kv_quant_sparse_flash_attention_pioneer_def.cpp     # OpDef：算子接口注册
│   ├── kv_quant_sparse_flash_attention_pioneer_infershape.cpp  # InferShape：输出 shape 推导
│   ├── kv_quant_sparse_flash_attention_pioneer_tiling.h    # Tiling 头文件：数据结构 + 类声明
│   ├── kv_quant_sparse_flash_attention_pioneer_tiling.cpp  # Tiling 实现：切分策略 + 参数校验
│   └── op_api/                                             # aclnn 接口封装
│       ├── aclnn_kv_quant_sparse_flash_attention_pioneer.cpp
│       ├── aclnn_kv_quant_sparse_flash_attention_pioneer.h
│       ├── kv_quant_sparse_flash_attention_pioneer.cpp
│       └── kv_quant_sparse_flash_attention_pioneer.h
│
└── op_kernel/                                              # ═══ Kernel 侧（NPU 端）═══
    ├── kv_quant_sparse_flash_attention_pioneer.cpp          # Kernel 入口：__global__ 函数
    ├── kv_quant_sparse_flash_attention_pioneer_common.h     # 公共常量、枚举、模板宏
    ├── kv_quant_sparse_flash_attention_pioneer_kernel_mla.h # 主 Kernel 类：Init + ProcessMainLoop
    ├── kv_quant_sparse_flash_attention_pioneer_kvcache.h    # KV Cache 参数计算（分核、S1/S2 循环）
    ├── kv_quant_sparse_flash_attention_pioneer_service_cube_mla.h   # Cube 服务：BMM1/BMM2
    ├── kv_quant_sparse_flash_attention_pioneer_service_vector_mla.h # Vector 服务：Vec0/Vec1/Vec2
    ├── kv_quant_sparse_flash_attention_pioneer_template_tiling_key.h # 模板参数组合声明
    ├── util_regbase.h                                      # 运行时参数结构体（RunInfo/ConstInfo/CVSharedParams）
    └── vf/                                                 # 微核函数（VF）
        ├── vf_basic_block_utils.h                          # VF 公共工具（CastTrait 定义）
        ├── vf_mul_sel_softmaxflashv2_cast_nz_qsfa.h       # Softmax 路由：按 s2RealSize 分档
        ├── vf_flashupdate_new_qsfa.h                       # Flash Update：O = O_prev * exp + O_cur
        ├── vf_basic_block_aligned128_no_update_qsfa.h      # Softmax VF：s2=128 对齐，首次（无 update）
        ├── vf_basic_block_aligned128_update_qsfa.h         # Softmax VF：s2=128 对齐，非首次（有 update）
        ├── vf_basic_block_unaligned64_no_update_qsfa.h     # Softmax VF：s2≤64 非对齐，首次
        ├── vf_basic_block_unaligned64_update_qsfa.h        # Softmax VF：s2≤64 非对齐，非首次
        ├── vf_basic_block_unaligned128_no_update_qsfa.h    # Softmax VF：64<s2<128 非对齐，首次
        └── vf_basic_block_unaligned128_update_qsfa.h       # Softmax VF：64<s2<128 非对齐，非首次
```

---

## 3. 输入/输出/属性规格

### 3.1 输入张量

| 序号 | 名称 | 必选 | 数据类型 | Shape | 说明 |
|------|------|------|---------|-------|------|
| 0 | query | 是 | BF16 / FP16 | BSND: `[B,S1,Q_N,576]`; TND: `[Q_T,Q_N,576]` | Q 输入，D=nope(512)+rope(64)=576。Q_N 支持 1/2/3/4/6/8/16/24/32/48/64 |
| 1 | key | 是 | FP8_E4M3 / HIFLOAT8 | PA_BSND: `[block_num,block_size,1,656]`; BSND: `[B,S2,1,656]`; TND: `[KV_T,1,656]` | K 输入，D=nope(512)+rope(128)+scale(16)=656。KV_N 仅支持 1。PA 场景支持非连续 |
| 2 | value | 是 | FP8_E4M3 / HIFLOAT8 | 同 key | V 输入，与 key 共享同一份数据（MLA 模式） |
| 3 | sparse_indices | 是 | INT32 | BSND: `[B,Q_S,KV_N,sparse_size]`; TND: `[Q_T,KV_N,sparse_size]` | 离散取 KV Cache 的索引，有效值在前、无效值在后 |
| 4 | key_dequant_scale | 否 | FP32 | - | 预留参数，当前 combine 模式下 scale 内嵌在 key 的 D 维度中 |
| 5 | value_dequant_scale | 否 | FP32 | - | 预留参数 |
| 6 | block_table | 否 | INT32 | `[B, S2/block_size]` | PA 场景的 block 映射表 |
| 7 | actual_seq_lengths_query | 否 | INT32 | `(B,)` | 每个 Batch 的 Q 有效 token 数。TND 时为前缀和 |
| 8 | actual_seq_lengths_kv | 否 | INT32 | `(B,)` | 每个 Batch 的 KV 有效 token 数。TND/PA 时必传 |
| 9 | key_sink | 否 | BF16 / FP16 | `[128, KV_N, D]` | Sink token 的 Key，sink_num 固定 128 |
| 10 | value_sink | 否 | BF16 / FP16 | `[128, KV_N, D]` | Sink token 的 Value |

### 3.2 输出张量

| 名称 | 数据类型 | Shape | 说明 |
|------|---------|-------|------|
| attention_out | BF16 / FP16（与 query 同类型） | BSND: `[B,S1,Q_N,512]`; TND: `[Q_T,Q_N,512]` | 输出 D=512（去掉 rope 维度） |

### 3.3 关键属性

| 属性 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| scale_value | FLOAT | 1.0 | 缩放系数 1/√d_k，实际使用 SOFTMAX_SCALE=0.0417 |
| key_quant_mode | INT64 | 1 | Key 量化模式，仅支持 2（per_tile） |
| value_quant_mode | INT64 | 1 | Value 量化模式，仅支持 2（per_tile） |
| sparse_block_size | INT64 | 1 | Sparse 阶段 block 大小，仅支持 1 |
| layout_query | STRING | "BSND" | Q 布局，支持 BSND / TND |
| layout_kv | STRING | "BSND" | KV 布局，支持 BSND / TND / PA_BSND |
| sparse_mode | INT64 | 3 | 0=全计算，3=rightDownCausal（右下三角 mask） |
| pre_tokens | INT64 | INT64_MAX | 前向 token 关联数 |
| next_tokens | INT64 | INT64_MAX | 后向 token 关联数 |
| attention_mode | INT64 | 0 | 仅支持 2（MLA-absorb 模式） |
| quant_scale_repo_mode | INT64 | 1 | 量化参数存放模式，1=combine（scale 内嵌在 KV 数据中） |
| tile_size | INT64 | 128 | Per-tile 量化的 tile 大小 |
| rope_head_dim | INT64 | 64 | MLA 下 rope 维度大小 |
| key_block_stride | INT64 | -1 | PA 场景 key block 的 stride（-1 表示自动计算） |

### 3.4 布局支持

| 布局 | 维度数 | 适用张量 | 说明 |
|------|--------|---------|------|
| BSND | 4 | query / key / value / out | 标准 Batch-Seq-Head-Dim 格式 |
| TND | 3 | query / key / value / out | 变长序列格式，T=所有 batch 的 token 总和 |
| PA_BSND | 4 | key / value | PageAttention 格式，`[block_num, block_size, N, D]` |

---

## 4. Host 侧逻辑

### 4.1 OpDef（kv_quant_sparse_flash_attention_pioneer_def.cpp）

注册 11 个输入（4 必选 + 7 可选）、1 个输出、15 个属性。支持 4 组数据类型组合：

| 组合 | query | key/value | attention_out |
|------|-------|-----------|---------------|
| 0 | BF16 | FP8_E4M3 | BF16 |
| 1 | BF16 | HIFLOAT8 | BF16 |
| 2 | FP16 | FP8_E4M3 | FP16 |
| 3 | FP16 | HIFLOAT8 | FP16 |

特殊配置：
- `DynamicCompileStaticFlag(true)`：支持动态编译静态化
- `PrecisionReduceFlag(true)`：允许精度降低优化
- 仅注册 `ascend950` SoC
- 编译选项：`--cce-auto-sync=off`（手动同步）、`-mllvm -cce-vf-remove-membar=false`

### 4.2 InferShape（kv_quant_sparse_flash_attention_pioneer_infershape.cpp）

输出 shape 推导规则：
- BSND：`attention_out = [B, S1, Q_N, D - rope_head_dim]` = `[B, S1, Q_N, 512]`
- TND：`attention_out = [Q_T, Q_N, D - rope_head_dim]` = `[Q_T, Q_N, 512]`
- 输出数据类型与 query 一致

### 4.3 Tiling 策略

#### 4.3.1 Tiling 主流程

```
TilingKvQuantSparseFlashAttentionPioneer()  // 注册入口
  ├── QSFAPInfoParser::Parse()              // 解析所有输入参数
  │   ├── CheckRequiredParaExistence()      // 必选参数存在性校验
  │   ├── GetOpParaInfo()                   // 获取输入/输出/属性信息
  │   ├── GetInOutDataType()                // 获取数据类型
  │   ├── GetBatchSize/S1Size/S2Size/...    // 解析各维度大小
  │   └── GenerateInfo()                    // 填充 QSFATilingInfo
  │
  ├── QSFAPTilingCheck::Process()           // 参数合法性校验
  │   ├── CheckSinglePara()                 // 单参数校验
  │   ├── CheckMultiParaConsistency()       // 多参数一致性校验
  │   ├── CheckKV()                         // KV shape/dtype 校验
  │   ├── CheckFeature()                    // 特性组合校验（MLA/Antiquant）
  │   └── CheckSink()                       // Sink 参数校验
  │
  └── QSFAPMlaTiling::DoOpTiling()          // 核心 Tiling 逻辑
      ├── GetPlatformInfo()                 // 获取 AIC/AIV 核数
      ├── InitParams()                      // 初始化切分参数
      ├── Split()                           // 分核策略
      │   └── SplitBalanced()               // 按 S1 维度均衡分核
      ├── CalcUbBmm()                       // 计算 UB 上 BMM 结果大小
      ├── FillTiling()                      // 填充 TilingData
      ├── CalcBlockDim()                    // 计算 blockDim
      ├── GetWorkspaceSize()                // 计算 workspace 大小
      └── GenTilingKey()                    // 生成模板 TilingKey
```

#### 4.3.2 关键切分参数

| 参数 | 值 | 说明 |
|------|-----|------|
| s1BaseSize（mBaseSize） | 64 | S1 方向基本块大小（Cube 侧 M 维度） |
| s2BaseSize | 128 | S2 方向基本块大小（Cube 侧 N 维度） |
| CV_RATIO | 2 | AIC:AIV = 1:2，每个 AIC 对应 2 个 AIV（subBlock0/1） |
| PRELOAD_NUM | 2 | 主循环尾部额外 2 轮用于排空流水 |

#### 4.3.3 TilingData 结构体

```
KvQuantSparseFlashAttentionPioneerTilingDataMla
├── baseParams (KvQuantSparseFlashAttentionPioneerBaseParamsMla)
│   ├── batchSize, seqSize, qSeqSize       // B, S2, S1
│   ├── blockSize, keyBlockStride           // PA 参数
│   ├── maxBlockNumPerBatch                 // PA 最大 block 数
│   ├── scaleValue                          // 缩放系数
│   ├── nNumOfQInOneGroup                   // GQA 分组数（Q_N / KV_N）
│   ├── actualLenDimsQ, actualLenDimsKV     // actual seq len 维度
│   ├── outputLayout                        // 输出布局枚举
│   ├── sparseMode, sparseBlockSize, sparseBlockCount  // 稀疏参数
│   ├── dSizeVInput                         // KV 输入 D 维度（656）
│   └── isActualLenDimsNull, isActualLenDimsKVNull     // 标志位
│
├── splitKVParams (KvQuantSparseFlashAttentionPioneerSplitKVParamsMla)
│   ├── s2                                  // S2 切分份数
│   ├── accumOutSize                        // FD workspace 累加输出大小
│   └── logSumExpSize                       // FD workspace LSE 大小
│
├── singleCoreParams (KvQuantSparseFlashAttentionPioneerSingleCoreParamsMla)
│   └── usedCoreNum                         // 实际使用的核数
│
├── singleCoreTensorSize (KvQuantSparseFlashAttentionPioneerSingleCoreTensorSizeMla)
│   ├── mmResUbSize                         // BMM1 结果 UB 大小
│   └── bmm2ResUbSize                       // BMM2 结果 UB 大小
│
└── innerSplitParams (KvQuantSparseFlashAttentionPioneerInnerSplitParams)
    ├── mBaseSize                           // M 方向基本块大小
    └── s2BaseSize                          // S2 方向基本块大小
```

#### 4.3.4 TilingKey 模板参数

TilingKey 由 5 个模板参数编码：

| 参数 | 取值 | 说明 |
|------|------|------|
| HAS_SINK | 0/1 | 是否有 sink token |
| PAGE_ATTENTION | 0/1 | 是否 PA 模式 |
| LAYOUT_T | 0(BSND)/1(TND) | Query 布局 |
| KV_LAYOUT_T | 0(BSND)/1(TND)/2(PA_BSND) | KV 布局 |
| TEMPLATE_MODE | 1(V_TEMPLATE) | 模板模式，当前仅支持 V_TEMPLATE |

支持 8 种合法组合（见 `_template_tiling_key.h`）。

---

## 5. Kernel 侧逻辑

### 5.1 整体架构：AIC/AIV 异构三级流水

```
┌─────────────────────────────────────────────────────────────────┐
│                        NPU AI Core (C310)                       │
│                                                                 │
│  ┌──────────────────┐    SSBuf     ┌──────────────────────┐    │
│  │   AIC (Cube核)    │◄──────────►│   AIV (Vector核×2)    │    │
│  │                   │  CVSharedParams │  subBlock0/1      │    │
│  │  BMM1: Q×K^T      │  CrossCore  │  Vec0: KV搬运+反量化  │    │
│  │  BMM2: P×V        │  同步通道    │  Vec1: Softmax+Cast  │    │
│  └────────┬──────────┘             │  Vec2: FlashUpdate   │    │
│           │                        └──────────┬───────────┘    │
│           ▼                                   ▼                │
│  ┌─────────────────────────────────────────────────────┐       │
│  │  L1 Buffer (512KB)                                   │       │
│  │  ├── l1QBuffers: Q 左矩阵 (64×576×2B)               │       │
│  │  └── l1RightBuffers: KV 右矩阵 (128×576×2B) ×3buf   │       │
│  ├─────────────────────────────────────────────────────┤       │
│  │  L0A/L0B (64KB each), L0C (256KB)                   │       │
│  │  ├── mmL0ABuffers: Matmul 左操作数 (DB, 16KB×2)      │       │
│  │  ├── mmL0BBuffers: Matmul 右操作数 (DB, 32KB×2)      │       │
│  │  └── mmL0CBuffers: Matmul 结果 (DB, 128KB×2)        │       │
│  ├─────────────────────────────────────────────────────┤       │
│  │  UB Buffer                                           │       │
│  │  ├── bmm1Buffers: BMM1 结果 (DB, 核间同步)           │       │
│  │  ├── bmm2Buffers: BMM2 结果 (SingleBuf, 核间同步)    │       │
│  │  ├── stage0InQue: KV 搬入 (2buf, 672×16×1B)         │       │
│  │  ├── stage0OutQue: 反量化输出 (2buf, 576×17×2B)      │       │
│  │  ├── stage1OutQue: Softmax 输出 (2buf, 65×128×2B)   │       │
│  │  ├── stage2OutBuf: Vec2 累加输出 (32×512×4B)         │       │
│  │  ├── softmax{Max,Sum,Exp}Buf: 各 256B ×2            │       │
│  │  └── commonTBuf: 512B 临时空间                       │       │
│  └─────────────────────────────────────────────────────┘       │
│                          │                                      │
│                          ▼                                      │
│  ┌─────────────────────────────────────────────────────┐       │
│  │              Global Memory (GM / HBM)                │       │
│  │  query, key, value, sparse_indices, block_table,     │       │
│  │  actual_seq_lengths_*, key_sink, value_sink,          │       │
│  │  attention_out, workspace                             │       │
│  └─────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Kernel 入口（kv_quant_sparse_flash_attention_pioneer.cpp）

`__global__` 函数签名：

```cpp
extern "C" __global__ __aicore__ void kv_quant_sparse_flash_attention_pioneer(
    GM_ADDR query, GM_ADDR key, GM_ADDR value,
    GM_ADDR sparseIndices, GM_ADDR keyScale, GM_ADDR valueScale,
    GM_ADDR blockTable, GM_ADDR actualSeqLengthsQ, GM_ADDR actualSeqLengths,
    GM_ADDR keySink, GM_ADDR valueSink,
    GM_ADDR attentionOut, GM_ADDR workspace, GM_ADDR tiling)
```

模板参数展开：9 个模板参数通过 `TEMPLATE_INTF` 宏定义：

| 模板参数 | 类型 | 说明 |
|---------|------|------|
| Q_T | typename | Query 数据类型（half / bfloat16_t） |
| KV_T | typename | KV 数据类型（int8_t / float8_e4m3 / hifloat8） |
| T | typename | 中间计算类型（float） |
| OUTPUT_T | typename | 输出数据类型（half / bfloat16_t） |
| hasSink | bool | 是否有 sink token |
| isPa | bool | 是否 PageAttention 模式 |
| LAYOUT_T | QSFA_LAYOUT | Query 布局枚举 |
| KV_LAYOUT_T | QSFA_LAYOUT | KV 布局枚举 |
| TEMPLATE_MODE | QSFATemplateMode | 模板模式（当前仅 V_TEMPLATE） |

AIC/AIV 实例化方式：
- 通过 `ASCEND_IS_AIC` / `ASCEND_IS_AIV` 编译期分支，AIC 使用 `QSFAMatmulService`（真实 Cube 服务），AIV 使用 `QSFAMatmulServiceDummy`（空实现）
- 反之 AIV 使用 `QSFAVectorService`（真实 Vector 服务），AIC 使用 `QSFAVectorServiceDummy`（空实现）
- 主 Kernel 类 `KvQuantSparseFlashAttentionPioneerMla<CubeBlockType, VecBlockType>` 通过模板参数接收两种服务

### 5.3 主 Kernel 类（kv_quant_sparse_flash_attention_pioneer_kernel_mla.h）

#### 5.3.1 初始化流程 Init()

```
Init()
  ├── 1. 获取核 ID 和 subBlockIdx
  │     AIC: aicIdx = GetBlockIdx(), aivIdx = 0
  │     AIV: aivIdx = GetBlockIdx(), aicIdx = aivIdx >> 1
  │
  ├── 2. 设置基本块大小
  │     s1BaseSize = 64, s2BaseSize = 128
  │
  ├── 3. AIV 初始化 VecBlock
  │     vecBlock.InitVecBlock() → 解析 TilingData → 填充 CVSharedParams
  │     → 通过 SSBuf 将 CVSharedParams 传递给 AIC
  │     → CrossCoreSetFlag(15) 通知 AIC
  │
  ├── 4. AIV 清零输出（needInit 时）
  │     vecBlock.CleanOutput() → InitOutput 将 attentionOut 清零
  │
  ├── 5. 初始化 BMM 结果 Buffer
  │     InitMMResBuf():
  │     ├── L1: 512KB, l1RightBuffers (3buf, 128×576×2B)
  │     ├── UB: bmm1Buffers (DB, 32×128×4B), bmm2Buffers (Single, 32×512×4B)
  │     └── 设置 CrossCore 标志
  │
  ├── 6. AIC 初始化 CubeBlock
  │     cubeBlock.InitCubeBlock() → 设置 queryGm
  │     → CrossCoreWaitFlag(15) 等待 AIV 的 CVSharedParams
  │     → 从 SSBuf 读取 CVSharedParams
  │
  ├── 7. 计算编译期常量 ComputeConstexpr()
  │     从 CVSharedParams 展开所有轴乘积到 ConstInfo
  │
  ├── 8. 初始化 GM Tensor
  │     InitGlobalBuffer(): 设置 actualSeqQlenAddr, actualSeqKvlenAddr
  │
  ├── 9. 分核计算 InitCalcParamsEach()
  │     按 S1 维度均衡分核，计算每个核的 [bN2Start, gS1Start] ~ [bN2End, gS1End]
  │
  └── 10. 初始化 Local Buffer
       InitLocalBuffer() → vecBlock.InitLocalBuffer()
```

#### 5.3.2 分核策略 InitCalcParamsEach()

分核在 Kernel 侧动态完成（而非 Host 侧静态分配），核心逻辑：

1. 遍历所有 batch，累加每个 batch 的 `actualSeqQ` 得到 `totalBaseNum`
2. 均分：`avgBaseNum = ceil(totalBaseNum / coreNum)`
3. 每个核负责 `[currCoreIdx * avgBaseNum, (currCoreIdx+1) * avgBaseNum)` 范围内的基本块
4. 通过 `GetAxisStartIdx()` 处理核间边界：上一个核的 End 位置 +1 = 当前核的 Start

#### 5.3.3 主循环 ProcessMainLoop()

```
ProcessMainLoop()
  ├── 适配分核左闭右开区间
  │
  ├── for bnIdx in [bN2Start, bN2End):                    // BN 外层循环
  │   ├── ComputeParamBatch()                              // 获取 actualS1/S2
  │   ├── ComputeS1LoopInfo()                              // 计算 S1 循环范围
  │   │
  │   ├── for gS1Index in [gs1LoopStart, gs1LoopEnd + 2):  // S1 循环（+2 排空流水）
  │   │   ├── ComputeParamS1() → ComputeSouterParam()      // 计算 S1 方向参数
  │   │   ├── ComputeS2LoopInfo()                           // 计算 S2 循环范围
  │   │   │
  │   │   ├── for s2LoopCount in [0, s2LoopEnd]:            // S2 内层循环
  │   │   │   │
  │   │   │   ├── ── taskId 对应的当前轮 ──
  │   │   │   │   AIC: cubeBlock.IterateBmm1()              // BMM1: Q × K^T → L0C → UB
  │   │   │   │   AIV: vecBlock.ProcessVec0()               // Vec0: GM→UB→反量化→L1
  │   │   │   │
  │   │   │   ├── ── taskId-1 对应的上一轮 ──
  │   │   │   │   AIV: vecBlock.ProcessVec1()               // Vec1: Softmax + Cast → L1
  │   │   │   │   AIC: cubeBlock.IterateBmm2()              // BMM2: P × V → L0C → UB
  │   │   │   │
  │   │   │   └── ── taskId-2 对应的上上轮 ──
  │   │   │       AIV: vecBlock.ProcessVec2()               // Vec2: FlashUpdate → GM
  │   │   │
  │   │   └── ++multiCoreInnerIdx
  │   └── gS1StartIdx = 0  // 非首个 BN 从 0 开始
  └── end
```

三级流水时序图：

```
taskId:    0          1          2          3          4       ...
AIC:    BMM1[0]    BMM1[1]    BMM1[2]    BMM1[3]    (排空)
                   BMM2[0]    BMM2[1]    BMM2[2]    BMM2[3]
AIV:    Vec0[0]    Vec0[1]    Vec0[2]    Vec0[3]    (排空)
                   Vec1[0]    Vec1[1]    Vec1[2]    Vec1[3]
                              Vec2[0]    Vec2[1]    Vec2[2]
```

关键设计：
- `RunInfo runInfo[3]`：3 份运行参数，通过 `taskId % 3` 轮转，支撑三级流水中 3 个阶段同时持有不同轮次的参数
- 尾部 `PRELOAD_NUM=2` 额外循环：`notLastTwoLoop=false` 时跳过 Vec0/BMM1，仅执行 Vec1/BMM2 和 Vec2，排空流水线
- `hasSink` 场景：S2 循环多一轮（`s2LoopEndIdx += 1`），第 0 轮处理 sink token

### 5.4 Cube 服务（kv_quant_sparse_flash_attention_pioneer_service_cube_mla.h）

#### 5.4.1 类结构

`QSFAMatmulService` 负责 AIC 侧的两次矩阵乘法（BMM1/BMM2）。

编译期常量：

| 常量 | 值 | 说明 |
|------|-----|------|
| s1BaseSize | 64 | M 方向基本块 |
| s2BaseSize | 128 | N 方向基本块 |
| dBaseSize | 576 | D 维度（nope 512 + rope 64） |
| dBaseMatmulSize | 128 | Matmul K 方向切分粒度 |
| sinkNum | 128 | Sink token 数量 |

Buffer 分配：

| Buffer | 类型 | 大小 | 说明 |
|--------|------|------|------|
| l1QBuffers | SingleBuffer/L1 | 64×576×2B = 72KB | Q 左矩阵，S2 循环内复用 |
| mmL0ABuffers | DB/L0A | 16KB×2 | Matmul 左操作数 |
| mmL0BBuffers | DB/L0B | 32KB×2 | Matmul 右操作数 |
| mmL0CBuffers | DB/L0C | 128KB×2 | Matmul 结果 |

#### 5.4.2 IterateBmm1QSFA — Q × K^T

```
IterateBmm1QSFA(outputBuf, inputRightBuf, runInfo, constInfo)
  ├── 1. 加载 Q 左矩阵到 L1
  │     if s2LoopCount == 0:
  │       CopyToL1Nd2Nz(l1QBuffers, queryGm[offset], mRealSize, dSize=576, mm1Ka)
  │       if hasSink: CopyToL1Nd2Nz(inputRightBuf, keySinkGm, 128, dSize, mm1Ka)
  │     else:
  │       复用 l1QBuffers.GetPre()（Q 在 S2 循环内驻留）
  │
  ├── 2. 等待 Vec0 完成 KV 搬运
  │     inputRightBuf.WaitCrossCore()  // 核间同步
  │
  ├── 3. 执行 BMM1 矩阵乘
  │     MatmulK<Q_T, Q_T, float, 64, 128, 128, MK, KN>(
  │       left=l1QBuffers,     // [M, 576] Q
  │       right=inputRightBuf, // [128, 576]^T KV (NZ 格式)
  │       result=mm1ResL0C)    // [M, 128] float
  │     K 方向按 128 切分，循环累加
  │
  ├── 4. Fixpipe L0C → UB
  │     FixpipeParamsC310<ROW_MAJOR>:
  │       nSize = Align8(s2RealSize)    // 128
  │       mSize = Align2(mRealSize)     // ≤64
  │       dualDstCtl = 1                // 双目标模式，按 M/2 拆分到两个 subBlock
  │       dstStride = s2BaseSize(128)   // ND 布局
  │
  └── 5. outputBuf.SetCrossCore()  // 通知 Vec1 BMM1 结果就绪
```

#### 5.4.3 IterateBmm2QSFA — P × V

```
IterateBmm2QSFA(outputBuf, inputLeftBuffers, inputRightBuf, runInfo, constInfo)
  ├── 1. inputRightBuf.WaitCrossCore()  // 等待 Vec1 完成 Softmax+Cast
  │
  ├── 2. 执行 BMM2 矩阵乘
  │     MatmulN<Q_T, Q_T, float, 64, 128, 128, MK, KN>(
  │       left  = inputRightBuf[s2BaseSize * dSizeNope],  // P 矩阵（rope 位置）
  │       right = inputRightBuf[0],                        // V 矩阵（nope 位置）
  │       result = mm2ResL0C)                              // [M, 512] float
  │     注意：P 和 V 共享同一份 L1 Buffer，通过偏移区分
  │
  ├── 3. inputRightBuf.SetCrossCore()  // 释放 KV L1 Buffer 供下一轮 Vec0 使用
  │
  ├── 4. Fixpipe L0C → UB
  │     nSize = Align8(dSizeNope=512)
  │     mSize = Align2(mRealSize)
  │     dualDstCtl = 1
  │     dstStride = Align16(dSizeNope)
  │
  └── 5. outputBuf.SetCrossCore()  // 通知 Vec2 BMM2 结果就绪
```

#### 5.4.4 Dummy 服务与 Traits 系统

- `QSFAMatmulServiceDummy`：空实现，AIV 侧实例化时使用，所有方法为空函数体
- `CubeBlockTraits<T>` + `ARGS_TRAITS` 宏：通过模板特化将 Cube 服务的类型和常量暴露给主 Kernel 类，实现编译期多态

### 5.5 Vector 服务（kv_quant_sparse_flash_attention_pioneer_service_vector_mla.h）

#### 5.5.1 类结构

`QSFAVectorService` 负责 AIV 侧的三阶段向量处理（Vec0/Vec1/Vec2）。

编译期常量：

| 常量 | 值 | 说明 |
|------|-----|------|
| s1BaseSize | 64 | M 方向基本块 |
| s2BaseSize | 128 | S2 方向基本块 |
| dVTemplateType | 512 | V 输出 D 维度（nope） |
| dTemplateAlign64 | 512 | dVTemplateType 64 对齐 |
| dVTemplateTypeInput | 672 | KV 输入 D 维度（656→672 对齐） |
| vec1Srcstride | 33 | Vec1 输出 stride = s1BaseSize/2 + 1 |

Buffer 分配：

| Buffer | 类型 | 大小 | 说明 |
|--------|------|------|------|
| stage0InQue | TQue/VECIN/2buf | 672×16×1B ×2 | Vec0 KV 搬入（每次 16 行） |
| stage0OutQue | TQue/VECOUT/2buf | 576×17×2B ×2 | Vec0 反量化输出（+1 解 bank conflict） |
| stage1OutQue[2] | TQue/VECOUT/1buf ×2 | 33×128×2B | Vec1 Softmax+Cast 输出 |
| stage2OutBuf | TBuf | 32×512×4B = 64KB | Vec2 FlashUpdate 累加输出 |
| softmaxMaxBuf[2] | TBuf | 256B ×2 | Softmax max 值 |
| softmaxSumBuf[2] | TBuf | 256B ×2 | Softmax sum 值 |
| softmaxExpBuf[2] | TBuf | 256B ×2 | Softmax exp(max) 值 |
| commonTBuf | TBuf | 512B | 临时空间 |

#### 5.5.2 ProcessVec0 — KV 搬运 + 反量化

```
ProcessVec0(outputL1, runInfo, constInfo)
  ├── outputL1.WaitCrossCore()  // 等待上一轮 BMM2 释放 L1
  │
  ├── if hasSink && s2LoopCount == 0:
  │     直接 SetCrossCore() 返回（sink 由 Cube 侧在 BMM1 中处理）
  │
  └── ProcessSparseKv(outputL1, runInfo, constInfo)
      ├── 计算 subBlock 分工：
      │     s2Loops = ceil(ceil(s2RealSize / 2) / 2)
      │     subBlock0: [0, s2Loops*2)
      │     subBlock1: [s2Loops*2, s2RealSize)
      │
      ├── while (s2 < s2End && !meetEnd):
      │   ├── AllocTensor(stage0InQue)
      │   ├── 内层循环（每次填满 16 行或遇到 -1）:
      │   │   ├── GetRealCmpS2Idx(token0, token1)  // 从 GM 读 sparse_indices
      │   │   │   topkBS1Idx = boIdx * s1Size * sparseBlockCount + s1oIdx * sparseBlockCount
      │   │   │   token0 = SparseIndicesGm[topkBS1Idx + s2IdxInBase + cmpS2LoopCnt * s2BaseSize]
      │   │   │   token1 = SparseIndicesGm[topkBS1Idx + s2IdxInBase + 1]
      │   │   │   若 index >= sparseBlockCount 则返回 -1（无效）
      │   │   │
      │   │   ├── CopyInKvSparse(kvInUb, startRow, token0, token1)
      │   │   │   ├── GetkeyOffset(): 计算 GM 偏移
      │   │   │   │   PA: blockTable[boIdx * maxBlockNum + blkIdx] * keyBlockStride + blkOffset * dSizeVInput
      │   │   │   │   BSND: (boIdx * s2Size + s2Idx) * dSizeVInput
      │   │   │   │   TND: (batchKvStart + s2Idx) * dSizeVInput
      │   │   │   ├── 尝试合并 2 行为 1 条 DataCopyPad（stride 合法时）
      │   │   │   │   blockLen = 672B（combine 模式），rightPadding 对齐到 32B
      │   │   │   └── stride 溢出/负数时退化为 2 条 CopyInSingleKv
      │   │   └── s2 += 2
      │   │
      │   ├── DequantKv(kvDequantOutUb, kvInUb, dealRow)
      │   │   ├── AntiquantVFFp8D448(): VF 反量化
      │   │   │   对 nope 512 维度按 tile=128 循环：
      │   │   │     LoadAlign FP8 数据 → Cast FP8→FP32 → Mul scale → Cast FP32→FP16/BF16
      │   │   │     DeInterleave + StoreAlign（blockStride=17 解 bank conflict）
      │   │   └── Copy rope 部分（64 维）到 NZ 格式输出的 rope 位置
      │   │
      │   └── CopyOutKvUb2L1(outputL1, kvDequantOutUb)
      │       DataCopy UB→L1，NZ 格式，blockCount=(512+64)/16=36
      │
      └── outputL1.SetCrossCore()  // 通知 AIC BMM1 右矩阵就绪
```

#### 5.5.3 ProcessVec1 — Softmax + Cast

```
ProcessVec1(outputBuf, bmm1ResBuf, runInfo, constInfo)
  ├── bmm1ResBuf.WaitCrossCore()  // 等待 AIC BMM1 结果
  │
  ├── 获取 Softmax 状态 Buffer：
  │     sumUb = softmaxSumBuf[multiCoreIdxMod2]
  │     maxUb = softmaxMaxBuf[multiCoreIdxMod2]
  │     expUb = softmaxExpBuf[taskIdMod2]
  │
  ├── Softmax VF 路由（6 条路径）：
  │     ┌─────────────────┬──────────────────────────────────────┐
  │     │ s2RealSize       │ 首次 (NoUpdate)    │ 非首次 (Update)  │
  │     ├─────────────────┼────────────────────┼──────────────────┤
  │     │ == 128           │ Impl128 NoUpdate   │ Impl128 Update   │
  │     │ ≤ 64             │ Impl64 NoUpdate    │ Impl64 Update    │
  │     │ 64 < x < 128     │ General128 NoUpdate│ General128 Update│
  │     └─────────────────┴────────────────────┴──────────────────┘
  │     输入：BMM1 结果 (float)，输出：Cast 后的 FP16/BF16 (NZ 格式)
  │     内部完成：Muls(scale) → RowMax → Sub → Exp → RowSum → Div → Cast
  │
  ├── bmm1ResBuf.SetCrossCore()  // 释放 BMM1 UB
  │
  ├── DataCopy Cast 结果 → L1（BMM2 左矩阵 P）
  │     dst = outputBuf[s2BaseSize * dSizeV]  // L1 中 P 矩阵位置（V 之后）
  │     blockCount = s2BaseSize / 16 = 8
  │     blockLen = halfMRealSize
  │
  ├── outputBuf.SetCrossCore()  // 通知 AIC BMM2 左矩阵就绪
  │
  └── if s2LoopCount != 0:
        QSFAUpdateExpSumAndExpMax()  // 更新 Online Softmax 状态
          expMax = FusedExpSub(inMax, max)  // exp(old_max - new_max)
          sum = expMax * old_sum + new_sum
```

#### 5.5.4 ProcessVec2 — Flash Update + 输出

```
ProcessVec2(bmm2ResBuf, runInfo, constInfo)
  ├── bmm2ResBuf.WaitCrossCore()  // 等待 AIC BMM2 结果
  │
  ├── WaitFlag<MTE3_V>(mte3ToVId[0])  // 等待上一轮输出搬运完成
  │
  ├── Flash Update 分支：
  │     if s2LoopCount == 0（首次）:
  │       DataCopy(vec2ResUb, mmRes)  // 直接拷贝，无需累加
  │     elif s2LoopCount < s2LoopLimit（中间轮次）:
  │       FlashUpdateNew: dst = pre * expMax + cur
  │         逐行广播 expMax，D 方向按 64 元素循环
  │     else（最后一轮）:
  │       FlashUpdateLastNew: dst = (pre * expMax + cur) / expSum
  │
  ├── bmm2ResBuf.SetCrossCore()  // 释放 BMM2 UB
  │
  ├── if s2LoopCount == s2LoopLimit（最后一轮）:
  │     if s2LoopCount == 0（仅一轮场景）:
  │       LastDivNew: dst = cur / expSum  // 单独除法
  │     CopyOutAttentionOut → Bmm2DataCopyOut:
  │       Cast(float → OUTPUT_T, CAST_ROUND)
  │       DataCopyPad(attentionOutGm[offset], attenOut,
  │         blockLen = dSizeV * sizeof(OUTPUT_T),  // 512 × 2B
  │         blockCount = vec2MRealSize,
  │         srcStride = (dTemplateAlign64 - dSizeV) >> 4,
  │         dstStride = attentionOutStride)
  │
  └── SetFlag<MTE3_V>(mte3ToVId[0])  // 标记输出搬运完成
```

#### 5.5.5 InitCubeVecSharedParams — CVSharedParams 初始化

AIV subBlock0 负责将 TilingData 中的关键参数填充到 `CVSharedParams` 结构体，并通过 SSBuf 传递给 AIC：

1. 从 `tilingData->baseParams` 提取 bSize、gSize、s1Size、s2Size 等
2. 硬编码 MLA 参数：dSize=576、dSizeRope=64、softmaxScale=SOFTMAX_SCALE
3. 遍历所有 batch 判断 `needInit`：若任一 batch 的 s1Size > s2Size，则需要初始化输出为 0
4. 通过 `reinterpret_cast<__ssbuf__ uint32_t*>(0)` 逐字写入 SSBuf
5. `CrossCoreSetFlag(15)` 通知 AIC 读取

### 5.6 微核函数（VF）

#### 5.6.1 Softmax 路由（vf_mul_sel_softmaxflashv2_cast_nz_qsfa.h）

`ProcessVec1Vf` 是 Softmax 的统一入口，通过两个维度分派到 6 个特化实现：

| 维度 | 取值 | 说明 |
|------|------|------|
| isUpdate | false / true | 首次迭代（需初始化 max/sum）vs 非首次（需更新） |
| OriginNRange | EQ_128_QSFA | s2RealSize == 128，最优路径 |
| | GT_0_AND_LTE_64_QSFA | s2RealSize ≤ 64，尾块场景 |
| | GT_64_AND_LTE_128_QSFA | 64 < s2RealSize < 128，非对齐场景 |

每个 VF 实现完成：`Muls(scale) → RowMax → Sub(x - max) → Exp → RowSum → Div(x / sum) → Cast(fp32→fp16/bf16) → ND2NZ`

`QSFAUpdateExpSumAndExpMax`：Online Softmax 状态更新 VF
- `expMax = FusedExpSub(inMax, max)` — 计算 exp(old_max - new_max)
- `sum = expMax * old_sum + new_sum` — 更新累积 sum

`DuplicateSumWithR0`：初始化 sum 为 1.0（首次迭代时使用）

#### 5.6.2 Flash Update（vf_flashupdate_new_qsfa.h）

三个 VF 函数，均使用 `__simd_vf__` 修饰：

| 函数 | 公式 | 使用场景 |
|------|------|---------|
| FlashUpdateNew | `dst = pre * expMax + cur` | S2 中间轮次 |
| FlashUpdateLastNew | `dst = (pre * expMax + cur) / expSum` | S2 最后一轮 |
| LastDivNew | `dst = cur / expSum` | S2 仅一轮场景 |

实现细节：
- 逐行处理：外层循环 `i ∈ [0, m)`，广播加载 `expMax[i]`（BRC_B32 模式）
- D 方向分块：内层循环 `j ∈ [0, srcD/64)`，每次处理 64 个 float 元素
- `MulDstAdd` 融合指令：`pre = pre * expMax + cur`（一条指令完成乘加）

#### 5.6.3 FP8 反量化 VF（AntiquantVFImplFp8D448）

Per-tile（tile=128）反量化流程：

```
for j in [0, nopeDim/128):          // nope 512 维度按 128 切分，共 4 轮
  for i in [0, dealRowCount):       // 逐行处理
    1. LoadAlign FP8 数据（UNPACK4_B8 模式，64+64 分两次加载）
    2. LoadAlign FP32 scale（BRC_B32 广播模式，1 个 scale 广播到 64 元素）
    3. Cast FP8 → FP32
    4. Mul FP32 × scale
    5. Cast FP32 → FP16/BF16
    6. DeInterleave 重排
    7. StoreAlign（blockStride=17 解 bank conflict）
```

输入布局：`[row, 672]` = nope(512) + rope(128) + scale(16) → 672 对齐
输出布局：`[row, 576×(16+1)]` = NZ 格式，stride=17 解 bank conflict

### 5.7 KV Cache 参数计算（kv_quant_sparse_flash_attention_pioneer_kvcache.h）

提供主循环中各层级的参数计算函数：

| 函数 | 调用层级 | 功能 |
|------|---------|------|
| GetSingleCoreParam | BN 循环 | 获取当前 batch 的 actualS1/S2Size，计算 preTokens/nextTokens |
| ComputeParamBatch | BN 循环 | 封装 GetSingleCoreParam |
| ComputeS1LoopInfo | BN 循环 | 计算 S1 循环范围 [gs1LoopStart, gs1LoopEnd)，qSNumInOneBlock=1 |
| ComputeSouterParam | S1 循环 | 计算 s1RealSize、mRealSize、halfM/S1、subBlock 偏移 |
| LoopSOuterOffsetInit | S1 循环 | 计算 attentionOutOffset（考虑 TND 前缀和、subBlock 偏移） |
| ComputeParamS1 | S1 循环 | 封装 ComputeSouterParam + LoopSOuterOffsetInit |
| ComputeS2LoopInfo | S1 循环 | 计算 S2 循环范围，sparse_mode 0 全计算 / 3 右下三角 mask |
| ClipSInnerTokenCube | S2 循环 | 裁剪 s2 token 到 [minValue, maxValue] |
| ComputeLastBN | BN 循环 | TND 下检测空 batch（actualSeqQ=0） |
| InitTaskParamByRun | S2 循环 | 将 RunParamStr 参数拷贝到 RunInfo |

sparse_mode 处理差异：
- `sparseMode == 0`（全计算）：s2LineStart=0, s2LineEnd=min(actualS2, sparseBlockCount)
- `sparseMode == 3`（rightDownCausal）：s2LineStart/End 由 cubeSOuterOffset ± preTokens/nextTokens 裁剪得到

---

## 6. 运行时参数结构体（util_regbase.h）

### 6.1 RunInfo — 运行时循环参数

每轮 S2 循环的动态参数，通过 `runInfo[taskId % 3]` 轮转支撑三级流水。

| 字段 | 类型 | 说明 |
|------|------|------|
| boIdx | int64_t | 当前 batch index |
| s1oIdx | int64_t | 当前 S1 index |
| n2oIdx | int64_t | 当前 N2 index |
| goIdx | int64_t | 当前 G index |
| s2LoopCount | int64_t | S2 循环当前 index |
| s2LoopLimit | int64_t | S2 循环上限 |
| s2StartIdx | int64_t | S2 起始位置 |
| s2EndIdx | int64_t | S2 结束位置 |
| s1RealSize | int32_t | S1 方向实际大小 |
| halfS1RealSize | int32_t | S1/2（subBlock 分工后） |
| mRealSize | int32_t | M 方向实际大小 = s1RealSize × gSize |
| halfMRealSize | int32_t | M/2（subBlock 分工后） |
| s2RealSize | int32_t | S2 方向基本块实际长度 |
| s2AlignedSize | int64_t | S2 对齐到 16 后的长度 |
| vec2S1BaseSize | int32_t | Vec2 切分后的 S1 大小 |
| vec2S1RealSize | int32_t | Vec2 切分后的 S1 尾块大小 |
| vec2MBaseSize | int32_t | Vec2 切分后的 M 大小 |
| vec2MRealSize | int32_t | Vec2 切分后的 M 尾块大小 |
| taskId | int64_t | 全局任务 ID |
| taskIdMod2 / taskIdMod3 | uint8_t | taskId 对 2/3 取模（Buffer 轮转） |
| multiCoreInnerIdx | int64_t | 核内循环计数 |
| multiCoreIdxMod2 / Mod3 | uint8_t | 核内循环对 2/3 取模 |
| attentionOutOffset | int64_t | 输出 GM 偏移 |
| actualS1Size / actualS2Size | int32_t | 当前 batch 的有效 Q/KV 长度 |
| preTokensPerBatch / nextTokensPerBatch | int64_t | Causal mask 参数 |
| qSNumInOneBlock | int64_t | 每个基本块的 Q 行数（=1） |
| kvLoopEndIdx | int64_t | KV 循环结束 index |

### 6.2 ConstInfo — 编译期/初始化期常量

Init 阶段从 CVSharedParams 展开，整个 Kernel 生命周期不变。

| 字段组 | 关键字段 | 说明 |
|--------|---------|------|
| 基本块 | s1BaseSize=64, s2BaseSize=128 | M/N 方向基本块 |
| D 维度 | dSize=576, dSizeV=512, dSizeVInput=656, dSizeNope=512, dSizeRope=64 | 各 D 维度 |
| 轴大小 | bSize, n2Size, gSize, s1Size, s2Size | 输入各轴 |
| 轴乘积 | s1D, gS1D, n2GS1D, s2D, n2S2D, ... | 预计算的偏移乘积 |
| Matmul | mm1Ka | BMM1 K 方向跳读参数 |
| 输出 | attentionOutStride | 输出行间 stride |
| 分核 | aivIdx, subBlockIdx, bN2Start/End, gS1Start/End | 核分配范围 |
| 推理 | isActualLenDimsNull, isActualLenDimsKVNull | actualSeqLen 标志 |
| PA | blockSize, keyBlockStride, maxBlockNumPerBatch | PageAttention 参数 |
| Softmax | softmaxScale | 缩放系数 |
| Sparse | sparseMode, sparseBlockCount, sparseBlockSize | 稀疏参数 |

### 6.3 CVSharedParams — 核间共享参数

位域压缩的结构体（≤32B），通过 SSBuf 从 AIV 传递到 AIC：

| 字段 | 位宽 | 说明 |
|------|------|------|
| s1BaseSize, s2BaseSize | 32 | 基本块大小 |
| bSize, n2Size, gSize, s1Size, s2Size | 32 each | 各轴大小 |
| dSize | 10 bit | D 维度（576） |
| dSizeVInput | 12 bit | KV 输入 D（656） |
| needInit | 4 bit | 是否需要初始化输出 |
| layoutType | 4 bit | 布局类型 |
| isActualSeqLengthsNull | 1 bit | Q actualSeqLen 标志 |
| isActualSeqLengthsKVNull | 1 bit | KV actualSeqLen 标志 |
| sparseBlockCount | 32 | 稀疏块数 |
| softmaxScale | 32 (float) | Softmax 缩放系数 |
| dSizeRope | 11 bit | rope 维度 |
| maskMode | 6 bit | sparse_mode |
| tileSize | 8 bit | 量化 tile 大小 |
| blockSize | 12 bit | PA block 大小 |
| maxBlockNumPerBatch | 32 | PA 最大 block 数 |
| keyBlockStride | 32 | PA key block stride |
| usedCoreNum | 32 | 实际使用核数 |

---

## 7. 端到端数据流

```
                              GM (HBM)
  ┌──────────────────────────────────────────────────────────────────┐
  │  query    key(FP8)  value(FP8)  sparse_indices  block_table     │
  │  [B,S1,   [B,S2,    (=key)      [B,S1,1,K]     [B,S2/blk]      │
  │   N,576]   1,656]                                                │
  └────┬─────────┬──────────────────────┬──────────────┬─────────────┘
       │         │                      │              │
       │    ┌────▼──────────────────────▼──────────────▼─────┐
       │    │  AIV Vec0: KV 搬运 + 反量化                      │
       │    │  1. GetRealCmpS2Idx: 读 sparse_indices          │
       │    │  2. GetkeyOffset: 计算 GM 偏移（支持 PA）         │
       │    │  3. DataCopyPad: GM→UB (672B/行, 每次16行)       │
       │    │  4. AntiquantVFFp8D448: FP8→FP32→×scale→FP16    │
       │    │  5. Copy rope 部分到 NZ 输出                      │
       │    │  6. DataCopy: UB→L1 (NZ格式, 576×s2RealSize)    │
       │    └────────────────────┬────────────────────────────┘
       │                         │ L1: KV [s2RealSize, 576] NZ
       │                         │ CrossCore Sync
  ┌────▼─────────────────────────▼────────────────────────────┐
  │  AIC BMM1: Q × K^T                                        │
  │  1. CopyToL1Nd2Nz: GM→L1 (Q, 首次S2循环加载, 后续复用)    │
  │  2. MatmulK: L1→L0A/L0B→L0C, K方向按128切分               │
  │     Q[M,576] × K^T[576,s2] → S[M,s2] (float)             │
  │  3. Fixpipe: L0C→UB (dualDstCtl=1, 按M/2拆分到2个subBlock)│
  └──────────────────────────────┬────────────────────────────┘
                                 │ UB: BMM1结果 S[M,s2] float
                                 │ CrossCore Sync
  ┌──────────────────────────────▼────────────────────────────┐
  │  AIV Vec1: Softmax + Cast                                  │
  │  1. ProcessVec1Vf: Muls→RowMax→Sub→Exp→RowSum→Div→Cast   │
  │     S[halfM,s2] float → P[halfM,s2] FP16/BF16 (NZ格式)   │
  │     路由: 6条路径 (isUpdate × s2RealSize分档)              │
  │  2. DataCopy: UB→L1 (P矩阵, 放在L1的rope偏移位置)         │
  │  3. QSFAUpdateExpSumAndExpMax: 更新 Online Softmax 状态    │
  └──────────────────────────────┬────────────────────────────┘
                                 │ L1: P[halfM,s2] + V[s2,512] NZ
                                 │ CrossCore Sync
  ┌──────────────────────────────▼────────────────────────────┐
  │  AIC BMM2: P × V                                          │
  │  1. MatmulN: L1→L0A/L0B→L0C                               │
  │     P[M,s2] × V[s2,512] → O_partial[M,512] (float)       │
  │     左矩阵P取L1 rope偏移, 右矩阵V取L1 nope偏移            │
  │  2. Fixpipe: L0C→UB (dualDstCtl=1)                        │
  └──────────────────────────────┬────────────────────────────┘
                                 │ UB: BMM2结果 O_partial[M,512] float
                                 │ CrossCore Sync
  ┌──────────────────────────────▼────────────────────────────┐
  │  AIV Vec2: Flash Update + 输出                             │
  │  首次:  O = O_partial                                      │
  │  中间:  O = O_prev × exp(old_max - new_max) + O_partial   │
  │  末次:  O = (O_prev × exp + O_partial) / sum              │
  │  单轮:  O = O_partial / sum                                │
  │                                                            │
  │  Cast float → OUTPUT_T (FP16/BF16)                        │
  │  DataCopyPad: UB→GM (attentionOut)                         │
  └──────────────────────────────┬────────────────────────────┘
                                 │
  ┌──────────────────────────────▼────────────────────────────┐
  │  GM: attention_out [B,S1,Q_N,512] / [Q_T,Q_N,512]         │
  └───────────────────────────────────────────────────────────┘
```

---

## 8. 关键设计总结

| 设计要点 | 实现方式 |
|---------|---------|
| MLA-Absorb 模式 | Q_D=576(nope512+rope64), KV_N=1 共享, 输出 D=512(去 rope) |
| FP8 量化 KV Cache | Combine 模式：KV_D=656(nope512+rope128+scale16), per-tile(128) FP32 scale 内嵌 |
| 离散稀疏访存 | sparse_indices 索引 → GetkeyOffset 计算 GM 偏移 → 2 行合并搬运优化 |
| AIC/AIV 异构 | Cube 核做 BMM1/BMM2, Vector 核做 KV 搬运反量化/Softmax/FlashUpdate |
| 三级流水 | Vec0+BMM1 / Vec1+BMM2 / Vec2 三级并行, RunInfo[3] 轮转 |
| 核间同步 | CrossCore Sync (WaitCrossCore/SetCrossCore) 保证数据依赖 |
| CVSharedParams | 位域压缩 ≤32B, AIV→SSBuf→AIC 传递 Tiling 参数 |
| 动态分核 | Kernel 侧按 actualSeqQ 均衡分配, 支持变长序列 |
| Sink Token | 固定 128 个, S2 循环多一轮, Cube 侧在 BMM1 首轮加载到 L1 |
| Softmax 分档 | 6 条 VF 路径 (isUpdate × s2RealSize), 编译期特化减少运行时分支 |
| Flash Update | VF 实现 O=O_prev×exp+O_cur, 逐行广播 expMax, D 方向按 64 分块 |
| Bank Conflict 规避 | stage0OutQue stride=17 (16+1), 避免 L1/UB bank conflict |
| Buffer 复用 | Q 在 S2 循环内驻留 L1 复用; L1 KV Buffer 3buf 轮转支撑流水 |
| 模板特化 | 9 个模板参数 → 5 维 TilingKey → 8 种合法组合, 编译期消除运行时分支 |
| PA 支持 | block_table 映射 → keyBlockStride 计算物理偏移, 支持非连续 KV Cache |
| 输出初始化 | needInit 标志: 当 s1 > s2 时清零输出, 避免脏数据 |

