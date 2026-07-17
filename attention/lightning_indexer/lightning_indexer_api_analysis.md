# lightning_indexer 算子 —— API 详细代码级解释

## 文件结构总览

```
attention/lightning_indexer/
├── op_host/
│   ├── lightning_indexer_def.cpp              # 算子原型定义
│   ├── lightning_indexer_infershape.cpp        # Shape/DType 推导
│   ├── lightning_indexer_tiling.h              # Tiling 数据结构定义
│   ├── lightning_indexer_tiling.cpp            # Tiling 计算逻辑
│   └── op_api/
│       ├── aclnn_lightning_indexer.h           # ACLNN 接口声明
│       └── aclnn_lightning_indexer.cpp         # ACLNN 接口实现
├── op_kernel/
│   ├── lightning_indexer.cpp                   # Kernel 入口
│   ├── lightning_indexer_common.h              # 公共常量/结构体定义
│   ├── lightning_indexer_template_tiling_key.h # 模板 TilingKey 定义
│   └── arch22/
│       ├── lightning_indexer_kernel.h          # 主 Kernel 逻辑
│       ├── lightning_indexer_service_cube.h    # Cube(矩阵乘)服务
│       ├── lightning_indexer_service_vector.h  # Vector(TopK)服务
│       └── lightning_indexer_vector.h          # Vector 辅助函数
```

---

## 一、算子功能概述

`lightning_indexer` 是一个**稀疏注意力索引算子**,用于为每个 Query token 从上下文 Key 序列中筛选出 Top-$k$ 个最相关的位置索引。其核心计算公式为:

$$Indices = \text{Top-}k\left\{[1]_{1\times g} @ \left[(W @ [1]_{1\times S_k}) \odot \text{ReLU}(Q_{index} @ K_{index}^T)\right]\right\}$$

**直观理解**:对于某个 token 的 Query $Q \in \mathbb{R}^{g \times d}$ 和上下文 Key $K \in \mathbb{R}^{S_k \times d}$:
1. 计算 $Q @ K^T$ 得到相关性分数矩阵 $(g, S_k)$
2. 经过 ReLU 过滤负相关信号
3. 乘以权重 $W$ 并沿 group 方向($g$ 轴)求和,得到该 token 与所有 Key token 的综合相关性向量 $(S_k,)$
4. 取 Top-$k$ 得到稀疏索引

---

## 二、算法详细执行流程(基于 arch22 kernel)

### 整体架构:Cube + Vector 协同计算

该算子采用 **AIC(Cube 矩阵核)+ AIV(Vector 向量核)协同**的架构(`lightning_indexer.cpp:48` `KERNEL_TYPE_MIX_AIC_1_2`,Cube:Vector = 1:2):

```
┌─────────────────────────────────────────────────────────┐
│  Cube 核 (AIC)           │  Vector 核 (AIV ×2)          │
│  计算 Q@K^T + ReLU       │  加权求和 + TopK + 归约输出   │
│  (矩阵乘法 + Fixpipe)    │  (Reduce + Sort + MergeSort) │
└─────────────────────────────────────────────────────────┘
```

### 执行步骤详解

#### 步骤 1:初始化与分核 (`Init` + `SplitCore`)

**`Init()`** (`lightning_indexer_kernel.h:397`):
- 识别当前核是 AIC 还是 AIV(AIV 的 blockIdx 除以 2 得到 aiCoreIdx)
- 初始化 TilingData(从 Host 传入的切分参数)
- 初始化变长序列信息(actualSeqLengths)
- **`SplitCore()`**:将所有计算任务按 (B, N2, gS1, S2) 四级循环切成基本块,均匀分配到各核

**`SplitCore()`** (`lightning_indexer_kernel.h:286`) 的核心逻辑:
- 统计所有 batch 的总基本块数 `GetTotalBaseBlockNum()`
- 按 `totalBlockNum / coreNum` 分配,余数给前面的核各多处理一块
- 每个核得到一个 `SplitCoreInfo{bN2Start, bN2End, gS1Start, gS1End, s2Start, s2End}`,定义其处理的四维区间
- 因果掩码(`attenMaskFlag`)场景下,每个 S1 基本块对应的 S2 有效长度不同,通过 `GetS2BaseBlockNumOnMask()` 动态计算

#### 步骤 2:主计算循环 (`ProcessMain`)

**`ProcessMain()`** (`lightning_indexer_kernel.h:617`) 遍历 `bN2 → gS1 → s2` 三级循环:

```
for bN2LoopIdx in [bN2Start, bN2End]:
    CalcGS1LoopParams()          // 计算当前 batch 的 S1/S2 实际长度、尾块大小
    if actSeqLen == 0:           // 空序列处理
        DealActSeqLenIsZero()
        continue
    for gS1LoopIdx in [gS1Start, gS1LoopEnd]:
        CalcS2LoopParams()       // 计算 S2 循环范围(考虑掩码)
        for s2LoopIdx in [s2Start, s2LoopEnd]:
            ProcessBaseBlock()   // 核心:一个基本块的计算
```

#### 步骤 3:基本块计算 (`ProcessBaseBlock`)

**`ProcessBaseBlock()`** (`lightning_indexer_kernel.h:664`) 是 Cube 和 Vector 协同的核心:

```
// AIC 侧:
CrossCoreWaitFlag(syncV1C1)     // 等待 Vector 上一次完成
matmulService.ComputeMm1(runInfo)  // 执行矩阵乘法 Q@K^T + ReLU
CrossCoreSetFlag(syncC1V1)      // 通知 Vector 可以处理

// AIV 侧:
CrossCoreWaitFlag(syncC1V1)     // 等待 Cube 完成
vectorService.ProcessVec(runInfo)  // 加权 + TopK
CrossCoreSetFlag(syncV1C1)      // 通知 Cube 可以下一次
```

#### 步骤 4a:Cube 侧矩阵乘法 (`ComputeMm1`)

**`ComputeMm1()`** (`lightning_indexer_service_cube.h:143`) 计算 $S = \text{ReLU}(Q @ K^T)$:

1. **数据搬运**:GM → L1(Nd2Nz 转换为分块矩阵布局)
   - `QueryNd2Nz()`:Query 从 GM 搬到 L1,支持 BSND/TND 布局
   - `KeyNd2Nz()` / `KeyNd2NzForPA()`:Key 搬到 L1,PA 场景通过 blockTable 查表
2. **L1 → L0A/L0B**:`LoadQueryToL0a()`(LoadData3D)和 `LoadKeyToL0b()`(LoadData2D)
3. **矩阵乘**:`ComuteL0c()` 调用 `Mmad` 计算 $C = Q_{L0A} \times K_{L0B}^T$
4. **Fixpipe 输出**:`Fixp()` 将 L0C 结果通过 `DataCopyCO12DstParams` 写回 GM 的 workspace,关键参数 `reluPre = 1` **在 Fixpipe 阶段直接执行 ReLU**,输出类型为 float

多 Buffer 流水线优化:`KEY_BUF_NUM=3`、`QUERY_BUF_NUM=2`、`L0_BUF_NUM=2`,通过 Event Flag 实现 MTE2/MTE1/M 多级流水线。

#### 步骤 4b:Vector 侧加权 + TopK (`ProcessVec`)

**`ProcessVec()`** (`lightning_indexer_service_vector.h:265`) 执行:

**(a) 加权求和**:$(W @ [1]_{1\times S_k}) \odot S$ 后沿 $g$ 轴 Reduce
```
for outerGidx in [0, outerG):           // 按 groupInner_=16 切分 g 轴
    CopyIn(mm1Res, weights)              // 从 GM 搬入 Cube 结果和权重
    DoScale(reduceCacheBuf, dbTmpUb, weightsInUb, ...)  // S[g,S2] * W[g] 逐元素乘
    // DoScale 内部同时对 g 轴做部分 Reduce
DoReduce(reduceOutInner, gRedCnt, s2BaseSize_)  // 完成沿 g 轴的最终归约 → (S2,)
```

**(b) TopK 排序**:对每个 S1 token 的相关性向量取 Top-k
- **小规模路径** (`actS1Size <= 4` 且 `sparseCount <= 2048`):
  - `Sort()` 对每块 S2 做**精排**
  - 缓存 4 个基本块后用 `MrgBasicBlock()` 合并
  - 超过 4 块用 `SparseTopK()` 做稀疏筛选
- **大规模路径** (`actS1Size > 4` 或 `sparseCount > 2048`):
  - `SortAll()` 精排每块 S2
  - `MergeSort()` 合并到全局 TopK 缓冲区

**(c) 结果输出**:
- **单核完成场景** (`needCopyOutGm`):S2 只有一个基本块,直接 `Extract` 分离 value 和 index,`CopyOut` 写回 GM
- **多核需要归约场景** (`needCopyWsGm`):将中间 TopK 结果和参数写入 workspace,留待 `ProcessLD` 跨核归约

#### 步骤 5:跨核归约 (`ProcessDecode` → `ProcessLD`)

**`ProcessLD()`** (`lightning_indexer_service_vector.h:530`):当 S2 被切分到多个核时,需要将各核的局部 TopK 合并为全局 TopK:

1. 遍历 S1 行,检查 workspace 中的 `needFd` 标志,找出需要归约的行
2. 读取当前核的尾规约结果 + 后续核的头规约结果
3. 用 `MrgSort4`(4 路归并排序)合并多个 2048 长度的有序列表
4. `Extract` 分离最终的 value/index,写回输出 GM

---

## 三、各 API/函数功能说明

### 3.1 ACLNN 接口层 (`aclnn_lightning_indexer.cpp`)

| API | 功能 |
|-----|------|
| `aclnnLightningIndexerGetWorkspaceSize()` | 第一段接口:入参校验、Shape/_dtype 检查、Tiling 计算、返回 workspace 大小和 executor |
| `aclnnLightningIndexer()` | 第二段接口:分配 workspace 内存、下发 kernel 执行任务 |

### 3.2 算子定义层 (`lightning_indexer_def.cpp`)

| API | 功能 |
|-----|------|
| `LightningIndexer` (OpDef) | 注册算子的输入/输出/属性。Input: query/key/weights/actualSeqLengthsQuery/actualSeqLengthsKey/blockTable;Output: sparseIndices/sparseValues;Attr: layout_query/layout_key/sparse_count/sparse_mode/pre_tokens/next_tokens/return_values |

### 3.3 Shape 推导层 (`lightning_indexer_infershape.cpp`)

| API | 功能 |
|-----|------|
| `InferShapeLightningIndexer()` | 根据 query shape + sparseCount 推导输出 shape。BSND → `[B,S1,N2,K]`,TND → `[T1,N2,K]`。`returnValues=false` 时 sparseValues shape 设为 `[0]` |
| `InferDataTypeLightningIndexer()` | sparseIndices 输出 INT32,sparseValues 输出与 query 相同的 dtype |

### 3.4 Tiling 层 (`lightning_indexer_tiling.h`)

| API | 功能 |
|-----|------|
| `LIInfoParser::ParseAndCheck()` | 解析并校验所有输入参数(shape/dtype/layout/attr),生成 `LITilingInfo` |
| `LightningIndexerTiling::DoTiling()` | 计算 TilingData:usedCoreNum、bSize、s1Size、s2Size、sparseCount、blockSize 等,写入 `LITilingData` |

### 3.5 Kernel 入口 (`lightning_indexer.cpp`)

| API | 功能 |
|-----|------|
| `lightning_indexer()` (`__global__`) | Kernel 入口,根据编译期模板参数(数据类型/PA/Layout)实例化 `LightningIndexerKernel`,调用 `Init()` + `Process()` |

### 3.6 Kernel 主逻辑 (`lightning_indexer_kernel.h`)

| API | 功能 |
|-----|------|
| `Init()` | 初始化 TilingData、变长序列、分核(`SplitCore`)、分配 workspace Buffer(GlobalTensor)、初始化 Cube/Vector Service |
| `SplitCore()` | 将 (B,N2,gS1,S2) 任务空间均匀切分到各核,返回每个核处理的四维区间 |
| `GetTotalBaseBlockNum()` | 统计所有 batch 的总基本块数(考虑掩码和 sparseCount>2048 的特殊路径) |
| `GetS2BaseBlockNumOnMask()` | 因果掩码场景下,计算某个 S1 基本块对应的有效 S2 块数 |
| `Process()` | 总入口:`ProcessMain()` + `ProcessDecode()`;无任务时走 `ProcessInvalid()` |
| `ProcessMain()` | 三级循环(bN2→gS1→s2)驱动基本块计算,Cube/Vector 通过 CrossCore Flag 同步 |
| `ProcessBaseBlock()` | 单个基本块:Cube 做 Mm1,AIV 做 ProcessVec,通过 `syncC1V1`/`syncV1C1` 同步 |
| `ProcessDecode()` | AIV 侧执行跨核归约(`ProcessLD`) |
| `ProcessInvalid()` | 无计算任务时(全空序列),AIV 直接将输出初始化为 -1/-inf |
| `DealActSeqLenIsZero()` | 处理 actualSeqLen=0 的 batch,将对应输出清为无效值(-1) |
| `CalcGS1LoopParams()` | 计算当前 batch 的实际 S1/S2 长度、gS1 循环次数、尾块大小 |
| `CalcS2LoopParams()` | 计算当前 gS1 块的 S2 循环范围(考虑掩码和分核边界) |
| `CalcRunInfo()` | 计算当前基本块的完整运行信息:各 tensor 的 GM offset、实际块大小、首尾标志等 |

### 3.7 Cube 服务层 (`lightning_indexer_service_cube.h`)

| API | 功能 |
|-----|------|
| `ComputeMm1()` | **核心**:执行 $S = \text{ReLU}(Q @ K^T)$。多级流水线:GM→L1(Nd2Nz)→L0A/L0B(LoadData)→Mmad→L0C→Fixpipe(含ReLU)→GM |
| `QueryNd2Nz()` | Query 从 GM 搬到 L1,ND 转 NZ 分块布局 |
| `KeyNd2Nz()` | Key 从 GM 搬到 L1(非 PA 场景) |
| `KeyNd2NzForPA()` | Key 从 GM 搬到 L1(PA 场景),通过 blockTable 查表获取物理地址 |
| `LoadQueryToL0a()` | Query 从 L1 加载到 L0A(LoadData3D,含 FMatrix 设置和 padding) |
| `LoadKeyToL0b()` | Key 从 L1 加载到 L0B(LoadData2D) |
| `ComuteL0c()` | 执行 Mmad 矩阵乘:$C_{L0C} = Q_{L0A} \times K_{L0B}^T$ |
| `Fixp()` | Fixpipe 将 L0C 结果写回 GM,设置 `reluPre=1` 在此阶段执行 ReLU,`nz2ndEn=true` 将 NZ 转 ND 布局 |

### 3.8 Vector 服务层 (`lightning_indexer_service_vector.h`)

| API | 功能 |
|-----|------|
| `ProcessVec()` | **核心**:加权求和 + TopK。流程:CopyIn → DoScale(逐元素乘+部分Reduce) → DoReduce(完成 g 轴归约) → Sort/MergeSort(TopK) → CopyOut/写 workspace |
| `ProcessLD()` | **跨核归约**:读取各核的局部 TopK(workspace),用 MrgSort4 合并为全局 TopK,Extract 分离 value/index 后写回输出 GM |
| `InitBuffers()` | 初始化 UB Buffer:输出队列、排序缓冲、索引缓冲、归约缓冲等;初始化有序索引 `ArithProgression(0,1,...)` 和 TopK 缓冲(-inf,-1) |
| `InitLDBuffers()` | 初始化 LD(Lookup Decode)专用 Buffer:待归并缓冲、临时缓冲、输出 value/index 缓冲 |
| `CleanInvalidOutput()` | 将指定位置的输出清为无效值(indices=-1,values=-inf) |
| `CopyIn()` | 从 GM 搬入 Cube 结果(mm1Res)和权重(weights)到 UB |
| `DoScale()` | 执行 $S[g,S2] \times W[g]$ 逐元素乘,并对 g 轴做部分 Reduce |
| `DoReduce()` | 完成沿 g 轴的最终归约,得到 $(S2,)$ 的相关性向量 |
| `SortAll()` | 对单块 S2 的相关性向量做完整排序 |
| `MergeSort()` | 将当前块的排序结果与全局 TopK 缓冲合并,保持 Top-k |
| `Extract()` | 从 interleaved 的 (value, index) 对中分离出纯 value 和纯 index |
| `CopyOut()` | 将结果从 UB 搬回输出 GM |

---

## 四、各 API 代码详细解释

### 4.1 ACLNN 接口层

#### 4.1.1 `aclnnLightningIndexerGetWorkspaceSize()`

**文件**: `op_host/op_api/aclnn_lightning_indexer.cpp:85-126`

```cpp
aclnnStatus aclnnLightningIndexerGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *weights,
    const aclTensor *actualSeqLengthsQueryOptional, const aclTensor *actualSeqLengthsKeyOptional,
    const aclTensor *blockTableOptional, char *layoutQueryOptional, char *layoutKeyOptional,
    int64_t sparseCount, int64_t sparseMode, int64_t preTokens, int64_t nextTokens,
    bool returnValues, const aclTensor *sparseIndicesOut, const aclTensor *sparseValuesOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
```

**功能**: 两段式接口第一段。完成入参校验后,调用内部实现 `aclnnInnerLightningIndexerGetWorkspaceSize` 执行 Tiling 计算,返回 workspace 大小和 executor。

**代码逐行解释**:
- **L104-108**: 检查 `query` 非空,获取其 dtype 作为后续 `sparseValuesOut` 的默认 dtype
- **L110-115**: 若 `returnValues=true` 但 `sparseValuesOut==nullptr`,报错返回 `ACLNN_ERR_PARAM_NULLPTR`
- **L116-120**: 使用 `TensorHolder` RAII 包装器处理 `sparseValuesOut` 为空的情况 —— 若用户传 nullptr,自动创建一个 shape=`{0}` 的占位 Tensor(地址设为 `0xff`),使后续 GE 图编译不报空指针
- **L122-125**: 委托给 `aclnnInnerLightningIndexerGetWorkspaceSize`(由 GE 框架自动生成的内部实现),它内部会:
  1. 调用 `TilingForLightningIndexer()` 做 Tiling
  2. 计算 workspace 大小
  3. 构建 `aclOpExecutor`

#### 4.1.2 `aclnnLightningIndexer()`

**文件**: `op_host/op_api/aclnn_lightning_indexer.cpp:128-132`

```cpp
aclnnStatus aclnnLightningIndexer(void *workspace, uint64_t workspaceSize,
                                  aclOpExecutor *executor, const aclrtStream stream)
```

**功能**: 两段式接口第二段。直接委托给 `aclnnInnerLightningIndexer`,由 GE 框架在指定 stream 上异步下发 kernel 任务。workspace 由调用方在 device 侧已分配。

---

### 4.2 算子原型定义层

#### 4.2.1 `LightningIndexer` (OpDef)

**文件**: `op_host/lightning_indexer_def.cpp:19-80`

```cpp
class LightningIndexer : public OpDef {
    explicit LightningIndexer(const char *name) : OpDef(name) {
        this->Input("query").ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16})
            ...
```

**功能**: 向 GE 图引擎注册算子的输入/输出/属性规格。

**代码逐段解释**:

| 代码段 | 行号 | 解释 |
|--------|------|------|
| `Input("query")` | L23-27 | 必选输入 Q。DataType 列表 4 种组合对应 4 种模板:BF16+BF16、FP16+FP16、BF16+BF16(W=FP32)、FP16+FP16(W=FP32)。`.AutoContiguous()` 表示框架自动做连续化 |
| `Input("key")` | L28-32 | 必选输入 K,DType 与 Q 一致 |
| `Input("weights")` | L33-37 | 必选输入 W。当 Q=BF16/FP16 时 W 可为 FP32(第3/4组合) |
| `Input("actual_seq_lengths_query")` | L38-42 | 可选输入,TND 布局时必传。表示每个 batch 的 Q 有效长度(前缀和) |
| `Input("actual_seq_lengths_key")` | L43-47 | 可选输入,TND/PA_BSND 布局时必传 |
| `Input("block_table")` | L48-52 | 可选输入,PA_BSND 布局时必传。PageAttention 的 block 映射表 |
| `Output("sparse_indices")` | L53-56 | 必选输出,INT32,存储 Top-k 索引 |
| `Output("sparse_values")` | L57-60 | 必选输出,DType 与 Q 一致,存储 Top-k 对应的值 |
| `Attr("layout_query")` | L61 | 默认 "BSND",支持 BSND/TND |
| `Attr("layout_key")` | L62 | 默认 "BSND",支持 BSND/TND/PA_BSND |
| `Attr("sparse_count")` | L63 | 默认 2048,Top-k 的 k 值 |
| `Attr("sparse_mode")` | L64 | 默认 3(下三角因果掩码) |
| `Attr("pre_tokens")` | L65 | 默认 INT64_MAX(当前仅支持此值) |
| `Attr("next_tokens")` | L66 | 默认 INT64_MAX(当前仅支持此值) |
| `Attr("return_values")` | L67 | 默认 false,是否输出 sparseValues |
| `AICore().AddConfig(...)` | L75-77 | 注册支持的芯片:ascend910b(arch22)、ascend910_93(arch22)、ascend950(arch35) |

---

### 4.3 Shape 推导层

#### 4.3.1 `InferShapeLightningIndexer()`

**文件**: `op_host/lightning_indexer_infershape.cpp:31-93`

```cpp
static ge::graphStatus InferShapeLightningIndexer(gert::InferShapeContext *context)
```

**功能**: 根据 query shape + sparseCount 推导输出 shape。

**代码逐段解释**:

- **L35-43**: 获取 query 和 key 的输入 shape,以及两个输出的 shape 指针
- **L47-52**: 获取属性:layout_query、layout_key、sparse_count
- **L55-58**: 校验 layout_query 必须为 "BSND" 或 "TND"

- **L61-69** (BSND 路径):
  ```cpp
  if (inputLayoutQueryPtrStr == "BSND") {
      sparseIndicesShape->SetDim(0, queryShape->GetDim(0)); // B
      sparseIndicesShape->SetDim(1, queryShape->GetDim(1)); // S1
      sparseIndicesShape->SetDim(2, keyShape->GetDim(2));   // N2 (来自 key 的 N 维)
      sparseIndicesShape->SetDim(3, *seleced_count);        // K (sparseCount)
  }
  ```
  输出 shape = `[B, S1, N2, K]`,其中 N2 取自 key 的第 2 维(K 的多头数,固定为 1)

- **L70-79** (TND 路径):
  ```cpp
  sparseIndicesShape->SetDim(0, queryShape->GetDim(0));   // T1
  int32_t nDimIndex = (inputLayoutKeyPtrStr == "PA_BSND") ? 2 : 1;
  sparseIndicesShape->SetDim(1, keyShape->GetDim(nDimIndex)); // N2
  sparseIndicesShape->SetDim(2, *seleced_count);              // K
  ```
  输出 shape = `[T1, N2, K]`。PA_BSND 时 N2 在 key 的第 2 维,TND 时在第 1 维

- **L81-89** (sparseValues shape):
  ```cpp
  if (returnValueFlag) {
      *sparseValuesShape = *sparseIndicesShape;  // 与 indices 相同
  } else {
      sparseValuesShape->SetDimNum(1);
      sparseValuesShape->SetDim(0, 0);           // 空 tensor [0]
  }
  ```

#### 4.3.2 `InferDataTypeLightningIndexer()`

**文件**: `op_host/lightning_indexer_infershape.cpp:95-107`

```cpp
context->SetOutputDataType(0, outputType);  // sparseIndices = INT32
context->SetOutputDataType(1, context->GetInputDataType(QUERY_INDEX)); // sparseValues = query.dtype
```

---

### 4.4 Tiling 层

#### 4.4.1 `LIInfoParser::ParseAndCheck()` — 参数解析与校验总入口

**文件**: `op_host/lightning_indexer_tiling.cpp:898-933`

```cpp
ge::graphStatus LIInfoParser::ParseAndCheck(LITilingInfo &liInfo) {
    GetOpName();              // 获取算子节点名
    GetNpuInfo();             // 获取平台信息(AIC/AIV 核数、NpuArch)
    GetOpParaInfo();          // 获取输入/输出/属性参数
    CheckRequiredParaExistence();  // 校验必选参数非空
    GetAndCheckInOutDataType();    // 校验 dtype 合法性
    GetQueryKeyAndOutLayout();     // 解析 layout 字符串为枚举
    GetAndCheckOptionalInput();    // 校验可选输入(TND 必传 seq_len, PA 必传 block_table)
    CheckShapeDim();               // 校验各 tensor 维度数
    GetN1Size();                   // 获取 Q 头数 N1
    GetAndCheckN2Size();           // 获取 K 头数 N2(必须为 1)
    GetGSize();                    // 计算 group size = N1/N2
    GetBatchSize();                // 获取 B
    GetS1Size();                   // 获取 S1
    GetHeadDim();                  // 获取 D(必须为 128)
    GetS2Size();                   // 获取 S2(根据 layout 不同方式获取)
    CheckFeatureMlaNoQuantShape(); // A5 专用额外校验
    ValidateInputShapesMatch();    // 校验所有 tensor 间的 shape 一致性
    GenerateInfo(liInfo);          // 填充 LITilingInfo 结构体
}
```

#### 4.4.2 `GetAndCheckAttrParaInfo()` — 属性校验

**文件**: `op_host/lightning_indexer_tiling.cpp:196-267`

关键校验逻辑:
- **L231-237**: 非 PA 场景下,layout_query 必须等于 layout_key
- **L246-252**: `sparse_count` 必须 >0 且 <=8192;>2048 时必须是 1024 的整数倍
- **L253-255**: `sparse_mode` 只支持 0(无掩码)或 3(下三角因果掩码)
- **L256-261**: `preTokens`/`nextTokens` 当前仅支持 INT64_MAX(即不限制)
- **L262-264**: `returnValues=true` 时不允许 `layout_key="PA_BSND"`(推理场景不返回 values)

#### 4.4.3 `GetAndCheckInOutDataType()` — 数据类型校验

**文件**: `op_host/lightning_indexer_tiling.cpp:279-333`

- **L287-297**: Q 和 K 的 dtype 必须一致,且只能是 FP16 或 BF16
- **L298-304** (A5/arch35 特有): Q/K/weights dtype 必须全一致
- **L306-319** (A2/A3/arch22): 若 weights 不是 FP32,则必须与 Q/K 一致;若是 FP32 则直接通过
- **L321-330**: `sparse_indices` 必须 INT32;`sparse_values` 必须与 Q dtype 一致

#### 4.4.4 `GetS2Size()` — S2 长度获取

**文件**: `op_host/lightning_indexer_tiling.cpp:617-630`

```cpp
ge::graphStatus LIInfoParser::GetS2Size() {
    if (kLayout_ == DataLayout::BnBsND) {        // PA_BSND
        return GetS2SizeForPageAttention();       // S2 = maxBlockNumPerBatch * blockSize
    } else if (kLayout_ == DataLayout::TND) {
        s2Size_ = key.shape->GetDim(0);           // T2 = key 的第 0 维
    } else {                                       // BSND
        s2Size_ = key.shape->GetDim(1);           // S2 = key 的第 1 维
    }
}
```

#### 4.4.5 `LightningIndexerTiling::DoTiling()` — Tiling 数据生成

**文件**: `op_host/lightning_indexer_tiling.cpp:975-1021`

```cpp
ge::graphStatus LightningIndexerTiling::DoTiling(LITilingInfo *tilingInfo) {
    // 1. 设置 BlockDim
    uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context_->SetBlockDim(blockDim);          // 总核数 = AIC + AIV

    // 2. 计算 Workspace 大小
    uint64_t workspaceSize = LiCalcWorkspaceSize(ascendcPlatform, tilingInfo->s2Size, aicNum);
    workSpaces[0] = workspaceSize;

    // 3. 填充 TilingData
    tilingData_.set_bSize(tilingInfo->bSize);
    tilingData_.set_s2Size(tilingInfo->s2Size);
    tilingData_.set_s1Size(tilingInfo->s1Size);
    tilingData_.set_sparseCount(tilingInfo->sparseCount);
    tilingData_.set_gSize(tilingInfo->gSize);
    tilingData_.set_blockSize(tilingInfo->blockSize);
    tilingData_.set_maxBlockNumPerBatch(tilingInfo->maxBlockNumPerBatch);
    tilingData_.set_sparseMode(tilingInfo->sparseMode);
    tilingData_.set_usedCoreNum(blockDim);
    // ... 保存到 RawTilingData

    // 4. 设置 TilingKey(编译期模板参数选择)
    uint64_t tilingKey = GET_TPL_TILING_KEY(inputQType, inputKType, outputType,
                                            pageAttentionFlag, inputQLayout, inputKLayout, weightTypeFlag);
    context_->SetTilingKey(tilingKey);
    context_->SetScheduleMode(1);  // batchmode
}
```

#### 4.4.6 `LiCalcWorkspaceSize()` — Workspace 大小计算

**文件**: `op_host/lightning_indexer_tiling.cpp:942-973`

```cpp
// arch22 路径:
liWorkspaceSize += liMm1ResSize * liMm1ResElemSize * liDoubleBuffer * liAicNum;
// mm1Res: 512*512 * 4B(float) * 2(DoubleBuffer) * AIC核数

liWorkspaceSize += 2 * liS1BaseSize * liV1ResElemType * liTopkMaxSize * liV1ResElemSize * liAicNum;
// vec1Res: 2(头尾) * 8(S1Base) * 2(idx+value) * 2048(TopK) * 4B(float) * AIC核数

liWorkspaceSize += 2 * liS1BaseSize * liV1DecodeParamNum * liV1DecodeParamElemSize * liAicNum;
// vec1Param: 2(头尾) * 8(S1Base) * 16(参数个数) * 8B(int64) * AIC核数
```

---

### 4.5 Kernel 入口

#### 4.5.1 `lightning_indexer()` (global kernel)

**文件**: `op_kernel/lightning_indexer.cpp:39-65`

```cpp
template <int DT_Q, int DT_K, int DT_OUT, int PAGE_ATTENTION, int LAYOUT_T, int K_LAYOUT_T, int DT_W_FLAG>
__global__ __aicore__ void lightning_indexer(
    __gm__ uint8_t *query, __gm__ uint8_t *key, __gm__ uint8_t *weights,
    __gm__ uint8_t *actualSeqLengthsQ, __gm__ uint8_t *actualSeqLengths,
    __gm__ uint8_t *blocktable, __gm__ uint8_t *sparseIndices,
    __gm__ uint8_t *sparseValues, __gm__ uint8_t *workspace, __gm__ uint8_t *tiling)
{
    TPipe tPipe;
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);  // Cube:Vector = 1:2

    // 根据 dtype 实例化模板
    if (ORIG_DTYPE_QUERY == DT_BF16) {
        INVOKE_LI_NO_KFC_OP_IMPL(LightningIndexerKernel, bfloat16_t, bfloat16_t, int32_t, ...);
    } else if (ORIG_DTYPE_QUERY == DT_FLOAT16) {
        INVOKE_LI_NO_KFC_OP_IMPL(LightningIndexerKernel, half, half, int32_t, ...);
    }
}
```

**宏 `INVOKE_LI_NO_KFC_OP_IMPL`** (L29-37):
```cpp
#define INVOKE_LI_NO_KFC_OP_IMPL(templateClass, ...)                            \
    templateClass<LIType<__VA_ARGS__>> op;                                      \
    GET_TILING_DATA_WITH_STRUCT(LITilingData, tiling_data_in, tiling);          \
    const LITilingData *__restrict tiling_data = &tiling_data_in;               \
    op.Init(query, key, weights, ..., tiling_data, &tPipe);                     \
    op.Process();
```
从 `tiling` buffer 反序列化 `LITilingData`,构造 Kernel 对象,调用 `Init()` + `Process()`。

---

### 4.6 Kernel 主逻辑 (`lightning_indexer_kernel.h`)

#### 4.6.1 `Init()` — 初始化

**文件**: `op_kernel/arch22/lightning_indexer_kernel.h:397-457`

```cpp
__aicore__ inline void LightningIndexerKernel<LIT>::Init(...) {
    // 1. 区分 AIC/AIV,计算 aiCoreIdx
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx();     // VEC: 0-47
        aiCoreIdx = tmpBlockIdx / 2;     // 2个AIV共享1个AIC
    } else {
        tmpBlockIdx = GetBlockIdx();     // Cube: 0-23
        aiCoreIdx = tmpBlockIdx;
    }

    // 2. 初始化 TilingData
    InitTilingData(tiling);  // 从 tiling buffer 读取 bSize, gSize, s2Size, sparseCount 等

    // 3. 初始化变长序列
    InitActualSeqLen(actualSeqLengthsQ, actualSeqLengths);

    // 4. 分核计算
    SplitCore(aiCoreIdx, usedCoreNum, splitCoreInfo);

    // 5. Workspace 内存布局:
    // |mm1ResGm(DoubleBuffer)|vec1ResGm(LD中间结果)|vec1ParamGm(LD参数)|
    uint64_t singleCoreMm1ResSize = WS_DOUBLE * mBaseSizeAlign * s2BaseSize * sizeof(float);
    mm1ResGm.SetGlobalBuffer(workspace + offset + aiCoreIdx * singleCoreMm1ResSize);
    // ... vec1ResGm, vec1ParamGm

    // 6. 分别初始化 AIC/AIV 的 GlobalTensor
    if ASCEND_IS_AIV {
        indiceOutGm.SetGlobalBuffer(sparseIndices);     // 输出索引
        valueOutGm.SetGlobalBuffer(sparseValues);       // 输出值
        weightsGm.SetGlobalBuffer(weights);             // 权重
        vectorService.InitVec1GlobalTensor(mm1ResGm, vec1ResGm, vec1ParamGm, weightsGm, ...);
    } else {
        queryGm.SetGlobalBuffer(query);
        if constexpr (PAGE_ATTENTION) { blockTableGm.SetGlobalBuffer(blockTable); }
        keyGm.SetGlobalBuffer(key);
        matmulService.InitMm1GlobalTensor(blockTableGm, keyGm, queryGm, mm1ResGm);
    }
    InitBuffers();
}
```

#### 4.6.2 `InitTilingData()` — Tiling 参数解析

**文件**: `op_kernel/arch22/lightning_indexer_kernel.h:157-188`

```cpp
__aicore__ inline void LightningIndexerKernel<LIT>::InitTilingData(...) {
    usedCoreNum = tilingData->usedCoreNum;
    constInfo.batchSize = tilingData->bSize;
    constInfo.qHeadNum = constInfo.gSize = tilingData->gSize;  // gSize = N1/N2
    constInfo.kSeqSize = tilingData->s2Size;
    constInfo.qSeqSize = tilingData->s1Size;
    constInfo.attenMaskFlag = (tilingData->sparseMode == 3);   // sparseMode=3 为因果掩码
    constInfo.kCacheBlockSize = tilingData->blockSize;
    constInfo.sparseCount = tilingData->sparseCount;

    // 关键: 根据 sparseCount 动态调整基本块大小
    constInfo.isSparseCountOver2K = (constInfo.sparseCount <= BASE_TOPK) ? false : true;
    // sparseCount <= 2048: s1BaseSize = 8
    // sparseCount > 2048:  s1BaseSize = 8192 / sparseCount * 2 (如 4096->4, 8192->2)
    constInfo.s1BaseSize = constInfo.isSparseCountOver2K
                         ? SPARSE_COUNT_8K / constInfo.sparseCount * 2 : 8;
    constInfo.mBaseSize = constInfo.s1BaseSize * constInfo.gSize;  // M = S1_base * g
    constInfo.mBaseSizeAlign = Align(constInfo.mBaseSize, 16);     // 对齐到 16(Cube 基本块)
}
```

**设计要点**: 当 sparseCount > 2048 时,减小 s1BaseSize 以控制 UB 内存使用(因为 TopK 缓冲区与 sparseCount 成正比)。

#### 4.6.3 `SplitCore()` — 多核任务切分

**文件**: `op_kernel/arch22/lightning_indexer_kernel.h:286-367`

```cpp
__aicore__ void LightningIndexerKernel<LIT>::SplitCore(uint32_t curCoreIdx,
                                                        uint32_t &coreNum, SplitCoreInfo &info) {
    // 1. 统计总基本块数
    uint32_t totalBlockNum = GetTotalBaseBlockNum();
    // 2. 均匀分配: minBlockPerCore = total/coreNum, 前 deal1MoreBlockCoreNum 个核各多1块
    uint32_t minBlockPerCore = totalBlockNum / coreNum;
    uint32_t deal1MoreBlockCoreNum = totalBlockNum % coreNum;

    // 3. 遍历 (B*N2, gS1, S2) 三维空间,按块分配给各核
    for (uint32_t bN2Idx = 0; bN2Idx < batchSize * kHeadNum; bN2Idx++) {
        GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size);
        for (uint32_t gS1Idx = 0; gS1Idx < s1GBaseNum; gS1Idx++) {
            // 因果掩码: 不同 S1 块的有效 S2 不同
            if (attenMaskFlag) {
                s2BaseNum = GetS2BaseBlockNumOnMask(gS1Idx, actS1Size, actS2Size);
            }
            for (uint32_t s2Idx = 0; s2Idx < s2Loop;) {
                // 找到当前核的起始点
                if (findLastCoreEnd) {
                    info.bN2Start = bN2Idx; info.gS1Start = gS1Idx; info.s2Start = s2Idx;
                    findLastCoreEnd = false;
                }
                // 累积块数达到当前核配额,设置终点
                if (lastGS1RemainBlockCnt + s2RemainBaseNum >= coreDealBlockCnt) {
                    info.bN2End = bN2Idx; info.gS1End = gS1Idx;
                    info.s2End = s2Idx + coreDealBlockCnt - lastGS1RemainBlockCnt - 1;
                    // S2 被切到多核时,只有第一个核需要做 LD 归约
                    if (s2Idx == 0 && info.s2End + 1 < s2BaseNum) {
                        info.isLD = true;
                    }
                    if (coreIdx == curCoreIdx) return;  // 找到当前核的任务
                    coreIdx++; findLastCoreEnd = true;
                }
            }
        }
    }
}
```

#### 4.6.4 `GetS2BaseBlockNumOnMask()` — 因果掩码下 S2 有效块数

**文件**: `op_kernel/arch22/lightning_indexer_kernel.h:244-256`

```cpp
__aicore__ inline uint32_t GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size,
                                                    uint32_t actS2Size) {
    uint32_t s1Offset = s1BaseSize * s1gIdx;                    // 当前 S1 块的起始位置
    int32_t validS2LenBase = (int32_t)actS2Size - (int32_t)actS1Size;  // S2-S1 偏移
    int32_t validS2Len = s1Offset + validS2LenBase + s1BaseSize;       // 当前 S1 块的有效 S2 长度
    validS2Len = Min(validS2Len, (int32_t)actS2Size);                  // 不超过 S2
    validS2Len = Max(validS2Len, 1);                                   // 至少 1
    return CeilDiv(validS2Len, s2BaseSize);                            // 向上取整为块数
}
```

**因果掩码原理**: `rightDownCausal` 模式下,第 i 个 query token 只能 attend 到前 i 个 key token。因此越靠后的 S1 块,有效 S2 越长。

#### 4.6.5 `Process()` — 主入口

**文件**: `op_kernel/arch22/lightning_indexer_kernel.h:569-578`

```cpp
__aicore__ inline void LightningIndexerKernel<LIT>::Process() {
    if (usedCoreNum == 0) {
        ProcessInvalid();   // 全空序列: 直接初始化输出为 -1/-inf
        return;
    }
    ProcessMain();          // 主计算: Cube 矩阵乘 + Vector TopK
    ProcessDecode();        // 跨核归约: AIV 执行 ProcessLD
}
```

#### 4.6.6 `ProcessMain()` — 主计算循环

**文件**: `op_kernel/arch22/lightning_indexer_kernel.h:617-661`

```cpp
__aicore__ inline void LightningIndexerKernel<LIT>::ProcessMain() {
    if (aiCoreIdx >= usedCoreNum) return;  // 无任务核退出

    // AIV 设置跨核同步标志
    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
        CrossCoreSetFlag<FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
        CrossCoreSetFlag<FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
    } else {
        matmulService.AllocEventID();
    }

    LICommon::RunInfo runInfo;
    uint32_t gloop = 0;
    // 三级循环: B*N2 → gS1 → S2
    for (uint32_t bN2LoopIdx = splitCoreInfo.bN2Start; bN2LoopIdx <= splitCoreInfo.bN2End; bN2LoopIdx++) {
        CalcGS1LoopParams(bN2LoopIdx);      // 计算 S1/S2 实际长度、尾块
        if (tempLoopInfo.curActSeqLenIsZero) {
            DealActSeqLenIsZero(...);       // 空序列清理
            continue;
        }
        for (uint32_t gS1LoopIdx = splitCoreInfo.gS1Start; gS1LoopIdx <= gS1LoopEnd; gS1LoopIdx++) {
            CalcS2LoopParams(bN2LoopIdx, gS1LoopIdx);  // S2 循环范围
            for (int s2LoopIdx = splitCoreInfo.s2Start; s2LoopIdx <= s2LoopEnd; s2LoopIdx++) {
                ProcessBaseBlock(gloop, s2LoopIdx, runInfo);  // 核心计算
                ++gloop;
            }
            splitCoreInfo.s2Start = 0;
        }
        // BSND 场景: S1 实际长度 < shape S1 时,清理多余输出
        if (tempLoopInfo.needDealActS1LessThanS1) {
            DealActSeqLenIsZero(..., tempLoopInfo.actS1Size);
        }
        splitCoreInfo.gS1Start = 0;
    }

    // AIC 等待 AIV 完成最后一次
    if ASCEND_IS_AIV { vectorService.FreeEventID(); }
    else {
        matmulService.FreeEventID();
        CrossCoreWaitFlag(constInfo.syncV1C1);
        CrossCoreWaitFlag(constInfo.syncV1C1);
    }
}
```

#### 4.6.7 `ProcessBaseBlock()` — 单基本块 Cube+Vector 协同

**文件**: `op_kernel/arch22/lightning_indexer_kernel.h:664-677`

```cpp
__aicore__ inline void LightningIndexerKernel<LIT>::ProcessBaseBlock(uint32_t loop,
                                                      uint64_t s2LoopIdx, RunInfo &runInfo) {
    CalcRunInfo(loop, s2LoopIdx, runInfo);  // 计算 offset、实际块大小等

    if ASCEND_IS_AIC {
        CrossCoreWaitFlag(constInfo.syncV1C1);        // 等 Vector 上一次完成
        matmulService.ComputeMm1(runInfo);             // Cube: Q@K^T + ReLU
        CrossCoreSetFlag<FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V1);  // 通知 Vector
    } else {
        CrossCoreWaitFlag(constInfo.syncC1V1);        // 等 Cube 完成
        vectorService.ProcessVec(runInfo);             // Vector: 加权+TopK
        CrossCoreSetFlag<FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1); // 通知 Cube
    }
}
```

#### 4.6.8 `CalcRunInfo()` — 运行信息计算

**文件**: `op_kernel/arch22/lightning_indexer_kernel.h:514-566`

```cpp
__aicore__ inline void LightningIndexerKernel<LIT>::CalcRunInfo(...) {
    runInfo.actMBaseSize = tempLoopInfo.actMBaseSize;     // 当前 M 轴实际大小
    runInfo.actualSingleProcessSInnerSize = s2BaseSize;   // S2 块大小

    // 尾块处理
    uint32_t s2SplitNum = CeilDiv(actS2Size, s2BaseSize);
    if (runInfo.s2Idx == s2SplitNum - 1) {
        runInfo.actualSingleProcessSInnerSize = s2BasicSizeTail;  // 尾块实际大小
    }
    runInfo.actualSingleProcessSInnerSizeAlign = Align(..., 32B); // 32B 对齐

    runInfo.isFirstS2InnerLoop = (s2LoopIdx == splitCoreInfo.s2Start);
    runInfo.isLastS2InnerLoop = (s2LoopIdx == tempLoopInfo.s2LoopEnd);
    runInfo.isAllLoopEnd = (bN2Idx==bN2End && gS1Idx==gS1End && s2Idx==s2End);

    // 首个 S2 循环时计算各 tensor 的 GM offset
    if (runInfo.isFirstS2InnerLoop) {
        // TND: 使用 actualSeqLengths 的前缀和计算偏移
        // BSND: 使用 bIdx * seqSize 计算偏移
        queryCoreOffset = tndBIdxOffset + gS1Idx * mBaseSize * headDim;
        keyCoreOffset = tndKeyBIdxOffset + n2Idx * headDim;
        weightsCoreOffset = actualSeqQPrefixSum * qHeadNum + n2Idx * gSize;
        indiceOutCoreOffset = actualSeqQPrefixSum * kHeadNum * sparseCount + n2Idx * sparseCount;
    }
    // Key offset 随 S2 块变化
    runInfo.tensorKeyOffset = keyCoreOffset + s2Idx * s2BaseSize * kHeadNum * headDim;
}
```

#### 4.6.9 `ProcessDecode()` — 跨核归约入口

**文件**: `op_kernel/arch22/lightning_indexer_kernel.h:680-690`

```cpp
__aicore__ inline void LightningIndexerKernel<LIT>::ProcessDecode() {
    if ASCEND_IS_AIV {
        vectorService.InitLDBuffers(pipe);  // 重置 pipe,分配 LD 专用 UB
        ICachePreLoad(LD_PREFETCH_LEN);     // 预取指令缓存
        SyncAll();                          // 全核同步,确保所有 workspace 数据就绪
        if (splitCoreInfo.isLD) {           // 只有 isLD=true 的核执行归约
            vectorService.ProcessLD();
        }
    }
}
```

#### 4.6.10 `ProcessInvalid()` — 空输出初始化

**文件**: `op_kernel/arch22/lightning_indexer_kernel.h:581-614`

```cpp
__aicore__ inline void LightningIndexerKernel<LIT>::ProcessInvalid() {
    if ASCEND_IS_AIV {
        // 总输出大小 = B * S1 * N2 * K
        uint64_t totalOutputSize = batchSize * qSeqSize * kHeadNum * sparseCount;
        // 每个核均分
        uint64_t singleCoreSize = Align(totalOutputSize / aivCoreNum, GM_ALIGN_BYTES / sizeof(OUT_T));
        uint64_t baseSize = tmpBlockIdx * singleCoreSize;
        if (baseSize < totalOutputSize) {
            uint64_t dealSize = Min(singleCoreSize, totalOutputSize - baseSize);
            // 索引初始化为 -1 (INVALID_IDX)
            AscendC::InitGlobalMemory(indiceOutGm[baseSize], dealSize, constInfo.INVALID_IDX);
            // 值初始化为 -inf (0xFC00 for fp16, 0xFF80 for bf16)
            if (constInfo.returnValue) {
                AscendC::InitGlobalMemory(valueOutGm[baseSize], dealSize, negInf);
            }
        }
    }
}
```

---

### 4.7 Cube 服务层 (`lightning_indexer_service_cube.h`)

#### 4.7.1 `ComputeMm1()` — 矩阵乘核心

**文件**: `op_kernel/arch22/lightning_indexer_service_cube.h:143-204`

```cpp
__aicore__ inline void LightningIndexerServiceCube<LIT>::ComputeMm1(const RunInfo &runInfo) {
    uint64_t s2GmBaseOffset = runInfo.s2Idx * s2BaseSize;
    uint64_t s1gProcessSize = runInfo.actMBaseSize;      // M 轴大小 = S1_base * g
    uint64_t s2ProcessSize = runInfo.actualSingleProcessSInnerSize;  // S2 轴大小

    // 外层循环: S2 方向,每次 S2_BASIC_BLOCK=256
    for (uint64_t s2GmOffset = 0; s2GmOffset < s2ProcessSize; s2GmOffset += S2_BASIC_BLOCK) {
        // ---- MTE2: Key GM → L1 (Nd2Nz) ----
        WaitFlag<MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
        if (PAGE_ATTENTION) {
            KeyNd2NzForPA(s2L1RealSize, s2GmBaseOffset + s2GmOffset, runInfo);  // PA: 查 blockTable
        } else {
            KeyNd2Nz(s2L1RealSize, s2GmOffset, runInfo);  // 普通: 直接搬
        }

        // 中层循环: S1*g 方向,每次 M_BASIC_BLOCK=256
        for (uint64_t s1gGmOffset = 0; s1gGmOffset < s1gProcessSize; s1gGmOffset += M_BASIC_BLOCK) {
            // ---- MTE2: Query GM → L1 (仅首个 S2 块搬一次,后续复用) ----
            if (runInfo.isFirstS2InnerLoop && s2GmOffset == 0) {
                WaitFlag<MTE1_MTE2>(QUERY_MTE1_MTE2_EVENT + queryL1Mte2BufIdx_ % QUERY_BUF_NUM);
                QueryNd2Nz(s1gL1RealSize, s1gGmOffset, runInfo);
            }

            // 内层循环: L1 → L0A/L0B → Mmad → L0C → Fixpipe
            for (s2L1Offset ...; s2L1Offset < s2L1RealSize; s2L1Offset += S2_BASIC_BLOCK_L0) {
                for (s1gL1Offset ...; s1gL1Offset < s1gL1RealSize; s1gL1Offset += M_BASIC_BLOCK_L0) {
                    WaitFlag<M_MTE1>(M_MTE1_EVENT + l0BufIdx_ % L0_BUF_NUM);

                    LoadQueryToL0a(...);   // L1 → L0A (LoadData3D)
                    LoadKeyToL0b(...);     // L1 → L0B (LoadData2D)

                    ComuteL0c(...);        // Mmad: C = Q × K^T

                    SetFlag<M_MTE1>(...);  // 释放 L0 Buffer

                    Fixp(...);             // L0C → GM (Fixpipe + ReLU)
                    l0BufIdx_++;
                }
            }
        }
        SetFlag<MTE1_MTE2>(KEY_MTE1_MTE2_EVENT + keyL1BufIdx_ % KEY_BUF_NUM);
        keyL1BufIdx_++;
    }
}
```

**流水线设计**:
- `KEY_BUF_NUM=3`: 3 个 Key L1 Buffer 轮转,实现 MTE2(搬运)与 MTE1(L1→L0)并行
- `QUERY_BUF_NUM=2`: Query 在首个 S2 块搬运后复用,2 Buffer 轮转
- `L0_BUF_NUM=2`: L0A/L0B/L0C 双 Buffer,实现 MTE1(L1→L0)与 M(矩阵乘)并行

#### 4.7.2 `QueryNd2Nz()` — Query ND→NZ 转换

**文件**: `op_kernel/arch22/lightning_indexer_service_cube.h:286-301`

```cpp
__aicore__ inline void QueryNd2Nz(uint64_t s1gL1RealSize, uint64_t s1gGmOffset, const RunInfo &runInfo) {
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = s1gL1RealSize;      // 行数 = M 方向实际大小
    nd2nzPara.dValue = headDim;             // 列数 = 128
    nd2nzPara.srcDValue = headDim;          // 源矩阵列步长 = 128
    nd2nzPara.dstNzC0Stride = CeilAlign(s1gL1RealSize, 16);  // NZ 布局 C0 步长,16 对齐
    nd2nzPara.dstNzNStride = 1;
    DataCopy(queryL1_[(queryL1Mte2BufIdx_ % 2) * QUERY_BUFFER_OFFSET],
             queryGm_[runInfo.tensorQueryOffset + s1gGmOffset * headDim], nd2nzPara);
}
```

**功能**: 将 Query 从 ND(行优先)布局转为 NZ(分块列优先)布局,以适配 Cube 的 LoadData3D 和 Mmad 接口要求。

#### 4.7.3 `KeyNd2NzForPA()` — PA 场景 Key 搬运

**文件**: `op_kernel/arch22/lightning_indexer_service_cube.h:244-282`

```cpp
__aicore__ inline void KeyNd2NzForPA(uint64_t s2L1RealSize, uint64_t s2GmOffset, const RunInfo &runInfo) {
    while (s2L1Offset < s2L1RealSize) {
        // 通过 blockTable 查表获取物理 block id
        uint64_t s2BlkId = (s2L1Offset + s2GmOffset) / kCacheBlockSize;
        uint64_t s2BlkOffset = (s2L1Offset + s2GmOffset) % kCacheBlockSize;
        // 物理地址 = blockTable[bIdx][s2BlkId] * blockSize * kHeadNum * headDim + s2BlkOffset * headDim
        uint64_t keyGmOffset = blkTableGm_.GetValue(bIdx * maxBlockNumPerBatch + s2BlkId)
                             * kCacheBlockSize * kHeadNum * headDim + s2BlkOffset * headDim;

        // 处理 block 边界: 不跨 block 搬运
        uint64_t s2Mte2Size = s2BlkOffset + s2Mte2Size >= kCacheBlockSize
                            ? kCacheBlockSize - s2BlkOffset : s2Mte2Size;

        Nd2NzParams nd2nzPara;
        nd2nzPara.nValue = s2Mte2Size;  // 行数 = 本次搬运的 S2 大小
        // ... Nd2Nz 转换
        s2L1Offset += s2Mte2Size;
    }
}
```

**功能**: PageAttention 场景下,Key 存储在非连续的物理 block 中。通过 `blockTable` 查表获取每个逻辑 S2 位置对应的物理 block 地址,跨 block 边界时分段搬运。

#### 4.7.4 `ComuteL0c()` — Mmad 矩阵乘

**文件**: `op_kernel/arch22/lightning_indexer_service_cube.h:357-372`

```cpp
__aicore__ inline void ComuteL0c(uint64_t s1gL0RealSize, uint64_t s2L0RealSize, const RunInfo &runInfo) {
    MmadParams mmadParams;
    mmadParams.m = CeilAlign(s1gL0RealSize, 16);   // M = S1*g (16 对齐)
    mmadParams.n = s2L0RealSize;                    // N = S2
    mmadParams.k = headDim;                         // K = 128
    mmadParams.cmatrixInitVal = true;               // L0C 初始化(累加模式)
    mmadParams.cmatrixSource = false;               // C 在 CO1(左半)
    Mmad(cL0_[...], queryL0_[...], keyL0_[...], mmadParams);  // C = Q × K^T

    // 小矩阵时加 PipeBarrier 防止 L0C 读写冲突
    if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
        PipeBarrier<PIPE_M>();
    }
}
```

#### 4.7.5 `Fixp()` — Fixpipe 输出(融合 ReLU)

**文件**: `op_kernel/arch22/lightning_indexer_service_cube.h:375-393`

```cpp
__aicore__ inline void Fixp(uint64_t s1gGmOffset, uint64_t s2GmOffset,
                             uint64_t s1gL0RealSize, uint64_t s2L0RealSize, const RunInfo &runInfo) {
    AscendC::DataCopyCO12DstParams intriParams;
    intriParams.mSize = CeilAlign(s1gL0RealSize, 16);    // M 方向 16 对齐
    intriParams.nSize = s2L0RealSize;                     // N 方向
    intriParams.dstStride = runInfo.actualSingleProcessSInnerSizeAlign;  // 目标 stride
    intriParams.quantPre = QuantMode_t::NoQuant;          // 无量化
    intriParams.nz2ndEn = true;                           // NZ → ND 转换
    intriParams.reluPre = 1;                              // ★ Fixpipe 阶段融合 ReLU
    AscendC::SetFixpipeNz2ndFlag(1, 1, 1);
    AscendC::DataCopy(mm1ResGm_[(runInfo.loop % 2) * mBaseSizeAlign * s2BaseSize + ...],
                      cL0_[...], intriParams);
}
```

**关键设计**: `reluPre=1` 让 Fixpipe 硬件在将 L0C 结果写回 GM 时自动执行 ReLU,避免在 Vector 侧额外做一次 ReLU,省去一次数据搬运。DoubleBuffer 通过 `runInfo.loop % 2` 实现。

---

### 4.8 Vector 服务层 (`lightning_indexer_service_vector.h`)

#### 4.8.1 `ProcessVec()` — 加权+TopK 核心

**文件**: `op_kernel/arch22/lightning_indexer_service_vector.h:265-527`

```cpp
__aicore__ inline void LightningIndexerServiceVector<LIT>::ProcessVec(const RunInfo &info) {
    int32_t cuBaseS1Idx = info.gS1Idx * s1BaseSize_;
    int32_t cuBaseS2Idx = info.s2Idx * s2BaseSize_;

    // DoubleBuffer offset: 偶数 loop → 前半, 奇数 loop → 后半
    int64_t mmGmOffset = (info.loop % 2) * (mBaseSizeAlign * s2BaseSize_);

    // 每个 AIV 处理 S1 的一半 (2个AIV/核)
    int32_t cuS1ProcNumPerAiv = blockId_ % 2 == 0 ? CeilDiv(cuS1ProcNum, 2) : (cuS1ProcNum / 2);
    cuS1BeginIdxPerAiv += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2);

    // G 轴切分: groupInner_ = 16
    int32_t outerG = CeilDiv(gSize_, groupInner_);

    // 非 M 首块时重置 TopK 缓冲
    if (info.loop != 0 && info.s2Idx == 0) {
        InitSortOutBuf(globalTopkUb_, ...);  // 重置为 (-inf, -1)
        blockS2StartIdx_ = 0;
    }

    // 因果掩码下的有效 S2 长度
    int32_t cuRealAcSeq = info.actS2Size;
    if (attenMaskFlag) {
        cuRealAcSeq = info.actS2Size - (info.actS1Size - cuS1BeginIdxPerAiv);
    }

    // ====== 逐 S1 token 处理 ======
    for (int innerS1Idx = 0; innerS1Idx < cuS1ProcNumPerAiv; innerS1Idx++) {
        if (attenMaskFlag) cuRealAcSeq += 1;  // 每个 S1 token 的有效 S2 递增 1
        int32_t cuS2Len = Min(cuBaseS2Idx + s2BaseSize_, cuRealAcSeq) - cuBaseS2Idx;

        if (cuRealAcSeq > 0 && cuS2Len > 0) {
            // ===== (a) 加权求和: S[g,S2] * W[g], 沿 g 轴 Reduce =====
            for (int outerGidx = 0; outerGidx < outerG; outerGidx++) {
                int32_t procGnum = ...;  // 当前 group 块大小
                int32_t pingpong = outerGidx % 2;  // DoubleBuffer

                // 从 GM 搬入 Cube 结果(mm1Res)和权重(weights)
                CopyIn(dbTmpUb, weightsInTUb, mm1ResGm, weightsGm, mmGmAllOffet, weightGmAllOffset, ...);

                // 逐元素乘: S[g,S2] * W[g]
                DoScale(reduceCacheBuf, dbTmpUb, weightsInUb, weightsInTUb, brcBuf, procGnum, s2BaseSize_, outerGidx);
                // outerGidx==0: Mul → reduceCacheBuf
                // outerGidx>0:  Mul → mmOutUb, 然后 Add → reduceCacheBuf (累加)
            }

            // 完成沿 g 轴的最终归约 → (S2,)
            DoReduce(reduceCacheBuf, reduceOutInner, gRedCnt, s2BaseSize_);

            // ===== (b) TopK 排序 =====
            // 准备排序输入: score + index 交错排列
            Adds(sortScoreUb, reduceOutInner, 0.0f, cuS2Len);  // 复制 score
            Adds(sortIndiceUbInt, globalTopkIndice_, cuBaseS2Idx, cuS2Len);  // index + 偏移

            if (actS1Size > 4 || isSparseCountOver2K) {
                // 大规模路径: SortAll + MergeSort
                SortAll(reduceOutBuff, tmpSortBuf, cuS2LenVecAlign);  // 精排当前 S2 块
                MergeSort(globalTopkUb_[innerS1Idx * virTopK * 2], virTopK,
                          reduceOutBuff, cuS2LenVecAlign, UbTmpSort);  // 合并到全局 TopK
            } else {
                // 小规模路径: Sort + MrgBasicBlock + SparseTopK
                Sort<float, true>(SortedBasicBlock_[...], reduceOutBuff, sortIndiceUbInt, tmpSortBuf, ...);
                // 缓存 4 块后合并
                if (globalTopkUbCacheIdx == 3 || isS2End || isAllLoopEnd) {
                    if (info.s2Idx - blockS2StartIdx_ < 4) {
                        MrgBasicBlock(globalTopkUb_[...], tt, globalTopkUbCacheIdx + 1, s2BaseSize_);
                    } else {
                        SparseTopK(globalTopkUb_[...], SortedBasicBlock_[...], tmpSortBuf, BASE_TOPK, ...);
                    }
                }
            }

            // ===== (c) 结果输出 =====
            bool needCopyOutGm = blockS2StartIdx_ == 0 && isS2End;  // 单核完成
            bool needCopyWsGm = isAllLoopEnd || isS2End;            // 需跨核归约

            if (needCopyOutGm) {
                // 直接输出: Extract 分离 value/index, CopyOut 写回 GM
                Extract(outValueUb, outIdxUb, globalTopkUb_[...], offset / 32);
                if (returnValue) Cast(valueULocal1, outValueUb, CAST_ROUND, copyLen);
                CopyOut(indiceOutGm[...], idxULocal1, copyLen);
                if (returnValue) CopyOut(valueOutGm[...], valueULocal1, copyLen);
            } else if (needCopyWsGm) {
                // 写入 workspace: TopK 结果 + LD 参数(needFd, s2AcSeq, s2Start, s2End, ...)
                tmpiBuff.SetValue(0, 1);              // needFd = 1
                tmpiBuff.SetValue(1, cuRealAcSeq);    // S2 实际长度
                tmpiBuff.SetValue(2, blockS2StartIdx_);  // S2 起始块
                tmpiBuff.SetValue(3, cuBaseS2Idx + cuS2Len);  // S2 结束位置
                tmpiBuff.SetValue(4, isS2End);        // 是否 S2 最后一块
                tmpiBuff.SetValue(8, indiceOutOffset + cuS1Idx * sparseCount);  // 输出地址偏移
                // 头尾判断: tail reduce → 写偏移 1; head reduce → 写偏移 0
                CopyOut(vec1ParamGm[wsInfoOffset], tmpiBuff, 16);
                CopyOut(vec1ResGm[wsOffset], globalTopkUb_[...], 2 * BASE_TOPK);
            }
        } else if (cuRealAcSeq <= 0) {
            CleanInvalidOutput(...);  // 无效 token 输出 -1
        }
    }

    // BSND 场景: 清理 S1 > actS1 的无效输出
    if (LAYOUT_T == BSND) {
        if (invalidS1Num > 0 && isS1LoopEnd && blockS2StartIdx_ == 0) {
            for (...) CleanInvalidOutput(...);
        }
    }
}
```

#### 4.8.2 `ProcessLD()` — 跨核 TopK 归约

**文件**: `op_kernel/arch22/lightning_indexer_service_vector.h:530-704`

```cpp
__aicore__ inline void LightningIndexerServiceVector<LIT>::ProcessLD() {
    int32_t curCubeId = blockId_ / 2;  // 当前 AIC id

    // 1. 扫描 workspace,找出需要归约的 S1 行(needFd==1)
    for (uint32_t innerS1Idx = 0; innerS1Idx < s1BaseSize_; innerS1Idx++) {
        needFd = vec1ParamGm.GetValue(paramGmCoreOffset + innerS1Idx * 2 * paramNum_ + paramNum_);
        if (needFd == 1) { s1ProcNum++; }
    }
    if (s1ProcNum == 0) return;

    // 2. 每个 AIV 处理一半 S1 行
    uint32_t s1VecNum = CeilDiv(s1ProcNum, 2);
    if (blockId_ % 2 == 1) { s1LdStartIdx += s1VecNum; s1VecNum = s1ProcNum - s1VecNum; }

    // 3. 逐行归约
    for (uint32_t innerS1Idx = s1LdStartIdx; innerS1Idx < s1LdStartIdx + s1VecNum; innerS1Idx++) {
        // 搬入当前核的尾规约结果(workspace 偏移 2*BASE_TOPK 处)
        DataCopyPad(curValueIdxUb, vec1ResGm[wsOffset], ...);
        acc_list_num = 1;

        // 向后扫描,搬入后续核的头规约结果(workspace 偏移 0 处)
        tmpCubeId++;
        needFd = vec1ParamGm.GetValue(wsInfoOffset);
        while (needFd == 1) {
            DataCopyPad(curValueIdxUb[valueOffset], vec1ResGm[wsOffset], ...);  // 头规约数据
            acc_list_num++;

            // 每满 4 个 list,用 MrgSort4 归并为 1 个
            if (acc_list_num == mrgListNum_) {
                MrgSort4Info params;
                params.elementLengths = {BASE_TOPK, BASE_TOPK, BASE_TOPK, BASE_TOPK};
                params.validBit = 0b1111;
                MrgSort(tmpUb, srcList, params);    // 4 路归并
                DataCopy(curValueIdxUb, tmpUb, 2 * BASE_TOPK);  // 结果写回第 1 块
                acc_list_num = 1;
                valueOffset = 2 * BASE_TOPK;
            }

            if (isS2End == 1) break;  // 到达 S2 末尾
            tmpCubeId++;
            needFd = vec1ParamGm.GetValue(wsInfoOffset);
        }

        // 4. 归约不足 4 个 list 的剩余数据
        if (acc_list_num == 2) { params.validBit = 0b0011; }
        else if (acc_list_num == 3) { params.validBit = 0b0111; }
        MrgSort(tmpUb, srcList, params);
        DataCopy(curValueIdxUb, tmpUb, 2 * BASE_TOPK);

        // 5. Extract 分离 value/index, 写回输出 GM
        Extract(outValueUb, outIdxUb, curValueIdxUb, BASE_TOPK / 32);
        if (!returnValue) {
            DataCopyPad(indiceOutGm[outOffset], idxULocal1, ...);  // 只输出索引
        } else {
            Cast(valueULocal1, outValueUb, CAST_ROUND, sparseCount);  // value 转 fp16/bf16
            DataCopyPad(indiceOutGm[outOffset], idxULocal1, ...);     // 输出索引
            DataCopyPad(valueOutGm[outOffset], valueULocal1, ...);    // 输出值
        }
    }
}
```

**归约原理**: 当 S2 被切分到多个核时,每个核维护自己的局部 Top-2048。归约时,当前核的"尾"(workspace 偏移 `2*BASE_TOPK`)与下一个核的"头"(偏移 `0`)做 MrgSort 归并。4 路归并后取前 2048,直到 S2 结束。

#### 4.8.3 `InitBuffers()` — UB 缓冲初始化

**文件**: `op_kernel/arch22/lightning_indexer_service_vector.h:132-172`

```cpp
__aicore__ inline void LightningIndexerServiceVector<LIT>::InitBuffers(TPipe *pipe) {
    virTopK = isSparseCountOver2K ? sparseCount : BASE_TOPK;  // 2048 或 sparseCount

    pipe->InitBuffer(outQueue_, 1, outNeedBufSize);           // 输出队列
    pipe->InitBuffer(tmpBuf_, (groupInner_ * s2BaseSize_ + s2BaseSize_) * 2 * sizeof(float));  // 临时缓冲(DoubleBuffer)
    pipe->InitBuffer(sortOutBuf_, CeilDiv(s1BaseSize_, 2) * virTopK * 2 * sizeof(float));  // TopK 排序结果
    pipe->InitBuffer(indexBuf_, s2BaseSize_ * sizeof(int32_t));  // 索引缓冲
    pipe->InitBuffer(reduceOutBuf_, s2BaseSize_ * 2 * sizeof(float));  // 归约输出
    pipe->InitBuffer(brcBuf_, groupInner_ * 8 * sizeof(float));  // 广播缓冲
    pipe->InitBuffer(paramBuf_, LD_PARAM_NUM * sizeof(int64_t)); // LD 参数缓冲

    // 初始化有序索引 [0, 1, 2, ..., s2BaseSize-1]
    ArithProgression<int32_t>(globalTopkIndice_, 0, 1, s2BaseSize_);
    // 初始化 TopK 缓冲为交错排列的 (-inf, -1)
    InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * virTopK * 2);
    // 清零 workspace 中的 LD 参数(needFd = -1)
    Duplicate(tmpBuff.ReinterpretCast<int32_t>(), -1, ...);
    CopyOut(vec1ParamGm[wsInfoOffset], tmpBuff.ReinterpretCast<int64_t>(), 16);
}
```

---

### 4.9 Vector 辅助函数 (`lightning_indexer_vector.h`)

#### 4.9.1 `CopyIn()` — GM→UB 数据搬运

**文件**: `op_kernel/arch22/lightning_indexer_vector.h:43-66`

```cpp
template <typename T>
__aicore__ inline void CopyIn(LocalTensor<float> &mmOutUb, LocalTensor<T> &weightsUb,
                              GlobalTensor<float> &mMoutGm, GlobalTensor<T> &weightScaleGm,
                              int64_t MMout_gmoffset, int64_t weights_gmoffset,
                              int64_t groupInner, int64_t s2Inner, int64_t mmUbStride) {
    // 搬运 Cube 结果: [groupInner, s2Inner] 的 float 矩阵
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyExtParams dataCopymMoutParams;
    dataCopymMoutParams.blockCount = groupInner;        // G 方向块数
    dataCopymMoutParams.blockLen = s2Inner * sizeof(float);  // 每块长度
    dataCopymMoutParams.srcStride = 0;                  // 源 stride(Cube 输出已紧凑)
    dataCopymMoutParams.dstStride = mmUbStride;         // 目标 stride(考虑对齐)
    DataCopyPad(mmOutUb, mMoutGm[MMout_gmoffset], dataCopymMoutParams, padParams);

    // 搬运权重: [groupInner] 的一维向量
    DataCopyPad(weightsUb, weightScaleGm[weights_gmoffset], dataCopyweightParams, padTParams);
}
```

#### 4.9.2 `DoScale()` — 加权逐元素乘 + 部分归约

**文件**: `op_kernel/arch22/lightning_indexer_vector.h:82-115`

```cpp
template <typename T>
__aicore__ inline void DoScale(const LocalTensor<float> &reduceCacheBuf, LocalTensor<float> &mmOutUb,
                               LocalTensor<float> &weightsUb, LocalTensor<T> &weightsTUb,
                               LocalTensor<float> &tmpBuff, int64_t groupInner, int64_t s2Inner, int32_t outerGidx) {
    // 1. 若 weights 非 float(如 fp16/bf16),先 Cast 为 float
    if constexpr (!IsSameType<T, float>::value) {
        Cast(weightsUb, weightsTUb, CAST_NONE, groupInner);
    }

    // 2. 权重广播: [groupInner, 1] → [groupInner, 8] (Brcb)
    Brcb(tmpBuff, weightsUb, CeilDiv(groupInner, 8), {1, 8});

    // 3. 逐行乘法: [1, s2Inner] * [1, s2Inner] → [1, s2Inner]
    for (int32_t i = 0; i < groupInner; i++) {
        if (outerGidx == 0) {
            // 首个 group 块: Mul → reduceCacheBuf
            Mul(reduceCacheBuf[i * s2Inner], mmOutUb[i * s2Inner], tmpBuff[i * 8], ...);
        } else {
            // 后续 group 块: Mul → mmOutUb, 然后 Add → reduceCacheBuf (累加前一块)
            Mul(mmOutUb[i * s2Inner], mmOutUb[i * s2Inner], tmpBuff[i * 8], ...);
        }
    }

    // 4. 后续块: Add 累加到 reduceCacheBuf
    if (outerGidx != 0) {
        Add(reduceCacheBuf, mmOutUb, reduceCacheBuf, groupInner * s2Inner);
    }
}
```

**设计要点**: 当 gSize > 16 时分多次处理。第一次直接写入 reduceCacheBuf,后续先乘到 mmOutUb 再 Add 累加,实现沿 g 轴的流式 Reduce。

#### 4.9.3 `DoReduce()` — 树形归约

**文件**: `op_kernel/arch22/lightning_indexer_vector.h:130-153`

```cpp
__aicore__ inline void DoReduce(const LocalTensor<float> &srcTensor, LocalTensor<float> &dstTensor,
                                int32_t rNum, int32_t aNum) {
    if (rNum == 1) {
        Adds(dstTensor, srcTensor, 0, aNum);  // 只有一行,直接拷贝
        return;
    }

    // 找到 <= rNum 的最大 2 的幂
    uint32_t dichotomizeAddPow = FindNearestPower2(rNum);
    // 处理非 2 的幂的余数部分
    uint32_t dichotomizeAddDiffSize = rNum - dichotomizeAddPow;
    if (dichotomizeAddDiffSize != 0) {
        Add(srcTensor, srcTensor, srcTensor[dichotomizeAddPow * aNum], dichotomizeAddDiffSize * aNum);
    }
    // 树形二分归约: 8→4→2→1
    int32_t nowRows = dichotomizeAddPow;
    while (nowRows > 2) {
        nowRows = nowRows / 2;
        Add(srcTensor, srcTensor, srcTensor[nowRows * aNum], nowRows * aNum);
    }
    Add(dstTensor, srcTensor, srcTensor[aNum], aNum);  // 最终: row[0] + row[1] → dst
}
```

**原理**: 将 `rNum` 行(g 轴)通过二分法逐层 Add 归约到 1 行。例如 rNum=16: 16→8→4→2→1,共 4 次 Add 操作。

#### 4.9.4 `SortAll()` — 完整排序(两个重载)

**文件**: `op_kernel/arch22/lightning_indexer_vector.h:190-266`

**重载 1** (value+index 交错输入):
```cpp
__aicore__ inline void SortAll(LocalTensor<float> &src, LocalTensor<float> &tmp, int64_t logitsNum) {
    // 第一步: Sort32 对每 32 元素排序
    int64_t sort32Repeats = logitsNum / 32;  // 32B 对齐
    Sort32(tmp, src, src[logitsNum].ReinterpretCast<uint32_t>(), sort32Repeats);

    // 第二步: 迭代 MrgSort4 归并,直到只剩 1 组
    int64_t mrgGroups = sort32Repeats;
    int64_t mrgElements = 32;
    while (true) {
        MrgSort4Info params;
        params.elementLengths = {mrgElements, mrgElements, mrgElements, mrgElements};
        params.validBit = 0b1111;
        if (mrgGroups <= 4) {
            // 最后一次归并
            params.repeatTimes = 1;
            MrgSort(dstTensor, srcList, params);
            break;
        } else {
            // 4 路归并,组数 /4
            params.repeatTimes = mrgGroups / 4;
            MrgSort(dstTensor, srcList, params);
            mrgElements *= 4;
            mrgGroups /= 4;
        }
    }
}
```

**重载 2** (value/index 分离输入):
```cpp
__aicore__ inline void SortAll(LocalTensor<float> &dst, LocalTensor<float> &srcValue,
                                LocalTensor<uint32_t> &srcIndex, LocalTensor<float> &tmpTensor, int64_t logitsNum) {
    int64_t sort32Repeats = logitsNum / 32;
    Sort<float, true>(dst, srcValue, srcIndex, tmpTensor, sort32Repeats);  // 直接排序
}
```

#### 4.9.5 `MergeSort()` — 全局 TopK 合并

**文件**: `op_kernel/arch22/lightning_indexer_vector.h:274-321`

```cpp
__aicore__ inline void MergeSort(const LocalTensor<float> &mrgDst, int32_t mrgDstNum,
                                 LocalTensor<float> &mrgSrc, int32_t mrgSrcNum, LocalTensor<float> &tmpTensor) {
    if (mrgDstNum <= 3072) {
        // 小规模: 2 路 MrgSort (dst + src → tmp → dst)
        MrgSort4Info params;
        params.elementLengths = {mrgSrcNum, mrgDstNum};  // src 和 dst 不同长度
        params.validBit = 0b0011;
        MrgSort(tmpTensor, srcList, params);
        DataCopy(mrgDst, tmpTensor, mrgDstNum * 2);  // 只保留前 mrgDstNum 个
    } else {
        // 大规模: 将 dst 分成 3 段 + src,4 路 MrgSort
        int64_t unitElements = 1024;
        int64_t segNum = mrgDstNum / unitElements;
        int64_t mrgQuelen_1 = (segNum + 2) / 3;
        int64_t mrgQuelen_2 = ((segNum - mrgQuelen_1) + 1) / 2;
        int64_t mrgQuelen_3 = segNum - mrgQuelen_1 - mrgQuelen_2;
        params.elementLengths = {mrgQuelen_1 * 1024, mrgQuelen_2 * 1024, mrgQuelen_3 * 1024, mrgSrcNum};
        params.validBit = 0b1111;
        MrgSort(tmpTensor, srcList, params);
        DataCopy(mrgDst, tmpTensor, mrgDstNum * 2);
    }
}
```

**设计要点**: 将当前 S2 块的排序结果(mrgSrc)与全局 TopK 缓冲(mrgDst)做归并排序。大规模时将 mrgDst 拆成 3 段利用 4 路 MrgSort 硬件,减少归并轮次。

#### 4.9.6 `SparseTopK()` — 稀疏 TopK 筛选

**文件**: `op_kernel/arch22/lightning_indexer_vector.h:373-397`

```cpp
template <bool needMrg = true>
__aicore__ inline void SparseTopK(const LocalTensor<float> &dst, const LocalTensor<float> &needsMerging,
                                  const LocalTensor<float> &tmp, int64_t topk, int64_t mergSize) {
    if (!needMrg) {
        DataCopy(dst, needsMerging, mergSize * 2);
        return;
    }
    // 2 路 MrgSort: dst(已有 topk 个) + needsMerging(mergSize 个) → tmp → dst(取前 topk)
    MrgSort4Info params;
    params.elementLengths = {topk, mergSize};      // 两个队列长度不同
    params.ifExhaustedSuspension = (topk == mergSize);  // 长度相等时挂起模式
    params.validBit = 0b0011;
    MrgSort(tmp, srcList, params);
    DataCopy(dst, tmp, topk * 2);  // 只保留前 topk 个
}
```

#### 4.9.7 `InitSortOutBuf()` — TopK 缓冲初始化

**文件**: `op_kernel/arch22/lightning_indexer_vector.h:160-182`

```cpp
__aicore__ inline void InitSortOutBuf(const LocalTensor<float> &src, int64_t eleNum) {
    // 使用 mask 交替写入 -inf(0xFF800000) 和 -1,实现交错排列 [value=-inf, index=-1, value=-inf, index=-1, ...]
    uint64_t mask1[2] = {0x5555555555555555, 0};  // 奇数位: value
    uint64_t mask0[2] = {0xaaaaaaaaaaaaaaaa, 0};  // 偶数位: index
    Duplicate(src.ReinterpretCast<int32_t>(), NEG_INF, mask1, repeatNum, 1, 8);     // 奇数位写 -inf
    Duplicate(src.ReinterpretCast<int32_t>(), INVALID_INDEX, mask0, repeatNum, 1, 8); // 偶数位写 -1
}
```

**原理**: TopK 排序缓冲区采用 value 和 index 交错排列的格式 `[..., value_i, index_i, value_j, index_j, ...]`。初始化时所有 value 设为 -inf(最小值),所有 index 设为 -1(无效),这样后续 MrgSort 归并时这些占位符自然排在最后。

#### 4.9.8 `ExtractIndex()` — 分离 value 和 index

**文件**: `op_kernel/arch22/lightning_indexer_vector.h:400-412`

```cpp
__aicore__ inline void ExtractIndex(const LocalTensor<uint32_t> &idxULocal,
                                    const LocalTensor<uint32_t> &sortLocal, int64_t extractNum) {
    GatherMaskParams gatherMaskParams;
    gatherMaskParams.repeatTimes = Ceil(extractNum * sizeof(float) * 2, 256);
    gatherMaskParams.src0RepeatStride = 8;
    uint8_t src1Pattern = 2;  // 模式 2: 提取奇数位置(即 index)
    GatherMask(idxULocal, sortLocal, src1Pattern, false, 0, gatherMaskParams, rsvdCnt);
}
```

**原理**: 排序结果中 value 和 index 交错排列。`GatherMask` 使用 pattern=2(奇数索引模式)提取所有奇数位置的元素(即 index),实现 value 和 index 的分离。

#### 4.9.9 `CopyOut()` — UB→GM 输出

**文件**: `op_kernel/arch22/lightning_indexer_vector.h:70-78`

```cpp
template <typename T>
__aicore__ inline void CopyOut(const GlobalTensor<T> &dstGm, const LocalTensor<T> &srcUb, int64_t copyCount) {
    DataCopyParams dataCopyOutyParams;
    dataCopyOutyParams.blockCount = 1;
    dataCopyOutyParams.blockLen = copyCount * sizeof(T);  // 按字节长度拷贝
    dataCopyOutyParams.srcStride = 0;
    dataCopyOutyParams.dstStride = 0;
    DataCopyPad(dstGm, srcUb, dataCopyOutyParams);
}
```

---

### 4.10 公共定义 (`lightning_indexer_common.h`)

#### 关键结构体

**`ConstInfo`** (L67-124): 运行时常量集合,包含:
- `syncC1V1`/`syncV1C1`: Cube↔Vector 跨核同步 EventID
- `mBaseSize`/`s1BaseSize`/`s2BaseSize`: 基本块大小
- `sparseCount`: Top-k 的 k 值
- `attenMaskFlag`: 是否因果掩码
- `isSparseCountOver2K`: sparseCount 是否 > 2048(影响 TopK 路径选择)
- `INVALID_IDX = -1`: 无效索引填充值

**`RunInfo`** (L40-65): 单个基本块的运行时信息,包含:
- `bIdx`/`n2Idx`/`gS1Idx`/`s2Idx`: 四维循环索引
- `actMBaseSize`/`actualSingleProcessSInnerSize`: 实际块大小
- `tensorQueryOffset`/`tensorKeyOffset`/`tensorWeightsOffset`/`indiceOutOffset`: 各 tensor 的 GM 偏移
- `isFirstS2InnerLoop`/`isLastS2InnerLoop`/`isAllLoopEnd`: 首/尾/全结束标志

**`SplitCoreInfo`** (L126-135): 分核信息,定义每个核处理的四维区间 `[bN2Start, bN2End] × [gS1Start, gS1End] × [s2Start, s2End]`,`isLD` 标记是否需要做跨核归约。

#### 关键工具函数

- `Align(num, rnd)`: 向上对齐 `((num + rnd - 1) / rnd) * rnd`
- `CeilDiv(num, rnd)`: 向上整除 `(num + rnd - 1) / rnd`
- `Min(a, b)` / `Max(a, b)`: 取小/取大

---

## 五、总结:数据流全景

```
GM(query, key, weights)
    │
    ├──[Cube AIC]── ComputeMm1()
    │     │
    │     ├─ QueryNd2Nz:   GM(Q) ──MTE2──→ L1(Q) [NZ布局]
    │     ├─ KeyNd2Nz[PA]: GM(K) ──MTE2──→ L1(K) [NZ布局, 查blockTable]
    │     ├─ LoadQueryToL0a: L1(Q) ──MTE1──→ L0A(Q) [LoadData3D]
    │     ├─ LoadKeyToL0b:  L1(K) ──MTE1──→ L0B(K) [LoadData2D]
    │     ├─ ComuteL0c:     Mmad: C = Q × K^T  [L0A × L0B → L0C]
    │     └─ Fixp:          L0C ──Fixpipe+ReLU──→ GM(mm1Res) [float, NZ→ND]
    │           │
    │     CrossCoreSetFlag(syncC1V1) ──────────→ 通知 Vector
    │
    ├──[Vector AIV]── ProcessVec()
    │     │
    │     ├─ CopyIn:    GM(mm1Res, weights) ──MTE2──→ UB
    │     ├─ DoScale:   UB: S[g,S2] × W[g] + 沿g轴部分Reduce ──V──→ UB(reduceCache)
    │     ├─ DoReduce:  UB: 沿g轴树形归约 → (S2,) ──V──→ UB(reduceOut)
    │     ├─ SortAll:   UB: 精排当前S2块 ──V──→ UB(sorted)
    │     ├─ MergeSort: UB: 合并到全局TopK ──V──→ UB(globalTopk)
    │     └─ CopyOut:   UB(globalTopk) ──MTE3──→ GM(sparseIndices, sparseValues)
    │     │           或 UB(globalTopk) ──MTE3──→ GM(workspace) [跨核归约场景]
    │
    └──[Vector AIV]── ProcessLD()  [跨核归约]
          │
          ├─ DataCopyPad: GM(workspace) ──MTE2──→ UB (各核局部TopK)
          ├─ MrgSort:     UB: 4路归并排序 ──V──→ UB(merged)
          ├─ Extract:     UB: GatherMask分离value/index ──V──→ UB(value, index)
          └─ DataCopyPad: UB ──MTE3──→ GM(sparseIndices, sparseValues) [最终输出]
```

---

## 六、关键设计要点

1. **Fixpipe 融合 ReLU**:ReLU 不在 Vector 侧做,而是在 Cube 的 Fixpipe 阶段通过 `reluPre=1` 硬件直接融合,减少一次数据搬运。

2. **多级流水线**:Cube 侧使用 3 个 Key Buffer + 2 个 Query Buffer + 2 个 L0 Buffer,通过 Event Flag 实现 MTE2→MTE1→M 的三级流水并行。

3. **TopK 两级架构**:先在各核内做局部 TopK(Sort + MergeSort),再通过 `ProcessLD` 跨核 MrgSort4 归约。这是为了处理 S2 被切分到多核的场景。

4. **sparseCount>2048 特殊路径**:`isSparseCountOver2K` 时,s1BaseSize 动态调整为 `8192/sparseCount*2`,S2 不分块(整块处理),使用 `SparseTopK` 替代常规 MergeSort。

5. **双 AIV 协作**:每个 AIC 配 2 个 AIV(`blockId_%2`),S1 方向对半切分,两个 AIV 分别处理前半和后半 S1,结果在 workspace 中按头尾排列,LD 阶段合并。
