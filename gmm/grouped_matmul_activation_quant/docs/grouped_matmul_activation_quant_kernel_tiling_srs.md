# GroupedMatmulActivationQuant Kernel/Tiling SRS

## 1. 迭代信息

| 项目 | 内容 |
| --- | --- |
| 迭代名称 | GroupedMatmulActivationQuant kernel/tiling 实现 |
| 算子类型 | GMM + Activation + MX FP8 dynamic quant 融合算子 |
| 覆盖范围 | Host def、InferShape、Tiling、AscendC Kernel、CSV tiling UT |
| 不覆盖范围 | aclnn/op_api、PTA wrapper、端到端精度 STC、性能 benchmark |
| 目标芯片 | Ascend950 / `__NPU_ARCH__ == 3510` |
| 迭代版本 | v0.1 |
| 更新时间 | 2026-06-08 |

## 2. 需求背景

当前模型图中存在如下计算链路：

```text
GroupedMatmul(MX FP8 input, MX FP8 weight NZ)
    -> dequant to fp32 accumulator
    -> GeluTanh activation
    -> dynamic MX FP8 quant
    -> y, y_scale
```

若将 GMM、Activation、Dynamic MX Quant 拆成多个算子，会产生额外 GM 读写、调度开销和中间张量生命周期。`GroupedMatmulActivationQuant` 的目标是将上述链路融合到一个 AscendC kernel 中：

- GMM 主体复用 grouped matmul 低阶 BasicAPI 能力。
- weight 使用 FP8 `FRACTAL_NZ` 存储。
- 输入、权重和输出只支持 FP8 `FLOAT8_E4M3FN` / `FLOAT8_E5M2` 场景。
- scale 使用 MX `FLOAT8_E8M0`，scale 形状按 `ceil(K / 64)` 和 `ceil(N / 64)` 分块。
- 当前 activation 仅支持 `GMM_ACT_TYPE_GELU_TANH = 2`。
- quant 逻辑参考 `dynamic_mx_quant_tail_axis_fp8.h` 的 tail axis FP8 MX quant 思路，并在 epilogue 中完成。

## 3. 方案简述

### 3.1 总体数据流

```mermaid
flowchart LR
    X[x: FP8 ND] --> MMAD[Grouped MX Matmul BasicAPI]
    XS[x_scale: E8M0] --> MMAD
    W[weight: FP8 FRACTAL_NZ] --> MMAD
    WS[weight_scale: E8M0] --> MMAD
    GL[group_list] --> Scheduler[Group Scheduler]
    Scheduler --> MMAD
    MMAD --> L0C[L0C fp32 tile]
    L0C --> Epi[Epilogue on AIV]
    Epi --> Act[GeluTanh]
    Act --> Quant[MX dynamic quant]
    Quant --> Y[y: FP8 ND]
    Quant --> YS[y_scale: E8M0 ND]
```

### 3.2 模块划分

| 模块 | 职责 |
| --- | --- |
| Host def | 定义输入、输出、属性顺序和 dtype/format 组合；注册 Ascend950 AICore 配置 |
| InferShape | 根据 `x` 和 `weight_scale` 推导 `y` / `y_scale` shape；根据 `quant_dtype` 推导 `y` dtype |
| Tiling | 校验 dtype、shape、attr；调用 GMM BasicAPI tiling；生成 `GMMActivationQuantTilingDataParams` |
| Kernel Dispatch | 根据 tiling key 选择 `weight` 是否转置，对应 `layout::Nz` / `layout::Zn` |
| GMM Kernel | 按 group 遍历，使用 low-level MX matmul BasicAPI 完成矩阵乘 |
| Epilogue | 对每个 matmul tile 做 GeluTanh + MX dynamic quant + GM 写回 |
| UT | 使用 CSV 驱动 tiling 白盒用例，覆盖合法和非法参数 |

### 3.3 当前实现边界

这篇 SRS 描述的是本轮 kernel/tiling 交付，不包含 aclnn/op_api 通路。为避免接口字段和实际数值路径混淆，当前实现边界如下：

- `activation_type` 是必选属性，但当前只允许 `GELU_TANH = 2`。tiling 会在 host 侧拦截其他值，kernel/epilogue 暂不做多 activation 分支。
- `x_scale` 在 def 中保持 optional，以便和 schema 演进兼容；当前 MX FP8 路径在 tiling 中会读取 `x_scale` desc/shape，因此本轮实际要求调用方传入有效 `x_scale`。
- `bias` 是 dynamic input 预留字段，kernel 入口接收该地址但当前通过 `(void)bias` 丢弃，不参与 GMM、Activation 或 Quant 数值计算。
- `weight`、`weight_scale`、`bias` 都是 dynamic input；当前 GMM 路径只消费 `weight[0]` 和 `weight_scale[0]`，多 weight list 不是本轮交付目标。
- `round_mode`、`scale_alg`、`dst_type_max` 仅按 MX FP8 当前可用组合收敛：`round_mode = "rint"`，`scale_alg in {0, 1}`，`dst_type_max = 0.0`。

### 3.4 类关系总览

```mermaid
classDiagram
    class GroupedMatmulActivationQuant {
        <<OpDef>>
        +x
        +group_list
        +x_scale
        +weight[]
        +weight_scale[]
        +bias[]
        +y
        +y_scale
        +activation_type
    }

    class GroupedMatmulActivationQuantTiling950 {
        +AnalyzeAttrs()
        +AnalyzeDtype()
        +AnalyzeInputs()
        +DoOpTiling()
        +DoLibApiTiling()
        +GetTilingKey()
    }

    class GroupedQmmBasicApiTiling {
        +GetShapeAttrsInfo()
        +SetGroupNum()
        +SetMKN()
        +CalL1Tiling()
    }

    class GMMActivationQuantTilingDataParams {
        +GMMActivationQuantParams
        +QuantBasicApiMMTiling
    }

    class grouped_matmul_activation_quant_apt {
        <<AICore_entry>>
        +DispatchRowMajorNz()
        +DispatchRowMajorZn()
    }

    class GmmActivationMxQuant {
        +AssembleBlockMmadMx()
        +AssembleBlockEpilogueActivationQuant()
        +AssembleKernelGmmActivationMixOnlineDynamic()
    }

    GroupedMatmulActivationQuant --> GroupedMatmulActivationQuantTiling950
    GroupedMatmulActivationQuantTiling950 --|> GroupedQmmBasicApiTiling
    GroupedMatmulActivationQuantTiling950 --> GMMActivationQuantTilingDataParams
    GMMActivationQuantTilingDataParams --> grouped_matmul_activation_quant_apt
    grouped_matmul_activation_quant_apt --> GmmActivationMxQuant
```

## 4. 输入输出规格

### 4.1 Host 字段顺序

def 文件需按以下顺序声明。该顺序同时决定 tiling/infershape 中 attr index 和 input index：

| 字段分组 | 字段名 | index | 当前实现要求 |
| --- | --- | ---: | --- |
| REQUIRED_INPUT | `x` | 0 | FP8 ND，形状 `[M, K]` |
| REQUIRED_INPUT | `group_list` | 1 | INT64 ND，形状 `[E]` |
| OPTIONAL_INPUT | `x_scale` | 2 | schema 为 optional；当前 MX FP8 kernel/tiling 实际必传，E8M0 ND，形状 `[M, ceil(K/64), 2]` |
| DYNAMIC_INPUT | `weight` | 3 | FP8 FRACTAL_NZ，动态 tensor list；当前仅消费第 0 个 tensor |
| DYNAMIC_INPUT | `weight_scale` | 4 | E8M0 ND，动态 tensor list；当前仅消费第 0 个 tensor |
| DYNAMIC_INPUT | `bias` | 5 | 预留 dynamic input；当前 kernel 接收但不消费 |
| OUTPUT | `y` | 0 | FP8 ND，形状 `[M, N]` |
| OUTPUT | `y_scale` | 1 | E8M0 ND，形状 `[M, ceil(N/64), 2]` |
| REQUIRED_ATTR | `activation_type` | 0 | 当前仅支持 `2`，即 GELU_TANH |
| OPTIONAL_ATTR | `dequant_dtype` | 1 | 当前仅支持 `ge::DT_FLOAT` |
| OPTIONAL_ATTR | `transpose_weight` | 2 | `false` 或 `true` |
| OPTIONAL_ATTR | `group_list_type` | 3 | `0` 累加 offsets；`1` 每组 token count |
| OPTIONAL_ATTR | `tuning_config` | 4 | 当前保留，默认 `{0}` |
| OPTIONAL_ATTR | `quant_dtype` | 5 | `FLOAT8_E4M3FN` 或 `FLOAT8_E5M2` |
| OPTIONAL_ATTR | `round_mode` | 6 | 当前仅支持 `"rint"` |
| OPTIONAL_ATTR | `scale_alg` | 7 | `0` OCP；`1` cuBLAS |
| OPTIONAL_ATTR | `dst_type_max` | 8 | 当前仅支持 `0.0` |

### 4.2 dtype 与 format

| 张量 | dtype | format | 说明 |
| --- | --- | --- | --- |
| `x` | `FLOAT8_E4M3FN` / `FLOAT8_E5M2` | `ND` | GMM A 矩阵 |
| `x_scale` | `FLOAT8_E8M0` | `ND` | MX per-token scale |
| `group_list` | `INT64` | `ND` | group 边界或长度 |
| `weight` | `FLOAT8_E4M3FN` / `FLOAT8_E5M2` | `FRACTAL_NZ` | GMM B 矩阵，dynamic input 第 0 个 tensor |
| `weight_scale` | `FLOAT8_E8M0` | `ND` | MX per-group weight scale |
| `bias` | `FLOAT` | `ND` | 接口预留；当前非空也不参与计算，后续策略待定 |
| `y` | `FLOAT8_E4M3FN` / `FLOAT8_E5M2` | `ND` | `y.dtype == quant_dtype` |
| `y_scale` | `FLOAT8_E8M0` | `ND` | 输出 MX scale |

### 4.3 shape 约束

设：

- `M` 为所有 group token 总数。
- `E` 为 group 数。
- `K` 为输入 hidden size。
- `N` 为输出 hidden size。
- `KBlocks = ceil(K / 64)`。
- `NBlocks = ceil(N / 64)`。

| 张量 | 非转置 weight | 转置 weight |
| --- | --- | --- |
| `x` | `[M, K]` | `[M, K]` |
| `x_scale` | `[M, KBlocks, 2]` | `[M, KBlocks, 2]` |
| `weight.origin` | `[E, K, N]` | `[E, N, K]` |
| `weight.storage` | `[E, ceil(N/32), ceil(K/16), 16, 32]` | `[E, ceil(K/32), ceil(N/16), 16, 32]` |
| `weight_scale` | `[E, KBlocks, N, 2]` | `[E, N, KBlocks, 2]` |
| `y` | `[M, N]` | `[M, N]` |
| `y_scale` | `[M, NBlocks, 2]` | `[M, NBlocks, 2]` |

必须满足：

- `x_scale.rank == 3`。
- `weight_scale.rank == 4`。
- `weight.storage.rank == 5`。
- `weight.storage[3] == 16`。
- `weight.storage[4] == 32`。
- `x_scale[M] == x[M]`。
- `x_scale[KBlock] == ceil(K / 64)`。
- `x_scale[pair] == 2`。
- `weight_scale[group] == E`。
- `weight_scale[N] == N`，转置时 N 维在 index 1，非转置时在 index 2。
- `weight_scale[KBlock] == ceil(K / 64)`，转置时 KBlock 维在 index 2，非转置时在 index 1。
- `weight_scale[pair] == 2`。

说明：

- `weight.storage` 表中的 leading dims 是 FP8 NZ layout 下的期望存储展开形式；当前 tiling 侧强校验 `rank == 5` 和尾两维 `16/32`，`M/K/N/E` 关系由 origin shape、BasicAPI tiling 和 kernel offset 共同约束。
- `x_scale` 和 `weight_scale` 的 scale 块维按 64 个原始 K/N 元素生成两个 E8M0 scale 字节，对应 MX FP8 的 pair dim = 2。
- `group_list` 的内容合法性依赖运行时数据，本轮 tiling 只校验 shape 和 `group_list_type`，不读取 GM 数据。

### 4.4 dtype/format/场景矩阵

| x dtype | weight dtype | y dtype | quant_dtype | weight format | transpose_weight | 是否支持 |
| --- | --- | --- | --- | --- | --- | --- |
| E4M3 | E4M3 | E4M3 | E4M3 | FRACTAL_NZ | false | 支持 |
| E4M3 | E5M2 | E4M3 | E4M3 | FRACTAL_NZ | false | 支持 |
| E5M2 | E4M3 | E5M2 | E5M2 | FRACTAL_NZ | false | 支持 |
| E5M2 | E5M2 | E5M2 | E5M2 | FRACTAL_NZ | false | 支持 |
| E4M3/E5M2 | E4M3/E5M2 | E4M3/E5M2 | 同 y | FRACTAL_NZ | true | 支持 |
| FP16/BF16/FP32 | 任意 | 任意 | 任意 | 任意 | 任意 | 不支持 |
| FP8 | FP8 | FP8 | 与 y 不一致 | FRACTAL_NZ | 任意 | 不支持 |
| FP8 | FP8 | FP8 | E4M3/E5M2 | ND weight | 任意 | 不支持 |

## 5. 交付标准

| 交付范围 | 当前迭代 |
| --- | --- |
| 交付通路 | Ascend950 AICore kernel，`opFile.value = grouped_matmul_activation_quant_apt` |
| 动态/静态要求 | 支持动态 shape；tiling 使用 `DynamicCompileStaticFlag(true)`、`DynamicShapeSupportFlag(true)` |
| 精度交付标准 | 与参考链路 `GMM(fp32 accum) -> GeluTanh -> Dynamic MX Quant` 对齐；具体容差待 STC 定义 |
| 性能交付标准 | 目标是减少中间 GM 读写；本 SRS 不给性能阈值，性能 benchmark 待补 |
| UT 交付 | CSV tiling UT 覆盖 dtype、shape、attr、core ratio、tiling key |
| 设计材料 | 本 SRS；代码路径见第 7 章 |
| 维测材料 | 暂无；依赖 tiling 日志和 UT 失败信息定位 |

## 6. 具体方案描述

### 6.1 Tiling 设计

#### 6.1.1 Tiling 数据结构

```cpp
BEGIN_TILING_DATA_DEF(GMMActivationQuantParams)
    uint32_t groupNum;
    uint8_t groupListType;
    uint8_t activationType;
    uint8_t quantDtype;
    uint8_t roundMode;
    uint8_t dequantDtype;
    uint8_t scaleAlg;
    uint32_t rowLen;      // 预留
    uint32_t ubAvail;     // 预留
    float dstTypeMax;
END_TILING_DATA_DEF;

BEGIN_TILING_DATA_DEF(GMMActivationQuantTilingDataParams)
    GMMActivationQuantParams gmmActivationQuantParams;
    QuantBasicApiMMTiling mmTilingData;
END_TILING_DATA_DEF;
```

`mmTilingData` 由 `GroupedQmmBasicApiTiling` 计算，kernel 侧转换成 `KernelGmmActivationMixOnlineDynamic::GMMTiling`。

#### 6.1.2 Tiling 主流程

```text
TilingPrepareForGroupedMatmulActivationQuant:
    platform = PlatformAscendC(context.platform)
    compileInfo.aicNum = platform.GetCoreNumAic()
    compileInfo.aivNum = platform.GetCoreNumAiv()
    compileInfo.socVersion = platform.GetSocVersion()
    compileInfo.npuArch = platform.GetCurNpuArch()
    compileInfo.ubSize = platform.GetCoreMemSize(UB)
    compileInfo.l1Size = platform.GetCoreMemSize(L1)
    compileInfo.l0ASize = platform.GetCoreMemSize(L0_A)
    compileInfo.l0BSize = platform.GetCoreMemSize(L0_B)
    compileInfo.l0CSize = platform.GetCoreMemSize(L0_C)
    compileInfo.l2Size = platform.GetCoreMemSize(L2)

GroupedMatmulActivationQuantTilingFunc:
    if context == nullptr:
        log "tiling context is nullptr"
        return GRAPH_FAILED

    // 当前只有 arch35/Ascend950 一套模板注册，registerList 固定为 {0}
    registerList = {0}
    return TilingRegistry.DoTilingImpl(context, registerList)

GroupedMatmulActivationQuantTiling950.DoTiling:
    Reset raw tiling data pointer
    GetShapeAttrsInfo from GroupedQmmBasicApiTiling
    AnalyzeAttrs
    AnalyzeDtype
    AnalyzeInputs
    DoOpTiling
    DoLibApiTiling
    GetWorkspaceSize
    SaveTilingDataToContext
```

#### 6.1.3 Attr 解析伪代码

```text
AnalyzeAttrs:
    attrs = context.GetAttrs()
    require attrs != nullptr

    transB = attr[transpose_weight] if exists else false
    transA = false

    groupListType = attr[group_list_type] if exists else default
    require groupListType in {0, 1}

    activationType = attr[activation_type]          // required attr
    dequantDtype = attr[dequant_dtype]              // optional attr with default FLOAT
    quantDtype = attr[quant_dtype]                  // optional attr with default E4M3
    roundMode = attr[round_mode]                    // optional attr with default "rint"
    scaleAlg = attr[scale_alg]                      // optional attr with default OCP
    dstTypeMax = attr[dst_type_max]                 // optional attr with default 0.0

    require all required quant attrs are non-null
    require activationType == 2  // GELU_TANH
    require dequantDtype == FLOAT
    require quantDtype in {FLOAT8_E4M3FN, FLOAT8_E5M2}
    require roundMode == "rint"
    require scaleAlg in {0, 1}
    require abs(dstTypeMax - 0.0) <= 1e-6

    roundMode_ = ROUND_MODE_RINT
    scaleAlg_ = scaleAlg
    activationType_ = activationType
    dstTypeMax_ = dstTypeMax
    quantDtype_ = quantDtype
    groupType = SPLIT_M
```

`tuning_config` 当前随 schema 保留，tiling 实现没有读取该字段；若后续需要指定 block shape 或策略开关，应在本流程中显式解析并写入 tiling data。

#### 6.1.4 dtype 解析伪代码

```text
AnalyzeDtype:
    xDesc = inputDesc[0]
    wDesc = inputDesc[3]        // dynamic weight list 当前按第 0 个 tensor 展平到该 index
    xScaleDesc = inputDesc[2]
    wScaleDesc = inputDesc[4]   // dynamic weight_scale list 当前按第 0 个 tensor 展平到该 index
    yDesc = outputDesc[0]
    yScaleDesc = outputDesc[1]
    require all desc are non-null

    aDtype = xDesc.dtype
    bDtype = wDesc.dtype
    perTokenScaleDtype = xScaleDesc.dtype
    scaleDtype = wScaleDesc.dtype
    outDataDtype = yDesc.dtype
    outScaleDtype = yScaleDesc.dtype
    bFormat = PrimaryFormat(wDesc.storageFormat)

CheckDtype:
    require IsFp8(aDtype)
    require IsFp8(bDtype)
    require perTokenScaleDtype == FLOAT8_E8M0
    require scaleDtype == FLOAT8_E8M0
    require IsFp8(outDataDtype)
    require outDataDtype == quantDtype_
    require outScaleDtype == FLOAT8_E8M0
```

#### 6.1.5 输入 shape 解析伪代码

```text
AnalyzeInputs:
    xShape = inputShape[x]
    weightShape = dynamicInputShape[weight][0]
    xScaleShape = inputShape[x_scale]
    weightScaleShape = dynamicInputShape[weight_scale][0]
    require all shape are non-null

    SetGroupNum(group_list_index)
    SetMKN(xShape.origin, weightShape.origin)

    aQuantMode = MX_PERGROUP_MODE
    bQuantMode = MX_PERGROUP_MODE
    isSingleX = true
    isSingleW = true
    isSingleY = true

    CheckMxScaleShape(xScaleShape.origin, weightScaleShape.origin)
    CheckWeightNzShape(weightShape.storage)
    CheckCoreNum()
```

#### 6.1.6 MX scale shape 校验伪代码

```text
CheckMxScaleShape(xScaleShape, wScaleShape):
    require rank(xScaleShape) == 3
    require rank(wScaleShape) == 4

    expectedKBlocks = ceil(kSize / 64)

    if transB:
        wScaleN = wScaleShape[1]
        wScaleK = wScaleShape[2]
    else:
        wScaleK = wScaleShape[1]
        wScaleN = wScaleShape[2]

    require xScaleShape[0] == mSize
    require xScaleShape[1] == expectedKBlocks
    require xScaleShape[2] == 2

    require wScaleShape[0] == groupNum
    require wScaleN == nSize
    require wScaleK == expectedKBlocks
    require wScaleShape[3] == 2
```

#### 6.1.7 weight NZ shape 校验伪代码

```text
CheckWeightNzShape(weightStorageShape):
    require bFormat == FRACTAL_NZ
    require rank(weightStorageShape) == 5
    require weightStorageShape[3] == 16
    require weightStorageShape[4] == 32
```

#### 6.1.8 BasicAPI tiling 伪代码

```text
DoLibApiTiling:
    GroupedQmmTiling.CalBasicBlock()

    baseM = min(basicTiling.baseM, 128)
    basicTiling.baseM = CeilAlign(baseM, CUBE_BLOCK)

    if GroupedQmmBasicApiTiling.CalL1Tiling() != GRAPH_SUCCESS:
        return GRAPH_FAILED

    mmTilingData.m = mSize
    mmTilingData.n = nSize
    mmTilingData.k = kSize
    mmTilingData.baseM = basicTiling.baseM
    mmTilingData.baseN = basicTiling.baseN
    mmTilingData.baseK = basicTiling.baseK
    mmTilingData.kAL1 = stepKa * baseK
    mmTilingData.kBL1 = stepKb * baseK
    mmTilingData.scaleKAL1 =
        min(max(scaleFactorA * stepKa, scaleFactorB * stepKb) * baseK, kSize)
    mmTilingData.scaleKBL1 = mmTilingData.scaleKAL1
    mmTilingData.isBias = 0
    mmTilingData.dbL0C = basicTiling.dbL0c
    return GRAPH_SUCCESS
```

#### 6.1.9 Op tiling data 写入伪代码

```text
DoOpTiling:
    gmmParams.groupNum = groupNum
    gmmParams.groupListType = groupListType
    gmmParams.activationType = activationType_
    if outDataDtype == FLOAT8_E5M2:
        gmmParams.quantDtype = FP8_E5M2_VALUE
    else:
        gmmParams.quantDtype = FP8_E4M3FN_VALUE
    gmmParams.roundMode = ROUND_MODE_RINT
    gmmParams.dequantDtype = FLOAT
    gmmParams.scaleAlg = scaleAlg_
    gmmParams.dstTypeMax = dstTypeMax_
```

`FP8_E5M2_VALUE`、`FP8_E4M3FN_VALUE` 使用当前 GE dtype enum 值写入 tiling data。文档中优先使用符号名，避免把 enum 数值误读成算法常量。

#### 6.1.10 tiling key

当前只生成两个有效 kernel 模板组合：

| tiling key 逻辑 | `QUANT_B_TRANS` | `QUANT_A_TRANS` | A layout | B layout | 是否支持 |
| --- | ---: | ---: | --- | --- | --- |
| no-trans weight | 0 | 0 | RowMajor | Nz | 支持 |
| trans weight | 1 | 0 | RowMajor | Zn | 支持 |
| trans x | 0/1 | 1 | - | - | 不支持 |

伪代码：

```text
GetTilingKey:
    return GET_TPL_TILING_KEY(transB, transA)

Kernel dispatch:
    if transB == 0 and transA == 0:
        call GmmActivationMxQuant<RowMajor, Nz>()
    else if transB == 1 and transA == 0:
        call GmmActivationMxQuant<RowMajor, Zn>()
```

#### 6.1.11 Tiling 类图

```mermaid
classDiagram
    class GroupedMatmulActivationQuantTiling950 {
        -GMMActivationQuantTilingDataParams tilingData_
        -uint8_t roundMode_
        -uint8_t scaleAlg_
        -uint8_t activationType_
        -float dstTypeMax_
        -ge::DataType quantDtype_
        +Reset()
        +GetShapeAttrsInfo()
        +AnalyzeAttrs()
        +AnalyzeDtype()
        +AnalyzeInputs()
        +CheckMxScaleShape()
        +CheckWeightNzShape()
        +CheckCoreNum()
        +DoOpTiling()
        +DoLibApiTiling()
        +GetTilingKey()
        +GetWorkspaceSize()
        +PostTiling()
    }

    class GroupedQmmBasicApiTiling {
        +CalBasicBlock()
        +CalL1Tiling()
        +SetGroupNum()
        +SetMKN()
        +CheckDtype()
    }

    class GMMActivationQuantParams {
        +groupNum
        +groupListType
        +activationType
        +quantDtype
        +roundMode
        +dequantDtype
        +scaleAlg
        +rowLen
        +ubAvail
        +dstTypeMax
    }

    class QuantBasicApiMMTiling {
        +m
        +n
        +k
        +baseM
        +baseN
        +baseK
        +kAL1
        +kBL1
        +scaleKAL1
        +scaleKBL1
        +isBias
        +dbL0C
    }

    class GMMActivationQuantTilingDataParams {
        +GMMActivationQuantParams gmmActivationQuantParams
        +QuantBasicApiMMTiling mmTilingData
    }

    GroupedMatmulActivationQuantTiling950 --|> GroupedQmmBasicApiTiling
    GroupedMatmulActivationQuantTiling950 --> GMMActivationQuantTilingDataParams
    GMMActivationQuantTilingDataParams *-- GMMActivationQuantParams
    GMMActivationQuantTilingDataParams *-- QuantBasicApiMMTiling
```

### 6.2 Kernel 设计

#### 6.2.1 Kernel 参数入口

```cpp
__global__ __aicore__ void grouped_matmul_activation_quant(
    GM_ADDR x,
    GM_ADDR groupList,
    GM_ADDR xScale,
    GM_ADDR weight,
    GM_ADDR weightScale,
    GM_ADDR bias,
    GM_ADDR y,
    GM_ADDR yScale,
    GM_ADDR workspace,
    GM_ADDR tiling);
```

说明：

- 参数顺序与 def 文件输入输出顺序一致。
- `bias` 为预留输入，当前 `(void)bias`，不进入 matmul 或 epilogue。
- kernel 会临时将 float overflow mode 控制位清 0，结束时恢复。

#### 6.2.2 Kernel 参数装配

```text
GmmActivationMxQuant:
    read gmmActivationQuantParams from tiling
    read mmTilingData from tiling

    BlockEpilogue =
        BlockEpilogueActivationQuant<L0TileShape, CType, float, E8M0, E8M0, true>

    BlockMmad =
        BlockMmadMx<MatmulWithScale, AType, LayoutA, BType, LayoutB, float>

    QGmmKernel =
        KernelGmmActivationMixOnlineDynamic<MatmulShape, BlockMmad, BlockEpilogue, Scheduler>

    gmmParams = {
        groupNum,
        m, n, k,
        baseM, baseN, baseK,
        kAL1, kBL1,
        scaleKAL1, scaleKBL1,
        isBias = 0,
        dbL0C,
        groupType = SPLIT_M,
        groupListType
    }

    params.mmadParams = {
        x, weight, weightScale, xScale, y, nullptr, groupList
    }

    params.epilogueParams = {
        y, yScale,
        x2Scale = nullptr,
        x1Scale = nullptr,
        bias = nullptr,
        baseM, baseN,
        roundMode,
        scaleAlg,
        dstTypeMax,
        activationType
    }

    QGmmKernel(params)
```

#### 6.2.3 Kernel 类图

```mermaid
classDiagram
    class grouped_matmul_activation_quant {
        <<global_kernel>>
        +SaveRestoreOverflowMode()
        +DispatchNzZnLayout()
    }

    class GmmActivationMxQuant {
        <<template_function>>
        +ReadTilingData()
        +BuildGmmTiling()
        +BuildParams()
        +InvokeQGmmKernel()
    }

    class KernelGmmActivationMixOnlineDynamic {
        +InitParamsAndTensor()
        +UpdateOffset()
        +ProcessSingleGroup()
        +operator()()
    }

    class BlockMmadMx {
        +UseMatmulWithScale()
        +UseARowMajor()
        +UseBNzOrZn()
        +UseCRowMajorFloat()
    }

    class BlockEpilogueActivationQuant {
        +Init()
        +UpdateGlobalAddr()
        +UpdateNextProblem()
        +VFDoActivationForMX()
        +ComputeMaxExp()
        +ComputeScale()
        +ComputeDataForQuantTargetFp8()
        +CopyOutputFromUb2Gm()
        +CopyScaleFromUb2Gm()
    }

    class GroupedMatmulAswtWithTailSplitScheduler {
        +GetTileIdx()
        +GetBlockShape()
        +GetQuantOffset()
    }

    grouped_matmul_activation_quant --> GmmActivationMxQuant
    GmmActivationMxQuant --> KernelGmmActivationMixOnlineDynamic
    KernelGmmActivationMixOnlineDynamic *-- BlockMmadMx
    KernelGmmActivationMixOnlineDynamic *-- BlockEpilogueActivationQuant
    KernelGmmActivationMixOnlineDynamic *-- GroupedMatmulAswtWithTailSplitScheduler
```

#### 6.2.4 分组调度流程

```mermaid
flowchart TD
    Start[Kernel operator()] --> Init[InitParamsAndTensor]
    Init --> AICInit{AIC?}
    AICInit -->|yes| InitMmad[Init BlockMmad with m,n,k/baseM/baseN/baseK/L1]
    AICInit -->|no| InitEpi
    InitMmad --> InitEpi[Init Epilogue]
    InitEpi --> Scheduler[Create tail-split scheduler]
    Scheduler --> LoopGroup[for groupIdx in groupNum]
    LoopGroup --> UpdateGroup[UpdateOffset + GetSplitValue]
    UpdateGroup --> Skip{M_i == 0?}
    Skip -->|yes| LoopGroup
    Skip -->|no| UpdateMmad[Update matmul problem shape]
    UpdateMmad --> L2[Set L2 cache hint]
    L2 --> TileLoop[ProcessSingleGroup tile loop]
    TileLoop --> LoopGroup
    LoopGroup --> End[Wait final vector if needed]
```

```mermaid
sequenceDiagram
    participant Kernel as KernelGmmActivationMixOnlineDynamic
    participant Scheduler as TailSplitScheduler
    participant AIC as AIC/BlockMmadMx
    participant AIV as AIV/BlockEpilogue
    participant GM as Global Memory

    Kernel->>Kernel: InitParamsAndTensor
    Kernel->>Scheduler: Create scheduler by group shape
    loop each group
        Kernel->>Kernel: UpdateOffset and split M_i
        alt M_i > 0
            Kernel->>Scheduler: GetTileIdx/GetBlockShape
            Scheduler-->>Kernel: tile shape and quant offsets
            AIC->>GM: load x, weight, x_scale, weight_scale
            AIC->>AIC: BlockMmadMx computes fp32 L0C tile
            AIC-->>AIV: NotifyVector
            AIV->>AIV: GeluTanh + MX dynamic quant
            AIV->>GM: write y and y_scale
            AIV-->>AIC: NotifyCube
        else M_i == 0
            Kernel->>Kernel: skip empty group
        end
    end
```

#### 6.2.5 group_list 语义

```text
GetSplitValueFromGroupList(groupIdx, groupListType):
    if groupListType == 0:
        // group_list 保存每组累加结束 offset
        offset = group_list[groupIdx]
        splitValue = offset - preOffset
        preOffset = offset
    else:
        // group_list 保存每组 token 数
        splitValue = group_list[groupIdx]
    return splitValue
```

#### 6.2.6 group 间 GM offset 更新

```text
UpdateOffset(groupIdx):
    if groupIdx == 0:
        return

    previousM = problemShape.m
    n = problemShape.n
    k = problemShape.k

    aOffset += previousM * k

    if B format is NZ:
        if transB:
            nAlign = align_up(n, BLOCK_CUBE)
            kAlign = align_up(k, c0Size)
        else:
            nAlign = align_up(n, c0Size)
            kAlign = align_up(k, BLOCK_CUBE)
        bOffset = nAlign * kAlign * groupIdx
    else:
        bOffset += n * k

    scaleK = ceil(k / 64) * 2
    xScaleOffset += previousM * scaleK
    wScaleOffset += n * scaleK
    yOffset += previousM * n
    yScaleOffset += previousM * ceil(n / 64) * 2
```

#### 6.2.7 tile 内 AIC/AIV 协同

```text
ProcessSingleGroup:
    bs.UpdateNextProblem({m_i, n, k, 0})
    epilogue.UpdateNextProblem({m_i, n, k, 0})
    UpdateGlobalBuffer(baseOffset)

    coord = Coordinate(m_i, n, k, baseM, baseN, baseK)

    while bs.GetTileIdx(tileIdx):
        singleShape = bs.GetBlockShape(tileIdx)
        if singleShape.m <= 0 or singleShape.n <= 0:
            return

        quantOffset = coord.GetQuantOffset(MX_PERGROUP_MODE, tileIdx, singleShape, c0Size)
        blockOffset.a = quantOffset.a
        blockOffset.b = quantOffset.b
        blockOffset.xScale = quantOffset.x1Scale
        blockOffset.wScale = quantOffset.x2Scale
        blockOffset.y = quantOffset.c
        blockOffset.yScale =
            mOffset * ceil(n / 64) * 2 + ceil(nOffset / 64) * 2

        if AIC:
            if previous tile exists:
                WaitForVector()
            mmSingleShape = {singleM, singleN, k}
            BlockMmad(aGlobal[aOffset], bGlobal[bOffset],
                      xScaleGlobal[xScaleOffset],
                      wScaleGlobal[wScaleOffset],
                      l0cOutUbFirst,
                      mmSingleShape)
            NotifyVector()

        if AIV:
            epilogueShape = {singleM, singleN, 0, 0}
            epilogueOffset = {yOffset, yScaleOffset, wScaleOffset, xScaleOffset, biasOffset}
            WaitForCube()
            Epilogue(epilogueShape, epilogueOffset)
            NotifyCube()
```

#### 6.2.8 Epilogue UB 规划

Epilogue 在 AIV 侧运行。UB 区域按固定偏移划分：

| UB tensor | 类型 | 大小/上限 | 用途 |
| --- | --- | --- | --- |
| `l0cOutUbFirst_` | `DataTypeIn` | `MAX_SINGLE_MN` | matmul 输出 tile，来源于 L0C -> UB |
| `quantOutput_` | `int8_t` | `MAX_SINGLE_MN` | FP8 量化后 y |
| `quantScaleOutput_` | `int8_t` | `MAX_SINGLE_MN / 32` | 临时 scale |
| `activationRes_` | `bfloat16_t` | `MAX_SINGLE_MN` | GeluTanh 后 BF16 中间结果 |
| `maxExp_` | `uint16_t` | `MAX_SINGLE_MN / 32` | 每 32 元素规约 max exponent |
| `halfScale_` | `uint16_t` | `MAX_SINGLE_MN / 32` | 反量化/量化用 half scale |
| `quantScaleBlockOutput_` | `int8_t` | `baseM / taskRatio * 32` | GM 写回前 scale layout |

#### 6.2.9 Epilogue 主流程伪代码

```text
BlockEpilogueActivationQuant::operator()(blockShape, blockCoord):
    singleM = blockShape.m
    singleN = blockShape.n
    scaleBlockN = ceil(singleN / 64) * 2

    halfSingleM = ceil(singleM / taskRatio)
    if subBlockIdx == 1:
        singleMInVec = singleM - halfSingleM
    else:
        singleMInVec = halfSingleM

    if singleMInVec == 0:
        return

    vlForHalfNumber = VECTOR_REG_WIDTH / sizeof(bfloat16_t)
    elementAfterReduce = VECTOR_REG_WIDTH / 32

    VFDoActivationForMX(singleMInVec)

    yOffset = blockCoord.y + subBlockIdx * halfSingleM * fullN
    yScaleOffset = blockCoord.yScale + subBlockIdx * halfSingleM * fullScaleN

    TransMxScaleLayout(singleMInVec, scaleBlockN)

    CopyOutputFromUb2Gm(singleMInVec, yOffset, quantOutput)
    CopyScaleFromUb2Gm(singleMInVec, yScaleOffset, quantScaleBlockOutput)
```

#### 6.2.10 GeluTanh 计算伪代码

当前实现使用 sigmoid 形式的 GeluTanh 近似：

```text
GeluTanh(x) = x / (1 + exp(-1.595769121 * (x + 0.044715 * x^3)))
```

代码中的等价计算：

```text
inputSqr = x * x
inputCub = inputSqr * x
inputCub = inputCub + x * (1 / 0.044715)
inputCub = inputCub * (-1.595769121 * 0.044715)
denominator = exp(inputCub) + 1
actOut = x / denominator
activationRes = cast_bf16_rint(actOut)
```

逐 tile 伪代码：

```text
VFDoActivationAndQuantForMX(src, mSize, nSize):
    sizePerRepeat = VECTOR_REG_WIDTH / sizeof(DataTypeIn)
    repeatsPerRow = ceil(nSize / sizePerRepeat)
    nSrcAligned = align_up(nSize, 32 / sizeof(DataTypeIn))
    nDstAligned = align_up(nSize, 32)

    for mIdx in [0, mSize):
        mask = UpdateMask(nSize)
        for vfBlockIdx in [0, repeatsPerRow):
            offset = mIdx * nSrcAligned + vfBlockIdx * sizePerRepeat
            input = load_fp32(src + offset)
            act = GeluTanh(input)
            store_bf16(activationRes + mIdx * nDstAligned + vfBlockIdx * sizePerRepeat, act)

    DynamicMxQuant(activationRes, nDstAligned)
```

#### 6.2.11 MX dynamic quant 伪代码

量化主流程：

```mermaid
flowchart LR
    A[activationRes BF16 tile] --> B{scale_alg}
    B -->|0 OCP| C[每 32 元素规约 max exponent]
    B -->|1 cuBLAS| D[每 32 元素规约 max abs]
    C --> E[生成 E8M0 mxScale 和 BF16 halfScale]
    D --> E
    E --> F[activationRes * halfScale]
    F --> G[saturate + rint cast to FP8]
    G --> H[quantOutput UB]
    E --> I[quantScaleOutput UB]
    I --> J[scale layout 转换]
    H --> K[y GM]
    J --> L[y_scale GM]
```

```text
DynamicMxQuant(activationRes, mSize, nDstAligned):
    // activationRes 已按 row 存成 BF16，nDstAligned 按 FP8 写回粒度对齐
    totalData = mSize * nDstAligned
    totalScaleGroup = totalData / 32

    if scaleAlg == OCP:
        maxStat = ReduceMaxExponentPerGroup32(activationRes, totalScaleGroup)
        mxScale, halfScale = BuildScaleByOcpExponent(maxStat)
    else:
        maxStat = ReduceMaxAbsPerGroup32(activationRes, totalScaleGroup)
        mxScale, halfScale = BuildScaleByCublasAbs(maxStat, dtypeMax)

    quantOutput = CastScaledDataToFp8(activationRes, halfScale)
    quantScaleOutput = PackMxScale(mxScale)
    return quantOutput, quantScaleOutput
```

OCP scale 分支按 BF16 exponent 选择共享 scale，和 `dynamic_mx_quant_tail_axis_fp8.h` 的 OCP 路径保持同一类思路：

```text
ReduceMaxExponentPerGroup32(src, totalScaleGroup):
    for groupIdx in [0, totalScaleGroup):
        maxExp = 0
        for value in src[groupIdx * 32 : groupIdx * 32 + 32):
            bf16Exp = bits(value) & BF16_EXP_MASK
            maxExp = max(maxExp, bf16Exp)
        maxExpBuf[groupIdx] = maxExp
    return maxExpBuf

BuildScaleByOcpExponent(maxExpBuf):
    for each maxExp in maxExpBuf:
        if maxExp represents INF or NaN:
            mxScale = FP8_NAN_SCALE
            halfScale = BF16_NAN
            continue

        // 未超过目标 FP8 exponent 范围时，不额外放大共享 exponent。
        clippedExp = max(maxExp, fp8MaxExp)
        sharedExp = clippedExp - fp8MaxExp

        mxScale = sharedExp >> BF16_EXP_SHIFT
        halfScale = BF16_EXP_BIAS - sharedExp

        if sharedExp == 0:
            halfScale = 0
        if sharedExp hits special small-exponent range:
            halfScale = SPECIAL_EXP_THRESHOLD

        store mxScale, halfScale
```

cuBLAS scale 分支先规约绝对值，再按目标 FP8 dtype 的最大值反推共享 exponent：

```text
ReduceMaxAbsPerGroup32(src, totalScaleGroup):
    for groupIdx in [0, totalScaleGroup):
        maxAbs = 0
        for value in src[groupIdx * 32 : groupIdx * 32 + 32):
            absValue = bits(value) & ABS_MASK_FOR_16BIT
            maxAbs = max(maxAbs, absValue)
        maxAbsBuf[groupIdx] = maxAbs
    return maxAbsBuf

BuildScaleByCublasAbs(maxAbsBuf, dtypeMax):
    for each maxAbs in maxAbsBuf:
        if maxAbs == 0:
            mxScale = 0
            halfScale = 0
            continue
        if maxAbs represents INF or NaN:
            mxScale = FP8_NAN_SCALE
            halfScale = BF16_NAN
            continue

        scaledMax = cast_bf16_to_fp32(maxAbs) * dtypeMax
        exp32 = extract_fp32_exponent(scaledMax)
        man32 = extract_fp32_mantissa(scaledMax)

        needRoundUp =
            (0 < exp32 < FP32_MAX_EXP and man32 > 0) or
            (exp32 == 0 and man32 > 0.5)
        sharedExp = exp32 + (needRoundUp ? 1 : 0)

        mxScale = low_u16(sharedExp)
        halfScale = FP32_EXP_BIAS_CUBLAS - (sharedExp << BF16_EXP_SHIFT)
        store mxScale, halfScale
```

FP8 输出 cast：

```text
CastScaledDataToFp8(activationRes, halfScale):
    for each vector chunk:
        data = load_bf16_interleaved(activationRes)
        scale = broadcast_group_scale(halfScale)
        scaled = data * scale

        fp32Lo, fp32Hi = cast_bf16_to_fp32_for_fp8_cast(scaled)
        fp8Lo, fp8Hi = cast_fp32_to_fp8_sat_rint(fp32Lo, fp32Hi)
        packedFp8 = pack_fp8_lanes(fp8Lo, fp8Hi)
        store packedFp8 to quantOutput
```

Scale layout 转换：

```text
TransMxScaleLayout(mSize, scaleBlockN):
    // quantScaleOutput 是紧凑 scale；quantScaleBlockOutput 按 GM DataCopyPad 写回粒度重排
    for mIdx in [0, mSize):
        src = quantScaleOutput + mIdx * scaleBlockN
        dst = quantScaleBlockOutput + mIdx * 32
        copy scaleBlockN bytes from src to dst
```

GM 写回：

```text
CopyOutputFromUb2Gm:
    blockCount = singleMInVec
    blockLen = singleN * sizeof(int8)
    dstStride = (fullN - singleN) * sizeof(int8)
    DataCopyPad(yGM + yOffset, quantOutput, params)

CopyScaleFromUb2Gm:
    blockScaleN = ceil(singleN / 64) * 2
    blockCount = singleMInVec
    blockLen = blockScaleN * sizeof(int8)
    dstStride = (fullScaleN - blockScaleN) * sizeof(int8)
    DataCopyPad(yScaleGM + yScaleOffset, quantScaleBlockOutput, params)
```

## 7. 软件设计和接口定义

### 7.1 文件级修改清单

| 文件 | 设计职责 |
| --- | --- |
| `op_host/grouped_matmul_activation_quant_def.cpp` | 定义 `GroupedMatmulActivationQuant` schema；按指定顺序声明输入/输出/属性 |
| `op_host/grouped_matmul_activation_quant_infershape.cpp` | 推导 `y` 和 `y_scale` shape/dtype；跟随 attr index 顺序 |
| `op_host/op_tiling/arch35/grouped_matmul_activation_quant_tiling.{h,cpp}` | 校验参数；生成 GMM BasicAPI tiling 和融合参数 tiling |
| `op_kernel/arch35/grouped_matmul_activation_quant_apt.cpp` | kernel 入口；根据 tiling key dispatch layout |
| `op_kernel/arch35/grouped_matmul_activation_quant_mxquant.h` | 组装 BlockMmad、BlockEpilogue、Kernel 参数 |
| `common/cgmct/kernel/kernel_gmm_activation_mxquant.h` | 分组 GMM kernel 主调度 |
| `common/cgmct/epilogue/block_epilogue_activation_mx_quant.h` | GeluTanh + MX dynamic quant epilogue |
| `tests/ut/op_host/test_grouped_matmul_activation_quant_tiling.*` | CSV tiling UT |

### 7.2 详细修改伪代码

#### 7.2.1 def schema

```text
DefineOp GroupedMatmulActivationQuant:
    Input("x").Required().Dtype(FP8_E4M3/E5M2).Format(ND)
    Input("group_list").Required().Dtype(INT64).Format(ND)
    Input("x_scale").Optional().Dtype(FLOAT8_E8M0).Format(ND)
    Input("weight").Dynamic().Dtype(FP8_E4M3/E5M2).Format(FRACTAL_NZ)
    Input("weight_scale").Dynamic().Dtype(FLOAT8_E8M0).Format(ND)
    Input("bias").Dynamic().Dtype(FLOAT).Format(ND)
    Output("y").Required().Dtype(FP8_E4M3/E5M2).Format(ND)
    Output("y_scale").Required().Dtype(FLOAT8_E8M0).Format(ND)

    Attr("activation_type").Required().Int()
    Attr("dequant_dtype").Optional().Int(FLOAT)
    Attr("transpose_weight").Optional().Bool(false)
    Attr("group_list_type").Optional().Int(0)
    Attr("tuning_config").Optional().ListInt({0})
    Attr("quant_dtype").Optional().Int(FLOAT8_E4M3FN)
    Attr("round_mode").Optional().String("rint")
    Attr("scale_alg").Optional().Int(0)
    Attr("dst_type_max").Optional().Float(0.0)

    AICore("ascend950")
        DynamicCompileStaticFlag(true)
        DynamicFormatFlag(true)
        DynamicRankSupportFlag(true)
        DynamicShapeSupportFlag(true)
        NeedCheckSupportFlag(false)
        PrecisionReduceFlag(true)
        opFile = "grouped_matmul_activation_quant_apt"
```

#### 7.2.2 InferShape

```text
InferShape:
    xShape = input[0].shape
    weightScaleShape = dynamicInput[4][0].shape
    transposeWeight = attr[2]

    M = xShape[0]
    if transposeWeight:
        N = weightScaleShape[1]
    else:
        N = weightScaleShape[2]

    y.shape = [M, N]
    if N == -1:
        yScale.shape = [M, -1, 2]
    else:
        yScale.shape = [M, ceil(N / 64), 2]

InferDataType:
    quantDtype = attr[5]
    y.dtype = quantDtype
    yScale.dtype = FLOAT8_E8M0
```

#### 7.2.3 Tiling

```text
Tiling constants:
    X_INDEX = 0
    GROUP_LIST_INDEX = 1
    X_SCALE_INDEX = 2
    WEIGHT_INDEX = 3
    WEIGHT_SCALE_INDEX = 4

    ATTR_ACTIVATION_TYPE = 0
    ATTR_DEQUANT_DTYPE = 1
    ATTR_TRANSPOSE_WEIGHT = 2
    ATTR_GROUP_LIST_TYPE = 3
    ATTR_TUNING_CONFIG = 4
    ATTR_QUANT_DTYPE = 5
    ATTR_ROUND_MODE = 6
    ATTR_SCALE_ALG = 7
    ATTR_DST_TYPE_MAX = 8

Tiling flow:
    Reset -> GetShapeAttrsInfo -> AnalyzeAttrs -> AnalyzeDtype -> AnalyzeInputs
    -> DoOpTiling -> DoLibApiTiling -> GetWorkspaceSize -> PostTiling
```

#### 7.2.4 Kernel dispatch

```text
KernelEntry:
    save overflowMode
    set overflowMode = 0

    switch (QUANT_B_TRANS, QUANT_A_TRANS):
        (0, 0):
            GmmActivationMxQuant<RowMajor, Nz>(x, weight, weightScale, xScale, groupList, y, yScale)
        (1, 0):
            GmmActivationMxQuant<RowMajor, Zn>(x, weight, weightScale, xScale, groupList, y, yScale)
        otherwise:
            no-op by template selection; not registered

    restore overflowMode
```

#### 7.2.5 Epilogue activation + quant

```text
EpilogueTile:
    input: L0C fp32 tile in UB
    for each row fragment assigned to subBlock:
        act = GeluTanh(l0c)
        actBf16 = cast_bf16(act)

    if scaleAlg == 0:
        mxScale, halfScale = OCP_MX_Scale(actBf16)
    else:
        mxScale, halfScale = cuBLAS_MX_Scale(actBf16)

    yFp8 = cast_fp8_sat_rint(actBf16 * halfScale)
    write yFp8 and mxScale to GM
```

## 8. UT 设计

### 8.1 UT 目标与边界

本轮 UT 以 tiling 白盒校验为主，目标是确认 host 侧 schema 顺序、dtype/format/shape/attr 准入、tiling key 和核心 compileInfo 约束符合设计。UT 不执行 AICore kernel，因此不覆盖 GeluTanh 数值精度、MX quant bit 级一致性和性能指标；这些内容需要后续 STC 或端到端精度测试补齐。

| 范围 | 覆盖方式 |
| --- | --- |
| def 顺序对齐 | CSV scalar 字段和 tensor 字段按 def 顺序构造 `TilingContextPara` |
| attr 准入 | 合法/非法 `activation_type`、`group_list_type`、`quant_dtype`、`round_mode`、`scale_alg`、`dst_type_max` |
| dtype/format 准入 | FP8 x/weight/y，E8M0 scale，NZ weight |
| shape 准入 | `x_scale`、`weight_scale`、`weight.storage` rank/tail dim |
| tiling key | no-trans 期望 key 0；trans weight 期望 key 1 |
| compileInfo | Ascend950，AIC/AIV 1:2 核比例 |
| kernel 数值 | 不在本 UT 覆盖范围内 |

### 8.2 UT 文件结构

| 文件 | 职责 |
| --- | --- |
| `tests/ut/op_host/test_grouped_matmul_activation_quant_tiling.cpp` | 读取 CSV、构造 `TilingContextPara`、执行 tiling、检查返回状态和 tiling key |
| `tests/ut/op_host/test_grouped_matmul_activation_quant_tiling.csv` | 表驱动用例数据，覆盖合法/非法场景 |
| `gmm_csv_ge_parse_utils.h` | 解析 dtype、format、shape、bool、字符串并生成安全参数名 |
| `tiling_case_executor.h` | 封装 `ExecuteTiling` / `ExecuteTestCase` |

执行结构如下：

```mermaid
flowchart TD
    CSV[test_grouped_matmul_activation_quant_tiling.csv] --> Loader[LoadCases]
    Loader --> Parse[Parse scalar attrs and tensor descs]
    Parse --> Context[Build TilingContextPara]
    Context --> Tiling{expectSuccess?}
    Tiling -->|false| FailPath[ExecuteTestCase expect GRAPH_FAILED]
    Tiling -->|true| SuccessPath[ExecuteTiling]
    SuccessPath --> KeyCheck[Compare tilingKey]
    FailPath --> End[case done]
    KeyCheck --> End
```

### 8.3 CSV 字段设计

CSV 采用“固定 scalar 字段 + 固定 tensor 字段”的方式，避免 C++ 侧为每类 case 写重复样板。

- scalar 字段数：`18`。
- 输入 tensor 数：`6`，顺序为 `x`、`group_list`、`x_scale`、`weight`、`weight_scale`、`bias`。
- 输出 tensor 数：`2`，顺序为 `y`、`y_scale`。
- 每个 tensor 固定 4 个字段：`origin`、`storage`、`dtype`、`format`。
- 总列数：`18 + (6 + 2) * 4 = 50`。

scalar 字段顺序如下：

| 顺序 | 字段 | 用途 |
| ---: | --- | --- |
| 1 | `socVersion` | 只加载 `Ascend950` 用例 |
| 2 | `caseName` | 原始用例名，便于解析失败定位 |
| 3 | `enable` | 用例开关 |
| 4 | `prefix` | gtest 参数名来源 |
| 5 | `expectSuccess` | true 走成功 tiling；false 期望 `GRAPH_FAILED` |
| 6 | `expectTilingKey` | 成功 case 校验 tiling key |
| 7 | `aicNum` | compileInfo AIC 数 |
| 8 | `aivNum` | compileInfo AIV 数 |
| 9 | `ubSize` | compileInfo UB 大小 |
| 10 | `activationType` | 对应 def attr index 0 |
| 11 | `dequantDtype` | 对应 def attr index 1 |
| 12 | `transposeWeight` | 对应 def attr index 2 |
| 13 | `groupListType` | 对应 def attr index 3 |
| 14 | `tuningConfig` | 对应 def attr index 4，当前默认 `0` |
| 15 | `quantDtype` | 对应 def attr index 5 |
| 16 | `roundMode` | 对应 def attr index 6 |
| 17 | `scaleAlg` | 对应 def attr index 7 |
| 18 | `dstTypeMax` | 对应 def attr index 8 |

tensor 字段顺序如下：

| 顺序 | tensor | 字段组 | 说明 |
| ---: | --- | --- | --- |
| 1 | `x` | `x_origin/x_storage/x_dtype/x_format` | REQUIRED_INPUT 0 |
| 2 | `group_list` | `grouplist_origin/grouplist_storage/grouplist_dtype/grouplist_format` | REQUIRED_INPUT 1 |
| 3 | `x_scale` | `xscale_origin/xscale_storage/xscale_dtype/xscale_format` | OPTIONAL_INPUT 2，当前 MX path 需要有效 desc |
| 4 | `weight` | `weight_origin/weight_storage/weight_dtype/weight_format` | DYNAMIC_INPUT 3，第 0 个 tensor |
| 5 | `weight_scale` | `wscale_origin/wscale_storage/wscale_dtype/wscale_format` | DYNAMIC_INPUT 4，第 0 个 tensor |
| 6 | `bias` | `bias_origin/bias_storage/bias_dtype/bias_format` | DYNAMIC_INPUT 5，当前允许 `NONE/UNDEFINED` 表示不传有效 bias |
| 7 | `y` | `y_origin/y_storage/y_dtype/y_format` | OUTPUT 0 |
| 8 | `y_scale` | `yscale_origin/yscale_storage/yscale_dtype/yscale_format` | OUTPUT 1 |

### 8.4 C++ 驱动伪代码

```text
LoadCases(socVersion):
    csvPath = ResolveCsvPath("test_grouped_matmul_activation_quant_tiling.csv")
    for each non-empty and non-comment line:
        items = split by comma
        if header or column count < 50:
            continue

        tc.socVersion = items[0]
        if tc.socVersion != socVersion:
            continue

        parse scalar fields in fixed order
        parse 6 input tensor descs
        parse 2 output tensor descs
        append tc
    return cases

GroupedMatmulActivationQuantTilingCase.Run():
    if !enable:
        return

    compileInfo = {
        aicNum, aivNum, ubSize,
        default L1/L2/L0A/L0B/L0C,
        socVersion = ASCEND950,
        npuArch = DAV_3510
    }

    attrs = {
        activation_type,
        dequant_dtype,
        transpose_weight,
        group_list_type,
        tuning_config,
        quant_dtype,
        round_mode,
        scale_alg,
        dst_type_max
    }

    context = TilingContextPara(
        opName = "GroupedMatmulActivationQuant",
        inputs = BuildTensorDescs(6 tensors),
        outputs = BuildTensorDescs(2 tensors),
        attrs,
        compileInfo
    )

    if expectSuccess == false:
        ExecuteTestCase(context, GRAPH_FAILED)
        return

    tilingInfo = ExecuteTiling(context)
    require tilingInfo.tilingKey == expectTilingKey
```

### 8.5 当前 CSV 用例覆盖

| case | 类型 | 关键输入 | 期望 |
| --- | --- | --- | --- |
| `valid_no_trans_e4m3` | 合法 | E4M3，`transposeWeight=false`，OCP scale | tiling 成功，key 0 |
| `valid_trans_e5m2_cublas` | 合法 | E5M2，`transposeWeight=true`，cuBLAS scale | tiling 成功，key 1 |
| `valid_grouplist_count` | 合法 | `group_list_type=1`，多 group count 模式 | tiling 成功，key 0 |
| `invalid_activation_type` | 非法 attr | `activation_type != 2` | `GRAPH_FAILED` |
| `invalid_group_list_type` | 非法 attr | `group_list_type` 不在 `{0,1}` | `GRAPH_FAILED` |
| `invalid_round_mode` | 非法 attr | `round_mode != "rint"` | `GRAPH_FAILED` |
| `invalid_scale_alg` | 非法 attr | `scale_alg` 不在 `{0,1}` | `GRAPH_FAILED` |
| `invalid_dst_type_max` | 非法 attr | `dst_type_max != 0.0` | `GRAPH_FAILED` |
| `invalid_quant_dtype` | 非法 attr | `quant_dtype` 不是 FP8 | `GRAPH_FAILED` |
| `invalid_y_dtype_mismatch` | 非法 dtype | `y.dtype != quant_dtype` | `GRAPH_FAILED` |
| `invalid_weight_nz_tail` | 非法 storage | NZ tail dim 不是 `16/32` | `GRAPH_FAILED` |
| `invalid_xscale_shape` | 非法 shape | `x_scale[1] != ceil(K/64)` | `GRAPH_FAILED` |
| `invalid_core_ratio` | 非法 compileInfo | AIC/AIV 不是 1:2 | `GRAPH_FAILED` |

### 8.6 建议补充用例

| 覆盖点 | 建议用例 | 目的 |
| --- | --- | --- |
| `x_scale` 缺失 | `x_scale` 使用 `NONE/UNDEFINED` 或空 shape | 明确 optional schema 在 MX path 下的失败行为 |
| `weight_scale` 转置维错误 | trans 场景交换 N/KBlock 维 | 覆盖 `transpose_weight=true` 下的 scale shape 校验 |
| `y_scale` dtype 错误 | `y_scale.dtype != FLOAT8_E8M0` | 补齐输出 scale dtype 负例 |
| weight format 错误 | `weight.format=ND` | 覆盖只支持 NZ weight 的负例 |
| bias 策略 | 非空 bias dynamic tensor | 明确当前“忽略”还是后续“非法”的产品策略 |
| group 数边界 | `group_list` shape 过大或空 group | 补充 group 调度边界 |
| mixed FP8 | `x`/`weight` 使用不同 FP8 dtype | 覆盖 dtype 组合矩阵 |

## 9. 外部依赖

| 依赖项 | 依赖内容 | 影响 |
| --- | --- | --- |
| CANN/AscendC | `kernel_operator.h`、MicroAPI、tiling registry、template argument API | 高 |
| Ascend950 | 需要 AIC/AIV 1:2 核比例；`__NPU_ARCH__ == 3510` epilogue 编译分支 | 高 |
| GMM BasicAPI | `GroupedQmmBasicApiTiling`、`BlockMmadMx`、`GroupedMatmulAswtWithTailSplitScheduler` | 高 |
| MX quant 约定 | `FLOAT8_E8M0` scale、`ceil(K/64)`、`ceil(N/64)`、pair dim = 2 | 高 |
| Dynamic MX Quant 参考实现 | tail-axis FP8 dynamic quant 的 max exponent、scale、cast 思路 | 中 |
| UT 框架 | `TilingContextPara`、CSV parse util、`ExecuteTiling` | 中 |

## 10. 待确认事项

1. `x_scale` 在 def 中为 optional，但当前 MX kernel/tiling 实际要求必传；是否需要在接口文档中显式声明“optional but required for MX path”。
2. `bias` 当前按 schema 预留为 dynamic input，但 kernel 不使用；后续是否需要支持 matmul bias 或 activation 前/后 bias。
3. `activation_type` 当前只支持 GELU_TANH=2；后续若增加 ReLU/Silu/FastGelu，需要扩展 epilogue 分支和 STC。
4. `rowLen`、`ubAvail` tiling 字段当前预留未写入；是否保留或清理。
5. 性能交付阈值和精度容差需要 STC/benchmark 补齐。
6. 当前本地环境缺少 CMake，UT 需在 CI 或标准开发环境中执行确认。

## 附录 A. UT/STC 交叉检查清单

| 检查域 | 设计约束 | 当前 CSV UT | 后续 STC/精度建议 |
| --- | --- | --- | --- |
| schema 顺序 | 输入、输出、attr 按第 4.1 节顺序声明和解析 | 通过 50 列 CSV 固定顺序覆盖 | aclnn/op_api 若后续补交付，需要同步接口文档 |
| activation | 仅 `activation_type=2` | 正例和非法 attr 已覆盖 | 端到端校验 GeluTanh 公式近似误差 |
| FP8 dtype | `x/weight/y/quant_dtype` 仅 E4M3/E5M2 | E4M3、E5M2、非法 quant、y mismatch 已覆盖 | 增加 mixed FP8 数值 case |
| scale dtype | `x_scale/weight_scale/y_scale` 为 E8M0 | 正例覆盖，`y_scale` dtype 负例待补 | STC 对齐 dynamic MX quant scale byte |
| weight format | 仅 FP8 `FRACTAL_NZ` | NZ tail dim 负例已覆盖 | 增加 ND format 负例 |
| transpose | `x` 不转置，`weight` 支持转置/非转置 | key 0/key 1 已覆盖 | STC 分别跑 no-trans/trans 数值 |
| group_list | 支持 offset 模式和 count 模式 | `group_list_type=0/1` 已覆盖 | 增加空 group、极端 group 数 |
| MX shape | `x_scale`、`weight_scale` rank 和块维匹配 | `x_scale` shape 负例已覆盖 | 增加 trans 下 `weight_scale` 维度负例 |
| core ratio | AIC:AIV 必须为 1:2 | 非法 core ratio 已覆盖 | CI 环境确认真实 Ascend950 compileInfo |
| bias | 当前 kernel 忽略 `bias` | 当前 CSV 使用 `NONE/UNDEFINED` | 非空 bias 的产品策略需确认后补测 |
| quant 数值 | OCP/cuBLAS 两条 scale 分支 | tiling 仅覆盖 `scale_alg` 准入 | STC 比对参考链路输出 `y/y_scale` |
| 性能 | 本轮只给融合方向，不给阈值 | 不适用 | benchmark 定义 GM 读写收益和性能阈值 |
