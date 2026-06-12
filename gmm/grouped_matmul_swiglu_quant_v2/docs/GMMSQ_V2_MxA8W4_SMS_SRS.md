# grouped_matmul_swiglu_quant_v2 MxA8W4 单多单场景支持 SRS

---

## 1. 文档信息

| 项目 | 内容 |
| --- | --- |
| 文档标题 | grouped_matmul_swiglu_quant_v2 MxA8W4 单多单场景支持 SRS |
| 迭代/特性名称 | GMMSQ V2 MxA8W4 单多单支持 |
| 算子类型 | 融合算子 |
| 主领域目录 | gmm/grouped_matmul_swiglu_quant_v2 |
| 责任人 | 待定 |
| 更新时间 | 2026-06-11 |
| 文档版本 | v0.1 |
| 关联 STC | 待补充 |
| 关联里程碑/需求单 | 无 |

---

## 2. 需求背景

### 2.1 背景

`grouped_matmul_swiglu_quant_v2` 在 arch35 MxA8W4 weight quant 路径下已有专用 tiling 与 kernel controller：

```text
op_host/op_tiling/arch35/grouped_matmul_swiglu_quant_v2_weight_quant_tiling.cpp
op_kernel/arch35/weight_quant_basic_block/gmmsq_weight_quant_resplit_controller.h
```

原路径默认 `weight` 与 `weightScale` 为单 tensor，group 维度在单 tensor 内部展开。group 计算完成后，`UpdateGmAddr` 按连续 offset 推进地址：

```cpp
weightGm_ += (nSize * kSize) >> 1;
weightScaleGm_ += nSize * CeilDivide(kAlignSize, static_cast<uint64_t>(tiling_->groupSize));
```

该逻辑适用于如下单 tensor 存储语义：

```text
weight      : view [E, N, K]，storage [E, ceil(K / 32), ceil(N / 16), 16, 32]
weightScale : [E, N, ceil(K / 64), 2]
```

新需求要求 MxA8W4 weight quant 路径支持“单多单”场景：

- `x` 为单 tensor；
- `weight` 为 tensor list，多 tensor，每个 group 对应一个独立 `weight[i]`；
- `weightScale` 为 tensor list，多 tensor，每个 group 对应一个独立 `weightScale[i]`；
- `y` / `output` 为单 tensor；
- `xScale`、`yScale` 仍为单 tensor。

单多单场景下，`weight` / `weightScale` 不能继续按单 tensor 内部 offset 推进，需要按 group 下标从 tensor list 中取址。

### 2.2 目标

本轮交付目标：

1. 为 GMMSQ V2 arch35 MxA8W4 weight quant 路径增加单多单能力；
2. 通过 TilingKey 传递 `IS_SINGLE_MULTI_SINGLE`，使 kernel 在编译期区分单多单和非单多单路径；
3. 在单多单场景下，kernel 使用 `GetTensorAddr(groupIdx, tensorList)` 切换 `weight` / `weightScale` 地址；
4. 保持 `x`、`xScale`、`y`、`yScale` 按单 tensor 内部 offset 推进；
5. host tiling 能识别 weight 多 tensor，校验 `weight` / `weightScale` 多 tensor shape，并生成对应 TilingKey；
6. 保持非 MxA8W4、非 weight quant、非单多单路径行为不变。

本期不做：

- 非 MxA8W4 数据流的单多单扩展；
- 非 arch35 路径改造；
- 非 weight quant 路径的模板参数变更；
- API utils 大范围重构；
- `weightAssistMatrix`、`bias`、`smoothScale` 在该 MxA8W4 weight quant 路径中的新增支持；
- 算子数学定义、输出布局、量化语义变更。

### 2.3 范围

- **包含**：
  - `grouped_matmul_swiglu_quant_v2_weight_quant_tiling_key.h` 新增 `IS_SINGLE_MULTI_SINGLE`；
  - `grouped_matmul_swiglu_quant_v2_apt.cpp` 的 weight quant kernel 模板新增 `IS_SINGLE_MULTI_SINGLE`；
  - `GMMSQWeightQuantResplitController` 新增 `isSingleMultiSingle` 模板参数；
  - `UpdateGmAddr` 支持单多单场景下 `weight` / `weightScale` tensor list 取址；
  - host tiling 识别 `weight` 多 tensor，设置 `GMMSQWeightQuantInputParams::isSingleMultiSingle`；
  - host tiling 校验单多单 `weight` / `weightScale` tensor list shape。
- **不包含**：
  - `gmm/grouped_matmul_swiglu_quant_v2/op_api/grouped_matmul_swiglu_quant_v2_utils.h` 修改；
  - 非 MxA8W4 数据流；
  - 非 arch35 / 非 weight quant 路径；
  - `groupListType` 语义变更；
  - `weightAssistMatrix`、`bias`、`smoothScale` 的新增可用路径。

---

## 3. 方案简述

| 层次 | 要点 |
| --- | --- |
| Host | 识别 `weight[1]` 是否存在以判断单多单；校验 MxA8W4 dtype / format / attrs；单多单下按 tensor list 校验每个 `weight[i]`、`weightScale[i]` shape；设置 `isSingleMultiSingle` |
| Tiling | `GMMSQWeightQuantInputParams` 新增 `isSingleMultiSingle`；`GET_TPL_TILING_KEY` 透传 `IS_SINGLE_MULTI_SINGLE`；`GMMSQWeightQuantTilingData` 继续携带 `groupNum / groupSize / kSize / nSize` |
| Kernel | MxA8W4 weight quant kernel 模板新增 `bool IS_SINGLE_MULTI_SINGLE`；controller 通过 `if constexpr` 选择 tensor list 取址或连续 offset 推进 |
| 测试/交付 | 覆盖 MxA8W4 单多单 happy path、非单多单回归、`weightScale` list size / shape 非法路径、TilingKey 白盒检查 |

---

## 4. 数学定义与语义

本需求不改变 `grouped_matmul_swiglu_quant_v2` 的数学语义，仅改变 MxA8W4 单多单场景下 group 间 `weight` / `weightScale` 的地址推进方式。

对于第 `g` 个 group：

```text
Matmul_g = Matmul(X_g, Weight_g, XScale_g, WeightScale_g)
Y_g, YScale_g = SwigluQuant(Matmul_g)
```

其中：

- `X_g` 来自单 tensor `x` 中按 `groupList` 切分出的第 `g` 段；
- `XScale_g` 来自单 tensor `xScale` 中与 `X_g` 对应的连续区域；
- 单多单场景下，`Weight_g` 来自 `weight[g]`；
- 单多单场景下，`WeightScale_g` 来自 `weightScale[g]`；
- `Y_g` 写入单 tensor `y` 中按 M 方向连续排列的位置；
- `YScale_g` 写入单 tensor `yScale` 中按 M 方向连续排列的位置。

SwiGLU 语义沿用既有实现：matmul 输出 N 方向包含激活值与 gate 值，最终 `y` 的 N 维为输入 `weight` N 维的一半，即 `[M, N / 2]`。

---

## 5. 输入输出规格

### 5.A 张量接口型

#### 5.A.1 输入 — 张量参数

| 参数名 | 必选/可选 | dtype | format | shape / 存储语义 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `x` | 必选 | FLOAT8_E4M3FN | ND | 单 tensor，shape `[M, K]` | M 方向按 `groupList` 切分 |
| `xScale` | 必选 | FLOAT8_E8M0 | ND / NCL | 单 tensor，shape `[M, ceil(K / 64), 2]` | 与 `x` 对应，仍按单 tensor offset 推进 |
| `groupList` | 必选 | INT64 | ND | `[groupNum]` | `groupListType` 为 0 时表示累积结束位置，为 1 时表示每组 M 大小 |
| `weight` | 必选 | FLOAT4_E2M1 | FRACTAL_NZ / FRACTAL_NZ_C0_32 | 单多单：tensor list，`weight[g]` view shape `[N, K]`，storage shape `[ceil(K / 32), ceil(N / 16), 16, 32]` | 单多单场景 tensor list 长度为 `groupNum` |
| `weightScale` | 必选 | FLOAT8_E8M0 | ND / NCHW | 单多单：tensor list，`weightScale[g]` shape `[N, ceil(K / 64), 2]` | tensor list 长度应与 `weight` 一致 |
| `weightAssistMatrix` | 可选 | 不涉及 | 不涉及 | 空指针或空 list | MxA8W4 weight quant 本路径不支持 |
| `bias` | 可选 | 不涉及 | 不涉及 | 空指针 | MxA8W4 weight quant 本路径不支持 |
| `smoothScale` | 可选 | 不涉及 | 不涉及 | 空指针 | MxA8W4 weight quant 本路径不支持 |

非单多单原有单 tensor 语义为：

```text
weight      : view [E, N, K]，storage [E, ceil(K / 32), ceil(N / 16), 16, 32]
weightScale : [E, N, ceil(K / 64), 2]
```

#### 5.A.2 输入 — 标量 / 属性

| 参数名 | 类型 | 取值 / 约束 | 说明 |
| --- | --- | --- | --- |
| `dequantMode` | int64 | 2 | MX quant |
| `dequantDtype` | int64 | FLOAT / DT_FLOAT | 反量化输出内部 dtype |
| `quantMode` | int64 | 2 | MX quant |
| `quantDtype` | int64 | FLOAT8_E4M3FN | 输出量化 dtype |
| `transposeWeight` | bool | true | MxA8W4 weight quant 当前仅支持 weight 转置语义 |
| `groupListType` | int64 | 0 / 1 | 保持既有语义 |
| `tuningConfig` | 可选 | 保持既有约束 | 本需求不新增语义 |

#### 5.A.3 输出

| 输出名 | dtype | format | shape | 说明 |
| --- | --- | --- | --- | --- |
| `y` / `output` | FLOAT8_E4M3FN | ND | `[M, N / 2]` | 单 tensor，按 group 输出连续写回 |
| `yScale` / `outputScale` | FLOAT8_E8M0 | ND | `[M, ceil(N / 2 / 64), 2]` | 单 tensor，按 group 输出 scale 连续写回 |

### 5.C dtype × format × 场景矩阵

| dtype | src_format | dst_format | 场景/备注 |
| --- | --- | --- | --- |
| x FLOAT8_E4M3FN / weight FLOAT4_E2M1 / xScale FLOAT8_E8M0 / weightScale FLOAT8_E8M0 | x ND；weight FRACTAL_NZ / FRACTAL_NZ_C0_32；scale ND / NCL / NCHW | y ND；yScale ND | GMMSQ V2 MxA8W4 weight quant；单多单时 weight / weightScale 为多 tensor |

---

## 6. 约束条件

| 约束项 | 条件 | 说明 / 校验失败表现 |
| --- | --- | --- |
| 场景约束 | 仅 MxA8W4 weight quant 路径 | `x=FLOAT8_E4M3FN`、`weight=FLOAT4_E2M1`、scale 为 FLOAT8_E8M0 |
| tensor 组合 | `x` 单 tensor、`weight` 多 tensor、`y` 单 tensor | 单多单由 `weight` tensor list 长度大于 1 识别 |
| `weightScale` 长度 | `weightScale` tensor list 长度与 `weight` 对齐 | host 校验 `weight` / `weightScale` tensor list 长度完全一致 |
| `groupNum` | `[1, 1024]` | 单多单下 `groupNum = weight tensor list length` |
| `M` | `M >= 1` | 来自 `x.shape[0]` |
| `K` | `K >= 32` 且 `K % 32 == 0` | MxA8W4 NZ / MX scale 约束 |
| `N` | `N >= 128` 且 `N % 128 == 0` | SwiGLU 输出为 `N / 2`，N 需满足分块约束 |
| `weight` shape | 单多单 `weight[g]` view `[N, K]`；storage `[ceil(K / 32), ceil(N / 16), 16, 32]` | 不满足时报 shape 错误 |
| `weightScale` shape | 单多单 `weightScale[g]` 为 `[N, ceil(K / 64), 2]` | 不满足时报 shape 错误 |
| `xScale` shape | `[M, ceil(K / 64), 2]` | 不满足时报 shape 错误 |
| `groupList` shape | `[groupNum]` | 单多单下与 weight list 长度一致 |
| 可选输入 | `weightAssistMatrix` 为空指针或空 list；`bias` / `smoothScale` 为空指针 | 当前 MxA8W4 weight quant 路径不支持这些输入 |
| TilingKey | `IS_SINGLE_MULTI_SINGLE` 与 kernel 模板参数顺序一致 | 参数顺序不一致会导致错误 kernel 实例 |

---

## 7. 交付标准

| 交付范围 | 当前迭代 |
| --- | --- |
| 交付通路 | aclnn / arch35 / MxA8W4 weight quant resplit controller 路径 |
| 动静要求 | 动态 groupNum；M / K / N 约束沿用当前 host tiling 检查 |
| 精度标准 | 与既有 GMMSQ V2 MxA8W4 单 tensor weight 场景对齐，按既有 golden / 容差策略验收 |
| 性能标准 | 非单多单路径不劣化；单多单路径使用编译期 `if constexpr` 避免运行时分支开销 |
| 客户场景索引 | 通用 GMMSQ V2 MxA8W4 单多单场景 |
| 相关设计文档索引 | `gmm/grouped_matmul_swiglu_quant_v2/docs/GMMSQ_V2_MxA8W4_SMS_SRS.md` |
| 维测相关索引 | 待补充 |

---

## 8. 具体方案描述

### 8.1 通路 / 调用链

```text
aclnn grouped_matmul_swiglu_quant_v2
  → op_host/op_tiling/arch35/grouped_matmul_swiglu_quant_v2_weight_quant_tiling.cpp
  → 识别 isSingleMultiSingle
  → 生成 TilingKey(QUANT_B_TRANS, QUANT_A_TRANS, IS_SINGLE_MULTI_SINGLE)
  → op_kernel/grouped_matmul_swiglu_quant_v2_apt.cpp
  → GMMSQWeightQuantResplitController<..., isSingleMultiSingle>
  → Process group loop
  → UpdateGmAddr 更新下一 group 地址
```

### 8.2 Tiling

- **输入约束**：MxA8W4 weight quant 下 `x` 单 tensor、`weight` 可为单 tensor 或多 tensor；当 `weight[1]` 存在时进入单多单分支。
- **切分策略**：沿用当前 resplit controller，N 方向按 SwiGLU 的前后半区处理；M 方向由 `groupList` 切分。
- **命中条件 / tiling key**：

  ```cpp
  return GET_TPL_TILING_KEY(static_cast<int64_t>(inputParams_.wTrans), 0L,
                            static_cast<uint64_t>(inputParams_.isSingleMultiSingle));
  ```

- **workspace / 资源**：workspace 沿用当前 `SYS_WORKSPACE_SIZE`，本需求不新增 workspace。

### 8.3 Kernel

- **执行流程**：
  1. `Init()` 保存 `weight`、`weightScale` tensor list 原始地址，并初始化第 0 个 tensor 地址；
  2. `Process()` 遍历 group；
  3. 当前 group 使用当前 `weightGm_ / weightScaleGm_` 计算；
  4. 当前 group 完成后调用 `UpdateGmAddr(groupIdx, ...)`；
  5. 单多单场景下切换到 `groupIdx + 1` 对应的 `weight` / `weightScale` tensor 地址。
- **关键逻辑**：
  - 单多单：`weightGm_ = GetTensorAddr<wType>(nextGroupIdx, weightTensorPtr_)`，`weightScaleGm_ = GetTensorAddr<weightScaleType>(nextGroupIdx, weightScaleTensorPtr_)`；
  - 非单多单：保持原连续 offset 推进。
- **Buffer 与同步**：沿用当前 `GMMSQWeightQuantVcvBasicBlock` 和 resplit 双 buffer / 同步策略，不新增同步。
- **异常与 early-exit**：仅当 `nextGroupIdx < tiling_->groupNum` 时更新下一 group 地址，避免最后一个 group 后越界访问。

### 8.4 代码片段

#### 8.4.1 TilingKey 新增 `IS_SINGLE_MULTI_SINGLE`

文件：`gmm/grouped_matmul_swiglu_quant_v2/op_kernel/arch35/weight_quant_basic_block/grouped_matmul_swiglu_quant_v2_weight_quant_tiling_key.h`

```cpp
#define GMM_SWIGLU_QUANT_NO_TRANS 0
#define GMM_SWIGLU_QUANT_TRANS 1
// 单多单场景标识：0表示weight/weightScale按group连续排布在单Tensor中，
// 1表示每个group使用独立Tensor。
#define GMM_SWIGLU_QUANT_NOT_SINGLE_MULTI_SINGLE 0
#define GMM_SWIGLU_QUANT_SINGLE_MULTI_SINGLE 1

// TilingKey模板参数新增IS_SINGLE_MULTI_SINGLE，供kernel编译期选择地址更新路径。
ASCENDC_TPL_ARGS_DECL(GroupedMatmulSwigluQuantV2, // 算子OpType
                      ASCENDC_TPL_UINT_DECL(QUANT_B_TRANS, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST,
                                            GMM_SWIGLU_QUANT_NO_TRANS, GMM_SWIGLU_QUANT_TRANS),
                      ASCENDC_TPL_UINT_DECL(QUANT_A_TRANS, ASCENDC_TPL_2_BW, ASCENDC_TPL_UI_LIST,
                                            GMM_SWIGLU_QUANT_NO_TRANS, GMM_SWIGLU_QUANT_TRANS),
                      ASCENDC_TPL_BOOL_DECL(IS_SINGLE_MULTI_SINGLE, GMM_SWIGLU_QUANT_SINGLE_MULTI_SINGLE,
                                            GMM_SWIGLU_QUANT_NOT_SINGLE_MULTI_SINGLE));

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_MIX_AIC_1_2),
                         ASCENDC_TPL_UINT_SEL(QUANT_B_TRANS, ASCENDC_TPL_UI_LIST, GMM_SWIGLU_QUANT_TRANS),
                         ASCENDC_TPL_UINT_SEL(QUANT_A_TRANS, ASCENDC_TPL_UI_LIST, GMM_SWIGLU_QUANT_NO_TRANS),
                         ASCENDC_TPL_BOOL_SEL(IS_SINGLE_MULTI_SINGLE,
                                              GMM_SWIGLU_QUANT_NOT_SINGLE_MULTI_SINGLE,
                                              GMM_SWIGLU_QUANT_SINGLE_MULTI_SINGLE)),
                         ASCENDC_TPL_TILING_STRUCT_SEL(GMMSQArch35Tiling::GMMSQWeightQuantTilingData),);
```

#### 8.4.2 Host 识别单多单并校验 shape

文件：`gmm/grouped_matmul_swiglu_quant_v2/op_host/op_tiling/arch35/grouped_matmul_swiglu_quant_v2_weight_quant_tiling.cpp`

```cpp
uint32_t GetDynamicInputTensorListSize(const gert::TilingContext& context, uint32_t inputIndex)
{
    uint32_t count = 0;
    for (; count <= MAX_GROUP_NUM; ++count) {
        if (context.GetDynamicInputShape(inputIndex, count) == nullptr) {
            break;
        }
    }
    return count;
}

bool CheckWeightAndScaleListSize(const gert::TilingContext& context, GMMSQWeightQuantInputParams& params)
{
    uint32_t weightListSize = GetDynamicInputTensorListSize(context, WEIGHT_INDEX);
    uint32_t weightScaleListSize = GetDynamicInputTensorListSize(context, WEIGHT_SCALE_INDEX);
    // 以weight list长度识别单多单；只要任一侧为多Tensor，就要求weight/weightScale长度完全一致。
    params.isSingleMultiSingle = weightListSize > 1;
    OP_CHECK_IF((weightListSize > 1 || weightScaleListSize > 1) && weightListSize != weightScaleListSize,
        OP_LOGE(&context, "weight and weightScale tensor list size should be equal, got %u and %u.",
            weightListSize, weightScaleListSize),
        return false);
    params.groupNum = params.isSingleMultiSingle ? static_cast<int64_t>(weightListSize) : params.groupNum;
    return true;
}
```

```cpp
OP_CHECK_IF(!CheckWeightAndScaleListSize(context, params),
    OP_LOGE(&context, "Check weight and weightScale tensor list size failed."), return false);
// 单多单下首个weight的origin shape为[N, K]，非单多单下为[groupNum, N, K]。
uint64_t expectedWeightDim = params.isSingleMultiSingle ? DIM_NUM_WEIGHT_ND - 1 : DIM_NUM_WEIGHT_ND;
OP_CHECK_IF(wShapeLen != expectedWeightDim,
    OP_LOGE(&context, "input dimension of weight should be %lu, but got %zu", expectedWeightDim, wShapeLen),
    return false);

// 单多单weight不携带group维度，N取首维；groupNum使用weight TensorList长度。
params.nSize = params.isSingleMultiSingle ? wShape.GetDim(DIM_0) : wShape.GetDim(DIM_1);
params.groupNum = params.isSingleMultiSingle ? params.groupNum : wShape.GetDim(DIM_0);
```

```cpp
if (params.isSingleMultiSingle) {
    // 单多单每个group对应一组独立weight/weightScale，逐个校验TensorList成员。
    const std::vector<int64_t> expectedWeightShape = {
        GroupedMatmul::CeilDiv<int64_t>(params.kSize, C0_SIZE),
        GroupedMatmul::CeilDiv<int64_t>(params.nSize, CUBE_BLOCK),
        CUBE_BLOCK,
        C0_SIZE
    };
    const std::vector<int64_t> expectedWeightScaleShape = {
        params.nSize,
        scaleKSize,
        MX_INNER_DIM
    };
    for (int64_t i = 0; i < params.groupNum; ++i) {
        uint32_t tensorIndex = static_cast<uint32_t>(i);
        auto* curWeightShapePtr = context.GetDynamicInputShape(WEIGHT_INDEX, tensorIndex);
        auto* curWeightScaleShapePtr = context.GetDynamicInputShape(WEIGHT_SCALE_INDEX, tensorIndex);
        OP_CHECK_IF(curWeightShapePtr == nullptr || curWeightScaleShapePtr == nullptr,
            OP_LOGE(params.opName, "weight or weightScale shape[%ld] should not be null", i), return false);
        OP_CHECK_IF(!CheckDynamicInputDesc(context, WEIGHT_INDEX, tensorIndex, "weight", ge::DT_FLOAT4_E2M1),
            OP_LOGE(params.opName, "Check dynamic input desc of weight[%ld] failed.", i), return false);
        OP_CHECK_IF(!CheckDynamicInputDesc(context, WEIGHT_SCALE_INDEX, tensorIndex, "weightScale",
                                          ge::DT_FLOAT8_E8M0),
            OP_LOGE(params.opName, "Check dynamic input desc of weightScale[%ld] failed.", i), return false);
        OP_CHECK_IF(!CheckWeightFormat(context, tensorIndex),
            OP_LOGE(params.opName, "Check weight[%ld] format failed.", i), return false);
        OP_CHECK_IF(!CheckWeightScaleFormat(context, tensorIndex, params.opName),
            OP_LOGE(params.opName, "Check weightScale[%ld] format failed.", i), return false);
        if (!CheckInputShape(&context, "weight", curWeightShapePtr->GetStorageShape(), expectedWeightShape)) {
            return false;
        }
        if (!CheckInputShape(&context, "weightScale", curWeightScaleShapePtr->GetOriginShape(),
                             expectedWeightScaleShape)) {
            return false;
        }
    }
    return true;
}
```

#### 8.4.3 Host 生成 TilingKey

文件：`gmm/grouped_matmul_swiglu_quant_v2/op_host/op_tiling/arch35/grouped_matmul_swiglu_quant_v2_weight_quant_tiling.cpp`

```cpp
uint64_t GroupedMatmulSwigluQuantV2WeightQuantTiling::GetTilingKey() const
{
    // 第3个参数对应IS_SINGLE_MULTI_SINGLE，需与kernel模板参数顺序保持一致。
    return GET_TPL_TILING_KEY(static_cast<int64_t>(inputParams_.wTrans), 0L,
                              static_cast<uint64_t>(inputParams_.isSingleMultiSingle));
}
```

#### 8.4.4 Kernel 模板参数透传

文件：`gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_apt.cpp`

```cpp
#if defined(GMM_SWIGLU_QUANT_WEIGHT_QUANT)
// MxA8W4单多单标识通过模板参数固化，kernel侧使用if constexpr消除无关取址分支。
template <int8_t QUANT_B_TRANS, int8_t QUANT_A_TRANS, bool IS_SINGLE_MULTI_SINGLE>
#else
template <int8_t QUANT_B_TRANS, int8_t QUANT_A_TRANS>
#endif
__global__ __aicore__ void grouped_matmul_swiglu_quant_v2(...)
```

```cpp
// IS_SINGLE_MULTI_SINGLE透传给controller，用于区分TensorList取址和连续offset推进。
GMMSQWeightQuant::GMMSQWeightQuantResplitController<
    DTYPE_X, DTYPE_WEIGHT, DTYPE_WEIGHT_SCALE, DTYPE_X_SCALE, DTYPE_Y, DTYPE_Y_SCALE,
    GMMSQWeightQuant::MXA8W4_NZNK, GMMSQWeightQuant::VEC_ANTIQUANT_CONFIG_DYNAMIC, IS_SINGLE_MULTI_SINGLE>
    controller;
```

#### 8.4.5 Controller 地址更新

文件：`gmm/grouped_matmul_swiglu_quant_v2/op_kernel/arch35/weight_quant_basic_block/gmmsq_weight_quant_resplit_controller.h`

```cpp
#define GMMSQ_WQ_RESPLIT_CONTROLLER_TEMPLATE_PARAM                                                                     \
    template <typename xType, typename wType, typename weightScaleType, typename xScaleType, typename yType,           \
              typename yScaleType, const WqmmConfig &wqmmConfig, const VecAntiQuantConfig &vecConfig,                  \
              bool isSingleMultiSingle>
```

```cpp
xGm_ = reinterpret_cast<__gm__ xType *>(x);
weightTensorPtr_ = weight;
weightScaleTensorPtr_ = weightScale;
weightGm_ = GetTensorAddr<wType>(0, weightTensorPtr_);
weightScaleGm_ = GetTensorAddr<weightScaleType>(0, weightScaleTensorPtr_);
xScaleGm_ = reinterpret_cast<__gm__ xScaleType *>(xScale);
yGm_ = reinterpret_cast<__gm__ yType *>(y);
yScaleGm_ = reinterpret_cast<__gm__ yScaleType *>(yScale);
```

```cpp
__aicore__ inline void GMMSQ_WQ_RESPLIT_CONTROLLER_CLASS::UpdateGmAddr(uint64_t groupIdx, uint64_t mSize,
                                                                       uint64_t kSize, uint64_t nSize)
{
    xGm_ += mSize * kSize;
    uint64_t kAlignSize = CeilAlign(kSize, K_ALIGNMENT64);
    if constexpr (isSingleMultiSingle) {
        // 单多单中每个group的weight/weightScale是独立Tensor，切到下个group时重新从TensorList取址。
        uint64_t nextGroupIdx = groupIdx + 1;
        if (nextGroupIdx < tiling_->groupNum) {
            weightGm_ = GetTensorAddr<wType>(nextGroupIdx, weightTensorPtr_);
            weightScaleGm_ = GetTensorAddr<weightScaleType>(nextGroupIdx, weightScaleTensorPtr_);
        }
    } else {
        weightGm_ += (nSize * kSize) >> 1;
        weightScaleGm_ += nSize * CeilDivide(kAlignSize, static_cast<uint64_t>(tiling_->groupSize));
    }
    xScaleGm_ += mSize * CeilDivide(kAlignSize, static_cast<uint64_t>(tiling_->groupSize));
    yGm_ += mSize * (nSize / 2);
    yScaleGm_ +=
        mSize * CeilDivide(CeilAlign(nSize / 2, static_cast<uint64_t>(64)), static_cast<uint64_t>(tiling_->groupSize));
}
```

---

## 9. 软件设计与接口总览

### 9.1 模块职责

| 模块 | 说明 |
| --- | --- |
| Host | 校验 MxA8W4 weight quant 输入，识别单多单，校验多 tensor `weight` / `weightScale` shape，生成 TilingKey |
| Tiling | 通过 `GMMSQWeightQuantTilingData` 下发 `groupNum / groupSize / kSize / nSize / groupListType` |
| Kernel | 根据 `IS_SINGLE_MULTI_SINGLE` 编译期参数选择 tensor list 取址或连续 offset 推进 |
| 测试 | 覆盖单多单、非单多单、shape 非法、TilingKey 白盒、地址切换正确性 |

### 9.2 接口列表（与实现对齐）

| 接口项 | 说明 |
| --- | --- |
| Host API | aclnn `grouped_matmul_swiglu_quant_v2` 两段式接口，API 名称沿用现有接口 |
| Tiling 数据结构 | `GMMSQWeightQuantInputParams::isSingleMultiSingle`；`GMMSQWeightQuantTilingData` 沿用现有字段 |
| Kernel entry / 模板参数 | `grouped_matmul_swiglu_quant_v2_apt.cpp` 中 weight quant 路径新增 `bool IS_SINGLE_MULTI_SINGLE` |
| Controller 模板参数 | `GMMSQWeightQuantResplitController<..., bool isSingleMultiSingle>` |
| 对外泄漏 API | 无新增对外 API；不修改 `op_api/grouped_matmul_swiglu_quant_v2_utils.h` |

---

## 10. 错误码与日志

| 场景 | 返回码 / 宏 | 日志关键字（英文） | 触发条件 |
| --- | --- | --- | --- |
| 参数非法 | `OP_CHECK_IF` / `GRAPH_FAILED` | `weight or weightScale shape[i] should not be null` | 单多单下指定 group 的 `weight` 或 `weightScale` 缺失 |
| shape 非法 | `OP_CHECK_IF` / `GRAPH_FAILED` | `Check input weight shape failed` | `weight[i]` storage shape 不等于期望 NZ shape |
| shape 非法 | `OP_CHECK_IF` / `GRAPH_FAILED` | `Check input weightScale shape failed` | `weightScale[i]` shape 不等于 `[N, ceil(K / 64), 2]` |
| TilingKey 非法 | `GET_TPL_TILING_KEY` 内部校验 | TilingKey 组合非法 | `IS_SINGLE_MULTI_SINGLE` 参数顺序或合法组合未注册 |

---

## 11. 外部依赖

| 依赖项 | 内容 | 影响级别 |
| --- | --- | --- |
| CANN / Ascend C 模板参数机制 | `ASCENDC_TPL_BOOL_DECL`、`ASCENDC_TPL_BOOL_SEL`、`GET_TPL_TILING_KEY` | 高 |
| 硬件型号 / SOC | arch35 MxA8W4 支持硬件 | 高 |
| grouped_matmul host util | `GroupedMatmul::CeilDiv` 等公共工具 | 中 |
| 测试框架与 golden | 现有 GMMSQ V2 MxA8W4 用例体系 | 中 |

---

## 12. 待确认事项

1. 是否需要在对外 API 文档中同步删除“weight / weightScale 目前仅支持 tensor list 长度为 1”的旧限制说明。
2. 是否需要为 `weightScale` list size 与 `weight` list size 不一致补充专门错误用例。
3. 是否需要为 `groupListType = 0 / 1` 分别补充单多单 STC。
4. 是否需要给 `IS_SINGLE_MULTI_SINGLE` 的 0 / 1 取语义化宏名，避免裸值含义不清。

---

## 13. 修订记录

| 版本 | 日期 | 作者 | 说明 |
| --- | --- | --- | --- |
| v0.1 | 2026-06-11 | Claude | 按脱敏母版重写 GMMSQ V2 MxA8W4 单多单 SRS |

---

## 附录 A：与 STC 的交叉检查清单

- [ ] MxA8W4 单多单 happy path：x 单 tensor、weight 多 tensor、weightScale 多 tensor、y 单 tensor。
- [ ] 非单多单 MxA8W4 回归：`IS_SINGLE_MULTI_SINGLE == 0`，weight / weightScale 连续 offset 推进行为不变。
- [ ] 单多单 TilingKey 白盒检查：`IS_SINGLE_MULTI_SINGLE == 1`。
- [ ] 非单多单 TilingKey 白盒检查：`IS_SINGLE_MULTI_SINGLE == 0`。
- [ ] `weightScale` 缺少某个 group tensor 时 host tiling 报错。
- [ ] `weight[i]` storage shape 非法时报错。
- [ ] `weightScale[i]` shape 非法时报错。
- [ ] `groupList` 长度与 `groupNum` 不一致时报错。
- [ ] `weightAssistMatrix` 非空时报错。
- [ ] `bias` 非空时报错。
- [ ] `smoothScale` 非空时报错。
