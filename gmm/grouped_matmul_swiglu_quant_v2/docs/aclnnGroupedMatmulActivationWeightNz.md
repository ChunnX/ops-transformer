# aclnnGroupedMatmulActivationWeightNz

[查看源码](https://gitcode.com/cann/ops-transformer/tree/master/gmm/grouped_matmul_swiglu_quant_v2)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

`aclnnGroupedMatmulActivationWeightNz` 融合 GroupedMatmul、MX 反量化和 SwiGLU，面向
Ascend 950PR/Ascend 950DT 的 A4W4 WeightNZ 场景。输入为 `FLOAT4_E2M1` 的 `x`、`weight`，
以及 `FLOAT8_E8M0` 的 `xScale`、`weightScale`；输出为 BF16 的 `y`，不创建、不写回 `yScale`。

接口复用已有 WeightNZ TensorAPI worker，但对外只支持以下组合：

- `dequantMode=2`、`dequantDtype=0`、`actType=0`；
- `swigluMode=2`，沿 GMM 结果最后一维前后分半；
- `bias=nullptr`，`tuningConfig` 为 `nullptr` 或空数组；
- `weight` 和 `weightScale` 均为长度 1 的 TensorList；
- `weight` 为 `FRACTAL_NZ`，`y` 为 `BFLOAT16`/`ND`。

### 计算公式

对 `groupList` 指定的每个 expert，执行 MX A4W4 grouped matmul：

$$
C_i[m,n] = \sum_{k=0}^{K-1}
\left(X_i[m,k] \times S^X_i[m,\lfloor k/64\rfloor]\right)
\left(W_i[k,n] \times S^W_i[\lfloor k/64\rfloor,n]\right)
$$

其中 `S^X`、`S^W` 为 E8M0 scale 解码值，scale 必须在 K 方向求和前参与计算。GMM 结果
`C` 的逻辑 shape 为 `(M,N)`，沿最后一维前后分半得到 `C_glu` 和 `C_gate`：

$$
u = \min(C_{glu}, clampLimit)
$$

$$
v = \operatorname{clip}(C_{gate},-clampLimit,clampLimit)+gluBias
$$

$$
y = \frac{u}{1+\exp(-gluAlpha\times u)}\times v
$$

中间 GMM 和 SwiGLU 使用 FP32 计算，结果在写回 `y` 时转换为 BF16。`clampLimit` 必须为有限
正数，`gluAlpha` 和 `gluBias` 必须为有限且可由 float32 表示的值。

## 函数原型

该算子采用两段式接口。第一段完成参数校验、shape/format 校验、tiling 和 executor 构造，
第二段提交 executor 到指定 stream 执行。

```cpp
aclnnStatus aclnnGroupedMatmulActivationWeightNzGetWorkspaceSize(
    const aclTensor* x, const aclTensor* xScale, const aclTensor* groupList,
    const aclTensorList* weight, const aclTensorList* weightScale, const aclTensor* bias,
    int64_t dequantMode, int64_t dequantDtype, int64_t actType, bool transposeWeight,
    int64_t groupListType, const aclIntArray* tuningConfig, int64_t swigluMode,
    double clampLimit, double gluAlpha, double gluBias, aclTensor* y,
    uint64_t* workspaceSize, aclOpExecutor** executor);

aclnnStatus aclnnGroupedMatmulActivationWeightNz(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
```

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 格式 | shape/约束 |
| --- | --- | --- | --- | --- | --- |
| `x` | 输入 | 激活矩阵。 | `FLOAT4_E2M1` | `ND` | `(M,K)` |
| `xScale` | 输入 | 激活 MX scale，K 方向每 64 个元素对应 2 个 scale。 | `FLOAT8_E8M0` | `ND` | `(M,ceil(K/64),2)` |
| `groupList` | 输入 | expert 分组信息。 | `INT64` | `ND` | `(E)` |
| `weight` | 输入 | WeightNZ 权重 TensorList。 | `FLOAT4_E2M1` | `FRACTAL_NZ` | 长度必须为 1 |
| `weightScale` | 输入 | 权重 MX scale TensorList。 | `FLOAT8_E8M0` | `ND` | 长度必须为 1，逻辑 shape `(E,ceil(K/64),N,2)` |
| `bias` | 可选输入 | 矩阵乘偏置。 | - | - | 必须为 `nullptr` |
| `dequantMode` | 属性 | 反量化模式。 | `INT64` | - | 仅支持 `2` |
| `dequantDtype` | 属性 | GMM 中间累加类型。 | `INT64` | - | 仅支持 `0`，表示 FLOAT32 |
| `actType` | 属性 | 激活类型。 | `INT64` | - | 仅支持 `0`，表示 SwiGLU |
| `transposeWeight` | 属性 | WeightNZ 转置布局标识。 | `BOOL` | - | 必须和 weight/scale 的 view、stride、storage 一致 |
| `groupListType` | 属性 | 分组列表解释方式。 | `INT64` | - | `0` 为 cumulative，`1` 为 count |
| `tuningConfig` | 可选输入 | tiling 调优配置。 | `aclIntArray` | - | 必须为 `nullptr` 或空数组 |
| `swigluMode` | 属性 | SwiGLU 计算模式。 | `INT64` | - | **仅支持 `2`** |
| `clampLimit` | 属性 | 激活和线性分支的裁剪上限。 | `DOUBLE` | - | 有限正数 |
| `gluAlpha` | 属性 | sigmoid 输入缩放系数。 | `DOUBLE` | - | 必须可由 FLOAT32 表示 |
| `gluBias` | 属性 | 线性分支偏置。 | `DOUBLE` | - | 必须可由 FLOAT32 表示 |
| `y` | 输出 | SwiGLU 结果。 | `BFLOAT16` | `ND` | `(M,N/2)` |
| `workspaceSize` | 输出 | Device workspace 大小。 | `uint64_t*` | - | 第一段接口返回 |
| `executor` | 输出 | 算子执行器。 | `aclOpExecutor**` | - | 传给第二段接口 |

## 约束说明

- 仅支持 Ascend 950PR/Ascend 950DT；`x` 和 `weight` 必须为 `FLOAT4_E2M1`，scale 必须为
  `FLOAT8_E8M0`，`groupList` 必须为 INT64，`y` 必须为 BF16。
- `weight` 和 `weightScale` 的 TensorList 长度必须均为 1，不能把各 expert 拆成多个 Tensor。
- `weight` 必须为 `FRACTAL_NZ`。非转置场景的逻辑权重为 `(E,K,N)`，转置场景需要按照 ZN
  storage 构造，并同步调整 `weightScale` 的 view/stride；不能只转置 weight。
- `x` 为二维 ND，K 必须与 weight 的逻辑 K 一致；逻辑 N 必须满足当前 A4W4 TensorAPI 的
  tile 对齐约束，通常要求 N 为 128 的整数倍，输出宽度为 `N/2`。
- `groupListType=0` 时，`groupList` 表示各 expert 的累计结束位置，必须非负、单调不减且不大于 M；
  `groupListType=1` 时，表示各 expert 的 token 数量，总和不能超过 M。
- `dequantMode/dequantDtype/actType` 必须为 `2/0/0`；`swigluMode` 必须为 `2`。
- `clampLimit`、`gluAlpha` 和 `gluBias` 必须为有限值，其中 `clampLimit` 必须大于 0。
- 不支持 bias、非空 tuningConfig、MXFP8 输出、FP8 输入、FP4 E1M2 或 WeightND。

## 返回值和错误码

本接口仅返回输出 Tensor `y`，不返回 `yScale` 或其他辅助输出。两段接口均返回 `aclnnStatus`。

| 返回码 | 错误码 | 含义 |
| --- | --- | --- |
| `ACLNN_ERR_PARAM_NULLPTR` | `161001` | 必选指针为空，或 TensorList 中包含空 Tensor |
| `ACLNN_ERR_PARAM_INVALID` | `161002` | dtype、format、shape、属性或布局不满足约束 |

## 调用示例

```cpp
// x: FLOAT4_E2M1 ND, (M, K)
// xScale: FLOAT8_E8M0 ND, (M, ceil(K / 64), 2)
// weight: 一个 FLOAT4_E2M1 FRACTAL_NZ Tensor，包含 E 个 expert
// weightScale: 一个 FLOAT8_E8M0 ND Tensor，逻辑 shape (E, ceil(K / 64), N, 2)
// y: BFLOAT16 ND, (M, N / 2)
constexpr int64_t dequantMode = 2;
constexpr int64_t dequantDtype = 0;
constexpr int64_t actType = 0;
constexpr int64_t groupListType = 0;
constexpr int64_t swigluMode = 2;
constexpr double clampLimit = 7.0;
constexpr double gluAlpha = 1.702;
constexpr double gluBias = 1.0;

uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
aclnnStatus ret = aclnnGroupedMatmulActivationWeightNzGetWorkspaceSize(
    x, xScale, groupList, weight, weightScale, nullptr, dequantMode, dequantDtype, actType,
    false, groupListType, nullptr, swigluMode, clampLimit, gluAlpha, gluBias,
    y, &workspaceSize, &executor);
// 分配 workspace 后执行：
ret = aclnnGroupedMatmulActivationWeightNz(workspace, workspaceSize, executor, stream);
```

## 确定性计算

默认支持确定性计算。中间 GMM 和 SwiGLU 保持 FP32 计算，最终仅在写回 `y` 时转换为 BF16。
