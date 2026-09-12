# aclnnGroupedMatmulSwigluQuantWeightNzV3

## 功能说明

该接口融合 grouped matmul、反量化、SwiGLU 和 MXFP8 量化。当前只支持单个 `weight` Tensor，且 `weight` 强制按 `FRACTAL_NZ` 格式处理；多 Tensor、非 NZ weight 和非 MXFP8 场景暂不支持。

SwiGLU 参数只影响 SwiGLU 计算阶段：

- `activateDim`：激活拆分维度，当前只支持 `-1`；
- `activateLeft`：是否激活拆分后的左半部分；
- `swigluMode`：SwiGLU 模式，支持 `0`～`3`；
- `clampLimit`：SwiGLU clamp 的正负对称边界，必须大于 `0`；
- `gluAlpha`、`gluBias`：带参数 SwiGLU 模式使用的 alpha 和 bias。

`axis`、`roundMode`、`scaleAlg`、`dstTypeMax` 是预留参数，当前只校验默认值（分别为 `-1`、`"rint"`、`0`、`0.0`），不改变 MX 量化逻辑。

## 函数原型

```cpp
aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzV3GetWorkspaceSize(
    const aclTensor *x,
    const aclTensorList *weight,
    const aclTensorList *weightScale,
    const aclTensorList *weightAssistMatrix,
    const aclTensor *bias,
    const aclTensor *xScale,
    const aclTensor *smoothScale,
    const aclTensor *groupList,
    int64_t dequantMode,
    int64_t dequantDtype,
    int64_t quantMode,
    int64_t groupListType,
    const aclIntArray *tuningConfigOptional,
    int64_t activateDim,
    bool activateLeft,
    int64_t swigluMode,
    double clampLimit,
    double gluAlpha,
    double gluBias,
    int64_t axis,
    const char *roundMode,
    int64_t scaleAlg,
    double dstTypeMax,
    aclTensor *output,
    aclTensor *outputScale,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzV3(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## 数据流

1. 按 `groupList` 将 `x` 分配到对应的 weight 分组；
2. 执行 grouped matmul，并结合 `xScale`、`weightScale` 完成反量化；
3. 按 `activateDim` 和 `activateLeft` 拆分 matmul 结果，执行 `swigluMode` 对应的 SwiGLU 计算，并在 SwiGLU 阶段应用 `clampLimit`、`gluAlpha`、`gluBias`；
4. 对 SwiGLU 结果执行 MXFP8 量化，输出量化结果和输出 scale。

## 调用顺序

该接口是两段式接口，必须先调用 `GetWorkspaceSize` 获取 workspace 和 executor，再调用 `aclnnGroupedMatmulSwigluQuantWeightNzV3` 执行计算。
