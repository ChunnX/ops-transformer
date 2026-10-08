# aclnnGroupedMatmulActivationWeightNz

`aclnnGroupedMatmulActivationWeightNz` fuses grouped MX dequantized matrix multiplication and SwiGLU. It is a
WeightNZ interface for Ascend 950PR and Ascend 950DT.

## Supported Configuration

- `x` and the single `weight` tensor: `FLOAT4_E2M1`.
- `xScale` and the single `weightScale` tensor: `FLOAT8_E8M0`.
- `weight`: `FRACTAL_NZ`; `x`, scales, `groupList`, and `y`: `ND`.
- `y`: `BFLOAT16`; no output scale is exposed by this interface.
- `dequantMode=2`, `dequantDtype=0`, and `actType=0`.
- `groupList`: `INT64`; `groupListType` is `0` (cumulative) or `1` (count).
- `bias` must be `nullptr`. `tuningConfig` must be `nullptr` or an empty array.
- The logical K and the concatenated N dimension must be divisible by 128.

`swigluMode=0` performs the traditional no-clamp SwiGLU:

```
y = silu(gemm[..., :N/2]) * gemm[..., N/2:]
```

`swigluMode=2` performs the existing front/back-half SwiGLU with clamp:

```
act = min(gemm[..., :N/2], clampLimit)
gate = clamp(gemm[..., N/2:], -clampLimit, clampLimit) + gluBias
y = (act * sigmoid(gluAlpha * act)) * gate
```

For mode 2, `clampLimit` must be finite and greater than zero, and `gluAlpha` and `gluBias` must be finite float32
values. Mode 0 does not use these three attributes.

## Function Prototypes

```cpp
aclnnStatus aclnnGroupedMatmulActivationWeightNzGetWorkspaceSize(
    const aclTensor* x, const aclTensor* xScale, const aclTensor* groupList, const aclTensorList* weight,
    const aclTensorList* weightScale, const aclTensor* bias, int64_t dequantMode, int64_t dequantDtype,
    int64_t actType, bool transposeWeight, int64_t groupListType, const aclIntArray* tuningConfig,
    int64_t swigluMode, double clampLimit, double gluAlpha, double gluBias, aclTensor* y,
    uint64_t* workspaceSize, aclOpExecutor** executor);

aclnnStatus aclnnGroupedMatmulActivationWeightNz(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
```

The implementation forwards to the existing A4W4 WeightNZ worker. Worker-only auxiliary tensors, rounding, scale
algorithm, and output scale are fixed internally; direct BF16 output does not read or write the internal output scale.
