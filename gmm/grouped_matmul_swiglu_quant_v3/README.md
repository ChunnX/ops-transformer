# GroupedMatmulSwigluQuantWeightNzV3

## 功能说明

本目录实现 `aclnnGroupedMatmulSwigluQuantWeightNzV3`。算子融合 grouped matmul、反量化、SwiGLU 和 MXFP8 量化，当前迭代只支持：

- 单个 weight Tensor；
- weight 为 `FRACTAL_NZ` 格式；
- MXFP8 输入和权重场景；
- 通过 aclnn 直调或上层 Torch 接口调用；
- SwiGLU 参数 `activate_dim`、`activate_left`、`swiglu_mode`、`clamp_limit`、`glu_alpha`、`glu_bias`。

`axis`、`round_mode`、`scale_alg` 和 `dst_type_max` 目前只做默认值校验，暂不改变 MX 量化逻辑。多 weight Tensor、非 MXFP8 和非 NZ weight 暂不支持。

## 接口资料

- [aclnnGroupedMatmulSwigluQuantWeightNzV3](docs/aclnnGroupedMatmulSwigluQuantWeightNzV3.md)

## 目录说明

- `op_api`：aclnn 两段式接口及 L0 调用封装；
- `op_host`：算子定义、shape inference 和 tiling；
- `op_kernel`：Device kernel；
- `examples`：aclnn 直调示例；
- `tests`：host、tiling 和 op_api 单元测试；
- `op_graph`：仅保留 proto 头文件，当前不提供图模式 fallback。
