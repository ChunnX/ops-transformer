# GroupedMatmulSwigluQuantWeightNzV3

## 功能说明

本目录预留 `aclnnGroupedMatmulSwigluQuantWeightNzV3`。本次迭代只面向 Ascend 950（arch35）的 MXFP8 weightNZ 场景，当前只支持：

- 单个 weight Tensor；
- weight 为 `FRACTAL_NZ` 格式；
- MXFP8 输入和权重场景；
- 通过 aclnn 直调或上层 Torch 接口调用；
- SwiGLU 参数 `activate_dim`、`activate_left`、`swiglu_mode`、`clamp_limit`、`glu_alpha`、`glu_bias`。

`axis` 和 `round_mode` 目前只做默认值校验（分别为 `-1` 和 `"rint"`）。MXFP8
支持 `scale_alg=0`（OCP）和 `scale_alg=1`（cuBLAS）；`scale_alg=2` 仅适用于
FLOAT4 输出，因此在本接口的 FP8 输出场景会报错。`dst_type_max` 当前只支持 `0.0`。
多 weight Tensor、非 MXFP8 和非 NZ weight 暂不支持。

## 接口资料

- [aclnnGroupedMatmulSwigluQuantWeightNzV3](docs/aclnnGroupedMatmulSwigluQuantWeightNzV3.md)

tiling 和 kernel 目前各保留一个 arch35 入口，具体实现待 950 环境具备后补充。

## 目录说明

- `op_api`：aclnn 两段式接口及 L0 调用封装；
- `op_host`：950 专用算子定义、shape inference 和 tiling 入口；
- `op_kernel`：950 专用 kernel 入口；
- `examples`：aclnn 直调示例；
- `tests`：op_api 单元测试；
- `op_graph`：仅保留 proto 头文件，当前不提供图模式 fallback。
