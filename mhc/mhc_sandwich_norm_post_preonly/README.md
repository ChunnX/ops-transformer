# MhcSandwichNormPostPreonly

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | × |
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品 | × |
<!-- npu="950" id1 -->
| Atlas A5 训练系列产品/Atlas 950 SuperPoD 推理系统 | √ |
<!-- npu="950" end -->

## 功能说明

融合 MHC Sandwich 结构中的 RMSNorm_0、MHC_Post、可选 RMSNorm_mid、MHC_Pre（PreOnly）和 RMSNorm_1，减少中间 Tensor 和 Host-Device 交互。

## 参数说明

输入包括 `h_out`、`residual`、`h_post`、`h_res`、`phi`、`alpha`、`bias`、`gamma_0`、`gamma_1` 和可选 `gamma_2`；输出为 `h_in_prime`、`x_2_out`，并可通过 `return_h_in_f32` 额外返回 `h_in_f32`。详细 shape、数据类型和属性说明参见接口文档。

## 约束说明

- 仅支持 Atlas A5 训练系列产品/Atlas 950 SuperPoD 推理系统。
- 支持 `[T,...]` 和 `[B,S,...]` 两种配套布局，当前实现要求 `N=4`、`D=2560/5120`。
- `h_out` 和 `residual` 支持 FLOAT16、BFLOAT16 且类型须一致，其余 Tensor 为 FLOAT。
- 当前仅支持 ND 格式。

## 调用说明

| 调用方式 | 说明 |
| --- | --- |
| ACLNN 接口 | [aclnnMhcSandwichNormPostPreonly](docs/aclnnMhcSandwichNormPostPreonly.md) |
| PyTorch 接口 | [mhc_sandwich_norm_post_preonly](docs/torchapi_mhc_sandwich_norm_post_preonly.md) |
