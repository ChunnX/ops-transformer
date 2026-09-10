# ClaGateQuant

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |
| <term>Kirin X90 处理器系列产品</term> | × |
| <term>Kirin 9030 处理器系列产品</term> | × |

## 功能说明

- 算子功能：融合算子，实现 CLA（Cross-Layer Attention）两路 head-wise gate 加权合并与双轴动态块量化。先对 Global/CLA 分支与 Local/SWA 分支的 Attention 输出分别施加 Sigmoid 门控并加权合并，再将合并结果 reshape 为 `[T, K]`（`K = N*D`），并在 K 方向（`[1,32]` block，row-wise）和 T 方向（`[32,1]` block，col-wise）进行基于块的动态量化，输出低精度的 FP8/FP4 张量和对应的 E8M0 缩放因子。

- 计算公式：

  **阶段1：CLA gate 融合**

  逐 token、逐 head 做两路 Sigmoid 门控加权：

  $$
  s_g = \sigma(z_g), \qquad s_l = \sigma(z_l)
  $$

  $$
  O_{t,n,d} = s_{g,t,n} \cdot O_{g,t,n,d} + s_{l,t,n} \cdot O_{l,t,n,d}
  $$

  量化前逻辑矩阵：

  $$
  X = \operatorname{reshape}(O, [T, K]), \qquad K = N \times D
  $$

  **阶段2：双轴动态块量化**

  - **Row-wise 量化（K 方向）**：将 X 在 K 维度上按 32 个数一组量化，得到 `rowScaleOut` 与 `rowDataOut`。
  - **Col-wise 量化（T 方向，仅 `dualAxisFlag=true`）**：将 X 在 T 维度上按 32 个数一组量化，得到 `colScaleOut` 与 `colDataOut`。

  block 内共享指数：

  $$
  shared\_exp = floor(\log_2(\max_i |V_i|)) - emax, \qquad scale = 2^{shared\_exp}, \qquad P_i = cast(V_i / scale)
  $$

  - emax: 对应数据类型的最大正则数的指数位。

    |   DataType    | emax |
    | :-----------: | :--: |
    |  FLOAT4_E2M1  |  2   |
    |  FLOAT4_E1M2  |  0   |
    | FLOAT8_E4M3FN |  8   |
    |  FLOAT8_E5M2  |  15  |

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 | shape |
| :----- | :------------ | :--- | :------- | :------- | :---- |
| global_attn | 输入 | Global/CLA 分支 Attention 输出 `O_g` | BFLOAT16、FLOAT16 | ND (TND) | `[T, N, D]` |
| local_attn | 输入 | Local/SWA 分支 Attention 输出 `O_l`，shape 与 global_attn 一致 | BFLOAT16、FLOAT16 | ND (TND) | `[T, N, D]` |
| global_gate_logits | 输入 | Global gate 的 Sigmoid 前值 `z_g` | BFLOAT16、FLOAT16 | ND | `[T, N]` |
| local_gate_logits | 输入 | Local gate 的 Sigmoid 前值 `z_l` | BFLOAT16、FLOAT16 | ND | `[T, N]` |
| row_data | 输出 | Row-wise 量化数据 `P_i` | FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2 | ND | `[T, N*D]` |
| row_scale | 输出 | Row-wise 每个 `[1,32]` 分组的 E8M0 缩放因子（两两一组存最后一维，偶数 pad 补 0） | FLOAT8_E8M0 | ND | `[T, ceil(N*D/64), 2]` |
| col_data | 输出 | Col-wise 量化数据 `P_j`；`dual_axis_flag=false` 时为空输出 | FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2 | ND | `[T, N*D]` |
| col_scale | 输出 | Col-wise 每个 `[32,1]` 分组的 E8M0 缩放因子（偶数 pad 补 0）；`dual_axis_flag=false` 时为空输出 | FLOAT8_E8M0 | ND | `[ceil(T/64), N*D, 2]` |
| dst_type | 属性 | 目标量化类型：35=FLOAT8_E5M2、36=FLOAT8_E4M3FN、40=FLOAT4_E2M1、41=FLOAT4_E1M2。默认 36 | INT64 | - | - |
| round_mode | 属性 | 舍入模式：FP8 仅支持 `"rint"`；FP4 支持 `"rint"`/`"floor"`/`"round"`。默认 `"rint"` | STRING | - | - |
| scale_alg | 属性 | scale 计算方法：1=cuBLAS，0=OCP；FP4 仅支持 0。默认 1 | INT64 | - | - |
| inputAttnLayout | 属性 | 输入 global_attn/local_attn 的排布格式，当前仅支持 `"TND"`。默认 `"TND"` | STRING | - | - |
| dual_axis_flag | 属性 | 量化模式：true=双轴量化（默认，同时输出 row/col 两套结果）；false=单轴量化（仅输出 row-wise，col_data/col_scale 为空输出） | BOOL | - | - |

> 说明：算子属性注册顺序为 `dst_type`、`round_mode`、`scale_alg`、`inputAttnLayout`、`dual_axis_flag`，与 tiling、aclnn 实现保持一致。

## 约束说明

- `global_attn` / `local_attn` 必须为 3 维张量且 shape 一致，为 `[T, N, D]`；`N ∈ [1,128]`，`D ∈ {128, 256}`。
- `global_gate_logits` / `local_gate_logits` 必须为 2 维张量 `[T, N]`，数据类型与 `global_attn`/`local_attn` 一致。
- 当 `dst_type` 为 40/41（FP4）时，`K = N*D` 必须可被 4 整除，且 `scale_alg` 必须为 0。
- FP8 输出类型仅支持 `"rint"` 舍入模式。
- `dual_axis_flag=false` 时，`col_data` / `col_scale` 为空输出（内部空形状 `[0]`），接口不访问该输出。
- 默认支持饱和模式。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :------- | :------- | :--- |
| aclnn 调用 | [test_aclnn_cla_gate_quant](./examples/arch35/test_aclnn_cla_gate_quant.cpp) | 通过 [aclnnClaGateQuant](./docs/aclnnClaGateQuant算子aclnn接口.md) 接口方式调用 ClaGateQuant 算子。 |
| 图模式 | - | 通过算子 IR 构图方式调用 ClaGateQuant 算子。 |
