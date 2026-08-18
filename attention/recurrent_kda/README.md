<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# RecurrentKda

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                  |    ×     |
| <term>Atlas 推理系列产品</term>                          |    ×     |
| <term>Atlas 训练系列产品</term>                          |    ×     |

## 功能说明

- 算子功能：完成 KDA（Kimi Delta Attention）的递归前向计算，面向 decode 和 MTP
  （Multi-Token Prediction）短序列场景。算子支持 BSND/TND、变长序列、state pool 槽位索引、
  V-first/K-first state、非连续 state view，以及可选的 q/k L2 normalize、raw gate 变换和
  beta sigmoid。

- 计算公式：

  对每个 token，令 $\bar q_t$ 为乘过 `scale` 的 query（可选先做 L2 normalize），
  $\bar k_t$ 为 key（可选做 L2 normalize），$d_t$ 为 gate 对应的衰减系数，
  $\bar\beta_t$ 为 beta 的有效值，则：

  $$
  \begin{aligned}
  \bar S_t &= S_{t-1}\odot d_t \\
  \Delta_t &= \bar\beta_t(v_t-\bar S_t\bar k_t) \\
  S_t &= \bar S_t+\Delta_t\bar k_t^T \\
  o_t &= S_t\bar q_t
  \end{aligned}
  $$

  `useGateInKernel=false` 时，输入 gate 是 step log gate，$d_t=\exp(gate_t)$。
  `useGateInKernel=true` 时，算子使用 `A_log` 和可选 `dt_bias` 在 kernel 内生成
  step log gate。完整 raw gate、safe gate 和 beta sigmoid 公式见
  [aclnnRecurrentKda](./docs/aclnnRecurrentKda.md)。

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| query | 输入 | BSND: `[B,T,H_k,128]`；TND: `[T,H_k,128]` | BFLOAT16 | ND |
| key | 输入 | shape 与 query 相同 | BFLOAT16 | ND |
| value | 输入 | BSND: `[B,T,H_v,V_dim]`；TND: `[T,H_v,V_dim]`，`V_dim` 为 128 或 256 | BFLOAT16 | ND |
| gate | 输入 | step log gate 或 raw gate，shape 尾维为 128 | FLOAT/BFLOAT16/FLOAT16 | ND |
| beta | 输入 | Delta 更新系数，BSND: `[B,T,H_v]`；TND: `[T,H_v]` | FLOAT/BFLOAT16/FLOAT16 | ND |
| initialStateRef | 输入/输出 | state pool；V-first 为 `[state_capacity,H_v,V_dim,128]`，K-first 为 `[state_capacity,H_v,128,V_dim]` | BFLOAT16/FLOAT | ND |
| cuSeqlensOptional | 可选输入 | 设备侧累计 offset tensor，shape 为 `[seq_num+1]` | INT32/INT64 | ND |
| ssmStateIndicesOptional | 可选输入 | packed `[T_capacity]` 或 speculative `[seq_num,max_step]` state slot 索引 | INT32/INT64 | ND |
| aLogOptional | 可选输入 | raw gate 参数，shape 为 `[H_v]` | FLOAT | ND |
| dtBiasOptional | 可选输入 | raw gate 偏置，shape 为 `[H_v*128]` 或 `[H_v,128]` | FLOAT | ND |
| numAcceptedTokensOptional | 可选输入 | 每条序列 accepted token 数，shape 为 `[seq_num]` | INT32/INT64 | ND |
| attnOut | 输出 | shape 与 value 相同 | BFLOAT16 | ND |
| finalState | 输出 | shape、dtype 与 initialStateRef 相同 | BFLOAT16/FLOAT | ND |

主要属性如下：

| 属性名 | 默认值 | 说明 |
| --- | :---: | --- |
| layout | `"BSND"` | 支持 `"BSND"`、`"TND"` |
| scale | 1.0 | 乘到 query 上，常用 `128^-0.5` |
| outputFinalState | false | 是否返回有效 finalState |
| inplaceFinalState | true | 是否原位更新 initialStateRef |
| useQkL2normInKernel | false | 是否在 kernel 内对 q/k 做 L2 normalize |
| useGateInKernel | false | 是否将 gate 解释为 raw gate |
| useBetaSigmoidInKernel | false | 是否在 kernel 内计算 sigmoid(beta) |
| allowNegEigval | false | beta sigmoid 后是否乘 2 |
| safeGate | false | 是否启用 safe gate 分支 |
| lowerBound | -5.0 | safe gate 下界，启用时范围为 `[-5,0)` |
| stateVFirst | false | state 尾部布局是否为 `[V_dim,K_dim]` |

## 约束说明

- 输入 shape、dtype、可选参数依赖、状态写回组合和两段式 aclnn 调用详见
  [aclnnRecurrentKda](./docs/aclnnRecurrentKda.md)。
- `H_v` 必须能被 `H_k` 整除；K 维固定为 128，V 维支持 128、256。
- 每条实际序列长度不超过 8。传 `cuSeqlensOptional` 时允许末 offset 小于 token capacity，
  padding tail 输出不保证有效。
- 未传 `ssmStateIndicesOptional` 时，state 第 0 维必须等于逻辑序列数；传入索引后可使用更大的
  state pool，索引值必须落在有效槽位范围内。
- `numAcceptedTokensOptional` 依赖 `ssmStateIndicesOptional`；
  `ssmStateIndicesOptional` 可以单独传入。
- 非连续 state 只允许 slot/head 外层维带间隔，内部二维 state 矩阵必须稠密且各地址区间不得重叠。
- `useGateInKernel=true` 时必须传 `aLogOptional`；`dtBiasOptional` 可选。
  `useGateInKernel=false` 时不能传 raw gate 相关可选 tensor，也不能启用 `safeGate`。

## 调用说明

| 调用方式 | 样例/文档 | 说明 |
| --- | --- | --- |
| aclnn 接口 | [aclnnRecurrentKda.md](./docs/aclnnRecurrentKda.md) | 两段式接口原型、参数说明和核心调用示例 |
| PyTorch 测试 | [test_accuracy.py](./tests/pytest/test_accuracy.py) | BSND/TND、dtype、state 布局、非连续 state、MTP 和融合开关覆盖 |
