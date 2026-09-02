# BlockAttentionResiduals

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：将 `prefixSum` 与 `blockRes` 拼接后完成 RMS 归一化、投影打分与 Softmax 加权融合，输出 `hiddenStates`；当 `needBackward` 为 true 时，同时输出反向所需的 `invNorm` 和 `probs`。
- 详细接口说明参见 [aclnnBlockAttentionResiduals](docs/aclnnBlockAttentionResiduals.md)。
