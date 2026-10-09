# LightningIndexerV2BatchInvariant

## 产品支持情况

| 产品 | 是否支持 |
| :-- | :--: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| 其他产品 | × |

## 功能说明

LightningIndexerV2BatchInvariant 的计算功能、输入输出和属性与
[LightningIndexerV2](../lightning_indexer_v2/README.md) 保持一致，用于得到每个 token 对应的 top-k 位置和可选分数。

本算子默认提供确定性和 batch 一致性：相同输入重复执行的输出完全一致；对于完全相同的单条样本输入和算子属性，
改变样本在 batch 中的位置或 batch 大小，该样本的输出仍保持完全一致。

## 接口说明

- ACLNN：[aclnnLightningIndexerV2BatchInvariant](./docs/aclnnLightningIndexerV2BatchInvariant.md)
- PyTorch：[torch.ops.cann_ops_transformer.lightning_indexer_batch_invariant](../../torch_extension/cann_ops_transformer/docs/zh/lightning_indexer_batch_invariant.md)

## 约束说明

- 仅支持 ND 格式。
- `q` 和 `k` 支持 FLOAT16、BFLOAT16，且数据类型必须一致；`w` 和 `sparseValues` 为 FLOAT32。
- head dim 仅支持 128；`q` 的 head 数支持 1~64，`k` 的 head 数仅支持 1。
- `topk` 支持 [1, 2048]，以及 3072、4096、5120、6144、7168、8192。
- 其余 shape、layout、mask、PageAttention 和可选输入约束与 A2/A3 上的 LightningIndexerV2 一致。
