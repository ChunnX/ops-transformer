 # aclnnQuantLightningIndexer

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/quant_lightning_indexer)

## 产品支持情况

| 产品                                                    | 是否支持 |
| ------------------------------------------------- | :-----------: |
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 接口功能：`QuantLightningIndexer`在`LightningIndexer`的基础上对输入query和key进行量化。

- 计算公式：
    $$
    out = \text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(\left(Scale_Q@Scale_K^T\right)\odot\left(Q_{index}^{Quant}@{\left(K_{index}^{Quant}\right)}^T\right)\right)\right]\right\}
    $$

## 函数原型



