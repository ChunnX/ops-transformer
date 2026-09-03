# BlockAttentionResidualsGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                      |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

BlockAttentionResidualsGrad 是正向算子 BlockAttentionResiduals（注意力残差）的反向传播算子。前向将token的残差状态按block组织，通过inv_norm逐block归一化，并结合proj_weight、norm_weight构成投影权重后经softmax得到probs；反向利用前向保存的inv_norm、probs和上游梯度grad_hidden_states，计算partial_block、block_res、proj_weight、norm_weight的梯度。

## 接口说明

- aclnn接口：[aclnnBlockAttentionResidualsGrad](./docs/aclnnBlockAttentionResidualsGrad.md)
