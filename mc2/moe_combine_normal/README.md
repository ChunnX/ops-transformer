# MoeCombineNormal

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Ascend 950PR/Ascend 950DT</term>                      |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    ×    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    ×    |
| <term>Atlas 训练系列产品</term>                              |    ×    |

## 功能说明

算子功能：该算子实现 MoE（Mixture of Experts）场景下的 Combine 阶段计算，将专家计算结果按路由权重加权求和，恢复为与输入相同形状的输出。

计算公式：

$$
Y[i] = \sum_{k=0}^{K-1} W[i,\, k] \cdot X_{recv}[\pi(i,\, k)]
$$

说明：对每个 Token $i$，根据路由索引 $\pi(i,k)$ 从接收到的专家结果中取出对应行，按 `topkWeights` 中的权重逐元素加权累加，得到最终输出 $Y$。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| recv_x | 输入 | 接收到的专家计算结果。 | BF16、FP16 | ND |
| token_src_info | 输入 | Token 来源信息，包含源 Rank、Token ID、TopK ID。 | INT32 | ND |
| ep_recv_counts | 输入 | 专家并行域接收计数。 | INT32 | ND |
| recv_topk_weights | 输入 | 每个 Token 的 topK 个专家权重。 | FP32 | ND |
| tp_recv_counts | 可选输入 | Tensor 并行域接收计数。 | INT32 | ND |
| ep_group_name | 属性 | 专家并行通信域名称。 | STRING | |
| ep_world_size | 属性 | 专家并行通信域大小。 | INT64 | |
| ep_rank_id | 属性 | 专家并行域中的 Rank ID。 | INT64 | |
| tp_group_name | 可选属性 | Tensor 并行通信域名称，默认为空字符串。 | STRING | |
| tp_world_size | 可选属性 | Tensor 并行通信域大小，默认为 0。 | INT64 | |
| tp_rank_id | 可选属性 | Tensor 并行域中的 Rank ID，默认为 0。 | INT64 | |
| moe_expert_num | 属性 | MoE 模型的总专家数量。 | INT64 | |
| global_bs | 可选属性 | 全局 batch size，默认为 0 表示自动计算。 | INT64 | |
| x | 输出 | 加权求和后的输出结果。 | BF16、FP16 | ND |
| combine_send_cost_stats | 可选输出 | 发送耗时统计信息。 | INT32 | ND |

## 约束说明

- **参数一致性约束**：
    - 调用算子过程中使用的 `ep_world_size`、`global_bs`、`HCCL_BUFFSIZE` 等参数取值，所有卡需保持一致。

- **通信域使用约束**：
    - 支持 EP 域和 TP 域。
    - 所有卡的 `moe_expert_num`、`ep_world_size` 参数取值需保持一致。

- **参数约束**：
  - BS（recv_x.dim0）范围 [1, 8000]。
  - H（recv_x.dim1）范围 [1024, 7168]。
  - topK（topkWeights.dim1）范围 [1, 16]。
  - epWorldSize 范围 [2, 384]。
  - moeExpertNum 范围 [epWorldSize, 512]，且 moeExpertNum % epWorldSize == 0。
  - globalBs 为 0 或满足 BS × epWorldSize <= globalBs 且 globalBs % epWorldSize == 0。

## 调用说明

该算子为 MC2 通算融合算子，需要在分布式环境下使用 HCCL 通信库。