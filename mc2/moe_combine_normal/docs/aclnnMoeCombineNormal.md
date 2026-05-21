# aclnnMoeCombineNormal

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

实现MoE Combine阶段计算，将经过专家处理后的token数据重新组合还原。该算子用于分布式MoE推理场景下的数据合并阶段，配合moe_dispatch_normal使用。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用 "aclnnMoeCombineNormalGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnMoeCombineNormal"接口执行计算。

```cpp
aclnnStatus aclnnMoeCombineNormalGetWorkspaceSize(
    const aclTensor *recvX,
    const aclTensor *tokenSrcInfo,
    const aclTensor *epRecvCount,
    const aclTensor *topkWeights,
    const aclTensor *tpRecvCount,
    const char *groupEp,
    int64_t epWorldSize,
    int64_t epRankId,
    const char *groupTpOptional,
    int64_t tpWorldSize,
    int64_t tpRankId,
    int64_t moeExpertNum,
    int64_t quantMode,
    int64_t globalBs,
    aclTensor *xOut,
    aclTensor *sendCostStatsOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
```

```cpp
aclnnStatus aclnnMoeCombineNormal(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
```

## aclnnMoeCombineNormalGetWorkspaceSize

- **参数说明**

    <table>
    <thead>
    <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
    <th>维度（shape）</th>
    </tr></thead>
    <tbody>
    <tr>
    <td>recvX</td>
    <td>输入</td>
    <td>接收到的token特征，来自dispatch输出。</td>
    <td>FLOAT16/BFLOAT16</td>
    <td>ND</td>
    <td>[total_recv_tokens, hidden]</td>
    </tr>
    <tr>
    <td>tokenSrcInfo</td>
    <td>输入</td>
    <td>token源信息，来自dispatch的assistInfoForCombine输出。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>[total_recv_tokens, 3]</td>
    </tr>
    <tr>
    <td>epRecvCount</td>
    <td>输入</td>
    <td>EP接收计数，来自dispatch输出。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>[num_experts]</td>
    </tr>
    <tr>
    <td>topkWeights</td>
    <td>输入</td>
    <td>TopK权重，shape为[BS, K]。</td>
    <td>FLOAT32</td>
    <td>ND</td>
    <td>[num_tokens, num_topk]</td>
    </tr>
    <tr>
    <td>tpRecvCount</td>
    <td>输入（可选）</td>
    <td>TP接收计数。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupEp</td>
    <td>输入</td>
    <td>EP通信域名称。</td>
    <td>char*</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epWorldSize</td>
    <td>输入</td>
    <td>EP通信域大小。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epRankId</td>
    <td>输入</td>
    <td>EP本卡ID。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>groupTpOptional</td>
    <td>输入</td>
    <td>TP通信域名称（可选）。</td>
    <td>char*</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tpWorldSize</td>
    <td>输入</td>
    <td>TP通信域大小。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>tpRankId</td>
    <td>输入</td>
    <td>TP本卡ID。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>moeExpertNum</td>
    <td>输入</td>
    <td>MoE专家数量。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>quantMode</td>
    <td>输入</td>
    <td>量化模式。当前仅支持传0。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>globalBs</td>
    <td>输入</td>
    <td>全局batch size。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>xOut</td>
    <td>输出</td>
    <td>合并后的输出特征。</td>
    <td>FLOAT16/BFLOAT16</td>
    <td>ND</td>
    <td>[num_tokens, hidden]</td>
    </tr>
    <tr>
    <td>sendCostStatsOut</td>
    <td>输出（可选）</td>
    <td>发送耗时统计信息。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>[epWorldSize]</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>输出</td>
    <td>返回workspace大小。</td>
    <td>UINT64</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>输出</td>
    <td>返回op执行器。</td>
    <td>aclOpExecutor*</td>
    <td>-</td>
    <td>-</td>
    </tr>
    </tbody></table>

- **返回值**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## aclnnMoeCombineNormal

- **参数说明**

    <table>
    <thead>
    <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    </tr></thead>
    <tbody>
    <tr>
    <td>workspace</td>
    <td>输入</td>
    <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>输入</td>
    <td>在Device侧申请的workspace大小。</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>输入</td>
    <td>op执行器。</td>
    </tr>
    <tr>
    <td>stream</td>
    <td>输入</td>
    <td>指定执行任务的Stream。</td>
    </tr>
    </tbody></table>

- **返回值**

    返回aclnnStatus状态码。

## 约束说明

- 数据类型约束：
  - recvX和xOut支持FLOAT16、BFLOAT16。
  - tokenSrcInfo、epRecvCount必须是INT32类型。
  - topkWeights必须是FLOAT32类型。
  - hidden维度范围：[1024, 7168]。

- 参数约束：
  - epWorldSize取值范围：[2, 384]。
  - tpWorldSize取值范围：[0, 2]。
  - moeExpertNum取值范围：[1, 512]，且必须能被epWorldSize整除。
  - num_topk取值范围：[1, 16]，且不能超过moeExpertNum。

## 调用示例

参见[示例代码](../examples/moe_dispatch_combine_normal_sample.py)。