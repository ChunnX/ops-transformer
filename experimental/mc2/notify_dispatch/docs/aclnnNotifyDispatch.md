# aclnnNotifyDispatch

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

实现Notify Dispatch通信操作，用于All-to-All通信的通知阶段。该算子用于在分布式MoE场景下，通知各rank准备接收数据。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用 "aclnnNotifyDispatchGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnNotifyDispatch"接口执行计算。

```cpp
aclnnStatus aclnnNotifyDispatchGetWorkspaceSize(
    const aclTensor *sendData,
    const aclTensor *tokenPerExpertData,
    int64_t sendCount,
    int64_t numTokens,
    const char *commGroup,
    int64_t rankSize,
    int64_t rankId,
    int64_t localRankSize,
    int64_t localRankId,
    aclTensor *sendDataOffset,
    aclTensor *recvData,
    aclTensor *totalRecvTokens,
    aclTensor *recvCount,
    aclTensor *recvOffset,
    aclTensor *maxBs,
    aclTensor *recvTokensPerExpert,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
```

```cpp
aclnnStatus aclnnNotifyDispatch(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
```

## aclnnNotifyDispatchGetWorkspaceSize

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
    <td>sendData</td>
    <td>输入</td>
    <td>发送的数据，包含每个expert的token数量、偏移和总数。</td>
    <td>FLOAT16/FLOAT32/INT32</td>
    <td>ND</td>
    <td>[num_experts * 3]</td>
    </tr>
    <tr>
    <td>tokenPerExpertData</td>
    <td>输入</td>
    <td>每个expert的token数量。</td>
    <td>INT32</td>
    <td>ND</td>
    <td>[num_experts]</td>
    </tr>
    <tr>
    <td>sendCount</td>
    <td>输入</td>
    <td>发送数量，token总数。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>numTokens</td>
    <td>输入</td>
    <td>Token总数。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commGroup</td>
    <td>输入</td>
    <td>通信域名称。</td>
    <td>char*</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>rankSize</td>
    <td>输入</td>
    <td>Rank总数。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>rankId</td>
    <td>输入</td>
    <td>当前Rank ID。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>localRankSize</td>
    <td>输入</td>
    <td>Local rank size，单机rank数量。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>localRankId</td>
    <td>输入</td>
    <td>Local rank ID。</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>sendDataOffset</td>
    <td>输出</td>
    <td>发送数据偏移。</td>
    <td>INT64</td>
    <td>ND</td>
    <td>[num_experts]</td>
    </tr>
    <tr>
    <td>recvData</td>
    <td>输出</td>
    <td>接收的数据。</td>
    <td>与sendData相同</td>
    <td>ND</td>
    <td>[rankSize * num_experts * 3]</td>
    </tr>
    <tr>
    <td>totalRecvTokens</td>
    <td>输出</td>
    <td>总接收token数。</td>
    <td>INT64</td>
    <td>ND</td>
    <td>[1]</td>
    </tr>
    <tr>
    <td>recvCount</td>
    <td>输出</td>
    <td>接收计数。</td>
    <td>INT64</td>
    <td>ND</td>
    <td>[num_experts]</td>
    </tr>
    <tr>
    <td>recvOffset</td>
    <td>输出</td>
    <td>接收偏移。</td>
    <td>INT64</td>
    <td>ND</td>
    <td>[num_experts]</td>
    </tr>
    <tr>
    <td>maxBs</td>
    <td>输出</td>
    <td>最大batch size。</td>
    <td>INT64</td>
    <td>ND</td>
    <td>[1]</td>
    </tr>
    <tr>
    <td>recvTokensPerExpert</td>
    <td>输出</td>
    <td>每个expert接收的token数。</td>
    <td>INT64</td>
    <td>ND</td>
    <td>[num_experts / rankSize]</td>
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

## aclnnNotifyDispatch

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
  - sendData支持FLOAT16、FLOAT32、INT32。
  - tokenPerExpertData必须是INT32类型。
  - 输出Tensor使用INT64类型。

- 参数约束：
  - rankSize取值范围：[1, 384]。
  - num_experts必须能被rankSize整除。
  - localRankSize必须是rankSize的整数倍。

## 调用示例

参见[示例代码](../examples/notify_dispatch_sample.py)。