# aclnnMoeEpDispatchEpilogue

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mc2/moe_ep_dispatch_epilogue)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |
| <term>Atlas A5 训练系列产品/Atlas A5 推理系列产品</term>     |    √     |

## 功能说明

- 接口功能：该接口为aclnnMoeEpDispatch的内部中间步骤算子，用于在dispatch的alltoallv通信完成后，将host pinned内存中的计数值读回并整理接收到的token数据，生成最终的recv输出张量。该接口由框架内部自动调用，用户无需直接使用。

- 计算公式：

    $$recvX = Rearrange(alltoallXOut, numRecvTokensPerRank, numRecvTokensPerExpert)$$

    $$recvTopkWeights = RearrangeTopkWeights(alltoallXOut, dstBufferSlotIdx) \quad (recvTopkWeights\ is\ optional)$$

    $$recvScales = RearrangeScales(alltoallXOut, numRecvTokensPerRank)$$

> 注意该接口为内部中间步骤算子，由aclnnMoeEpDispatch内部自动调用，用户无需直接使用。

## 函数原型

每个算子分为两段式接口，必须先调用"aclnnMoeEpDispatchEpilogueGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnMoeEpDispatchEpilogue"接口执行计算。

```cpp
aclnnStatus aclnnMoeEpDispatchEpilogueGetWorkspaceSize(
    const aclTensor*  context,
    const aclTensor*  dstBufferSlotIdx,
    const aclTensor*  numRecvTokensPerRank,
    const aclTensor*  numRecvTokensPerExpert,
    const aclTensor*  cachedRecvSrcMetadata,
    int64_t           epWorldSize,
    int64_t           epRankId,
    int64_t           numExperts,
    int64_t           numMaxTokensPerRank,
    int64_t           cclBufferSize,
    int64_t           expertAlignment,
    aclTensor*        recvX,
    aclTensor*        recvTopkWeights,
    aclTensor*        recvSrcMetadata,
    aclTensor*        recvScales,
    uint64_t*         workspaceSize,
    aclOpExecutor**   executor)
```

```cpp
aclnnStatus aclnnMoeEpDispatchEpilogue(
    void            *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream     stream)
```

## aclnnMoeEpDispatchEpilogueGetWorkspaceSize

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1567px"> <colgroup>
    <col style="width: 120px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 330px">
    <col style="width: 212px">
    <col style="width: 100px">
    <col style="width: 190px">
    <col style="width: 145px">
    </colgroup>
    <thead>
    <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>使用说明</th>
    <th>数据类型</th>
    <th>数据格式</th>
    <th>维度(shape)</th>
    <th>非连续Tensor</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>context（aclTensor*）</td>
    <td>输入</td>
    <td>HCCL通信域上下文。</td>
    <td>-</td>
    <td>INT32</td>
    <td>ND</td>
    <td>1</td>
    <td>×</td>
    </tr>
    <tr>
    <td>dstBufferSlotIdx（aclTensor*）</td>
    <td>输入</td>
    <td>slot索引张量，由aclnnMoeEpDispatch输出。</td>
    <td><ul><li>不支持空Tensor。</li><li>shape为(BS, K)。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>numRecvTokensPerRank（aclTensor*）</td>
    <td>输入</td>
    <td>各卡接收token数量的张量，由aclnnMoeEpDispatch输出。</td>
    <td><ul><li>不支持空Tensor。</li><li>shape为(epWorldSize,)。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>1</td>
    <td>×</td>
    </tr>
    <tr>
    <td>numRecvTokensPerExpert（aclTensor*）</td>
    <td>输入</td>
    <td>每个本地专家接收token数量的张量，由aclnnMoeEpDispatch输出。</td>
    <td><ul><li>不支持空Tensor。</li><li>shape为(numLocalExperts,)。</li></ul></td>
    <td>INT64</td>
    <td>ND</td>
    <td>1</td>
    <td>×</td>
    </tr>
    <tr>
    <td>cachedRecvSrcMetadata（aclTensor*）</td>
    <td>输入</td>
    <td>cached模式下复用的接收元数据张量。</td>
    <td><ul><li>Optional参数，非cached模式下传入空指针。</li><li>cached模式下必选，shape为(A, 3)。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>epWorldSize（int64_t）</td>
    <td>输入</td>
    <td>EP通信域大小。</td>
    <td><ul><li>取值范围[2, 128]。</li></ul></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epRankId（int64_t）</td>
    <td>输入</td>
    <td>EP域中的rank编号。</td>
    <td><ul><li>取值范围[0, epWorldSize)。</li></ul></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>numExperts（int64_t）</td>
    <td>输入</td>
    <td>MoE专家总数量。</td>
    <td><ul><li>取值范围[2, 2048]，且满足numExperts % epWorldSize = 0。</li></ul></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>numMaxTokensPerRank（int64_t）</td>
    <td>输入</td>
    <td>每张卡上的最大token数量上限。</td>
    <td><ul><li>必须大于0。</li></ul></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>cclBufferSize（int64_t）</td>
    <td>输入</td>
    <td>HCCL通信域缓存区大小（字节）。</td>
    <td><ul><li>必须大于0。</li></ul></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>expertAlignment（int64_t）</td>
    <td>输入</td>
    <td>专家对齐数。</td>
    <td><ul><li>Optional参数，默认为1。</li></ul></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>recvX（aclTensor*）</td>
    <td>输出</td>
    <td>接收的token数据，对应公式中recvX。</td>
    <td><ul><li>不支持空Tensor。</li><li>shape为(A, H)。</li></ul></td>
    <td>BFLOAT16、FLOAT16、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>recvTopkWeights（aclTensor*）</td>
    <td>输出</td>
    <td>接收的topk权重，对应公式中recvTopkWeights。</td>
    <td><ul><li>Optional参数，未请求topk权重输出时传入空指针。</li><li>非空时shape为(A)。</li></ul></td>
    <td>FLOAT</td>
    <td>ND</td>
    <td>1</td>
    <td>×</td>
    </tr>
    <tr>
    <td>recvSrcMetadata（aclTensor*）</td>
    <td>输出</td>
    <td>整理后的接收元数据，字段为[src_token_idx, src_rank_id, src_topk_idx]。</td>
    <td><ul><li>不支持空Tensor。</li><li>shape为(A, 3)。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>recvScales（aclTensor*）</td>
    <td>输出</td>
    <td>接收的动态量化缩放因子，用于FP8量化场景。</td>
    <td><ul><li>Optional参数，非量化场景传入空指针。</li></ul></td>
    <td>FLOAT、FLOAT8_E8M0</td>
    <td>ND</td>
    <td>-</td>
    <td>×</td>
    </tr>
    <tr>
    <td>workspaceSize（uint64_t*）</td>
    <td>输出</td>
    <td>返回需要在Device侧申请的workspace大小。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor（aclOpExecutor**）</td>
    <td>输出</td>
    <td>返回op执行器，包含了算子计算流程。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    </tbody></table>

- **返回值**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

    第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed; width: 1000px"> <colgroup>
    <col style="width: 300px">
    <col style="width: 150px">
    <col style="width: 550px">
    </colgroup>
    <thead>
    <tr>
    <th>返回值</th>
    <th>错误码</th>
    <th>描述</th>
    </tr>
    </thead>
    <tbody>
    <tr>
    <td>ACLNN_ERR_PARAM_NULLPTR</td>
    <td>161001</td>
    <td>context、dstBufferSlotIdx等必选参数存在空指针。</td>
    </tr>
    <tr>
    <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
    <td rowspan="2">161002</td>
    <td>dstBufferSlotIdx的shape维度不为2。</td>
    </tr>
    <tr>
    <td>numExperts不能被epWorldSize整除。</td>
    </tr>
    </tbody></table>

## aclnnMoeEpDispatchEpilogue

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1000px"> <colgroup>
    <col style="width: 180px">
    <col style="width: 120px">
    <col style="width: 700px">
    </colgroup>
    <thead>
    <tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr>
    </thead>
    <tbody>
    <tr><td>workspace</td><td>输入</td><td>在Device侧申请的workspace内存地址。</td></tr>
    <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnMoeEpDispatchEpilogueGetWorkspaceSize获取。</td></tr>
    <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
    <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
    </tbody></table>

- **返回值**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

- 确定性说明：aclnnMoeEpDispatchEpilogue默认确定性实现。
- 该接口为内部中间步骤算子，由aclnnMoeEpDispatch内部自动调用，用户无需直接使用。
- dstBufferSlotIdx、numRecvTokensPerRank、numRecvTokensPerExpert等输入参数需与aclnnMoeEpDispatch中对应输出参数保持一致。
- 参数里Shape使用的变量如下：
    - A：表示本卡接收的最大token数量，A = epWorldSize \* numMaxTokensPerRank \* MIN(K, numLocalExperts)。
    - H：表示hidden size隐藏层大小。取值范围为(0, 8192]。
    - BS：表示batch sequence size，即本卡的token数量。
    - K：表示选取topK个专家，取值范围为1 ≤ K ≤ 32。
    - numLocalExperts：表示本卡专家数量，numLocalExperts = numExperts / epWorldSize。
- <term>Atlas A5 训练系列产品/Atlas A5 推理系列产品</term>：recvX支持BFLOAT16、FLOAT16、FLOAT8_E5M2、FLOAT8_E4M3FN数据类型输出。

## 调用示例

该接口由aclnnMoeEpDispatch内部自动调用，无需用户直接调用。详见aclnnMoeEpDispatch调用示例。