# aclnnMoeEpCombine

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mc2/moe_ep_combine)

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

- 接口功能：需与aclnnMoeEpDispatch配套使用，相当于按dispatch算子收集数据的路径原路返回。该接口将专家处理后的token数据根据dispatch阶段记录的元数据信息，通过逆向路由和加权聚合，将token数据组合还原为原始序列顺序。
  - 支持带topkWeightsOptional加权聚合和纯累加两种模式；
  - 当前版本不支持bias参数。

- 计算公式：

    - 当提供topkWeightsOptional时：

    $$combinedX_i = \sum_{k=0}^{K-1} topkWeights_{i,k} \times x_{slot(i,k)}$$

    - 当不提供topkWeightsOptional时（纯累加）：

    $$combinedX_i = \sum_{k=0}^{K-1} x_{slot(i,k)}$$

> 注意该接口必须与aclnnMoeEpDispatch配套使用。

## 函数原型

每个算子分为两段式接口，必须先调用"aclnnMoeEpCombineGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnMoeEpCombine"接口执行计算。

```cpp
aclnnStatus aclnnMoeEpCombineGetWorkspaceSize(
    const aclTensor*  context,
    const aclTensor*  x,
    const aclTensor*  topkIdx,
    const aclTensor*  recvSrcMetadata,
    const aclTensor*  numRecvTokensPerExpert,
    const aclTensor*  topkWeightsOptional,
    const aclTensor*  biasOptional0,
    const aclTensor*  biasOptional1,
    int64_t           epWorldSize,
    int64_t           epRankId,
    int64_t           numExperts,
    int64_t           numMaxTokensPerRank,
    int64_t           cclBufferSize,
    aclTensor*        combinedX,
    aclTensor*        combinedTopkWeightsOptional,
    uint64_t*         workspaceSize,
    aclOpExecutor**   executor)
```

```cpp
aclnnStatus aclnnMoeEpCombine(
    void            *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream     stream)
```

## aclnnMoeEpCombineGetWorkspaceSize

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
    <td>x（aclTensor*）</td>
    <td>输入</td>
    <td>表示经过专家计算后的token数据，对应公式中x。</td>
    <td><ul><li>不支持空Tensor。</li><li>数据类型仅支持BFLOAT16。</li></ul></td>
    <td>BFLOAT16</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>topkIdx（aclTensor*）</td>
    <td>输入</td>
    <td>topk专家索引张量。</td>
    <td><ul><li>不支持空Tensor。</li><li>shape为(BS, K)。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>recvSrcMetadata（aclTensor*）</td>
    <td>输入</td>
    <td>接收元数据，由aclnnMoeEpDispatchEpilogue输出。</td>
    <td><ul><li>不支持空Tensor。</li><li>shape为(A, 4)。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>numRecvTokensPerExpert（aclTensor*）</td>
    <td>输入</td>
    <td>本地moe专家接收token数量，由aclnnMoeEpDispatchEpilogue输出。</td>
    <td><ul><li>不支持空Tensor。</li><li>shape为(numLocalExperts,)。</li></ul></td>
    <td>INT64</td>
    <td>ND</td>
    <td>1</td>
    <td>×</td>
    </tr>
    <tr>
    <td>topkWeightsOptional（aclTensor*）</td>
    <td>输入</td>
    <td>表示每个token对应的topK专家权重，用于加权聚合，对应公式中topkWeights。</td>
    <td><ul><li>Optional参数，传入空指针时进行纯累加combine。</li><li>shape为(A,)。</li></ul></td>
    <td>FLOAT</td>
    <td>ND</td>
    <td>1</td>
    <td>×</td>
    </tr>
    <tr>
    <td>biasOptional0（aclTensor*）</td>
    <td>输入</td>
    <td>预留bias参数。</td>
    <td><ul><li>预留参数，当前暂不支持，传入空指针即可。</li></ul></td>
    <td>BFLOAT16</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>biasOptional1（aclTensor*）</td>
    <td>输入</td>
    <td>预留bias参数。</td>
    <td><ul><li>预留参数，当前暂不支持，传入空指针即可。</li></ul></td>
    <td>BFLOAT16</td>
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
    <td>combinedX（aclTensor*）</td>
    <td>输出</td>
    <td>表示combine后的token数据，还原为原始序列顺序，对应公式中combinedX。</td>
    <td><ul><li>不支持空Tensor。</li><li>不支持非连续Tensor。</li><li>shape为(BS, H)。</li></ul></td>
    <td>BFLOAT16</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>combinedTopkWeightsOptional（aclTensor*）</td>
    <td>输出</td>
    <td>表示combine后的topK专家权重。</td>
    <td><ul><li>Optional参数，当topkWeightsOptional传入空指针时也传入空指针。</li><li>shape为(BS, K)。</li></ul></td>
    <td>FLOAT</td>
    <td>ND</td>
    <td>2</td>
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
    <td>context、x、topkIdx等必选参数存在空指针。</td>
    </tr>
    <tr>
    <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
    <td rowspan="3">161002</td>
    <td>x的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
    <td>biasOptional0或biasOptional1不为空指针（当前版本不支持bias）。</td>
    </tr>
    <tr>
    <td>numExperts不能被epWorldSize整除。</td>
    </tr>
    </tbody></table>

## aclnnMoeEpCombine

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
    <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnMoeEpCombineGetWorkspaceSize获取。</td></tr>
    <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
    <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
    </tbody></table>

- **返回值**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

- 确定性说明：aclnnMoeEpCombine默认确定性实现。
- 该接口必须与aclnnMoeEpDispatch配套使用。
- 当前版本不支持bias参数，biasOptional0和biasOptional1必须传入空指针。
- 调用接口过程中使用的epWorldSize、numExperts、numMaxTokensPerRank参数需与aclnnMoeEpDispatch保持一致。
- 参数里Shape使用的变量如下：
    - A：表示本卡接收的最大token数量，A = epWorldSize \* numMaxTokensPerRank \* MIN(K, numLocalExperts)。
    - H：表示hidden size隐藏层大小。取值范围为(0, 8192]。
    - BS：表示batch sequence size，即本卡的原始token数量。
    - K：表示选取topK个专家，取值范围为1 ≤ K ≤ 32。
    - numLocalExperts：表示本卡专家数量，numLocalExperts = numExperts / epWorldSize。
- HCCL通信域缓存区大小：
    调用本接口前需检查通信域缓存区大小取值是否合理，单位MB，不配置时默认为200MB。
    - 通信域缓存区大小可通过调用MoEElasticBuffer.get_moe_ep_ccl_buffer_size接口计算。
    - 设置大小要求 >= 2 \* (epWorldSize \* numMaxTokensPerRank \* Align512(Align32(2 \* H)))。
    - 其中 Align512(x) = ((x+512-1)/512)\*512，Align32(x) = ((x+32-1)/32)\*32。
- 通信域使用约束：
    - 一个模型中的aclnnMoeEpDispatch、aclnnMoeEpDispatchEpilogue和aclnnMoeEpCombine算子仅支持相同EP通信域，且该通信域中不允许有其他算子。

## 调用示例
