# aclnnMoeEpDispatch

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mc2/moe_ep_dispatch)

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

- 接口功能：需与aclnnMoeEpCombine和aclnnMoeEpDispatchEpilogue配套使用，完成MoE的Expert Parallelism（EP）并行部署下的token dispatch第一阶段。该接口根据每个token的topK专家索引，进行slot分配和EP域的alltoallv通信，输出中间结果供epilogue步骤处理。
  - 支持cached模式，即第二次dispatch时可复用第一次的handle，跳过slot分配阶段；
  - 支持CPU同步等待（doCpuSync）。

- 计算公式：

    - 非cached模式：

    $$alltoallXOut = AllToAllV(X, scales)$$

    $$dstBufferSlotIdx = SlotAssignment(topkIdx)$$

    - cached模式：

    $$alltoallXOut = AllToAllV(X, scales)$$

> 注意该接口必须与aclnnMoeEpCombine和aclnnMoeEpDispatchEpilogue配套使用。

## 函数原型

每个算子分为两段式接口，必须先调用"aclnnMoeEpDispatchGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnMoeEpDispatch"接口执行计算。

```cpp
aclnnStatus aclnnMoeEpDispatchGetWorkspaceSize(
    const aclTensor*  context,
    const aclTensor*  x,
    const aclTensor*  topkIdx,
    const aclTensor*  topkWeights,
    const aclTensor*  scales,
    const aclTensor*  cachedHandleDstBufferSlotIdx,
    int64_t           epWorldSize,
    int64_t           epRankId,
    int64_t           numExperts,
    int64_t           numMaxTokensPerRank,
    int64_t           cclBufferSize,
    int64_t           expertAlignment,
    bool              doCpuSync,
    int64_t           hostPinnedCounterAddr,
    aclTensor*        numRecvTokensPerRank,
    aclTensor*        numRecvTokensPerExpert,
    aclTensor*        dstBufferSlotIdx,
    uint64_t*         workspaceSize,
    aclOpExecutor**   executor)
```

```cpp
aclnnStatus aclnnMoeEpDispatch(
    void            *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream     stream)
```

## aclnnMoeEpDispatchGetWorkspaceSize

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
    <td>表示计算使用的token数据，需根据topkIdx来发送给其他卡，对应公式中X。</td>
    <td><ul><li>不支持空Tensor。</li><li>数据类型支持BFLOAT16、FLOAT16、FLOAT8_E5M2、FLOAT8_E4M3FN。</li></ul></td>
    <td>BFLOAT16、FLOAT16、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>topkIdx（aclTensor*）</td>
    <td>输入</td>
    <td>表示每个token的topK个专家索引，决定每个token要发给哪些专家，对应公式中topkIdx。</td>
    <td><ul><li>Optional参数，cached模式下传入空指针。</li><li>非cached模式下必选。</li><li>张量里value取值范围为[0, numExperts)。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>topkWeights（aclTensor*）</td>
    <td>输入</td>
    <td>表示每个token对应的topK专家权重。</td>
    <td><ul><li>Optional参数，不需要时传入空指针。</li></ul></td>
    <td>FLOAT</td>
    <td>ND</td>
    <td>2</td>
    <td>×</td>
    </tr>
    <tr>
    <td>scales（aclTensor*）</td>
    <td>输入</td>
    <td>动态量化缩放因子，用于FP8量化场景。</td>
    <td><ul><li>Optional参数，非量化场景传入空指针。</li></ul></td>
    <td>FLOAT、FLOAT8_E8M0</td>
    <td>ND</td>
    <td>-</td>
    <td>×</td>
    </tr>
    <tr>
    <td>cachedHandleDstBufferSlotIdx（aclTensor*）</td>
    <td>输入</td>
    <td>表示cached模式下复用的slot索引张量。</td>
    <td><ul><li>Optional参数，非cached模式下传入空指针。</li><li>cached模式下必选。</li></ul></td>
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
    <td>doCpuSync（bool）</td>
    <td>输入</td>
    <td>是否进行CPU同步等待。</td>
    <td><ul><li>Optional参数，非cached模式默认为true，cached模式必须为false。</li></ul></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>hostPinnedCounterAddr（int64_t）</td>
    <td>输入</td>
    <td>Host-pinned counter的Device侧地址，kernel将actualA写入此地址供CPU侧SpinWait读取。</td>
    <td><ul><li>Optional参数，默认为0。</li><li>doCpuSync=true时传入HostPinnedCounter的devPtr，doCpuSync=false时传入0。</li></ul></td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>numRecvTokensPerRank（aclTensor*）</td>
    <td>输出</td>
    <td>各卡接收token数量的张量。</td>
    <td><ul><li>不支持空Tensor。</li><li>shape为(epWorldSize,)。</li></ul></td>
    <td>INT32</td>
    <td>ND</td>
    <td>1</td>
    <td>×</td>
    </tr>
    <tr>
    <td>numRecvTokensPerExpert（aclTensor*）</td>
    <td>输出</td>
    <td>每个本地专家接收token数量的张量。</td>
    <td><ul><li>不支持空Tensor。</li><li>shape为(numLocalExperts,)。</li></ul></td>
    <td>INT64</td>
    <td>ND</td>
    <td>1</td>
    <td>×</td>
    </tr>
    <tr>
    <td>dstBufferSlotIdx（aclTensor*）</td>
    <td>输出</td>
    <td>slot索引张量，供aclnnMoeEpDispatchEpilogue和aclnnMoeEpCombine使用。</td>
    <td><ul><li>不支持空Tensor。</li><li>shape为(BS, K)。</li></ul></td>
    <td>INT32</td>
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
    <td>context、x等必选参数存在空指针。</td>
    </tr>
    <tr>
    <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
    <td rowspan="3">161002</td>
    <td>x的数据类型不在支持的范围之内（仅支持BFLOAT16、FLOAT16、FLOAT8_E5M2、FLOAT8_E4M3FN）。</td>
    </tr>
    <tr>
    <td>x的shape维度不为2。</td>
    </tr>
    <tr>
    <td>numExperts不能被epWorldSize整除。</td>
    </tr>
    </tbody></table>

## aclnnMoeEpDispatch

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
    <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnMoeEpDispatchGetWorkspaceSize获取。</td></tr>
    <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
    <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
    </tbody></table>

- **返回值**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

- 确定性说明：aclnnMoeEpDispatch默认确定性实现。
- 该接口必须与aclnnMoeEpCombine和aclnnMoeEpDispatchEpilogue配套使用。
- cached模式下，topkIdx和topkWeights必须为空指针，cachedHandleDstBufferSlotIdx为必选参数，doCpuSync必须为false；非cached模式下，topkIdx为必选参数，topkWeights和cachedHandleDstBufferSlotIdx必须为空指针。
- 调用接口过程中使用的numExperts、numMaxTokensPerRank参数取值所有卡需保持一致，且和aclnnMoeEpCombine对应参数也保持一致。
- 参数里Shape使用的变量如下：
    - A：表示本卡接收的最大token数量，A = epWorldSize \* numMaxTokensPerRank \* MIN(K, numLocalExperts)。
    - H：表示hidden size隐藏层大小。取值范围为(0, 8192]。
    - BS：表示batch sequence size，即本卡的token数量。
    - K：表示选取topK个专家，取值范围为1 ≤ K ≤ 32。
    - numLocalExperts：表示本卡专家数量，numLocalExperts = numExperts / epWorldSize，应满足0 < numLocalExperts \* epWorldSize ≤ 2048。
- HCCL通信域缓存区大小：
    调用本接口前需检查HCCL_BUFFSIZE环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。
    - 通信域缓存区大小可通过调用MoEElasticBuffer.get_moe_ep_ccl_buffer_size接口计算。
    - 设置大小要求 >= 2 \* (epWorldSize \* numMaxTokensPerRank \* Align512(Align32(2 \* H)))。
    - 其中 Align512(x) = ((x+512-1)/512)\*512，Align32(x) = ((x+32-1)/32)\*32。
- 通信域使用约束：
    - 一个模型中的aclnnMoeEpDispatch、aclnnMoeEpDispatchEpilogue和aclnnMoeEpCombine算子仅支持相同EP通信域，且该通信域中不允许有其他算子。
- <term>Atlas A5 训练系列产品/Atlas A5 推理系列产品</term>：x输入支持BFLOAT16、FLOAT16、FLOAT8_E5M2、FLOAT8_E4M3FN数据类型。

## 调用示例

