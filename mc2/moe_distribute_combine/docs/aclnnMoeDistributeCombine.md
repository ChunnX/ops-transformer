# aclnnMoeDistributeCombine

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明

算子功能：当存在TP域通信时，先进行ReduceScatterV通信，再进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）；当不存在TP域通信时，进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）。

>注意该接口必须与`aclnnMoeDistributeDispatch`配套使用，相当于按`MoeDistributeDispatch`算子收集数据的路径原路返还。

## 函数原型

每个算子分为两段式接口，必须先调用 `aclnnMoeDistributeCombineGetWorkspaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnMoeDistributeCombine`接口执行计算。

```cpp
aclnnStatus aclnnMoeDistributeCombineGetWorkspaceSize(
    const aclTensor* expandX,
    const aclTensor* expertIds,
    const aclTensor* expandIdx,
    const aclTensor* epSendCounts,
    const aclTensor* expertScales,
    const aclTensor* tpSendCounts,
    const aclTensor* xActiveMask,
    const aclTensor* activationScale,
    const aclTensor* weightScale,
    const aclTensor* groupList,
    const aclTensor* expandScales,
    const char*      groupEp,
    int64_t          epWorldSize,
    int64_t          epRankId,
    int64_t          moeExpertNum,
    const char*      groupTp,
    int64_t          tpWorldSize,
    int64_t          tpRankId,
    int64_t          expertShardType,
    int64_t          sharedExpertNum,
    int64_t          sharedExpertRankNum,
    int64_t          globalBs,
    int64_t          outDtype,
    int64_t          commQuantMode,
    int64_t          groupListType,
    aclTensor*       x,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnMoeDistributeCombine(
    void            *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream     stream)
```

## aclnnMoeDistributeCombineGetWorkspaceSize

### 参数说明

<table style="undefined;table-layout: fixed; width: 1576px;">
 <colgroup>
  <col style="width: 170px;">
  <col style="width: 170px;">
  <col style="width: 800px;">
  <col style="width: 800px;">
  <col style="width: 200px;">
 </colgroup>
 <thead>
  <tr>
   <th>参数名</th>
   <th>输入/输出</th>
   <th>描述</th>
   <th>数据类型</th>
   <th>数据格式</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>expandX</td>
   <td>输入</td>
   <td>根据expertIds进行扩展过的token特征，Device侧的aclTensor，要求为2D Tensor，shape为 (max(tpWorldSize, 1) * A , H)；支持非连续的Tensor。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertIds</td>
   <td>输入</td>
   <td>每个token的topK个专家索引，Device侧的aclTensor，要求为2D Tensor，shape为 (BS, K)；支持非连续的Tensor。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expandIdx</td>
   <td>输入</td>
   <td>对应aclnnMoeDistributeDispatch中的expandIdx输出，Device侧的aclTensor，要求为1D Tensor，shape为 (BS*K, )；支持非连续的Tensor。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>epSendCounts</td>
   <td>输入</td>
   <td>对应aclnnMoeDistributeDispatch中的epRecvCounts输出，Device侧的aclTensor，要求为1D Tensor；支持非连续的Tensor。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertScales</td>
   <td>输入</td>
   <td>每个token的topK个专家的权重，Device侧的aclTensor，要求为2D Tensor，shape为 (BS, K)；支持非连续的Tensor。</td>
   <td>FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tpSendCounts</td>
   <td>输入</td>
   <td>对应aclnnMoeDistributeDispatch中的tpRecvCounts输出，Device侧的aclTensor；有TP域通信需传参，无TP域通信传空指针。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>xActiveMask</td>
   <td>输入</td>
   <td>Device侧的aclTensor，预留参数。当前版本不支持，传空指针即可。</td>
   <td>-</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>activationScale</td>
   <td>输入</td>
   <td>Device侧的aclTensor，预留参数。当前版本不支持，传空指针即可。</td>
   <td>-</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>weightScale</td>
   <td>输入</td>
   <td>Device侧的aclTensor，预留参数。当前版本不支持，传空指针即可。</td>
   <td>-</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>groupList</td>
   <td>输入</td>
   <td>Device侧的aclTensor，预留参数。当前版本不支持，传空指针即可。</td>
   <td>-</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expandScales</td>
   <td>输入</td>
   <td>对应aclnnMoeDistributeDispatch中的expandScales输出，Device侧的aclTensor。</td>
   <td>FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>groupEp</td>
   <td>输入</td>
   <td>EP通信域名称（专家并行通信域），字符串长度范围为[1, 128)，不能和groupTp相同。</td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>epWorldSize</td>
   <td>输入</td>
   <td>EP通信域大小。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>epRankId</td>
   <td>输入</td>
   <td>EP域本卡Id，取值范围[0, epWorldSize)，同一个EP通信域中各卡的epRankId不重复。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>moeExpertNum</td>
   <td>输入</td>
   <td>MoE专家数量，取值范围(0, 512]，且满足moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>groupTp</td>
   <td>输入</td>
   <td>TP通信域名称（数据并行通信域）。</td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tpWorldSize</td>
   <td>输入</td>
   <td>TP通信域大小。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tpRankId</td>
   <td>输入</td>
   <td>TP域本卡Id。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertShardType</td>
   <td>输入</td>
   <td>表示共享专家卡分布类型。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sharedExpertNum</td>
   <td>输入</td>
   <td>表示共享专家数量（一个共享专家可复制部署到多个卡上）。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sharedExpertRankNum</td>
   <td>输入</td>
   <td>表示共享专家卡数量。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>globalBs</td>
   <td>输入</td>
   <td>EP域全局的batch size大小。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>outDtype</td>
   <td>输入</td>
   <td>用于指定输出x的数据类型，预留参数，当前版本不支持，传0即可。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>commQuantMode</td>
   <td>输入</td>
   <td>通信量化类型。<br><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：取值范围0或2，0表示通信不量化，2表示通信int8量化（2仅当HCCL_INTRA_PCIE_ENABLE=1、HCCL_INTRA_ROCE_ENABLE=0且驱动版本≥25.0.RC1.1时支持）。<br><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值范围0或2，0表示通信不量化，2表示通信int8量化。<br><term>昇腾910_95 AI处理器</term>：当前版本仅支持0，0表示通信不量化。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>groupListType</td>
   <td>输入</td>
   <td>group List格式，预留参数，当前版本不支持，传0即可。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>x</td>
   <td>输出</td>
   <td>表示处理后的token，Device侧的aclTensor，要求为2D Tensor，shape为 (BS, H)；数据类型、数据格式与expandX保持一致。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>workspaceSize</td>
   <td>输出</td>
   <td>返回需要在Device侧申请的workspace大小。</td>
   <td>UINT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>executor</td>
   <td>输出</td>
   <td>返回op执行器，包含了算子的计算流程。</td>
   <td>aclOpExecutor*</td>
   <td>ND</td>
  </tr>
 </tbody>
</table>

- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：
    - 不支持共享专家场景。
    - epSendCounts 的shape为 (moeExpertNum + 2 * globalBs * K * serverNum, )，前moeExpertNum个数表示从EP通信域各卡接收的token数，2 * globalBs * K * serverNum存储机间机内通信前combine可提前做reduce的token个数和通信区偏移，globalBs=0时按Bs * epWorldSize计算。
    - 当前不支持TP域通信。
    - expandScales 要求为1D Tensor，shape为 (A, )；支持非连续的Tensor。
    - epWorldSize 取值支持16、32、64。
    - moeExpertNum 还需满足moeExpertNum / (epWorldSize - sharedExpertRankNum) <= 24。
    - groupTp 当前版本不支持，传空字符即可。
    - tpWorldSize、tpRankId、expertShardType、sharedExpertNum、sharedExpertRankNum当前版本不支持，传0即可。
    - 各rank Bs一致时，globalBs = Bs * epWorldSize 或 0；各rank Bs不一致时，globalBs = maxBs * epWorldSize 或 256 * epWorldSize（maxBs为单rank BS最大值，建议按maxBs * epWorldSize传入）。

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    - epSendCounts 的shape为 (epWorldSize * max(tpWorldSize, 1) * localExpertNum, )。
    - 有TP域通信时 tpSendCounts 为1D Tensor，shape为 (tpWorldSize, )；支持非连续的Tensor。
    - expandScales 为预留参数，当前版本不支持，传空指针即可。
    - epWorldSize 取值支持8、16、32、64、128、144、256、288。
    - groupTp 字符串长度范围为[1, 128)，不能和groupEp相同。
    - tpWorldSize 取值范围[0, 2]，0和1表示无TP域通信，有TP域通信时仅支持2。
    - tpRankId 取值范围[0, 1]，同一个TP通信域中各卡的tpRankId不重复；无TP域通信时传0即可。
    - expertShardType 当前仅支持传0，表示共享专家卡排在MoE专家卡前面。
    - sharedExpertNum 当前取值范围[0, 1]，0表示无共享专家，1表示一个共享专家，当前版本仅支持1。
    - sharedExpertRankNum 当前取值范围[0, epWorldSize)，不为0时需满足epWorldSize % sharedExpertRankNum = 0。
    - 各rank Bs一致时，globalBs = Bs * epWorldSize 或 0；各rank Bs不一致时，globalBs = maxBs * epWorldSize（maxBs为单卡BS最大值）。

- <term>昇腾910_95 AI处理器</term>：
    - epSendCounts 的shape为 (epWorldSize * max(tpWorldSize, 1) * localExpertNum, )。
    - 当前不支持TP域通信。
    - expandScales 为预留参数，当前版本不支持，传空指针即可。
    - epWorldSize 取值支持4、8、16、32、64、128、144、256、288。
    - groupTp 当前版本不支持，传空字符即可。
    - tpWorldSize 当前版本不支持，传1即可。
    - tpRankId 当前版本不支持，传0即可。
    - expertShardType 当前仅支持传0，表示共享专家卡排在MoE专家卡前面。
    - sharedExpertNum 当前取值范围[0, 1]，0表示无共享专家，1表示一个共享专家，当前版本仅支持1。
    - sharedExpertRankNum 当前取值范围[0, epWorldSize)，不为0时需满足epWorldSize % sharedExpertRankNum = 0。
    - 各rank Bs一致时，globalBs = Bs * epWorldSize 或 0。

### 返回值

返回aclnnStatus状态码，具体参见aclnn返回码。

第一段接口完成入参校验，出现以下场景时报错：

<table style="undefined;table-layout: fixed; width: 1576px;">
 <colgroup>
  <col style="width: 170px;">
  <col style="width: 170px;">
  <col style="width: 400px;">
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
   <td>输入和输出的必选参数Tensor是空指针。</td>
  </tr>
  <tr>
   <td>ACLNN_ERR_PARAM_INVALID</td>
   <td>161002</td>
   <td>输入和输出的数据类型不在支持的范围内。</td>
  </tr>
  <tr>
   <td>ACLNN_ERR_INNER_TILING_ERROR</td>
   <td>561002</td>
   <td>1. 输入和输出的shape不在支持的范围内；<br>2. 参数的取值不在支持的范围。</td>
  </tr>
 </tbody>
</table>

## aclnnMoeDistributeCombine

### 参数说明

<table style="undefined;table-layout: fixed; width: 1576px;">
 <colgroup>
  <col style="width: 170px;">
  <col style="width: 170px;">
  <col style="width: 800px;">
 </colgroup>
 <thead>
  <tr>
   <th>参数名</th>
   <th>输入/输出</th>
   <th>描述</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>workspace</td>
   <td>输入</td>
   <td>在Device侧申请的workspace内存地址。</td>
  </tr>
  <tr>
   <td>workspaceSize</td>
   <td>输入</td>
   <td>在Device侧申请的workspace大小，由第一段接口`aclnnMoeDistributeCombineGetWorkspaceSize`获取。</td>
  </tr>
  <tr>
   <td>executor</td>
   <td>输入</td>
   <td>op执行器，包含了算子计算流程。</td>
  </tr>
  <tr>
   <td>stream</td>
   <td>输入</td>
   <td>指定执行任务的Stream。</td>
  </tr>
 </tbody>
</table>

### 返回值

返回aclnnStatus状态码，具体参见aclnn返回码。

## 约束说明

1. `aclnnMoeDistributeDispatch`接口与`aclnnMoeDistributeCombine`接口必须配套使用，具体参考调用示例。

2. 在不同产品型号、不同通信算法或不同版本中，`aclnnMoeDistributeDispatch`的Tensor输出`expandIdx`、`epRecvCounts`、`tpRecvCounts`、`expandScales`中的元素值可能不同，使用时直接将上述Tensor传给`aclnnMoeDistributeCombine`对应参数即可，模型其他业务逻辑不应对其存在依赖。

3. 调用接口过程中使用的`groupEp`、`epWorldSize`、`moeExpertNum`、`groupTp`、`tpWorldSize`、`expertShardType`、`sharedExpertNum`、`sharedExpertRankNum`、`globalBs`参数取值所有卡需保持一致，网络中不同层中也需保持一致，且和`aclnnMoeDistributeDispatch`对应参数也保持一致。

4. <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。

5. 参数说明里shape格式说明：
    - **A**：表示本卡需要分发的最大token数量，取值范围如下：
      - 对于共享专家，需满足 (A = BS * epWorldSize * sharedExpertNum / sharedExpertRankNum)。
      - 对于MoE专家，当`globalBs`为0时，需满足 (A >= BS * epWorldSize * min(localExpertNum, K))；当`globalBs`非0时，需满足 (A >= globalBs * min(localExpertNum, K))。
    - **H**：表示hidden size（隐藏层大小）：
      - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：取值范围(0, 7168]，且需为32的整数倍。
      - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值为7168。
    - **BS**：表示batch sequence size（本卡最终输出的token数量）：
      - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：取值范围为 (0 < BS ≤ 256)。
      - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值范围为 (0 < BS ≤ 512)。
    - **K**：表示选取topK个专家，需满足 (0 < K ≤ moeExpertNum)：
      - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：取值范围为 (0 < K ≤ 16)。
      - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>昇腾910_95 AI处理器</term>：取值范围为 (0 < K ≤ 8)。
    - **serverNum**：表示服务器的节点数，取值仅支持2、4、8。
    - **localExpertNum**：表示本卡专家数量：
      - 对于共享专家卡，(localExpertNum = 1)。
      - 对于MoE专家卡，(localExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum))；当(localExpertNum > 1)时，不支持TP域通信。

6. **HCCL_BUFFSIZE**：
   调用本接口前需检查`HCCL_BUFFSIZE`环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB：
   - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：要求 (≥ 2 * (BS * epWorldSize * min(localExpertNum, K) * H * sizeof(uint16) + 2MB))。
   - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：要求 (≥ 2) 且满足 (1024^2 * (HCCL_BUFFSIZE - 2) / 2 ≥ BS * 2 * (H + 128) * (epWorldSize * localExpertNum + K + 1))，其中`localExpertNum`需使用MoE专家卡的本卡专家数。
   - <term>昇腾910_95 AI处理器</term>：要求 (≥ aivNum * 32 + 2 * epWorldSize * BS * H * 2 * localExpertNum)，其中`aivNum`表示核数，`localExpertNum`需使用MoE专家卡的本卡专家数。

7. **HCCL_INTRA_PCIE_ENABLE和HCCL_INTRA_ROCE_ENABLE**：
   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：设置环境变量`HCCL_INTRA_PCIE_ENABLE = 1`和`HCCL_INTRA_ROCE_ENABLE = 0`可减少跨机通信数据量，可能提升算子性能。此时，`HCCL_BUFFSIZE`要求 (≥ moeExpertNum * BS * (H * sizeof(dtypeX) + 4 * ((K + 7) / 8 * 8) * sizeof(uint32)) + 4MB + 100MB)；且对于入参`moeExpertNum`，仅要求 (moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0)，不要求 (moeExpertNum / (epWorldSize - sharedExpertRankNum) ≤ 24)。

8. 本文公式中的“/”表示整除。

9. 通信域使用约束：
   - 一个模型中的`aclnnMoeDistributeCombine`和`aclnnMoeDistributeDispatch`仅支持相同EP通信域，且该通信域中不允许有其他算子。
   - 一个模型中的`aclnnMoeDistributeCombine`和`aclnnMoeDistributeDispatch`仅支持相同TP通信域或都不支持TP通信域；有TP通信域时，该通信域中不允许有其他算子。

## 调用示例
