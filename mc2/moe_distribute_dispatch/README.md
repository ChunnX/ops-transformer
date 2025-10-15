# MoeDistributeDispatch

> 注意：
> 本文档仅仅是算子功能的简介，不支持用户直接调用，因为当前不支持kernel直调，等后续支持再完善文档!!!!!!

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |




## 功能说明

算子功能：对Token数据进行量化（可选），当存在TP域通信时，先进行EP（Expert Parallelism）域的AllToAllV通信，再进行TP（Tensor Parallelism）域的AllGatherV通信；当不存在TP域通信时，进行EP（Expert Parallelism）域的AllToAllV通信。

注意该接口必须与aclnnMoeDistributeCombine配套使用。


## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px">
<colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 1000px">
<col style="width: 1000px">
<col style="width: 200px">
</colgroup>
<thead>
<tr>
<th>参数名</th>
<th>输入/输出/属性</th>
<th>描述</th>
<th>数据类型</th>
<th>数据格式</th>
</tr>
</thead>
<tbody>
<tr>
<td>x</td>
<td>输入</td>
<td>本卡发送的token数据，Device侧的aclTensor，要求为2D Tensor。</td>
<td>FLOAT16、BFLOAT16、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8</td>
<td>ND</td>
</tr>
<tr>
<td>expertIds</td>
<td>输入</td>
<td>每个token的topK个专家索引，Device侧的aclTensor，要求为2D Tensor。</td>
<td>INT32</td>
<td>ND</td>
</tr>
<tr>
<td>scales</td>
<td>输入</td>
<td>每个专家的平滑权重、融合量化平滑权重的量化系数或量化系数，Device侧的aclTensor，要求为1D或2D Tensor。</td>
<td>FLOAT32、FLOAT8_E8M0</td>
<td>ND</td>
</tr>
<tr>
<td>xActiveMask</td>
<td>输入</td>
<td>Device侧的aclTensor，预留参数，当前版本不支持，传空指针即可。</td>
<td>-</td>
<td>ND</td>
</tr>
<tr>
<td>expertScales</td>
<td>输入</td>
<td>每个Token的topK个专家权重，Device侧的aclTensor，要求为2D Tensor。</td>
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
<td>表示共享专家卡分布类型，当前仅支持传0，表示共享专家卡排在MoE专家卡前面。</td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>sharedExpertNum</td>
<td>输入</td>
<td>表示共享专家数量（一个共享专家可复制部署到多个卡上），当前取值范围[0, 1]，0表示无共享专家，1表示一个共享专家，当前版本仅支持传1。</td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>sharedExpertRankNum</td>
<td>输入</td>
<td>表示共享专家卡数量，当前取值范围[0, epWorldSize)，不为0时需满足epWorldSize % sharedExpertRankNum = 0。</td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>quantMode</td>
<td>输入</td>
<td>表示量化模式，支持0：非量化，1：静态量化，2：pertoken动态量化，3：pergroup动态量化，4：mx量化。</td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>globalBs</td>
<td>输入</td>
<td>EP域全局的batch size大小，各rank Bs一致时，globalBs = Bs * epWorldSize 或 0；各rank Bs不一致时，globalBs = maxBs * epWorldSize（maxBs为单卡/单rank BS最大值）。</td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>expertTokenNumsType</td>
<td>输入</td>
<td>输出expertTokenNums中值的语义类型，支持0：expertTokenNums中的输出为每个专家处理的token数的前缀和，1：expertTokenNums中的输出为每个专家处理的token数量。</td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>expandX</td>
<td>输出</td>
<td>根据expertIds进行扩展过的token特征，Device侧的aclTensor，要求为2D Tensor。</td>
<td>FLOAT16、BFLOAT16、INT8、FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8</td>
<td>ND</td>
</tr>
<tr>
<td>dynamicScales</td>
<td>输出</td>
<td>Device侧的aclTensor，要求为1D或2D Tensor；支持非连续的Tensor。</td>
<td>FLOAT32、FLOAT32_E8M0</td>
<td>ND</td>
</tr>
<tr>
<td>expandIdx</td>
<td>输出</td>
<td>表示给同一专家发送的token个数（对应aclnnMoeDistributeCombine中的expandIdx），Device侧的aclTensor，要求为1D Tensor。</td>
<td>INT32</td>
<td>ND</td>
</tr>
<tr>
<td>expertTokenNums</td>
<td>输出</td>
<td>表示每个专家收到的token个数，Device侧的aclTensor，要求为1D Tensor。</td>
<td>INT64</td>
<td>ND</td>
</tr>
<tr>
<td>epRecvCounts</td>
<td>输出</td>
<td>从EP通信域各卡接收的token数（对应aclnnMoeDistributeCombine中的epSendCounts），Device侧的aclTensor，要求为1D Tensor。</td>
<td>INT32</td>
<td>ND</td>
</tr>
<tr>
<td>tpRecvCounts</td>
<td>输出</td>
<td>从TP通信域各卡接收的token数（对应aclnnMoeDistributeCombine中的tpSendCounts），Device侧的aclTensor；若有TP域通信则有该输出，若无TP域通信则无该输出，有TP域通信时要求为1D Tensor。</td>
<td>INT32</td>
<td>ND</td>
</tr>
<tr>
<td>expandScales</td>
<td>输出</td>
<td>表示本卡输出Token的权重（对应aclnnMoeDistributeCombine中的expandScales），Device侧的aclTensor，要求为1D Tensor。</td>
<td>FLOAT32</td>
<td>ND</td>
</tr>
</tbody>
</table>


## 约束说明

- aclnnMoeDistributeDispatch接口与aclnnMoeDistributeCombine接口必须配套使用，具体参考调用示例。

- 在不同产品型号、不同通信算法或不同版本中，aclnnMoeDistributeDispatch的Tensor输出expandIdx、epRecvCounts、tpRecvCounts、expandScales中的的元素值可能不同，使用时直接将上述Tensor传给aclnnMoeDistributeCombine对应参数即可，模型其他业务逻辑不应对其存在依赖。

- 调用接口过程中使用的groupEp、epWorldSize、moeExpertNum、groupTp、tpWorldSize、expertShardType、sharedExpertNum、sharedExpertRankNum、globalBs参数取值所有卡需保持一致，网络中不同层中也需保持一致，且和aclnnMoeDistributeCombine对应参数也保持一致。

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。

- 参数说明里shape格式说明：
    - A：表示本卡可能接收的最大token数量，取值范围如下：
        - 对于共享专家，要满足A = BS \* epWorldSize \* sharedExpertNum / sharedExpertRankNum。
        - 对于MoE专家，当globalBs为0时，要满足A >= BS \* epWorldSize \* min(localExpertNum, K)；当globalBs非0时，要满足A >= globalBs \* min(localExpertNum, K)。
    - H：表示hidden size隐藏层大小。
        - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：取值范围(0, 7168]，且保证是32的整数倍。
    - BS：表示batch sequence size，即本卡最终输出的token数量。
        - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：取值范围为0 < BS ≤ 256。
    - K：表示选取topK个专家，需满足0 < K ≤ moeExpertNum。
        - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：取值范围为0 < K ≤ 16。
    - serverNum：表示服务器的节点数，取值只支持2、4、8。
    - localExpertNum：表示本卡专家数量。
        - 对于共享专家卡，localExpertNum = 1
        - 对于MoE专家卡，localExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum)，localExpertNum > 1时，不支持TP域通信。

- HCCL_BUFFSIZE：
    调用本接口前需检查HCCL_BUFFSIZE环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：要求 >= 2 \* (BS \* epWorldSize \* min(localExpertNum, K) \* H \* sizeof(uint16) + 2MB)。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：要求 >= 2且满足1024 ^ 2 \* (HCCL_BUFFSIZE - 2) / 2 >= BS \* 2 \* (H + 128) \* (epWorldSize \* localExpertNum + K + 1)，localExpertNum需使用MoE专家卡的本卡专家数。

- HCCL_INTRA_PCIE_ENABLE和HCCL_INTRA_ROCE_ENABLE：
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：设置环境变量HCCL_INTRA_PCIE_ENABLE = 1和HCCL_INTRA_ROCE_ENABLE = 0可以减少跨机通信数据量，可能提升算子性能。
    此时，HCCL_BUFFSIZE要求 >= moeExpertNum \* BS \* (H \* sizeof(dtypeX) + 4 \* ((K + 7) / 8 \* 8) \* sizeof(uint32)) + 4MB + 100MB。并且，对于入参moeExpertNum，只要求moeExpertNum \% (epWorldSize - sharedExpertRankNum) = 0，不要求moeExpertNum / (epWorldSize - sharedExpertRankNum) <= 24。
    
- 本文公式中的"/"表示整除。

- 通信域使用约束：
    - 一个模型中的aclnnMoeDistributeCombine和aclnnMoeDistributeDispatch仅支持相同EP通信域，且该通信域中不允许有其他算子。
    - 一个模型中的aclnnMoeDistributeCombine和aclnnMoeDistributeDispatch仅支持相同TP通信域或都不支持TP通信域，有TP通信域时该通信域中不允许有其他算子。

- quantMode相关约束：
    - quantMode取值为0时，表示非量化场景，expandX的数据类型支持FLOAT16、BFLOAT16、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8。
        - expandX的数据类型为FLOAT16、BFLOAT16时，输入scales必须传入空指针。
        - expandX的数据类型为FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8时，输入scales必须传入有效数据,且输入scales的shape第1维必须等于BS。
    - quantMode取值为1时，表示静态量化场景，expandX的数据类型支持INT8、HIFLOAT8。
        - expandX的数据类型为INT8时，输入scales为量化系数时，shape为 \(1, \)；输入scales为每个专家共享的平滑权重时，shape为 \(H，\)。输入scales为融了每个专家的平滑权重的量化系数时，若有共享专家卡，其shape为 \(sharedExpertNum + moeExpertNum, H\)，若无共享专家卡，其shape为 \(moeExpertNum, H\)。
        - expandX的数据类型为HIFLOAT8时，scales的shape必须为 \(1, \)。
    - quantMode取值为2时，表示pertoken动态量化场景，expandX的数据类型支持INT8、FLOAT8_E4M3FN、FLOAT8_E5M2。
        - 输入scales可传入空指针。
        - 若输入scales传入有效数据且存在共享专家卡时，其shape为 \(sharedExpertNum + moeExpertNum, H\)。
        - 若输入scales传入有效数据且不存在共享专家卡时，其shape为 \(moeExpertNum, H\)。
    - quantMode取值为3时，表示pergroup动态量化场景，expandX的数据类型支持FLOAT8_E4M3FN、FLOAT8_E5M2。
        - 输入scales可传入空指针。
        - 若输入scales传入有效数据且存在共享专家卡时，其shape为 \(sharedExpertNum + moeExpertNum, H\)。
        - 若输入scales传入有效数据且不存在共享专家卡时，其shape为 \(moeExpertNum, H\)。
    - quantMode取值为4时，表示mx量化场景，expandX的数据类型支持FLOAT8_E4M3FN、FLOAT8_E5M2，输入scales必须传入空指针。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| aclnn接口 | [test_aclnn_moe_distribute_dispatch.cpp](./examples/test_aclnn_moe_distribute_dispatch.cpp) | 通过[aclnnMoeDistributeDispatch](./docs/aclnnMoeDistributeDispatch.md)接口方式调用moe_distribute_dispatch算子。 |