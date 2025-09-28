# aclnnMoeDistributeDispatch

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

算子功能：对Token数据进行量化（可选），当存在TP域通信时，先进行EP（Expert Parallelism）域的AllToAllV通信，再进行TP（Tensor Parallelism）域的AllGatherV通信；当不存在TP域通信时，进行EP（Expert Parallelism）域的AllToAllV通信。

注意该接口必须与aclnnMoeDistributeCombine配套使用。

## 函数原型

每个算子分为两段式接口，必须先调用 “aclnnMoeDistributeDispatchGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeDistributeDispatch”接口执行计算。

* `aclnnStatus aclnnMoeDistributeDispatchGetWorkspaceSize(const aclTensor* x, const aclTensor* expertIds, const aclTensor* scales, const aclTensor* xActiveMask, const aclTensor* expertScales, const char* groupEp, int64_t epWorldSize, int64_t epRankId, int64_t moeExpertNum, const char* groupTp, int64_t tpWorldSize, int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum, int64_t sharedExpertRankNum, int64_t quantMode, int64_t globalBs, int64_t expertTokenNumsType, aclTensor* expandX, aclTensor* dynamicScales, aclTensor* expandIdx, aclTensor* expertTokenNums, aclTensor* epRecvCounts, aclTensor* tpRecvCounts, aclTensor* expandScales, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnStatus aclnnMoeDistributeDispatch(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnMoeDistributeDispatchGetWorkspaceSize

-   **参数说明**：
    - x（aclTensor\*，计算输入）：表示本卡发送的token数据，Device侧的aclTensor。要求为一个2D的Tensor，shape为 \(BS, H\)，其中BS为batch size，H为hidden size，即隐藏层大小。数据格式要求为ND，支持非连续的Tensor。
        - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT16、BFLOAT16。
    - expertIds（aclTensor\*，计算输入）：每个token的topK个专家索引，Device侧的aclTensor，要求为一个2D的Tensor，shape为 \(BS, K\)，数据类型支持INT32，数据格式要求为ND，支持非连续的Tensor。
    - scales（aclTensor\*，计算输入）：每个专家的平滑系数或者融合了每个专家的量化平滑系数的量化系数或者量化系数，Device侧的aclTensor，要求是一个1D的Tensor或者2D的Tensor，数据格式要求为ND，支持非连续的Tensor。
        - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：当HCCL_INTRA_PCIE_ENABLE为1且HCCL_INTRA_ROCE_ENABLE为0时，要求传nullptr。
        - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT32。quantMode取值为0时传空指针。quantMode取值为2时，可以选择传入有效数据或者传空指针，传入有效数据时且有共享专家卡时shape为 \(sharedExpertNum + moeExpertNum, H\)，无共享专家卡时shape为 \(moeExpertNum, H\)。
    - xActiveMask（aclTensor\*，计算输入）：Device侧的aclTensor，预留参数。
    - expertScales（aclTensor\*，计算输入）：每个Token的topK个专家权重，Device侧的aclTensor。
        - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：要求是一个2D的shape \(BS, K\)。数据类型支持FLOAT32，数据格式要求为ND，支持非连续的Tensor。
    - groupEp（char\*，计算输入）：EP通信域名称，专家并行的通信域，string数据类型。字符串长度范围为[1, 128)，不能和groupTp相同。
    - epWorldSize（int64_t，计算输入）：EP通信域size，数据类型支持INT64。
        - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：取值支持16、32、64。
        - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值支持8、16、32、64、128、144、256、288。
    - epRankId（int64_t，计算输入）: EP域本卡Id，数据类型支持INT64，取值范围[0, epWorldSize)。同一个EP通信域中各卡的epRankId不重复。
    - moeExpertNum（int64_t，计算输入）: MoE专家数量，数据类型支持INT64，取值范围(0, 512]，并满足moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0。
        - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：还需满足moeExpertNum / (epWorldSize - sharedExpertRankNum) <= 24。
    - groupTp（char\*，计算输入）：TP通信域名称，数据并行的通信域，string数据类型。
        - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：字符串长度范围为[1, 128)，不能和groupEp相同。
    - tpWorldSize（int64_t，计算输入）：TP通信域size，int数据类型。
        - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：当前版本不支持，传0即可。
        - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值范围[0, 2]，0和1表示无tp域通信，有tp域通信时仅支持2。
    - tpRankId（int64_t，计算输入）：TP域本卡Id，数据类型支持INT64。
        - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值范围[0, 1]，同一个TP通信域中各卡的tpRankId不重复。无TP域通信时，传0即可。
    - expertShardType（int64_t，计算输入）：表示共享专家卡分布类型，数据类型支持INT64。
        - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：当前版本不支持，传0即可。
    - sharedExpertNum（int64_t，计算输入）：表示共享专家数量，一个共享专家可以复制部署到多个卡上，数据类型支持INT64。
        - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：当前版本不支持，传0即可。
        - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当前取值范围[0, 1]。0表示无共享专家，1表示一个共享专家，当前版本仅支持传1。
    - sharedExpertRankNum（int64_t，计算输入）：表示共享专家卡数量，数据类型支持INT64。
        - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：当前版本不支持，传0即可。
    - quantMode（int64_t，计算输入）：表示量化模式，支持0：非量化，1：静态量化，2：pertoken动态量化，3：pergroup动态量化，4：mx量化。
        - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持quantMode取值为0或2。
    - globalBs（int64_t，计算输入）：EP域全局的batch size大小，数据类型支持INT64。
        - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：当每个rank的Bs数一致时，globalBs = Bs \* epWorldSize 或 globalBs = 0；当每个rank的Bs数不一致时，globalBs = maxBs \* epWorldSize或者globalBs = 256 \* epWorldSize，其中maxBs表示表示单rank BS最大值，建议按maxBs \* epWorldSize传入，固定按256 \* epWorldSize传入在后续版本bs支持大于256的场景下会无法支持。
        - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当每个rank的Bs数一致时，globalBs = Bs \* epWorldSize 或 globalBs = 0；当每个rank的Bs数不一致场景下，globalBs = maxBs \* epWorldSize，其中maxBs表示单卡Bs最大值。
    - expertTokenNumsType（int64_t，计算输入）：输出expertTokenNums中值的语义类型。支持0：expertTokenNums中的输出为每个专家处理的token数的前缀和，1：expertTokenNums中的输出为每个专家处理的token数量。
    - expandX（aclTensor\*，计算输出）：根据expertIds进行扩展过的token特征，Device侧的aclTensor，要求为一个2D的Tensor，shape为 \(max(tpWorldSize, 1) \* A, H\)。
        - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT16、BFLOAT16、INT8，数据格式要求为ND，支持非连续的Tensor。
    - dynamicScales（aclTensor\*，计算输出）：数据类型FLOAT32，要求为一个1D的Tensor或者2D的Tensor，数据格式要求为ND，支持非连续的Tensor。
        - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT32，其shape为 \(A, \)，当quantMode为2时，才有该输出。
    - expandIdx（aclTensor\*，计算输出）：表示给同一专家发送的token个数，对应aclnnMoeDistributeCombine中的expandIdx，Device侧的aclTensor，要求是一个1D的shape \(BS\*K, \)。数据类型支持INT32，数据格式要求为ND，支持非连续的Tensor。
    - expertTokenNums（aclTensor\*，计算输出）：表示每个专家收到的token个数，Device侧的aclTensor，数据类型INT64，要求为一个1D的Tensor，shape为 \(localExpertNum, \)，数据格式要求为ND，支持非连续的Tensor。
    - epRecvCounts（aclTensor\*，计算输出）：从EP通信域各卡接收的token数，对应aclnnMoeDistributeCombine中的epSendCounts，Device侧的aclTensor，数据类型INT32，要求为一个1D的Tensor，数据格式要求为ND，支持非连续的Tensor。
        - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：要求shape为 \(moeExpertNum + 2 \* globalBs \* K \* serverNum, \)，前moeExpertNum个数表示从EP通信域各卡接收的token数，2 \* globalBs \* K \* serverNum存储了机间机内做通信前combine可以提前做reduce的token个数和token在通信区中的偏移，globalBs传入0时在此处应当按照Bs \* epWorldSize计算。
    - tpRecvCounts（aclTensor\*，计算输出）：从TP通信域各卡接收的token数，对应aclnnMoeDistributeCombine中的tpSendCounts，Device侧的aclTensor。若有TP域通信则有该输出，若无TP域通信则无该输出。
        - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当有TP域通信时，要求是一个1D的Tensor，shape为 \(tpWorldSize, \)。数据类型支持INT32，数据格式要求为ND，支持非连续的Tensor。
    - expandScales（aclTensor\*，计算输出）：表示本卡输出Token的权重，对应aclnnMoeDistributeCombine中的expandScales，Device侧的aclTensor。
        - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：要求是一个1D的Tensor，shape为 \(A, \)，数据类型支持FLOAT32，数据格式要求为ND，支持非连续的Tensor。
    - workspaceSize（uint64\_t\*，出参）：返回需要在Device侧申请的workspace大小。
    - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。
    
-   **返回值**

    返回aclnnStatus状态码，具体参见aclnn返回码。
    ```
    第一段接口完成入参校验，出现以下场景时报错：
    161001(ACLNN_ERR_PARAM_NULLPTR): 1. 输入和输出的必选参数Tensor是空指针。
    161002(ACLNN_ERR_PARAM_INVALID): 1. 输入和输出的数据类型不在支持的范围内。
    561002(ACLNN_ERR_INNER_TILING_ERROR): 1. 输入和输出的shape不在支持的范围内。
                                          2. 参数的取值不在支持的范围。 
    ```

## aclnnMoeDistributeDispatch

-   **参数说明：**
    
    - workspace（void\*，入参）：在Device侧申请的workspace内存地址。
    - workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnMoeDistributeDispatchGetWorkspaceSize获取。
    - executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
    - stream（aclrtStream，入参）：指定执行任务的Stream。
    
-   **返回值：**

    返回aclnnStatus状态码，具体参见aclnn返回码。

## 约束说明

- aclnnMoeDistributeDispatch接口与aclnnMoeDistributeCombine接口必须配套使用，具体参考[调用示例](#调用示例)。

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

## 调用示例

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：示例代码请参考aclnnMoeDistributeDispatchV2接口调用过程，仅供参考，请根据实际情况配置；

- 文件准备：    
  1.新建combineDemo目录，按照下方指导在combineDemo下新建aclnnCombineDemo.cpp，buildCombine.sh，rank_table_m2.json文件并修改。
  2.将combineDemo项目拷贝到两台服务器中，并根据机器的device ip配置rank_table_m2.json文件内容。注意两机rank_table_m2.json文件保持一致。
  3.安装cann包，并根据下方指导编译运行combineDemo。

- 关于rankTable:
  开发者可以通过ranktable文件配置参与集合通信的NPU资源信息，详细配置请参考[《集合通信用户指南》](https://hiascend.com/document/redirect/CannCommercialHcclUg)中“通信功能开发>集群信息配置>ranktable文件配置资源信息”。

  使用`cat /etc/hccn.conf` 或者`for i in seq 0 7; do echo "===================> dev$i, NPU$((i+1))"; hccn_tool -i $i -ip -g; done`查询机器的device ip。然后参考集合通信文档填写json文件。

  注意：device_id范围是[0, 8)，且可自由选择几张卡； rank_id依次增加。以两机16卡为例，两机器的device ip都是0~7，其中一机器rank_id为0~7，则另一台机器的rank_id为8~15。

-  编译脚本
    ```bash
    #!/bin/bash
    cann_path="/path/to/cann_env" # 更改cann包环境的路径
    g++ "aclnnCombineDemo.cpp" -o combineDemo -I"$cann_path/latest/include/" -I"$cann_path/latest/include/aclnnop/" \
                        -L="$cann_path/latest/lib64/" -lascendcl -lnnopbase -lopapi -lop_common -lpthread -lhccl
    ```
- 编译与运行：

    ```bash
    # source cann环境
    source /path/to/cann_env/latest/bin/setenv.bash

    # 编译aclnnCombineDemo.cpp
    bash buildCombine.sh

    # 运行前需设置两个环境变量
    ## FIRST_RANK_ID说明：以两机16卡为例，其中一机器设置为0，另一机器设置为8
    ## 如export FIRST_RANK_ID=0
    export RANK_TABLE_FILE=/home/path/to/rank_table_m2.json
    export FIRST_RANK_ID=<设备的起始rank_id>

    # 两机同时运行
    ./combineDemo
    ```

- 示例代码如下，仅供参考
  - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：
    ```Cpp
    #include <unistd.h>
    #include <sys/types.h>
    #include <sys/wait.h>
    #include <iostream>
    #include <string>
    #include <vector>
    #include "acl/acl.h"
    #include "hccl/hccl.h"
    #include "aclnnop/aclnn_moe_distribute_dispatch.h"
    #include "aclnnop/aclnn_moe_distribute_combine.h"
    #include "aclnn/opdev/fp16_t.h"
    #include <random>

    #define CHECK_RET(cond, return_expr) \
        do {                             \
            if (!(cond)) {               \
                return_expr;             \
            }                            \
        } while (0)

    #define LOG_PRINT(message, ...)         \
        do {                                \
            printf(message, ##__VA_ARGS__); \
        } while(0)
    #define ACLCHECK(ret) do { \
        if(ret != ACL_SUCCESS)\
        {\
            printf("acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret);\
        }\
    } while(0)
    constexpr int EP_WORLD_SIZE = 16;
    constexpr int TP_WORLD_SIZE = 0;
    int FIRST_RANK_ID = 0;

    int64_t GetShapeSize(const std::vector<int64_t> &shape)
    {
        int64_t shape_size = 1;
        for (auto i : shape) {
            shape_size *= i;
        }
        return shape_size;
    }

    template<typename T>
    int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
        aclDataType dataType, aclTensor **tensor)
    {
        auto size = GetShapeSize(shape) * sizeof(T);
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i +1] * strides[i + 1];
        }
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
            shape.data(), shape.size(), *deviceAddr);
        return 0;
    }

    struct Args {
        int rankId;
        int epRankId;
        char* groupEpName;
        HcclComm hcclEpComm;
        aclrtStream stream;
    };

    int launchOneProcess_MoeDistributeCombine(Args &args)
    {
        int64_t BS = 8;
        int64_t H = 7168;
        int64_t K = 8;
        int64_t shardType = 0; // dispatch need
        int64_t quantMode = 0; // dispatch need
        bool isQuant = false;  // dispatch need
        int64_t expertTokenNumsType = 0; // dispatch need
        int64_t expertShardType = 0;
        int64_t sharedExpertRankNum = 0;
        int64_t sharedExpertNum = 0;
        int64_t moeExpertNum = 16;
        int64_t globalBS = BS * EP_WORLD_SIZE;      // tiling里处理成BS*world_size
        int64_t outDtype = 0;
        int64_t commQuantMode = 0;
        int64_t groupList_type = 0;
        const char* groupTpName = "";
        int64_t tpWorldSize = 0;
        int64_t tpRankId = 0;

        int64_t localMoeExpertNum = moeExpertNum / (EP_WORLD_SIZE - sharedExpertRankNum);
        int64_t A = 0;
        if (args.epRankId < sharedExpertRankNum) { // 共享专家
            A = BS * EP_WORLD_SIZE / sharedExpertRankNum;
            localMoeExpertNum = 1;
        } else { // Moe专家
            A = BS * EP_WORLD_SIZE * localMoeExpertNum;
        }
        int64_t epWorldSize = EP_WORLD_SIZE;
        auto outDataType = aclDataType::ACL_BF16;
        if (isQuant) {
            outDataType = aclDataType::ACL_INT8;
        }

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        void *workspaceAddr = nullptr;
        std::vector<int64_t> scalesShape{moeExpertNum, H};              // dispatch need
        std::vector<int64_t> dynamicScalesShape{A};                     // dispatch need
        std::vector<int64_t> expertTokenNumsShape{localMoeExpertNum};   // dispatch need
        std::vector<int64_t> expandScalesShape{A}; // dispatch & combine
        std::vector<int64_t> expandXShape{A, H};
        std::vector<int64_t> expertIdsShape{BS, K};
        std::vector<int64_t> expandIdxShape{BS * K};
        std::vector<int64_t> epSendCountsShape{localMoeExpertNum * EP_WORLD_SIZE};
        std::vector<int64_t> expertScalesShape{BS, K};
        std::vector<int64_t> tpSendCountsShape{1};
        std::vector<int64_t> xActiveMaskShape{BS};
        std::vector<int64_t> activationScaleShape{A};
        std::vector<int64_t> weightScaleShape{1, H};
        std::vector<int64_t> groupListShape{1};
        std::vector<int64_t> xShape{BS, H};

        void *scalesDeviceAddr = nullptr;           // dispatch need
        void *dynamicScalesDeviceAddr = nullptr;    // dispatch need
        void *expertTokenNumsDeviceAddr = nullptr;  // dispatch need
        void *expandScalesDeviceAddr = nullptr;     // dispatch & combine need
        void *expandXDeviceAddr = nullptr;
        void *expertIdsDeviceAddr = nullptr;
        void *expandIdxDeviceAddr = nullptr;
        void *epSendCountsDeviceAddr = nullptr;
        void *expertScalesDeviceAddr = nullptr;
        void *tpSendCountsDeviceAddr = nullptr;
        void *xActiveMaskDeviceAddr = nullptr; 
        void *activationScaleDeviceAddr = nullptr;
        void *weightScaleDeviceAddr = nullptr;
        void *groupListDeviceAddr = nullptr;
        void *xDeviceAddr = nullptr;
    
        aclTensor *scales = nullptr;            // dispatch need
        aclTensor *dynamicScales = nullptr;     // dispatch need
        aclTensor *expertTokenNums = nullptr;   // dispatch need
        aclTensor *expandScales = nullptr;      // dispatch & combine need
        aclTensor *expandX = nullptr;
        aclTensor *expertIds = nullptr;
        aclTensor *expandIdx = nullptr;
        aclTensor *epSendCounts = nullptr;
        aclTensor *expertScales = nullptr;
        aclTensor *tpSendCounts = nullptr;
        aclTensor *xActiveMask = nullptr; 
        aclTensor *activationScale = nullptr;
        aclTensor *weightScale = nullptr;
        aclTensor *groupList = nullptr;
        aclTensor *x = nullptr;
    
        long long scalesShapeSize = GetShapeSize(scalesShape);                      // dispatch need
        long long dynamicScalesShapeSize = GetShapeSize(dynamicScalesShape);        // dispatch need
        long long expertTokenNumsShapeSize = GetShapeSize(expertTokenNumsShape);    // dispatch need
        long long expandScalesShapeSize = GetShapeSize(expandScalesShape);          // dispatch & combine need
        long long expandXShapeSize = GetShapeSize(expandXShape);
        long long expertIdsShapeSize = GetShapeSize(expertIdsShape);
        long long expandIdxShapeSize = GetShapeSize(expandIdxShape);
        long long epSendCountsShapeSize = GetShapeSize(epSendCountsShape);
        long long expertScalesShapeSize = GetShapeSize(expertScalesShape);
        long long tpSendCountsShapeSize = GetShapeSize(tpSendCountsShape);
        long long xActiveMaskShapeSize = GetShapeSize(xActiveMaskShape);
        long long activationScaleShapeSize = GetShapeSize(activationScaleShape);
        long long weightScaleShapeSize = GetShapeSize(weightScaleShape);
        long long groupListShapeSize = GetShapeSize(groupListShape);
        long long xShapeSize = GetShapeSize(xShape);
    
        std::vector<float> scalesHostData(scalesShapeSize, 0);                      // dispatch need
        std::vector<float> dynamicScalesHostData(dynamicScalesShapeSize, 0);        // dispatch need
        std::vector<int64_t> expertTokenNumsHostData(expertTokenNumsShapeSize, 0);  // dispatch need
        std::vector<float> expandScalesHostData(expandScalesShapeSize, 0);          // dispatch & combine need
        std::vector<op::fp16_t> expandXHostData(expandXShapeSize, 0);
        std::vector<int32_t> expertIdsHostData(expertIdsShapeSize, 0);
        std::random_device rd; // 随机数设备
        std::mt19937 gen(rd()); // 以随机数设备作为种子的Mersenne Twister生成器
        std::uniform_int_distribution<> dis(sharedExpertRankNum, EP_WORLD_SIZE - 1);
        for (auto& val : expertIdsHostData) {
            val = dis(gen); // 为每个元素生成一个2到15之间的随机数
        }
        std::vector<int32_t> expandIdxHostData(expandIdxShapeSize, 0);
        std::vector<int32_t> epSendCountsHostData(epSendCountsShapeSize, 0);
        std::vector<int32_t> tpSendCountsHostData(tpSendCountsShapeSize, 0);
        std::vector<float> expertScalesHostData(expertScalesShapeSize, 0);
        std::vector<int8_t> xActiveMaskHostData(xActiveMaskShapeSize, 0);
        std::vector<float> activationScaleHostData(activationScaleShapeSize,0);
        std::vector<float> weightScaleHostData(weightScaleShapeSize,0);
        std::vector<int32_t> groupListHostData(groupListShapeSize,0);
        std::vector<op::fp16_t> xHostData(xShapeSize, 0);
    
        auto ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);                                // dispatch need
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(dynamicScalesHostData, dynamicScalesShape, &dynamicScalesDeviceAddr, aclDataType::ACL_FLOAT, &dynamicScales);         // dispatch need
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertTokenNumsHostData, expertTokenNumsShape, &expertTokenNumsDeviceAddr, aclDataType::ACL_INT64, &expertTokenNums); // dispatch need
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expandScalesHostData, expandScalesShape, &expandScalesDeviceAddr, aclDataType::ACL_FLOAT, &expandScales);             // dispatch & combine need
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expandXHostData, expandXShape, &expandXDeviceAddr, aclDataType::ACL_BF16, &expandX);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertIdsHostData, expertIdsShape, &expertIdsDeviceAddr, aclDataType::ACL_INT32, &expertIds);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expandIdxHostData, expandIdxShape, &expandIdxDeviceAddr, aclDataType::ACL_INT32, &expandIdx);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(epSendCountsHostData, epSendCountsShape, &epSendCountsDeviceAddr, aclDataType::ACL_INT32, &epSendCounts);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(tpSendCountsHostData, tpSendCountsShape, &tpSendCountsDeviceAddr, aclDataType::ACL_INT32, &tpSendCounts);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(expertScalesHostData, expertScalesShape, &expertScalesDeviceAddr, aclDataType::ACL_FLOAT, &expertScales);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(xActiveMaskHostData, xActiveMaskShape, &xActiveMaskDeviceAddr, aclDataType::ACL_BOOL, &xActiveMask);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(activationScaleHostData, activationScaleShape, &activationScaleDeviceAddr, aclDataType::ACL_FLOAT, &activationScale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(weightScaleHostData, weightScaleShape, &weightScaleDeviceAddr, aclDataType::ACL_FLOAT, &weightScale);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT32, &groupList);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    
        /******************************先调用dispatch,因为combine需要使用dispatch的数据********************************************/
        ret = aclnnMoeDistributeDispatchGetWorkspaceSize(x, expertIds, 
                (isQuant? scales : nullptr), xActiveMask, 
                expertScales, args.groupEpName, epWorldSize, args.epRankId, moeExpertNum, groupTpName, tpWorldSize, tpRankId, expertShardType, 
                sharedExpertNum,sharedExpertRankNum, quantMode, globalBS, expertTokenNumsType, expandX, dynamicScales, 
                expandIdx, expertTokenNums, epSendCounts, tpSendCounts, expandScales, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchGetWorkspaceSize failed. ret = %d \n", ret);
            return ret;
        }
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeDispatchGetWorkspaceSize failed. ret = %d \n", ret); return ret);
        // 根据第一阶段接口计算出的workspaceSize申请device内存
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
        // 调用第二阶段接口
        ret = aclnnMoeDistributeDispatch(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeDispatch failed. ret = %d \n", ret);
            return ret);

        /**************************************** 然后调用combine ********************************************/
        // 调用第一阶段接口
        ret = aclnnMoeDistributeCombineGetWorkspaceSize(expandX, expertIds,
                                                         expandIdx, epSendCounts,
                                                         expertScales, tpSendCounts,
                                                         xActiveMask, activationScale,
                                                         weightScale, groupList, expandScales, 
                                                         args.groupEpName, EP_WORLD_SIZE, 
                                                         args.epRankId, moeExpertNum,
                                                         groupTpName, tpWorldSize, tpRankId,
                                                         expertShardType, sharedExpertNum, sharedExpertRankNum,globalBS, outDtype, commQuantMode,
                                                         groupList_type, x,
                                                         &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("[ERROR] aclnnMoeDistributeCombineGetWorkspaceSize failed. ret = %d \n", ret); return ret);
        // 根据第一阶段接口计算出的workspaceSize申请device内存
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
        }
    
        // 调用第二阶段接口
        ret = aclnnMoeDistributeCombine(workspaceAddr, workspaceSize, executor, args.stream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnMoeDistributeCombine failed. ret = %d \n", ret);
            return ret);
        // （固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStreamWithTimeout(args.stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSynchronizeStreamWithTimeout failed. ret = %d \n", ret);
            return ret);
        LOG_PRINT("[INFO] device_%d aclnnMoeDistributeCombine execute successfully.\n", args.rankId);
    
        // 释放device资源，需要根据具体API的接口定义修改
        if (scales != nullptr) {                // dispatch need
            aclDestroyTensor(scales);
        }
        if (dynamicScales != nullptr) {         // dispatch need
            aclDestroyTensor(dynamicScales);
        }
        if (expertTokenNums != nullptr) {       // dispatch need
            aclDestroyTensor(expertTokenNums);
        }
        if (expandScales != nullptr) {          // dispatch & combine need
            aclDestroyTensor(expandScales);
        }
        if (expandX != nullptr) {
            aclDestroyTensor(expandX);
        }
        if (expertIds != nullptr) {
            aclDestroyTensor(expertIds);
        }
        if (expandIdx != nullptr) {
            aclDestroyTensor(expandIdx);
        }
        if (epSendCounts != nullptr) {
            aclDestroyTensor(epSendCounts);
        }
        if (tpSendCounts != nullptr) {
            aclDestroyTensor(tpSendCounts);
        }
        if (expertScales != nullptr) {
            aclDestroyTensor(expertScales);
        }
        if (x != nullptr) {
            aclDestroyTensor(x);
        }
        if (xDeviceAddr != nullptr) {
            aclrtFree(xDeviceAddr);
        }
        if (expandXDeviceAddr != nullptr) {
            aclrtFree(expandXDeviceAddr);
        }
        if (expertIdsDeviceAddr != nullptr) {
            aclrtFree(expertIdsDeviceAddr);
        }
        if (expandIdxDeviceAddr != nullptr) {
            aclrtFree(expandIdxDeviceAddr);
        }
        if (epSendCountsDeviceAddr != nullptr) {
            aclrtFree(epSendCountsDeviceAddr);
        }
        if (tpSendCountsDeviceAddr != nullptr) {
            aclrtFree(tpSendCountsDeviceAddr);
        }
        if (expertScalesDeviceAddr != nullptr) {
            aclrtFree(expertScalesDeviceAddr);
        }
        if (workspaceSize > 0) {
            aclrtFree(workspaceAddr);
        }
        aclrtDestroyStream(args.stream);
        HcclCommDestroy(args.hcclEpComm);
        aclrtResetDevice(args.rankId);
        return 0;
    }
    
    void RunInProcess(int rank, int rankSize)
    {
        // 1. acl init
        Args args;
        aclrtStream stream;
        ACLCHECK(aclInit(nullptr));
        ACLCHECK(aclrtSetDevice(rank));
        ACLCHECK(aclrtCreateStream(&stream));
    
        // 2. create HcclComm by rankFile
        char commName[128] = "";
        HcclComm hcclComm = nullptr;
        char *rankTableFile = getenv("RANK_TABLE_FILE");
    
        std::string rankTableFileStr(rankTableFile);
        std::cout << "rankTableFilePath is :" << rankTableFileStr << std::endl;
        int rank_id = rank + FIRST_RANK_ID;
        auto ret = HcclCommInitClusterInfo(rankTableFile, rank_id, &hcclComm);
        if (ret != HCCL_SUCCESS || hcclComm == nullptr) {
            std::cout << "HCCL CommInitClusterInfo ERROR" << ret << " should check rankTableFile config" << std::endl;
            return;
        }
        std::cout << "HcclCommInitClusterInfo success, rank_id:" << rank_id << ", rankSize:" << rankSize
                      << ", hcclComm:" << hcclComm;
        HcclGetCommName(hcclComm, commName);
        if (commName == "") { std::cout << "rankTableFile CommName should not be null" << std::endl;}
    
        // 3. launch one process for MoeDistributeCombine
        args.rankId = rank;
        args.groupEpName = commName;
        args.hcclEpComm = hcclComm;
        args.epRankId = rank_id;
        args.stream = stream;
        LOG_PRINT("[INFO] rank = %d, groupEpName = %s, stream = %p\n", args.rankId, commName, args.stream);
    
        int res = launchOneProcess_MoeDistributeCombine(args);
        if (res != ACL_SUCCESS) {
            std::cout << "run launchOneProcess_MoeDistributeCombine failed, ret = " << res << std::endl;
            return;
        }
    }
    
    int main(int argc, char *argv[])
    {
        char* env_rankID = getenv("FIRST_RANK_ID");
        if (!env_rankID) {
            std::cerr << "FIRST_RANK_ID环境变量未设置！\n";
            return 1;
        }
        FIRST_RANK_ID = std::stoi(std::string(env_rankID));
        std::cout << "FIRST_RANK_ID is: " << FIRST_RANK_ID << std::endl;
    
        // 所需的进程数量
        const int processCount = 8;
        pid_t pids[processCount];
    
        for (int i = 0; i < processCount; ++i) {
            pids[i] = fork();
            if (pids[i] < 0) {
                std::cout << "fork failed ! " << pids[i] << std::endl;
            } else if (pids[i] == 0) {
                // 子进程，完成任务后退出
                RunInProcess(i, processCount);
                exit(0);
            }
        }
    
        // 父进程等待所有子进程完成
        for (int i = 0; i < processCount; ++i) {
            waitpid(pids[i], NULL, 0);
        }
    
        return 0;
    }
    ```