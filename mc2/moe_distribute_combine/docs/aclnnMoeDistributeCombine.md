# aclnnMoeDistributeCombine

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

算子功能：当存在TP域通信时，先进行ReduceScatterV通信，再进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）；当不存在TP域通信时，进行AlltoAllV通信，最后将接收的数据整合（乘权重再相加）。

注意该接口必须与`aclnnMoeDistributeDispatch`配套使用，相当于按`MoeDistributeDispatch`算子收集数据的路径原路返还。

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
    const char* groupEp,
    int64_t epWorldSize,
    int64_t epRankId,
    int64_t moeExpertNum,
    const char* groupTp,
    int64_t tpWorldSize,
    int64_t tpRankId,
    int64_t expertShardType,
    int64_t sharedExpertNum,
    int64_t sharedExpertRankNum,
    int64_t globalBs,
    int64_t outDtype,
    int64_t commQuantMode,
    int64_t groupListType,
    aclTensor* x,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
```

```cpp
aclnnStatus aclnnMoeDistributeCombine(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
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
   <td>根据expertIds进行扩展过的token特征，Device侧的aclTensor，要求为2D Tensor，shape为 \(max(tpWorldSize, 1) * A , H\)；支持非连续的Tensor。<br><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：不支持共享专家场景。</td>
   <td>FLOAT16、BFLOAT16</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertIds</td>
   <td>输入</td>
   <td>每个token的topK个专家索引，Device侧的aclTensor，要求为2D Tensor，shape为 \(BS, K\)；支持非连续的Tensor。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expandIdx</td>
   <td>输入</td>
   <td>对应aclnnMoeDistributeDispatch中的expandIdx输出，Device侧的aclTensor，要求为1D Tensor，shape为 \(BS*K, \)；支持非连续的Tensor。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>epSendCounts</td>
   <td>输入</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertScales</td>
   <td>输入</td>
   <td>每个token的topK个专家的权重，Device侧的aclTensor，要求为2D Tensor，shape为 \(BS, K\)；支持非连续的Tensor。</td>
   <td>FLOAT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tpSendCounts</td>
   <td>输入</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>xActiveMask</td>
   <td>输入</td>
   <td>-</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>activationScale</td>
   <td>输入</td>
   <td>-</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>weightScale</td>
   <td>输入</td>
   <td>-</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>groupList</td>
   <td>输入</td>
   <td>-</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expandScales</td>
   <td>输入</td>
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
   <td>MoE专家数量，取值范围(0, 512]，且满足moeExpertNum % (epWorldSize - sharedExpertRankNum) = 0。<br><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：还需满足moeExpertNum / (epWorldSize - sharedExpertRankNum) <= 24。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>groupTp</td>
   <td>输入</td>
   <td>STRING</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tpWorldSize</td>
   <td>输入</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>tpRankId</td>
   <td>输入</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>expertShardType</td>
   <td>输入</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sharedExpertNum</td>
   <td>输入</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>sharedExpertRankNum</td>
   <td>输入</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>globalBs</td>
   <td>输入</td>
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
   <td>表示处理后的token，Device侧的aclTensor，要求为2D Tensor，shape为 \(BS, H\)；数据类型、数据格式与expandX保持一致。</td>
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

1. `aclnnMoeDistributeDispatch`接口与`aclnnMoeDistributeCombine`接口必须配套使用，具体参考[调用示例](#调用示例)。

2. 在不同产品型号、不同通信算法或不同版本中，`aclnnMoeDistributeDispatch`的Tensor输出`expandIdx`、`epRecvCounts`、`tpRecvCounts`、`expandScales`中的元素值可能不同，使用时直接将上述Tensor传给`aclnnMoeDistributeCombine`对应参数即可，模型其他业务逻辑不应对其存在依赖。

3. 调用接口过程中使用的`groupEp`、`epWorldSize`、`moeExpertNum`、`groupTp`、`tpWorldSize`、`expertShardType`、`sharedExpertNum`、`sharedExpertRankNum`、`globalBs`参数取值所有卡需保持一致，网络中不同层中也需保持一致，且和`aclnnMoeDistributeDispatch`对应参数也保持一致。

4. <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明里的“本卡”均表示单DIE。

5. 参数说明里shape格式说明：
    - **A**：表示本卡需要分发的最大token数量，取值范围如下：
      - 对于共享专家，需满足 \(A = BS * epWorldSize * sharedExpertNum / sharedExpertRankNum\)。
      - 对于MoE专家，当`globalBs`为0时，需满足 \(A >= BS * epWorldSize * min(localExpertNum, K)\)；当`globalBs`非0时，需满足 \(A >= globalBs * min(localExpertNum, K)\)。
    - **H**：表示hidden size（隐藏层大小）：
      - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：取值范围(0, 7168]，且需为32的整数倍。
      - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值为7168。
    - **BS**：表示batch sequence size（本卡最终输出的token数量）：
      - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：取值范围为 \(0 < BS ≤ 256\)。
      - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：取值范围为 \(0 < BS ≤ 512\)。
    - **K**：表示选取topK个专家，需满足 \(0 < K ≤ moeExpertNum\)：
      - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：取值范围为 \(0 < K ≤ 16\)。
    - **serverNum**：表示服务器的节点数，取值仅支持2、4、8。
    - **localExpertNum**：表示本卡专家数量：
      - 对于共享专家卡，\(localExpertNum = 1\)。
      - 对于MoE专家卡，\(localExpertNum = moeExpertNum / (epWorldSize - sharedExpertRankNum)\)；当\(localExpertNum > 1\)时，不支持TP域通信。

6. **HCCL_BUFFSIZE**：
   调用本接口前需检查`HCCL_BUFFSIZE`环境变量取值是否合理，该环境变量表示单个通信域占用内存大小，单位MB，不配置时默认为200MB：
   - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：要求 \(≥ 2 * (BS * epWorldSize * min(localExpertNum, K) * H * sizeof(uint16) + 2MB)\)。
   - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：要求 \(≥ 2\) 且满足 \(1024^2 * (HCCL_BUFFSIZE - 2) / 2 ≥ BS * 2 * (H + 128) * (epWorldSize * localExpertNum + K + 1)\)，其中`localExpertNum`需使用MoE专家卡的本卡专家数。

7. **HCCL_INTRA_PCIE_ENABLE和HCCL_INTRA_ROCE_ENABLE**：
   <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：设置环境变量`HCCL_INTRA_PCIE_ENABLE = 1`和`HCCL_INTRA_ROCE_ENABLE = 0`可减少跨机通信数据量，可能提升算子性能。此时，`HCCL_BUFFSIZE`要求 \(≥ moeExpertNum * BS * (H * sizeof(dtypeX) + 4 * ((K + 7) / 8 * 8) * sizeof(uint32)) + 4MB + 100MB\)；且对于入参`moeExpertNum`，仅要求 \(moeExpertNum \% (epWorldSize - sharedExpertRankNum) = 0\)，不要求 \(moeExpertNum / (epWorldSize - sharedExpertRankNum) ≤ 24\)。

8. 本文公式中的“/”表示整除。

9. 通信域使用约束：
   - 一个模型中的`aclnnMoeDistributeCombine`和`aclnnMoeDistributeDispatch`仅支持相同EP通信域，且该通信域中不允许有其他算子。
   - 一个模型中的`aclnnMoeDistributeCombine`和`aclnnMoeDistributeDispatch`仅支持相同TP通信域或都不支持TP通信域；有TP通信域时，该通信域中不允许有其他算子。



## 调用示例

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：示例代码请参考aclnnMoeDistributeCombineV2接口调用过程，仅供参考，请根据实际情况配置；

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
        int64_t groupListType = 0;
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
                                                         groupListType, x,
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