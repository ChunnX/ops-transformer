# aclnnMoeInitRoutingV3

## 支持的产品型号

- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>。
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>。

## 功能说明

- 算子功能：MoE的routing计算，根据[aclnnMoeGatingTopKSoftmaxV2](../../moe_gating_top_k_softmax_v2/docs/aclnnMoeGatingTopKSoftmaxV2.md)的计算结果做routing处理，支持不量化和动态量化模式。本接口针对V2接口[aclnnMoeInitRoutingV2](../../moe_init_routing_v2/docs/aclnnMoeInitRoutingV2.md)做了如下功能变更，请根据实际情况选择合适的接口：

    1.增加动态量化功能，支持输出expendX的 int8动态量化输出

    2.增加参数activeExpertRangeOptional，支持筛选有效范围内的expertId

    3.删除属性expertTokensBeforeCapacityFlag、删除输出expertTokensBeforeCapacityOut (使用expertTokensCountOrCumsumOut进行输出)

- 计算公式：  

  1.对输入expertIdx做排序，得出排序后的结果sortedExpertIdx和对应的序号sortedRowIdx：

    $$
    sortedExpertIdx, sortedRowIdx=keyValueSort(expertIdx,rowIdx)
    $$

  2.以sortedRowIdx做位置映射得出expandedRowIdxOut：

    $$
    expandedRowIdxOut[sortedRowIdx[i]]=i
    $$

  3.在drop模式下，对sortedExpertIdx的每个专家统计直方图结果，得出expertTokensCountOrCumsumOutOptional：

    $$
    expertTokensCountOrCumsumOutOptional[i]=Histogram(sortedExpertIdx)
    $$

  4.计算quant结果：
    - 动态quant：
        - 若不输入scale：
            $$
            dynamicQuantScaleOutOptional = row\_max(abs(x)) / 127
            $$

            $$
            quantResult = round(x / dynamicQuantScaleOutOptional)
            $$
        - 若输入scale:
            $$
            dynamicQuantScaleOutOptional = row\_max(abs(x * scaleOptional)) / 127
            $$

            $$
            quantResult = round(x / dynamicQuantScaleOutOptional)
            $$
  
  5.对quantResult取前NUM\_ROWS个sortedRowIdx的对应位置的值，得出expandedXOut：

    $$
    expandedXOut[i]=quantResult[sortedRowIdx[i]\%NUM\_ROWS]
    $$

  6.expandedRowIdxOut的有效元素数量availableIdxNum计算方式为，expertIdx中activeExpertRangeOptional范围内的元素的个数
    $$
    availableIdxNum = |\{x\in expertIdx| expert\_start \le x<expert\_end \ \}|
    $$
  

## 函数原型

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用 “aclnnMoeInitRoutingV3GetWorkspaceSize”接口获取入参并计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeInitRoutingV3”接口执行计算。

* `aclnnStatus aclnnMoeInitRoutingV3GetWorkspaceSize(const aclTensor *x, const aclTensor *expertIdx, const aclTensor *scaleOptional, const aclTensor *offsetOptional, int64_t activeNum, int64_t expertCapacity, int64_t expertNum, int64_t dropPadMode, int64_t expertTokensNumType, bool expertTokensNumFlag, int64_t quantMode, const aclIntArray *activeExpertRangeOptional, int64_t rowIdxType, const aclTensor *expandedXOut, const aclTensor *expandedRowIdxOut, const aclTensor *expertTokensCountOrCumsumOut, const aclTensor *expandedScaleOut, uint64_t *workspaceSize, aclOpExecutor **executor)`

* `aclnnStatus aclnnMoeInitRoutingV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnMoeInitRoutingV3GetWorkspaceSize

-   **参数说明**：
    -   x（aclTensor\*，计算输入）：MOE的输入即token特征输入，要求为一个2D的Tensor，shape为(NUM\_ROWS, H)，H代表每个Token的长度，数据类型支持FLOAT16、BFLOAT16、FLOAT32、INT8，[数据格式](common/数据格式.md)要求为ND。   
    -   expertIdx （aclTensor\*，计算输入）：[aclnnMoeGatingTopKSoftmaxV2](../../moe_gating_top_k_softmax_v2/docs/aclnnMoeGatingTopKSoftmaxV2.md)的输出每一行特征对应的K个处理专家，要求是一个2D的shape (NUM\_ROWS, K)，且里面元素专家id不能超过专家数。数据类型支持INT32，[数据格式](common/数据格式.md)要求为ND。
    -   scaleOptional （aclTensor\*，计算输入）：表示用于计算quant结果的参数，数据类型支持FLOAT32，[数据格式](common/数据格式.md)要求为ND。如果不输入表示计算时不使用scale，且输出expandedScaleOut中的值未定义。
        -   非量化场景下，如果输入则要求为1D的Tensor，shape为(NUM_ROWS,)。
        -   动态quant场景下，如果输入则要求为2D的Tensor，shape为(expertEnd-expertStart, H)。
    -   offsetOptional（aclTensor\*，计算输入）：表示用于计算quant结果的偏移值。数据类型支持FLOAT32，[数据格式](common/数据格式.md)要求为ND。
        -   在非量化场景下不输入。
        -   动态quant场景下不输入。
    -   activeNum（int64\_t，计算输入）：表示总的最大处理row数，输出expandedXOut只有这么多行是有效的，当前入参校验需大于等于0。当前未使用，校验需等于NUM_ROWS*K。
    -   expertCapacity（int64\_t， 计算输入）：表示每个专家能够处理的tokens数，取值范围大于等于0。当前未使用，仅校验非空。
    -   expertNum（int64\_t， 计算输入）：表示专家数，expertTokensNumType为key\_value模式时，取值范围为[0, 5120], 其它模式取值范围[0, 10240]。
    -   dropPadMode（int64\_t， 计算输入）：表示是否为DropPad场景，取值为0和1（当前仅支持0）。
        - 0：表示Dropless场景，该场景下不校验expertCapacity。
        - 1：表示DropPad场景。
    -   expertTokensNumType（int64\_t， 计算输入）：取值为0、1和2 （当前仅支持1和2）。
        - 0：表示comsum模式。
        - 1：表示count模式，即输出的值为各个专家处理的token数量的累计值。
        - 2：表示key\_value模式，即输出的值为专家和对应专家处理token数量的累计值。
    -   expertTokensNumFlag（bool，计算输入）：取值为false和true（当前仅支持true）。
        - false：表示不输出expertTokensCountOrCumsumOut。
        - true：表示输出expertTokensCountOrCumsumOut。
    -   quantMode（int64\_t， 计算输入）：取值为0、1、-1（当前仅支持1和-1）。
        - 0：表示静态quant场景。
        - 1：表示动态quant场景。
        - -1：表示不量化场景。
    -   activeExpertRangeOptional（aclIntArray *， 计算输入）, 长度为2，数组内的值为[expertStart, expertEnd], 表示活跃的expert范围在expertStart和expertEnd之间，左闭右开。要求值大于等于0，并且expertEnd不大于expertNum。
    -   rowIdxType（int64\_t， 计算输入）：表示expandedRowIdxOut使用的索引类型，取值为0、1。（性能模板仅支持1）
        - 0：表示gather类型的索引。
        - 1：表示scatter类型的索引。
    -   expandedXOut（aclTensor\*，计算输出）：根据expertIdx进行扩展过的特征，要求为2D的Tensor，shape为(NUM\_ROWS \* K, H)。非量化场景下数据类型同x，量化场景下数据类型支持INT8，[数据格式](common/数据格式.md)要求为ND。前availableIdxNum\*H个元素为有效数据，其余为无效数据。量化场景下，当x的数据类型为INT8时，输出值未定义。
    -   expandedRowIdxOut（aclTensor\*，计算输出）：expandedXOut和x的索引映射关系， 要求是一个1D的Tensor，Shape为(NUM\_ROWS\*K, )，数据类型支持INT32，[数据格式](common/数据格式.md)要求为ND。前availableIdxNum\*H个元素为有效数据，其余无效数据由rowIdxType决定。当rowIdxType为0时，无效数据由-1填充；当rowIdxType为1时，无效数据未初始化。
    -   expertTokensCountOrCumsumOut（aclTensor\*，计算输出）：在expertTokensNumType为1的场景下，要求是1D的Tensor，表示activeExpertRangeOptional范围内expert对应的处理token的总数。shape为(expertEnd-expertStart, )；在expertTokensNumType为2的场景下，要求是2D的Tensor，shape为(expertNum,  2)，表示activeExpertRangeOptional范围内token总数为非0的expert，以及对应expert处理token的总数；expert id在activeExpertRangeOptional范围内且剔除对应expert处理token为0的元素对为有效元素对，存放于Tensor头部并保持原序。数据类型支持INT64，[数据格式](common/数据格式.md)要求为ND。
    -   expandedScaleOut（aclTensor\*，计算输出）：数据类型支持FLOAT32，[数据格式](common/数据格式.md)要求为ND。
        -   非量化场景下，shape为(NUM\_ROWS\*K,)。当scaleOptional未输入时，输出值未定义。当scaleOptional输入时，输出表示一个1D的Tensor；当scaleOptional输入时，输出表示一个1D的Tensor，前availableIdxNum\*H个元素为有效数据，其余为无效数据。
        -   动态quant场景下，输出量化计算过程中scaleOptional的中间值，shape为(NUM\_ROWS \*K)。当scaleOptional未输入时，输出值未定义；当scaleOptional输入时，输出表示一个1D的Tensor，前availableIdxNum个元素为有效数据，其余为无效数据，若x的数据类型为INT8，输出值未定义。。
    -   workspaceSize（uint64\_t\*，出参）：返回用户需要在Device侧申请的workspace大小。
    -   executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

-   **返回值**

    返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。
    ```
    第一段接口完成入参校验，出现以下场景时报错：
    161001(ACLNN_ERR_PARAM_NULLPTR)：1. 计算输入和必选计算输出是空指针
    161002(ACLNN_ERR_PARAM_INVALID)：1. 计算输入和输出的数据类型和格式不在支持的范围内
    561002(ACLNN_ERR_INNER_TILING_ERROR): 1. 输入、输出Tensor的shape不在支持的范围内
                                          2. 输入的属性不在支持的取值范围内
    ```

## aclnnMoeInitRoutingV3

-   **参数说明：**
    -   workspace（void\*，入参）：在Device侧申请的workspace内存地址。
    -   workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnMoeInitRoutingV3GetWorkspaceSize获取。
    -   executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
    -   stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明

该算子支持两种性能模板，进入两种性能模板需要分别额外满足以下条件，不满足条件则进入通用模板：

- 进入低时延性能模板需要同时满足以下条件：
    - x、expertIdx、scaleOptional输入Shape要求分别为：(1, 7168)、(1, 8)、(256, 7168)
    - x数据类型要求：BFLOAT16
    - 属性要求：activeExpertRangeOptional=[0,256]、 quantMode=1、expertTokensNumType=2、expertNum=256

- 进入大batch性能模板需要同时满足以下条件：
    - NUM_ROWS范围为[384, 8192]
    - K=8
    - expertNum=256
    - expertEnd-expertStart<=32
    - quantMode=-1
    - rowIdxType=1
    - expertTokensNumType=1

此外，针对该算子部分性能用例相较于MoeInitRoutingV2发生回退的情况，在aclnn l2接口中将特定性能用例分流至MoeInitRoutingV2。

- 进入调用MoeInitRoutingV2算子接口的案例约束如下：

    - expertNum=128
    - activeExpertRangeOptional取值范围为[0, 128]
    - H=2048
    - quantMode=-1

    
## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_moe_init_routing_v3.h"
#include <iostream>
#include <vector>
#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)
#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)
int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}
int Init(int32_t deviceId, aclrtStream *stream)
{
    // 固定写法，AscendCL初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}
template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(),
        shape.size(),
        dataType,
        strides.data(),
        0,
        aclFormat::ACL_FORMAT_ND,
        shape.data(),
        shape.size(),
        *deviceAddr);
    return 0;
}
int main()
{
    // 1. 固定写法，device/stream初始化, 参考acl对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口定义构造
    std::vector<int64_t> xShape = {3, 2};
    std::vector<int64_t> expertIdxShape = {3, 4};
    std::vector<int64_t> scaleShape = {3};
    std::vector<int64_t> offsetShape = {1};

    std::vector<int64_t> expandedXOutShape = {12, 2};
    std::vector<int64_t> expandedRowIdxOutShape = {12};
    std::vector<int64_t> expertTokensCountOrCumsumOutOptionalShape = {4};
    std::vector<int64_t> expandedScaleOutOptionalShape = {12};

    std::vector<int64_t> activeExpertRangeArray = {0, 4};

    void *xDeviceAddr = nullptr;
    void *expertIdxDeviceAddr = nullptr;
    void *scaleDeviceAddr = nullptr;
    void *offsetDeviceAddr = nullptr;

    void *expandedXOutDeviceAddr = nullptr;
    void *expandedRowIdxOutDeviceAddr = nullptr;
    void *expertTokensCountOrCumsumOutOptionalDeviceAddr = nullptr;
    void *expandedScaleOutOptionalDeviceAddr = nullptr;

    aclTensor *x = nullptr;
    aclTensor *expertIdx = nullptr;
    aclTensor *scale = nullptr;
    aclTensor *offset = nullptr;

    int64_t activeNum = 12;
    int64_t expertCapacity = 4;
    int64_t expertNum = 256;
    int64_t dropPadMode = 0;
    int64_t expertTokensNumType = 1;
    bool expertTokensNumFlag = true;
    int64_t quantMode = -1;
    aclIntArray *activeExpertRange = aclCreateIntArray(activeExpertRangeArray.data(), activeExpertRangeArray.size());
    int64_t rowIdxType = 1;

    aclTensor *expandedXOut = nullptr;
    aclTensor *expandedRowIdxOut = nullptr;
    aclTensor *expertTokensCountOrCumsumOutOptional = nullptr;
    aclTensor *expandedScaleOutOptional = nullptr;

    std::vector<float> xHostData = {0.1, 0.1, 0.2, 0.2, 0.3, 0.3};
    std::vector<int> expertIdxHostData = {1, 2, 0, 3, 0, 2, 1, 3, 0, 1, 3, 2};
    std::vector<float> scaleHostData = {0.3423, 0.1652, 0.2652};
    std::vector<float> offsetHostData = {1.8369};

    std::vector<int8_t> expandedXOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int> expandedRowIdxOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int64_t> expertTokensCountOrCumsumOutOptionalHostData = {0, 0, 0, 0};
    std::vector<float> expandedScaleOutOptionalHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // 创建self aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertIdxHostData, expertIdxShape, &expertIdxDeviceAddr, aclDataType::ACL_INT32, &expertIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(offsetHostData, scaleShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(
        expandedXOutHostData, expandedXOutShape, &expandedXOutDeviceAddr, aclDataType::ACL_INT8, &expandedXOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedRowIdxOutHostData,
        expandedRowIdxOutShape,
        &expandedRowIdxOutDeviceAddr,
        aclDataType::ACL_INT32,
        &expandedRowIdxOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertTokensCountOrCumsumOutOptionalHostData,
        expertTokensCountOrCumsumOutOptionalShape,
        &expertTokensCountOrCumsumOutOptionalDeviceAddr,
        aclDataType::ACL_INT64,
        &expertTokensCountOrCumsumOutOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expandedScaleOutOptionalHostData,
        expandedScaleOutOptionalShape,
        &expandedScaleOutOptionalDeviceAddr,
        aclDataType::ACL_FLOAT,
        &expandedScaleOutOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // 调用aclnnMoeInitRoutingV3第一段接口
    ret = aclnnMoeInitRoutingV3GetWorkspaceSize(x,
        expertIdx,
        scale,
        offset,
        activeNum,
        expertCapacity,
        expertNum,
        dropPadMode,
        expertTokensNumType,
        expertTokensNumFlag,
        quantMode,
        activeExpertRange,
        rowIdxType,
        expandedXOut,
        expandedRowIdxOut,
        expertTokensCountOrCumsumOutOptional,
        expandedScaleOutOptional,
        &workspaceSize,
        &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingV3GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnMoeInitRoutingV3第二段接口
    ret = aclnnMoeInitRoutingV3(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMoeInitRoutingV3 failed. ERROR: %d\n", ret); return ret);
    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto expandedXSize = GetShapeSize(expandedXOutShape);
    std::vector<int8_t> expandedXData(expandedXSize, 0);
    ret = aclrtMemcpy(expandedXData.data(),
        expandedXData.size() * sizeof(expandedXData[0]),
        expandedXOutDeviceAddr,
        expandedXSize * sizeof(int8_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedXSize; i++) {
        LOG_PRINT("expandedXData[%ld] is: %d\n", i, expandedXData[i]);
    }
    auto expandedRowIdxSize = GetShapeSize(expandedRowIdxOutShape);
    std::vector<int> expandedRowIdxData(expandedRowIdxSize, 0);
    ret = aclrtMemcpy(expandedRowIdxData.data(),
        expandedRowIdxData.size() * sizeof(expandedRowIdxData[0]),
        expandedRowIdxOutDeviceAddr,
        expandedRowIdxSize * sizeof(int32_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expandedRowIdxSize; i++) {
        LOG_PRINT("expandedRowIdxData[%ld] is: %d\n", i, expandedRowIdxData[i]);
    }
    auto expertTokensBeforeCapacitySize = GetShapeSize(expertTokensCountOrCumsumOutOptionalShape);
    std::vector<int> expertTokenIdxData(expertTokensBeforeCapacitySize, 0);
    ret = aclrtMemcpy(expertTokenIdxData.data(),
        expertTokenIdxData.size() * sizeof(expertTokenIdxData[0]),
        expertTokensCountOrCumsumOutOptionalDeviceAddr,
        expertTokensBeforeCapacitySize * sizeof(int32_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < expertTokensBeforeCapacitySize; i++) {
        LOG_PRINT("expertTokenIdxData[%ld] is: %d\n", i, expertTokenIdxData[i]);
    }

    auto dynamicQuantScaleSize = GetShapeSize(expandedScaleOutOptionalShape);
    std::vector<float> dynamicQuantScaleData(dynamicQuantScaleSize, 0);
    ret = aclrtMemcpy(dynamicQuantScaleData.data(),
        dynamicQuantScaleData.size() * sizeof(dynamicQuantScaleData[0]),
        expandedScaleOutOptionalDeviceAddr,
        dynamicQuantScaleSize * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < dynamicQuantScaleSize; i++) {
        LOG_PRINT("dynamicQuantScaleData[%ld] is: %f\n", i, dynamicQuantScaleData[i]);
    }
    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(expertIdx);
    aclDestroyTensor(scale);
    aclDestroyTensor(offset);
    aclDestroyTensor(expandedXOut);
    aclDestroyTensor(expandedRowIdxOut);
    aclDestroyTensor(expertTokensCountOrCumsumOutOptional);
    aclDestroyTensor(expandedScaleOutOptional);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(xDeviceAddr);
    aclrtFree(expertIdxDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    aclrtFree(offsetDeviceAddr);
    aclrtFree(expandedXOutDeviceAddr);
    aclrtFree(expandedRowIdxOutDeviceAddr);
    aclrtFree(expertTokensCountOrCumsumOutOptionalDeviceAddr);
    aclrtFree(expandedScaleOutOptionalDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```