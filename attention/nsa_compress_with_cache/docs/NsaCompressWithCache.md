声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# NsaCompressWithCache

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品</term>|      √     |
|<term>Atlas 800I A2 推理产品</term>|      ×     |
|<term>A200I A2 Box 异构组件</term>|      ×     |

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 功能说明

- 算子功能：用于Native-Sparse-Attention推理阶段的KV压缩，每次推理每个batch会产生一个新的token，每当某个batch的token数量凑满一个compress\_block时，该算子会将该batch的后compress\_block个token压缩成一个compress\_token，算法流程如下：

1. 检查act\_seq\_lens是否有满足满足$s \ge compressBlockSize$ 且 $(s - compressBlockSize) \% stride ==0$的序列长度；
2. 找到满足序列长度的batchIdx，根据block\_table找到该batch的后compress\_block\_size个token压缩；
3. 执行压缩算法；
4. 根据slot\_mapping写回到output\_cache中。

- 计算公式

$$
compressIdx=(s-compressBlockSize)/stride\\ 
ouputCacheRef[slotMapping[i]] = input[compressIdx*stride : compressIdx*stride+compressBlockSize]*weight[:]
$$


## 实现原理

详细实现原理参考[NsaCompressWithCache算子设计介绍](./common/NsaCompressWithCache算子设计介绍.md)。

## 算子执行接口

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用 “aclnnNsaCompressWithCacheGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnNsaCompressWithCache”接口执行计算。

* `aclnnStatus aclnnNsaCompressWithCacheGetWorkspaceSize(const aclTensor *input, const aclTensor *weight, const aclTensor *slotMapping, const aclIntArray *actSeqLenOptional,const aclTensor *blockTableOptional, char *layoutOptional, int64_t compressBlockSize, int64_t compressStride, int64_t actSeqLenType, int64_t pageBlockSize, aclTensor *outputCache, uint64_t *workspaceSize, aclOpExecutor **executor);`
* `aclnnStatus aclnnNsaCompressWithCache(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnNsaCompressWithCacheGetWorkspaceSize

-   **参数说明**：
    -   input（aclTensor \*，计算输入）：Device侧的aclTensor, 表示待压缩张量。当传入blockTable时shape为[blockNum, pageBlockSize, N, D]，数据类型支持BFLOAT16、FLOAT16，[数据格式](common/数据格式.md)支持ND，支持[非连续的Tensor](common/非连续的Tensor.md)，不支持空Tensor。N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸。
    -   weight（aclTensor \*，计算输入）：Device侧的aclTensor，压缩的权重。shape支持[compressBlockSize, N]，weight与input的shape满足broadcast关系，数据类型与inpu保持一致，[数据格式](common/数据格式.md)支持ND，支持[非连续的Tensor](common/非连续的Tensor.md)，不支持空Tensor。N（Head-Num）表示多头数。
    -   slotMapping (aclTensor \*，计算输入)：Device侧的aclTensor，[数据格式](common/数据格式.md)支持ND，shape为[B,]，存储每个batch尾部压缩数据存储的位置的索引，数据类型支持INT32，不支持[非连续的Tensor](common/非连续的Tensor.md)，不支持空Tensor。B（Batch）表示输入样本批量大小。
    -   actSeqLenOptional（aclTensor \*，计算输入）：可选参数，Host侧的aclIntArray，数据类型支持INT64，[数据格式](common/数据格式.md)支持ND，描述了每个Batch对应的S大小。在TND排布场景下需要该输入，其余场景输入nullptr。S（Seq-Length）表示输入样本序列长度。
    -   blockTableOptional （aclTensor \*，计算输入）：可选参数，Device侧的aclTensor，数据类型支持INT32。[数据格式](common/数据格式.md)支持ND。表示PageAttention中KV存储使用的block映射表，如不使用该功能可传入nullptr。
    -   layoutOptional （char \*，计算输入）：可选参数，Host侧的string，数据类型支持String，代表输入input的数据排布格式，支持BSH、SBH、BSND、BNSD、TND。当前仅支持TND，当传入blockTableOptional时此参数无效，否则为必选参数。
        - 说明：数据排布格式支持从多种维度解读，其中T是B和S合轴紧密排列的数据（每个batch的actSeqLen）、B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Head-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N。
    -   compressBlockSize（int64_t，计算输入）：Host侧的int64_t，压缩滑窗大小。
    -   compressStride（int64_t，计算输入）：Host侧的int64_t，两次压缩滑窗间隔大小。
    -   actSeqLenType（int64_t，计算输入）：Host侧的int64_t，actSeqLenOptional有输入时生效，可取值0或1，0代表actSeqLenOptional中数值为前继batch的系列大小的cumsum结果（累积和），1代表actSeqLenOptional中数值为每个batch中序列大小，当前仅支持1。
    -   pageBlockSize（int64_t，计算输入）：Host侧的int64_t，指定page attention场景下page的blocksize大小。
    -   outputCache（aclTensor \*，计算输入输出）：Device侧的aclTensor，[数据格式](common/数据格式.md)支持ND，数据类型与input保持一致，不支持[非连续的Tensor](common/非连续的Tensor.md)，不支持空Tensor。
    -   workspaceSize（uint64\_t \*，出参）：返回用户需要在Device侧申请的workspace大小。
    -   executor（aclOpExecutor \*\*，出参）：返回op执行器，包含了算子计算流程。

-   **返回值**

    返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。
    ```
    第一段接口完成入参校验，出现以下场景时报错：
    161001(ACLNN_ERR_PARAM_NULLPTR)：1. 计算输入和必选计算输出是空指针
    161002(ACLNN_ERR_PARAM_INVALID)：1. 计算输入和输出的数据类型和格式不在支持的范围内
    561002(ACLNN_ERR_INNER_TILING_ERROR): 1. input和weight不满足broadcast关系，即input的第三维大小与weight的第二维大小不相等
                                          2. activeNum、expertNum、expertCapacity的值小于0
                                          3. compress_block_size、compress_stride 、不是16的整数倍，或者compress_block_size<compress_stride
                                          4. seq_lens_type!=1或者layout取值不是BSH、SBH、BSND、BNSD、TND中的一个
                                          5. page_block_size取值不是64或者128
                                          6. headDim未对齐16
    ```

### aclnnNsaCompressWithCache    

-   **参数说明：**
    -   workspace（void\*，入参）：在Device侧申请的workspace内存地址。
    -   workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnNsaCompressWithCacheGetWorkspaceSize获取。
    -   executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
    -   stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明

-   input和weight满足broadcast关系，input的第三维大小与weight的第二维大小相等
-   compressBlockSize、compressStride 必须是16的整数倍，且compressBlockSize>=compressStride，compressBlockSize <= 64, 
-   actSeqLenType目前仅支持取值1
-   layoutOptional取值可以是BSH、SBH、BSND、BNSD、TND，但是不会生效
-   pageBlockSize 只能是64或者128
-   headDim是16的整数倍，且headDim <= 256
-   不支持input/weight/outputCache为空输入
-   slotMapping的值无重复，否则会导致计算结果不稳定
-   blockTableOptional的值不超过blockNum，否则会发生越界
-   actSeqLenOptional的值不应该超过序列最大长度
-   headNum <= 64，且headNum>50时headNum%2=0, 

## 算子原型

```c++
REG_OP(NasCompressOpWithCache)
    .INPUT(kv_cache, TensorType({DT_BF16,DT_FP16}))
    .INPUT(weight, TensorType({DT_BF16,DT_FP16}))
    .INPUT(act_seq_lens, list)
    .INPUT(seq_lens_type, int)
    .INPUT(block_table, TensorType({DT_INT32}))
    .INPUT(slot_mapping, TensorType({DT_INT32}))
    .OUTPUT(compress_kv_cache, TensorType({DT_BF16,DT_FP16}))
    .ATTR(compress_block_size, int)
    .ATTR(compress_stride, int) 
    .ATTR(page_block_size, int) 
    .OP_END_FACTORY_REG(NasCompressWithCache)
```
参数解释请参见**算子执行接口**。

## 调用示例

aclnn单算子调用示例代码如下（以<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>为例），仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

```c++
#include "acl/acl.h"
#include "aclnnop/aclnn_nsa_compress_with_cache.h"
#include <iostream>
#include <vector>
#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)
#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)
int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}
int Init(int32_t deviceId, aclrtStream* stream) {
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
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
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}
int main() {
    // 输入shape相关参数设置
    constexpr int64_t compress_block_size = 32;
    constexpr int64_t compress_stride = 16;
    constexpr int64_t heads_num = 24;
    constexpr int64_t heads_dim = 192;
    constexpr int64_t batch_size = 4;
    constexpr int64_t page_block_size = 128;
    constexpr int64_t max_seq_len = 512;
    constexpr int64_t result_len = 512;
    constexpr int64_t block_num_per_batch = max_seq_len / page_block_size;
    constexpr int64_t blocks_num = block_num_per_batch * batch_size;
    // 1. 固定写法，device/stream初始化, 参考acl对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口定义构造
    std::vector<int64_t> inputShape = {blocks_num, page_block_size, heads_num, heads_dim};
    std::vector<int64_t> weightShape = {compress_block_size, heads_num};
    std::vector<int64_t> slotMappingShape = {batch_size};
    std::vector<int64_t> outputCacheRefShape = {result_len, heads_num, heads_dim};
    std::vector<int64_t> actSeqLenShape = {batch_size};
    std::vector<int64_t> blockTableShape = {batch_size, block_num_per_batch};

    void *inputDeviceAddr = nullptr;
    void *weightDeviceAddr = nullptr;
    void *slotMappingDeviceAddr = nullptr;
    void *outputCacheRefDeviceAddr = nullptr;
    void *actSeqLenDeviceAddr = nullptr;
    void *blockTableDeviceAddr = nullptr;

    aclTensor *input = nullptr;
    aclTensor *weight = nullptr;
    aclTensor *slotMapping = nullptr;
    aclTensor *outputCacheRef = nullptr;
    aclIntArray *actSeqLen = nullptr;
    aclTensor *blockTable = nullptr;

    std::vector<aclFloat16> inputHostData(inputShape[0] * inputShape[1] * inputShape[2] * inputShape[3],
                                          aclFloatToFloat16(1.0));
    std::vector<aclFloat16> weightHostData(weightShape[0] * weightShape[1], aclFloatToFloat16(1.0));
    std::vector<int32_t> slotMappingHostData(slotMappingShape[0], 0);
    std::vector<aclFloat16> outputCacheRefHostData(outputCacheRefShape[0] * outputCacheRefShape[1] *
                                                   outputCacheRefShape[2], aclFloatToFloat16(1.0));
    std::vector<int64_t> actSeqLenHostData(actSeqLenShape[0], 0);
    std::vector<int32_t> blockTableHostData(blockTableShape[0] * blockTableShape[1]);
    actSeqLenHostData[0]=32;
    // 创建self aclTensor
    ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT16, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(slotMappingHostData, slotMappingShape, &slotMappingDeviceAddr, aclDataType::ACL_INT32,
                          &slotMapping);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outputCacheRefHostData, outputCacheRefShape, &outputCacheRefDeviceAddr,
                          aclDataType::ACL_FLOAT16, &outputCacheRef);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    actSeqLen = aclCreateIntArray(actSeqLenHostData.data(), actSeqLenHostData.size());
    ret = CreateAclTensor(blockTableHostData, blockTableShape, &blockTableDeviceAddr, aclDataType::ACL_INT32,
                          &blockTable);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    char layout[4] = "TND";
    int64_t actSeqLenType = 1;
    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnNsaCompressWithCache第一段接口
    ret = aclnnNsaCompressWithCacheGetWorkspaceSize(input, weight, slotMapping, actSeqLen, blockTable, layout,
                                                    compress_block_size, compress_stride, actSeqLenType,
                                                    page_block_size, outputCacheRef, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompressWithCacheGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnNsaCompressWithCache第二段接口
    ret = aclnnNsaCompressWithCache(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompressWithCache failed. ERROR: %d\n", ret); return ret);
    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outputCacheRefShape);
    std::vector<aclFloat16> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(aclFloat16), outputCacheRefDeviceAddr,
                      size * sizeof(aclFloat16), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = heads_dim * heads_num - 16; i < heads_dim * heads_num + 16; i++) {
        printf("outputCache[%d]:%f\n", i, aclFloat16ToFloat(resultData[i]));
    }
    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(input);
    aclDestroyTensor(weight);
    aclDestroyTensor(slotMapping);
    aclDestroyTensor(outputCacheRef);
    aclDestroyIntArray(actSeqLen);
    aclDestroyTensor(blockTable);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(inputDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(slotMappingDeviceAddr);
    aclrtFree(outputCacheRefDeviceAddr);
    aclrtFree(blockTableDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```