声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# NsaSelectedAttentionInfer

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>|      √     |

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。


## 功能说明

-   **算子功能**：Native Sparse Attention推理过程中，Selected Attention的计算。

-   **计算公式**：

    self-attention（自注意力）利用输入样本自身的关系构建了一种注意力模型。其原理是假设有一个长度为$n$的输入样本序列$x$，$x$的每个元素都是一个$d$维向量，可以将每个$d$维向量看作一个token embedding，将这样一条序列经过3个权重矩阵变换得到3个维度为$n*d$的矩阵。

    Selected Attention的计算由topK索引取数与attention计算融合而成，外加paged attention取kvCache。首先，通过$topkIndices$索引从$key$中取出$key_{topk}$，从$value$中取出$value_{topk}$，计算self_attention公式如下：

    $$ 
      Attention(query,key,value)=Softmax(\frac{query · key_{topk}^T}{\sqrt{d}})value_{topk}
    $$
    其中$query$和$key_{topk}^T$的乘积代表输入$x$的注意力，为避免该值变得过大，通常除以$d$的开根号进行缩放，并对每行进行softmax归一化，与$value_{topk}$相乘后得到一个$n*d$的矩阵。


## 算子执行接口

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnNsaSelectedAttentionInferGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnNsaSelectedAttentionInfer”接口执行计算。

* `aclnnStatus aclnnNsaSelectedAttentionInferGetWorkspaceSize(const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *topkIndices, const aclTensor *attenMaskOptional, const aclTensor *blockTableOptional, const aclIntArray *actualQSeqLenOptional, const aclIntArray *actualKvSeqLenOptional, char *layoutOptional, int64_t numHeads, int64_t numKeyValueHeads, int64_t selectBlockSize, int64_t selectBlockCount, int64_t pageBlockSize, double scaleValue, int64_t sparseMode, aclTensor *output, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnNsaSelectedAttentionInfer(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnNsaSelectedAttentionInferGetWorkspaceSize

- **参数说明：**
  - query（aclTensor \*，计算输入）：Device侧的aclTensor，attention结构的Query输入，数据类型保持与key、value的数据类型一致。[数据格式](common/数据格式.md)支持ND，数据类型支持FLOAT16、BFLOAT16，支持输入的维度是3/4维，不支持[非连续的Tensor](common/非连续的Tensor.md)。  

  - key（aclTensor \*，计算输入）：Device侧的aclTensor，attention结构的Key输入，数据类型保持与query、value的数据类型一致。[数据格式](common/数据格式.md)支持ND，数据类型支持FLOAT16、BFLOAT16，支持输入的维度是3/4维，不支持[非连续的Tensor](common/非连续的Tensor.md)。

  - value（aclTensor \*，计算输入）：Device侧的aclTensor，attention结构的Value输入，数据类型保持与query、key的数据类型一致。[数据格式](common/数据格式.md)支持ND，数据类型支持FLOAT16、BFLOAT16，支持输入的维度是3/4维，不支持[非连续的Tensor](common/非连续的Tensor.md)。

  - topkIndices （aclTensor \*，计算输入）：Device侧的aclTensor， NSA里的topK索引，[数据格式](common/数据格式.md)支持ND，数据类型支持INT32，支持输入的维度是3/4维，不支持[非连续的Tensor](common/非连续的Tensor.md)。

  - attenMask（aclTensor \*，计算输入）：Device侧的aclTensor，可选参数，表示attention掩码矩阵，[数据格式](common/数据格式.md)支持ND，如不使用该功能时可传入nullptr。**预留参数，暂未使用**。 

  - blockTableOptional（aclTensor \*，计算输入）：Device侧的aclTensor，表示paged attention中KV存储使用的block映射表，数据类型支持INT32，支持输入的维度是2维，[数据格式](common/数据格式.md)支持ND，不支持[非连续的Tensor](common/非连续的Tensor.md)。
  
  - actualQSeqLenOptional（aclIntArray \*，计算输入）：Host侧的aclIntArray，表示query的S轴实际长度，[数据格式](common/数据格式.md)支持ND，数据类型支持INT64，如不使用该功能时可传入nullptr。

  - actualSelKvSeqLenOptional（aclIntArray \*，计算输入）：Host侧的aclIntArray，表示算子处理的key和value的S轴实际长度，[数据格式](common/数据格式.md)支持ND，数据类型支持INT64。
  
  - inputLayoutOptional（char \*，计算输入）：Host侧的字符指针，用于标识输入query、key、value的数据排布格式，当前支持BSH/BSND/TND，当不传入该参数时，默认为“BSND”，分别对应query、key、value 3/4维。

    **说明：** 
    query的数据排布格式中，B即Batch，S即Seq-Length，N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N。key和value的数据排布格式当前（paged attention）支持（blocknum, blocksize, H），（blocknum, blocksize, N, D），H（Head-Size）表示隐藏层的大小，H = N * D。
  
  - numHeads（int64\_t，计算输入 ）：Host侧的int64\_t，代表head个数，数据类型支持INT64。

  - numKeyValueHeads（int64\_t，计算输入 ）：Host侧的int64\_t，代表kvHead个数，数据类型支持INT64。

  - selectBlockSize（int64\_t，计算输入 ）：Host侧的int64\_t，代表select阶段的block大小，在计算importance score时使用，数据类型支持INT64。

  - selectBlockCount（int64\_t，计算输入 ）：Host侧的int64\_t，代表topK阶段需要保留的block数量，数据类型支持INT64。

  - pageBlockSize（int64\_t，计算输入 ）：Host侧的int64\_t，代表paged attention的block大小，在kv cache取数时使用，数据类型支持INT64。

  - scaleValue（double，计算输入）：Host侧的double，公式中d开根号的倒数，代表缩放系数，作为计算流中Muls的scalar值，数据类型支持DOUBLE。

  - sparseMode（int64\_t，计算输入）：Host侧的int64\_t，表示sparse的模式，控制有attentionMask输入时的稀疏计算。数据类型支持INT64，**预留参数，暂未使用**。

  - output（aclTensor \*，计算输出）：Device侧的aclTensor，attention的输出，[数据格式](common/数据格式.md)支持ND。数据类型支持FLOAT16、BFLOAT16。
  
  - workspaceSize（uint64\_t \*，出参）：返回用户需要在Device侧申请的workspace大小。

  - executor（aclOpExecutor \*\*，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：query、key、value、topkIndices、attenMask、blockTableOptional、actualQSeqLenOptional、actualSelKvSeqLenOptional、output的数据类型和数据格式不在支持的范围内。
  ```

### aclnnNsaSelectedAttentionInfer

-   **参数说明：**
    -   workspace（void\*，入参）：在Device侧申请的workspace内存地址。
    -   workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnNsaSelectedAttentionInferGetWorkspaceSize获取。
    -   executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
    -   stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明

- 参数query中的N和numHeads值相等，key、value的N和numKeyValueHeads值相等，并且numHeads是numKeyValueHeads的倍数关系。
- 参数query中的D和key的D(H/numKeyValueHeads)值相等，value的D(H/numKeyValueHeads)和output的D值相等。
- query，key，value输入，功能使用限制如下：
  -   支持B轴小于等于3072；
  -   支持key/value的N轴小于等于256；
  -   支持query的N轴与key/value的N轴（H/D）的比值（即GQA中的group大小）小于等于16；
  -   支持query与Key的D轴等于192；
  -   支持value的D轴等于128；
  -   支持Key与Value的blockSize等于64或128；
  -   普通场景下仅支持query的S轴等于1;
  -   多token推理场景下，仅支持query的S轴最大等于4，并且此时要求每个batch单独的actualQSeqLen <= actualSelKvSeqLen;
  -   仅支持paged attention。
  -   仅支持selectBlockSize取值为16的整数倍，最大支持到128。
  -   selectBlockCount上限满足selectBlockCount * selectBlockSize <= MaxKvSeqlen，MaxKvSeqlen = Max(actualSelKvSeqLenOptional)。

## 算子原型

```c++
REG_OP(NsaSelectedAttentionInfer)
    .INPUT(query, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(key, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(value, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(topk_indices, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(attentionMaskOptional, TensorType({DT_BOOL, DT_UINT8}))
    .OPTIONAL_INPUT(blockTableOptional, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(actualQSeqLenOptional, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actualKvSeqLenOptional, TensorType({DT_INT64}))
    .OUTPUT(output, TensorType({DT_FLOAT16, DT_BF16})) 
    .ATTR(layoutOptional, Str)   
    .ATTR(numHeads, Int, 1)  
    .ATTR(numKeyValueHeads, Int, 0)  
    .ATTR(selectBlockSize, Int, 0)
    .ATTR(selectBlockCount, Int, 0)
    .ATTR(pageBlockSize, Int, 0) 
    .ATTR(scaleValue, Float, 1.0)   
    .ATTR(sparseMode, Int, 0)  
    .OP_END_FACTORY_REG(NsaSelectedAttentionInfer)
```
参数解释请参见**算子执行接口**。

## 调用示例

- PyTorch框架调用

  如果通过PyTorch单算子方式调用该融合算子，则需要参考PyTorch融合算子[torch_npu.npu_nsa_select_attention_infer](https://hiascend.com/document/redirect/PyTorchAPI)；如果用户定制了该融合算子，则需要参考《Ascend C算子开发》手册[适配PyTorch框架](https://hiascend.com/document/redirect/CannCommunityAscendCInvorkOnNetwork)。

- aclnn单算子调用方式

  通过aclnn单算子调用示例代码如下（以<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>为例），仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

```c++
#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <sys/stat.h>
#include <cstring>
#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_nsa_select_attention_infer.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                            *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
    }
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
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}


int main(int argc, char **argv)
{
    // 1. （固定写法）device/context/stream初始化，参考AscendCL对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    // 如果需要修改shape值，需要同步修改../scripts/fa_generate_data.py中 test_nsa_selected_attention_infer 分支下生成
    // query、key、value对应的shape值，并重新gen data，再执行

    int64_t batch = 21;
    int sequenceLengthK = 97;
    aclIntArray * actualCmpKvSeqLen = nullptr;
    aclIntArray * actualCmpQSeqLen = nullptr;
    // 创建actualCmpKvSeqLen aclIntArray
    std::vector<int64_t> actualCmpKvSeqLenVector(batch, sequenceLengthK);
    actualCmpKvSeqLen = aclCreateIntArray(actualCmpKvSeqLenVector.data(), actualCmpKvSeqLenVector.size());
    // 创建actualCmpQSeqLen aclIntArray
    std::vector<int64_t> actualCmpQSeqLenVector(batch, 1);
    actualCmpQSeqLen = aclCreateIntArray(actualCmpQSeqLenVector.data(), actualCmpQSeqLenVector.size());
    int64_t d1 = 192;
    int64_t d2 = 128;
    int64_t g = 4;
    int64_t s1 = 1;
    int64_t n2 = 1;
    int64_t blockSize = 128;
    int64_t selectBlockSize = 64;
    int64_t selectBlockCount = 2;
    int64_t blockTableLength = 1;
    int64_t numBlocks = batch * blockTableLength;
    std::vector<int64_t> queryShape = {batch, s1, n2 * g, d1};
    std::vector<int64_t> keyShape = {numBlocks, blockSize, n2, d1};
    std::vector<int64_t> valueShape = {numBlocks, blockSize, n2, d2};
    std::vector<int64_t> topkIndicesShape = {batch, n2, selectBlockCount};
    std::vector<int64_t> blockTableOptionalShape = {batch, blockTableLength};
    std::vector<int64_t> outputShape = {batch, s1, n2 * g, d2};

    long long queryShapeSize = GetShapeSize(queryShape);
    long long keyShapeSize = GetShapeSize(keyShape);
    long long valueShapeSize = GetShapeSize(valueShape);
    long long blockTableOptionalShapeSize = GetShapeSize(blockTableOptionalShape);
    long long outputShapeSize = GetShapeSize(outputShape);
    long long topkIndicesShapeSize = GetShapeSize(topkIndicesShape);

    std::vector<int16_t> queryHostData(queryShapeSize, 1);
    std::vector<int16_t> keyHostData(keyShapeSize, 1);
    std::vector<int16_t> valueHostData(valueShapeSize, 1);
    std::vector<int32_t> blockTableOptionalHostData(blockTableOptionalShapeSize, 0);
    std::vector<int16_t> outputHostData(outputShapeSize, 1);
    
    std::vector<int32_t> topkIndicesHostData;
    for (int b = 0; b < batch; ++b) {
        for (int h = 0; h < n2; ++h) {
            for (int s = 0; s < selectBlockCount; ++s) {
                if (s == 0) {
                    topkIndicesHostData.push_back(s);
                } else {
                    topkIndicesHostData.push_back(-1);
                }
            }
        }
    }
    // attr
    double scaleValue = 1.0;
    int64_t sparseMod = 0;
    int64_t numHeads= static_cast<int64_t>(n2 * g);
    std::string sLayerOut = "BSND";
    char layOut[sLayerOut.length()];
    std::strcpy(layOut, sLayerOut.c_str());

    void *queryDeviceAddr = nullptr;
    void *keyDeviceAddr = nullptr;
    void *valueDeviceAddr = nullptr;
    void *blockTableOptionalDeviceAddr = nullptr;
    void *outputDeviceAddr = nullptr;
    void *topkIndicesDeviceAddr = nullptr;
    aclTensor *queryTensor = nullptr;
    aclTensor *keyTensor = nullptr;
    aclTensor *valueTensor = nullptr;
    aclTensor *blockTableOptionalTensor = nullptr;
    aclTensor *outputTensor = nullptr;
    aclTensor *topkIndicesTensor = nullptr;
    

    uint64_t workspaceSize = 0;
    void *workspaceAddr = nullptr;

    if (argv == nullptr || argv[0] == nullptr) {
        LOG_PRINT("Environment error, Argv=%p, Argv[0]=%p", argv, argv == nullptr ? nullptr : argv[0]);
        return 0;
    }
    // 创建query aclTensor
    ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed. ERROR: %d\n", ret); return ret);
    // 创建key aclTensor
    ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed. ERROR: %d\n", ret); return ret);
    // 创建value aclTensor
    ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed. ERROR: %d\n", ret); return ret);
    // 创建blockTableOptional aclTensor
    ret = CreateAclTensor(blockTableOptionalHostData, blockTableOptionalShape, &blockTableOptionalDeviceAddr, aclDataType::ACL_INT32, &blockTableOptionalTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed. ERROR: %d\n", ret); return ret);
    // 创建output aclTensor
    ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT16, &outputTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed. ERROR: %d\n", ret); return ret);
    // 创建topkIndices aclTensor
    ret = CreateAclTensor(topkIndicesHostData, topkIndicesShape, &topkIndicesDeviceAddr, aclDataType::ACL_INT32, &topkIndicesTensor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("failed. ERROR: %d\n", ret); return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    aclOpExecutor *executor;

    // 调用aclnnNsaSelectedAttention第一段接口
    ret = aclnnNsaSelectedAttentionInferGetWorkspaceSize(queryTensor, keyTensor, valueTensor, topkIndicesTensor, nullptr,
                blockTableOptionalTensor, actualCmpQSeqLen, actualCmpKvSeqLen, layOut,
                numHeads, n2, selectBlockSize, selectBlockCount, blockSize,
                scaleValue, sparseMod, outputTensor,
                &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaSelectedAttentionInfer allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnNsaSelectedAttention第二段接口
    ret = aclnnNsaSelectedAttentionInfer(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaSelectedAttentionInfer failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaSelectedAttentionInfer aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("aclnn execute success : %d\n", ret);
    
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outputShape);
    std::vector<op::fp16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputDeviceAddr,
                size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy [attn] result from device to host failed. ERROR: %d\n", ret); return ret);
    uint64_t printNum = 10;
    for (int64_t i = 0; i < printNum; i++) {
        std::cout << "index: " << i << ": " << static_cast<float>(resultData[i]) << std::endl;
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改; 释放device资源
    aclDestroyTensor(queryTensor);
    aclDestroyTensor(keyTensor);
    aclDestroyTensor(valueTensor);
    aclDestroyTensor(outputTensor);
    aclDestroyTensor(topkIndicesTensor);
    aclDestroyTensor(blockTableOptionalTensor);
    aclrtFree(queryDeviceAddr);
    aclrtFree(keyDeviceAddr);
    aclrtFree(valueDeviceAddr);
    aclrtFree(outputDeviceAddr);
    aclrtFree(topkIndicesDeviceAddr);
    aclrtFree(blockTableOptionalDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```