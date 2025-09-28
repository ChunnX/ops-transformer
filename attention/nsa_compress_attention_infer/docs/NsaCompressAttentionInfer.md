声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。

# NsaCompressAttentionInfer

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>|      √     |

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)。


## 功能说明

-   **算子功能**：Native Sparse Attention推理过程中，Compress Attention的计算。

-   **计算公式**：

    self-attention（自注意力）利用输入样本自身的关系构建了一种注意力模型。其原理是假设有一个长度为$n$的输入样本序列$x$，$x$的每个元素都是一个$d$维向量，可以将每个$d$维向量看作一个token embedding，将这样一条序列经过3个权重矩阵变换得到3个维度为$n*d$的矩阵。

    Compress Attention的计算由三阶段的attention计算、importance score与topK三个过程融合而成，首先，$query$和$key^T$的乘积进行softmax得到注意力分数$P_{cmp}$

    $$ 
    P_{cmp}= Softmax(scale * query · key^T) 
    $$

    一方面，注意力分数$P_{cmp}$与$value$相乘得到自注意力的结果
    $$ 
    attentionOut = P_{cmp} · value 
    $$

    另一方面，注意力分数$P_{cmp}$被用于计算selection block的重要性分数$P_{slc}[j]$
    $$ 
    P_{slc}[j] = \sum\limits_{m=0}^{l'/d -1} \sum\limits_{n = 0}^{l/d -1} P_{cmp} [l'/d * j -m - n]
    $$

    接着，重要性分数在group内进行累加，得到共享的重要性分数$P_{slc'}$
    $$
    P_{slc'} = \sum\limits_{h=1}^{H}  P_{slc} ^h
    $$

    最后，选出重要性分数最高的K个selection block，其下标数组$topkIndices$为topK的输出
    $$
    topkIndices = topk(P_{slc'})
    $$


## 算子执行接口

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnNsaCompressAttentionInferGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnNsaCompressAttentionInfer”接口执行计算。

* `aclnnStatus aclnnNsaCompressAttentionInferGetWorkspaceSize(const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *attentionMaskOptional, const aclTensor *blockTableOptional, const aclIntArray *actualQSeqLenOptional, const aclIntArray *actualCmpKvSeqLenOptional, const aclIntArray *actualSelKvSeqLenOptional, const aclTensor *topKMaskOptional, int64_t numHeads, int64_t numKeyValueHeads, int64_t selectBlockSize, int64_t selectBlockCount, int64_t compressBlockSize, int64_t compressBlockStride, double scaleValue, char *layoutOptional, int64_t pageBlockSize, int64_t sparseMode, const aclTensor *output, const aclTensor *topKOutput, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnNsaCompressAttentionInfer(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnNsaCompressAttentionInferGetWorkspaceSize

- **参数说明：**
  - query（aclTensor \*，计算输入）：Device侧的aclTensor，attention结构的Query输入，[数据格式](common/数据格式.md)支持ND，数据类型支持FLOAT16、BFLOAT16，支持输入的维度是3维和4维，不支持[非连续的Tensor](common/非连续的Tensor.md)。  

  - key（aclTensor \*，计算输入）：Device侧的aclTensor，attention结构的Key输入，[数据格式](common/数据格式.md)支持ND，数据类型支持FLOAT16、BFLOAT16，支持输入的维度是3维，不支持[非连续的Tensor](common/非连续的Tensor.md)。

  - value（aclTensor \*，计算输入）：Device侧的aclTensor，attention结构的Value输入，[数据格式](common/数据格式.md)支持ND，数据类型支持FLOAT16、BFLOAT16，支持输入的维度是3维，不支持[非连续的Tensor](common/非连续的Tensor.md)。
  
  - attentionMaskOptional（aclTensor \*，计算输入）：Device侧的aclTensor，可选参数，表示attention掩码矩阵，[数据格式](common/数据格式.md)支持ND，如不使用该功能时可传入nullptr。attention掩码矩阵仅在Q_S大于1的情况下生效。

  - blockTableOptional（aclTensor \*，计算输入）：Device侧的aclTensor，可选参数，表示paged attention中KV存储使用的block映射表，数据类型支持INT32，[数据格式](common/数据格式.md)支持ND，不支持[非连续的Tensor](common/非连续的Tensor.md)，当前算子只支持paged attention，该参数必须传入。
  
  - actualQSeqLenOptional（aclIntArray \*，计算输入）：Host侧的aclIntArray，可选参数，表示query的S轴实际长度，[数据格式](common/数据格式.md)支持ND，数据类型支持INT64，长度必须为B，如不使用该功能时可传入nullptr。

  - actualCmpKvSeqLenOptional（aclIntArray \*，计算输入）：Host侧的aclIntArray，可选参数，表示经过压缩后的key和value的S轴实际长度，也即该算子处理的key和value的S轴实际长度，[数据格式](common/数据格式.md)支持ND，数据类型支持INT64，由于该算子当前只支持paged attention，因此该参数必须传入。
  
  - actualSelKvSeqLenOptional（aclIntArray \*，计算输入）：Host侧的aclIntArray，可选参数，表示压缩前的key和value的S轴实际长度，[数据格式](common/数据格式.md)支持ND，数据类型支持INT64，如不使用该功能时可传入nullptr。actualSelKvSeqLenOptional仅在Q_S大于1的情况下生效，且在Q_S大于1的情况下必须传入，且长度必须为B。

  - topKMaskOptional（aclTensor \*，计算输入）：Device侧的aclTensor，可选参数，表示topK计算中的掩码矩阵，[数据格式](common/数据格式.md)支持ND，如不使用该功能时可传入nullptr。**预留参数，暂未使用**。
  
  - numHeads（int64\_t，计算输入 ）：Host侧的int64\_t，代表head个数，数据类型支持INT64。

  - numKeyValueHeads（int64\_t，计算输入 ）：Host侧的int64\_t，代表kvHead个数，数据类型支持INT64。

  - selectBlockSize（int64\_t，计算输入 ）：Host侧的int64\_t，代表select阶段的block大小，在计算importance score时使用，数据类型支持INT64。

  - selectBlockCount（int64\_t，计算输入 ）：Host侧的int64\_t，代表topK阶段需要保留的block数量，数据类型支持INT64。

  - compressBlockSize（int64\_t，计算输入 ）：Host侧的int64\_t，代表压缩时的滑窗大小，数据类型支持INT64。

  - compressBlockStride（int64\_t，计算输入 ）：Host侧的int64\_t，代表两次压缩间的滑窗间隔大小，数据类型支持INT64。

  - scaleValue（double，计算输入）：Host侧的double，公式中d开根号的倒数，代表缩放系数，作为计算流中Muls的scalar值，数据类型支持DOUBLE。

  - layoutOptional（char \*，计算输入）：Host侧的字符指针，用于标识输入query、key、value的数据排布格式，当前支持取值“TND”和“BSND”。

    **说明：** 
    query的数据排布格式当前支持（B, N1, D1）和（B, S1, N1, D1）, B（Batch）表示Batch数，N1（Head-Num）表示numHeads，D1表示headSizeQK，S1表示qSeqLen。
    key的数据排布格式当前（paged attention）支持（blocknum, blocksize, N2 * D1），N2（Head-Num）表示numKeyValueHeads, D1表示headSizeQK。
    value的数据排布格式当前（paged attention）支持（blocknum, blocksize, N2 * D2），N2（Head-Num）表示numKeyValueHeads, D2表示headSizeVO。
    blockTableOptional的数据排布格式当前支持（B, maxBlockPerQuery），B（Batch）表示Batch数，maxBlockPerQuery表示最长的keyValueSeqLen需要占用的block数量。

  - pageBlockSize（int64\_t，计算输入 ）：Host侧的int64\_t，代表blockTable中一个block的大小，数据类型支持INT64。

  - sparseMode（int64\_t，计算输入）：Host侧的int64\_t，表示sparse的模式，控制有attentionMask输入时的稀疏计算。数据类型支持INT64，**预留参数，暂未使用**。

  - output（aclTensor \*，计算输出）：Device侧的aclTensor，attention的输出，[数据格式](common/数据格式.md)支持ND。数据类型支持FLOAT16、BFLOAT16。

  - topKOutput（aclTensor \*，计算输出）：Device侧的aclTensor，topK的输出，[数据格式](common/数据格式.md)支持ND，数据类型支持INT32。
  
  - workspaceSize（uint64\_t \*，出参）：返回用户需要在Device侧申请的workspace大小。

  - executor（aclOpExecutor \*\*，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：query、key、value、blockTableOptional、attentionOut、topKOutput的数据类型不在支持的范围内。
  - 返回361001（ACLNN_ERR_RUNTIME_ERROR）：API内存调用npu runtime的接口异常。
  ```

### aclnnNsaCompressAttentionInfer

-   **参数说明：**
    -   workspace（void\*，入参）：在Device侧申请的workspace内存地址。
    -   workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnIncreFlashAttentionV2GetWorkspaceSize获取。
    -   executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
    -   stream（aclrtStream，入参）：指定执行任务的AscendCL stream流。

-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明

- 参数query中的N和numHeads值相等，key、value的N和numKeyValueHeads值相等，并且numHeads是numKeyValueHeads的倍数关系。
- 参数query中的D和key的D(H/numKeyValueHeads)值相等，value的D(H/numKeyValueHeads)和output的D值相等。
- 参数query中的B是[1, 10000]区间内的整数。
- 参数query中的B与blockTable中的B与actualCmpKvSeqLenOptional数组的长度相等。
- 参数key中的numBlocks和参数value中的numBlocks值相等。
- 参数key中的blockSize、参数value中的blockSize和pageBlockSize值相等。
- query，key，value输入，功能使用限制如下：
  -   支持query的N轴必须是key/value的N轴（H/D）的整数倍。
  -   支持query的N轴与key/value的N轴（H/D）的比值（即GQA中的group大小）小于等于128，且128是group的整数倍。
  -   支持query与Key的D轴小于等于192。
  -   支持value的D轴小于等于128。
  -   支持query与Key的D轴大于等于value的D轴。
  -   支持key与value的blockSize小于等于128，且是16的整数倍。
  -   仅支持query，key，value输入的数据类型完全相同，为FLOAT16或BFLOAT16。
  -   仅支持query的S轴小于等于4。
  -   仅支持paged attention。
  -   仅支持key/value的S轴小于等于8192。
  -   仅支持compressBlockSize取值16、32、48、64、80、96、112、128。
  -   仅支持compressBlockStride取值16、32、48、64。
  -   仅支持selectBlockSize取值16、32、48、64、80、96、112、128。
  -   仅支持compressBlockSize大于等于compressBlockStride , selectBlockSize大于等于compressBlockSize , selectBlockSize是compressBlockStride的整数倍。
  -   压缩前的kvSeqlen的上限可以表示为：NoCmpKvSeqlenCeil =（cmpKvSeqlen - 1）* compressBlockStride + compressBlockSize，需要满足NoCmpKvSeqlenCeil / selectBlockSize <= 4096，且需要满足selectBlockCount <= NoCmpKvSeqlenCeil / selectBlockSize。

## 算子原型

```c++
REG_OP(NsaCompressAttentionInfer)
    .INPUT(query, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(key, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(value, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(attentionMaskOptional, TensorType({DT_BOOL, DT_BOOL}))  
    .OPTIONAL_INPUT(blockTableOptional, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(actualQSeqLenOptional, TensorType({DT_INT64}))  
    .OPTIONAL_INPUT(actualCmpKvSeqLenOptional, TensorType({DT_INT64}))  
    .OPTIONAL_INPUT(actualSelKvSeqLenOptional, TensorType({DT_INT64}))  
    .OPTIONAL_INPUT(topkMaskOptional, TensorType({DT_BOOL}))  
    .OPTIONAL_INPUT(actualKvSeqLenOptional, TensorType({DT_INT64}))
    .OUTPUT(output, TensorType({DT_FLOAT16, DT_BF16}))  
    .OUTPUT(topkIndicesOut, TensorType({DT_FLOAT16, DT_BF16}))  
    .OUTPUT(topk_index, TensorType({DT_INT32})) 
    .ATTR(numHeads, Int, 1)  
    .ATTR(numKeyValueHeads, Int, 0)  
    .ATTR(selectBlockSize, Int, 0)
    .ATTR(selectBlockCount, Int, 0)
    .ATTR(compressBlockSize, Int, 0)
    .ATTR(compressBlockStride, Int, 0)
    .ATTR(scaleValue, Float, 1.0)  
    .ATTR(layoutOptional, Str)  
    .ATTR(pageBlockSize, Int, 0)  
    .ATTR(sparseMode, Int, 0)  
    .OP_END_FACTORY_REG(NsaCompressAttentionInfer)
```
参数解释请参见**算子执行接口**。

## 调用示例

- PyTorch框架调用

  如果通过PyTorch单算子方式调用该融合算子，则需要参考PyTorch融合算子[torch_npu.npu_nsa_compress_attention_infer](https://hiascend.com/document/redirect/PyTorchAPI)；如果用户定制了该融合算子，则需要参考《Ascend C算子开发》手册[适配PyTorch框架](https://hiascend.com/document/redirect/CannCommunityAscendCInvorkOnNetwork)。

- aclnn单算子调用方式

  通过aclnn单算子调用示例代码如下（以<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>为例），仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

    ```c++
    #include <iostream>
    #include <vector>
    #include <cstring>
    #include "acl/acl.h"
    #include "aclnn/opdev/fp16_t.h"
    #include "aclnnop/aclnn_nsa_compress_attention_infer.h"

    using namespace std;

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
      int64_t shapeSize = 1;
      for (auto i : shape) {
        shapeSize *= i;
      }
      return shapeSize;
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
      // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. 构造输入与输出，需要根据API的接口自定义构造
      int32_t batchSize = 20;
      int32_t headDimsQK = 192;
      int32_t blockNum = 640;
      int32_t headDimsV = 128;
      int32_t sequenceLengthK = 4096;
      int32_t maxNumBlocksPerSeq = 32;
      // attr
        int64_t numHeads = 64;
        int64_t numKeyValueHeads = 4;
        int64_t selectBlockSize = 64;
        int64_t selectBlockCount = 16;
        int64_t compressBlockSize = 32;
        int64_t compressStride = 16;
        double scaleValue = 0.088388;
      string sLayerOut = "TND";
      char layOut[sLayerOut.length()];
      strcpy(layOut, sLayerOut.c_str());
        int64_t pageBlockSize = 128;
        int64_t sparseMod = 0;
      std::vector<int64_t> queryShape = {batchSize, numHeads, headDimsQK};
      std::vector<int64_t> keyShape = {blockNum, pageBlockSize, numKeyValueHeads * headDimsQK};
      std::vector<int64_t> valueShape = {blockNum, pageBlockSize, numKeyValueHeads * headDimsV};
      std::vector<int64_t> blockTableOptionalShape = {batchSize, maxNumBlocksPerSeq};
        std::vector<int64_t> outputShape = {batchSize, numHeads, headDimsV};
        std::vector<int64_t> topkIndicesShape = {batchSize, numKeyValueHeads, selectBlockCount};
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
      std::vector<op::fp16_t> queryHostData(batchSize * numHeads * headDimsQK, 1.0);
      std::vector<op::fp16_t> keyHostData(blockNum * pageBlockSize * numKeyValueHeads * headDimsQK, 1.0);
      std::vector<op::fp16_t> valueHostData(blockNum * pageBlockSize * numKeyValueHeads * headDimsV, 1.0);
      std::vector<int32_t> blockTableOptionalHostData(batchSize * maxNumBlocksPerSeq, 1);
      std::vector<op::fp16_t> outputHostData(batchSize * numHeads * headDimsV, 1.0);
      std::vector<int32_t> topkIndicesHostData(batchSize * numKeyValueHeads * selectBlockCount, 1);

      // 创建query aclTensor
      ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, aclDataType::ACL_FLOAT16, &queryTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建key aclTensor
      ret = CreateAclTensor(keyHostData, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT16, &keyTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建v aclTensor
      ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT16, &valueTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建blockTableOptional aclTensor
      ret = CreateAclTensor(blockTableOptionalHostData, blockTableOptionalShape, &blockTableOptionalDeviceAddr, aclDataType::ACL_INT32, &blockTableOptionalTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建output aclTensor
      ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT16, &outputTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建topkIndices aclTensor
      ret = CreateAclTensor(topkIndicesHostData, topkIndicesShape, &topkIndicesDeviceAddr, aclDataType::ACL_INT32, &topkIndicesTensor);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

        std::vector<int64_t> actualCmpKvSeqLenVector(batchSize, sequenceLengthK);
        auto actualCmpKvSeqLen = aclCreateIntArray(actualCmpKvSeqLenVector.data(), actualCmpKvSeqLenVector.size());
        
      // 3. 调用CANN算子库API
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      // 调用第一段接口
      ret = aclnnNsaCompressAttentionInferGetWorkspaceSize(queryTensor, keyTensor, valueTensor, nullptr, blockTableOptionalTensor, nullptr, actualCmpKvSeqLen,
            nullptr, nullptr,
            numHeads, numKeyValueHeads, selectBlockSize, selectBlockCount, compressBlockSize, compressStride,
            scaleValue, layOut, pageBlockSize, sparseMod, outputTensor, topkIndicesTensor, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompressAttentionInferGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void* workspaceAddr = nullptr;
      if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // 调用第二段接口
      ret = aclnnNsaCompressAttentionInfer(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNsaCompressAttentionInfer failed. ERROR: %d\n", ret); return ret);

      // 4. （固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

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
        auto topksize = GetShapeSize(topkIndicesShape);
      std::vector<op::fp16_t> topkresultData(topksize, 0);
      ret = aclrtMemcpy(topkresultData.data(), topkresultData.size() * sizeof(topkresultData[0]), topkIndicesDeviceAddr,
                topksize * sizeof(topkresultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy [top k] result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < printNum; i++) {
        std::cout << "topk index: " << i << ": " << static_cast<int32_t>(topkresultData[i]) << std::endl;
      }

      // 6. 释放资源
      aclDestroyTensor(queryTensor);
      aclDestroyTensor(keyTensor);
      aclDestroyTensor(valueTensor);
      aclDestroyTensor(blockTableOptionalTensor);
      aclDestroyIntArray(actualCmpKvSeqLen);
      aclDestroyTensor(outputTensor);
      aclDestroyTensor(topkIndicesTensor);
      aclrtFree(queryDeviceAddr);
      aclrtFree(keyDeviceAddr);
      aclrtFree(valueDeviceAddr);
      aclrtFree(blockTableOptionalDeviceAddr);
      aclrtFree(outputDeviceAddr);
      aclrtFree(topkIndicesDeviceAddr);
      if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 0;
    }
    ```