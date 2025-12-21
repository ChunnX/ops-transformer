## aclnnNsaCompressAttentionInfer

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/nsa_compress_attention_infer)

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>昇腾910_95 AI处理器</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |
|<term>Atlas 200I/300/500 推理产品</term>|      ×     |

## 功能说明

- 接口功能：Native Sparse Attention推理过程中，Compress Attention的计算。
- 计算公式：

<center>

  $$
  P_{cmp}= Softmax(scale * query · key^T) \\
  attentionOut = P_{cmp} · value\\
  P_{slc}[j] = \sum\limits_{m=0}^{l'/d -1} \sum\limits_{n = 0}^{l/d -1} P_{cmp} [l'/d * j -m - n]\\
  P_{slc'} = \sum\limits_{g=1}^{G}  P_{slc} ^g,\quad 
  \text{其中 } G = \text{GroupSize（分组大小），即：} G = \frac{\text{numHeads}}{\text{numKeyValueHeads}} \\
  topkIndices = topk(P_{slc'})\\
  $$

</center>

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnNsaCompressAttentionInferGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnNsaCompressAttentionInfer”接口执行计算。

```cpp
aclnnStatus aclnnNsaCompressAttentionInferGetWorkspaceSize(
    const aclTensor    *query,
    const aclTensor    *key,
    const aclTensor    *value,
    const aclTensor    *attentionMaskOptional,
    const aclTensor    *blockTableOptional,
    const aclIntArray  *actualQSeqLenOptional,
    const aclIntArray  *actualCmpKvSeqLenOptional,
    const aclIntArray  *actualSelKvSeqLenOptional,
    const aclTensor    *topKMaskOptional,
    int64_t             numHeads,
    int64_t             numKeyValueHeads,
    int64_t             selectBlockSize,
    int64_t             selectBlockCount,
    int64_t             compressBlockSize,
    int64_t             compressBlockStride,
    double              scaleValue,
    char               *layoutOptional,
    int64_t             pageBlockSize,
    int64_t             sparseMode,
    const aclTensor    *output,
    const aclTensor    *topKOutput,
    uint64_t           *workspaceSize,
    aclOpExecutor     **executor
)
```

```cpp
aclnnStatus aclnnNsaCompressAttentionInfer(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream
)
```

## aclnnNsaCompressAttentionInferGetWorkspaceSize

- **参数说明**
  
  <table style="undefined;table-layout: fixed; width: 1567px">
  <colgroup>
    <col style="width: 170px">  <!-- 参数名 -->
    <col style="width: 120px">  <!-- 输入/输出 -->
    <col style="width: 300px">  <!-- 描述 -->
    <col style="width: 330px">  <!-- 使用说明 -->
    <col style="width: 212px">  <!-- 数据类型 -->
    <col style="width: 100px">  <!-- 数据格式 -->
    <col style="width: 190px">  <!-- 维度(shape) -->
    <col style="width: 145px">  <!-- 非连续的tensor -->
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
      <td>query</td>
      <td>输入</td>
      <td>attention结构的Query输入。</td>
      <td>
        <ul style="list-style-type: circle;">
          <li>query中的B是[1, 10000]区间内的整数，且与blockTableOptional中的B以及actualCmpKvSeqLenOptional数组的长度相等。</li>
          <li>query的S轴小于等于4。</li>
          <li>query中的N和numHeads值相等，且N轴必须是key/value的N轴（H/D）的整数倍，此外，query的N轴与key/value的N轴（H/D）的比值（即GQA中的group大小）小于等于128，且128是group的整数倍。</li>
          <li>query中的D和key的D(H/numKeyValueHeads)值相等，小于等于192且大于等于value的D轴。</li>
          <li>query，key，value输入的数据类型完全相同，为FLOAT16或BFLOAT16。</li>
        </ul>
      </td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[BSND]、[TND]</td>
      <td>x</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输入</td>
      <td>attention结构的Key输入。</td>
      <td>
        <ul style="list-style-type: circle;">
          <li>key中的numBlocks和参数value中的numBlocks值相等。</li>
          <li>key中的blockSize和pageBlockSize值相等，且blockSize小于等于128，且是16的整数倍。</li>
          <li>key的S轴小于等于8192。</li>
          <li>key的N和numKeyValueHeads值相等。</li>
          <li>key的D轴小于等于192且大于等于value的D轴。</li>
        </ul>
      </td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[numBlocks, blockSize, numKeyValueHeads * headDimsQK]</td>
      <td>x</td>
    </tr>
    <tr>
      <td>value</td>
      <td>输入</td>
      <td>attention结构的Value输入。</td>
      <td>
        <ul style="list-style-type: circle;">
          <li>value的N和numKeyValueHeads值相等。</li>
          <li>value的D(H/numKeyValueHeads)和output的D值相等。</li>
  	  <li>value的D轴小于等于128。</li>
          <li>value中的blockSize和pageBlockSize值相等。</li>
          <li>value的S轴小于等于8192。</li>
        </ul>
      </td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[numBlocks, blockSize, numKeyValueHeads * headDimsV]</td>
      <td>x</td>
    </tr>
    <tr>
      <td>attentionMaskOptional</td>
      <td>可选输入</td>
      <td>attention掩码矩阵。</td>
      <td>仅在Q_S大于1的情况下生效</td>
      <td>-</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockTableOptional</td>
      <td>输入</td>
      <td>paged attention中KV存储使用的block映射表。</td>
      <td>当前只支持paged attention，因此该参数必须传入。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>-</td>
      <td>x</td>
    </tr>
    <tr>
      <td>actualQSeqLenOptional</td>
      <td>可选输入</td>
      <td>query的S轴实际长度。</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>[B]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>actualCmpKvSeqLenOptional</td>
      <td>可选输入</td>
      <td>经过压缩后的key和value的S轴实际长度，也即该算子处理的key和value的S轴实际长度。</td>
      <td>当前只支持paged attention，因此该参数必须传入。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>actualSelKvSeqLenOptional</td>
      <td>可选输入</td>
      <td>压缩前的key和value的S轴实际长度。</td>
      <td>-</td>
      <td>INT64</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topKMaskOptional</td>
      <td>可选输入</td>
      <td>topK计算中的掩码矩阵。</td>
      <td>预留参数暂未使用。</td>
      <td>-</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeads</td>
      <td>输入</td>
      <td>head个数。</td>
      <td>numHeads是numKeyValueHeads的倍数关系。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numKeyValueHeads</td>
      <td>输入</td>
      <td>kvHead个数。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>selectBlockSize</td>
      <td>输入</td>
      <td>select阶段的block大小，在计算importance score时使用。</td>
      <td>仅支持selectBlockSize取值16、32、48、64、80、96、112、128，且selectBlockSize大于等于compressBlockSize，并且是compressBlockStride的整数倍。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>selectBlockCount</td>
      <td>输入</td>
      <td>topK阶段需要保留的block数量。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>compressBlockSize</td>
      <td>输入</td>
      <td>压缩时的滑窗大小。</td>
      <td>仅支持compressBlockSize取值16、32、48、64、80、96、112、128，且需要大于等于compressBlockStride。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>compressBlockStride</td>
      <td>输入</td>
      <td>两次压缩间的滑窗间隔大小。</td>
      <td>仅支持compressBlockStride取值16、32、48、64。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleValue</td>
      <td>输入</td>
      <td>缩放系数，作为计算流中Muls的scalar值。</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutOptional</td>
      <td>输入</td>
      <td>输入query、key、value的数据排布格式。</td>
      <td>当前支持取值“TND”和“BSND”。</td>
      <td>CHAR</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pageBlockSize</td>
      <td>输入</td>
      <td>blockTableOptional中一个block的大小。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseMode</td>
      <td>输入</td>
      <td>sparse的模式，控制有attentionMaskOptional输入时的稀疏计算。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输出</td>
      <td>attention的输出。</td>
      <td>-</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[BSND]、[TND]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topKOutput</td>
      <td>输出</td>
      <td>topK的输出。</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>[T, N, selectBlockCount]、[B, S, N, selectBlockCount]</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  
  </table>
- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：
  
  <table style="undefined;table-layout: fixed; width: 1030px">
  <colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 650px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入参数是必选输入，输出或者必选属性，且是空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>query、key、value、blockTableOptional、attentionOut、topKOutput的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>API内存调用npu runtime的接口异常。</td>
    </tr>
  </tbody>
  </table>

## aclnnNsaCompressAttentionInfer

- **参数说明**
  
  <table><thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnNsaCompressAttentionInfer获取。</td>
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
- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnNsaCompressAttentionInfer默认确定性实现。
* 参数query仅支持TND、BSND输入。T是B和S合轴紧密排列的数据（每个batch的actualQSeqLenOptional）、B（batch）表示输入样本批量大小、S（qSeqlen）表示输入样本序列长度、N（numHeads）表示多头数、D（headDimsQK）表示隐藏层最小的单元尺寸。
* 压缩前的kvSeqlen的上限可以表示为：actualSelKvSeqLenCeil=(actualCmpKvSeqLenOptional-1)*compressBlockStride+compressBlockSize，需要满足actualSelKvSeqLenCeil/selectBlockSize<=4096，且需要满足selectBlockCount<=actualSelKvSeqLenCeil/selectBlockSize。如果actualSelKvSeqLenOptional不满足actualCmpKvSeqLenOptional=(actualSelKvSeqLenOptional-compressBlockSize)/compressBlockStride+1，或者actualCmpKvSeqLenOptional的长度和blockTableOptional的batch维度不同，则会默认进入单token推理场景。
* 多token推理场景下，actualQSeqLenOptional参数必须传入，actualQSeqLenOptional的长度必须和blockTableOptional的batch维度相等，仅支持query的S轴最大等于4，并且此时要求每个batch单独的actualQSeqLenOptional<=actualSelKvSeqLenOptional。如果actualQSeqLenOptional的长度和blockTableOptional的batch维度不同，或者actualQSeqLenOptional的值小于1或者大于4，则会默认进入单token推理场景。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

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
  // 固定写法，资源初始化
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
