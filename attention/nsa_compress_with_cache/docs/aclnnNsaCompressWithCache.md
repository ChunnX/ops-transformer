## aclnnNsaCompressWithCache

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/nsa_compress_with_cache)

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

- 接口功能：用于Native-Sparse-Attention推理阶段的KV压缩，每次推理每个batch会产生一个新的token，每当某个batch的token数量凑满一个compress_block时，该算子会将该batch的后compress_block个token压缩成一个compress_token。
- 计算公式：

$$
compressIdx=(s-compressBlockSize)/stride\\ 
outputCacheRef[slotMapping[i]] = input[compressIdx*stride : compressIdx*stride+compressBlockSize]*weight[:]
$$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnNsaCompressWithCacheGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnNsaCompressWithCache”接口执行计算。

```c++
aclnnStatus aclnnNsaCompressWithCacheGetWorkspaceSize(
   const aclTensor   *input,
   const aclTensor   *weight,
   const aclTensor   *slotMapping,
   const aclIntArray *actSeqLenOptional,
   const aclTensor   *blockTableOptional,
   char              *layoutOptional,
   int64_t            compressBlockSize,
   int64_t            compressStride,
   int64_t            actSeqLenType,
   int64_t            pageBlockSize,
   aclTensor         *outputCache,
   uint64_t          *workspaceSize,
   aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnNsaCompressWithCache(
   void          *workspace,
   uint64_t       workspaceSize,
   aclOpExecutor *executor,
   aclrtStream    stream)
```

## aclnnNsaCompressWithCacheGetWorkspaceSize

- **参数说明**
  
  <table style="undefined; table-layout: fixed; width: 1567px">
    <colgroup>
      <col style="width: 170px"> <!-- 参数名 -->
      <col style="width: 120px"> <!-- 输入/输出 -->
      <col style="width: 300px"> <!-- 描述 -->
      <col style="width: 330px"> <!-- 使用说明 -->
      <col style="width: 212px"> <!-- 数据类型 -->
      <col style="width: 100px"> <!-- 数据格式 -->
      <col style="width: 190px"> <!-- 维度(shape) -->
      <col style="width: 145px"> <!-- 非连续Tensor -->
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
        <td>input</td>
        <td>输入</td>
        <td>待压缩张量。</td>
        <td>
          <ul style="list-style-type: circle;">
            <li>不支持空Tensor。</li>
            <li>input和weight满足broadcast关系，input的第三维大小与weight的第二维大小相等。</li>
            <li>headDim是16的整数倍，且headDim<=256。</li>
            <li>headNum<=64，且headNum>50时headNum%2=0。</li>
            <li>N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸。</li>
          </ul>
        </td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>[blockNum, pageBlockSize, N, D]、[TND]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weight</td>
        <td>输入</td>
        <td>压缩的权重。</td>
        <td>
          <ul style="list-style-type: circle;">
            <li>不支持空Tensor。</li>
            <li>数据类型与input保持一致。</li>
            <li>N（Head-Num）表示多头数。</li>
          </ul>
        </td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>[compressBlockSize, N]</td>
        <td>√</td>
      </tr>
      <tr>
        <td>slotMapping</td>
        <td>输入</td>
        <td>每个batch尾部压缩数据存储的位置的索引。</td>
        <td>
          <ul style="list-style-type: circle;">
            <li>不支持空Tensor。</li>
            <li>slotMapping的值无重复，否则会导致计算结果不稳定。</li>
          </ul>
        </td>
        <td>INT32</td>
        <td>ND</td>
        <td>[B]</td>
        <td>x</td>
      </tr>
      <tr>
        <td>actSeqLenOptional</td>
        <td>可选输入</td>
        <td>每个Batch对应的S大小。</td>
        <td>
          <ul style="list-style-type: circle;">
            <li>在TND排布场景下需要该输入，其余场景输入nullptr。</li>
            <li>S（Seq-Length）表示输入样本序列长度。</li>
            <li>actSeqLenOptional的值不应该超过序列最大长度。</li>
          </ul>
        </td>
        <td>INT64</td>
        <td>ND</td>
        <td>[B]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>blockTableOptional</td>
        <td>可选输入</td>
        <td>PageAttention中KV存储使用的block映射表。</td>
        <td>
          <ul style="list-style-type: circle;">
            <li>使用该功能可传入nullptr。</li>
            <li>blockTableOptional的值不超过blockNum，否则会发生越界。</li>
          </ul>
        </td>
        <td>INT32</td>
        <td>ND</td>
        <td>[batch, blockNumPerBatch]</td>
        <td>-</td>
      </tr>
      <tr>
        <td>layoutOptional</td>
        <td>可选输入</td>
        <td>输入input的数据排布格式。</td>
        <td>
          <ul style="list-style-type: circle;">
            <li>当前仅支持"TND"，当传入blockTableOptional时此参数无效，否则为必选参数。</li>
            <li>其中T是B和S合轴紧密排列的数据（每个batch的actSeqLen）、B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸。</li>
          </ul>
        </td>
        <td>CHAR</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>compressBlockSize</td>
        <td>输入</td>
        <td>压缩滑窗大小。</td>
        <td>必须是16的整数倍，且compressBlockSize>=compressStride，compressBlockSize<=64。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>compressStride</td>
        <td>输入</td>
        <td>两次压缩间的滑窗间隔大小。</td>
        <td>仅支持compressBlockStride取值16、32、48、64。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>actSeqLenType</td>
        <td>输入</td>
        <td>actSeqLenOptional的不同表达形式</td>
        <td>actSeqLenOptional有输入时生效，可取值0或1，0代表actSeqLenOptional中数值为前继batch的序列大小的cumsum结果（累积和），1代表actSeqLenOptional中数值为每个batch中序列大小，当前仅支持1。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>pageBlockSize</td>
        <td>输入</td>
        <td>page attention场景下page的blocksize大小。</td>
        <td>只能是64或者128。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>outputCache</td>
        <td>输出</td>
        <td>压缩之后的cache</td>
        <td>数据类型与input保持一致。</td>
        <td>INT64</td>
        <td>-</td>
        <td>[result_len, N, D]</td>
        <td>x</td>
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
    <table><thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>计算输入和必选计算输出是空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>计算输入和输出的数据类型和格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_RUNTIME_ERROR</td>
      <td rowspan="6">561002</td>
      <td>input和weight不满足broadcast关系，即input的第三维大小与weight的第二维大小不相等。</td>
    </tr>
    <tr>
      <td>activeNum、expertNum、expertCapacity的值小于0。</td>
    </tr>
    <tr>
      <td>compress_block_size、compress_stride 、不是16的整数倍，或者compress_block_size<compress_stride。</td>
    </tr>
    <tr>
      <td>seq_lens_type!=1或者layout取值不是BSH、SBH、BSND、BNSD、TND中的一个。</td>
    </tr>
    <tr>
      <td>page_block_size取值不是64或者128。</td>
    </tr>
    <tr>
      <td>headDim未对齐16。</td>
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnNsaCompressWithCache获取。</td>
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
  - aclnnNsaCompressWithCache默认确定性实现。
- outputCache的N和D和input一致，而且要满足result_len>(blockNum*pageBlockSize-compressBlockSize)/compressStride。
- page attention场景下input的shape支持[blockNum,pageBlockSize,N,D]，其余场景下input的shape支持[T,N,D]。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

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
        printf("outputCache[%ld]:%f\n", i, aclFloat16ToFloat(resultData[i]));
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
