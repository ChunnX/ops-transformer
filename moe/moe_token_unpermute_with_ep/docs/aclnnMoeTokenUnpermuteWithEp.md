# aclnnMoeTokenUnpermuteWithEp

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/moe/moe_token_unpermute_with_ep)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- **接口功能：** 用于MoE专家并行场景，根据sortedIndices记录的全局token索引，从本卡rangeOptional范围内的permutedTokens中取回token，并按numTopk分组累加；传入probsOptional时，累加前会乘以对应的专家概率。

- **计算公式：**

  设numTopk为$K$，原始token数为$N$，隐藏维度为$H$，rangeOptional为$[r_0,r_1)$。对于$i\in[0,N)$、$j\in[0,K)$和$h\in[0,H)$，令：

  $$
  s_{i,j}=sortedIndices_{iK+j}
  $$

  $$
  v_{i,j,h}=\begin{cases}
  permutedTokens_{s_{i,j}-r_0,h}, & r_0\le s_{i,j}<r_1 \\
  0, & \text{其他情况}
  \end{cases}
  $$

  传入probsOptional时：

  $$
  out_{i,h}=\sum_{j=0}^{K-1}v_{i,j,h}\times probsOptional_{i,j}
  $$

  未传入probsOptional时：

  $$
  out_{i,h}=\sum_{j=0}^{K-1}v_{i,j,h}
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnMoeTokenUnpermuteWithEpGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMoeTokenUnpermuteWithEp”接口执行计算。

```c++
aclnnStatus aclnnMoeTokenUnpermuteWithEpGetWorkspaceSize(
    const aclTensor   *permutedTokens,
    const aclTensor   *sortedIndices,
    const aclTensor   *probsOptional,
    int64_t            numTopk,
    const aclIntArray *rangeOptional,
    bool               paddedMode,
    const aclIntArray *restoreShapeOptional,
    const aclTensor   *out,
    uint64_t          *workspaceSize,
    aclOpExecutor     **executor);
```

```c++
aclnnStatus aclnnMoeTokenUnpermuteWithEp(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream);
```

## aclnnMoeTokenUnpermuteWithEpGetWorkspaceSize

- **参数说明**

  <table style="table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 320px">
  <col style="width: 250px">
  <col style="width: 120px">
  <col style="width: 140px">
  <col style="width: 140px">
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
  </tr></thead>
  <tbody>
  <tr>
      <td>permutedTokens（aclTensor*）</td>
      <td>输入</td>
      <td>经过EP切分后的本地token，对应公式中的permutedTokens。</td>
      <td><ul><li>不能传入空指针，不支持空Tensor。</li><li>必须为二维Tensor，shape为(R,H)。传入rangeOptional时，要求R不小于r1-r0，其中rangeOptional为[r0,r1)。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>(R,H)</td>
      <td>√</td>
  </tr>
  <tr>
      <td>sortedIndices（aclTensor*）</td>
      <td>输入</td>
      <td>每个原始token所对应专家输出的全局索引，对应公式中的sortedIndices。</td>
      <td><ul><li>不能传入空指针，不支持空Tensor。</li><li>必须为一维Tensor，长度为N×K，其中N为原始token数，K为numTopk。</li><li>超出[r0,r1)的索引对应的输出贡献为0。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>(N×K,)</td>
      <td>√</td>
  </tr>
  <tr>
      <td>probsOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>每个原始token对应的专家概率，对应公式中的probsOptional。</td>
      <td><ul><li>允许传入空指针；传入空指针时不进行概率加权。</li><li>传入Tensor时不支持空Tensor，必须为二维Tensor，shape为(N,K)，且K等于numTopk。</li><li>数据类型可以与permutedTokens不同。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>(N,K)</td>
      <td>√</td>
  </tr>
  <tr>
      <td>numTopk（int64_t）</td>
      <td>输入</td>
      <td>每个原始token选中的专家数量，对应公式中的K。</td>
      <td>取值必须大于等于1；传入probsOptional时，取值必须小于等于512。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>rangeOptional（aclIntArray*）</td>
      <td>输入</td>
      <td>本卡EP切分的全局token索引范围，对应公式中的[r0,r1)。</td>
      <td><ul><li>允许传入空指针；传入空指针时使用默认值{0,0}，没有索引落入有效范围，输出为全0。</li><li>传入aclIntArray时，元素数量必须为2。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>(2,)</td>
      <td>-</td>
  </tr>
  <tr>
      <td>paddedMode（bool）</td>
      <td>输入</td>
      <td>表示是否启用padded模式。</td>
      <td>当前仅支持false。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>restoreShapeOptional（aclIntArray*）</td>
      <td>输入</td>
      <td>padded模式下用于恢复输出shape的预留参数。</td>
      <td>当前计算不使用该参数，传入空指针即可。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>按原始token顺序合并后的输出，对应公式中的out。</td>
      <td><ul><li>不能传入空指针，不支持空Tensor。</li><li>必须为二维Tensor，shape为(N,H)。</li><li>数据类型必须与permutedTokens一致。</li></ul></td>
      <td>与permutedTokens一致</td>
      <td>ND</td>
      <td>(N,H)</td>
      <td>×</td>
  </tr>
  <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
  </tr>
  <tr>
      <td>executor（aclOpExecutor**）</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="table-layout: fixed; width: 1000px"><colgroup>
      <col style="width: 300px">
      <col style="width: 150px">
      <col style="width: 550px">
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
          <td>permutedTokens、sortedIndices、out或executor为空指针。</td>
          </tr>
          <tr>
          <td>ACLNN_ERR_PARAM_INVALID</td>
          <td>161002</td>
          <td>输入和输出的数据类型和数据格式不在支持的范围之内。</td>
          </tr>
      </tbody>
  </table>

## aclnnMoeTokenUnpermuteWithEp

- **参数说明**

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 700px">
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
          <td>在Device侧申请的workspace大小，由第一段接口aclnnMoeTokenUnpermuteWithEpGetWorkspaceSize获取。</td>
          </tr>
          <tr>
          <td>executor</td>
          <td>输入</td><td> op执行器，包含了算子计算流程。</td>
          </tr>
          <tr>
          <td>stream</td>
          <td>输入</td>
          <td> 指定执行任务的Stream。</td>
          </tr>
      </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性说明：aclnnMoeTokenUnpermuteWithEp默认确定性实现。
- 不支持Broadcast。
- aclTensor的shape必须为运行时确定的实际值，不支持使用-1表示动态维度或使用-2表示动态Rank。
- permutedTokens的shape为(R,H)。传入rangeOptional时要求r0小于等于r1，且R不小于r1-r0；sortedIndices的长度为N×K，out的shape为(N,H)。传入probsOptional时，其shape必须为(N,K)。
- numTopk必须大于等于1；传入probsOptional时，numTopk还必须小于等于512。
- rangeOptional传入aclIntArray时必须包含两个元素。传入空指针时使用默认值{0,0}，不会回退调用其他算子。
- paddedMode仅支持false，restoreShapeOptional传入空指针即可。

## 调用示例

以下示例展示基本调用流程，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp

#include "acl/acl.h"
#include "aclnnop/aclnn_moe_token_unpermute_with_ep.h"
#include <iostream>
#include <vector>

#define CHECK_RET(cond, return_expr)                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      return_expr;                                                             \
    }                                                                          \
  } while (0)

#define LOG_PRINT(message, ...)                                                \
  do {                                                                         \
    printf(message, ##__VA_ARGS__);                                            \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
      return );
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream *stream) {
  // 初始化ACL和Stream
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
            return ret);
  return 0;
}

template <typename T>
int CreateAclIntArray(const std::vector<T> &hostData, aclIntArray **intArray) {
  *intArray = aclCreateIntArray(hostData.data(), hostData.size());
  CHECK_RET(*intArray != nullptr, LOG_PRINT("aclCreateIntArray failed.\n"); return 1);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData,
                    const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
            return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size,
                    ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
            return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                            strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. 初始化Device和Stream
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2. 构造输入与输出

  std::vector<float> permutedTokensData = {2, 2, 1, 1, 3, 3, 2, 2};
  std::vector<int64_t> permutedTokensShape = {4, 2};
  void *permutedTokensAddr = nullptr;
  aclTensor *permutedTokens = nullptr;

  ret = CreateAclTensor(permutedTokensData, permutedTokensShape,
                        &permutedTokensAddr, aclDataType::ACL_FLOAT,
                        &permutedTokens);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int> sortedIndicesData = {2, 0, 4, 1, 5, 3};
  std::vector<int64_t> sortedIndicesShape = {6};
  void *sortedIndicesAddr = nullptr;
  aclTensor *sortedIndices = nullptr;

  ret =
      CreateAclTensor(sortedIndicesData, sortedIndicesShape, &sortedIndicesAddr,
                      aclDataType::ACL_INT32, &sortedIndices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<float> probsOptionalData = {1, 1, 1, 1, 1, 1};
  std::vector<int64_t> probsOptionalShape = {3, 2};
  void *probsOptionalAddr = nullptr;
  aclTensor *probsOptional = nullptr;

  ret =
      CreateAclTensor(probsOptionalData, probsOptionalShape, &probsOptionalAddr,
                      aclDataType::ACL_FLOAT, &probsOptional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  int64_t numTopk = 2;
  aclIntArray *range = nullptr;
  std::vector<int64_t> rangeHostData = {1, 5};
  ret = CreateAclIntArray(rangeHostData, &range);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<float> outData = {0, 0, 0, 0, 0, 0};
  std::vector<int64_t> outShape = {3, 2};
  void *outAddr = nullptr;
  aclTensor *out = nullptr;

  ret = CreateAclTensor(outData, outShape, &outAddr, aclDataType::ACL_FLOAT,
                        &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);


  // 3. 调用CANN算子库API
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // 调用aclnnMoeTokenUnpermuteWithEp第一段接口
  ret = aclnnMoeTokenUnpermuteWithEpGetWorkspaceSize(permutedTokens, sortedIndices,
                                                     probsOptional, numTopk, range, false, nullptr,
                                                     out, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnMoeTokenUnpermuteWithEpGetWorkspaceSize failed. ERROR: %d\n",
                ret);
      return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // 调用aclnnMoeTokenUnpermuteWithEp第二段接口
  ret = aclnnMoeTokenUnpermuteWithEp(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnMoeTokenUnpermuteWithEp failed. ERROR: %d\n", ret);
            return ret);

  // 4. 同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5. 将输出从Device侧拷贝至Host侧
  PrintOutResult(outShape, &outAddr);

  // 6. 释放aclTensor和aclIntArray
  aclDestroyTensor(permutedTokens);
  aclDestroyTensor(sortedIndices);
  aclDestroyTensor(probsOptional);
  aclDestroyTensor(out);
  aclDestroyIntArray(range);

  // 7. 释放device资源
  aclrtFree(permutedTokensAddr);
  aclrtFree(sortedIndicesAddr);
  aclrtFree(probsOptionalAddr);
  aclrtFree(outAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
