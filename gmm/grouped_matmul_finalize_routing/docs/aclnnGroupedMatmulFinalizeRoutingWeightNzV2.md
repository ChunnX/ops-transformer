# aclnnGroupedMatmulFinalizeRoutingWeightNzV2

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/gmm/grouped_matmul_finalize_routing)

## 产品支持情况

| 产品                                                                | 是否支持 |
|:------------------------------------------------------------------|:----:|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>                      |  √   |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |  √   |


## 功能说明

- 接口功能：GroupedMatmul和MoeFinalizeRouting的融合算子，GroupedMatmul计算后的输出按照索引做combine动作，支持w为AI处理器亲和数据排布格式(NZ)。
本接口相较于[aclnnGroupedMatmulFinalizeRoutingWeightNz](aclnnGroupedMatmulFinalizeRoutingWeightNz.md)，新增对INT4类型weight矩阵的支持，并新增入参offsetOptional、antiquantScaleOptional、antiquantOffsetOptional、tuningConfigOptional，其中前三个参数当前为预留参数，暂不生效，传入空指针即可。tuningConfigOptional是调优参数，数组中的第一个值表示各个专家处理的token数的预期值，算子tiling时会按照该预期值合理进行tiling切分，性能更优。请根据实际情况选择合适的接口。


## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGroupedMatmulFinalizeRoutingWeightNzV2”接口执行计算。
```cpp
aclnnStatus aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize(
    const aclTensor   *x1,
    aclTensor         *x2,
    const aclTensor   *scale,
    const aclTensor   *bias,
    const aclTensor   *offsetOptional,
    const aclTensor   *antiquantScaleOptional,
    const aclTensor   *antiquantOffsetOptional,
    const aclTensor   *pertokenScaleOptional,
    const aclTensor   *groupList,
    const aclTensor   *sharedInput,
    const aclTensor   *logit,
    const aclTensor   *rowIndex,
    int64_t            dtype,
    float              sharedInputWeight,
    int64_t            sharedInputOffset,
    bool               transposeX1,
    bool               transposeX2,
    int64_t            groupListType,
    const aclIntArray *tuningConfigOptional,
    aclTensor         *out,
    uint64_t          *workspaceSize,
    aclOpExecutor     **executor)
```

```cpp
aclnnStatus aclnnGroupedMatmulFinalizeRoutingWeightNzV2(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1494px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 400px">
  <col style="width: 230px">
  <col style="width: 212px">
  <col style="width: 100px">
  <col style="width: 190px">
  <col style="width: 145px">
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
      <td>x1</td>
      <td>输入</td>
      <td>输入x(左矩阵)。</td>
      <td>无</td>
      <td>INT8</td>
      <td>ND</td>
      <td>(m, k)，维度m的取值范围为[1,16\*1024\*8]，k支持2048</td>
      <td>×</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>输入weight(右矩阵)</td>
      <td>无</td>
      <td>INT4、INT8</td>
      <td>NZ</td>
      <td>shape支持三维，当输入为INT32时维度为(e, k, n / 8)，输入转为INT4时维度为(e, k, n)，e取值范围[1,256]，k支持2048，n支持7168</td>
      <td>×</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td>量化参数中的缩放因子，perchannel量化参数</td>
      <td>无</td>
      <td>INT64、FLOAT32、BF16</td>
      <td>ND</td>
      <td>shape支持三维，维度为(e, 1, n)，e、n和w的e、n一致</td>
      <td>×</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>输入</td>
      <td>矩阵的偏移</td>
      <td>无</td>
      <td>BF16、FLOAT32</td>
      <td>ND</td>
      <td>shape支持二维，维度为(e, n)，e、n和w的e、n一致</td>
      <td>×</td>
    </tr>
    <tr>
      <td>offsetOptional</td>
      <td>输入</td>
      <td>非对称量化的偏移量</td>
      <td>无</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>shape支持三维，维度为(e, 1, n)，e、n和w的e、n一致</td>
      <td>×</td>
    </tr>
    <tr>
      <td>antiquantScaleOptional</td>
      <td>输入</td>
      <td>伪量化的缩放因子</td>
      <td>目前暂未启用</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td></td>
      <td>×</td>
    </tr>
    <tr>
      <td>antiquantOffsetOptional</td>
      <td>输入</td>
      <td>伪量化的偏移量</td>
      <td>目前暂未启用</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td></td>
      <td>×</td>
    </tr>
    <tr>
      <td>pertokenScaleOptional</td>
      <td>输入</td>
      <td>矩阵计算的反量化参数</td>
      <td></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>shape支持一维，维度为(m)，m和x的m一致</td>
      <td>×</td>
    </tr>
    <tr>
      <td>groupList</td>
      <td>输入</td>
      <td>输入和输出分组轴方向的matmul大小分布</td>
      <td></td>
      <td>INT64</td>
      <td>ND</td>
      <td>shape支持一维，维度为(e)，e和w的e一致</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sharedInput</td>
      <td>输入</td>
      <td>moe计算中共享专家的输出，需要与moe专家的输出进行combine操作</td>
      <td></td>
      <td>BF16</td>
      <td>ND</td>
      <td>shape支持一维，维度为(e)，e和w的e一致</td>
      <td>×</td>
    </tr>
    <tr>
      <td>logit</td>
      <td>输入</td>
      <td>moe专家对各个token的logit大小</td>
      <td></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>shape支持一维，维度为(m)，m和x的m一致</td>
      <td>×</td>
    </tr>
    <tr>
      <td>rowIndex</td>
      <td>输入</td>
      <td>moe专家输出按照该rowIndex进行combine，其中的值即为combine做scatter add的索引</td>
      <td></td>
      <td>INT64、INT32</td>
      <td>ND</td>
      <td>shape支持一维，维度为(m)，m和x的m一致</td>
      <td>×</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>输入</td>
      <td>计算的输出类型：0：FLOAT32；1：FLOAT16；2：BFLOAT16。目前仅支持0。</td>
      <td></td>
      <td>INT64</td>
      <td></td>
      <td></td>
      <td>×</td>
    </tr>
    <tr>
      <td>sharedInputWeight</td>
      <td>输入</td>
      <td>共享专家与moe专家进行combine的系数，sharedInput先与该参数乘，然后在和moe专家结果累加。</td>
      <td></td>
      <td>FLOAT32</td>
      <td></td>
      <td></td>
      <td>×</td>
    </tr>
    <tr>
      <td>sharedInputOffset</td>
      <td>输入</td>
      <td>共享专家输出的在总输出中的偏移。</td>
      <td></td>
      <td>INT64</td>
      <td></td>
      <td></td>
      <td>×</td>
    </tr>
    <tr>
      <td>transposeX</td>
      <td>输入</td>
      <td>左矩阵是否转置，仅支持false。</td>
      <td></td>
      <td>BOOL</td>
      <td></td>
      <td></td>
      <td>×</td>
    </tr>
    <tr>
      <td>transposeX2</td>
      <td>输入</td>
      <td>右矩阵是否转置，仅支持false。</td>
      <td></td>
      <td>BOOL</td>
      <td></td>
      <td></td>
      <td>×</td>
    </tr>
    <tr>
      <td>groupListType</td>
      <td>输入</td>
      <td>分组模式：配置为0：cumsum模式，即为前缀和；配置为1：count模式。</td>
      <td></td>
      <td>INT64</td>
      <td></td>
      <td></td>
      <td>×</td>
    </tr>
    <tr>
      <td>tuningConfigOptional</td>
      <td>输入</td>
      <td>数组中的第一个元素表示各个专家处理的token数的预期值，算子tiling时会按照数组的第一个元素合理进行tiling切分，性能更优。数组中的第二个元素设置为1，则算子tiling时会根据实际输入尝试使用更适合的算法，当k<=2048的时候，性能可能更优。从第三个元素开始预留，用户无须填写。未来会进行扩展。兼容历史版本，用户如不使用该参数，不传入(即为nullptr)即可。</td>
      <td></td>
      <td>INT64</td>
      <td></td>
      <td></td>
      <td>×</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>输出结果。</td>
      <td>shape与self相同。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>(batch, n)</td>
      <td>×</td>
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

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 700px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的x1、x2、scale、bias或out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>x1、x2、scale、bias、offsetOptional、antiquantScaleOptional、antiquantOffsetOptional、pertokenScaleOptional、groupList、sharedInputOptional、logit、rowIndex、sharedInputWeight、sharedInputOffset、transposeX、transposeX2、或out的数据类型或数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>x1、x2、scale、bias、offsetOptional、antiquantScaleOptional、antiquantOffsetOptional、pertokenScaleOptional、groupList、sharedInputOptional、logit、rowIndex或out的shape不满足校验条件。</td>
    </tr>
    <tr>
      <td>x1、x2、scale、bias、offsetOptional、antiquantScaleOptional、antiquantOffsetOptional、pertokenScaleOptional、groupList、sharedInputOptional、logit、rowIndex或out的shape是空tensor。</td>
    </tr>
    <tr>
      <td>x1、x2、scale、bias、offsetOptional、antiquantScaleOptional、antiquantOffsetOptional、pertokenScaleOptional、groupList、sharedInputOptional、logit、rowIndex或out的shape是空tensor。</td>
    </tr>
  </tbody></table>

## aclnnGroupedMatmulFinalizeRoutingWeightNzV2

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
    <col style="width: 173px">
    <col style="width: 112px">
    <col style="width: 668px">
    </colgroup>
    <thead>
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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize获取。</td>
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

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
- 确定性计算：
  - aclnnGroupedMatmulFinalizeRoutingWeightNzV2默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。

输入和输出支持以下数据类型组合：

| x1    | x2    | scale   | bias    | offsetOptional  | antiquantScaleOptional | antiquantOffsetOptional | pertokenScaleOptional| groupList | sharedInput | logit   | rowIndex | out   | tuningConfigOptional |
|------|------|---------|---------|---------|----------------|-----------------|---------------|-----------|-------------|---------|----------|-------|----------------------|
| INT8 | INT8 | FLOAT32 | null    | null    | null           | null            | FLOAT32       | INT64     | BFLOAT16    | FLOAT32 | INT64    | FLOAT | IntArray             |
| INT8 | INT8 | FLOAT32 | null    | null    | null           | null            | FLOAT32       | INT64     | BFLOAT16    | FLOAT32 | INT64    | FLOAT | IntArray             |
| INT8 | INT4 | INT64   | FLOAT32 | FLOAT32 | null           | null            | FLOAT32       | INT64     | BFLOAT16    | FLOAT32 | INT64    | FLOAT | IntArray             |
| INT8 | INT4 | INT64   | FLOAT32 | null    | null           | null            | FLOAT32       | INT64     | BFLOAT16    | FLOAT32 | INT64    | FLOAT | IntArray             |

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_permute.h"
  #include "aclnnop/aclnn_grouped_matmul_finalize_routing_weight_nz_v2.h"
  #include "aclnnop/aclnn_trans_matmul_weight.h"

  #define CHECK_RET(cond, return_expr) \
      do {                             \
          if (!(cond)) {               \
              return_expr;             \
          }                            \
      } while (0)

  #define CHECK_FREE_RET(cond, return_expr) \
      do {                                  \
          if (!(cond)) {                    \
              Finalize(deviceId, stream);   \
              return_expr;                  \
          }                                 \
      } while (0)

  #define LOG_PRINT(message, ...)         \
      do {                                \
          printf(message, ##__VA_ARGS__); \
      } while (0)

  int64_t GetShapeSize(const std::vector<int64_t> &shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  int Init(int32_t deviceId, aclrtStream *stream)
  {
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
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      return 0;
  }

  template <typename T>
  int CreateAclTensorWeight(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclTensor **tensor)
  {
      auto size = static_cast<uint64_t>(GetShapeSize(shape));

      const aclIntArray *mat2Size = aclCreateIntArray(shape.data(), shape.size());
      auto ret = aclnnCalculateMatmulWeightSizeV2(mat2Size, dataType, &size);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSizeV2 failed. ERROR: %d\n", ret);
                return ret);
      size *= sizeof(T);

      // 调用aclrtMalloc申请device侧内存
      ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // 计算连续tensor的strides
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      std::vector<int64_t> storageShape;
      storageShape.push_back(GetShapeSize(shape));

      // 调用aclCreateTensor接口创建aclTensor
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                storageShape.data(), storageShape.size(), *deviceAddr);
      return 0;
  }

    int main() {
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init stream failed. ERROR: %d\n", ret); return ret);

      // 2. 构造输入与输出，需要根据API的接口自定义构造
      int64_t m = 192;
      int64_t k = 2048;
      int64_t n = 7168;
      int64_t e = 4;
      int64_t batch = 24;
      int64_t bsdp = 8;
      int64_t dtype = 0;
      float shareInputWeight = 1.0;
      int64_t shareInputOffset = 0;
      bool transposeX = false;
      bool transposeW = false;
      int64_t groupListType = 1;
      
      std::vector<int64_t> xShape = {m, k};
      std::vector<int64_t> wShape = {e, k, n};
      std::vector<int64_t> scaleShape = {e, n};
      std::vector<int64_t> pertokenScaleShape = {m};
      std::vector<int64_t> groupListShape = {e};
      std::vector<int64_t> sharedInputShape = {bsdp, n};
      std::vector<int64_t> logitShape = {m};
      std::vector<int64_t> rowIndexShape = {m};
      std::vector<int64_t> outShape = {batch, n};
      std::vector<int64_t> tuningConfigVal = {1}; 

      void *xDeviceAddr = nullptr;
      void *wDeviceAddr = nullptr;
      void *biasDeviceAddr = nullptr;
      void *scaleDeviceAddr = nullptr;
      void *pertokenScaleDeviceAddr = nullptr;
      void *groupListDeviceAddr = nullptr;
      void *sharedInputDeviceAddr = nullptr;
      void *logitDeviceAddr = nullptr;
      void *rowIndexDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      void *tuningConfigDeviceAddr = nullptr;

      aclTensor* x = nullptr;
      aclTensor* w = nullptr;
      aclTensor* bias = nullptr;
      aclTensor* groupList = nullptr;
      aclTensor* scale = nullptr;
      aclTensor* pertokenScale = nullptr;
      aclTensor* sharedInput = nullptr;
      aclTensor* logit = nullptr;
      aclTensor* rowIndex = nullptr;
      aclTensor* out = nullptr;

      std::vector<int8_t> xHostData(GetShapeSize(xShape));
      std::vector<int8_t> wHostData(GetShapeSize(wShape));
      std::vector<float> scaleHostData(GetShapeSize(scaleShape));
      std::vector<float> pertokenScaleHostData(GetShapeSize(pertokenScaleShape));
      std::vector<int64_t> groupListHostData(GetShapeSize(groupListShape));
      groupListHostData[0] = 7;
      groupListHostData[0] = 32;
      groupListHostData[0] = 40;
      groupListHostData[0] = 64;

      std::vector<uint16_t> sharedInputHostData(GetShapeSize(sharedInputShape));
      std::vector<int64_t> logitHostData(GetShapeSize(logitShape));
      std::vector<float> rowIndexHostData(GetShapeSize(rowIndexShape));
      std::vector<float> outHostData(GetShapeSize(outShape));  // 实际上是float16半精度方式

      // 创建x aclTensor
      ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_INT8, &x);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> xTensorPtr(x, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建AI处理器亲和数据排布格式的w aclTensor
      ret = CreateAclTensorWeight(wHostData, wShape, &wDeviceAddr, aclDataType::ACL_INT8, &w);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> wTensorPtr(w, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> wDeviceAddrPtr(wDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建scale aclTensor
      ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建pertokenScale aclTensor
      ret = CreateAclTensor(pertokenScaleHostData, pertokenScaleShape, &pertokenScaleDeviceAddr, aclDataType::ACL_FLOAT, &pertokenScale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> pertokenScaleTensorPtr(pertokenScale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> pertokenScaleDeviceAddrPtr(pertokenScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建groupList aclTensor
      ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupList);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> groupListTensorPtr(groupList, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建sharedInput aclTensor
      ret = CreateAclTensor(sharedInputHostData, sharedInputShape, &sharedInputDeviceAddr, aclDataType::ACL_BF16, &sharedInput);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> sharedInputTensorPtr(sharedInput, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> sharedInputDeviceAddrPtr(sharedInputDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建logit aclTensor
      ret = CreateAclTensor(logitHostData, logitShape, &logitDeviceAddr, aclDataType::ACL_FLOAT, &logit);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> logitTensorPtr(logit, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> logitDeviceAddrPtr(logitDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建rowIndex aclTensor
      ret = CreateAclTensor(rowIndexHostData, rowIndexShape, &rowIndexDeviceAddr, aclDataType::ACL_INT64, &rowIndex);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> rowIndexTensorPtr(rowIndex, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> rowIndexDeviceAddrPtr(rowIndexDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建out aclTensor
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建tuningConfig aclIntArray
      aclIntArray *tuningConfig = aclCreateIntArray(tuningConfigVal.data(), tuningConfigVal.size());
      std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> tuningConfigIntArrayPtr(tuningConfig, aclDestroyIntArray);
      std::unique_ptr<void, aclError (*)(void *)> tuningConfigDeviceAddrPtr(tuningConfigDeviceAddr, aclrtFree);
      CHECK_RET(tuningConfig == nullptr, -1);
      // 3. 调用CANN算子库API，需要修改为具体的Api名称
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor;
      void *workspaceAddr = nullptr;

      // 调用aclnnTransMatmulWeight第一段接口
      ret = aclnnTransMatmulWeightGetWorkspaceSize(w, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeightGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // 调用aclnnTransMatmulWeight第二段接口
      ret = aclnnTransMatmulWeight(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeight failed. ERROR: %d\n", ret); return ret);

      // 调用aclnnGroupedMatmulFinalizeRoutingWeightNzV2第一段接口
      workspaceSize = 0;                                               
      ret = aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize(x, w, scale, nullptr, nullptr, nullptr, nullptr, pertokenScale, groupList, sharedInput, logit, rowIndex, dtype, shareInputWeight, shareInputOffset, transposeX, transposeW, groupListType, tuningConfig, out, &workspaceSize, &executor);

      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存

      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // 调用aclnnGroupedMatmulFinalizeRoutingWeightNzV2第二段接口
      ret = aclnnGroupedMatmulFinalizeRoutingWeightNzV2(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedMatmulFinalizeRoutingWeightNzV2 failed. ERROR: %d\n", ret); return ret);

      // 4. （固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(
          size, 0);  // C语言中无法直接打印fp16的数据，需要用uint16读出来，自行通过二进制转成fp16
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %u\n", i, resultData[i]);
      }

      // 6. 释放aclTensor资源，需要根据具体API的接口定义修改
      aclDestroyTensor(x);
      aclDestroyTensor(w);
      aclDestroyTensor(scale);
      aclDestroyTensor(pertokenScale);
      aclDestroyTensor(groupList);
      aclDestroyTensor(sharedInput);
      aclDestroyTensor(logit);
      aclDestroyTensor(rowIndex);
      aclDestroyTensor(out);

      // 7.释放device资源，需要根据具体API的接口定义修改
      aclrtFree(xDeviceAddr);
      aclrtFree(wDeviceAddr);
      aclrtFree(scaleDeviceAddr);
      aclrtFree(pertokenScaleDeviceAddr);
      aclrtFree(groupListDeviceAddr);
      aclrtFree(sharedInputDeviceAddr);
      aclrtFree(logitDeviceAddr);
      aclrtFree(rowIndexDeviceAddr);
      aclrtFree(outDeviceAddr);
      aclDestroyIntArray(tuningConfig);

      if (workspaceSize > 0) {
          aclrtFree(workspaceAddr);
      }
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
      return 0;
  }
  ```