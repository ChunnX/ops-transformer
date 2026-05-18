# aclnnAiInfraRotaryPositionEmbeddingV2

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------- |
| <term>Ascend 950PR/Ascend 950DT</term>                   | ×       |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √       |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √       |
| <term>Atlas 200I/500 A2 推理产品</term>                  | ×       |
| <term>Atlas 推理系列产品</term>                          | ×       |
| <term>Atlas 训练系列产品</term>                          | ×       |

## 功能说明

- 接口功能：给定输入张量 **x**（形状任意，最后一维为特征维度），位置编码张量 **cos** 和 **s**in（与 **x** 广播兼容），输出 **y** 的计算分为三步：
  
  1、将输入沿最后一维切分为两半
  
  $$
  x_1 = x[..., : D/2], \quad x_2 = x[..., D/2 :]
  $$
  
  其中 D=x.shape[−1]，要求 D 为偶数。
  
  2、构造旋转向量
  
  $$
  x_{\text{rotate}} = \operatorname{concat}(-x_2,\; x_1,\; \text{dim} = -1)
  $$
  
  即前半部分为 x2 的相反数，后半部分为 x1。
  
  3、执行旋转编码
  
  $$
  y = x \odot \cos + x_{\text{rotate}} \odot \sin
  $$
  
  其中 ⊙ 表示逐元素乘法，广播规则遵循 NumPy/PyTorch 的广播语义。

## 函数原型

每个算子分为两段式接口，必须先调用"aclnnAiInfraRotaryPositionEmbeddingV2GetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnAiInfraRotaryPositionEmbeddingV2"接口执行计算。

```cpp
aclnnStatus aclnnAiInfraRotaryPositionEmbeddingV2GetWorkspaceSize(
const aclTensor *x,
const aclTensor *cos,
const aclTensor *sin,
int64_t mode,
const aclTensor *rotateOptional,
const aclTensor *out,
uint64_t *workspaceSize,
aclOpExecutor **executor)
```

```cpp
aclnnStatus aclnnAiInfraRotaryPositionEmbeddingV2(
  void *workspace,
  uint64_t workspaceSize,
  aclOpExecutor *executor,
  aclrtStream stream)
```

## aclnnAiInfraRotaryPositionEmbeddingV2GetWorkspaceSize

### 参数说明

| 参数名         | 输入/输出 | 描述                                  | 使用说明                                                          | 数据类型                   | 数据格式 | 维度(shape) | 非连续Tensor |
| -------------- | --------- | ------------------------------------- | ----------------------------------------------------------------- | -------------------------- | -------- | ----------- | ------------ |
| x              | 输入      | 待执行旋转位置编码的张量，公式中的x   | 最后一维必须为偶数                                                | BFLOAT16、FLOAT16、FLOAT32 | ND       | 3,4         | √           |
| cos            | 输入      | 位置编码张量，公式中的cos             | shape需与x广播兼容                                                | BFLOAT16、FLOAT16、FLOAT32 | ND       | 3,4         | √           |
| sin            | 输入      | 位置编码张量，公式中的sin             | shape需与x广播兼容                                                | BFLOAT16、FLOAT16、FLOAT32 | ND       | 3,4         | √           |
| mode           | 输入      | 旋转模式                              | 0表示half模式，1表示interleave模式，<font color="red">当前仅支持0 | INT64                      | -        | -           | -            |
| rotateOptional | 输入      | 旋转矩阵                              | 可选参数，传入nullptr时使用默认旋转构造逻辑，当前只支持nullptr         | BFLOAT16、FLOAT16、FLOAT32 | ND       | 2           | -            |
| out            | 输出      | 旋转位置编码计算结果，公式中的y       | 需预先分配，shape与x一致                                          | BFLOAT16、FLOAT16、FLOAT32 | ND       | 3,4         | √           |
| workspaceSize  | 输出      | 返回需要在Device侧申请的workspace大小 | -                                                                 | -                          | -        | -           | -            |
| executor       | 输出      | 返回op执行器，包含算子计算流程        | -                                                                 | -                          | -        | -           | -            |

- **返回值：**

aclnnStatus：返回状态码，具体参见aclnn返回码文档。

第一段接口完成入参校验，出现以下场景时报错：

| 返回值                       | 错误码 | 描述                                                                                                          |
| ---------------------------- | ------ | ------------------------------------------------------------------------------------------------------------- |
| ACLNN_ERR_INNER_NULLPTR      | 161001 | 参数x、cos、sin、out为空指针                                                                                  |
| ACLNN_ERR_PARAM_INVALID      | 161002 | x最后一维不为偶数；x、cos、sin、out数据类型不一致；mode取值不在有效范围内[0,1]；cos/sin的shape与x无法广播兼容；rotateOptional不为空 |
| ACLNN_ERR_INNER_TILING_ERROR | 561002 | Tiling计算失败                                                                                                |

## aclnnAiInfraRotaryPositionEmbeddingV2

### 参数说明

| 参数名        | 输入/输出 | 描述                                                                                         |
| ------------- | --------- | -------------------------------------------------------------------------------------------- |
| workspace     | 输入      | 在Device侧申请的workspace内存地址                                                            |
| workspaceSize | 输入      | 在Device侧申请的workspace大小，由第一段接口aclnnAiInfraRotaryPositionEmbeddingV2GetWorkspaceSize获取 |
| executor      | 输入      | op执行器，包含了算子计算流程                                                                 |
| stream        | 输入      | 指定执行任务的AscendCL Stream流                                                              |

### 返回值

aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  
  - aclnnAiInfraRotaryPositionEmbeddingV2默认确定性实现。
- 输入shape限制：
  
  - x的维度支持3维或4维，最后一维D必须为偶数。
  - cos和sin的维度支持3维或4维，shape需与x广播兼容。
  - rotateOptional为2维矩阵。
- 输入值域限制：
  
  - mode取值范围为0或1，<font color="red">当前仅支持0（half模式）</font>，1（interleave模式）暂不支持。
- 数据类型限制：
  
  - x、cos、sin、rotateOptional、out的数据类型必须一致，支持BFLOAT16、FLOAT16、FLOAT32。
  - mode的数据类型为INT64。
- 格式限制：
  - 所有Tensor参数的数据格式要求为ND。
- <font color="red">满足以下条件的shape支持自适应分核策略，优先满载UB，从而减少启动核数。</font>
  - 输入input (x_dim0，x_dim1, x_dim2, x_dim3)或(x_dim1, x_dim2, x_dim3)
  - 输入r1/r2 (r_dim0, r_dim1, r_dim2, r_dim3)或(r_dim1, r_dim2, r_dim3)
  - r_dim2 等于 1
  - (x_dim0==r_dim0 && x_dim1==r_dim1）||（r_dim0 * r_dim1 * r_dim2==1)
  - x_dim3 <= 64
  - 对于float32类型输入：8 * x_dim2 * x_dim3 + 8 * x_dim3 <= 180 * 1024
  - 对于float16/bfloat16类型输入：10 * x_dim2 * x_dim3 + 12 * x_dim3 <= 180 * 1024
- 支持非连续Tensor：
  
  - x、cos、sin支持非连续Tensor输入，out支持非连续Tensor输出。

## 调用示例
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas 训练系列产品</term>：
以下为C++调用示例，需结合AscendCL环境编译运行。
```cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_rotary_position_embedding_v2.h"
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

std::vector<float> get_interleave_matrix(int64_t n) {
    std::vector<float> matrix(n * n, 0.0f);
    for (int64_t i = 0; i < n; i += 2) {
        if (i + 1 >= n) {
            break;
        }
        matrix[i * n + (i + 1)] = 1.0f;
        matrix[(i + 1) * n + i] = 1.0f;
    }
    return matrix;
}

int main() {
    // 1. 固定写法，device/stream初始化, 参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口定义构造
    std::vector<int64_t> xShape = {1, 1, 1, 128};
    std::vector<int64_t> cosShape = {1, 1, 1, 128};
    std::vector<int64_t> sinShape = {1, 1, 1, 128};
    std::vector<int64_t> outShape = {1, 1, 1, 128};
    std::vector<int64_t> rotateShape = {128, 128};
    int64_t mode = 1;

    void* xDeviceAddr = nullptr;
    void* cosDeviceAddr = nullptr;
    void* sinDeviceAddr = nullptr;
    void* rotateDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* cos = nullptr;
    aclTensor* sin = nullptr;
    aclTensor* rotate = nullptr;
    aclTensor* out = nullptr;

    std::vector<float> xHostData = {74, 54, 84, 125, 23, 78, 37, 72, 27, 98, 34, 107, 29, 23, 54, 60, 70, 49,
                                    119, 54, 29, 54, 41, 99, 27, 62, 5, 46, 108, 39, 24, 123, 33, 82, 6, 40, 88,
                                    24, 6, 116, 38, 119, 110, 5, 30, 79, 87, 18, 29, 100, 90, 24, 21, 93, 63, 68,
                                    34, 112, 119, 48, 74, 43, 85, 64, 14, 49, 128, 59, 18, 37, 123, 76, 14, 63, 10,
                                    39, 107, 124, 79, 16, 17, 76, 80, 47, 90, 41, 58, 82, 75, 80, 69, 37, 74, 36, 54,
                                    26, 32, 54, 13, 100, 105, 15, 13, 69, 122, 26, 94, 59, 29, 14, 60, 8, 24, 17, 45,
                                    33, 107, 122, 63, 111, 75, 128, 68, 31, 105, 6, 82, 99};
    std::vector<float> cosHostData = {41, 37, 17, 25, 49, 25, 22, 24, 110, 120, 107, 3, 82, 66, 75, 86, 85, 115, 110, 56, 52,
                                      39, 86, 23, 36, 71, 20, 73, 113, 25, 114, 56, 125, 80, 95, 82, 31, 63, 99, 62, 23, 55, 30,
                                      99, 42, 121, 15, 24, 97, 87, 81, 67, 43, 21, 13, 9, 33, 29, 117, 10, 114, 61, 98, 15, 78,
                                      108, 48, 97, 1, 3, 78, 109, 57, 46, 47, 56, 50, 66, 81, 77, 17, 128, 68, 121, 47, 91, 114,
                                      125, 51, 108, 31, 15, 47, 78, 109, 115, 113, 26, 53, 97, 1, 111, 103, 58, 106, 68, 11,
                                      104, 22, 79, 61, 127, 86, 39, 33, 123, 102, 39, 64, 41, 119, 120, 61, 29, 94, 68, 36, 12};
    std::vector<float> sinHostData = {46, 56, 56, 101, 66, 10, 96, 16, 86, 57, 102, 66, 12, 105, 76, 58, 90, 6, 79, 128, 126,
                                      82, 41, 3, 45, 7, 66, 4, 46, 22, 31, 26, 37, 63, 97, 84, 91, 90, 47, 77, 90, 34, 41, 83,
                                      91, 108, 120, 13, 90, 32, 85, 37, 119, 31, 51, 82, 122, 125, 7, 116, 121, 108, 38, 56,
                                      100, 20, 97, 119, 10, 4, 53, 13, 46, 82, 103, 119, 124, 80, 23, 67, 78, 56, 119, 122, 40,
                                      58, 128, 27, 30, 52, 71, 42, 123, 69, 4, 5, 116, 97, 38, 107, 8, 4, 65, 120, 40, 22, 60,
                                      44, 48, 66, 68, 125, 4, 93, 112, 112, 113, 90, 94, 23, 104, 39, 85, 84, 64, 128, 96, 119};
    std::vector<float> rotateHostData = get_interleave_matrix(128);
    std::vector<float> outHostData(128, 0);

    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建cos aclTensor
    ret = CreateAclTensor(cosHostData, cosShape, &cosDeviceAddr, aclDataType::ACL_FLOAT, &cos);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建sin aclTensor
    ret = CreateAclTensor(sinHostData, sinShape, &sinDeviceAddr, aclDataType::ACL_FLOAT, &sin);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建rotate aclTensor
    ret = CreateAclTensor(rotateHostData, rotateShape, &rotateDeviceAddr, aclDataType::ACL_FLOAT, &rotate);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnRotaryPositionEmbeddingV2第一段接口
    ret = aclnnRotaryPositionEmbeddingV2GetWorkspaceSize(x, cos, sin, mode, rotate, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRotaryPositionEmbeddingV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnRotaryPositionEmbeddingV2第二段接口
    ret = aclnnRotaryPositionEmbeddingV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRotaryPositionEmbeddingV2 failed. ERROR: %d\n", ret); return ret);
    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(cos);
    aclDestroyTensor(sin);
    aclDestroyTensor(rotate);
    aclDestroyTensor(out);

    // 7. 释放device 资源
    aclrtFree(xDeviceAddr);
    aclrtFree(cosDeviceAddr);
    aclrtFree(sinDeviceAddr);
    aclrtFree(rotateDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```