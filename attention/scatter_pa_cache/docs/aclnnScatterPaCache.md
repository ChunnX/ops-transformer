# aclnnScatterPaCache

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 接口功能：更新KCache中指定位置的key。

- 计算公式：
  - 场景一：
    ```
    key:[batch * seq_len, num_head, k_head_size]
    keyCache:[num_blocks, block_size, num_head, k_head_size]
    slotMapping:[batch * seq_len]
    cacheMode:"Norm"
    ```  
    $$
    keyCache = slotMapping(key)
    $$

  - 场景二：
    ```
    key:[batch, seq_len, num_head, k_head_size]
    keyCache:[num_blocks, block_size, 1, k_head_size]
    slotMapping:[batch, num_head]
    compressLensOptional:[batch, num_head]
    compressSeqOffsetOptional:[batch * num_head]
    seqLensOptional:[batch]
    cacheMode:"Norm"
    ```
    $$
    \begin{aligned}
    keyCache =\ & slotMapping(key[: compressSeqOffset], \\
    & ReduceMean(key[compressSeqOffset : compressSeqOffset + compressLens]), \\
    & key[compressSeqOffset + compressLens : seqLens])
    \end{aligned}
    $$

  - 场景三：
    ```
    key:[batch, seq_len, num_head, k_head_size]
    keyCache:[num_blocks, block_size, 1, k_head_size]
    slotMapping:[batch, num_head]
    compressLensOptional:[batch * num_head]
    seqLensOptional:[batch]
    cacheMode:"Norm"
    ```
    $$
    keyCache = slotMapping(key[seqLens - compressLens : seqLens])
    $$
  
  上述场景根据构造的参数来区别，符合第一种入参构造走场景一，符合第二种构造走场景二，符合第三种构造走场景三。场景一没有compressLensOptional、seqLensOptional、compressSeqOffsetOptional这三个可选参数，场景三没有compressSeqOffsetOptional可选参数。

## 函数原型

每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnScatterPaCacheGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnScatterPaCache”接口执行计算。

```c++
aclnnStatus aclnnScatterPaCacheGetWorkspaceSize(
  const aclTensor *key, 
  const aclTensor *keyCacheRef, 
  const aclTensor *slotMapping, 
  const aclTensor *compressLensOptional, 
  const aclTensor *compressSeqOffsetOptional, 
  const aclTensor *seqLensOptional, 
  char            *cacheMode, 
  uint64_t        *workspaceSize, 
  aclOpExecutor  **executor)
```

```c++
aclnnStatus aclnnScatterPaCache(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnScatterPaCacheGetWorkspaceSize

- **参数说明：**

  | 参数名 | 输入/输出<div style="width: 70px"> | 描述 | 使用说明 | 数据类型 | 数据格式<div style="width: 70px"> | 维度(shape)<div style="width: 80px"> | 非连续tensor <div style="width: 90px">|
  |------|------|------|------|------|------|------|------|
  | <term>key</term>  |  输入 | 待更新的key值，公式中的key。 |<ul><li>支持空tensor。</li><li>HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2仅支持key是3维的场景。</li><li>shape满足[batch * seq_len, num_head, k_head_size]或[batch, seq_len, num_head, k_head_size]，FLOAT4_E2M1、FLOAT4_E1M2情况下，k_head_size必须是偶数。</li><ul> | FLOAT16、FLOAT、BFLOAT16、INT8、UINT8、INT16、UINT16、INT32、UINT32、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2 | ND | 3-4 | × |
  | keyCacheRef |    输入输出   | 需要更新的keyCache，公式中的keyCache。 | <ul><li>支持空tensor。</li><li>当key是3维时，shape满足[num_blocks, block_size, num_head, k_head_size]，当key是4维时，shape满足[num_blocks, block_size, 1, k_head_size]。</li><ul> | 与key一致。 | ND | 4 | × |
  | slotMapping | 输入 | key的每个token在cache中的存储偏移，公式中的slotMapping。 | <ul><li>支持空tensor。</li><li>当key是3维时，shape满足[batch * seq_len]，当key是4维时，shape满足[batch, num_head]。</li><li>值范围为[0, num_blocks * block_size-1]，且元素值不能重复，重复时不保证正确性。</li><ul> | INT32、INT64 | ND | 1-2 | × |
  | compressLensOptional | 输入 | 压缩量，公式中的compressLens。 | <ul><li>支持空tensor。</li><li>当key是4维且compressSeqOffsetOptional不为空指针时，shape满足[batch, num_head]，当key是4维且compressSeqOffsetOptional为空指针时，shape满足[batch * num_head]。</li><li>场景一传空指针。</li><ul> | 与slotMapping一致。 | ND | 1-2 | × |
  | compressSeqOffsetOptional | 输入 | 每个batch中每个head的压缩起点，公式中的compressSeqOffset。| <ul><li>支持空tensor。</li><li>shape满足[batch * num_head]。</li><li>场景一和场景三传空指针。</li><ul> | 与slotMapping一致。 | ND | 1 | × |
  | seqLensOptional | 输入 | 每个batch的实际seqLens，公式中的seqLens。 | <ul><li>支持空tensor。</li><li>shape满足[batch]。</li><li>场景一传空指针。</li><ul> | 与slotMapping一致。 | ND | 1 | × |
  | cacheMode | 输入 | keyCacheRef的内存排布格式。 | <ul><li>预留参数，当前数据格式只支持ND，该参数不生效。</li><ul> | - | - | - | - |
  | workspaceSize | 输出 | 返回需要在Device侧申请的workspace大小。 | - | - | - | - | - |
  | executor | 输出 | 返回op执行器，包含了算子计算流程。 | - | - | - | - | - |

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  |返回值|错误码|描述|
  |------|------|------|
  |ACLNN_ERR_PARAM_NULLPTR|161001|1. key、keyCacheRef、slotMapping存在空指针。|
  |ACLNN_ERR_PARAM_INVALID|161002|1. key、slotMapping、compressLensOptional、compressSeqOffsetOptional或seqLensOptional的数据类型不在支持的范围之内。<br>2. key或keyCacheRef的数据类型不匹配。<br>3. slotMapping、compressLensOptional、compressSeqOffsetOptional或seqLensOptional的数据类型不匹配。<br>4. key、slotMapping、compressLensOptional、compressSeqOffsetOptional或seqLensOptional的shape维度不在支持的范围之内。|

## aclnnScatterPaCache

- **参数说明：**
  |参数名|输入/输出|描述|
  |------|------|------|
  |workspace|输入|在Device侧申请的workspace内存地址。|
  |workspaceSize|输入|在Device侧申请的workspace大小，由第一段接口aclnnScatterPaCacheGetWorkspaceSize获取。|
  |executor|输入|op执行器，包含了算子计算流程。|
  |stream|输入|指定执行任务的Stream。|

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

## 约束说明
- 确定性计算：aclnnScatterPaCache默认为确定性实现，暂不支持非确定性实现，[确定性计算](common/确定性计算.md)配置也不会生效。
- 参数说明中shape使用的变量说明如下：
  - batch：当前输入的序列数量（一次处理的样本数），取值为正整数。
  - seq_len：序列的长度，取值为正整数。
  - num_head：多头注意力中“头”的数量，取值为正整数。
  - k_head_size：每个注意力头中key的特征维度（单头key的长度），取值为正整数。
  - num_blocks：keyCache中预分配的块总数，用于存储所有序列的key数据，取值为正整数。
  - block_size：每个缓存块包含的token数量，取值为正整数。
- 输入值域限制：seqLensOptional和compressLensOptional里面的每个元素值必须满足公式：reduceSum(seqLensOptional[i] - compressLensOptional[i] + 1) <= num_blocks * block_size（对应场景二、三）。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

- <term>Ascend 950PR/Ascend 950DT</term> ：

  ```c++
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_scatter_pa_cache.h"

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
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> keyShape = {2, 2, 3, 7};
    std::vector<int64_t> keyCacheShape = {2, 2, 1, 7};
    std::vector<int64_t> slotMappingShape = {2, 3};

    std::vector<int64_t> compressLensShape = {2, 3};
    std::vector<int64_t> compressSeqOffsetShape = {6};
    std::vector<int64_t> seqLensShape = {2};
    void* keyDeviceAddr = nullptr;
    void* slotMappingDeviceAddr = nullptr;
    void* keyCacheDeviceAddr = nullptr;
    void* compressLensDeviceAddr = nullptr;
    void* compressSeqOffsetDeviceAddr = nullptr;
    void* seqLensDeviceAddr = nullptr;

    aclTensor* key = nullptr;
    aclTensor* slotMapping = nullptr;
    aclTensor* keyCache = nullptr;
    aclTensor* compressLens = nullptr;
    aclTensor* compressSeqOffset = nullptr;
    aclTensor* seqLens = nullptr;
    char* cacheMode = const_cast<char*>("Norm");

    std::vector<float> hostKey = {1};
    std::vector<int32_t> hostSlotMapping = {0, 3, 6, 9, 12, 15};
    std::vector<float> hostKeyCacheRef = {1};
    std::vector<int32_t> hostCompressLens = {1, 0, 0, 0, 1, 0};
    std::vector<int32_t> hostCompressSeqOffset = {0, 0, 1, 0, 1, 1};
    std::vector<int32_t> hostSeqLens = {2, 1};

    // 创建key aclTensor
    ret = CreateAclTensor(hostKey, keyShape, &keyDeviceAddr, aclDataType::ACL_FLOAT, &key);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建slotMappitng aclTensor
    ret = CreateAclTensor(hostSlotMapping, slotMappingShape, &slotMappingDeviceAddr, aclDataType::ACL_INT32, &slotMapping);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建keyCache aclTensor
    ret = CreateAclTensor(hostKeyCacheRef, keyCacheShape, &keyCacheDeviceAddr, aclDataType::ACL_FLOAT, &keyCache);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建compressLens aclTensor
    ret = CreateAclTensor(hostCompressLens, compressLensShape, &compressLensDeviceAddr, aclDataType::ACL_INT32, &compressLens);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建compressSeqOffset aclTensor
    ret = CreateAclTensor(hostCompressSeqOffset, compressSeqOffsetShape, &compressSeqOffsetDeviceAddr, aclDataType::ACL_INT32, &compressSeqOffset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建seqLens aclTensor
    ret = CreateAclTensor(hostSeqLens, seqLensShape, &seqLensDeviceAddr, aclDataType::ACL_INT32, &seqLens);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnScatterPaCache第一段接口
    ret = aclnnScatterPaCacheGetWorkspaceSize(key, keyCache, slotMapping, compressLens, compressSeqOffset, seqLens, cacheMode, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterPaCacheGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnScatterPaCache第二段接口
    ret = aclnnScatterPaCache(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterPaCache failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(keyShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), keyDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(key);
    aclDestroyTensor(slotMapping);
    aclDestroyTensor(keyCache);
    aclDestroyTensor(compressLens);
    aclDestroyTensor(compressSeqOffset);
    aclDestroyTensor(seqLens);
    // 7. 释放device资源，需要根据具体API的接口定义参数
    aclrtFree(keyDeviceAddr);
    aclrtFree(slotMappingDeviceAddr);
    aclrtFree(keyCacheDeviceAddr);
    aclrtFree(compressLensDeviceAddr);
    aclrtFree(compressSeqOffsetDeviceAddr);
    aclrtFree(seqLensDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
  }
  ```