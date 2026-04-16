# aclnnQuantGroupedMatmulInplaceAdd

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>昇腾910_95 AI处理器</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>|      ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |
|<term>Atlas 200I/300/500 推理产品</term>|      ×     |

产品形态详细说明请参见[昇腾产品形态说明](https://www.hiascend.com/document/redirect/CannCommunityProductForm)

## 功能说明

-   算子功能：在micro-batch训练场景，需要做micro-batch的梯度累计，会存在大量GroupedMatMul后接InplaceAdd的融合场景。QuantGroupedMatmulInplaceAdd算子将上述算子融合起来，提高网络性能。实现分组矩阵乘计算和加法计算，基本功能为矩阵乘和加法的组合，如T-C量化场景下$y_i[m,n]=(x1_i[m,k_i] \times x2_i[k_i,n]) * scale2_i[n] * scale1_i + y_i[m,n], i=1...g$，其中g为分组个数，$m/k_i/n$为对应的维度。


    相较于[GroupedMatmulV4](GroupedMatmulV4.md)接口，**此接口变化：**
    - 输入输出参数类型均为aclTensor。
    - 在GroupedMatMul计算结束后增加了InplaceAdd计算。
    - 仅支持量化场景（1.mx量化；2.T-C量化）。量化方式请参见[量化介绍](../../../docs/zh/context/量化介绍.md)。
    - 仅支持x1、x2是FLOAT8_E5M2、FLOAT8_E4M3FN、HIFLOAT8的输入。

-   计算公式：
    - **mx量化：**
    $$
     y_i[m,n] = \sum_{j=0}^{kLoops-1} ((\sum_{k=0}^{gsK-1} (x1Slice_i * x2Slice_i)) * (scale1_i[m, j] * scale2_i[j, n])) + y_i[m,n]
    $$
    其中，gsK代表K轴的量化的block size即32，$x1Slice_i$代表$x1_i$第m行长度为gsK的向量，$x2Slice_i$代表$x2_i$第n列长度为gsK的向量，K轴均从$j*gsK$起始切片，j的取值范围[0, kLoops), kLoops=ceil($K_i$ / gsK)，支持最后的切片长度不足gsK。
    - **T-C量化：**
    $$
     y_i=(x1_i\times x2_i) * scale2_i * scale1_i + y_i
    $$


## 实现原理

QuantGroupedMatmulInplaceAdd算子主要由1个GroupedMatmul和1个InplaceAdd组成，计算过程如下：
GroupedMatmul的计算结果和输入y在内存中相加，将y输出；

## 算子执行接口

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnQuantGroupedMatmulInplaceAdd”接口执行计算。

* `aclnnStatus aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *scale1Optional, const aclTensor *scale2, const aclTensor *groupList, aclTensor *yRef, int64_t groupListType, int64_t groupSize, uint64_t *workspaceSize, aclOpExecutor **executor)`
* `aclnnStatus aclnnQuantGroupedMatmulInplaceAdd(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

**说明**：

- 算子执行接口对外屏蔽了算子内部实现逻辑以及不同代际NPU的差异，且开发者无需编译算子，实现了算子的精简调用。
- 若开发者不使用算子执行接口的调用算子，也可以定义基于Ascend IR的算子描述文件，通过ATC工具编译获得算子om文件，然后加载模型文件执行算子，详细调用方法可参见《应用开发指南》的[单算子调用 > 单算子模型执行](https://hiascend.com/document/redirect/CannCommunityCppOpcall)章节。

### aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize

- **参数说明：**
  -   x1（aclTensor *，计算输入）：Device侧的aclTensor，公式中的输入x1，2维tensor，shape为(K，M)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，数据类型支持FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8。
  -   x2（aclTensor *，计算输入）：Device侧的aclTensor，公式中的输入x2，2维tensor，shape为(K，N)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，数据类型支持FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8。
  -   scale1Optional（aclTensor *，计算输入）：可选参数，Device侧的aclTensor，代表量化参数中的由x1量化引入的缩放因子，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，数据类型支持FLOAT32、FLOAT8_E8M0。综合约束请参见[约束说明](#约束说明)。
  -   scale2（aclTensor *，计算输入）：Device侧的aclTensor，代表量化参数中的由x2量化引入的缩放因子，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，数据类型支持FLOAT32、FLOAT8_E8M0。综合约束请参见[约束说明](#约束说明)。
  -   groupList（aclTensor *，计算输入）：Device侧的aclTensor类型，代表输入和输出分组轴方向的matmul大小分布，数据类型支持INT64，1维tensor，shape为(g，)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。需注意：
        * 当groupListType为0时，groupList必须为非负单调非递减数列，当groupListType为1时，groupList必须为非负数列。
        * groupList中的最后一个值约束了输出数据的有效部分，groupList中未指定的部分将不会参与更新。
  -   yRef（aclTensor *，计算输入输出）：Device侧的aclTensor，公式中的输入输出y，3维tensor，shape为(g，M，N)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND，数据类型支持FLOAT32。
  -   groupListType（int64\_t，计算输入）：整数型参数，支持的取值如下：
        * 0: groupList中数值为分组轴大小的cumsum结果（累积和）;
        * 1: groupList中数值为分组轴上每组大小；
  -   groupSize（int64\_t，计算输入）：整数型参数，用于输入m、n、k方向上的量化分组大小。groupSize输入由3个方向的groupSizeM，groupSizeN，groupSizeK三个值拼接组成，每个值占16位，共占用int64_t类型groupSize的低48位（groupSize中的高16位的数值无效），计算公式为：groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32。
      - 当前只支持传0。
  -   workspaceSize（uint64\_t *，出参）：返回需要在Device侧申请的workspace大小。
  -   executor（aclOpExecutor **，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，若出现以下错误码，则对应原因为：
  - 返回161001（ACLNN_ERR_PARAM_NULLPTR）：
  1.如果传入参数是必选输入、输出或者必选属性，且是空指针。
  - 返回161002（ACLNN_ERR_PARAM_INVALID）：
  1.x1、x2、scale2、groupList、yRef、scale1Optional、groupListType、groupSize的数据类型和数据格式不在支持的范围内。
  ```

### aclnnQuantGroupedMatmulInplaceAdd

-   **参数说明：**
    -   workspace（void\*，入参）：在Device侧申请的workspace内存地址。
    -   workspaceSize（uint64\_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize获取。
    -   executor（aclOpExecutor\*，入参）：op执行器，包含了算子计算流程。
    -   stream（aclrtStream，入参）：指定执行任务的Stream。

-   **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
  - x1和x2的每一维大小在32字节对齐后都应小于int32的最大值2147483647，且内轴大小需小于2097152。
    - 动态量化（T-C量化）场景支持的输入类型为：
      - 不为空的参数支持的数据类型组合要满足下表：
        | x1       | x2  | scale2 | scale1Optional |yRef     |
        |:-------:|:-------:| :------      | :------   | :------ |
        |HIFLOAT8  |HIFLOAT8| FLOAT32    | FLOAT32   | FLOAT32 |
      - scale1Optional/scale2要满足以下约束（其中g为matmul组数即分组数）：
        | 参数 | shape限制 |
        |:---------:| :------ |
        |scale1Optional| 2维tensor或1维tensor，shape为(g, 1)或(g,)|
        |scale2| 2维tensor，shape为(g, N)|
    - 动态量化（mx量化）场景支持的数据类型为：
      - 数据类型组合要满足下表：
        | x1       | x2  |  scale2  | scale1Optional |yRef     |
        |:-------:|:-------:| :-------    | :------   | :------ |
        |FLOAT8_E5M2/FLOAT8_E4M3FN  |FLOAT8_E5M2/FLOAT8_E4M3FN| FLOAT8_E8M0   | FLOAT8_E8M0    | FLOAT32 |
      - scale1Optional/scale2要满足以下约束（其中g为matmul组数即分组数，g\_i为第i个分组（下标从0开始））：
        | 参数 | shape限制 |
        |:---------:| :------ |
        |scale1Optional| 3维tensor，shape为((K / 64) + g, M, 2)，scale\_i起始地址偏移为((K\_0 + K\_1 + ...+ K\_{i-1})/ 64 + g\_i) \* M \* 2，即scale_0的起始地址偏移为0，scale_1的起始地址偏移为(K\_0 / 64 + 1) \* M \* 2， scale_2的起始地址偏移为((K\_0 + K\_1) / 64 + 2) \* M \* 2, 依此类推|
        |scale2| 3维tensor，shape为((K / 64) + g, N, 2), 起始地址偏移与scale1Optional同理|
## 算子原型
```c++
REG_OP(QuantGroupedMatmulInplaceAdd)
    .INPUT(x1, TensorType({DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .INPUT(x2, TensorType({DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .INPUT(scale2, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .INPUT(group_list, TensorType({DT_INT64}))
    .INPUT(y, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(scale1, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(group_list_type, Int, 0)
    .ATTR(group_size, Int, 0)
    .OP_END_FACTORY_REG(QuantGroupedMatmulInplaceAdd)
```

参数解释请参见**算子执行接口**。

## 调用示例
- aclnn单算子调用方式

  通过aclnn单算子调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

    ```c++
  #include <iostream>
  #include <vector>
  #include <memory>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_quant_grouped_matmul_inplace_add.h"

  #define CHECK_RET(cond, return_expr) \
      do {                               \
        if (!(cond)) {                   \
          return_expr;                   \
        }                                \
      } while (0)

  #define CHECK_FREE_RET(cond, return_expr) \
      do {                                  \
          if (!(cond)) {                    \
              Finalize(deviceId, stream);   \
              return_expr;                  \
          }                                 \
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
      // 调用aclrtMalloc申请Device侧内存
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

      // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
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

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  int aclnnQuantGroupedMatmulInplaceAddTest(int32_t deviceId, aclrtStream &stream) {
      auto ret = Init(deviceId, &stream);
      // check根据自己的需要处理
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. 构造输入与输出，需要根据API的接口自定义构造
      std::vector<int64_t> x1Shape = {2, 2};
      std::vector<int64_t> x2Shape= {2, 2};
      std::vector<int64_t> scale2Shape = {2, 2};
      std::vector<int64_t> yShape = {2, 2, 2};
      std::vector<int64_t> scale1Shape = {2, 1};
      std::vector<int64_t> groupListShape = {2};
      void* x1DeviceAddr = nullptr;
      void* x2DeviceAddr = nullptr;
      void* scale2DeviceAddr = nullptr;
      void* scale1DeviceAddr = nullptr;
      void* yDeviceAddr = nullptr;
      void* groupListDeviceAddr = nullptr;
      aclTensor* x1 = nullptr;
      aclTensor* x2 = nullptr;
      aclTensor* groupList = nullptr;
      aclTensor* scale2 = nullptr;
      aclTensor* yRef = nullptr;
      aclTensor* scale1 = nullptr;
      aclTensor* out = nullptr;
      int64_t groupListType = 0;
      int64_t groupSize = 0;
      std::vector<uint8_t> xData(GetShapeSize(x1Shape), 0X10); // hifloat8 2.0 转16进制 0X10
      std::vector<int64_t> groupListData = {1, 2};
      std::vector<float> scale2Data(GetShapeSize(scale2Shape), 1);
      std::vector<float> yData(GetShapeSize(yShape), 1);
      std::vector<float> scale1Data(GetShapeSize(scale1Shape), 1);

      // 创建x aclTensor
      ret = CreateAclTensor<uint8_t>(xData, x1Shape, &x1DeviceAddr, aclDataType::ACL_HIFLOAT8, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建x2 aclTensor
      ret = CreateAclTensor<uint8_t>(xData, x2Shape, &x2DeviceAddr, aclDataType::ACL_HIFLOAT8, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2TensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建scale2 aclTensor
      ret = CreateAclTensor<float>(scale2Data, scale2Shape, &scale2DeviceAddr, aclDataType::ACL_FLOAT, &scale2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scale2TensorPtr(scale2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> scale2DeviceAddrPtr(scale2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建y aclTensor
      ret = CreateAclTensor<float>(yData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &yRef);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> yTensorPtr(yRef, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建group_list aclTensor
      ret = CreateAclTensor<int64_t>(groupListData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64, &groupList);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> groupListTensorPtr(groupList, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建scale1 aclTensor
      ret = CreateAclTensor<float>(scale1Data, scale1Shape, &scale1DeviceAddr, aclDataType::ACL_FLOAT, &scale1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scale1TensorPtr(scale1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> scale1DeviceAddrPtr(scale1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;

      // 3. 调用CANN算子库API
      // 调用aclnnQuantGroupedMatmulInplaceAdd第一段接口
      ret = aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize(x1, x2, scale1, scale2, groupList, yRef, groupListType, groupSize, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void* workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // 调用aclnnQuantGroupedMatmulInplaceAdd第二段接口
      ret = aclnnQuantGroupedMatmulInplaceAdd(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantGroupedMatmulInplaceAdd failed. ERROR: %d\n", ret); return ret);

      // 4. （固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
      auto size = GetShapeSize(yShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), size * sizeof(uint32_t), yDeviceAddr,
                        size * sizeof(uint32_t), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t j = 0; j < size; j++) {
          LOG_PRINT("result[%ld] is: %f\n", j, resultData[j]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnQuantGroupedMatmulInplaceAddTest(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantGroupedMatmulInplaceAddTest failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
    ```

