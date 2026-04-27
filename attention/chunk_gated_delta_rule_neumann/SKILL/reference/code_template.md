# 测试代码完整模板

## 基础模板

以下是调用 aclnn 单算子的完整 C++ 测试代码模板。

```cpp
#include <iostream>
#include <vector>
#include <cstring>
#include "acl/acl.h"
#include "aclnnop/aclnn_${算子名}.h"

// 宏定义
#define CHECK_RET(cond, return_expr) \
  do { if (!(cond)) { return_expr; } } while (0)

#define LOG_PRINT(message, ...) \
  do { printf(message, ##__VA_ARGS__); } while (0)

// 计算 shape 大小
int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) { shapeSize *= i; }
  return shapeSize;
}

// ACL 初始化
int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  
  return 0;
}

// 创建 aclTensor（通用模板）
template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, 
                    const std::vector<int64_t>& shape, 
                    void** deviceAddr, 
                    aclDataType dataType, 
                    aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  
  // 分配设备内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  
  // 拷贝数据到设备
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
  
  // 计算 strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  
  // 创建 tensor
  *tensor = aclCreateTensor(
      shape.data(), 
      shape.size(), 
      dataType, 
      strides.data(), 
      0, 
      aclFormat::ACL_FORMAT_ND, 
      shape.data(), 
      shape.size(), 
      *deviceAddr
  );
  
  return 0;
}

// 创建空 aclTensor（用于输出）
int CreateEmptyAclTensor(const std::vector<int64_t>& shape, 
                         void** deviceAddr, 
                         aclDataType dataType, 
                         aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(uint16_t);  // BF16 = 2 bytes
  
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }
  
  *tensor = aclCreateTensor(
      shape.data(), 
      shape.size(), 
      dataType, 
      strides.data(), 
      0, 
      aclFormat::ACL_FORMAT_ND, 
      shape.data(), 
      shape.size(), 
      *deviceAddr
  );
  
  return 0;
}

int main() {
  // ==================== 1. 初始化 ====================
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  // ==================== 2. 定义参数 ====================
  // 参考 tests/ut/op_kernel/test_${算子名}.cpp 的默认参数
  int64_t B = 2;
  int64_t T = 128;     // B * seqlen
  int64_t nk = 4;
  int64_t nv = 4;
  int64_t dk = 64;
  int64_t dv = 64;
  float scale = 1.0f;
  // ... 其他参数
  
  // ==================== 3. 创建输入 Tensor ====================
  // 示例：query tensor (T, nk, dk) BF16
  std::vector<int64_t> queryShape = {T, nk, dk};
  int64_t querySize = GetShapeSize(queryShape);
  
  // 准备 host 数据（BF16 编码）
  // 注意：应使用合理输入数据，而非全 1.0
  std::vector<uint16_t> queryHostData(querySize);
  // ... 填充数据（可从 Python 导出）
  
  void* queryDeviceAddr = nullptr;
  aclTensor* queryTensor = nullptr;
  ret = CreateAclTensor(queryHostData, queryShape, &queryDeviceAddr, ACL_BF16, &queryTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  // 创建其他输入 tensor...
  // key, value, beta, initial_state, actual_seq_lengths, g (可选)
  
  // ==================== 4. 创建输出 Tensor ====================
  std::vector<int64_t> outShape = {T, nv, dv};  // 根据算子输出 shape
  void* outDeviceAddr = nullptr;
  aclTensor* outTensor = nullptr;
  ret = CreateEmptyAclTensor(outShape, &outDeviceAddr, ACL_BF16, &outTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  // final_state tensor...
  
  // ==================== 5. 获取 workspace 大小 ====================
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  
  // 调用算子特定的 GetWorkspaceSize 函数
  // 参数顺序参考 aclnn_${算子名}.h 头文件
  ret = aclnn${算子名}GetWorkspaceSize(
      queryTensor,
      keyTensor,
      valueTensor,
      // ... 其他输入参数
      outTensor,
      finalStateTensor,
      &workspaceSize,
      &executor
  );
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclnn${算子名}GetWorkspaceSize failed. ERROR: %d\n", ret); 
            return ret);
  
  LOG_PRINT("Workspace size: %lu bytes\n", workspaceSize);
  
  // ==================== 6. 分配 workspace ====================
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc workspace failed.\n"); return ret);
  }
  
  // ==================== 7. 执行算子 ====================
  LOG_PRINT("Calling aclnn${算子名}...\n");
  ret = aclnn${算子名}(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclnn${算子名} failed. ERROR: %d\n", ret); 
            return ret);
  
  // ==================== 8. 同步等待 ====================
  LOG_PRINT("Synchronizing stream...\n");
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Synchronize failed.\n"); return ret);
  
  // ==================== 9. 获取结果 ====================
  int64_t outSize = GetShapeSize(outShape);
  std::vector<uint16_t> outHostData(outSize);
  ret = aclrtMemcpy(outHostData.data(), outSize * sizeof(uint16_t), 
                    outDeviceAddr, outSize * sizeof(uint16_t), 
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  
  // 打印部分结果
  LOG_PRINT("Output sample: ");
  for (int i = 0; i < 5 && i < outSize; i++) {
    // BF16 解码（简化）
    float val = reinterpret_cast<float&>(outHostData[i]);  // 实际需要正确解码
    LOG_PRINT("%.4f ", val);
  }
  LOG_PRINT("\n");
  
  // ==================== 10. 释放资源 ====================
  aclDestroyTensor(queryTensor);
  aclDestroyTensor(keyTensor);
  aclDestroyTensor(valueTensor);
  aclDestroyTensor(outTensor);
  // ... 释放其他 tensor
  
  aclrtFree(queryDeviceAddr);
  aclrtFree(keyDeviceAddr);
  aclrtFree(valueDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceAddr) aclrtFree(workspaceAddr);
  // ... 释放其他 device memory
  
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  
  LOG_PRINT("Test completed successfully!\n");
  return 0;
}
```

## BF16 数据处理

### BF16 编码说明

BF16 (bfloat16) 使用 16 位表示浮点数：
- 1 bit sign
- 8 bits exponent
- 7 bits mantissa

```cpp
// float32 → BF16 转换（截断 mantissa）
uint16_t FloatToBf16(float f) {
  uint32_t* p = reinterpret_cast<uint32_t*>(&f);
  return static_cast<uint16_t>(*p >> 16);
}

// BF16 → float32 转换
float Bf16ToFloat(uint16_t bf) {
  uint32_t f = static_cast<uint32_t>(bf) << 16;
  return reinterpret_cast<float&>(f);
}
```

### 生成随机 BF16 数据

```cpp
#include <random>

void GenerateRandomBf16(std::vector<uint16_t>& data, float min = -1.0f, float max = 1.0f) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(min, max);
  
  for (auto& val : data) {
    float f = dis(gen);
    val = FloatToBf16(f);
  }
}
```

## 编译命令模板

```bash
#!/bin/bash

source ${ASCEND_HOME_PATH}/bin/setenv.bash

ARCH_INFO=$(uname -m)
OP_NAME="chunk_gated_delta_rule"  # 替换为实际算子名

INCLUDE_PATH="${ASCEND_HOME_PATH}/include"
EAGER_INCLUDE_OPP_ACLNNOP_PATH="${ASCEND_HOME_PATH}/${ARCH_INFO}-linux/include/aclnnop"
EAGER_LIBRARY_PATH="${ASCEND_HOME_PATH}/${ARCH_INFO}-linux/lib64"
CUST_INCLUDE_PATH="${ASCEND_OPP_PATH}/vendors/custom_transformer/op_api/include"
CUST_LIBRARY_PATH="${ASCEND_OPP_PATH}/vendors/custom_transformer/op_api/lib"

g++ examples/test_aclnn_${OP_NAME}.cpp \
    -I ${CUST_INCLUDE_PATH} -I ${CUST_INCLUDE_PATH}/aclnnop \
    -I ${INCLUDE_PATH} -I ${EAGER_INCLUDE_OPP_ACLNNOP_PATH} \
    -L ${CUST_LIBRARY_PATH} -L ${EAGER_LIBRARY_PATH} \
    -lcust_opapi -lascendcl -lnnopbase -lc_sec \
    -Wl,-rpath=${CUST_LIBRARY_PATH} -Wl,-rpath=${EAGER_LIBRARY_PATH} \
    -o test_aclnn_${OP_NAME} \
    -std=c++17 -O2
```

## 运行命令模板

```bash
#!/bin/bash

source ${ASCEND_HOME_PATH}/bin/setenv.bash

TILING_LIBRARY_PATH="${ASCEND_OPP_PATH}/vendors/custom_transformer/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64"
CUST_LIBRARY_PATH="${ASCEND_OPP_PATH}/vendors/custom_transformer/op_api/lib"
EAGER_LIBRARY_PATH="${ASCEND_HOME_PATH}/${ARCH_INFO}-linux/lib64"

export LD_LIBRARY_PATH=${TILING_LIBRARY_PATH}:${CUST_LIBRARY_PATH}:${EAGER_LIBRARY_PATH}:${LD_LIBRARY_PATH}

./test_aclnn_${OP_NAME}
```