# aclnnMhcSinkhornBackward

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|    ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|    ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |
|<term>Atlas 200/300/500 推理产品</term>|      ×     |

## 功能说明

- 接口功能：MhcSinkhornBackward是MhcSinkhorn的反向算子。mHC（Manifold-Constrained Hyper-Connections）架构中的MhcSinkhorn算子对输入矩阵做sinkhorn变换得到双随机矩阵$\mathbf{H}_{\text{res}}$，输出的双随机矩阵的所有元素≥0、每一行之和为1且每一列之和为1 (具有范数保持、组合封闭性和凸组合几何解释三大特性)。

## 计算公式

对mHC架构中双随机矩阵$\mathbf{H}_{\text{res}}$矩阵的梯度进行sinkhorn变换的反向计算得到输入$\mathbf{H}'_{\text{res}}$的梯度

- 前 $\mathbf{num\_iters}-1$ 次迭代:

$$
\begin{aligned}
    \mathbf{dot\_prod}_{2i+1} &= \sum_{\dim=-2,\text{keepdim}=\text{True}} (\mathbf{grad}_{curr}\ {⋅}\ \mathbf{norm\_out}_{2i+1}),  \\
    \mathbf{grad}_{curr} &← \frac{\mathbf{grad}_{curr} - \mathbf{dot\_prod}_{2i+1}}{\mathbf{sum\_out}_{2i+1}}, \\
    \mathbf{dot\_prod}_{2i} &= \sum_{\dim=-1,\text{keepdim}=\text{True}} (\mathbf{grad}_{curr}\ {⋅}\ \mathbf{norm\_out}_{2i}),  \\
    \mathbf{grad}_{curr} &← \frac{\mathbf{grad}_{curr} - \mathbf{dot\_prod}_{2i}}{\mathbf{sum\_out}_{2i}}, \\
\end{aligned}
$$

- 最后一次迭代：

$$
\begin{aligned}
    \mathbf{dot\_prod}_{1} &= \sum_{\dim=-2,\text{keepdim}=\text{True}} (\mathbf{grad}_{curr}\ {⋅}\ \mathbf{norm\_out}_{1}),  \\
    \mathbf{grad}_{curr} &← \frac{\mathbf{grad}_{curr} - \mathbf{dot\_prod}_{1}}{\mathbf{sum\_out}_{1}}, \\
    \mathbf{dot\_prod}_{0} &= \sum_{\dim=-1,\text{keepdim}=\text{True}} (\mathbf{grad}_{curr}\  {⋅}\  \mathbf{norm\_out}_{0}),  \\
    \mathbf{grad}_{input} &← ({\mathbf{grad}_{curr} - \mathbf{dot\_prod}_{0}})\  {⋅}\  \mathbf{norm\_out}_{0} \\
\end{aligned}
$$

### 🔍 符号说明

| 符号 | 含义 |
|------|------|
| $\mathbf{grad}_\text{curr}$ | 当前梯度（迭代中间变量） |
| $\mathbf{grad}_\text{input}$ | 最终输入梯度（输出） |
| $\mathbf{norm\_out}_\text{k}$ | 第$k$个归一化方向向量 |
| $\mathbf{sum\_out}_\text{k}$ | 对应的缩放系数 |
| $\mathbf{⋅}$ | 点积（内积） |
| $\sum_{\dim=d,\text{keepdim}=\text{True}}$ | 在指定维度$d$上求和并保持维度 |

## 函数原型

算子采用两段式接口调用：需先调用`aclnnMhcSinkhornBackwardGetWorkspaceSize`获取计算所需的Device侧内存大小，再调用`aclnnMhcSinkhornBackward`执行实际计算。

```c++
aclnnStatus aclnnMhcSinkhornBackwardGetWorkspaceSize(
    const aclTensor *gradOutput, 
    const aclTensor *normOut, 
    const aclTensor *sumOut, 
    aclTensor *out, 
    uint64_t *workspaceSize, 
    aclOpExecutor **executor)
```

```c++
aclnnStatus aclnnMhcSinkhornBackward(
    void *workspace, 
    uint64_t workspaceSize, 
    aclOpExecutor *executor,
    aclrtStream stream)
```

## aclnnMhcSinkhornBackwardGetWorkspaceSize

### 参数说明

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度(shape) | 非连续Tensor |
|:--- |:--- |:--- |:--- |:--- |:--- |:--- |:--- |
| gradOutput | 输入 | Sinkhorn变换输出的$\mathbf{H}_{\text{res}}$的梯度 | 不能为空Tensor | FLOAT32 | ND | (B,S,n,n)、(T,n,n) | √ |
| normOut| 输入 | Sinkhorn变换正向计算保存的中间norm结果| 不支持空Tensor | FLOAT32 | ND | [2\*num_iters\*n\*align_n\*B\*S]、[2\*num_iters\*n\*align_n\*T] | √ |
| sumOut| 输入 | Sinkhorn变换正向计算保存的中间sum结果 | 不支持空Tensor | FLOAT32 | ND | [2\*num_iters\*align_n\*B\*S]、[2\*num_iters\*align_n\*T] | √ |
| out| 输出 | Sinkhorn变换的输入的$\mathbf{H}_{\text{res}}$的梯度 | - | FLOAT32 | ND | (B,S,n,n)、(T,n,n) | - |
| workspaceSize | 输出 | 计算所需的Device侧workspace内存大小（字节） | 由算子内部计算得出，用于后续申请内存 | UINT64 | - | - | - |
| executor | 输出 | 算子执行器，包含计算流程和参数信息 | 需传递给第二段接口使用 | aclOpExecutor* | - | - | - |

### 返回值

返回`aclnnStatus`状态码，第一段接口完成入参校验，出现以下场景时报错：

| 返回值 | 错误码 | 描述 |
|:--- |:--- |:--- |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 必选参数或者输出是空指针。 |
| ACLNN_ERR_PARAM_INVALID | 161002 | gradOutput、normOut、sumOut等输入变量的数据类型和数据格式不在支持的范围内。 |
| ACLNN_ERR_RUNTIME_ERROR | 361001 | API内存调用npu runtime的接口异常。 |

## aclnnMhcSinkhornBackward

### 参数说明

| 参数名 | 输入/输出 | 描述 |
|:--- |:--- |:--- |
| workspace | 输入 | 在Device侧申请的workspace内存地址。 |
| workspaceSize | 输入 | Device侧workspace内存大小，由`aclnnMhcSinkhornBackwardGetWorkspaceSize`获取。 |
| executor | 输入 | op执行器，包含了算子计算流程。 |
| stream | 输入 | 指定执行任务的Stream。|

### 返回值

返回`aclnnStatus`状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

### 确定性计算

- aclnnMhcSinkhornBackward默认确定性实现。

### 公共约束

- 入参为空的场景处理：gradOutput等输入参数不能为空Tensor，否则返回失败。

### 规格约束

| 规格项 | 规格 | 规格说明 |
|:--- |:--- |:--- |
| numIters | 1~100 | 迭代次数超出该范围会返回参数无效错误。 |
| n | 4、6、8 | 输入Tensor最后两维的大小（矩阵维度）仅支持这三个值。 |
| align_n | 8 | n按block_size(32B, 对于FP32下为8)对齐后的大小。 |
| 维度数 | 3/4 | 输入gradOutput支持3维(T,n,n)或4维(B,S,n,n)，其他维度数不支持。 |

## 调用示例

以下为C++调用示例，需结合AscendCL环境编译运行，具体流程参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_sinkhorn_backward.h"

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

// 计算Tensor形状对应的总元素数
int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t size = 1;
  for (int64_t dim : shape) {
    size *= dim;
  }
  return size;
}

// 向上对齐到align的倍数
int64_t CeilAlign(int64_t value, int64_t align) {
  return (value + align - 1) / align * align;
}

// 将Device侧Tensor数据拷贝到Host侧并打印
void PrintTensorData(const std::vector<int64_t>& shape, void* device_addr, const char* name) {
  int64_t size = GetShapeSize(shape);
  std::vector<float> host_data(size, 0.0f);

  // Device -> Host 数据拷贝
  aclError ret = aclrtMemcpy(
      host_data.data(), size * sizeof(float),
      device_addr, size * sizeof(float),
      ACL_MEMCPY_DEVICE_TO_HOST
  );
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("Memcpy device to host failed, error: %d\n", ret); 
            return);

  // 打印前10个元素
  LOG_PRINT("%s (first 10 elements): ", name);
  for (int i = 0; i < std::min((int64_t)10, size); ++i) {
    LOG_PRINT("%.6f ", host_data[i]);
  }
  LOG_PRINT("\n");
}

// 初始化AscendCL环境（Device/Context/Stream）
int InitAcl(int32_t device_id, aclrtContext& context, aclrtStream& stream) {
  // 1. 初始化ACL
  aclError ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclInit failed, error: %d\n", ret); 
            return -1);

  // 2. 设置Device
  ret = aclrtSetDevice(device_id);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtSetDevice failed, error: %d\n", ret); 
            return -1);

  // 3. 创建Context
  ret = aclrtCreateContext(&context, device_id);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtCreateContext failed, error: %d\n", ret); 
            return -1);

  // 4. 设置当前Context
  ret = aclrtSetCurrentContext(context);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtSetCurrentContext failed, error: %d\n", ret); 
            return -1);

  // 5. 创建Stream
  ret = aclrtCreateStream(&stream);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtCreateStream failed, error: %d\n", ret); 
            return -1);

  return 0;
}

// 创建Device侧aclTensor（含数据拷贝）
int CreateAclTensor(
    const std::vector<float>& host_data,
    const std::vector<int64_t>& shape,
    void*& device_addr,
    aclTensor*& tensor) {
  // 1. 计算内存大小
  int64_t size = GetShapeSize(shape) * sizeof(float);

  // 2. 申请Device侧内存
  aclError ret = aclrtMalloc(&device_addr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtMalloc failed, error: %d\n", ret); 
            return -1);

  // 3. Host -> Device 数据拷贝
  ret = aclrtMemcpy(
      device_addr, size,
      host_data.data(), size,
      ACL_MEMCPY_HOST_TO_DEVICE
  );
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtMemcpy failed, error: %d\n", ret); 
            return -1);

  // 4. 计算Tensor的strides（连续Tensor）
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }

  // 5. 创建aclTensor
  tensor = aclCreateTensor(
      shape.data(), shape.size(),
      ACL_FLOAT, strides.data(), 0,
      ACL_FORMAT_ND, shape.data(), shape.size(),
      device_addr
  );
  CHECK_RET(tensor != nullptr, 
            LOG_PRINT("aclCreateTensor failed\n"); 
            return -1);

  return 0;
}

// 创建输出Tensor（仅申请内存，无初始数据）
int CreateOutputTensor(
    const std::vector<int64_t>& shape,
    void*& device_addr,
    aclTensor*& tensor) {
  int64_t size = GetShapeSize(shape) * sizeof(float);
  
  aclError ret = aclrtMalloc(&device_addr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtMalloc output failed, error: %d\n", ret); 
            return -1);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }

  tensor = aclCreateTensor(
      shape.data(), shape.size(),
      ACL_FLOAT, strides.data(), 0,
      ACL_FORMAT_ND, shape.data(), shape.size(),
      device_addr
  );
  CHECK_RET(tensor != nullptr, 
            LOG_PRINT("aclCreateTensor output failed\n"); 
            return -1);

  return 0;
}

int main() {
  // ========== 1. 初始化环境 ==========
  int32_t device_id = 0;  // 根据实际Device ID调整
  aclrtContext context = nullptr;
  aclrtStream stream = nullptr;

  int ret = InitAcl(device_id, context, stream);
  CHECK_RET(ret == 0, 
            LOG_PRINT("InitAcl failed, error: %d\n", ret); 
            return -1);

  LOG_PRINT("ACL environment initialized successfully!\n");

  // ========== 2. 定义算子参数 ==========
  // grad_y形状可以是(T, N, N)或(B, S, N, N)
  // 这里使用T形状作为示例
  int64_t T = 128;       // total length
  int64_t N = 4;         // n size (N x N矩阵)
  int64_t num_iters = 20; // Sinkhorn迭代次数
  
  // 计算对齐后的N大小（对齐到8）
  int64_t align_unit = 8;
  int64_t n_align_size = CeilAlign(N, align_unit);
  
  LOG_PRINT("Parameters: T=%lld, N=%lld, num_iters=%lld, n_align_size=%lld\n", 
            T, N, num_iters, n_align_size);

  // ========== 3. 构造输入/输出张量 ==========
  // 输入grad_y形状：(T, N, N)
  std::vector<int64_t> grad_y_shape = {T, N, N};
  int64_t grad_y_size = GetShapeSize(grad_y_shape);
  std::vector<float> grad_y_host_data(grad_y_size, 1.0f);
  for (int64_t i = 0; i < grad_y_size; ++i) {
    grad_y_host_data[i] = 0.1f * static_cast<float>(i % 100);  // 填充测试数据
  }

  // 输入norm形状：[2 * num_iters * T * N * n_align_size]
  int64_t norm_size = 2 * num_iters * T * N * n_align_size;
  std::vector<int64_t> norm_shape = {norm_size};
  std::vector<float> norm_host_data(norm_size, 0.5f);
  for (int64_t i = 0; i < norm_size; ++i) {
    norm_host_data[i] = 1.0f / (1.0f + static_cast<float>(i % 1000) * 0.001f);  // 填充测试数据
  }

  // 输入sum形状：[2 * num_iters * T * n_align_size]
  int64_t sum_size = 2 * num_iters * T * n_align_size;
  std::vector<int64_t> sum_shape = {sum_size};
  std::vector<float> sum_host_data(sum_size, 1.0f);
  for (int64_t i = 0; i < sum_size; ++i) {
    sum_host_data[i] = 1.0f + static_cast<float>(i % 500) * 0.01f;  // 填充测试数据
  }

  // 输出grad_input形状与grad_y相同：(T, N, N)
  std::vector<int64_t> grad_input_shape = grad_y_shape;

  // 创建Device侧Tensor
  void* grad_y_device_addr = nullptr;
  aclTensor* grad_y_tensor = nullptr;
  ret = CreateAclTensor(grad_y_host_data, grad_y_shape, grad_y_device_addr, grad_y_tensor);
  CHECK_RET(ret == 0, LOG_PRINT("Create grad_y_tensor failed\n"); return -1);

  void* norm_device_addr = nullptr;
  aclTensor* norm_tensor = nullptr;
  ret = CreateAclTensor(norm_host_data, norm_shape, norm_device_addr, norm_tensor);
  CHECK_RET(ret == 0, LOG_PRINT("Create norm_tensor failed\n"); return -1);

  void* sum_device_addr = nullptr;
  aclTensor* sum_tensor = nullptr;
  ret = CreateAclTensor(sum_host_data, sum_shape, sum_device_addr, sum_tensor);
  CHECK_RET(ret == 0, LOG_PRINT("Create sum_tensor failed\n"); return -1);

  void* grad_input_device_addr = nullptr;
  aclTensor* grad_input_tensor = nullptr;
  ret = CreateOutputTensor(grad_input_shape, grad_input_device_addr, grad_input_tensor);
  CHECK_RET(ret == 0, LOG_PRINT("Create grad_input_tensor failed\n"); return -1);

  LOG_PRINT("All tensors created successfully!\n");
  LOG_PRINT("grad_y shape: [%lld, %lld, %lld]\n", grad_y_shape[0], grad_y_shape[1], grad_y_shape[2]);
  LOG_PRINT("norm shape: [%lld]\n", norm_shape[0]);
  LOG_PRINT("sum shape: [%lld]\n", sum_shape[0]);
  LOG_PRINT("grad_input shape: [%lld, %lld, %lld]\n", grad_input_shape[0], grad_input_shape[1], grad_input_shape[2]);

  // ========== 4. 调用第一段接口：获取Workspace大小 ==========
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;

  aclnnStatus aclnn_ret = aclnnMhcSinkhornBackwardGetWorkspaceSize(
      grad_y_tensor,
      norm_tensor,
      sum_tensor,
      grad_input_tensor,
      &workspace_size,
      &executor
  );
  CHECK_RET(aclnn_ret == ACL_SUCCESS, 
            LOG_PRINT("aclnnMhcSinkhornBackwardGetWorkspaceSize failed, error: %d\n", aclnn_ret); 
            return -1);

  LOG_PRINT("Workspace size: %lu bytes\n", workspace_size);

  // ========== 5. 申请Workspace内存 ==========
  void* workspace_addr = nullptr;
  if (workspace_size > 0) {
    ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, 
              LOG_PRINT("aclrtMalloc workspace failed, error: %d\n", ret); 
              return -1);
  }

  // ========== 6. 调用第二段接口：执行MhcSinkhornBackward计算 ==========
  aclnn_ret = aclnnMhcSinkhornBackward(
      workspace_addr,
      workspace_size,
      executor,
      stream
  );
  CHECK_RET(aclnn_ret == ACL_SUCCESS, 
            LOG_PRINT("aclnnMhcSinkhornBackward failed, error: %d\n", aclnn_ret); 
            return -1);

  // ========== 7. 同步Stream ==========
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtSynchronizeStream failed, error: %d\n", ret); 
            return -1);

  LOG_PRINT("MhcSinkhornBackward compute success!\n");

  // ========== 8. 打印结果 ==========
  PrintTensorData(grad_input_shape, grad_input_device_addr, "grad_input");

  // ========== 9. 释放资源 ==========
  // 销毁Tensor
  aclDestroyTensor(grad_y_tensor);
  aclDestroyTensor(norm_tensor);
  aclDestroyTensor(sum_tensor);
  aclDestroyTensor(grad_input_tensor);

  // 释放Device内存
  aclrtFree(grad_y_device_addr);
  aclrtFree(norm_device_addr);
  aclrtFree(sum_device_addr);
  aclrtFree(grad_input_device_addr);
  if (workspace_size > 0) {
    aclrtFree(workspace_addr);
  }

  // 销毁Stream/Context，重置Device
  aclrtDestroyStream(stream);
  aclrtDestroyContext(context);
  aclrtResetDevice(device_id);
  aclFinalize();

  LOG_PRINT("All resources released successfully!\n");
  return 0;
}
```
