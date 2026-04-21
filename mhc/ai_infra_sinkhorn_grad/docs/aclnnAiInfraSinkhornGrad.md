# aclnnAiInfraSinkhornGrad

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>昇腾910_95 AI处理器</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|    √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|    √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |
|<term>Atlas 200/300/500 推理产品</term>|      ×     |

## 功能说明

- 接口功能：对mHC架构中双随机矩阵Hres的梯度进行sinkhorn变换的反向计算得到输入H`res的梯度。


## 计算公式：

设当前梯度为 $\mathbf{grad}_{\text{curr}} = \mathbf{grad}_{\text{output}}$，共进行 $\mathbf{num\_iters}$ 次迭代。

#### 前 $\mathbf{num\_iters} - 1$ 次迭代  
对迭代编号 $i = \mathbf{num\_iters} - 1, \mathbf{num\_iters} - 2, \dots, 1$，依次执行：

$$
\begin{aligned}
    \mathbf{dot\_prod}_{2i+1} &= \sum_{\dim=-2,\text{keepdim}=\text{True}} \left( \mathbf{grad}_{\text{curr}} \cdot \mathbf{norm\_out}_{2i+1} \right) \\
    \\
    \mathbf{grad}_{\text{curr}} &\gets \frac{{ \mathbf{grad}_{\text{curr}}} - \mathbf{dot\_prod}_{2i+1}} { \mathbf{sum\_out}_{2i+1} }, \\
    \\
    \mathbf{dot\_prod}_{2i} &= \sum_{\dim=-1,\text{keepdim}=\text{True}} \left( \mathbf{grad}_{\text{curr}} \cdot \mathbf{num\_out}_{2i} \right), \\
    \\
    \mathbf{grad}_{\text{curr}}  &\gets  \frac{{ \mathbf{grad}_{\text{curr}}} - \mathbf{dot\_prod}_{2i}} { \mathbf{sum\_out}_{2i} }, \\
\end{aligned}
$$

#### 最后一次迭代  

$$
\begin{aligned}
    \mathbf{dot\_prod}_{1} &= \sum_{\dim=-2,\text{keepdim}=\text{True}} \left( \mathbf{grad}_{\text{curr}} \cdot \mathbf{norm\_out}_{1} \right), \\
    \\
    \mathbf{grad}_{\text{curr}} &\gets \frac{{ \mathbf{grad}_{\text{curr}}} - \mathbf{dot\_prod}_{1}} { \mathbf{sum\_out}_{1} }, \\
    \\
    \mathbf{dot\_prod}_{0} &= \sum_{\dim=-1,\text{keepdim}=\text{True}} \left( \mathbf{grad}_{\text{curr}} \cdot \mathbf{num\_out}_{0} \right), \\
    \\
    \mathbf{grad}_{\text{input}}  &\gets  \left( \mathbf{grad}_{\text{curr}} -  \mathbf{dot\_prod}_{0} \right) \cdot { \mathbf{norm\_out}_{0} }, \\
\end{aligned}
$$

---

###  符号说明：

| 符号 | 含义 |
|------|------|
| $\mathbf{grad}_{\text{curr}}$ | 当前梯度（迭代中间变量） |
| $\mathbf{grad}_{\text{input}}$ | 最终输入梯度（输出） |
| $\mathbf{num\_out}_k$ | 第 $k$ 个归一化方向向量（对应 `norm_out[k]`） |
| $\mathbf{sum\_out}_k$ | 对应的缩放系数（对应 `sum_out[k]`） |
| $\cdot$ | 点积（内积） |
| $\sum_{\dim=d,\text{keepdim}=\text{True}}$ | 在指定维度上求和并保持维度 |



## 函数原型

算子分为两段式接口，必须先调用“aclnnAiInfraSinkhornGradGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnAiInfraSinkhornGrad”接口执行计算。

```c++
aclnnStatus aclnnAiInfraSinkhornGradGetWorkspaceSize(
    const aclTensor *gradOutput,
    const aclTensor *normOut,
    const aclTensor *sumOut,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
```

```c++
aclnnStatus aclnnAiInfraSinkhornGrad(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
```

## aclnnAiInfraSinkhornGradGetWorkspaceSize

- **参数说明:**
  
    <table style="undefined;table-layout: fixed; width: 1550px">
        <colgroup>
            <col style="width: 220px">
            <col style="width: 120px">
            <col style="width: 300px">  
            <col style="width: 400px">  
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
            <td>gradOutput</td>
            <td>输入</td>
            <td>Sinkhorn变换输出的H_res的梯度</td>
            <td>数据类型与key/queryIndex/keyIndex保持一致。</td>
            <td>FLOAT32 </td>
            <td>ND</td>
            <td>(B,S,n,n)、(T,n,n)</td>
            <td>√</td>
        </tr>
        <tr>
            <td>normOut</td>
            <td>输入</td>
            <td>Sinkhorn变换正向计算保存的中间norm结果</td>
            <td>数据类型与query/queryIndex/keyIndex保持一致。</td>
            <td>FLOAT32 </td>
            <td>ND</td>
            <td>[2*num_iters,n,n,B,S]、[2*num_iters,n,n,T]</td>
            <td>√</td>
        </tr>
        <tr>
            <td>sumOut</td>
            <td>输入</td>
            <td>Sinkhorn变换正向计算保存的中间sum结果</td>
            <td>数据类型与query/key/keyIndex保持一致。</td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>[2*num_iters,n,B,S]、[2*num_iters,n,T]</td>
            <td>√</td>
        </tr>
        <tr>
            <td>out</td>
            <td>输出</td>
            <td>Sinkhorn变换的输入的H_res的梯度</td>
            <td>-</td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>(B,S,n,n)、(T,n,n)</td>
            <td>-</td>
        </tr>
        </tbody>
    </table>

- **返回值：**

    返回aclnnStatus状态码。

    第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
    <col style="width: 319px">
    <col style="width: 144px">
    <col style="width: 671px">
    </colgroup>
        <thead>
            <th>返回值</th>
            <th>错误码</th>
            <th>描述</th>
        </thead>
        <tbody>
            <tr>
                <td>ACLNN_ERR_PARAM_NULLPTR</td>
                <td>161001</td>
                <td>必选参数或者输出是空指针。</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_PARAM_INVALID</td>
                <td>161002</td>
                <td>gradOutput、normOut、sumOut等输入变量的数据类型和数据格式不在支持的范围内。</td>
            </tr>
            <tr>
                <td>ACLNN_ERR_RUNTIME_ERROR</td>
                <td>361001</td>
                <td>API内存调用npu runtime的接口异常。</td>
            </tr>
        </tbody>
    </table>

## aclnnAiInfraSinkhornGrad

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
    <col style="width: 144px">
    <col style="width: 125px">
    <col style="width: 700px">
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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnAiInfraSinkhornGradGetWorkspaceSize获取。</td>
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
  - aclnnAiInfraSinkhornGrad默认确定性实现。
- 公共约束
    - 入参为空的场景处理：
        - gradOutput等输入参数不能为空Tensor，否则返回失败。
- 规格约束
    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 300px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>规格项</th>
                <th>规格</th>
                <th>规格说明</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>num_iters</td>
            <td>1~100</td>
            <td>num_iters支持1~100范围以内</td>
        </tr>
        <tr>
            <td>n</td>
            <td>4， 6 ，8</td>
            <td>n值目前支持4， 6， 8</td>
        </tr>
        </tbody>
    </table>
- 典型值
    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 300px">
        </colgroup>
        <thead>
            <tr>
                <th>规格项</th>
                <th>典型值</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>T或 B*S</td>
            <td>1024/2048/4096</td>
        </tr>
        <tr>
            <td>n</td>
            <td>n = 4</td>
        </tr>
        <tr>
            <td>num_iters</td>
            <td>num_iters = 20</td>
        </tr>
        </tbody>
    </table>

## 调用示例

PTA调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include <algorithm>
#include "acl/acl.h"
#include "aclnnop/aclnn_ai_infra_sinkhorn_grad.h"

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

// 将Device侧Tensor数据拷贝到Host侧并打印
void PrintTensorData(const std::vector<int64_t>& shape, void* device_addr) {
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

  // 打印前10个元素（示例）
  LOG_PRINT("Tensor data (first 10 elements): ");
  for (int i = 0; i < std::min((int64_t)10, size); ++i) {
    LOG_PRINT("%f ", host_data[i]);
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

// 创建空的Device侧aclTensor（仅申请内存，无初始数据）
int CreateEmptyAclTensor(
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

  // 3. 计算Tensor的strides（连续Tensor）
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }

  // 4. 创建aclTensor
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

int main() {
  // ========== 1. 初始化环境 ==========
  int32_t device_id = 0;  // 根据实际Device ID调整
  aclrtContext context = nullptr;
  aclrtStream stream = nullptr;
  
  int ret = InitAcl(device_id, context, stream);
  CHECK_RET(ret == 0, 
            LOG_PRINT("InitAcl failed, error: %d\n", ret); 
            return -1);

  // ========== 2. 构造反向算子输入/输出参数 ==========
  // 关键参数配置（与正向保持一致）
  int64_t num_iters = 20;  // 迭代次数，需和正向一致
  int64_t n = 4;           // 维度n，支持4/6/8
  int64_t T = 1024;        // T = B*S

  // ---------------- 输入1：gradOutput (H_res的输出梯度) ----------------
  std::vector<int64_t> grad_output_shape = {T, n, n};  // (1024,4,4)
  int64_t grad_output_size = GetShapeSize(grad_output_shape);
  std::vector<float> grad_output_host_data(grad_output_size, 0.1f);  // 模拟梯度数据
  void* grad_output_device_addr = nullptr;
  aclTensor* grad_output_tensor = nullptr;
  ret = CreateAclTensor(grad_output_host_data, grad_output_shape, grad_output_device_addr, grad_output_tensor);
  CHECK_RET(ret == 0, 
            LOG_PRINT("Create grad_output_tensor failed\n"); 
            return -1);

  // ---------------- 输入2：normOut (正向保存的norm中间结果) ----------------
  std::vector<int64_t> norm_out_shape = {2 * num_iters, n, n, T};  // (40,4,4,1024)
  int64_t norm_out_size = GetShapeSize(norm_out_shape);
  std::vector<float> norm_out_host_data(norm_out_size, 1.0f);  // 模拟正向输出的norm数据
  void* norm_out_device_addr = nullptr;
  aclTensor* norm_out_tensor = nullptr;
  ret = CreateAclTensor(norm_out_host_data, norm_out_shape, norm_out_device_addr, norm_out_tensor);
  CHECK_RET(ret == 0, 
            LOG_PRINT("Create norm_out_tensor failed\n"); 
            return -1);

  // ---------------- 输入3：sumOut (正向保存的sum中间结果) ----------------
  std::vector<int64_t> sum_out_shape = {2 * num_iters, n, T};  // (40,4,1024)
  int64_t sum_out_size = GetShapeSize(sum_out_shape);
  std::vector<float> sum_out_host_data(sum_out_size, 1.0f);  // 模拟正向输出的sum数据
  void* sum_out_device_addr = nullptr;
  aclTensor* sum_out_tensor = nullptr;
  ret = CreateAclTensor(sum_out_host_data, sum_out_shape, sum_out_device_addr, sum_out_tensor);
  CHECK_RET(ret == 0, 
            LOG_PRINT("Create sum_out_tensor failed\n"); 
            return -1);

  // ---------------- 输出：out (H_res的输入梯度) ----------------
  std::vector<int64_t> out_shape = grad_output_shape;  // 与gradOutput形状一致 (1024,4,4)
  void* out_device_addr = nullptr;
  aclTensor* out_tensor = nullptr;
  ret = CreateEmptyAclTensor(out_shape, out_device_addr, out_tensor);
  CHECK_RET(ret == 0, 
            LOG_PRINT("Create out_tensor failed\n"); 
            return -1);

  // ========== 3. 调用第一段接口：获取Workspace大小和executor ==========
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  
  aclnnStatus aclnn_ret = aclnnAiInfraSinkhornGradGetWorkspaceSize(
      grad_output_tensor,    // 输入：H_res的输出梯度
      norm_out_tensor,       // 输入：正向保存的norm中间结果
      sum_out_tensor,        // 输入：正向保存的sum中间结果
      out_tensor,            // 输出：H_res的输入梯度
      &workspace_size,       // 输出：所需workspace大小
      &executor              // 输出：op执行器
  );
  CHECK_RET(aclnn_ret == ACLNN_SUCCESS, 
            LOG_PRINT("aclnnAiInfraSinkhornGradGetWorkspaceSize failed, error: %d\n", aclnn_ret); 
            return -1);
  LOG_PRINT("Required workspace size: %lu bytes\n", workspace_size);

  // ========== 4. 申请Workspace内存 ==========
  void* workspace_addr = nullptr;
  if (workspace_size > 0) {
    ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, 
              LOG_PRINT("aclrtMalloc workspace failed, error: %d\n", ret); 
              return -1);
    LOG_PRINT("Allocate workspace success, addr: %p\n", workspace_addr);
  }

  // ========== 5. 调用第二段接口：执行AiInfraSinkhornGrad计算 ==========
  aclnn_ret = aclnnAiInfraSinkhornGrad(
      workspace_addr,    // workspace内存地址
      workspace_size,    // workspace大小
      executor,          // op执行器
      stream             // 执行流
  );
  CHECK_RET(aclnn_ret == ACLNN_SUCCESS, 
            LOG_PRINT("aclnnAiInfraSinkhornGrad failed, error: %d\n", aclnn_ret); 
            return -1);

  // ========== 6. 同步Stream并打印结果 ==========
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtSynchronizeStream failed, error: %d\n", ret); 
            return -1);

  LOG_PRINT("AiInfraSinkhornGrad compute success!\n");
  LOG_PRINT("Output grad input tensor data: ");
  PrintTensorData(out_shape, out_device_addr);

  // ========== 7. 释放资源 ==========
  // 销毁Tensor
  aclDestroyTensor(grad_output_tensor);
  aclDestroyTensor(norm_out_tensor);
  aclDestroyTensor(sum_out_tensor);
  aclDestroyTensor(out_tensor);

  // 释放Device内存
  aclrtFree(grad_output_device_addr);
  aclrtFree(norm_out_device_addr);
  aclrtFree(sum_out_device_addr);
  aclrtFree(out_device_addr);
  if (workspace_size > 0) {
    aclrtFree(workspace_addr);
  }

  // 销毁executor（重要：反向算子的executor需要手动销毁）
  if (executor != nullptr) {
    aclOpExecutorDestroy(executor);
    executor = nullptr;
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