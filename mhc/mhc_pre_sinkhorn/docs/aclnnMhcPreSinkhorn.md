aclnnMhcPreSinkhorn

aclnnMhcPreSinkhorn

## 产品支持情况


| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                  |    ×    |
| <term>Atlas 推理系列产品</term>                          |    ×    |
| <term>Atlas 训练系列产品</term>                          |    ×    |
| <term>Atlas 200/300/500 推理产品</term>                  |    ×    |

## 功能说明

- 接口功能：基于一系列计算得到MHC架构中hidden层的$\mathbf{H}'_{\text{res}}$和$\mathbf{H}_{\text{post}}$投影矩阵以及Attention或MLP层的输入矩阵$\mathbf{h}_{\text{in}}$。对$\mathbf{H}'_{\text{res}}$矩阵执行Sinkhorn迭代归一化变换，最终得到双随机矩阵$\mathbf{H}_{\text{res}}$；支持输出中间计算结果，用于反向梯度计算。包括sigmoid计算之后的$\mathbf{H^{pre}_l}$矩阵、$\vec{x^{'}_{l}}$与$\mathbf{\varphi}$矩阵乘的结果，输入x的RmsNorm结果$\mathbf{\vec{x^{'}_{l}}}$、迭代过程中的中间归一化结果和$\mathbf{normOut}$和求和结果$\mathbf{sumOut}$。
- 计算公式

$$
\begin{aligned}
\vec{x^{'}_{l}} &= \frac{1}{\sqrt{\frac{1}{d} \sum_{\dim=-2,\text{keepdim}=\text{True}} x_i^2 + \epsilon}}\\
H^{pre}_l &= \alpha^{pre}_{l} ·(\vec{x^{'}_{l}}\varphi^{pre}_{l}) + b^{pre}_{l}\\
H^{post}_l &= \alpha^{post}_{l} ·(\vec{x^{'}_{l}}\varphi^{post}_{l}) + b^{post}_{l}\\
H^{res}_l &= \alpha^{res}_{l} ·(\vec{x^{'}_{l}}\varphi^{res}_{l}) + b^{res}_{l}\\
H^{pre}_l &= \sigma (H^{pre}_{l})\\
H^{post}_l &= 2\sigma (H^{post}_{l})\\
h_{in} &=\vec{x_{l}}H^{pre}_l
\end{aligned}
$$

- 将$\mathbf{H^{res}_l}$作为输入，Sinkhorn变换共执行$\mathbf{numIters}$次迭代，迭代过程中生成中间归一化结果$\mathbf{normOut}[k]$和求和结果$\mathbf{sumOut}[k]$，最终输出最后一次迭代的$\mathbf{normOut}$作为变换结果。

第一次迭代（初始化）：

$$
\begin{aligned}
    \mathbf{normOut}[0] &= \text{softmax}(\mathbf{H^{res}_l},  \dim=-1) + \epsilon, \\
    \mathbf{sumOut}[1] &= \sum_{\dim=-2,\text{keepdim}=\text{True}} \mathbf{normOut}[0] + \epsilon, \\
    \mathbf{normOut}[1] &= \frac{\mathbf{normOut}[0]}{\mathbf{sum\_out}[1]}, \\
\end{aligned}
$$

第$i$次迭代（$i = 1, 2, \dots, \mathbf({num\_iters}-1)$）：

$$
\begin{aligned}
    \mathbf{sumOut}[2i] &= \sum_{\dim=-1,\text{keepdim}=\text{True}} \mathbf{normOut}[2i-1] + \epsilon, \\
    \mathbf{normOut}[2i] &= \frac{\mathbf{normOut}[2i-1]}{\mathbf{sum\_out}[2i]}, \\
    \mathbf{sumOut}[2i+1] &= \sum_{\dim=-2,\text{keepdim}=\text{True}} \mathbf{normOut}[2i] + \epsilon, \\
    \mathbf{normOut}[2i+1] &= \frac{\mathbf{normOut}[2i]}{\mathbf{sum\_out}[2i+1]}, \\
\end{aligned}
$$

### 最终输出

$$
\mathbf{normOut}[2 \times \mathbf{num\_iters} - 1]
$$

$$
\mathbf{sumOut}[2 \times \mathbf{num\_iters} - 1]
$$

---

### 符号说明


| 符号                                       | 含义                                              |
| ------------------------------------------ | ------------------------------------------------- |
| $\mathbf{x}$                               | 输入张量（mHC层的$\mathbf{H}'_{\text{res}}$矩阵） |
| $\epsilon$                                 | 防除零参数（对应入参`eps`）                       |
| $\text{softmax}(\cdot, \dim=-1)$           | 在最后一维执行softmax归一化                       |
| $\sum_{\dim=d,\text{keepdim}=\text{True}}$ | 在指定维度$d$上求和并保持维度                     |
| $\mathbf{normOut}[k]$                      | 第$k$步归一化中间结果                             |
| $\mathbf{sumOut}[k]$                       | 第$k$步求和中间结果                               |
| $\mathbf{numIters}$                        | 迭代次数（入参）                                  |

## 函数原型

算子采用两段式接口调用：需先调用`aclnnMhcPreSinkhornGetWorkspaceSize`获取计算所需的Device侧内存大小，再调用`aclnnMhcPreSinkhorn`执行实际计算。

```c++
aclnnStatus aclnnMhcPreSinkhornGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *phi,
    const aclTensor *alpha,
    const aclTensor *bias,
    int              hcMult,
    int              numIters,
    double           hcEps,
    double           normEps,
    bool             outFlag,
    aclTensor        *hin,
    aclTensor        *hPost,
    aclTensor        *hRes,
    aclTensor        *hPre,
    aclTensor        *hcBeforeNorm,
    aclTensor        *invRms,
    aclTensor        *sumOut,
    aclTensor        *normOut,
    uint64_t         **workspaceSize,
    aclOpExecutor **executor)
```

```c++
aclnnStatus aclnnMhcPreSinkhorn(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
```

## aclnnMhcPreSinkhornGetWorkspaceSize

### 参数说明


| 参数名       | 输入/输出 | 描述                                                | 使用说明                           | 数据类型 | 数据格式 | 维度(shape)                            | 非连续Tensor |
| :----------- | :-------- | :-------------------------------------------------- | :--------------------------------- | :------- | :------- | :------------------------------------- | :----------- |
| x            | 输入      | 待计算数据，表示网络中mHC层的输入数据               | 支持空Tensor                       | BFLOAT16 | ND       | (bs, seq_len, n, c)                    | √           |
| phi          | 输入      | mHC的参数矩阵                                       | 支持空Tensor                       | FLOAT32  | ND       | (n * n + 2 * n, n * c)                 | √           |
| alpha        | 输入      | mHC的缩放参数                                       | 支持空Tensor                       | FLOAT32  | ND       | (3)                                    | √           |
| bias         | 输入      | mHC的bias参数                                       | 支持空Tensor                       | FLOAT32  | ND       | (n * n + 2 * n)                        | √           |
| hcMult       | 输入      | 残差流数量，HC维度大小                              | 必选参数，`当前仅支持4`            | INT32    | -        | -                                      | -            |
| numIters     | 输入      | 表示sinkhorn算法迭代次数                            | 必选参数，`当前仅支持20`           | INT32    | -        | -                                      | -            |
| hcEps        | 输入      | $H_{pre}$的sigmoid后的eps参数                       | 必选参数，建议值：1e-6             | FLOAT64  | -        | -                                      | -            |
| normEps      | 输入      | RmsNorm的防除零参数                                 | 必选参数，建议值：1e-6             | FLOAT64  | -        | -                                      | -            |
| needGrad     | 输入      | 是否需要输出额外属性                                | 必选参数，建议值为true             | BOOL     | -        | -                                      | -            |
| hin          | 输出      | 输出的h_in作为Atten/MLP层的输入                     | 必选参数                           | BFLOAT16 | ND       | (bs, seq_len, c)                       | x            |
| hPost        | 输出      | 输出的mHC的h_post变换矩阵                           | 必选参数                           | FLOAT32  | ND       | (bs, seq_len, n)                       | x            |
| hRes         | 输出      | 输出的mHC的h_res变换矩阵（未做sinkhorn变换）        | 必选参数                           | FLOAT32  | ND       | (bs, seq_len, n * n)                   | x            |
| hPre         | 输出      | 需要反向时输出，做完sigmoid计算之后的hPre矩阵       | 可选输出，根据needGrad决定是否输出 | FLOAT32  | ND       | (bs, seq_len, n)                       | x            |
| hcBeforeNorm | 输出      | 需要反向时输出，x与phi矩阵乘的结果                  | 可选输出，根据needGrad决定是否输出 | FLOAT32  | ND       | (bs, seq_len, n*n + 2*n)               | x            |
| invRms       | 输出      | 需要反向时输出，RmsNorm计算得到的1/r                | 可选输出，根据needGrad决定是否输出 | FLOAT32  | ND       | (bs, seq_len, 1)                       | x            |
| sumOut       | 输出      | 需要反向时输出，每一次迭代的colSum/rowSum结果       | 可选输出，根据needGrad决定是否输出 | FLOAT32  | ND       | (sk_iter_count * 2, bs, seq_len, n)    | x            |
| normOut      | 输出      | 需要反向时输出，每一次colSum/rowSum迭代后的comb结果 | 可选输出，根据needGrad决定是否输出 | FLOAT32  | ND       | (sk_iter_count * 2, bs, seq_len, n, n) | x            |

### 返回值

返回`aclnnStatus`状态码，第一段接口主要完成入参校验，异常场景如下：


| 返回值                  | 错误码 | 描述                                                                             |
| :---------------------- | :----- | :------------------------------------------------------------------------------- |
| ACLNN_ERR_PARAM_NULLPTR | 161001 | 必选参数或者输出是空指针。                                                       |
| ACLNN_ERR_PARAM_INVALID | 161002 | 1.等输入变量的数据类型和数据格式不在支持的范围内；2. numIters不为20；3. n值非4。 |
| ACLNN_ERR_RUNTIME_ERROR | 361001 | 调用NPU Runtime接口申请内存/创建Tensor失败。                                     |

## aclnnMhcPreSinkhorn

### 参数说明


| 参数名        | 输入/输出 | 描述                                                                         |
| :------------ | :-------- | :--------------------------------------------------------------------------- |
| workspace     | 输入      | Device侧申请的workspace内存地址，需与第一段接口返回的workspaceSize匹配。     |
| workspaceSize | 输入      | Device侧workspace内存大小，由`aclnnMhcPreSinkhornGetWorkspaceSize`接口返回。 |
| executor      | 输入      | 算子执行器，由第一段接口创建，包含计算流程和参数信息。                       |
| stream        | 输入      | 指定执行计算任务的Stream，需提前创建并绑定Device。                           |

### 返回值

返回`aclnnStatus`状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

### 确定性计算

- aclnnMhcPreSinkhorn默认采用确定性实现，相同输入多次调用结果一致。

### 规格约束


| 规格项   | 规格               | 规格说明                               |
| :------- | :----------------- | :------------------------------------- |
| numIters | 20                 | 迭代次数超出该范围会返回参数无效错误。 |
| n        | 4                  | 目前只支持4。                          |
| C        | [1280, 1920, 2560] | 目前只支持这三个数                     |

## 调用示例

以下为C++调用示例，需结合AscendCL环境编译运行，具体流程参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_mhc_pre_sinkhorn.h"

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
  CHECK_RET(ret(ret == ACL_SUCCESS, 
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

int main() {
  // ========== 1. 初始化环境 ==========
  int32_t device_id = 0;  // 根据实际Device ID调整
  aclrtContext context = nullptr;
  aclrtStream stream = nullptr;

  int ret = InitAcl(device_id, context, stream);
  CHECK_RET(ret == 0, 
            LOG_PRINT("InitAcl failed, error: %d\n", ret); 
            return -1);

  // ========== 2. 构造输入/输出参数 ==========
  // 输入h_res的形状：B=1, S=1024, n=4 → (1024,4,4)（合并B*S为T=1024）
  std::vector<int64_t> h_res_shape = {1024, 4, 4};
  int64_t h_res_size = GetShapeSize(h_res_shape);
  std::vector<float> h_res_host_data(h_res_size, 1.0f);  // 初始化输入数据为1.0

  // 输出h_res_sinkhorn的形状与h_res一致
  std::vector<int64_t> output_shape = h_res_shape;
  void* output_device_addr = nullptr;
  aclTensor* output_tensor = nullptr;

  // 输入h_res的Device Tensor
  void* h_res_device_addr = nullptr;
  aclTensor* h_res_tensor = nullptr;
  ret = CreateAclTensor(h_res_host_data, h_res_shape, h_res_device_addr, h_res_tensor);
  CHECK_RET(ret == 0, 
            LOG_PRINT("Create h_res_tensor failed\n"); 
            return -1);

  // 输出h_res_sinkhorn的Device Tensor（仅申请内存，无初始数据）
  ret = aclrtMalloc(&output_device_addr, GetShapeSize(output_shape)*sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("Malloc output failed, error: %d\n", ret); 
            return -1);
  output_tensor = (aclCreateTensor(
      output_shape.data(), output_shape.size(),
      ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND,
      output_shape.data(), output_shape.size(),
      output_device_addr
  );

  // MhcPreSinkhorn算子参数
  float eps = 1e-6f;       // 防除零参数
  int64_t num_iters = 20;  // 迭代次数
  int out_flag = 0;        // 输出标志位（当前版本仅支持0）

  // ========== 3. 调用第一段接口：获取Workspace大小 ==========
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;

  aclnnStatus aclnn_ret = aclnnMhcPreSinkhornGetWorkspaceSize(
      h_res_tensor,
      eps,
      num_iters,
      out_flag,
      output_tensor,
      nullptr,  // norm_out（outFlag=0时为nullptr）
      nullptr,  // sum_out（outFlag=0时为nullptr）
      &workspace_size,
      &executor
  );
  CHECK_RET(aclnn_ret == ACL_SUCCESS, 
            LOG_PRINT("aclnnMhcPreSinkhornGetWorkspaceSize failed, error: %d\n", aclnn_ret); 
            return -1);

  // ========== 4. 申请Workspace内存 ==========
  void* workspace_addr = nullptr;
  if (workspace_size > 0) {
    ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, 
              LOG_PRINT("aclrtMalloc workspace failed, error: %d\n", ret); 
              return -1);
  }

  // ========== 5. 调用第二段接口：执行MhcPreSinkhorn计算 ==========
  aclnn_ret = aclnnMhcPreSinkhorn(
      workspace_addr,
      workspace_size,
      executor,
      stream
  );
  CHECK_RET(aclnn_ret == ACL_SUCCESS, 
            LOG_PRINT("aclnnMhcPreSinkhorn failed, error: %d\n", aclnn_ret); 
            return -1);

  // ========== 6. 同步Stream并打印结果 ==========
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, 
            LOG_PRINT("aclrtSynchronizeStream failed, error: %d\n", ret); 
            return -1);

  LOG_PRINT("MhcPreSinkhorn compute success!\n");
  LOG_PRINT("Output tensor data: ");
  PrintTensorData(output_shape, output_device_addr);

  // ========== 7. 释放资源 ==========
  // 销毁Tensor
  aclDestroyTensor(h_res_tensor);
  aclDestroyTensor(output_tensor);

  // 释放Device内存
  aclrtFree(h_res_device_addr);
  aclrtFree(output_device_addr);
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
