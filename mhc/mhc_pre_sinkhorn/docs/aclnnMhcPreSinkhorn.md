# aclnnMhcPreSinkhorn

## 产品支持情况


| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    x    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                  |    ×    |
| <term>Atlas 推理系列产品</term>                          |    ×    |
| <term>Atlas 训练系列产品</term>                          |    ×    |

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

  - 最终输出

  $$
  \mathbf{normOut}[2 \times \mathbf{num\_iters} - 1]
  $$

  $$
  \mathbf{sumOut}[2 \times \mathbf{num\_iters} - 1]
  $$

  - 符号说明

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

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用`aclnnMhcPreSinkhornGetWorkspaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnMhcPreSinkhorn`执行实际计算。

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
    aclTensor       *hin,
    aclTensor       *hPost,
    aclTensor       *hRes,
    aclTensor       *hPre,
    aclTensor       *hcBeforeNorm,
    aclTensor       *invRms,
    aclTensor       *sumOut,
    aclTensor       *normOut,
    uint64_t       **workspaceSize,
    aclOpExecutor  **executor)
```

```c++
aclnnStatus aclnnMhcPreSinkhorn(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnMhcPreSinkhornGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1435px"><colgroup>
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 250px">
    <col style="width: 250px">
    <col style="width: 100px">
    <col style="width: 120px">
    <col style="width: 200px">
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
        <td>x（aclTensor*）</td>
        <td>输入</td>
        <td>待计算数据，表示网络中mHC层的输入数据。</td>
        <td>支持空Tensor。</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>(bs, seq_len, n, c)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>phi（aclTensor*）</td>
        <td>输入</td>
        <td>mHC的参数矩阵。</td>
        <td>支持空Tensor。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>(n * n + 2 * n, n * c)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>alpha（aclTensor*）</td>
        <td>输入</td>
        <td>mHC的缩放参数。</td>
        <td>支持空Tensor。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>(3)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>bias（aclTensor*）</td>
        <td>输入</td>
        <td>mHC的bias参数。</td>
        <td>支持空Tensor。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>(n * n + 2 * n)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>hcMult（int64_t）</td>
        <td>输入</td>
        <td>残差流数量，HC维度大小。</td>
        <td>当前仅支持4。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>numIters（int64_t）</td>
        <td>输入</td>
        <td>表示sinkhorn算法迭代次数。</td>
        <td>当前仅支持20。</td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>hcEps（double）</td>
        <td>输入</td>
        <td>$H_{pre}$的sigmoid后的eps参数。</td>
        <td>建议值：1e-6。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>normEps（double）</td>
        <td>输入</td>
        <td>RmsNorm的防除零参数。</td>
        <td>建议值：1e-6。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>needGrad（bool）</td>
        <td>输入</td>
        <td>是否需要输出额外属性。</td>
        <td>建议值为true。</td>
        <td></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>hin（aclTensor*）</td>
        <td>输出</td>
        <td>输出的h_in作为Atten/MLP层的输入。</td>
        <td>-</td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>(bs, seq_len, c)</td>
        <td>×</td>
      </tr>
      <tr>
        <td>hPost（aclTensor*）</td>
        <td>输出</td>
        <td>输出的mHC的h_post变换矩阵。</td>
        <td>-</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>(bs, seq_len, n)</td>
        <td>×</td>
      </tr>
      <tr>
        <td>hRes（aclTensor*）</td>
        <td>输出</td>
        <td>输出的mHC的h_res变换矩阵。</td>
        <td>-</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>(bs, seq_len, n * n)</td>
        <td>×</td>
      </tr>
      <tr>
        <td>hPre（aclTensor*）</td>
        <td>可选输出</td>
        <td>需要反向时输出，做完sigmoid计算之后的hPre矩阵。</td>
        <td>根据needGrad决定是否输出。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>(bs, seq_len, n)</td>
        <td>×</td>
      </tr>
      <tr>
        <td>hcBeforeNorm（aclTensor*）</td>
        <td>可选输出</td>
        <td>需要反向时输出，x与phi矩阵乘的结果。</td>
        <td>根据needGrad决定是否输出。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>(bs, seq_len, n*n + 2*n)</td>
        <td>×</td>
      </tr>
      <tr>
        <td>invRms（aclTensor*）</td>
        <td>可选输出</td>
        <td>需要反向时输出，RmsNorm计算得到的1/r。</td>
        <td>根据needGrad决定是否输出。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>(bs, seq_len, 1)</td>
        <td>×</td>
      </tr>
      <tr>
        <td>sumOut（aclTensor*）</td>
        <td>可选输出</td>
        <td>需要反向时输出，每一次迭代的colSum/rowSum结果。</td>
        <td>根据needGrad决定是否输出。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>(sk_iter_count * 2, bs, seq_len, n)</td>
        <td>×</td>
      </tr>
      <tr>
        <td>normOut（aclTensor*）</td>
        <td>可选输出</td>
        <td>需要反向时输出，每一次colSum/rowSum迭代后的comb结果。</td>
        <td>根据needGrad决定是否输出。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>(sk_iter_count * 2, bs, seq_len, n, n)</td>
        <td>×</td>
      </tr>
      <tr>
        <td>workspaceSize（uint64_t**）</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  
  <table style="undefined;table-layout: fixed;width: 1202px"><colgroup>
  <col style="width: 262px">
  <col style="width: 121px">
  <col style="width: 819px">
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
      <td>传入参数是必选输入，输出或者必选属性，且是空指针。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>输入变量的数据类型和数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>numIters不为20。</td>
    </tr>
    <tr>
      <td>n值非4。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>调用NPU Runtime接口申请内存/创建Tensor失败。</td>
    </tr>
  </tbody>
  </table>

## aclnnMhcPreSinkhorn

- **参数说明**
  <table style="undefined;table-layout: fixed; width: 1154px"><colgroup>
  <col style="width: 153px">
  <col style="width: 121px">
  <col style="width: 880px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnMhcPreSinkhornGetWorkspaceSize获取。</td>
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

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算

  - aclnnMhcPreSinkhorn默认采用确定性实现，相同输入多次调用结果一致。

- 规格约束

  | 规格项   | 规格               | 规格说明                               |
  | :------- | :----------------- | :------------------------------------- |
  | numIters | 20                 | 迭代次数超出该范围会返回参数无效错误。 |
  | n        | 4                  | 目前只支持4。                          |
  | C        | [1280, 1920, 2560] | 目前只支持这三个数                     |

## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

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
