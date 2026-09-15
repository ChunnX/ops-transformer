# aclnnMhcPreSplitPostRes

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

- 算子功能：mhc_pre_split_post_res算子是mhc_pre算子的一部分，负责计算mHC（Manifold-Constrained Hyper Connection）架构中x2输入对应的post分支和comb/Res分支，输出h_post和h_res两个变换矩阵。具体流程为：将输入x展平并转FP32后计算RMSNorm的InvRMS因子，执行线性投影得到x_hat，然后切分为post分支和comb分支分别计算h_post和h_res。

- 计算公式：
  $$
  X_{flat} = \mathrm{reshape}(x, [B,S,N \cdot D])
  $$
  $$
  \mathrm{inv\_rms} = \frac{1}{\sqrt{\frac{1}{N \cdot D}\sum_{k} X_{flat,k}^2 + \mathrm{norm\_eps}}}
  $$
  $$
  X_{hat} = (X_{flat} \cdot \phi^T) \odot \mathrm{inv\_rms}
  $$
  $$
  [X_{post}, X_{comb}] = \mathrm{split}(X_{hat}, [N, N^2])
  $$
  $$
  h_{post} = 2 \cdot \sigma(\alpha_0 \cdot X_{post} + b_{post})
  $$
  $$
  h_{res} = \alpha_1 \cdot \mathrm{reshape}(X_{comb}, [B,S,N,N]) + \mathrm{reshape}(b_{comb}, [N,N])
  $$

---

## 函数原型

算子执行接口为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用"aclnnMhcPreSplitPostResGetWorkspaceSize"接口获取入参并根据计算流程计算所需workspace大小，再调用"aclnnMhcPreSplitPostRes"接口执行计算。

```c++
aclnnStatus aclnnMhcPreSplitPostResGetWorkspaceSize(
    const aclTensor     *x,
    const aclTensor     *phi,
    const aclTensor     *alpha,
    const aclTensor     *bias,
    double               norm_eps,
    const aclTensor     *h_post_out,
    const aclTensor     *h_res_out,
    uint64_t            *workspaceSize,
    aclOpExecutor      **executor)
```
```c++
aclnnStatus aclnnMhcPreSplitPostRes(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnMhcPreSplitPostResGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1550px">
        <colgroup>
            <col style="width: 220px">
            <col style="width: 120px">
            <col style="width: 500px">
            <col style="width: 212px">
            <col style="width: 100px">
            <col style="width: 190px">
            <col style="width: 145px">
            </colgroup>
        <thead>
        <tr>
            <th>参数名</th>
            <th>输入/输出</th>
            <th>描述说明</th>
            <th>数据类型</th>
            <th>数据格式</th>
            <th>非连续Tensor</th>
        </tr></thead>
        <tbody>
        <tr>
            <td>x</td>
            <td>输入</td>
            <td>待计算的数据，表示网络中mHC层的输入数据x2。<br>维度shape支持[B,S,n,D]，或者为[T,n,D]。不支持空Tensor。</td>
            <td>FLOAT16、BFLOAT16</td>
            <td>ND</td>
            <td>√</td>
        </tr>
        <tr>
            <td>phi</td>
            <td>输入</td>
            <td>mHC的变换矩阵phi。<br>维度shape支持[n<sup>2</sup>+n, n*D]。不支持空Tensor。</td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>√</td>
        </tr>
        <tr>
            <td>alpha</td>
            <td>输入</td>
            <td>mHC的缩放参数。<br>维度shape支持[2]。不支持空Tensor。</td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>√</td>
        </tr>
        <tr>
            <td>bias</td>
            <td>输入</td>
            <td>mHC的bias参数。<br>维度shape支持[n<sup>2</sup>+n]。不支持空Tensor。</td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>√</td>
        </tr>
        <tr>
            <td>norm_eps</td>
            <td>输入</td>
            <td>RMSNorm防除0参数，默认值为1e-6。</td>
            <td>DOUBLE</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>h_post_out</td>
            <td>输出</td>
            <td>mHC的h_post变换矩阵。<br>维度shape支持[B,S,n]，或者为[T,n]。不支持空Tensor。</td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>×</td>
        </tr>
        <tr>
            <td>h_res_out</td>
            <td>输出</td>
            <td>mHC的comb分支输出矩阵。<br>维度shape支持[B,S,n,n]，或者为[T,n,n]。不支持空Tensor。</td>
            <td>FLOAT32</td>
            <td>ND</td>
            <td>×</td>
        </tr>
        <tr>
            <td>workspaceSize</td>
            <td>输出</td>
            <td>返回用户需要在Device侧申请的workspace大小</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td>executor</td>
            <td>输出</td>
            <td>返回op执行器，包含了算子计算流程</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        </tbody>
    </table>
- **返回值：**

  返回aclnnStatus状态码。

## aclnnMhcPreSplitPostRes

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 598px"><colgroup>
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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnMhcPreSplitPostResGetWorkspaceSize获取。</td>
        </tr>
        <tr>
        <td>executor</td>
        <td>输入</td>
        <td>op执行器，包含了算子计算流程。</td>
        </tr>
        <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的AscendCL stream流。</td>
        </tr>
    </tbody>
    </table>
- **返回值：**

  返回aclnnStatus状态码。

## 约束说明

### 确定性计算

- MhcPreSplitPostRes 算子默认采用确定性实现，相同输入多次调用结果一致。

### 公共约束
- 输入约束：
   - 所有输入参数不能为空，否则返回失败。
   - 所有输入/输出Tensor的数据格式仅支持`ACL_FORMAT_ND`。
   - Workspace内存需在Device侧申请，且大小需严格匹配第一段接口返回值。
   - 非连续Tensor无需提前转为连续，算子内部自动处理。

### 规格约束

| 规格项 | 规格 | 规格说明 |
|:--- |:--- |:--- |
| T或B*S | 1~512K | B*S 或T支持1~512K范围以内。|
| n | 4 | n为固定值4。|
| D | 2560/5120 | D支持2560或5120。|

### 典型值

| 规格项 | 典型值 |
|:--- |:--- |
| T或B*S | 4 |
| n | 4 |
| D | 2560 |

## 调用示例

以下为C++调用示例，需结合AscendCL环境编译运行。

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_mhc_pre_split_post_res.h"

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
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  const int64_t B = 1, S = 4, n = 4, D = 2560;
  const int64_t phiRows = n * n + n;  // 20
  std::vector<int64_t> xShape = {B, S, n, D};
  std::vector<int64_t> phiShape = {phiRows, n * D};
  std::vector<int64_t> alphaShape = {2};
  std::vector<int64_t> biasShape = {phiRows};
  std::vector<int64_t> hPostShape = {B, S, n};
  std::vector<int64_t> hResShape = {B, S, n, n};

  void *xAddr = nullptr, *phiAddr = nullptr, *alphaAddr = nullptr, *biasAddr = nullptr;
  void *hPostAddr = nullptr, *hResAddr = nullptr;
  aclTensor *x = nullptr, *phi = nullptr, *alpha = nullptr, *bias = nullptr;
  aclTensor *hPost = nullptr, *hRes = nullptr;

  std::vector<uint16_t> xHostData(GetShapeSize(xShape), 0x3C00);
  std::vector<float> phiHostData(GetShapeSize(phiShape), 0.01f);
  std::vector<float> alphaHostData = {1.0f, 1.0f};
  std::vector<float> biasHostData(GetShapeSize(biasShape), 0.0f);
  std::vector<float> hPostHostData(GetShapeSize(hPostShape), 0.0f);
  std::vector<float> hResHostData(GetShapeSize(hResShape), 0.0f);

  // 创建输入aclTensor
  ret = CreateAclTensor(xHostData, xShape, &xAddr, aclDataType::ACL_BF16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(phiHostData, phiShape, &phiAddr, aclDataType::ACL_FLOAT, &phi);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(alphaHostData, alphaShape, &alphaAddr, aclDataType::ACL_FLOAT, &alpha);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(biasHostData, biasShape, &biasAddr, aclDataType::ACL_FLOAT, &bias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建输出aclTensor
  ret = CreateAclTensor(hPostHostData, hPostShape, &hPostAddr, aclDataType::ACL_FLOAT, &hPost);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(hResHostData, hResShape, &hResAddr, aclDataType::ACL_FLOAT, &hRes);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  double normEps = 1e-6;
  // 调用aclnnMhcPreSplitPostRes第一段接口
  ret = aclnnMhcPreSplitPostResGetWorkspaceSize(x, phi, alpha, bias, normEps, hPost, hRes,
                                                       &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMhcPreSplitPostResGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnMhcPreSplitPostRes第二段接口
  ret = aclnnMhcPreSplitPostRes(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMhcPreSplitPostRes failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧
  auto hPostSize = GetShapeSize(hPostShape);
  std::vector<float> hPostResult(hPostSize, 0);
  ret = aclrtMemcpy(hPostResult.data(), hPostResult.size() * sizeof(float), hPostAddr,
                    hPostSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy h_post result failed. ERROR: %d\n", ret); return ret);

  auto hResSize = GetShapeSize(hResShape);
  std::vector<float> hResResult(hResSize, 0);
  ret = aclrtMemcpy(hResResult.data(), hResResult.size() * sizeof(float), hResAddr,
                    hResSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy h_res result failed. ERROR: %d\n", ret); return ret);

  for (int64_t i = 0; i < hPostSize; i++) {
      LOG_PRINT("hPostResult[%ld] is: %f\n", i, hPostResult[i]);
  }

  for (int64_t i = 0; i < hResSize; i++) {
      LOG_PRINT("hResResult[%ld] is: %f\n", i, hResResult[i]);
  }

  // 6. 释放aclTensor
  aclDestroyTensor(x);
  aclDestroyTensor(phi);
  aclDestroyTensor(alpha);
  aclDestroyTensor(bias);
  aclDestroyTensor(hPost);
  aclDestroyTensor(hRes);
  // 7. 释放device资源
  aclrtFree(xAddr);
  aclrtFree(phiAddr);
  aclrtFree(alphaAddr);
  aclrtFree(biasAddr);
  aclrtFree(hPostAddr);
  aclrtFree(hResAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
