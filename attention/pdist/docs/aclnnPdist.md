# aclnnPdist

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：Pdist算子计算行向量两两之间的p-范数距离，输出为上三角行优先的一维向量，等价于 `torch.nn.functional.pdist`。

  两段式 aclnn 入口直接复用 CANN 内置 `aclnnPdist`（libopapi.so，本仓不再提供手写 op_api 实现与 `aclnnop/aclnn_pdist.h` 头文件）；本仓 vendor 包仅提供算子注册、tiling 与预编译二进制，覆盖内置实现的 dispatch。参数校验行为以内置实现为准（下述错误码表沿用本仓 kernel 合同的语义约束，供参考）。

  主要计算过程为：
  1. 按上三角行优先顺序枚举行对 (i, j)，0 ≤ i < j ≤ N-1。
  2. 逐维计算绝对差 d_k = |x_i,k − x_j,k|。
  3. 按p分支归约：p=0统计不相等维度数；p=+inf取最大绝对差；0<p<inf先求幂和再取1/p次方。
  4. 以idx = i*(2N-i-1)/2 + (j-i-1)写出结果。

- 计算公式：

  $$
  y_{idx}=\left\|x_{i}-x_{j}\right\|_{p}=\left(\sum_{k=0}^{M-1}\left|x_{i,k}-x_{j,k}\right|^{p}\right)^{1/p},\quad idx=\frac{i\cdot(2N-i-1)}{2}+(j-i-1),\quad 0\le i<j\le N-1
  $$

## 函数原型

算子执行接口为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnPdistGetWorkspaceSize"接口获取入参并根据计算流程计算所需 workspace 大小，再调用"aclnnPdist"接口执行计算。

注意：与标准 aclnn 约定不同，内置 `aclnnPdist` 的范数阶参数 `p` 为 **float**（非 double），调用方传 double 会导致 ABI 不匹配（p 被读取为错误位模式）；torch_extension 侧已在 csrc 中显式窄化。

```c++
aclnnStatus aclnnPdistGetWorkspaceSize(
    const aclTensor  *x,
    float             p,
    aclTensor        *out,
    uint64_t         *workspaceSize,
    aclOpExecutor   **executor);
```

```c++
aclnnStatus aclnnPdist(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream);
```

## aclnnPdistGetWorkspaceSize

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
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>输入张量。</td>
      <td><ul><li>N：2~65535，至少为2（至少一个行对）。</li><li>M：1~65535，支持泛化，超过片上内存时自动分块规约。</li><li>要求行主序连续。</li></ul></td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>(N,M)</td>
      <td>×</td>
     </tr>
     <tr>
      <td>p（float）</td>
      <td>输入</td>
      <td>范数阶。</td>
      <td><ul><li>p≥0或+inf，默认2.0。</li><li>p=0：统计不相等维度数（NaN维度计为不相等）。</li><li>p=+inf：最大绝对差。</li><li>0<p<inf：幂和的1/p次方。</li><li>p<0或NaN报参数错误。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
     </tr>
     <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>两两距离向量，上三角行优先排列。</td>
      <td><ul><li>shape必须为(N*(N-1)/2,)。</li><li>dtype必须与x一致。</li></ul></td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>(N*(N-1)/2,)</td>
      <td>×</td>
     </tr>
     <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在 Device 侧申请的 workspace 大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回 op 执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     </tbody>
    </table>

- **返回值：**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

    第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed;width: 1155px">
        <colgroup>
            <col style="width: 319px">
            <col style="width: 144px">
            <col style="width: 671px">
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
                <td>参数中存在非法的nullptr。</td>
            </tr>
            <tr>
                <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
                <td rowspan="4">161002</td>
                <td>输入的数据类型不满足支持类型（仅支持FLOAT16/FLOAT32，且out与x同dtype）。</td>
            </tr>
            <tr>
                <td>x、out必选输入/输出未传。</td>
            </tr>
            <tr>
                <td>x不为二维(N, M)或N&lt;2、M&lt;1，out的shape不为(N*(N-1)/2,)。</td>
            </tr>
            <tr>
                <td>p&lt;0或p为NaN。</td>
            </tr>
        </tbody>
    </table>

## aclnnPdist

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
    <col style="width: 184px">
    <col style="width: 134px">
    <col style="width: 833px">
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
      <td>在 Device 侧申请的 workspace 内存地址。</td>
     </tr>
     <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在 Device 侧申请的 workspace 大小，由第一段接口 aclnnPdistGetWorkspaceSize 获取。</td>
     </tr>
     <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op 执行器，包含了算子计算流程。</td>
     </tr>
     <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的 Stream 流。</td>
     </tr>
    </tbody>
    </table>

- **返回值：**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

  - 确定性说明：aclnnPdist默认确定性实现。每个行对由唯一lane以固定求和次序计算，静态切分。

  - 规格约束

    <table style="undefined;table-layout: fixed; width: 909px"><colgroup>
    <col style="width: 125px">
    <col style="width: 182px">
    <col style="width: 602px">
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
      <td>N</td>
      <td>2~65535</td>
      <td>行数，至少为2。</td>
    </tr>
    <tr>
      <td>M</td>
      <td>1~65535</td>
      <td>每行维度数，支持分块规约。</td>
    </tr>
    <tr>
      <td>p</td>
      <td>[0, +inf)</td>
      <td>p≥0或+inf；p<0或NaN为参数错误。</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>FLOAT16/FLOAT32</td>
      <td>输入输出同dtype；fp16中间计算使用FP32累加，出口窄化。</td>
    </tr>
    </tbody>
    </table>

  - 特殊约束
    - x必须为行主序连续的二维张量。
    - NaN语义：p=0时NaN维度计为不相等；p=+inf与一般p（非0/0.5/1/2）含NaN维度的行对输出NaN；p∈{0.5, 1, 2}按IEEE算术自然传播。
    - 输出上溢行为与PyTorch对齐：fp16输出超出[0, 65504]时上溢为inf，不属验收范围。

## 调用示例

调用示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。完整用例（fp32全部p分支、多M分块、fp16）见[test_aclnn_pdist.cpp](../examples/test_aclnn_pdist.cpp)。

```c++
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>
#include "securec.h"
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"

// aclnnPdist 由 CANN 内置 libopapi.so 提供；本仓不提供 aclnnop/aclnn_pdist.h，
// 编译环境无该头文件时可直接声明两段式入口（注意 p 为 float）。
#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnPdistGetWorkspaceSize(const aclTensor *x, float p, aclTensor *out,
                                       uint64_t *workspaceSize, aclOpExecutor **executor);
aclnnStatus aclnnPdist(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                       aclrtStream stream);

#ifdef __cplusplus
}
#endif

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

int main() {
  int32_t deviceId = 0;
  aclrtContext context;
  aclrtStream stream;
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = aclrtCreateContext(&context, deviceId);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = aclrtSetCurrentContext(context);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = aclrtCreateStream(&stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // x: [N, M] fp32, p = 2.0 (Euclidean); y: [N*(N-1)/2] fp32
  int64_t N = 64;
  int64_t M = 16;
  double p = 2.0;
  std::vector<int64_t> xShape = {N, M};
  std::vector<int64_t> yShape = {N * (N - 1) / 2};

  void* xDeviceAddr = nullptr;
  void* yDeviceAddr = nullptr;
  aclTensor* x = nullptr;
  aclTensor* y = nullptr;

  std::vector<float> xHostData(N * M, 0.5f);
  std::vector<float> yHostData(yShape[0], 0.0f);

  auto xSize = N * M * sizeof(float);
  ret = aclrtMalloc(&xDeviceAddr, xSize, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = aclrtMemcpy(xDeviceAddr, xSize, xHostData.data(), xSize, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = aclrtMalloc(&yDeviceAddr, yShape[0] * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> strides = {1};
  x = aclCreateTensor(xShape.data(), 2, aclDataType::ACL_FLOAT, strides.data(), 0,
                      aclFormat::ACL_FORMAT_ND, xShape.data(), 2, xDeviceAddr);
  y = aclCreateTensor(yShape.data(), 1, aclDataType::ACL_FLOAT, strides.data(), 0,
                      aclFormat::ACL_FORMAT_ND, yShape.data(), 1, yDeviceAddr);

  // 两段式接口
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnPdistGetWorkspaceSize(x, p, y, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
  }

  ret = aclnnPdist(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<float> resultData(yShape[0], 0.0f);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), yDeviceAddr,
                    yShape[0] * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  printf("y[0] is: %f\n", resultData[0]);

  aclDestroyTensor(x);
  aclDestroyTensor(y);
  aclrtFree(xDeviceAddr);
  aclrtFree(yDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtDestroyContext(context);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
