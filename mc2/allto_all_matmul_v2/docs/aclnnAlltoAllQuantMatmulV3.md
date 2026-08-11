# aclnnAlltoAllQuantMatmulV3

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

完成 All-to-All 通信 + MX 量化 FP8 矩阵乘法的融合计算。在多卡场景下，通过 URMA all-to-all 在各卡间重新分布 A 矩阵的行和列，再与 B 矩阵做 MX FP8 量化矩阵乘法，输出 BF16 或 FP16 结果。

- 接口功能：All-to-All 通信 + MX FP8 矩阵乘法融合。
- 通信方式：AIV 通过 URMA 完成多卡 all-to-all，AIC 完成矩阵乘法计算。
- 计算公式：

  $$
  A_{\text{permuted}}[M_{\text{per\_rank}}, K_{\text{total}}] = \text{AllToAll}(A_{\text{local}}[M_{\text{total}}, K_a])
  $$

  $$
  Y[M_{\text{per\_rank}}, N] = A_{\text{permuted}} \times X2^T
  $$

  其中 $M_{\text{per\_rank}} = M_{\text{total}} / \text{world\_size}$，$K_{\text{total}} = K_a \times \text{world\_size}$。

## 函数原型

每个算子分为两段式接口，必须先调用 `aclnnAlltoAllQuantMatmulV3GetWorkspaceSize` 接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用 `aclnnAlltoAllQuantMatmulV3` 接口执行计算。

```cpp
aclnnStatus aclnnAlltoAllQuantMatmulV3GetWorkspaceSize(
    const aclTensor *context,
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *biasOptional,
    const aclTensor *x1ScaleOptional,
    const aclTensor *x2ScaleOptional,
    const char *group,
    int64_t worldSize,
    int64_t hcclBufferSize,
    int64_t x1QuantMode,
    int64_t x2QuantMode,
    int64_t groupSize,
    const char *commMode,
    int64_t precisionMode,
    const aclTensor *output,
    const aclTensor *alltoAllOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
```

```cpp
aclnnStatus aclnnAlltoAllQuantMatmulV3(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
```

## aclnnAlltoAllQuantMatmulV3GetWorkspaceSize

- **参数说明**

  <table style="table-layout:fixed;width:1500px"><colgroup>
  <col style="width:180px"><col style="width:120px"><col style="width:300px">
  <col style="width:350px"><col style="width:250px"><col style="width:100px">
  <col style="width:100px"><col style="width:100px"></colgroup>
  <thead><tr>
  <th>参数名</th><th>输入/输出</th><th>描述</th><th>使用说明</th>
  <th>数据类型</th><th>数据格式</th><th>维度(shape)</th><th>非连续Tensor</th>
  </tr></thead><tbody>
  <tr>
  <td>context（aclTensor*）</td><td>输入</td>
  <td>apace通信上下文张量，存储CommContext数据。</td>
  <td>不支持空Tensor。</td>
  <td>INT32、UINT8</td><td>ND</td><td>1</td><td>×</td>
  </tr><tr>
  <td>x1（aclTensor*）</td><td>输入</td>
  <td>左矩阵，对应公式中A_local。</td>
  <td>不支持空Tensor。shape：M_total能被worldSize整除。</td>
  <td>FLOAT8_E4M3FN、FLOAT8_E5M2</td><td>ND</td><td>2</td><td>×</td>
  </tr><tr>
  <td>x2（aclTensor*）</td><td>输入</td>
  <td>右矩阵，对应公式中X2。transposeX2=true时shape为[N,K_total]。</td>
  <td>不支持空Tensor。</td>
  <td>FLOAT8_E4M3FN、FLOAT8_E5M2</td><td>ND</td><td>2</td><td>×</td>
  </tr><tr>
  <td>biasOptional（aclTensor*）</td><td>输入</td>
  <td>偏置项，预留参数。</td>
  <td>暂不支持，传入nullptr。</td><td>-</td><td>-</td><td>-</td><td>-</td>
  </tr><tr>
  <td>x1ScaleOptional（aclTensor*）</td><td>输入</td>
  <td>x1的MX量化scale参数。</td>
  <td>不支持空Tensor。shape：[M_total,Ceil(Ka/64),2]。</td>
  <td>FLOAT8_E8M0</td><td>ND</td><td>3</td><td>×</td>
  </tr><tr>
  <td>x2ScaleOptional（aclTensor*）</td><td>输入</td>
  <td>x2的MX量化scale参数。</td>
  <td>不支持空Tensor。shape：[N,Ceil(K_total/64),2]。</td>
  <td>FLOAT8_E8M0</td><td>ND</td><td>3</td><td>×</td>
  </tr><tr>
  <td>group（const char*）</td><td>输入</td>
  <td>通信域标识，由HcclGetCommName获取。</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>
  </tr><tr>
  <td>worldSize（int64_t）</td><td>输入</td>
  <td>通信域大小（卡数）。</td><td>必须≥2，且为2/4/8/16之一。</td><td>-</td><td>-</td><td>-</td><td>-</td>
  </tr><tr>
  <td>hcclBufferSize（int64_t）</td><td>输入</td>
  <td>HCCL共享buffer大小。</td><td>预留，传入0即可。</td><td>-</td><td>-</td><td>-</td><td>-</td>
  </tr><tr>
  <td>x1QuantMode（int64_t）</td><td>输入</td>
  <td>x1量化模式，MX量化为6。</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>
  </tr><tr>
  <td>x2QuantMode（int64_t）</td><td>输入</td>
  <td>x2量化模式，MX量化为6。</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>
  </tr><tr>
  <td>groupSize（int64_t）</td><td>输入</td>
  <td>预留参数。</td><td>传入0即可。</td><td>-</td><td>-</td><td>-</td><td>-</td>
  </tr><tr>
  <td>commMode（const char*）</td><td>输入</td>
  <td>通信模式，仅支持"urma"。</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>
  </tr><tr>
  <td>precisionMode（int64_t）</td><td>输入</td>
  <td>精度模式。0=REMOTE，1=LOCAL+REMOTE(atomicAdd)，2=DEFERRED_SYNC。</td><td>建议使用0。</td><td>-</td><td>-</td><td>-</td><td>-</td>
  </tr><tr>
  <td>output（aclTensor*）</td><td>输出</td>
  <td>输出结果，对应公式中Y。</td>
  <td>不支持空Tensor。数据类型BF16或FP16。shape：[M_per_rank,N]。</td>
  <td>BF16、FP16</td><td>ND</td><td>2</td><td>×</td>
  </tr><tr>
  <td>alltoAllOutOptional（aclTensor*）</td><td>输出</td>
  <td>all-to-all通信中间输出，预留参数。</td>
  <td>暂不支持，传入nullptr。</td><td>-</td><td>-</td><td>-</td><td>-</td>
  </tr><tr>
  <td>workspaceSize（uint64_t*）</td><td>输出</td>
  <td>返回需要在Device侧申请的workspace大小。</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>
  </tr><tr>
  <td>executor（aclOpExecutor**）</td><td>输出</td>
  <td>返回op执行器，包含了算子计算流程。</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>
  </tr></tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="table-layout:fixed;width:1000px"><colgroup>
  <col style="width:300px"><col style="width:150px"><col style="width:550px"></colgroup>
  <thead><tr><th>返回值</th><th>错误码</th><th>描述</th></tr></thead><tbody>
  <tr><td>ACLNN_ERR_PARAM_NULLPTR</td><td>161001</td>
  <td>x1、x2、x1ScaleOptional、x2ScaleOptional、output存在空指针。</td></tr>
  <tr><td rowspan="4">ACLNN_ERR_PARAM_INVALID</td><td rowspan="4">161002</td>
  <td>x1或x2的数据类型不在FLOAT8_E4M3FN、FLOAT8_E5M2范围内。</td></tr>
  <tr><td>x1或x2的维度小于2。</td></tr>
  <tr><td>M_total不能被worldSize整除。</td></tr>
  <tr><td>CeilDiv(Ka,64)不是偶数。</td></tr>
  </tbody></table>

## aclnnAlltoAllQuantMatmulV3

- **参数说明**

  <table style="table-layout:fixed;width:1000px"><colgroup>
  <col style="width:180px"><col style="width:120px"><col style="width:700px"></colgroup>
  <thead><tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr></thead><tbody>
  <tr><td>workspace</td><td>输入</td><td>在Device侧申请的workspace内存地址。</td></tr>
  <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnAlltoAllQuantMatmulV3GetWorkspaceSize获取。</td></tr>
  <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
  <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

- 确定性说明：aclnnAlltoAllQuantMatmulV3默认确定性实现。
- 仅支持Ascend 950系列产品。
- 输入数据类型仅支持FP8_E4M3FN和FP8_E5M2。
- x1的shape[M_total,Ka]需满足：M_total能被worldSize整除，CeilDiv(Ka,64)为偶数。
- x2的K_total需满足 K_total = Ka × worldSize。
- 输出数据类型支持BF16和FP16，shape为[M_per_rank,N]，其中M_per_rank = M_total / worldSize。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。

```cpp
#include <acl/acl.h>
#include <hccl/hccl.h>
#include "aclnnop/aclnn_allto_all_matmul_v2.h"

// 1. 初始化HCCL通信域
aclInit(nullptr);
int32_t devices[4] = {0, 1, 2, 3};
for (int i = 0; i < 4; i++) aclrtSetDevice(devices[i]);
HcclComm comms[4];
HcclCommInitAll(4, devices, comms);

// 2. 创建tensor
// x1: [M_total, Ka] FP8_E4M3FN
// x2: [N, K_total] FP8_E4M3FN
// x1_scale: [M_total, Ceil(Ka/64), 2] FP8_E8M0
// x2_scale: [N, Ceil(K_total/64), 2] FP8_E8M0
// output: [M_per_rank, N] BF16

// 3. 获取workspace大小
uint64_t wsSize = 0;
aclOpExecutor *exec = nullptr;
aclnnAlltoAllQuantMatmulV3GetWorkspaceSize(
    contextTensor, x1Tensor, x2Tensor, nullptr,
    x1ScaleTensor, x2ScaleTensor,
    hcomName, 4, 0, 6, 6, 0, "urma", 0,
    outputTensor, nullptr, &wsSize, &exec);

// 4. 执行计算
void *ws = nullptr;
aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
aclnnAlltoAllQuantMatmulV3(ws, wsSize, exec, stream);
aclnrtSynchronizeStream(stream);
```