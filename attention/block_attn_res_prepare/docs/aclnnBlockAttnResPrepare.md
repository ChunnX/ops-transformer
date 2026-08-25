# aclnnBlockAttnResPrepare

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/attention/block_attn_res_prepare)

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持。

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

## 功能说明

* 接口功能：Attention Residuals的历史残差注意力两段式计算的阶段一：一次性并行完成全部 S 层的块间注意力计算；同时返回输出结果与 softmax 统计量(O, M, L)，供阶段二融合当前残差并形成下一子层输入。
* 计算公式：



$$
\operatorname{block\_attn\_res\_prepare}
\left(
V,Q,\operatorname{valid\_blocks};\epsilon
\right)
\longrightarrow
(O,M, L).
$$

其中：

$$
V_{t,n,:}\in\mathbb R^D
$$

表示 token \(t\) 的第 \(n\) 个有效历史来源。

RMS 归一化因子，每个 token、每个历史来源独立计算：

$$
R_{t,n}
=
\sqrt{
\frac1D
\sum_{d=0}^{D-1}V_{t,n,d}^2
+\epsilon
},
$$

$$
I_{t,n}=R_{t,n}^{-1}.
$$

历史来源 logits， 对每个 slot、token 和历史来源：

$$
Z_{s,t,n}
=
I_{t,n}
\sum_{d=0}^{D-1}
Q_{s,d}V_{t,n,d}.
$$

历史 softmax 统计量，对当前有效历史 source 范围 (N_v) 取最大值：

$$
m_{s,t}
=
\max_{0\le n<N_v}Z_{s,t,n}.
$$

计算 max-shift 后的指数：

$$
E_{s,t,n}
=
\exp\left(Z_{s,t,n}-m_{s,t}\right).
$$

历史指数和：

$$
L_{s,t}
=
\sum_{n=0}^{N_v-1}E_{s,t,n}.
$$

历史加权输出：

$$
O_{s,t,d}
=
\sum_{n=0}^{N_v-1}
E_{s,t,n}V_{t,n,d}.
$$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnBlockAttnResPrepareGetWorkspaceSize”接口获取计算所需workspace大小，再调用“aclnnBlockAttnResPrepare”接口执行计算。

```cpp
aclnnStatus aclnnBlockAttnResPrepareGetWorkspaceSize(
    const aclTensor *blockRes, const aclTensor *validBlocks,
    const aclTensor *pseudoQuery, aclTensor *numerator,
    aclTensor *logitMax, aclTensor *expSum, float eps,
    uint64_t *workspaceSize, aclOpExecutor **executor);
```
```
aclnnStatus aclnnBlockAttnResPrepare(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);
```


## 参数说明


统一符号：

| 符号                                                                                                                                                                                                                                                                                                                                          | 含义                                           |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| $T$                                                   | 扁平 token 数；原始 padded 布局下可理解为$\left(B\times T_{\mathrm{seq}}\right)$。$B$：$\mathrm{batch\_size}$，批次大小，一次送入模型多少条样本；$T_{\mathrm{seq}}$：$\mathrm{sequence\ length}$，序列长度，单条样本 token 数目                                           |                                                |
| $N$                                                                                                                                                                                                                                                                                                                                         | 历史 block/source 缓冲区的固定容量             |
| $N\_v$                                                                                                                                                                                                                                                                                                                                       | 当前有效历史 block/source 数                   |
| $D$                                                                                                                                                                                                                                                                                                                                         | hidden dimension，隐藏层维度                   |
| $S$                                                                                                                                                                                                                                                                                                                                         | 当前$\mathrm{AttnRes}$ block 的目标 slot 数  |
| $t$                                                                                                                                                                                                                                                                                                             | token 下标，$0\le t<T$ |                                                |
| $n$     | 有效历史 source 下标，$0\le n<N_v\le N$                                                                                                                                                                                                                                                                                         |                                                |
| $d$     | hidden channel 下标，$0\le d<D$                                                                                                                                                                                                                                                                                                 |                                                |
| $s$     | slot 下标，$0\le s<S$                                                                                                                                                                                                                                                                                                           |                                                |

### aclnnBlockAttnResPrepareGetWorkspaceSize

- **参数说明：**

  <table style="table-layout: fixed; width: 1503px"><colgroup>
  <col style="width: 130px">
  <col style="width: 97px">
  <col style="width: 308px">
  <col style="width: 488px">
  <col style="width: 197px">
  <col style="width: 77px">
  <col style="width: 115px">
  <col style="width: 95px">
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>blockRes</td>
      <td>输入</td>
      <td>表示残差块，对应公式中的输入V。</td>
      <td>shape为[T, N, D]。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>3</td>
      <td>x</td>
    </tr>
    <tr>
      <td>validBlocks</td>
      <td>输入</td>
      <td>表示当前有效历史块数。</td>
      <td>shape为[1]，tensor中的值为N_v时，计算时blockRes的N维进前N_v参与运算。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>x</td>
    </tr>
    <tr>
      <td>pseudoQuery</td>
      <td>输入</td>
      <td>表示伪查询，对应公式中的输入Q。</td>
      <td>shape为[S, D]，最后一维必须与blockRes的最后一维相同。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>2</td>
      <td>x</td>
    </tr>
    <tr>
      <td>numerator</td>
      <td>输出</td>
      <td>表示softmax加权分子，对应公式中的输出O。</td>
      <td>shape为[S, T, D]。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>3</td>
      <td>x</td>
    </tr>
    <tr>
      <td>logitMax</td>
      <td>输出</td>
      <td>表示softmax最大值，对应公式中的输出M。</td>
      <td>shape为[S, T]。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>2</td>
      <td>x</td>
    </tr>
    <tr>
      <td>expSum</td>
      <td>输出</td>
      <td>表示softmax指数和，对应公式中的输出L。</td>
      <td>shape为[S, T]。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>2</td>
      <td>x</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>输入</td>
      <td>表示RMS归一化的稳定项，通常为1e-6。</td>
      <td>属性，必须为有限正数。</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
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

### aclnnBlockAttnResPrepare

- **参数说明：**

  <table style="table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
  </colgroup>
  <thead>
    <tr>
      <th>参数说明</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnBlockAttnResPrepareGetWorkspaceSize获取。</td>
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

## 约束说明

- `T >= 0`，`S >= 0`，`0 <= N <= 64`，`0 <= D <= 8192`。
- `blockRes` 与 `pseudoQuery` 的最后一维必须相同。
- 输入输出不支持非连续 Tensor。
- `T == 0` 或 `S == 0` 时返回对应形状的空输出。
- 仅支持 Ascend 950。

## 返回值

| 返回值                      | 说明                                   |
| --------------------------- | -------------------------------------- |
| `ACLNN_SUCCESS`           | 成功。                                 |
| `ACLNN_ERR_PARAM_NULLPTR` | 必选参数为空。                         |
| `ACLNN_ERR_PARAM_INVALID` | dtype、rank、shape 或 eps 不满足约束。 |
| `ACLNN_ERR_RUNTIME_ERROR` | 平台不支持或执行失败。                 |

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```c++
#include <cstdint>
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_block_attn_res_prepare.h"

#define CHECK_RET(condition, message)           \
    do {                                        \
        if (!(condition)) {                     \
            std::printf("%s\n", (message));    \
            return 1;                           \
        }                                       \
    } while (0)

namespace {

int64_t GetElementCount(const std::vector<int64_t> &shape)
{
    int64_t count = 1;
    for (const int64_t dim : shape) {
        count *= dim;
    }
    return count;
}

template <typename T>
int CreateTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, aclDataType dataType,
                 void **deviceAddress, aclTensor **tensor)
{
    const size_t bytes = static_cast<size_t>(GetElementCount(shape)) * sizeof(T);
    aclError ret = aclrtMalloc(deviceAddress, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        return ret;
    }
    ret = aclrtMemcpy(*deviceAddress, bytes, hostData.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        return ret;
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddress);
    return *tensor == nullptr ? ACL_ERROR_INVALID_PARAM : ACL_SUCCESS;
}

} // namespace

int main()
{
    constexpr int32_t deviceId = 0;
    aclError ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, "aclInit failed");
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtSetDevice failed");

    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtCreateStream failed");

    const std::vector<int64_t> blockResShape = {1, 2, 4};
    const std::vector<int64_t> validBlocksShape = {1};
    const std::vector<int64_t> pseudoQueryShape = {2, 4};
    const std::vector<int64_t> numeratorShape = {2, 1, 4};
    const std::vector<int64_t> statsShape = {2, 1};

    const std::vector<float> blockResData = {
        1.0F, 2.0F, 3.0F, 4.0F,
        2.0F, 0.0F, -1.0F, 1.0F,
    };
    const std::vector<int64_t> validBlocksData = {2};
    const std::vector<float> pseudoQueryData = {
        1.0F, 0.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 0.0F, 0.0F,
    };
    std::vector<float> numeratorData(GetElementCount(numeratorShape), 0.0F);
    std::vector<float> logitMaxData(GetElementCount(statsShape), 0.0F);
    std::vector<float> expSumData(GetElementCount(statsShape), 0.0F);

    void *blockResAddress = nullptr;
    void *validBlocksAddress = nullptr;
    void *pseudoQueryAddress = nullptr;
    void *numeratorAddress = nullptr;
    void *logitMaxAddress = nullptr;
    void *expSumAddress = nullptr;
    aclTensor *blockRes = nullptr;
    aclTensor *validBlocks = nullptr;
    aclTensor *pseudoQuery = nullptr;
    aclTensor *numerator = nullptr;
    aclTensor *logitMax = nullptr;
    aclTensor *expSum = nullptr;

    ret = CreateTensor(blockResData, blockResShape, ACL_FLOAT, &blockResAddress, &blockRes);
    CHECK_RET(ret == ACL_SUCCESS, "create blockRes failed");
    ret = CreateTensor(validBlocksData, validBlocksShape, ACL_INT64, &validBlocksAddress, &validBlocks);
    CHECK_RET(ret == ACL_SUCCESS, "create validBlocks failed");
    ret = CreateTensor(pseudoQueryData, pseudoQueryShape, ACL_FLOAT, &pseudoQueryAddress, &pseudoQuery);
    CHECK_RET(ret == ACL_SUCCESS, "create pseudoQuery failed");
    ret = CreateTensor(numeratorData, numeratorShape, ACL_FLOAT, &numeratorAddress, &numerator);
    CHECK_RET(ret == ACL_SUCCESS, "create numerator failed");
    ret = CreateTensor(logitMaxData, statsShape, ACL_FLOAT, &logitMaxAddress, &logitMax);
    CHECK_RET(ret == ACL_SUCCESS, "create logitMax failed");
    ret = CreateTensor(expSumData, statsShape, ACL_FLOAT, &expSumAddress, &expSum);
    CHECK_RET(ret == ACL_SUCCESS, "create expSum failed");

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnBlockAttnResPrepareGetWorkspaceSize(blockRes, validBlocks, pseudoQuery, numerator, logitMax,
                                                    expSum, 1.0e-6F, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnBlockAttnResPrepareGetWorkspaceSize failed");

    void *workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, "allocate workspace failed");
    }
    ret = aclnnBlockAttnResPrepare(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclnnBlockAttnResPrepare failed");
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, "aclrtSynchronizeStream failed");

    ret = aclrtMemcpy(numeratorData.data(), numeratorData.size() * sizeof(float), numeratorAddress,
                      numeratorData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, "copy numerator failed");
    ret = aclrtMemcpy(logitMaxData.data(), logitMaxData.size() * sizeof(float), logitMaxAddress,
                      logitMaxData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, "copy logitMax failed");
    ret = aclrtMemcpy(expSumData.data(), expSumData.size() * sizeof(float), expSumAddress,
                      expSumData.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, "copy expSum failed");

    for (size_t i = 0; i < numeratorData.size(); ++i) {
        std::printf("numerator[%zu] = %.6f\n", i, numeratorData[i]);
    }
    for (size_t i = 0; i < logitMaxData.size(); ++i) {
        std::printf("logit_max[%zu] = %.6f, exp_sum[%zu] = %.6f\n",
                    i, logitMaxData[i], i, expSumData[i]);
    }

    aclDestroyTensor(blockRes);
    aclDestroyTensor(validBlocks);
    aclDestroyTensor(pseudoQuery);
    aclDestroyTensor(numerator);
    aclDestroyTensor(logitMax);
    aclDestroyTensor(expSum);
    aclrtFree(blockResAddress);
    aclrtFree(validBlocksAddress);
    aclrtFree(pseudoQueryAddress);
    aclrtFree(numeratorAddress);
    aclrtFree(logitMaxAddress);
    aclrtFree(expSumAddress);
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}

```
