# aclnnCausalConv1dBackward

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- **接口功能**：

    完成因果一维深度卷积前向算子（`aclnnCausalConv1dFn`，prefill 模式，bias=None、冷启动）对输入 `x` 与权重 `weight` 的反向梯度计算，用于训练反向传播。给定上游梯度 `gradY` 以及前向输入 `x`、`weight`，内部重算 pre-activation 并求激活导数，输出 `gradX` 与 `gradWeight`。结构信息（文档/序列边界）由 `queryStartLocOptional` 承担：每段是一个独立的、无掩码的因果卷积反向（段内冷启动左补零、段尾右补零）。仅支持 prefill 模式反向，不支持 decode/update 增量模式。

- **计算公式**：

    前向（prefill、冷启动、无 bias）：

    $$
    pre[t, d] = \sum_{k=0}^{W-1} weight[k, d] \cdot x[t-(W-1-k), d] \cdot m(t, t-(W-1-k))
    $$

    $$
    y[t, d] = Act(pre[t, d])
    $$

    其中 $m(t,u)$ 在 token $t$ 与 $u$ 属于同一序列且 $u \ge 0$ 时为 1。序列为 `query_start_loc` 在展平 token 轴上划出的段；`query_start_loc` 为 None 时整轴一段。**`weight[k,d]` 配 `x[t-(W-1-k),d]`，即 `weight[W-1]` 配当前 token `x[t]`、`weight[0]` 配最老 token `x[t-(W-1)]`**——与 `CausalConv1d` 前向同布局同约定（vllm causal_conv1d_fn 一致），正反向共用同一份权重张量，无需翻序。

    反向：

    $$
    dout[t, d] = gradY[t, d] \cdot Act'(pre[t, d])
    $$

    $$
    gradX[s, d] = \sum_{j=0}^{W-1} weight[W-1-j, d] \cdot m(s+j, s) \cdot dout[s+j, d]
    $$

    $$
    gradWeight[k, d] = \sum_{t} m(t, t-(W-1-k)) \cdot dout[t, d] \cdot x[t-(W-1-k), d]
    $$

    `Act'`：`silu'(z) = σ(z)·(1 + z·(1-σ(z)))`；`activation_mode="none"` 时 `Act' ≡ 1`。`Act'(pre)` 与全部累加在 FP32 中进行，结果只 cast 一次回输入 dtype；`gradWeight` 的累加项数最多可达 batch×seqlen，禁止半精度累加。

- **说明**：

  - 本算子与 `aclnnCausalConv1dFn` 配对，通过 torch autograd 注册实现训练反向传播。
  - 不输出历史状态缓存（convStates）的梯度；不支持 bias 梯度（前向契约 bias=None）。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用 `aclnnCausalConv1dBackwardGetWorkspaceSize` 接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用 `aclnnCausalConv1dBackward` 接口执行计算。

```cpp
aclnnStatus aclnnCausalConv1dBackwardGetWorkspaceSize(
    const aclTensor *gradY,
    const aclTensor *x,
    const aclTensor *weight,
    const aclTensor *queryStartLocOptional,
    const char      *activation,
    aclTensor       *gradXOut,
    aclTensor       *gradWeightOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnCausalConv1dBackward(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnCausalConv1dBackwardGetWorkspaceSize

- **参数说明**：

  <table style="table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 230px">
  <col style="width: 100px">
  <col style="width: 320px">
  <col style="width: 400px">
  <col style="width: 180px">
  <col style="width: 80px">
  <col style="width: 130px">
  <col style="width: 110px">
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
      <td>gradY（aclTensor*）</td>
      <td>输入</td>
      <td>上游梯度，表示对前向输出 y 的梯度。</td>
      <td><ul><li>不支持空tensor。</li><li>shape与前向输出y一致：固定batch为[batch, seqLen, dim]，变长为[cuSeqLen, dim]。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>前向输入序列，用于重算 pre-activation 及计算 gradWeight。</td>
      <td><ul><li>不支持空tensor。</li><li>shape与gradY一致。</li></ul></td>
      <td>数据类型与gradY一致</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight（aclTensor*）</td>
      <td>输入</td>
      <td>前向卷积权重，用于重算 pre-activation 及计算 gradX。</td>
      <td><ul><li>不支持空tensor。</li><li>shape为[W, dim]，W∈{2,3,4}。</li><li>布局与因果序与前向CausalConv1d的weight完全相同。</li></ul></td>
      <td>数据类型与gradY一致</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>queryStartLocOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>序列起始位置索引，记录各序列在展平token轴中的起始位置。</td>
      <td><ul><li>变长序列场景必须提供，固定batch场景可传空指针。</li><li>shape为[num_seq+1]。queryStartLoc[0]必须为0，queryStartLoc[-1]必须等于token总数，值必须非递减。</li><li>不传时整轴视为一个序列（无文档边界）。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>activation（const char*）</td>
      <td>输入</td>
      <td>正向所用的激活函数类型，须与前向一致。</td>
      <td>"silu"：计算SiLU导数；"none"：无激活，dout等于gradY。</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradXOut（aclTensor*）</td>
      <td>输出</td>
      <td>对输入x的梯度。</td>
      <td><ul><li>不支持空tensor。</li><li>shape与x一致。</li><li>需为连续张量。</li></ul></td>
      <td>数据类型与gradY一致</td>
      <td>ND</td>
      <td>2-3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>gradWeightOut（aclTensor*）</td>
      <td>输出</td>
      <td>对卷积权重weight的梯度。</td>
      <td><ul><li>不支持空tensor。</li><li>shape为[W, dim]，与weight一致。</li><li>需为连续张量。</li></ul></td>
      <td>数据类型与gradY一致</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回用户需要在Device侧申请的workspace大小。</td>
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
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="table-layout: fixed;width: 1000px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 550px">
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
      <td>传入的gradY、x、weight、gradXOut、gradWeightOut是空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>561001</td>
      <td>
      输入和输出的数据类型不在支持的范围内。<br>
      gradY、x、weight、gradXOut、gradWeightOut的数据类型不一致。<br>
      queryStartLocOptional的数据类型不是INT32。<br>
      gradY与x的shape不一致，或gradXOut与x、gradWeightOut与weight的shape不一致。<br>
      weight的shape不是[W, dim]或W不在[2,4]范围内。<br>
      2D输入（变长）未提供queryStartLocOptional，或queryStartLocOptional不是size≥2的一维张量。<br>
      activation不是"silu"或"none"。<br>
      </td>
    </tr>
  </tbody></table>

## aclnnCausalConv1dBackward

- **参数说明**：

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 700px">
  </colgroup>
  <thead>
    <tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr>
  </thead>
  <tbody>
    <tr><td>workspace</td><td>输入</td><td>在Device侧申请的workspace内存地址。</td></tr>
    <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnCausalConv1dBackwardGetWorkspaceSize获取。</td></tr>
    <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
    <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
  </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnCausalConv1dBackward默认确定性实现（按通道维切核、每核独占完整通道在核内完成归约，无原子操作，gradWeight全程FP32累加、末尾cast一次）。

- 输入输出一致性限制：
  - gradY、x、weight、gradXOut、gradWeightOut的数据类型必须一致（FLOAT16或BFLOAT16）。
  - gradY与x的shape必须一致；gradXOut与x的shape必须一致；gradWeightOut与weight的shape必须一致。

- 输入shape限制：
  - prefill场景（固定batch）：gradY、x为3维[batch, seqLen, dim]，token轴为展平的batch×seqLen。
  - prefill场景（变长序列）：gradY、x为2维[cuSeqLen, dim]，queryStartLocOptional必须提供。
  - weight、gradWeightOut为2维[W, dim]，W∈{2,3,4}。
  - dim需满足 (dim * dtypeSize) % 32 == 0（32字节对齐，与正向CausalConv1d一致）。
  - 建议范围（非硬校验）：dim为64的倍数时性能最优（非64整除走逐行兜底路径，慢但正确）。

- 输入值域限制：
  - queryStartLocOptional[0]必须为0，queryStartLocOptional[-1]必须等于token总数，值必须非递减。
  - 输入不支持有±inf和nan的情况。

- 模式限制：
  - 仅支持prefill模式反向（bias=None、冷启动语义），不支持decode/update增量模式。
  - 卷积核宽度W仅支持2、3、4。
  - 不输出历史状态缓存convStates的梯度，不支持bias梯度。

- 权重序约定：
  - weight[k, d]是x[t-(W-1-k), d]的系数：weight[W-1]配当前token x[t]，weight[0]配最老token x[t-(W-1)]。与CausalConv1d前向完全一致，正反向共用同一份权重张量，无需翻序。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_causal_conv1d_backward.h"

#define CHECK_RET(cond, return_expr)                                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            return_expr;                                                                                               \
        }                                                                                                              \
    } while (0)

#define LOG_PRINT(message, ...)                                                                                        \
    do {                                                                                                               \
        printf(message, ##__VA_ARGS__);                                                                                \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtContext *context, aclrtStream *stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateContext(context, deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetCurrentContext(*context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetCurrentContext failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

int main()
{
    int32_t deviceId = 0;
    aclrtContext context;
    aclrtStream stream;
    auto ret = Init(deviceId, &context, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 变长场景：2 个序列，长度分别为 4 与 4（cuSeqLen = 8）
    int64_t cuSeqLen = 8;
    int64_t dim = 64;
    int64_t width = 4;
    int64_t numSeq = 2;

    std::vector<int64_t> gradYShape = {cuSeqLen, dim};
    std::vector<int64_t> weightShape = {width, dim};
    std::vector<int64_t> qslShape = {numSeq + 1};

    std::vector<uint16_t> gradYHostData(GetShapeSize(gradYShape), 0x3C00);  // bf16 1.0
    std::vector<uint16_t> xHostData(GetShapeSize(gradYShape), 0x3C00);
    std::vector<uint16_t> weightHostData(GetShapeSize(weightShape), 0x3C00);
    std::vector<int32_t> qslHostData = {0, 4, 8};
    std::vector<uint16_t> gradXHostData(GetShapeSize(gradYShape), 0);
    std::vector<uint16_t> gradWeightHostData(GetShapeSize(weightShape), 0);

    void *gradYDeviceAddr = nullptr; void *xDeviceAddr = nullptr; void *weightDeviceAddr = nullptr;
    void *qslDeviceAddr = nullptr; void *gradXDeviceAddr = nullptr; void *gradWeightDeviceAddr = nullptr;
    aclTensor *gradY = nullptr; aclTensor *x = nullptr; aclTensor *weight = nullptr;
    aclTensor *qsl = nullptr; aclTensor *gradX = nullptr; aclTensor *gradWeight = nullptr;

    ret = CreateAclTensor(gradYHostData, gradYShape, &gradYDeviceAddr, aclDataType::ACL_BF16, &gradY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(xHostData, gradYShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_BF16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(qslHostData, qslShape, &qslDeviceAddr, aclDataType::ACL_INT32, &qsl);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradXHostData, gradYShape, &gradXDeviceAddr, aclDataType::ACL_BF16, &gradX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradWeightHostData, weightShape, &gradWeightDeviceAddr, aclDataType::ACL_BF16, &gradWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    const char *activation = "silu";

    ret = aclnnCausalConv1dBackwardGetWorkspaceSize(gradY, x, weight, qsl, activation, gradX, gradWeight,
                                                    &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCausalConv1dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnCausalConv1dBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCausalConv1dBackward failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    aclDestroyTensor(gradY); aclDestroyTensor(x); aclDestroyTensor(weight);
    aclDestroyTensor(qsl); aclDestroyTensor(gradX); aclDestroyTensor(gradWeight);
    aclrtFree(gradYDeviceAddr); aclrtFree(xDeviceAddr); aclrtFree(weightDeviceAddr);
    aclrtFree(qslDeviceAddr); aclrtFree(gradXDeviceAddr); aclrtFree(gradWeightDeviceAddr);
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
