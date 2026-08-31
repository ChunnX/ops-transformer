# aclnnAttentionToFfnV2

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mc2/attention_to_ffn_v2)

## 产品支持情况

<!-- npu="960" id1 -->
- <term>Ascend 960PR/Ascend 960DT</term>：不支持
<!-- end id1 -->
<!-- npu="950" id2 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id2 -->
<!-- npu="A3" id3 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="910b" id4 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
<!-- end id4 -->
<!-- npu="310b" id5 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id5 -->
<!-- npu="310p" id6 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id6 -->
<!-- npu="910" id7 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id7 -->

## 功能说明

将Attention节点上token数据发往FFN节点。

相较于`aclnnAttentionToFFN`接口，该接口变更如下：
1. 新增`context`入参，用于传递HCCL Channel通信域的窗口内存地址信息，替代原接口依赖的`HCCL_BUFFSIZE`环境变量；
2. 新增`cclBufferSize`入参，表示CCL通信缓冲区总大小（Bytes），由`get_buffer_for_attention_to_ffn_v2`接口内部自动计算；
3. 扩展了`quantMode`取值，新增MX/MX_CLIP量化模式（quantMode 3/4/5/6/7），支持FP8/FP4低精度输出。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnAttentionToFfnV2GetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnAttentionToFfnV2”接口执行计算。

```c++
aclnnStatus aclnnAttentionToFfnV2GetWorkspaceSize(
    const aclTensor    *context,
    const aclTensor    *x,
    const aclTensor    *sessionId,
    const aclTensor    *microBatchId,
    const aclTensor    *layerId,
    const aclTensor    *expertIds,
    const aclTensor    *expertRankTable,
    const aclTensor    *scalesOptional,
    const aclTensor    *activeMaskOptional,
    const char         *group,
    int64_t             worldSize,
    const aclIntArray  *ffnTokenInfoTableShape,
    const aclIntArray  *ffnTokenDataShape,
    const aclIntArray  *attnTokenInfoTableShape,
    int64_t             moeExpertNum,
    int64_t             quantMode,
    int64_t             syncFlag,
    int64_t             ffnStartRankId,
    int64_t             cclBufferSize,
    uint64_t           *workspaceSize,
    aclOpExecutor     **executor)
```

```c++
aclnnStatus aclnnAttentionToFfnV2(
    void            *workspace,
    uint64_t        workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream     stream)
```

## aclnnAttentionToFfnV2GetWorkspaceSize

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
      <col style="width: 170px">
      <col style="width: 120px">
      <col style="width: 300px">
      <col style="width: 300px">
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
        <td>context</td>
        <td>输入</td>
        <td>本卡通信域信息数据，包含HCCL Channel通信域的窗口内存地址。</td>
        <td>由通信缓冲区初始化接口创建（如Python侧的<code>get_buffer_for_attention_to_ffn_v2</code>接口），内部携带各rank的窗口内存地址，shape为(1,)。</td>
        <td>INT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x</td>
        <td>输入</td>
        <td>本卡发送的token数据。</td>
        <td>shape为(X, Bs, H)。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>sessionId</td>
        <td>输入</td>
        <td>表示当前Attention Worker节点的Id。</td>
        <td>shape为(X, )，取值范围为[0, attentionWorkerNum)。</td>
        <td>INT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>microBatchId</td>
        <td>输入</td>
        <td>表示microBatch的Id。</td>
        <td>shape为(X, )。</td>
        <td>INT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>layerId</td>
        <td>输入</td>
        <td>表示当前模型层数的Id。</td>
        <td>shape为(X, )。</td>
        <td>INT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>expertIds</td>
        <td>输入</td>
        <td>表示每个micro batch组中每个token的topK个专家索引。</td>
        <td>shape为(X, Bs, K)，取值区间为[0, moeExpertNum)。</td>
        <td>INT32</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>expertRankTable</td>
        <td>输入</td>
        <td>每个micro batch组中专家Id到FFN卡专家部署的映射表（外部需保证值正确）。</td>
        <td>shape为(L, moeExpertNum + sharedExpertNum, M)。</td>
        <td>INT32</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>scalesOptional</td>
        <td>输入</td>
        <td>每个专家的量化平滑参数。</td>
        <td>非量化场景下必须传空指针，PERTOKEN量化场景可传有效数据或空指针，MX/MX_CLIP量化场景必须传空指针，shape为(L, moeExpertNum + sharedExpertNum, H)。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>activeMaskOptional</td>
        <td>输入</td>
        <td>表示token是否参与通信，可传有效数据或空指针。</td>
        <td>传空指针时，默认所有token参与通信；传值时，shape为(X, Bs)，true表示该token参与通信，且true需排在false前。</td>
        <td>BOOL</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>group</td>
        <td>输入</td>
        <td>通信域名称（专家并行）。</td>
        <td>字符串长度范围为[1, 128)。</td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>worldSize</td>
        <td>输入</td>
        <td>表示通信域大小。</td>
        <td>取值区间[2, 1024]。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>ffnTokenInfoTableShape</td>
        <td>输入</td>
        <td>表示FFN节点上token信息表格shape大小的列表。</td>
        <td>长度为3，包括Attention节点的数量、microBatchSize的大小以及每个token对应的相关发送状态信息shape的大小。</td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>ffnTokenDataShape</td>
        <td>输入</td>
        <td>表示FFN节点上token数据表格shape大小的列表。</td>
        <td>长度为5，包括Attention节点的数量、microBatchSize的大小、batchSize大小、每个token需发送的专家数量（包括共享专家）、单个token的长度。</td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>attnTokenInfoTableShape</td>
        <td>输入</td>
        <td>表示Attention节点上token信息表格shape大小的列表。</td>
        <td>长度为3，包括microBatchSize的大小、batchSize大小、每个token需发送的专家数量（包括共享专家）。</td>
        <td>INT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>moeExpertNum</td>
        <td>输入</td>
        <td>MoE专家数量。</td>
        <td>取值范围(0, 1024]。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantMode</td>
        <td>输入</td>
        <td>表示量化模式。值同时编码算法和输出类型。</td>
        <td>支持0（非量化）、2（PERTOKEN+INT8）、3（MX+FP8_E5M2）、4（MX+FP8_E4M3）、5（MX+FP4_E2M1）、6（MX_CLIP+FP8_E5M2）、7（MX_CLIP+FP8_E4M3）。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>syncFlag</td>
        <td>输入</td>
        <td>表示FFN节点同步模式。</td>
        <td>仅支持0（同步）、1（异步）。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>ffnStartRankId</td>
        <td>输入</td>
        <td>表示FFN节点的起始ID。</td>
        <td>取值范围为[0, worldSize)。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>cclBufferSize</td>
        <td>输入</td>
        <td>CCL通信缓冲区总大小（Bytes）。</td>
        <td>由<code>get_buffer_for_attention_to_ffn_v2</code>接口内部自动计算，需 >= tokenInfoSize + tokenDataSize（按2MB向上对齐）。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回Device侧需申请的workspace大小。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>输出</td>
        <td>返回包含算子计算流程的op执行器。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

- **量化模式说明：**

    `quantMode` 值同时编码量化算法和输出数据类型：

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 150px">
    <col style="width: 150px">
    <col style="width: 200px">
    <col style="width: 650px">
    </colgroup>
    <thead>
      <tr>
        <th>quantMode</th>
        <th>算法</th>
        <th>输出类型</th>
        <th>说明</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>0</td>
        <td>非量化</td>
        <td>跟随输入</td>
        <td>FP16/BF16直接传输</td>
      </tr>
      <tr>
        <td>2</td>
        <td>PERTOKEN</td>
        <td>INT8</td>
        <td>每个token动态计算scale并量化为INT8</td>
      </tr>
      <tr>
        <td>3</td>
        <td>MX</td>
        <td>FP8_E5M2</td>
        <td>每32个元素为一组计算MX scale（E8M0格式），输出FP8_E5M2</td>
      </tr>
      <tr>
        <td>4</td>
        <td>MX</td>
        <td>FP8_E4M3</td>
        <td>每32个元素为一组计算MX scale（E8M0格式），输出FP8_E4M3</td>
      </tr>
      <tr>
        <td>5</td>
        <td>MX</td>
        <td>FP4_E2M1</td>
        <td>每32个元素为一组计算MX scale（E8M0格式），输出FP4_E2M1</td>
      </tr>
      <tr>
        <td>6</td>
        <td>MX_CLIP</td>
        <td>FP8_E5M2</td>
        <td>每32个元素为一组计算MX scale（带1e-4下限clipping），输出FP8_E5M2</td>
      </tr>
      <tr>
        <td>7</td>
        <td>MX_CLIP</td>
        <td>FP8_E4M3</td>
        <td>每32个元素为一组计算MX scale（带1e-4下限clipping），输出FP8_E4M3</td>
      </tr>
    </tbody></table>

- **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

    第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed; width: 1150px"> <colgroup>
    <col style="width: 280px">
    <col style="width: 100px">
    <col style="width: 900px">
      </colgroup><thead>
      <tr>
        <th>返回值</th>
        <th>错误码</th>
        <th>描述</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>输入和输出的必选参数Tensor是空指针。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>输入和输出的数据类型不在支持的范围内。</td>
      </tr>
      <tr>
        <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
        <td rowspan="2">561002</td>
        <td>输入和输出的shape不在支持的范围内。</td>
      </tr>
      <tr>
        <td>参数的取值不在支持的范围内。</td>
      </tr>
    </tbody></table>

## aclnnAttentionToFfnV2

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1150px"> <colgroup>
    <col style="width: 150px">
    <col style="width: 100px">
    <col style="width: 900px">
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
        <td>在Device侧申请的workspace大小，由第一段接口<code>aclnnAttentionToFfnV2GetWorkspaceSize</code>获取。</td>
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
    </tbody></table>

- **返回值：**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- **确定性约束**：
  - `aclnnAttentionToFfnV2`默认确定性实现。

- **参数一致性约束**：
  - 调用接口过程中使用的`group`、`worldSize`、`moeExpertNum`、`ffnTokenInfoTableShape`、`ffnTokenDataShape`、`ffnStartRankId`参数及`cclBufferSize`取值所有卡需保持一致，网络中不同层中也需保持一致，且和分离场景系列算子对应参数也保持一致。

- **产品特定约束**：

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：该场景下单卡包含双DIE（简称为“晶粒”或“裸片”），因此参数说明中的“本卡”均表示单DIE。

  <!-- end id7 -->

- **Shape变量约束**：

  | 变量         | 定义与取值范围                                                                 |
  | :----------- | :----------------------------------------------------------------------------- |
  | X            | 表示micro batch sequence size（token组数），当前版本只支持<code>X = 1</code>。 |
  | H（hidden size） | 表示hidden size隐藏层大小，取值范围为<code>[1024, 8192]</code>。 |
  | Bs           | 表示batch sequence size（本卡最终输出的token数量），取值范围为<code>0 < Bs ≤ 512</code>。 |
  | K            | 表示选取topK个专家，取值范围为<code>0 < K ≤ 16</code>，且<code>0 < K ≤ moeExpertNum</code>。 |
  | L            | 表示模型层数，当前版本只支持<code>L = 1</code>。 |
  | M            | 表示expertRankTable最后一维的长度，具体体现为部署在FFN节点上数量最多的专家部署信息列表的长度，取值范围为<code>1 < M ≤ FFNWorkerNum * 2 + 1</code>。 |
  | moeExpertNum | 表示MoE专家数量，取值范围 <code>(0, 1024]</code>。                   |
  | sharedExpertNum | 表示共享专家数量（一个共享专家可以复制部署到多个ffnRank卡上），取值范围 <code>[0, 4]</code>。                   |
  | moeExpertRankNum | 表示部署MoE专家的FFN节点数量，取值范围 <code>0 < moeExpertRankNum < FFNWorkerNum</code>。                   |
  | sharedExpertRankNum | 表示部署共享专家的FFN节点数量，取值范围 <code>0 < sharedExpertRankNum ≤ FFNWorkerNum</code>。                 |
  | FFNWorkerNum | 表示FFN节点的数量，取值范围为 <code>0 < FFNWorkerNum < worldSize</code>，且满足<code>FFNWorkerNum = moeExpertRankNum + sharedExpertRankNum</code>。                   |
  | AttentionWorkerNum | 表示Attention节点的数量，取值范围 <code>0 < AttentionWorkerNum < worldSize</code>。

- **CCL通信缓冲区约束**：
  - `cclBufferSize`为HBM上分配的CCL通信缓冲区**总大小**（Bytes），需满足：

  $$cclBufferSize \ge \mathrm{CeilAlign}(\mathrm{tokenInfoSize} + \mathrm{tokenDataSize},\ 2\,\mathrm{MB})$$

  其中：

    - `tokenInfoSize = attentionWorkerNum × microBatchNum × infoTableLastDim × 4B`
    - 非量化模式：`tokenDataSize = attentionWorkerNum × microBatchNum × BS × (K+shared) × HS × 2B`
    - INT8/FP8量化模式：`tokenDataSize = attentionWorkerNum × microBatchNum × BS × (K+shared) × HS × 1B`
    - FP4量化模式：`HS`已按打包字节计算，`tokenDataSize = attentionWorkerNum × microBatchNum × BS × (K+shared) × HS × 1B`

  该大小由`get_buffer_for_attention_to_ffn_v2`接口内部自动计算，用户无需自行计算或设置`cclBufferSize`。

- **MX/MX_CLIP量化约束**：
  - MX模式（quantMode 3/4/5）及MX_CLIP模式（quantMode 6/7）不支持传入scales参数。
  - FP4输出（quantMode=5）时H维度需为偶数。
  - MX_CLIP模式（quantMode 6/7）仅支持FP8输出（E5M2/E4M3），不支持FP4。

- **通信域使用约束**：
  - AttentionToFFN算子的通信域中不允许有其他算子。

## 调用示例

- 环境变量配置：

    无需额外配置环境变量。相较于`aclnnAttentionToFFN`接口，本接口通过`cclBufferSize`入参替代`HCCL_BUFFSIZE`环境变量。

算子编译命令如下：

```bash
bash build.sh --pkg --soc=ascend950 --ops=attention_to_ffn_v2
```

示例算子执行命令如下：

```bash
bash build.sh --run_example --ops=attention_to_ffn_v2 eager cust
```

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```c++
#include <thread>
#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "aclnn/opdev/fp16_t.h"
#include "aclnnop/aclnn_attention_to_ffn_v2.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while(0)

struct Args {
    uint32_t rankId;
    HcclComm hcclComm;
    aclrtStream attentionToFfnV2Stream;
    aclrtContext context;
};

constexpr uint32_t WORLD_SIZE = 16;
constexpr uint32_t FFN_WORKER_NUM = 5;
constexpr uint32_t ATTENTION_WORKER_NUM = WORLD_SIZE - FFN_WORKER_NUM;

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

template<typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc failed. ret: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMemcpy failed. ret: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i +1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
        aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int LaunchOneProcessAttentionToFfnV2(Args &args)
{
    int ret = aclrtSetCurrentContext(args.context);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetCurrentContext failed, ret %d\n", ret); return ret);

    char hcomName[128] = {0};
    ret = HcclGetCommName(args.hcclComm, hcomName);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] HcclGetCommName failed, ret %d\n", ret); return -1);
    LOG_PRINT("[INFO] rank = %d, hcomName = %s, attentionToFfnV2Stream = %p, context = %p\n",
              args.rankId, hcomName, args.attentionToFfnV2Stream, args.context);

    int64_t X = 1;
    int64_t L = 1;
    int64_t Bs = 8;
    int64_t H = 7168;
    int64_t K = 4;
    int64_t sharedExpertNum = 1;
    int64_t sharedExpertRankNum = 1;
    int64_t moeExpertNum = 8;
    int64_t expertNumPerToken = K + sharedExpertNum;
    int64_t M = 2 *(FFN_WORKER_NUM - sharedExpertNum * sharedExpertRankNum) + 1;
    int64_t quantMode = 0;
    int64_t syncFlag = 0;
    int64_t ffnStartRankId = 0;
    int64_t attentionWorkerNum = ATTENTION_WORKER_NUM;
    int64_t infoTableLastDim = 2 + Bs * expertNumPerToken;
    // cclBufferSize: 非量化模式下 tokenInfoSize + tokenDataSize (按2MB向上对齐)
    // tokenInfoSize = attentionWorkerNum * X * infoTableLastDim * 4
    // tokenDataSize = attentionWorkerNum * X * Bs * expertNumPerToken * H * 2
    int64_t tokenInfoSize = attentionWorkerNum * X * infoTableLastDim * 4;
    int64_t tokenDataSize = attentionWorkerNum * X * Bs * expertNumPerToken * H * 2;
    int64_t cclBufferSize = ((tokenInfoSize + tokenDataSize + 2 * 1024 * 1024 - 1) / (2 * 1024 * 1024)) * (2 * 1024 * 1024);
    int64_t ffnTokenInfoTableShapeData[] = {ATTENTION_WORKER_NUM , X, 2 + Bs * expertNumPerToken};
    int64_t ffnTokenDataShapeData[] = {ATTENTION_WORKER_NUM, X, Bs, expertNumPerToken, H};
    int64_t attnTokenInfoTableShapeData[] = {X, Bs, expertNumPerToken};

    /* 根据当前场景，构造device侧输入输出变量 */
    // 声明device侧输入输出变量
    void *contextDeviceAddr = nullptr;
    void *xDeviceAddr = nullptr;
    void *sessionIdDeviceAddr = nullptr;
    void *microBatchIdDeviceAddr = nullptr;
    void *layerIdDeviceAddr = nullptr;
    void *expertIdsDeviceAddr = nullptr;
    void *expertRankTableDeviceAddr = nullptr;
    void *scalesDeviceAddr = nullptr;

    aclTensor *contextTensor = nullptr;
    aclTensor *x = nullptr;
    aclTensor *sessionId = nullptr;
    aclTensor *microBatchId = nullptr;
    aclTensor *layerId = nullptr;
    aclTensor *expertIds = nullptr;
    aclTensor *expertRankTable = nullptr;
    aclTensor *scales = nullptr;
    aclIntArray *ffnTokenInfoTableShape = aclCreateIntArray(ffnTokenInfoTableShapeData, 3);
    aclIntArray *ffnTokenDataShape = aclCreateIntArray(ffnTokenDataShapeData, 5);
    aclIntArray *attnTokenInfoTableShape = aclCreateIntArray(attnTokenInfoTableShapeData, 3);

    // 定义当前场景下各变量维度
    // context: 通信域信息数据，由通信缓冲区初始化接口创建，此处使用占位tensor
    std::vector<int64_t> contextShape{1};
    std::vector<int64_t> xShape{X, Bs, H};
    std::vector<int64_t> sessionIdShape{X};
    std::vector<int64_t> microBatchIdShape{X};
    std::vector<int64_t> layerIdShape{X};
    std::vector<int64_t> expertIdsShape{X, Bs, K};
    std::vector<int64_t> expertRankTableShape{L, moeExpertNum + sharedExpertNum, M};
    std::vector<int64_t> scalesShape{L, moeExpertNum + sharedExpertNum, H};

    int64_t contextShapeSize = GetShapeSize(contextShape);
    int64_t xShapeSize = GetShapeSize(xShape);
    int64_t sessionIdShapeSize = GetShapeSize(sessionIdShape);
    int64_t microBatchIdShapeSize = GetShapeSize(microBatchIdShape);
    int64_t layerIdShapeSize = GetShapeSize(layerIdShape);
    int64_t expertIdsShapeSize = GetShapeSize(expertIdsShape);
    int64_t expertRankTableShapeSize = GetShapeSize(expertRankTableShape);
    int64_t scalesShapeSize = GetShapeSize(scalesShape);

    // 构造host侧变量
    std::vector<int32_t> contextHostData(contextShapeSize, 0);
    std::vector<int16_t> xHostData(xShapeSize, 1);
    std::vector<int16_t> sessionIdHostData(sessionIdShapeSize, args.rankId - FFN_WORKER_NUM);
    std::vector<int16_t> microBatchIdHostData(microBatchIdShapeSize, 0);
    std::vector<int16_t> layerIdHostData(layerIdShapeSize, 0);
    std::vector<int32_t> expertIdsHostData;
    for (int32_t micro_batch_id = 0; micro_batch_id < expertIdsShape[0]; micro_batch_id++) {
        for (int32_t token_id = 0; token_id < expertIdsShape[1]; token_id++) {
            for (int32_t k_id = 0; k_id < expertIdsShape[2]; k_id++) {
                expertIdsHostData.push_back(k_id);
            }
        }
    }

    std::vector<int32_t> expertRankTableHostData = {4, 2, 4, 3, 7, 1, 3, 2, 5, 2, 2, 5, 1, 2, 0, 0, 0, 0,
                                                    3, 2, 5, 0, 0, 3, 7, 0, 0, 4, 1, 3, 0, 1, 2, 4, 3, 7,
                                                    4, 0, 0, 3, 6, 1, 3, 2, 5, 3, 3, 7, 2, 4, 1, 2, 0, 0,
                                                    2, 2, 5, 0, 0, 0, 0, 0, 0, 3, 3, 6, 2, 5, 3, 7, 0, 0,
                                                    1, 4, 8, 0, 0, 0, 0, 0, 0};

    std::vector<float> scalesHostData(scalesShapeSize, 0.1);

    ret = CreateAclTensor(contextHostData, contextShape, &contextDeviceAddr, aclDataType::ACL_INT32, &contextTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sessionIdHostData, sessionIdShape, &sessionIdDeviceAddr, aclDataType::ACL_INT32, &sessionId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(microBatchIdHostData, microBatchIdShape, &microBatchIdDeviceAddr, aclDataType::ACL_INT32, &microBatchId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(layerIdHostData, layerIdShape, &layerIdDeviceAddr, aclDataType::ACL_INT32, &layerId);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertIdsHostData, expertIdsShape, &expertIdsDeviceAddr, aclDataType::ACL_INT32, &expertIds);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expertRankTableHostData, expertRankTableShape, &expertRankTableDeviceAddr, aclDataType::ACL_INT32, &expertRankTable);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scalesHostData, scalesShape, &scalesDeviceAddr, aclDataType::ACL_FLOAT, &scales);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t attentionToFfnV2WorkspaceSize = 0;
    aclOpExecutor *attentionToFfnV2Executor = nullptr;
    void *attentionToFfnV2WorkspaceAddr = nullptr;

    /**************************************** 调用AttentionToFfnV2 ********************************************/
    // 调用第一阶段接口
    ret = aclnnAttentionToFfnV2GetWorkspaceSize(
        contextTensor, x, sessionId, microBatchId, layerId, expertIds, expertRankTable,
        (quantMode > 0 ? scales : nullptr), nullptr, hcomName, WORLD_SIZE, ffnTokenInfoTableShape,
        ffnTokenDataShape, attnTokenInfoTableShape, moeExpertNum, quantMode, syncFlag, ffnStartRankId,
        cclBufferSize, &attentionToFfnV2WorkspaceSize, &attentionToFfnV2Executor);

    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("[ERROR] aclnnAttentionToFfnV2GetWorkspaceSize failed. ret = %d \n", ret); return ret);

    // 根据第一阶段接口计算出的workspaceSize申请device内存
    if (attentionToFfnV2WorkspaceSize > 0) {
        ret = aclrtMalloc(&attentionToFfnV2WorkspaceAddr, attentionToFfnV2WorkspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtMalloc workspace failed. ret = %d \n", ret); return ret);
    }

    if (args.rankId < FFN_WORKER_NUM) {  // FFN Worker
        // 等待Attention Worker任务执行结束
        LOG_PRINT("[INFO] device_%d is FFN worker, skipping aclnnAttentionToFfnV2 execute.\n", args.rankId);
        std::this_thread::sleep_for(std::chrono::seconds(30));
    } else {    // Attention Worker
        // 调用第二阶段接口
        ret = aclnnAttentionToFfnV2(attentionToFfnV2WorkspaceAddr, attentionToFfnV2WorkspaceSize,
                                    attentionToFfnV2Executor, args.attentionToFfnV2Stream);

        //（固定写法）同步等待任务执行结束
        ret = aclrtSynchronizeStreamWithTimeout(args.attentionToFfnV2Stream, 10000);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclnnAttentionToFfnV2 failed. ret = %d \n", ret);  \
            return ret);

        LOG_PRINT("[INFO] device_%d aclnnAttentionToFfnV2 execute successfully.\n", args.rankId);
    }

    // 释放device资源
    if (attentionToFfnV2WorkspaceSize > 0) {
        aclrtFree(attentionToFfnV2WorkspaceAddr);
    }

    if (contextTensor != nullptr) {
        aclDestroyTensor(contextTensor);
    }
    if (x != nullptr) {
        aclDestroyTensor(x);
    }
    if (sessionId != nullptr) {
        aclDestroyTensor(sessionId);
    }
    if (microBatchId != nullptr) {
        aclDestroyTensor(microBatchId);
    }
    if (layerId != nullptr) {
        aclDestroyTensor(layerId);
    }
    if (expertIds != nullptr) {
        aclDestroyTensor(expertIds);
    }
    if (expertRankTable != nullptr) {
        aclDestroyTensor(expertRankTable);
    }
    if (scales != nullptr) {
        aclDestroyTensor(scales);
    }
    if (ffnTokenInfoTableShape != nullptr) {
        aclDestroyIntArray(ffnTokenInfoTableShape);
    }
    if (ffnTokenDataShape != nullptr) {
        aclDestroyIntArray(ffnTokenDataShape);
    }
    if (attnTokenInfoTableShape != nullptr) {
        aclDestroyIntArray(attnTokenInfoTableShape);
    }

    if (contextDeviceAddr != nullptr) {
        aclrtFree(contextDeviceAddr);
    }
    if (xDeviceAddr != nullptr) {
        aclrtFree(xDeviceAddr);
    }
    if (sessionIdDeviceAddr != nullptr) {
        aclrtFree(sessionIdDeviceAddr);
    }
    if (microBatchIdDeviceAddr != nullptr) {
        aclrtFree(microBatchIdDeviceAddr);
    }
    if (layerIdDeviceAddr != nullptr) {
        aclrtFree(layerIdDeviceAddr);
    }
    if (expertIdsDeviceAddr != nullptr) {
        aclrtFree(expertIdsDeviceAddr);
    }
    if (expertRankTableDeviceAddr != nullptr) {
        aclrtFree(expertRankTableDeviceAddr);
    }
    if (scalesDeviceAddr != nullptr) {
        aclrtFree(scalesDeviceAddr);
    }

    HcclCommDestroy(args.hcclComm);
    aclrtDestroyStream(args.attentionToFfnV2Stream);
    aclrtDestroyContext(args.context);
    aclrtResetDevice(args.rankId);

    return 0;
}

int main(int argc, char *argv[])
{
    // 本样例基于Ascend 950实现，必须在Ascend 950上运行
    int ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtInit failed, ret = %d\n", ret); return ret);

    aclrtStream attentionToFfnV2Stream[WORLD_SIZE];
    aclrtContext context[WORLD_SIZE];

    for (uint32_t rankId = 0; rankId < WORLD_SIZE; rankId++) {
        ret = aclrtSetDevice(rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtSetDevice failed, ret = %d\n", ret); return ret);
        ret = aclrtCreateContext(&context[rankId], rankId);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateContext failed, ret = %d\n", ret); return ret);
        ret = aclrtCreateStream(&attentionToFfnV2Stream[rankId]);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[ERROR] aclrtCreateStream failed, ret = %d\n", ret); return ret);
    }

    int32_t devices[WORLD_SIZE];
    for (int32_t id = 0; id < WORLD_SIZE; id++) {
        devices[id] = id;
    }

    HcclComm comms[WORLD_SIZE];
    ret = HcclCommInitAll(WORLD_SIZE, devices, comms);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[ERROR] HcclCommInitAll failed, ret %d\n", ret); return ret);

    Args args[WORLD_SIZE];
    std::vector<std::unique_ptr<std::thread>> threads(WORLD_SIZE);
    for (uint32_t rankId = 0; rankId < WORLD_SIZE; rankId++) {
        args[rankId].rankId = rankId;
        args[rankId].hcclComm = comms[rankId];
        args[rankId].attentionToFfnV2Stream = attentionToFfnV2Stream[rankId];
        args[rankId].context = context[rankId];
        threads[rankId].reset(new(std::nothrow) std::thread(&LaunchOneProcessAttentionToFfnV2, std::ref(args[rankId])));
    }

    for(uint32_t rankId = 0; rankId < WORLD_SIZE; rankId++) {
        threads[rankId]->join();
    }

    aclFinalize();
    LOG_PRINT("[INFO] aclFinalize success\n");

    return 0;
}
```
