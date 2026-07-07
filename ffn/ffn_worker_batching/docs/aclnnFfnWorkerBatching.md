# aclnnFfnWorkerBatching

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

> 本版本新增 <term>Ascend 950PR/Ascend 950DT</term> 支持，支持范围与 <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> 对齐。

## 功能说明

- 接口功能：`FfnWorkerBatching` 在 Attention 与 FFN 分离部署场景下，完成 FFN worker 上的 token 重排。Attention 将 token 按专家路由发送到对应 FFN worker 的预分配数据区，本算子从调度上下文（`scheduleContext`）中扫描/读取调度信息，按专家维度聚合并重排 token，产出各专家对应的连续 token 数据块，供后续 GroupedMatmul 消费。

- 计算逻辑（三步）：

  1. 对所有 token 的专家 ID 排序（被 mask 的无效 token 初始化为大值排到末尾），生成 gather 索引。
  2. 多核并行按 gather 索引从 token 数据区提取 token 的 hidden states 与 dynamic scale，同时查表得到对应的 `session_id`、`micro_batch_id`、`token_id`、`expert_offset`。
  3. 单核扫描排序后的专家 ID 序列，查找跳变点，生成 `groupList`（每个专家处理的 token 起止），并统计有效 token 总数 `actualTokenNum`。

- 输出规模：

  $$
  Y = A \times BS \times (topK+1)
  $$

  其中 $A$ 为 Attention worker 数量，$BS$ 为 micro batch size，$topK+1$ 为 topK 加共享专家数。`y` 的 shape 为 $[Y, H]$，$H$ 为 hidden size。

## 函数原型

每个算子分为两段式接口，必须先调用 “aclnnFfnWorkerBatchingGetWorkspaceSize” 接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用 “aclnnFfnWorkerBatching” 接口执行计算。

```cpp
aclnnStatus aclnnFfnWorkerBatchingGetWorkspaceSize(
  const aclTensor    *scheduleContext,
  int64_t             expertNum,
  const aclIntArray  *maxOutShape,
  int64_t             tokenDtype,
  int64_t             needSchedule,
  int64_t             layerNum,
  aclTensor          *y,
  aclTensor          *groupList,
  aclTensor          *sessionIds,
  aclTensor          *microBatchIds,
  aclTensor          *tokenIds,
  aclTensor          *expertOffsets,
  aclTensor          *dynamicScale,
  aclTensor          *actualTokenNum,
  uint64_t           *workspaceSize,
  aclOpExecutor     **executor)
```

```cpp
aclnnStatus aclnnFfnWorkerBatching(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

## aclnnFfnWorkerBatchingGetWorkspaceSize

- **参数说明**

  <table style="table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 350px">
  <col style="width: 250px">
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 100px">
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
      <td>scheduleContext（aclTensor*）</td>
      <td>输入</td>
      <td>调度上下文数据结构，含 CommonArea/ControlArea/AttentionArea/FfnArea，算子从中读取待重排 token 数据与路由信息。</td>
      <td><ul><li>不支持空Tensor。</li><li>为 1 维、1024 字节的调度上下文结构体。</li></ul></td>
      <td>INT8</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>expertNum（int64_t）</td>
      <td>输入</td>
      <td>本卡专家总数，等于每层本卡专家数 × layerNum，用于推导 groupList 输出大小。</td>
      <td>取值范围 (0, 8192]。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxOutShape（aclIntArray*）</td>
      <td>输入</td>
      <td>输出 shape 上限，格式为 {A, BS, topK+1, H}，用于推导 Y = A × BS × (topK+1) 与 H。</td>
      <td><ul><li>长度必须为 4。</li><li>A∈(0,1024]，topK+1∈(0,64]，BS>0，H>0。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>4</td>
      <td>-</td>
    </tr>
    <tr>
      <td>tokenDtype（int64_t）</td>
      <td>输入</td>
      <td>输入 token 的数据类型。0=FP16；1=BF16；2=INT8 动态量化（INT8 数据与 FP32 dynamic scale 连续排布）。</td>
      <td>取值范围 [0,2]，默认 0。取值为 2 时需输出 dynamicScale。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>needSchedule（int64_t）</td>
      <td>输入</td>
      <td>调度模式。0=仅做 batching 不扫描数据；1=先扫描数据再做 batching。</td>
      <td>取值范围 [0,1]，默认 0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layerNum（int64_t）</td>
      <td>输入</td>
      <td>层数，每层专家独立索引。</td>
      <td>取值范围 [0, expertNum]，默认 0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y（aclTensor*）</td>
      <td>输出</td>
      <td>重排后的 token hidden states，按专家 ID 排序后连续存放。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型随 tokenDtype：0→FLOAT16、1→BFLOAT16、2→INT8。</li></ul></td>
      <td>FLOAT16、BFLOAT16、INT8</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>groupList（aclTensor*）</td>
      <td>输出</td>
      <td>每个专家处理的 token 范围，每行 [expert_id, expert_token_num]，未使用专家填 [0,0]。</td>
      <td>不支持空Tensor。shape 为 [expertNum, 2]。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sessionIds（aclTensor*）</td>
      <td>输出</td>
      <td>每个输出 token 对应的 Attention session ID。</td>
      <td>不支持空Tensor。shape 为 [Y]。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>microBatchIds（aclTensor*）</td>
      <td>输出</td>
      <td>每个输出 token 对应的 micro batch ID。</td>
      <td>不支持空Tensor。shape 为 [Y]。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>tokenIds（aclTensor*）</td>
      <td>输出</td>
      <td>每个输出 token 在原始输入中的位置索引。</td>
      <td>不支持空Tensor。shape 为 [Y]。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>expertOffsets（aclTensor*）</td>
      <td>输出</td>
      <td>每个输出 token 在其所属专家分组内的偏移。</td>
      <td>不支持空Tensor。shape 为 [Y]。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>dynamicScale（aclTensor*）</td>
      <td>输出</td>
      <td>动态量化的 scale 值，仅在 tokenDtype=2 时有效；tokenDtype 为 0 或 1 时为空 tensor。</td>
      <td>shape 为 [Y]。tokenDtype 非 2 时输出空 tensor。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>actualTokenNum（aclTensor*）</td>
      <td>输出</td>
      <td>所有专家有效 token 数之和，标量输出。</td>
      <td>不支持空Tensor。shape 为 [1]。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
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
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回 op 执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="table-layout: fixed; width: 1000px"><colgroup>
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
      <td>scheduleContext 或任一必选输出为空指针。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>scheduleContext 数据类型不是 INT8 或不是 1 维。</td>
    </tr>
    <tr>
      <td>maxOutShape 长度不等于 4。</td>
    </tr>
    <tr>
      <td>expertNum、maxOutShape 各分量、tokenDtype、needSchedule 或 layerNum 取值超出支持范围。</td>
    </tr>
    <tr>
      <td>输出 dtype 与 tokenDtype 推导结果不匹配。</td>
    </tr>
  </tbody></table>

## aclnnFfnWorkerBatching

- **参数说明**

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 700px">
  </colgroup>
  <thead>
    <tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr>
  </thead>
  <tbody>
    <tr><td>workspace</td><td>输入</td><td>在 Device 侧申请的 workspace 内存地址。</td></tr>
    <tr><td>workspaceSize</td><td>输入</td><td>在 Device 侧申请的 workspace 大小，由第一段接口 aclnnFfnWorkerBatchingGetWorkspaceSize 获取。</td></tr>
    <tr><td>executor</td><td>输入</td><td>op 执行器，包含了算子计算流程。</td></tr>
    <tr><td>stream</td><td>输入</td><td>指定执行任务的 Stream。</td></tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

- 确定性说明：aclnnFfnWorkerBatching 默认确定性实现（相同 scheduleContext 输入产出二进制一致的输出）。
- scheduleContext 必须为 INT8、1 维、1024 字节的调度上下文结构体。
- maxOutShape 长度必须为 4，格式为 {A, BS, topK+1, H}，其中 A ≤ 1024、topK+1 ≤ 64；micro batch 数 M ≤ 64；BS、Y、H 支持泛化（受内存限制）。
- tokenDtype=2 时，输入 INT8 数据与 FP32 scale 在 token 数据区中连续排布，此时 dynamicScale 输出有效；tokenDtype 为 0 或 1 时 dynamicScale 为空 tensor。
- 本算子无 broadcast 语义；输出 shape 完全由 expertNum 与 maxOutShape 静态推导。
- <term>Ascend 950PR/Ascend 950DT</term>：支持范围与 <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> 一致。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。

```Cpp
// 示例代码占位，待开发阶段代码完成后补充。
// 图模式调用示例参见 examples/test_geir_ffn_worker_batching.cpp。
```
