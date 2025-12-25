# AttentionWorkScheduler

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 算子功能：对输入的每个元素进行反余弦操作后输出。

- 计算公式：

$$
\text{Initialize:} \quad \text{ready_count} = 0, \quad \text{flag_num} = \text{micro_batch_size} \times \text{selected_expert_num}
$$

$$
\text{Check if run_flag is 0:}
\quad \text{if run_flag} = 0, \quad \text{exit and log}
$$

$$
\text{Loop:} \quad \text{while run_flag} \neq 0:
\quad \text{ready_count} = \sum_{i=1}^{\text{flag_num}} \mathbf{1}_{\{ \text{flag}[i] = 1 \}}; \quad \text{if ready_count} = \text{flag_num}, \quad \text{break}
$$

$$
\text{Reset flags:}
\quad \text{flag}[i] = 0 \quad \text{for} \quad i = 1, 2, \dots, \text{flag_num}
$$

$$
\text{Set micro_batch_id:} \quad \text{micro_batch_id} = (\text{micro_batch_id} + 1) \% \text{micro_batch_num}
$$

备注：micro_batch_size、selected_expert_num、run_flag、micro_batch_id是入参ScheduleContext结构体的参数，该结构体信息在[调用示例](#调用示例)中进行展示说明。


## 参数说明

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1565px"><colgroup>
  <col style="width: 146px">
  <col style="width: 135px">
  <col style="width: 326px">
  <col style="width: 246px">
  <col style="width: 275px">
  <col style="width: 101px">
  <col style="width: 190px">
  <col style="width: 146px">
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
      <td>scheduleContextRef</td>
      <td>输入/输出</td>
      <td>Attention侧接收的待处理数据，表示输入scheduleContext信息，详细结构见调用示例。</td>
      <td>不支持空Tensor。</td>
      <td>INT8</td>
      <td>ND</td>
      <td>1维，shape固定为(1024)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td rowspan="1">输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td rowspan="1">输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1134px"><colgroup>
  <col style="width: 319px">
  <col style="width: 144px">
  <col style="width: 671px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>参数scheduleContextRef是空指针</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>参数scheduleContextRef维度不为1。</td>
    </tr>
    <tr>
      <td>161002</td>
      <td>参数scheduleContextRef是空tensor。</td>
    </tr>
  </tbody>
  </table>

## 调用说明

| 调用方式 | 调用样例                                                                         | 说明                                                                                                                        |
|---------|------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_attention_worker_scheduler](./examples/test_aclnn_attention_worker_scheduler.cpp) | 通过[aclnnInplaceAttentionWorkerScheduler](./docs/aclnnInplaceAttentionWorkerScheduler.md)接口方式调用AttentionWorkerScheduler算子。 |
