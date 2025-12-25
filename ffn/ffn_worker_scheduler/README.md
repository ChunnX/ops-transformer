# FfnWorkScheduler

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 算子功能：对输入的每个元素进行反余弦操作后输出。

- 计算公式：
1. 初始化，根据入参ScheduleContext中的session_num和sync_group_size计算分组个数。
  2. 若分组个数为1，表示全同步处理数据，待全部session数据准备就绪后，进行数据整理。
  3. 若分组个数不为1，表示非全同步处理数据，待group内的session数据准备就绪后，进行数据整理。

     $$
     \text{Initialize:} \quad\text{group_num} = \frac{\text{session_num}}{\text{sync_group_size}}
     $$

$$
\text{Process} =
\begin{cases}
\text{check_all_session_ready()} \quad \text{data_reorganization()} & \text{if } \text{group_num} = 1 \\
\text{check_all_sessions_of_group_ready()} \quad \text{data_reorganization()} & \text{otherwise}
\end{cases}
$$

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
      <td>FFN侧接收的待处理数据，表示ScheduleContext信息，详细结构参见调用示例</td>
      <td>不支持空tensor。</td>
      <td>INT8</td>
      <td>ND</td>
      <td>1维，shape为(1024)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>syncGroupSize</td>
      <td>输入</td>
      <td>每个同步组处理的session个数。</td>
      <td>取值范围为(0，session_num]，session_num表示待处理数据的最大会话数，即调用示例中结构体ScheduleContext中CommonArea域的session_num字段。</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executeMode</td>
      <td>输入</td>
      <td>执行模式。</td>
      <td>只支持模式0， 表示执行完一次退出。</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
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
      <td>参数scheduleContextRef是空指针。</td>
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
    <tr>
      <td>161002</td>
      <td>参数executeMode非0。</td>
    </tr>
  </tbody>
  </table>


## 约束说明
- aclnnInplaceFfnWorkerScheduler默认为确定性实现，暂不支持非确定性实现，确定性计算配置也不会生效。

## 调用说明

| 调用方式 | 调用样例                                             | 说明                                                                             |
|---------|--------------------------------------------------|--------------------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_ffn_worker_scheduler](./examples/test_aclnn_ffn_worker_scheduler.cpp) | 通过[aclnnFfnWorkerScheduler](./docs/aclnnSinh&aclnnInplaceSinh.md)接口方式调用FfnWorkerScheduler算子。 |
