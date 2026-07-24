# EngramFetchGrad

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    ×    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                               |    ×    |
| <term>Atlas 训练系列产品</term>                              |    ×    |

## 功能说明

- 算子功能：EngramFetchGrad 算子是 Engram 存储的训练态反向算子，需与 [EngramFetch](../engram_fetch/README.md)（训练模式）配套使用。根据前向保存的通信元数据（桶排列序、收发计数等），将 `grad_fetched` 沿前向路径反向交换并按 local entry 稀疏累加，产出稀疏梯度用于优化器更新。

- 计算公式：

  - 输入：
    - $\mathbf{G}^{\text{fetched}} \in \mathbb{R}^{\text{num\_tokens} \times \text{hidden\_size}}$：前向 `fetched` 的梯度，对应入参 `grad_fetched`。
    - $\mathbf{P} \in \mathbb{Z}^{\text{num\_tokens}}$：前向保存的桶排列序索引，对应入参 `perm`。
    - $\mathbf{C}^{\text{send}} \in \mathbb{Z}^{\text{rank\_size}}$：前向保存的每 rank 发送计数，对应入参 `send_counts`。
    - $\mathbf{C}^{\text{recv}} \in \mathbb{Z}^{\text{rank\_size}}$：前向保存的每 rank 接收计数，对应入参 `recv_counts`。
    - $\mathbf{L}^{\text{recv}} \in \mathbb{Z}^{\text{total\_recv}}$：前向保存的接收 local entry 索引，对应入参 `recv_local_entry`。
    - $\mathbf{N}^{\text{recv}} \in \mathbb{Z}$：前向保存的实际接收行数，对应入参 `num_recv`。

  - 输出：
    - $\mathbf{G}^{\text{unique}} \in \mathbb{R}^{\text{K} \times \text{hidden\_size}}$：去重累加后的稀疏梯度，对应出参 `grad_unique_out`。$\text{K}$ 为去重后的 local entry 数量（运行时决定），前 $\text{K}$ 行有效。数据类型与 `grad_fetched` 一致（kernel 内部按 FP32 累加后转回输出 dtype）。
    - $\mathbf{L}^{\text{unique}} \in \mathbb{Z}^{\text{K}}$：去重后的 local entry 索引，对应出参 `unique_local_entry_out`。$\mathbf{G}^{\text{unique}}[k]$ 对应 local entry $\mathbf{L}^{\text{unique}}[k]$ 的梯度。
    - $\mathbf{N}^{\text{unique}} \in \mathbb{Z}$：实际去重后的行数 $\text{K}$，对应出参 `num_unique_out`。

  - 计算过程（3 步融合 kernel）：

    1. **梯度重排**：按 $\mathbf{P}$ 重排 $\mathbf{G}^{\text{fetched}}$，恢复桶排序顺序，得到 $\mathbf{G}^{\text{sorted}}$。
    2. **a2a 反向交换**：通过 all-to-all 将 $\mathbf{G}^{\text{sorted}}$ 沿前向路径反向交换（$\mathbf{C}^{\text{send}} \leftrightarrow \mathbf{C}^{\text{recv}}$），使每张卡收到属于自己的梯度 $\mathbf{G}^{\text{recv}}$。
    3. **unique + scatter-add**：按 $\mathbf{L}^{\text{recv}}$ 对 $\mathbf{G}^{\text{recv}}$ 做去重和 FP32 累加，对相同 local entry 的梯度求和，输出去重后的稀疏梯度 $\mathbf{G}^{\text{unique}}$、对应索引 $\mathbf{L}^{\text{unique}}$ 和行数 $\mathbf{N}^{\text{unique}}$。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1392px">
 <colgroup>
  <col style="width: 160px">
  <col style="width: 120px">
  <col style="width: 400px">
  <col style="width: 150px">
  <col style="width: 80px">
 </colgroup>
 <thead>
  <tr>
   <th>参数名</th>
   <th>输入/输出/属性</th>
   <th>描述</th>
   <th>数据类型</th>
   <th>数据格式</th>
  </tr>
 </thead>
 <tbody>
  <tr>
   <td>comm_context</td>
   <td>输入</td>
   <td>本卡通信域信息数据。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>grad_fetched</td>
   <td>输入</td>
   <td>前向 fetched 的梯度。dim0 为 num_tokens，dim1 为 hidden_size（从 dim1 推断，不通过属性传入）。</td>
   <td>BF16、FP16、FP32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>perm</td>
   <td>输入</td>
   <td>前向保存的桶排列序索引。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>send_counts</td>
   <td>输入</td>
   <td>前向保存的每 rank 发送计数。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>recv_counts</td>
   <td>输入</td>
   <td>前向保存的每 rank 接收计数。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>recv_local_entry</td>
   <td>输入</td>
   <td>前向保存的接收 local entry 索引。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>num_recv</td>
   <td>输入</td>
   <td>前向保存的实际接收行数。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>grad_unique_out</td>
   <td>输出</td>
   <td>去重累加后的稀疏梯度，前 num_unique_out 行有效。数据类型与 grad_fetched 一致（内部 FP32 累加后转回）。</td>
   <td>BF16、FP16、FP32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>unique_local_entry_out</td>
   <td>输出</td>
   <td>去重后的 local entry 索引，与 grad_unique_out 行一一对应。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>num_unique_out</td>
   <td>输出</td>
   <td>实际去重后的行数 K。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>num_entries_per_rank</td>
   <td>属性</td>
   <td>每 rank 的 embedding 条目数。</td>
   <td>INT</td>
   <td></td>
  </tr>
  <tr>
   <td>comm_buffer_size</td>
   <td>属性</td>
   <td>HCCL 通信 buffer 大小（由 HCCL 框架管理，通过 HcclGetHcclBufferFunc 查询获得）。</td>
   <td>INT</td>
   <td></td>
  </tr>
 </tbody>
</table>

## 约束说明

- **维度约束**：
  - `grad_fetched` 必须为 2 维张量，dim1（hidden_size）必须大于 0 且为 128 对齐。
  - `perm` 必须为 1 维张量，dim0 必须等于 `grad_fetched` 的 dim0。
  - `send_counts` 和 `recv_counts` 必须为 1 维张量。
  - `recv_local_entry` 必须为 1 维张量。
  - `num_recv` 必须为 1 维张量，dim0 为 1。
  - `grad_unique_out` 必须为 2 维张量。
  - `unique_local_entry_out` 必须为 1 维张量。
  - `num_unique_out` 必须为 1 维张量，dim0 为 1。

- **数据类型约束**：
  - `comm_context`、`perm`、`send_counts`、`recv_counts`、`recv_local_entry`、`num_recv`、`unique_local_entry_out`、`num_unique_out` 必须为 `INT32`。
  - `grad_fetched` 和 `grad_unique_out` 支持 `BF16`、`FP16`、`FP32`，且两者数据类型须一致。

- **属性约束**：
  - `num_entries_per_rank` 取值范围为 $[0, \text{INT32\_MAX}]$。
  - `comm_buffer_size` 必须大于 0。

- **配套约束**：
  - 必须与 [EngramFetch](../engram_fetch/README.md)（训练模式）配套使用，`perm`、`send_counts`、`recv_counts`、`recv_local_entry`、`num_recv` 必须来自同一次前向调用。
  - `grad_fetched` 的 shape 必须与前向 `fetched` 一致。

- **通信域约束**：
  - Engram 通信域 `world_size` 范围 $[2, 1024]$。
  - 通信 buffer 使用 HCCL 默认 buffer（大小受 `HCCL_BUFFSIZE` 环境变量控制，默认 200MB），由框架管理。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| PyTorch接口调用 | - | 通过 [ElasticBuffer](../../torch_extension/cann_ops_transformer/docs/zh/elastic_buffer.md) PyTorch 接口，以 `with_grad=True` 调用 `engram_fetch_grad` 方式间接调用 EngramFetchGrad 算子。 |
