# EngramFetch

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

- 算子功能：EngramFetch 算子是 Engram 存储的前向算子，支持推理和训练两种模式，通过 `with_grad` 属性区分。

  - **推理模式**（`with_grad=0`）：通过 RDMA one-sided read 从远端 rank 抓取 Engram 数据。仅需 `commContext`、`indices` 和 `fetched` 三个张量，可选输入输出传 `nullptr`。
  - **训练模式**（`with_grad=1`）：通过 all-to-all 通信从远端 rank 抓取 Engram 数据，同时输出反向所需的通信元数据（桶排列序、收发计数等），供 [EngramFetchGrad](../engram_fetch_grad/README.md) 使用。需要传入 `local_storage_addr` 和全部可选输出。

- 计算公式：

  - 输入：
    - $\mathbf{I} \in \mathbb{Z}^{\text{num\_tokens}}$：查询索引向量，对应入参 `indices`。$\text{num\_tokens}$ 是查询的 token 数量。每个元素 $\mathbf{I}[i]$ 为全局 embedding 索引，取值范围为 $[0, \text{world\_size} \times \text{num\_entries\_per\_rank})$。
    - $\mathbf{S} \in \mathbb{R}^{\text{num\_entries\_per\_rank} \times \text{hidden\_size}}$：本 rank 的 embedding 表分片（存储在 host-pinned 内存中，通过 `local_storage_addr` 输入传入设备映射地址，仅训练模式需要）。

  - 输出：
    - $\mathbf{F} \in \mathbb{R}^{\text{num\_tokens} \times \text{hidden\_size}}$：抓取到的 embedding 矩阵，对应出参 `fetched`。$\mathbf{F}[i] = \mathbf{S}_{\text{owner}(\mathbf{I}[i])}[\mathbf{I}[i] \bmod \text{num\_entries\_per\_rank}]$，其中 $\text{owner}(j) = \lfloor j / \text{num\_entries\_per\_rank} \rfloor$。

  - Save-for-backward 输出（仅训练模式，供反向算子使用）：
    - $\mathbf{P} \in \mathbb{Z}^{\text{num\_tokens}}$：桶排列序索引，对应出参 `perm_out`。将 `indices` 按目标 rank 分组排序后的排列。
    - $\mathbf{C}^{\text{send}} \in \mathbb{Z}^{\text{rank\_size}}$：每 rank 发送的 index 数量，对应出参 `send_counts_out`。
    - $\mathbf{C}^{\text{recv}} \in \mathbb{Z}^{\text{rank\_size}}$：每 rank 接收的 index 数量，对应出参 `recv_counts_out`。
    - $\mathbf{L}^{\text{recv}} \in \mathbb{Z}^{\text{total\_recv}}$：接收到的 local entry 索引，对应出参 `recv_local_entry_out`。$\text{total\_recv} = \text{num\_max\_tokens\_per\_rank} \times \text{rank\_size}$。
    - $\mathbf{N}^{\text{recv}} \in \mathbb{Z}$：实际接收行数，对应出参 `num_recv_out`。

  - 计算过程：

    **推理模式**（RDMA read）：

    通过 UBC channel 对远端 rank 的 host-pinned 内存执行 one-sided RDMA read，直接将远端 embedding 读取到本地 `fetched` 输出。

    **训练模式**（5 步融合 kernel）：

    1. **桶排序**：按 $\text{owner}(\mathbf{I}[i])$ 对 `indices` 分组，生成排列序 $\mathbf{P}$、发送计数 $\mathbf{C}^{\text{send}}$ 和排序后的 local entry 序列。
    2. **a2a 交换 indices**：通过 all-to-all 将排序后的 local entry 发送给对应 rank，收到远端请求的 local entry $\mathbf{L}^{\text{recv}}$ 和接收计数 $\mathbf{C}^{\text{recv}}$。
    3. **本地 gather**：从本地 storage（`local_storage_addr`）按 $\mathbf{L}^{\text{recv}}$ 读取 embedding，得到 $\text{recvTokens}$。
    4. **a2a 交换 data**：通过 all-to-all 将 $\text{recvTokens}$ 发回请求方，得到排序序的 gathered 数据。
    5. **还原**：按 $\mathbf{P}$ 将 gathered 数据还原为原始顺序，输出 $\mathbf{F}$。

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
   <td>indices</td>
   <td>输入</td>
   <td>查询索引，表示要抓取的条目全局索引。元素取值范围为 [0, world_size × num_entries_per_rank)。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>local_storage_addr</td>
   <td>可选输入</td>
   <td>本地 storage 的 host-pinned 设备映射地址，作为 gather 源。shape 为 (1,)，存储一个 int64 地址值。仅训练模式需要，推理模式传 nullptr。</td>
   <td>INT64</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>fetched</td>
   <td>输出</td>
   <td>抓取到的 embedding 数据。</td>
   <td>BF16、FP16、FP32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>perm_out</td>
   <td>可选输出</td>
   <td>桶排列序索引，记录 indices 按目标 rank 分组排序后的排列，供反向使用。仅训练模式需要，推理模式传 nullptr。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>send_counts_out</td>
   <td>可选输出</td>
   <td>每 rank 发送的 index 数量。仅训练模式需要。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>recv_counts_out</td>
   <td>可选输出</td>
   <td>每 rank 接收的 index 数量。仅训练模式需要。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>recv_local_entry_out</td>
   <td>可选输出</td>
   <td>接收到的 local entry 索引，dim0 为 num_max_tokens_per_rank × rank_size。仅训练模式需要。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>num_recv_out</td>
   <td>可选输出</td>
   <td>实际接收行数 R。仅训练模式需要。</td>
   <td>INT32</td>
   <td>ND</td>
  </tr>
  <tr>
   <td>hidden_size</td>
   <td>属性</td>
   <td>每个条目的隐藏维度，必须为 128 对齐。</td>
   <td>INT</td>
   <td></td>
  </tr>
  <tr>
   <td>num_entries_per_rank</td>
   <td>属性</td>
   <td>每 rank 的 embedding 条目数。</td>
   <td>INT</td>
   <td></td>
  </tr>
  <tr>
   <td>num_max_tokens_per_rank</td>
   <td>可选属性</td>
   <td>每 rank 的最大 token 数量上限。仅训练模式需要。</td>
   <td>INT</td>
   <td></td>
  </tr>
  <tr>
   <td>comm_buffer_size</td>
   <td>可选属性</td>
   <td>HCCL 通信 buffer 大小（由 HCCL 框架管理，通过 HcclGetHcclBufferFunc 查询获得）。仅训练模式需要。</td>
   <td>INT</td>
   <td></td>
  </tr>
  <tr>
   <td>with_grad</td>
   <td>可选属性</td>
   <td>是否开启训练模式。0 表示推理模式（默认），1 表示训练模式。</td>
   <td>INT</td>
   <td></td>
  </tr>
 </tbody>
</table>

## 约束说明

- **维度约束**：
  - `indices` 必须为 1 维张量。
  - `local_storage_addr` 必须为 1 维张量，dim0 为 1。仅训练模式校验。
  - `fetched` 必须为 2 维张量，dim0 必须等于 `indices` 的 dim0，dim1 必须等于 `hidden_size`。
  - `perm_out` 必须为 1 维张量，dim0 必须等于 `indices` 的 dim0。仅训练模式校验。
  - `send_counts_out` 和 `recv_counts_out` 必须为 1 维张量。仅训练模式校验。
  - `recv_local_entry_out` 必须为 1 维张量，dim0 为 `num_max_tokens_per_rank × rank_size`。仅训练模式校验。
  - `num_recv_out` 必须为 1 维张量，dim0 为 1。仅训练模式校验。

- **数据类型约束**：
  - `comm_context`、`indices`、`perm_out`、`send_counts_out`、`recv_counts_out`、`recv_local_entry_out`、`num_recv_out` 必须为 `INT32`。
  - `local_storage_addr` 必须为 `INT64`。
  - `fetched` 支持 `BF16`、`FP16`、`FP32`。

- **属性约束**：
  - `hidden_size` 必须大于 0 且为 128 对齐。
  - `num_entries_per_rank` 取值范围为 $[0, \text{INT32\_MAX}]$。
  - `num_max_tokens_per_rank` 必须大于 0。仅训练模式校验。
  - `comm_buffer_size` 必须大于 0。仅训练模式校验。
  - `with_grad` 取值为 0（推理）或 1（训练）。

- **通信域约束**：
  - Engram 通信域 `world_size` 范围 $[2, 1024]$。
  - 训练模式下，通信 buffer 使用 HCCL 默认 buffer（大小受 `HCCL_BUFFSIZE` 环境变量控制，默认 200MB），由框架管理。

- **配套约束**：
  - 训练模式下，必须与 [EngramFetchGrad](../engram_fetch_grad/README.md) 配套使用，`perm_out`、`send_counts_out`、`recv_counts_out`、`recv_local_entry_out`、`num_recv_out` 需原样传递给反向算子。

## 调用说明

| 调用方式  | 样例代码                                  | 说明                                                     |
| :--------: | :----------------------------------------: | :-------------------------------------------------------: |
| PyTorch接口调用 | - | 通过 [ElasticBuffer](../../torch_extension/cann_ops_transformer/docs/zh/elastic_buffer.md) PyTorch 接口调用 `engram_fetch` 间接调用 EngramFetch 算子。推理模式 `with_grad=False`，训练模式 `with_grad=True`。 |
