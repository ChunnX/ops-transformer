# ffn_to_attention_v2

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

- **接口功能**：

  - ffn_to_attention_v2算子将FFN节点上的token数据发送回Attention节点，用于MoE（Mixture of Experts）场景下Attention与FFN分离部署时的反向数据通信。算子根据每个token所属的Attention Worker索引，将FFN计算完成的token数据路由回原始Attention卡，实现专家计算结果的汇聚。
  - 该算子提供了ffn_to_attention_v2与FFNToAttentionV2Buffer、get_ffn_to_attention_v2_ccl_buffer_size等接口配套使用。
  - get_ffn_to_attention_v2_ccl_buffer_size：用于计算ffn_to_attention_v2算子运行所需的CCL通信缓冲区大小。
  - FFNToAttentionV2Buffer：用于封装输入参数并创建通信上下文（context），生成`context`和`ccl_buffer_size`等ffn_to_attention_v2算子运行所需信息。

- **计算公式**：

  - 输入：
    - $\mathbf{X} \in \mathbb{R}^{\text{Y} \times \text{H}}$：FFN节点上的token数据矩阵，对应入参 `x`。$\text{Y}$ 是本卡需要分发的最大token数量，$\text{H}$ 是隐藏层维度。
    - $\mathbf{S} \in \mathbb{Z}^{\text{Y}}$：每个token的Attention Worker节点索引，对应入参 `session_ids`。$\text{S}[i] \in \{0, \dots, \text{attnRankNum}-1\}$。
    - $\mathbf{B} \in \mathbb{Z}^{\text{Y}}$：每个token的microBatch索引，对应入参 `micro_batch_ids`。$\text{B}[i] \in \{0, \dots, \text{microBatchNum}-1\}$。
    - $\mathbf{T} \in \mathbb{Z}^{\text{Y}}$：每个token在microBatch中的token索引，对应入参 `token_ids`。$\text{T}[i] \in \{0, \dots, \text{BS}-1\}$。
    - $\mathbf{O} \in \mathbb{Z}^{\text{Y}}$：每个token在tokenInfoTable中PerTokenExpertNum的索引，对应入参 `expert_offsets`。$\text{O}[i] \in \{0, \dots, \text{ExpertNumPerToken}-1\}$。
    - $\text{N}$：本卡发送的实际token总数，对应入参 `actual_token_num`。$\text{N} \in [0, \text{Y}]$。
  - 输出：
    - 无host可见输出。数据通过HCCL窗口发送至目标Attention卡，Attention卡从其CCL窗口中读取token数据。
  - 约定：
    - $\text{attnRankNum}$：Attention Worker数量。
    - $\text{ffnRankNum}$：FFN Worker数量。
    - $\text{ExpertNumPerToken}$：每个token对应的专家总数（含共享专家）。
    - $\text{HS}$：hidden与scale隐藏层大小。

- 计算说明：

    **数据路由**

    对于FFN Worker上的每个token $\text{token}_i$（$i \in \{0, 1, \dots, \text{N}-1\}$），根据其Attention Worker索引 $\text{S}[i]$ 确定目标Attention卡。若提供了 `attn_rank_table`，则通过该表将Worker索引映射为实际的Rank Id；若未提供，则采用默认策略（Worker Id == Rank Id）。

    **数据发送**

    对token $\text{token}_i$，将其数据 $\mathbf{x}_i$ 写入目标Attention卡的CCL窗口中对应的token数据区域，区域位置由 $\text{B}[i]$（microBatch索引）、$\text{T}[i]$（token索引）和 $\text{O}[i]$（expertOffset）共同确定。同时将元信息写入目标Attention卡的信息表区域，供Attention卡接收时识别。

    $$\text{sendData}_i = \mathbf{x}_i \quad \in \mathbb{R}^{1 \times \text{H}}$$

    ffn_to_attention_v2不支持量化，token数据始终以原始精度（FP16/BF16）传输。

## 函数原型

先调用get_ffn_to_attention_v2_ccl_buffer_size接口计算CCL通信缓冲区大小，再通过FFNToAttentionV2Buffer创建通信上下文，最后调用ffn_to_attention_v2接口进行数据发送。

```python
get_ffn_to_attention_v2_ccl_buffer_size(token_info_table_shape, token_data_shape) -> int
```

```python
FFNToAttentionV2Buffer(group, world_size, token_info_table_shape, token_data_shape)
```

```python
ffn_to_attention_v2(context, x, session_ids, micro_batch_ids, token_ids, expert_offsets, actual_token_num, group, world_size, token_info_table_shape, token_data_shape, *, ccl_buffer_size=0, attn_rank_table=None) -> None
```

## 参数说明

### get_ffn_to_attention_v2_ccl_buffer_size

<table style="undefined;table-layout: fixed; width:840px"><colgroup>
<col style="width: 180px">
<col style="width: 140px">
<col style="width: 80px">
<col style="width: 440px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>token_info_table_shape</td>
        <td>List[int]</td>
        <td>必选</td>
        <td>Token信息列表大小，长度为3，格式为<code>[microBatchNum, BS, expertNumPerToken]</code>。</td>
    </tr>
    <tr>
        <td>token_data_shape</td>
        <td>List[int]</td>
        <td>必选</td>
        <td>Token数据列表大小，长度为4，格式为<code>[microBatchNum, BS, expertNumPerToken, HS]</code>。</td>
    </tr>
</tbody>
</table>

### FFNToAttentionV2Buffer

<table style="undefined;table-layout: fixed; width:840px"><colgroup>
<col style="width: 180px">
<col style="width: 140px">
<col style="width: 80px">
<col style="width: 440px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>group</td>
        <td>torch.distributed.ProcessGroup</td>
        <td>必选</td>
        <td>EP通信域的ProcessGroup对象。</td>
    </tr>
    <tr>
        <td>world_size</td>
        <td>int</td>
        <td>必选</td>
        <td>通信域大小。</td>
    </tr>
    <tr>
        <td>token_info_table_shape</td>
        <td>List[int]</td>
        <td>必选</td>
        <td>Token信息列表大小，长度为3。</td>
    </tr>
    <tr>
        <td>token_data_shape</td>
        <td>List[int]</td>
        <td>必选</td>
        <td>Token数据列表大小，长度为4。</td>
    </tr>
</tbody>
</table>

### ffn_to_attention_v2

<table style="undefined;table-layout: fixed; width:1400px"><colgroup>
<col style="width: 120px">
<col style="width: 120px">
<col style="width: 90px">
<col style="width: 320px">
<col style="width: 160px">
<col style="width: 120px">
<col style="width: 260px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
    <th>维度(shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>context</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>由<a href="#ffntoattentionv2buffer">FFNToAttentionV2Buffer</a>创建的通信上下文tensor。</td>
        <td>int32</td>
        <td>ND</td>
        <td>(contextSize,)</td>
    </tr>
    <tr>
        <td>x</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>本卡发送的token数据。</td>
        <td>float16、bfloat16</td>
        <td>ND</td>
        <td>(Y, H)</td>
    </tr>
    <tr>
        <td>session_ids</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>每个token的Attention Worker节点索引。元素取值范围为<code>[0, attnRankNum-1]</code>。</td>
        <td>int32</td>
        <td>ND</td>
        <td>(Y,)</td>
    </tr>
    <tr>
        <td>micro_batch_ids</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>每个token的microBatch索引。元素取值范围为<code>[0, microBatchNum-1]</code>。</td>
        <td>int32</td>
        <td>ND</td>
        <td>(Y,)</td>
    </tr>
    <tr>
        <td>token_ids</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>每个token在microBatch中的token索引。元素取值范围为<code>[0, BS-1]</code>。</td>
        <td>int32</td>
        <td>ND</td>
        <td>(Y,)</td>
    </tr>
    <tr>
        <td>expert_offsets</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>每个token在tokenInfoTable中PerTokenExpertNum的索引。元素取值范围为<code>[0, expertNumPerToken-1]</code>。</td>
        <td>int32</td>
        <td>ND</td>
        <td>(Y,)</td>
    </tr>
    <tr>
        <td>actual_token_num</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>本卡发送的实际token总数。取值范围为<code>[0, Y]</code>。</td>
        <td>int64</td>
        <td>ND</td>
        <td>(1,)</td>
    </tr>
    <tr>
        <td>group</td>
        <td>str | ProcessGroup</td>
        <td>必选</td>
        <td>EP通信域名称或ProcessGroup对象。</td>
        <td>str</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>world_size</td>
        <td>int</td>
        <td>必选</td>
        <td>通信域大小。</td>
        <td>int</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>token_info_table_shape</td>
        <td>List[int]</td>
        <td>必选</td>
        <td>Token信息列表大小，长度为3，格式为<code>[microBatchNum, BS, expertNumPerToken]</code>。</td>
        <td>int</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>token_data_shape</td>
        <td>List[int]</td>
        <td>必选</td>
        <td>Token数据列表大小，长度为4，格式为<code>[microBatchNum, BS, expertNumPerToken, HS]</code>。</td>
        <td>int</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>ccl_buffer_size</td>
        <td>int</td>
        <td>可选</td>
        <td>CCL通信缓冲区总大小（Bytes），由<code>FFNToAttentionV2Buffer</code>自动计算并填充。默认值为0。</td>
        <td>int</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>attn_rank_table</td>
        <td>Tensor</td>
        <td>可选</td>
        <td>映射每一个Attention Worker对应的卡Id。Attention Worker必须从0卡开始连续部署；若输入None，采用默认策略：每张卡的Id作为对应Attention Worker的Id。默认值为None。</td>
        <td>int32</td>
        <td>ND（支持非连续Tensor）</td>
        <td>(attnRankNum,)</td>
    </tr>
</tbody>
</table>

## 返回值说明

无返回值。数据通过HCCL窗口发送至目标Attention卡，无host可见的输出tensor。

## 约束说明

- 调用算子过程中使用的`group`、`world_size`、`token_info_table_shape`、`token_data_shape`参数及`ccl_buffer_size`取值所有卡需保持一致，网络中不同层中也需保持一致。

- `ccl_buffer_size`为HBM上分配的CCL通信缓冲区**总大小**（Bytes），需满足：

$$ccl\_buffer\_size \ge \mathrm{CeilAlign}(\mathrm{tokenInfoSize} + \mathrm{tokenDataSize},\ 2\,\mathrm{MB})$$

其中：
  - `tokenInfoSize = microBatchNum × BS × expertNumPerToken × 4B`
  - `tokenDataSize = microBatchNum × BS × expertNumPerToken × HS × 2B`

可通过 `get_ffn_to_attention_v2_ccl_buffer_size` 接口计算所需最小值。

- 参数说明里shape格式说明：
    - `Y`：表示本卡需要分发的最大token数量。
    - `BS`：表示各Attention节点上的发送token数，取值范围为0 < `BS` ≤ 512。
    - `H`：表示hidden size（隐藏层大小），取值范围为1024 ≤ `H` ≤ 8192。
    - `HS`：表示hidden与scale隐藏层大小，取值范围为1152 ≤ `HS` ≤ 8320。
    - `ffnRankNum`：表示选取ffnRankNum个卡作为FFNWorker，取值范围为0 < `ffnRankNum` < `worldSize`。
    - `attnRankNum`：表示选取attnRankNum个卡作为AttnWorker，取值范围为0 < `attnRankNum` < `worldSize`。
    - `sharedExpertNum`：表示共享专家数量（一个共享专家可以复制部署到多个ffnRank卡上），取值范围为0 ≤ `sharedExpertNum` ≤ 4。
    - `worldSize`：通信域大小，取值区间[2, 768]。

- 通信域使用约束：
    - ffn_to_attention_v2算子的通信域中不允许有其他算子。
    - 通信域各节点的驱动版本应当相同。

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    - 该场景下单卡包含双DIE（简称为"晶粒"或"裸片"），因此参数说明里的"本卡"均表示单DIE。

## 确定性计算

默认支持确定性计算。

## 调用示例

- 单算子模式调用：

  下面示例展示了ffn_to_attention_v2的完整调用流程：先初始化通信域，再用FFNToAttentionV2Buffer构造通信上下文，最后调用ffn_to_attention_v2发送token数据。

  ```python
  import os
  import torch
  import torch_npu
  import torch.distributed as dist
  import torch.multiprocessing as mp
  from torch.multiprocessing import Process, Manager
  from cann_ops_transformer.ops import (
      ffn_to_attention_v2,
      FFNToAttentionV2Buffer,
      get_ffn_to_attention_v2_ccl_buffer_size,
  )

  WORLD_SIZE = 8
  ATTENTION_WORKER_NUM = 8
  microBatchNum = 1
  tokenNum = 8
  hiddenSize = 7168
  routedExpertNum = 7
  sharedExpertNum = 1
  expertNumPerToken = routedExpertNum + sharedExpertNum


  def set_device(rank):
      torch_npu.npu.set_device(rank % (WORLD_SIZE // 2))


  def init_hccl_comm(rank):
      dist.init_process_group(
          backend="hccl",
          rank=rank,
          world_size=WORLD_SIZE,
          init_method="tcp://127.0.0.1:50002",
      )
      ep_group = dist.new_group(backend="hccl", ranks=list(range(WORLD_SIZE)))
      return ep_group


  def run_ffn_to_attention_v2(rank):
      set_device(rank)
      ep_group = init_hccl_comm(rank)

      token_info_table_shape = [microBatchNum, tokenNum, expertNumPerToken]
      token_data_shape = [microBatchNum, tokenNum, expertNumPerToken, hiddenSize]

      # 步骤1：创建通信上下文
      buffer = FFNToAttentionV2Buffer(
          ep_group,
          WORLD_SIZE,
          token_info_table_shape,
          token_data_shape,
      )
      context = buffer.context
      ccl_buffer_size = buffer.ccl_buffer_size

      # 步骤2：构造输入数据
      Y = tokenNum * expertNumPerToken
      x = torch.randn((Y, hiddenSize), dtype=torch.bfloat16, device="npu")
      session_ids = torch.tensor([i % ATTENTION_WORKER_NUM for i in range(Y)],
                                 dtype=torch.int32, device="npu")
      micro_batch_ids = torch.zeros(Y, dtype=torch.int32, device="npu")
      token_ids = torch.tensor([i % tokenNum for i in range(Y)], dtype=torch.int32, device="npu")
      expert_offsets = torch.tensor([i % expertNumPerToken for i in range(Y)],
                                    dtype=torch.int32, device="npu")
      actual_token_num = torch.tensor([Y], dtype=torch.int64, device="npu")
      attn_rank_table = torch.tensor(list(range(ATTENTION_WORKER_NUM)),
                                      dtype=torch.int32, device="npu")

      # 步骤3：调用ffn_to_attention_v2发送token数据
      ffn_to_attention_v2(
          context,
          x,
          session_ids,
          micro_batch_ids,
          token_ids,
          expert_offsets,
          actual_token_num,
          ep_group,
          WORLD_SIZE,
          token_info_table_shape,
          token_data_shape,
          ccl_buffer_size=ccl_buffer_size,
          attn_rank_table=attn_rank_table,
      )

      torch.npu.synchronize()
      buffer.destroy()
      dist.barrier()
      dist.destroy_process_group()
      print(f"[INFO] rank {rank} ffn_to_attention_v2 finished")


  if __name__ == "__main__":
      mp.set_start_method("forkserver", force=True)
      proc_list = []
      for rank in range(WORLD_SIZE):
          proc = Process(target=run_ffn_to_attention_v2, args=(rank,))
          proc.start()
          proc_list.append(proc)
      for proc in proc_list:
          proc.join()
  ```
