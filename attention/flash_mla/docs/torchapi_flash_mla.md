# flash\_mla

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：不支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
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

  调用FlashMla算子完成旋转位置编码（Rotary Position Embedding，RoPE）分离的多头潜在注意力（Multi-head Latent Attention，MLA）前向计算。支持float16、bfloat16非量化和int8全量化场景，KV可采用连续存储或分页注意力（Paged Attention，PA）存储。

  调用前需通过[flash_mla_metadata](../../flash_mla_metadata/docs/torchapi_flash_mla_metadata.md)生成负载均衡结果。query_rope和key_rope须为已完成旋转位置编码的输入。

- **计算公式**：

  非量化场景下，每个batch、每个Query头的计算公式如下：

  $$
  S=\mathrm{scale\_value}\left(QK^T+Q_{\mathrm{rope}}K_{\mathrm{rope}}^T\right)+M
  $$

  $$
  \mathrm{attention\_out}=\operatorname{softmax}(S)V,\qquad
  \mathrm{softmax\_lse}_i=\log\sum_j\exp(S_{ij})
  $$

  $Q$、$K$、$V$分别对应query、key[0]和value[0]，$Q_{\mathrm{rope}}$和$K_{\mathrm{rope}}$分别对应query_rope和key_rope。$M$表示掩码和窗口共同确定的屏蔽关系，保留位置为0，屏蔽位置为负无穷。INT8场景先使用dequant_scale_query、key_antiquant_scale和value_antiquant_scale恢复Q、K、V的数值尺度，再按上述公式计算，RoPE部分使用bfloat16。

## 函数原型

`*`之后的参数仅支持以关键字方式传入。

```python
cann_ops_transformer.flash_mla(
    query,
    key,
    value,
    query_rope,
    key_rope,
    metadata,
    num_heads,
    *,
    atten_mask=None,
    actual_seq_lengths_q=None,
    actual_seq_lengths_kv=None,
    block_table=None,
    dequant_scale_query=None,
    key_antiquant_scale=None,
    value_antiquant_scale=None,
    scale_value=1.0,
    pre_tokens=2147483647,
    next_tokens=2147483647,
    input_layout="BSH",
    num_key_value_heads=1,
    sparse_mode=0,
    inner_precise=0,
    block_size=0,
    softmax_lse_flag=False,
    key_antiquant_mode=0,
    value_antiquant_mode=0,
    query_quant_mode=0,
) -> (Tensor, Tensor)
```

## 参数说明

> [!NOTE]
>
> B表示输入样本批量大小，N表示Query头数，S_q和S_kv表示Query和KV的存储序列长度，T_q和T_kv表示TND布局下各batch序列长度的累加和，P表示KV缓存的物理块数，L表示block_size，max_blocks表示每个batch最多使用的块数。KV头数固定为1，非RoPE维度为512，RoPE维度为64。

所有输入Tensor须位于同一NPU。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| query | Tensor | 必选 | 公式中的Q，即Query的非RoPE部分，每个注意力头的维度为512。数据格式为ND，支持非连续Tensor。 | float16、bfloat16、int8 | 见本节对应的布局表。 |
| key | List[Tensor] | 必选 | 公式中的K，即Key的非RoPE部分。列表中仅包含一个Tensor，头数为1，每个注意力头的维度为512。数据格式为ND，支持非连续Tensor。 | float16、bfloat16、int8 | 见本节对应的布局表。 |
| value | List[Tensor] | 必选 | 公式中的V。列表中仅包含一个Tensor，shape和数据类型与key[0]一致；可与key共享Tensor。数据格式为ND，支持非连续Tensor。 | float16、bfloat16、int8 | 与key[0]一致。 |
| query_rope | Tensor | 必选 | 公式中的Q_rope，即已完成旋转位置编码的Query部分，每个注意力头的维度为64。非量化时数据类型与query一致；INT8时为bfloat16。数据格式为ND，支持非连续Tensor。 | float16、bfloat16 | 见本节对应的布局表。 |
| key_rope | Tensor | 必选 | 公式中的K_rope，即已完成旋转位置编码的Key部分，数据类型与query_rope一致。数据格式为ND，支持非连续Tensor。 | float16、bfloat16 | 见本节对应的布局表。 |
| metadata | Tensor | 必选 | 由flash_mla_metadata生成的负载均衡结果，须原样传入。数据格式为ND，不支持非连续Tensor。 | int32 | (480,) |
| num_heads | int | 必选 | Query头数N，取值范围为[1, 128]；INT8时须为2的幂。 | int64 | - |
| atten_mask | Tensor | 可选 | 注意力掩码，True或1表示屏蔽，False或0表示保留。默认值为None。数据格式为ND，支持非连续Tensor。 | bool、int8、uint8 | 见[掩码与窗口](#约束说明)。 |
| actual_seq_lengths_q | Tensor | 可选 | 每个batch的Query实际序列长度。TND采用不含起始0的累计长度，batch非空时必传。默认值为None。数据格式为ND，不支持非连续Tensor。 | int64 | (B,)；非TND的广播及缺省规则见[序列实长](../../flash_mla_metadata/docs/torchapi_flash_mla_metadata.md#约束说明)。 |
| actual_seq_lengths_kv | Tensor | 可选 | 每个batch的KV实际序列长度。连续TND KV采用累计长度，PA采用各batch的实际长度；PA或TND且batch非空时必传。默认值为None。数据格式为ND，不支持非连续Tensor。 | int64 | (B,)；非TND的广播及缺省规则见[序列实长](../../flash_mla_metadata/docs/torchapi_flash_mla_metadata.md#约束说明)。 |
| block_table | Tensor | 可选 | PA场景下的分页表，元素为物理块索引。PA时必传，连续KV时不传。默认值为None。数据格式为ND，支持非连续Tensor。 | int32 | (B, max_blocks) |
| dequant_scale_query | Tensor | 可选 | Query按token和head量化的反量化系数，INT8时必传，非量化时不传。默认值为None。数据格式为ND，支持非连续Tensor。 | float32 | BSH/BSND：(B, S_q, N)；TND：(T_q, N)。输出布局后缀不影响shape。 |
| key_antiquant_scale | Tensor | 可选 | Key按Tensor量化的反量化系数，INT8时必传，非量化时不传。默认值为None。数据格式为ND，支持非连续Tensor。 | float32 | (1,) |
| value_antiquant_scale | Tensor | 可选 | Value按Tensor量化的反量化系数，INT8时必传，非量化时不传。默认值为None。数据格式为ND，支持非连续Tensor。 | float32 | (1,) |
| scale_value | float | 可选 | 公式中的缩放系数scale_value，作用于两路QK乘积之和。默认值为1.0。 | double | - |
| pre_tokens | int | 可选 | 默认值为2147483647，左窗口参数，按sparse_mode解释。 | int64 | - |
| next_tokens | int | 可选 | 默认值为2147483647，右窗口参数，按sparse_mode解释。 | int64 | - |
| input_layout | str | 可选 | 默认值为"BSH"，支持下表中的8种输入/输出布局组合，区分大小写。 | string | - |
| num_key_value_heads | int | 可选 | 默认值为1。非量化非空输入仅支持1；INT8在num_heads=1时也支持0，此时有效KV头数仍为1。 | int64 | - |
| sparse_mode | int | 可选 | 默认值为0。非量化支持0、3、4、9；INT8支持0、3、9。 | int64 | - |
| inner_precise | int | 可选 | 默认值为0。非量化支持0、1、2、3，其中2和3开启行无效修正；INT8仅支持0和1。 | int64 | - |
| block_size | int | 可选 | 默认值为0。PA时为不大于1024的正16倍数；INT8时必须为128。 | int64 | - |
| softmax_lse_flag | bool | 可选 | 默认值为False。True时返回有效的softmax_lse，False时第二个输出为空张量。 | bool | - |
| key_antiquant_mode | int | 可选 | 默认值为0，当前仅支持0，即Key使用per-tensor反量化。 | int64 | - |
| value_antiquant_mode | int | 可选 | 默认值为0，当前仅支持0，即Value使用per-tensor反量化。 | int64 | - |
| query_quant_mode | int | 可选 | 默认值为0。非量化仅支持0；INT8必须显式设为3，即按Query的token和head量化。 | int64 | - |

query、query_rope和attention_out在各布局下的shape如下：

| input_layout | query的shape | query_rope的shape | attention_out的shape |
| :--- | :--- | :--- | :--- |
| BSH | (B, S_q, N×512) | (B, S_q, N×64) | (B, S_q, N×512) |
| BSND | (B, S_q, N, 512) | (B, S_q, N, 64) | (B, S_q, N, 512) |
| BNSD | (B, N, S_q, 512) | (B, N, S_q, 64) | (B, N, S_q, 512) |
| TND | (T_q, N, 512) | (T_q, N, 64) | (T_q, N, 512) |
| BSH_NBSD | (B, S_q, N×512) | (B, S_q, N×64) | (N, B, S_q, 512) |
| BSND_NBSD | (B, S_q, N, 512) | (B, S_q, N, 64) | (N, B, S_q, 512) |
| BNSD_NBSD | (B, N, S_q, 512) | (B, N, S_q, 64) | (N, B, S_q, 512) |
| TND_NTD | (T_q, N, 512) | (T_q, N, 64) | (N, T_q, 512) |

连续KV存储时，KV的输入布局与Query相同，忽略输出后缀：

| 输入布局 | key[0]、value[0]的shape | key_rope的shape |
| :--- | :--- | :--- |
| BSH | (B, S_kv, 512) | (B, S_kv, 64) |
| BSND | (B, S_kv, 1, 512) | (B, S_kv, 1, 64) |
| BNSD | (B, 1, S_kv, 512) | (B, 1, S_kv, 64) |
| TND | (T_kv, 1, 512) | (T_kv, 1, 64) |

PA时由KV张量的维度确定存储布局，三者必须采用相同的布局和物理块数：

| PA存储布局 | key[0]、value[0]的shape | key_rope的shape | 适用数据类型 |
| :--- | :--- | :--- | :--- |
| BSH | (P, L, 512) | (P, L, 64) | float16、bfloat16 |
| BNSD | (P, 1, L, 512) | (P, 1, L, 64) | float16、bfloat16 |
| NZ | (P, 1, 32, L, 16) | (P, 1, 4, L, 16) | float16、bfloat16 |
| NZ | (P, 1, 16, 128, 32) | (P, 1, 4, 128, 16) | int8，RoPE为bfloat16 |

NZ表示表中给出的五维数据排布，Tensor的数据格式仍为ND。

## 返回值说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| attention_out | Tensor | 必选 | 公式中的注意力计算结果。非量化时数据类型与query一致，INT8时为bfloat16。数据格式为ND，不支持非连续Tensor。 | float16、bfloat16 | 见[参数说明](#参数说明)。 |
| softmax_lse | Tensor | 条件输出 | 每个Query位置的Softmax log-sum-exp结果。softmax_lse_flag为True时输出有效结果，为False时返回空Tensor。数据格式为ND，不支持非连续Tensor。 | float32 | 非TND：(B, N, S_q, 1)；TND：(T_q, N, 1)；关闭时：(0,)。 |

返回二元组(attention_out, softmax_lse)。两个输出均位于query所在的NPU，softmax_lse的布局不随NBSD或NTD输出后缀变化。

## 约束说明

- 所有Tensor必须位于同一NPU。`key`和`value`必须为各含一个Tensor的列表，不支持多张量KV列表或不同batch分开传入。
- 非量化时query、key[0]、value[0]、query_rope、key_rope的数据类型必须全部一致，三个反量化系数均应传None。
- 除metadata和实际序列长度Tensor外，其余输入支持非连续ND格式Tensor。非连续输入可能产生连续化拷贝。
- actual_seq_lengths_q和actual_seq_lengths_kv必须一维连续，支持非零存储偏移的连续切片。metadata须直接使用flash_mla_metadata的输出，以满足连续、存储偏移为0及64字节对齐要求。
- 逻辑B不超过65536；Query存储S_q/T_q不超过4294967295，KV存储S_kv/T_kv或PA单batch容量不超过2147483647。实际可用shape还受设备内存和调度容量限制，INT8长度另有约束。
- PA中，每个batch实际使用的块数为ceil(KV实长/L)，不得超过block_table的列数；被访问的块索引必须位于[0, P)。非空计算不能用空block_table代替None。
- TND的序列实长是长度为B的累计数组，不含起始0；PA KV始终为逐batch绝对长度。完整缺省、广播、末项与容量规则见[序列实长约束](../../flash_mla_metadata/docs/torchapi_flash_mla_metadata.md#约束说明)。

- 掩码与窗口：

  atten_mask使用torch.bool时，False表示保留、True表示屏蔽；使用torch.int8或torch.uint8时，只能填0或1，1表示屏蔽。未使用掩码时传None，并在metadata接口设置`has_atten_mask=False`。

  | sparse_mode | 含义 | atten_mask要求 |
  | :--- | :--- | :--- |
  | 0 | 默认模式。未传掩码时计算全部有效KV；传入掩码时结合左右窗口计算。 | 非量化支持(B, Q_m, K_m)、(1, Q_m, K_m)、(B, 1, Q_m, K_m)、(1, 1, Q_m, K_m)，不支持二维普通掩码。INT8仅支持无掩码组合，另见下文。 |
  | 3 | 右下对齐的因果模式。 | 必须传入上三角屏蔽模板，标准shape为(2048, 2048)；非量化也支持(1, 2048, 2048)、(1, 1, 2048, 2048)。 |
  | 4 | 右下对齐的带状窗口模式，由pre_tokens和next_tokens指定可见窗口。 | 仅非量化支持，模板shape与模式3相同。 |
  | 9 | 树形注意力模式，用于推测解码。 | 必须传入tree mask。非TND按(B, S_q, S_q)组织；TND按每个batch的Q实长平方大小拼接为一维张量，长度为各batch的Q实长平方之和。 |

  - 普通掩码的Q_m至少覆盖非TND的S_q，TND时至少覆盖最大Query实长；K_m至少覆盖最大KV实长，PA时应覆盖该长度向上取整到block_size后的范围。
  - 模式3和4使用的标准模板可由`torch.ones(2048, 2048, dtype=torch.bool).triu(1)`构造。模板按窗口位置偏移读取，不能直接替换成完整的任意形状掩码。
  - 模式0无掩码时忽略pre_tokens和next_tokens；模式3和9将有效窗口参数设为2147483647和0。模式0有掩码或模式4时，窗口值先截断到[-2147483647, 2147483647]，再检查两者之和不小于0。
  - Tree mask描述本轮Query对应的树形可见关系。有效非空序列应满足Query长度不大于KV长度，并按上表布局准备足够的掩码数据。
  - 非量化时，若存在被全部屏蔽的行，可设置inner_precise为2或3开启行无效修正；INT8不支持这两个值。

- INT8全量化：

  - query、key[0]、value[0]为torch.int8；query_rope、key_rope和attention_out为torch.bfloat16。三个反量化系数必须同时提供，且均为torch.float32，不支持反量化offset。
  - num_heads只能为1、2、4、8、16、32、64、128；num_key_value_heads使用1，query_quant_mode必须为3，key_antiquant_mode和value_antiquant_mode必须为0。
  - 必须使用PA、NZ KV和block_size=128；Query仅支持BSH、BSND、TND及其对应输出后缀。
  - 非TND时S_q不超过32，Query实长参数的值和项数被忽略，实际使用存储S_q。S_q=1时仅支持sparse_mode=0且无掩码；S_q>1时必须使用sparse_mode=3或9并提供掩码。
  - TND时T_q不超过1048576，每个batch的Query实长在[0, 32]内；sparse_mode=0时不得提供掩码，sparse_mode=3或9时必须提供掩码。

- 仅支持前向计算和单算子模式（eager），不支持反向传播和TorchAir图模式。

- 调用`flash_mla_metadata`时，其`input_layer`对应本接口的`input_layout`；非量化设`quant_mode=0`，INT8设`quant_mode=1`。完整原型、参数、输出及逐项对应关系见[flash_mla_metadata接口文档](../../flash_mla_metadata/docs/torchapi_flash_mla_metadata.md)。

- 在同一设备、同一stream上依次完成序列实长更新、metadata生成和flash_mla计算，两次调用使用相同的有效核数。必须传入相同版本的序列实长，并保持头数、存储长度、布局、分页与稀疏配置一致。序列值或描述、调度配置变化后重新生成metadata；在计算完成前保持序列张量与metadata有效且不被修改。

## 确定性计算

默认支持确定性计算。

## 调用示例

调用前需安装包含FlashMla和FlashMlaMetadata的算子包，并按照[Torch扩展编译与安装说明](../../../torch_extension/README.md#build--installation)安装cann_ops_transformer。

- 单算子模式调用（非量化连续KV）：

  以下示例使用BNSD布局和float16输入。

  ```python
  import math

  import torch
  import torch_npu
  import cann_ops_transformer as ops

  torch.npu.set_device(0)
  device = torch.device("npu:0")
  torch.manual_seed(7)
  batch, heads, q_size, kv_size = 2, 4, 1, 128
  scale = 1.0 / math.sqrt(512 + 64)
  q = torch.randn(batch, heads, q_size, 512, dtype=torch.float16) * 0.1
  k = torch.randn(batch, 1, kv_size, 512, dtype=torch.float16) * 0.1
  v = torch.randn(batch, 1, kv_size, 512, dtype=torch.float16)
  qr = torch.randn(batch, heads, q_size, 64, dtype=torch.float16) * 0.1
  kr = torch.randn(batch, 1, kv_size, 64, dtype=torch.float16) * 0.1
  query, key, value, query_rope, key_rope = [
      x.to(device) for x in (q, k, v, qr, kr)
  ]
  seq_q = torch.full((batch,), q_size, dtype=torch.int64, device=device)
  seq_kv = torch.tensor([128, 96], dtype=torch.int64, device=device)
  metadata = ops.flash_mla_metadata(
      heads, batch, q_size, kv_size,
      actual_seq_lengths_q=seq_q,
      actual_seq_lengths_kv=seq_kv,
      input_layer="BNSD",
  )
  out, lse = ops.flash_mla(
      query, [key], [value], query_rope, key_rope, metadata, heads,
      actual_seq_lengths_q=seq_q,
      actual_seq_lengths_kv=seq_kv,
      input_layout="BNSD",
      scale_value=scale,
      softmax_lse_flag=True,
  )
  torch.npu.synchronize()
  print(out.shape, lse.shape)  # (2, 4, 1, 512), (2, 4, 1, 1)
  ```

- 单算子模式调用（INT8分页KV）：

  下面直接创建符合NZ维度要求的连续ND缓存，展示全量化所需的配置和系数shape。

  ```python
  import math

  import torch
  import torch_npu
  import cann_ops_transformer as ops

  torch.npu.set_device(0)
  device = torch.device("npu:0")
  batch, heads, block = 2, 4, 128
  query = torch.zeros((batch, 1, heads * 512), dtype=torch.int8, device=device)
  query_rope = torch.zeros(
      (batch, 1, heads * 64), dtype=torch.bfloat16, device=device
  )
  key = torch.randint(
      -8, 8, (batch, 1, 16, block, 32), dtype=torch.int8, device=device
  )
  value = key.clone()
  key_rope = torch.zeros(
      (batch, 1, 4, block, 16), dtype=torch.bfloat16, device=device
  )
  seq_kv = torch.tensor([96, 32], dtype=torch.int64, device=device)
  block_table = torch.tensor([[0], [1]], dtype=torch.int32, device=device)
  q_scale = torch.full(
      (batch, 1, heads), 0.03125, dtype=torch.float32, device=device
  )
  kv_scale = torch.full((1,), 0.0625, dtype=torch.float32, device=device)
  metadata = ops.flash_mla_metadata(
      heads, batch, 1, block,
      actual_seq_lengths_kv=seq_kv,
      input_layer="BSH", block_size=block, quant_mode=1,
  )
  out, lse = ops.flash_mla(
      query, [key], [value], query_rope, key_rope, metadata, heads,
      actual_seq_lengths_kv=seq_kv,
      block_table=block_table,
      dequant_scale_query=q_scale,
      key_antiquant_scale=kv_scale,
      value_antiquant_scale=kv_scale,
      scale_value=1.0 / math.sqrt(512 + 64),
      input_layout="BSH", block_size=block, query_quant_mode=3,
      softmax_lse_flag=True,
  )
  torch.npu.synchronize()
  assert out.dtype == torch.bfloat16
  print(out.shape, lse.shape)  # (2, 1, 2048), (2, 4, 1, 1)
  ```

TND与PA配套示例参见[flash_mla_metadata调用示例](../../flash_mla_metadata/docs/torchapi_flash_mla_metadata.md#调用示例)。包含序列实长更新和CPU结果比较的完整样例参见[test_torch_flash_mla.py](../examples/test_torch_flash_mla.py)。
