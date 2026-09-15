# flash\_mla\_metadata

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

  flash_mla_metadata是[flash_mla](../../flash_mla/docs/torchapi_flash_mla.md)的前置接口，根据Query和KV的实际序列长度、存储长度、布局、稀疏配置和量化模式，生成各计算核的任务划分信息，用于FlashMla的负载均衡。

  支持连续KV存储和分页注意力（Paged Attention，PA）存储。实际序列长度以NPU Tensor传入，调用方无需将其复制到CPU。接口返回metadata，由flash_mla使用该结果完成注意力计算。

## 函数原型

`*`之后的参数只能按关键字传入。

```python
cann_ops_transformer.flash_mla_metadata(
    num_heads,
    batch_size=None,
    q_seq_size=None,
    kv_seq_size=None,
    *,
    actual_seq_lengths_q=None,
    actual_seq_lengths_kv=None,
    input_layer="BSH",
    block_size=0,
    sparse_mode=0,
    pre_tokens=2147483647,
    next_tokens=2147483647,
    has_atten_mask=False,
    quant_mode=0,
    device=None,
) -> Tensor
```

## 参数说明

> [!NOTE]
>
> B表示batch_size，即输入样本批量大小；N表示num_heads，即Query头数；S表示非TND布局的存储序列长度；T表示TND布局下各batch序列长度的累加和。

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| num_heads | int | 必选 | Query头数N，取值范围为[1, 128]；INT8时必须为2的幂。KV头数固定为1。 | int64 | - |
| batch_size | int | 可选 | 默认值为None，TND时从Query实长张量的项数推导；其他布局必须显式传入。逻辑batch数B，取值范围为[0, 65536]。TND时必须等于Query和KV实长张量的项数。 | int64 | - |
| q_seq_size | int | 可选 | 默认值为None，TND且B>0时从Query累计实长末项推导；其他情况必须显式传入。非TND下为Query存储S，TND下为Query存储总T，取值范围为[0, 4294967295]；INT8另有上限。 | int64 | - |
| kv_seq_size | int | 可选 | 默认值为None，连续TND KV且B>0时从KV累计实长末项推导；其他情况必须显式传入。连续KV存储时为KV的S/T；PA时为block_table.shape[1]乘block_size。取值范围为[0, 2147483647]。 | int64 | - |
| actual_seq_lengths_q | Tensor | 可选 | 默认值为None，Query实长。TND下使用累计长度；其他布局的缺省、广播规则见约束说明。必须为连续NPU张量。数据格式为ND，不支持非连续Tensor。 | int64 | (B,)；非TND的广播及缺省规则见[序列实长](#约束说明)。 |
| actual_seq_lengths_kv | Tensor | 可选 | 默认值为None，KV实长。连续TND KV使用累计长度；PA使用绝对长度。必须为连续NPU张量。数据格式为ND，不支持非连续Tensor。 | int64 | (B,)；非TND的广播及缺省规则见[序列实长](#约束说明)。 |
| input_layer | str | 可选 | 默认值为"BSH"，对应flash_mla的input_layout，支持BSH、BSND、BNSD、TND、BSH_NBSD、BSND_NBSD、BNSD_NBSD、TND_NTD。 | string | - |
| block_size | int | 可选 | 默认值为0，表示连续KV存储。PA时为不大于1024的正16倍数；INT8时必须为128。 | int64 | - |
| sparse_mode | int | 可选 | 默认值为0。非量化支持0、3、4、9；INT8支持0、3、9。与flash_mla的sparse_mode一致。 | int64 | - |
| pre_tokens | int | 可选 | 默认值为2147483647，与flash_mla一致的左窗口参数。 | int64 | - |
| next_tokens | int | 可选 | 默认值为2147483647，与flash_mla一致的右窗口参数。 | int64 | - |
| has_atten_mask | bool | 可选 | 默认值为False。后续flash_mla使用非空atten_mask时设为True；不使用掩码时设为False。 | bool | - |
| quant_mode | int | 可选 | 量化模式，0表示float16/bfloat16非量化，1表示int8全量化。默认值为0。 | int64 | - |
| device | torch.device或str | 可选 | 默认值为None。优先采用actual_seq_lengths_q的设备，其次采用actual_seq_lengths_kv的设备；两者均未传入时使用当前NPU。显式指定时必须与已传入的实长张量位于同一NPU。 | - | - |

## 返回值说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| metadata | Tensor | 必选 | FlashMla的负载均衡结果，作为flash_mla的metadata输入。数据格式为ND，不支持非连续Tensor。 | int32 | (480,) |

输出位于device指定或推导出的NPU。batch_size为0时仍返回上述shape的Tensor。

## 约束说明

- 尺寸参数推导：

  - batch_size、q_seq_size和kv_seq_size分别支持None，省略时按以下规则推导；显式传入时使用传入值。显式传入0表示对应尺寸为0。
  - TND和TND_NTD下，batch_size可从actual_seq_lengths_q的项数推导；两实长张量的项数必须与最终B一致。
  - B>0时，TND Query的q_seq_size可从Query累计实长末项推导；连续TND KV的kv_seq_size可从KV累计实长末项推导。推导在设备侧完成。
  - B=0时没有累计末项，q_seq_size和kv_seq_size必须显式传入。非TND尺寸和PA的kv_seq_size也必须显式传入。
  - 推导得到的尺寸仍须与后续flash_mla的实际张量shape和分页表容量一致。

- 序列实长：

  - 传入的实长张量必须一维连续，dtype为torch.int64，且均位于输出NPU。允许非零存储偏移的连续切片，不支持非连续实长张量。
  - 对采用逐batch绝对长度的输入，单项张量表示广播到所有batch；不少于B项时只使用前B项。除下表中的INT8非TND Query外，参与计算的实长应非负且不超过对应存储容量。
  - TND下两实长张量均须恰好包含B项。其中采用累计长度的张量应非负、单调不减，最后一项等于对应T；首项为第一个batch的实长，不在数组前额外添加0。PA KV使用绝对长度，不受累计规则约束。

  | 场景 | Query实长 | KV实长 |
  | :--- | :--- | :--- |
  | 非量化、非TND、连续KV | None或空张量表示使用q_seq_size；支持单项广播或不少于B项绝对长度。 | None或空张量表示使用kv_seq_size；支持单项广播或不少于B项绝对长度。 |
  | 非量化、非TND、PA | 与上一行相同。 | B>0时必须提供非空张量；支持单项广播或不少于B项绝对长度，每项不超过kv_seq_size。 |
  | 非量化、TND、连续KV | B>0时必传(B,)累计长度，末项等于q_seq_size。 | B>0时必传(B,)累计长度，末项等于kv_seq_size。 |
  | TND、PA | B>0时必传(B,)累计长度，末项等于q_seq_size。 | B>0时必传(B,)绝对长度，每项不超过kv_seq_size；不进行累加。 |
  | INT8、非TND、PA | 实际使用存储q_seq_size；忽略传入Query实长的值和项数，仍检查设备、dtype、维度和连续性。 | B>0时必须提供非空张量；支持单项广播或不少于B项绝对长度。 |

  例如，两个batch的Query实长为1和2时，TND的actual_seq_lengths_q应为`[1, 3]`；KV实长为96和32时，连续TND KV的累计数组为`[96, 128]`，PA KV的绝对长度数组为`[96, 32]`。

- 布局、分页与量化：

  - input_layer描述Query输入布局及可选的输出布局后缀。连续KV与Query输入布局一致；PA KV的BSH、BNSD或NZ存储排布由flash_mla的KV张量shape确定，本接口不接收独立的KV布局参数。
  - q_seq_size和kv_seq_size描述存储长度或容量，不能用最大实长替代。PA时kv_seq_size必须为分页表列数乘block_size，不能使用物理缓存块总数计算。
  - quant_mode=0用于FLOAT16和BFLOAT16；quant_mode=1要求num_heads为1、2、4、8、16、32、64、128之一，block_size=128，Query仅支持BSH、BSND、TND及对应输出后缀。
  - INT8非TND的q_seq_size不超过32；TND总q_seq_size不超过1048576，且每个batch的Query实长在[0, 32]内。配套主接口还要求NZ KV、BFLOAT16 RoPE、三个FLOAT32反量化系数及query_quant_mode=3，详见[INT8全量化约束](../../flash_mla/docs/torchapi_flash_mla.md#约束说明)。

- 稀疏模式与窗口：

  - sparse_mode=0表示默认模式；3表示右下对齐因果模式；4表示右下对齐带状窗口模式，仅非量化支持；9表示树形注意力模式。
  - sparse_mode非0时必须设置has_atten_mask=True。窗口值与主接口保持一致，其归一化规则和实际掩码shape见[掩码与窗口](../../flash_mla/docs/torchapi_flash_mla.md#约束说明)。本接口只接收掩码是否存在，不接收掩码张量。
  - INT8非TND时，q_seq_size=1仅支持sparse_mode=0且has_atten_mask=False；q_seq_size>1必须使用sparse_mode=3或9且has_atten_mask=True。
  - INT8 TND时，sparse_mode=0必须设置has_atten_mask=False；sparse_mode=3或9必须设置has_atten_mask=True。

- 配套调用约束：

  - 在同一NPU、同一stream中依次更新实长、调用本接口、调用flash_mla。生成和使用metadata时的有效核数必须一致。
  - 两次调用应使用相同版本的实长张量，包含相同的存在性、项数和数值；在flash_mla执行完成前保持这些张量与metadata有效且不被修改。
  - 实长、序列描述或调度配置变化后重新生成metadata。配置不变时，FLOAT16与BFLOAT16可共用quant_mode=0的结果；输出布局后缀不影响调度。
  - metadata须原样传入flash_mla，以满足连续、存储偏移为0及64字节对齐要求；不支持手工构造、修改或切片。
  - 仅支持单算子模式（eager），不支持反向传播和TorchAir图模式。

  本接口的输出传给[flash_mla](../../flash_mla/docs/torchapi_flash_mla.md)，主接口的完整原型、参数和输出见该文档。两接口的配置对应关系如下：

  | flash_mla_metadata参数 | flash_mla中的对应项 |
  | :--- | :--- |
  | num_heads | num_heads。 |
  | batch_size | 非TND时为query.shape[0]；TND时为actual_seq_lengths_q.numel()。 |
  | q_seq_size | BSH/BSND为query.shape[1]，BNSD为query.shape[2]，TND为query.shape[0]。输出后缀不影响取值。 |
  | kv_seq_size | 连续KV存储时取key[0]的S/T；PA时取block_table.shape[1]乘block_size。 |
  | actual_seq_lengths_q、actual_seq_lengths_kv | 同名实长张量。 |
  | input_layer | input_layout。 |
  | block_size | block_size；连续KV时为0，且主接口block_table=None。 |
  | sparse_mode、pre_tokens、next_tokens | 同名稀疏模式及窗口参数。 |
  | has_atten_mask | 是否使用非空atten_mask。 |
  | quant_mode | query为FLOAT16/BFLOAT16时为0，query为INT8时为1。 |
  | device | query所在NPU。 |

## 确定性计算

默认支持确定性计算。

## 调用示例

调用前需安装包含FlashMla和FlashMlaMetadata的算子包，并按照[Torch扩展编译与安装说明](../../../torch_extension/README.md#build--installation)安装cann_ops_transformer。

- 单算子模式调用（TND与分页KV）：

  以下示例使用bfloat16输入、TND Query和分页KV。Query实际序列长度为累计值，KV实际序列长度为各batch的长度。batch_size和q_seq_size从Query实际序列长度推导，PA场景下的kv_seq_size须显式传入。

  ```python
  import math

  import torch
  import torch_npu
  import cann_ops_transformer as ops

  torch.npu.set_device(0)
  device = torch.device("npu:0")
  batch, heads, total_q, block = 2, 4, 3, 128
  query = torch.randn(
      (total_q, heads, 512), dtype=torch.bfloat16, device=device
  ) * 0.1
  query_rope = torch.randn(
      (total_q, heads, 64), dtype=torch.bfloat16, device=device
  ) * 0.1
  kv = torch.randn((2, block, 512), dtype=torch.bfloat16, device=device)
  key_rope = torch.randn((2, block, 64), dtype=torch.bfloat16, device=device)
  seq_q = torch.tensor([1, 3], dtype=torch.int64, device=device)
  seq_kv = torch.tensor([96, 32], dtype=torch.int64, device=device)
  block_table = torch.tensor([[1], [0]], dtype=torch.int32, device=device)
  metadata = ops.flash_mla_metadata(
      heads,
      kv_seq_size=block_table.shape[1] * block,
      actual_seq_lengths_q=seq_q,
      actual_seq_lengths_kv=seq_kv,
      input_layer="TND_NTD",
      block_size=block,
      device=device,
  )
  out, lse = ops.flash_mla(
      query, [kv], [kv], query_rope, key_rope, metadata, heads,
      actual_seq_lengths_q=seq_q,
      actual_seq_lengths_kv=seq_kv,
      block_table=block_table,
      input_layout="TND_NTD",
      block_size=block,
      scale_value=1.0 / math.sqrt(512 + 64),
      softmax_lse_flag=True,
  )
  torch.npu.synchronize()
  assert metadata.shape == (480,) and metadata.dtype == torch.int32
  assert out.shape == (heads, total_q, 512)
  assert lse.shape == (total_q, heads, 1)
  print(out.shape, lse.shape)
  ```
