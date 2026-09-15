# flash\_mla\_metadata

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | × |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

`cann_ops_transformer.flash_mla_metadata`是[flash_mla](../../flash_mla/docs/torchapi_flash_mla.md)的前置接口。AI CPU根据设备上的Query和KV序列实长、存储长度、布局、稀疏配置、量化模式及当前有效核数生成调度元数据，供后续FlashMla注意力计算使用。

接口返回固定大小的NPU INT32张量，不返回注意力结果。序列实长由AI CPU在执行时读取，调用方无需将NPU实长张量复制到CPU。

非量化模式供FLOAT16和BFLOAT16共用。TND布局的Query和连续KV使用累计长度；分页注意力（Paged Attention，PA）的KV使用逐batch绝对长度。若累计数组为$c$，第$b$个batch的实长为：

$$
\ell_b=\begin{cases}
c_0,&b=0,\\
c_b-c_{b-1},&b>0.
\end{cases}
$$

累计数组不包含额外的起始0，项数等于逻辑batch数。本文中，B表示batch_size，N表示num_heads，S表示非TND存储序列长度，T表示TND存储总长度。

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

| 参数名 | 参数类型 | 必选/可选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| num_heads | int | 必选 | Query头数N，取值范围为[1, 128]；INT8时必须为2的幂。KV头数固定为1。 | int | - | - |
| batch_size | Optional[int] | 可选 | 默认None，TND时从Query实长张量的项数推导；其他布局必须显式传入。逻辑batch数B，取值范围为[0, 65536]。TND时必须等于Query和KV实长张量的项数。 | int | - | - |
| q_seq_size | Optional[int] | 可选 | 默认None，TND且B>0时从Query累计实长末项推导；其他情况必须显式传入。非TND下为Query存储S，TND下为Query存储总T，取值范围为[0, 4294967295]；INT8另有上限。 | int | - | - |
| kv_seq_size | Optional[int] | 可选 | 默认None，连续TND KV且B>0时从KV累计实长末项推导；其他情况必须显式传入。连续KV存储时为KV的S/T；PA时为block_table.shape[1]乘block_size。取值范围为[0, 2147483647]。 | int | - | - |
| actual_seq_lengths_q | Optional[Tensor] | 可选 | 默认None，Query实长。TND下使用累计长度；其他布局的缺省、广播规则见约束说明。必须为连续NPU张量。 | torch.int64 | ND | 1 |
| actual_seq_lengths_kv | Optional[Tensor] | 可选 | 默认None，KV实长。连续TND KV使用累计长度；PA使用绝对长度。必须为连续NPU张量。 | torch.int64 | ND | 1 |
| input_layer | str | 可选 | 默认"BSH"，对应flash_mla的input_layout，支持BSH、BSND、BNSD、TND、BSH_NBSD、BSND_NBSD、BNSD_NBSD、TND_NTD。 | str | - | - |
| block_size | int | 可选 | 默认0，表示连续KV存储。PA时为不大于1024的正16倍数；INT8时必须为128。 | int | - | - |
| sparse_mode | int | 可选 | 默认0。非量化支持0、3、4、9；INT8支持0、3、9。与flash_mla的sparse_mode一致。 | int | - | - |
| pre_tokens | int | 可选 | 默认2147483647，与flash_mla一致的左窗口参数。 | int | - | - |
| next_tokens | int | 可选 | 默认2147483647，与flash_mla一致的右窗口参数。 | int | - | - |
| has_atten_mask | bool | 可选 | 默认False。后续flash_mla使用非空atten_mask时设为True；不使用掩码时设为False。 | bool | - | - |
| quant_mode | int | 可选 | 默认0。整数0表示FLOAT16/BFLOAT16非量化，整数1表示INT8全量化；不接受torch.dtype作为该参数。 | int | - | - |
| device | Optional[Union[torch.device, str]] | 可选 | 默认None。优先采用actual_seq_lengths_q的设备，其次采用actual_seq_lengths_kv的设备；两者均未传入时使用当前NPU。显式指定时必须与已传入的实长张量位于同一NPU。 | torch.device、str | - | - |

## 返回值说明

| 输出名 | 参数类型 | shape | 数据类型 | 数据格式 | 说明 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| metadata | Tensor | (480,) | torch.int32 | ND | 新分配的NPU调度元数据，连续、存储偏移为0、首地址按64字节对齐，占1920字节。直接作为flash_mla的metadata入参。 |

batch_size为0时仍返回上述固定shape的调度张量。`device="meta"`仅用于输出shape和dtype推导，得到的Meta张量不能用于NPU计算。

## 约束说明

### 尺寸参数推导

- batch_size、q_seq_size、kv_seq_size分别独立支持None；显式传入时使用传入值并保留原有范围、序列及配套调用一致性校验。0是合法尺寸，不表示省略。
- TND和TND_NTD下，batch_size可从actual_seq_lengths_q的项数推导；两实长张量的项数必须与最终B一致。
- B>0时，TND Query的q_seq_size可从Query累计实长末项推导；连续TND KV的kv_seq_size可从KV累计实长末项推导。AI CPU读取累计值、校验并完成推导，Host不读取或复制设备实长值。
- B=0时没有累计末项，q_seq_size和kv_seq_size必须显式传入。非TND尺寸和PA的kv_seq_size也必须显式传入；不从广播项数、最大实长或按block_size向上取整的实长猜测存储容量。
- 推导得到的尺寸仍须与后续flash_mla的实际张量shape和分页表容量一致。Meta/FakeTensor只检查描述层面的推导条件，不读取累计末项。

### 序列实长

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

### 布局、分页与量化

- input_layer描述Query输入布局及可选的输出布局后缀。连续KV与Query输入布局一致；PA KV的BSH、BNSD或NZ存储排布由flash_mla的KV张量shape确定，本接口不接收独立的KV布局参数。
- q_seq_size和kv_seq_size描述存储长度或容量，不能用最大实长替代。PA时kv_seq_size必须为分页表列数乘block_size，不能使用物理缓存块总数计算。
- quant_mode=0用于FLOAT16和BFLOAT16；quant_mode=1要求num_heads为1、2、4、8、16、32、64、128之一，block_size=128，Query仅支持BSH、BSND、TND及对应输出后缀。
- INT8非TND的q_seq_size不超过32；TND总q_seq_size不超过1048576，且每个batch的Query实长在[0, 32]内。配套主接口还要求NZ KV、BFLOAT16 RoPE、三个FLOAT32反量化系数及query_quant_mode=3，详见[INT8全量化约束](../../flash_mla/docs/torchapi_flash_mla.md#int8全量化)。

### 稀疏模式与窗口

- sparse_mode=0表示默认模式；3表示右下对齐因果模式；4表示右下对齐带状窗口模式，仅非量化支持；9表示树形注意力模式。
- sparse_mode非0时必须设置has_atten_mask=True。窗口值与主接口保持一致，其归一化规则和实际掩码shape见[掩码与窗口](../../flash_mla/docs/torchapi_flash_mla.md#掩码与窗口)。本接口只接收掩码是否存在，不接收掩码张量。
- INT8非TND时，q_seq_size=1仅支持sparse_mode=0且has_atten_mask=False；q_seq_size>1必须使用sparse_mode=3或9且has_atten_mask=True。
- INT8 TND时，sparse_mode=0必须设置has_atten_mask=False；sparse_mode=3或9必须设置has_atten_mask=True。

### 配套调用一致性

- 在同一NPU、同一stream中依次更新实长、调用本接口、调用flash_mla。生成和使用metadata时的有效核数必须一致。
- 两次调用应使用相同版本的实长张量，包含相同的存在性、项数和数值；在flash_mla执行完成前保持这些张量与metadata有效且不被修改。
- 实长、序列描述或调度配置变化后重新生成metadata。配置不变时，FLOAT16与BFLOAT16可共用quant_mode=0的结果；输出布局后缀不影响调度。
- metadata须原样传入配套主接口，不能手工构造、修改、切片，或使用flash_attn_metadata、sparse_flash_mla_metadata等其他接口的结果替代。
- 当前仅提供前向eager调用，暂不支持反向传播和torchair图模式。

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

- 默认支持确定性计算。

## 调用示例

### 单算子模式：TND与分页KV

运行前按[编译与安装说明](../../../torch_extension/README.md#build--installation)安装torch扩展，并安装包含两个算子的算子包及FlashMlaMetadata的AI CPU实现。下面使用BFLOAT16、TND Query与三维PA缓存；Query实长为累计值，KV实长为绝对值。

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
    heads, batch, total_q, block_table.shape[1] * block,
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

### 图模式调用（暂不支持）

当前未提供torchair图转换接口，请使用单算子模式。Meta/FakeTensor仅用于输出shape和dtype推导。
