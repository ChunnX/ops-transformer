# flash\_mla

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

`cann_ops_transformer.flash_mla`调用FlashMla算子，完成旋转位置编码（Rotary Position Embedding，RoPE）分离的多头潜在注意力（Multi-head Latent Attention，MLA）前向计算。支持FLOAT16、BFLOAT16非量化及INT8全量化输入，可使用连续KV存储或分页注意力（Paged Attention，PA）。

调用前必须使用[flash_mla_metadata](../../flash_mla_metadata/docs/torchapi_flash_mla_metadata.md)生成调度元数据。Query的非RoPE部分与RoPE部分分开传入；接口使用已经完成位置编码的张量，不在内部执行RoPE变换。

非量化场景下，每个batch、每个Query头的计算为：

$$
S=\mathrm{scale\_value}\left(QK^T+Q_{\mathrm{rope}}K_{\mathrm{rope}}^T\right)+M
$$

$$
\mathrm{attention\_out}=\operatorname{softmax}(S)V,\qquad
\mathrm{softmax\_lse}_i=\log\sum_j\exp(S_{ij})
$$

$Q$、$K$、$V$分别对应`query`、`key[0]`、`value[0]`，两个RoPE变量分别对应`query_rope`和`key_rope`。$M$由掩码和窗口共同确定，保留位置为0，屏蔽位置在数学上为负无穷。INT8场景使用`dequant_scale_query`、`key_antiquant_scale`、`value_antiquant_scale`分别恢复Q、K、V的数值尺度，RoPE部分使用BFLOAT16。

本文中，B表示逻辑batch数，N表示Query头数，S_q和S_kv表示存储序列长度，T_q和T_kv表示TND布局下的存储总长度，P表示缓存的物理块数，L表示block_size。Query、Key和Value的每头维度均为512，RoPE部分的每头维度为64，KV头数固定为1。

## 函数原型

通过`cann_ops_transformer.flash_mla`调用，完整签名如下。`*`之后的参数只能按关键字传入。

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

所有输入Tensor均须位于`query`所在的NPU，数据格式为ND。非连续张量的支持情况见约束说明。

| 参数名 | 参数类型 | 必选/可选 | 描述 | 数据类型 | 数据格式 | 维度 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| query | Tensor | 必选 | 非RoPE Query，每头维度为512，shape见Query布局表。 | torch.float16、torch.bfloat16、torch.int8 | ND | 3或4 |
| key | List[Tensor] | 必选 | 必须恰好包含一个Key张量，shape见KV布局表。 | 与query一致 | ND | 3、4或5 |
| value | List[Tensor] | 必选 | 必须恰好包含一个Value张量，shape与key[0]一致。共享KV场景可将同一张量分别放入两个列表。 | 与query一致 | ND | 3、4或5 |
| query_rope | Tensor | 必选 | Query的RoPE部分，每头维度为64，shape见Query布局表。 | 非量化与query一致；INT8时为torch.bfloat16 | ND | 3或4 |
| key_rope | Tensor | 必选 | Key的RoPE部分，shape见KV布局表。 | 与query_rope一致 | ND | 3、4或5 |
| metadata | Tensor | 必选 | flash_mla_metadata返回的张量，shape固定为(480,)，连续、零存储偏移、64字节对齐。 | torch.int32 | ND | 1 |
| num_heads | int | 必选 | Query头数N，取值范围为[1, 128]；INT8时须为2的幂。 | int | - | - |
| atten_mask | Optional[Tensor] | 可选 | 默认None。shape由sparse_mode决定，详见掩码约束。 | torch.bool、torch.int8、torch.uint8 | ND | 1、2、3或4 |
| actual_seq_lengths_q | Optional[Tensor] | 可选 | 默认None，连续实长张量；TND时非空batch必须提供(B,)累计长度。 | torch.int64 | ND | 1 |
| actual_seq_lengths_kv | Optional[Tensor] | 可选 | 默认None，连续实长张量；PA或TND时非空batch必须提供。 | torch.int64 | ND | 1 |
| block_table | Optional[Tensor] | 可选 | 默认None。PA时必选，shape为(B, max_blocks)；连续KV存储时传None。 | torch.int32 | ND | 2 |
| dequant_scale_query | Optional[Tensor] | 可选 | 默认None，INT8时必选；BSH/BSND为(B, S_q, N)，TND为(T_q, N)，输出布局后缀不影响shape。 | torch.float32 | ND | 2或3 |
| key_antiquant_scale | Optional[Tensor] | 可选 | 默认None，INT8时必选；shape为(1,)，表示Key的per-tensor反量化系数。 | torch.float32 | ND | 1 |
| value_antiquant_scale | Optional[Tensor] | 可选 | 默认None，INT8时必选；shape为(1,)，表示Value的per-tensor反量化系数。 | torch.float32 | ND | 1 |
| scale_value | float | 可选 | 默认1.0，对两路QK乘积之和进行缩放。接口不会自动计算缩放系数；示例采用1/sqrt(512+64)。 | float | - | - |
| pre_tokens | int | 可选 | 默认2147483647，左窗口参数，按sparse_mode解释。 | int | - | - |
| next_tokens | int | 可选 | 默认2147483647，右窗口参数，按sparse_mode解释。 | int | - | - |
| input_layout | str | 可选 | 默认"BSH"，支持下表中的8种输入/输出布局组合，区分大小写。 | str | - | - |
| num_key_value_heads | int | 可选 | 默认1。非量化非空输入仅支持1；INT8在num_heads=1时也支持0，此时有效KV头数仍为1。 | int | - | - |
| sparse_mode | int | 可选 | 默认0。非量化支持0、3、4、9；INT8支持0、3、9。 | int | - | - |
| inner_precise | int | 可选 | 默认0。非量化支持0、1、2、3，其中2和3开启行无效修正；INT8仅支持0和1。 | int | - | - |
| block_size | int | 可选 | 默认0。PA时为不大于1024的正16倍数；INT8时必须为128。 | int | - | - |
| softmax_lse_flag | bool | 可选 | 默认False。True时返回有效的softmax_lse，False时第二个输出为空张量。 | bool | - | - |
| key_antiquant_mode | int | 可选 | 默认0，当前仅支持0，即Key使用per-tensor反量化。 | int | - | - |
| value_antiquant_mode | int | 可选 | 默认0，当前仅支持0，即Value使用per-tensor反量化。 | int | - | - |
| query_quant_mode | int | 可选 | 默认0。非量化仅支持0；INT8必须显式设为3，即按Query的token和head量化。 | int | - | - |

### Query布局

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

### KV布局

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
| BSH | (P, L, 512) | (P, L, 64) | FLOAT16、BFLOAT16 |
| BNSD | (P, 1, L, 512) | (P, 1, L, 64) | FLOAT16、BFLOAT16 |
| NZ | (P, 1, 32, L, 16) | (P, 1, 4, L, 16) | FLOAT16、BFLOAT16 |
| NZ | (P, 1, 16, 128, 32) | (P, 1, 4, 128, 16) | INT8，RoPE为BFLOAT16 |

这里的NZ表示五维数据排布，Tensor仍须使用基础ND存储。不能以`torch_npu.npu_format_cast`得到的私有存储格式代替上述五维ND张量。

## 返回值说明

返回二元组`(attention_out, softmax_lse)`，两个输出均位于输入NPU，采用ND格式。

| 输出名 | shape | dtype | 说明 |
| :--- | :--- | :--- | :--- |
| attention_out | 见Query布局表 | 非量化与query一致；INT8输入为torch.bfloat16 | 注意力计算结果。 |
| softmax_lse | 开启时：非TND为(B, N, S_q, 1)，TND为(T_q, N, 1)；关闭时为(0,) | torch.float32 | 每个Query位置的log-sum-exp；不随NBSD或NTD输出后缀转置，关闭时返回空Tensor而非None。 |

## 约束说明

### 参数一致性与存储

- 所有Tensor必须位于同一NPU。`key`和`value`必须为各含一个Tensor的列表，不支持多张量KV列表或不同batch分开传入。
- 非量化时query、key[0]、value[0]、query_rope、key_rope的数据类型必须全部一致，三个反量化系数均应传None。
- query、query_rope、atten_mask、block_table及量化系数支持非连续ND张量，由执行器连续化。KV缓存和key_rope在torch封装中按需连续化；非量化PA在运行时支持TensorV2时可保留仅第0维不连续的正stride视图，其他情况会复制为连续张量。
- actual_seq_lengths_q、actual_seq_lengths_kv必须一维连续，支持非零存储偏移的连续切片；metadata必须连续、零存储偏移、首地址64字节对齐。
- 逻辑B不超过65536；Query存储S_q/T_q不超过4294967295，KV存储S_kv/T_kv或PA单batch容量不超过2147483647。实际可用shape还受设备内存和调度容量限制，INT8长度另有约束。
- PA中，每个batch实际使用的块数为ceil(KV实长/L)，不得超过block_table的列数；被访问的块索引必须位于[0, P)。非空计算不能用空block_table代替None。
- TND的序列实长是长度为B的累计数组，不含起始0；PA KV始终为逐batch绝对长度。完整缺省、广播、末项与容量规则见[序列实长约束](../../flash_mla_metadata/docs/torchapi_flash_mla_metadata.md#序列实长)。

### 掩码与窗口

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

### INT8全量化

- query、key[0]、value[0]为torch.int8；query_rope、key_rope和attention_out为torch.bfloat16。三个反量化系数必须同时提供，且均为torch.float32，不支持反量化offset。
- num_heads只能为1、2、4、8、16、32、64、128；num_key_value_heads使用1，query_quant_mode必须为3，key_antiquant_mode和value_antiquant_mode必须为0。
- 必须使用PA、NZ KV和block_size=128；Query仅支持BSH、BSND、TND及其对应输出后缀。
- 非TND时S_q不超过32，Query实长参数的值和项数被忽略，实际使用存储S_q。S_q=1时仅支持sparse_mode=0且无掩码；S_q>1时必须使用sparse_mode=3或9并提供掩码。
- TND时T_q不超过1048576，每个batch的Query实长在[0, 32]内；sparse_mode=0时不得提供掩码，sparse_mode=3或9时必须提供掩码。

### 执行模式

- 仅支持前向eager调用，暂不支持反向传播；Meta/FakeTensor实现用于推导输出shape和dtype。
- 图模式调用（暂不支持）：当前未提供torchair图转换接口。

调用`flash_mla_metadata`时，其`input_layer`对应本接口的`input_layout`；非量化设`quant_mode=0`，INT8设`quant_mode=1`。完整原型、参数、输出及逐项对应关系见[flash_mla_metadata接口文档](../../flash_mla_metadata/docs/torchapi_flash_mla_metadata.md)。

在同一设备、同一stream上依次完成序列实长更新、metadata生成和flash_mla计算，两次调用使用相同的有效核数。必须传入相同版本的序列实长，并保持头数、存储长度、布局、分页与稀疏配置一致。序列值或描述、调度配置变化后重新生成metadata；在计算完成前保持序列张量与metadata有效且不被修改。

## 确定性计算

- 默认支持确定性计算。比较结果时应保持输入、软件版本、硬件及有效核数等条件一致。

## 调用示例

### 单算子模式：非量化连续KV

先按[编译与安装说明](../../../torch_extension/README.md#build--installation)安装torch扩展，并安装包含FlashMla与FlashMlaMetadata的算子包及AI CPU实现。以下示例使用默认NPU stream，计算FP16注意力并与CPU参考值比较。

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
actual, actual_lse = out.cpu().double(), lse.cpu().double()
for b, length in enumerate([128, 96]):
    scores = (
        q[b].double() @ k[b, :, :length].double().transpose(-1, -2)
        + qr[b].double() @ kr[b, :, :length].double().transpose(-1, -2)
    ) * scale
    expected = scores.softmax(-1) @ v[b, :, :length].double()
    torch.testing.assert_close(actual[b], expected, rtol=4e-3, atol=4e-3)
    torch.testing.assert_close(
        actual_lse[b], scores.logsumexp(-1, keepdim=True), rtol=4e-3, atol=4e-3
    )
print(out.shape, lse.shape)  # (2, 4, 1, 512), (2, 4, 1, 1)
```

### 单算子模式：INT8分页KV

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

TND与PA配套示例见[metadata调用示例](../../flash_mla_metadata/docs/torchapi_flash_mla_metadata.md#调用示例)。包含自定义stream、序列实长更新及CPU参考比较的可执行脚本见[test_torch_flash_mla.py](../examples/test_torch_flash_mla.py)；其他布局与量化模式可参考[设备测试脚本](../../flash_mla_metadata/tests/st/test_flash_mla_metadata.py)。

### 图模式调用（暂不支持）

当前未提供torchair图模式示例，请使用上述单算子模式。
