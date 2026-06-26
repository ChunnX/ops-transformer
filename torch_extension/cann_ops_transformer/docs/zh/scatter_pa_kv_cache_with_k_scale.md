# scatter\_pa\_kv\_cache\_with\_k\_scale

## 产品支持情况

- <term>Ascend 950PR/Ascend 950DT</term>：支持
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
- <term>Atlas 推理系列产品</term>：不支持
- <term>Atlas 训练系列产品</term>：不支持

## 功能说明

- 接口功能：

  `scatter_pa_kv_cache_with_k_scale`接口用于将FP8格式的key/value和对应的key_scale按token偏移更新到KV cache中，主要应用于大语言模型推理场景下的PagedAttention KV Cache更新。

## 函数原型

```python
cann_ops_transformer.scatter_pa_kv_cache_with_k_scale(key, value, key_cache, value_cache, slot_mapping, key_scale, key_scale_cache, *, cache_layout='BNBD') -> Tuple[Tensor, Tensor, Tensor]
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| key | Tensor | 必选 | 表示需要更新的key输入。数据格式为ND，支持非连续的Tensor。 | float8_e5m2/float8_e4m3fn | (num_tokens, num_head, k_head_size) |
| value | Tensor | 必选 | 表示需要更新的value输入，dtype需与key一致。数据格式为ND，支持非连续的Tensor。 | float8_e5m2/float8_e4m3fn | (num_tokens, num_head, v_head_size) |
| key_cache | Tensor | 必选 | 表示被更新的key cache，dtype需与key一致。数据格式为ND，支持非连续的Tensor。 | float8_e5m2/float8_e4m3fn | (num_blocks, num_head, block_size, k_head_size) |
| value_cache | Tensor | 必选 | 表示被更新的value cache，dtype需与key一致。数据格式为ND，支持非连续的Tensor。 | float8_e5m2/float8_e4m3fn | (num_blocks, num_head, block_size, v_head_size) |
| slot_mapping | Tensor | 必选 | 表示每个token在cache中的偏移索引，取值范围为\[0, num_blocks\*block_size-1\]，超出范围的slot将被跳过。数据格式为ND，支持非连续的Tensor。 | int32/int64 | (num_tokens,) |
| key_scale | Tensor | 必选 | 表示需要更新的key scale输入，与key配合用于FP8反量化。数据格式为ND，支持非连续的Tensor。 | float32 | (num_tokens, num_head) |
| key_scale_cache | Tensor | 必选 | 表示被更新的key scale cache。数据格式为ND，支持非连续的Tensor。 | float32 | (num_blocks, num_head, block_size, 1) |
| cache_layout | str | 可选 | 表示cache的布局格式，当前仅支持"BNBD"，默认值为"BNBD"。 | string | - |

## 返回值说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
|--------|----------|-----------|------|----------|-------------|
| key_cache_out | Tensor | 必选 | 表示更新后的key cache，shape和dtype与输入key_cache相同。数据格式为ND。 | float8_e5m2/float8_e4m3fn | (num_blocks, num_head, block_size, k_head_size) |
| value_cache_out | Tensor | 必选 | 表示更新后的value cache，shape和dtype与输入value_cache相同。数据格式为ND。 | float8_e5m2/float8_e4m3fn | (num_blocks, num_head, block_size, v_head_size) |
| key_scale_cache_out | Tensor | 必选 | 表示更新后的key scale cache，shape和dtype与输入key_scale_cache相同。数据格式为ND。 | float32 | (num_blocks, num_head, block_size, 1) |

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和aclgraph模式。
- key和value的dtype必须一致，且必须为float8_e5m2或float8_e4m3fn。
- key_cache和value_cache的dtype必须与key一致。
- key_scale和key_scale_cache的dtype必须为float32。
- slot_mapping的取值范围为\[0, num_blocks\*block_size-1\]，超出范围的slot将被跳过，不更新对应cache。
- num_tokens表示当前需要更新到cache中的token数量。
- num_blocks表示KV cache分块的总数，block_size表示每个分块包含的token数。
- num_head表示注意力头数，k_head_size和v_head_size分别表示key和value的头维度大小。

## 确定性计算

- 默认支持确定性计算

## 调用说明

### FP8 KV Cache更新示例

下面示例演示如何将若干token的FP8格式key/value及其对应的key_scale按`slot_mapping`更新到PagedAttention的KV cache中。

```python
import torch
import torch_npu
import cann_ops_transformer

torch_npu.npu.set_device(0)

# 形状定义
num_tokens = 4       # 本次需要写入的token数量
num_head = 8         # 注意力头数
k_head_size = 128    # key头维度
v_head_size = 128    # value头维度
num_blocks = 2       # KV cache分块总数
block_size = 16      # 每个分块包含的token数

# FP8 dtype（float8_e5m2 与 float8_e4m3fn 均支持，此处以 e4m3fn 为例）
kv_dtype = torch.float8_e4m3fn

# 构造输入：key/value 为待写入的新数据
key = torch.randn(num_tokens, num_head, k_head_size, dtype=torch.float32, device="npu").to(kv_dtype)
value = torch.randn(num_tokens, num_head, v_head_size, dtype=torch.float32, device="npu").to(kv_dtype)

# 构造KV cache（被inplace更新的目标，初始置0便于校验）
key_cache = torch.zeros(num_blocks, num_head, block_size, k_head_size, dtype=kv_dtype, device="npu")
value_cache = torch.zeros(num_blocks, num_head, block_size, v_head_size, dtype=kv_dtype, device="npu")

# 构造slot_mapping：每个token在cache中的偏移，取值范围 [0, num_blocks*block_size-1]
slot_mapping = torch.tensor([0, 1, 16, 17], dtype=torch.int32, device="npu")

# 构造key_scale及其cache（per-token-head的FP8反量化scale）
key_scale = torch.randn(num_tokens, num_head, dtype=torch.float32, device="npu")
key_scale_cache = torch.zeros(num_blocks, num_head, block_size, 1, dtype=torch.float32, device="npu")

# 调用算子，将key/value/key_scale按slot_mapping写入cache
key_cache_out, value_cache_out, key_scale_cache_out = cann_ops_transformer.ops.scatter_pa_kv_cache_with_k_scale(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    key_scale,
    key_scale_cache,
    cache_layout='BNBD',
)

torch_npu.npu.synchronize()

# 输出shape与输入cache一致
assert key_cache_out.shape == key_cache.shape
assert value_cache_out.shape == value_cache.shape
assert key_scale_cache_out.shape == key_scale_cache.shape
assert key_cache_out.dtype == kv_dtype
assert key_scale_cache_out.dtype == torch.float32
```

