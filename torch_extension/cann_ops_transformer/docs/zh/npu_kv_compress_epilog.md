# npu\_kv\_compress\_epilog

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- API功能：在KV Cache的Epilog阶段对cache进行原地逐组动态量化压缩更新，封装aclnnKvCompressEpilog。将bfloat16激活值x按尾轴每64个元素一组动态量化为FP8（以uint8打包），并按slot_mapping将量化结果散写到cache，slot_mapping中值为-1的token跳过不处理。支持group(bf16 scale)、group(e8m0 scale)两种量化模式。
- 语义说明：底层aclnn接口对cacheRef为原地实现；本Python接口为函数式语义，对输入cache做拷贝后在副本上更新并返回，未被slot_mapping命中的行保留原值。调用方需使用返回的cache。

## 函数原型

```python
npu_kv_compress_epilog(Tensor cache, Tensor x, Tensor slot_mapping, *, int quant_group_size=64, int quant_mode=1, bool round_scale=True, float x_scale=1.0) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明 | 数据类型 | 数据格式 |
|--------|-----------|------|----------|----------|
| cache | 输入 | 当前层的KV Cache向量缓存，shape为[num_slots, kvCacheCol]，每行包含rope/fp8/scale/pad各区段。 | UINT8 | $ND$ |
| x | 输入 | 待量化的激活输入，shape为[bs, d]，d需满足d % 64 == 0且d > 64。 | BFLOAT16 | $ND$ |
| slot_mapping | 输入 | token到cache slot的索引映射，shape为[bs]，元素取值范围[-1, num_slots - 1]，-1表示跳过该token；维度需等于x的维度减1。 | INT32、INT64 | $ND$ |
| quant_group_size | 可选属性 | 量化分组大小，与内核实现一致固定为64。默认值64。 | INT | - |
| quant_mode | 可选属性 | 量化模式：0=group(bf16 scale)、1=group(e8m0 scale)。默认值1。 | INT | - |
| round_scale | 可选属性 | group模式下是否对每组scale向上取到2的幂。默认值True。 | BOOL | - |
| x_scale | 可选属性 | 预留参数，当前未使用。默认值1.0。 | FLOAT | - |

## 输出说明

| 输出 | 说明 | 数据类型 | 数据格式 |
|------|------|----------|----------|
| cache | 更新后的KV Cache，shape与dtype与输入cache相同。 | UINT8 | $ND$ |

## 约束说明

- 仅支持<term>Ascend 950PR/Ascend 950DT</term>。
- slot_mapping的维度应等于x的维度减1，即slot_mapping为x除最后一维外的所有维度展平。
- x的最后一维（d轴）需满足d % 64 == 0且d > 64，按每64个元素一组进行逐组量化。
- slot_mapping中值为-1的token会被跳过不处理；其余有效元素取值范围为[0, num_slots - 1]，且元素值应保证不重复，重复时不保证结果正确性。
- 图模式调用（暂不支持）。

## 调用示例

```python
import torch
import torch_npu
import npu_ops_transformer

num_slots, kv_cache_col = 2048, 384
bs, d = 1024, 256

cache = torch.zeros(num_slots, kv_cache_col, dtype=torch.uint8).npu()
x = torch.randn(bs, d, dtype=torch.bfloat16).npu()
slot_mapping = torch.randint(0, num_slots, (bs,), dtype=torch.int32).npu()

cache = npu_ops_transformer.ops.npu_kv_compress_epilog(
    cache, x, slot_mapping, quant_group_size=64, quant_mode=1, round_scale=True, x_scale=1.0)
print(cache.shape, cache.dtype)
```