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

- API功能：在KV Cache的Epilog阶段对cache进行原地量化压缩更新，封装aclnnKvCompressEpilog。将bfloat16激活值x量化压缩后，按slot_mapping散写到cache，slot_mapping中值为-1的token跳过不处理。x尾轴d的后64列为rope段、前d-64列为nope段。支持三种量化模式：
  - quant_mode=0/1：逐组动态量化压缩为FP8（uint8打包），rope段保留bfloat16；
  - quant_mode=2：rope段hifloat8**静态**量化（乘x_scale后取hifloat8），nope段per-group **FLOAT4_E2M1 动态**量化（scale=组内amax/6以bfloat16写出）。
- 语义说明：底层aclnn接口对cacheRef为原地实现；本Python接口为函数式语义，对输入cache做拷贝后在副本上更新并返回，未被slot_mapping命中的行保留原值。调用方需使用返回的cache。

## 函数原型

```python
npu_kv_compress_epilog(Tensor cache, Tensor x, Tensor slot_mapping, *, int quant_group_size=64, int quant_mode=1, bool round_scale=True, float x_scale=1.0) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明 | 数据类型 | 数据格式 |
|--------|-----------|------|----------|----------|
| cache | 输入 | 当前层的分页KV Cache，**仅支持四维shape** `[blockNum, blockSize, 1, headDim]`（num_slots = blockNum × blockSize），**headDim须 ≥ kvCacheCol**（kvCacheCol计算见约束说明）。在**blockNum维支持非连续**（分页）。 | UINT8 | $ND$ |
| x | 输入 | 待量化的激活输入，shape为[bs, d]，d需满足d % 64 == 0且d > 64。 | BFLOAT16 | $ND$ |
| slot_mapping | 输入 | token到cache slot的索引映射，shape为[bs]，元素取值范围[-1, num_slots - 1]（num_slots = blockNum × blockSize），-1表示跳过该token；维度需等于x的维度减1。 | INT32、INT64 | $ND$ |
| quant_group_size | 可选属性 | 量化分组大小，默认值64。quant_mode=0/1时内核固定按64处理；quant_mode=2时仅支持16/32/64，且要求(d-64) % quant_group_size == 0。 | INT | - |
| quant_mode | 可选属性 | 量化模式：0=group(bf16 scale)、1=group(e8m0 scale)、2=rope hifloat8静态 + nope FLOAT4_E2M1动态。默认值1。 | INT | - |
| round_scale | 可选属性 | group模式下是否对每组scale向上取到2的幂。默认值True。quant_mode=2下不生效。 | BOOL | - |
| x_scale | 可选属性 | quant_mode=2时为rope段hifloat8静态量化的缩放系数；quant_mode=0/1下预留未使用。默认值1.0。 | FLOAT | - |

## 输出说明

| 输出 | 说明 | 数据类型 | 数据格式 |
|------|------|----------|----------|
| cache | 更新后的KV Cache，shape与dtype与输入cache相同。 | UINT8 | $ND$ |

## 约束说明

- 仅支持<term>Ascend 950PR/Ascend 950DT</term>。
- **cache仅支持四维shape** `[blockNum, blockSize, 1, headDim]`，倒数第二维固定为1（每token一个压缩向量），不支持其他维数。
- cache**仅在blockNum维支持非连续**：各block可不紧密排布，但block内（blockSize、headDim维）须连续。
- **headDim约束**：cache末维headDim必须 ≥ 每行写出字节数kvCacheCol = 对齐(concatCol)，concatCol按量化模式计算：
  - quant_mode=0/1：concatCol = (d−64) + 128 + ⌈(d−64)/64⌉ × scaleBytes，scaleBytes为mode0=2、mode1=1；
  - quant_mode=2：concatCol = 64 + (d−64)/2 + (d−64)/quant_group_size × 2；
  - 对齐规则：quant_mode=1不补齐，kvCacheCol = concatCol；quant_mode=0/2按32B对齐，kvCacheCol = ⌈concatCol / 32⌉ × 32。
  - 示例：d=512、quant_mode=1 → kvCacheCol=583 → headDim ≥ 583（示例中取640）。
- slot_mapping的维度应等于x的维度减1，即slot_mapping为x除最后一维外的所有维度展平。
- x的最后一维（d轴）需满足d % 64 == 0且d > 64，按每64个元素一组进行逐组量化。
- quant_mode=2时，quant_group_size仅支持16/32/64，且nope段长度(d-64)需能被quant_group_size整除。
- slot_mapping中值为-1的token会被跳过不处理；其余有效元素取值范围为[0, num_slots - 1]，且元素值应保证不重复，重复时不保证结果正确性。
- 支持图模式（torchair）调用：经graph_convert注册的GE converter下沉为KvCompressEpilog算子，eager与图模式结果一致。

## 调用示例

### 单算子（eager）调用

```python
import torch
import torch_npu
import npu_ops_transformer

# cache 为四维 [blockNum, blockSize, 1, headDim]，num_slots = blockNum*blockSize
# headDim 须 >= kvCacheCol（d=256/quant_mode=1 时 kvCacheCol=323，取 head_dim=384 满足）
block_num, block_size, head_dim = 128, 16, 384
bs, d = 1024, 256

cache = torch.zeros(block_num, block_size, 1, head_dim, dtype=torch.uint8).npu()
x = torch.randn(bs, d, dtype=torch.bfloat16).npu()
slot_mapping = torch.randint(0, block_num * block_size, (bs,), dtype=torch.int32).npu()

cache = npu_ops_transformer.ops.npu_kv_compress_epilog(
  cache, x, slot_mapping, quant_group_size=64, quant_mode=1, round_scale=True, x_scale=1.0)
print(cache.shape, cache.dtype)
```

### 图模式（torchair）调用

复用上面单算子示例构造的 cache/x/slot_mapping，用torchair后端编译后调用即可：

```python
import torchair

def fn(cache, x, slot_mapping):
  return torch.ops.npu_ops_transformer.npu_kv_compress_epilog(
    cache, x, slot_mapping, quant_group_size=64, quant_mode=1, round_scale=True, x_scale=1.0)

npu_backend = torchair.get_npu_backend(compiler_config=torchair.CompilerConfig())
compiled = torch.compile(fn, backend=npu_backend, dynamic=False)
cache = compiled(cache, x, slot_mapping)
print(cache.shape, cache.dtype)
```