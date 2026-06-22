# npu\_indexer\_quant\_cache

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

- API功能：在Indexer注意力机制的Epilog阶段对KV Cache进行原地动态量化压缩更新，封装aclnnIndexerQuantCache。将float16/bfloat16激活值x按量化粒度（MX-FP8/MX-FP4模式每32个元素一组，Normal/HiFloat8模式整行一组）动态量化为FP8（E4M3/E5M2）、INT8（uint8）或MX-FP4，并按slot_mapping将量化结果与对应scale散写到cache/cache_scale，slot_mapping中值为-1的token跳过不处理。支持MX-FP8、Normal、HiFloat8、MX-FP4四种量化模式。
- 语义说明：底层aclnn接口对cacheRef、cacheScaleRef为原地实现；本Python接口为函数式语义，对输入cache、cache_scale做拷贝后在副本上更新并返回，未被slot_mapping命中的行保留原值。调用方需使用返回的cache与cache_scale。

## 函数原型

```python
npu_indexer_quant_cache(Tensor cache, Tensor cache_scale, Tensor x, Tensor slot_mapping, *, int quant_mode=1, bool round_scale=True, float x_scale=1.0) -> (Tensor, Tensor)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 | 数据类型 | 数据格式 |
|--------|-----------|------|----------|----------|
| cache | 输入 | 当前层的分页KV Cache，**仅支持四维shape** `[blockNum, blockSize, 1, headDim]`（num_slots = blockNum × blockSize）。算子按slot_mapping把每个token的量化结果写入对应行前部，**headDim须 ≥ 每token量化-x长度（=d，MX-FP4以fp4元素计）**，行内余下保持原值。在**blockNum维支持非连续**（分页）。MX-FP4模式下cache为FP4（每字节打包2个fp4值）。 | FLOAT8_E4M3FN、FLOAT8_E5M2、UINT8、FLOAT4_E2M1、FLOAT4_E1M2 | $ND$ |
| cache_scale | 输入 | 每块量化的scale因子，**仅支持四维shape** `[blockNum, blockSize, 1, scaleHeadDim]`，**scaleHeadDim须 ≥ scaleCol**（MX-FP8/MX-FP4=⌈d/32⌉，Normal/HiFloat8=1）。**所有量化模式（含HiFloat8/quant_mode=2）均校验，无例外**；HiFloat8模式scaleCol=1，故scaleHeadDim须 ≥ 1（kernel仍按scaleCol=1为每token写出一个float scale，必须提供合法cache_scale）。 | FLOAT、FLOAT8_E8M0 | $ND$ |
| x | 输入 | 待量化的激活输入，shape为[bs, d]，d需满足d % 32 == 0。 | FLOAT16、BFLOAT16 | $ND$ |
| slot_mapping | 输入 | token到cache slot的索引映射，shape为[bs]，元素取值范围[-1, num_slots - 1]，-1表示跳过该token；维度需等于x的维度减1。 | INT32 | $ND$ |
| quant_mode | 可选属性 | 量化模式：0=MX-FP8、1=Normal、2=HiFloat8、3=MX-FP4。默认值1。 | INT | - |
| round_scale | 可选属性 | MX-FP8模式（quant_mode=0）下是否对scale进行舍入。默认值True。 | BOOL | - |
| x_scale | 可选属性 | HiFloat8模式（quant_mode=2）下的全局scale乘数。默认值1.0。 | FLOAT | - |

## 输出说明

| 输出 | 说明 | 数据类型 | 数据格式 |
|------|------|----------|----------|
| cache | 更新后的KV Cache，shape与dtype与输入cache相同。 | 与输入cache一致 | $ND$ |
| cache_scale | 更新后的scale因子，shape与dtype与输入cache_scale相同。 | 与输入cache_scale一致 | $ND$ |

## 约束说明

- 仅支持<term>Ascend 950PR/Ascend 950DT</term>。
- cache与cache_scale的数据类型组合需与quant_mode匹配：Normal/HiFloat8模式cache_scale为FLOAT；MX-FP8/MX-FP4模式cache_scale为FLOAT8_E8M0；MX-FP4模式cache为FP4（uint8打包）。
- **cache 与 cache_scale（所有量化模式，含HiFloat8）均仅支持四维shape** `[blockNum, blockSize, 1, headDim]`，倒数第二维固定为1（每token一个量化向量）；num_slots = blockNum × blockSize。
- cache/cache_scale**仅在blockNum维支持非连续**（分页）：各block可不紧密排布，但block内须连续。
- **headDim长度约束**（算子每行只写固定长度，余下保持原值）：
  - **cache.headDim ≥ d**（MX-FP4以fp4元素计，d个fp4值占⌈d/2⌉字节）；
  - **cache_scale.headDim ≥ scaleCol**，scaleCol = MX-FP8/MX-FP4：⌈d/32⌉；Normal/HiFloat8：1；
  - 示例：d=128、quant_mode=0 → scaleCol=4 → cache.headDim ≥ 128 且 cache_scale.headDim ≥ 4。
- **cache_scale在所有量化模式（含HiFloat8/quant_mode=2）下均完整校验，无例外**：HiFloat8模式scaleCol=1，cache_scale须为四维 `[blockNum, blockSize, 1, headDim]`、dim2==1 且 headDim ≥ 1。（说明：kernel在HiFloat8下仍按scaleCol=1为每token散写一个float scale，故必须提供合法cache_scale，否则越界写。）
- slot_mapping的维度应等于x的维度减1，即slot_mapping为x除最后一维外的所有维度展平。
- x的最后一维（d轴）按量化粒度分组：MX-FP8（quant_mode=0）/ MX-FP4（quant_mode=3）按每32个元素一组（标准MX块）量化出一个float8_e8m0 scale；Normal（quant_mode=1）/ HiFloat8（quant_mode=2）整行一组量化出一个float32 scale。
- slot_mapping中值为-1的token会被跳过不处理；其余有效元素取值范围为[0, num_slots - 1]，且元素值应保证不重复，重复时不保证结果正确性。
- 支持图模式（torchair）调用：经graph_convert注册的GE converter下沉为IndexerQuantCache算子，eager与图模式结果一致。

## 调用示例

### 单算子（eager）调用

```python
import torch
import torch_npu
import npu_ops_transformer

# cache/cache_scale 为四维 [blockNum, blockSize, 1, headDim]，num_slots = blockNum*blockSize
block_num, block_size, d = 128, 16, 128   # num_slots = 2048; mode1 headDim=d=128 >= d
bs = 1024

cache = torch.zeros(block_num, block_size, 1, d, dtype=torch.float8_e4m3fn).npu()
cache_scale = torch.zeros(block_num, block_size, 1, 1, dtype=torch.float32).npu()  # mode1 scaleCol=1
x = torch.randn(bs, d, dtype=torch.float16).npu()
slot_mapping = torch.randint(0, block_num * block_size, (bs,), dtype=torch.int32).npu()

cache, cache_scale = npu_ops_transformer.ops.npu_indexer_quant_cache(
    cache, cache_scale, x, slot_mapping, quant_mode=1, round_scale=True, x_scale=1.0)
print(cache.shape, cache.dtype, cache_scale.shape, cache_scale.dtype)
```

### 图模式（torchair）调用

复用上面单算子示例构造的 cache/cache_scale/x/slot_mapping，用torchair后端编译后调用即可：

```python
import torchair

def fn(cache, cache_scale, x, slot_mapping):
    return torch.ops.npu_ops_transformer.npu_indexer_quant_cache(
        cache, cache_scale, x, slot_mapping, quant_mode=1, round_scale=True, x_scale=1.0)

npu_backend = torchair.get_npu_backend(compiler_config=torchair.CompilerConfig())
compiled = torch.compile(fn, backend=npu_backend, dynamic=False)
cache, cache_scale = compiled(cache, cache_scale, x, slot_mapping)
print(cache.shape, cache.dtype, cache_scale.shape, cache_scale.dtype)
```