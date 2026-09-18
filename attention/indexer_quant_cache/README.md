# IndexerQuantCache

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：在Indexer注意力机制的Epilog阶段对KV Cache进行原地压缩更新。算子按逐块动态量化（Per-Block Dynamic Quantization）将`float16`/`bfloat16`激活值`x`压缩为FP8（E4M3/E5M2）、HiFloat8（uint8 位存储）或MX-FP4格式，并按`slot_mapping`将量化结果与对应`cache_scale`散写（scatter）到`cache`，`slot_mapping`中值为 -1的token跳过不处理。支持MX-FP8、Normal、HiFloat8、MX-FP4四种量化模式。

- 计算公式：

  对`x`的最后一维（d轴）进行量化，记第g组为$x_g$：

  $$
  scale_g = \frac{\max(|x_g|)}{Q\_MAX}, \quad q_i = \mathrm{round}\left(\frac{x_i}{scale_g}\right)
  $$

  - `quant_mode=0`（MX-FP8）：scale存储为`float8_e8m0`，`round_scale=true`时对scale进行舍入。
  - `quant_mode=1`（Normal）：动态逐块量化，scale存储为`float32`。
  - `quant_mode=2`（HiFloat8）：输出为$x \times x\_scale$后的hifloat8。
  - `quant_mode=3`（MX-FP4）：按标准MX块（每32元素）量化为FP4（uint8每字节打包2个fp4值），scale存储为`float8_e8m0`。

## 参数说明

| 参数名       | 输入/输出/属性 | 描述                                                                          | 数据类型                                                      | 数据格式 |
| ------------ | -------------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------ | -------- |
| cache        | 输入/输出      | 当前层的KV Cache向量缓存，原地更新，即公式中的量化目标。MX-FP4模式下为uint8，每字节打包2个fp4值。 | FLOAT8_E4M3FN、FLOAT8_E5M2、UINT8、FLOAT4_E2M1、FLOAT4_E1M2   | ND       |
| cache_scale  | 输入/输出      | 每块量化的scale因子，原地更新，即公式中`scale`。                            | FLOAT、FLOAT8_E8M0                                            | ND       |
| x            | 输入           | 待量化的激活输入，即公式中`x`。shape为[bs, d]。                              | FLOAT16、BFLOAT16                                             | ND       |
| slot_mapping | 输入           | token到cache slot的索引映射，值为 -1表示跳过该token。shape为[bs]。       | INT32                                                        | ND       |
| cache        | 输出           | 原地更新后的KV Cache，与输入`cache`为同一tensor。                            | FLOAT8_E4M3FN、FLOAT8_E5M2、UINT8、FLOAT4_E2M1、FLOAT4_E1M2   | ND       |
| cache_scale  | 输出           | 原地更新后的scale，与输入`cache_scale`为同一tensor。                         | FLOAT、FLOAT8_E8M0                                            | ND       |
| quant_mode   | 可选属性       | 量化模式：0=MX-FP8，1=Normal，2=HiFloat8，3=MX-FP4。默认值为1。                | INT                                                          | -        |
| round_scale  | 可选属性       | MX-FP8/Normal模式（quant_mode=0/1）下是否对scale进行舍入。默认值为true。              | BOOL                                                         | -        |
| x_scale      | 可选属性       | HiFloat8模式（quant_mode=2）下的全局scale乘数。默认值为1.0。                 | FLOAT                                                        | -        |

## 约束说明

- `x`的最后一维`d`须能被32整除且 `d ≤ 8192`。
- `slot_mapping`的维度应等于`x`的维度减1，且`slot_mapping.shape`须与`x.shape[:-1]`一致。
- 数据类型组合需匹配量化模式：Normal模式cache为FP8/UINT8、HiFloat8模式cache为UINT8，二者cache_scale均为FLOAT；MX-FP8模式cache为FP8且cache_scale为FLOAT8_E8M0；MX-FP4模式cache为FP4且cache_scale为FLOAT8_E8M0。
- `slot_mapping`中值为 -1的token会被跳过不处理；其余有效元素取值范围为[0, num_slots - 1]，且元素值应保证不重复，重复时不保证结果正确性。

## 调用说明

| 调用方式  | 样例代码                                                                          | 说明                                                                          |
| --------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| aclnn接口 | [test_aclnn_indexer_quant_cache](./examples/test_aclnn_indexer_quant_cache.cpp)    | 通过[aclnnIndexerQuantCache](./docs/aclnnIndexerQuantCache.md)调用IndexerQuantCache算子。 |
| 图模式    | [GEIR 示例](./examples/test_geir_indexer_quant_cache.cpp)                                                                                 | 通过[算子IR](./op_graph/indexer_quant_cache_proto.h)接入GE图模式调用IndexerQuantCache算子。 |
| PyTorch API | - | 通过[indexer_quant_cache](../../attention/indexer_quant_cache/docs/torchapi_indexer_quant_cache.md)接口调用indexer_quant_cache算子。 |

## 输入与框架范围

- `cache`、`cache_scale` 必须均为四维 `[blockNum, blockSize, 1, headDim]`，前两维相同且为正。cache 行宽至少为 d，scale 行宽在 MX 模式至少为 d/32，其余模式至少为 1；HiFloat8 保留原 scale。
- `x` 的 rank 至少为 2；`slot_mapping.shape == x.shape[:-1]`，各实际维度为正，`0 < d <= 8192` 且 d 是 32 的倍数。标量、空张量、运行时仍含未知维（-1）或未知 rank（-2）的输入不支持。图构建时的动态维需在执行前解析为满足这些约束的具体 shape；不承诺动态 rank。
- 支持 ACLNN、GE IR 和 PyTorch 扩展；本目录不提供 TensorFlow 适配器或 TensorFlow 测试入口。
- ACLNN/PyTorch eager 支持 cache 在 blockNum 维的分页间隔，block 内必须连续。GE/torchair 示例仅覆盖连续张量；不承诺任意 strided view 的图模式原地更新，不能通过复制 cache 来代替原地写回。
- 有效 slot 必须唯一。设备对小于 0 或不小于 `blockNum * blockSize` 的 slot 跳过写入以避免越界；该保护不返回逐元素错误，调用者仍须提供约定范围内的索引。
- GEIR 示例接受命令行参数 `0|1|2|3`（默认 1），分别构造四种模式匹配的 dtype/shape，并使用唯一 slot 和一个 -1 跳过项。

## 特殊值与比较规则

- Normal 模式仅使用有限值求整行绝对值最大值。全零行输出零 cache 和零 scale（两种 round_scale 均如此）；全非有限值行 scale 为零。NaN/Inf 在目标格式转换前保留，最终表示由目标格式决定（E4M3 无无穷表示，E5M2 有无穷表示，UINT8 承载 HiFloat8 编码）。
- MX-FP8 的分组最大值至少取 `1e-4`，零行因此仍有非零 scale；非有限分组使用 E8M0 特殊值编码。MX-FP4 的特殊值依照对应 FP4/E8M0 转换规则，不与 Normal 的有限值筛选规则混用。
- HiFloat8 对 `x * x_scale` 做格式转换，cache_scale 原样保留。`x_scale` 在其余模式不参与计算，保留统一默认属性。
- 测试 Golden 显式配置各输出 dtype 的比较标准；打包格式按原始编码比较，并检查未命中的 cache 行及 HiFloat8 scale 保持不变。
