# FlashMla

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

- 算子功能：完成旋转位置编码（Rotary Position Embedding，RoPE）分离的多头潜在注意力（Multi-head Latent Attention，MLA）前向计算。支持连续KV存储、分页注意力（Paged Attention，PA）以及INT8全量化场景。
- 调用流程：先通过[FlashMlaMetadata](../flash_mla_metadata/README.md)读取设备上的序列实长并生成调度元数据，再由FlashMla执行注意力计算。序列实长更新后需要重新生成元数据。
- 计算公式：非量化场景下，对每个batch、每个Query头计算：

  $$
  S=\mathrm{scale\_value}\left(QK^{T}+Q_{\mathrm{rope}}K_{\mathrm{rope}}^{T}\right)+M
  $$

  $$
  O=\operatorname{softmax}(S)V,\qquad
  \mathrm{LSE}=\log\sum_j\exp(S_{ij})
  $$

  其中，$Q$、$K$、$V$对应`query`、`key[0]`、`value[0]`，$Q_{\mathrm{rope}}$、$K_{\mathrm{rope}}$对应已经完成位置编码的`query_rope`、`key_rope`。$M$表示掩码与窗口共同确定的屏蔽关系，保留位置为0，屏蔽位置在数学上为负无穷。$O$为`attention_out`；开启`softmax_lse_flag`时输出有效的LSE。KV头数为1，供所有Query头共享；`key[0]`和`value[0]`可以使用同一张量，也可以分别传入。

## 参数说明

下表按[算子定义](op_host/flash_mla_def.cpp)中的输入、输出和属性顺序列出参数。shape的完整说明见[torch接口文档](docs/torchapi_flash_mla.md)。

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| query | 输入 | Query的非RoPE部分，每个头的维度为512。 | FLOAT16、BFLOAT16、INT8 | ND |
| key | 动态输入 | 当前仅支持一个Key张量，头数为1，每个头的维度为512。 | 与query一致 | ND |
| value | 动态输入 | 当前仅支持一个Value张量，shape与key中的张量一致。 | 与query一致 | ND |
| atten_mask | 可选输入 | 注意力掩码。BOOL类型中True表示屏蔽；INT8、UINT8类型取0或1，1表示屏蔽。 | BOOL、INT8、UINT8 | ND |
| seq_q | 可选输入 | Query序列实长；torch参数名为actual_seq_lengths_q。TND布局使用不含起始0的累计长度。 | INT64 | ND |
| seq_kv | 可选输入 | KV序列实长；torch参数名为actual_seq_lengths_kv。连续TND KV使用累计长度，PA KV使用逐batch绝对长度。 | INT64 | ND |
| block_table | 可选输入 | PA分页表，shape为(B, max_blocks)。连续KV存储时不传入。 | INT32 | ND |
| key_antiquant_scale | 可选输入 | INT8全量化时必选，Key的反量化系数，shape为(1,)。非量化时不传入。 | FLOAT | ND |
| value_antiquant_scale | 可选输入 | INT8全量化时必选，Value的反量化系数，shape为(1,)。非量化时不传入。 | FLOAT | ND |
| query_rope | 输入 | Query的RoPE部分，每个头的维度为64；INT8场景为BFLOAT16，其他场景与query一致。 | FLOAT16、BFLOAT16 | ND |
| key_rope | 输入 | Key的RoPE部分，头数为1，每个头的维度为64；数据类型与query_rope一致。 | FLOAT16、BFLOAT16 | ND |
| dequant_scale_query | 可选输入 | INT8全量化时必选，按Query的token和head提供反量化系数。非量化时不传入。 | FLOAT | ND |
| metadata | 输入 | FlashMlaMetadata生成的调度元数据，shape固定为(480,)，占1920字节。 | INT32 | ND |
| attention_out | 输出 | 注意力结果，每个头的维度为512；INT8输入对应BFLOAT16输出，其他场景与query一致。 | FLOAT16、BFLOAT16 | ND |
| softmax_lse | 输出 | softmax的log-sum-exp。非TND为(B, N, S, 1)，TND为(T, N, 1)；关闭时为(0,)。 | FLOAT | ND |
| num_heads | 属性 | Query头数，取值范围为[1, 128]。 | INT | - |
| scale_value | 可选属性 | 注意力分数的缩放系数，默认值为1.0。 | FLOAT | - |
| pre_tokens | 可选属性 | 左窗口参数，默认值为2147483647；具体语义由sparse_mode决定。 | INT | - |
| next_tokens | 可选属性 | 右窗口参数，默认值为2147483647；具体语义由sparse_mode决定。 | INT | - |
| input_layout | 可选属性 | Query输入布局及可选的输出布局后缀，默认值为BSH。 | STRING | - |
| num_key_value_heads | 可选属性 | 算子定义默认值为0。非量化非空输入必须显式设为1；INT8仅在num_heads为1时兼容0。 | INT | - |
| sparse_mode | 可选属性 | 非量化支持0、3、4、9，INT8支持0、3、9，默认值为0。 | INT | - |
| inner_precise | 可选属性 | 计算精度及行无效修正配置。算子定义默认值为1；非量化支持0、1、2、3，INT8支持0、1。 | INT | - |
| block_size | 可选属性 | 连续KV存储时设为0；PA时为不大于1024的正16倍数，INT8时必须为128。默认值为0。 | INT | - |
| softmax_lse_flag | 可选属性 | 是否输出有效LSE，默认值为false。 | BOOL | - |
| key_antiquant_mode | 可选属性 | Key反量化模式，当前仅支持0，默认值为0。 | INT | - |
| value_antiquant_mode | 可选属性 | Value反量化模式，当前仅支持0，默认值为0。 | INT | - |
| query_quant_mode | 可选属性 | 非量化时为0，INT8时必须设为3，默认值为0。 | INT | - |

PyTorch接口的默认值为`num_key_value_heads=1`、`inner_precise=0`，与算子定义的默认值不同；使用torch接口时以其函数原型为准。

## 约束说明

- 仅提供前向计算，当前torch接口支持单算子模式（eager），暂不支持反向传播和torchair图模式。
- Query布局支持BSH、BSND、BNSD、TND，以及BSH_NBSD、BSND_NBSD、BNSD_NBSD、TND_NTD。后缀仅指定attention_out的布局，LSE布局不随输出后缀变化。
- 连续KV与Query采用相同输入布局；PA KV支持三维BSH、四维BNSD或五维NZ排布。NZ需使用按规定维度组织的ND张量，具体shape见torch接口文档。
- INT8全量化必须使用PA和NZ KV，Query头数为2的幂，Query布局不支持BNSD及BNSD_NBSD。必须同时传入三个FLOAT反量化系数张量。
- 序列实长必须为一维连续NPU INT64张量；TND下两个实长张量的项数均等于逻辑batch数，PA非空batch必须提供KV实长。
- metadata必须连续、存储偏移为0且首地址按64字节对齐。应直接使用配套接口返回的张量，并保持到FlashMla执行完成；禁止修改其内容或用其他metadata接口的输出替代。
- 两次调用必须使用同一设备、同一stream及相同的有效核数。头数、存储长度、输入布局、分页配置、稀疏配置、量化模式和序列描述必须匹配；序列值变化后必须重新生成metadata。
- 掩码shape、INT8序列长度上限及非连续张量的支持情况见[torch接口约束说明](docs/torchapi_flash_mla.md#约束说明)。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| PyTorch API | [test_torch_flash_mla.py](examples/test_torch_flash_mla.py) | 先调用[flash_mla_metadata](../flash_mla_metadata/docs/torchapi_flash_mla_metadata.md)，再通过[flash_mla](docs/torchapi_flash_mla.md)完成计算。样例与CPU参考结果比较attention_out和LSE。 |

[Torch ST](tests/st/flash_mla/)提供执行器`executor_flash_mla.py`和固定精度用例`atk_flash_mla.json`，共600条，覆盖FP16、BF16、INT8各200条。每个用例通过PyTorch接口依次调用`flash_mla_metadata`和`flash_mla`，与CPU参考结果比较。

安装配套算子包、Torch扩展和ATK后，在用例目录运行以下命令。`--devices`指定测试使用的NPU设备。

```bash
cd attention/flash_mla/tests/st/flash_mla
atk node --backend npu --devices 0 node --backend cpu task \
    -c atk_flash_mla.json -p executor_flash_mla.py --task accuracy -bd cpu
```
