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

<table style="table-layout: fixed; width: 1390px">
  <colgroup>
    <col style="width: 190px">
    <col style="width: 150px">
    <col style="width: 750px">
    <col style="width: 200px">
    <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>query</td>
      <td>输入</td>
      <td>Query的非RoPE部分，每个头的维度为512。</td>
      <td>FLOAT16、BFLOAT16、INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key</td>
      <td>动态输入</td>
      <td>当前仅支持一个Key张量，头数为1，每个头的维度为512。</td>
      <td>FLOAT16、BFLOAT16、INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value</td>
      <td>动态输入</td>
      <td>当前仅支持一个Value张量，shape与key中的张量一致。</td>
      <td>FLOAT16、BFLOAT16、INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>atten_mask</td>
      <td>可选输入</td>
      <td>注意力掩码。BOOL类型中True表示屏蔽；INT8、UINT8类型取0或1，1表示屏蔽。</td>
      <td>BOOL、INT8、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seq_q</td>
      <td>可选输入</td>
      <td>Query序列实长；PyTorch接口参数名为actual_seq_lengths_q。TND布局使用不含起始0的累计长度。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>seq_kv</td>
      <td>可选输入</td>
      <td>KV序列实长；PyTorch接口参数名为actual_seq_lengths_kv。连续TND KV使用累计长度，PA KV使用逐batch绝对长度。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>block_table</td>
      <td>可选输入</td>
      <td>PA分页表，shape为(B, max_blocks)。连续KV存储时不传入。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key_antiquant_scale</td>
      <td>可选输入</td>
      <td>INT8全量化时必选，Key的反量化系数，shape为(1,)。非量化时不传入。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value_antiquant_scale</td>
      <td>可选输入</td>
      <td>INT8全量化时必选，Value的反量化系数，shape为(1,)。非量化时不传入。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>query_rope</td>
      <td>输入</td>
      <td>Query的RoPE部分，每个头的维度为64；INT8场景为BFLOAT16，其他场景与query一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key_rope</td>
      <td>输入</td>
      <td>Key的RoPE部分，头数为1，每个头的维度为64；数据类型与query_rope一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dequant_scale_query</td>
      <td>可选输入</td>
      <td>INT8全量化时必选，按Query的token和head提供反量化系数。非量化时不传入。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>metadata</td>
      <td>输入</td>
      <td>FlashMlaMetadata生成的负载均衡结果，shape为(480,)。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>attention_out</td>
      <td>输出</td>
      <td>注意力结果，每个头的维度为512；INT8输入对应BFLOAT16输出，其他场景与query一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>softmax_lse</td>
      <td>输出</td>
      <td>softmax的log-sum-exp。非TND为(B, N, S, 1)，TND为(T, N, 1)；关闭时为(0,)。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_heads</td>
      <td>属性</td>
      <td>Query头数，取值范围为[1, 128]。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale_value</td>
      <td>可选属性</td>
      <td>注意力分数的缩放系数，默认值为1.0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pre_tokens</td>
      <td>可选属性</td>
      <td>左窗口参数，默认值为2147483647；具体语义由sparse_mode决定。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>next_tokens</td>
      <td>可选属性</td>
      <td>右窗口参数，默认值为2147483647；具体语义由sparse_mode决定。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>input_layout</td>
      <td>可选属性</td>
      <td>Query输入和attention_out输出的数据排布格式，支持BSH、BSND、BNSD、TND、BSH_NBSD、BSND_NBSD、BNSD_NBSD、TND_NTD。默认值为BSH。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_key_value_heads</td>
      <td>可选属性</td>
      <td>Key和Value的头数。非量化非空输入仅支持1；INT8场景在num_heads为1时也支持0，此时有效KV头数为1。算子属性默认值为0。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_mode</td>
      <td>可选属性</td>
      <td>非量化支持0、3、4、9，INT8支持0、3、9，默认值为0。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inner_precise</td>
      <td>可选属性</td>
      <td>计算精度及行无效修正配置。非量化支持0、1、2、3，其中2和3开启行无效修正；INT8支持0、1。算子属性默认值为1。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>block_size</td>
      <td>可选属性</td>
      <td>连续KV存储时设为0；PA时为不大于1024的正16倍数，INT8时必须为128。默认值为0。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmax_lse_flag</td>
      <td>可选属性</td>
      <td>是否输出有效LSE，默认值为false。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>key_antiquant_mode</td>
      <td>可选属性</td>
      <td>Key反量化模式，当前仅支持0，默认值为0。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>value_antiquant_mode</td>
      <td>可选属性</td>
      <td>Value反量化模式，当前仅支持0，默认值为0。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>query_quant_mode</td>
      <td>可选属性</td>
      <td>非量化时为0，INT8时必须设为3，默认值为0。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

PyTorch接口中num_key_value_heads的默认值为1，inner_precise的默认值为0。接口参数及shape说明参见[flash_mla](docs/torchapi_flash_mla.md)。

## 约束说明

- 仅支持前向计算，支持单算子模式（eager），不支持反向传播和TorchAir图模式。
- query、key和value的非RoPE维度均为512，query_rope和key_rope的维度均为64，Key和Value的头数为1。
- key和value各包含一个Tensor；二者可以共享同一个Tensor，也可以分别传入。非量化时，query、key、value、query_rope和key_rope的数据类型必须一致。
- 连续KV与Query采用相同的输入布局；分页注意力（PA）的KV支持三维BSH、四维BNSD和五维NZ排布。所有输入的数据格式均为ND，具体shape参见[PyTorch接口参数说明](docs/torchapi_flash_mla.md#参数说明)。
- INT8全量化仅支持PA场景，KV采用NZ排布，block_size为128，Query头数为2的幂，Query布局支持BSH、BSND、TND及对应输出布局后缀。query_rope和key_rope为BFLOAT16，三个反量化系数均为FLOAT。
- seq_q和seq_kv必须是一维连续的设备Tensor。TND的Query及连续KV使用不含起始0的累计长度；PA的KV使用各batch的实际长度。PA和TND场景在batch非空时必须提供seq_kv。
- metadata必须由配套的FlashMlaMetadata生成并原样传入。生成metadata和执行FlashMla应使用相同的设备、stream、有效核数及调度参数；序列实长或调度参数变化后，需要重新生成metadata。
- 掩码、窗口、序列长度及量化场景的完整约束参见[PyTorch接口约束说明](docs/torchapi_flash_mla.md#约束说明)。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| PyTorch API | [test_torch_flash_mla](examples/test_torch_flash_mla.py) | 通过[flash_mla](docs/torchapi_flash_mla.md)接口调用FlashMla算子，调用前需通过[flash_mla_metadata](../flash_mla_metadata/docs/torchapi_flash_mla_metadata.md)生成metadata。 |
