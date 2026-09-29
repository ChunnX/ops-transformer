# Ascendir_FusedInferAttentionScore

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3训练系列产品/Atlas A3推理系列产品</term>|      √     |
|<term>Atlas A2训练系列产品/Atlas A2推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：适配增量（decode）&全量（prefill）推理场景的FlashAttention算子，既可以支持全量计算场景（[PromptFlashAttention](../prompt_flash_attention/README.md)），也可支持增量计算场景（[IncreFlashAttention](../incre_flash_attention/README.md)）。支持KV Cache、PageAttention、GQA/MLA、伪量化/全量化/后量化等特性。

- 计算公式：

  self-attention（自注意力）利用输入样本自身的关系构建了一种注意力模型。其原理是假设有一个长度为$n$的输入样本序列$x$，$x$的每个元素都是一个$d$维向量，可以将每个$d$维向量看作一个token embedding，将这样一条序列经过3个权重矩阵变换得到3个维度为$n*d$的矩阵。

  $$
  Attention(Q,K,V)=Score(Q,K)V
  $$

  本算子中Score函数采用Softmax函数，self-attention计算公式为：

  $$
  Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
  $$

  其中：$Q$和$K^T$的乘积代表输入$x$的注意力，为避免该值变得过大，通常除以$d$的平方根进行缩放，并对每行进行softmax归一化，与$V$相乘后得到一个$n*d$的矩阵。

## Ascend IR定义

Ascend IR定义所在头文件路径为[op_graph/fused_infer_attention_score_proto.h](./op_graph/fused_infer_attention_score_proto.h)。

```cpp
REG_OP(FusedInferAttentionScore)
    .INPUT(query, TensorType({DT_INT8, DT_FLOAT16, DT_BF16, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .DYNAMIC_INPUT(key, TensorType({DT_INT8, DT_FLOAT16, DT_BF16, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN,
                                    DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, DT_INT4}))
    .DYNAMIC_INPUT(value, TensorType({DT_INT8, DT_FLOAT16, DT_BF16, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN,
                                      DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, DT_INT4}))
    .OPTIONAL_INPUT(pse_shift, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(atten_mask, TensorType({DT_FLOAT16, DT_BOOL, DT_UINT8, DT_INT8}))
    .OPTIONAL_INPUT(actual_seq_lengths, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(actual_seq_lengths_kv, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(dequant_scale1, TensorType({DT_UINT64, DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale1, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(dequant_scale2, TensorType({DT_UINT64, DT_FLOAT}))
    .OPTIONAL_INPUT(quant_scale2, TensorType({DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(quant_offset2, TensorType({DT_FLOAT32, DT_BF16}))
    .OPTIONAL_INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(block_table, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(query_padding_size, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(kv_padding_size, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(key_antiquant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(key_antiquant_offset, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(value_antiquant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(value_antiquant_offset, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(key_shared_prefix, TensorType({DT_INT8, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(value_shared_prefix, TensorType({DT_INT8, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(actual_shared_prefix_len, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(query_rope, TensorType({DT_INT8, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(key_rope, TensorType({DT_INT8, DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(key_rope_antiquant_scale, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(dequant_scale_query, TensorType({DT_FLOAT32}))
    .OPTIONAL_INPUT(learnable_sink, TensorType({DT_BF16}))
    .OPTIONAL_INPUT(q_start_idx, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(kv_start_idx, TensorType({DT_INT64}))
    .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_INT8, DT_BF16}))
    .OUTPUT(softmax_lse, TensorType({DT_FLOAT32}))
    .REQUIRED_ATTR(num_heads, Int)
    .ATTR(scale, Float, 1.0)
    .ATTR(pre_tokens, Int, 2147483647)
    .ATTR(next_tokens, Int, 2147483647)
    .ATTR(input_layout, String, "BSH")
    .ATTR(num_key_value_heads, Int, 0)
    .ATTR(sparse_mode, Int, 0)
    .ATTR(inner_precise, Int, 1)
    .ATTR(block_size, Int, 0)
    .ATTR(antiquant_mode, Int, 0)
    .ATTR(softmax_lse_flag, Bool, false)
    .ATTR(key_antiquant_mode, Int, 0)
    .ATTR(value_antiquant_mode, Int, 0)
    .ATTR(query_quant_mode, Int, 0)
    .ATTR(pse_type, Int, 0)
    .ATTR(out_dtype, Int, 0)
    .OP_END_FACTORY_REG(FusedInferAttentionScore)
```

### 参数说明

<table style="undefined;table-layout: fixed; width: 1340px"><colgroup>
  <col style="width: 150px">
  <col style="width: 120px">
  <col style="width: 260px">
  <col style="width: 280px">
  <col style="width: 210px">
  <col style="width: 80px">
  <col style="width: 240px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>query（tensor）</td>
      <td>输入</td>
      <td>公式中的输入query。</td>
      <td>不支持空tensor。</td>
      <td>int8、float16、bf16、hifloat8、float8_e5m2、float8_e4m3fn</td>
      <td>ND</td>
      <td>见input_layout</td>
    </tr>
    <tr>
      <td>key（tensor）</td>
      <td>动态输入</td>
      <td>公式中的输入key，支持tensorlist。</td>
      <td>非连续场景下tensorlist中的batch只能为1，个数等于query的B，N和D需要相等。</td>
      <td>int8、float16、bf16、hifloat8、float8_e5m2、float8_e4m3fn、float4_e2m1、float4_e1m2、int4</td>
      <td>ND</td>
      <td>见input_layout</td>
    </tr>
    <tr>
      <td>value（tensor）</td>
      <td>动态输入</td>
      <td>公式中的输入value，支持tensorlist。</td>
      <td>shape与key的shape需要完全一致。</td>
      <td>int8、float16、bf16、hifloat8、float8_e5m2、float8_e4m3fn、float4_e2m1、float4_e1m2、int4</td>
      <td>ND</td>
      <td>见input_layout</td>
    </tr>
    <tr>
      <td>pse_shift（tensor）</td>
      <td>可选输入</td>
      <td>公式中的pse，表示位置编码。</td>
      <td>不使用该功能时可传入nullptr；MLA、D不等长、GQA全量化场景不支持。</td>
      <td>float16、bf16</td>
      <td>ND</td>
      <td>[B,Q_N,Q_S,KV_S]、[1,Q_N,Q_S,KV_S]</td>
    </tr>
    <tr>
      <td>atten_mask（tensor）</td>
      <td>可选输入</td>
      <td>公式中的atten_mask，表示注意力掩码。取值为1代表该位不参与计算（不生效），为0代表该位参与计算。</td>
      <td>不使用该功能时可传入nullptr。</td>
      <td>float16、bool、uint8、int8</td>
      <td>ND</td>
      <td>[Q_S,KV_S]、[B,Q_S,KV_S]、[B,N,Q_S,KV_S]</td>
    </tr>
    <tr>
      <td>actual_seq_lengths（tensor）</td>
      <td>可选输入</td>
      <td>不同Batch中query的有效序列长度。</td>
      <td>不指定序列长度时可传入nullptr，表示与query的shape的S长度相同。</td>
      <td>int64</td>
      <td>ND</td>
      <td>1、B</td>
    </tr>
    <tr>
      <td>actual_seq_lengths_kv（tensor）</td>
      <td>可选输入</td>
      <td>不同Batch中key/value的有效序列长度。</td>
      <td>不指定序列长度时可传入nullptr，表示与key/value的shape的S长度相同。</td>
      <td>int64</td>
      <td>ND</td>
      <td>1、B</td>
    </tr>
    <tr>
      <td>dequant_scale1（tensor）</td>
      <td>可选输入</td>
      <td>BMM1后面的反量化因子。</td>
      <td>支持per-tensor；不使用该功能时可传入nullptr。</td>
      <td>uint64、float32</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_scale1（tensor）</td>
      <td>可选输入</td>
      <td>BMM2前面的量化因子。</td>
      <td>支持per-tensor、MxFP8；不使用该功能时可传入nullptr。</td>
      <td>float32</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dequant_scale2（tensor）</td>
      <td>可选输入</td>
      <td>BMM2后面的反量化因子。</td>
      <td>支持per-tensor；不使用该功能时可传入nullptr。</td>
      <td>uint64、float32</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_scale2（tensor）</td>
      <td>可选输入</td>
      <td>输出的量化因子。</td>
      <td>支持per-tensor、per-channel；不使用该功能时可传入nullptr。</td>
      <td>float32、bf16</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quant_offset2（tensor）</td>
      <td>可选输入</td>
      <td>输出的量化偏移。</td>
      <td>shape必须与quant_scale2保持一致；不使用该功能时可传入nullptr。</td>
      <td>float32、bf16</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquant_scale（tensor）</td>
      <td>可选输入</td>
      <td>伪量化因子。</td>
      <td>预留参数，暂未使用。</td>
      <td>float16、bf16、float32</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquant_offset（tensor）</td>
      <td>可选输入</td>
      <td>伪量化偏移。</td>
      <td>预留参数，暂未使用。</td>
      <td>float16、bf16、float32</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>block_table（tensor）</td>
      <td>可选输入</td>
      <td>PageAttention中KV存储使用的block映射表。</td>
      <td>第一维长度需等于B，第二维长度不能小于maxBlockNumPerSeq。</td>
      <td>int32</td>
      <td>ND</td>
      <td>[B, KV_S_max/block_size]</td>
    </tr>
    <tr>
      <td>query_padding_size（tensor）</td>
      <td>可选输入</td>
      <td>表示query中每个batch的数据是否右对齐，且右对齐的个数是多少。</td>
      <td>仅Q_S大于1场景生效；不使用该功能时可传入nullptr。</td>
      <td>int64</td>
      <td>ND</td>
      <td>1</td>
    </tr>
    <tr>
      <td>kv_padding_size（tensor）</td>
      <td>可选输入</td>
      <td>表示key/value中每个batch的数据是否右对齐，且右对齐的个数是多少。</td>
      <td>不使用该功能时可传入nullptr。</td>
      <td>int64</td>
      <td>ND</td>
      <td>1</td>
    </tr>
    <tr>
      <td>key_antiquant_scale（tensor）</td>
      <td>可选输入</td>
      <td>KV伪量化参数分离场景下key的反量化因子。</td>
      <td>不使用该功能时可传入nullptr；支持per-tensor、per-channel、per-token、per-token-group、per-tensor叠加per-head、per-token叠加per-head等模式。</td>
      <td>float16、bf16、float32、float8_e8m0</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>key_antiquant_offset（tensor）</td>
      <td>可选输入</td>
      <td>KV伪量化参数分离场景下key的反量化偏移。</td>
      <td>使用时shape必须与key_antiquant_scale保持一致；不使用该功能时可传入nullptr。</td>
      <td>float16、bf16、float32</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>value_antiquant_scale（tensor）</td>
      <td>可选输入</td>
      <td>KV伪量化参数分离场景下value的反量化因子。</td>
      <td>不使用该功能时可传入nullptr；模式编号与key_antiquant_scale一致。</td>
      <td>float16、bf16、float32、float8_e8m0</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>value_antiquant_offset（tensor）</td>
      <td>可选输入</td>
      <td>KV伪量化参数分离场景下value的反量化偏移。</td>
      <td>使用时shape必须与value_antiquant_scale保持一致；不使用该功能时可传入nullptr。</td>
      <td>float16、bf16、float32</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>key_shared_prefix（tensor）</td>
      <td>可选输入</td>
      <td>attention结构中key的系统前缀部分的输入。</td>
      <td>不使用该功能时可传入nullptr。</td>
      <td>int8、float16、bf16、int4</td>
      <td>ND</td>
      <td>[1,prefix_S,H]、[1,prefix_S,KV_N,KV_D]、[1,KV_N,prefix_S,KV_D]</td>
    </tr>
    <tr>
      <td>value_shared_prefix（tensor）</td>
      <td>可选输入</td>
      <td>attention结构中value的系统前缀部分的输入。</td>
      <td>shape与key_shared_prefix保持一致；不使用该功能时可传入nullptr。</td>
      <td>int8、float16、bf16、int4</td>
      <td>ND</td>
      <td>[1,prefix_S,H]、[1,prefix_S,KV_N,KV_D]、[1,KV_N,prefix_S,KV_D]</td>
    </tr>
    <tr>
      <td>actual_shared_prefix_len（tensor）</td>
      <td>可选输入</td>
      <td>key_shared_prefix/value_shared_prefix的有效序列长度。</td>
      <td>不使用该功能时可传入nullptr，表示与系统前缀的S长度相同。</td>
      <td>int64</td>
      <td>ND</td>
      <td>1</td>
    </tr>
    <tr>
      <td>query_rope（tensor）</td>
      <td>可选输入</td>
      <td>MLA结构中query的rope信息。</td>
      <td>shape中D为64，其余维度与query一致。</td>
      <td>int8、float16、bf16</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>key_rope（tensor）</td>
      <td>可选输入</td>
      <td>MLA结构中key的rope信息。</td>
      <td>shape中D为64，其余维度与key一致。</td>
      <td>int8、float16、bf16</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>key_rope_antiquant_scale（tensor）</td>
      <td>可选输入</td>
      <td>MLA结构中key的rope信息的反量化因子。</td>
      <td>预留参数，当前版本不生效，传入nullptr即可。</td>
      <td>float16、bf16</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dequant_scale_query（tensor）</td>
      <td>可选输入</td>
      <td>对query进行反量化的因子。</td>
      <td>全量化场景涉及，支持per-token-group、per-token叠加per-head模式；不使用该功能时可传入nullptr。</td>
      <td>float32</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>learnable_sink（tensor）</td>
      <td>可选输入</td>
      <td>表示通过可学习的"Sink Token"起到吸收Attention Score的作用。</td>
      <td>仅支持非量化场景；不使用该功能时可传入nullptr。</td>
      <td>bf16</td>
      <td>ND</td>
      <td>[Q_N]</td>
    </tr>
    <tr>
      <td>q_start_idx（tensor）</td>
      <td>可选输入</td>
      <td>外切场景下，当前分块query的sequence在全局中的起始索引。</td>
      <td>可传入nullptr。</td>
      <td>int64</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kv_start_idx（tensor）</td>
      <td>可选输入</td>
      <td>外切场景下，当前分块key和value的sequence在全局中的起始索引。</td>
      <td>可传入nullptr。</td>
      <td>int64</td>
      <td>ND</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_heads（int）</td>
      <td>必要属性</td>
      <td>query的head个数。</td>
      <td>在BNSD、BSND、TND、NTD等场景下需与shape中query的N轴shape值相同。</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale（float）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>公式中d开根号的倒数，表示缩放系数。</li>
          <li>默认值为1.0。</li>
        </ul>
      </td>
      <td>用户不特意指定时建议传入1.0。</td>
      <td>float</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pre_tokens（int）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>用于稀疏计算，表示attention需要和前几个Token计算关联。</li>
          <li>默认值为2147483647。</li>
        </ul>
      </td>
      <td>用户不特意指定时建议传入2147483647。</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>next_tokens（int）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>用于稀疏计算，表示attention需要和后几个Token计算关联。</li>
          <li>默认值为2147483647。</li>
        </ul>
      </td>
      <td>用户不特意指定时建议传入2147483647。</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>input_layout（string）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>代表输入query、key、value的数据排布格式。</li>
          <li>默认值为"BSH"。</li>
        </ul>
      </td>
      <td>支持BSH、BSND、BNSD、BNSD_BSND、BSND_BNSD、BSH_BNSD、BNSD_NBSD、BSND_NBSD、BSH_NBSD、TND、NTD、NTD_TND、TND_NTD。排布格式带下划线时，下划线左边表示输入query的layout，下划线右边表示输出attention_out的格式。</td>
      <td>string</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>num_key_value_heads（int）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>key、value中head个数。</li>
          <li>默认值为0。</li>
        </ul>
      </td>
      <td>取值0表示key/value和query的head个数相等；需满足num_heads整除num_key_value_heads。</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_mode（int）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>表示sparse的模式。0表示defaultMask模式，1表示allMask，2表示leftUpCausal，3表示rightDownCausal，4表示band。</li>
          <li>默认值为0。</li>
        </ul>
      </td>
      <td>支持配置值为0~4，5~8暂不支持。</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inner_precise（int）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>表示高精度或者高性能选择。0表示高精度模式且不做行无效修正，1表示高性能模式且不做行无效修正，2表示高精度模式且做行无效修正，3表示高性能模式且做行无效修正。</li>
          <li>默认值为1。</li>
        </ul>
      </td>
      <td>当计算过程中"参与计算的mask部分"存在某整行全为1的情况时，建议配置为2或3开启行无效修正以提升精度，但该配置会导致性能下降。</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>block_size（int）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>PageAttention中KV存储每个block中最大的token个数。</li>
          <li>默认值为0。</li>
        </ul>
      </td>
      <td>不传时按照0处理。</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquant_mode（int）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>伪量化的方式。</li>
          <li>默认值为0。</li>
        </ul>
      </td>
      <td>预留参数，暂未使用。</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmax_lse_flag（bool）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>是否输出softmax_lse。</li>
          <li>默认值为false。</li>
        </ul>
      </td>
      <td>支持S轴外切（增加输出）。</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>key_antiquant_mode（int）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>key的反量化方式。0表示per-channel（含per-tensor），1表示per-token，2表示per-tensor叠加per-head，3表示per-token叠加per-head，4表示per-token叠加使用page attention模式管理scale/offset，5表示per-token叠加per-head并使用page attention模式管理scale/offset，6表示per-token-group。</li>
          <li>默认值为0。</li>
        </ul>
      </td>
      <td>除key_antiquant_mode为0且value_antiquant_mode为1的场景外，需与value_antiquant_mode一致。</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>value_antiquant_mode（int）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>value的反量化方式。模式编号0~6与key_antiquant_mode一致，8表示per-channel-group。</li>
          <li>默认值为0。</li>
        </ul>
      </td>
      <td>-</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>query_quant_mode（int）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>query反量化的模式，模式编号与key_antiquant_mode一致。</li>
          <li>默认值为0。</li>
        </ul>
      </td>
      <td>仅支持模式3（per-token叠加per-head）。</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pse_type（int）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>pse的方式。0表示外部传入pse，先mul再add。</li>
          <li>默认值为0。</li>
        </ul>
      </td>
      <td>仅支持配置值为0（pseType=1推理场景不支持）。</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out_dtype（int）</td>
      <td>可选属性</td>
      <td>
        <ul>
          <li>attention_out的数据类型。5表示fp16，15表示bf16，23表示fp8_e5m2，24表示fp8_e4m3，290表示hifp8。</li>
          <li>默认值为0。</li>
        </ul>
      </td>
      <td>仅在PTA图模式下支持。</td>
      <td>int</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>attention_out（tensor）</td>
      <td>输出</td>
      <td>公式中的attention_out。</td>
      <td>D维度与value的D保持一致，其余维度需要与query的shape保持一致。</td>
      <td>float16、int8、bf16</td>
      <td>ND</td>
      <td>见input_layout</td>
    </tr>
    <tr>
      <td>softmax_lse（tensor）</td>
      <td>输出</td>
      <td>ring attention算法对query乘key的结果先取max得到softmax_max，再减去softmax_max取exp求sum得到softmax_sum，最后对softmax_sum取log并加上softmax_max得到的结果。</td>
      <td>softmax_lse_flag为True时生效；数据为inf的代表无效数据。</td>
      <td>float32</td>
      <td>ND</td>
      <td>[B,N,Q_S,1]，TND相关场景下为[T,N,1]</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
- 入参为空的处理：算子内部需要判断参数query是否为空，如果是空则直接返回。参数query不为空tensor，参数key、value为空tensor（即KV_S为0），则attention_out填充为全零。attention_out为空tensor时，AscendCLNN框架会处理。
- 参数key、value中对应tensor的shape需要完全一致；非连续场景下key、value的tensorlist中的batch只能为1，个数等于query的B，N和D需要相等。由于tensorlist限制，非连续场景下B不能大于256。
- 当Q_S大于1时，query、key、value输入，功能使用限制如下：
    - 支持B轴小于等于65536。
    - 如果输入类型为INT8且D轴不是32字节对齐，则B轴的最大支持值为128。若输入类型为FLOAT16或BFLOAT16且D轴不是16字节对齐，B轴同样仅支持到128。
    - 支持N轴小于等于256，支持D轴小于等于512。input_layout为BSH或者BSND时，建议N*D小于65535。
    - S支持小于等于20971520（20M）。部分长序列场景下，如果计算量过大可能会导致算子执行超时（aicore error类型报错，errorStr为：timeout or trap error），此场景下建议做S切分处理，注：这里计算量会受B、S、N、D等的影响，值越大计算量越大。
    - D轴限制：query、key、value或attention_out类型包含INT8时，D轴需要32对齐；类型包含INT4时，D轴需要64对齐；类型全为FLOAT16、BFLOAT16时，D轴需16对齐。
- 当Q_S等于1时，query、key、value输入，功能使用限制如下：
    - 支持B轴小于等于65536，支持N轴小于等于256，支持D轴小于等于512。
    - query、key、value输入类型均为INT8的场景暂不支持。
    - 在INT4伪量化场景下，aclnn单算子调用支持KV INT4输入或者INT4拼接成INT32输入（建议通过dynamicQuant生成INT4格式的数据）。
    - 在INT4伪量化场景下，若KV INT4拼接成INT32输入，那么KV的N、D或者H是实际值的八分之一（prefix同理）。
    - key、value输入类型为INT4（INT32）时，D轴需要64对齐（INT32仅支持D 8对齐）。
- query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Head-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N，T表示所有Batch输入样本序列长度的累加和。
- FA场景约束：
    - GQA：非量化场景下，当query/key/value三组Head-Dim均小于等于128且layout不为NTD/NTD_TND/BSH_BNSD/BSND_BNSD/BNSD_BSND时，支持query/key的Head-Dim与value的Head-Dim不相等；query/key的Head-Dim与value的Head-Dim不相等时，除atten_mask参数组外，其余参数组均不支持。
    - Prefill MLA：支持query/key的Head-Dim为192，value的Head-Dim为128场景，其余场景query/key/value/attention_out的Head-Dim需保持一致；不支持全量化、伪量化。
    - Decode MLA：query/key/value/attention_out的Head-Dim需保持一致；不支持tensorlist、左padding、伪量化、prefix。
    - MLA场景下，不支持pse，不能传入pse_shift。
    - query的layout为TND/NTD时，不支持pse_shift、tensorlist。
    - PagedAttention场景下，当input_layout为BNSD、TND、BSH、BSND时，key/value排布支持BnBsH（blockNum, blockSize, H）、BnNBsD（blockNum, KV_N, blockSize, D）和NZ（blockNum，KV_N，D/16，blockSize，16）三种格式。
- pse_shift约束：
    - pse_type为0时，query的数据类型为FLOAT16或INT8时pse_shift的数据类型必须为FLOAT16，query的数据类型为BFLOAT16时pse_shift的数据类型必须为BFLOAT16。
    - pse_shift shape的维度必须为4，第2维应等于Q_N，第3维应大于等于Q_S，第4维应大于等于KV_S（prefix场景下为KV_S + actual_shared_prefix_len）。
    - PagedAttention场景下，pse_shift shape的最后一维应大于等于maxBlockNumPerBatch * block_size。
    - alibi场景下，Q_S应等于KV_S。
- atten_mask约束：
    - 输入维度仅支持2/3/4；如果输入atten_mask shape中的Q_S、KV_S非32B对齐，可以向上取到对齐的Q_S、KV_S。
    - sparse_mode为2、3、4时，需要传入优化后的atten_mask矩阵（2048*2048）。
    - sparse_mode为0、4时，须保证atten_mask与pre_tokens、next_tokens的范围一致。
- num_heads需可整除num_key_value_heads。全量化场景下，当query/key/value的类型为FLOAT8_E4M3FN且input_layout为TND时，num_heads与num_key_value_heads的比值仅支持1、2、4、6、8、12、16、24、32、48、64、96、128；其他全量化场景num_heads与num_key_value_heads的比值仅支持1、2、4、8、16、32、64、128。
- input_layout=BSH_BNSD、BSND_BNSD、NTD、NTD_TND仅支持Q_D=K_D=V_D都等于64或128，或Q_D=K_D等于192，V_D等于128；input_layout=BNSD_BSND仅支持Q_D=K_D=V_D都16对齐（attention_out数据类型为INT8时为32对齐），或Q_D=K_D等于192，V_D等于128。
- GQA全量化场景下（query/key/value类型为FLOAT8_E4M3FN），Head-Dim仅支持128，不支持pse。
- MxFP8场景下，Head-Dim仅支持64或128。
- <term>Ascend 950PR/Ascend 950DT</term>：
    - antiquant_scale/antiquant_offset（统一伪量化）不支持，需使用key/value分离的伪量化参数。
- <term>Atlas A3训练系列产品/Atlas A3推理系列产品</term>、<term>Atlas A2训练系列产品/Atlas A2推理系列产品</term>：
    - Prefill MLA场景下，不支持tensorlist、左padding。
    - MLA场景下，不支持后量化。

## 调用示例

- aclnn调用（<term>Ascend 950PR/Ascend 950DT</term>）：[test_aclnn_fused_infer_attention_score_v5](./examples/arch35/test_aclnn_fused_infer_attention_score_v5.cpp)，通过[aclnnFusedInferAttentionScoreV5](./docs/aclnnFusedInferAttentionScoreV5.md)接口调用。
- 图模式调用：参考[算子调用](../../docs/zh/invocation/quick_op_invocation.md#ge图模式)的GE图模式说明，通过torchair（`torch.compile` + `torchair.get_npu_backend`）构图调用本算子的完整用例见[gqa_no_quant_bnsd_bsnd_ge](./tests/pytest/gqa_no_quant_bnsd_bsnd_ge.py)。

> **说明**：当前版本中，本算子在<term>Ascend 950PR/Ascend 950DT</term>上的算子二进制采用`folded_with_desc`动态参数模式，GE的C++接口（`ge::Session` + `ge::Operator`构图）在该模式下对未绑定的可选输入下发参数校验存在限制，图模式调用请优先使用torchair方式；该限制不影响aclnn调用方式。
