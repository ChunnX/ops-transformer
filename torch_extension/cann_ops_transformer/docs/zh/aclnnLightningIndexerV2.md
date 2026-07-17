# aclnnLightningIndexerV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：LightningIndexerV2基于一系列操作得到每一个token对应的top-k个位置。

- 计算公式：

  $$
  Top-k \left\{  \left[ 1 \left] \mathop{{}}\nolimits_{{1 \times \text{ }g}}\text{@} \left[  \left( W\text{@} \left[ 1 \left] \mathop{{}}\nolimits_{{1\text{ } \times \text{ }S\mathop{{}}\nolimits_{{k}}}} \left) \text{ } \odot \text{ }ReLU \left( Q\mathop{{}}\nolimits_{{index}}\text{@}K\mathop{{}}\nolimits_{{T}}^{{index}} \left)  \left]  \right\} \right. \right. \right. \right. \right. \right. \right. \right. \right. \right.
  $$

- 主要计算过程为：

  1. 将某个token对应的输入参数`q`（$Q_{index}\in\R^{g\times d}$）乘以给定上下文`k`（$K_{index}\in\R^{S_{k}\times d}$），得到相关性。
  2. 通过激活函数$ReLU$过滤无效负相关信号后，得到当前Token与所有前序Token的相关性分数向量。
  3. 将其与权重系数`w`（$W$）相乘后，沿g的方向，选取前$Top-k$个索引值得到输出$sparseIndices$，并输出对应的$sparseValues$，作为Attention的输入。

## 函数原型

```python
cann_ops_transformer.lightning_indexer_v2(
    q,
    k,
    w,
    cu_seqlens_q=None,
    cu_swqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    block_talbe=None,
    output_idx_offset=None,
    metadata=None,
    topk=2048,
    max_seqlen_q=-1,
    layout_q="BSND",
    layout_k="BSND",
    mask_mode=0,
    cmp_ratio=1,
    return_value=0,
) -> (Tensor, Tensor)
```


## 参数说明<a name="zh-cn_topic_0000002168254826_section112637109429"></a>

### lightning_indexer_v2

> [!NOTE]
> - query、key、weights参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、N（Head Num）表示多头数、D（Head Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
> - S1表示query shape中的S，S2表示key shape中的S，T1表示query shape中的T，N1表示query shape中的N，N2表示key shape中的N。

  <table style="undefined;table-layout: fixed; width: 1601px"><colgroup>
  <col style="width: 264px">
  <col style="width: 132px">
  <col style="width: 232px">
  <col style="width: 330px">
  <col style="width: 164px">
  <col style="width: 119px">
  <col style="width: 215px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>q</td>
      <td>输入</td>
      <td>公式中的$Q$。</td>
      <td>不支持空tensor。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_query为BSND时，shape为(B,S1,N1,D)。</li>
                <li>layout_query为TND时，shape为(T1,N1,D)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>k</td>
      <td>输入</td>
      <td>公式中的$K$。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>block_num为PageAttention时block总数，block_size为一个block的token数。</li>
                <li>layout_key为PA_BSND时，shape为(block_num, block_size, N2, D)。</li>
                <li>Ascend 950PR/Ascend 950DT额外支持layout_key为BSND（shape为(B, K_S, N2, D)）和TND（shape为(K_T, N2, D)）。</li>
          </ul>
      </td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_key为PA_BSND时，shape为(block_num, block_size, N2, D)。</li>
          </ul>
      </td>
      <td>支持0轴非连续</td>
    </tr>
    <tr>
      <td>w</td>
      <td>输入</td>
      <td>公式中的$W$。</td>
      <td>不支持空tensor。</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_query为BSND时，shape为(B,S1,N1)。</li>
                <li>layout_query为TND时，shape为(T1,N1)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>cu_seqlens_q</td>
      <td>输入</td>
      <td>每个Batch中，Query的有效token数（TND场景使用cu_seqlens格式）。</td>
      <td>
          <ul>
                <li>如果不指定可传入None。</li>
                <li>当layout_query为TND时，该入参必须传入，且以该入参元素的数量作为B值，该入参中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B+1,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>cu_seqlens_k</td>
      <td>输入</td>
      <td>每个Batch中，Key的有效token数（TND场景使用cu_seqlens格式）。</td>
      <td>
          <ul>
                <li>如果不指定可传入None。</li>
                <li>当layout_key为TND时，该入参必须传入。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B+1,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>seqused_q</td>
      <td>输入</td>
      <td>每个Batch中，Query的有效token数（BSND场景使用seqused格式）。</td>
      <td>
          <ul>
                <li>如果不指定可传入None，表示和query的shape的S长度相同。</li>
                <li>该入参中每个Batch的有效token数不超过query中的维度S大小且不小于0。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>seqused_k</td>
      <td>输入</td>
      <td>每个Batch中，Key的有效token数（BSND场景使用seqused格式）。</td>
      <td>
          <ul>
                <li>如果不指定可传入None，表示和key的shape的S长度相同。</li>
                <li>该入参中每个Batch的有效token数不超过key中的维度S大小且不小于0。</li>
                <li>当layout_key为PA_BSND时，该入参必须传入。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>cmp_residual_k</td>
      <td>输入</td>
      <td>压缩场景下Key的残余长度。</td>
      <td>如果不指定可传入None。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>block_table</td>
      <td>输入</td>
      <td>表示PageAttention中K存储使用的block映射表。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>PageAttention场景下，block_table必须为二维，第一维长度需要等于B，第二维长度不能小于maxBlockNumPerSeq。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B, S2/block_size)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>output_idx_offset</td>
      <td>输入</td>
      <td>输出索引的偏移量。</td>
      <td>如果不指定可传入None。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>-</td>
      <td>x</td>
    </tr>
    <tr>
      <td>metadata</td>
      <td>输入</td>
      <td>QuantLightningIndexerV2Metadata算子传入的分核信息。</td>
      <td>
          <ul>
                <li>包含使用核数、分块大小以及每个核处理数据的起始点等内容。</li>
                <li>shape大小为[1024]，当前不支持传空。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(1024,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>topk</td>
      <td>输入</td>
      <td>topK阶段需要保留的block数量。</td>
      <td>支持[1, 2048]，默认值2048。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_seqlen_q</td>
      <td>输入</td>
      <td>Query的最大序列长度。</td>
      <td>默认值-1，表示任意可能长度。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_q</td>
      <td>输入</td>
      <td>用于标识输入Query的数据排布格式。</td>
      <td>
          <ul>
                <li>用户不特意指定时可传入默认值"BSND"。</li>
                <li>当前支持BSND、TND。</li>
          </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_k</td>
      <td>输入</td>
      <td>用于标识输入Key的数据排布格式。</td>
      <td>
          <ul>
                <li>用户不特意指定时可传入默认值"PA_BSND"。</li>
                <li>Atlas A3 训练系列产品/Atlas A3 推理系列产品、Atlas A2 训练系列产品/Atlas A2 推理系列产品仅支持PA_BSND。</li>
                <li>Ascend 950PR/Ascend 950DT支持PA_BSND、BSND、TND。</li>
          </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mask_mode</td>
      <td>输入</td>
      <td>表示sparse的模式。</td>
      <td>
          <ul>
                <li>0代表defaultMask模式。</li>
                <li>3代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。</li>
                <li>默认值0。</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_ratio</td>
      <td>输入</td>
      <td>key的压缩倍数。</td>
      <td>
          <ul>
                <li>Atlas A3 训练系列产品/Atlas A3 推理系列产品、Atlas A2 训练系列产品/Atlas A2 推理系列产品支持2的幂次方且范围为[1, 128]，即1/2/4/8/16/32/64/128。</li>
                <li>Ascend 950PR/Ascend 950DT支持(0, 128]范围内的正整数。</li>
                <li>默认值1。</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>return_value</td>
      <td>输入</td>
      <td>表示是否输出sparseValuesOut。</td>
      <td>
          <ul>
                <li>1表示输出，0表示不输出；默认值为0。</li>
                <li>仅在Ascend 950PR/Ascend 950DT上支持，Atlas A3 训练系列产品/Atlas A3 推理系列产品、Atlas A2 训练系列产品/Atlas A2 推理系列产品不支持。</li>
          </ul>
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

## 返回值说明<a name="zh-cn_topic_0000002168254826_section22231435517"></a>

  <table style="undefined;table-layout: fixed; width: 1601px"><colgroup>
  <col style="width: 264px">
  <col style="width: 132px">
  <col style="width: 232px">
  <col style="width: 330px">
  <col style="width: 164px">
  <col style="width: 119px">
  <col style="width: 215px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>sparseIndicesOut</td>
      <td>输出</td>
      <td>公式中的Indices输出。</td>
      <td>不支持空tensor。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_query为"BSND"时输出shape为[B, S1, N2, topk]。</li>
                <li>layout_query为"TND"时输出shape为[T1, N2, topk]。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>sparseValuesOut</td>
      <td>输出</td>
      <td>公式中的Indices输出对应的value值。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>仅当returnValue为1时输出有效值，否则输出bf16负无穷。</li>
          </ul>
      </td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>shape与sparseIndicesOut保持一致。</td>
      <td>x</td>
    </tr>
  </tbody>
  </table>

## 约束说明<a name="zh-cn_topic_0000002168254826_section12345537164214"></a>

- 声明
  - 参数cu_seqlens_q、cu_seqlens_k、seqused_q、seqused_k及block_table属于tensor。由于算子在Tiling阶段无法获取tensor的具体数值，tiling侧不对值进行校验，正确性需要用户自行保证。若上述参数传入非法值，会触发未定义行为（精度问题、非法内存访问导致的程序崩溃等）。
  - lightning_indexer_v2_metadata和lightning_indexer_v2的入参在调用时应该保持一致。由于算子分为两个接口分段调用，算子无法自行校验，正确性需要由客户自行保证。若接口传入参数不一致，会发生未定义行为（精度问题、非法内存访问导致的程序崩溃等）。

### 特性参数组

|      特性参数组      |     参数字段名称     |    字段分组    |  字段类型  |
| :-------------------: | :-------------------: | :-------------: | :--------: |
|      公共参数组      |         q         |      INPUT      |   Tensor   |
|                      |          k          |      INPUT      | Tensor |
|                      |          w          |      INPUT      | Tensor |
|                      |         metadata        |      INPUT(OPTIONAL)      | Tensor |
|                      |         output_idx_offset  |      INPUT(OPTIONAL)      | Tensor |
|                      |      topk      | ATTR(OPTIONAL) |   int   |
|                      |      layout_q      | ATTR(OPTIONAL) |   string   |
|                      |      layout_k      | ATTR(OPTIONAL) |   string   |
|                      |     sparse_indices     |     OUTPUT     |   Tensor   |
|                      |     sparse_values     |     OUTPUT     |   Tensor   |
|      Mask参数组      |       mask_mode       | ATTR(OPTIONAL) |   int   |
| SeqLens参数组  |   cu_seqlens_q   | INPUT(OPTIONAL) |  Tensor  |
|                      |  cu_seqlens_k  | INPUT(OPTIONAL) |  Tensor  |
|                      |  seqused_q  | INPUT(OPTIONAL) |  Tensor  |
|                      |  seqused_k  | INPUT(OPTIONAL) |  Tensor  |
|                      |  max_seqlen_q  | ATTR(OPTIONAL) |  int  |
| Paged Attention参数组 |      block_table      | INPUT(OPTIONAL) |   Tensor   |
|   compress参数组   |    cmp_ratio    | ATTR(OPTIONAL) |    int    |
|                      |      cmp_residual_k      |     INPUT(OPTIONAL)     |   Tensor   |

### 基准信息说明

资料约束中，常见字段释义如下：

|    命名    |                            含义                            |
| :---------: | :---------------------------------------------------------: |
|      B      |                Batch,表示输入样本批量大小                |
|     Q_N     |        输入q tensor的头数，对应q shape中的N        |
|    K_N    |    输入k tensor的头数，对应k/v shape中的N    |
|     Q_S     |      输入q tensor的序列长度，对应q shape中的S      |
|    K_S    |  输入k tensor的序列长度，对应k/v shape中的S  |
|     Q_T     |          输入q tensor所有batch序列长度的累加和          |
|     K_T     |          输入k tensor所有batch序列长度的累加和          |
|     D     |          输入q/k tensor隐藏层最小的单元尺寸headdim         |

### 参数组约束

#### 公共参数组
- 入参为空的场景处理：
    - 空Tensor指必选输入和输出的shape size为0,即有任意轴为0。
    - 触发空tensor的用例将全部拦截报错。

- q、k、v、sparse_indices、sparse_values校验
<table style="undefined;table-layout: fixed; width:1625px"><colgroup>
<col style="width: 147px">
<col style="width: 232px">
<col style="width: 232px">
<col style="width: 293px">
<col style="width: 185px">
</colgroup>
<thead>
<tr>
    <th>参数</th>
    <th>单参数校验</th>
    <th>存在性校验</th>
    <th>一致性校验</th>
    <th>特性交叉校验</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>q</td>
        <td>
            <ul>
                <li>tensor_type支持BFLOAT16和FLOAT16</li>
                <li>BSND -> (B, S_Q, N_Q, D)</li>
                <li>TND -> (T_Q, N_Q, D)</li>
            </ul>
        </td>
        <td rowspan="4">
            必须存在
        </td>
        <td rowspan="4">
            <ul>
                <li>q、k的数据类型需相同</li>
                <li>k、v的shape需相同</li>
                <li>Layout校验规则见layout匹配关系表</li>
            </ul>
        </td>
        <td rowspan="4">
            轴校验：
            <ul>
                <li>65536 > B > 0</li>
                <li>Q_T > 0</li>
                <li>K_T > 0</li>
                <li>Q_N > 0</li>
                <li>K_N > 0</li>
                <li>Q_S > 0</li>
                <li>K_S > 0</li>
                <li>D = 128</li>
                <li>Q_N % K_N == 0且Q_N / K_N > 0</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>k</td>
        <td rowspan="1">
            <ul>
                <li>tensor_type支持BFLOAT16和FLOAT16</li>
                <li>BSND -> (B, S_K, N_K, D)</li>
                <li>TND -> (T_K, N_K, D)</li>
                <li>PA_BBND -> (num_blocks, block_size, K_N, D)</li>
                <li>1024 >= block_size >= 16，block_size % 16 == 0</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>sparse_indices</td>
        <td rowspan="1">
            <ul>
                <li>tensor_type支持INT32</li>
                <li>shape维度支持3、4</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>sparse_values</td>
        <td rowspan="1">
            <ul>
                <li>tensor_type支持BFLOAT16</li>
                <li>shape维度支持3、4</li>
            </ul>
        </td>
    </tr>
</tbody>
</table>


layout匹配关系表：
<table style="undefined;table-layout: fixed; width:1625px"><colgroup>
<col style="width: 247px">
<col style="width: 132px">
<col style="width: 232px">
<col style="width: 293px">
<col style="width: 185px">
<col style="width: 119px">
<col style="width: 272px">
<col style="width: 145px">
</colgroup>
<thead>
<tr>
    <th>layout_q</th>
    <th>layout_k</th>
    <th>layout_out</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>BSND</td>
        <td>
          <li>BSND</li>
          <li>PA_BBND</li>
        </td>
        <td>BSND</td>
    </tr>
    <tr>
        <td>TND</td>
        <td>
          <li>TND</li>
          <li>PA_BBND</li>
        </td>
        <td>TND</td>
    </tr>
</tbody>
</table>

metadata校验
<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
            <th>一致性校验</th>
            <th>特性交叉校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>metadata</td>
            <td>
                <ul>
                    <li>tensor_type仅支持INT32</li>
                    <li>shape由lightning_indexer_v2_metadata动态计算</li>
                    <li>当前不支持不传入，未传入将发出拦截报警</li>
                </ul>
            </td>
            <td>可选参数</td>
            <td>无</td>
            <td>传入时需与lightning_indexer_v2_metadata生成的结果一致</td>
        </tr>
    </tbody>
</table>

mask_mode参数解释
<ul>
    <li>mask_mode=0，全计算模式（默认值）</li>
    <li>mask_mode=3，Causal模式</li>
</ul>

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>mask_mode</td>
            <td>
                <ul>
                    <li>data_type支持INT</li>
                    <li>支持输入范围仅为0、3，默认值为0</li>
                </ul>
            </td>
            <td>
                可选输入，如果不传该参数，默认值为0
            </td>
        </tr>
    </tbody>
</table>

#### SeqLengths参数组

<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
            <th>一致性校验</th>
            <th>特性交叉校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>seqused_q</td>
            <td rowspan="2">
                <ul>
                    <li>tensor_type支持INT32</li>
                    <li>tensor_shape为(B,)</li>
                    <li>仅支持非负整数</li>
                    <li>seqused_q中的值需小于等于Q_S</li>
                    <li>seqused_k中的值需小于等于K_S</li>
                </ul>
            </td>
            <td rowspan="6">可选参数</td>
            <td rowspan="6">无</td>
            <td rowspan="2">无</td>
        </tr>
        <tr>
            <td>seqused_k</td>
        </tr>
        <tr>
            <td>cu_seqlens_q</td>
            <td>
                <ul>
                    <li>tensor_type支持INT32</li>
                    <li>tensor_shape为(B+1,)</li>
                    <li>值仅支持非负整数</li>
                    <li>其值应非递减（大于等于前一个值）排列，第一个元素为0且最后一个元素等于Q_T</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>当layout_q为TND时，必须传入</li>
                    <li>当layout_q不为TND时，不支持传入</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>cu_seqlens_k</td>
            <td>
                <ul>
                    <li>tensor_type支持INT32</li>
                    <li>tensor_shape为(B+1,)</li>
                    <li>值仅支持非负整数</li>
                    <li>其值应非递减（大于等于前一个值）排列，第一个元素为0且最后一个元素等于K_T</li>
                </ul>
            </td>
            <td>
                <ul>
                    <li>当layout_k为TND时，必须传入</li>
                    <li>当layout_k不为TND时，不支持传入</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>max_seqlen_q</td>
            <td rowspan="2">
                <ul>
                    <li>data_type支持INT</li>
                    <li>暂不生效，仅支持-1</li>
                    <li>默认值为-1</li>
                </ul>
            </td>
            <td rowspan="2">
                <ul>
                    <li>暂不生效，仅支持传入-1</li>
                </ul>
            </td>
        </tr>
        <tr>
            <td>max_seqlen_k</td>
        </tr>
    </tbody>
</table>

#### Paged Attention参数组
当block_table不为空时，开启Paged Attention
<table style="undefined;table-layout: fixed; width:1625px">
    <colgroup>
        <col style="width: 147px">
        <col style="width: 232px">
        <col style="width: 232px">
        <col style="width: 293px">
        <col style="width: 185px">
    </colgroup>
    <thead>
        <tr>
            <th>参数</th>
            <th>单参数校验</th>
            <th>存在性校验</th>
            <th>一致性校验</th>
            <th>特性交叉校验</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>block_table</td>
            <td>
                <ul>
                    <li>tensor_type仅支持INT32</li>
                    <li>tensor_shape为(B, max_num_blocks_per_seq)</li>
                    <li>值只能为正整数</li>
                </ul>
            </td>
            <td>可选参数</td>
            <td>无</td>
            <td>
                <ul>
                    <li>PagedAttention开启情况下，必须传入seqused_k</li>
                    <li>PagedAttention开启情况下，block_table必须不为空</li>
                </ul>
            </td>
        </tr>
    </tbody>
</table>

  
## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include "securec.h"
#include "acl/acl.h"
#include "aclnnop/aclnn_lightning_indexer_v2.h"

using namespace std;

namespace {

#define CHECK_RET(cond) ((cond) ? true :(false))

#define LOG_PRINT(message, ...)     \
  do {                              \
    (void)printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
    return ret;
  }
  ret = aclrtSetDevice(deviceId);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
    return ret;
  }
  ret = aclrtCreateStream(stream);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
    return ret;
  }
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
    return ret;
  }

  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
    LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
    return ret;
  }

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

struct TensorResources {
    void* queryDeviceAddr = nullptr;
    void* keyDeviceAddr = nullptr;
    void* weightsDeviceAddr = nullptr;
    void* cmpResidualKDeviceAddr = nullptr;
    void* sparseIndicesDeviceAddr = nullptr;
    void* sparseValuesDeviceAddr = nullptr;

    aclTensor* queryTensor = nullptr;
    aclTensor* keyTensor = nullptr;
    aclTensor* weightsTensor = nullptr;
    aclTensor* cmpResidualKTensor = nullptr;
    aclTensor* sparseIndicesTensor = nullptr;
    aclTensor* sparseValuesTensor = nullptr;
};

int InitializeTensors(TensorResources& resources) {
    int64_t B = 2;
    int64_t S1 = 64;
    int64_t N1 = 16;
    int64_t D = 128;
    int64_t N2 = 1;
    int64_t topk = 32;
    int64_t cmpRatio = 4;
    int64_t S2Orig = 130;
    int64_t S2Compressed = S2Orig / cmpRatio;
    int64_t resK = S2Orig % cmpRatio;

    std::vector<int64_t> queryShape = {B, S1, N1, D};
    std::vector<int64_t> keyShape = {B, S2Compressed, N2, D};
    std::vector<int64_t> weightsShape = {B, S1, N1};
    std::vector<int64_t> cmpResidualKShape = {B};
    std::vector<int64_t> sparseIndicesShape = {B, S1, N2, topk};
    std::vector<int64_t> sparseValuesShape = {B, S1, N2, topk};

    int64_t queryShapeSize = GetShapeSize(queryShape);
    int64_t keyShapeSize = GetShapeSize(keyShape);
    int64_t weightsShapeSize = GetShapeSize(weightsShape);
    int64_t sparseIndicesShapeSize = GetShapeSize(sparseIndicesShape);
    int64_t sparseValuesShapeSize = GetShapeSize(sparseValuesShape);

    std::vector<uint16_t> queryHostData(queryShapeSize, 0x3C00);
    std::vector<uint16_t> keyHostData(keyShapeSize, 0x3C00);
    std::vector<float> weightsHostData(weightsShapeSize, 1.0f);
    std::vector<int32_t> cmpResidualKHostData(B, static_cast<int32_t>(resK));
    std::vector<int32_t> sparseIndicesHostData(sparseIndicesShapeSize, 0);
    std::vector<float> sparseValuesHostData(sparseValuesShapeSize, 0.0f);

    int ret = CreateAclTensor(queryHostData, queryShape, &resources.queryDeviceAddr,
                              aclDataType::ACL_FLOAT16, &resources.queryTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(keyHostData, keyShape, &resources.keyDeviceAddr,
                          aclDataType::ACL_FLOAT16, &resources.keyTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(weightsHostData, weightsShape, &resources.weightsDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.weightsTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(cmpResidualKHostData, cmpResidualKShape, &resources.cmpResidualKDeviceAddr,
                          aclDataType::ACL_INT32, &resources.cmpResidualKTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(sparseIndicesHostData, sparseIndicesShape, &resources.sparseIndicesDeviceAddr,
                          aclDataType::ACL_INT32, &resources.sparseIndicesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    ret = CreateAclTensor(sparseValuesHostData, sparseValuesShape, &resources.sparseValuesDeviceAddr,
                          aclDataType::ACL_FLOAT, &resources.sparseValuesTensor);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
      return ret;
    }

    return ACL_SUCCESS;
}

int ExecuteLightningIndexerV2(TensorResources& resources, aclrtStream stream,
                                 void** workspaceAddr, uint64_t* workspaceSize) {
    int64_t topk = 32;
    int64_t maxSeqlenQ = 64;
    int64_t maskMode = 3;
    int64_t cmpRatio = 4;
    int64_t returnValue = 0;
    constexpr const char layoutQueryStr[] = "BSND";
    constexpr const char layoutKeyStr[] = "BSND";
    constexpr size_t layoutQueryLen = sizeof(layoutQueryStr);
    constexpr size_t layoutKeyLen = sizeof(layoutKeyStr);
    char layoutQuery[layoutQueryLen];
    char layoutKey[layoutKeyLen];
    errno_t memcpyRet = memcpy_s(layoutQuery, sizeof(layoutQuery), layoutQueryStr, layoutQueryLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutQuery failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    memcpyRet = memcpy_s(layoutKey, sizeof(layoutKey), layoutKeyStr, layoutKeyLen);
    if (!CHECK_RET(memcpyRet == 0)) {
        LOG_PRINT("memcpy_s layoutKey failed. ERROR: %d\n", memcpyRet);
        return -1;
    }
    aclOpExecutor* executor;

    int ret = aclnnLightningIndexerV2GetWorkspaceSize(
        resources.queryTensor, resources.keyTensor, resources.weightsTensor,
        nullptr, nullptr, nullptr, nullptr,
        resources.cmpResidualKTensor, nullptr, nullptr, nullptr,
        topk, maxSeqlenQ,
        layoutQuery, layoutKey,
        maskMode, cmpRatio, returnValue,
        resources.sparseIndicesTensor, resources.sparseValuesTensor,
        workspaceSize, &executor);

    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnLightningIndexerV2GetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret;
    }

    if (*workspaceSize > 0ULL) {
        ret = aclrtMalloc(workspaceAddr, *workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (!CHECK_RET(ret == ACL_SUCCESS)) {
            LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
            return ret;
        }
    }

    ret = aclnnLightningIndexerV2(*workspaceAddr, *workspaceSize, executor, stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclnnLightningIndexerV2 failed. ERROR: %d\n", ret);
        return ret;
    }

    return ACL_SUCCESS;
}

int PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<int32_t> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
        return ret;
  }
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("sparse_indices result[%ld] is: %d\n", i, resultData[i]);
  }
  return ACL_SUCCESS;
}

void CleanupResources(TensorResources& resources, void* workspaceAddr,
                     aclrtStream stream, int32_t deviceId) {
    if (resources.queryTensor) {
      aclDestroyTensor(resources.queryTensor);
    }
    if (resources.keyTensor) {
      aclDestroyTensor(resources.keyTensor);
    }
    if (resources.weightsTensor) {
      aclDestroyTensor(resources.weightsTensor);
    }
    if (resources.cmpResidualKTensor) {
      aclDestroyTensor(resources.cmpResidualKTensor);
    }
    if (resources.sparseIndicesTensor) {
      aclDestroyTensor(resources.sparseIndicesTensor);
    }
    if (resources.sparseValuesTensor) {
      aclDestroyTensor(resources.sparseValuesTensor);
    }

    if (resources.queryDeviceAddr) {
      aclrtFree(resources.queryDeviceAddr);
    }
    if (resources.keyDeviceAddr) {
      aclrtFree(resources.keyDeviceAddr);
    }
    if (resources.weightsDeviceAddr) {
      aclrtFree(resources.weightsDeviceAddr);
    }
    if (resources.cmpResidualKDeviceAddr) {
      aclrtFree(resources.cmpResidualKDeviceAddr);
    }
    if (resources.sparseIndicesDeviceAddr) {
      aclrtFree(resources.sparseIndicesDeviceAddr);
    }
    if (resources.sparseValuesDeviceAddr) {
      aclrtFree(resources.sparseValuesDeviceAddr);
    }

    if (workspaceAddr) {
      aclrtFree(workspaceAddr);
    }
    if (stream) {
      aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceId);
    aclFinalize();
}

} // namespace

int main() {
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    TensorResources resources = {};
    void* workspaceAddr = nullptr;
    uint64_t workspaceSize = 0;
    int64_t B = 2;
    int64_t S1 = 64;
    int64_t N2 = 1;
    int64_t topk = 32;
    std::vector<int64_t> sparseIndicesShape = {B, S1, N2, topk};
    int ret = ACL_SUCCESS;

    ret = Init(deviceId, &stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
        return ret;
    }

    ret = InitializeTensors(resources);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("InitializeTensors failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = ExecuteLightningIndexerV2(resources, stream, &workspaceAddr, &workspaceSize);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("ExecuteLightningIndexerV2 failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    ret = aclrtSynchronizeStream(stream);
    if (!CHECK_RET(ret == ACL_SUCCESS)) {
        LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        CleanupResources(resources, workspaceAddr, stream, deviceId);
        return ret;
    }

    PrintOutResult(sparseIndicesShape, &resources.sparseIndicesDeviceAddr);

    CleanupResources(resources, workspaceAddr, stream, deviceId);
    return 0;
}
```