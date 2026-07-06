# QuantLightningIndexerV2

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列加速卡产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

-   API功能：QuantLightningIndexerV2是推理场景下，稀疏attention前处理的计算，选出关键的稀疏token，并对输入query和key进行量化实现存8算8，获取最大收益。

-   计算公式：
    $$out = \text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(\left(Scale_Q@Scale_K^T\right)\odot\left(Q_{index}^{Quant}@{\left(K_{index}^{Quant}\right)}^T\right)\right)\right]\right\}$$
    主要计算过程为：
    1. 将某个token对应的输入参数`query`（$Q_{index}^{Quant}\in\R^{g\times d}$）乘以给定上下文`key`（$K_{index}^{Quant}\in\R^{S_{k}\times d}$），得到相关性。
    2. 相关性结果与`query`和`key`对应的反量化系数`query_dequant_scale`（$Scale_Q$）和`key_dequant_scale`（$Scale_K^T$）相乘，通过激活函数$ReLU$过滤无效负相关信号后，得到当前Token与所有前序Token的相关性分数向量。
    3. 将其与权重系数`weights`（$W$）相乘后，沿g的方向，选取前$Top-k$个索引值得到输出$out$，作为Attention的输入。

## 参数说明

> **说明：**<br>
> 参数维度含义：B表示Batch Size、Q_S和K_S分别表示query和key的Sequence Length、Q_N和K_N分别表示query和key的Head Num、D表示Head Dim（Q_D和K_D取值相等为128）、Q_T和K_T分别表示query和key的Total Tokens、sparse_count表示最后选取的索引个数（topK）、block_num和block_size分别表示PageAttention场景下的block总数和每个block的token数。K_N仅支持1。

  <table style="undefined;table-layout: fixed; width: 1080px"><colgroup>
  <col style="width: 200px">
  <col style="width: 150px">
  <col style="width: 480px">
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
    </tr></thead>
  <tbody>
    <tr>
      <td>query</td>
      <td>输入</td>
      <td>
          <ul>
                <li>公式中的$Q_{index}^{Quant}\in\R^{g\times d}$，表示输入Index Query。</li>
                <li>不支持空tensor和非连续。</li>
                <li>layout_query为BSND时，shape为[B, Q_S, Q_N, D]；layout_query为TND时，shape为[Q_T, Q_N, D]。</li>
          </ul>
      </td>
      <td>INT8、FLOAT8_e4m3fn、HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输入</td>
      <td>
          <ul>
                <li>公式中的$K_{index}^{Quant}\in\R^{S_{k}\times d}$，表示压缩后的输入Index Key。</li>
                <li>支持0轴非连续。</li>
                <li>layout_key为PA_BSND时，shape为[block_num, block_size, K_N, D]。</li>
                <li>Ascend 950PR/Ascend 950DT额外支持layout_key为BSND（shape为[B, K_S, K_N, D]）和TND（shape为[K_T, K_N, D]）。</li>
          </ul>
      </td>
      <td>INT8、FLOAT8_e4m3fn、HIFLOAT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weights</td>
      <td>输入</td>
      <td>
          <ul>
                <li>公式中的$W$，表示权重系数。</li>
                <li>不支持空tensor和非连续。</li>
                <li>layout_query为BSND时，shape为[B, Q_S, Q_N]；layout_query为TND时，shape为[Q_T, Q_N]。</li>
          </ul>
      </td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>query_dequant_scale</td>
      <td>输入</td>
      <td>
          <ul>
                <li>公式中的$Scale_Q$，表示Index Query的反量化系数。</li>
                <li>不支持空tensor和非连续。</li>
                <li>layout_query为BSND时，shape为[B, Q_S, Q_N]；layout_query为TND时，shape为[Q_T, Q_N]。</li>
          </ul>
      </td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>key_dequant_scale</td>
      <td>输入</td>
      <td>
          <ul>
                <li>公式中的$Scale_K$，表示Index Key的反量化系数。</li>
                <li>不支持空tensor和非连续。</li>
                <li>layout_key为PA_BSND时，shape为[block_num, block_size, K_N]。</li>
          </ul>
      </td>
      <td>FLOAT16、FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>actual_seq_lengths_query</td>
      <td>可选输入</td>
      <td>
          <ul>
                <li>表示不同Batch中query的有效token数。</li>
                <li>可传入None表示与query的Q_S长度相同。</li>
                <li>支持长度为B的一维tensor，且每个Batch的有效token数不超过query中的Q_S大小且不小于0。</li>
                <li>layout_query为TND时该入参必须传入，并以元素数量作为B值。每个元素表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>actual_seq_lengths_key</td>
      <td>可选输入</td>
      <td>
          <ul>
                <li>表示不同Batch中key的有效token数。</li>
                <li>可传入None表示与key的K_S长度相同。</li>
                <li>支持长度为B的一维tensor，且每个Batch的有效token数不超过key中的K_S大小且不小于0。</li>
                <li>layout_key为PA_BSND时该入参必须传入。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>block_table</td>
      <td>可选输入</td>
      <td>
          <ul>
                <li>表示PageAttention中KV存储使用的block映射表。</li>
                <li>PageAttention场景下，block_table必须为二维，第一维长度需要等于B，第二维长度不能小于maxBlockNumPerSeq（每个batch中最大actual_seq_lengths_key对应的block数量）。</li>
                <li>shape支持[B, K_S/block_size]。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>metadata</td>
      <td>可选输入</td>
      <td>
          <ul>
                <li>QuantLightningIndexerV2Metadata算子传入的分核信息，包含使用核数、分块大小以及每个核处理数据的起始点等内容，shape大小为[1024]，当前不支持传空。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>query_quant_mode</td>
      <td>属性</td>
      <td>
          <ul>
                <li>用于标识输入query的量化模式。</li>
                <li>Atlas A3 训练系列产品/Atlas A3 推理系列产品仅支持传入2（Per-Token-Head量化）。</li>
                <li>Ascend 950PR/Ascend 950DT支持1（FLOAT8_e4m3fn量化）和4（HIFLOAT8量化）。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>key_quant_mode</td>
      <td>属性</td>
      <td>
          <ul>
                <li>用于标识输入key的量化模式。</li>
                <li>当前与query_quant_mode取值一致。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_query</td>
      <td>可选属性</td>
      <td>
          <ul>
                <li>用于标识输入query的数据排布格式。</li>
                <li>当前支持BSND、TND，默认值"BSND"。</li>
          </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_key</td>
      <td>可选属性</td>
      <td>
          <ul>
                <li>用于标识输入key的数据排布格式。</li>
                <li>Atlas A3 训练系列产品/Atlas A3 推理系列产品仅支持传入PA_BSND，默认值"PA_BSND"。</li>
                <li>Ascend 950PR/Ascend 950DT支持PA_BSND、BSND、TND，默认值"PA_BSND"。</li>
          </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_count</td>
      <td>可选属性</td>
      <td>
          <ul>
                <li>代表topK阶段需要保留的block数量。</li>
                <li>支持[1, 2048]，默认值2048。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_mode</td>
      <td>可选属性</td>
      <td>
          <ul>
                <li>表示sparse的模式，支持0/3。</li>
                <li>sparse_mode为0时，代表defaultMask模式。</li>
                <li>sparse_mode为3时，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pre_tokens</td>
      <td>可选属性</td>
      <td>预留参数，表示attention需要和前几个Token计算关联，仅支持默认值2^63-1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>next_tokens</td>
      <td>可选属性</td>
      <td>预留参数，表示attention需要和后几个Token计算关联，仅支持默认值2^63-1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmp_ratio</td>
      <td>可选属性</td>
      <td>
          <ul>
                <li>用于稀疏计算，表示key的压缩倍数。</li>
                <li>Atlas A3 训练系列产品/Atlas A3 推理系列产品支持2的幂次方且范围为[1, 128]，即1/2/4/8/16/32/64/128。</li>
                <li>Ascend 950PR/Ascend 950DT支持(0, 128]范围内的正整数。</li>
                <li>默认值1。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>return_value</td>
      <td>可选属性</td>
      <td>
          <ul>
                <li>表示是否输出sparse_values。</li>
                <li>True表示输出，False表示不输出；默认值为False。</li>
                <li>仅在Ascend 950PR/Ascend 950DT上支持，Atlas A3 训练系列产品/Atlas A3 推理系列产品不支持。</li>
          </ul>
      </td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_indices</td>
      <td>输出</td>
      <td>
          <ul>
                <li>公式中的输出Out，参与稀疏attention计算的token索引值。</li>
                <li>layout_query为BSND时输出shape为[B, Q_S, K_N, sparse_count]；layout_query为TND时输出shape为[Q_T, K_N, sparse_count]。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sparse_values</td>
      <td>输出</td>
      <td>
          <ul>
                <li>公式中的Indices输出对应的value值。</li>
                <li>shape与sparse_indices保持一致。</li>
                <li>仅当return_value为True时输出有效值，否则输出bf16负无穷。</li>
          </ul>
      </td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

- <term>Ascend 950PR/Ascend 950DT</term>：query、key在quant_mode为1时支持FLOAT8_e4m3fn，quant_mode为4时支持HIFLOAT8，不支持INT8；weights、query_dequant_scale和key_dequant_scale支持FLOAT32，不支持FLOAT16。
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：query、key支持INT8，不支持FLOAT8_e4m3fn和HIFLOAT8；weights、query_dequant_scale和key_dequant_scale支持FLOAT16，不支持FLOAT32。

## 约束说明

-   该接口支持图模式。
-   该接口要求$W \odot Scale_Q$的结果在`float16`(Atlas A3)/`float32`(Ascend 950PR/Ascend 950DT)的表示范围内。
-   该接口的TopK过程对NAN排序是未定义行为。
-   参数query中的D轴和参数key中的D轴值相等为128。
-   参数query和key中的N轴分别支持：Atlas A3 训练系列产品/Atlas A3 推理系列产品仅支持Q_N=64、K_N=1（即group size=64）；Ascend 950PR/Ascend 950DT支持Q_N=32或64、K_N=1（即group size=32或64）。
-   block_size取值为16的整数倍，最大支持到1024。
-   当`layout_query`为TND时，`actual_seq_lengths_query`必须传入，且以该入参元素的数量作为B值，该入参中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。不能出现负值。
-   当`layout_key`为PA_BSND时，`actual_seq_lengths_key`该入参必须传入。
-   当`layout_key`不为PA_BSND时，`layout_query`和`layout_key`必须一致。
-   PageAttention场景下，`block_table`必须为二维，第一维长度需要等于B，第二维长度不能小于maxBlockNumPerSeq（每个batch中最大actual_seq_lengths_key对应的block数量），支持block_size取值为16的整数倍，最大支持到1024。
-   query、key、weights、query_dequant_scale、key_dequant_scale数据排布格式支持从多种维度解读，其中B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、N（Head Num）表示多头数、D（Head Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
-   当`quant_mode`为4时（仅Ascend 950PR/Ascend 950DT），`query_dequant_scale`和`key_dequant_scale`的shape维度必须为1且shape[0]=1（即Per-Tensor量化）。
-   `return_value`仅在Ascend 950PR/Ascend 950DT上支持，Atlas A3 训练系列产品/Atlas A3 推理系列产品不支持。

## 调用示例

<table class="tg"><thead>
  <tr>
    <th class="tg-0pky">调用方式</th>
    <th class="tg-0pky">样例代码</th>
    <th class="tg-0pky">说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="6">aclnn接口</td>
    <td class="tg-0pky">
    <a href="./examples/test_aclnn_quant_lightning_indexer_v2.cpp">test_aclnn_quant_lightning_indexer_v2
    </a>
    </td>
    <td class="tg-lboi" rowspan="6">
    通过
    <a href="./docs/aclnnQuantLightningIndexerV2.md">aclnnQuantLightningIndexerV2
    </a>
    接口方式调用算子
    </td>
  </tr>
</tbody></table>
