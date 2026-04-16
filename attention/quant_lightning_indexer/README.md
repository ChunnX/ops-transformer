# QuantLightningIndexer

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列加速卡产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：QuantLightningIndexer是推理场景下，SparseFlashAttention（SFA）前处理的计算，选出关键的稀疏token，并对输入query和key进行量化实现存8算8，获取最大收益。

- 计算公式：
    $$out = \text{Top-}k\left\{[1]_{1\times g}@\left[(W@[1]_{1\times S_{k}})\odot\text{ReLU}\left(\left(Scale_Q@Scale_K^T\right)\odot\left(Q_{index}^{Quant}@{\left(K_{index}^{Quant}\right)}^T\right)\right)\right]\right\}$$
    主要计算过程为：
    1. 将某个token对应的输入参数`query`（$Q_{index}^{Quant}\in\R^{g\times d}$）乘以给定上下文`key`（$K_{index}^{Quant}\in\R^{S_{k}\times d}$），得到相关性。
    2. 相关性结果与`query`和`key`对应的反量化系数`query_dequant_scale`（$Scale_Q$）和`key_dequant_scale`（$Scale_K^T$）相乘，通过激活函数$ReLU$过滤无效负相关信号后，得到当前Token与所有前序Token的相关性分数向量。
    3. 将其与权重系数`weights`（$W$）相乘后，沿g的方向，选取前$Top-k$个索引值得到输出$out$，作为SparseFlashAttention的输入。

## 参数说明
>**说明：**<br> 
>
>- query、key、weights、query_dequant_scale、key_dequant_scale参数维度含义：B（Batch Size）表示输入样本批量大小、S（Sequence Length）表示输入样本序列长度、H（Head Size）表示hidden层的大小、N（Head Num）表示多头数、D（Head Dim）表示hidden层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
>- 使用S1和S2分别表示query和key的输入样本序列长度，N1和N2分别表示query和key对应的多头数，k表示最后选取的索引个数。参数query中的D和参数key中的D值相等为128。T1和T2分别表示query和key的输入样本序列长度的累加和。
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
      <td>query</td>
      <td>输入</td>
      <td>公式中的输入Q。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>Ascend 950PR/Ascend 950DT N1仅支持16、24、32、64。</li>
          </ul>
      </td>
      <td>
          <ul>
                <li>Atlas A3 推理系列产品数据类型支持int8。</li>
                <li>Ascend 950PR/Ascend 950DT数据类型支持float8_e4m3fn、hifloat8。</li>
          </ul>
      </td>
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
      <td>key</td>
      <td>输入</td>
      <td>公式中的输入K。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>block_num为PageAttention时block总数，block_size为一个block的token数。</li>
                <li>block_size取值为16的整数倍，最大支持到1024。</li>
                <li>N2仅支持1。</li>
          </ul>
      </td>
      <td>
          <ul>
                <li>Atlas A3 推理系列产品数据类型支持int8。</li>
                <li>Ascend 950PR/Ascend 950DT数据类型支持float8_e4m3fn、hifloat8。</li>
          </ul>
      </td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_key为PA_BSND时，shape为(block_num, block_size, N2, D)。</li>
                <li>layout_kv为BSND时，shape为(B, S2, N2, D)。</li>
                <li>layout_kv为TND时，shape为(T2, N2, D)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>weights</td>
      <td>输入</td>
      <td>公式中的输入W。</td>
      <td>不支持空tensor。</td>
      <td>
          <ul>
                <li>Atlas A3 推理系列产品数据类型支持float16。</li>
                <li>Ascend 950PR/Ascend 950DT 数据类型支持bfloat16和float16。</li>
          </ul>
      </td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_query为BSND时，shape为(B,S1,N1)。</li>
                <li>layout_query为TND时，shape为(T,N1)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>query_dequant_scale</td>
      <td>输入</td>
      <td>公式中Query的反量化系数Scale_Q。</td>
      <td>不支持空tensor。</td>
      <td>
          <ul>
                <li>Atlas A3 推理系列产品数据类型支持float16。</li>
                <li>Ascend 950PR/Ascend 950DT 数据类型支持float和float16。</li>
          </ul>
      </td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_query为BSND时，shape为(B,S1,N1)。</li>
                <li>layout_query为TND时，shape为(T,N1)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>key_dequant_scale</td>
      <td>输入</td>
      <td>公式中Key的反量化系数Scale_K。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>block_num为PageAttention时block总数，block_size为一个block的token数。</li>
          </ul>
      </td>
      <td>
          <ul>
                <li>Atlas A3 推理系列产品数据类型支持float16。</li>
                <li>Ascend 950PR/Ascend 950DT 数据类型支持float和float16。</li>
          </ul>
      </td>
      <td>ND</td>
      <td>
          <ul>
                <li>layout_query为BSND时，shape为(B,S1,N2)。</li>
                <li>layout_query为TND时，shape为(T,N2)。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
    <tr>
      <td>actual_seq_lengths_query</td>
      <td>可选输入</td>
      <td>每个Batch中，Query的有效token数。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>如果不指定seqlen可传入None，表示和query的shape的S长度相同。</li>
                <li>该入参中每个Batch的有效token数不超过query中的维度S大小且不小于0，支持长度为B的一维tensor。</li>
                <li>当layout_query为TND时，该入参必须传入，且以该入参元素的数量作为B值，该入参中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。</li>
                <li>不能出现负值。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>actual_seq_lengths_key</td>
      <td>可选输入</td>
      <td>每个Batch中，Key的有效token数。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>如果不指定seqlen可传入None，表示和key的shape的S长度相同。</li>
                <li> 该参数中每个Batch的有效token数不超过key/value中的维度S大小且不小于0，支持长度为B的一维tensor。</li>
                <li>当layout_key为TND或PA_BSND时，该入参必须传入，layout_key为TND，该参数中每个元素的值表示当前batch与之前所有batch的token数总和，即前缀和，因此后一个元素的值必须大于等于前一个元素的值。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>(B,)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>block_table</td>
      <td>可选输入</td>
      <td>表示PageAttention中KV存储使用的block映射表。</td>
      <td>
          <ul>
                <li>不支持空tensor。</li>
                <li>PageAttention场景下，block_table必须为二维，第一维长度需要等于B，第二维长度不能小于maxBlockNumPerSeq（maxBlockNumPerSeq为每个batch中最大actual_seq_lengths_key对应的block数量）</li>
                <li>支持block_size取值为16的整数倍。</li>
                <li>最大支持到1024。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>ND</td>
      <td>shape支持(B,S2_max/block_size)</td>
      <td>x</td>
    </tr>
    <tr>
      <td>query_quant_mode</td>
      <td>可选输入</td>
      <td>用于标识输入Query的量化模式。</td>
      <td>
          <ul>
                <li>当前支持Per-Token-Head量化模式。</li>
                <li>当前仅支持传入0。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>key_quant_mode</td>
      <td>可选输入</td>
      <td>用于标识输入Key的量化模式。</td>
      <td>
          <ul>
                <li>当前支持Per-Token-Head量化模式。</li>
                <li>当前仅支持传入0。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout_query</td>
      <td>可选输入</td>
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
      <td>layout_key</td>
      <td>可选输入</td>
      <td>用于标识输入Key的数据排布格式。</td>
      <td>
          <ul>
                <li>用户不特意指定时可传入默认值"BSND"。</li>
                <li>当前支持PA_BSND、BSND、TND。</li>
                <li>在非PageAttention场景下，layout_key应与layout_query保持一致。</li>
          </ul>
      </td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_count</td>
      <td>可选输入</td>
      <td>topK阶段需要保留的block数量。</td>
      <td>支持[1, 2048]。</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_mode</td>
      <td>可选输入</td>
      <td>表示sparse的模式。</td>
      <td>
          <ul>
                <li>sparse_mode为0时，代表defaultMask模式。</li>
                <li>sparse_mode为3时，代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pre_tokens</td>
      <td>可选输入</td>
      <td>用于稀疏计算，表示attention需要和前几个Token计算关联。</td>
      <td>仅支持默认值2^63-1。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>next_tokens</td>
      <td>输入</td>
      <td>用于稀疏计算，表示attention需要和后几个Token计算关联。</td>
      <td>仅支持默认值2^63-1。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>query_dtype</td>
      <td>可选输入</td>
      <td>用于支持Query为hifloat8数据类型。</td>
      <td>
          <ul>
                <li>如果Query的数据类型为hifloat8，则将该变量赋值为torch_npu.hifloat8。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>key_dtype</td>
      <td>可选输入</td>
      <td>用于支持Key为hifloat8数据类型。</td>
      <td>
          <ul>
                <li>如果Key的数据类型为hifloat8，则将该变量赋值为torch_npu.hifloat8。</li>
          </ul>
      </td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse_indices</td>
      <td>输出</td>
      <td>公式中的Indices输出。</td>
      <td>不支持空tensor。</td>
      <td>INT32</td>
      <td>-</td>
      <td>
          <ul>
                <li>layout_query为"BSND"时输出shape为[B, S1, N2, sparse_count]。</li>
                <li>layout_query为"TND"时输出shape为[T1, N2, sparse_count]。</li>
          </ul>
      </td>
      <td>x</td>
    </tr>
  </tbody>
  </table>

## 约束说明

- 该接口支持图模式。
- 该接口要求$W \odot Scale_Q$的结果在`float16`的表示范围内。
- 该接口的TopK过程对NAN排序是未定义行为。
- 对于Ascend 950PR/Ascend 950DT，当query和key的数据类型为`float8_e4m3fn`时，支持weights、query_dequant_scale、key_dequant_scale的数据类型为`bfloat16、float32、float32`或`float16、float16、float16`；当query和key的数据类型为`hifloat8`时，仅支持weights、query_dequant_scale、key_dequant_scale数据类型为`bfloat16、float32、float32`。
