# npu_flash_attn<a name="ZH-CN_TOPIC_0000002309174912"></a>

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |


## 功能说明<a name="zh-cn_topic_0000002168254826_section14441124184110"></a>

-   API功能:

    支持Flash Attention训练推理归一化。
-   计算公式:
    
    self-attention（自注意力）利用输入样本自身的关系构建了一种注意力模型。其原理是假设有一个长度为$n$的输入样本序列$x$，$x$的每个元素都是一个$d$维向量，可以将每个$d$维向量看作一个token embedding，将这样一条序列经过3个权重矩阵变换得到3个维度为$n \times d$的矩阵。

    self-attention的计算公式一般定义如下，其中$Q、K、V$为输入样本的重要属性元素，是输入样本经过空间变换得到，且可以统一到一个特征空间中。公式及算子名称中的"Attention"为"self-attention"的简写。

    $$
    Attention(Q,K,V)=Score(Q,K)V
    $$

    本算子中Score函数采用Softmax函数，self-attention计算公式为:

    $$
    Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
    $$

    其中$Q$和$K^T$的乘积代表输入$x$的注意力，为避免该值变得过大，通常除以$\sqrt{d}$进行缩放，并对每行进行softmax归一化，与$V$相乘后得到一个$n \times d$的矩阵。

    增加**sink**之后计算逻辑如下所示，主要修改相关softmax_max和softmax_sum逻辑计算部分。
    
    $$
    S = \frac{QK^T}{\sqrt{d}}
    $$
    
    $$
    m = max(sink, max(S))
    $$
    
    $$
    Attention = \frac{e^{S - m} \times V}{\sum e^{S-m} + e^{sink - m}}
    $$

    开启**return_softmax_lse**之后，返回值softmax_lse计算逻辑如下所示：

    $$
    S = \frac{QK^T}{\sqrt{d}}
    $$
    
    $$
    softmax\_max = max(S)
    $$

    $$
    softmax\_lse = log{\sum e^{S-softmax\_max}} + softmax\_max
    $$

**说明**
-   q、k、v数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小batch_size、S（Seq-Length）表示输入样本序列长度、H（Hidden-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸headdim，且满足D=H/N、Q_T表示所有query Batch输入样本序列长度的累加和，KV_T表示所有k、v Batch输入样本序列长度的累加和。
-   Q_S表示seqlen_q，Q_N表示nheads_q，KV_S表示seqlen_kv，KV_N表示nheads_kv。

## 函数原型<a name="zh-cn_topic_0000002168254826_section45077510411"></a>

```
npu_flash_attn(q, k, v, block_table=None, cu_seqlens_q=None, cu_seqlens_kv=None, seqused_q=None, seqused_kv=None, sinks=None, metadata=None, softmax_scale=1.0, mask_mode=0, attn_mask=None, win_left=-1, win_right=-1, max_seqlen_q=-1, max_seqlen_kv=-1, layout_q="BSND", layout_kv="BSND", layout_out="BSND", return_softmax_lse=0) -> (Tensor, Tensor)
```

## 参数说明<a name="zh-cn_topic_0000002168254826_section112637109429"></a>

<table style="undefined;table-layout: fixed; width:1625px"><colgroup>
<col style="width: 147px">
<col style="width: 132px">
<col style="width: 232px">
<col style="width: 293px">
<col style="width: 185px">
<col style="width: 319px">
<col style="width: 272px">
<col style="width: 45px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
    <th>维度(shape)</th>
    <th>非连续Tensor</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>q</td>
        <td>必选输入</td>
        <td>公式中的q</td>
        <td>BFLOAT16/FLOAT16</td>
        <td>ND</td>
        <td>
            <ul>
                <li>(B, Q_S, Q_N, D)</li>
                <li>(B, Q_N, Q_S, D)</li>
                <li>(Q_T, Q_N, D)</li>
            </ul>      
        </td>
        <td>✓</td>
    </tr>
    <tr>
        <td>k</td>
        <td>必选输入</td>
        <td>公式中的k</td>
        <td>BFLOAT16/FLOAT16</td>
        <td>ND</td>
        <td>
            <ul>
                <li>(B, KV_S, KV_N, D)</li>
                <li>(B, KV_N, KV_S, D)</li>
                <li>(KV_T, KV_N, D)</li>
                <li>(num_blocks, block_size, KV_N, D)</li>
                <li>(num_blocks, KV_N, block_size, D)</li>
            </ul>
        </td>
        <td>×</td>
    </tr>
    <tr>
        <td>v</td>
        <td>必选输入</td>
        <td>公式中的v</td>
        <td>BFLOAT16/FLOAT16</td>
        <td>ND</td>
        <td>
            <ul>
                <li>(B, KV_S, KV_N, D)</li>
                <li>(B, KV_N, KV_S, D)</li>
                <li>(KV_T, KV_N, D)</li>
                <li>(num_blocks, block_size, KV_N, D)</li>
                <li>(num_blocks, KV_N, block_size, D)</li>
            </ul>    
        </td>
        <td>×</td>
    </tr>
    <tr>
        <td>block_table</td>
        <td>可选输入</td>
        <td>用于分块注意力计算中的块索引映射</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(B, max_num_blocks_per_seq)</td>
        <td>✓</td>
    </tr>
    <tr>
        <td>cu_seqlens_q</td>
        <td>可选输入</td>
        <td>累积序列长度，用于处理变长序列</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(B+1,)</td>
        <td>✓</td>
    </tr>
    <tr>
        <td>cu_seqlens_kv</td>
        <td>可选输入</td>
        <td>累积序列长度，用于处理变长序列</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(B+1,)</td>
        <td>✓</td>
    </tr>
    <tr>
        <td>seqused_q</td>
        <td>可选输入</td>
        <td>指定每batch中实际使用的序列长度，截断冗余运算</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(B,)</td>
        <td>✓</td>
    </tr>
    <tr>
        <td>seqused_kv</td>
        <td>可选输入</td>
        <td>指定每batch中实际使用的序列长度，截断冗余运算</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(B,)</td>
        <td>✓</td>
    </tr>
    <tr>
        <td>sinks</td>
        <td>可选输入</td>
        <td>暂不支持</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>(Q_N,)</td>
        <td>✓</td>
    </tr>
    <tr>
        <td>metadata</td>
        <td>可选输入</td>
        <td>metadata</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(max_schedule_size,)</td>
        <td>✓</td>
    </tr>
    <tr>
        <td>softmax_scale</td>
        <td>可选输入</td>
        <td>可显式设置缩放因子，覆盖默认计算</td>
        <td>FLOAT32</td>
        <td>N/A</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>mask_mode</td>
        <td>可选输入</td>
        <td>掩码模式</td>
        <td>INT</td>
        <td>N/A</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>attn_mask</td>
        <td>可选输入</td>
        <td>掩码矩阵</td>
        <td>INT8</td>
        <td>ND</td>
        <td>(2048, 2048)</td>
        <td>✓</td>
    </tr>
    <tr>
        <td>win_left</td>
        <td>可选输入</td>
        <td>window左界限</td>
        <td>INT</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>win_right</td>
        <td>可选输入</td>
        <td>window右界限</td>
        <td>INT</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>max_seqlen_q</td>
        <td>可选输入</td>
        <td>指定查询q序列的长度上限</td>
        <td>INT</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>max_seqlen_kv</td>
        <td>可选输入</td>
        <td>指定键k和值v序列的长度上限</td>
        <td>INT</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>layout_q</td>
        <td>可选输入</td>
        <td>定义输入q张量的布局格式</td>
        <td>STRING</td>
        <td>N/A</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>layout_kv</td>
        <td>可选输入</td>
        <td>定义输入k和v张量的布局格式</td>
        <td>STRING</td>
        <td>N/A</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>layout_out</td>
        <td>可选输入</td>
        <td>定义输出张量的布局格式</td>
        <td>STRING</td>
        <td>N/A</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>return_softmax_lse</td>
        <td>可选输入</td>
        <td>是否需要获取softmax的LSE结果</td>
        <td>INT</td>
        <td>N/A</td>
        <td>-</td>
        <td>-</td>
    </tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000002168254826_section22231435517"></a>

<table style="undefined;table-layout: fixed; width:1475px"><colgroup>
<col style="width: 250px">
<col style="width: 130px">
<col style="width: 230px">
<col style="width: 290px">
<col style="width: 185px">
<col style="width: 270px">
<col style="width: 120px">

</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
    <th>维度(shape)</th>
    <th>非连续Tensor</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>attn_out</td>
        <td>固定输出</td>
        <td>flash_attn的计算输出</td>
        <td>BFLOAT16/FLOAT16</td>
        <td>ND</td>
        <td>
            <ul>
                <li>(B, Q_S, Q_N, D)</li>
                <li>(B, Q_N, Q_S, D)</li>
                <li>(Q_T, Q_N, D)</li>
            </ul>      
        </td>
        <td>✓</td>
    </tr>
    <tr>
        <td>softmax_lse</td>
        <td>可选输出</td>
        <td>softmax的LSE结果</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>
            <ul>
                <li>(B, Q_N, Q_S)</li>
                <li>(Q_N, Q_T)</li>
            </ul>      
        </td>
        <td>✓</td>
    </tr>
</tbody>
</table>

**说明**
-   attn_out：Tensor类型，公式中的输出，数据类型支持float16、bfloat16。数据格式支持ND。限制：该输出参数的D维度与value的D保持一致，其余维度需要与入参query的shape保持一致。
-   softmax_lse：Tensor类型，ring attention算法对query乘key的结果，先取max得到softmax_max。query乘key的结果减去softmax_max，再取exp，最后取sum，得到softmax_sum，最后对softmax_sum取log，再加上softmax_max得到的结果。数据类型支持float32，return_softmax_lse为1时，一般情况下，输出shape为(B, Q_N, Q_S)的Tensor，当input_q为TND时，输出shape为(Q_N, Q_T)的Tensor；return_softmax_lse为0时，则输出shape为[1]的值为0的Tensor。

## 约束说明<a name="zh-cn_topic_0000002168254826_section12345537164214"></a>

- 声明
  - 参数cu_seqlens_q、cu_seqlens_kv、seqused_q、seqused_kv、block_table及attn_mask属于tensor。由于算子在Tiling阶段无法获取tensor的具体数值，tiling侧不对值进行校验，正确性需要用户自行保证。若上述参数传入非法值，会触发未定义行为（精度问题、非法内存访问导致的程序崩溃等）。

### 特性参数组

|      特性参数组      |     参数字段名称     |    字段分组    |  字段类型  |
| :-------------------: | :-------------------: | :-------------: | :--------: |
|      公共参数组      |         q         |      INPUT      |   Tensor   |
|                      |          k          |      INPUT      | Tensor |
|                      |         v         |      INPUT      | Tensor |
|                      |      softmax_scale      | ATTR(OPTIONAL) |   double   |
|                      |      layout_q      | ATTR(OPTIONAL) |   string   |
|                      |      layout_kv      | ATTR(OPTIONAL) |   string   |
|                      |      layout_out      | ATTR(OPTIONAL) |   string   |
|                      |     attn_out     |     OUTPUT     |   Tensor   |
|      Mask参数组      |       mask_mode       | ATTR(OPTIONAL) |   int   |
|                      |       win_left       | ATTR(OPTIONAL) |   int   |
|                      |      win_right      | ATTR(OPTIONAL) |   int   |
|                      |      attn_mask      | INPUT(OPTIONAL) |   Tensor   |
| SeqLens参数组  |   cu_seqlens_q   | INPUT(OPTIONAL) |  Tensor  |
|                      |  cu_seqlens_kv  | INPUT(OPTIONAL) |  Tensor  |
|                      |  seqused_q  | INPUT(OPTIONAL) |  Tensor  |
|                      |  seqused_kv  | INPUT(OPTIONAL) |  Tensor  |
|                      |  max_seqlen_q  | ATTR(OPTIONAL) |  int  |
|                      |  max_seqlen_kv  | ATTR(OPTIONAL) |  int  |
| Paged Attention参数组 |      block_table      | INPUT(OPTIONAL) |   Tensor   |
|  Sinks参数组  |     sinks     | INPUT(OPTIONAL) |   Tensor   |
|   SoftmaxLSE参数组   |    return_softmax_lse    | ATTR(OPTIONAL) |    int    |
|                      |      softmax_lse      |     OUTPUT     |   Tensor   |

### 基准信息说明

资料约束中，常见字段释义如下：

|    命名    |                            含义                            |
| :---------: | :---------------------------------------------------------: |
|      B      |                Batch,表示输入样本批量大小                |
|     Q_N     |        输入q tensor的头数，对应q shape中的N        |
|    KV_N    |    输入k/v tensor的头数，对应k/v shape中的N    |
|     Q_S     |      输入q tensor的序列长度，对应q shape中的S      |
|    KV_S    |  输入k/v tensor的序列长度，对应k/v shape中的S  |
|     Q_T     |          输入q tensor所有batch序列长度的累加和          |
|     KV_T     |          输入k/v tensor所有batch序列长度的累加和          |
|     D     |          输入q/k/v tensor以及输出attn_out隐藏层最小的单元尺寸headdim         |

### 参数组约束

#### 公共参数组
- 入参为空的场景处理：
    - 空Tensor指必选输入和输出的shape size为0,即有任意轴为0。
    - 触发空tensor的用例将全部拦截报错。
- q、k、v、attn_out校验
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
                <li>BNSD -> (B, N_Q, S_Q, D)</li>
                <li>BSND -> (B, S_Q, N_Q, D)</li>
                <li>TND -> (T_Q, N_Q, D)</li>
            </ul>
        </td>
        <td rowspan="4">
            必须存在
        </td>
        <td rowspan="4">
            <ul>
                <li>q、k、v、attn_out的数据类型需相同</li>
                <li>k、v的shape需相同</li>
                <li>Layout校验规则见layout匹配关系表</li>
            </ul>
        </td>
        <td rowspan="4">
            轴校验：
            <ul>
                <li>65536 > B > 0</li>
                <li>Q_T > 0</li>
                <li>KV_T > 0</li>
                <li>Q_N > 0</li>
                <li>KV_N > 0</li>
                <li>Q_S > 0</li>
                <li>KV_S > 0</li>
                <li>D = 128</li>
                <li>Q_N % KV_N == 0且Q_N / KV_N > 0</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>k</td>
        <td rowspan="2">
            <ul>
                <li>tensor_type支持BFLOAT16和FLOAT16</li>
                <li>BNSD -> (B, N_KV, S_KV, D)</li>
                <li>BSND -> (B, S_KV, N_KV, D)</li>
                <li>TND -> (T_KV, N_KV, D)</li>
                <li>PA_BBND -> (num_blocks, block_size, KV_N, D)</li>
                <li>PA_BNBD -> (num_blocks, KV_N, block_size, D)</li>
                <li>1024 >= block_size >= 16，block_size % 16 == 0</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>v</td>
    </tr>
    <tr>
        <td>attn_out</td>
        <td>
            <ul>
                <li>data_type支持BFLOAT16和FLOAT16</li>
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
    <th>layout_kv</th>
    <th>layout_out</th>
    <th>layout_softmax_lse</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>BNSD</td>
        <td>
          <li>BNSD</li>
          <li>PA_BBND</li>
          <li>PA_BNBD</li>
        </td>
        <td>
          <li>BNSD</li>
          <li>BSND</li>
        </td>
        <td>(B, Q_N, Q_S)</td>
    </tr>
    <tr>
        <td>BSND</td>
        <td>
          <li>BSND</li>
          <li>PA_BBND</li>
          <li>PA_BNBD</li>
        </td>
        <td>BSND</td>
        <td>(B, Q_N, Q_S)</td>
    </tr>
    <tr>
        <td>TND</td>
        <td>
          <li>TND</li>
          <li>PA_BBND</li>
          <li>PA_BNBD</li>
        </td>
        <td>TND</td>
        <td>(Q_N, Q_T)</td>
    </tr>
</tbody>
</table>

#### Mask参数组

mask_mode参数解释
<ul>
    <li>mask_mode=0，全计算模式（默认值）</li>
    <li>mask_mode=3，Causal模式</li>
</ul>

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
        <td rowspan="4">
            <ul>
                <li>当mask_mode为0时，win_left和win_right必须为-1以及不支持传入attn_mask</li>
                <li>当mask_mode为3时，win_left和win_right必须为-1且必须传入attn_mask矩阵</li>
            </ul>
        </td>
        <td rowspan="4">
            <ul>
                <li>无</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>attn_mask</td>
        <td>
            <ul>
                <li>tensor_type支持INT8</li>
                <li>tensor_shape维度为(2048, 2048)，该矩阵固定为2048*2048的下三角矩阵对角线全0</li>
            </ul>
        </td>
        <td rowspan="3">
            无
        </td>
    </tr>
    <tr>
        <td>win_left</td>
        <td rowspan="2">
            <ul>
                <li>data_type支持INT</li>
                <li>不传或者传入-1,代表该值为正无穷</li>
                <li>值需要 >= -1</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>win_right</td>
    </tr>
</tbody>
</table>

#### SeqLengths参数组

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
        <td>seqused_q</td>
        <td rowspan="2">
            <ul>
                <li>tensor_type支持INT32</li>
                <li>tensor_shape为(B,)</li>
                <li>仅支持非负整数</li>
                <li>seqused_q中的值需小于等于Q_S</li>
                <li>seqused_kv中的值需小于等于KV_S</li>
            </ul>
        </td>
        <td rowspan="6">
            可选参数
        </td>
        <td rowspan="6">
            无
        </td>
        <td rowspan="2">
            无
        </td>
    </tr>
    <tr>
        <td>seqused_kv</td>
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
        <td>cu_seqlens_kv</td>
        <td>
            <ul>
                <li>tensor_type支持INT32</li>
                <li>tensor_shape为(B+1,)</li>
                <li>值仅支持非负整数</li>
                <li>其值应非递减（大于等于前一个值）排列，第一个元素为0且最后一个元素等于KV_T</li>
            </ul>
        </td>
        <td>
            <ul>
                <li>当layout_kv为TND时，必须传入</li>
                <li>当layout_kv不为TND时，不支持传入</li>
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
        <td>max_seqlen_kv</td>
    </tr>
</tbody>
</table>

#### Paged Attention参数组
当block_table不为空时，开启Paged Attention
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
        <td>block_table</td>
        <td>
            <ul>
                <li>tensor_type仅支持INT32</li>
                <li>tensor_shape为(B, max_num_blocks_per_seq)</li>
                <li>值只能为正整数</li>
            </ul>
        </td>
        <td>
            可选参数
        </td>
        <td>
            无
        </td>
        <td>
            <ul>
                <li>PagedAttention开启情况下，必须传入seqused_kv</li>
                <li>PagedAttention开启情况下，block_table必须不为空</li>
            </ul>
        </td>
    </tr>
</tbody>
</table>

#### Sinks参数组

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
        <td>sinks</td>
        <td>
            <ul>
                <li>暂不支持</li>
            </ul>
        </td>
        <td>
            可选参数
        </td>
        <td>
            无
        </td>
        <td>
            暂不支持
        </td>
    </tr>
</tbody>
</table>

#### SoftmaxLSE参数组

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
        <td>return_softmax_lse</td>
        <td>
            <ul>
                <li>data_type仅支持INT</li>
                <li>值仅支持0和1，1代表开启softmax_lse，0代表关闭softmax_lse</li>
            </ul>
        </td>
        <td>
            可选属性，默认值为0
        </td>
        <td rowspan="2">
            <ul>
                <li>当return_softmax_lse为0时，输出shape为[1]的值为0的Tensor</li>
                <li>当return_softmax_lse为1时，输出shape见layout匹配关系表</li>
            </ul>
        </td>
        <td>
            无
        </td>
    </tr>
    <tr>
        <td>softmax_lse</td>
        <td>
            <ul>
                <li>data_type仅支持FLOAT32</li>
            </ul>
        </td>
        <td>
            无
        </td>
        <td>
            无
        </td>
    </tr>
</tbody>
</table>

## 调用示例<a name="zh-cn_topic_0000002168254826_section14459801435"></a>

-   单算子模式调用

    ```python
    import os
    import torch
    import random
    import torch_npu
    import numpy as np

    ```

-   调用

    ```python
    import os
    import torch
    import random
    import torch_npu
    import torchair
    import numpy as np

    ```