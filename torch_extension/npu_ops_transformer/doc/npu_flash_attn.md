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
-   q、k、v数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小batch_size、S（Seq-Length）表示输入样本序列长度、H（Hidden-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸headdim，且满足D=H/N、T_Q表示所有query Batch输入样本序列长度的累加和，T_KV表示所有k、v Batch输入样本序列长度的累加和。
-   S_Q表示seqlen_q，N_Q表示nheads_q，S_KV表示seqlen_kv，N_KV表示nheads_kv。

## 函数原型<a name="zh-cn_topic_0000002168254826_section45077510411"></a>

```
npu_flash_attn(q, k, v, block_table=None, cu_seqlens_q=None, cu_seqlens_kv=None, seqused_q=None, seqused_kv=None, sinks=None, metadata=None, softmax_scale=1.0, mask_mode=1, attn_mask=None, win_left=-1, win_right=-1, max_seqlen_q=-1, max_seqlen_kv=-1, layout_q="BSND", layout_kv="BSND", layout_out="BSND", return_softmax_lse=0, deterministic=0) -> (Tensor, Tensor)
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
                <li>(B, S_Q, N_Q, D)</li>
                <li>(B, N_Q, S_Q, D)</li>
                <li>(T_Q, N_Q, D)</li>
            </ul>      
        </td>
        <td>×</td>
    </tr>
    <tr>
        <td>k</td>
        <td>必选输入</td>
        <td>公式中的k</td>
        <td>BFLOAT16/FLOAT16</td>
        <td>ND</td>
        <td>
            <ul>
                <li>(B, S_KV, N_KV, D)</li>
                <li>(B, N_KV, S_KV, D)</li>
                <li>(T_KV, N_KV, D)</li>
                <li>(num_blocks, block_size, N_KV, D)</li>
                <li>(num_blocks, N_KV, block_size, D)</li>
                <li>(num_blocks, N_KV, D/16, block_size, 16)</li>
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
                <li>(B, S_KV, N_KV, D)</li>
                <li>(B, N_KV, S_KV, D)</li>
                <li>(T_KV, N_KV, D)</li>
                <li>(num_blocks, block_size, N_KV, D)</li>
                <li>(num_blocks, N_KV, block_size, D)</li>
                <li>(num_blocks, N_KV, D/16, block_size, 16)</li>
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
        <td>×</td>
    </tr>
    <tr>
        <td>cu_seqlens_q</td>
        <td>可选输入</td>
        <td>累积序列长度,用于处理变长序列</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(B+1,)</td>
        <td>×</td>
    </tr>
    <tr>
        <td>cu_seqlens_kv</td>
        <td>可选输入</td>
        <td>累积序列长度,用于处理变长序列</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(B+1,)</td>
        <td>×</td>
    </tr>
    <tr>
        <td>seqused_q</td>
        <td>可选输入</td>
        <td>指定每batch中实际使用的序列长度，截断冗余运算</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(B, )</td>
        <td>×</td>
    </tr>
    <tr>
        <td>seqused_kv</td>
        <td>可选输入</td>
        <td>指定每batch中实际使用的序列长度，截断冗余运算</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(B, )</td>
        <td>×</td>
    </tr>
    <tr>
        <td>sinks</td>
        <td>可选输入</td>
        <td>提供解决注意力计算中集中前置一部分token问题的功能</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>(N_Q, )</td>
        <td>×</td>
    </tr>
    <tr>
        <td>metadata</td>
        <td>可选输入</td>
        <td>metadata</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(max_schedule_size, )</td>
        <td>×</td>
    </tr>
    <tr>
        <td>softmax_scale</td>
        <td>可选输入</td>
        <td>可显式设置缩放因子，覆盖默认计算</td>
        <td>FLOAT32</td>
        <td>N/A</td>
        <td>-</td>
        <td>×</td>
    </tr>
    <tr>
        <td>mask_mode</td>
        <td>可选输入</td>
        <td>掩码模式</td>
        <td>INT</td>
        <td>N/A</td>
        <td>-</td>
        <td>×</td>
    </tr>
    <tr>
        <td>attn_mask</td>
        <td>可选输入</td>
        <td>掩码矩阵</td>
        <td>INT8</td>
        <td>N/A</td>
        <td>(2048, 2048)</td>
        <td>×</td>
    </tr>
    <tr>
        <td>win_left</td>
        <td>可选输入</td>
        <td>window左界限</td>
        <td>INT</td>
        <td>-</td>
        <td>-</td>
        <td>×</td>
    </tr>
    <tr>
        <td>win_right</td>
        <td>可选输入</td>
        <td>window右界限</td>
        <td>INT</td>
        <td>-</td>
        <td>-</td>
        <td>×</td>
    </tr>
    <tr>
        <td>max_seqlen_q</td>
        <td>可选输入</td>
        <td>指定查询q序列的长度上限</td>
        <td>INT</td>
        <td>-</td>
        <td>-</td>
        <td>×</td>
    </tr>
    <tr>
        <td>max_seqlen_kv</td>
        <td>可选输入</td>
        <td>指定键k和值v序列的长度上限</td>
        <td>INT</td>
        <td>-</td>
        <td>-</td>
        <td>×</td>
    </tr>
    <tr>
        <td>layout_q</td>
        <td>可选输入</td>
        <td>定义输入q张量的布局格式</td>
        <td>STRING</td>
        <td>N/A</td>
        <td>-</td>
        <td>×</td>
    </tr>
    <tr>
        <td>layout_kv</td>
        <td>可选输入</td>
        <td>定义输入k和v张量的布局格式</td>
        <td>STRING</td>
        <td>N/A</td>
        <td>-</td>
        <td>×</td>
    </tr>
    <tr>
        <td>layout_out</td>
        <td>可选输入</td>
        <td>定义输出张量的布局格式</td>
        <td>STRING</td>
        <td>N/A</td>
        <td>-</td>
        <td>×</td>
    </tr>
    <tr>
        <td>return_softmax_lse</td>
        <td>可选输入</td>
        <td>是否需要获取softmax的LSE结果</td>
        <td>INT</td>
        <td>N/A</td>
        <td>-</td>
        <td>×</td>
    </tr>
    <tr>
        <td>deterministic</td>
        <td>可选输入</td>
        <td>是否开启确定性计算</td>
        <td>INT</td>
        <td>N/A</td>
        <td>-</td>
        <td>×</td>
    </tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000002168254826_section22231435517"></a>

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
                <li>(B, S_Q, N_Q, D)</li>
                <li>(B, N_Q, S_Q, D)</li>
                <li>(T_Q, N_Q, D)</li>
            </ul>      
        </td>
        <td>×</td>
    </tr>
    <tr>
        <td>softmax_lse</td>
        <td>可选输出</td>
        <td>softmax的LSE结果</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>
            <ul>
                <li>(B, N_Q, S_Q)</li>
                <li>(N_Q, T_Q)</li>
            </ul>      
        </td>
        <td>×</td>
    </tr>
</tbody>
</table>

**说明**
-   attn_out：Tensor类型，公式中的输出，数据类型支持float16、bfloat16。数据格式支持ND。限制：该输出参数的D维度与value的D保持一致，其余维度需要与入参query的shape保持一致。
-   softmax_lse：Tensor类型，ring attention算法对query乘key的结果，先取max得到softmax_max。query乘key的结果减去softmax_max，再取exp，最后取sum，得到softmax_sum，最后对softmax_sum取log，再加上softmax_max得到的结果。数据类型支持float32，return_softmax_lse 为1时，一般情况下，输出shape为(B, N_Q, S_Q)的Tensor，当input_q为TND时，输出shape为(N_Q, T_Q)的Tensor；return_softmax_lse为0时，则输出shape为[1]的值为0的Tensor。

layout匹配关系如下：
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
            <ul>
                <li>BNSD</li>
                <li>PA_BBND</li>
                <li>PA_BNBD</li>
                <li>PA_Nz</li>
            </ul>
        </td>
        <td>
            <ul>
                <li>BNSD</li>
                <li>BSND</li>
            </ul>
        </td>
        <td>(B, N_Q, S_Q)</td>
    </tr>
    <tr>
        <td>BSND</td>
        <td>
            <ul>
                <li>BSND</li>
                <li>PA_BBND</li>
                <li>PA_BNBD</li>
                <li>PA_Nz</li>
            </ul>
        </td>
        <td>BSND</td>
        <td>(B, N_Q, S_Q)</td>
    </tr>
    <tr>
        <td>TND</td>
        <td>
            <ul>
                <li>TND</li>
                <li>PA_BBND</li>
                <li>PA_BNBD</li>
                <li>PA_Nz</li>
            </ul>
        </td>
        <td>TND</td>
        <td>(N_Q, T_Q)</td>
    </tr>
</tbody>
</table>

## 约束说明<a name="zh-cn_topic_0000002168254826_section12345537164214"></a>

npu_flash_attn算子约束分为4个档位，按约束复杂程度递增分为 单参数约束、存在性约束、一致性约束和特性交叉约束，各档位约束内容和示例如下：

- 单参数约束：对于单个接口参数的约束，包含npu_flash_attn算子接口中的 Tensor 和 Attributes
  - 对于Tensor单参数约束中包含如下校验
    - 校验 shape，包括shape维度dim、每一维度 dim value
    - 校验 dtype
    - 校验 format
  - 对于属性 Attribute
    - 校验属性取值
- 存在性约束：约束特定场景下，特性参数组内，必须传入某参数，或不支持传入某参数
- 一致性约束：特性参数组内，各个参数间约束。
  - Example 1：属性mask_mode和输入tensor attn_mask均属于Attention Mask参数组，mask_mode取值为3/4时，attn_mask shape必须为（2048, 2048），此类约束即为参数组内的一致性约束
- 特性交叉约束：涉及多个参数组，不同参数组间交叉约束

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
|                      |     deterministic     | ATTR(OPTIONAL) |   int64   |
|                      |     attn_out     |     OUTPUT     |   Tensor   |
|      Mask参数组      |       mask_mode       | INPUT(OPTIONAL) |   int   |
|                      |       win_left       | ATTR(OPTIONAL) |   int   |
|                      |      win_right      | ATTR(OPTIONAL) |   int   |
|                      |      attn_mask      | ATTR(OPTIONAL) |   Tensor   |
| SeqLens参数组  |   cu_seqlens_q   | INPUT(OPTIONAL) |  Tensor  |
|                      |  cu_seqlens_kv  | INPUT(OPTIONAL) |  Tensor  |
|                      |  seqused_q  | INPUT(OPTIONAL) |  Tensor  |
|                      |  seqused_kv  | INPUT(OPTIONAL) |  Tensor  |
|                      |  max_seqlen_q  | INPUT(OPTIONAL) |  int  |
|                      |  max_seqlen_kv  | INPUT(OPTIONAL) |  int  |
| Paged Attention参数组 |      block_table      | INPUT(OPTIONAL) |   Tensor   |
|  Sinks参数组  |     sinks     | INPUT(OPTIONAL) |   Tensor   |
|   SoftmaxLSE参数组   |    return_softmax_lse    | ATTR(OPTIONAL) |    int    |
|                      |      softmax_lse      |     OUTPUT     |   Tensor   |

### 基准信息说明

资料约束中，常见字段释义如下：

|    命名    |                            含义                            |
| :---------: | :---------------------------------------------------------: |
|      B      |                Batch, 表示输入样本批量大小                |
|     N_Q     |        输入q tensor的头数，对应q shape中的N        |
|    N_KV    |    输入k/v tensor的头数，对应k/v shape中的N    |
|     S_Q     |      输入q tensor的序列长度，对应q shape中的S      |
|    S_KV    |  输入k/v tensor的序列长度，对应k/v shape中的S  |
|     T_Q     |          输入q tensor所有batch序列长度的累加和          |
|     T_KV     |          输入k/v tensor所有batch序列长度的累加和          |
|      G     |        Group，表示查询头的分组数，对应Q_N与KV_N的比值         |

### 参数组约束

#### 公共参数组
- 确定性计算：
  - npu_flash_attn通过可选参数deterministic控制是否开启确定性计算，1代表开启确定性计算，0代表关闭确定性计算。
- 入参为空的场景处理：
    - 空Tensor指必选输入和输出的shape size为0, 即有任意轴为0。
    - 触发空tensor的用例将全部拦截报错。
- 单参数约束
  - 无
- 存在性约束
  - 无
- 一致性约束

    <table style="undefined;table-layout: fixed; width: 800px">
        <colgroup>
            <col style="width: 200px">
            <col style="width: 200px">
            <col style="width: 200px">
            <col style="width: 200px">
            <col style="width: 200px">
            <col style="width: 200px">
        </colgroup>
        <thead>
            <tr>
                <th>tensor</th>
                <th>B</th>
                <th>N</th>
                <th>S</th>
                <th>D</th>
                <th>N_Q/N_KV</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>q</td>
                <td>
                    B轴支持大于0且小于等于65536
                </td>
                <td>
                    大于0
                </td>
                <td>
                    大于0
                </td>
                <td>
                    D轴小于等于512，16对齐，且大于0
                </td>
                <td rowspan="3">
                    N_Q是N_KV的整数倍
                </td>
            </tr>
            <tr>
                <td>k</td>
                <td>
                    B轴支持大于0且小于等于65536
                </td>
                <td>
                    大于0
                </td>
                <td>
                    大于0
                </td>
                <td>
                    D轴小于等于512，16对齐，且大于0
                </td>
            </tr>
            <tr>
                <td>v</td>
                <td>
                    B轴支持大于0且小于等于65536
                </td>
                <td>
                    大于0
                </td>
                <td>
                    大于0
                </td>
                <td>
                    D轴小于等于512，16对齐，且大于0
                </td>
            </tr>
        </tbody>
    </table>
- 特性交叉约束
    <table style="undefined;table-layout: fixed; width: 1000px">
        <colgroup>
            <col style="width: 200px">
            <col style="width: 400px">
            <col style="width: 400px">
        </colgroup>
        <thead>
            <tr>
                <th>场景</th>
                <th>PagedAttention场景</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>npu_flash_attn</td>
                <td>
                    <ul>
                        <li>k/v排布支持PA_BBND（blockNum, blockSize, N_KV, D）、PA_BNBD（blockNum, N_KV,
                            blockSize, D）和PA_Nz中（blockNum，N_KV，D/16，blockSize，16）三种格式</li>
                    </ul>
                </td>
            </tr>
        </tbody>
    </table>
#### Mask参数组

- 单参数约束
  - 公共
    - 入参 mask_mode 需要满足以下条件:

      - mask_mode 支持输入范围为 1、3、4，默认值为 1
      - mask_mode 含义如下表所示

      | mask_mode |                                                                               含义                                                                                     |                                             备注                                             |
      | :--------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------: |
      |     1     |                                                全计算模式，忽略win_left和win_right                                               |                  不需要传入attn_mask                                            |
      |     3     |                                           Causal模式，忽略win_left和win_right                                            | 必须传入atte_mask矩阵，该矩阵固定为2048*2048的下三角矩阵，<br />以右下顶点为参数起点划分，对角线全0 |
      |     4     |                                           band模式，计算win_left和win_right之间的部分                                            | 必须传入atte_mask矩阵，该矩阵固定为2048*2048的下三角矩阵，<br />以右下顶点为参数起点划分，对角线全0 |
    - 入参 win_left和win_right应满足以下条件:
        - 只在mask_mode为4情况下生效
        - 如传入-1则代表win_left和win_right的值为正无穷
- 存在性约束
  - 无
- 一致性约束
  - mask_mode为3或4时，必须传入atte_mask矩阵。
- 特性交叉约束
  - 无
#### SeqLengths参数组

- 单参数约束
  - 公共
    - 入参 seqused_q应满足以下条件:
      - 长度应满足以下条件:
        - 长度应等于 batch 数
      - 入参中的数值应满足以下条件:
        - 其值应为非负整数
    - 入参 seqused_kv应满足以下条件:
      - 长度应满足以下条件:
        - 长度应等于 batch 数
      - 入参中的数值应满足以下条件:
        - 其值应为非负整数
    - 入参 cu_seqlens_q应满足以下条件:
      - 长度应满足以下条件:
        - 长度应等于 batch 数 + 1
      - 入参中的数值应满足以下条件:
        - 当 layout_q 为 TND 时，其值应非递减（大于等于前一个值）排列
        - 当 layout_q 为 TND 时，第一个元素为0且最后一个元素应等于 T_Q
        - 其值应为非负整数
    - 入参 cu_seqlens_kv应满足以下条件:
      - 长度应满足以下条件:
        - 长度应等于 batch 数 + 1
      - 入参中的数值应满足以下条件:
        - 当 layout_kv 为 TND 时，其值应非递减（大于等于前一个值）排列
        - 当 layout_kv 为 TND 时，第一个元素为0且最后一个元素应等于 T_KV
        - 其值应为非负整数
- 存在性约束
  - 公共
    - 入参 seqused_kv 应满足以下条件:
      - PagedAttention场景，必须传入 seqused_kv
    - 入参 cu_seqlens_q 应满足以下条件:
      - 当 layout_q 为 TND 时，必须传入 cu_seqlens_q
    - 入参 cu_seqlens_kv 应满足以下条件:
      - 当 layout_kv 为 TND 时，必须传入 cu_seqlens_kv
- 一致性约束
  - 无
- 特性交叉约束
  - 无

#### Paged Attention参数组

- 单参数约束
  - 公共
    - 入参 blockTable 需要满足以下条件：
      - tensor dtype 为int32类型
      - tensor shape 为2维，每一维dim value取值均不能为0，第一维长度需等于Batch size，第二维长度不能小于max_num_blocks_per_seq（max_num_blocks_per_seq为每个batch中最大seqused_kv对应的block数量）
- 存在性约束
  - 公共
    - PagedAttention 使能情况下，必须传入 seqused_kv
- 一致性约束
  - 无
- 特性交叉约束
  - 无

#### Sinks参数组

- 单参数约束
  - 公共
    - 入参 sinks 需要满足以下条件
      - tensor dtype 为 FLOAT32
      - tensor shape 为 (N_Q,)
- 存在性约束
  - 无
- 一致性约束
  - 无
- 特性交叉约束
  - 无
#### SoftmaxLSE参数组

- 单参数约束
  - 公共
    - 输出 softmax_lse 仅支持数据类型 FLOAT32
- 存在性约束
  - 公共
    - softmaxLSE 场景下，输出 softmax_lse 不应为空
- 一致性约束
  - 无
- 特性交叉约束
  - 公共
    - softmaxLSE 且非空tensor场景下，如输出 layout 为 TND则 softmax_lse 输出维度应为 2，且 shape 匹配 [N_Q, T_Q]
    - softmaxLSE 且非空tensor场景下，如输出 layout 不为 TND则 softmax_lse 输出维度应为 3，且 shape 匹配 [B, N_Q, S_Q]
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
