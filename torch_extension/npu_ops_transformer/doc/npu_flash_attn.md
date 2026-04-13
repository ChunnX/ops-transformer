# npu\_flash_attn<a name="ZH-CN_TOPIC_0000002309174912"></a>

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

    支持flashattention训练推理归一化
-   计算公式:
    
    self-attention（自注意力）利用输入样本自身的关系构建了一种注意力模型。其原理是假设有一个长度为$n$的输入样本序列$x$，$x$的每个元素都是一个$d$维向量，可以将每个$d$维向量看作一个token embedding，将这样一条序列经过3个权重矩阵变换得到3个维度为$n*d$的矩阵。

    self-attention的计算公式一般定义如下，其中$Q、K、V$为输入样本的重要属性元素，是输入样本经过空间变换得到，且可以统一到一个特征空间中。公式及算子名称中的"Attention"为"self-attention"的简写。

    $$
    Attention(Q,K,V)=Score(Q,K)V
    $$

    本算子中Score函数采用Softmax函数，self-attention计算公式为:

    $$
    Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
    $$

    其中$Q$和$K^T$的乘积代表输入$x$的注意力，为避免该值变得过大，通常除以$d$的开根号进行缩放，并对每行进行softmax归一化，与$V$相乘后得到一个$n*d$的矩阵。

    增加**sink**之后计算逻辑见下，主要修改相关softmax_max和softmax_sum逻辑计算部分
    
    $$
    S = \frac{QK^T}{\sqrt{d}}
    $$
    
    $$
    m = max(sink, max(S))
    $$
    
    $$
    Attention = \frac{e^{S - m} * V}{\sum e^{S-m} + e^{sink - m}}
    $$

    开启**return_softmax_lse**之后，返回值softmax_lse计算逻辑见下：

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
-   query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小batch_size、S（Seq-Length）表示输入样本序列长度、H（Hidden-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸headdim，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。Q表示total_q，KV表示total_kv。
-   S_Q表示seqlen_q，N_Q表示nheads_q，S_KV表示seqlen_kv，N_KV表示nheads_kv。

## 函数原型<a name="zh-cn_topic_0000002168254826_section45077510411"></a>

```
npu_flash_attn(q, k, v, block_table=None, cu_seqlens_q=None, cu_seqlens_kv=None, seqused_q=None, seqused_kv=None, sinks=None, metadata=None, softmax_scale=1.0, mask_mode=1, win_left=-1, win_right=-1, max_seqlen_q=-1, max_seqlen_kv=-1, layout_q="BSND", layout_kv="BSND", layout_out="BSND", return_softmax_lse=0, deterministic=0) -> (Tensor, Tensor)
```

## 参数说明<a name="zh-cn_topic_0000002168254826_section112637109429"></a>

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
</thread>
<tbody>
    <tr>
        <td>q</td>
        <td>必选输入</td>
        <td>公式中的q</td>
        <td>BF16/FP16</td>
        <td>ND</td>
        <td>
            <ul>
                <li>(B, S_Q, N_Q, D)</li>
                <li>(B, N_Q, S_Q, D)</li>
                <li>(Q, N_Q, D)</li>
            </ul>      
        </td>
        <td>√</td>
    </tr>
    <tr>
        <td>k</td>
        <td>必选输入</td>
        <td>公式中的k</td>
        <td>BF16/FP16</td>
        <td>ND</td>
        <td>
            <ul>
                <li>(B, S_KV, N_KV, D)</li>
                <li>(B, N_KV, S_KV, D)</li>
                <li>(KV, N_KV, D)</li>
                <li>(number_blocks, B, N_KV, D)</li>
                <li>(num_blocks, N_KV, block_size, D)</li>
                <li>(num_blocks, N_KV, D/16, block_size, 16)</li>
            </ul>
        </td>
        <td>√</td>
    </tr>
    <tr>
        <td>v</td>
        <td>必选输入</td>
        <td>公式中的v</td>
        <td>BF16/FP16</td>
        <td>ND</td>
        <td>
            <ul>
                <li>(B, S_KV, N_KV, D)</li>
                <li>(B, N_KV, S_KV, D)</li>
                <li>(KV, N_KV, D)</li>
                <li>(number_blocks, B, N_KV, D)</li>
                <li>(num_blocks, N_KV, block_size, D)</li>
                <li>(num_blocks, N_KV, D/16, block_size, 16)</li>
            </ul>    
        </td>
        <td>√</td>
    </tr>
    <tr>
        <td>block_table</td>
        <td>可选输入</td>
        <td>用于分块注意力计算中的块索引映射</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(B, max_num_blocks_per_seq)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>cu_seqlens_q</td>
        <td>可选输入</td>
        <td>累积序列长度,用于处理变长序列</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(B + 1, )</td>
        <td>√</td>
    </tr>
    <tr>
        <td>cu_seqlens_kv</td>
        <td>可选输入</td>
        <td>累积序列长度,用于处理变长序列</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(B + 1, )</td>
        <td>√</td>
    </tr>
    <tr>
        <td>seqused_q</td>
        <td>可选输入</td>
        <td>指定每batch中实际使用的序列长度，截断冗余运算</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(B, )</td>
        <td>√</td>
    </tr>
    <tr>
        <td>seqused_kv</td>
        <td>可选输入</td>
        <td>指定每batch中实际使用的序列长度，截断冗余运算</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(B, )</td>
        <td>√</td>
    </tr>
    <tr>
        <td>sinks</td>
        <td>可选输入</td>
        <td>提供解决注意力计算中集中前置一部分token问题的功能</td>
        <td>FP32</td>
        <td>ND</td>
        <td>(N_Q, )</td>
        <td>√</td>
    </tr>
    <tr>
        <td>metadata</td>
        <td>可选输入</td>
        <td>metadata</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(max_schedule_size, )</td>
        <td>√</td>
    </tr>
    <tr>
        <td>softmax_scale</td>
        <td>可选输入</td>
        <td>可显式设置缩放因子，覆盖默认计算</td>
        <td>FLOAT</td>
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
        <td>指定查询q序列的最大长度上限</td>
        <td>INT</td>
        <td>-</td>
        <td>-</td>
        <td>×</td>
    </tr>
    <tr>
        <td>max_seqlen_kv</td>
        <td>可选输入</td>
        <td>指定键k和值v序列的最大长度上限</td>
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
</thread>
<tbody>
    <tr>
        <td>attn_out</td>
        <td>固定输出</td>
        <td>flash_attn的计算输出</td>
        <td>BF16/FP16</td>
        <td>ND</td>
        <td>
            <ul>
                <li>(B, S_Q, N_Q, D)</li>
                <li>(B, N_Q, S_Q, D)</li>
                <li>(Q, N_Q, D)</li>
            </ul>      
        </td>
        <td>√</td>
    </tr>
    <tr>
        <td>softmax_lse</td>
        <td>可选输出</td>
        <td>softmax的LSE结果</td>
        <td>FP32</td>
        <td>ND</td>
        <td>
            <ul>
                <li>(B, N_Q, S_Q)</li>
                <li>(N_Q, T_Q)</li>
            </ul>      
        </td>
        <td>√</td>
    </tr>
</tbody>
</table>

**说明**
-   attention_out：Tensor类型，公式中的输出，数据类型支持float16、bfloat16。数据格式支持ND。限制：该入参的D维度与value的D保持一致，其余维度需要与入参query的shape保持一致。
-   softmaxLse：Tensor类型，ring attention算法对query乘key的结果，先取max得到softmax_max。query乘key的结果减去softmax_max，再取exp，最后取sum，得到softmax_sum，最后对softmax_sum取log，再加上softmax_max得到的结果。数据类型支持float32，return_softmax_lse 为1时，一般情况下，输出shape为(B, N, Q_S)的Tensor，当input_layout为TND/NTD_TND时，输出shape为(T_Q,N_Q)的Tensor；softmax_lse_flag为False时，则输出shape为[1]的值为0的Tensor。

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
    <th>layout_softmax_lse_layout</th>
</tr>
</thread>
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

-   该接口支持训练正向和推理场景下使用。

- 确定性计算：
  - npu_flash_attn通过可选参数deterministic控制是否开启确定性计算，1代表开启确定性计算，0代表关闭确定性计算。
#### 公共参数组
    - 入参为空的场景处理：
        - 空Tensor指必选输入和输出的shapeSize为0。在空Tensor场景下，若attentionOut为空，返回空，否则返回全0；若有lse且lse为空时返回空，lse不为空则返回全inf。非空Tensor时输入正常拦截。
        - query，attentionOut所有tensor的shapeSize为0，属于空Tensor。
        - query，attentionOut所有tensor的shapeSize不为0，若有lse且lse不为空，并且key，value中所有tensor的shapeSize为0，属于空Tensor。
        - attentionOut和lse都为空时，属于空Tensor。
        - 属于空Tensor时，跳过校验流程；否则，走正常校验流程。
- 单参数约束
  - 无
- 存在性约束

  <table style="undefined;table-layout: fixed; width: 700px">
        <colgroup>
            <col style="width: 400px">
            <col style="width: 300px">
        </colgroup>
        <thead>
            <tr>
                <th>场景</th>
                <th>结果预期</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>query,key,value,attentionOut存在nullptr</td>
                <td>正常拦截</td>
            </tr>
            <tr>
                <td>query,attentionOut的tensor的shapeSize为0</td>
                <td>attentionOut为返回空tensor</td>
            </tr>
            <tr>
                <td>query,attentionOut的tensor的shapeSize不为0,且key,value的tensor的shapeSize为0</td>
                <td>attentionOut返回全0</td>
            </tr>
        </tbody>
    </table>
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
                <th>N_Q/KV_N</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>q</td>
                <td>
                    支持B轴小于等于65536
                </td>
                <td>
                    N轴无限制
                </td>
                <td>
                    S轴无限制
                </td>
                <td>
                    D轴小于等于512, 且16对齐
                </td>
                <td rowspan="3">
                    N_Q是N_KV的整数倍
                </td>
            </tr>
            <tr>
                <td>k</td>
                <td>
                    支持B轴小于等于65536
                </td>
                <td>
                    N轴无限制
                </td>
                <td>
                    S轴无限制
                </td>
                <td>
                    D轴小于等于512, 且16对齐
                </td>
            </tr>
            <tr>
                <td>v</td>
                <td>
                    支持B轴小于等于65536
                </td>
                <td>
                    N轴无限制
                </td>
                <td>
                    S轴无限制
                </td>
                <td>
                    D轴小于等于512, 且16对齐
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
                        <li>Key/Value排布支持PA_BBND（blockNum, blockSize, KV_N, D）、PA_BNBD（blockNum, KV_N,
                            blockSize, D）和PA_NZ（blockNum，KV_N，D/16，blockSize，16）三种格式</li>
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

      | mask_mode |                                                                               含义                                                                                     |
      | :--------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------: |
      |     1     |                                                全计算模式，忽略win_left和win_right                                                 |
      |     3     |                                           Causal模式，忽略win_left和win_right                                           |
      |     4     |                                           band模式，计算win_left和win_right之间的部分                                            |
- 存在性约束
  - 无
- 一致性约束
  - 无
- 特性交叉约束
  - 无
#### SeqLengths参数组

- 单参数约束
  - 公共
    - 入参 seqused_q应满足以下条件:
      - 长度应满足以下条件:
        - 长度应等于 batch 数
      - 入参中的数值应满足以下条件:
        - 其值应为非负数
    - 入参 seqused_kv应满足以下条件:
      - 长度应满足以下条件:
        - 长度应等于 batch 数
      - 入参中的数值应满足以下条件:
        - 其值应为非负数
    - 入参 cu_seqlens_q应满足以下条件:
      - 长度应满足以下条件:
        - 长度应等于 batch 数 + 1
      - 入参中的数值应满足以下条件:
        - 当 layout_q 为 TND 时，其值应递增(大于等于前一个值)排列
        - 当 layout_q 为 TND 时，第一个元素为0且最后一个元素应等于 T_Q
        - 其值应为非负数
    - 入参 cu_seqlens_kv应满足以下条件:
      - 长度应满足以下条件:
        - 长度应等于 batch 数 + 1
      - 入参中的数值应满足以下条件:
        - 当 layout_kv 为 TND 时，其值应递增(大于等于前一个值)排列
        - 当 layout_kv 为 TND 时，第一个元素为0且最后一个元素应等于 T_KV
        - 其值应为非负数
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
      - tensor shape 为2维，每一维dim value取值均不能为0，第一维长度需等于Batch size，第二维长度不能小于maxBlockNumPerSeq（maxBlockNumPerSeq为每个batch中最大actualSeqLengthsKv对应的block数量）
- 存在性约束
  - 公共
    - PagedAttention 使能情况下，必须传入 seqused_kv
- 一致性约束
  - 无
- 特性交叉约束
  - 无

#### LearnableSink参数组

- 单参数约束
  - 公共
    - 入参 sinks 需要满足以下条件
      - tensor dtype 为 FP32
      - tensor shape 为 (N_Q)
- 存在性约束
  - 无
- 一致性约束
  - 无
- 特性交叉约束
  - 无
#### SoftmaxLSE参数组

- 单参数约束
  - 公共
    - 输出 lseOut 仅支持数据类型 FP32
- 存在性约束
  - 公共
    - softmaxLSE 场景下，输出 lseOut 不应为空
- 一致性约束
  - 无
- 特性交叉约束
  - 公共
    - softmaxLSE 且非空tensor场景下，如输出 layout 为 TND则 lseOut 输出维度应为 2，且 shape 匹配 [N_Q，T_Q]
    - softmaxLSE 且非空tensor场景下，如输出 layout 不为 TND则 lseOut 输出维度应为 3，且 shape 匹配 [B，N_Q，Q_S]
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
