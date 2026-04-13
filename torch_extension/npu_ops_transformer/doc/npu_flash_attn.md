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

    其中增加**sink**之后计算逻辑见下，主要修改相关softmax_max和softmax_sum逻辑计算部分
    
    $$
    S = Q * K^{T}
    $$
    
    $$
    m = max(sink, max(S))
    $$
    
    $$
    Attention = \frac{e^{S - m} * V}{\sum e^{S-m} + e^{sink - m}}
    $$

    **说明**:
        <blockquote>query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小batch_size、S（Seq-Length）表示输入样本序列长度、H（Hidden-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸headdim，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。Q表示total_q，KV表示total_kv。
        <br>S_Q表示seqlen_q，N_Q表示nheads_q，S_KV表示seqlen_kv，N_KV表示nheads_kv。</blockquote>



## 函数原型<a name="zh-cn_topic_0000002168254826_section45077510411"></a>

```
npu_flash_attn(q, k, v, block_table=None, cu_seqlens_q=None, cu_seqlens_kv=None, seqused_q=None, seqused_kv=None, sinks=None, metadata=None, softmax_scale=1.0, mask_mode=1, win_left=-1, win_right=-1, max_seqlen_q=-1, max_seqlen_kv=-1, layout_q="BSND", layout_kv="BSND"，layout_out="BSND", return_softmax_lse=0, deterministic=0) -> attn_out(Tensor), softmax_lse(Tensor)
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
    <th>使用说明</th>
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
        <td>-</td>
        <td>BF16/FP16</td>
        <td>ND</td>
        <td>
            <ul>
                <li>可为(B, S_Q, N_Q, D)、(B, N_Q, S_Q, D)和(Q, N_Q, D)其中一种</li>
                <li>具体shape需满足后续layout匹配规则</li>
            </ul>      
        </td>
        <td>√</td>
    </tr>
    <tr>
        <td>k</td>
        <td>必选输入</td>
        <td>公式中的k</td>
        <td>-</td>
        <td>BF16/FP16</td>
        <td>ND</td>
        <td>
            <ul>
                <li>可为(B, S_KV, N_KV, D)、(B, N_KV, S_KV, D)、(KV, N_KV, D)、(number_blocks, B, N_KV, D)、(num_blocks, N_KV, block_size, D)和(num_blocks, N_KV, D/16, block_size, 16)其中一种</li>
                <li>具体shape需满足后续layout匹配规则</li>
            </ul>      
        </td>
        <td>√</td>
    </tr>
    <tr>
        <td>v</td>
        <td>必选输入</td>
        <td>公式中的v</td>
        <td>-</td>
        <td>BF16/FP16</td>
        <td>ND</td>
        <td>
            <ul>
                <li>可为(B, S_KV, N_KV, D)、(B, N_KV, S_KV, D)、(KV, N_KV, D)、(number_blocks, B, N_KV, D)、(num_blocks, N_KV, block_size, D)和(num_blocks, N_KV, D/16, block_size, 16)其中一种</li>
                <li>具体shape需满足后续layout匹配规则</li>
            </ul>      
        </td>
        <td>√</td>
    </tr>
    <tr>
        <td>block_table</td>
        <td>可选输入</td>
        <td>用于分块注意力计算中的块索引映射</td>
        <td>-</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(B, max_num_blocks_per_seq)</td>
        <td>√</td>
    </tr>
    <tr>
        <td>cu_seqlens_q</td>
        <td>可选输入</td>
        <td>累积序列长度,用于处理变长序列</td>
        <td>
            <ul>
                <li>仅在layout_q为TND时有效</li>
                <li>传入不为空时仅支持一维张量</li>
            </ul>
        </td>
        <td>INT32</td>
        <td>ND</td>
        <td>(B + 1, )</td>
        <td>√</td>
    </tr>
    <tr>
        <td>cu_seqlens_kv</td>
        <td>可选输入</td>
        <td>累积序列长度,用于处理变长序列</td>
        <td>
            <ul>
                <li>仅在layout_kv为TND时有效</li>
                <li>传入不为空时仅支持一维张量</li>
            </ul>
        </td>
        <td>INT32</td>
        <td>ND</td>
        <td>(B + 1, )</td>
        <td>√</td>
    </tr>
    <tr>
        <td>seqused_q</td>
        <td>可选输入</td>
        <td>指定每batch中实际使用的序列长度，截断冗余运算</td>
        <td>
            <ul>
                <li>若提供该参数，仅处理每batch元素中前seqused_q[i]个元素，其余元素忽略</li>
                <li>传入不为空时仅支持一维张量</li>
            </ul>
        </td>
        <td>INT32</td>
        <td>ND</td>
        <td>(B, )</td>
        <td>√</td>
    </tr>
    <tr>
        <td>seqused_kv</td>
        <td>可选输入</td>
        <td>指定每batch中实际使用的序列长度，截断冗余运算</td>
        <td>
            <ul>
                <li>若提供该参数，仅处理每batch元素中前seqused_kv[i]个元素，其余元素忽略</li>
                <li>传入不为空时仅支持一维张量</li>
            </ul>
        </td>
        <td>INT32</td>
        <td>ND</td>
        <td>(B, )</td>
        <td>√</td>
    </tr>
    <tr>
        <td>sinks</td>
        <td>可选输入</td>
        <td>提供解决注意力计算中集中前置一部分token问题的功能</td>
        <td>传入不为空时仅支持一维张量</td>
        <td>FP32</td>
        <td>ND</td>
        <td>(N_Q, )</td>
        <td>√</td>
    </tr>
    <tr>
        <td>metadata</td>
        <td>可选输入</td>
        <td>metadata</td>
        <td>传入不为空时仅支持一维张量</td>
        <td>INT32</td>
        <td>ND</td>
        <td>(max_schedule_size, )</td>
        <td>√</td>
    </tr>
    <tr>
        <td>softmax_scale</td>
        <td>可选输入</td>
        <td>可显式设置缩放因子，覆盖默认计算</td>
        <td>传入默认值为None,当为None时默认为1/sqrt(D)</td>
        <td>FLOAT</td>
        <td>N/A</td>
        <td>-</td>
        <td>×</td>
    </tr>
    <tr>
        <td>mask_mode</td>
        <td>可选输入</td>
        <td>掩码模式</td>
        <td>
            <ul>仅支持mask_mode为1、3或者4</ul>
            <ul>mode默认值为1</ul>
            <ul>
                <li>mode为1时为全计算，无掩码</li>
                <li>mode为3时为causal</li>
                <li>mode为4时为window，需配合win_left与win_right</li>
            </ul>
        </td>
        <td>INT</td>
        <td>N/A</td>
        <td>-</td>
        <td>×</td>
    </tr>
    <tr>
        <td>win_left</td>
        <td>可选输入</td>
        <td>window左界限</td>
        <td>
            <ul>
                <li>仅在mask_mode为4时有效</li>
                <li>值域范围为-1和非负整数，当为-1时表示无限窗口大小</li>
            </ul>
        </td>
        <td>INT</td>
        <td>-</td>
        <td>-</td>
        <td>×</td>
    </tr>
    <tr>
        <td>win_right</td>
        <td>可选输入</td>
        <td>window右界限</td>
        <td>
            <ul>
                <li>仅在mask_mode为4时有效</li>
                <li>值域范围为-1和非负整数，当为-1时表示无限窗口大小</li>
            </ul>
        </td>
        <td>INT</td>
        <td>-</td>
        <td>-</td>
        <td>×</td>
    </tr>
    <tr>
        <td>max_seqlen_q</td>
        <td>可选输入</td>
        <td>指定查询q序列的最大长度上限</td>
        <td>
            <ul>
                <li>值域范围为-1和非负整数，默认值为-1</li>
                <li>若设置为-1，算子会根据输入自动推断最大序列长度</li>
                <li>若设置为非负整数，则强制限制Q的序列长度不超过该值</li>
            </ul>
        </td>
        <td>INT</td>
        <td>-</td>
        <td>-</td>
        <td>×</td>
    </tr>
    <tr>
        <td>max_seqlen_kv</td>
        <td>可选输入</td>
        <td>指定键k和值v序列的最大长度上限</td>
        <td>
            <ul>
                <li>值域范围为-1和非负整数，默认值为-1</li>
                <li>若设置为-1，算子会根据输入自动推断最大序列长度</li>
                <li>若设置为非负整数，则强制限制k/v的序列长度不超过该值</li>
            </ul>
        </td>
        <td>INT</td>
        <td>-</td>
        <td>-</td>
        <td>×</td>
    </tr>
    <tr>
        <td>layout_q</td>
        <td>可选输入</td>
        <td>定义输入q张量的布局格式</td>
        <td>
            <ul>
                <li>layout可以为BSND、TND或者BNSD</li>
                <li>默认layout为BSND</li>
                <li>layout_q、layout_kv和layout_out需满足后续layout匹配规则</li>
            </ul>
        </td>
        <td>STRING</td>
        <td>N/A</td>
        <td>-</td>
        <td>×</td>
    </tr>
    <tr>
        <td>layout_kv</td>
        <td>可选输入</td>
        <td>定义输入k和v张量的布局格式</td>
        <td>
            <ul>
                <li>layout可以为BSND、TND、BNSD、PA_BBND、PA_BNBD或者PA_Nz</li>
                <li>默认layout为BSND</li>
                <li>layout_q、layout_kv和layout_out需满足后续layout匹配规则</li>
            </ul>
        </td>
        <td>STRING</td>
        <td>N/A</td>
        <td>-</td>
        <td>×</td>
    </tr>
    <tr>
        <td>layout_out</td>
        <td>可选输入</td>
        <td>定义输入k和v张量的布局格式</td>
        <td>
            <ul>
                <li>layout可以为BSND、TND或者BNSD</li>
                <li>默认layout为BSND</li>
                <li>layout_q、layout_kv和layout_out需满足后续layout匹配规则</li>
            </ul>
        </td>
        <td>STRING</td>
        <td>N/A</td>
        <td>-</td>
        <td>×</td>
    </tr>
    <tr>
        <td>return_softmax_lse</td>
        <td>可选输入</td>
        <td>是否需要获取softmax的LSE结果</td>
        <td>
            <ul>
                <li>值域范围为0和1，默认值为0</li>
                <li>当设置为1时会输出softmax_lse张量，为0时不输出该张量</li>
                <li>输出格式依赖于layout_q和layout_out，layout匹配规则如后续约束中所示</li>
            </ul>
        </td>
        <td>INT</td>
        <td>N/A</td>
        <td>-</td>
        <td>×</td>
    </tr>
    <tr>
        <td>deterministic</td>
        <td>可选输入</td>
        <td>是否开启确定性计算</td>
        <td>
            <ul>
                <li>值域范围为0和1，默认值为0</li>
                <li>当设置为1时会开启确定性计算，为0时不开启</li>
            </ul>
        </td>
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
    <th>使用说明</th>
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
        <td>与q输入布局一致</td>
        <td>BF16/FP16</td>
        <td>ND</td>
        <td></td>
        <td>√</td>
    </tr>
    <tr>
        <td>softmax_lse</td>
        <td>可选输出</td>
        <td>softmax的LSE结果</td>
        <td>是否输出取决于可选输入return_softmax_lse参数</td>
        <td>FP32</td>
        <td>ND</td>
        <td>
            <ul>
                <li>可为(B, N_Q, S_Q)和(N_Q, Q)</li>
                <li>shape取决于layout_q和layout_out，可见后续layout匹配规则</li>
            </ul>      
        </td>
        <td>√</td>
    </tr>
</tbody>
</table>

## 约束说明<a name="zh-cn_topic_0000002168254826_section12345537164214"></a?>
## 约束说明

### 一、约束类型说明

npu_flash_attn算子约束分为4个档位，按约束复杂程度递增分为 单参数约束、存在性约束、一致性约束和特性交叉约束，各档位约束内容和示例如下:

- npu_flash_attn算子约束分为4个档位，按约束复杂程度递增分为 Tensor 和 Attributes
  - 对于Tensor，单参数约束中包含如下校验
    - 校验 shape，包括shape维度dim、每一维度 dim value
    - 校验 dtype
    - 校验 format
  - 对于属性 Attribute
    - 校验属性取值
- 存在性约束:约束特定场景下，特性参数组内，必须传入某参数，或不支持传入某参数
- 一致性约束:特性参数组内，各个参数间约束。
  - Example 1:属性sparseMode和输入tensor attenMask均属于Attention Mask参数组，sparseMode取值为2/3/4时，attenMask shape必须为（2048，2048），此类约束即为参数组内的一致性约束
- 特性交叉约束:涉及多个参数组，不同参数组间交叉约束
  - Example 1:输入tensor blockTable 和 属性 blockSize 属于Paged Attention（同PA）参数组，输入tensor attenMask属于 Mask参数组；在PA场景下，attenMask输入最后一维（attenMaskS2）需要大于等于maxBlockNumPerSeq * blockSize，此类约束即为多参数组间的交叉约束；**且为保证风格统一，此约束会放在入参顺序靠后的 Paged Attention参数组中**

### 特性参数组

|      特性参数组       |     参数字段名称     |    字段分组    |  字段类型  |
| :--------------------: | :-------------------: | :-------------: | :--------: |
|      公共参数组      |         q         |      INPUT      |   Tensor   |
|                      |          k         |      INPUT      | Tensor |
|                      |         v         |      INPUT      | Tensor |
|                      |      softmax_scale      | ATTR(OPTIONAL) |   float   |
|                      |      layout_q      | ATTR(OPTIONAL) |    string    
|                      |      layout_kv      | ATTR(OPTIONAL) |    string    |
|                      |      layout_out      | ATTR(OPTIONAL) |    string    |
|                      |     attentionOut     |     OUTPUT     |   Tensor   |
|      Mask参数组      |      sparseMode      | ATTR(OPTIONAL) |   int64   |
|                      |       win_left       | ATTR(OPTIONAL) |   int64   |
|                      |      win_right      | ATTR(OPTIONAL) |   int64   |
|  SeqLengths参数组       |   seqused_q   | INPUT(OPTIONAL) |  Tensor  |
|                      |   seqused_kv   | INPUT(OPTIONAL) |  Tensor  |
|                      |   cu_seqlens_q   | INPUT(OPTIONAL) |  Tensor  |
|                      |   cu_seqlens_kv   | INPUT(OPTIONAL) |  Tensor  |
| Paged Attention参数组 |      blockTable      | INPUT(OPTIONAL) |   Tensor   |
|  LearnableSink参数组  |     sinks     | INPUT(OPTIONAL) |   Tensor   |
|   SoftmaxLSE参数组   |    return_softmax_lse    | ATTR(OPTIONAL) |    int    |
|                      |      softmaxLse      |     OUTPUT(OPTIONAL)     |   Tensor   |

### 基准信息说明

资料约束中，常见字段释义如下:

|    命名    |                            含义                            |
| :---------: | :---------------------------------------------------------: |
|      B      |                Batch, 表示输入样本批量大小                |
|     Q_N     |        输入query tensor的头数，对应query shape中的N        |
|    KV_N    |    输入key/value tensor的头数，对应key/value shape中的N    |
|     Q_S     |      输入query tensor的序列长度，对应query shape中的S      |
|    KV_S    |  输入key/value tensor的序列长度，对应key/value shape中的S  |
|     Q_T     |          输入query tensor所有batch序列长度的累加和          |

### 参数组约束

<!--
#### 整体规则
1. 按照 特性参数组 -> 四层约束 -> 量化类型（非量化/伪量化/全量化） -> FA场景（GQA/Prefill MLA/Decode MLA） -> 芯片代际（Atlas A2/Ascend 950PR）层级维护
2. 每个特性组，缩进风格一致，表格宽高无固定约束，保持简洁清晰即可
3. 约束中参数命名，第一次出现必须和接口中参数名一致，后续可通过 “attenMask（同 mask/后称 mask）”方式缩写
4. 对于输入各轴缩写，缩写需要具有自解释性，如QueryS、KeyNumHead、PseShiftS2等，禁止出现 S1、B、n2 等字眼；最好在基准信息说明中补充各个变量含义
-->
#### 公共参数组（ShapeChecker）

- 单参数约束

  - layout约束
    <table style="undefined;table-layout: fixed; width: 1100px">
        <colgroup>
            <col style="width: 200px">
            <col style="width: 250px">
            <col style="width: 200px">
            <col style="width: 450px">
        </colgroup>
        <thead>
            <tr>
                <th>layout_q</th>
                <th>layout_kv</th>
                <th>layout_out</th>
                <th>layout_softmax_lse_layout</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>BNSD</td>
                <td>
                    <li>BNSD</li>
                    <li>PA_BBND</li>
                    <li>PA_BNBD</li>
                    <li>PA_Nz</li>
                </td>
                <td>
                    <li>BNSD</li>
                    <li>BSND</li>
                </td>
                <td>
                    (B, N_Q, S_Q)
                </td>
            </tr>
            <tr>
                <td>BSND</td>
                <td>
                    <li>BSND</li>
                    <li>PA_BBND</li>
                    <li>PA_BNBD</li>
                    <li>PA_Nz</li>
                </td>
                <td>
                    <li>BSND</li>
                </td>
                <td>
                    (B, N_Q, S_Q)
                </td>
            </tr>
            <tr>
            <td>TND</td>
                <td>
                    <li>TND</li>
                    <li>PA_BBND</li>
                    <li>PA_BNBD</li>
                    <li>PA_Nz</li>
                </td>
                <td>
                    <li>TND</li>
                </td>
                <td>
                    (N_Q, Q)
                </td>
            </tr>
        </tbody>
    </table>
  - 公共约束

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
                <td>attentionOut为返回空tesor</td>
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
                <th>Q_N/KV_N</th>
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
                    Q_N需可整除KV_N，Q_N与KV_N的比值无限制
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
            </tr>
        </tbody>
    </table>
- 特性交叉约束

  - 非量化场景
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
                        <li>Key/Value排布支持BnBsH（blockNum, blockSize, H）、BnNBsD（blockNum, KV_N,
                            blockSize, D）和NZ（blockNum，KV_N，D/16，blockSize，16）三种格式</li>
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
      |     3     |                                           rightDownCausal模式，忽略win_left和win_right                                           |
      |     4     |                                           band模式，计算win_left和win_right之间的部分                                            |
- 存在性约束
  - 无
- 一致性约束
  - 公共
    - Q_S 大于1时，win_left 与 win_right 应满足 win_left * (-1) <= win_right，以确保具有有效数据
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
        - 当 layout_q 为 TND 时，第一个元素为0且最后一个元素应等于 T
        - 其值应为非负数
    - 入参 cu_seqlens_kv应满足以下条件:
      - 长度应满足以下条件:
        - 长度应等于 batch 数 + 1
      - 入参中的数值应满足以下条件:
        - 当 layout_kv 为 TND 时，其值应递增(大于等于前一个值)排列
        - 当 layout_kv 为 TND 时，第一个元素为0且最后一个元素应等于 T
        - 其值应为非负数
- 存在性约束
  - 公共
    - 入参 seqused_kv 应满足以下条件:
      - PagedAttention场景，必须传入 cu_seqlens_kv
    - 入参 cu_seqlens_q 应满足以下条件:
      - 当 layout_q 为 TND 时，必须传入 cu_seqlens_q
    - 入参 cu_seqlens_kv 应满足以下条件:
      - 当 layout_kv 为 TND 时，必须传入 cu_seqlens_kv
- 一致性约束
  - 无
- 特性交叉约束
  - 无
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
