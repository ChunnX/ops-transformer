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

-   API功能：

    支持flashattention训练推理归一化
-   计算公式：
    
    self-attention（自注意力）利用输入样本自身的关系构建了一种注意力模型。其原理是假设有一个长度为$n$的输入样本序列$x$，$x$的每个元素都是一个$d$维向量，可以将每个$d$维向量看作一个token embedding，将这样一条序列经过3个权重矩阵变换得到3个维度为$n*d$的矩阵。

    self-attention的计算公式一般定义如下，其中$Q、K、V$为输入样本的重要属性元素，是输入样本经过空间变换得到，且可以统一到一个特征空间中。公式及算子名称中的"Attention"为"self-attention"的简写。

    $$
    Attention(Q,K,V)=Score(Q,K)V
    $$

    本算子中Score函数采用Softmax函数，self-attention计算公式为：

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

    **说明**：
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

## 约束说明<a name="zh-cn_topic_0000002168254826_section12345537164214"></a>

-   该接口支持训练正向和推理场景下使用。
-   layout匹配关系如下：
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
            <td>(N_Q, Q)</td>
        </tr>
    </tbody>
    </table>

- 确定性计算：
  - npu_flash_attn通过可选参数deterministic控制是否开启确定性计算，1代表开启确定性计算，0代表关闭确定性计算。
- 公共约束
    - 入参为空的场景处理：
        - 空Tensor指必选输入和输出的shapeSize为0。在空Tensor场景下，若attentionOut为空，返回空，否则返回全0；若有lse且lse为空时返回空，lse不为空则返回全inf。非空Tensor时输入正常拦截。
        - query，attentionOut所有tensor的shapeSize为0，属于空Tensor。
        - query，attentionOut所有tensor的shapeSize不为0，若有lse且lse不为空，并且key，value中所有tensor的shapeSize为0，属于空Tensor。
        - attentionOut和lse都为空时，属于空Tensor。
        - 属于空Tensor时，跳过校验流程；否则，走正常校验流程。
- <a id="public"></a>通用场景
    <table style="undefined;table-layout: fixed; width: 1000px">
        <colgroup>
            <col style="width: 150px">
            <col style="width: 100px">
            <col style="width: 750px">
        </colgroup>
        <thead>
            <tr>
                <th>参数</th>
                <th>维度</th>
                <th>限制</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td rowspan="4">q/k/v</td>
                <td>B</td>
                <td>
                    <ul>
                        <li>支持B轴小于等于65536</li>
                        <li>非连续场景下 key、value的tensorlist中的batch只能为1，个数等于query的B，N和D需要相等。由于tensorlist限制, 非连续场景下B不能大于256</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>N</td>
                <td>
                    <ul>
                        <li>N轴无限制</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>S</td>
                <td>
                    <ul>
                        <li>S轴无限制</li>
                    </ul>
                </td>
            </tr>
            <tr>
                <td>D</td>
                <td>
                    <ul>
                        <li>D轴无限制</li>
                    </ul>
                </td>
            </tr>
        </tbody>
    </table>
- <a id="Mask"></a>Mask
    <table style="undefined;table-layout: fixed; width: 820"><colgroup>
        <col style="width: 100px">
        <col style="width: 360px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>sparseMode</th>
                <th>含义</th>
                <th>备注</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>1</td>
            <td>allMask</td>
            <td>
                无
            </td>
        </tr>
        <tr>
            <td>3</td>
            <td>rightDownCausal模式的mask，对应以右顶点为划分的下三角场景</td>
            <td>
                无
            </td>
        </tr>
        <tr>
            <td>4</td>
            <td>band模式的mask</td>
            <td>
                无
            </td>
        </tr>
        </tbody>
    </table>
- <a id="seqused"></a>seqused
    <table style="undefined;table-layout: fixed; width: 1148px"><colgroup>
    <col style="width: 195px">
    <col style="width: 156px">
    <col style="width: 608px">
    <col style="width: 189px">
        </colgroup>
        <thead>
            <tr>
                <th>参数</th>
                <th>Layout</th>
                <th>说明</th>
                <th>限制</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>seqused_q</td>
                <td>不为TND</td>
                <td>该入参为可选入参，其长度为1或大于等于query的batch值，该入参中的值代表每个batch的实际长度，其值应该不大于Q_S</td>
                <td rowspan="4">传入时应为非负数</td>
            </tr>
            <tr>
                <td>cu_seqused_q</td>
                <td>TND</td>
                <td>该入参必须传入，第b+1个值表示前b个batch的Q_S轴累加长度，其值应递增（大于等于前一个值）排列，且该入参长度代表总batch_size + 1</td>
            </tr>
            <tr>
                <td>seqused_kv</td>
                <td>不为TND</td>
                <td>
                在非PA场景下，该入参为可选入参，其长度为1或大于等于key/value的batch值，该入参中的值代表每个batch的实际长度，其值应该不大于KV_S。</br>
                在PA场景下，该入参为必选入参，其长度等于key/value的batch值，代表每个batch的实际长度，值不大于KV_S
                </td>
            </tr>
            <tr>
                <td>cu_seqused_kv</td>
                <td>TND</td>
                <td>该入参必须传入</br>
                    在非PA场景下，第b + 1个值表示前b个batch的KV_S轴累加长度，其值应递增（大于等于前一个值）排列，且该入参长度代表总batch数</br>
                </td>
            </tr>
        </tbody>
    </table>

- <a id="PagedAttention"></a>PagedAttention
    <table style="undefined;table-layout: fixed; width: 1354px">
        <colgroup>
            <col style="width: 155px">
            <col style="width: 169px">
            <col style="width: 550px">
            <col style="width: 600px">
        </colgroup>
        <thead>
            <tr>
                <th>参数所属场景或特性</th>
                <th>参数</th>
                <th>约束</th>
                <th>备注</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td rowspan="2">PagedAttention</td>
                <td>blockSize</td>
                <td>
                    <li>在使能PagedAttention，blockSize需要传入非0值</li>
                </td>
                <td>blockSize是用户自定义的参数，该参数的取值会影响PagedAttention的性能，通常情况下，PagedAttention可以提高吞吐量，但会带来性能上的下降。</td>
            </tr>
            <tr>
                <td>blockTable</td>
                <td>PagedAttention场景下，blockTable必须为二维，第一维长度需等于B，第二维长度不能小于maxBlockNumPerSeq（maxBlockNumPerSeq为每个batch中最大actualSeqLengthsKv对应的block数量）。
                </td>
                <td>-</td>
            </tr>
            <tr>
                <td rowspan="2">通用场景</td>
                <td>k、v</td>
                <td>
                    <ul>
                        <li>支持key、value dtype为FLOAT16/BFLOAT16</li>
                </td>
                <td>
                <ul>
                    <li>PagedAttention场景下，kv cache排布为BnNBsD时性能通常优于kv cache排布为BnBsH时的性能，建议优先选择BnNBsD格式。</li>
                    <li>blocknum不能小于根据actualSeqLengthsKv和blockSize计算的每个batch的block数量之和。且key和value的shape需保证一致。</li>
                    <li>PagedAttention场景下，当输入kv cache排布格式为BnBsH（blocknum, blocksize, H），且 KV_N * D 超过65535时，受硬件指令约束，会被拦截报错。可通过使能GQA（减小 KV_N）或调整kv cache排布格式为BnNBsD（blocknum, KV_N, blocksize, D）解决。</li>
                </ul>
                </td>
            </tr>
            <tr>
                <td>seqused_kv</td>
                <td>PagedAttention场景下，必须传入seqused_kv。</td>
                <td>-</td>
            </tr>
            <tr>
                <td colspan="4">
                <ul><li>PagedAttention 不支持tensorlist场景</li>
                <li>PagedAttention的使能必要条件是blocktable存在且有效，同时k、v是按照blocktable中的索引在一片连续内存中排布，在该场景下k、v的inputLayout参数无效</li>
                </ul></td>
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
