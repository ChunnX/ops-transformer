# npu\_flash_attn<a name="ZH-CN_TOPIC_0000002309174912"></a>

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A5 训练系列产品/Atlas A5 推理系列产品</term>           |    √     |


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
        <blockquote>query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Hidden-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
        <br>Q_S表示query shape中的S，KV_S表示key和value shape中的S，Q_N表示num_query_heads，KV_N表示num_key_value_heads。P表示Softmax(<span>(QK<sup class="superscript">T</sup>) / <span class="sqrt">d</span></span>)的计算结果。</blockquote>



## 函数原型<a name="zh-cn_topic_0000002168254826_section45077510411"></a>

```
npu_flash_attn(q, k, v, block_table=None, cu_seqlens_q=None, cu_seqlens_kv=None, seqused_q=None, seqused_kv=None, sinks=None, metadata=None, softmax_scale=1.0, mask_mode=1, win_left=-1, win_right=-1, max_seqlen_q=-1, max_seqlen_kv=-1, layout_q="BSND", layout_kv="BSND"，layout_out="BSND", return_softmax_lse=0, deterministic=0) -> attn_out(Tensor), softmax_lse(Tensor)
```

## 参数说明<a name="zh-cn_topic_0000002168254826_section112637109429"></a>

-   **q** (`Tensor`)：必选输入参数，公式中的输入Q，要求为张量，shape为\(B, S_Q, N_Q, D\)、\(B, N_Q, S_Q, D\)或\(Q, N_Q, D\)。数据类型支持`bfloat16`和`float16`，数据格式为$ND$，默认值为N/A，对应[npu\_flash_attn](npu_flash_attn.md)的`q`输入。

-   **k** (`Tensor`)：必选输入参数，公式中的输入K，要求为张量，shape为\(B, S_KV, N_KV, D\)、\(B, N_KV, S_KV, D\)、\(KV, N_KV, D\)、\(num_blocks, block_size, N_KV, D\)、\(num_blocks, N_KV, block_size, D\)或\(num_blocks, N_KV, D/16, block_size, 16\)。数据类型支持`bfloat16`和`float16`，数据格式为$ND$，默认值为N/A，对应[npu\_flash_attn](npu_flash_attn.md)的`k`输入。

-   **v** (`Tensor`)：必选输入参数，公式中的输入V，要求为张量，shape为\(B, S_KV, N_KV, D\)、\(B, N_KV, S_KV, D\)、\(KV, N_KV, D\)、\(num_blocks, block_size, N_KV, D\)、\(num_blocks, N_KV, block_size, D\)或\(num_blocks, N_KV, D/16, block_size, 16\)。数据类型支持`bfloat16`和`float16`，数据格式为$ND$，默认值为N/A，对应[npu\_flash_attn](npu_flash_attn.md)的`v`输入。

-   **block\_table** (`Tensor`)：可选输入参数，要求为二维张量，shape为\(B, max_num_blocks_per_seq\)。数据类型支持`int32`，数据格式为$ND$，默认值为N/A。对应[npu\_flash_attn](npu_flash_attn.md)的`block_table`输入。

-   **cu\_seqlens\_q** (`Tensor`)：可选输入参数，要求为张量，shape为\(B + 1, \)。数据类型支持`int32`，数据格式为$ND$，默认值为N/A。对应[npu\_flash_attn](npu_flash_attn.md)的`cu_seqlens_q`输入。

-   **cu\_seqlens\_kv** (`Tensor`)：可选输入参数，要求为张量，shape为\(B + 1, \)。数据类型支持`int32`，数据格式为$ND$，默认值为N/A。对应[npu\_flash_attn](npu_flash_attn.md)的`cu_seqlens_kv`输入。

-   **seqused\_q** (`Tensor`)：可选输入参数，要求为张量，shape为\(B, \)。数据类型支持`int32`，数据格式为$ND$，默认值为N/A。对应[npu\_flash_attn](npu_flash_attn.md)的`seqused_q`输入。

-   **seqused\_kv** (`Tensor`)：可选输入参数，要求为张量，shape为\(B, \)。数据类型支持`int32`，数据格式为$ND$，默认值为N/A。对应[npu\_flash_attn](npu_flash_attn.md)的`seqused_kv`输入。

-   **sinks** (`Tensor`)：可选输入参数，公式中的sink，提供解决注意力计算时集中在前面一部分token问题的功能，要求不为空时需要为一维张量，shape为\(N_Q, \)。数据类型支持`float32`，数据格式为$ND$，默认值为N/A。对应[npu\_flash_attn](npu_flash_attn.md)的`sinks`输入。

-   **metadata** (`Tensor`)：可选输入参数，公式中的metadata，要求为张量，shape为\(max_schedule_size, \)。数据类型支持`int32`，数据格式为$ND$，默认值为N/A。对应[npu\_flash_attn](npu_flash_attn.md)的`metadata`输入。

-   **softmax\_scale** (`FLOAT`)：可选字段。数据类型支持`float`，数据格式为$N/A$，传入默认值为None, 当为None时使用时默认值为1/sqrt(D)。对应[npu\_flash_attn](npu_flash_attn.md)的`softmax_scale`输入。

-   **mask\_mode** (`INT`)：可选字段。数据类型支持`int`，仅支持mode传入为1、3或者4。数据格式为$N/A$，传入默认值为1, 当mask_mode传入为1时为全计算模式，当传入为3时为causal模式，当传入为4时为window模式。对应[npu\_flash_attn](npu_flash_attn.md)的`mask_mode`输入。

-   **win\_left** (`INT`)：可选字段，此字段仅在mask_mode非4时合法。数据类型支持`int`，输入为-1或为非负整数。传入默认值为-1，当为-1不做限制。对应[npu\_flash_attn](npu_flash_attn.md)的`win_left`输入。

-   **win\_right** (`INT`)：可选字段，此字段仅在mask_mode非4时合法。数据类型支持`int`，输入为-1或为非负整数。传入默认值为-1，当为-1不做限制。对应[npu\_flash_attn](npu_flash_attn.md)的`win_right`输入。

-   **max\_seqlen\_q** (`INT`)：可选字段。数据类型支持`int`，输入为-1或为非负整数。传入默认值为-1。对应[npu\_flash_attn](npu_flash_attn.md)的`max_seqlen_q`输入。

-   **max\_seqlen\_kv** (`INT`)：可选字段。数据类型支持`int`，输入为-1或为非负整数。传入默认值为-1。对应[npu\_flash_attn](npu_flash_attn.md)的`max_seqlen_kv`输入。

-   **layout\_q** (`STRING`)：可选字段。数据类型支持`String`，输入为BSND、TND或者BNSD。传入默认值为BSND，数据格式为N/A。对应[npu\_flash_attn](npu_flash_attn.md)的`layout_q`输入。

-   **layout\_kv** (`STRING`)：可选字段。数据类型支持`String`，输入为BSND、TND、BNSD、PA_BBND、PA_BNBD或者PA_Nz。传入默认值为BSND，数据格式为N/A。对应[npu\_flash_attn](npu_flash_attn.md)的`layout_kv`输入。

-   **layout\_out** (`STRING`)：可选字段。数据类型支持`String`，输入为BSND、TND或者BNSD。传入默认值为BSND，数据格式为N/A。对应[npu\_flash_attn](npu_flash_attn.md)的`layout_out`输入。

-   **return\_softmax\_lse** (`INT`)：可选字段。数据类型支持`int`，输入为0或者1。传入默认值为0，数据格式为N/A。对应[npu\_flash_attn](npu_flash_attn.md)的`return_softmax_lse`输入。

-   **deterministic** (`INT`)：可选字段，代表是否开启确定性计算。数据类型支持`int`，输入为0或者1。传入默认值为0，数据格式为N/A。对应[npu\_flash_attn](npu_flash_attn.md)的`deterministic`输入。

## 返回值说明<a name="zh-cn_topic_0000002168254826_section22231435517"></a>

-   **attn\_out** (`Tensor`)：固定输出。数据类型支持`bfloat16`或者`float16`，输出为张量，shape为\(B, S_Q, N_Q, D\)、\(B, N_Q, S_Q, D\)或者\(Q, N_Q, D\)，默认值为N/A，数据格式为ND。对应[npu\_flash_attn](npu_flash_attn.md)的`attn_out`输出。

-   **softmax\_lse** (`Tensor`)：可选输出，与可选输入return_softmax_lse传入值有关。数据类型支持`float32`，输出为张量，shape为\(B, N_Q, S_Q\)或者\(N_Q, Q\)，默认值为N/A，数据格式为ND。对应[npu\_flash_attn](npu_flash_attn.md)的`softmax_lse`输出。

## 约束说明<a name="zh-cn_topic_0000002168254826_section12345537164214"></a>

-   该接口支持训练和推理场景下使用。
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
