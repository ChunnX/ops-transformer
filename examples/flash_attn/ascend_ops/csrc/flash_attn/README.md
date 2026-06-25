# FA

## 核心交付件
1. `csrc/flash_attn/op_host` 算子host实现
2. `csrc/flash_attn/op_kernel`算子kernel实现
3. `csrc/flash_attn/torch_interface.cpp`算子pytorch层和C++实现层接口
4. `csrc/flash_attn/CMakeLists.txt` 算子cmake配置

## 计算公式

self-attention（自注意力）利用输入样本自身的关系构建了一种注意力模型。其原理是假设有一个长度为$n$的输入样本序列$x$，$x$的每个元素都是一个$d$维向量，可以将每个$d$维向量看作一个token embedding，将这样一条序列经过3个权重矩阵变换得到3个维度为$n*d$的矩阵。

self-attention的计算公式一般定义如下，其中$Q、K、V$为输入样本的重要属性元素，是输入样本经过空间变换得到，且可以统一到一个特征空间中。公式及算子名称中的"Attention"为"self-attention"的简写。

$$
Attention(Q,K,V)=Softmax(\frac{QK^T}{\sqrt{d}})V
$$

其中$Q$和$K^T$的乘积代表输入$x$的注意力，为避免该值变得过大，通常除以$d$的开根号进行缩放，并对每行进行softmax归一化，与$V$相乘后得到一个$n*d$的矩阵。

**说明**：
<blockquote>query、key、value数据排布格式支持从多种维度解读，其中B（Batch）表示输入样本批量大小、S（Seq-Length）表示输入样本序列长度、H（Head-Size）表示隐藏层的大小、N（Head-Num）表示多头数、D（Head-Dim）表示隐藏层最小的单元尺寸，且满足D=H/N、T表示所有Batch输入样本序列长度的累加和。
<br>Q_S表示query shape中的S，KV_S表示key和value shape中的S，Q_N表示num_query_heads，KV_N表示num_key_value_heads。P表示Softmax(<span>(QK<sup class="superscript">T</sup>) / <span class="sqrt">d</span></span>)的计算结果。</blockquote>


## 接口参数说明

- <a id="query"></a>**Q/K/V**
    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 740px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>属性</th>
                <th>含义</th>
                <th>备注</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>Batch(B)</td>
            <td>输入样本批量大小</td>
            <td>-</td>
        </tr>
        <tr>
            <td>Head-Num(N)</td>
            <td>多头数</td>
            <td>支持Q矩阵的N与KV矩阵的N不同，但需要保证Q矩阵的N是KV矩阵的N的整数倍</td>
        </tr>
        <tr>
            <td>Seq-Length(S)</td>
            <td>输入样本序列长度</td>
            <td>支持Q矩阵的S与KV矩阵的S不同，但需要保证Q、K、V矩阵的S都是128对齐的</td>
        </tr>
        <tr>
            <td>Head-Dim(D)</td>
            <td>隐藏层最小的单元尺寸</td>
            <td>目前只支持D=128</td>
        </tr>
        <tr>
            <td>数据类型</td>
            <td>Q、K、V矩阵中的数据类型</td>
            <td>目前只支持bfloat16</td>
        </tr>
        </tbody>
    </table>
**QKV目前不支持空Tensor传入**

- <a id="mask"></a>**Mask**
    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 740px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>属性</th>
                <th>含义</th>
                <th>备注</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>Shape</td>
            <td>Mask矩阵的传入形状</td>
            <td>必选输入，支持以(QS,KvS), (1,QS,KvS), (1,1,QS,KvS)形状传入</td>
        </tr>
        </tbody>
    </table>

- <a id="sparseMode"></a>**SparseMode（必传属性）**
    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 740px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>属性</th>
                <th>含义</th>
                <th>备注</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>1</td>
            <td>allMask，必须传入完整的attenmask矩阵</td>
            <td>-</td>
        </tr>
        </tbody>
    </table>

- <a id="inputLayout"></a>**InputLayout（必传属性）**
    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 740px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>属性</th>
                <th>含义</th>
                <th>备注</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>BNSD</td>
            <td>Q、K、V以及输出矩阵的排布格式</td>
            <td>目前只支持BNSD格式</td>
        </tr>
        </tbody>
    </table>
- <a id="attr"></a>**其他参数**
    <table style="undefined;table-layout: fixed; width: 942px"><colgroup>
        <col style="width: 100px">
        <col style="width: 740px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>属性</th>
                <th>含义</th>
                <th>备注</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>ScaleValue</td>
            <td>Q与K矩阵相乘之后的缩放系数</td>
            <td>必传属性，等于1表示不缩放，默认为  1/√d</td>
        </tr>
        <tr>
            <td>Num_heads</td>
            <td>Q矩阵的N</td>
            <td>必传属性</td>
        </tr>
        <tr>
            <td>NumKeyValueHeads</td>
            <td>K、V矩阵的N</td>
            <td>必传属性、KV矩阵的N要一致</td>
        </tr>
        </tbody>
    </table>



## 代码架构
算子执行在Device侧（NPU，执行Kernel函数），但是所有的输入都在Host侧（CPU，执行Tiling函数），Device侧无法直接读取Host侧的数据，两者存在数据隔离。在Host侧准备好的数据需要通过TilingData(用户自定义的数据结构)传递给Device侧，Device侧拿到这个数据再去执行算子。因此代码主要分为如下三部分：
 - **csrc/flash_attn/torch_interface.cpp** 里面包含了算子执行的所有流程，上层python入参传递到该文件，经过Tiling函数计算得到TilingData，将TilingData传递给算子的Kernel函数进行执行，最终得到结果。
 - **csrc/flash_attn/op_host** 算子从入参计算得到TilingData的所有文件都在该目录下
 - **csrc/flash_attn/op_kernel**算子的Kernel函数所有文件都在该目录下

