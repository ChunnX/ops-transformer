# SparseFlashMlaSoftmaxL1Norm

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

-   算子功能：计算`SparseFlashMla`注意力的Softmax L1Norm结果，支持Sliding Window Attention、Compressed Attention以及Sparse Compressed Attention场景。该算子为`aclnnSparseFlashMlaGrad`反向算子的配套正向接口，输出可用于反向梯度计算。

-   计算公式：

    阶段一：根据不同cmp_ratio场景，对输入k进行选择

    * 当cmp_ratio = 1 (SWA)：

    $$
    selectedKv\text{ }=\text{ }k
    $$

    * 当cmp_ratio = 4 (SCFA)：

    $$
    selectedKv\text{ }=concat(k, \text{ }Gather \left( cmpkv,topkIndices \left[ i \left]  \left)) ,\text{ }0\text{ } < =i < \text{ }selectBlockCount\right. \right. \right. \right.
    $$

    * else (CFA):

    $$
    selectedKv\text{ }=concat(k, \text{ }cmpkv)
    $$

    阶段二：计算P（SimpleSoftmax）

    $$
    P = SimpleSoftmax(Mask(Q \text{ }@\text{ } selectedKv^{{T}} \cdot \text{ } scale), softmaxLse)
    $$

    阶段三：计算Softmax L1Norm

    $$
    softmaxL1Norm = \frac{ReduceSum(P, dim=G)}{G}
    $$

    其中，$G$ 为group数（$G = N1 / N2$）。


## 参数说明

<table style="undefined;table-layout: fixed; width: 1080px">
    <colgroup>
        <col style="width: 170px">
        <col style="width: 120px">
        <col style="width: 300px">
        <col style="width: 212px">
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
        <td>q</td>
        <td>输入</td>
        <td>attention结构的输入Q。</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>k</td>
        <td>输入</td>
        <td>attention结构的输入K(V)。</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>softmaxLse</td>
        <td>输入</td>
        <td>注意力正向计算的输出softmaxLse，计算公式详见sparse_flash_mla文档。</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>sparseIndicesOptional</td>
        <td>输入</td>
        <td>稀疏场景下选择的k中权重较高的注意力索引。</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>cuSeqlensQOptional</td>
        <td>输入</td>
        <td>每个Batch中，Query的有效token数。</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>cuSeqlensKOptional</td>
        <td>输入</td>
        <td>每个Batch中，Key的有效token数。</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>sequsedQOptional</td>
        <td>输入</td>
        <td>表示不同batch中q实际参与运算的token数。</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>sequsedKOptional</td>
        <td>输入</td>
        <td>表示不同batch中k实际参与运算的token数。</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>cmpResidualKOptional</td>
        <td>输入</td>
        <td>表示每个batch S2 // cmpRatio后的余数。</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>topkLengthOptional</td>
        <td>输入</td>
        <td>表示每行q对应的k实际可选的topk长度。</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>metadataOptional</td>
        <td>输入</td>
        <td>表示tiling下沉的aicpu算子输出结果。</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>softmaxScale</td>
        <td>属性</td>
        <td>缩放系数。</td>
        <td>FLOAT32</td>
        <td>-</td>
    </tr>
    <tr>
        <td>cmpRatio</td>
        <td>属性</td>
        <td>表示对k的压缩率。</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>maskMode</td>
        <td>属性</td>
        <td>表示q和k计算的mask模式。</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>layoutQOptional</td>
        <td>属性</td>
        <td>表示输入q的数据排布格式。</td>
        <td>STRING</td>
        <td>-</td>
    </tr>
    <tr>
        <td>layoutKOptional</td>
        <td>属性</td>
        <td>表示输入k的数据排布格式。</td>
        <td>STRING</td>
        <td>-</td>
    </tr>
    <tr>
        <td>softmaxL1Norm</td>
        <td>输出</td>
        <td>表示q与k计算得出的softmax L1Norm结果，公式为reduceG(softmax)/G。</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    </tbody>
</table>

## 约束说明

- 仅支持BSND或TND layout；关于数据shape的约束如下：
  - B：泛化支持。
  - S1、S2：泛化支持；S1、S2支持不等长。
  - N1：支持1~128。
  - N2：仅支持1。
  - D：仅支持512。

- maskMode支持：
  - 0：不做mask操作。
  - 3：rightDownCausal模式的mask，对应以右顶点为划分的下三角场景。

- 确定性计算：aclnnSparseFlashMlaSoftmaxL1Norm默认确定性实现。

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
    <a href="./examples/test_aclnn_sparse_flash_mla_softmax_l1_norm.cpp">test_aclnn_sparse_flash_mla_softmax_l1_norm
    </a>
    </td>
    <td class="tg-lboi" rowspan="6">
    通过
    <a href="./docs/aclnnSparseFlashMlaSoftmaxL1Norm.md">aclnnSparseFlashMlaSoftmaxL1Norm
    </a>
    接口方式调用算子
    </td>
  </tr>
</tbody></table>
