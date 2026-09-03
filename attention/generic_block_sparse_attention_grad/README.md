# GenericBlockSparseAttentionGrad

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|     √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|    x     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|    x     |
|<term>Atlas 200I/500 A2 推理产品</term>|      x     |
|<term>Atlas 推理系列产品</term>|      x     |
|<term>Atlas 训练系列产品</term>|      x     |

## 算子功能

- 算子功能：GenericBlockSparseAttentionGrad是通用块稀疏注意力的反向计算算子。该算子依据`rsvdBlockIdx`/`rsvdBlockCount`（稀疏块索引表）定义的索引，仅在被选中的Q块上计算和传播梯度，在反向传播时精确路由梯度，支持动态、可变长的分块稀疏模式。调用前须先通过`aclnnGenericBlockSparseAttentionGradMetadata`生成分核`metadata`。

- 计算公式：

  1. 计算 Softmax 后的注意力权重矩阵

  $$
  P = SimpleSoftmax(Mask(Q @ selectedK^{T} \cdot scale), lse)
  $$

  2. 计算 $dP$（Softmax 梯度的直接部分）

  $$
  dP = dO @ selectedV^{T}
  $$

  3. 计算 $dS$（注意力分数梯度）

  $$
  dS = P \odot (dP - SoftmaxGrad(dO, O))
  $$

  4. 计算 $dQ$、$dK$、$dV$

  $$
  dQ = dS @ selectedK \cdot scale
  $$

  $$
  dK = dS^{T} @ Q \cdot scale
  $$

  $$
  dV = P^{T} @ dO
  $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px">
    <colgroup>
        <col style="width: 170px">
        <col style="width: 170px">
        <col style="width: 310px">
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
        <td>query</td>
        <td>输入</td>
        <td>attention结构的输入Q。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>key</td>
        <td>输入</td>
        <td>attention结构的输入K。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>value</td>
        <td>输入</td>
        <td>attention结构的输入V，shape与key相同。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>dout</td>
        <td>输入</td>
        <td>注意力输出矩阵的梯度。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>out</td>
        <td>输入</td>
        <td>注意力输出矩阵。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>lse</td>
        <td>输入</td>
        <td>注意力正向计算的输出lse，layout与正向输出保持一致。</td>
        <td>FLOAT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>rsvdBlockIdx</td>
        <td>输入</td>
        <td>稀疏块索引数组，指定每个KV块选择的Q块/token索引。</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>rsvdBlockCount</td>
        <td>输入</td>
        <td>指定每个KV块实际选择的Q数量。</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>metadata</td>
        <td>输入</td>
        <td>由GenericBlockSparseAttentionGradMetadata生成的分核信息。</td>
        <td>INT64</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>attenMaskOptional</td>
        <td>可选输入</td>
        <td>atten_mask，当前暂不支持，应传nullptr。</td>
        <td>BOOL</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>cuSeqLengthsOptional</td>
        <td>可选输入</td>
        <td>TND layout下query的累积序列长度，shape为(B+1,)。</td>
        <td>INT64</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>cuSeqLengthsKvOptional</td>
        <td>可选输入</td>
        <td>TND layout下key/value的累积序列长度，shape为(B+1,)。</td>
        <td>INT64</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>sequsedQOptional</td>
        <td>可选输入</td>
        <td>表示每个batch实际使用的query长度。</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>sequsedKvOptional</td>
        <td>可选输入</td>
        <td>表示每个batch实际使用的kv长度。</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>blockShape</td>
        <td>属性</td>
        <td>稀疏块形状[blockShapeX, blockShapeY]，当前支持[1, 128]。</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>isPackedGqa</td>
        <td>属性</td>
        <td>同一group内qHead是否共享稀疏pattern，当前仅支持1。</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>qInputLayout</td>
        <td>属性</td>
        <td>query侧layout格式。</td>
        <td>STRING</td>
        <td>-</td>
    </tr>
    <tr>
        <td>kvInputLayout</td>
        <td>属性</td>
        <td>key/value侧layout格式。</td>
        <td>STRING</td>
        <td>-</td>
    </tr>
    <tr>
        <td>scaleValue</td>
        <td>属性</td>
        <td>缩放系数，建议值为1/sqrt(D)。</td>
        <td>FLOAT</td>
        <td>-</td>
    </tr>
    <tr>
        <td>maskType</td>
        <td>属性</td>
        <td>mask模式，当前仅支持1。</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>softmaxPrecision</td>
        <td>属性</td>
        <td>Softmax精度级别，取值0或1，当前实现传0。</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>windowSizeLeft</td>
        <td>属性</td>
        <td>滑窗向前包含token数，不使能时须为-1。</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>windowSizeRight</td>
        <td>属性</td>
        <td>滑窗向后包含token数，不使能时须为-1。</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>dQuery</td>
        <td>输出</td>
        <td>query的梯度。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>dKey</td>
        <td>输出</td>
        <td>key的梯度。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>dValue</td>
        <td>输出</td>
        <td>value的梯度。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
    </tr>
    </tbody>
</table>

## 约束说明

- 确定性计算：
  - GenericBlockSparseAttentionGrad外部接口默认为非确定性实现，暂不支持确定性实现，确定性计算配置后不会生效。
- 公共约束
    - 参数query、key、value、dout、out、dQuery、dKey、dValue的数据类型应保持一致，支持FLOAT16和BFLOAT16。
    - 参数lse的数据类型应为FLOAT32。
    - 参数rsvdBlockIdx、rsvdBlockCount、sequsedQOptional、sequsedKvOptional的数据类型应为INT32。
    - 参数cuSeqLengthsOptional、cuSeqLengthsKvOptional、metadata的数据类型应为INT64。
    - qInputLayout和kvInputLayout当前支持TND、BNSD、BSND，且必须保持一致。
    - 当qInputLayout为TND时，需要传入cuSeqLengthsOptional；当kvInputLayout为TND时，需要传入cuSeqLengthsKvOptional。
    - 须先调用GenericBlockSparseAttentionGradMetadata生成metadata，再调用本算子。
    - Softmax LSE的head/seq轴语义须与query布局一致。
    - `rsvdBlockIdx`第4维maxS1应≥`rsvdBlockCount`中所有元素的最大值。
    - `windowSizeLeft`和`windowSizeRight`不使能时必须为-1。
    - `isPackedGqa`当前仅支持1。
    - `attenMaskOptional`当前暂不支持，应传nullptr。
    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
        <col style="width: 100px">
        <col style="width: 700px">
        <col style="width: 350px">
        </colgroup>
        <thead>
            <tr>
                <th>maskType</th>
                <th>含义</th>
                <th>备注</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>1</td>
            <td>causal mask（RIGHT_DOWN_CAUSAL）</td>
            <td>支持；attenMaskOptional应传nullptr，windowSizeLeft/Right为-1/-1</td>
        </tr>
        </tbody>
    </table>

- 规格约束
    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
        <col style="width: 100px">
        <col style="width: 550px">
        <col style="width: 500px">
        </colgroup>
        <thead>
            <tr>
                <th>规格项</th>
                <th>规格</th>
                <th>规格说明</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>D</td>
            <td>128</td>
            <td>query/key/value最后一维固定为128。</td>
        </tr>
        <tr>
            <td>N1</td>
            <td>1~128</td>
            <td>须能被N2整除。</td>
        </tr>
        <tr>
            <td>N2</td>
            <td>1~128</td>
            <td>-</td>
        </tr>
        <tr>
            <td>layout</td>
            <td>TND/BNSD/BSND</td>
            <td>q与kv layout须一致。</td>
        </tr>
        <tr>
            <td>blockShape</td>
            <td>[1, 128]</td>
            <td>blockShapeX仅支持1；blockShapeY当前仅支持128。</td>
        </tr>
        <tr>
            <td>isPackedGqa</td>
            <td>1</td>
            <td>当前仅支持1。</td>
        </tr>
        <tr>
            <td>maskType</td>
            <td>1</td>
            <td>当前仅支持1。</td>
        </tr>
        </tbody>
    </table>

  - 参数B、S1、S2的支持情况:
    - <term>Ascend 950PR/Ascend 950DT</term>：B>0，S1>0，S2>0。

- 典型值
    <table style="undefined;table-layout: fixed; width: 900px"><colgroup>
        <col style="width: 100px">
        <col style="width: 800px">
        </colgroup>
        <thead>
            <tr>
                <th>规格项</th>
                <th>典型值</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>query</td>
            <td>N1=1~128，D=128</td>
        </tr>
        <tr>
            <td>key/value</td>
            <td>N2=1~128，D=128，且N1%N2==0</td>
        </tr>
        <tr>
            <td>blockShape</td>
            <td>[1, 128]</td>
        </tr>
        <tr>
            <td>metadata</td>
            <td>长度≥80+B×N1×J×4</td>
        </tr>
        </tbody>
    </table>

## 调用说明

| 调用方式           | 调用样例                                                                                    | 说明                                                                                                  |
|----------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| aclnn调用 | [aclnnGenericBlockSparseAttentionGrad](./docs/aclnnGenericBlockSparseAttentionGrad.md) | 通过[aclnnGenericBlockSparseAttentionGrad](./docs/aclnnGenericBlockSparseAttentionGrad.md)接口方式调用GenericBlockSparseAttentionGrad算子。 |
| aclnn示例 | [test_aclnn_generic_block_sparse_attention_grad](./examples/test_aclnn_generic_block_sparse_attention_grad.cpp) | 通过aclnn两段式接口完成Metadata生成与Grad计算的可运行示例。 |
| torch extension调用 | [generic_block_sparse_attention_grad](./torch_extension/generic_block_sparse_attention_grad.py) | 通过torch extension接口进行功能和精度验证。 |
