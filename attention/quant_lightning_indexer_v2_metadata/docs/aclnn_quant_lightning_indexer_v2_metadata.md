# aclnnQuantLightingIndexerV2Metadata

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
|<term>Ascend 950PR/Ascend 950DT</term> |      √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    x     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    x     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：该算子为AICPU算子，是aclnnQuantLightningIndexerV2算子的前置算子。根据aclnnQuantLightningIndexerV2算子的输入信息，计算并输出负载均衡结果。输出结果可以作为aclnnQuantLightningIndexerV2算子的输入，减少aclnnQuantLightningIndexerV2算子的执行耗时。

  **该算子不建议单独使用，建议与aclnnQuantLightningIndexerV2算子配合使用，形成完整的工作流。**
    1. 接受aclnnQuantLightningIndexerV2算子输入数据shape信息，包含batchSize，qSeqlen，kSeqlen, mask。通过对输入分块并模拟计算耗时，均匀分配分块到可用核上，以降低aclnnQuantLightningIndexerV2算子的整体计算耗时，并提高硬件利用率。
    2. 分配结果输出后，后续作为输入供aclnnQuantLightningIndexerV2算子使用。

- 基本块计算公式：

  $$
  \begin{aligned}
    BN2轴长度 \quad \text{BN2\_Num} &= BatchSize * numHeadsK  \\
    G &= numHeadsK / numHeadsQ          \\
    M轴基本块长度 \quad mBaseSize &= 256 \\
    S2基本块长度 \quad s2BaseSize &= 128 \\
    当前Batch的S1G轴(M轴)长度 \quad M_{batchIdx} &= G * ActualSeqLenQuery(batchIdx) \\
    当前Batch的S1G轴(M轴)切分数量 \quad \text{M\_Num}_{batchIdx} &= (S1G_{batchIdx} + mBaseSize - 1) / mBaseSize \\
    当前Batch的S2轴长度量 \quad \text{S2}_{batchIdx} &= ActualSeqLenKey(batchIdx) \\
    当前Batch的S2轴切分数量 \quad \text{S2\_Num}_{batchIdx} &= (\text{S2}_{batchIdx} + s2BaseSize - 1) / s2BaseSize \\
    总块数 BlockNum &= \sum_{bn2=0}^{\text{BN2\_Num}}\sum_{m=0}^{\text{M\_Num}}\text{S2\_Num}
  \end{aligned}
  $$

## aclnnQuantLightingIndexerV2MetadataGetWorkspaceSize
## 参数说明

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1600px"><colgroup>
  <col style="width: 150px">
  <col style="width: 100px">
  <col style="width: 350px">
  <col style="width: 150px">
  <col style="width: 70px">
  <col style="width: 70px">
  <col style="width: 190px">
  <col style="width: 80px">
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
      <td>cuSeqlensQOptional</td>
      <td>可选输入</td>
      <td>表示不同Batch中query的有效Sequence Length</br>TND场景下必传，以该入参的数量作为Batch值</br>第一个值为额外值并固定为0。</td>
      <td>支持空Tensor</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape固定为(B+1)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>cuSeqlensKOptional</td>
      <td>可选输入</td>
      <td>表示不同Batch中key的有效Sequence Length</br>TND场景下必传，以该入参的数量作为Batch值</br>第一个值为额外值并固定为0</td>
      <td>支持空Tensor</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape固定为(B+1)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sequsedQOptional</td>
      <td>可选输入</td>
      <td>表示不同Batch中query实际参与运算的Sequence Length</td>
      <td>支持空Tensor</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape固定为(B)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sequsedKOptional</td>
      <td>可选输入</td>
      <td>表示不同Batch中key实际参与运算的Sequence Length</td>
      <td>支持空Tensor</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape固定为(B)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>cmpResidualKOptional</td>
      <td>可选输入</td>
      <td>预留接口参数。表示不同Batch中key的Sequence Length的余数</td>
      <td>支持空Tensor</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape固定为(B)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>numHeadsQ</td>
      <td>必选输入</td>
      <td>表示query的head个数</td>
      <td>支持非负数</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeadsK</td>
      <td>必选输入</td>
      <td>表示key的head个数</td>
      <td>支持非负数</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headDim</td>
      <td>必选输入</td>
      <td>表示toekn数</td>
      <td>支持非负数</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topk</td>
      <td>必选输入</td>
      <td>表示从Query中筛选出的关键稀疏token的个数</td>
      <td>支持非负数</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>qQuantMode</td>
      <td>必选输入</td>
      <td>表示query的量化模式</td>
      <td>2: per-toekn-head</br> 3: group-scaling</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>kQuantMode</td>
      <td>必选输入</td>
      <td>表示key的量化模式</td>
      <td>2: per-toekn-head</br> 3: group-scaling</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batchSizeOptional</td>
      <td>可选输入</td>
      <td>表示batch数量</td>
      <td>支持非负数</br>默认为0</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenQOptional</td>
      <td>可选输入</td>
      <td>表示query的最长seqlen</td>
      <td>支持非负数</br>默认为0</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenKOptional</td>
      <td>可选输入</td>
      <td>表示key的最长seqlen</td>
      <td>支持非负数</br>默认为0</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQOptional</td>
      <td>可选输入</td>
      <td>表示query的排列格式</td>
      <td>支持 BSND/TND</br>默认为BSND</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKOptional</td>
      <td>可选输入</td>
      <td>表示key的排列格式</td>
      <td>支持 BSND/TND/PA_BBND</br>默认为BSND</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskModeOptional</td>
      <td>可选输入</td>
      <td>表示sparse模式</td>
      <td>0: No mask</br>3: Causal </br>默认为0 </td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpRatioOptional</td>
      <td>可选输入</td>
      <td>预留接口参数,表示key的压缩率</td>
      <td>取值范围[1, 128]</td>
      <td>INT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metaData</td>
      <td>必选输出</td>
      <td>表示负载均衡结果输出</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape固定为(1024)</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>必选输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>必选输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## aclnnQuantLightingIndexerV2Metadata
- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 168px">
    <col style="width: 128px">
    <col style="width: 854px">
    </colgroup>
    <thead>
        <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
        </tr></thead>
    <tbody>
        <tr>
        <td>workspace</td>
        <td>输入</td>
        <td>在Device侧申请的workspace内存地址。</td>
        </tr>
        <tr>
        <td>workspaceSize</td>
        <td>输入</td>
        <td>在Device侧申请的workspace大小，由第一段接口aclnnQuantLightingIndexerV2MetadataGetWorkspaceSize获取。</td>
        </tr>
        <tr>
        <td>executor</td>
        <td>输入</td>
        <td>op执行器，包含了算子计算流程。</td>
        </tr>
        <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的Stream。</td>
        </tr>
    </tbody>
    </table>

- **返回值：**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
<details>
<summary><a id="Mask"></a>Mask</summary>
    &nbsp;&nbsp;<table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
        <col style="width: 165px">
        <col style="width: 625px">
        <col style="width: 360px">
        </colgroup>
        <thead>
            <tr>
                <th>mask_mode</th>
                <th>含义</th>
                <th>备注</th>
            </tr>
        </thead>
        <tbody>
        <tr>
            <td>0</td>
            <td>无mask</td>
            <td>-</td>
        </tr>
        <tr>
            <td>3</td>
            <td>rightDownCausal模式的mask，对应以右顶点为划分的下三角场景，需要传入优化后的attenmask矩阵</td>
        </tr>
        </tbody>
    </table>
</details>

<details>
<summary><a id="特殊约束"></a>特殊约束</summary>

  - Batch取值规则
    - 优先获取sequsedKOptional中的batch的shape信息
    - 如果未传入sequsedKOptional，优先获取cuSeqlensQOptional中的batch的shape信息
    - 如果未传入sequsedKOptional，且layoutQOptional为TND，则必获取cuSeqlensQOptional中的batch的shape信息
    - 除上所述，使用batchSizeOptional
  - Seqlen取值规则
    - 优先获取sequsedKOptional中的seqlen的shape信息
    - 如果未传入sequsedKOptional，且layoutKOptional为TND，则必获取cuSeqlensQOptional中的batch的shape信息
    - 除上所述，使用maxSeqlenQOptional
    - key与query的获取规则一致
  - cmpResidualKOptional启用规则
    - 预留参数，暂无效果，只做约束
    - 如果cmpRatioOptional不为1，且mask为3，则必须传入cmpResidualKOptional
  - layout约束
    - 当layoutKOptional为PA_BBND时，layoutQOptional可以任意取值
    - 除上所述，layoutQOptional必须与layoutKOptional保持一致

</details>

## 调用说明

| 调用方式  | 样例代码                                                                | 说明                                                                                          |
| ----------- | ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| aclnn接口 | [test_aclnn_quant_lightning_indexer_v2_metadata.cpp](./examples/test_aclnn_quant_lightning_indexer_v2_metadata.cpp) | 通过[`aclnnQuantLightningIndexerV2Metadata`](./docs/aclnnQuantLightningIndexerV2Metadata.md)接口方式调用 |
