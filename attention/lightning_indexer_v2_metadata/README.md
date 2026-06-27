# aclnnLightningIndexerV2Metadata

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT |      x      |
| Atlas A3训练系列产品/Atlas A3推理系列产品     |   √     |
| Atlas A2训练系列产品/Atlas A2推理系列产品 |     √       |
| Atlas 200I/500 A2推理产品                      |    x     |
| Atlas推理系列产品                              |    x     |
| Atlas训练系列产品                              |    x     |

## 功能说明

- 接口功能：该接口为AI CPU算子接口，是aclnnLightningIndexerV2算子的前置算子接口。根据aclnnLightningIndexerV2算子接口的输入信息，计算并输出负载均衡结果。输出结果可以作为aclnnLightningIndexerV2算子接口的输入，减少aclnnLightningIndexerV2算子接口的执行耗时。

  **该算子不建议单独使用，建议与aclnnLightningIndexerV2算子配合使用，形成完整的工作流。**
    1. 接受aclnnLightningIndexerV2算子接口输入数据shape信息，包含batchSize、qSeqlen、kSeqlen、mask。通过对输入分块并模拟计算耗时，均匀分配分块到可用核上，以降低aclnnLightningIndexerV2算子的整体计算耗时，并提高硬件利用率。
    2. 分配结果输出后，后续作为输入供aclnnLightningIndexerV2算子使用。
    3. 分配结果包含每个AIC核基本块的起始点和终止点，已经每个AIV核的FD任务信息。详细内容可以参考[调用示例](#调用示例)。

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1600px"><colgroup>
  <col style="width: 150px">
  <col style="width: 100px">
  <col style="width: 350px">
  <col style="width: 200px">
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 190px">
  <col style="width: 100px">
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
      <td>输入</td>
      <td>表示不同Batch中Query的有效Sequence Length</br>TND场景下必传，以该入参的数量作为Batch值</br>第一个值为额外值并固定为0。</td>
      <td>支持空Tensor</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape固定为(B+1，)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cuSeqlensKOptional</td>
      <td>输入</td>
      <td>表示不同Batch中Key的有效Sequence Length</br>TND场景下必传，以该入参的数量作为Batch值</br>第一个值为额外值并固定为0。</td>
      <td>支持空Tensor</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape固定为(B+1，)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedQOptional</td>
      <td>输入</td>
      <td>表示不同Batch中Query实际参与运算的Sequence Length。</td>
      <td>支持空Tensor</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape固定为(B，)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sequsedKOptional</td>
      <td>输入</td>
      <td>表示不同Batch中Key实际参与运算的Sequence Length。</td>
      <td>支持空Tensor</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape固定为(B，)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cmpResidualKOptional</td>
      <td>输入</td>
      <td>预留参数，表示不同Batch中Key的Sequence Length的余数，当前不影响算子计算效果。</br>如果cmpRatio不为1，且mask为3，则必须传入cmpResidualKOptional。</td>
      <td>支持空Tensor</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape固定为(B，)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>numHeadsQ</td>
      <td>输入</td>
      <td>表示Query的head个数。</td>
      <td>支持非负数</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numHeadsK</td>
      <td>输入</td>
      <td>表示Key的head个数。</td>
      <td>支持非负数</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>headDim</td>
      <td>输入</td>
      <td>表示token数。</td>
      <td>支持非负数</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>topk</td>
      <td>输入</td>
      <td>表示从Query中筛选出的关键稀疏token的个数。</td>
      <td>支持非负数</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batchSize</td>
      <td>输入</td>
      <td>表示Batch数量。</td>
      <td>支持非负数</br>建议值为0</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenQ</td>
      <td>输入</td>
      <td>表示Query的最长Sequence Length。</td>
      <td>支持非负数</br>建议值为0</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxSeqlenK</td>
      <td>输入</td>
      <td>表示Key的最长Sequence Length。</td>
      <td>支持非负数</br>建议值为0</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutQOptional</td>
      <td>输入</td>
      <td>表示Query的排列格式。</td>
      <td>支持BSND、TND</br>建议值为BSND</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layoutKOptional</td>
      <td>输入</td>
      <td>表示Key的排列格式。</td>
      <td>支持BSND、TND、PA_BBND</br>建议值为BSND</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maskMode</td>
      <td>输入</td>
      <td>表示sparse模式。</td>
      <td>0: No mask</br>3: rightDownCausal模式的mask，对应以右顶点为划分的下三角场景 </br>建议值为0</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cmpRatio</td>
      <td>输入</td>
      <td>预留参数，表示Key的压缩率，当前不影响算子计算效果。</br>建议值1，表示无压缩。</td>
      <td>取值范围[1，128]</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>metaData</td>
      <td>输出</td>
      <td>表示负载均衡结果输出。</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1维，shape固定为(1024)</td>
      <td>×</td>
    </tr>
  </tbody>
  </table>

## 约束说明
  - aclnnLightningIndexerV2Metadata默认确定性实现

  - Batch取值规则
    - 优先获取sequsedQOptional中的Batch信息
    - 如果未传入sequsedQOptional，优先获取cuSeqlensQOptional中的Batch信息
    - 如果未传入sequsedQOptional，且layoutQOptional为TND，则必获取cuSeqlensQOptional中的Batch信息
    - 除上所述，使用batchSize
  - Sequence Length取值规则
    - 优先获取sequsedQOptional中的Sequence Length信息
    - 如果未传入sequsedQOptional，且layoutQOptional为TND，则必获取cuSeqlensQOptional中的Sequence Length信息
    - 除上所述，使用maxSeqlenQ
    - Key与Query的获取规则一致
  - layout约束
    - 当layoutKOptional为PA_BBND时，layoutQOptional可以任意取值
    - 除上所述，layoutQOptional必须与layoutKOptional保持一致
  - BSND场景
    - 当传入的layoutQOptional为"BSND"时，视为使用BSND场景
    - 在未传入cuSeqlensQOptional和sequsedQOptional的情况下，必传batchSize、maxSeqlenQ、maxSeqlenK参数
  - TND场景
    - 当传入的layoutQOptional为"TND"时，视为使用TND场景
    - 必传cuSeqlensQOptional、cuSeqlensKOptional参数

## 调用示例
| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_lightning_indexer_v2_metadata](./examples/test_aclnn_lightning_indexer_v2_metadata.cpp) | 通过[aclnnLightningIndexerV2Metadata](./docs/aclnnLightningIndexerV2Metadata.md)接口方式调用LightningIndexerV2Metadata算子。 |