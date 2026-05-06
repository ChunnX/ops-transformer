# ChunkGatedDeltaRule

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      ×     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：完成chunk版的Gated Delta Rule计算。

- 计算公式：

  Gated Delta Rule（门控Delta规则，GDR）是一种应用于循环神经网络的算子，也被应用于一种线性注意力机制中。在每个时间步 $t$，GDR根据当前的输入 $q_t$、$k_t$、$v_t$、上一个隐藏状态 $S_{t-1}$、衰减系数 $\alpha_t$ 以及更新强度 $\beta_t$，计算当前的注意力输出 $o_t$ 和新的隐藏状态 $S_t$，其计算公式如下：
  $$
  S_t := S_{t-1}(\alpha_t(I - \beta_t k_t k_t^T)) + \beta_t v_t k_t^T = \alpha_t S_{t-1} + \beta_t (v_t - \alpha_t S_{t-1}k_t)k_t^T
  $$
  $$
  o_t := S_t (q_t \cdot scale)
  $$

  其中，$S_{t-1},S_t \in R^{D_v \times D_k}$，$q_t, k_t \in R^{D_k}$，$v_t \in R^{D_v}$，$\alpha_t \in R$，$\beta_t \in R$，$o \in R^{D_v}$。
  
  Chunked Gated Delta Rule (CGDR)是GDR的chunk版实现([参考论文](https://arxiv.org/abs/2412.06464))，它通过将输入序列切块，实现了一定的并行效果，在长上下文场景其计算效率相对Recurrent Gated Delta Rule更高，适用于prefill阶段。输入一个长度为L的序列，CGDR算子可以计算出每一步的输出 $o_t, t \in \{1, .., L\}$ 以及最终的状态矩阵 $S_L$。

为了方便理解后续支持的具体排布格式（如 BNSD、TND 等），此处先对排布格式中各缩写字母所代表的维度含义进行统一说明：

* B：表示输入样本批量大小（Batch）。
* T：令$L_i$ 表示第i个序列的长度，则 $T=\sum_i^B L_i$ 表示累积序列长度。
* Nk：表示query和key的头数。
* Nv：表示value的头数。
* Dk：表示query和key的hidden size。
* Dv：表示value的hidden size。

## 参数说明

<table style="undefined;table-layout: fixed; width: 900px"><colgroup>
<col style="width: 180px">
<col style="width: 120px">
<col style="width: 200px">
<col style="width: 300px">
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
    <td>公式中的q。</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>key</td>
    <td>输入</td>
    <td>公式中的输入k。</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>value</td>
    <td>输入</td>
    <td>公式中的输入v。</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>beta</td>
    <td>输入</td>
    <td>公式中的β。</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>initial_state</td>
    <td>输入</td>
    <td>初始状态矩阵，公式中的输入S_0。</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>actual_seq_lengths</td>
    <td>输入</td>
    <td>每个batch的序列长度。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>g</td>
    <td>输入</td>
    <td>衰减系数，公式中的α=e^g</td>
    <td>FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>out</td>
    <td>输出</td>
    <td>每一步的attention结果，公式中的o_t。</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>final_state</td>
    <td>输出</td>
    <td>最终的状态矩阵，公式中的S_L。</td>
    <td>BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>scale_value</td>
    <td>可选属性</td>
    <td>query的缩放因子，公式中的scale。默认为1.0</td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
</tbody>
</table>

## 约束说明

* 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
* 当前仅支持“TND”的layout，其中query和key的形状为（T，Nk，Dk），value和out的形状为（T，Nv，Dv），beta和g的形状为（T，Nv），actual_seq_lengths的形状为（B，），initial_state和final_state的形状为（B，Nv，Dv，Dk）。各维度大小需满足约束：
  - $0 \lt Nv \le 64$，$0 \lt Nk \le 64$，且Nv是Nk的整数倍。
  - $0 \lt Dv \le 128$, $0 \lt Dk \le 128$。
  - $B \gt 0$, $T \gt 0$。
* 由于算法特性，用户需保障以下数值约束，否则计算结果可能出现溢出：
  - $0 < query < 1$。
  - $0 < key < 1$。
  - $g < 0$。
  - $0 < beta < 1$。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn | [test_aclnn_chunk_gated_delta_rul.cpp](./examples/test_aclnn_chunk_gated_delta_rule.cpp) | 通过[aclnnChunkGatedDeltaRule](./docs/aclnnChunkGatedDeltaRule.md)调用aclnnChunkGatedDeltaRule算子 |
