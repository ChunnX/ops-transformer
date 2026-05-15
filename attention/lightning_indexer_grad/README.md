# LightningIndexerGrad

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      x     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|      √     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|      √     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- 算子功能：`lightning_indexer_grad`基于一系列操作得到每一个token对应的Top-$k$个位置。

- 计算公式：
  LightningIndexer反向计算公式如下：

  $$
  S = Relu(Matmul(Query, Gather(Key, Indices)))
  $$

  $$
  Y = Dy*Weights
  $$

  $$
  dW = Reduce(S * dy)
  $$

  $$
  dQ = Matmul(ReluGrad(Y, S), Gather(Key, Indices))
  $$

  $$
  dK = ScatterAdd(Matmul(ReluGrad(Y, S), Q), Indices)
  $$

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1565px">
  <colgroup>
    <col style="width: 146px">
    <col style="width: 135px">
    <col style="width: 326px">
    <col style="width: 246px">
    <col style="width: 275px">
    <col style="width: 101px">
    <col style="width: 190px">
    <col style="width: 146px">
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
  </thead>
  <tbody>
    <tr>
      <td>query</td>
      <td>输入</td>
      <td>公式中的query。</td>
      <td>shape支持[B, S1, N1, D]/[T1, N1, D]。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>3-4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>key</td>
      <td>输入</td>
      <td>公式中的key。</td>
      <td>shape支持[B, S2, N2, D]/[T2, N2, D]。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>3-4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dy</td>
      <td>输入</td>
      <td>公式中的value。</td>
      <td>shape支持[B, S1, N1, D]/[T1, N1, D]。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>3-4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>sparseIndices</td>
      <td>输入</td>
      <td>公式中的sparseIndices。</td>
      <td>shape支持[B, S1, K]/[T1, K]。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weights</td>
      <td>输入</td>
      <td>公式中的weights。</td>
      <td>shape支持[B, S1, N1]/[T1, N1]。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>actualSeqLengthsQuery</td>
      <td>输入</td>
      <td>表示query每个Batch S的累加和长度。</td>
      <td>TND排布时需要输入，其余场景输入nullptr。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>actualSeqLengthsKey</td>
      <td>输入</td>
      <td>表示key每个Batch S的累加和长度。</td>
      <td>TND排布时需要输入，其余场景输入nullptr。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>headNum</td>
      <td>输入</td>
      <td>代表head个数。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>layout</td>
      <td>输入</td>
      <td>代表query/key的数据排布格式。</td>
      <td>当前支持TND/BSND。</td>
      <td>String</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparseMode</td>
      <td>输入</td>
      <td>表示sparse模式。</td>
      <td>支持取值0/3。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>preTokens</td>
      <td>输入</td>
      <td>用于稀疏计算 ，表示slides window的左边界。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nextTokens</td>
      <td>输入</td>
      <td>用于稀疏计算 ，表示slides window的右边界。</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deterministic</td>
      <td>输入</td>
      <td>表示当前是否支持确定性计算。</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dQuery</td>
      <td>输出</td>
      <td>dQuery梯度。</td>
      <td>数据类型与query一致。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>3-4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dKey</td>
      <td>输出</td>
      <td>dKey梯度。</td>
      <td>数据类型与key一致。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>3-4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dWeights</td>
      <td>输出</td>
      <td>dWeights梯度。</td>
      <td>数据类型与weights一致。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

## 约束说明

- 确定性计算：
  - aclnnLightningIndexerGrad默认非确定性实现，不支持通过aclrtCtxSetSysParamOpt开启确定性。
- 该接口与PyTorch配合使用时，需要保证CANN相关包与PyTorch相关包的版本匹配。
- inputLayout支持TND/BSND。
- 关于数据shape的约束，以Layout的BSND举例。其中：
  
  - B（Batchsize）：取值范围为1\~1024。
  - N（Head-Num）：取值为64。
  - G（Group）：取值为64。
  - S1（Seq-LengthQ）：取值范围为1\~128K。
  - S2（Seq-LengthK）：取值范围为topK\~128K。
  - D（Head-Dim）：取值为128。
  - TopK：取值为2048

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
    <a href="./examples/test_aclnn_lightning_indexer_grad.cpp">test_aclnn_lightning_indexer_grad
    </a>
    </td>
    <td class="tg-lboi" rowspan="6">
    通过
    <a href="./docs/aclnnLightningIndexerGrad.md">aclnnLightningIndexerGrad
    </a>
    接口方式调用算子
    </td>
  </tr>
</tbody></table>