# aclnnStemOamPrepVarlenQ

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

> 仅支持 arch35 (DAV_3510) 架构芯片。

## 功能说明

- 接口功能：完成 Stem OAM block-sparse attention 中 Q 侧预处理计算。将变长 Q tensor 从 paged 存储格式转化为按 stem block (128 token) 分组的 flattened qFlat 输出，供后续 OAM score 计算。
- 计算公式：

  阶段1 Scale Fusion:
  $$q\_scale[b, h, pos] = qscale[b, h, pos]$$

  阶段2 De-page Varlen:
  $$Q\_dense[b] = Cast(q[cu\_seqlens\_q[b]:cu\_seqlens\_q[b]+q\_len[b], :, :], \text{FP32})$$

  阶段3 Weighted Group Sum (自然顺序，NO flip):
  $$Q\_group\_sum[b,h,qb,g,:] = \sum_{r=0}^{R-1} Q\_blocks[b,h,qb,r,g,:] \times q\_scale[b,h,position(qb,r,g)]$$

  阶段4 Flatten:
  $$qflat[b, h, qb, g \times D : (g+1) \times D] = Q\_group\_sum[b, h, qb, g, :]$$

  阶段5 Cast输出:
  $$qflat\_out = qflat.to(\text{BF16})$$

- 关键特性：Q 侧 stride 维度为自然顺序（不翻转），与 K 侧的翻转处理不同。

## 函数原型

每个算子分为两段式接口，必须先调用"aclnnStemOamPrepVarlenQGetWorkspaceSize"接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用"aclnnStemOamPrepVarlenQ"接口执行计算。

```cpp
aclnnStatus aclnnStemOamPrepVarlenQGetWorkspaceSize(
  const aclTensor   *q,
  const aclIntArray *qSeqLens,
  const aclIntArray *cuSeqLensQ,
  const aclTensor   *qScale,
  int64_t            stemBlockSize,
  int64_t            stemStride,
  aclTensor         *qFlat,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```

```cpp
aclnnStatus aclnnStemOamPrepVarlenQ(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

## aclnnStemOamPrepVarlenQGetWorkspaceSize

- **参数说明**

  <table style="table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 350px">
  <col style="width: 250px">
  <col style="width: 100px">
  <col style="width: 100px">
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
      <td>q（aclTensor*）</td>
      <td>输入</td>
      <td>表示变长Q tensor，所有batch的token拼接存储。对应公式中q。</td>
      <td><ul><li>支持空Tensor，空Tensor时直接返回，workspaceSize为0。</li><li>最后一维必须等于128。</li><li>数据类型仅支持FLOAT8_E4M3FN。</li></ul></td>
      <td>FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>3 ([total_tokens, H_q, D])，D=128</td>
      <td>-</td>
    </tr>
    <tr>
      <td>qSeqLens（aclIntArray*）</td>
      <td>输入</td>
      <td>表示每个batch的Q序列长度。对应公式中q_seq_lens。</td>
      <td><ul><li>长度等于batch，取值范围为(0, 1024]。</li><li>每个值≥0。</li><li>长度为0时按空Tensor处理，直接返回。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>1 ([batch])</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cuSeqLensQ（aclIntArray*）</td>
      <td>输入</td>
      <td>表示Q的累积序列长度偏移量，用于varlen索引。对应公式中cu_seqlens_q。</td>
      <td><ul><li>长度等于batch+1。</li><li>cuSeqLensQ[0]必须为0。</li><li>cuSeqLensQ[batch]必须等于total_tokens。</li><li>单调递增（允许相等）。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>1 ([batch+1])</td>
      <td>-</td>
    </tr>
    <tr>
      <td>qScale（aclTensor*）</td>
      <td>输入</td>
      <td>表示Q的per-token scale factor。对应公式中qscale。</td>
      <td><ul><li>必填，q为FLOAT8_E4M3FN时不可为nullptr。</li><li>shape为[total_tokens, H_q]。</li><li>数据类型必须为FLOAT。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>2 ([total_tokens, H_q])</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stemBlockSize（int64_t）</td>
      <td>输入</td>
      <td>表示stem block大小,对应公式中B。控制每个stem block的token数量,决定Q Processing的分组粒度。</td>
      <td><ul><li>仅支持128。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stemStride（int64_t）</td>
      <td>输入</td>
      <td>表示stem stride大小,对应公式中S。控制stem block内stride group的token数量,决定qFlat的维度粒度。</td>
      <td><ul><li>仅支持16。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>qFlat（aclTensor*）</td>
      <td>输出</td>
      <td>表示flattened Q输出，供OAM score计算使用。对应公式中qflat_out。</td>
      <td><ul><li>数据类型固定为BFLOAT16。</li><li>shape为[batch, H_q, max_Qb, kflat_dim]，其中max_Qb=ceil(max(qSeqLens)/stemBlockSize)，kflat_dim=stemStride×D=16×128=2048。</li></ul></td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>4 ([batch, H_q, max_Qb, kflat_dim])</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 550px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>q、qSeqLens、cuSeqLensQ、qFlat、workspaceSize、executor存在空指针。</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>q的数据类型不为FLOAT8_E4M3FN。</td>
    </tr>
    <tr>
      <td>q为FP8时qScale为nullptr。</td>
    </tr>
    <tr>
      <td>qScale的数据类型不为FLOAT。</td>
    </tr>
    <tr>
      <td>qFlat的数据类型不为BFLOAT16。</td>
    </tr>
    <tr>
      <td>q的维度不为3或最后一维不等于128；qScale维度不为2；qFlat维度不为4。</td>
    </tr>
    <tr>
      <td>qSeqLens长度超出(0, 1024]范围；cuSeqLensQ长度不等于batch+1、cuSeqLensQ[0]不为0、非单调递增、或cuSeqLensQ[batch]不等于total_tokens。</td>
    </tr>
    <tr>
      <td>stemBlockSize不为128或stemStride不为16。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER</td>
      <td>361001</td>
      <td>内部错误，如Tiling计算失败或Kernel查找失败。</td>
    </tr>
  </tbody></table>

## aclnnStemOamPrepVarlenQ

- **参数说明**

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 700px">
  </colgroup>
  <thead>
    <tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr>
  </thead>
  <tbody>
    <tr><td>workspace</td><td>输入</td><td>在Device侧申请的workspace内存地址。</td></tr>
    <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnStemOamPrepVarlenQGetWorkspaceSize获取。</td></tr>
    <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
    <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

- 确定性说明：aclnnStemOamPrepVarlenQ默认确定性实现。
- <term>Ascend 950PR/Ascend 950DT</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：仅支持arch35架构，不支持arch22。
- q的最后一维D必须等于128。
- stemBlockSize仅支持128，stemStride仅支持16。派生值：R = stemBlockSize / stemStride = 8, kflat_dim = stemStride × D = 2048。
- qFlat输出维度kflat_dim = stemStride × D = 16 × 128 = 2048。
- Q侧stride维度为自然顺序（g ∈ [0, S-1]），不翻转（与K侧处理不同）。
- 当qSeqLens中某batch值为0时，该batch对应的qFlat输出填充为零。
- q仅支持FLOAT8_E4M3FN数据类型，qScale必填且数据类型为FLOAT，qFlat输出固定为BFLOAT16。
- 支持空Tensor输入：当q为空或qSeqLens长度为0时，直接返回，workspaceSize为0。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。

<term>Ascend 950PR/Ascend 950DT</term>、<term>Atlas A3 推理系列产品</term>：

```Cpp
// 1. 创建输入tensor
int64_t total_tokens = 256;
int64_t H_q = 32;
int64_t D = 128;
int64_t batch = 2;

int64_t qDims[] = {total_tokens, H_q, D};
aclTensor *q = aclCreateTensor(qDims, 3, ACL_FLOAT8_E4M3FN, ACL_FORMAT_ND, nullptr, 0);

int64_t qScaleDims[] = {total_tokens, H_q};
aclTensor *qScale = aclCreateTensor(qScaleDims, 2, ACL_FLOAT, ACL_FORMAT_ND, nullptr, 0);

int64_t qSeqLens_data[] = {128, 128};
int64_t cuSeqLensQ_data[] = {0, 128, 256};
aclIntArray *qSeqLens = aclCreateIntArray(qSeqLens_data, 2);
aclIntArray *cuSeqLensQ = aclCreateIntArray(cuSeqLensQ_data, 3);

int64_t stemBlockSize = 128;
int64_t stemStride = 16;

int64_t max_Qb = 1;
int64_t qFlatDims[] = {batch, H_q, max_Qb, 2048};
aclTensor *qFlat = aclCreateTensor(qFlatDims, 4, ACL_BFLOAT16, ACL_FORMAT_ND, nullptr, 0);

// 2. 获取workspace大小
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
aclnnStatus ret = aclnnStemOamPrepVarlenQGetWorkspaceSize(
    q, qSeqLens, cuSeqLensQ, qScale,
    stemBlockSize, stemStride, qFlat,
    &workspaceSize, &executor);

// 3. 申请workspace
void *workspace = nullptr;
aclrtMalloc(&workspace, workspaceSize, ACL_RT_MEM_MALLOC_NORMAL_ONLY);

// 4. 执行计算
ret = aclnnStemOamPrepVarlenQ(workspace, workspaceSize, executor, stream);

// 5. 释放资源
aclrtFree(workspace);
aclDestroyTensor(q);
aclDestroyTensor(qScale);
aclDestroyIntArray(qSeqLens);
aclDestroyIntArray(cuSeqLensQ);
aclDestroyTensor(qFlat);
```
