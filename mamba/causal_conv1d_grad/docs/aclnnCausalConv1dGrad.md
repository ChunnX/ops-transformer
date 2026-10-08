# aclnnCausalConv1dGrad

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- **接口功能**：

    完成因果一维卷积前向算子（`aclnnCausalConv1dFn`，prefill 模式）的反向计算，用于训练反向传播。给定上游梯度 `gradY` 以及前向输入 `x`、`weight`、`bias`，计算对三个前向输入的全量梯度：`dx`（对输入序列的梯度）、`dweight`（对卷积权重的梯度）、`dbias`（对偏置的梯度）。由于前向未保存激活前的中间结果，反向在算子内部用 `x`、`weight`、`bias` 重算卷积得到 pre-activation，再求 SiLU 激活导数。仅支持 prefill 模式，不支持 decode/update 增量模式。

- **计算公式**：

    设 pre-activation 为 $z_{t,d}$，前向输出 $y_{t,d} = \text{Activation}(z_{t,d})$，其中 $W$ 为卷积核宽（支持2、3、4）：

    $$
    z_{t,d} = \sum_{j=0}^{W-1} w_{j,d} \cdot x_{t-j,d} + b_d
    $$

    SiLU 导数（$\sigma$ 为 sigmoid）：

    $$
    \text{silu}'(z) = \sigma(z)\,(1 + z\,(1 - \sigma(z)))
    $$

    pre-activation 梯度（`activation="none"` 时 $grad\_z = gradY$）：

    $$
    grad\_z_{t,d} = gradY_{t,d} \cdot \text{silu}'(z_{t,d})
    $$

    三个输出梯度：

    $$
    dbias_d = \sum_t grad\_z_{t,d}, \quad dweight_{j,d} = \sum_t grad\_z_{t,d} \cdot x_{t-j,d}, \quad dx_{t,d} = \sum_{j=0}^{W-1} grad\_z_{t+j,d} \cdot w_{j,d}
    $$

    其中 $dx$ 的求和中 $t+j$ 超出序列长度时对应 $grad\_z$ 取0，$dweight$ 的求和中 $t-j<0$ 时取历史状态（冷启动为0）。按 batch/序列分段独立计算，变长场景用 `queryStartLoc` 切分。

- **说明**：

  - 本算子与 `aclnnCausalConv1dFn` 配对，通过 torch autograd 注册实现训练反向传播。
  - 不输出历史状态缓存（`convStates`）的梯度。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用 `aclnnCausalConv1dGradGetWorkspaceSize` 接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用 `aclnnCausalConv1dGrad` 接口执行计算。

```cpp
aclnnStatus aclnnCausalConv1dGradGetWorkspaceSize(
    const aclTensor *gradY,
    const aclTensor *x,
    const aclTensor *weight,
    const aclTensor *biasOptional,
    const aclTensor *queryStartLocOptional,
    const aclTensor *convStatesOptional,
    const aclTensor *cacheIndicesOptional,
    const aclTensor *initialStateModeOptional,
    const char      *activation,
    int64_t          nullBlockId,
    aclTensor       *dx,
    aclTensor       *dweight,
    aclTensor       *dbiasOptional,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnCausalConv1dGrad(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnCausalConv1dGradGetWorkspaceSize

- **参数说明**：

  <table style="table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 230px">
  <col style="width: 100px">
  <col style="width: 320px">
  <col style="width: 400px">
  <col style="width: 180px">
  <col style="width: 80px">
  <col style="width: 130px">
  <col style="width: 110px">
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
      <td>gradY（aclTensor*）</td>
      <td>输入</td>
      <td>上游梯度，对应公式中gradY，表示对前向输出y的梯度。</td>
      <td><ul><li>不支持空tensor。</li><li>shape与前向输出y一致：固定batch为[batch, seqLen, dim]，变长为[cuSeqLen, dim]。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>前向输入序列，对应公式中x，用于重算pre-activation及计算dweight。</td>
      <td><ul><li>不支持空tensor。</li><li>shape与gradY一致。</li></ul></td>
      <td>数据类型与gradY一致</td>
      <td>ND</td>
      <td>2-3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>weight（aclTensor*）</td>
      <td>输入</td>
      <td>前向卷积权重，对应公式中w，用于重算pre-activation及计算dx。</td>
      <td><ul><li>不支持空tensor。</li><li>shape为[K, dim]，K∈{2,3,4}。</li></ul></td>
      <td>数据类型与gradY一致</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>biasOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>前向偏置，对应公式中b，用于重算pre-activation。</td>
      <td><ul><li>shape为[dim]。</li><li>传入空指针表示前向无偏置，此时dbiasOptional也应传空指针。</li></ul></td>
      <td>数据类型与gradY一致</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>queryStartLocOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>序列起始位置索引，记录各序列在拼接张量中的起始位置。</td>
      <td><ul><li>变长序列场景必须提供，固定batch场景传空指针。</li><li>shape为[batch+1]。queryStartLoc[0]必须为0，queryStartLoc[-1]必须等于cuSeqLen，值必须非递减。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>convStatesOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>历史状态缓存（只读），存储各序列的历史token数据。</td>
      <td><ul><li>当initialStateModeOptional中存在值为1的序列时须提供。</li><li>shape为[numCacheLines, stateLen, dim]，stateLen ≥ K-1。</li></ul></td>
      <td>数据类型与gradY一致</td>
      <td>ND</td>
      <td>3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>cacheIndicesOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>缓存索引，指定每个序列对应的历史状态在convStates中的索引。</td>
      <td><ul><li>不传时使用恒等映射。</li><li>shape为[batch]。值∈[0, numCacheLines)，或等于nullBlockId表示跳过该序列。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>initialStateModeOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>初始状态标志，表示各序列是否使用缓存的初始状态。</td>
      <td><ul><li>不传时全部从零初始化。</li><li>shape为[batch]。值：0=零初始化，1=使用缓存状态。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>activation（const char*）</td>
      <td>输入</td>
      <td>激活函数类型，须与前向一致。</td>
      <td>"silu"：计算SiLU导数；"none"：无激活，grad_z等于gradY。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>nullBlockId（int64_t）</td>
      <td>输入</td>
      <td>用于跳过不需要参与计算的batch。</td>
      <td>当cacheIndices[i]==nullBlockId时跳过该序列，对应dx段填零且不贡献dweight/dbias。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dx（aclTensor*）</td>
      <td>输出</td>
      <td>对输入x的梯度，对应公式中dx。</td>
      <td><ul><li>不支持空tensor。</li><li>shape与x一致。</li></ul></td>
      <td>数据类型与gradY一致</td>
      <td>ND</td>
      <td>2-3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>dweight（aclTensor*）</td>
      <td>输出</td>
      <td>对卷积权重weight的梯度，对应公式中dweight。</td>
      <td><ul><li>不支持空tensor。</li><li>shape为[K, dim]，与weight一致。</li></ul></td>
      <td>数据类型与gradY一致</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>dbiasOptional（aclTensor*）</td>
      <td>输出</td>
      <td>对偏置bias的梯度，对应公式中dbias。</td>
      <td><ul><li>shape为[dim]，与bias一致。</li><li>biasOptional传空指针时此处也传空指针。</li></ul></td>
      <td>数据类型与gradY一致</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回用户需要在Device侧申请的workspace大小。</td>
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
    </tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

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
      <td>传入的gradY、x、weight、dx、dweight是空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>
      输入和输出的数据类型不在支持的范围内。<br>
      gradY、x、weight、biasOptional、dx、dweight、dbiasOptional的数据类型不一致。<br>
      queryStartLocOptional、cacheIndicesOptional、initialStateModeOptional的数据类型不是INT32。<br>
      gradY与x的shape不一致，或dx与x的shape不一致。<br>
      输入、输出Tensor的shape不在支持的范围内。<br>
      dim不满足对齐要求（需满足 (dim * dtypeSize) % 32 == 0）。<br>
      K不在[2,4]范围内。<br>
      变长场景未提供queryStartLocOptional。<br>
      </td>
    </tr>
  </tbody></table>

## aclnnCausalConv1dGrad

- **参数说明**：

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
    <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnCausalConv1dGradGetWorkspaceSize获取。</td></tr>
    <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
    <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnCausalConv1dGrad默认确定性实现（dweight/dbias的跨序列、跨核累加采用固定归约顺序，使用fp32中间累加）。

- 输入输出一致性限制：
  - gradY、x、weight、biasOptional、dx、dweight、dbiasOptional的数据类型必须一致（FLOAT16或BFLOAT16）。
  - gradY与x的shape必须一致；dx与x的shape必须一致；dweight与weight的shape必须一致；dbiasOptional与biasOptional的shape必须一致。
  - biasOptional与dbiasOptional必须同时提供或同时传空指针。

- 输入shape限制：
  - prefill场景（固定batch）：gradY、x为3维[batch, seqLen, dim]，seqLen > 0。
  - prefill场景（变长序列）：gradY、x为2维[cuSeqLen, dim]，queryStartLocOptional必须提供。
  - weight、dweight为2维[K, dim]，K∈{2,3,4}。
  - dim需满足 (dim * dtypeSize) % 32 == 0。

- 输入值域限制：
  - queryStartLocOptional[0]必须为0，queryStartLocOptional[-1]必须等于cuSeqLen，值必须非递减。
  - cacheIndicesOptional中的值∈[0, numCacheLines)，或等于nullBlockId表示跳过。
  - initialStateModeOptional值∈{0,1}。
  - 输入不支持有±inf和nan的情况。

- 模式限制：
  - 仅支持prefill模式反向，不支持decode/update增量模式。
  - 卷积核宽度K仅支持2、3、4。
  - 不输出历史状态缓存convStates的梯度。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
// 示例代码占位，待开发阶段代码完成后补充。
```
