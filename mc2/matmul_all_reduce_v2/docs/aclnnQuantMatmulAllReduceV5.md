# aclnnQuantMatmulAllReduceV2V5

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mc2/matmul_all_reduce_v2)

## 产品支持情况

| 产品                                                                                     | 是否支持 |
| :--------------------------------------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                                                                      |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>                        |    ×    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √    |
| <term>Atlas 200I/500 A2 推理产品</term>                                         |    ×    |
| <term>Atlas 推理系列产品</term>                                                 |    ×    |
| <term>Atlas 训练系列产品</term>                                                 |    ×    |

## 功能说明

- **接口功能**：兼容`aclnnQuantMatmulAllReduceV2V4`支持的功能，新增通信模式参数。
- **计算公式**：与`aclnnQuantMatmulAllReduceV2V4`一致，详见[aclnnQuantMatmulAllReduceV2V4](aclnnQuantMatmulAllReduceV2V4.md)。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用`aclnnQuantMatmulAllReduceV2V5GetWorkspaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnQuantMatmulAllReduceV2V5`接口执行计算。

```cpp
aclnnStatus aclnnQuantMatmulAllReduceV2V5GetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *biasOptional,
    const aclTensor *x3Optional,
    const aclTensor *x1ScaleOptional,
    const aclTensor *x2ScaleOptional,
    const aclTensor *commQuantScale1Optional,
    const aclTensor *commQuantScale2Optional,
    const char      *group,
    const char      *reduceOp,
    const char      *commMode,
    int64_t          commTurn,
    int64_t          streamMode,
    int64_t          groupSize,
    int64_t          commQuantMode,
    const aclTensor *output,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnQuantMatmulAllReduceV2V5(
    void              *workspace,
    uint64_t           workspaceSize,
    aclOpExecutor     *executor,
    const aclrtStream  stream)
```

## aclnnQuantMatmulAllReduceV2V5GetWorkspaceSize

- **参数说明**
    <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
      <col style="width: 170px">
      <col style="width: 120px">
      <col style="width: 300px">
      <col style="width: 330px">
      <col style="width: 212px">
      <col style="width: 100px">
      <col style="width: 190px">
      <col style="width: 145px">
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
          <td>x1</td>
          <td>输入</td>
          <td>MatMul计算的左矩阵，即计算公式中的x1。</td>
          <td><ul><li>当前版本仅支持二维或者三维输入。</li><li>支持不转置场景。</li></ul></td>
          <td>INT8、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8、FLOAT4_E2M1。</td>
          <td>ND</td>
          <td>2-3</td>
          <td>×</td>
        </tr>
        <tr>
          <td>x2</td>
          <td>输入</td>
          <td>MatMul计算的右矩阵，即计算公式中的x2。</td>
          <td><ul><li>当前版本仅支持二维输入。</li><li>支持转置/不转置场景。</li><li>ND格式下支持最后两轴转置情况下的非连续的tensor，其他非连续tensor不支持</li></ul></td>
          <td>INT8、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8、FLOAT4_E2M1。</td>
          <td>ND</td>
          <td>2</td>
          <td>×</td>
        </tr>
        <tr>
          <td>biasOptional</td>
          <td>输入</td>
          <td>bias偏移，即计算公式中的biasOptional。</td>
          <td><ul><li>当前版本仅支持一维输入。</li></ul></td>
          <td>INT32、FLOAT32</td>
          <td>ND</td>
          <td>1</td>
          <td>√</td>
        </tr>
        <tr>
          <td>x3Optional</td>
          <td>输入</td>
          <td>MatMul计算后的add计算，即计算公式中的x3Optional。</td>
          <td><ul><li>低比特通信场景下仅支持输出为BFLOAT16场景，且仅支持非空输入，要求数据类型、维度与output的维度一致。</li></ul></td>
          <td>FLOAT16、BFLOAT16、FLOAT32</td>
          <td>ND</td>
          <td>2-3</td>
          <td>√</td>
        </tr>
        <tr>
          <td>x1ScaleOptional</td>
          <td>输入</td>
          <td>MatMul计算后的pertoken去量化系数，即计算公式中的x1ScaleOptional。</td>
          <td><ul><li>pertoken场景下，x1为(b, m, k)时shape为(b*m)，x1为(m, k)时shape为(m)。</li><li>perblock场景下，x1为(b, m, k)时shape为[b, ceilDiv(m, 128), ceilDiv(k, 128)]，x1为(m, k)时shape为[ceilDiv(m, 128), ceilDiv(k, 128)]。</li><li>数据类型为FLOAT8_E8M0时，shape为[m, ceilDiv(k, 64), 2], x1为FLOAT4_E2M1时，必须保证ceilDiv(k, 32)为偶数。</li></ul></td>
          <td>FLOAT32、FLOAT8_E8M0</td>
          <td>ND</td>
          <td>1-3</td>
          <td>√</td>
        </tr>
        <tr>
          <td>x2ScaleOptional</td>
          <td>输入</td>
          <td>MatMul计算后的去量化系数，即计算公式中的x2Scale。</td>
          <td><ul><li>shape在pertensor场景为(1)，perchannel场景为(n)/(1, n)。</li><li>输入为int8且输出为BFLOAT16时，直接将BFLOAT16类型的x2ScaleOptional传入本接口。</li><li>输出为FLOAT16且输入为INT8时，x1ScaleOptional不为空，可直接将FLOAT32类型的x2ScaleOptional传入本接口，如果x1ScaleOptional为空，则需提前调用TransQuantParamV2算子的aclnn接口来将x2ScaleOptional转成INT64/UINT64数据类型。</li><li>数据类型为FLOAT8_E8M0时，仅支持转置，shape为[n, ceilDiv(k, 64), 2]。</li> <li>x2为FLOAT4_E2M1时，必须保证ceilDiv(k, 32)为偶数.</li><li>perblock场景下，x2的shape为[ceilDiv(k, 128), ceilDiv(n, 128)]，x2转置时，x2ScaleOptional的shape为[ceilDiv(n, 128), ceilDiv(k, 128)]。</li></ul></td>
          <td>INT64、UINT64、FLOAT32、BFLOAT16、FLOAT8_E8M0</td>
          <td>ND</td>
          <td>1-3</td>
          <td>√</td>
        </tr>
        <tr>
          <td>commQuantScale1Optional</td>
          <td>输入</td>
          <td>MatMul+Add计算后的perchannel量化系数，即计算公式中的commQuantScale1Optional。</td>
          <td><ul><li>当前版本仅在输入为int8类型时支持，其他场景传入空。</li><li>x2为(k, n)时, shape可为(n)或者(1,n)</li></ul></td>
          <td>BFLOAT16、FLOAT16</td>
          <td>ND</td>
          <td>1-2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>commQuantScale2Optional</td>
          <td>输入</td>
          <td>AllGather计算后的perchannel量化系数，即计算公式中的commQuantScale2Optional。</td>
          <td><ul><li>当前版本仅在输入为int8类型时支持，其他场景传入空。</li><li>x2为(k, n)时, shape可为(n)或者(1,n)</li></ul></td>
          <td>BFLOAT16、FLOAT16</td>
          <td>ND</td>
          <td>1-2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>group</td>
          <td>输入</td>
          <td>通信域名称。</td>
          <td><ul><li>通过Hccl提供的接口"extern HcclResult HcclGetCommName(HcclComm comm, char* commName);"获取，其中commName即为group。</li></ul></td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>reduceOp</td>
          <td>输入</td>
          <td>reduce操作类型。</td>
          <td><ul><li>当前版本仅支持输入"sum"。</li></ul></td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>commMode</td>
          <td>输入</td>
          <td>通信模式参数。</td>
          <td><ul><li>用于指定通信实现的模式，具体功能待实现。</li></ul></td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>commTurn</td>
          <td>输入</td>
          <td>通信数据切分数，即总数据量/单次通信量。</td>
          <td><ul><li>当前版本仅支持输入0。</li></ul></td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>streamMode</td>
          <td>输入</td>
          <td>流模式的枚举。</td>
          <td><ul><li>当前版本仅支持枚举值1。</li></ul></td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>groupSize</td>
          <td>输入</td>
          <td>用于表示反量化中x1Scale/x2Scale输入的一个数在其所在的对应维度方向上可以用于该方向x1/x2输入的多少个数的反量化。</td>
          <td><ul><li>groupSize输入由3个方向的groupSizeM，groupSizeN，groupSizeK三个值拼接组成，每个值占16位，计算公式为：groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32。</li><li>perblock场景仅支持groupSizeM，groupSizeN，groupSizeK = 128,128,128</li><li>MXFP场景仅支持groupSizeM，groupSizeN，groupSizeK = 1,1,32</li><li>支持参数自动推导，任一参数输入0时，算子自动推导该参数值，全部输入0时，自动推导全部参数值</li></ul></td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>commQuantMode</td>
          <td>输入</td>
          <td>静态量化和动态量化的标志位。</td>
          <td><ul><li>数值为0和1。仅在x1和x2为FLOAT8_E4M3FN或FLOAT8_E5M2时支持1，为1时走Pertile量化Fp8通信场景。</li></ul></td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>output</td>
          <td>输出</td>
          <td>MatMul计算与AllReduce通信的结果，即计算公式中的output。</td>
          <td><ul><li>output的维数与x1一致。</li></ul></td>
          <td>FLOAT16、BFLOAT16、FLOAT32</td>
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
          <td>返回op执行器，包含了算子计算流程。</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
      </tbody>
    </table>

- **返回值**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## aclnnQuantMatmulAllReduceV2V5

- **参数说明**

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
      <td>在Device侧申请的workspace大小，由第一段接口<code>aclnnQuantMatmulAllReduceV2V5GetWorkspaceSize</code>获取。</td>
  </tr>
  <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
  </tr>
  <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的stream。</td>
  </tr>
  </tbody></table>

- **返回值**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 约束与`aclnnQuantMatmulAllReduceV2V4`一致，详见[aclnnQuantMatmulAllReduceV2V4](aclnnQuantMatmulAllReduceV2V4.md)。
- 当前版本commMode参数功能待实现。

## 调用示例

调用示例与`aclnnQuantMatmulAllReduceV2V4`类似，详见[aclnnQuantMatmulAllReduceV2V4](aclnnQuantMatmulAllReduceV2V4.md)。