# aclnnMatmulAllReduceV3

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mc2/matmul_all_reduce)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

**说明：** 使用该接口时，请确保驱动固件包和CANN包都为配套的8.0.RC2版本或者配套的更高版本，否则将会引发报错，比如Bus Error等。

## 功能说明

- **接口功能**：兼容`aclnnMatmulAllReduceV2`支持的功能，新增通信模式参数。
- **计算公式**：

    $$
    output = AllReduce(x1 @ x2 + bias + x3)
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用`aclnnMatmulAllReduceV3GetWorkspaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnMatmulAllReduceV3`接口执行计算。

```cpp
aclnnStatus aclnnMatmulAllReduceV3GetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *bias,
    const aclTensor *x3,
    const char      *group,
    const char      *reduceOp,
    const char      *commMode,
    int64_t          commTurn,
    int64_t          streamMode,
    const aclTensor *output,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnMatmulAllReduceV3(
    void              *workspace,
    uint64_t           workspaceSize,
    aclOpExecutor     *executor,
    const aclrtStream  stream)
```

## aclnnMatmulAllReduceV3GetWorkspaceSize

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
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>2-3</td>
          <td>×</td>
        </tr>
        <tr>
          <td>x2</td>
          <td>输入</td>
          <td>MatMul计算的右矩阵，即计算公式中的x2。</td>
          <td><ul><li>当前版本仅支持两维输入。</li><li>支持转置/不转置场景。</li><li>ND格式下支持最后两轴转置情况下的非连续的tensor，其他非连续tensor不支持</li></ul></td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>×</td>
        </tr>
        <tr>
          <td>x3</td>
          <td>输入</td>
          <td>MatMul计算后的add计算，即计算公式中的x3。</td>
          <td>shape与MatMul计算后的shape一致。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
        </tr>
        <tr>
          <td>bias</td>
          <td>输入</td>
          <td>对应计算公式中的bias偏移。</td>
          <td>当前版本仅支持一维输入。</td>
          <td>FLOAT16、BFLOAT16</td>
          <td>ND</td>
          <td>0-1</td>
          <td>√</td>
        </tr>
        <tr>
          <td>group</td>
          <td>输入</td>
          <td>通信域名称。</td>
          <td>通过Hccl提供的接口"extern HcclResult HcclGetCommName(HcclComm comm, char* commName);"获取，其中commName即为group。</td>
          <td>String</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>reduceOp</td>
          <td>输入</td>
          <td>reduce操作类型。</td>
          <td>当前版本仅支持输入"sum"。</td>
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
          <td>当前版本仅支持输入0。</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>streamMode</td>
          <td>输入</td>
          <td>AscendCL流模式的枚举。</td>
          <td>当前版本仅支持枚举值1。</td>
          <td>INT64</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>
        <tr>
          <td>output</td>
          <td>输出</td>
          <td>MatMul计算与AllReduce通信的结果，即计算公式中的output。</td>
          <td>output的维数与x1一致。</td>
          <td>FLOAT16、BFLOAT16</td>
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

## aclnnMatmulAllReduceV3

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
        <td>在Device侧申请的workspace大小，由第一段接口<code>aclnnMatmulAllReduceV3GetWorkspaceSize</code>获取。</td>
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
    </tbody></table>

- **返回值**

    返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 约束与`aclnnMatmulAllReduceV2`一致，详见[aclnnMatmulAllReduceV2](aclnnMatmulAllReduceV2.md)。
- 当前版本commMode参数功能待实现。

## 调用示例

调用示例与`aclnnMatmulAllReduceV2`类似，详见[aclnnMatmulAllReduceV2](aclnnMatmulAllReduceV2.md)。