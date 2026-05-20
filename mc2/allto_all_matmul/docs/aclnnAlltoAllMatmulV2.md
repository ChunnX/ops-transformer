# aclnnAlltoAllMatmulV2

[📄 查看源码](https://gitcode.com/cann/ops-transformer/tree/master/mc2/allto_all_matmul_v2)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 接口功能：完成AlltoAll通信、Permute（保证通信后地址连续）和Matmul计算的融合，**先通信后计算**。

  - <term>Ascend 950PR/Ascend 950DT</term>：新增`commMode`参数，根据该参数指定通信类型。

- 计算公式：假设x1输入shape为(BS, H)，rankSize为NPU卡数

  $$
  commOut = AlltoAll(x1.view(rankSize, BS/rankSize, H)) \\
  permutedOut = commOut.permute(1, 0, 2).view(BS/rankSize, rankSize*H) \\
  output = permutedOut @ x2 + bias \\
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用 “aclnnAlltoAllMatmulV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnAlltoAllMatmulV2”接口执行计算。

```cpp
aclnnStatus aclnnAlltoAllMatmulV2GetWorkspaceSize(
  const aclTensor*   x1, 
  const aclTensor*   x2,
  const aclTensor*   biasOptional,
  const aclIntArray* alltoAllAxesOptional,
  const char*        group,
  bool               transposeX1,
  bool               transposeX2,
  const char*        commMode,
  const aclTensor*   output,
  const aclTensor*   alltoAllOutOptional,
  uint64_t*          workspaceSize,
  aclOpExecutor**    executor)
```

```cpp
aclnnStatus aclnnAlltoAllMatmulV2(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnAlltoAllMatmulV2GetWorkspaceSize

- ​**参数说明**​

    <table style="undefined;table-layout: fixed; width: 1556px"> <colgroup>
    <col style="width: 154px">
    <col style="width: 123px">
    <col style="width: 270px">
    <col style="width: 325px">
    <col style="width: 245px">
    <col style="width: 120px">
    <col style="width: 203px">
    <col style="width: 116px">
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
    <th>非连续tensor</th>
    </tr></thead>
    <tbody>
    <tr>
    <td>x1</td>
    <td>输入</td>
    <td>融合算子的左矩阵输入，对应公式中的x1。</td>
    <td>该输入进行AlltoAll通信与Permute操作后结果作为MatMul计算的左矩阵输入。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>2维，shape为(BS, H)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>x2</td>
    <td>输入</td>
    <td>融合算子的右矩阵输入，也是MatMul计算的右矩阵。</td>
    <td>直接作为MatMul计算的右矩阵输入。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>2维，shape为(H*rankSize, N)</td>
    <td>不同设备型号支持情况不同，参见<a href="#约束说明">约束说明</a>。</td>
    </tr>
    <tr>
    <td>biasOptional</td>
    <td>输入</td>
    <td>矩阵乘运算后累加的偏置，对应公式中的bias。</td>
    <td>支持传入空指针场景，根据设备型号对数据类型有不同限制，详细参见<a href="#约束说明">约束说明</a>。</td>
    <td>FLOAT16、BFLOAT16、FLOAT32</td>
    <td>ND</td>
    <td>1维，shape为(N)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>alltoAllAxesOptional</td>
    <td>输入</td>
    <td>AlltoAll和Permute数据交换的方向。</td>
    <td>支持配置空或者[-2,-1]，传入空时默认按[-2,-1]处理，表示将输入由(BS, H)转为(BS/rankSize, rankSize*H)。</td>
    <td>aclIntArray*(元素类型INT64)</td>
    <td>-</td>
    <td>1维，shape为(2)</td>
    <td>-</td>
    </tr>
    <tr>
    <td>group</td>
    <td>输入</td>
    <td>Host侧标识列组的字符串，即通信域名称，通过Hccl接口HcclGetCommName获取commName作为该参数。</td>
    <td>字符串长度要求(0, 128)。</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>commMode</td>
    <td>输入</td>
    <td>指定当前通信类型。</td>
    <td>支持输入"default"、"aicpu"或"ccu"。</td>
    <td>STRING</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>transposeX1</td>
    <td>输入</td>
    <td>标识左矩阵是否转置过。</td>
    <td>暂不支持配为True。</td>
    <td>bool</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>transposeX2</td>
    <td>输入</td>
    <td>标识右矩阵是否转置过。</td>
    <td>配置为True时右矩阵Shape为(N, rankSize*H)。</td>
    <td>bool</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>output</td>
    <td>输入</td>
    <td>最终的计算结果。</td>
    <td>数据类型与输入x1保持一致。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>2维，shape为(BS/rankSize, N)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>alltoAllOutOptional</td>
    <td>输出</td>
    <td>接收AlltoAll和Permute后的内容。</td>
    <td>传入nullptr时表示不输出通信输出。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>2维，shape为(BS/rankSize, H*rankSize)</td>
    <td>x</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>输出</td>
    <td>返回需要在Device侧申请的workspace大小。</td>
    <td></td>
    <td>UINT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor</td>
    <td>输出</td>
    <td>返回op执行器，包含了算子的计算流程。</td>
    <td></td>
    <td>aclOpExecutor*</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
    <col style="width:282px">
    <col style="width:120px">
    <col style="width:747px">
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
      <td>输入和输出的必选参数Tensor是空指针。</td>
    </tr>
    <tr>
        <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="7">161002</td>
        <td>输入和输出的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
        <td>输入Tensor为空Tensor。</td>
    </tr>
    <tr>
        <td>alltoAllAxesOptional非法。</td>
    </tr>
    <tr>
        <td>transposeX1为true。</td>
    </tr>
    <tr>
        <td>通信域长度非法。</td>
    </tr>
    <tr>
        <td>输入输出Tensor维度不合法。</td>
    </tr>
    <tr>
        <td>输入输出format为私有格式。</td>
    </tr>
      </tbody>
  </table>

## aclnnAlltoAllMatmulV2

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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnAlltoAllMatmulV2GetWorkspaceSize获取。</td>
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

* 默认支持确定性计算。
* NPU卡数（rankSize），根据设备型号有不同限制：
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持2、4、8卡。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持2、4、8、16卡。
  - <term>Ascend 950PR/Ascend 950DT</term>：支持2、4、8、16卡。
* 参数说明中shape使用的变量BS必须整除NPU卡数。
* BS和N的值不得超过2147483647（INT32_MAX），BS的值不得小于0，N的值不得小于1。
* H*rankSize范围，根据设备型号有不同限制：
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持[1, 35000]。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：支持[2, 65535]。
* 空tensor的支持度根据不同设备型号有不同的限制：
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持任何空tensor。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：仅支持输入x1的第一维度（BS）为0的空tensor，其它空tensor均不支持。
* 非连续tensor的支持度根据不同设备型号有不同的限制：
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持任何非连续tensor。
  - <term>Ascend 950PR/Ascend 950DT</term>：仅支持x2为非连续tensor，其它非连续tensor均不支持。
* x1、x2计算输入的数据类型要和output、alltoAllOutOptional计算输出的数据类型一致，传入的x1、x2与output均不为空指针。
* biasOptional的数据类型根据不同设备型号有不同的限制：
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：x1/x2计算输入的数据类型为FLOAT16时，biasOptional计算输入的数据类型支持FLOAT16；x1/x2计算输入的数据类型为BFLOAT16时，biasOptional计算输入的数据类型支持FLOAT32。
  - <term>Ascend 950PR/Ascend 950DT</term>：x1/x2计算输入的数据类型为FLOAT16时，biasOptional计算输入的数据类型支持FLOAT16和FLOAT32；x1/x2计算输入的数据类型为BFLOAT16时，biasOptional计算输入的数据类型支持BFLOAT16和FLOAT32。
* 通算融合算子不支持并发调用，不同的通算融合算子也不支持并发调用。
* 不支持跨超节点通信，只支持超节点内。
* 通信引擎约束：
   - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持MTE通信。
   - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持AICPU通信。
   - <term>Ascend 950PR/Ascend 950DT</term>：支持CCU通信。

## 调用示例

## 调用示例

待补充

## 附录

### 归属领域

ops-transformer/mc2

### Python AtenIR

```
func: npu_all_to_all_matmul (Tensor x1, Tensor x2, str hcom, int world_size, Tensor? bias=None, int[]? all2all_axes=None, bool all2all_out_flag, str? comm_mode=None) -> (Tensor，Tensor)
```
