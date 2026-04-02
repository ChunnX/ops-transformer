# AllGatherMatmulV2

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | x |
| <term>Atlas 推理系列产品</term> | x |
| <term>Atlas 训练系列产品</term> | x |

## 功能说明

- **算子功能**：
  完成AllGather通信与MatMul计算融合。在支持x1和x2输入类型为FLOAT16/BFLOAT16的基础上，同时也支持低精度数据类型，此时算子在Matmul计算后会做对应的反量化计算，支持的低精度数据类型与量化方式如下：
      
  - <term>Ascend 950PR/Ascend 950DT</term>：
          
    新增了对低精度数据类型FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8的支持。支持pertensor、perblock、mx[量化方式](../../../docs/zh/context/量化介绍.md)。
      
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
          
    新增了对低精度数据类型INT8/INT4的支持。支持pertoken/perchannel[量化方式](../../../docs/zh/context/量化介绍.md)。

- **计算公式**：
    - 情形1：如果x1和x2数据类型为FLOAT16/BFLOAT16时，入参x1进行AllGather后，对x1、x2进行MatMul计算。

    $$
    output=AllGather(x1)@x2 + bias
    $$

    $$
    gatherOut=AllGather(x1)
    $$

    - 情形2：如果x1和x2数据类型为FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8的pertensor场景，或者x1和x2数据类型为INT8/INT4的perchannel、pertoken场景，且不输出amaxOut，入参x1进行AllGather后，对x1、x2进行MatMul计算，然后进行dequant操作。

    $$
    output=(x1Scale*x2Scale)*(AllGather(x1)@x2 + bias)
    $$

    $$
    gatherOut=AllGather(x1)
    $$

    - 情形3：如果x1和x2数据类型为FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8的perblock场景，且不输出amaxOut, 当x1为(m, k)、x2为(k, n)时， x1Scale为(ceildiv(m, 128)， ceildiv(k, 128))、x2Scale为(ceildiv(k, 128), ceildiv(n, 128))时，入参x1和x1Scale进行AllGather后，对x1、x2进行perblock量化MatMul计算，然后进行dequant操作。

        $$
        output=\sum_{0}^{\left \lfloor \frac{k}{blockSize=128} \right \rfloor} (AllGather(x1)_{pr}@x2_{rq}*(AllGather(x1Scale)_{pr}*x2Scale_{rq}))
        $$

        $$
        gatherOut=AllGather(x1)
        $$

    - 情形4：如果x1和x2数据类型为FLOAT8_E4M3FN/FLOAT8_E5M2的mx量化场景，x1为(m, k)、x2 为(n, k)，且x1Scale为(m, ceilDiv(k, 64), 2)、x2Scale为(n, ceilDiv(k, 64), 2)，入参x1和x1Scale进行AllGather后，对x1、x2进行MatMul计算，然后进行dequant操作；

        $$
        output=\sum_{0}^{\left \lfloor \frac{k}{blockSize=32} \right \rfloor} (AllGather(x1)_{pr}@x2_{rq}*(AllGather(x1Scale)_{pr}*x2Scale_{rq}))
        $$

        $$
        gatherOut=AllGather(x1)
        $$

## 参数说明

- **参数说明：**
    <table style="undefined;table-layout: fixed; width: 1607px"><colgroup>
    <col style="width: 190px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 330px">
    <col style="width: 212px">
    <col style="width: 120px">
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
        <td>x1 (aclTensor*)</td>
        <td>输入</td>
        <td>MM左矩阵，即计算公式中的x1。</td>
        <td>当前版本仅支持两维输入，shape为[m, k]，且仅支持不转置场景。</td>
        <td>FLOAT16、BFLOAT16、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8、INT8、INT4</td>
        <td>ND</td>
        <td>2</td>
        <td>-</td>
    </tr>
    <tr>
        <td>x2 (aclTensor*)</td>
        <td>输入</td>
        <td>MM右矩阵，即计算公式中的x2。</td>
        <td><ul><li>当前版本仅支持二维输入，shape为[k, n]，支持转置/不转置场景。</li><li>仅支持两根轴转置情况下的<a href="../../../docs/zh/context/非连续的Tensor.md">[非连续的Tensor]</a>。</li></ul></td>
        <td>FLOAT16、BFLOAT16、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8、INT8、INT4</td>
        <td>ND</td>
        <td>2</td>
        <td>√（仅适用转置场景）</td>
    </tr>
    <tr>
        <td>bias (aclTensor*)</td>
        <td>输入</td>
        <td>即计算公式中的bias。</td>
        <td><ul><li><term>Ascend 950PR/Ascend 950DT</term>：支持传入一维输入或者空指针。</li><li><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：当前版本仅支持传入空指针。</li></ul></td>
        <td>FLOAT16、BFLOAT16、FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>x1Scale (aclTensor*)</td>
        <td>输入</td>
        <td>mm左矩阵反量化参数。</td>
        <td>支持传入空指针场景。</td>
        <td>FLOAT16、BFLOAT16、FLOAT</td>
        <td>ND</td>
        <td>1-3</td>
        <td>-</td>
    </tr>
    <tr>
        <td>x2Scale (aclTensor*)</td>
        <td>输入</td>
        <td>mm右矩阵反量化参数。</td>
        <td>支持传入空指针场景。</td>
        <td>FLOAT16、BFLOAT16、FLOAT</td>
        <td>ND</td>
        <td>1-3</td>
        <td>√（仅适用转置场景）</td>
    </tr>
    <tr>
        <td>quantScale (aclTensor*)</td>
        <td>输入</td>
        <td>即计算公式中的bias。</td>
        <td>当前仅支持传入空指针场景。</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>blockSize (int64_t)</td>
        <td>输入</td>
        <td>用于表示mm输出矩阵在M轴方向上和N轴方向上可以用于对应方向上的多少个数的量化。</td>
        <td>blockSize由blockSizeM、blockSizeN、blockSizeK三个值拼接而成，每个值占16位，计算公式为blockSize = blockSizeK | blockSizeN << 16 | blockSizeM << 32，mm输出矩阵不涉及K轴，blockSizeK固定为0, 当前版本只支持blockSizeM=blockSizeN=0。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>group (char*)</td>
        <td>输入</td>
        <td>通信域名称。</td>
        <td>通过Hccl提供的接口“extern HcclResult HcclGetCommName(HcclComm comm, char* commName);”获取，其中commName即为group。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>gatherIndex (int64_t)</td>
        <td>输入</td>
        <td>标识gather目标。</td>
        <td><ul><li>0表示目标为x1，1表示目标为x2。</li><li>当前版本仅支持输入0。</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commTurn (int64_t)</td>
        <td>输入</td>
        <td>通信数据切分数，即总数据量/单次通信量。</td>
        <td>当前版本仅支持输入0。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>streamMode (int64_t)</td>
        <td>输入</td>
        <td>流模式的枚举。</td>
        <td>当前只支持枚举值1。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>groupSize (int64_t)</td>
        <td>输入</td>
        <td>用于表示反量化中x1Scale/x2Scale输入的一个数在其所在的对应维度方向上可以用于该方向x1/x2输入的多少个数的反量化。</td>
        <td>groupSize输入由3个方向的groupSizeM、groupSizeN、groupSizeK三个值拼接组成，每个值占16位，计算公式为groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commMode (char*)</td>
        <td>输入</td>
        <td>通信模式。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>output (aclTensor*)</td>
        <td>输出</td>
        <td>AllGather通信与MatMul计算的结果，即计算公式中的output。</td>
        <td><ul><li><term>Ascend 950PR/Ascend 950DT</term>：支持空Tensor。</li><li><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持空Tensor。</li></ul></td>
        <td>FLOAT16、BFLOAT16、FLOAT</td>
        <td>ND</td>
        <td>2</td>
        <td>-</td>
    </tr>
    <tr>
        <td>gatherOut (aclTensor*)</td>
        <td>输出</td>
        <td>仅输出all_gather通信后的结果。即公式中的gatherOut。</td>
        <td><ul><li><term>Ascend 950PR/Ascend 950DT</term>：支持空Tensor。</li><li><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持空Tensor。</li><li>数据类型与x1的数据类型保持一致。</li></ul></td>
        <td>FLOAT16、BFLOAT16、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8、INT8、INT4</td>
        <td>ND</td>
        <td>2</td>
        <td>-</td>
    </tr>
    <tr>
        <td>amaxOut (aclTensor*)</td>
        <td>输出</td>
        <td>MM计算的最大值结果，即公式中的amaxOut。</td>
        <td>当前版本仅支持nullptr或空tensor。</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
    </tr>
    <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>输出</td>
        <td>返回需要在Device侧申请的workspace大小。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor (aclOpExecutor**)</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tbody></table>

## 约束说明

- 确定性计算：
  - 该算子默认确定性实现。
- <term>Ascend 950PR/Ascend 950DT</term>：
    - 输入x1为2维，其维度为\(m, k\)。x2必须是2维，其维度为\(k, n\)，轴满足mm算子入参要求，k轴相等，且k轴取值范围为\[256, 65535\)。
    - bias为1维，shape为\(n,\)。
    - 输出output为2维，其维度为\(m*rank\_size, n\)，rank\_size为卡数。
    - 输出gatherout为2维，其维度为\(m*rank\_size, k\)，rank\_size为卡数。
    - 当x1、x2的数据类型为FLOAT16/BFLOAT16时，output计算输出数据类型和x1、x2保持一致。
    - 当x1、x2的数据类型为FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8时，output输出数据类型支持FLOAT16、BFLOAT16、FLOAT。
    - 当x1、x2的数据类型为FLOAT16/BFLOAT16/HIFLOAT8时，x1和x2数据类型需要保持一致。
    - 当x1、x2数据类型为FLOAT8_E4M3FN/FLOAT8_E5M2时，x1和x2数据类型可以为其中一种。
    - 当x1、x2数据类型为FLOAT16/BFLOAT16/HIFLOAT8/FLOAT8_E4M3FN/FLOAT8_E5M2时，x2矩阵支持转置/不转置场景，x1矩阵只支持不转置场景。
    - 当groupSize取值为549764202624，bias必须为空。
    - 支持2、4、8、16、32、64卡。
    - allgather(x1)集合通信数据总量不能超过16*256MB，集合通信数据总量计算方式为：m * k * sizeof(x1_dtype) * 卡数。由于shape不同，算子内部实现可能存在差异，实际支持的总通信量可能略小于该值。

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
    - 只支持x2矩阵转置/不转置，x1矩阵仅支持不转置场景。
    - 输入x1必须是2维，其shape为\(m, k\)。
    - 输入x2必须是2维，其shape为\(k, n\)，轴满足mm算子入参要求，k轴相等，且k轴取值范围为\[256, 65535\)。
    - bias仅支持输入nullptr。
    - 输出为2维，其shape为\(m*rank\_size, n\), rank\_size为卡数。
    - 不支持空tensor。
    - x1和x2的数据类型需要保持一致。
    - x1和x2数据类型为INT4时，k与n必须为偶数。
    - 支持2、4、8卡。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_all_gather_matmul_v2](./examples/test_aclnn_all_gather_matmul_v2.cpp) | 通过[aclnnAllGatherMatmulV2](./docs/aclnnAllGatherMatmulV2.md)接口方式调用AllGatherMatmulV2算子。 |
