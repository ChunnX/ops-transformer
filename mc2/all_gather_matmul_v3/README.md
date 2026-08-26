# AllGatherMatmulV3

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   | x |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | x |
| <term>Atlas 200I/500 A2 推理产品</term> | x |
| <term>Atlas 推理系列产品</term> | x |
| <term>Atlas 训练系列产品</term> | x |

## 功能说明

- **算子功能**：
  完成AllGather通信与MX量化MatMul计算融合（apace URMA通算融合路径）。入参x1与x1Scale经AllGather后，与本rank的x2、x2Scale进行MX量化MatMul计算并反量化输出。支持MX[量化方式](../../docs/zh/context/quant_mode_introduction.md)（group size固定为[1, 1, 32]），支持的低精度数据类型为FLOAT8_E4M3FN/FLOAT8_E5M2/FLOAT4_E2M1。

- **计算公式**：x1为(m, k)、x2为(n, k)，x1Scale为(m, ceilDiv(k, 64), 2)、x2Scale为(n, ceilDiv(k, 64), 2)时，入参x1和x1Scale进行AllGather后，对x1、x2进行MX量化MatMul计算，然后进行dequant操作：

    $$
    output=\sum_{0}^{\left \lfloor \frac{k}{blockSize=32} \right \rfloor} (AllGather(x1)_{pr}@x2_{rq}^{T}*(AllGather(x1Scale)_{pr}*x2Scale_{rq})) + bias
    $$

    $$
    gatherOut=AllGather(x1)
    $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
    <col style="width: 170px">
    <col style="width: 170px">
    <col style="width: 200px">
    <col style="width: 200px">
    <col style="width: 170px">
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
        <td>context </td>
        <td>输入</td>
        <td>通信上下文，由通信建链接口生成并透传。</td>
        <td>INT32</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>x1 </td>
        <td>输入</td>
        <td>MM左矩阵，即计算公式中的x1，维度为(m, k)。</td>
        <td>FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E2M1</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>x2 </td>
        <td>输入</td>
        <td>MM右矩阵，即计算公式中的x2，维度为(n, k)，调用时以(n, k)存储的转置视图(k, n)传入。</td>
        <td>FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E2M1</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>bias </td>
        <td>可选输入</td>
        <td>即计算公式中的bias，维度为(n,)。</td>
        <td>FLOAT</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>x1Scale </td>
        <td>可选输入</td>
        <td>mm左矩阵MX量化系数，维度为(m, ceilDiv(k, 64), 2)，MX场景必传。</td>
        <td>FLOAT8_E8M0</td>
        <td>ND、NCL</td>
    </tr>
    <tr>
        <td>x2Scale </td>
        <td>可选输入</td>
        <td>mm右矩阵MX量化系数，维度为(n, ceilDiv(k, 64), 2)，MX场景必传。</td>
        <td>FLOAT8_E8M0</td>
        <td>ND、NCL</td>
    </tr>
    <tr>
        <td>y </td>
        <td>输出</td>
        <td>AllGather通信与MX量化MatMul计算的结果，即计算公式中的output，维度为(m*rank_size, n)。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>gatherOut </td>
        <td>输出</td>
        <td>AllGather通信结果，即公式中的gatherOut，维度为(m*rank_size, k)。当前版本未实现，返回空tensor。</td>
        <td>FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E2M1</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>amaxOut </td>
        <td>可选输出</td>
        <td>MM计算的最大值结果，即公式中的amaxOut。当前版本未实现。</td>
        <td>FLOAT</td>
        <td>ND</td>
    </tr>
    <tr>
        <td>group</td>
        <td>属性</td>
        <td>通信域名称，长度需在[1, 128)范围内。</td>
        <td>STRING</td>
        <td>-</td>
    </tr>
    <tr>
        <td>hcclBufferSize</td>
        <td>属性</td>
        <td>HCCL通信buffer大小（字节），需大于0。</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>isTransA</td>
        <td>可选属性</td>
        <td>x1是否转置，当前仅支持false。</td>
        <td>BOOL</td>
        <td>-</td>
    </tr>
    <tr>
        <td>isTransB</td>
        <td>可选属性</td>
        <td>x2是否转置，当前仅支持true（x2为(n, k)布局）。</td>
        <td>BOOL</td>
        <td>-</td>
    </tr>
    <tr>
        <td>rankSize</td>
        <td>可选属性</td>
        <td>通信域rank数量，缺省(<=0)时从group反查。</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>groupSize</td>
        <td>可选属性</td>
        <td>量化group大小，MX场景仅支持[1, 1, 32]；取0时由scale shape自动推导。</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>yDtype</td>
        <td>可选属性</td>
        <td>输出y的数据类型。</td>
        <td>INT64</td>
        <td>-</td>
    </tr>
    <tr>
        <td>commMode</td>
        <td>可选属性</td>
        <td>通信模式，当前仅支持urma。</td>
        <td>STRING</td>
        <td>-</td>
    </tr>
    </tbody></table>

## 约束说明

- 确定性计算：
  - 该算子默认确定性实现。
- <term>Ascend 950PR/Ascend 950DT</term>：
    - 输入x1必须是2维，其维度为\(m, k\)。
    - 输入x2必须是2维，存储布局为\(n, k\)，调用时以转置视图\(k, n\)传入；k轴与x1相等，取值范围为\[256, 65535\)。
    - x1、x2数据类型为FLOAT8_E4M3FN/FLOAT8_E5M2时可混合搭配；数据类型为FLOAT4_E2M1时，x1和x2必须同时为FLOAT4_E2M1，且k必须为偶数。
    - x1Scale/x2Scale必须是3维，shape为\(m, ceilDiv(k, 64), 2\)/\(n, ceilDiv(k, 64), 2\)。
    - bias为1维，shape为\(n,\)，数据类型仅支持FLOAT。
    - 输出y为2维，其维度为\(m*rank\_size, n\)，数据类型支持FLOAT16、BFLOAT16，rank\_size为卡数。
    - 输出gatherOut为2维，其维度为\(m*rank\_size, k\)（当前版本未实现，返回空tensor）。
    - groupSize仅支持\[1, 1, 32\]，取0时从scale shape自动推导。
    - rankSize支持2、4、8、16、32、64卡，且不能超过1:1核配比下实际可用block数（min(aicNum, aivNum)）。
    - 仅支持URMA通信引擎。
    - 通信数据总量（x1 data + x1Scale）不能超过hcclBufferSize，可通过HCCL_BUFFSIZE环境变量调整通信buffer大小。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_all_gather_matmul_v3](./tests/ut/op_api/test_aclnn_all_gather_matmul_v3.cpp) | 通过aclnnAllGatherQuantMatmulV3接口方式调用AllGatherMatmulV3算子。 |
| torch接口  | [all_gather_quant_matmul](./torch_extension/all_gather_quant_matmul.py) | 通过cann_ops_transformer的all_gather_quant_matmul接口方式调用AllGatherMatmulV3算子。 |
