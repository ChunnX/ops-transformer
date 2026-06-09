# mhc\_post<a name="ZH-CN_TOPIC_0000002309174913"></a>

## 产品支持情况

- <term>Ascend 950PR/Ascend 950DT</term>：支持
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
- <term>Atlas 推理系列产品</term>：不支持
- <term>Atlas 训练系列产品</term>：不支持

## 功能说明<a name="zh-cn_topic_0000002168254826_section14441124184110"></a>

-   API功能：

    实现MHC（Manifold-Constraint Hyper-Connection）Post组件的前向计算，用于Transformer模型中多层残差连接的后处理阶段。该算子将残差矩阵变换与输出状态投影融合为单次计算，避免多次独立算子调用带来的额外开销。

-   计算公式：

    MHC Post算子的核心计算公式为：

    $$x_{l+1} = (H_{l}^{res})^{T} \cdot x_{l} + h_{l}^{out} \cdot H_{t}^{post}$$

    其中：

    - $(H_{l}^{res})^{T} \cdot x_{l}$ 表示对输入$x_{l}$进行残差矩阵的转置矩阵乘法。对于输出中的第$i$行（对应第$i$个head），计算过程为：

      $$x_{l+1}[i] = \sum_{j=0}^{n-1} H_{l}^{res}[j, i] \cdot x_{l}[j]$$

      即将$H_{l}^{res}$矩阵按转置方式与$x_{l}$做矩阵乘法，$H_{l}^{res}[j, i]$为标量，对$x_{l}$的第$j$行做标量乘法后累加到第$i$行输出。

    - $h_{l}^{out} \cdot H_{t}^{post}$ 表示输出状态$h_{l}^{out}$与后处理权重$H_{t}^{post}$的逐元素乘法与广播。对于第$i$个head：

      $$x_{l+1}[i] += H_{t}^{post}[i] \cdot h_{l}^{out}$$

      即$H_{t}^{post}[i]$为标量，对$h_{l}^{out}$整行做标量乘法后加到第$i$行输出。

    综合完整计算过程为：

    $$x_{l+1}[i, :] = H_{t}^{post}[i] \cdot h_{l}^{out}[:] + \sum_{j=0}^{n-1} H_{l}^{res}[j, i] \cdot x_{l}[j, :]$$

    其中，$x_{l}$表示参数`x`，$H_{l}^{res}$表示参数`h_res`，$h_{l}^{out}$表示参数`h_out`，$H_{t}^{post}$表示参数`h_post`，$x_{l+1}$表示输出`y`。

**说明**
- 输入支持两种维度格式：BSND（4维）和TND（3维）。其中B（Batch）表示批量大小，S（Seq-Length）表示序列长度，T表示所有Batch序列长度的累加和（$T = B \times S$），n表示头数（head数量），D表示每个头的隐藏维度大小（headdim）。

## 函数原型<a name="zh-cn_topic_0000002168254826_section45077510411"></a>

```
mhc_post(x, h_res, h_out, h_post) -> Tensor
```

## 参数说明<a name="zh-cn_topic_0000002168254826_section112637109429"></a>

<table style="undefined;table-layout: fixed; width:1625px"><colgroup>
<col style="width: 147px">
<col style="width: 132px">
<col style="width: 232px">
<col style="width: 293px">
<col style="width: 185px">
<col style="width: 319px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>维度</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>x</td>
        <td>输入</td>
        <td>必选</td>
        <td>当前层的输入token特征</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>
            <ul>
                <li>(B, S, n, D)</li>
                <li>(T, n, D)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>h\_res</td>
        <td>输入</td>
        <td>必选</td>
        <td>残差连接矩阵</td>
        <td>FLOAT32</td>
        <td>
            <ul>
                <li>(B, S, n, n)</li>
                <li>(T, n, n)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>h\_out</td>
        <td>输入</td>
        <td>必选</td>
        <td>上一层的输出状态</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>
            <ul>
                <li>(B, S, D)</li>
                <li>(T, D)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>h\_post</td>
        <td>输入</td>
        <td>必选</td>
        <td>后处理权重矩阵</td>
        <td>FLOAT32</td>
        <td>
            <ul>
                <li>(B, S, n)</li>
                <li>(T, n)</li>
            </ul>
        </td>
    </tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000002168254826_section22231435517"></a>

<table style="undefined;table-layout: fixed; width:1475px"><colgroup>
<col style="width: 250px">
<col style="width: 130px">
<col style="width: 230px">
<col style="width: 290px">
<col style="width: 185px">
<col style="width: 270px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>维度</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>y</td>
        <td>输出</td>
        <td>必选</td>
        <td>MHC Post计算输出</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>
            <ul>
                <li>(B, S, n, D)</li>
                <li>(T, n, D)</li>
            </ul>
        </td>
    </tr>
</tbody>
</table>

**说明**
- y：Tensor类型，公式中的输出$x_{l+1}$，数据类型与输入`x`保持一致，shape与输入`x`保持一致。

## 约束说明<a name="zh-cn_topic_0000002168254826_section12345537164214"></a>

-   该接口支持训练、推理场景下使用。
-   该接口支持单算子模式和TorchAir图模式调用。
-   数据类型约束：
    - `x`和`h_out`的数据类型必须相同，支持BFLOAT16或FLOAT16。
    - `h_res`和`h_post`的数据类型必须为FLOAT32。
    - 输出`y`的数据类型与`x`保持一致。

-   维度约束：
    - `x`的维度仅支持3维（TND格式）或4维（BSND格式），不支持其他维度。
    - `h_res`的维度需与`x`维度格式匹配：4维时为(B, S, n, n)，3维时为(T, n, n)。
    - `h_out`的维度：4维时为(B, S, D)，3维时为(T, D)。
    - `h_post`的维度：4维时为(B, S, n)，3维时为(T, n)。

-   Shape一致性约束：
    - 4维（BSND）格式下：
        - `h_res`的(B, S)维度需与`x`的(B, S)维度一致，`h_res`的后两维为(n, n)，其中n与`x`的第3维一致。
        - `h_out`的(B, S)维度需与`x`的(B, S)维度一致，`h_out`的D维度需与`x`的D维度一致。
        - `h_post`的(B, S)维度需与`x`的(B, S)维度一致，`h_post`的n维度需与`x`的n维度一致。
    - 3维（TND）格式下：
        - `h_res`的T维度需与`x`的T维度一致，后两维为(n, n)。
        - `h_out`的T维度需与`x`的T维度一致，D维度需与`x`的D维度一致。
        - `h_post`的T维度需与`x`的T维度一致，n维度需与`x`的n维度一致。

-   所有输入Tensor的shape各维度值必须为正数（大于0）。

-   参数里Shape使用的变量如下：
    -   B：表示Batch批量大小。
    -   S：表示Seq-Length序列长度。
    -   T：表示所有Batch序列长度的累加和，$T = B \times S$。
    -   n：表示head数量（头数）。
    -   D：表示每个头的隐藏维度大小（headdim）。

## 调用说明<a name="zh-cn_topic_0000002168254826_section_calling"></a>

<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：

| 调用方式 | 样例代码 | 说明 |
| :---: | :---: | :--- |
| aclnn API | [test_aclnn_mhc_post](https://gitcode.com/cann/ops-transformer/tree/master/mhc/mhc_post/examples/test_aclnn_mhc_post.cpp) | 通过aclnnMhcPost接口调用MhcPost算子。 |
| GE图模式 | - | 通过算子IR（[mhc_post_proto.h](https://gitcode.com/cann/ops-transformer/tree/master/mhc/mhc_post/op_graph/mhc_post_proto.h)）构图方式调用MhcPost算子。 |
| PyTorch API | - | 通过torch.mhc_post接口调用MhcPost算子。 |

<term>Ascend 950PR/Ascend 950DT</term>：

| 调用方式 | 样例代码 | 说明 |
| :---: | :---: | :--- |
| aclnn API | [test_aclnn_mhc_post](https://gitcode.com/cann/ops-transformer/tree/master/mhc/mhc_post/examples/test_aclnn_mhc_post.cpp) | 通过aclnnMhcPost接口调用MhcPost算子。 |
| GE图模式 | - | 通过算子IR（[mhc_post_proto.h](https://gitcode.com/cann/ops-transformer/tree/master/mhc/mhc_post/op_graph/mhc_post_proto.h)）构图方式调用MhcPost算子。 |
| PyTorch API | - | 通过torch.mhc_post接口调用MhcPost算子。 |

## 调用示例<a name="zh-cn_topic_0000002168254826_section14459801435"></a>

-   单算子模式调用

    ```python
    import torch
    import torch_npu
    from npu_ops_transformer.ops import mhc_post

    B = 2
    S = 8
    n = 4
    D = 128

    x = torch.randn(B, S, n, D, dtype=torch.bfloat16).npu()
    h_res = torch.randn(B, S, n, n, dtype=torch.float32).npu()
    h_out = torch.randn(B, S, D, dtype=torch.bfloat16).npu()
    h_post = torch.randn(B, S, n, dtype=torch.float32).npu()

    y = mhc_post(x, h_res, h_out, h_post)
    print(f"output shape: {y.shape}")
    ```