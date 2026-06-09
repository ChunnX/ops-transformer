# mhc\_post\_backward<a name="ZH-CN_TOPIC_0000002309174914"></a>

## 产品支持情况


- <term>Ascend 950PR/Ascend 950DT</term>：支持
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
- <term>Atlas 推理系列产品</term>：不支持
- <term>Atlas 训练系列产品</term>：不支持

## 功能说明<a name="zh-cn_topic_0000002168254826_section14441124184110"></a>

-   接口功能：

    实现MHC_Post组件的反向计算，需与[mhc\_post](mhc_post.md)配套使用。该算子根据前向计算的公式，对输入`x`、`h_res`、`h_out`、`h_post`分别计算梯度。

-   计算公式：

    MHC Post前向计算公式为：

    $$x_{l+1}[i, :] = H_{t}^{post}[i] \cdot h_{l}^{out}[:] + \sum_{j=0}^{n-1} H_{l}^{res}[j, i] \cdot x_{l}[j, :]$$

    反向计算中，给定输出梯度$\frac{\partial L}{\partial x_{l+1}}$（即`grad_output`），各输入梯度推导如下：

    - $\frac{\partial L}{\partial x_{l}}$（即`grad_x`）：对$x_{l}$的梯度，由$H_{l}^{res}$的转置与输出梯度做矩阵乘法得到：

      $$\frac{\partial L}{\partial x_{l}}[j, :] = \sum_{i=0}^{n-1} H_{l}^{res}[j, i] \cdot \frac{\partial L}{\partial x_{l+1}}[i, :]$$

    - $\frac{\partial L}{\partial H_{l}^{res}}$（即`grad_h_res`）：对$H_{l}^{res}$的梯度，由输出梯度与$x_{l}$的外积得到：

      $$\frac{\partial L}{\partial H_{l}^{res}}[j, i] = \sum_{d=0}^{D-1} \frac{\partial L}{\partial x_{l+1}}[i, d] \cdot x_{l}[j, d]$$

    - $\frac{\partial L}{\partial h_{l}^{out}}$（即`grad_h_out`）：对$h_{l}^{out}$的梯度，由$H_{t}^{post}$与输出梯度做加权求和得到：

      $$\frac{\partial L}{\partial h_{l}^{out}}[:] = \sum_{i=0}^{n-1} H_{t}^{post}[i] \cdot \frac{\partial L}{\partial x_{l+1}}[i, :]$$

    - $\frac{\partial L}{\partial H_{t}^{post}}$（即`grad_h_post`）：对$H_{t}^{post}$的梯度，由输出梯度与$h_{l}^{out}$的内积得到：

      $$\frac{\partial L}{\partial H_{t}^{post}}[i] = \sum_{d=0}^{D-1} \frac{\partial L}{\partial x_{l+1}}[i, d] \cdot h_{l}^{out}[d]$$

    其中，$\frac{\partial L}{\partial x_{l+1}}$表示参数`grad_output`，$x_{l}$表示参数`x`，$H_{l}^{res}$表示参数`h_res`，$h_{l}^{out}$表示参数`h_out`，$H_{t}^{post}$表示参数`h_post`。

**说明**
- 输入支持两种维度格式：BSND（4维）和TND（3维）。其中B（Batch）表示批量大小，S（Seq-Length）表示序列长度，T表示所有Batch序列长度的累加和（$T = B \times S$），n表示头数（head数量），D表示每个头的隐藏维度大小（headdim）。

## 函数原型<a name="zh-cn_topic_0000002168254826_section45077510411"></a>

```
mhc_post_backward(grad_output, x, h_res, h_out, h_post) -> (Tensor, Tensor, Tensor, Tensor)
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
        <td>grad_output</td>
        <td>输入</td>
        <td>必选</td>
        <td>表示网络中MHC层的输入数据，前向输出的梯度</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>
            <ul>
                <li>(B, S, n, D)</li>
                <li>(T, n, D)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>x</td>
        <td>输入</td>
        <td>必选</td>
        <td>表示网络中MHC层的输入数据，前向计算的输入token特征（保存用于反向计算）</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>
            <ul>
                <li>(B, S, n, D)</li>
                <li>(T, n, D)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>h_res</td>
        <td>输入</td>
        <td>必选</td>
        <td>mHC的h_res变换矩阵，前向计算的残差连接矩阵（保存用于反向计算）</td>
        <td>FLOAT32</td>
        <td>
            <ul>
                <li>(B, S, n, n)</li>
                <li>(T, n, n)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>h_out</td>
        <td>输入</td>
        <td>必选</td>
        <td>Atten/MLP层的输出，前向计算的输出状态（保存用于反向计算）</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>
            <ul>
                <li>(B, S, D)</li>
                <li>(T, D)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>h_post</td>
        <td>输入</td>
        <td>必选</td>
        <td>MHC的h_post变换矩阵，前向计算的后处理权重矩阵（保存用于反向计算）</td>
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
        <td>grad_x</td>
        <td>输出</td>
        <td>必选</td>
        <td>网络中MHC层的输入数据x的梯度</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>
            <ul>
                <li>(B, S, n, D)</li>
                <li>(T, n, D)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>grad_h_res</td>
        <td>输出</td>
        <td>必选</td>
        <td>网络中MHC层的输入数据h_res的梯度</td>
        <td>FLOAT32</td>
        <td>
            <ul>
                <li>(B, S, n, n)</li>
                <li>(T, n, n)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>grad_h_out</td>
        <td>输出</td>
        <td>必选</td>
        <td>网络中MHC层的输入数据h_out的梯度</td>
        <td>BFLOAT16、FLOAT16</td>
        <td>
            <ul>
                <li>(B, S, D)</li>
                <li>(T, D)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>grad_h_post</td>
        <td>输出</td>
        <td>必选</td>
        <td>网络中MHC层的输入数据h_post的梯度</td>
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

**说明**
- grad\_x：Tensor类型，对输入`x`的梯度，数据类型与`x`保持一致，shape与`x`保持一致。
- grad\_h\_res：Tensor类型，对输入`h_res`的梯度，数据类型为FLOAT32，shape与`h_res`保持一致。
- grad\_h\_out：Tensor类型，对输入`h_out`的梯度，数据类型与`h_out`保持一致，shape与`h_out`保持一致。
- grad\_h\_post：Tensor类型，对输入`h_post`的梯度，数据类型为FLOAT32，shape与`h_post`保持一致。

## 约束说明<a name="zh-cn_topic_0000002168254826_section12345537164214"></a>

-   该接口支持训练、推理场景下使用。
-   该接口支持单算子模式和TorchAir图模式调用。
-   数据类型约束：
    - `grad_output`、`x`和`h_out`的数据类型必须相同，支持BFLOAT16或FLOAT16。
    - `h_res`和`h_post`的数据类型必须为FLOAT32。
    - 输出`grad_x`和`grad_h_out`的数据类型与`grad_output`保持一致。
    - 输出`grad_h_res`和`grad_h_post`的数据类型为FLOAT32。

-   维度约束：
    - `grad_output`和`x`的维度仅支持3维（TND格式）或4维（BSND格式），不支持其他维度。
    - `h_res`的维度需与`grad_output`维度格式匹配：4维时为(B, S, n, n)，3维时为(T, n, n)。
    - `h_out`的维度：4维时为(B, S, D)，3维时为(T, D)。
    - `h_post`的维度：4维时为(B, S, n)，3维时为(T, n)。

-   Shape一致性约束：
    - `grad_output`的shape需与`x`的shape完全一致。
    - 4维（BSND）格式下：
        - `h_res`的(B, S)维度需与`grad_output`的(B, S)维度一致，`h_res`的后两维为(n, n)，其中n与`grad_output`的第3维一致。
        - `h_out`的(B, S)维度需与`grad_output`的(B, S)维度一致，`h_out`的D维度需与`grad_output`的D维度一致。
        - `h_post`的(B, S)维度需与`grad_output`的(B, S)维度一致，`h_post`的n维度需与`grad_output`的n维度一致。
    - 3维（TND）格式下：
        - `h_res`的T维度需与`grad_output`的T维度一致，后两维为(n, n)。
        - `h_out`的T维度需与`grad_output`的T维度一致，D维度需与`grad_output`的D维度一致。
        - `h_post`的T维度需与`grad_output`的T维度一致，n维度需与`grad_output`的n维度一致。

-   所有输入Tensor的shape各维度值必须为正数（大于0）。

-   输入Tensor不支持非连续格式，算子内部会自动进行AutoContiguous处理。

-   mhc\_post\_backward需与[mhc\_post](mhc_post.md)配套使用，反向计算的输入`x`、`h_res`、`h_out`、`h_post`需与前向计算对应的输入保持一致。

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
| aclnn API | [test_aclnn_mhc_post_backward](https://gitcode.com/cann/ops-transformer/tree/master/mhc/mhc_post_backward/examples/test_aclnn_mhc_post_backward.cpp) | 通过aclnnMhcPostBackward接口调用MhcPostBackward算子。 |
| PyTorch API | - | 通过torch.mhc_post_backward接口调用MhcPostBackward算子。 |

<term>Ascend 950PR/Ascend 950DT</term>：

| 调用方式 | 样例代码 | 说明 |
| :---: | :---: | :--- |
| aclnn API | [test_aclnn_mhc_post_backward](https://gitcode.com/cann/ops-transformer/tree/master/mhc/mhc_post_backward/examples/test_aclnn_mhc_post_backward.cpp) | 通过aclnnMhcPostBackward接口调用MhcPostBackward算子。 |
| PyTorch API | - | 通过torch.mhc_post_backward接口调用MhcPostBackward算子。 |

## 调用示例<a name="zh-cn_topic_0000002168254826_section14459801435"></a>

-   单算子模式调用

    ```python
    import torch
    import torch_npu
    from npu_ops_transformer.ops import mhc_post, mhc_post_backward

    B = 2
    S = 8
    n = 4
    D = 128

    x = torch.randn(B, S, n, D, dtype=torch.bfloat16).npu()
    h_res = torch.randn(B, S, n, n, dtype=torch.float32).npu()
    h_out = torch.randn(B, S, D, dtype=torch.bfloat16).npu()
    h_post = torch.randn(B, S, n, dtype=torch.float32).npu()

    y = mhc_post(x, h_res, h_out, h_post)

    grad_output = torch.randn_like(y).npu()
    grad_x, grad_h_res, grad_h_out, grad_h_post = mhc_post_backward(grad_output, x, h_res, h_out, h_post)

    print(f"grad_x shape: {grad_x.shape}")
    print(f"grad_h_res shape: {grad_h_res.shape}")
    print(f"grad_h_out shape: {grad_h_out.shape}")
    print(f"grad_h_post shape: {grad_h_post.shape}")
    ```