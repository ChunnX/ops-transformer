# cann_ops_transformer.mhc_post_backward

## 产品支持情况

- <term>Ascend 950PR/Ascend 950DT</term>：不支持
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
- <term>Atlas 推理系列产品</term>：不支持
- <term>Atlas 训练系列产品</term>：不支持

## 功能说明

- 接口功能：

    实现MHC_Post组件的反向计算，需与[mhc_post](mhc_post.md)配套使用。该算子根据前向计算的公式，对输入`x`、`h_res`、`h_out`、`h_post`分别计算梯度。

- 计算公式：

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

## 函数原型

```
cann_ops_transformer.mhc_post_backward(grad_output, x, h_res, h_out, h_post) -> (Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

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
    <th>维度维度(shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>grad_output</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>前向输出的梯度，对应公式中的partial L/partial x<sub>l+1</sub></td>
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
        <td>Tensor</td>
        <td>必选</td>
        <td>前向计算的输入token特征（保存用于反向计算），对应公式中的x<sub>l</sub></td>
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
        <td>Tensor</td>
        <td>必选</td>
        <td>前向计算的残差连接矩阵（保存用于反向计算），对应公式中的H<sub>l</sub><sup>res</sup></td>
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
        <td>Tensor</td>
        <td>必选</td>
        <td>前向计算的输出状态（保存用于反向计算），对应公式中的H<sub>l</sub><sup>out</sup></td>
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
        <td>Tensor</td>
        <td>必选</td>
        <td>前向计算的后处理权重矩阵（保存用于反向计算），对应公式中的H<sub>t</sub><sup>post</sup></td>
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

## 返回值说明

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
    <th>维度(shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>grad_x</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>对输入`x`的梯度，对应公式中的∂L/∂x<sub>l</sub>，数据类型与`x`保持一致，shape与`x`保持一致</td>
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
        <td>Tensor</td>
        <td>必选</td>
        <td>对输入`h_res`的梯度，对应公式中的partial L/partial H<sub>l</sub><sup>res</sup>，数据类型为FLOAT32，shape与`h_res`保持一致</td>
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
        <td>Tensor</td>
        <td>必选</td>
        <td>对输入`h_out`的梯度，对应公式中的∂L/∂h<sub>l</sub><sup>out</sup>，数据类型与`h_out`保持一致，shape与`h_out`保持一致</td>
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
        <td>Tensor</td>
        <td>必选</td>
        <td>对输入`h_post`的梯度，对应公式中的partial L/partial H<sub>t</sub><sup>post</sup>，数据类型为FLOAT32，shape与`h_post`保持一致</td>
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

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式和TorchAir图模式调用。
- 数据类型约束：
    - `grad_output`、`x`和`h_out`的数据类型必须相同。
    - 输出`grad_x`和`grad_h_out`的数据类型与`grad_output`保持一致。

- 维度约束：
    - `h_res`的维度需与`grad_output`维度格式匹配：4维时为(B, S, n, n)，3维时为(T, n, n)。

- Shape一致性约束：
    - `grad_output`的shape需与`x`的shape完全一致。
    - 4维（BSND）格式下：
        - `h_res`的(B, S)维度需与`grad_output`的(B, S)维度一致，`h_res`的后两维为(n, n)，其中n与`grad_output`的第3维一致。
        - `h_out`的(B, S)维度需与`grad_output`的(B, S)维度一致，`h_out`的D维度需与`grad_output`的D维度一致。
        - `h_post`的(B, S)维度需与`grad_output`的(B, S)维度一致，`h_post`的n维度需与`grad_output`的n维度一致。
    - 3维（TND）格式下：
        - `h_res`的T维度需与`grad_output`的T维度一致，后两维为(n, n)。
        - `h_out`的T维度需与`grad_output`的T维度一致，D维度需与`grad_output`的D维度一致。
        - `h_post`的T维度需与`grad_output`的T维度一致，n维度需与`grad_output`的n维度一致。

- 所有输入Tensor的shape各维度值必须为正数（大于0）。

- 输入Tensor不支持非连续格式，算子内部会自动进行AutoContiguous处理。

- mhc_post_backward需与[mhc_post](mhc_post.md)配套使用，反向计算的输入`x`、`h_res`、`h_out`、`h_post`需与前向计算对应的输入保持一致。

- 参数里Shape使用的变量如下：
    - B：表示Batch批量大小。
    - S：表示Seq-Length序列长度。
    - T：表示所有Batch序列长度的累加和，$T = B \times S$。
    - n：表示head数量（头数）。
    - D：表示每个头的隐藏维度大小（headdim）。

## 确定性计算

`mhc_post_backward`默认确定性实现。

## 调用示例

- 单算子模式调用

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