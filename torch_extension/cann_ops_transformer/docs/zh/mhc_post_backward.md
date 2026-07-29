# mhc_post_backward

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：

    实现MHC Post组件的反向计算，用于Transformer模型中多层残差连接后处理阶段的梯度回传。该接口为[mhc_post](./mhc_post.md)的反向算子，输入正向计算的`x`、`h_res`、`h_out`、`h_post`以及正向输出`y`的梯度`grad_output`，返回各正向输入对应的梯度。

- 计算公式：

  - MHC Post算子的反向计算公式为：

    $$grad\_x_{l} = H_{l}^{res} \times grad\_output$$

    $$grad\_h\_res_{l} = x_{l} \times grad\_output^{T}$$

    $$grad\_h\_out = (grad\_output \times H_{t}^{post}.unsqueeze(-1)).sum(dim=-2)$$

    $$grad\_h\_post = (grad\_output \times h_{l}^{out}.unsqueeze(-2)).sum(dim=-1)$$

    其中，`grad_output`对应正向输出`y`的梯度，`x`对应公式中的$x_{l}$，`h_res`对应$H_{l}^{res}$，`h_out`对应$h_{l}^{out}$，`h_post`对应$H_{t}^{post}$，`grad_x`、`grad_h_res`、`grad_h_out`、`grad_h_post`分别为对应正向输入的梯度。

  >**说明：**
  >
  > 输入支持两种维度格式：BSND（4维）和TND（3维）。其中B（Batch）表示批量大小，S（Seq-Length）表示序列长度，T表示所有Batch有效序列长度的累加和，n表示头数（head数量），D表示每个头的隐藏维度大小（headdim）。

## 函数原型

```python
cann_ops_transformer.mhc_post_backward(grad_output, x, h_res, h_out, h_post) -> (Tensor, Tensor, Tensor, Tensor)
```

## 参数说明

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 120px">
<col style="width: 120px">
<col style="width: 100px">
<col style="width: 300px">
<col style="width: 120px">
<col style="width: 200px">
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
        <td>grad_output</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>正向输出y的梯度，对应公式中的grad_output。</td>
        <td>bfloat16、float16</td>
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
        <td>当前层的输入token特征，对应公式中的x<sub>l</sub>。</td>
        <td>bfloat16、float16</td>
        <td>
            <ul>
                <li>(B, S, n, D)</li>
                <li>(T, n, D)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>h_res</td>
        <td>Tensor或None</td>
        <td>可选</td>
        <td>残差连接矩阵，对应公式中的H<sub>l</sub><sup>res</sup>。仅<term>Ascend 950PR/Ascend 950DT</term>在eager模式下支持置为None或传入空Tensor（二者均按未传入处理）；<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>与<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>必须传入非空Tensor。</td>
        <td>float32</td>
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
        <td>上一层的输出状态，对应公式中的h<sub>l</sub><sup>out</sup>。</td>
        <td>bfloat16、float16</td>
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
        <td>后处理权重矩阵，对应公式中的H<sub>t</sub><sup>post</sup>。</td>
        <td>float32</td>
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

<table style="undefined;table-layout: fixed; width:1200px"><colgroup>
<col style="width: 120px">
<col style="width: 120px">
<col style="width: 100px">
<col style="width: 300px">
<col style="width: 120px">
<col style="width: 200px">
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
        <td>x的梯度，对应公式中的grad_x<sub>l</sub>，数据类型与输入`x`保持一致，shape与`x`保持一致。</td>
        <td>bfloat16、float16</td>
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
        <td>h_res的梯度，对应公式中的grad_h_res<sub>l</sub>。数据类型始终为float32，与`h_res`是否传入无关。当`h_res`未传入时，shape按`x`推导。</td>
        <td>float32</td>
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
        <td>h_out的梯度，对应公式中的grad_h_out，数据类型与输入`h_out`保持一致，shape与`h_out`保持一致。</td>
        <td>bfloat16、float16</td>
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
        <td>h_post的梯度，对应公式中的grad_h_post，数据类型与输入`h_post`保持一致，shape与`h_post`保持一致。</td>
        <td>float32</td>
        <td>
            <ul>
                <li>(B, S, n)</li>
                <li>(T, n)</li>
            </ul>
        </td>
    </tr>
</tbody>
</table>

> **说明**：当`h_res`为`None`或空Tensor时（仅<term>Ascend 950PR/Ascend 950DT</term> eager模式支持），`grad_x`等于`grad_output`，`grad_h_res`仍以float32输出，shape按`x`推导为`(B, S, n, n)`（BSND格式）或`(T, n, n)`（TND格式）。

## 约束说明

- 该接口支持训练场景下使用。
- 该接口支持单算子模式与TorchAir图模式调用。
- `h_res`为可选输入：仅<term>Ascend 950PR/Ascend 950DT</term>在eager模式下支持置为`None`或传入空Tensor（二者均按未传入处理）；<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>与<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>必须传入非空`h_res`。图模式下`h_res`不可为`None`或空Tensor（底层IR将`h_res`标记为必选输入）。
- 除`h_res`在<term>Ascend 950PR/Ascend 950DT</term> eager模式下可为空外，其余所有输入Tensor必须非空（各维度值大于0），且所有输入Tensor必须位于同一NPU设备上。
- 数据类型约束：
  - `grad_output`、`x`、`h_out`的数据类型必须相同（bfloat16或float16）。
  - `grad_x`的数据类型与`x`保持一致，`grad_h_out`与`h_out`保持一致，`grad_h_post`与`h_post`保持一致（float32）。
  - `grad_h_res`的数据类型始终为float32，与`h_res`是否传入无关。
- 维度约束：
  - `h_res`的维度需与`x`维度格式匹配：4维时为(B, S, n, n)，3维时为(T, n, n)。
- Shape一致性约束：
  - 4维（BSND）格式下：
    - `grad_output`的shape需与`x`一致。
    - `h_res`的(B, S)维度需与`x`的(B, S)维度一致，`h_res`的后两维为(n, n)，其中n与`x`的第3维一致。
    - `h_out`的(B, S)维度需与`x`的(B, S)维度一致，`h_out`的D维度需与`x`的D维度一致。
    - `h_post`的(B, S)维度需与`x`的(B, S)维度一致，`h_post`的n维度需与`x`的n维度一致。
  - 3维（TND）格式下：
    - `grad_output`的shape需与`x`一致。
    - `h_res`的T维度需与`x`的T维度一致，后两维为(n, n)。
    - `h_out`的T维度需与`x`的T维度一致，D维度需与`x`的D维度一致。
    - `h_post`的T维度需与`x`的T维度一致，n维度需与`x`的n维度一致。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：`n`仅支持4；`D`需按128对齐，取值范围为[1, 100000]。
- <term>Ascend 950PR/Ascend 950DT</term>：`n`支持4、6、8。

## 确定性计算

默认支持确定性计算。

## 调用说明

- 单算子模式调用：

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer.ops import mhc_post_backward

  B = 2
  S = 8
  n = 4
  D = 128

  grad_output = torch.randn(B, S, n, D, dtype=torch.float16).npu()
  x = torch.randn(B, S, n, D, dtype=torch.float16).npu()
  h_res = torch.randn(B, S, n, n, dtype=torch.float32).npu()
  h_out = torch.randn(B, S, D, dtype=torch.float16).npu()
  h_post = torch.randn(B, S, n, dtype=torch.float32).npu()

  grad_x, grad_h_res, grad_h_out, grad_h_post = mhc_post_backward(
      grad_output, x, h_res, h_out, h_post
  )
  print(f"grad_x shape: {grad_x.shape}")
  ```

  `h_res`不传入的场景（仅<term>Ascend 950PR/Ascend 950DT</term> eager模式支持，此时`grad_x`等于`grad_output`）：

  ```python
  grad_x, grad_h_res, grad_h_out, grad_h_post = mhc_post_backward(
      grad_output, x, None, h_out, h_post
  )
  ```

- 图模式调用：

  图模式下`h_res`必须传入非空Tensor，不可为`None`或空Tensor。

  ```python
  import torch
  import torch_npu
  import torchair
  from cann_ops_transformer.ops import mhc_post_backward

  B = 2
  S = 8
  n = 4
  D = 128

  grad_output = torch.randn(B, S, n, D, dtype=torch.float16).npu()
  x = torch.randn(B, S, n, D, dtype=torch.float16).npu()
  h_res = torch.randn(B, S, n, n, dtype=torch.float32).npu()
  h_out = torch.randn(B, S, D, dtype=torch.float16).npu()
  h_post = torch.randn(B, S, n, dtype=torch.float32).npu()

  compiled = torch.compile(mhc_post_backward, backend=torchair.get_npu_backend(), dynamic=False)
  grad_x, grad_h_res, grad_h_out, grad_h_post = compiled(grad_output, x, h_res, h_out, h_post)
  ```
