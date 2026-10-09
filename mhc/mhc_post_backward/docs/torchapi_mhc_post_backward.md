# mhc_post_backward

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
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

- 接口功能：

    mhc_post对上一层输出$h_{l}^{out}$执行Post Mapping、对上一层输入$x_{l}$执行Res Mapping，再残差连接得到下一层输入$x_{l+1}$。`mhc_post_backward`实现该过程的反向计算：给定上游梯度`grad_output`，分别求解`x`、`h_res`、`h_out`、`h_post`的梯度。

- 计算公式：

  - 当传入`h_res`时：

    $$grad\_x = H_{l}^{res} \cdot grad\_output$$

    $$grad\_h\_res = x_{l} \cdot (grad\_output)^{T}$$

    $$grad\_h\_out = \sum_{i=0}^{n-1} grad\_output[i] \cdot H_{l}^{post}[i]$$

    $$grad\_h\_post[i] = \sum_{d=0}^{D-1} grad\_output[i, d] \cdot h_{l}^{out}[d]$$

    其中，$grad\_output$表示参数`grad_output`，$x_{l}$表示参数`x`，$H_{l}^{res}$表示参数`h_res`，$h_{l}^{out}$表示参数`h_out`，$H_{l}^{post}$表示参数`h_post`。

  - 当`h_res`传入None时，跳过Res Mapping，$grad\_x$退化为直接透传上游梯度：

    $$grad\_x = grad\_output$$

    $grad\_h\_res$、$grad\_h\_out$、$grad\_h\_post$的计算公式不变。

  >**说明：**
  >
  > 输入支持两种维度格式：BSND（4维）和TND（3维）。其中B（Batch）表示批量大小，S（Seq-Length）表示序列长度，T表示所有Batch序列长度的累加和（$T = B \times S$），n表示头数（head数量），D表示每个头的隐藏维度大小（headdim）。

## 函数原型

```python
cann_ops_transformer.mhc_post_backward(grad_output, x, h_res, h_out, h_post) -> (Tensor, Tensor, Tensor, Tensor)
```

返回值依次为`grad_x`、`grad_h_res`、`grad_h_out`、`grad_h_post`。

其中`h_res`为可选输入，可传入None（仅Ascend 950PR&950DT系列产品支持）。

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
        <td>下一层输入$x_{l+1}$的上游梯度，对应公式中的$grad\_output$。</td>
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
        <td>当前层的输入token特征，对应公式中的x<sub>l</sub>，shape与`grad_output`一致。</td>
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
        <td>Tensor</td>
        <td>可选</td>
        <td>残差连接矩阵，对应公式中的H<sub>l</sub><sup>res</sup>。传入None时跳过Res Mapping，仅Ascend 950PR&950DT系列产品支持，其他产品形态传入None会报错。</td>
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
        <td>后处理权重矩阵，对应公式中的H<sub>l</sub><sup>post</sup>。</td>
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
        <td>输入`x`的梯度，数据类型与shape均与`x`保持一致。</td>
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
        <td>残差连接矩阵`h_res`的梯度。`h_res`传入None时，shape由`x`推导（末维取`x`的倒数第2维，即n）。</td>
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
        <td>上一层输出状态`h_out`的梯度，数据类型与shape均与`h_out`保持一致。</td>
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
        <td>后处理权重`h_post`的梯度，数据类型与shape均与`h_post`保持一致。</td>
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

## 约束说明

- 该接口支持训练场景下使用。
- 该接口支持单算子模式和图模式。

- 数据类型约束：
  - `grad_output`、`x`、`h_out`的数据类型为bfloat16或float16。
  - `h_res`、`h_post`的数据类型仅支持float32。
  - `grad_x`与`x`、`grad_h_out`与`h_out`、`grad_h_post`与`h_post`的数据类型保持一致。

- `h_res`可选约束：
  - `h_res`传入None时跳过Res Mapping，仅Ascend 950PR&950DT系列产品支持；Atlas A2/A3系列产品`h_res`为必传参数，传入None会返回`ACLNN_ERR_PARAM_NULLPTR`错误码。
  - 图模式下`h_res`是算子定义的必选输入，不可省略该输入；传入None时接口层会转换为float32空Tensor下发，Ascend 950PR&950DT系列产品按该输入的元素个数是否为0决定是否启用Res Mapping。

- 维度约束：
  - `grad_output`与`x`的shape必须完全一致。
  - `h_res`的维度格式需与`x`匹配：4维时为(B, S, n, n)，3维时为(T, n, n)。
  - `h_out`比`x`少n维度：4维时为(B, S, D)，3维时为(T, D)。
  - `h_post`比`x`少D维度：4维时为(B, S, n)，3维时为(T, n)。

- Shape一致性约束：
  - 4维（BSND）格式下：
    - `h_res`的(B, S, n)维度需与`x`的前3维一致，末维为n。
    - `h_out`的(B, S)维度需与`x`的(B, S)维度一致，D维度需与`x`的D维度一致。
    - `h_post`的(B, S)维度需与`x`的(B, S)维度一致，n维度需与`x`的n维度一致。
  - 3维（TND）格式下：
    - `h_res`的T维度需与`x`的T维度一致，后两维为(n, n)。
    - `h_out`的T维度需与`x`的T维度一致，D维度需与`x`的D维度一致。
    - `h_post`的T维度需与`x`的T维度一致，n维度需与`x`的n维度一致。

<!-- npu="A3,910b" id7 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：维度n的取值目前仅支持4，维度D的取值需要128对齐并且取值范围为[1,100000]。
<!-- end id7 -->
<!-- npu="950" id8 -->
- <term>Ascend 950PR&950DT系列产品</term>：维度n的取值目前仅支持4、6和8。
<!-- end id8 -->

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

  grad_output = torch.randn(B, S, n, D, dtype=torch.bfloat16).npu()
  x = torch.randn(B, S, n, D, dtype=torch.bfloat16).npu()
  h_res = torch.randn(B, S, n, n, dtype=torch.float32).npu()
  h_out = torch.randn(B, S, D, dtype=torch.bfloat16).npu()
  h_post = torch.randn(B, S, n, dtype=torch.float32).npu()

  grad_x, grad_h_res, grad_h_out, grad_h_post = mhc_post_backward(
      grad_output, x, h_res, h_out, h_post
  )
  print(f"grad_x shape: {grad_x.shape}")

  # h_res缺省时（仅Ascend 950PR&950DT系列产品支持）
  grad_x, grad_h_res, grad_h_out, grad_h_post = mhc_post_backward(
      grad_output, x, None, h_out, h_post
  )
  print(f"grad_h_res shape: {grad_h_res.shape}")
  ```

- 图模式调用：

  ```python
  import torch
  import torch_npu
  import torchair
  from cann_ops_transformer.ops import mhc_post_backward

  torch_npu.npu.set_device(0)

  B = 2
  S = 8
  n = 4
  D = 128

  class MhcPostBackwardModel(torch.nn.Module):
      def forward(self, grad_output, x, h_res, h_out, h_post):
          return mhc_post_backward(grad_output, x, h_res, h_out, h_post)

  model = MhcPostBackwardModel().npu()
  npu_backend = torchair.get_npu_backend()
  model = torch.compile(model, backend=npu_backend, dynamic=False)

  grad_output = torch.randn(B, S, n, D, dtype=torch.bfloat16, device="npu")
  x = torch.randn(B, S, n, D, dtype=torch.bfloat16, device="npu")
  h_res = torch.randn(B, S, n, n, dtype=torch.float32, device="npu")
  h_out = torch.randn(B, S, D, dtype=torch.bfloat16, device="npu")
  h_post = torch.randn(B, S, n, dtype=torch.float32, device="npu")

  grad_x, grad_h_res, grad_h_out, grad_h_post = model(
      grad_output, x, h_res, h_out, h_post
  )
  ```
