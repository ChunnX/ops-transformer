# mhc_pre_split_post_res

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id5 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id5 -->
<!-- npu="310p" id4 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id4 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：

    mhc_pre_split_post_res是mhc_pre算子的一部分，负责计算mHC（Manifold-Constrained Hyper-Connections）架构中x2输入对应的post分支和comb/Res分支，输出h_post和h_res两个变换矩阵（h_res为可选输出：phi行数为n^2+n时输出，为n时仅输出h_post）。该算子作为独立的纯Vector算子，可与Atten/MLP层的Cube算子做多流并行，掩盖该部分计算时延。

- 计算公式：

  - RMSNorm因子计算（其中 $n = N \cdot D$）：

    $$inv\_rms = \frac{1}{\sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2 + norm\_eps}}$$

  - 线性投影与归一化融合：

    $$x\_hat = \left(x\_flat \cdot phi^{T}\right) \odot inv\_rms$$

  - post分支（取x_hat前N维）：

    $$h\_post = \text{Sigmoid}(\alpha_0 \cdot x\_hat[0:N] + bias[0:N]) * 2$$

  - comb/Res分支（取x_hat后N·N维）：

    $$h\_res = \alpha_1 \cdot reshape(x\_hat[N:N^2+N], [N, N]) + reshape(bias[N:N^2+N], [N, N])$$

  >**说明：**
  >
  > 输入支持两种维度格式：BSND（4维）和TND（3维）。其中B（Batch）表示批量大小，S（Seq-Length）表示序列长度，T表示所有Batch序列长度的累加和（$T = B \times S$），n表示mHC超连接路数，D表示隐藏层最小的单元尺寸。

## 函数原型

```python
cann_ops_transformer.mhc_pre_split_post_res(x, phi, alpha, bias, norm_eps=1e-6) -> (Tensor, Tensor)
```

其中`norm_eps`为关键字可选参数；`*`之前为位置相关必选参数。

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
        <td>x</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>待计算的数据，表示网络中mHC层的输入数据x2。支持非连续的Tensor。</td>
        <td>bfloat16、float16</td>
        <td>
            <ul>
                <li>(B, S, n, D)</li>
                <li>(T, n, D)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>phi</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>mHC的变换矩阵phi（线性投影矩阵）。支持非连续的Tensor。</td>
        <td>float32</td>
        <td>
            <ul>
                <li>(n<sup>2</sup>+n, n·D)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>alpha</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>mHC的缩放参数，对应post/comb两分支的缩放系数。支持非连续的Tensor。</td>
        <td>float32</td>
        <td>
            <ul>
                <li>(2)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>bias</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>mHC的bias参数，对应post/comb两分支的偏置。支持非连续的Tensor。</td>
        <td>float32</td>
        <td>
            <ul>
                <li>(n<sup>2</sup>+n)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>norm_eps</td>
        <td>float</td>
        <td>可选</td>
        <td>RMSNorm防除0参数（数值稳定项），必须大于0。</td>
        <td>float32</td>
        <td>-</td>
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
        <td>h_post</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>mHC post分支变换矩阵，对应公式中的h_post。</td>
        <td>float32</td>
        <td>
            <ul>
                <li>(B, S, n)</li>
                <li>(T, n)</li>
            </ul>
        </td>
    </tr>
    <tr>
        <td>h_res</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>mHC comb/Res分支输出矩阵。完整形态（phi行数为n²+n）时返回实际结果；仅post形态（phi行数为n）时返回0元素Tensor，以保持二元组返回结构。</td>
        <td>float32</td>
        <td>
            <ul>
                <li>(B, S, n, n)</li>
                <li>(T, n, n)</li>
            </ul>
        </td>
    </tr>
</tbody>
</table>

## 约束说明

- 该接口支持推理场景下使用。
- 该接口支持单算子模式和图模式调用。
- 数据类型约束：
  - `x`的数据类型为bfloat16或float16；`phi`、`alpha`、`bias`的数据类型必须为float32。
  - 输出`h_post`、`h_res`的数据类型均为float32。

- 维度约束：
  - `x`支持4维（B, S, n, D）或3维（T, n, D）格式，其中n固定为4，D仅支持白名单{2560, 5120, 6144, 7168}；T无上限（>=1）。
  - `phi`的维度为(n<sup>2</sup>+n, n·D)（完整形态，输出h_res）或(n, n·D)（仅post形态，h_res返回0元素Tensor），其中n、D与`x`保持一致。
  - `alpha`的维度为(2)。
  - `bias`的维度与`phi`的第0维保持一致（n<sup>2</sup>+n或n）。

- Shape一致性约束：
  - 4维（BSND）格式下：`h_post`的(B, S)维度需与`x`的(B, S)维度一致；`h_res`的(B, S)维度需与`x`的(B, S)维度一致。
  - 3维（TND）格式下：`h_post`、`h_res`的T维度需与`x`的T维度一致。

- T或B*S至少为1，当前实现不设置上限。
- `norm_eps`必须大于0。
- 输入Tensor不支持空Tensor；输出`h_res`在仅post形态（phi行数为n）时为0元素Tensor（允许）。
- 所有输入Tensor的shape各维度值必须为正数（大于0）。

## 确定性计算

默认支持确定性计算。

## 调用说明

- 单算子模式调用：

  ```python
  import torch
  import torch_npu
  from cann_ops_transformer.ops import mhc_pre_split_post_res

  B = 1
  S = 4
  n = 4
  D = 5120

  x = torch.randn(B, S, n, D, dtype=torch.bfloat16).npu()
  phi = torch.randn(n * n + n, n * D, dtype=torch.float32).npu()
  alpha = torch.randn(2, dtype=torch.float32).npu()
  bias = torch.randn(n * n + n, dtype=torch.float32).npu()

  h_post, h_res = mhc_pre_split_post_res(x, phi, alpha, bias)
  print(f"h_post shape: {h_post.shape}")
  print(f"h_res shape: {h_res.shape}")

  # 显式指定norm_eps
  h_post, h_res = mhc_pre_split_post_res(x, phi, alpha, bias, norm_eps=1e-5)
  print(f"h_post shape: {h_post.shape}")
  ```

- 图模式调用：

  ```python
  import torch
  import torch_npu
  import torchair
  from cann_ops_transformer.ops import mhc_pre_split_post_res

  torch_npu.npu.set_device(0)

  B = 1
  S = 4
  n = 4
  D = 5120

  class MhcPreSplitPostResModel(torch.nn.Module):
      def forward(self, x, phi, alpha, bias):
          return mhc_pre_split_post_res(x, phi, alpha, bias)

  model = MhcPreSplitPostResModel().npu()
  npu_backend = torchair.get_npu_backend()
  model = torch.compile(model, backend=npu_backend, dynamic=False)

  x = torch.randn(B, S, n, D, dtype=torch.bfloat16, device="npu")
  phi = torch.randn(n * n + n, n * D, dtype=torch.float32, device="npu")
  alpha = torch.randn(2, dtype=torch.float32, device="npu")
  bias = torch.randn(n * n + n, dtype=torch.float32, device="npu")

  h_post, h_res = model(x, phi, alpha, bias)
  print(f"h_post shape: {h_post.shape}")
  print(f"h_res shape: {h_res.shape}")
  ```
