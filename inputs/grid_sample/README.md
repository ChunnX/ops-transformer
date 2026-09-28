# GridSample

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Kirin X90 处理器系列产品</term> | √ |
|  <term>Kirin 9030 处理器系列产品</term> | √ |

## 功能说明

- 算子功能：提供一个输入tensor以及一个对应的grid网格，然后根据grid中每个位置提供的坐标信息，将input中对应位置的像素值填充到网格指定的位置，得到最终的输出。
- 计算公式：

  - 计算流程：
    1. 根据grid存储的(x, y)或者(x, y, z)值，计算出映射到input上坐标，这些坐标和align_corners、padding_mode有关。
    2. 坐标根据输入的interpolation_mode，选择使用bilinear、nearest、bicubic不同插值模式计算输出值。

  - 其中：
    - 3D场景：

      输入input、grid网格、输出output的尺寸如下：

      $$
      input: (N, C, D_{in}, H_{in}, W_{in})\\
      grid: (N, D_{out}, H_{out}, W_{out}, 3)\\
      output: (N, C, D_{out}, H_{out}, W_{out})
      $$

      其中input、grid、output中的N是一致的，input和output中的C是一致的，grid和output中的$D_{out}$、$H_{out}$、$W_{out}$是一致的，grid最后一维大小为3，表示input像素位置信息为(x, y, z)，会将x、y、z的取值范围归一化到[-1, 1]之间。

      - 反归一化的计算公式：
        - align_corners=true，表示特征值位于像素中心。

         $$
         x' = (grid\_x + 1) / 2 * (W_{in} - 1)
         $$

         $$
         y' = (grid\_y +1) / 2 * (H_{in} - 1)
         $$

         $$
         z' = (grid\_z +1) / 2 * (D_{in} - 1)
         $$

        - align_corners=false，表示特征值位于像素的角点。

         $$
         x' = ((grid\_x +1) * W_{in} - 1) / 2
         $$

         $$
         y' = ((grid\_y +1) * H_{in} - 1) / 2
         $$

         $$
         z' = ((grid\_z +1) * D_{in} - 1) / 2
         $$

      - 对于超出范围的坐标，会根据paddingMode进行不同处理：
        - paddingMode=0，表示对越界位置用0填充。
        - paddingMode=1，表示对越界位置用边界值填充。
        - paddingMode=2，表示对越界位置用边界值的对称值填充。
      - 对input采样时，会根据interpolationMode进行不同处理：
        - interpolationMode="bilinear"，表示取input中(x, y, z)周围八个坐标的加权平均值。

          $$
          {output(N, C, D_{out}, H_{out}, W_{out})} = \sum_{i=0}^{1}\sum_{j=0}^{1}\sum_{k=0}^{1}{w(i, j, k)} * {f(x', y', z')}
          $$

          其中：
          - $f(x', y', z')$是原图像在$(x', y', z')$的像素值。
          - $w(i, j, k)$是双线性插值周边8个点的权重，计算公式为：

            $$
              w(i) = \begin{cases}
              1 - |x'_i - x_i| & |x'_i - x_i| < 1 \\
              0 & otherwise
              \end{cases}
            $$

            $$
              w(j) = \begin{cases}
              1 - |y'_j - y_j| & |y'_j - y_j| < 1 \\
              0 & otherwise
              \end{cases}
            $$

            $$
              w(k) = \begin{cases}
              1 - |z'_k - z_k| & |z'_k - z_k| < 1 \\
              0 & otherwise
              \end{cases}
            $$

            $$
            w(i, j, k)= w(i) * w(j) * w(k)
            $$

        - interpolationMode="nearest"，表示取input中距离(x, y, z)最近的坐标值。

          $$
          output(N, C, D_{out}, H_{out}, W_{out}) = input(N, C, D_{in}, H_{in}, W_{in})
          $$

          其中：

          $$
          W_{out} = min(round(x'),  W_{in} - 1)\\
          W_{out} = max(round(x'),  0)
          $$

          $$
          H_{out} = min(round(y'),  H_{in} - 1)\\
          H_{out} = max(round(y'),  0)
          $$

          $$
          D_{out} = min(round(z'),  D_{in} - 1)\\
          D_{out} = max(round(z'),  0)
          $$

    - 2D场景：

      输入input、grid网格、输出output的尺寸如下：

      $$
      input: (N, C, H_{in}, W_{in})\\
      grid: (N, H_{out}, W_{out}, 2)\\
      output: (N, C, H_{out}, W_{out})
      $$

      其中input、grid、output中的N是一致的，input和output中的C是一致的，grid和output中的$H_{out}$、$W_{out}$是一致的，grid最后一维大小为2，表示input像素位置信息为(x, y)，会将x和y的取值范围归一化到[-1, 1]之间，(-1, 1)表示左上角坐标，(1, -1)表示右下角坐标。

      - 反归一化的计算公式：
        - align_corners=true，表示特征值位于像素中心。

          $$
          x' = (grid\_x + 1) / 2 * (W_{in} - 1)
          $$

          $$
          y' = (grid\_y +1) / 2 * (H_{in} - 1)
          $$

        - align_corners=false，表示特征值位于像素的角点。

          $$
          x' = ((grid\_x +1) * W_{in} - 1) / 2
          $$

          $$
          y' = ((grid\_y +1) * H_{in} - 1) / 2
          $$

      - 对于超出范围的坐标，会根据paddingMode进行不同处理：

        - paddingMode=0，表示对越界位置用0填充。
        - paddingMode=1，表示对越界位置用边界值填充。
        - paddingMode=2，表示对越界位置用边界值的对称值填充。

      - 对input采样时，会根据interpolationMode进行不同处理：

        - interpolationMode=0，表示取(x, y)周围四个坐标的加权平均值。

          $$
          output(N, C, H_{out}, W_{out}) = \sum_{i=0}^{1}\sum_{j=0}^{1}{w(i, j)} * {f(x', y')}
          $$

          其中：
          - $f(x', y')$是原图像在$(x', y')$的像素值。
          - $w(i, j)$是双线性插值周边4个点的权重，计算公式为：

            $$
              w(i) = \begin{cases}
              1 - |x'_i - x_i| & |x'_i - x_i| < 1 \\
              0 & otherwise
              \end{cases}
            $$

            $$
              w(j) = \begin{cases}
              1 - |y'_j - y_j| & |y'_j - y_j| < 1 \\
              0 & otherwise
              \end{cases}
            $$

            $$
            w(i, j)= w(i) * w(j)
            $$

        - interpolationMode=1，表示取input中距离(x, y)最近的坐标值。

          $$
          output(N, C, H_{out}, W_{out}) = input(N, C, H_{in}, W_{in})
          $$

          其中：

          $$
          W_{out} = min(round(x'),  W_{in} - 1)\\
          W_{out} = max(round(x'),  0)
          $$

          $$
          H_{out} = min(round(y'),  H_{in} - 1)\\
          H_{out} = max(round(y'),  0)
          $$

        - interpolationMode=2，表示取(x, y)周围十六个坐标的加权平均值。

          $$
          {output(N, C, H_{out}, W_{out})}=\sum_{i=0}^{3}\sum_{j=0}^{3}{W(i, j)}*{f(x', y')}
          $$

          其中：
          - i和j是$W(i, j)$的索引变量。
          - $f(x', y')$是原图像在$(x', y')$的像素值。
          - $W(i, j)$是双三次插值的权重，定义为：

            $$
            W(d) =\begin{cases}
            (a+2)|d|^3-(a+3)|d|^2+1 & |d|\leq1 \\
            a|d|^3-5a|d|^2+8a|d|-4a & 1<|d|<2 \\
            0 & otherwise
            \end{cases}
            $$

            其中：
            - $a=-0.75$
            - $d = |(x_i, y_j) - (x'_i, y'_j)|$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
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
      <td>x</td>
      <td>输入</td>
      <td>进行插值计算的输入张量，对应公式中描述的`input`。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grid</td>
      <td>输入</td>
      <td>采样的网格，对应公式中描述的`grid`。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>interpolation_mode</td>
      <td>可选属性</td>
      <td><ul><li>表示插值模式，对应公式描述中的`interpolation_mode`。支持bilinear（双线性插值）、nearest（最邻近插值）、bicubic（双三次插值）。</li><li>默认值为"bilinear"。</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding_mode</td>
      <td>可选属性</td>
      <td><ul><li>表示填充模式，对应公式描述中的`padding_mode`。支持zeros(0)、border(1)、reflection(2)三种模式。</li><li>默认值为"zeros"。</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>align_corners</td>
      <td>可选属性</td>
      <td><ul><li>表示设定特征图坐标与特征值的对应方式，设定为true时，特征值位于像素中心。设定为false时，特征值位于像素的角点。</li><li>默认值为false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>channel_last</td>
      <td>可选属性</td>
      <td><ul><li>指定输入x的C轴是否在最后。如果为true，则x的形状为`[batch, height, width, channels]`，否则为`[batch, channels, height, width]`。</li><li>默认值为false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scheduler_mode</td>
      <td>可选属性</td>
      <td><ul><li>调度模式，执行的操作方式。0：一般；1：滑动窗口。仅当channel_last的值为true时，配置为1有效。</li><li>默认值为1。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>插值计算的最终输出结果，对应公式中描述的`output`。
      </td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas 推理系列产品</term>：

  - 2D场景下，需要同时满足以下条件：
    - 输入和输出的数据类型仅支持FLOAT32。
    - 属性`interpolation_mode`需要设置为`bilinear`。
    - 属性`padding_mode`设置为`zeros`。
    - 输入`x`的C轴的值为32或者$（C轴的大小 * H轴的大小 * W轴的大小） < 20k$。
  - 不支持3D场景。

- <term>Kirin X90 处理器系列产品</term>、<term>Kirin 9030 处理器系列产品</term>：不支持BFLOAT16。

## 约束说明

- 3D场景下
  - 属性`interpolation_mode`不支持配置为"bicubic"。
  - 输入`x`的$（D轴的大小 * H轴的大小 * W轴的大小） < INT32的最大值$。
- 2D场景下，输入`x`的$（H轴的大小 * W轴的大小） < INT32的最大值$。
- `x`和`grid`的shape，所有维度都必须大于0。
- 输入`x`的$（D轴的大小 * H轴的大小 * W轴的大小） < INT32的最大值$。
- grid的$输入值 * 图片（长或宽）> 24位的二进制数（16777216）$，采样点可能存在误差，精度可能产生偏差。
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：如果grid存在超出[-1, 1]范围的数据，使用bicubic插值时，小值域数据计算可能存在误差，精度可能产生偏差。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_grid_sample2_d](examples/test_aclnn_grid_sample2_d.cpp) | 通过[aclnnGridSampler2D](docs/aclnnGridSampler2D.md)接口方式调用GridSample算子。 |
| aclnn接口  | [test_aclnn_grid_sample3_d](examples/test_aclnn_grid_sample3_d.cpp) | 通过[aclnnGridSampler3D](docs/aclnnGridSampler3D.md)接口方式调用GridSample算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/grid_sample_proto.h)构图方式调用GridSample算子。         |
