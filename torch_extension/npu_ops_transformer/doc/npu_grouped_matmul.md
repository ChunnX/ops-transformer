# npu\_grouped\_matmul<a name="ZH-CN_TOPIC_0000002309174913"></a>

## 产品支持情况

| 产品                                                         | 是否支持 |
| ------------------------------------------------------------ | :------: |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>           |    √     |


## 功能说明<a name="zh-cn_topic_0000002168254827_section14441124184110"></a>

-   API功能：

    分组矩阵乘法（Grouped Matrix Multiplication），对多组输入矩阵和权重矩阵分别进行矩阵乘法运算，支持偏置、激活函数、量化等功能。
     - 支持M轴分组：各组的k、n维度相同，m维度可以不同；
     - 支持K轴分组：各组的m、n维度相同，k维度可以不同；
     - 支持量化模式：INT8、INT4、FP8量化；
     - 支持激活函数：ReLU、GELU、SiLU等。
-   计算公式：
    - 基础模式：

      $$y_i = x_i \times weight_i + bias_i$$

    - 激活函数模式：

      $$y_i = activation(x_i \times weight_i + bias_i)$$

    - 量化模式：

      $$y_i = dequant(quant(x_i) \times quant(weight_i)) + bias_i$$

    其中，$i = 1, 2, ..., g$，$g$为分组数量。


## 函数原型<a name="zh-cn_topic_0000002168254827_section45077510411"></a>

```
npu_grouped_matmul_v5(x, weight, bias=None, scale=None, offset=None, antiquant_scale=None, antiquant_offset=None, per_token_scale=None, group_list=None, activation_input=None, activation_quant_scale=None, activation_quant_offset=None, split_item=0, group_type=-1, group_list_type=0, act_type=0, tuning_config=None) -> (Tensor[], Tensor[]?, Tensor[]?)
```

## 参数说明<a name="zh-cn_topic_0000002168254827_section112637109429"></a>

-   **x** (`Tensor[]`)：必选参数，输入张量列表，每个张量要求为2维或更高维度，shape为\(..., M_i, K_i\)，数据类型支持`bfloat16`、`float16`、`int8`、`int4`，数据格式为$ND$，支持非连续的Tensor。

-   **weight** (`Tensor[]`)：必选参数，权重张量列表，每个张量要求为2维或更高维度，shape为\(..., K_i, N_i\)。数据类型需与`x`保持一致，数据格式为$ND$，支持非连续的Tensor。要求`len(weight) == len(x)`。

-   **bias** (`Tensor[]`)：可选参数，偏置张量列表，每个张量要求为1维，shape为\(N_i, \)。数据类型支持`bfloat16`、`float16`、`float32`，数据格式为$ND$，支持非连续的Tensor。当传入有效数据时，要求`len(bias) == len(x)`。

-   **scale** (`Tensor[]`)：可选参数，量化缩放因子张量列表，用于权重量化场景。每个张量shape根据量化模式而定。数据类型支持`float32`、`bfloat16`、`float16`，数据格式为$ND$，支持非连续的Tensor。

-   **offset** (`Tensor[]`)：可选参数，量化偏移量张量列表，用于权重量化场景。每个张量shape根据量化模式而定。数据类型支持`float32`、`bfloat16`、`float16`，数据格式为$ND$，支持非连续的Tensor。

-   **antiquant\_scale** (`Tensor[]`)：可选参数，反量化缩放因子张量列表。每个张量shape根据量化模式而定。数据类型支持`float32`、`bfloat16`、`float16`，数据格式为$ND$，支持非连续的Tensor。

-   **antiquant\_offset** (`Tensor[]`)：可选参数，反量化偏移量张量列表。每个张量shape根据量化模式而定。数据类型支持`float32`、`bfloat16`、`float16`，数据格式为$ND$，支持非连续的Tensor。

-   **per\_token\_scale** (`Tensor[]`)：可选参数，逐token缩放因子张量列表，用于动态量化场景。每个张量要求为1维或2维，shape为\(M_i, \)或\(M_i, 1\)。数据类型支持`float32`、`bfloat16`、`float16`，数据格式为$ND$，支持非连续的Tensor。

-   **group\_list** (`Tensor`)：可选参数，分组列表张量，用于M轴或K轴分组场景。要求为1维张量，shape为\(g, \)，其中$g$为分组数量。数据类型支持`int64`、`int32`，数据格式为$ND$，支持非连续的Tensor。根据`group_list_type`参数，可以是累加和格式或每组大小格式。

-   **activation\_input** (`Tensor[]`)：预留参数，当前版本不支持，传默认值None即可。

-   **activation\_quant\_scale** (`Tensor[]`)：预留参数，当前版本不支持，传默认值None即可。

-   **activation\_quant\_offset** (`Tensor[]`)：预留参数，当前版本不支持，传默认值None即可。

-   **split\_item** (`int`)：可选参数，输出模式。取值范围：
    -   0或1：多张量输出模式，输出`g`个张量，每个张量shape为\(M_i, N_i\)；
    -   2或3：单张量输出模式，输出1个张量，shape为\(sum(M_i), N\)。

    默认值为0。

-   **group\_type** (`int`)：可选参数，分组类型。取值范围：
    -   -1：无分组，所有输入独立计算；
    -   0：M轴分组，各组的K、N维度相同，M维度可以不同；
    -   1：N轴分组（当前版本不支持）；
    -   2：K轴分组，各组的M、N维度相同，K维度可以不同。

    默认值为-1。

-   **group\_list\_type** (`int`)：可选参数，分组列表类型。取值范围：
    -   0：累加和格式，`group_list[i]`表示前`i+1`组的累加和；
    -   1：每组大小格式，`group_list[i]`表示第`i`组的大小。

    默认值为0。

-   **act\_type** (`int`)：可选参数，激活函数类型。取值范围：
    -   0：无激活函数；
    -   1：ReLU；
    -   2：GELU (tanh近似)；
    -   3：GELU (erf精确)；
    -   4：Fast GELU；
    -   5：SiLU (Swish)。

    默认值为0。

-   **tuning\_config** (`int[]`)：可选参数，性能调优配置参数列表。具体取值由底层算子实现决定，一般情况下传默认值None即可。

## 返回值说明<a name="zh-cn_topic_0000002168254827_section22231435517"></a>
`Tuple[Tensor[], Tensor[]?, Tensor[]?]`

返回一个三元组：
-   **out** (`Tensor[]`)：输出张量列表。
    -   当`split_item`为0或1时，返回`g`个张量，每个张量shape为\(M_i, N_i\)；
    -   当`split_item`为2或3时，返回1个张量，shape为\(sum(M_i), N\)。

    数据类型与输入`x`保持一致（量化场景除外），数据格式为$ND$。

-   **activation\_feature\_out** (`Tensor[]?`)：预留输出，当前版本返回None。

-   **dyn\_quant\_scale\_out** (`Tensor[]?`)：预留输出，当前版本返回None。

## 约束说明<a name="zh-cn_topic_0000002168254827_section12345537164214"></a>

-   该接口支持训练和推理场景下使用。
-   该接口支持单算子模式和静态图模式。
-   参数里Shape使用的变量如下：
    -   $g$：表示分组数量，取值范围\[1, 1024\]。
    -   $M_i$：表示第$i$组输入的M维度大小，取值范围\[1, 65536\]。
    -   $K_i$：表示第$i$组输入的K维度大小，取值范围\[1, 65536\]。
    -   $N_i$：表示第$i$组输出的N维度大小，取值范围\[1, 65536\]。

-   M轴分组约束：
    -   所有组的$K_i$和$N_i$必须相同，即$K_1 = K_2 = ... = K_g$，$N_1 = N_2 = ... = N_g$；
    -   各组的$M_i$可以不同；
    -   需要提供`group_list`参数指定各组的M维度大小。

-   K轴分组约束：
    -   所有组的$M_i$和$N_i$必须相同，即$M_1 = M_2 = ... = M_g$，$N_1 = N_2 = ... = N_g$；
    -   各组的$K_i$可以不同；
    -   需要提供`group_list`参数指定各组的K维度大小。

-   量化场景约束：
    -   输入`x`和`weight`的数据类型为`int8`或`int4`时，必须提供`scale`参数；
    -   量化参数（`scale`、`offset`、`antiquant_scale`、`antiquant_offset`）的shape需与量化模式匹配；
    -   量化输出的数据类型由`antiquant_scale`决定，一般为`float16`或`bfloat16`。

-   激活函数约束：
    -   激活函数在矩阵乘法和偏置加法之后应用；
    -   GELU有两种实现：tanh近似（速度快）和erf精确（精度高）。

## 调用示例<a name="zh-cn_topic_0000002168254827_section14459801435"></a>

-   单算子模式调用 - 基础用法

    ```python
    import torch
    import torch_npu
    import npu_ops_transformer

    # 创建输入数据
    num_groups = 8
    x_list = [torch.randn(128, 256, dtype=torch.float16).npu() for _ in range(num_groups)]
    weight_list = [torch.randn(256, 512, dtype=torch.float16).npu() for _ in range(num_groups)]

    # 方式1：使用高层封装接口
    gmm = npu_ops_transformer.ops.GroupedMatmul()
    out, _, _ = gmm(x_list, weight_list)

    print(f"Input: {num_groups} tensors of shape {x_list[0].shape}")
    print(f"Weight: {num_groups} tensors of shape {weight_list[0].shape}")
    print(f"Output: {len(out)} tensors of shape {out[0].shape}")

    # 方式2：使用便捷函数
    out, _, _ = npu_ops_transformer.ops.grouped_matmul(x_list, weight_list)

    # 方式3：直接调用底层算子
    out, _, _ = torch.ops.npu_ops_transformer.npu_grouped_matmul_v5(
        x_list, weight_list, None, None, None, None, None, None,
        None, None, None, None, 0, -1, 0, 0, None
    )
    ```

-   单算子模式调用 - 带偏置和激活函数

    ```python
    import torch
    import torch_npu
    import npu_ops_transformer

    num_groups = 8
    x_list = [torch.randn(128, 256, dtype=torch.float16).npu() for _ in range(num_groups)]
    weight_list = [torch.randn(256, 512, dtype=torch.float16).npu() for _ in range(num_groups)]
    bias_list = [torch.randn(512, dtype=torch.float16).npu() for _ in range(num_groups)]

    gmm = npu_ops_transformer.ops.GroupedMatmul()

    # 使用ReLU激活函数
    out, _, _ = gmm.forward_with_activation(
        x_list, weight_list, bias_list,
        act_type=npu_ops_transformer.ops.GroupedMatmulConfig.ACT_RELU
    )

    # 或者使用GELU激活函数
    out, _, _ = gmm(
        x_list, weight_list, bias_list,
        act_type=npu_ops_transformer.ops.GroupedMatmulConfig.ACT_GELU_TANH
    )

    print(f"Output with activation: {out[0].shape}")
    ```

-   单算子模式调用 - 量化模式

    ```python
    import torch
    import torch_npu
    import npu_ops_transformer

    num_groups = 8
    # INT8量化输入
    x_list = [torch.randint(-128, 127, (128, 256), dtype=torch.int8).npu()
              for _ in range(num_groups)]
    weight_list = [torch.randint(-128, 127, (256, 512), dtype=torch.int8).npu()
                   for _ in range(num_groups)]
    scale_list = [torch.tensor([0.01], dtype=torch.float32).npu()
                  for _ in range(num_groups)]

    gmm = npu_ops_transformer.ops.GroupedMatmul()
    out, _, _ = gmm.forward_quantized(
        x_list, weight_list, scale_list
    )

    print(f"Quantized output: {out[0].shape}, dtype: {out[0].dtype}")
    ```

-   单算子模式调用 - M轴分组

    ```python
    import torch
    import torch_npu
    import npu_ops_transformer

    num_groups = 8
    group_sizes = [128, 256, 512, 128, 256, 512, 128, 256]  # 各组M维度大小

    # 创建不同M维度的输入张量
    x_list = [torch.randn(m, 256, dtype=torch.float16).npu() for m in group_sizes]
    weight_list = [torch.randn(256, 512, dtype=torch.float16).npu() for _ in range(num_groups)]

    # 创建分组列表（累加和格式）
    group_list = npu_ops_transformer.ops.GroupedMatmul.create_group_list_cumsum(group_sizes)

    gmm = npu_ops_transformer.ops.GroupedMatmul()
    out, _, _ = gmm(
        x_list, weight_list,
        group_list=group_list.npu(),
        group_type=npu_ops_transformer.ops.GroupedMatmulConfig.GROUP_M_AXIS,
        group_list_type=npu_ops_transformer.ops.GroupedMatmulConfig.GROUP_LIST_CUMSUM
    )

    print(f"M-axis grouped output: {len(out)} tensors")
    for i, tensor in enumerate(out):
        print(f"  Group {i}: shape {tensor.shape}")
    ```

-   单算子模式调用 - 单张量输出

    ```python
    import torch
    import torch_npu
    import npu_ops_transformer

    num_groups = 8
    x_list = [torch.randn(128, 256, dtype=torch.float16).npu() for _ in range(num_groups)]
    weight_list = [torch.randn(256, 512, dtype=torch.float16).npu() for _ in range(num_groups)]

    gmm = npu_ops_transformer.ops.GroupedMatmul()
    out, _, _ = gmm(
        x_list, weight_list,
        split_item=npu_ops_transformer.ops.GroupedMatmulConfig.SPLIT_SINGLE_TENSOR
    )

    # 输出为单个张量，shape为(128*8, 512) = (1024, 512)
    print(f"Single tensor output: {out[0].shape}")
    ```

-   单算子模式调用 - 不同激活函数对比

    ```python
    import torch
    import torch_npu
    import npu_ops_transformer

    num_groups = 4
    x_list = [torch.randn(64, 128, dtype=torch.float16).npu() for _ in range(num_groups)]
    weight_list = [torch.randn(128, 256, dtype=torch.float16).npu() for _ in range(num_groups)]
    bias_list = [torch.randn(256, dtype=torch.float16).npu() for _ in range(num_groups)]

    gmm = npu_ops_transformer.ops.GroupedMatmul()
    config = npu_ops_transformer.ops.GroupedMatmulConfig

    activations = [
        ("None", config.ACT_NONE),
        ("ReLU", config.ACT_RELU),
        ("GELU (tanh)", config.ACT_GELU_TANH),
        ("GELU (erf)", config.ACT_GELU_ERF),
        ("Fast GELU", config.ACT_FAST_GELU),
        ("SiLU", config.ACT_SILU),
    ]

    for act_name, act_type in activations:
        out, _, _ = gmm(x_list, weight_list, bias_list, act_type=act_type)
        print(f"{act_name}: Output shape {out[0].shape}")
    ```

-   图模式调用

    ```python
    import torch
    import torch_npu
    import torchair
    import npu_ops_transformer

    class GroupedMatmulModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gmm = npu_ops_transformer.ops.GroupedMatmul()

        def forward(self, x_list, weight_list, bias_list):
            out, _, _ = self.gmm(x_list, weight_list, bias_list,
                                act_type=npu_ops_transformer.ops.GroupedMatmulConfig.ACT_RELU)
            return out

    # 创建输入数据
    num_groups = 8
    x_list = [torch.randn(128, 256, dtype=torch.float16).npu() for _ in range(num_groups)]
    weight_list = [torch.randn(256, 512, dtype=torch.float16).npu() for _ in range(num_groups)]
    bias_list = [torch.randn(512, dtype=torch.float16).npu() for _ in range(num_groups)]

    # 编译模型
    model = GroupedMatmulModel().npu()
    npu_backend = torchair.get_npu_backend()
    model = torch.compile(model, backend=npu_backend, dynamic=False)

    # 执行推理
    output = model(x_list, weight_list, bias_list)
    torch.npu.synchronize()

    print(f"Graph mode output: {len(output)} tensors")
    print(f"First tensor shape: {output[0].shape}")
    ```
