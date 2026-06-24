# inplace_partial_rotary_mul

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

-   接口功能：

    执行单路旋转位置编码的Inplace计算，直接修改输入张量`x`，不产生新的输出张量。通过`partial_slice`参数支持对输入张量最后一维进行部分旋转位置编码计算，仅对指定范围的数据进行旋转编码。

-   计算公式：

    interleave模式（`rotary_mode`为"interleave"）：

    $$x_1 = x[..., ::2]$$

    $$x_2 = x[..., 1::2]$$

    $$x_{rotate} = \text{cat}(-x_2, x_1)$$

    $$x_{out} = x \cdot \cos + x_{rotate} \cdot \sin$$

    其中，$x$表示参数`x`，$\cos$表示参数`r1`（cos分量），$\sin$表示参数`r2`（sin分量），$x_{out}$为Inplace写入`x`的结果。

-   `partial_slice`参数说明：通过`partial_slice`指定$[start, end)$范围，仅对该范围内的数据进行旋转位置编码。`partial_slice`作用于输入张量的最后一维（D维）。

## 函数原型

```
inplace_partial_rotary_mul(x, r1, r2, *, rotary_mode, partial_slice) -> None
```

## 参数说明

<table style="undefined;table-layout: fixed; width:1625px"><colgroup>
<col style="width: 147px">
<col style="width: 132px">
<col style="width: 232px">
<col style="width: 293px">
<col style="width: 185px">
<col style="width: 200px">
<col style="width: 140px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
    <th>维度(shape)</th>
    <th>非连续Tensor</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>x</td>
        <td>必选输入（Inplace输出）</td>
        <td>待执行旋转位置编码的张量，公式中的x。Inplace模式，计算结果直接写回x。</td>
        <td>BFLOAT16、FLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>(B, S, N, D)，即BSND布局</td>
        <td>×</td>
    </tr>
    <tr>
        <td>r1</td>
        <td>必选输入</td>
        <td>位置编码张量，公式中的cos。</td>
        <td>BFLOAT16、FLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>4维</td>
        <td>×</td>
    </tr>
    <tr>
        <td>r2</td>
        <td>必选输入</td>
        <td>位置编码张量，公式中的sin。</td>
        <td>BFLOAT16、FLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>4维</td>
        <td>×</td>
    </tr>
    <tr>
        <td>rotary_mode</td>
        <td>必选属性</td>
        <td>旋转模式，当前仅支持"interleave"。</td>
        <td>str</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>partial_slice</td>
        <td>必选属性</td>
        <td>部分旋转的切片范围[start, end)，作用于最后一维（D维）。</td>
        <td>List[int]</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
</tbody>
</table>

**说明**

- `r1`的数据类型需与`x`一致，或者为FLOAT32。
- `r2`的数据类型需与`x`一致，或者为FLOAT32。
- `r1`和`r2`的数据类型必须一致。
- `rotary_mode`当前仅支持"interleave"（对应值1），暂不支持"half"（值0）、"quarter"（值2）、"interleave-half"（值3）。
- `partial_slice`为两个整数组成的列表，指定$[start, end)$范围。不指定时默认整D轴做旋转编码；当$start$和$end$相等时不做旋转编码。

## 输出说明

本算子为Inplace操作，计算结果直接写回输入张量`x`，不产生新的输出张量。`x`在计算后shape和数据类型保持不变，其最后一维中`partial_slice`指定的$[start, end)$范围内的数据被旋转位置编码结果替换，其余位置的数据保持原值。

## 约束说明

- 该接口支持训练、推理场景下使用。
- 该接口支持单算子模式调用。
- 不支持非连续Tensor。
- 输入张量`x`最后一维（D）大小不超过1024。
- interleave模式（`rotary_mode`为"interleave"）下，`x`最后一维（D）大小必须为2的倍数。
- interleave模式下，`partial_slice`切片长度（即$end - start$）必须为2的倍数。
- 输入张量`r1`、`r2`最后一维大小必须相同，且必须等于`partial_slice`的切片长度（即$end - start$）。
- `partial_slice`约束：$start \geq 0$，$end \geq 0$，$end \leq D$（`x`最后一维大小），$end - start \geq 0$。当$end$和$start$相同时，不做旋转位置编码，直接返回。
- 仅支持interleave模式（`rotary_mode`为"interleave"）。
- `x`的shape为BSND。
- `r1`/`r2`的shape必须与`x`满足广播关系，且存在如下约束：
    - <term>Ascend 950PR/Ascend 950DT</term>：`r1`/`r2`的shape当前只支持BSND、B1ND、B11D、111D排布。
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：`r1`/`r2`的shape当前只支持BS1D、B11D排布。

## 确定性计算

默认支持确定性计算。

## 调用示例

-   单算子模式调用

    ```python
    import torch
    import torch_npu
    from cann_ops_transformer.ops import inplace_partial_rotary_mul

    torch_npu.npu.set_device(0)

    # x采用BSND布局，D维为128。
    B, S, N, D = 2, 32, 8, 128

    # 仅对x最后一维的[0, 64)范围执行旋转位置编码。
    slice_start, slice_end = 0, 64

    x = torch.randn(B, S, N, D, device="npu", dtype=torch.float16)

    # r1和r2分别表示cos和sin分量，最后一维需等于partial_slice的长度。
    r1 = torch.randn(B, S, 1, slice_end - slice_start, device="npu", dtype=torch.float16)
    r2 = torch.randn(B, S, 1, slice_end - slice_start, device="npu", dtype=torch.float16)

    # Inplace接口无返回值，计算结果直接写回x。
    inplace_partial_rotary_mul(x, r1, r2, rotary_mode="interleave", partial_slice=[slice_start, slice_end])
    ```

-   图模式调用

    ```python
    import torch
    import torch_npu
    import torchair
    from cann_ops_transformer.ops import inplace_partial_rotary_mul

    torch_npu.npu.set_device(0)

    # x采用BSND布局，D维为128。
    B, S, N, D = 2, 32, 8, 128

    # 仅对x最后一维的[0, 64)范围执行旋转位置编码。
    slice_start, slice_end = 0, 64

    class InplacePartialRotaryMulModel(torch.nn.Module):
        def __init__(self, rotary_mode, partial_slice):
            super().__init__()
            self.rotary_mode = rotary_mode
            self.partial_slice = partial_slice

        def forward(self, x, r1, r2):
            inplace_partial_rotary_mul(x, r1, r2, rotary_mode=self.rotary_mode, partial_slice=self.partial_slice)
            return x

    model = InplacePartialRotaryMulModel("interleave", [slice_start, slice_end]).npu()
    npu_backend = torchair.get_npu_backend()
    model = torch.compile(model, backend=npu_backend, dynamic=False)

    x = torch.randn(B, S, N, D, device="npu", dtype=torch.float16)

    # r1和r2需与x满足广播关系，最后一维需等于partial_slice的长度。
    r1 = torch.randn(B, S, 1, slice_end - slice_start, device="npu", dtype=torch.float16)
    r2 = torch.randn(B, S, 1, slice_end - slice_start, device="npu", dtype=torch.float16)

    # 图模式下forward返回写回后的x。
    output = model(x, r1, r2)
    ```
