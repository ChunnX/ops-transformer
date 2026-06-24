# InplacePartialRotaryMulGrad

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    ×    |
| <term>Atlas 训练系列产品</term>                              |    ×    |

## 功能说明

- **算子功能**：执行局部旋转位置编码InplacePartialRotaryMul的反向计算。该算子对输入dy的D维度上切片[start, end)区域执行旋转位置编码梯度计算，dx'的结果inplace写回dy的[start, end)区间。dy的[start, end)之外的数据保持不变。

- **计算公式**：

    取旋转位置编码的正向计算中，broadcast的轴列表为`dims`，在D维度上的切片范围为`[start, end)`，令参与计算的切片数据为：
    $$
    dy' = dy[..., start:end]
    $$
    $$
    cos' = cos[..., start:end]
    $$
    $$
    sin' = sin[..., start:end]
    $$
    $$
    x' = x[..., start:end]
    $$

    则梯度计算公式可表达如下：

    （1）half模式（rotary_mode等于0）：
    $$
    dy1', dy2' = chunk(dy', chunks=2, dim=-1)
    $$

    $$
    cos1', cos2' = chunk(cos', chunks=2, dim=-1)
    $$

    $$
    sin1', sin2' = chunk(sin', chunks=2, dim=-1)
    $$

    $$
    x1', x2' = chunk(x', chunks=2, dim=-1)
    $$

    $$
    dx' = cat((cos1' * dy1' + sin2' * dy2', cos2' * dy2' - sin1' * dy1'), dim=-1)
    $$

    $$
    dcos = sum(dy' * x', dims)
    $$

    $$
    dsin = sum(dy' * cat((-x2', x1'), dim=-1), dims)
    $$

    （2）interleave模式（rotary_mode等于1）：
    $$
    dy1', dy2' = dy'[..., :: 2], dy'[..., 1 :: 2]
    $$

    $$
    cos1', cos2' = cos'[..., :: 2], cos'[..., 1 :: 2]
    $$

    $$
    sin1', sin2' = sin'[..., :: 2], sin'[..., 1 :: 2]
    $$

    $$
    x1', x2' = x'[..., :: 2], x'[..., 1 :: 2]
    $$

    $$
    dx' = stack((cos1' * dy1' + sin2' * dy2', cos2' * dy2' - sin1' * dy1'), dim=-1).reshape(dy'.shape)
    $$

    $$
    dcos = sum(dy' * x', dims)
    $$

    $$
    dsin = sum(dy' * stack((-x2', x1'), dim=-1).reshape(dy'.shape), dims)
    $$

    （3）quarter模式（rotary_mode等于2）：
    $$
    dy1', dy2', dy3', dy4' = chunk(dy', chunks=4, dim=-1)
    $$

    $$
    cos1', cos2', cos3', cos4' = chunk(cos', chunks=4, dim=-1)
    $$

    $$
    sin1', sin2', sin3', sin4' = chunk(sin', chunks=4, dim=-1)
    $$

    $$
    x1', x2', x3', x4' = chunk(x', chunks=4, dim=-1)
    $$

    $$
    dx' = cat((cos1' * dy1' + sin2' * dy2', cos2' * dy2' - sin1' * dy1', cos3' * dy3' + sin4' * dy4', cos4' * dy4' - sin3' * dy3'), dim=-1)
    $$

    $$
    dcos = sum(dy' * x', dims)
    $$

    $$
    dsin = sum(dy' * cat((-x2', x1', -x4', x3'), dim=-1), dims)
    $$

    （4）interleave-half模式（rotary_mode等于3）：
    $$
    dy1', dy2' = chunk(dy', chunks=2, dim=-1)
    $$

    $$
    cos1', cos2' = chunk(cos', chunks=2, dim=-1)
    $$

    $$
    sin1', sin2' = chunk(sin', chunks=2, dim=-1)
    $$

    $$
    x1', x2' = x'[..., :: 2], x'[..., 1 :: 2]
    $$

    $$
    dx' = stack((cos1' * dy1' + sin2' * dy2', cos2' * dy2' - sin1' * dy1'), dim=-1).reshape(dy'.shape)
    $$

    $$
    dcos = sum(dy' * cat((x1', x2'), dim=-1), dims)
    $$

    $$
    dsin = sum(dy' * cat((-x2', x1'), dim=-1), dims)
    $$

## 参数说明

<table style="table-layout: auto; width: 100%">
<colgroup>
  <col style="width: 10%">
  <col style="width: 10%">
  <col style="width: 40%">
  <col style="width: 20%">
  <col style="width: 10%">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出/属性</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>dyRef</td>
    <td>输入</td>
    <td>公式中的dy，表示正向计算输出y的导数，inplace更新为正向输入x的导数。Inplace模式，dyRef同时作为输出写入结果。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>cos</td>
    <td>输入</td>
    <td>公式中的cos，正向计算输入，需与sin数据类型一致。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>sin</td>
    <td>输入</td>
    <td>公式中的sin，正向计算输入，需与cos数据类型一致。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>xOptional</td>
    <td>可选输入</td>
    <td>公式中的x，正向计算输入，需与dyRef数据类型一致。**当前版本暂不支持，请传入空指针。** 如果为空指针，则不计算dcosOut和dsinOut。</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>rotaryMode</td>
    <td>属性</td>
    <td>旋转模式，0=half，1=interleave，2=quarter，3=interleave-half。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>partialSlice</td>
    <td>属性</td>
    <td>D维度上的切片范围[start, end)，默认{0, 0}表示不做有效计算。start须在[0, D]内，end须在[start, D]内。</td>
    <td>IntArray</td>
    <td>-</td>
  </tr>
  <tr>
    <td>dcosOut</td>
    <td>输出</td>
    <td>公式中的dcos，输入cos的导数，需与cos数据类型一致。**当前版本暂不支持，输出无意义。**</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>dsinOut</td>
    <td>输出</td>
    <td>公式中的dsin，输入sin的导数，需与sin数据类型一致。**当前版本暂不支持，输出无意义。**</td>
    <td>BFLOAT16、FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明

- 该算子仅支持Ascend 950 AI Processor。
- 该算子仅支持连续Tensor，不支持非连续Tensor。
- **该算子当前版本仅支持 interleave 模式（`rotary_mode=1`）**。half（0）、quarter（2）、DeepSeek interleave（3）模式暂未实现。

- 输入张量dyRef支持BSND排布以及其B/S/N维度的广播变体（如111D、1SND、B1ND、BS1D、11ND、B11D、1S1D等）。各参数的shape约束可以描述如下：
  - 输入张量dyRef和xOptional的最后一维大小D必须相同，且D ≤ 1024。
  - 输入张量cos、sin及输出张量dcosOut、dsinOut的最后一维大小必须等于切片长度(end - start)。
  - 输入张量cos和sin的shape必须完全相同，cos和sin的B、S、N维度需要与dyRef满足[broadcast关系](../../docs/zh/context/broadcast关系.md)，且广播后的B、S、N必须等于dyRef的B、S、N。
  - half、interleave和interleave-half模式下，切片长度(end - start)必须能被2整除。
  - quarter模式下，切片长度(end - start)必须能被4整除。
  - 当start等于end时，算子不执行有效计算，直接返回。
  - 输入张量cos和sin的数据类型必须相同。
  - 输入张量xOptional与dyRef的数据类型必须相同。
  - 输出张量dcosOut与cos的数据类型必须相同。
  - 输出张量dsinOut与sin的数据类型必须相同。
  - 当xOptional为空指针时，不计算dcosOut和dsinOut，此时dcosOut和dsinOut无意义。
  - **注意**：当前版本暂不支持xOptional（即dcos/dsin）的梯度计算。xOptional输入及dcosOut/dsinOut输出参数保留用于后续版本兼容，现阶段请传入空指针。


## 调用说明

| 调用方式           | 调用样例                                                                                    | 说明                                                                                                  |
|----------------|-----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_inplace_partial_rotary_mul_grad](./examples/arch35/test_aclnn_inplace_partial_rotary_mul_grad.cpp) | 通过[aclnnInplacePartialRotaryMulGrad](./docs/aclnnInplacePartialRotaryMulGrad.md)接口方式调用InplacePartialRotaryMulGrad算子。             |
