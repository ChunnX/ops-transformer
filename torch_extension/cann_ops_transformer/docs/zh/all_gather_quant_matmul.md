# all_gather_quant_matmul

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

- **接口功能**：

  AllGatherMatmulV3 算子将 AllGather 通信与 MX 量化 Matmul 计算融合为单个算子，实现通信和计算的掩盖。每个 Rank 先通过 AllGather 收集所有 Rank 的左矩阵 $x_1$ 及其 MX 缩放因子，再与本地的右矩阵 $x_2$ 完成 MX 量化矩阵乘。

- **计算公式**：

  - 输入：
    - $\mathbf{X}_1 \in \mathbb{F}^{M \times K}$：本 Rank 的左矩阵（激活），对应入参 `x1`。$M$ 为单 Rank 的 M 轴长度，$K$ 为隐藏维度。$\mathbb{F}$ 为 FLOAT8_E4M3FN、FLOAT8_E5M2 或 FLOAT4_E2M1。
    - $\mathbf{X}_2 \in \mathbb{F}^{N \times K}$：右矩阵（权重），对应入参 `x2`，需以 `[N, K]` 存储的 `.t()` 转置视图传入。
    - $\mathbf{S}_1 \in \mathbb{E}^{M \times \lceil K/64 \rceil \times 2}$：`x1` 的 MX 缩放因子，对应入参 `x1_scale`，$\mathbb{E}$ 为 FLOAT8_E8M0。
    - $\mathbf{S}_2 \in \mathbb{E}^{N \times \lceil K/64 \rceil \times 2}$：`x2` 的 MX 缩放因子，对应入参 `x2_scale`。
  - 输出：
    - $\mathbf{Y} \in \mathbb{R}^{(M \times \text{rank\_size}) \times N}$：输出矩阵，对应出参 `y`。
  - 计算过程：
    $$
    \mathbf{Y} = \mathrm{AllGather}(\mathbf{X}_1) \cdot \mathbf{X}_2^{\top} + \text{bias}
    $$
    其中 AllGather 沿 M 轴收集全部 rank_size 个 Rank 的 $\mathbf{X}_1$（及对应缩放因子），Matmul 在 kernel 内部对 MX 量化数据隐式执行逐组反量化（group size = 32）。bias 为可选叠加项。

## 函数原型

```python
all_gather_quant_matmul(x1, x2, group, *, bias=None, x1_scale=None, x2_scale=None, group_sizes=None, x1_dtype=None, x2_dtype=None, x1_scale_dtype=None, x2_scale_dtype=None, y_dtype=None, comm_mode="urma") -> (Tensor, Tensor, Tensor)
```

## 参数说明

<table style="undefined;table-layout: fixed; width:1400px"><colgroup>
<col style="width: 150px">
<col style="width: 120px">
<col style="width: 90px">
<col style="width: 420px">
<col style="width: 220px">
<col style="width: 100px">
<col style="width: 300px">
</colgroup>
<thead>
<tr>
    <th>参数名</th>
    <th>参数类型</th>
    <th>可选/必选</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
    <th>维度(shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>x1</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>本 Rank 的左矩阵（激活）。</td>
        <td>float8_e4m3fn、float8_e5m2、uint8(fp4 压缩存储，需搭配 x1_dtype=296)</td>
        <td>ND</td>
        <td>(M, K)，K 取值范围 [256, 65535)，fp4 场景 K 须为偶数</td>
    </tr>
    <tr>
        <td>x2</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>右矩阵（权重），须传入 [N, K] 存储的 .t() 转置视图。fp4 场景 x1/x2 必须同为 fp4。</td>
        <td>float8_e4m3fn、float8_e5m2、uint8(fp4 压缩存储，需搭配 x2_dtype=296)</td>
        <td>ND</td>
        <td>存储 (N, K)，视图 (K, N)</td>
    </tr>
    <tr>
        <td>group</td>
        <td>ProcessGroup</td>
        <td>必选</td>
        <td>AllGather 通信的 torch.distributed 通信域。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>bias</td>
        <td>Tensor</td>
        <td>可选</td>
        <td>Matmul 后叠加的偏置。默认值为 None。</td>
        <td>float32</td>
        <td>ND</td>
        <td>(N,)</td>
    </tr>
    <tr>
        <td>x1_scale</td>
        <td>Tensor</td>
        <td>可选</td>
        <td>x1 的 MX 缩放因子，MX 场景必传。默认值为 None。</td>
        <td>float8_e8m0fnu（或 uint8 存储搭配 x1_scale_dtype=293）</td>
        <td>ND</td>
        <td>(M, ⌈K/64⌉, 2)</td>
    </tr>
    <tr>
        <td>x2_scale</td>
        <td>Tensor</td>
        <td>可选</td>
        <td>x2 的 MX 缩放因子，MX 场景必传。默认值为 None。</td>
        <td>float8_e8m0fnu（或 uint8 存储搭配 x2_scale_dtype=293）</td>
        <td>ND</td>
        <td>(N, ⌈K/64⌉, 2)</td>
    </tr>
    <tr>
        <td>group_sizes</td>
        <td>list[int]</td>
        <td>可选</td>
        <td>量化分组大小，MX 场景仅支持 [1, 1, 32]。默认值为 None，表示从 scale shape 自动推导。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>x1_dtype</td>
        <td>int</td>
        <td>可选</td>
        <td>x1 真实数据类型枚举（用于 uint8 压缩存储的 bitcast）。fp4 场景必须传 296（fp4_e2m1）。默认值为 None，表示使用 tensor 自身 dtype。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>x2_dtype</td>
        <td>int</td>
        <td>可选</td>
        <td>x2 真实数据类型枚举，同 x1_dtype。默认值为 None。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>x1_scale_dtype</td>
        <td>int</td>
        <td>可选</td>
        <td>x1_scale 真实数据类型枚举。uint8 存储时必须传 293（fp8_e8m0）。默认值为 None。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>x2_scale_dtype</td>
        <td>int</td>
        <td>可选</td>
        <td>x2_scale 真实数据类型枚举，同 x1_scale_dtype。默认值为 None。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>y_dtype</td>
        <td>int</td>
        <td>可选</td>
        <td>输出数据类型枚举，仅支持 5（float16）/ 15（bfloat16）。默认值为 None，表示 bfloat16。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>comm_mode</td>
        <td>str</td>
        <td>可选</td>
        <td>通信引擎，仅支持 "urma"。默认值为 "urma"。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
</tbody>
</table>

## 返回值说明

返回三元组 `(y, gather_out, amax_out)`：

| 返回值 | 数据类型 | 维度(shape) | 描述 |
| --- | --- | --- | --- |
| y | bfloat16、float16 | (M × rank_size, N) | AllGather + Matmul 融合计算结果。 |
| gather_out | 同 x1 | (0,) | 当前版本未实现，返回空占位 tensor。 |
| amax_out | float32 | (0,) | 当前版本未实现，返回空占位 tensor。 |

## 约束说明

- 仅支持 <term>Ascend 950PR/Ascend 950DT</term>，rank_size 仅支持 2/4/8/16。
- 通信数据总量（x1 数据 + x1_scale）不能超过 HCCL 通信 buffer 大小，大 shape 场景需提前设置环境变量 `HCCL_BUFFSIZE`（单位 MB）。
- x1/x2 dtype 支持 float8_e4m3fn、float8_e5m2 及 fp4_e2m1（uint8 压缩存储 + dtype 枚举 296）；fp4 场景 x1/x2 必须同为 fp4，且 K 为偶数。
- y 仅支持 bfloat16/float16 输出。

## 调用示例

```python
import torch
import torch_npu
import torch.distributed as dist
from cann_ops_transformer import all_gather_quant_matmul

dist.init_process_group(backend="hccl")
group = dist.group.WORLD

m_per_rank, k, n = 2048, 4096, 2560
x1 = torch.randn(m_per_rank, k).to(torch.float8_e4m3fn).npu()
x2 = torch.randn(n, k).to(torch.float8_e4m3fn).npu()
x1_scale = torch.ones(m_per_rank, (k + 63) // 64, 2, dtype=torch.float8_e8m0fnu).npu()
x2_scale = torch.ones(n, (k + 63) // 64, 2, dtype=torch.float8_e8m0fnu).npu()

y, _, _ = all_gather_quant_matmul(x1, x2.t(), group, x1_scale=x1_scale, x2_scale=x2_scale)
```
