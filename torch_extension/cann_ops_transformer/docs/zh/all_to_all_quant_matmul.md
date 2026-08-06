# all_to_all_quant_matmul

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

  - AlltoAll + Matmul融合算子，将AlltoAll通信、Permute数据重排和Matmul矩阵乘融合为单个算子，实现通信与计算的掩盖。算子语义为"先通信后计算"：各rank先通过AlltoAll交换x1的数据，再对交换后的数据做permute重排，最后与权重矩阵x2做matmul。
  - 支持MX量化场景（x1_quant_mode=6, x2_quant_mode=6），输入为float8_e4m3fn/float8_e5m2数据配合float8_e8m0缩放因子，输出为bfloat16/float16/float32。
  - 接口内部自动创建通信context，用户只需传入torch.distributed的ProcessGroup即可，无需额外管理通信资源。

- **计算公式**：

  - 输入：

    - $\mathbf{X1} \in \mathbb{R}^{\text{BS} \times \text{H}}$：激活矩阵，对应入参 `x1`。$\text{BS}$ 为batch size（总token数），$\text{H}$ 为每rank的hidden size。
    - $\mathbf{X2} \in \mathbb{R}^{\text{N} \times (\text{H} \times \text{world\_size})}$：权重矩阵，对应入参 `x2`。$\text{N}$ 为输出维度，$\text{H} \times \text{world\_size}$ 即总K维度。传入前需调用 `.t()` 转置为 $(\text{H} \times \text{world\_size}, \text{N})$ 的非连续视图，算子内部自动检测并处理。
    - $\mathbf{S1} \in \mathbb{R}^{\text{BS} \times \lceil\text{H}/64\rceil \times 2}$：x1的MX量化缩放因子，对应入参 `x1_scale`，数据类型为float8_e8m0。
    - $\mathbf{S2} \in \mathbb{R}^{\text{N} \times \lceil\text{H} \times \text{world\_size}/64\rceil \times 2}$：x2的MX量化缩放因子，对应入参 `x2_scale`，数据类型为float8_e8m0。

  - 输出：

    - $\mathbf{Y} \in \mathbb{R}^{(\text{BS}/\text{world\_size}) \times \text{N}}$：matmul输出矩阵，对应出参 `y`。
    - $\mathbf{A} \in \mathbb{R}^{(\text{BS}/\text{world\_size}) \times (\text{H} \times \text{world\_size})}$：AlltoAll通信后的中间结果，对应出参 `all2all_out`。`alltoall_out_flag=False`时不输出。

  - 约定：

    - $\text{world\_size}$：参与AlltoAll通信的rank数量。
    - $\text{local\_bs} = \text{BS} / \text{world\_size}$：每个rank经AlltoAll后接收的token数。
    - $\text{K} = \text{H} \times \text{world\_size}$：matmul的总K维度。

  - 计算说明（MX量化场景）：

    **1. AlltoAll通信**

    每个rank将其 $\mathbf{X1}$ 按 $\text{local\_bs}$ 行为一块，共 $\text{world\_size}$ 块，通过AlltoAll发送给所有rank。每个rank接收来自所有rank的第 $\text{dst\_rank}$ 块数据，按src rank顺序拼接为 $\mathbf{A} \in \mathbb{R}^{\text{BS} \times \text{H}}$。

    **2. Permute重排**

    将AlltoAll结果 $\mathbf{A}$ 从 $(\text{BS}, \text{H})$ 重排为 $(\text{local\_bs}, \text{K})$，即将各rank的数据块沿K轴拼接：

    $$\mathbf{A}_{\text{perm}} = \text{Permute}(\mathbf{A}) \in \mathbb{R}^{\text{local\_bs} \times \text{K}}$$

    **3. MX反量化 + Matmul**

    对 $\mathbf{A}_{\text{perm}}$ 和 $\mathbf{X2}^T$ 分别用各自的MX缩放因子反量化为float32，再做matmul：

    $$\mathbf{Y} = \text{DQ}_{\text{MX}}(\mathbf{A}_{\text{perm}}, \mathbf{S1}) \cdot \text{DQ}_{\text{MX}}(\mathbf{X2}^T, \mathbf{S2}) \in \mathbb{R}^{\text{local\_bs} \times \text{N}}$$

    其中 $\text{DQ}_{\text{MX}}$ 表示MX逐组反量化（group size = 32），在matmul内部隐式执行。最终输出按 `y_dtype` 转换为目标数据类型。

## 函数原型

```python
all_to_all_quant_matmul(x1, x2, group, *, world_size=0, alltoall_out_flag=True, bias=None, x1_scale=None, x2_scale=None, comm_scale=None, x1_offset=None, x2_offset=None, x1_quant_mode=0, x2_quant_mode=0, comm_quant_mode=0, group_sizes=None, all2all_axes=None, comm_quant_dtype=-1, x1_quant_dtype=-1, x1_dtype=-1, x2_dtype=-1, x1_scale_dtype=-1, x2_scale_dtype=-1, output_scale_dtype=-1, comm_scale_dtype=-1, y_dtype=-1, comm_mode=None, precision_mode=0) -> (Tensor, Tensor)
```

## 参数说明

<table style="undefined;table-layout: fixed; width:1400px"><colgroup>
<col style="width: 120px">
<col style="width: 120px">
<col style="width: 90px">
<col style="width: 320px">
<col style="width: 160px">
<col style="width: 120px">
<col style="width: 260px">
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
        <td>激活矩阵，AlltoAll通信的输入数据。各rank的x1数据通过AlltoAll交换后与x2做matmul。</td>
        <td>float8_e4m3fn、float8_e5m2</td>
        <td>ND</td>
        <td>(BS, H)</td>
    </tr>
    <tr>
        <td>x2</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>权重矩阵。传入前需调用 <code>.t()</code> 转置，使viewShape为 (H*world_size, N) 但内存布局按 (N, H*world_size) 存储。算子内部自动检测非连续并处理。</td>
        <td>float8_e4m3fn、float8_e5m2</td>
        <td>ND</td>
        <td>(H*world_size, N)（转置后）</td>
    </tr>
    <tr>
        <td>group</td>
        <td>ProcessGroup</td>
        <td>必选</td>
        <td>torch.distributed进程组，用于AlltoAll通信。接口内部通过该group获取HCCL通信句柄并创建通信context。</td>
        <td>ProcessGroup</td>
        <td>不涉及</td>
        <td>不涉及</td>
    </tr>
    <tr>
        <td>world_size</td>
        <td>int</td>
        <td>必选</td>
        <td>参与AlltoAll通信的rank数量。取值为2、4、8、16。</td>
        <td>int</td>
        <td>不涉及</td>
        <td>不涉及</td>
    </tr>
    <tr>
        <td>alltoall_out_flag</td>
        <td>bool</td>
        <td>可选</td>
        <td>是否输出AlltoAll通信后的中间结果。True时输出all2all_out，False时不输出。默认值为True。</td>
        <td>bool</td>
        <td>不涉及</td>
        <td>不涉及</td>
    </tr>
    <tr>
        <td>bias</td>
        <td>Tensor</td>
        <td>可选</td>
        <td>matmul后加的偏置向量。</td>
        <td>float32</td>
        <td>ND</td>
        <td>(N,)</td>
    </tr>
    <tr>
        <td>x1_scale</td>
        <td>Tensor</td>
        <td>可选</td>
        <td>x1的MX量化缩放因子。MX量化场景下为必选。</td>
        <td>float8_e8m0</td>
        <td>ND</td>
        <td>(BS, CeilDiv(H, 64), 2)</td>
    </tr>
    <tr>
        <td>x2_scale</td>
        <td>Tensor</td>
        <td>可选</td>
        <td>x2的MX量化缩放因子。MX量化场景下为必选。</td>
        <td>float8_e8m0</td>
        <td>ND</td>
        <td>(N, CeilDiv(H*world_size, 64), 2)</td>
    </tr>
    <tr>
        <td>comm_scale</td>
        <td>预留参数</td>
        <td>可选</td>
        <td>预留参数，通信量化缩放因子，当前版本不支持，使用默认值即可。</td>
        <td>预留参数</td>
        <td>不涉及</td>
        <td>不涉及</td>
    </tr>
    <tr>
        <td>x1_offset</td>
        <td>预留参数</td>
        <td>可选</td>
        <td>预留参数，x1量化偏移，当前版本不支持，使用默认值即可。</td>
        <td>预留参数</td>
        <td>不涉及</td>
        <td>不涉及</td>
    </tr>
    <tr>
        <td>x2_offset</td>
        <td>预留参数</td>
        <td>可选</td>
        <td>预留参数，x2量化偏移，当前版本不支持，使用默认值即可。</td>
        <td>预留参数</td>
        <td>不涉及</td>
        <td>不涉及</td>
    </tr>
    <tr>
        <td>x1_quant_mode</td>
        <td>int</td>
        <td>可选</td>
        <td>x1量化模式。0表示非量化，6表示MX量化，7表示pertoken动态量化。默认值为0。</td>
        <td>int</td>
        <td>不涉及</td>
        <td>不涉及</td>
    </tr>
    <tr>
        <td>x2_quant_mode</td>
        <td>int</td>
        <td>可选</td>
        <td>x2量化模式。0表示非量化，2表示perchannel量化，6表示MX量化。默认值为0。</td>
        <td>int</td>
        <td>不涉及</td>
        <td>不涉及</td>
    </tr>
    <tr>
        <td>comm_quant_mode</td>
        <td>int</td>
        <td>可选</td>
        <td>预留参数，通信量化模式，当前版本不支持。默认值为0。</td>
        <td>int</td>
        <td>不涉及</td>
        <td>不涉及</td>
    </tr>
    <tr>
        <td>group_sizes</td>
        <td>List[int]</td>
        <td>可选</td>
        <td>量化分组大小，格式为 <code>[group_m, group_n, group_k]</code>。MX量化场景下默认值为<code>[1, 1, 32]</code>。</td>
        <td>List[int]</td>
        <td>不涉及</td>
        <td>不涉及</td>
    </tr>
    <tr>
        <td>all2all_axes</td>
        <td>List[int]</td>
        <td>可选</td>
        <td>AlltoAll数据交换方向，支持空或<code>[-2, -1]</code>。为空时默认按<code>[-2, -1]</code>处理。默认值为None。</td>
        <td>List[int]</td>
        <td>不涉及</td>
        <td>不涉及</td>
    </tr>
    <tr>
        <td>comm_quant_dtype</td>
        <td>int</td>
        <td>可选</td>
        <td>预留参数，通信量化数据类型枚举，当前版本不支持。默认值为-1。</td>
        <td>int</td>
        <td>不涉及</td>
        <td>不涉及</td>
    </tr>
    <tr>
        <td>x1_quant_dtype</td>
        <td>int</td>
        <td>可选</td>
        <td>x1量化数据类型枚举。MX量化场景下该参数不生效，使用默认值即可。默认值为-1。</td>
        <td>int</td>
        <td>不涉及</td>
        <td>不涉及</td>
    </tr>
    <tr>
        <td>x1_dtype</td>
        <td>int</td>
        <td>可选</td>
        <td>x1的数据类型枚举，用于bitcast重解释。-1表示使用tensor实际数据类型。默认值为-1。</td>
        <td>int</td>
        <td>不涉及</td>
        <td>不涉及</td>
    </tr>
    <tr>
        <td>x2_dtype</td>
        <td>int</td>
        <td>可选</td>
        <td>x2的数据类型枚举，用于bitcast重解释。-1表示使用tensor实际数据类型。默认值为-1。</td>
        <td>int</td>
        <td>不涉及</td>
        <td>不涉及</td>
    </tr>
    <tr>
        <td>x1_scale_dtype</td>
        <td>int</td>
        <td>可选</td>
        <td>x1_scale的数据类型枚举。float8_e8m0时设为293。-1表示使用tensor实际数据类型。默认值为-1。</td>
        <td>int</td>
        <td>不涉及</td>
        <td>不涉及</td>
    </tr>
    <tr>
        <td>x2_scale_dtype</td>
        <td>int</td>
        <td>可选</td>
        <td>x2_scale的数据类型枚举。float8_e8m0时设为293。-1表示使用tensor实际数据类型。默认值为-1。</td>
        <td>int</td>
        <td>不涉及</td>
        <td>不涉及</td>
    </tr>
    <tr>
        <td>output_scale_dtype</td>
        <td>预留参数</td>
        <td>可选</td>
        <td>预留参数，输出scale数据类型枚举，当前版本不支持。默认值为-1。</td>
        <td>预留参数</td>
        <td>不涉及</td>
        <td>不涉及</td>
    </tr>
    <tr>
        <td>comm_scale_dtype</td>
        <td>预留参数</td>
        <td>可选</td>
        <td>预留参数，通信scale数据类型枚举，当前版本不支持。默认值为-1。</td>
        <td>预留参数</td>
        <td>不涉及</td>
        <td>不涉及</td>
    </tr>
    <tr>
        <td>y_dtype</td>
        <td>int</td>
        <td>可选</td>
        <td>输出y的数据类型枚举。5表示float16，6表示float32，15表示bfloat16。-1表示默认float32。默认值为-1。</td>
        <td>int</td>
        <td>不涉及</td>
        <td>不涉及</td>
    </tr>
    <tr>
        <td>comm_mode</td>
        <td>str</td>
        <td>可选</td>
        <td>通信引擎选择。<code>"ai_cpu"</code>表示使用AICPU通信引擎。None时默认使用<code>"ai_cpu"</code>。默认值为None。</td>
        <td>str</td>
        <td>不涉及</td>
        <td>不涉及</td>
    </tr>
    <tr>
        <td>precision_mode</td>
        <td>int</td>
        <td>可选</td>
        <td>精度模式。0表示默认精度。默认值为0。</td>
        <td>int</td>
        <td>不涉及</td>
        <td>不涉及</td>
    </tr>
</tbody>
<tfoot>
<tr>
    <td colspan="7">表格中的<code>CeilDiv(<var>x</var>, <var>y</var>) = ⌈<var>x</var> / <var>y</var>⌉ = ⌊(<var>x</var> + <var>y</var> - 1) / <var>y</var>⌋</code></td>
</tr>
</tfoot>
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
        <td>y</td>
        <td>Tensor</td>
        <td>必选</td>
        <td>matmul输出矩阵，数据类型由y_dtype决定。</td>
        <td>bfloat16、float16、float32</td>
        <td>(BS/world_size, N)</td>
    </tr>
    <tr>
        <td>all2all_out</td>
        <td>Tensor</td>
        <td>可选</td>
        <td>AlltoAll通信后的中间结果。alltoall_out_flag=False时不输出。</td>
        <td>与x1一致</td>
        <td>(BS/world_size, H*world_size)</td>
    </tr>
</tbody>
</table>

## 约束说明

- **参数一致性约束**：
  - 所有rank的 `world_size`、`x1` 的shape、`x2` 的shape、量化参数（`x1_quant_mode`、`x2_quant_mode`、`group_sizes` 等）必须保持一致。
  - `BS` 必须能被 `world_size` 整除。

- **Shape约束**：
  - `x1`：2维张量，shape为 `(BS, H)`，`H` 必须能被64整除（MX量化场景）。
  - `x2`：2维张量，传入前需调用 `.t()` 转置。原始存储shape为 `(N, H*world_size)`，转置后viewShape为 `(H*world_size, N)`。
  - `x1_scale`：3维张量，shape为 `(BS, CeilDiv(H, 64), 2)`，第3维必须为2。
  - `x2_scale`：3维张量，shape为 `(N, CeilDiv(H*world_size, 64), 2)`，第3维必须为2。

- **world_size约束**：
  - 取值为 `2`、`4`、`8`、`16`。

- **数据类型约束**：

  当前版本仅支持MX量化场景（`x1_quant_mode=6, x2_quant_mode=6`），各参数的数据类型配套关系如下：

  <table>
  <thead>
      <tr>
      <th>场景</th>
      <th>x1</th>
      <th>x2</th>
      <th>x1_scale</th>
      <th>x2_scale</th>
      <th>y</th>
      <th>x1_quant_mode</th>
      <th>x2_quant_mode</th>
      <th>group_sizes</th>
      <th>x1_scale_dtype</th>
      <th>x2_scale_dtype</th>
      </tr>
  </thead>
  <tbody>
      <tr>
      <td><strong>MX FP8 E4M3FN</strong></td>
      <td>float8_e4m3fn</td>
      <td>float8_e4m3fn</td>
      <td>float8_e8m0</td>
      <td>float8_e8m0</td>
      <td>bfloat16</td>
      <td>6</td>
      <td>6</td>
      <td>[1, 1, 32]</td>
      <td>293</td>
      <td>293</td>
      </tr>
      <tr>
      <td><strong>MX FP8 E5M2</strong></td>
      <td>float8_e5m2</td>
      <td>float8_e5m2</td>
      <td>float8_e8m0</td>
      <td>float8_e8m0</td>
      <td>bfloat16</td>
      <td>6</td>
      <td>6</td>
      <td>[1, 1, 32]</td>
      <td>293</td>
      <td>293</td>
      </tr>
  </tbody>
  </table>

- **x2转置约束**：
  - x2在传入前必须调用 `.t()` 进行转置，使viewShape为 `(H*world_size, N)`。算子内部通过检测stride自动识别转置状态，无需用户手动传入 `transpose_x2` 参数。
  - 转置后x2为非连续tensor，算子内部会自动处理。

- **通信域约束**：
  - 调用前需初始化HCCL通信域（`dist.init_process_group(backend="hccl")`）。
  - 所有rank必须使用相同的通信域。
  - HCCL通信缓存区大小需通过环境变量 `HCCL_BUFFSIZE` 配置，单位为MB。

- **预留参数约束**：
  - `comm_scale`、`x1_offset`、`x2_offset`、`comm_quant_mode`、`comm_quant_dtype`、`output_scale_dtype`、`comm_scale_dtype` 为预留参数，当前版本不支持，使用默认值即可。

## 确定性计算

默认支持确定性计算。

## 调用示例

- 单算子模式调用（MX FP8 E4M3FN量化场景，多卡分布式）：

```python
import os
import argparse
import math
import multiprocessing as mp
from torch.multiprocessing import Process, Manager

import numpy as np
import torch
import torch_npu
import torch.distributed as dist
from ml_dtypes import float8_e4m3fn as np_fp8_e4m3fn
from ml_dtypes import float8_e8m0fnu as np_fp8_e8m0fnu

from cann_ops_transformer import all_to_all_quant_matmul

MASTER_IP = "127.0.0.1"
MASTER_PORT = 50010

os.environ['HCCL_BUFFSIZE'] = '6000'

# at::ScalarType dtype enums
Y_DTYPE_BF16 = 15
SCALE_DTYPE_FP8_E8M0 = 293

# MX quant constants
MX_QUANT_MODE = 6
MX_GROUP_SIZES = [1, 1, 32]

torch_npu.npu.config.allow_internal_format = True


def set_device(rank, rank_per_dev):
    real_device = rank % rank_per_dev
    torch_npu.npu.set_device(real_device)


def init_hccl_comm(rank, world_size):
    dist.init_process_group(
        backend='hccl',
        rank=rank,
        world_size=world_size,
        init_method=f'tcp://{MASTER_IP}:{MASTER_PORT}'
    )
    global_group = dist.new_group(ranks=list(range(world_size)))
    return global_group


def alltoallmm_worker(queue, rank, world_size, bs, h, n):
    try:
        set_device(rank, world_size)
        group = init_hccl_comm(rank, world_size)

        k = h * world_size
        local_bs = bs // world_size

        # 构造输入数据
        np.random.seed(42)

        # x2: 原始存储shape为 (N, K)，传入前需 .t() 转置
        x2_np = np.random.uniform(1, 8, (n, k)).astype(np_fp8_e4m3fn)
        x2_scale_np = np.random.uniform(1, 8, (n, math.ceil(k / 64), 2)).astype(np_fp8_e8m0fnu)

        # x1: shape (BS, H)
        x1_np = np.random.uniform(1, 8, (bs, h)).astype(np_fp8_e4m3fn)
        x1_scale_np = np.random.uniform(1, 8, (bs, math.ceil(h / 64), 2)).astype(np_fp8_e8m0fnu)

        # numpy fp8 -> torch fp8
        def np_fp8_to_torch(np_arr, torch_dtype):
            return torch.from_numpy(np_arr.view(np.uint8)).view(torch_dtype)

        x1 = np_fp8_to_torch(x1_np, torch.float8_e4m3fn).npu()
        x2 = np_fp8_to_torch(x2_np, torch.float8_e4m3fn).npu()
        x2 = x2.t()  # 转置为 (K, N) 的非连续视图
        x1_scale = np_fp8_to_torch(x1_scale_np, torch.float8_e8m0fnu).npu()
        x2_scale = np_fp8_to_torch(x2_scale_np, torch.float8_e8m0fnu).npu()

        # 调用算子
        y, all2all_out = all_to_all_quant_matmul(
            x1,
            x2,
            group,
            world_size=world_size,
            x1_scale=x1_scale,
            x2_scale=x2_scale,
            y_dtype=Y_DTYPE_BF16,
            x1_quant_mode=MX_QUANT_MODE,
            x2_quant_mode=MX_QUANT_MODE,
            group_sizes=MX_GROUP_SIZES,
            x1_scale_dtype=SCALE_DTYPE_FP8_E8M0,
            x2_scale_dtype=SCALE_DTYPE_FP8_E8M0,
            comm_mode="ai_cpu",
            alltoall_out_flag=True,
        )
        torch.npu.synchronize()

        print(f"[rank {rank}] y.shape={y.shape}, dtype={y.dtype}, "
              f"all2all_out.shape={all2all_out.shape}")

        dist.barrier()
        torch.npu.synchronize()

        queue.put([rank, "success", None])

    except Exception as e:
        print(f"[ERROR] rank {rank} failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        queue.put([rank, f"error: {e}", None])

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        torch.npu.empty_cache()


if __name__ == "__main__":
    world_size = 2
    bs = 1024
    h = 4096
    n = 8192

    proc_list = []
    manager = Manager()
    result_queue = manager.Queue()
    mp.set_start_method('spawn', force=True)

    for rank in range(world_size):
        proc = Process(
            target=alltoallmm_worker,
            args=(result_queue, rank, world_size, bs, h, n)
        )
        proc.start()
        proc_list.append(proc)

    for proc in proc_list:
        proc.join()

    for _ in range(world_size):
        rank_id, result, _ = result_queue.get()
        print(f"[INFO] rank {rank_id} result: {result}")
```
