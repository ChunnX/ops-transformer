# chunk_kda_fwd

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持。
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持。
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持。

## 功能说明

`chunk_kda_fwd` 完成不涉及 CP 切分的 KDA 分块正向计算。在一次 `ChunkKdaFwd` device kernel
launch 内执行 chunk-local gate 累计、块内矩阵构造与求逆、W/U 更新、跨 chunk FP32 状态传播和
输出计算。矩阵乘输入为 BF16/FP16，使用 FP32 累加；三角矩阵求逆和跨 chunk 状态均保持 FP32。

当 `use_gate_in_kernel=True` 时，gate 计算为：

$$
\mathrm{gate}=
\begin{cases}
-\exp(A_{log})\operatorname{softplus}(g+dt\_bias), & safe\_gate=false,\\
lower\_bound\cdot\operatorname{sigmoid}(\exp(A_{log})(g+dt\_bias)), & safe\_gate=true.
\end{cases}
$$

随后在每个 chunk 内计算 $gk=\operatorname{cumsum}(\mathrm{gate})/\ln 2$。

## 函数原型

```python
from cann_ops_transformer.ops import chunk_kda_fwd

outputs = chunk_kda_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    chunk_indices=None,
    safe_gate=False,
    lower_bound=None,
    use_gate_in_kernel=False,
    A_log=None,
    dt_bias=None,
    chunk_size=64,
    disable_recompute=False,
    return_intermediate_states=False,
    state_v_first=False,
    layout="BSND",
)
```

## 参数说明

| 参数名 | 类型 | 必选性 | 数据类型 | 说明 |
| --- | --- | --- | --- | --- |
| `q/k` | Tensor | 必选 | BF16/FP16 | Query/Key；shape 由 `layout` 解释。 |
| `v` | Tensor | 必选 | 与 `q` 相同 | Value，head 数为 `H_v`。 |
| `g` | Tensor | 必选 | BF16/FP32 | raw gate 或已激活的自然对数 gate。 |
| `beta` | Tensor | 必选 | BF16/FP32 | Delta 系数。 |
| `scale` | float | 必选 | float | 通常取 `K**-0.5`。 |
| `initial_state` | Tensor | 可选 | FP32 | `[N,H_v,K,V]`；`state_v_first=True` 时为 `[N,H_v,V,K]`。 |
| `output_final_state` | bool | 可选 | - | 是否返回 `final_state`，默认 `False`。 |
| `cu_seqlens` | Sequence/Tensor | 可选 | 整数 | 变长序列累计长度。 |
| `cu_seqlens_cpu` | Sequence/Tensor | 可选 | 整数 | 已有 CPU 长度副本；非空时优先使用，避免额外 D2H。 |
| `chunk_indices` | Sequence/Tensor | 可选 | 整数 | canonical `(seq_id, chunk_id)` 序列。 |
| `safe_gate` | bool | 可选 | - | 是否使用有界 gate，默认 `False`。 |
| `lower_bound` | float | 可选 | float | safe gate 下取值范围 `[-5,0)`，默认 `-5.0`。 |
| `use_gate_in_kernel` | bool | 可选 | - | 是否从 raw gate 计算激活，默认 `False`。 |
| `A_log` | Tensor | 条件必选 | BF16/FP32 | `use_gate_in_kernel=True` 时必选，shape 为 `[H_v]`。BF16 在包装层提升为 FP32。 |
| `dt_bias` | Tensor | 可选 | BF16/FP32 | shape 为 `[H_v*K]`。BF16 在包装层提升为 FP32。 |
| `chunk_size` | int | 可选 | - | 支持 64/128，默认 64。 |
| `disable_recompute` | bool | 可选 | - | 是否保留反向中间量，默认 `False`。 |
| `return_intermediate_states` | bool | 可选 | - | 是否额外返回 `h`，默认 `False`。 |
| `state_v_first` | bool | 可选 | - | 是否交换状态末两维，默认 `False`。 |
| `layout` | str | 可选 | - | 输入布局，支持 `BSND/BNSD/TND/NTD`，默认 `BSND`。 |

## 返回值说明

返回 12 项：

```text
(attn_out, final_state, gk, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state)
```

- `attn_out` 固定为 BSND/TND；`Aqk/Akk` 始终返回并保持 head-major。
- `final_state` 仅在 `output_final_state=True` 时返回。
- `gk` 在 `use_gate_in_kernel=False` 或 `disable_recompute=True` 时返回。
- `w/u/qg/kg/v_new` 仅在 `disable_recompute=True` 时返回。
- `h` 在 `disable_recompute=True` 或 `return_intermediate_states=True` 时返回。
- 第 12 项原样返回 `initial_state`，不是 aclnn 输出。

## 约束说明

- `H_v >= H` 且 `H_v % H == 0`，head 数不超过 128。
- `K/V` 为 `[16,256]` 内 16 的倍数；重点场景为 `K=V=128, chunk_size=64`。
- rank-4 变长输入要求 `B=1`，逻辑序列数不超过 1024。
- `layout` 只描述输入；输出布局按返回值说明固定。
- 默认支持确定性计算。

## 调用示例

```python
import torch
import torch_npu
from cann_ops_transformer.ops import chunk_kda_fwd

B, T, H, D = 1, 128, 4, 128
q = torch.randn(B, T, H, D, device="npu", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
g = torch.randn(B, T, H, D, device="npu", dtype=torch.float32)
beta = torch.randn(B, T, H, device="npu", dtype=torch.bfloat16)
a_log = torch.randn(H, device="npu", dtype=torch.bfloat16)
dt_bias = torch.randn(H * D, device="npu", dtype=torch.bfloat16)

attn_out, final_state, *_ = chunk_kda_fwd(
    q=q,
    k=k,
    v=v,
    g=g,
    beta=beta,
    scale=D ** -0.5,
    output_final_state=True,
    safe_gate=True,
    lower_bound=-5.0,
    use_gate_in_kernel=True,
    A_log=a_log,
    dt_bias=dt_bias,
    chunk_size=64,
)
```
