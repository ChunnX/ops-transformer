# SageAttention2 与 CUDA 原生实现接口对齐与契约规范

本文档详述华为昇腾 Atlas 950PR 平台上的 `SageAttention2` 实现与上游官方开源实现（`thu-ml/SageAttention`）在 API 契约、参数定义、数值逻辑与调用行为方面的逐项对齐规范。

## 一、API 签名与参数定义对齐

本项目在 Python 层提供与上游 CUDA 扩展完全对齐的函数入口：

### 1. 核心算子入口 `sageattn_qk_int8_pv_fp8_asc`

```python
def sageattn_qk_int8_pv_fp8_asc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: float | None = None,
    pv_accum_dtype: str = "fp32+fp16",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
```

### 2. 逐项参数契约与对齐规范

- `q, k, v`:
  - 支持数据类型：`torch.float16` 与 `torch.bfloat16`（三者数据类型必须严格一致，且同设备）。
  - 张量布局：支持 `"HND"`（`[B, H, S, D]`）与 `"NHD"`（`[B, S, H, D]`）。末维连续性要求 `stride(-1) == 1`。
- `tensor_layout`:
  - 默认为 `"HND"`。对于 `"NHD"` 布局，内部自动通过内存连续视图转换转置为 `"HND"` 格式参与昇腾 Cube/Vector 处理，并在输出时还原为 `"NHD"` 视图。
- `is_causal`:
  - 默认为 `False`。当为 `True` 时要求输入序列满足 `Sq == Sk`，采用因果三角遮蔽，且硬件调度应用 FA4-LPT 逆序调度避免尾核空转。
- `qk_quant_gran`:
  - 默认为 `"per_thread"`（可选 `"per_warp"`）。
  - 契约保证：`per_thread` 与 `per_warp` 均有真实设备态与仿真态向量微架构支持，量化尺度与精度表现具有真实物理差异。
- `sm_scale`:
  - 默认值为 `1.0 / math.sqrt(head_dim_original)`。在输入经过 Head Dim 自动对齐填充（如 48->64，96->128）时，缩放系数严格以原始维度计算，确保数学等价性。
- `pv_accum_dtype`:
  - 默认为 `"fp32+fp16"`（可选 `"fp32+fp32"` 与 `"fp32"`）。
  - 契约保证：三种累加精度档位均为原生实现。
- `smooth_k`:
  - 默认为 `True`。开启时在 Attention 阶段前自动执行 Key 均值中心化平移操作，大幅压平外点，降低 INT8 量化截断误差。
- `smooth_v`:
  - 默认为 `False`。在 `pv_accum_dtype="fp32"` 档位下可选生效；在混合精度累加档下严格触发 `UserWarning` 并被安全旁路，防止多余数值噪声。
- `return_lse`:
  - 默认为 `False`。当为 `True` 时返回 `(output, lse)`，`lse` 为自然对数域形状 `[B, Hq, Sq]` 的单精度浮点张量。

## 二、自动填充与数学等价性保护

针对非 64/128 对齐的特征维度（如 $D=48$ 与 $D=96$）：

1. **自动前向对齐补零**：
   - $D \in (0, 64] \to$ 零填充至 64；
   - $D \in (64, 128] \to$ 零填充至 128；
   - $D > 128$ 或 $D \le 0 \to$ 严格抛出 `ValueError`。
2. **零贡献性数学保障**：
   - 填充维度赋值为 0，在 INT8 对称量化与 FP8 矩阵乘计算中，填充列内积贡献恒为 0，绝不干扰有效维度的乘加累加值。
3. **输出自动切片裁回**：
   - 核心输出由宽向量指令写回后，在统一出口严格执行 `out[..., :d_original]` 裁切，调用方感知零侵入。

## 三、LogSumExp (LSE) 自然域闭环

在开启 `smooth_k=True` 且 `return_lse=True` 时，LSE 的计算对齐开源官方数学定义：

$$\text{lse}_{\text{final}} = \ln(\text{denom}) + m' + s \cdot (q \cdot \mu_k)$$

其中：
- $\mu_k$ 为 Key 均值向量，在 GQA 场景下严格按照组比率 `repeat_interleave` 广播；
- $s$ 为 `sm_scale`；
- $m'$ 为全局最大值平移量；
- 自然域闭式等价计算完全由算子内部融合完成，消除了二次访存开销。

## 四、相关文件索引

- 核心实现：[python/sageattention2/core.py](file:///Users/z/git/hw/ops-transformer/experimental/attention/sageattention2/python/sageattention2/core.py)
- 接口校验：[python/sageattention2/_ops.py](file:///Users/z/git/hw/ops-transformer/experimental/attention/sageattention2/python/sageattention2/_ops.py)
- 验收矩阵：[tests/test_functional_matrix.py](file:///Users/z/git/hw/ops-transformer/experimental/attention/sageattention2/tests/test_functional_matrix.py)
