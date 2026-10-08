# causal_conv1d_grad

`causal_conv1d_fn`（因果一维深度卷积 prefill 模式）的反向算子，用于训练反向传播。给定上游梯度 `grad_y` 及前向输入 `x`、`weight`、`bias`，计算三路全量梯度 `dx` / `dweight` / `dbias`。

## 数学定义

前向未保存 pre-activation，反向在算子内部重算：

```
z_t       = Σ_{j=0..3} w_j · x_{t-j} + b        （冷启动，左侧 pad 3 零）
silu'(z)  = σ(z) · (1 + z · (1 - σ(z)))
grad_z    = grad_y · silu'(z)
dbias     = Σ_{B,T} grad_z
dweight_j = Σ_{B,T} grad_z_t · x_{t-j}
dx_t      = Σ_{j=0..3} grad_z_{t+j} · w_j        （右边界截断）
```

中间计算与累加全程 fp32，输出 cast 回输入 dtype。

## 支持范围

| 项 | 值 |
|----|----|
| dtype | bfloat16（已验证，matched_ratio=1.0000）；float16 已实现待验证 |
| shape | x/grad_y `[B, T, 12288]`，weight `[4, 12288]`，bias `[12288]` |
| 约束 | kW=4 / activation=silu / 冷启动 / 固定 batch 3D prefill |
| 芯片 | Ascend950PR（arch35 / DAV_3510，已验证 Ascend950PR_9579） |

## PyTorch 绑定

通过 `mamba/causal_conv1d/torch_extension/causal_conv1d_grad_ext.py` 提供 autograd 绑定（前向 `causal_conv1d_fn` ↔ 本算子反向）：

```python
from mamba.causal_conv1d.torch_extension import causal_conv1d_grad_ext
y = causal_conv1d_grad_ext.run(x, weight, bias, conv_states, query_start_loc)  # requires_grad=True 时可 backward()
```

## 性能（Ascend950PR 实测，bf16，T=8192）

| 指标 | 数值 |
|------|------|
| 每 token | 0.280 us（优化全程 0.932 → 0.708 → 0.280，3.33x） |
| B4_T8192 kernel | 9.1 ms |
| vs PyTorch 小算子拼接 | 15 shape geomean 3.364（15/15 反超） |
| 精度 | 三路 matched_ratio=1.0000；拼接方案 bf16 长归约 T=8192 时降至 0.735 |

## 测试

```bash
# CPU golden 自测
python3 tests/golden.py
# NPU ST（bf16 4 用例，需先构建安装算子包）
cd tests/st_cpp && bash run.sh --real
```

详细设计文档与开发记录见开发仓 `cannbot-skills/plugins-official/ops-registry-invoke/operators/causal_conv1d_grad/docs/`。
