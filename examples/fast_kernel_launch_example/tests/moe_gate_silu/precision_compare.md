# moe_gate_silu 精度对比

## 算子接口

```python
torch.ops.ascend_ops.moe_gate_silu(x, total_rows) -> Tensor
```

- `x`: NPU contiguous tensor，shape 为 `[rows, hidden]`，dtype 为 `torch.bfloat16`，`hidden` 为偶数且不大于8192。
- `total_rows`: NPU contiguous `torch.int64` 一维 tensor，最后一个元素表示实际参与计算的行数；常规全量测试可传 `torch.tensor([rows], dtype=torch.int64).npu()`。
- 输出: shape 为 `[rows, hidden / 2]`，dtype 与 `x` 一致。

## PyTorch 参考实现

```python
hidden = x.shape[-1] // 2
expected = torch.nn.functional.silu(x.float()[:, :hidden]) * x.float()[:, hidden:]
expected = expected.to(x.dtype)
```

即输入最后一维前半段作为 gate，后半段作为 up，计算 `silu(gate) * up`。

## 测试与阈值

测试脚本: `tests/moe_gate_silu/test_moe_gate_silu.py`

默认覆盖 case:

| rows | hidden | dtype |
|---:|---:|---|
| 1 | 32 | bfloat16 |
| 7 | 64 | bfloat16 |
| 40 | 128 | bfloat16 |
| 128 | 256 | bfloat16 |
| 2 | 8192 | bfloat16 |

判定阈值:

| dtype | rtol | atol |
|---|---:|---:|
| bfloat16 | 1e-2 | 1e-2 |

## 运行方式

```bash
pytest tests/moe_gate_silu -v -s
```

也可以直接运行脚本输出 CSV 格式精度指标:

```bash
python tests/moe_gate_silu/test_moe_gate_silu.py --dtype bfloat16
python tests/moe_gate_silu/test_moe_gate_silu.py --rows 128 --hidden 256 --dtype bfloat16
```

输出字段:

```text
rows,hidden,dtype,max_abs,mean_abs,max_rel,allclose,rtol,atol
```

其中 `max_abs` / `mean_abs` / `max_rel` 分别为 NPU 结果与 PyTorch CPU 参考结果转换到 `float32` 后的最大绝对误差、平均绝对误差和最大相对误差。

## 本机验证结果

验证环境中执行:

```bash
python3 tests/moe_gate_silu/test_moe_gate_silu.py --dtype bfloat16
```

得到以下精度对比结果:

| rows | hidden | dtype | max_abs | mean_abs | max_rel | allclose | rtol | atol |
|---:|---:|---|---:|---:|---:|---|---:|---:|
| 1 | 32 | bfloat16 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | True | 1e-2 | 1e-2 |
| 7 | 64 | bfloat16 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | True | 1e-2 | 1e-2 |
| 40 | 128 | bfloat16 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | True | 1e-2 | 1e-2 |
| 128 | 256 | bfloat16 | 6.103516e-05 | 3.725290e-09 | 4.629630e-03 | True | 1e-2 | 1e-2 |
| 2 | 8192 | bfloat16 | 0.000000e+00 | 0.000000e+00 | 0.000000e+00 | True | 1e-2 | 1e-2 |
