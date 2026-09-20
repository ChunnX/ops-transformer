# IndexerPrologueK

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练/推理系列产品 | × |
| Atlas A2 训练/推理系列产品 | × |

## 功能说明

`indexer_prologue_k` 融合完成 Indexer K 路的 BF16 投影、RMSNorm、尾部 RoPE、
MXFP4 量化和分页 cache 写入。算子原地更新 `k_cache`，mode 0 可同时原地更新
独立的 `k_scale_cache`，并返回传入的 `k_cache`。

Kernel 由 `cannbot-arena-ds41` 的 net-ops wheel 提供，本目录只负责 PyTorch
schema、Meta/FakeTensor 和 PrivateUse1 dispatcher 适配。调用前必须安装包含
`ops.indexer_prologue_k` 的 net-ops wheel。

## 接口

```python
indexer_prologue_k(
    latent,
    wk,
    norm_weight,
    rope_sin,
    rope_cos,
    k_cache,
    k_scale_cache=None,
    *,
    cache_index,
    storage_mode,
    norm_eps,
    combined_block_size=-1,
) -> Tensor
```

| 参数 | dtype / format | shape |
| :--- | :--- | :--- |
| `latent` | BF16 / ND | `[T, H]` |
| `wk` | BF16 / FRACTAL_NZ | `[D, H]` |
| `norm_weight` | FP32 / ND | `[D]` |
| `rope_sin`, `rope_cos` | FP32 / ND | `[T, Dr]` |
| `cache_index` | INT64 / ND | `[T]`，`-1` 表示跳过 |
| `k_cache` (mode 0) | UINT8 / ND | `[Bn, Bs, 1, D/2]` |
| `k_scale_cache` (mode 0) | UINT8 / ND，可选 | `[Bn, Bs, 1, D/32]` |
| `k_cache` (mode 1) | UINT8 / ND | `[Bn, Bs/G, 1, G*(D/2+D/32)]` |

其中 `G=combined_block_size`。约束为 `T>0`、`H%16==0`、`D%32==0`、
`0<Dr<=D` 且 `Dr` 为偶数。mode 1 要求 `G>0`。

## 调用示例

```python
import torch_npu
import cann_ops_transformer.ops
from cann_ops_transformer.ops.ds41 import indexer_prologue_k

wk_nz = torch_npu.npu_format_cast(wk.npu(), torch_npu.Format.FRACTAL_NZ)
result = indexer_prologue_k(
    latent.npu(),
    wk_nz,
    norm_weight.npu(),
    rope_sin.npu(),
    rope_cos.npu(),
    k_cache,
    k_scale_cache,
    cache_index=cache_index.npu(),
    storage_mode=0,
    norm_eps=1e-6,
)
assert result is k_cache
```

也可通过 `cann_ops_transformer.ops.indexer_prologue_k` 或
`torch.ops.cann_ops_transformer.ds41.indexer_prologue_k` 调用。

## 构建与测试

先安装提供 DSL kernel 的 net-ops wheel，再构建本仓整包：

```bash
bash build.sh --torch_extension
python3 -m pip install build_out/*.whl --force-reinstall --no-deps
```

执行真机精度用例：

```bash
bash experimental/attention/indexer_prologue_k_dsl/tests/test.sh
```
