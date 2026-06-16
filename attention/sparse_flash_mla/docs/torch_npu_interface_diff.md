# SMLA torch_npu接口原型差异说明

本文对比“已经两段式、但算子原型尚未对齐”的旧版 `sparse_flash_mla_metadata` / `sparse_flash_mla` 适配，与当前原型对齐后的 torch_npu 调用差异。旧版已经是两段式调用，差异不在调用流程，而在 metadata 字段、属性命名、默认值和长度语义。

## 1. 调用形态不变

旧版和当前版本都保持两段式调用：

```python
from cann_ops_transformer.ops import sparse_flash_mla, sparse_flash_mla_metadata

metadata = sparse_flash_mla_metadata(
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_ori_kv=cu_seqlens_ori_kv,
    cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
    num_heads_q=num_heads_q,
    num_heads_kv=num_heads_kv,
    head_dim=head_dim,
    layout_q="TND",
    layout_kv="TND",
    # ...
)

attn_out, softmax_lse = sparse_flash_mla(
    q,
    ori_kv=ori_kv,
    cmp_kv=cmp_kv,
    metadata=metadata,
    layout_q="TND",
    layout_kv="TND",
    # ...
)
```

因此迁移时不需要把单段调用改成两段调用；需要检查的是参数名、默认值和长度相关输入是否按当前原型传递。

## 2. metadata接口差异

| 旧版两段式原型 | 当前原型 | torch_npu调用影响 |
| --- | --- | --- |
| `max_seqlen_kv` | `max_seqlen_ori_kv` + `max_seqlen_cmp_kv` | 需要分别传入 `ori_kv` 和 `cmp_kv` 的最大序列长度；旧版单一 `kv` 最大长度不能表达两段长度不同的场景。 |
| 无 `seqused_cmp_kv` | 新增 `seqused_cmp_kv` | 可显式描述每个 batch 中 `cmp_kv` 参与计算的有效长度。 |
| 无 `cmp_residual_kv` | 新增 `cmp_residual_kv` | 用于按 `cmp_len * cmp_ratio + residual` 恢复压缩前长度，进而恢复 cmp 侧 causal mask 语义。 |
| 无 `ori_topk_length` / `cmp_topk_length` | 新增 `ori_topk_length` / `cmp_topk_length` | 仅作接口预留，当前版本不支持传入非空 Tensor；调用侧应保持 `None`。 |
| `cmp_ratio` 默认 `-1` | `cmp_ratio` 默认 `1` | 不传或传 `1` 都表示非压缩/SWA 默认语义；`0` 在主算子侧也按 SWA 语义处理。 |
| 需要 `soc_version` / `aic_core_num` / `aiv_core_num` 属性 | torch_npu调用侧不再传这些属性 | 平台和核数信息由算子内部获取或由底层编译环境处理，调用方不用再显式传入。 |

## 3. 主算子接口差异

主算子仍是 `sparse_flash_mla(...)`，且仍接收 `metadata` 作为可选输入。当前原型对齐后，调用侧需要重点关注：

- `layout_q` 默认值为 `BSND`，可取值为 `BSND` / `TND`。
- `layout_kv` 默认值为 `BSND`，公开可取值仅为 `PA_BNBD` / `BSND` / `TND`。
- `cmp_residual_kv`、`seqused_cmp_kv` 与 metadata 侧保持同一语义；需要端到端生效时，metadata 和主算子侧应传入一致的数据。
- `ori_topk_length`、`cmp_topk_length` 为预留输入，当前版本不支持传入非空 Tensor；metadata 和主算子侧都应保持 `None`。
- `ori_kv_stride` / `cmp_kv_stride` 不作为 torch_npu Python 接口参数传递；当前由 tiling 从输入 Tensor 的 stride 中获取，再通过 tilingData 传递给 kernel。

## 4. 长度语义差异

### `seqused_ori_kv`

- `TND` 场景：优先由 `cu_seqlens_ori_kv` 表达每个 batch 的 `ori_kv` 有效长度。
- `BSND` 场景：可用于覆盖 `ori_kv` 的逻辑有效长度；未传时使用输入 shape 推导。
- `PA_BNBD` 场景：用于表达 paged KV 的逻辑有效长度。

### `seqused_cmp_kv`

`seqused_cmp_kv` 当前在 `BSND` / `TND` / `PA_BNBD` 场景均可用于表达 `cmp_kv` 的逻辑有效长度。未传时：

- `TND` 使用 `cu_seqlens_cmp_kv` 推导。
- `BSND` / `PA_BNBD` 使用输入 shape 或对应 batch 的逻辑长度推导。

### `cmp_residual_kv`

`cmp_residual_kv` 当前是主算子和 metadata 的可选输入。它不表示 `cmp_kv` 起点，而是表示压缩长度无法被 `cmp_ratio` 整除时的余数：

```text
压缩前长度 = cmp_len * cmp_ratio + cmp_residual_kv
```

该长度用于恢复 cmp 侧 mask 的压缩前坐标，典型用于 CP 切分、长短序列混合、以及 `ori_kv` 有效长度不等于 `cmp_len * cmp_ratio` 的场景。

## 5. 迁移示例

旧版两段式调用中，如果 metadata 使用单一 KV 长度：

```python
metadata = sparse_flash_mla_metadata(
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_ori_kv=cu_seqlens_ori_kv,
    cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
    max_seqlen_kv=max_seqlen_kv,
    cmp_ratio=cmp_ratio,
    layout_q="TND",
    layout_kv="TND",
    # ...
)
```

当前应改为分别传递 `ori` / `cmp` 最大长度，并在需要时补充 cmp 侧有效长度和 residual：

```python
metadata = sparse_flash_mla_metadata(
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_ori_kv=cu_seqlens_ori_kv,
    cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
    seqused_cmp_kv=seqused_cmp_kv,
    cmp_residual_kv=cmp_residual_kv,
    max_seqlen_ori_kv=max_seqlen_ori_kv,
    max_seqlen_cmp_kv=max_seqlen_cmp_kv,
    cmp_ratio=cmp_ratio,
    layout_q="TND",
    layout_kv="TND",
    # ...
)

attn_out, softmax_lse = sparse_flash_mla(
    q,
    ori_kv=ori_kv,
    cmp_kv=cmp_kv,
    cmp_sparse_indices=cmp_sparse_indices,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_ori_kv=cu_seqlens_ori_kv,
    cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
    seqused_cmp_kv=seqused_cmp_kv,
    cmp_residual_kv=cmp_residual_kv,
    metadata=metadata,
    cmp_ratio=cmp_ratio,
    layout_q="TND",
    layout_kv="TND",
    return_softmax_lse=True,
)
```

## 6. 迁移检查项

- 将 `max_seqlen_kv` 拆成 `max_seqlen_ori_kv` 和 `max_seqlen_cmp_kv`。
- 需要 cmp 侧有效长度覆盖时，同时向 metadata 和主算子传入 `seqused_cmp_kv`。
- 需要恢复压缩前 mask 长度时，同时向 metadata 和主算子传入 `cmp_residual_kv`。
- `ori_topk_length` / `cmp_topk_length` 保持 `None`，当前版本传入非空 Tensor 会触发参数拦截。
- `layout_kv` 只使用 `PA_BNBD` / `BSND` / `TND`，默认按 `BSND` 处理。
- torch_npu 调用侧不传 `soc_version`、`aic_core_num`、`aiv_core_num`。
