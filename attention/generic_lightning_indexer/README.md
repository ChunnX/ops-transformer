# GenericLightningIndexer

通用 Indexer 融合算子：对**已经完成前处理**的 Index Query / Index Key 做

`Matmul → 激活 → 沿 head 规约 →（可选 Key 轴规约）→ TopK →（可选 Expand / 追加尾 token）`。

不包含 RoPE、RMSNorm、AvgPool、KeyPool 等 Q/K 前处理；通过 `cmp_ratio` 消费压缩后的 Key。

完整需求与收编映射见 [docs/generic_lightning_indexer_design.md](docs/generic_lightning_indexer_design.md)。

## 产品支持

| 产品 | Step 1 |
|---|---|
| Ascend 950PR / 950DT | 支持（主路径） |
| Atlas A2 / A3 | 支持（非量化 FP16/BF16） |
| 其他 | 不支持 |

## 计算公式

$$
s = \mathrm{Reduce}_h\bigl(\phi(\alpha\, Q K^\top)\bigr),\quad
idx = \mathrm{Post}\bigl(\mathrm{TopK}(s \odot M)\bigr)
$$

- $\phi$：`act_type` = ReLU 或 Identity
- $\mathrm{Reduce}_h$：加权和或累加和（`has_weighted_sum`）
- $M$：由 `sparse_mode` 与 `cmp_ratio` 决定的可见性
- $\mathrm{Post}$：identity / Expand / Expand+AppendTail

## 收编剖面（属性速查）

| 剖面 | cmp_ratio | act | weighted | sparse_mode | expand | append |
|---|---|---|---|---|---|---|
| LI / QLI | 1 | relu | yes | 0/3 | no | no |
| LIv2 / QLIv2 | r | relu | yes | 0/3 | no | no |
| PoolKeyIndexer | pool_size | relu | yes | 0/3 | yes | yes |
| Qwen CLI | 4 | relu | **no** | **4** | yes | yes |
| MSA（Step 2） | 1 | identity | no | 0/3 | — | — |

## 调用

| 方式 | 说明 |
|---|---|
| aclnn | `examples/test_aclnn_generic_lightning_indexer.cpp` |
| 设计文档 | `docs/generic_lightning_indexer_design.md` |
