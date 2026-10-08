# QuantSparseLightningIndexer 测试（两级TopK 第二级）

本目录是第二级 consumer 算子（`quant_sparse_lightning_indexer`）的 pytest 用例。它必须与
第一级算子 [QuantLightningIndexerV2](../../quant_lightning_indexer_v2/README.md) 配合使用：
第一级按块粒度挑出候选块并输出块级索引，本算子在候选块集合内做 token 级 TopK。

## 文件

| 文件 | 说明 |
|---|---|
| `qli_consumer_two_level.py` | 第二级精度测试：两条路径分别与 torch 参考实现比对 —— ① 吃 golden 候选（隔离 consumer 自身精度）；② 吃第一级 NPU 候选（生产链路端到端）。用例表为 shape 矩阵：cmp_ratio 1/2、mask 0/3、g 32/64、BSND/TND、B=4 变长、候选容量边界（pin_overflow）、单行长上下文（decode_long_m1）等 |
| `qli_cand_orig_consumer.py` | 由主算子测试 `qli_cand_orig.py`（其上溯 /opt/tjj 的原始 harness）按 `cand_mode` 切片迁移而来：原 harness 中全部 33 个 `cand_mode=2` 用例（大 shape/1M、TND、pa_gap 0 轴非连续、offset 等形态），候选输入由 golden 候选准备；第一级/现网路径的用例仍留在主算子目录 |

第一级与现网（候选关闭）路径的用例见
[../quant_lightning_indexer_v2/tests/pytest/qli_cand_orig.py](../../quant_lightning_indexer_v2/tests/pytest/qli_cand_orig.py)；其中
`cand_mode=2`（第二级）的 33 个用例已按本目录迁移，主算子文件只保留 `cand_mode` 1（第一级）/ 3（现网）。

## 运行

```bash
source <CANN>/set_env.sh          # 需要能 import 到含本算子的 cann_ops_transformer 包
export ASCEND_CUSTOM_OPP_PATH=<算子包 vendors 目录>   # 如用的是共享 CANN，可省略
ASCEND_RT_VISIBLE_DEVICES=<空闲卡> pytest qli_consumer_two_level.py [-k <用例 id>]
ASCEND_RT_VISIBLE_DEVICES=<空闲卡> pytest qli_cand_orig_consumer.py [-k <用例 id>]   # 文件名非 test_*.py，需显式指定
```

比对规则取自第一级算子的测试目录（`result_compare_method.py`，与本算子共用一套边界容忍判据）。
