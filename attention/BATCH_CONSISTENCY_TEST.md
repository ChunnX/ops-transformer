# MLA Batch 一致性测试说明

## 测试命令

在仓库根目录分别执行：

```bash
cd attention/sparse_flash_mla/tests/pytest && bash test_run.sh single --batch-consistency on
```

```bash
cd attention/quant_sparse_flash_mla/tests/pytest && bash test_run.sh single --batch-consistency on
```

```bash
cd attention/mixed_quant_sparse_flash_mla/tests/pytest && bash test_run.sh single --batch-consistency on
```

开启 Batch 一致性后，测试使用 `npu_deterministic_level=3`，同时进行 Batch 一致性比较和精度比较。

## 测试用例

| 用例 | 真实计算负载 | Template | Q Layout | KV Layout | ori/cmp mask mode |
|---|---:|---|---|---|---|
| `bc_load_4096_swa_tnd_pa` | 4096 | SWA | TND | PA_BBND | 4 / 0 |
| `bc_load_8192_hca_tnd_tnd` | 8192 | HCA | TND | TND | 3 / 3 |
| `bc_load_2048_csa_bsnd_bsnd` | 2048 | CSA | BSND | BSND | 4 / 3 |
| `bc_load_5120_ori_sparse_bsnd_pa` | 5120 | ORI_SPARSE | BSND | PA_BBND | 0 / 0 |
| `bc_load_4096_ori_cmp_sparse_tnd_pa` | 4096 | ORI_CMP_SPARSE | TND | PA_BBND | 0 / 0 |

所有用例均使用 `B=2`、`S1=2`，并执行 batch reorder、batch split、token split 和 Q shape change 四种一致性模式。

## 测试结果

测试日期：2026-08-31。

| 用例 | SMLA | QSMLA | MQSMLA |
|---|---|---|---|
| `bc_load_4096_swa_tnd_pa` | 失败 | 失败 | 失败 |
| `bc_load_8192_hca_tnd_tnd` | 通过 | 失败 | 失败 |
| `bc_load_2048_csa_bsnd_bsnd` | 失败 | 失败 | 失败 |
| `bc_load_5120_ori_sparse_bsnd_pa` | 通过 | 失败 | 失败 |
| `bc_load_4096_ori_cmp_sparse_tnd_pa` | 通过 | 失败 | 失败 |
| **汇总** | **3 通过，2 失败** | **0 通过，5 失败** | **0 通过，5 失败** |

SMLA 的两个失败用例数值精度比较均为 100%，失败原因为部分变换未达到位级一致。QSMLA 和 MQSMLA 的失败包含精度比较失败；MQSMLA 还存在 Batch 一致性比较失败。
