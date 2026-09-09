# Quant Sparse Lightning Indexer pytest

该目录按主线 ASC pytest 的流程组织，但所有数据生成、golden、算子调用和比较均为 QSLI 独立实现，不引用 ASC 测试代码。

## 文件结构

- `test_quant_sparse_lightning_indexer_paramset.py`：single 参数集，含 decode、prefill 和 8 条交叉用例。
- `test_quant_sparse_lightning_indexer_single.py`：paramset 展开、按名称筛选、可选 PT 保存和 NPU 精度比较。
- `test_quant_sparse_lightning_indexer_batch.py`：读取 PT、批量执行并调用 `result_compare_method`。
- `qsli_test_utils.py`：参数生成、连续/axis-0 非连续 PA、NPU 调用和比较公共流程。
- `quant_sparse_lightning_indexer_golden.py`：原有独立 CPU golden，内容未修改。
- `batch/quant_sparse_lightning_indexer_pt_save.py`：从 Excel 生成 PT。
- `batch/list_pt_from_excel.py`：按 Excel `Testcase_Name` 筛选已有 PT。
- `test_run.sh`：single、batch、batch_exec 统一入口。
- `batch_isolated_run.sh`：每条 PT 独立 pytest 进程。

## 运行

```bash
bash test_run.sh single
bash test_run.sh single -C qsli_decode,qsli_prefill
bash test_run.sh batch -E ./excel/qsli_transfer_cases.xlsx -P ./pt_path
bash test_run.sh batch_exec -E ./excel/qsli_transfer_cases.xlsx -P ./pt_path
bash batch_isolated_run.sh ./pt_path 0
```

Excel 默认读取 `TestCases` sheet。输出索引和 value 均使用本目录的 `result_compare_method.py` 判定。
