# Quant Sparse Lightning Indexer pytest

## 分离部署与 Torch 注册

测试导入 `cann_ops_transformer.ops.attention.quant_sparse_lightning_indexer_dsl`，
调用 `torch.ops.cann_ops_transformer.ds41.quant_sparse_lightning_indexer`。
扩展从独立 DSL 包的 `ops.quant_sparse_lightning_indexer_dsl` 导入实现，
不依赖本项目 src 或相邻目录。部署时安装匹配的 DSL/extension 包；源码验收
通过 PYTHONPATH 指定两个独立包目录。可设置 CANNBOTDSL_TEST_OPS_DIR 核对
DSL 来源并输出实际扩展路径，不强制加载相邻目录下的扩展。

`ds41.quant_sparse_lightning_indexer_metadata` 同样独立注册，返回 int32[1024]。
Torch 层使用 layout_kv，映射至 DSL 的 layout_k。主算子默认 PA_BBND，metadata
默认 TND。显式 metadata 原样传入，省略时通过注册入口生成。PA 使用 packed K，
TND 使用独立 K/scale；return_value=False 返回空 values。

激活 `cannbotdsl-0.5.0-20260910`。原 Python 参数集保持不变。
回迁仅执行代表性接口与精度检查，不要求全量 Excel 回归。

```bash
export QSLI_DEVICE_ID=1 OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
export TMPDIR=/tmp/xyh/LI_ds41/sync_back
export CANNBOTDSL_CACHE_DIR="$TMPDIR/cache"
python -B -m pytest -q -p no:cacheprovider experimental/attention/quant_sparse_lightning_indexer_dsl/pytest/test_quant_sparse_lightning_indexer_single.py -k torch_extension
```

已有 single 文件包含 PA/TND、values 开关、显式/自动 metadata 与 DSL 直调对照。
test_run.sh 可从任意工作目录启动；PT、日志及缓存统一放临时目录。

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
