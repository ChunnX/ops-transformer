# Quant Lightning Indexer V2-compatible pytest

## Checkout 验证与接口

主算子和 metadata 支持 key PA_BBND/TND。调用方必须先生成 metadata 并显式传入；省略时沿用底层错误，不自动生成。
公开 schema、dtype、返回顺序及空输出约定保持不变。

激活匹配的 Torch/NPU/DSL 环境并加载 CANN 后，设置：

```bash
export CANNBOTDSL_TEST_OPS_DIR=/path/to/cannbot-arena-ds41/net/ops
export OPS_TRANSFORMER_TORCH_EXTENSION_DIR=/path/to/ops-transformer_dsv41/torch_extension
export QLIV2_DEVICE_ID=1 QSLI_DEVICE_ID=1
export PYTHONDONTWRITEBYTECODE=1
export PYTEST_ADDOPTS="-p no:cacheprovider"
```

测试强制从指定 kernel 目录及本仓算子 torch_extension 加载，记录来源和 SHA256；不回退到集中开发仓或已安装的旧算子。
公共帮助函数位于 experimental/attention/indexer_test_runtime.py。ACL Graph 测试显式捕获 metadata→主算子链路。
产物默认写入 /tmp/ops_transformer_indexer/，也可通过测试脚本环境变量指定。

This directory follows the mainline QLI V2 pytest workflow while keeping the
DSL CPU Golden and fused NPU implementation independent from ASC source code.

## Single/paramset execution

```bash
./test_run.sh single
./test_run.sh single -C decode_b8_s2_8k
```

`test_quant_lightning_indexer_v2_paramset.py` retains the original PA cases and adds 16 adapted TND cases from the existing Excel files.
`_excel_cases()` records the selected rows and coverage adaptations; these are not unchanged Excel rows.

## Save and batch-run PT cases

```bash
QLIV2_SINGLE_SAVE_PT_DIR="$TMPDIR/pt_path" python3 -m pytest -q test_quant_lightning_indexer_v2_single.py
QLIV2_TESTCASE_DIR="$TMPDIR/pt_path" python3 -m pytest -q test_quant_lightning_indexer_v2_batch.py
```

Use `QLIV2_CASE_NAMES` or `QLIV2_CASE_INDEXES` to select an ordered subset.
Every run compares CPU and NPU Sparse and Candidate outputs through the unified
ASC-style `result_compare_method` entry.

## Excel to PT

`batch.quant_lightning_indexer_v2_pt_save.load_excel_test_cases()` reads a
sheet containing `Testcase_Name` and every column in `qliv2_test_utils.PARAM_NAMES`.
`save_test_case()` writes one deterministic `.pt` input per row.
