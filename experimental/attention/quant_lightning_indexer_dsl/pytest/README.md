# Quant Lightning Indexer V2-compatible pytest

This directory follows the mainline QLI V2 pytest workflow while keeping the
DSL CPU Golden and fused NPU implementation independent from ASC source code.

## Single/paramset execution

```bash
./test_run.sh
QLIV2_CASE_NAMES=decode_b8_s2_8k ./test_run.sh
```

`test_quant_lightning_indexer_v2_paramset.py` contains one decode case, one
prefill case, and ten cross-axis/branch regression cases.

## Save and batch-run PT cases

```bash
QLIV2_SINGLE_SAVE_PT_DIR=pt_path python3 -m pytest -q test_quant_lightning_indexer_v2_single.py
QLIV2_TESTCASE_DIR=pt_path python3 -m pytest -q test_quant_lightning_indexer_v2_batch.py
```

Use `QLIV2_CASE_NAMES` or `QLIV2_CASE_INDEXES` to select an ordered subset.
Every run compares CPU and NPU Sparse and Candidate outputs through the unified
ASC-style `result_compare_method` entry.

## Excel to PT

`batch.quant_lightning_indexer_v2_pt_save.load_excel_test_cases()` reads a
sheet containing `Testcase_Name` and every column in `qliv2_test_utils.PARAM_NAMES`.
`save_test_case()` writes one deterministic `.pt` input per row.
