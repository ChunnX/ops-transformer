# Quant Lightning Indexer V2-compatible pytest

## Torch extension and separate DSL package

Activate `cannbotdsl-0.5.0-20260910` before running. Tests import
`cann_ops_transformer.ops.attention.quant_lightning_indexer_dsl` and call
`torch.ops.cann_ops_transformer.ds41.quant_lightning_indexer`.
The extension imports `ops.quant_lightning_indexer_dsl` from the separately
installed DSL package. Install matching DSL and extension packages; neither
the project root nor an adjacent extension directory is an implementation fallback.
For source validation, put independent extension and DSL package copies on
`PYTHONPATH`. Set `CANNBOTDSL_TEST_OPS_DIR` to the expected `ops` directory to
assert DSL origins and report the actual extension path; this never forces an
adjacent extension to load.

`ds41.quant_lightning_indexer_metadata` is independently registered and returns
`int32[1024]`. Both Torch APIs use `layout_kv`, mapped to DSL `layout_k`.
An explicit metadata tensor is passed through; omitting it generates metadata
through the registered operator. Main layout defaults to PA_BBND; metadata
layout defaults to TND. Both support PA_BBND and TND.

The default paramset selects 16 rows from both Excel files already present in
`excel/`. Source row selections and TND adaptations are explicit in
`test_quant_lightning_indexer_v2_paramset.py::_excel_cases`.
These are adapted Excel cases, not unmodified original rows. Existing PA
paramset definitions remain available as `TEST_PARAMS`.

```bash
export QLIV2_DEVICE_ID=1
export TMPDIR=/tmp/xyh/LI_ds41/sync_back
export CANNBOTDSL_CACHE_DIR="$TMPDIR/cache"
export OMP_NUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
python -B -m pytest -p no:cacheprovider experimental/attention/quant_lightning_indexer_dsl/pytest/test_quant_lightning_indexer_v2_single.py -q -k 'excel_01 or excel_05'
python -B -m pytest -p no:cacheprovider experimental/attention/quant_lightning_indexer_dsl/pytest/test_qliv2_test_utils.py -q -k 'meta_and_fake or metadata_without_tensor'
```

The utility test file checks shape rejection, PA/TND and LD/no-LD equivalence,
offsets/strides, Meta/FakeTensor without DSL execution, candidate/value switches,
explicit/automatic metadata, and metadata calls with no tensor inputs.
`test_run.sh` stores generated results and PT files under the temporary
artifact directory. Do not add new helper/test files to this copied facility.

This directory follows the mainline QLI V2 pytest workflow while keeping the
DSL CPU Golden and fused NPU implementation independent from ASC source code.

## Single/paramset execution

```bash
bash test_run.sh single
PYTEST_ADDOPTS='-k excel_01' bash test_run.sh single
```

Run these shell examples from this pytest directory, or pass the absolute
script path from another working directory. The default is the 16 adapted
Excel cases; select representative cases for interface-only changes.

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
