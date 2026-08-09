# QBSA FP8 STC TTK 用例

`qbsa_fp8_stc_cases.csv` 由 pytest 的 `batch/cases_stc.csv` 生成，只包含
`enable=TRUE` 的 FP8 STC 用例。当前共有 13 条，用例名、shape、数据范围、
稀疏模式、block-table 模式和算子属性均来自 pytest 源表。

重新生成或检查表格：

```bash
python3 generate_ttk_cases.py
python3 generate_ttk_cases.py --check
```

在 `ops-test-kit` 仓库根目录设置公共路径：

```bash
export OPS_TRANSFORMER_ROOT=/home/w00644563/ops-transformer_ttk
export QBSA_ASSETS=${OPS_TRANSFORMER_ROOT}/experimental/attention/quant_block_sparse_attn/tests/assets
export QBSA_FP8_STC_CASES=${OPS_TRANSFORMER_ROOT}/experimental/attention/quant_block_sparse_attn/tests/ttk/bsa_fullquant_fp8_test/qbsa_fp8_stc_cases.csv
export PYTHONPATH=${QBSA_ASSETS}:${PYTHONPATH:-}
```

先校验 CSV：

```bash
python3 -m ttk e2e \
  -i "${QBSA_FP8_STC_CASES}" \
  --plugin "${QBSA_ASSETS}" \
  --validate
```

执行全部 FP8 STC Eager 用例：

```bash
python3 -m ttk e2e \
  -i "${QBSA_FP8_STC_CASES}" \
  --plugin "${QBSA_ASSETS}" \
  --pc 1 --proc-no-reuse --proc-timeout 600 \
  --warmup=false --run 1 --single-log
```

执行全部 FP8 STC ACLGraph 用例：

```bash
python3 -m ttk e2e \
  -i "${QBSA_FP8_STC_CASES}" \
  --plugin "${QBSA_ASSETS}" \
  --aclgraph \
  --pc 1 --proc-no-reuse --proc-timeout 600 \
  --warmup=false --run 1 --single-log
```

单独执行一条时增加 `-t`：

```bash
python3 -m ttk e2e \
  -i "${QBSA_FP8_STC_CASES}" \
  --plugin "${QBSA_ASSETS}" \
  -t stc_factor_n2_1_sc1_b1_g1_ntd_gt_align_an \
  --pc 1 --proc-no-reuse --proc-timeout 600 \
  --warmup=false --run 1 --single-log
```
