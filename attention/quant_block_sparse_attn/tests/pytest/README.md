# QuantBlockSparseAttn pytest

## 目录结构

```text
pytest/
├── common_utils/
│   ├── qbsa_extension.py
│   ├── result_compare_fp8.py
│   └── result_compare_mxfp8.py
├── bsa_fullquant_mxfp8_test/
│   ├── qbsa_mxfp8_golden.py
│   ├── qbsa_mxfp8_test_cases.py
│   ├── case_files/
│   │   ├── qbsa_mxfp8_redline.csv
│   │   ├── qbsa_mxfp8_stc.csv
│   │   └── qbsa_mxfp8_stc_perf.csv
│   └── utils/
│       └── golden_cache.py
├── bsa_fullquant_fp8_test/
│   ├── quant_block_sparse_attn_golden.py
│   ├── test_quant_block_sparse_attn_*.py
│   ├── test_run.sh
│   └── batch/
├── conftest.py
└── pytest.ini
```

FP8 保留原有文件和 `batch/` 结构。公共扩展加载、两套精度比较工具集中在 `common_utils`，各自的精度规则保持不变。
TTK 的输入生成与 Golden 适配器引用这里的 FP8/MXFP8 文件。

pytest 和 TTK 使用相同的目录名，但 CSV 格式不同。TTK 的 `-i` 应指向
`tests/ttk/bsa_fullquant_mxfp8_test/` 或 `tests/ttk/bsa_fullquant_fp8_test/` 下的 CSV。
将 pytest 参数表传给 TTK 会报缺少 `api_name`、`tensor_view_shapes`、`tensor_dtypes`。

在已安装 ops-test-kit 的环境中，从本仓库根目录运行 MXFP8 STC：

```bash
python -m ttk e2e \
  -i attention/quant_block_sparse_attn/tests/ttk/bsa_fullquant_mxfp8_test/qbsa_mxfp8_stc.csv \
  --plugin attention/quant_block_sparse_attn/tests/assets
```

追加 `--validate` 可只检查用例，不执行算子。

## 前置条件

使用已配置 CANN、torch、torch_npu、torchair 和 pytest 的环境，并先安装编译后的 QBSA 算子包。
统一扩展从仓库根目录安装：

```bash
python -m pip install ./torch_extension --no-build-isolation
```

## MXFP8

在仓库根目录列出默认 Python 用例：

```bash
python attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py --list_cases
```

列出 STC CSV 用例：

```bash
python attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py \
  --csv --case_files qbsa_mxfp8_stc.csv --list_cases
```

`--csv` 不指定文件时读取 `case_files/qbsa_mxfp8_redline.csv`。
执行模式与缓存用法见 [MXFP8 使用说明](bsa_fullquant_mxfp8_test/README.md)。

## FP8

从仓库根目录启动：

```bash
bash attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_fp8_test/test_run.sh single
bash attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_fp8_test/test_run.sh batch
bash attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_fp8_test/test_run.sh graph
```

脚本自动切换到 `bsa_fullquant_fp8_test`，PT 与结果文件继续由 FP8 测试入口管理。
直接运行 pytest 时，可从 `pytest/` 或 `bsa_fullquant_fp8_test/` 启动，公共配置负责添加包的搜索路径。
FP8 参数和 CSV 使用方式见 [FP8 使用说明](bsa_fullquant_fp8_test/README.md)。
