# QBSA MXFP8 Pytest 用例说明

本目录用于运行 QuantBlockSparseAttn 的 MXFP8 full-quant golden/NPU 对比用例。

## 文件结构

- `qbsa_mxfp8_golden.py`：执行入口，负责生成输入、CPU golden、NPU 调用和精度对比。
- `qbsa_mxfp8_test_cases.py`：默认 testcase 文件，使用 Python dict 保存 case。
- `qbsa_mxfp8_test_cases.xlsx`：可选 Excel testcase 文件，每行对应一个 case。
- `golden_cache.py`：输入、CPU 输出、NPU 输出缓存工具。
- `result_compare_method.py`：精度比较工具。

## 前置条件

运行 NPU 路径前需要先完成 PTA 扩展编译/安装，确保 `custom_ops` 可以导入，并注册：

```bash
cd experimental/attention/quant_block_sparse_attn/torch_ops_extension
bash build_and_install.sh
```

## 查看用例

从仓库根目录执行：

```bash
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py \
  --list_cases
```

输出会列出 case 名称和 `enable` 状态。

## 默认执行

不指定数据源和 case 名称时，脚本默认执行 `qbsa_mxfp8_test_cases.py` 中所有 `enable=True` 的 case。

```bash
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py
```

默认 `--mode all`，等价于依次执行：

- `gen`：生成输入并保存缓存
- `cpu`：生成 CPU golden
- `npu`：调用 NPU 算子
- `compare`：比较 CPU/NPU 输出

每个 case 开始时会打印 `CASE START [idx/total]` 标志；一次运行多个 case 时，脚本末尾会输出 `QBSA MXFP8 CASE SUMMARY` 汇总表。

## 指定单个 Case

通过 `--case_name` 指定 case 名称：

```bash
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py \
  --case_name PA_BNBD_B1_QS256_KVS512_Nq1_Nkv1_D128_SP3
```

`--case_id` 保留为兼容参数，效果等同于 `--case_name`。

也支持用逗号一次指定多个 case：

```bash
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py \
  --case_name PA_BNBD_B1_QS256_KVS512_Nq1_Nkv1_D128_SP3,MIX_01
```

## 使用 Case Excel

使用 Excel 数据源时必须增加 `--excel`。此时未指定 `--case_files` 会读取同目录下的
`qbsa_mxfp8_test_cases.xlsx`，默认读取名为 `test` 的工作表：

Excel 模式依赖 `pandas` 和 `openpyxl`；默认 Python testcase 模式不会加载这两个依赖。

```bash
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py \
  --excel
```

通过 `--sheet_name` 指定其他工作表：

```bash
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py \
  --excel --sheet_name case_group_1
```

此时不指定 `--case_name`，会运行 Excel 中所有 `enable=true` 的 case；指定单个 case 的方式为：

```bash
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py \
  --excel --case_name MIX_01
```

指定其他 Excel 或一次加载多个 Excel：

```bash
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py \
  --excel --case_files /path/to/cases_a.xlsx,/path/to/cases_b.xlsx
```

不带 `--excel` 时，`--case_files` 按 Python testcase 文件解析：

```bash
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py \
  --case_files /path/to/custom_test_cases.py
```

## 指定执行阶段

`--mode` 支持 `all/gen/cpu/npu/compare`，也支持逗号组合：

```bash
# 只生成输入
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py \
  --mode gen

# 使用已有缓存，只跑 NPU 并比较
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py \
  --mode npu,compare
```

## 缓存目录

默认缓存目录为本目录下的 `golden_cache/`。可以通过 `--cache-dir` 指定：

```bash
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py \
  --cache-dir /tmp/qbsa_mxfp8_cache
```

缓存文件名前缀默认使用 case 名称。需要自定义时使用：

```bash
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py \
  --cache_case_name my_case
```

## 新增 Case

默认方式是在 `qbsa_mxfp8_test_cases.py` 的 `TestCases` 中新增 dict。也可以在
`qbsa_mxfp8_test_cases.xlsx` 的 `test` 工作表中新增一行，并通过 `--excel` 执行。两种方式都要求
`name` 唯一；`enable=True` 表示默认运行，关闭的 case 仍可通过 `--case_name` 单独执行。

字段格式：

- `actual_seq_q`、`actual_seq_kv` 使用 JSON 数组文本，例如 `[256]`。
- `enable`、`is_contiguous`、`return_softmax_lse` 使用 Excel 布尔值 `TRUE/FALSE`。
- `softmax_scale` 留空时，脚本按 `1 / sqrt(D)` 自动补齐。
- Excel 可直接扩展到多行；case 名称在一个或多个输入文件之间都不能重复。
