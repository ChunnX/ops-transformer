# QBSA MXFP8 Pytest 用例说明

本目录用于运行 QuantBlockSparseAttn 的 MXFP8 full-quant golden/NPU 对比用例。

## 文件结构

- `qbsa_mxfp8_golden.py`：执行入口，负责生成输入、CPU golden、NPU 调用和精度对比。
- `qbsa_mxfp8_test_cases.py`：默认 testcase 文件，保留少量典型 case，便于日常快速回归。
- `qbsa_mxfp8_test_cases.csv`：批量 testcase 文件，每行对应一个 case。
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
  --case_name TYP_PAGE_REUSE_BN1_BS128_S1_LT_S2
```

`--case_id` 保留为兼容参数，效果等同于 `--case_name`。

也支持用逗号一次指定多个 case：

```bash
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py \
  --case_name TYP_PAGE_REUSE_BN1_BS128_S1_LT_S2,TYP_NO_MASK_S1_EQ_S2_BS64_BOTH_UNALIGNED
```

## 使用 Case CSV

使用 CSV 数据源时必须增加 `--csv`。此时未指定 `--case_files` 会读取同目录下的
`qbsa_mxfp8_test_cases.csv`。CSV 使用 Python 标准库读取，不依赖 `pandas` 或 `openpyxl`。

```bash
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py \
  --csv
```

此时不指定 `--case_name`，会运行 CSV 中所有 `enable=true` 的 case；指定单个 case 的方式为：

```bash
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py \
  --csv --case_name F_BS64_R5_N3_BN1_GT_AA
```

指定其他 CSV 或一次加载多个 CSV：

```bash
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py \
  --csv --case_files /path/to/cases_a.csv,/path/to/cases_b.csv
```

不带 `--csv` 时，`--case_files` 按 Python testcase 文件解析：

```bash
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py \
  --case_files /path/to/custom_test_cases.py
```

## 批量用例设计

CSV 当前包含 500 条 case：

- 352 条重点全交叉：`block_size=64/128`、`N1/N2=5/16`、`N2=3/8`、
  `block_num=1/128/256/512`，再与 11 组可实现的 S1/S2 大小关系及对齐状态完全交叉；
- 72 条非重点 head 与 `mask_mode=0` 补充用例；
- 48 条 `block_num=1024/2048/4096/8192` 用例；
- 27 条非重点 head、稀疏模式和 block-table 模式的混合边界用例；
- 1 条专用空 Tensor 用例：`actualSeqLengths` 和 `actualSeqLengthsKv` 传空 Tensor，序列长度仍由
  `cuSeqlensQ/cuSeqlensKv` 表达。

其中 48 条大 `block_num` 用例默认 `enable=false`，避免直接执行 `--csv` 时占用过多内存；仍可使用
`--case_name` 点名执行。其余 452 条默认启用。

S1/S2 相等时，“S1 不对齐 128、S2 对齐 512”在数学上不可实现：一个数只要对齐 512，就必然对齐
128。因此三个关系与四种对齐状态共有 11 个有效组合，而不是 12 个。

默认 Python testcase 文件当前包含 14 条典型覆盖用例，覆盖两种
block size、重点 head 组合、三种 S1/S2 关系、四种对齐状态、`mask_mode=0/3`、大 `block_num`、空
`actualSeqLengths`、64大小KV块拼接、变长多batch、反序block/sparse、双S2循环和关闭LSE；其中大
`block_num` 典型用例默认关闭。

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
`qbsa_mxfp8_test_cases.csv` 中新增一行，并通过 `--csv` 执行。两种方式都要求
`name` 唯一；`enable=True` 表示默认运行，关闭的 case 仍可通过 `--case_name` 单独执行。

字段格式：

- `actual_seq_q`、`actual_seq_kv` 使用 JSON 数组文本，例如 `[256]`。
- `enable`、`is_contiguous`、`return_softmax_lse`、`empty_actual_seq` 使用 `true/false`。
- `softmax_scale` 留空时，脚本按 `1 / sqrt(D)` 自动补齐。
- `N1` 必须等于 `N2 * n1_n2_ratio`。
- `block_size`、`sparse_q_block_size`、`sparse_kv_block_size` 必须相等。
- `s1_s2_relation` 与 `alignment` 会根据 `S1/S2` 及 `s1_base_size/s2_base_size` 做一致性校验。
- CSV 可直接扩展到多行；case 名称在一个或多个输入文件之间都不能重复。
