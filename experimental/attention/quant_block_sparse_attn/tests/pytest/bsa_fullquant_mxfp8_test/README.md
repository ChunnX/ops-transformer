# QBSA MXFP8 Pytest 用例说明

本目录用于运行 QuantBlockSparseAttn 的 MXFP8 full-quant golden/NPU 对比用例。

## 文件结构

- `qbsa_mxfp8_golden.py`：执行入口，负责生成输入、CPU golden、NPU 调用和精度对比。
- `qbsa_mxfp8_test_cases.py`：testcase 文件，只存放 case 数据。
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

不指定 case 名称时，脚本默认执行 `qbsa_mxfp8_test_cases.py` 中所有 `enable=True` 的 case。

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

在 `qbsa_mxfp8_test_cases.py` 的 `TestCases` 列表中新增一个 dict。字段 `name` 是 case 名称，`enable=True` 表示默认执行。

```python
TestCases = [
    {
        "name": "PA_BNBD_B1_QS256_KVS512_Nq1_Nkv1_D128_SP3",
        "enable": True,
        "B": 1,
        "N1": 1,
        "N2": 1,
        "D": 128,
        "actual_seq_q": [256],
        "actual_seq_kv": [512],
        "S1": 256,
        "S2": 512,
        "block_size": 128,
        "mask_mode": 3,
        "fp8_dtype": "float8_e4m3fn",
        "scale_dtype": "float8_e8m0fnu",
        "quant_group_size": 32,
        "layout_q": "TND",
        "layout_kv": "PA_BNSD",
        "layout_out": "TND",
        "layout_sparse_indices": "B_N_Qb_Kb",
        "softmax_scale": None,
    },
]
```

`softmax_scale=None` 时，脚本会按 `1 / sqrt(D)` 自动补齐。
