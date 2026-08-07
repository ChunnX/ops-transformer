# QBSA MXFP8 TTK 测试

本目录把 `pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_test_cases.py` 中 `enable=True` 的参数集转换为 TTK E2E 用例。TTK 负责用例解析、调度、NPU 执行和结果汇总；`tests/assets` 中的 TestSpec 复用原 pytest 数据生成、CPU Golden 和精度标准。

## 前置条件

1. 已安装与当前源码匹配的 CANN、PyTorch 和 `torch_npu`。
2. 已构建并安装 QBSA PTA 扩展：

   ```bash
   cd ${OPS_TRANSFORMER_ROOT}/experimental/attention/quant_block_sparse_attn/torch_ops_extension
   bash build_and_install.sh
   ```

3. 在 `ops-test-kit` 仓库根目录执行 TTK 命令。

## 校验与执行

```bash
export OPS_TRANSFORMER_ROOT=/path/to/ops-transformer
export QBSA_ASSETS=${OPS_TRANSFORMER_ROOT}/experimental/attention/quant_block_sparse_attn/tests/assets
export QBSA_MXFP8_CASES=${OPS_TRANSFORMER_ROOT}/experimental/attention/quant_block_sparse_attn/tests/ttk/bsa_fullquant_mxfp8_test/qbsa_mxfp8_cases.csv
export PYTHONPATH=${QBSA_ASSETS}:${PYTHONPATH:-}

python3 -m ttk -v
python3 -m ttk info
python3 -m ttk e2e -i "${QBSA_MXFP8_CASES}" --plugin "${QBSA_ASSETS}" --validate
python3 -m ttk list -i "${QBSA_MXFP8_CASES}"
```

先执行一条小用例冒烟：

```bash
python3 -m ttk e2e \
  -i "${QBSA_MXFP8_CASES}" \
  --plugin "${QBSA_ASSETS}" \
  -t STC_B1_BS128_MIN_PAGE_EQ_MASK3 \
  --pc 1 --single-log --dump-on-fail
```

以 Eager 模式执行全部 71 条启用用例：

```bash
python3 -m ttk e2e \
  -i "${QBSA_MXFP8_CASES}" \
  --plugin "${QBSA_ASSETS}" \
  --pc 1 --proc-no-reuse --proc-timeout 600 \
  --warmup=false --run 1 --single-log
```

以 ACLGraph 模式执行全部 71 条启用用例：

```bash
python3 -m ttk e2e \
  -i "${QBSA_MXFP8_CASES}" \
  --plugin "${QBSA_ASSETS}" \
  --aclgraph \
  --pc 1 --proc-no-reuse --proc-timeout 600 \
  --warmup=false --run 1 --single-log
```

`PYTHONPATH` 必须包含 `tests/assets`，因为 CSV 的 `api_name` 是 `bsa_ttk_ops.quant_block_sparse_attn`；`--plugin` 则用于发现同目录下的 `spec.py`。

`return_softmax_lse=False` 时，PTA 返回的第二个输出是 shape `()` 的无效占位 Tensor。TTK Golden 将该槽位保存为 `None` marker 并跳过数值比较；已有 manual-data 目录需要重新 prepare 才会生成新的 `golden_1_none.bin`。

## 重新生成 CSV

在 `ops-transformer` 仓库根目录执行：

```bash
python3 experimental/attention/quant_block_sparse_attn/tests/ttk/bsa_fullquant_mxfp8_test/generate_ttk_cases.py
python3 experimental/attention/quant_block_sparse_attn/tests/ttk/bsa_fullquant_mxfp8_test/generate_ttk_cases.py --check
```

生成器只读取 `qbsa_mxfp8_test_cases.py`，并只输出其中 `enable=True` 的用例；当前共 71 条。生成结果保留 ragged `actual_seq_q/actual_seq_kv`、page block size、稀疏模式、数据范围、随机种子和 LSE 开关。
