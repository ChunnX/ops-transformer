# MXFP8 泛化用例生成说明

本目录维护一套确定性、Pytest/TTK 成对、受资源上限约束的 MXFP8 泛化用例。生成器固定输出 2000 条，不包含超大用例，不向 CSV 写入 `quant_matmul` 属性，也不修改 FP8 路径。

## 输出文件

- Pytest：`../qbsa_mxfp8_generalized_2000.csv`
- TTK：`../../../ttk/bsa_fullquant_mxfp8_test/qbsa_mxfp8_generalized_2000.csv`
- 覆盖数据：`qbsa_mxfp8_factor_coverage_2000.json`
- 可视化报告：`outputs/20260823/qbsa_mxfp8_factor_coverage_report_2000.xlsx`

## 资源约束

- `S1_max <= 4096`，`S2_max <= 4096`
- 单用例保守估算占用不超过 256 MiB
- 不包含旧集合中的大 B/大 S 压力用例；这些边界仍由现有 STC 覆盖
- `quant_matmul` 仍是执行开关，正常 CSV 默认不输入

## 定向回归

32 条 `MXREG_*` 用例用于固定历史问题和高风险数学边界：

- Q/K/V 的 `+Inf`、`-Inf`、`NaN`、`[-Inf,+Inf]`
- 每个张量和每类非有限值均成对覆盖单 tile（S2=511）和多 tile（S2=513）
- 原始极大 Q × 负 K 导致 score 全为 `-Inf` 的 S2=511/513/1024/1025
- 第二个 merge tile 引入 NaN、GQA、ragged、zero-Q、zero-KV、`p_scale=None`
- V 双 Inf、多 tile、GQA，且 Q/K 为零的数学边界

另有 42 条 `generalized_boundary` 用例覆盖 1、63/64/65、127/128/129、255/256/257、511/512/513、1023/1024/1025、2047/2048/2049、4095/4096 等边界及其 S1/S2 关系。

## 生成与校验

```bash
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/case_generation/generate_qbsa_mxfp8_generalized.py
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/case_generation/generate_qbsa_mxfp8_generalized.py --check
```

Pytest 运行定向回归示例：

```bash
python experimental/attention/quant_block_sparse_attn/tests/pytest/bsa_fullquant_mxfp8_test/qbsa_mxfp8_golden.py \
  --csv \
  --case_files qbsa_mxfp8_generalized_2000.csv \
  --case_name MXREG_LSE_EXTREME_NEG_SINGLE_TILE,MXREG_LSE_EXTREME_NEG_MULTI_TILE
```

## 覆盖结论

- 当前 104 条 STC：计划单因子 70/70，组合 106/167（63.47%）。
- 新增 2000 条：资源范围内单因子 64/64，计划组合 167/167。
- STC 的主要组合缺口是 `N2 × N1/N2`；该缺口由泛化集合补齐，因此当前无需继续扩充 STC。
- 非连续输入仍保留在 Pytest 定向用例中；当前 TTK CSV 没有 stride 列，不能把它伪装成等价的成对覆盖。
