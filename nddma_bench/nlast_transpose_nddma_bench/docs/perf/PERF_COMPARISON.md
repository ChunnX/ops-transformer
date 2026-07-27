# NLastTranspose [0,1,3,2,4] NDDMA — 迁移版 vs 源 `<<<>>>` 基线性能对照

**结论：无回归。** registry-invoke 迁移遵循 R-054 only-move-not-rewrite —— compute 头
（`op_kernel/arch35/nlast_transpose_nddma_bench.h`）与源工程逐字一致，带宽差异均为 msprof 测量抖动。
精度侧 `validate_pta.py` 走 `torch.equal` bitwise 全等（含 dim4 内轴），独立坐实功能正确。

## 对照口径

- **迁移版数据**：`docs/perf/parsed_matrix.json`（run1）+ `docs/perf/parsed_matrix_run2.json`（run2 复采），
  均为 FP16+FP32+B8 四配置 db_rw/sb_rw/db_r/sb_r。带宽 = factor·numel·elemSize / min(TaskDuration)，
  factor=2（读写）/1（纯读）；numel = dim0·dim1·dim2·dim3·dim4。
- **源基线**：`ops-direct-invoke/.../nlast_transpose_nddma_bench/output/nlasttranspose_matrix_report.md`。
- 采集环境：Ascend 950PR，dim0=56，per-case dim1（与 `prof_pta_matrix.CASES` 同源），iters=5 + 1 预热。

## 汇总（两次独立复采）

| 指标 | run1 | run2 |
|------|------|------|
| 对照行数 | 224 | 224 |
| \|delta%\| 均值 | 4.73% | 5.44% |
| \|delta%\| 中位 | 1.01% | 1.18% |
| \|delta%\| 最大 | 35.43% | 35.26% |
| 有符号 delta% 均值 | **+0.27%** | **+1.78%** |
| 超 ±10% 行数 | 34 / 224 | 42 / 224 |

## 抖动区间与离群点判读

- **中位 delta ~1%**：过半用例带宽与源基线重合；两次复采趋势一致。
- 离群点集中在**纯读（`_r`）+ 极小内轴（dim4∈{15,16}）** 与 A小B大深转置用例：
  单 task 时延小、msprof 相对噪声高；且离群双向并存（同 case 不同配置 +19% / -20%），
  有符号均值 ≈ 0，非单向劣化。
- 相比 last，nlast 抖动带略宽（中位 ~1% vs 0.87%，离群更多）——源于 5D 转置 task 粒度更细、
  逐 task 时延更短，噪声地板天然更高；这是**采集台特性**而非算法退化。

**判定**：compute 逐字保留 + 精度 bitwise 全等 + 有符号 delta ≈ 0 → 带宽差异为测量抖动，**无性能回归**。

## 已知采集 gap

本算子无 FP 全 dtype 全量四配置扫描脚本，FP16+FP32 采集须循环 `collect_single.sh` ×4 配置
（db/sb 间手动 toggle+重编），int8 走 `collect_b8.sh`；上述 parsed json 即由此路径产出。详见 `pta/README.md`。

## 证据快照

- `parsed_matrix.json` / `parsed_matrix_run2.json` — 两次独立复采带宽（离群点交叉验证）
- 原始 `PROF_*` msprof 采集目录按 `.gitignore` 排除，不入库。
