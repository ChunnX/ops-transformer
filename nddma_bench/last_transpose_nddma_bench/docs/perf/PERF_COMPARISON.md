# LastTranspose NDDMA — 迁移版 vs 源 `<<<>>>` 基线性能对照

**结论：无回归。** registry-invoke 迁移遵循 R-054 only-move-not-rewrite —— compute 头
（`op_kernel/arch35/last_transpose_nddma_bench.h`）与源工程逐字一致，故任何带宽差异均为 msprof
测量抖动，非算法改动。精度侧 `validate_pta.py` 走 `torch.equal` bitwise 全等，与带宽无关地独立坐实功能正确。

## 对照口径

- **迁移版数据**：`docs/perf/parsed_matrix.json`（FP16+FP32+B8，四配置 db_rw/sb_rw/db_r/sb_r）
  + `docs/perf/parsed_b8.json`（int8-only 复采）。带宽 = factor·numel·elemSize / min(TaskDuration)，
  factor=2（isMte3=1 读写）/1（isMte3=0 纯读）。
- **源基线**：`ops-direct-invoke/.../last_transpose_nddma_bench/output/lasttranspose_matrix_report.md`。
- 采集环境：Ascend 950PR，dim0=56（物理 vector core 数），iters=5（另加 1 预热），取 min 峰值带宽。

## 汇总（parsed_matrix.json 全量四配置 × FP16/FP32/B8）

| 指标 | 值 |
|------|------|
| 对照行数 | 216 |
| \|delta%\| 均值 | 2.48% |
| \|delta%\| 中位 | 0.87% |
| \|delta%\| 最大 | 36.72% |
| 有符号 delta% 均值 | **-0.14%**（无系统性偏移） |
| 超 ±10% 行数 | 12 / 216 |

B8-only 复采（parsed_b8.json，80 行）独立复核：\|delta%\| 均值 3.12%、中位 0.49%、
有符号均值 +0.47%、超 ±10% 9 行——与主采一致。

## 抖动区间与离群点判读

- **中位 delta 0.87%** 表明绝大多数用例带宽与源基线重合。
- 所有超 ±10% 离群点集中在**纯读（`_r`）+ 小内轴（d3∈{15,16,31,32}）** 用例：
  单 task 时延 sub-µs 量级，msprof 计时相对噪声天然偏高；且离群**双向对称**
  （+36.7% 与 -26.9% 同 case 不同配置并存），有符号均值 ≈ 0，非单向劣化。
- 读写（`_rw`）大块用例（如 256×256）delta 普遍在 ±3% 内，落在采集台噪声地板。

**判定**：compute 逐字保留 + 精度 bitwise 全等 + 有符号 delta ≈ 0 → 带宽差异为测量抖动，**无性能回归**。

## 证据快照

- `parsed_matrix.json` — 迁移版四配置全量带宽（本次归档证据）
- `parsed_b8.json` — int8 复采带宽（离群点交叉验证）
- 原始 `PROF_*` msprof 采集目录体积庞大，按 `.gitignore` 排除，不入库。
