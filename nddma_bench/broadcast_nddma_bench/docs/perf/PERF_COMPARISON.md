# Broadcast NDDMA（3D NDDMA 方案，DESIGN §9）— 迁移版 vs 源 `<<<>>>` 基线性能对照

**结论：无回归。** registry-invoke 迁移遵循 R-054 only-move-not-rewrite —— compute 头
（`op_kernel/arch35/broadcast_nddma_bench.h`）与源工程逐字一致，带宽差异均为 msprof 测量抖动。
精度侧 `validate_pta.py` 走 `torch.equal` bitwise 全等（`broadcast_to` golden），独立坐实功能正确。

## 对照口径

- **迁移版数据**：`docs/perf/parsed5d.json`（5D 场景全量，四配置 db_rw/sb_rw/db_r/sb_r）
  + `docs/perf/analysis5d.json`（PipeUtilization/aiv_cycles 分析）。
  带宽（§9.6/§7.7 满口径，fp32 elemSize=4）：numel_dst = dim0·dim1·dim2·dim3·dim4，
  isMte3=1 → (R+W)=2·R_logical；isMte3=0 → R_logical。
- **源基线**：`ops-direct-invoke/.../broadcast_nddma_bench/output/broadcast_5d_matrix_report.md`。
- 采集环境：Ascend 950PR，四场景 AB/BA/BAB/ABA，iters=5 + 1 预热，取 min 峰值带宽。

## 汇总（parsed5d.json 全量四配置 × 四场景 × dtype 变体）

| 指标 | 值 |
|------|------|
| 对照行数 | 64 |
| \|delta%\| 均值 | 1.17% |
| \|delta%\| 中位 | 0.19% |
| \|delta%\| 最大 | 23.60% |
| 有符号 delta% 均值 | **+0.40%**（无系统性偏移） |
| 超 ±10% 行数 | 2 / 64 |

## 抖动区间与离群点判读

- **中位 delta 0.19%**：广播用例带宽与源基线几近完全重合——三算子中噪声最低
  （广播 task 时延偏大、msprof 相对噪声天然更小）。
- 仅 2 个离群点，均为 **SCENE_BAB（scene=2）的 `sb_rw`**（AB_2 +23.6%、BA_2 +21.9%）：
  BAB 双轴广播 tile 最大、SB 整块无双缓冲，单次 msprof 会话尾核抖动放大所致；
  DB 对应用例 delta 落在 ±1% 内，佐证非算法问题。

**判定**：compute 逐字保留 + 精度 bitwise 全等 + 中位 delta <0.2% → 带宽差异为测量抖动，**无性能回归**。

## 证据快照

- `parsed5d.json` — 迁移版 5D 四配置全量带宽（本次归档证据）
- `analysis5d.json` — PipeUtilization / aiv_total_cycles 分析（辅助佐证 pipe 行为一致）
- 原始 `PROF_*` msprof 采集目录按 `.gitignore` 排除，不入库。
