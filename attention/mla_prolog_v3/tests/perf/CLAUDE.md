# CLAUDE.md — MLA Prolog Performance Modeling

本文件为 `attention/mla_prolog_v3/tests/perf/` 性能模型子模块的说明文档。当 Claude Code 在该目录工作时，应优先阅读本文件。

---

## 1. Purpose

该目录用 Python 模拟 mla_prolog kernel 的执行流程，输出每个 matmul/vector 阶段的瓶颈、关键路径、优化空间。**无需硬件**即可评估 tiling/切分/量化方案的理论收益。

主要应用场景：
- 评估新优化方案的预期收益（在写 kernel 代码之前）
- 定位现有 kernel 的瓶颈（HBM-bound、cube-bound、L0B-bound 等）
- 搜索最优 tiling 参数（baseM/baseN/baseK、stepBatchSize、Split-KN 配置）
- 跨芯片对比（910B chip-total 模型 vs 950 per-cycle 模型）

---

## 2. Module Layout

```
perf_model.py       — 硬件规格 + per-cycle matmul 时间模型 + Cache 容量校验
                      核心类: AscendHWSpec, MatmulBlockSpec, OperatorParams, QuantMode
                      核心函数: estimate_matmul_detailed, estimate_matmul_split_kn,
                                estimate_matmul_split_m, estimate_all_stages
tiling_sim.py       — Tiling 计算的 Python 复刻 (对应 mla_prolog_tiling.cpp)
                      核心函数: compute_tiling, search_split_kn, search_split_m,
                                search_prefill_optimal, search_matmul_block_sizes
pipeline_model.py   — AIC/AIV pipeline DAG + 关键路径求解 + Roofline 分析
                      核心函数: build_pipeline_dag, format_timeline, roofline_analysis
perf_analyzer.py    — CLI 入口，dispatch 到 11 种分析模式
                      mode_bound, mode_pipeline, mode_estimate, mode_advice,
                      mode_search, mode_report, mode_roofline, mode_block_search,
                      mode_split_kn, mode_split_m, mode_prefill
testcases.py        — 性能用例集 (对应 PERF_CASE_NAMES)
__init__.py         — 包初始化
```

文档与报告：
```
README.md                       — CLI 使用说明 + 各 mode 详解
methodology.md                  — 性能建模方法学 (含 worked examples)
report_ascend950.md             — 950 跨量化模式分析报告
report_split_m.md               — Split-M 详细推导
report_prefill_cube_bound.md    — K-outer 优化报告 (3.3× 加速)
blog_mla_prolog.md              — 从 0 到 1 完整教程
test_split_kn_perf.py           — Split-KN 性能评估 mini test suite (16 cases)
```

---

## 3. Core Hypotheses (重要假设)

性能模型基于以下假设。**理解这些假设是正确解读输出的前提**——任何与硬件实测的偏差都应首先回到这些假设上排查。

### H1. 双流水重叠假设
AIC/AIV pipeline 完全并行，任何阶段的实际时间 = `max(AIC busy, AIV busy, sync overhead)`。模型不模拟硬件级流水气泡（如 MTE2 与 MTE3 的物理资源冲突）。

### H2. Cache 命中率假设
- **跨核 A 复用**：所有核共享同一 A 矩阵时，首核读 HBM，其余命中 L2。由 `estimate_matmul_detailed` 的 `a_shared_cores` 参数显式驱动，带宽通过 `effective_load_bw(active_cores, reuse_count=a_shared_cores)` 用调和平均建模。`a_shared_cores=1`（默认）表示每核独占 A（split-M 语义），走纯 HBM。
- **跨 matmul 复用** (`a_reused=True`)：MM2 的 tokenX 假设 100% 命中 L2（来自 MM1 的预取），走纯 `l2_bw`。与 `a_shared_cores` 正交，`a_reused=True` 时优先生效。
- **A-matrix bandwidth decision (三路分支)**：
  - `a_reused=True` → `l2_bw`（数据已在 L2）
  - `a_reused=False, a_shared_cores>1` → `effective_load_bw(C, reuse_count=a_shared_cores)`（首核冷 HBM + 其余暖 L2 的调和平均）；`a_source="MIX"`
  - `a_reused=False, a_shared_cores=1` → `hbm_bw`（每核独占 A）
- **Weight L2 复用**：Split-M 模式下，B 矩阵假设跨核 L2 100% 命中。

### H2b. 核内 N-blocking 与 A 矩阵重复搬运
- 当 `per_core_N > baseN` 时，核内需 `n_blocks = ceil(per_core_N / baseN)` 次循环搬 B 到 L1
- **A 矩阵在每个 N-block 都需要在 L1 中**：因此 A 被搬运 n_blocks 次
  - `nb=0`：A 首次从 HBM 加载（cold，HBM 带宽）
  - `nb≥1`：A 从 L2 加载（warm，L2 带宽 ~3.25× HBM）
- **B 矩阵每个 N-block 是不同 N 偏移**，均首次从 HBM 加载
- **MTE2 A+B 共享带宽**：MTE2（HBM/L2→L1）可同时搬运 A 和 B，但共享 HBM→L1 带宽。带宽打满时（典型情况），时间 = `(A_bytes + B_bytes) / per_core_bw`，等价于 `A_time + B_time`
- **MTE1 A+B 并行**：MTE1（L1→L0A/L0B）可并行搬运 A 和 B（不同目的地址），时间 = `max(l0a_time, l0b_time)`
- 跨 kL1 迭代通过双缓冲流水重叠：本迭代的 MTE1 与上一迭代的内层计算+FixPipe 重叠
- L1、L0 默认 **双缓冲 (2×)**；当且仅当矩阵可一次性全载且带来性能收益时用单缓冲 (1×)
- **L0C 双缓冲显式判定**：L0C 内部始终以 FP32 存储（与输出 dtype 无关），单 tile 大小 = `baseM × baseN × 4`。当 `2 × tile ≤ l0c_size`（950 = 256 KB）时双缓冲成立，`per_nb = max(MTE1, inner, fixpipe)`（fixpipe 与下一 tile 的 MMAD 并行）；否则单缓冲，fixpipe 必须跟在同一 tile 的 inner 后串行：`per_nb = max(MTE1, inner + fixpipe)`。decode 常见 tile ≤ 16 KB 轻松双缓冲；prefill + K-outer 取 baseM=192/256 搭配 baseN=768 时会撞线。

### H2c. FixPipe 带宽与目的端
- FixPipe 的物理吞吐有两个串联约束：核内 `l0c_to_out` 引擎吞吐 和 目的端的落盘带宽
- **`dest="UB"`/`"L1"`（核内）**：走 `l0c_to_out_per_cycle × freq` 的核内峰值（950 ≈ 422 GB/s/core），不受跨核竞争影响。baseline MM1/2/3/4 都走此路径（L0C→UB→后续 vector op）
- **`dest="GM"`（跨核写 HBM）**：在 `min(l0c_to_out 峰值, chip_hbm / active_cores)` 中取 min。950 上 32 核 contention 下 HBM-write cap ≈ 50 GB/s/core，**比核内峰值低 ~8.4×**
- **Split-KN (`k_groups > 1`) 的 cube 输出**：FP32 partial sum 必须落到 GM 供 V 核 MTE2 读回累加，因此 `fixpipe_dest="GM"`。`matmul_split_kn.py::estimate_matmul_split_kn` 和 `estimate_mm2_split_k_cross_core` 内部根据 `k_groups > 1` / `k_cores > 1` 自动选择 `"GM"`
- **`estimate_matmul_{detailed,k_outer,split_m,split_m_l0c_ub,msplit}` + `_estimate_full_load`** 都在签名里加了 `fixpipe_dest="UB"` 参数，默认 UB；仅 Split-KN 路径主动传 `"GM"`
- Helper 函数：`hw.fixpipe_bw(active_cores, dest)`（`hardware.py`），与 `effective_load_bw` 对称设计
- **不建模**：fixpipe 的 quantize/cast cycle 开销（FP32→INT8/FP8 quantize 等视为零）、min burst alignment、scatter 写降速

### H3. MMAD 利用率假设
- BF16: `mmad_ops_per_cycle_fp16 = 4096` (Ascend 950) 假设 100% 利用率
- FP8: `mmad_ops_per_cycle_fp8 = 8192`
- 实际硬件由于发射带宽、L0B→Reg 等限制可能只有 ~85%
- **校准建议**：与实际 profiling 对比时，若 MMAD-bound 阶段误差 >15%，调整一个 dtype-specific derate 因子

### H4. Vector 算力假设
- 950 per-cycle 建模：每条 vector 指令吞吐量 **256 B/cycle**，IPC = **1.3**
  - 指令耗时 = `data_volume(B) / 256`（cycles）
  - 基本块耗时 = `sum(指令耗时) / IPC`
  - 所有指令在 float32 (4 字节/元素) 上操作
  - 具体指令数查表见 `vector_ops.py::vector_instr_count`
- **量化场景差异**：INT8/FP8 模式在 RmsNorm 前增加 dequant 指令（cast + mul），KV 量化模式在 RmsNormCkvKr 后增加 quantize 指令（abs + reducemax + mul + cast），使得 RmsNormCkvKr 从 6 条指令增加到 10 条
- Scatter 效率 0.4（间接寻址降速 60%）
- Legacy 910B path: chip-total `vector_ops_per_sec_bf16 = 10.0e12`，不区分指令（当 `has_vec_instr_spec=False` 时走此路径）

### H5. HBM 带宽分摊假设
- 单核带宽 = `min(per_cycle * freq, chip_total / active_cores)`
- 950: 单核 cycle 33.3 B/cycle × 1.65 GHz = 54.9 GB/s
- chip total 1.6 TB/s，32 核满载时单核被 cap 到 50 GB/s（contention）
- **不建模**：地址通道竞争、bank conflict、burst 长度差异

### H6. Sync overhead 假设
- **建模约定：不计入 CrossCoreSetFlag/WaitFlag 的耗时**（`sync_overhead_us = 0.0`，950 与 910B 同）
- DAG 中的 sync 节点（`sync_signal_*` / `aic_wait_*` / `aiv_wait_*`）保留以维持依赖关系结构，但时长为 0
- 此约定使所有 baseline 数字纯反映 cube/vector 计算耗时，便于跨方案对比
- 不建模 PIPE_FIX vs PIPE_MTE3 的实际硬件延迟差异
- **历史**：曾使用 0.8 µs/点（950）、1.0 µs/点（910B）；现统一归零

### H7. Split-K 累加精度假设
- K-split 引入累加非确定性
- 模型假设 vector 侧 **按固定顺序** 累加（保证位重现）
- 不计入额外精度损失到性能成本中

### H8. Tiling 对齐假设
- 所有 baseN 假设 32B 对齐（BLOCK_SIZE）
- baseM 假设 16 对齐
- 实际 kernel 可能因 stride 限制使用更严格对齐

### H9. Split-KN 建模约定（chip-total / contention）
- `estimate_matmul_split_kn` 与其他 5 个 timing 模型 (`estimate_matmul_detailed`、`estimate_matmul_legacy`、`estimate_matmul_split_m`、`estimate_matmul_k_outer`、`estimate_vector_op`) 采用 **统一的 chip-total / contention 约定**：传 chip-total 维度 + 总活跃核数，函数内部 `chip_bw / active_cores` 得到 per-core 带宽，反映多核同时竞争 HBM
- **Cube 调用**：`estimate_matmul_detailed(M, K_per_core, N_per_core × total_cores, active_cores=total_cores, ...)`。每核处理 N_per_core 列（n-group 切分）、K_per_core 行（k-group 切分），但 total_cores=n_groups×k_groups 个核同时竞争 chip-total HBM 带宽，per_core_bw = `chip_bw / total_cores`
- **per_core_N 计算**：`per_core_N = ceil((N_per_core × total_cores) / total_cores) = N_per_core`。K-cores within one n-group 都做同一个 N 切片（只切 K，不切 N），**不再被 k_groups 错除**
- **FixPipe 压力**：cube 输出 dtype 在 k_groups > 1 时为 FP32 (4 字节)；per_core FixPipe bytes = `baseM × N_per_core × 4`，正确反映 K-split 引入的搬出压力
- **Vector 累加调用**：`estimate_vector_op(load=k_groups × M × N_full × 4, compute=M × N_full × (k_groups − 1), store=M × N_full × out_size, active_cores=vec_cores)`，所有 V 核分摊 chip-total 累加工作
- **流水掩盖**：累加 stage (`AccumCq` / `AccumCkvKr`) 与后续 AIC matmul 的重叠由 `build_pipeline_dag` 在 DAG 层处理，本函数不做掩盖。这与 H7、`feedback_pipeline_overlap.md` 一致
- **修复历史**：commit 9d17f206b 一度误改为 `active_cores=1`（per-AI-core peak，全 per-core 带宽），与其他 5 个模型不一致；后续 commit 修正回 chip-total / contention 约定
- **解读**：报告的 `cube_us` 是 chip-total wall-clock 时间，等同于一个 AI core 在 chip 整体带宽竞争下完成自己 tile 所需的时间。"per-AI-core" 在这里指 *per-AI-core 的工作切片*，而非 *per-AI-core 的独占带宽*

---

## 4. Design Prototypes (关键设计原型)

模型经过多次迭代发展出以下 5 个核心原型，每个都有对应的 search/optimize 入口：

### Prototype 1: Per-Cycle Inner-Loop Model (Ascend 950)
**目的**：精确建模 matmul 的 L0/L1 流水

**结构**：每个 kL1 迭代分解为三段：
1. **MTE1**：HBM/L2 → L1 加载 (`max(load_A_time, load_B_time)`)
2. **Inner stepK**：L1 → L0A/L0B + MMAD (`stepK × max(l0a, l0b, mmad)`)
3. **FixPipe**：L0C → HBM/UB 写回

三段在 kL1 之间双缓冲重叠：
```python
per_kl1 = max(mte1_time, total_inner, fixpipe_time)
total = kL1Loops × per_kl1
```

**实现**：`perf_model.py::estimate_matmul_detailed`
**优化空间**：增大 baseM 让 MMAD 时间超过 L0B 加载 → cube-bound

### Prototype 2: Pipeline DAG Model
**目的**：捕捉 AIC/AIV 跨阶段同步开销与重叠收益

**结构**：
- **节点**：MM1, MM2, RmsNormCq, RmsNormCkv, MM3, RoPE, MM4, Scatter 等
- **边**：CrossCoreSetFlag/WaitFlag 形成的依赖
- **求解**：Kahn 拓扑排序 + 关键路径回溯

**关键洞察**：
- AIV 累加时间常被下一个 AIC matmul 覆盖（feedback_pipeline_overlap.md）
- 不能简单把 cube_us + accum_us 相加

**实现**：`pipeline_model.py::build_pipeline_dag`
**输出**：`format_timeline()` 生成 ASCII 时序图

### Prototype 3: Split-KN 2D 切分搜索
**动机**：当 N 太小时单核 N 不足以填满核数，且 K 太大造成 HBM 带宽瓶颈。Split-KN 把核数分成 `n_groups × k_groups` 网格，K 切分核产生 float32 partial，AIV 顺序累加。

**配置空间**：
```
n_groups × k_groups = aic_num (32 on 950)
合法组合: 1×32, 2×16, 4×8, 8×4, 16×2, 32×1
```

**关键约束**：必须做 **full pipeline 评估**——不能孤立看 MM1/MM2，因为：
- AIV 累加可能与下一个 AIC matmul 重叠
- 累加 sync overhead 对总耗时的影响

**实现**：`tiling_sim.py::search_split_kn`，迭代调用 `estimate_matmul_split_kn` + `build_pipeline_dag`
**典型增益**：MM1/MM2 在 decode 场景 1.5–1.7× 加速（见 `test_split_kn_perf.py`）

### Prototype 4: Split-M for Prefill
**动机**：prefill 场景 T 大、N 小，K 不再是瓶颈，转为按 M 切分以减少跨核同步。

**结构**：
- 每核处理 `M/m_groups` 个 token + **全量 N**
- B 矩阵跨核 L2 复用（feedback_hbm_l2_bandwidth.md）
- 3 级 tiling：`m_loops × kL1 × n_blocks`

**关键洞察**：B 矩阵从 L2 加载的成本由所有核摊销，但 m_groups 越大每核 M 越小，需要权衡

**实现**：`perf_model.py::estimate_matmul_split_m`，搜索 `tiling_sim.py::search_split_m`
**适用场景**：prefill T ≥ 256，small N (≤16) 时收益最大（4.8×）

### Prototype 5: K-Outer Loop for Cube-Bound Prefill
**动机**：标准 split-N 在 prefill 仍然 HBM 限速，需要更大的 baseM 让 MMAD 主导，同时让 K 在外层循环以保持 B 在 L1 内跨 N-block 复用。

**关键洞察**：当 `baseM ≥ sizeof(B) × mmad_tput / l0b_bw ≈ 88` (BF16) 时 MMAD 时间超过 L0B 加载时间 → cube-bound

**结构**：将 K 提到外层，每个 K-slab 内对所有 N-block 复用 B
```
for kL1:
    load B_slab to L1
    for n_block:
        for stepK:
            MMAD using B_slab
```

**实现**：`tiling_sim.py::search_prefill_optimal`
**典型增益**：3.3× pipeline 加速（见 `report_prefill_cube_bound.md`）

---

## 5. How To Use

### 基本调用

```bash
cd attention/mla_prolog_v3/tests/perf

# 单点分析（默认 BF16, 950）
python perf_analyzer.py --chip 950 --batch-size 1 --head-num 128 --mode full

# 跨场景报告
python perf_analyzer.py --chip 950 --mode report

# 加载预定义 testcase
python perf_analyzer.py --chip 950 --from-testcase perf_decode_bs1 --mode pipeline

# 单一分析模式
python perf_analyzer.py --chip 950 --batch-size 32 --head-num 128 --mode block-search
```

### 15 种分析模式

| Mode | 用途 | 仅 950 |
|------|------|:------:|
| `bound` | 各阶段 MTE1/L0/MMAD/FixPipe 时间分解 | |
| `pipeline` | AIC/AIV 时序图 + 关键路径 | |
| `estimate` | 单次内核耗时估算 | |
| `advice` | 自动瓶颈检测 + 优化建议 | |
| `search` | stepBatchSize 搜索 | |
| `report` | 全 testcase 对比 | |
| `roofline` | Roofline 模型 | ✓ |
| `block-search` | baseM/baseN/baseK 搜索 | ✓ |
| `split-kn` | 2D N×K 切分搜索 | ✓ |
| `split-kn-fused` | split-kn 补充：融合 K-accum + RmsNorm in M-split V 核 | ✓ |
| `split-m` | M 轴切分搜索 | ✓ |
| `prefill` | K-outer + cube-bound 优化 | ✓ |
| `prefill-msplit` | M-Split + K-outer 消除全核 barrier，cube-bound prefill | ✓ |
| `dequant-opt` | UB-pipelined dequant+RoPE (L0C→UB→L1, intra-core sync) | ✓ |
| `full` | 上面全部（不含 `dequant-opt`、`split-kn-fused`、`prefill-msplit`，避免破坏 regression goldens） | |

### 程序化调用

```python
from perf_model import OperatorParams, ASCEND_950, estimate_all_stages
from tiling_sim import compute_tiling
from pipeline_model import build_pipeline_dag

params = OperatorParams(batch_size=1, seq_len=1, head_num=128)
tiling = compute_tiling(params, ASCEND_950)
stages = estimate_all_stages(params, tiling, ASCEND_950)
result = build_pipeline_dag(params, tiling, ASCEND_950, stages)
print(f"Pipeline total: {result.total_us:.2f} us")
print(f"Critical path: {' → '.join(result.critical_path)}")
```

---

## 6. Files & Documentation

| 文件 | 作用 |
|------|------|
| `README.md` | CLI 使用说明 + 11 种 mode 详解 |
| `methodology.md` | 性能建模方法学，含 Roofline、Cache 层次、Worked Examples |
| `report_ascend950.md` | 950 全 testcase 跨量化模式分析 |
| `report_split_m.md` | Split-M 详细推导与场景对比 |
| `report_prefill_cube_bound.md` | K-outer 优化（3.3× 加速） |
| `blog_mla_prolog.md` | 从 0 到 1 的完整教程文档 |
| `test_split_kn_perf.py` | Split-KN 性能评估 mini test suite (16 cases) |

---

## 7. When Modifying the Model

修改性能模型时的 5 条核对清单：

1. **校验 1 — 硬件常数**：修改 `perf_model.py` 中的 `ASCEND_950` 常数前，对照 Ascend 950 datasheet（特别是 `chip_hbm_bw_bytes_per_sec`、`mmad_ops_per_cycle_*`、cache 大小）

2. **校验 2 — README 同步**：新增 / 修改分析模式时，**必须**在 `README.md` 同步更新模式列表与示例（避免 stale documentation）

3. **校验 3 — HBM/L2 分离建模**：修改 split-kn / split-m 模型时，HBM 与 L2 必须分开建模——绝不可用单一带宽数字代替。Ascend 950 上 L2 ≈ 3.25× HBM，混淆会产生 ~50% 误差

4. **校验 4 — Pipeline 重叠**：计算累加耗时时，AIV 累加可与下一个 AIC matmul 重叠，**不可** 简单 `cube_us + accum_us`。必须通过 `build_pipeline_dag` 走 DAG 评估

5. **校验 5 — Tiling 同步**：任何 host 侧的 tiling 调整（如 `mla_prolog_tiling.cpp` 改 `mm1BlockNum`）**必须**同步到 `tiling_sim.py::compute_tiling`，否则性能模型与实际 kernel 行为脱节

6. **校验 6 — FixPipe 建模约定**：`fixpipe_us` summary 按 **每 (m_block, n_block) 一次** 计（不是 per kL1）；critical path max 公式随 L0C 双缓冲可行性分支 (`2 × baseM × baseN × 4 ≤ l0c_size`)；目的端 GM（Split-KN FP32 partial）带宽 ≠ 目的端 UB（baseline）。改 matmul timing 模型时：新路径必须保持这三条一致，否则会偷偷回归到已修复的 Bug 1/2/3（见 `perf/model/matmul_{detailed,k_outer,split_m,msplit}.py` 的 `l0c_double_buffered` 分支 + 只有 `matmul_split_kn` 传 `fixpipe_dest="GM"` 的 pattern）

---

## 8. Known Limitations

性能模型 **不建模** 以下因素，使用结果时需注意：

1. **GM 一致性开销**：AIC 的 GM 写入对 AIV 的可见性延迟未建模
2. **ICache miss**：所有指令假设全部命中
3. **TLB miss**：默认全命中，实际大 batch 可能 miss
4. **chip-total 模型 (910B)**：仅给出 chip 总时间估算，不分解到 inner loop（与 950 的 per-cycle 模型不可比）
5. **量化模式覆盖**：完整支持 BF16 / INT8 partial / INT8 full / FP8。per-tile 量化（kvQuantMode=3）的 dequant scale 加载未精确建模
6. **MM3/MM4 的 Split-KN**：仅建模 MM1/MM2 的 Split-KN，MM3/MM4 始终用 Split-N（因为 MM4 矩阵小、MM3 N 已经够大）
7. **kNopeClipAlpha**：作为标量参数，对性能影响可忽略，未纳入模型
8. **变长序列 (PA_BLK_BSND/PA_BLK_NZ)**：cacheIndex 间接寻址成本未单独建模
9. **dequant-opt intra-core HardEvent 延迟**：`build_pipeline_dag_dequant_opt` 假设 `FIX_V`、`V_MTE3`、`MTE3_M` 等核内 HardEvent 同步开销为 0，与 H6 相符。实际硬件可能有 1–2 cycle stalls，需 profiling 校准。
10. **split-kn-fused 在新建模约定下退化为与 split-kn 等价**：本模式以 `hw.sync_overhead_us` 作为"独立 Accum 步骤"在真实 kernel 中需要插入的"would-be barrier"成本。在新约定（H6）下 `sync_overhead_us = 0`，该成本归零，split-kn-fused 与 split-kn 输出相同的 best config。该 mode 仍保留作为"如果未来需要重新计入 sync 成本时的开关"。现有 kernel barriers (line 854, 859, 867) 与此优化无关，**不动**。

---

## Quick Reference

| 场景 | 推荐 mode | 关注指标 |
|------|----------|---------|
| Decode 性能调优 | `pipeline` + `split-kn` | 关键路径 + Cube 利用率 |
| Prefill 性能调优 | `prefill` + `block-search` | MM3 bound 类型 |
| 跨芯片对比 | `report` | 总 us + 加速比 |
| 新优化方案验证 | `full` | bottleneck 变化 |
| 量化收益评估 | `report` (跨 quant mode) | 总 us 缩减 |
