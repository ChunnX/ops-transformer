# MC2 QuantAllReduce / QuantReduceScatter 优化分阶段汇总（handoff）

> **目的**：本文档为单一入口，供后续对话/分析者快速理解 QAR + QRS 两个算子的全部优化历程、数据资产和测试方法。读完本文档即可独立开展进一步分析与论证。
>
> **当前 HEAD**：`stage3_target_iter` 分支（基于 stage2 `5223bcbd`）
> **覆盖算子**：QuantAllReduce (QAR) + QuantReduceScatter (QRS)
> **环境**：4×Ascend950PR (aivNum=56, AIV:AIC=2:1)，CANN 9.0.0，world_size=4

---

## 1. 优化阶段时间线（4 个数据点）

| 阶段 | commit | xPerBlock 策略 | QAR 实测 vs 紧前一阶段 | QRS 实测 vs 紧前一阶段 |
|------|--------|---------------|------------------------|------------------------|
| **baseline** | `fdbcd206` | `X_PRE_BLOCK_NUM = 1024U` 硬编（无动态计算）| — | — |
| **stage1** | `f2f9b546` | host UB 最大化 → 实际 ub_max ≈ 7168 | +5.27% 回退 | +2.14% 回退 |
| **stage2** | `5223bcbd` | kernel 内 `cap = 2048U` 强制上限 | −9.95% 改善（22 baseline 用例提升 6.5~14.2%）| −6.5~11.4% 改善 |
| **stage3 (HEAD)** | branch `stage3_target_iter` | host TARGET_ITER 公式 + DataCopyPad | **−1.34%** 进一步改善 | **−0.33%（持平）** |

*回退/改善对照基线均为该阶段紧前一阶段；最终 stage3 vs stage2 的对比见 §3*

### 三次优化（stage1/2/3）的核心动作

- **stage1**："让 UB 充分利用" → 拆动态 buffer，允许 xPerBlock 跑到 ub_max ~7168
  - 失败原因：每核 iter 数被压到 1-2，DoubleBuffer 失效；且活跃核数过少
- **stage2**："限制最大块" → kernel 加 `cap = 2048U` 硬编码上限
  - 成功原因：每核 iter 回到 3-6，DB 流水满载
  - 局限：cap=2048 是经验值，不随 (xNums, world_size, aivNum) 自适应
- **stage3**："公式化分块" → host 侧基于 `xPerBlock = max(per_core/T, MIN)` 反推
  - 关键发现：QAR/QRS 物理特性不同（compute-bound vs comm-bound），需差异化参数
  - 最终配置：QAR T=3 MIN=1024，QRS T=3 MIN=2048（QRS 公式实际不激活，靠 MIN 兜底）

---

## 2. 数据资产清单（按算子分组）

### QuantAllReduce 数据

```
mc2/quant_all_reduce/docs/perf/
├── qar_perf/round_001/                 # stage1 阶段数据
│   ├── baseline_perf.csv               # commit fdbcd206 (xPerBlock=4096 硬编 QAR)
│   ├── optimized_perf.csv              # commit f2f9b546 (stage1 ub_max ~7168)
│   ├── baseline_mc2_pro.csv            # 简版 mc2_prof
│   ├── optimized_mc2_pro.csv
│   └── experiment_log.md               # stage1 实验日志
├── round_002/                          # stage2 准备阶段 profiling 原始数据
│   └── raw/                            # 各种 ASCEND_PROFILER_OUTPUT
├── x_block_2048_perf.csv               # stage2 (cap=2048) 24 用例完整数据
├── x_block_2048_perf_report.md         # stage1 vs stage2 对照报告
├── stage3_perf.csv                     # stage3 (本次) 24 用例完整数据
├── stage3_perf_report.md               # stage2 vs stage3 详细对照
└── MC2_QUANT_STAGES_HANDOFF.md         # ← 本文档
```

### QuantReduceScatter 数据

```
mc2/quant_reduce_scatter/docs/perf/
├── qrs_perf/round_001/
│   ├── baseline_perf.csv               # commit fdbcd206 (xPerBlock=1024 硬编 QRS)
│   ├── optimized_perf.csv              # commit f2f9b546 (stage1 ub_max ~7168)
│   ├── baseline_mc2_pro.csv
│   ├── optimized_mc2_pro.csv
│   └── summary.md                      # stage1 实验总结（含 QAR 关联）
├── round_001/ralpha/                   # stage2 阶段 profiling 原始数据
│   ├── baseline_raw/                   # 优化前 4 卡 profile
│   ├── raw/                            # 优化后 4 卡 profile
│   └── batch_perf*.log                 # 批跑日志
├── x_block_2048_perf.csv               # stage2 (cap=2048) 24 用例
├── x_block_2048_perf_report.md         # stage1 vs stage2 对照
├── stage3_perf.csv                     # stage3 24 用例
└── stage3_perf_report.md               # 含 §xPerBlock=512 假设证伪详细分析（重点）
```

---

## 3. stage3 最终成绩（与 stage2 对照）

| 算子 | 精度 | 24 用例总耗时 | Δ vs stage2 | 真实回退 |
|------|------|--------------|-------------|---------|
| QAR | 96/96 PASS, 0 FAIL | 661.93 us（去 ID0013 异常） | **−1.34%** | 0 |
| QRS | 96/96 PASS, 0 FAIL | 232.33 us（去 ID0014 baseline 异常） | **−0.33%（持平）** | 0 |

*1 用例 ID0013 / ID0014 异常源于 profiler 状态残留污染，已分别重测确认*

### 最终代码配置

QAR `quant_all_reduce_tiling.cpp`:
```cpp
constexpr uint32_t TARGET_ITER = 3U;
constexpr uint32_t MIN_BLOCK = 1024U;
constexpr uint32_t ALIGN_BLOCK = 1024U;
xPerBlock = max(ceil(xNums / aivNum / TARGET_ITER), MIN_BLOCK);
xPerBlock = floor(xPerBlock, ALIGN_BLOCK);
```

QRS `quant_reduce_scatter_tiling.cpp`:
```cpp
constexpr uint32_t TARGET_ITER = 3U;     // 与 QAR 对称（QRS 上不激活）
constexpr uint32_t MIN_BLOCK = 2048U;    // 关键：QRS 公式失效，靠 MIN 兜底
constexpr uint32_t ALIGN_BLOCK = 1024U;
xSliceSizeNums = xNums / rankSize;       // QRS 先除 rankSize
xPerBlock = max(ceil(xSliceSizeNums / aivNum / TARGET_ITER), MIN_BLOCK);
xPerBlock = floor(xPerBlock, ALIGN_BLOCK);
```

两边 kernel：
```cpp
constexpr static uint32_t X_BLOCK_ALIGN_NUM = 1024U;
// 读 host 推荐值，与 ub_max 取 min
uint32_t hostRecommended = tiliingDatainfo.xPerBlock;
if (hostRecommended != 0U && maxXPerBlock > hostRecommended) {
    maxXPerBlock = hostRecommended;
}
// scale DataCopy → DataCopyPad（鲁棒性提升，支持 sub-32B 对齐）
```

### xPerBlock 实际下发值（aivNum=56, rankSize=4）

| op | shape | per_core | T=3 raw | floor | iter/核 |
|----|-------|----------|---------|-------|---------|
| QAR | 96×4096 (qwen) | 7023 | 2341 | 2048 | 4 |
| QAR | 96×7168 (DSV3) | 12289 | 4097 | 4096 | 3 |
| QRS | 48×5120 (qwen) | 1097 | 366 | **2048（MIN 主导）** | 1 |
| QRS | 48×7168 (DSV3) | 1536 | 512 | **2048（MIN 主导）** | 1 |

---

## 4. 关键技术结论

### 4.1 QAR (compute-friendly) — 公式有效

- T=3 + MIN=1024 是 (aivNum=56) 下甜点
- DSV3 系列 12 用例由 stage2 的 2048（6 iter）升 4096（3 iter），减少同步开销 −0.9~−3.6% 加速
- qwen 系列 12 用例公式自然落 2048，与 stage2 持平

**T 扫参实测**（详见 stage3_perf_report.md）：
| T | 24 用例 vs stage2 | 回退 |
|---|---------------------|------|
| 6 | +0.92% | 12 用例 +5~8%（h=4096） |
| 3 | **−2.17%** ✅ | 0 |
| 2 | +1.82% | 6 用例（h=7168 块过大 iter=3 不足）|

### 4.2 QRS (comm-bound) — 公式失效，MIN 主导

- per_core ≤ 1536（h=7168）/ 1097（h=5120），任何 T ≥ 1 都让公式 raw < MIN，永远落 MIN
- 实测 MIN=1024 vs MIN=2048：MIN=2048 优 6.5~11.4%（详见 §4.3 假设证伪）
- 最终 MIN=2048 等效 stage2 cap=2048 → **stage3 在 QRS 上持平 stage2**

### 4.3 xPerBlock=512 假设证伪（深度分析）

> 详见 `stage3_perf_report.md` §"xPerBlock=512 假设证伪 — 详细分析"段

stage3 中曾尝试将 X_BLOCK_ALIGN_NUM 从 1024 降到 512 让 QRS 跑 3 iter 启 DB 流水。**实测劣化 +12.14%**，原因：
1. QRS comm-bound：T_compute << T_comm，DB 流水"无可重叠"
2. 块小 4 倍 → iter 多 3 倍 → 通信 N×T_setup 开销线性增长
3. 通信节省（小块）远不及 setup 开销

**核心公式**（论证机制）：
```
mc2_prof ≈ N_iter × (T_setup + T_data + T_compute) - DB_overlap
QAR (compute-friendly):  T_compute ≈ T_data → DB 重叠收益 > N×T_setup → 多 iter 优
QRS (comm-bound):        T_compute << T_data → DB 无重叠 → N×T_setup 主导 → 少 iter 优
```

**推论**：减小 xPerBlock 启动 DB 仅在 compute-bound 时有效，comm-bound 反向（少 iter 大块更优）。

### 4.4 DataCopyPad 改造（边界鲁棒性）

为支持 sub-1024 对齐尝试（已证伪），将所有 scale 相关 DataCopy 替换为 DataCopyPad（共 4 处：QAR/QRS 各 1 处 Win→UB，mte_comm.h 2 处 GM↔UB↔Win）。最终配置回到 1024 对齐，但 DataCopyPad 保留作为鲁棒性提升 + 未来 sub-1024 实验接口预留。

---

## 5. 测试方法（next session 复现）

### 5.1 环境激活

```bash
source $HOME/SUB/vendors/custom_transformer/bin/set_env.bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate mc2_py310
```

### 5.2 编译 & 安装（改动算子代码后）

```bash
cd ~/ops-transformer
bash build.sh --pkg --soc=ascend950 --ops=quant_all_reduce,quant_reduce_scatter
cp build_out/cann-ops-transformer-custom_linux-x86_64.run ~/SUB/
cd ~/SUB && bash cann-ops-transformer-custom_linux-x86_64.run --quiet --install-path=$HOME/SUB
```

### 5.3 NPU 占用检查（每次跑测前必做）

```bash
npu-smi info | tail -12
# 确认 4 张卡 "No running processes found"，否则等空闲（共享服务器）
```

### 5.4 精度 + 性能测试命令

```bash
cd ~/mc2_test_0416

# 跑前清空 csv（避免不同批次混入误读）
rm -f aclnnQuantAllReduce_performance_result.csv aclnnQuantAllreduce_result.csv \
      aclnnQuantReduceScatter_performance_result.csv aclnnQuantReducescatter_result.csv \
      mc2_aclnnQuantAllReduce_pro_result.csv mc2_aclnnQuantReduceScatter_pro_result.csv

# QAR 全 24 用例（精度 + 性能合一，--scene one_hccl）
python3 mc2_test.py --xlsx david_excel/aclnnQuantAllReduce.xlsx --sheet level0 \
    --num 1 24 --scene one_hccl --world_size 4 --soc 910D

# QRS 同
python3 mc2_test.py --xlsx david_excel/aclnnQuantReduceScatter.xlsx --sheet level0 \
    --num 1 24 --scene one_hccl --world_size 4 --soc 910D
```

scene 选项说明：
- `one_hccl`: 4 卡协同跑，**精度对比 + 性能采集合一**（推荐用法）
- `one_hccl_npu`: 仅精度对比批跑（不采性能）
- `npu`: 单卡（功能性 sanity，仍需 world_size=4 启 HCCL）

### 5.5 单用例重测（用于异常点确认）

```bash
# 例如重测 ID0013
python3 mc2_test.py --xlsx david_excel/aclnnQuantAllReduce.xlsx --sheet level0 \
    --num 13 13 --scene one_hccl --world_size 4 --soc 910D
```

注意 `--num` 必须为 `start end` 两个值，单值会报错。

### 5.6 结果分析模板

```python
import csv
with open('aclnnQuantAllReduce_performance_result.csv') as f:
    rows = list(csv.reader(f))
data = rows[1:]
# 列含义: r[1]=case_name, r[8]=mc2_prof(us), r[11]=recordtime
# 多次运行 append，按 case 取 latest 或自行筛选

# 对照 baseline
with open('mc2/quant_all_reduce/docs/perf/x_block_2048_perf.csv') as f:
    bl = {r[1]: float(r[8]) for r in list(csv.reader(f))[1:]}
```

---

## 6. 已知陷阱与注意事项

1. **profiler 状态残留**：批跑时偶有单点 mc2_prof 异常飙升（ID0013/ID0014 等），4× 偏离常态，需单独 `--num N N` 重测确认。所有报告中此类异常已标注。

2. **mc2_test csv 是 append 模式**：不同批次结果混叠。每次跑前必须 `rm` 清空。

3. **NPU 共享冲突**：服务器多人共用，跑前必查 `npu-smi info`。错误码 561000 / EADDRINUSE 多为冲突，看似 bug 实为占用。

4. **测试 shape 数集**：
   - QAR：96×4096 (qwen) + 96×7168 (DSV3) 各 12 用例
   - QRS：48×5120 / 1×48×5120 (qwen) + 48×7168 / 1×48×7168 (DSV3) 各 12 用例
   - **QAR 与 QRS shape 不同**，per_core_elem 计算时容易混淆

5. **aivNum 实际值 = 56**：可通过 `asys info -r=status -d=0` 验证。AIV:AIC = 2:1（aicNum=28）。stage3 公式与该值耦合，换硬件需重扫 T。

6. **ID0013 / ID0014 异常的来源**：profiler 状态残留，与代码无关。可通过单独 `--num N N` 重测确认。stage2 报告 `x_block_2048_perf_report.md` 也提到过同样问题。

---

## 7. 后续可探索方向

1. **是否对 QRS 引入 cost-model**：根据 (per_core_elem, group_size) 自动判定 compute-bound vs comm-bound，分别使用不同公式
2. **sub-1024 alignment 重新尝试**：stage3 已加 DataCopyPad 接口，但需要解决 GM 步长 vs DataCopy 字节对齐的 trade-off（QRS 实测 +12% 已证伪 512 直接降的方案）
3. **TARGET_ITER 跨硬件扫参**：T=3 在 aivNum=56 下最优，aivNum=24/40/48 等其他平台需重扫
4. **更大规模 shape 测试**：当前 24 用例覆盖 bs=48/96, h=4096-7168。更大 bs 或更长 h 下公式行为待验证
5. **fp4 dtype 兼容**：`X_BLOCK_ALIGN_NUM` 注释提到"fp4 需要另外算"，当前 24 用例不含 fp4，未来需补测
6. **统一 QAR/QRS 配置**：当前 QAR MIN=1024 / QRS MIN=2048 不对称，后续是否能用 host 自动判别一致化

---

## 8. 关键参考文件

| 用途 | 文件 |
|------|------|
| QAR stage3 详细数据 + 调优过程 | `mc2/quant_all_reduce/docs/perf/stage3_perf_report.md` |
| QRS stage3 详细数据 + 假设证伪分析 | `mc2/quant_reduce_scatter/docs/perf/stage3_perf_report.md` |
| stage1→stage2 对照报告 (QAR) | `mc2/quant_all_reduce/docs/perf/x_block_2048_perf_report.md` |
| stage1→stage2 对照报告 (QRS) | `mc2/quant_reduce_scatter/docs/perf/x_block_2048_perf_report.md` |
| stage1 vs baseline (QAR) | `mc2/quant_all_reduce/docs/perf/qar_perf/round_001/experiment_log.md` |
| stage1 vs baseline (QRS) | `mc2/quant_reduce_scatter/docs/perf/qrs_perf/round_001/summary.md` |

读完本文档 + 上面 6 份子报告，已具备对 3 次（实际 4 个数据点）做进一步分析与论证的全部信息。
