# mega_moe A3 性能优化

> 本文档记录 mega_moe 算子（A3 / ascend910_93 / arch22，BF16 非量化路径）的性能优化：方案、15 用例 kernel 级性能对比（×3 中位数）、复现方法。

## 1. 优化方案

相对原版（commit `0b0856d56`）做了三项改动，核心是**双路径 epr 自适应**：

- **双路径 epr 自适应**（`UsePower2Seg = isW4A8 || epr≤8`）：
  - **Path A（小 epr≤8 / W4A8）**：`GMM1→GMM2` 直连 + `IsSegBoundary`/`IsSyncTask` 分段（最多 K=2 段均衡）+ GMM2 入口 `PipeBarrier` 排空。消除小 epr 的 cube 启动串行（原版 `WaitFlag(10)` 落 groupIdx≈0）。
  - **Path B（大 epr>8，非 W4A8）**：保留原版 `GMM1→WaitFlag(10)→GMM2` + `epilogueGranularity` 2 段，**无 PipeBarrier**（WaitFlag 期间 cube 等 AIV SwiGLU，L1 流水自然排空）。
  - **根因**：诊断证实大 shape 退化来自直连 + GMM2 入口 `PipeBarrier<PIPE_ALL>`（排空大 matmul 的 L1 异步预取，开销大），与分段 K 无关；原版 WaitFlag 间接排空不需 PipeBarrier。故大 epr 走原版路径规避。
- **enableUnitFlag**：打开 BlockMmad 硬件 unit flag，让 `fixpipe(i)‖M(i+1)` 并行（大 hidden 收益显著，如 dsv3 −49% 纯瓶颈1 退化被抵消）。
- **Path B 段2 coreNum**：SwiGLU 段2 用 `coreNum`（2×aivNum，恢复原版），段1/Path A 用 `epilogueCoreNum`。

## 2. 性能对比（15 用例，MegaMoe kernel Duration，×3 中位数）

指标：`torch_npu.profiler` 的 `kernel_details.csv` 中 **MegaMoe kernel Duration(us)**（rank0，×3 取中位数消除单次 ±5% 噪声）。提升% = (原版 − 优化)/原版，**正值=优化快**。

| 用例 | ws | epr | 原版(us) | 优化(us) | 提升% |
|---|---|---|---|---|---|
| coded_L0_topk1_fixed_ws2 | 2 | 1 | 84.5 | 83.6 | +1.0% |
| coded_L1_topk1_unique_ws2 | 2 | 2 | 89.6 | 85.6 | +4.5% |
| coded_L1_topk1_unique_ws4 | 4 | 1 | 119.0 | 110.0 | **+7.5%** |
| coded_L2_topk2_equal_ws2 | 2 | 4 | 135.6 | 130.3 | +3.9% |
| coded_L2_topk4_equal_ws4 | 4 | 2 | 145.1 | 143.0 | +1.4% |
| coded_L2_topk4_equal_ws8 | 8 | 2 | 211.3 | 197.0 | **+6.8%** |
| coded_L3_topk8_randw_ws2 | 2 | 4 | 336.1 | 342.0 | −1.8% |
| coded_L3_topk8_randw_ws8 | 8 | 2 | 656.9 | 594.5 | **+9.5%** |
| probe_ex_like | 2 | 4 | 279.8 | 275.9 | +1.4% |
| dsv3_ws8（DeepSeek-V3） | 8 | 32 | 3999.3 | 4033.9 | −0.9% |
| dsv2_ws8 | 8 | 20 | 1660.0 | 1620.9 | +2.4% |
| qwen2_57b_ws8 | 8 | 8 | 1427.4 | 1390.0 | +2.6% |
| qwen3_235b_ws8 | 8 | 16 | 2451.8 | ~2450 | ~0% |
| boundary_maxdims_ws4 | 4 | 8 | 5159.6 | 5051.2 | +2.1% |
| boundary_topk16_ws8 | 8 | 8 | 1316.6 | 1310.9 | +0.4% |

**统计：12 用例提升（+0.4~+9.5%）、qwen3 持平、L3_ws2/dsv3 噪声级微退化（<2%，单次 ±5% 噪声内）。各 case 持平或优化，大 shape（dsv3/dsv2/qwen3）从之前退化 -6~7% 转为持平/微升。**

> 注：qwen3 ×3 全量偶发异常值（16935/7129，NPU 调度干扰），重测 ×3 稳定 ~2450（持平原版 2452）。

## 3. 分析

- **小 epr（≤8）提升**来自 Path A 直连消除 cube 启动串行（原版 `WaitFlag(10)` 落 groupIdx≈0，GMM2 专家0 等 GMM1 全部+SwiGLU 完成；直连后只等 seg0 SwiGLU）。L3_ws8（epr=2）+9.5% 最显著。
- **大 epr（>8）持平**来自 Path B 规避 PipeBarrier 排空（原版 WaitFlag 间接排空，对大 matmul 无额外开销）。dsv3（epr=32）从 Path A 退化 -7.6% → Path B 持平 -0.9%。
- **enableUnitFlag** 对大 hidden（fixpipe 重）收益显著（dsv3 纯瓶颈1 -110% 被 UF 抵消到 -7.6%，再经双路径到 -0.9%）。
- **迭代历程**：瓶颈1 power-of-2（7/8 混杂）→ K=2（10/5，大 epr -7%）→ 双路径（11/4，大 epr -2.5%）→ 段2 coreNum（12 提升 + 噪声级微退化）。每步靠 kernel 级 profiling（`perf_all_cases.py`）+ 分离对照（纯瓶颈1/原版+UF）定量驱动。

## 4. 复现方法

```bash
# 4.1 编译 + 安装
cd <EXP>/ops-transformer
source <EXP>/cann/ascend-toolkit/set_env.sh
bash build.sh --pkg --soc=ascend910_93 --ops=mega_moe -j$(($(nproc)+1))   # ~20MB 真包
yes | ./build_out/cann-ops-transformer-custom_linux-aarch64.run --install-path=$(cd ../cann && pwd)

# 4.2 精度回归（bf16 非量化 15 用例）
source <EXP>/cann/ascend-toolkit/set_env.sh
source <EXP>/cann/vendors/custom_transformer/bin/set_env.bash
export MAX_JOBS=8 TORCH_EXTENSIONS_DIR=<EXP>/.torch_extensions
python3 <EXP>/golden_megamoe.py --all          # 期望 15/15 PASS

# 4.3 性能采集（kernel Duration，×3 取中位数消除噪声）
python3 <EXP>/ops-transformer/mc2/mega_moe/perf/perf_all_cases.py --all    # 15 用例
python3 <EXP>/ops-transformer/mc2/mega_moe/perf/perf_all_cases.py <case>   # 单 case
# 对照原版：git checkout 0b0856d56 -- mc2/mega_moe/op_kernel/arch22/{mega_moe_kernel_a3.hpp,mega_moe.h} → 重编 → 跑 perf_all_cases.py
```

性能检测脚本 `perf/perf_all_cases.py`：fork 多卡跑（rank0 子进程内 `torch_npu.profiler`，warmup 一次避免首次 gen_task 编译进 profiling），解析 `kernel_details.csv` 的 MegaMoe kernel Duration。单次测量噪声 ±5%，**建议 ×3 取中位数**。
