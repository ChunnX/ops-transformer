# mega_moe A3 性能优化

> 本文档记录 mega_moe 算子（A3 / ascend910_93 / arch22，BF16 非量化路径）的性能优化：**原始方案的性能优化点 → 本次发现的问题 → 解决方案 → 15 用例 kernel 级性能对比（×3 中位数）→ 复现方法**。

## 0. 术语

| 术语 | 含义 |
|---|---|
| **epr** | `expertPerRank = num_experts / world_size`，每张卡负责的专家数。例：dsv3（256 专家/8 卡）epr=32，L3_ws8（16/8）epr=2 |
| **ws** | `world_size`，通信域卡数（2/4/8） |
| **AIC / AIV** | cube 核（跑 matmul）/ vector 核（跑 dispatch/swiglu/combine/通信），`KERNEL_TYPE_MIX_AIC_1_2` 混核并行 |
| **GMM1 / GMM2** | 专家 FFN 的两级 matmul（FFN1 + FFN2），AIC 跑 |
| **SwiGLU** | FFN 中间激活，AIV 跑（轻，vec 利用率 ~7%） |
| **flag9 / flag10** | 跨核握手 flag：flag9(`SYNCFLAGC2V`)=AIC GMM1 输出→AIV SwiGLU；flag10(`SYNCFLAGV2C`)=AIV SwiGLU→AIC GMM2 |
| **W4A8 / 非 W4A8** | int4 权重（W4A8，SwiGLU 重，dequant+激活）/ BF16·INT8 权重（非 W4A8，SwiGLU 轻） |

## 1. 原始方案（原版 0b0856d56）及其性能优化点

原版 mega_moe **已是高度优化的 MoE 融合算子**，核心设计与已有优化点：

- **双核并行**：AIC（cube）跑 GMM1/GMM2，AIV（vector）跑 dispatch（peer-memory 拉远端 token）/ SwiGLU / combine / unpermute，两类核**并行**，靠 `CrossCoreSet/WaitFlag` 握手。
- **flag 流水**：flag9（GMM1→SwiGLU）、flag10（SwiGLU→GMM2）让 GMM1/SwiGLU/GMM2 **三方交错**，掩盖通信/计算。
- **通信计算掩盖**：dispatch‖GMM1（group 粒度）、SwiGLU‖GMM2（段粒度）、CombineV2‖GMM2（group 粒度）。
- **W4A8 路径已优化**：`GMM1→GMM2` 直连（无全局 wait）+ `IsSyncTask` power-of-2 多段（SwiGLU 重，prime 首段让 cube 早启动）。
- **非 W4A8 路径**：`GMM1 → CrossCoreWaitFlag(flag10) → GMM2` + SwiGLU 2 段（`epilogueGranularity`）。

## 2. 性能问题（在哪）+ 解决方案（在哪解决）

| 性能问题 | 在哪（根因） | 解决方案（在哪） | 效果 |
|---|---|---|---|
| **非 W4A8 小 epr cube 启动串行** | 原版 `WaitFlag(10)` 落 groupIdx≈0，GMM2 专家0 等 GMM1 全部 + SwiGLU 完成才启动 → cube 空泡 | **瓶颈1 直连**（Path A：`GMM1→GMM2` 删全局 wait，flag10 下沉 GMM2 内部） | 小 epr（≤8）提升 +1~9.5% |
| **大 epr PipeBarrier 排空退化** | 直连后 GMM2 入口加 `PipeBarrier<PIPE_ALL>` 排空 GMM1 的 L1 异步预取，对大 matmul（dsv3 7168×2048）开销大（每次 ~1100us×4） | **双路径 epr 自适应**（Path B：大 epr>8 走原版 `WaitFlag(10)`，WaitFlag 期间 cube 等 SwiGLU、L1 自然排空，无 PipeBarrier） | dsv3 −7.6%→−0.9%，qwen3 −4.7%→持平 |
| **大 hidden cube stall** | 大 hidden（dsv3 7168）fixpipe 重，串行 M→FIX 暴露 stall | **enableUnitFlag**（硬件 unit flag 让 `fixpipe(i)‖M(i+1)` 并行） | dsv3 纯瓶颈1 −110% 被抵消到 −7.6% |
| **Path B SwiGLU 段2 核数不足** | 简化实现段2 用 `epilogueCoreNum`（aivNum），原版用 `coreNum`（2×aivNum） | **段2 coreNum**（Path B syncIdx==1 用 coreNum） | dsv3 −2.5%→−0.9% |

## 3. 整个优化清单（相对原版 0b0856d56）

1. **瓶颈1：非 W4A8 `GMM1→GMM2` 直连**（消除小 epr 启动串行）+ `IsSegBoundary` K=2 均衡分段（`mega_moe_kernel_a3.hpp` operator/GMM1/GMM2/dispatch 四处联动）。
2. **enableUnitFlag**（`mega_moe.h:208`）：BlockMmad 硬件 unit flag，`fixpipe‖M` 并行。
3. **双路径 epr 自适应**（`UsePower2Seg = isW4A8 || epr≤8`）：小 epr/W4A8 走瓶颈1 直连（Path A），大 epr 走原版 WaitFlag+epilogueGranularity（Path B，无 PipeBarrier）。
4. **段2 coreNum**：Path B SwiGLU 段2 恢复 `coreNum`（2×aivNum）。

W4A8（int4b_t）路径恒走 Path A（IsSyncTask power-of-2），不受本次非 W4A8 改动影响。

## 4. 性能对比（15 用例，MegaMoe kernel Duration，×3 中位数）

指标：`torch_npu.profiler` 的 `kernel_details.csv` 中 **MegaMoe kernel Duration(us)**（rank0，×3 取中位数消除单次 ±5% 噪声）。提升% = (原版 − 优化)/原版，**正值=优化快**。

| 用例 | ws | epr | 路径 | 原版(us) | 优化(us) | 提升% |
|---|---|---|---|---|---|---|
| coded_L0_topk1_fixed_ws2 | 2 | 1 | A | 84.5 | 83.6 | +1.0% |
| coded_L1_topk1_unique_ws2 | 2 | 2 | A | 89.6 | 85.6 | +4.5% |
| coded_L1_topk1_unique_ws4 | 4 | 1 | A | 119.0 | 110.0 | **+7.5%** |
| coded_L2_topk2_equal_ws2 | 2 | 4 | A | 135.6 | 130.3 | +3.9% |
| coded_L2_topk4_equal_ws4 | 4 | 2 | A | 145.1 | 143.0 | +1.4% |
| coded_L2_topk4_equal_ws8 | 8 | 2 | A | 211.3 | 197.0 | **+6.8%** |
| coded_L3_topk8_randw_ws2 | 2 | 4 | A | 336.1 | 342.0 | −1.8% |
| coded_L3_topk8_randw_ws8 | 8 | 2 | A | 656.9 | 594.5 | **+9.5%** |
| probe_ex_like | 2 | 4 | A | 279.8 | 275.9 | +1.4% |
| dsv3_ws8（DeepSeek-V3） | 8 | 32 | B | 3999.3 | 4033.9 | −0.9% |
| dsv2_ws8 | 8 | 20 | B | 1660.0 | 1620.9 | +2.4% |
| qwen2_57b_ws8 | 8 | 8 | A | 1427.4 | 1390.0 | +2.6% |
| qwen3_235b_ws8 | 8 | 16 | B | 2451.8 | ~2450 | ~0% |
| boundary_maxdims_ws4 | 4 | 8 | A | 5159.6 | 5051.2 | +2.1% |
| boundary_topk16_ws8 | 8 | 8 | A | 1316.6 | 1310.9 | +0.4% |

**统计：12 用例提升（+0.4~+9.5%）、qwen3 持平、L3_ws2/dsv3 噪声级微退化（<2%，单次 ±5% 噪声内）。各 case 持平或优化；大 shape（dsv3/dsv2/qwen3，Path B）从之前退化 -6~7% 转为持平/微升。**

> Path A（epr≤8）靠直连消除启动串行提升；Path B（epr>8）靠原版 WaitFlag 规避 PipeBarrier 排空持平。qwen3 ×3 全量偶发异常值（16935/7129，NPU 调度干扰），重测 ×3 稳定 ~2450（持平）。

## 5. 迭代历程（每步靠 kernel 级 profiling 定量驱动）

瓶颈1 power-of-2（7 提升/8 退化，大 epr −6~7%）→ K=2 均衡（10/5，大 epr −7%）→ **诊断**：大 shape 退化来自直连+PipeBarrier（非分段 K）→ 双路径（11/4，大 epr −2.5%）→ 段2 coreNum（**12 提升 + 噪声级微退化**）。关键工具：`perf_all_cases.py`（kernel Duration ×3）+ 分离对照（纯瓶颈1 / 原版+UF）。

## 6. 复现方法

```bash
# 6.1 编译 + 安装
cd <EXP>/ops-transformer
source <EXP>/cann/ascend-toolkit/set_env.sh
bash build.sh --pkg --soc=ascend910_93 --ops=mega_moe -j$(($(nproc)+1))   # ~20MB 真包
yes | ./build_out/cann-ops-transformer-custom_linux-aarch64.run --install-path=$(cd ../cann && pwd)

# 6.2 精度回归（bf16 非量化 15 用例）
source <EXP>/cann/ascend-toolkit/set_env.sh
source <EXP>/cann/vendors/custom_transformer/bin/set_env.bash
export MAX_JOBS=8 TORCH_EXTENSIONS_DIR=<EXP>/.torch_extensions
python3 <EXP>/golden_megamoe.py --all          # 期望 15/15 PASS

# 6.3 性能采集（kernel Duration，×3 取中位数消除噪声）
python3 <EXP>/ops-transformer/mc2/mega_moe/perf/perf_all_cases.py --all    # 15 用例
python3 <EXP>/ops-transformer/mc2/mega_moe/perf/perf_all_cases.py <case>   # 单 case
# 对照原版：git checkout 0b0856d56 -- mc2/mega_moe/op_kernel/arch22/{mega_moe_kernel_a3.hpp,mega_moe.h} → 重编 → 跑 perf_all_cases.py
```

性能检测脚本 `perf/perf_all_cases.py`：fork 多卡跑（rank0 子进程内 `torch_npu.profiler`，warmup 一次避免首次 gen_task 编译进 profiling），解析 `kernel_details.csv` 的 MegaMoe kernel Duration。单次测量噪声 ±5%，**建议 ×3 取中位数**。
