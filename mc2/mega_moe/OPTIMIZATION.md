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

---

# 第二轮：A3 ws16 **decode** 场景优化（2026-08-04）

> 上一轮（§1~§6）的验证场景是 **ws=2/4/8 的 prefill 用例**。本轮针对 **A3 单节点 ws=16 decode**
> （DeepSeek-V3.2/V4、Kimi-K2.5、GLM-5、Qwen3-235B × batch 16/32/64，共 14 用例）重新做 profiling，
> 结论与 prefill 场景**完全不同**，故单独成章。

## 7. 核心结论：decode 是 HBM 权重搬运带宽受限

用 5 个真实模型 + 一组受控探针（只扫 epr、其余全固定）两条独立路径回归，得到同一个模型：

```
Duration ≈ 权重字节数 / 1300 GB/s  +  85 us
每卡权重字节数 = epr × (hidden×2×inter + inter×hidden) × 2B
```

拟合误差 4/5 模型 < 1.5%。profiling 佐证（dsv3_ws16_dec32，master）：

| 指标 | 值 | 含义 |
|---|---|---|
| `aic_mte2_ratio` | **0.92~0.96** | AIC 九成以上时间在 GM→L1 搬权重，pipe 近乎满载 |
| `aic_mac_ratio` | 0.12~0.28 | cube 算力严重欠载 |
| `aiv_vec_ratio` | 0.002~0.026 | AIV 占满全程却几乎不干活，在等 AIC |

根因：BF16 下**算术强度 ≈ 每专家平均 token 数**（`AI = 2·rows·W /(2B·W) = rows`），decode 每专家仅 5~64 行
→ AI ≈ 8 FLOP/Byte，而 A3 roofline 拐点在 150+。**权重必须整份读一遍，与 token 数无关。**

直接佐证：dec16/dec32/dec64 三档 token 差 4 倍，Duration 几乎不变（DSV3.2 1178/1213/1215us）。

各模型带宽利用率（master 基线）：

| 模型 | epr | 权重 | Duration | 带宽利用 | 空间 |
|---|---|---|---|---|---|
| DeepSeek-V4 | 16 | 2.11 GB | 1867~1893us | 94~96% | 已撞墙 |
| Kimi-K2.5 | 24 | 2.11 GB | 1724~1752us | 102~104% | 已撞墙 |
| DeepSeek-V3.2 | 16 | 1.41 GB | 1178~1215us | 98~101% | 已撞墙 |
| GLM-5 | 16 | 1.21 GB | 1009~1033us | 99~101% | 已撞墙 |
| **Qwen3-235B** | 8 | 0.30 GB | 320~355us | **72~80%** | **有** |

> **对上一轮优化的启示**：§1~§4 的双路径 epr 自适应 / 瓶颈1 直连 / enableUnitFlag 都是**流水与同步类**优化。
> 在 decode 场景实测（megamoe-dev vs master，14 用例）为 **0~3.2% 的微弱正收益，不劣化**，
> 但收益远小于 prefill 场景 —— 因为此时瓶颈已不在流水，而在 HBM 带宽。
> **decode 下唯一可回收的是那 85us 固定串行开销。**

## 8. 固定开销 85us 的实测分解

用 kernel 内已有的 `ExceptionDump::UpdateStage` 的 `GetSystemCycle` 打点（50MHz，`t_us = cycle/50`），
从 HCCL peermem 的 dump 区读出各阶段时间戳（probe_epr1：Duration 91.9us，其中权重搬运仅 5.3us）：

| 归类 | us | 占固定开销 | 可优化性 |
|---|---|---|---|
| GMM 流水固定截距（DISPATCH 9.0 + SWIGLU 9.3 + COMBINE 17.6） | 35.9 | 42% | 潜力最大但定位不明，不宜盲改 |
| HEAD（块派发梯度 6.5 + tiling 读 2.1 + initBuffer 5.3~6.5 + ADump 2.9~5.4 + routing 尾 3.4） | 21.8 | 25% | 部分可动 |
| 同步/串行（CrossRankSync 8.8 + ALLGATHER 6.7 + **CUMSUM 4.8** + RESET 1.2） | 21.5 | 25% | CUMSUM 可动 |
| UNPERMUTE 4.8 + kernel 外 2.1 | 6.9 | 8% | 不值得 |

## 9. 本轮采纳的优化

### 9.1 编译期关闭 ExceptionDump 阶段插桩

`UpdateStage()` 每次执行 `GetSystemCycle` + 2 次 GM 写 + `DataCacheCleanAndInvalid`，
且 `blockStage_` 落在 **HCCL 对称窗口**内，DCCI 刷 cacheline 比刷普通 HBM 更贵。
kernel 里 11 处调用，**无任何编译开关，恒开**。

改法：Policy 加 `kEnableStageTrace = false`，`UpdateStage` 用 `if constexpr` 短路成零指令。
保留 `Init`/`Dump`/`DumpTilingData` 的异常转储能力；用 detection idiom 保证未声明该常量的其它算子行为不变。

**实测**：固定开销 81.8 → 74.0us，**有效带宽 1164 → 1308 GB/s（+12.4%）**
（DCCI 不再干扰内存子系统，所以带宽也受益）。

### 9.2 CUMSUM 单核前缀和倍增化 + ExpertTokenNums 直出

`GetCumsumForMMAIV` 原实现 `for i in 1..EP { Add(row i, row i-1); PipeBarrier<PIPE_V>; }`
是 O(EP) 的真 RAW 依赖链，EP=16 → **15 次「小 Add + 全流水排空」**，每次只累加 `expertPerRank` 个 int32。
实测 4.8us（epr1/epr16 完全一致，纯固定开销），是全 kernel 性价比最低的一段。

改法：换 Hillis-Steele 倍增扫描，依赖链 O(EP) → O(log2 EP)，**PipeBarrier 15→4、向量指令 15→8**；
ping-pong 双 buffer 避免 dst/src 部分重叠。同时 `ExpertTokenNums` 直接从 UB 写出，
省掉 `SyncAll` 后的一趟 GM→UB→GM（含 4 对 flag）。

**实测**：probe_epr1 再降 4.9us（−6.1%）。

## 10. 本轮**被实测否决**的方案（负面结论，避免重复劳动）

| 方案 | 假设 | 实测 | 否决原因 |
|---|---|---|---|
| 移除 GMM1 每专家 `PipeBarrier<PIPE_ALL>` | 它排空 preload 预取、压低带宽（GMM2 与 A2 kernel 都没有它） | 探针 −2.9%~+4.5%，噪声内 | MTE2 已饱和，barrier 开销被掩盖 |
| unpermute 全 AIV 化（BF16 路径原只用一半） | 尾部裸露段可减半 | 探针 **+3.1%~+14.1%（劣化）** | UNPERMUTE 仅 4.9us，全核化最多省 2us，而多加的 `SyncAll` 全核栅栏更贵 |
| 缓存本 rank HCCL window 基址 | 省掉 Init/HEAD/ALLGATHER 路径十余次 GM 标量读 | 探针 **+10.9%（劣化）** | `HcclShmem` 被 combine epilogue **以拷贝方式持有**，加 2 个指针成员使热路径每次对象拷贝多搬 16B，放大后远超收益 |
| 优化 ALLGATHER 的跨 rank 自旋 | 它是最大的同步段 | — | 实测其中真正被 skew 浪费的只有 2.5~3us，且屏障本质是"等最慢的 rank"，挪走只会转移到 CrossRankSync |

> **块派发梯度 6.5us 不可优化**：用 kernel 第一条语句打点证实 b00→b39 单调爬坡 0.16us/块，
> 发生在 kernel 序言之前，属硬件/运行时行为。

## 11. 第二轮性能对比（14 个 ws16 decode 用例，×5 中位数）

基线 = master 0b0856d56；优化 = megamoe-dev + §9.1 + §9.2。**提升% 为正表示更快。**

| 用例 | master(us) | 优化(us) | 提升% |
|---|---|---|---|
| qwen3_ws16_dec32 | 333.3 | 317.5 | **+4.7%** |
| qwen3_ws16_dec64 | 355.1 | 339.0 | **+4.5%** |
| dsv3_ws16_dec32 | 1213.3 | 1191.0 | +1.8% |
| glm5_ws16_dec16 | 1008.9 | 991.4 | +1.7% |
| dsv3_ws16_dec16 | 1177.8 | 1165.0 | +1.1% |
| dsv3_ws16_dec64 | 1215.2 | 1203.1 | +1.0% |
| kimi_ws16_dec16 | 1724.1 | 1709.0 | +0.9% |
| glm5_ws16_dec64 | 1033.1 | 1024.0 | +0.9% |
| glm5_ws16_dec32 | 1019.9 | 1011.3 | +0.8% |
| kimi_ws16_dec32 | 1752.3 | 1738.2 | +0.8% |
| dsv4_ws16_dec64 | 1892.6 | 1877.1 | +0.8% |
| qwen3_ws16_dec16 | 320.1 | 317.5 | +0.8% |
| dsv4_ws16_dec32 | 1873.2 | 1867.0 | +0.3% |
| dsv4_ws16_dec16 | 1866.9 | 1863.3 | +0.2% |

**14/14 全部提升，无一劣化。** 固定开销占比越高的用例收益越大（Qwen3 固定开销占 27% → +4.7%；
DSV4 占 5% → +0.2%），与"削减固定开销"的机理一致。

精度回归 **6/6 PASS**（cos 0.999985~0.999995，与 master 逐位同量级），
其中特意覆盖了 `expertPerRank` 非 8 倍数的边界（coded_L3_ws16 epr=1、glm45_ws16 epr=10、dsv2_ws8 epr=20），
以验证 §9.2 倍增扫描读取 UB padding lane 的安全性。

## 12. 下一步可摘的果子（未做，供后续参考）

1. **GMM 流水固定截距 ~36us**（占固定开销 42%）：在 probe_epr1 上 DISPATCH+SWIGLU+COMBINE 墙钟 40.3us，
   而权重搬运仅 5.3us、GEMM 数学 <1us —— 约 35us 是流水填充/排空 + 逐专家 CrossCore flag 握手。
   建议用 `msprof` simulator trace 定位 AIC 到底在等什么，**不要盲改**。
2. **DISPATCH 内远端 rank window 指针的链式 GM 读**：`shmem(0, dstEpIdx)` 是两级解引用，
   EP=16/epr=16 时约 512 次标量 GM 读。注意 §10 的教训——若要缓存，务必避免增大被热路径拷贝的对象。
3. **kernel 侧 `initBuffer` 每核 5.3~6.5us**：offset 计算可移到 host tiling 预算（需改 TilingData 结构，本轮未做）。

## 13. 本轮新增的工具

| 文件 | 作用 |
|---|---|
| `perf_decode.py` | decode 场景性能采集：一次 fork 内重复 N 次取中位数（噪声 ±1%，优于 golden 单次的 ±5%）；抓全量 pipe 指标；自动算权重字节数/有效带宽/带宽利用率；`--probe` 跑受控探针回归固定开销 |
| `stage_trace.py` / `stage_report.py` / `stage_summary.py` | 从 HCCL peermem 的 ExceptionDump 区读出各阶段 `GetSystemCycle` 时间戳，做阶段级耗时分解 |

> **探针用例的价值**：`probe_epr1`（ws16/experts16/hidden1024/inter1024/topk1/tokens32）的 90us 里
> 权重搬运仅 5.3us，**固定开销占 94%**，相当于把灵敏度放大约 20 倍。
> 真实用例上被 ±1% 噪声淹没的改动，在这里能看出明确信号 —— 本轮三个被否决的方案都是靠它快速判定的。
