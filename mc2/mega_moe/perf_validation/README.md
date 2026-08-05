# mega_moe A3 decode 场景性能优化 — 收益说明与验证方法

> 分支：`megamoe-optimize`（基线 master `0b0856d56`）
> 环境：A3 / ascend910_93 / arch22，BF16 非量化，单 die **20 AIC / 40 AIV**
> 场景：单节点 **world_size=16 decode**（DeepSeek-V3.2/V4、Kimi-K2.5、GLM-5、Qwen3-235B）

本分支包含 **4 个 commit，每一个都是经实测验证有效的独立优化**，不含任何重构。

---

## 1. 收益汇总

### 1.1 逐场景收益（world_size=16，BF16 非量化）

| 模型 | rank 数 | 全局专家数 | **每 rank 专家数(epr)** | hidden | intermediate | topk | batch | 每卡权重 | master(us) | 优化后(us) | **提升** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DeepSeek-V3.2 | 16 | 256 | 16 | 7168 | 2048 | 8 | 16 | 1.41 GB | 1177.8 | 1164.2 | **+1.2%** |
| DeepSeek-V3.2 | 16 | 256 | 16 | 7168 | 2048 | 8 | 32 | 1.41 GB | 1213.3 | 1178.7 | **+2.8%** |
| DeepSeek-V3.2 | 16 | 256 | 16 | 7168 | 2048 | 8 | 64 | 1.41 GB | 1215.2 | 1193.2 | **+1.8%** |
| DeepSeek-V4 | 16 | 256 | 16 | 7168 | 3072 | 6 | 16 | 2.11 GB | 1866.9 | 1847.2 | **+1.1%** |
| DeepSeek-V4 | 16 | 256 | 16 | 7168 | 3072 | 6 | 32 | 2.11 GB | 1873.2 | 1867.6 | **+0.3%** |
| DeepSeek-V4 | 16 | 256 | 16 | 7168 | 3072 | 6 | 64 | 2.11 GB | 1892.6 | 1872.7 | **+1.0%** |
| Kimi-K2.5 | 16 | 384 | 24 | 7168 | 2048 | 8 | 16 | 2.11 GB | 1724.1 | 1713.8 | **+0.6%** |
| Kimi-K2.5 | 16 | 384 | 24 | 7168 | 2048 | 8 | 32 | 2.11 GB | 1752.3 | 1727.8 | **+1.4%** |
| GLM-5 | 16 | 256 | 16 | 6144 | 2048 | 8 | 16 | 1.21 GB | 1008.9 | 994.0 | **+1.5%** |
| GLM-5 | 16 | 256 | 16 | 6144 | 2048 | 8 | 32 | 1.21 GB | 1019.9 | 1003.0 | **+1.7%** |
| GLM-5 | 16 | 256 | 16 | 6144 | 2048 | 8 | 64 | 1.21 GB | 1033.1 | 1025.4 | **+0.7%** |
| **Qwen3-235B** | 16 | 128 | **8** | 4096 | 1536 | 8 | 16 | 0.30 GB | 320.1 | 310.9 | **+2.9%** |
| **Qwen3-235B** | 16 | 128 | **8** | 4096 | 1536 | 8 | 32 | 0.30 GB | 333.3 | 319.8 | **+4.1%** |
| **Qwen3-235B** | 16 | 128 | **8** | 4096 | 1536 | 8 | 64 | 0.30 GB | 355.1 | 329.7 | **+7.2%** |

**14/14 用例全部提升，无一劣化。** 指标为 `torch_npu.profiler` 的 `kernel_details.csv` 中 MegaMoe kernel
`Duration(us)`，rank0、×5 取中位数（噪声 ±1%）。

### 1.2 非 decode 大 batch 用例（不劣化验证）

| 用例 | rank 数 | epr | hidden | inter | batch | master(us) | 优化后(us) | 变化 |
|---|---|---|---|---|---|---|---|---|
| coded_L3_ws16 | 16 | 1 | 2048 | 1024 | 256 | 413.6 | 378.6 | +8.5% |
| qwen3_235b_ws16 | 16 | 8 | 4096 | 1536 | 1024 | 2293.5 | 2242.3 | +2.2% |
| kimi_k2_ws16 | 16 | 24 | 6144 | 2048 | 512 | 2729.7 | 2710.6 | +0.7% |
| dsv3_ws16 | 16 | 16 | 7168 | 2048 | 512 | 2805.3 | 2785.1 | +0.7% |
| llama4_mav_ws16 | 16 | 8 | 5120 | 2048 | 512 | 1718.7 | 1708.0 | +0.6% |
| glm45_ws16 | 16 | 10 | 8192 | 3072 | 512 | 5131.2 | 5122.6 | +0.2% |

**6/6 持平或提升。** 这批用例 token 数 256~1024/卡，`aic_mac_ratio` 达 0.57~0.65（**算力受限**），
与 decode 的带宽受限形态不同，故收益较小 —— 但不劣化。

### 1.3 为什么收益随模型不同

实测建立的性能模型（两条独立路径互相印证，4/5 模型拟合误差 <1.5%）：

```
Duration ≈ 权重字节数 / 1425 GB/s  +  74 us
             ↑ 物理下界                ↑ 固定串行开销
每卡权重字节数 = epr × (hidden × 2·inter + inter × hidden) × 2B
```

- **第一项不可压缩**：decode 下 BF16 的**算术强度 ≈ 每专家 token 数 ≈ 8 FLOP/Byte**，
  远低于 A3 roofline 拐点（150~250）→ 纯 HBM 权重搬运受限。
  profiling 佐证：`aic_mte2_ratio` 0.92~0.96（搬权重近乎满载）、`aic_mac_ratio` 仅 0.12~0.28（算力欠载）、
  `aiv_vec_ratio` 仅 0.002~0.026（AIV 占满全程却几乎不干活）。
  另一佐证：dec16/32/64 三档 token 数差 4 倍，Duration 几乎不变。
- **本分支 4 项优化削减的全部是第二项**（固定开销），故 **固定开销占比越高的模型收益越大**：

| 模型 | 固定开销占比 | 实测提升 |
|---|---|---|
| Qwen3-235B | **27%** | **+2.9~7.2%** |
| GLM-5 | 10.0% | +0.7~1.7% |
| DeepSeek-V3.2 | 8.6% | +1.2~2.8% |
| Kimi-K2.5 | 5.9% | +0.6~1.4% |
| DeepSeek-V4 | 5.4% | +0.3~1.1% |

---

## 2. 四个优化方案

### commit 1 — `enableUnitFlag`（`op_kernel/arch22/mega_moe.h:208`，改 1 行）

`constexpr bool enableUnitFlag = false → true`，经 `MmadDispatchPolicyFor_t` 传入 `BlockMmad`，
启用硬件 unit flag 使 `fixpipe(i) ‖ Mmad(i+1)` 并行。

**实测**：`aic_fixpipe_ratio` 0.004~0.013 → **0.042~0.073**（5~10 倍跃升，证实 fixpipe 已与 mmad 重叠），
11 个 decode 用例 +0.2~1.2%。编译期模板参数，不触及 flag 握手计数，无死锁风险。

### commit 2 — 编译期关闭 ExceptionDump 阶段插桩

`UpdateStage()` 在 kernel 里有 11 处调用，每次执行 `GetSystemCycle()` + 2 次 GM 写 +
`DataCacheCleanAndInvalid`，且 **无任何编译开关、恒开**。其 `blockStage_` 落在 **HCCL 对称窗口**内，
DCCI 刷 cacheline 到那里比刷普通 HBM 更贵，还会干扰内存子系统。

改法：Policy 加 `kEnableStageTrace = false`，`UpdateStage` 用 `if constexpr` 短路成零指令；
用 detection idiom 保证**未声明该常量的其它算子行为不变**（`mc2_exception_dump.h` 是 mc2 公共文件）。
保留 `Init`/`Dump`/`DumpTilingData` 的异常转储能力。

**实测**：固定开销 81.8 → **74.0us**；有效带宽 1164 → **1308 GB/s（+12.4%）**。

### commit 3 — CUMSUM 前缀和倍增化 + ExpertTokenNums 直出

`GetCumsumForMMAIV` 原实现是 `for i in 1..EP { Add(row i, row i-1); PipeBarrier<PIPE_V>; }`，
**O(EP) 真 RAW 依赖链** → EP=16 时 15 次「小 Add + 全流水排空」，每次只累加几十个 int32。
计算量可忽略，**代价全在流水排空**。

改法：Hillis-Steele 倍增扫描，依赖链 O(EP) → O(log₂EP)，**PipeBarrier 15→4、向量指令 15→8**；
ping-pong 双 buffer 避免 dst/src 部分重叠。顺带 `ExpertTokenNums` 直接从 UB 写出，
省掉 `SyncAll` 后的一趟 GM→UB→GM（含 4 对 flag）。

**实测**：受控探针 −4.9us（−6.1%）。阶段打点独立验证：CUMSUM 段 4.81 → **0.80us**。

### commit 4 — ExpertTokenCount 按专家区间分摊到所有 sort 核

`ComputeExpertTokenCountOrCumsum`（`moe_v2_init_routing_fullload.h`）原本**只由一个核**执行
（`blockIdx_ == sortNeedCoreNum_-1`），其余 36 核在 `SyncAll` 干等。
它逐 token 做标量 `GetValue`/`SetValue`（每次约 22ns 的 UB 往返），dec64 下耗时 **11.34us**。

改法：按**专家区间**分摊到所有 sort 核。零同步代价，靠两个结构事实：
① fullload 下 `CopyIn` 不带 blockIdx 偏移 → **每个核 UB 里已有完整排序数组**，分核统计不需额外搬运；
② 切的是**输出域**（专家区间）→ 各核 GM 切片互不重叠、**无需跨核归约**，调用侧本就有 `SyncAll`。
关键技巧：二分得到的 `tokenIdx` 同时就是该区间的起始累加值，**各核无需知道彼此结果**即可算出全局前缀和。

> **为什么不用向量化**：把已排序序列变直方图本质是 **scatter**，而 `Scatter` 原语 **A2/A3 均不支持**（仅 A5 有）。
> **为什么不用二分替代整个循环**：`expertNum = epr × world_size` = 全局 256，与 `totalLength`(256~512) 同量级
> → 二分需 `256 × log₂256 = 2048` 次，比线性扫 256 次**差 8 倍**。二分只用于定位区间起点（每核 1 次）。

**实测**：dsv3_dec64 **−9.4us（−0.8%）**、qwen3_dec64 −2.7%。
**正确性**：Python 逐行复刻改前/改后两版对拍 **52668 组用例**（覆盖 `expertNum >/=/< totalLength`、
非 8/64 倍数、9 种分布含全零/全同专家/大量空洞/极端倾斜），fails=0 且每个专家槽恰好写一次。

---

## 3. 验证方法

### 3.1 编译与安装

```bash
cd <REPO>/ops-transformer
source <CANN>/ascend-toolkit/set_env.sh
bash build.sh --pkg --soc=ascend910_93 --ops=mega_moe -j$(($(nproc)+1))
# 产物须约 20MB；22KB 说明 --soc 写错（必须小写）导致空包
ls -la build_out/cann-ops-transformer-custom_linux-aarch64.run
yes | ./build_out/cann-ops-transformer-custom_linux-aarch64.run --install-path=<VENDOR_DIR>
```

### 3.2 运行环境

```bash
source <CANN>/ascend-toolkit/set_env.sh
source <VENDOR_DIR>/vendors/custom_transformer/bin/set_env.bash
export MAX_JOBS=8 TORCH_EXTENSIONS_DIR=<某个独立目录>/.torch_extensions
```

### 3.3 精度验证

```bash
python3 golden_megamoe.py --list                    # 列出全部用例
python3 golden_megamoe.py --case dsv3_ws16_dec32    # 单用例
python3 golden_megamoe.py --all                     # 全量
```
16 个 rank 各自与 CPU golden 比对，输出 `cos` 与 `expert_token_nums` 一致性。
**通过标准**：`cos ≥ 0.99998` 且 `etn_ok=True`。

> **必测的边界用例**：commit 4 的倍增扫描会读 UB padding lane，而 5 个 decode 模型的 `epr` 恰好都是
> 8 的倍数（8/16/24），**这个风险点在 decode 用例里测不到**。必须覆盖 `epr` 非 8 倍数的用例：
> `coded_L3_ws16`(epr=1)、`glm45_ws16`(epr=10)、`dsv2_ws8`(epr=20)。

### 3.4 性能采集

```bash
python3 perf_decode.py --tag <版本名> --repeat 5              # 14 个 ws16 decode 用例
python3 perf_decode.py --tag <版本名> --case dsv3_ws16_dec32  # 单用例
python3 perf_decode.py --tag <版本名> --probe --repeat 7      # 受控探针（见下）
python3 perf_decode.py --tag <版本名> --filter ws16           # 含大 batch 用例
```

`perf_decode.py` 相对 golden 的改进：
- **一次 fork 内重复 N 次**取中位数 → 噪声由 ±5% 降到 **±1%**（import + HCCL 建链只付一次）
- 抓 `kernel_details.csv` **全量 pipe 指标**（`aic_mac/mte1/mte2/fixpipe_ratio`、`aiv_vec_ratio` 等）
- 自动计算权重字节数、有效带宽、带宽利用率
- **权重构造走 `make_inputs_fast`**：epr 个专家共享同一 CPU tensor（`.npu()` 后仍是地址独立的显存），
  避开 golden 那份 22.5GB 的随机权重构造 —— 单用例从数分钟降到 13~15 秒。
  权重内容不影响性能（搬运量只由 shape 决定，L2 按地址索引），但**精度验证必须用 `golden_megamoe.py`**。

### 3.5 受控探针（`--probe`）

固定 `hidden=1024, inter=1024, tokens=32, ws=16`，**只扫 epr**：

| 探针 | epr | 权重 | Duration | 固定开销占比 |
|---|---|---|---|---|
| probe_epr1 | 1 | 6.29 MB | ~74 us | **94%** |
| probe_epr16 | 16 | 100.7 MB | ~156 us | ~47% |

`probe_epr1` 的固定开销占 94%，**相当于把灵敏度放大约 20 倍** —— 真实用例上被 ±1% 噪声淹没的改动，
在这里能看出明确信号。对 Duration 做线性回归可分别得到「有效带宽」（斜率）与「固定开销」（截距）两个独立量。

> ⚠️ **探针在「相对比较」上可靠（同探针比两个版本），在「绝对量级归因」上不可靠**：
> 探针 h=1024 时 GMM1 只有 8 个 tile、GMM2 只有 4 个，而 AIC 有 20 个 → 并行度只有 20%/40%，
> 不能用它的绝对耗时去推断真实用例的阶段构成。

### 3.6 测量注意事项

- **交错复测**：验证小改动时用 `A→B→A→B` 交替跑，避免把环境漂移算进结论
- **离群值**：NPU 被其它任务抢占时会出现数倍的离群采样，`--repeat` 取中位数已可过滤
- **`--soc` 必须小写**：大写会产出 22KB 空包且**编译不报错**，只能靠包大小判断

---

## 4. 已否决方案备案（避免重复投入）

本轮共验证 10+ 个方案，以下**均已实测否决**，请勿重复尝试：

| 方案 | 实测结果 | 否决原因 |
|---|---|---|
| **双路径 epr 自适应 + 瓶颈1 直连** | qwen3 −0.9%/−1.1%/+2.5%（噪声带） | 收益不可辨；45~90 行改动涉 flag 握手对称（改错即死锁）；会让 epr≤8 的 **INT8 路径**也走 Path A 而 INT8 从未验证 |
| **L2 预取**（去掉 `SetL2CacheHint(CACHE_MODE_DISABLE)`） | **+9.2% 劣化** | decode 下 `currentM ≤ 128` 等价 `loopsM == 1` → swizzle 退化 `mIdx ≡ 0` → **B 的每个 N 切片一对一分给一个核，零跨核复用**，每个 B 元素只 load 一次。纯流式读进 L2 只会冲刷掉真正有复用的数据。**这判死了整条 L2 预取路线** |
| **提前发 flag0**（`SyncAll#2` 之前） | 未改代码即作废 | ① `CrossCoreSetFlag<0x2>` 是 **per-AI-Core 非全局**，且需配对两 AIV 都发 → 只 core0 发会永久阻塞；② 该 SyncAll **首要是 AIV 自己的 producer→consumer 屏障**（40 个 AIV 都读 core0 独写的 `cumsumMM`），删掉即**静默精度错误**；③ `pipe` 参数不提供跨核可见性 |
| **unpermute 全 AIV 化** | **+3~14% 劣化** | 该段仅 4.9us，全核化最多省 2us，而新增的一次全核 `SyncAll` 更贵（实测 **1.15us/次**） |
| **缓存 HCCL window 基址** | **+10.9% 劣化** | `HcclShmem` 被 combine epilogue **以拷贝方式持有**，加 2 个指针成员使热路径每次拷贝多搬 16B，放大后远超收益 |
| **`GetTensorAddr` 上提** | 噪声内 / +2.4% | 依据数字来自观测者效应 +11~12% 的构建版本，很可能是打点假象 |
| **移除 GMM1 每专家 `PipeBarrier`** | 噪声内 | MTE2 已饱和（ratio 0.93~0.96），气泡会被排队请求自动填上 |
| **末轮 skew 整平** | 名义 79us → 真实可回收 7.7us | **skew ≠ 可回收，高估 5~9 倍**；轮转已整平到理论最小值 ±1 块；最佳手段（GMM2 动态窃取）**会死锁**；`split-N` 实测 **+32.4%** |
| **SyncAll 合并** | 天花板仅 2.4~7.2us | 单次 `SyncAll` = 1.15us，但**位置决定成本**：流水内几乎免费（dsv3 插 128 次仅 +0.9us），只有头部满价。且 dispatch 的栅栏**结构上不可删**（生产者按源 rank 切、消费者是全部 AIC，集合无法对齐） |
| SwiGLU 增段 / L1Tile 调小 / N 256→512 / GMM1-GMM2 交错 | 均判负 | 详见分支外的 `PERF_ANALYSIS.md` |

**不可优化项**：块派发梯度（AIV 6.5us / AIC 7.4~8.8us）发生在 kernel 序言之前，属运行时行为；
核启动错峰 14.35us 同理。

### 已验证有效但需外部契约变更（未实施）

**B 矩阵 stride padding**：B 是 RowMajor，`Nd2Nz` 跨步读（每次读 256 列 = 512B，然后跳 stride = 4N 字节），
**stride 落进不利档位会导致 HBM bank 冲突**。

| 模型 | stride | GMM2 带宽 | ΔDuration | 显存 |
|---|---|---|---|---|
| dsv3 | 7168→**7680** | 63.2→69.0 | **−39.0us (−3.4%)** | **+7.1%** |
| dsv3 | 7168→8192 | 63.2→72.9 | −60.8us (−5.3%) | +14.3% |
| kimi | 7168→8192 | 61.0→69.6 | −56.3us (−3.3%) | +14.3% |
| glm5 | 6144→8192 | 67.2→73.7 | −33.5us (−3.4%) | +33.3% |
| qwen3 | 4096（已最优） | — | 无收益 | — |

**关键发现**：`Nd2Nz` 的 `dValue`（读多少列）与 `srcDValue`（行跨度）**相互独立**，
`layout::RowMajor` 有三参构造 `(rows, cols, ldm)` → **padding 一个字节都不多搬**，代价只有显存。
另：**"补到 2 的幂"不是规则** —— dsv3 补到 7680（非 2 的幂）即可拿到八成收益，
而 6656/7168 用在 glm5 上是负收益，必须逐 shape 实测。

**为什么算子内部做不了**：内部 padding 需把 470MB 权重拷到 workspace（读 470MB + 写 470MB = 940MB），
而 GMM2 全程才读 470MB —— **为省 60us 先付 660us，净亏 10 倍**。
必须由调用方按 padded leading dimension 分配 `weight2`（逻辑 shape 与接口签名均不变）。
