# 代码概要

算子: mega_moe (A3/910_93, arch22) | 功能: MoE 专家并行 dispatch→GMM1→swiglu→GMM2→combine/unpermute 通算融合；本次改动新增"权重 L2 预取"机制（重点：W-c 窗口 = GMM2 段界 V2C wait 前 AIC 侧 GM→L1 哑拷贝 W2） | 侧别: Kernel 侧（两文件均在 op_kernel/ 下，`__aicore__ inline` / `CATLASS_DEVICE` / `AscendC::` API，无任何 Tiling 侧代码）

> 检视背景聚焦：① W-c 窗口（GMM2 段界 W2 预取）功能正确性；② 两文件本次全部改动的编译正确性。本次改动 = HEAD 提交 eb0c4ea46（"A3 MegaMoe支持权重预取"）：weight_l2_prefetch.hpp 为新增文件（332 行），mega_moe_kernel_a3.hpp +156/-4（其余 +23 行在 block_mmad_w4a4.hpp / block_mmad_preload_async_fixpipe_quant.hpp 的 GetL1EndOffset，非本次 file_input）。

## 代码脉络

**入口**: `__global__ __aicore__ void mega_moe(...)` (op_kernel/arch22/mega_moe.cpp:34) → `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`（每 block 1 AIC + 2 AIV）→ `MegaMoe<...>::Init/Process`（A3 非 A2 路径，mega_moe.cpp:68-78）→ `MegaMoeImpl` 包装（mega_moe_arch22.h: `using ArchTag = Arch::AtlasA2`:209，组装 BlockMmad/Epilogue 模板实参并构造 `MegaMoeKernel::Params`，:295-413 `MatmulKernel kernel(params); kernel(params);`）→ `MegaMoeKernel::operator()<AIC>`(kernel_a3.hpp:259) / `operator()<AIV>`(:321) → 触发条件：框架按 tilingKey 实例化模板后 launch。

**数据流**:
GM 输入（x/topkIds/probs/weight1/weight2/scales）+ SHMEM peermem（跨卡 expandedX/D/tokenPerExpert）→ AIV: permute prologue（expandedX 写对端 peermem）→ AllGather 计数 + cumsum（GM workspace，chunk 级多份缓冲）→ 路由表刷 UB 顶部常驻缓存 → dispatch（远端 token 拉 GM workspace 的 A 区）→ AIC: GMM1（BlockMmad cube 流水，gmC）→ AIV: swiglu epilogue（gmA2I4/gmPermutedToken）→ AIC: GMM2（gmC2）→ AIV: combine（写回 peermem D 区）→ unpermute（加权汇合写 ptrOutput GM）。本次改动插入的旁路：权重 ptrB1/ptrB2 经 GM→L1/GM→UB 哑拷贝"路过 L2"预热，UB/L1 scratch 数据从不消费。

**计算核心**:
- AIC 侧主循环 = `operator()<AIC>` 的 chunk×round 双层循环（:267/:276）内 `GMM1`(:864，专家循环 groupIdx×blockScheduler loopIdx 双层) 与 `GMM2`(:1013，同构)；循环语义：groupIdx 遍历本卡 expertPerRank 个专家，loopIdx 按核交错遍历该专家 M×N block。
- AIV 侧主循环 = `DispatchAndCombine`(:1657) 的 chunk 循环 → `ProcessRecvRound`(:1422) 的 dispatch groupIdx 循环 + swiglu 分段循环。
- 本次改动核心 = `GMM2` 段界处（int4: :1089 `if (groupIdx == 0 || IsSyncTask(groupIdx - 1, epr))`；非 int4: :1108 第二 dequant 段界）在 `CrossCoreWaitFlag<0x2>(SYNCFLAGV2C...)`(:1105/:1118) 之前调用 `WeightL2PrefetchAicRange`(weight_l2_prefetch.hpp:278) 发射 W2 哑拷贝。

**分支覆盖**（仅列与本次改动及相关开关路径）:
| 分支条件 | 位置(文件:行) | 触发场景 | 处理逻辑 | 涉及 API |
|---------|-------------|---------|---------|----------|
| `WEIGHT_L2_PREFETCH_ENABLE_W2_GMM2_SECTION && groupIdx > 0U && w2ScratchFits`（int4 路径） | kernel_a3.hpp:1094 | GMM2 每个段界（groupIdx==0 或 IsSyncTask(groupIdx-1)）且非首段 | 计算 sectionEnd（从 groupIdx 起推进到下一 IsSyncTask 边界或 epr-1），发射 [groupIdx, sectionEnd+1) 的 W2 预取，随后 V2C wait | DataCopy(GM→L1)、CrossCoreWaitFlag |
| 同上（非 int4 路径） | kernel_a3.hpp:1111 | `groupIdx+1 == epr - lastDequantExpertNum`（第二 dequant 段界） | 一次性发射 [groupIdx, epr) 的 W2 预取，随后 V2C wait | 同上 |
| `kGmm2WeightPrefetched`（constexpr，当前 true） | kernel_a3.hpp:1069-1074 | 编译期 | 开启→`if constexpr(!true)` 整体裁剪，gmB2 保持默认 NORMAL L2 hint；关闭→恢复基线条件 `currentM<=L1TileShape::M` 时 SetL2CacheHint(DISABLE) | SetL2CacheHint |
| `kGmm1WeightPrefetched`（constexpr，当前 false） | kernel_a3.hpp:909-915 | 编译期 | W1 两条预取路径均关→保持基线：`currentM<=L1TileShape::M` 时 gmB1 SetL2CacheHint(DISABLE) | SetL2CacheHint |
| `WEIGHT_L2_PREFETCH_ENABLE_W1_AIC_COLDSTART`（constexpr false，休眠） | kernel_a3.hpp:284-294 → weight_l2_prefetch.hpp:318 | chunk0/round0 首个轮就绪 wait 前（当前编译为死代码，ColdStartW1 内部还有第二重 early-return:322） | BlockMmad l1EndProbe 探针取 GetL1EndOffset 对齐 16KB 为 L1 scratch，w1ScratchFits 守卫（L1_SIZE-256B）后发射 W1 前缀专家哑拷贝 | DataCopy(GM→L1)、CrossCoreWaitFlag |
| `WEIGHT_L2_PREFETCH_ENABLE_WINDOW0`（constexpr false，休眠，`if constexpr`） | kernel_a3.hpp:1706-1714（Init+Retarget）、:1244-1246（BurstAll 主发射，位于 AllGather push 与自旋两循环之间） | 编译期全裁剪 | 全体 AIV 交错认领 W1 头部专家固定区；在飞块由 :1278 PipeBarrier<PIPE_ALL> 与 :1295 SyncAll<true> 收口 | DataCopyPad(GM→UB)、PipeBarrier、SyncAll |
| `w2ScratchFits`（运行时） | kernel_a3.hpp:1044-1046 | `AlignUp(GetL1EndOffset,16KB)+16KB > L1_SIZE-256B`（理论不触发） | 段界发射整体跳过（性能退化非功能错误） | — |
| 段界判定 `IsSyncTask(task_id, n)` | kernel_a3.hpp:1321-1330 | `(n - task_id) > 0 且为 2 的幂` | int4 swiglu/GMM2 分段边界（与 ProcessRecvRound:1509 的 nSyncSwiglu 分段同源） | — |

**关键变量流转**:
| 变量 | 来源 | 用途 | 流转路径 |
|------|------|------|----------|
| WEIGHT_L2_PREFETCH_ENABLE_W2_GMM2_SECTION=true | 编译期常量 (weight_l2_prefetch.hpp:54) | W-c 窗口总开关 | 决定 kernel:1069 hint 裁剪与 :1094/:1111 发射守卫 |
| WEIGHT_L2_PREFETCH_W2_GMM2_CHUNKS=10 | 编译期常量 (:88) | 每专家每 AIC 核 chunk 配额上限 | GMM2:1102/:1115 → AicRange capChunks → limit=min(capChunks×coreNum×16KB, bytesPerExpert) |
| WEIGHT_L2_PREFETCH_CHUNK_BYTES=16384 | 编译期常量 (:105) | chunk 粒度（AIC GM→L1 固定 16KB；L1 scratch 环写块大小） | GMM2:1044 对齐 / AicRange:288,297-303 |
| WEIGHT_L2_PREFETCH_ARCH_RESERVED_BYTES=256 | 编译期常量 (:112) | 融合场景官方 L1 预留 | kernel:288/:1046 scratch 上界收紧 |
| w2ScratchOffset / w2ScratchFits | 运行时（blockMmad.GetL1EndOffset() 对齐） | L1 scratch 起始偏移与容量守卫 | GMM2:1044-1046 → GetBufferByByte<uint8_t>(w2ScratchOffset):1103/:1116 |
| PrefetchW2BytesPerExpert(problemShape) | 编译期 dtype 分支公式 (kernel:1386-1396) | 每专家 W2 字节（int4: (n/2)×k/2；int8: (n/2)×k；其他×2） | GMM2 → AicRange bytesPerExpert（区域尾钳制 + stacked 模式步进） |
| params.ptrB2 / listLen / expertPerRank | Tiling 传递（Params） | W2 权重 GM tensor-list、每专家独立(≠1)/堆叠(==1)模式、专家数 | GMM2:1100/:1112 → AicRange → GetTensorAddr |
| coreIdx / coreNum | 硬件配置（构造函数 :239-247；AIC=GetBlockIdx()/GetBlockNum()，AIV=get_block_idx()+get_subblockid()×get_block_num()） | 交错认领 (mod) 与核数配额 | GMM2 → AicRange coreIdx/coreNum |
| w0PrefetchCursor_ | 成员声明 :2229（无条件），仅 WINDOW0 开启时 Init(:1708) | AIV W1 固定区游标（当前休眠不构造） | DispatchAndCombine → AllGather BurstAll（全裁剪） |
| cumsumMM / chunkCumsumMM | workspace GM 指针（chunk 级多份） | 每专家 currentM 行数、轮数推导 | operator() :271-272 / GMM1 :885 / GMM2 :1049 |

**核心 API**: AscendC::DataCopy（GM→L1 哑拷贝, weight_l2_prefetch.hpp:303）、DataCopyPad（GM→UB 哑拷贝 :251）、CrossCoreSetFlag/CrossCoreWaitFlag<0x2>（C2V=9/V2C=10/syncgmm 0-8）、SyncAll<true>、PipeBarrier<PIPE_ALL/PIPE_V/PIPE_MTE3>、SetFlag/WaitFlag<HardEvent>（MTE2/MTE3/V/S 各类）、SetL2CacheHint、GetTensorAddr、LocalTensor::GetBufferByByte/SetSize、BlockMmad（blockMmad(...)、SynchronizeBlock、Finalize、GetL1EndOffset）、BlockEpilogue1/2/3、icache_preload(8)、gm_signal_wait_until_ne。

**输出**: GMM2 结果 gmC2 → combine 写 peermem D 区 → unpermute 加权写 params.ptrOutput（GM）；ExpertTokenNums 写 GM。同步机制：本类不走 TPipe EnQue/DeQue，核内用 HardEvent Set/WaitFlag + PipeBarrier，核间用 CrossCoreSetFlag/WaitFlag，全核 SyncAll<true>；预取哑拷贝本身无同步（fire-and-forget，靠 wait 窗口与队列程序序排空）。

## 算子业务语义（Kernel 侧）

**数学运算**: `Y[token] = Σ_{k∈topK} probs[token,k] · (W2ᵀ · act(W1ᵀ · x[token,k]))`，其中 GMM1 = X_perm(M×K)×W1(K×N)，swiglu = act 前半/后半门控（int4 路径还含 per-token dequant/quant int8→int4），GMM2 = X2(M×N/2)×W2(N/2×K)；**输入输出**: x/topkIds/topkWeights/weight1/weight2/weightScales1/2/(bias1/2 可选)/xActiveMask/maskBuffer → yOut/expertTokenNumsOut；W1/W2 是 B 矩阵（GEMM 右操作数），即本次预取对象。
**计算模式**: 模式 5（AIC-AIV MIX Cooperative：`ASCEND_IS_AIC`/`ASCEND_IS_AIV` + CrossCoreSetFlag/WaitFlag）× 模式 6（5-Stage Cube Pipeline：BlockMmad copyGmToL1B(MTE2)→LoadData(MTE1)→MMAD→Fixpipe）混合。
**同步契约**: C2V(flag 9) = AIC GMM 段完成→放行 AIV swiglu（int4 按 IsSyncTask 分段多次；非 int4 按 epilogueGranularity 两段）；V2C(flag 10) = AIV swiglu 段完成→放行 AIC GMM2（W-c 预取即发射于此 wait 之前）；syncgmm(flag 0-8) = dispatch 分组就绪 AIV→AIC；SyncAll<true> = AIV 全核段界；HardEvent/PipeBarrier = 核内 MTE2/MTE3/Vector/Scalar 交接。flag 槽位按 chunk×round 复用（PERMUTE_CHUNK_FLAG_STRIDE=0，kernel:66-69 注释约束 0-13 占用）。

### 分支业务含义
| 分支条件 | 位置(文件:行) | 业务含义 | 处理逻辑 |
|---------|-------------|---------|---------|
| int4 vs 非 int4 ElementB | kernel:306-311, 1088-1120 | W4A8 量化路径 vs int8/bf16 路径，GMM2 段界数量不同（多段 vs 两段） | int4 段界逐 IsSyncTask 边界预取；非 int4 只在第二 dequant 段界预取一次 |
| `params.listLen != 1` | weight_l2_prefetch.hpp:154-157, 292-295 | 权重组织方式：每专家独立 tensor（list）vs 单堆叠 tensor | 独立：GetTensorAddr(expertIdx)；堆叠：GetTensorAddr(0)+expertIdx×bytesPerExpert 步进（与 kernel gmGroupOffsetB :994/:1165 元素步进同源） |
| scratch 不适配（w2ScratchFits=false） | kernel:1044-1046 | L1 余量不足（理论不触发） | 段界预取整体跳过，保功能弃性能 |
| W1/W2 hint 联动 | kernel:906-915, 1066-1074 | 被预取权重必须 NORMAL（DISABLE 双向 bypass 使预取失效）；未预取侧保持基线 bypass | `if constexpr` 按窗口开关联动裁剪 |
| currentM 钳制 maxOutputSize | kernel:886-890, 1050-1054 | 单轮接收预算 B 内截断专家行数 | 空专家 currentM=0 仍参与 flag 推进（死预取接受，见 :314 注释） |

### 模板参数语义
| 参数 | 取值 | 业务含义 |
|------|------|---------|
| BlockMmad | mega_moe_arch22.h:259-262（isW4A8 选 QuantTileCopy 变体） | Cube 流水实现（本次 GetL1EndOffset 的提供者） |
| BlockEpilogue1/2/3 | mega_moe_arch22.h:266-290 | swiglu / combineV1 / combineV2 尾处理 |
| ElementB（类模板） | int4b_t / int8_t / bf16(f16) | 权重 dtype，决定 PrefetchW1/W2BytesPerExpert 公式分支与 GMM2 段界结构 |
| ElementABefore | int8_t（当 ElementA 为 int4b_t） | dispatch 阶段量化前元素类型 |

## Tiling 业务语义（Tiling 侧）

> 本文件不含 Tiling 代码；此处记录被 Kernel 消费的 Tiling 字段语义（来源 op_host/op_tiling/arch22/mega_moe_tiling_arch22.cpp，经 mega_moe_arch22.h:174/178/193/197 读 tilingData.common.* 转入 Params）。

**切分策略**: 按 chunk（PERMUTE_CHUNK=1024 token 发送侧）× 接收轮（R=Σcumsum 行数，轮数=ceil(R/B)，B=maxOutputSize）× 专家 groupIdx × blockScheduler M×N block；**Buffer策略**: BlockMmad L1/L0 双/多 stage（mega_moe_arch22.h:240-244 preloadStages=1, l1Stages=2, l0A/B=2, l0C=1），epilogue ubStages=2。
**TilingKey 轴**: `(QuantMode == MEGA_MOE_QUANT_MODE_PER_TOKEN)` → MegaMoeTilingDataQuant vs NonQuant（mega_moe.cpp:45-46）。

### 校验策略（host 侧，与本次改动强相关）
| 校验条件 | 位置(文件:行) | 数学不变量 |
|---------|-------------|-----------|
| dim1 (hidden_size, K) 512 对齐 | mega_moe_tiling_arch22.cpp:703 | 保证 PrefetchW1/W2BytesPerExpert 的乘除结果 32B 对齐（DataCopy/L1 块粒度安全，kernel:1370 注释据此） |
| dim0 (intermediate_hidden, N) 512 对齐且 >=512 | mega_moe_tiling_arch22.cpp:953, 959 | 同上；n/2 半宽至少 512 |
| weight1 dtype ∈ {BF16, INT8, INT4}、ND/NZ 格式 | mega_moe_tiling_arch22.cpp:864-873 | Prefetch 字节公式三分支完备性 |
| weight1 每 tensor 2D 且 dim0==hidden_size | mega_moe_tiling_arch22.cpp:884-894 | bytesPerExpert 一致性（每专家同形） |
| expertPerRank = GetDynamicInputTensorListLen(WEIGHT1_INDEX) | mega_moe_tiling_arch22.cpp:878, 1218-1220 | listLen==expertPerRank（per-expert 模式）；listLen==1 时为堆叠模式 |

### 切分变量语义
| 变量 | 公式 | 业务含义 |
|------|------|---------|
| maxOutputSize (B) | host 计算（bs/ep/topK 相关，:526 附近自动推导 maxRecvTokenNum） | 单轮接收预算；GMM currentM 钳制上限 |
| recvRoundsMax | host 下发（kernel:134 默认 1） | 轮表份数上限（workspace 分配同步） |
| gmm1InputExpertSlotWidth | `max(K, N/2)`（Params ctor kernel:230-231） | dispatch/swiglu 复用 A 区的专家槽宽（防前专家 K2 宽输出覆盖后专家未消费输入） |
| gmmOutPreRowStride | `max(N, K)`（:229） | gmC 行步长（D 区复用） |

### TilingKey 语义
| 轴 | 取值 | 业务含义 |
|----|------|---------|
| QuantMode | PER_TOKEN / NO_QUANT | 量化路由（int8/int4 数据通路 + tiling 结构体选择） |
| TPL_ARCH | SOC_ASCEND910B(else) | A2 走 MegaMoeA2 旧 kernel，A3(910_93) 走本 MegaMoeKernel |

## 变量溯源

| 变量 | 声明(文件:行) | 初始化(文件:行) | 校验(文件:行) | 来源类型 |
|------|-------------|----------------|-------------|---------|
| w0PrefetchCursor_ | kernel_a3.hpp:2229（WeightL2PrefetchCursor，默认 active=false） | 仅 WINDOW0 开启时 kernel:1708-1713 Init+Retarget（当前编译期裁剪） | WeightL2PrefetchInit 下限判定 weight_l2_prefetch.hpp:222-223（chunkBytes>=2048 否则 inactive） | 编译期常量开关 |
| coreIdx/coreNum | kernel_a3.hpp:2189-2190 | ctor :239-247（AIC: GetBlockIdx/GetBlockNum；AIV: get_block_idx+get_subblockid×get_block_num 等） | 硬件保证 | 硬件配置 |
| syncgmmIdx | kernel_a3.hpp:2194（uint16_t） | :264 归零，每轮 :312 归零 | 注释契约 :66-69（flag 槽位约束） | Kernel 内部状态 |
| maskBufferAddr/rankMaskLocal | kernel_a3.hpp:2219-2220 | initBuffer :376-383（CopyRankMaskToLocal MTE2_S 收口 :421-422） | nullptr 可选（IsRankMasked :426-432） | 外部输入（可选） |
| params.listLen | kernel_a3.hpp:128 | mega_moe_arch22.h:178/197 ← tilingData.common.listLen ← host tiling_arch22.cpp:1220 (=expertPerRank) | host :878 动态列表长度 | TilingData 传递（已校验） |
| params.expertPerRank | kernel_a3.hpp:129 | mega_moe_arch22.h:174/193 ← host :878/:1218 | host 校验 weight 每 tensor 同形 :884-894 | TilingData 传递（已校验） |
| params.problemShape.k/n | kernel_a3.hpp:105 | mega_moe_arch22.h:298-305 ← tiling | host :703/:953/:959 强制 512 对齐 | TilingData 传递（已校验） |
| WEIGHT_L2_PREFETCH_* 全部常量 | weight_l2_prefetch.hpp:49-112 | constexpr 编译期 | static_assert :114（32B 对齐）、:121（UB 余量） | 编译期常量 |
| w2ScratchOffset/w2ScratchFits | kernel_a3.hpp:1044-1046（局部） | blockMmad.GetL1EndOffset()（block_mmad_w4a4.hpp:128 / block_mmad_preload_async_fixpipe_quant.hpp:135，InitL1 真实占用末偏移） | `<= ArchTag::L1_SIZE - 256` 运行时守卫 | 硬件配置 + 编译期常量 |

> 来源类型判定：TilingData 传递 → 已校验；硬件配置 → 硬件保证；编译期常量 → 编译期固定。

## 函数清单

### weight_l2_prefetch.hpp（全部为自由函数 `__aicore__ inline`）
| 函数 | 签名 | 行范围 | 角色 |
|------|------|--------|------|
| WeightL2PrefetchSetupRegion | void (WeightL2PrefetchCursor&, uint64_t skipInRegion) | 146-166 | 辅助：regionIndex→(tensor,专家基址,区域字节)，定位本 worker 起 chunk |
| WeightL2PrefetchAdvanceRegion | void (WeightL2PrefetchCursor&) | 169-173 | 辅助：推进下一专家 |
| WeightL2PrefetchRetarget | void (WeightL2PrefetchCursor&, uint32_t firstExpert, uint32_t lastExpert) | 180-198 | 辅助：重定位专家区间（越界钳制到 regionCount） |
| WeightL2PrefetchInit | void (cursor&, GM_ADDR, bool isPerExpert, uint64_t bytesPerExpert, uint32_t expertCount, uint32_t workerIdx, uint32_t workerCount, uint64_t budgetBytes, LocalTensor<uint8_t> scratch, uint32_t chunkBytes=16KB) | 206-231 | 初始化：AIV 游标（当前裁剪未用） |
| WeightL2PrefetchTryIssue | bool (WeightL2PrefetchCursor&) | 238-259 | 计算核心（AIV 路径）：发一个 chunk GM→UB DataCopyPad |
| WeightL2PrefetchBurstAll | void (WeightL2PrefetchCursor&) | 263-267 | 辅助：循环 TryIssue 到耗尽（窗口0主发射，当前裁剪） |
| WeightL2PrefetchAicRange | void (GM_ADDR, bool, uint64_t bytesPerExpert, uint32_t firstExpert, uint32_t lastExpert, uint32_t capChunks, uint32_t coreIdx, uint32_t coreNum, LocalTensor<uint8_t> l1Scratch) | 278-306 | **计算核心（W-c 窗口发射函数）**：[first,last) 逐专家 GM→L1 DataCopy 哑拷贝，(coreIdx,coreNum) mod 交错认领 16KB chunk，limit=min(capChunks×coreNum×16KB, bytesPerExpert)，尾块按剩余截断，环写同一 L1 scratch |
| WeightL2PrefetchAicColdStartW1 | void (GM_ADDR, bool, uint64_t, uint32_t expertPerRank, uint32_t coreIdx, uint32_t coreNum, LocalTensor<uint8_t>) | 318-330 | 辅助：W1 冷启动封装（常量 false 双重守卫，休眠） |

### mega_moe_kernel_a3.hpp（class MegaMoeKernel，★=本次改动涉及）
| 函数 | 签名 | 行范围 | 角色 |
|------|------|--------|------|
| Params（结构+2 ctor） | — | 103-233 | 数据结构：全量 GM 指针/布局/EP/listLen/topK 等 |
| MegaMoeKernel | (Params const&) | 236-250 | 初始化：coreIdx/coreNum + initBuffer |
| ~MegaMoeKernel | — | 252-253 | 析构 |
| operator()<AIC> | void (Params const&) | 258-318 | **AIC 入口**：chunk×round 循环；★:284-294 W1 冷启动（休眠）；:297 轮就绪 wait；:305 GMM1；:307 V2C wait（非 int4）；:308/310 GMM2 |
| operator()<AIV> | void (Params const&) | 320-327 | **AIV 入口**：rankMask 早退 → DispatchAndCombine |
| RuntimeRank | int32_t (Params const&) const | 330-335 | 辅助：shmem.Rank() |
| initBuffer | void (Params const&) | 337-409 | 初始化：workspace/peermem/GT 绑定、UB 顶部 rankMask+routeCache 常驻区（:373-400，与 ComputeAivPrefetchHoleBytes 镜像） |
| CopyRankMaskToLocal | void (Params const&) | 411-423 | 辅助：mask GM→UB + MTE2_S 收口 |
| IsRankMasked | bool (int32_t) | 425-432 | 辅助：弹性扩缩容屏蔽判定 |
| CopyGMToGM | template<T> void (GT dst, GT src, int32_t, int32_t) | 434-476 | 辅助：GM→GM 中转（双 32KB UB ping-pong） |
| CopyGMToGMPerToken | template<T> void (…, int32_t&) | 478-520 | 辅助：quant 路径逐 token GM→GM（行尾 scale 拆出） |
| FetchAndPreprocessInt8ToInt4 | void (uint64_t, GT, GT, int32_t, int32_t) | 522-657 | 计算核心：int8→int4 量化预处理（高/低 nibble 两级 Cast） |
| GetCumsumForMMAIV | void (GT&, GT&, uint32_t×3) | 659-689 | 计算核心：跨 rank tokenPerExpert cumsum（core0） |
| CacheRouteTablesToUb | void (Params const&) | 694-713 | 辅助：路由表 GM→UB 常驻缓存 |
| NumChunks | static int64_t (int64_t) | 716-719 | 辅助：发送 chunk 数 |
| ChunkCumsumPtr | GM_ADDR (Params const&, int64_t) const | 722-726 | 辅助：chunk 级 cumsum 基址 |
| ComputeNumRecvRounds | int64_t (Params const&) | 732-741 | 辅助：ceil(R/B) 轮数推导 |
| RoundTablePtr | GM_ADDR (…, RoundTableSeg) const | 754-763 | 辅助：轮表基址 |
| BuildRoundTables | void (Params const&, int64_t, int64_t) | 771-829 | 计算核心：轮表生成（core0） |
| SetRoundTableBuffers | void (Params const&, int64_t, int64_t) | 833-846 | 辅助：cumsumMM/preSum 切轮表 |
| CacheRoundTablesToUb | void () | 850-862 | 辅助：轮表刷 UB |
| GMM1 | void (Params const&, int64_t) | 864-1011 | **计算核心**：专家循环 cube GEMM；★:906-915 W1 hint 联动 |
| GMM2 | void (Params const&, int64_t) | 1013-1182 | **计算核心**：专家循环 cube GEMM；★:1041-1046 W2 L1 scratch；★:1066-1074 W2 hint 联动；★:1090-1105 段界 W2 预取（int4）；★:1110-1118 段界 W2 预取（非 int4） |
| CalculateTaskInfoEachCore | void (uint32_t&, uint32_t&, uint32_t) | 1184-1199 | 辅助：均分任务 |
| AllGatherTokenPerExpert | void (Params const&, int64_t) | 1201-1296 | **计算核心**：计数 push→★:1244-1246 窗口0主发射（休眠）→自旋→规范化；SyncAll 收口 :1295 |
| ResetTokenPerExpert | void (int32_t) | 1298-1319 | 辅助：peermem 计数清零 |
| IsSyncTask | bool (int32_t, int32_t) | 1321-1330 | 辅助：段界判定（(n-id) 为 2 的幂） |
| CombineSetFlag / CombineWaitFlag | void () | 1332-1348 / 1350-1366 | 辅助：combine 前后 HardEvent 配对 |
| ★PrefetchW1BytesPerExpert | static uint64_t (GemmCoord const&) | 1374-1384 | 辅助（新增）：W1 每专家字节（int4/int8/2B 三分支） |
| ★PrefetchW2BytesPerExpert | static uint64_t (GemmCoord const&) | 1386-1396 | 辅助（新增）：W2 每专家字节（同上） |
| ★ComputeAivPrefetchHoleBytes | uint32_t (Params const&) const | 1404-1415 | 辅助（新增）：AIV UB 洞宽（镜像 initBuffer :378-394 布局算术，钳 16KB、向下 32B 对齐） |
| ProcessRecvRound | void (Params const&, int64_t) | 1422-1655 | **计算核心**：dispatch→swiglu 分段→combine（AIV 侧轮内流水）；V2C SetFlag :1586/:1611/:1638 |
| DispatchAndCombine | void (Params const&) | 1657-1884 | **计算核心**（AIV 主循环）：★:1706-1714 窗口0游标构造（休眠）；prologue/AllGather/cumsum/轮循环/unpermute |
| CombineV1 | void (Params const&, BlockEpilogue2&) | 1886-1933 | 计算核心：int8 大 M combine |
| CombineV2 | void (Params const&, BlockEpilogue3&) | 1935-2034 | 计算核心：W4A8/int8 小 M/bf16 combine |
| WorkspaceInfo / PeermemInfo | 结构体 | 2037-2127 / 2129-2185 | 数据结构：workspace/peermem 偏移布局算术 |

## 调用关系图

### weight_l2_prefetch.hpp
| 函数 | 调用者 | 调用者计数 | 无外部调用者? | 重复调用链? |
|------|--------|-----------|--------------|------------|
| WeightL2PrefetchSetupRegion | AdvanceRegion:172; Retarget:197; Init:230 | 3 | 否 | 空 |
| WeightL2PrefetchAdvanceRegion | TryIssue:242 | 1 | 否 | 空 |
| WeightL2PrefetchRetarget | kernel DispatchAndCombine:1713（if constexpr WINDOW0=false，编译期裁剪） | 1（当前裁剪为 0 活调用） | 否 | 空 |
| WeightL2PrefetchInit | kernel DispatchAndCombine:1708（同上裁剪） | 1（裁剪） | 否 | 空 |
| WeightL2PrefetchTryIssue | BurstAll:265 | 1 | 否 | 空 |
| WeightL2PrefetchBurstAll | kernel AllGatherTokenPerExpert:1245（if constexpr 裁剪） | 1（裁剪） | 否 | 空 |
| WeightL2PrefetchAicRange | ColdStartW1:328（开关 false 死代码）; kernel GMM2:1100（int4 段界，**活**）; kernel GMM2:1112（非 int4 段界，**活**） | 3（活 2） | 否 | GMM2 int4/非 int4 两互斥分支各 1 次 |
| WeightL2PrefetchAicColdStartW1 | kernel operator()<AIC>:289（常量 false 守卫，死代码） | 1（裁剪） | 否 | 空 |

### mega_moe_kernel_a3.hpp（关键函数）
| 函数 | 调用者 | 调用者计数 | 无外部调用者? | 重复调用链? |
|------|--------|-----------|--------------|------------|
| operator()<AIC>/<AIV> | mega_moe_arch22.h:413 `kernel(params)`（launch 包装） | 1 | 否（Kernel 入口白名单） | 空 |
| GMM1 | operator()<AIC>:305 | 1 | 否 | 空 |
| GMM2 | operator()<AIC>:308（非 int4）/ :310（int4） | 2 | 否 | int4/非 int4 互斥分支各 1 次 |
| PrefetchW2BytesPerExpert | GMM2:1101（int4，活）; GMM2:1113（非 int4，活） | 2 | 否 | 两互斥分支各 1 次 |
| PrefetchW1BytesPerExpert | operator()<AIC>:291（W1 冷启动，裁剪）; DispatchAndCombine:1710（窗口0 Init，裁剪） | 2（均裁剪） | 否 | 空 |
| ComputeAivPrefetchHoleBytes | DispatchAndCombine:1707（裁剪） | 1（裁剪） | 否 | 空 |
| IsSyncTask | GMM1:969; GMM2:1089; GMM2:1097（sectionEnd 推进，★新增）; ProcessRecvRound:1509 | 4 | 否 | 空 |
| AllGatherTokenPerExpert | DispatchAndCombine:1759 | 1 | 否 | 空 |
| ProcessRecvRound | DispatchAndCombine:1805 | 1 | 否 | 空 |
| DispatchAndCombine | operator()<AIV>:326 | 1 | 否 | 空 |

## API 调用索引

### weight_l2_prefetch.hpp
| API | 行号 | 上下文 |
|-----|------|--------|
| GetTensorAddr<uint8_t> | 154 | SetupRegion：per-expert(list) 模式按 expertIdx 解引用，stacked 模式取 0 |
| GlobalTensor::SetGlobalBuffer | 250 | TryIssue：以 chunkAddr 建 GT |
| DataCopyPad | 251 | TryIssue：GM→UB fire-and-forget，`{1, (uint16_t)copyBytes, 0, 0}`（copyBytes ≤ 16KB，块粒度对齐） |
| LocalTensor::SetSize | 221 / 288 | Init 设 chunkBytes / AicRange 设 16KB |
| GetTensorAddr<uint8_t> | 292 | AicRange：与 154 同构 |
| DataCopy | 303 | **AicRange：GM→L1 哑拷贝 chunkBytes（≤16KB，W-c 窗口核心发射）** |

### mega_moe_kernel_a3.hpp（★=本次改动；其余列同步/搬运关键点）
| API | 行号 | 上下文 |
|-----|------|--------|
| GetBlockIdx/GetBlockNum（AIC）| 240-241 | ctor：coreIdx/coreNum |
| get_block_idx/get_subblockid/get_block_num（AIV） | 245-246 | ctor：逻辑核索引 |
| CrossCoreWaitFlag<0x2> | 297 | 轮就绪 wait（GMM1 前） |
| ★BlockMmad::GetL1EndOffset | 286 / 1044 | W1 冷启动 / W2 段界 L1 scratch 偏移 |
| ★LocalTensor::GetBufferByByte<uint8_t> | 292 / 1103 / 1116 / 1712 | L1/UB scratch 获取 |
| ★WeightL2PrefetchAicRange（见上） | 1100 / 1112 | GMM2 段界 W2 发射（int4 / 非 int4） |
| ★SetL2CacheHint(CACHE_MODE_DISABLE) | 913 / 1072 | gmB1（当前编译保留，条件执行）/ gmB2（当前编译裁剪，恒 NORMAL） |
| CrossCoreWaitFlag<0x2>(V2C) | 307 / 1105 / 1118 | GMM2 前置 wait（非 int4 外层 / int4 段界 / 非 int4 第二段界） |
| CrossCoreWaitFlag<0x2>(syncgmm) | 929 / 1001 | GMM1 分组就绪握手 |
| CrossCoreSetFlag<0x2, PIPE_MTE3> | 1431 / 1497 / 1586 / 1611 / 1638 | 轮就绪 / dispatch 分组就绪 / V2C（swiglu 段完成） |
| CrossCoreWaitFlag<0x2>(C2V) | 1568 / 1589 / 1614 | AIV 等 GMM 段完成 |
| SyncAll<true> | 1295 / 1496 / 1569 / 1585 / 1610 / 1636 / 1654 / 1755 / 1764 / 1788 / 1801 / 1880 / 1898 | AIV 全核屏障（窗口0在飞块收口之一 :1295） |
| PipeBarrier<PIPE_ALL> | 903 / 1278 | GMM1 前（非 int4）/ AllGather 自旋循环内（窗口0在飞块收口 :1278） |
| DataCopyPad | 419 / 515 / 517 / 609 / 669 / 686 / 702 / 708 / 819-825 / 855-857 / 1290 | rankMask / per-token 拆分 / cumsum / 路由表 / 轮表 |
| DataCopy | 504 / 604 / 622 / 645 / 1261 / 1272 / 1274 / 1315 / 1689 / 1776-1783 | 量化搬运 / 计数规范化 / 清零 / ExpertTokenNums |
| SetFlag/WaitFlag（HardEvent MTE2_S/MTE3_MTE2/MTE2_V/V_MTE3/MTE2_MTE3/S_MTE3/MTE3_V/V_MTE2/M_MTE1/FIX_*） | 421-422 / 438-475 / 563-656 / 674-684 / 711-712 / 813-814 / 860-861 / 1225-1234 / 1306-1317 / 1690-1691 / 1778-1785 等 | 核内 MTE/Vector/Scalar 阶段交接 |
| gm_signal_wait_until_ne | 1258 | AllGather 自旋等对端计数标记 |
| icache_preload(8) | 867 / 1016 / 1443 / 1660 / 1894 / 1959 | 指令缓存预热 |
| blockMmad(...)（BlockMmad::operator()） | 947-963 / 1141-1154 | cube GEMM 发射 |
| BlockMmad::Finalize / SynchronizeBlock | 971-982 / 1005-1010 / 1171-1181 | 段完成置 flag |
| BlockEpilogue1/2/3::operator() / Finalize | 1578-1607 / 1626-1634 / 1925-1933 / 2005-2013 | swiglu / combine |
| KernelMoeTokenUnpermute::Init/Process | 1855-1859 / 1872-1876 | unpermute |

## 常量清单

### weight_l2_prefetch.hpp（★=本次检视主对象）
| 常量 | 值 | 位置(行) | 用途 |
|------|-----|---------|------|
| WEIGHT_L2_PREFETCH_ENABLE_WINDOW0 | false | 49 | 窗口0（AIV W1 固定区）总开关，if constexpr 全裁剪 |
| WEIGHT_L2_PREFETCH_ENABLE_W1_AIC_COLDSTART | false | 51 | W1 AIC 冷启动开关（休眠） |
| WEIGHT_L2_PREFETCH_ENABLE_W2_GMM2_SECTION | true | 54 | **W-c 窗口总开关（当前唯一开启）** |
| WEIGHT_L2_PREFETCH_W0_EXPERTS | 2U | 68 | 窗口0 固定区头部专家数（休眠） |
| WEIGHT_L2_PREFETCH_W0_BUDGET_BYTES | 16MiB | 71 | 窗口0 全局字节预算（休眠） |
| WEIGHT_L2_PREFETCH_W0_AIC_COLDSTART_EXPERTS | 4U | 81 | W1 冷启动专家上限（休眠） |
| WEIGHT_L2_PREFETCH_W0_AIC_COLDSTART_CHUNKS | 64U | 82 | W1 冷启动 chunk 上限（休眠） |
| WEIGHT_L2_PREFETCH_W2_GMM2_CHUNKS | 10U | 88 | **W2 段界每专家每 AIC 核 chunk 上限**（注释公式 ceil(bytesPerExpert/(coreNum×16KB))，按 24 AIC 核 int8 3.5MB/专家推导=10，尾块按专家字节截断即精确全覆盖） |
| WEIGHT_L2_PREFETCH_UB_OFFSET | 179552 | 104 | AIV scratch 下界（swiglu BUFFER_SIZE 146784 + MIN_SHARED_TMP 32768；文件头注释给出各阶段功能区上界合约 dispatch<=75K/swiglu<=179552/combine<=150528/unpermute≈10K+7.5KB） |
| WEIGHT_L2_PREFETCH_CHUNK_BYTES | 16384 | 105 | chunk 粒度（AIC GM→L1 固定；L1 scratch 大小；对齐单位） |
| WEIGHT_L2_PREFETCH_AIV_MIN_CHUNK_BYTES | 2048 | 107 | AIV 洞最小 chunk（不足则 inactive） |
| WEIGHT_L2_PREFETCH_ARCH_RESERVED_BYTES | 256 | 112 | 融合场景官方 L1 预留（L1 scratch 上界收紧用；UB 侧不收紧的理由见 :108-111 注释） |
| static_assert | UB_OFFSET%32==0 / OFFSET+16KB+512<=UB_SIZE | 114 / 121 | 静态兜底（实际防重叠靠运行时洞宽） |

### mega_moe_kernel_a3.hpp
| 常量 | 值 | 位置(行) | 用途 |
|------|-----|---------|------|
| SYNCFLAGC2V / SYNCFLAGV2C | 9 / 10 | 59-60 | C2V/V2C 跨核 flag |
| UB_MOVE_NUM | 16384 | 61 | CopyGMToGM 单次搬运元素 |
| PERMUTE_CHUNK | 1024 | 64 | 发送 chunk 大小 |
| PERMUTE_CHUNK_FLAG_STRIDE | 0 | 69 | flag 按 chunk 复用（不偏移） |
| PERMUTE_MAX_CORES | 128 | 71 | prologue 计数区核数上限 |
| ROUND_TABLE_SEGMENTS | 3 | 750 | 轮表每份段数 |
| kRoutingIsQuant（静态） | — | 2233-2234 | dump 引擎量化路由选择 |
| ALIGN_128/ALIGN_512/CACHE_LINE/kMaxDequantSyncGroups 等 | 128/512/512/16 | utils/const_args.hpp:23-32 | 对齐/缓存行/段数上界（const_args.hpp 内 constexpr） |

## 跨文件防御摘要

| 关联文件 | 关键发现 | 位置(文件:行) | 影响范围 |
|---------|---------|-------------|---------|
| utils/block_mmad_w4a4.hpp | `GetL1EndOffset()` 返回 InitL1 真实 L1 末偏移（注释明确供权重预取 scratch 用，哑拷贝 MTE2 与 LoadData MTE1 跨队列无碰撞） | block_mmad_w4a4.hpp:125-132 | kernel:286/:1044 scratch 偏移来源（W4A8 变体） |
| utils/block_mmad_preload_async_fixpipe_quant.hpp | 同上定义 | :132-139 | 同上（int8/bf16 变体） |
| utils/get_tensor_addr.hpp | `GetTensorAddr<T>(index, ptr)` 解 GM tensor-list（首 u64 为偏移，再按 index 取项）；weight_l2_prefetch 与 kernel gmB1/gmB2/S 均用它，两处步进语义一致（stacked=0+字节步进） | get_tensor_addr.hpp:22-30 | 预取地址计算与 GEMM B 矩阵同源 |
| utils/const_args.hpp | SYNCFLAGC2V/V2C(9/10) 重复定义（kernel:59-60 亦定义）；ALIGN_128/512、CACHE_LINE=512、kMaxDequantSyncGroups=16 | const_args.hpp:23-32 | flag 槽位与对齐常量唯一来源（kernel 内同名 constexpr 与之数值一致） |
| mega_moe_arch22.h | 宿主包装：ArchTag=Arch::AtlasA2(:209)；tilingData.common.expertPerRank/listLen 读入(:174/:178)；Params 组装(:295-413)；epilogueGranularity 按 dtype 推导(:307-320，int4=0/int8=epr-3 或 epr-1/bf16=epr-2) | mega_moe_arch22.h | GMM2 非 int4 段界位置（lastDequantExpertNum）由此决定 |
| op_kernel/arch22/mega_moe.cpp | `__global__` 入口；KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)（1 AIC+2 AIV/block） | mega_moe.cpp:34,41 | coreIdx/coreNum 语义、CombineV2 aicCoreNum=coreNum/2 |
| op_host/op_tiling/arch22/mega_moe_tiling_arch22.cpp | K 512 对齐(:703)、N 512 对齐且>=512(:953/:959)——预取字节公式除法/32B 对齐的前提；info.listLen=expertPerRank(:1220)；aivNum=GetCoreNumAiv(:1236)、aicNum=GetCoreNumAic(:1421) | tiling_arch22.cpp | PrefetchW1/W2BytesPerExpert、AicRange limit |
| 第三方 catlass（cmake/third_party/catlass.cmake 从 gitcode.com/cann/catlass.git 拉取，本地未检出） | Arch::AtlasA2::UB_SIZE=192KB、L1_SIZE=512KB、BYTE_PER_BLK=32（以同仓 vendored 同源副本佐证：mc2/3rd/template_linear_algebra/.../tla_arch_arch.hpp:19-20 同值）；Arch::Resource 提供 ubBuf/l1Buf GetBufferByByte | catlass/arch/arch.hpp、catlass/arch/resource.hpp | static_assert:121、w2ScratchFits:1045、ComputeAivPrefetchHoleBytes:1410 |

## TilingData 值域溯源

> Kernel 经 Params（mega_moe_arch22.h 构造）间接消费 tiling；tiling 结构体为 MegaMoeTilingDataQuant/NonQuant（op_kernel/arch22/mega_moe_tiling_a2a3.h）。本次改动涉及的字段：

| TilingData 字段 | Host 侧计算(文件:行) | 公式 | 输入参数 | 约束 |
|---------------|---------------------|------|---------|------|
| common.listLen | tiling_arch22.cpp:1220 → arch22.h:178/197 | `info.listLen = expertPerRank` | GetDynamicInputTensorListLen(WEIGHT1_INDEX)（:878） | =expertPerRank（per-expert list）；==1 时为堆叠模式（此时 expertPerRank 亦为 1 或 host 走堆叠分支） |
| common.expertPerRank | tiling_arch22.cpp:1218 → arch22.h:174/193 | `= weight1 动态列表长度` | 动态输入 shape | ≥1；GMM2 段界 while 循环上界 expertPerRank-1 |
| common.K（hidden_size） | tiling_arch22.cpp:703 校验 | x.dim1 | x shape | 512 对齐（保证 PrefetchW1/W2 每专家字节为 32 的倍数） |
| common.N（intermediate） | tiling_arch22.cpp:953/959 校验 | weight1.dim1 | weight shape | 512 对齐且 ≥512（N/2 即 W2 的 K 维） |
| （Params）maxOutputSize | host tiling 计算（单轮接收预算 B） | bs/ep/topK 相关 | :526 附近 | GMM currentM 钳制与轮数推导 |
| （Params）recvRoundsMax | host 下发 | 轮表份数 | — | kernel:134 默认 1；workspace 按 numChunks×recvRoundsMax×3 同步分配（kernel:2082-2084 注释） |
| aivNum/aicNum | tiling_arch22.cpp:1236/1421 | GetCoreNumAiv()/GetCoreNumAic() | 平台信息 | A3 典型 40 逻辑核（1AIC+2AIV×20block）；W2_CHUNKS=10 注释按 24 AIC 核推导 |

## 芯片架构参数

| 参数 | 值 | 来源 | 影响范围 |
|------|-----|------|---------|
| ArchTag | Catlass::Arch::AtlasA2 | mega_moe_arch22.h:209 | 全部模板实例 |
| UB_SIZE | 192KB | catlass arch.hpp（第三方；同仓 vendored 同源副本 tla_arch_arch.hpp:19 佐证） | AIV scratch 洞上界（:121 static_assert、ComputeAivPrefetchHoleBytes:1410）、routeCache 布局（initBuffer:394） |
| L1_SIZE | 512KB | 同上 :20 | w1/w2ScratchFits 上界（kernel:288/:1046，减 256B 预留） |
| 核型 | MIX_AIC_1_2（1 AIC + 2 AIV/block，AICore 整体共享 MTE2） | mega_moe.cpp:41；weight_l2_prefetch.hpp:311 注释 | W-c 预取与 GMM B-tile 装载同队列错峰设计 |
| 平台预留 | 256B（ASC_L1_SIZE/ASC_UB_SIZE=总量-256，2201 规） | weight_l2_prefetch.hpp:108-112 注释 | L1 scratch 上界收紧；UB 不收紧的理由（预留区 API 零使用 + 洞上界让出常驻区） |
| aivNum/aicNum | 运行时平台 API | tiling_arch22.cpp:1236/1421 | W2_CHUNKS 剂量推导（注释按 24 AIC 核） |
| 32B 块粒度 | DataCopy/DataCopyPad 块对齐 | weight_l2_prefetch.hpp:114/219（chunkBytes & ~31） | copyBytes/尾块对齐 |

## 代码关联

**上游文件**:
| 文件路径 | 关联方式 | 依据 |
|----------|----------|------|
| op_kernel/arch22/utils/weight_l2_prefetch.hpp | include（kernel:51） | 预取机制全部符号 |
| op_kernel/arch22/utils/get_tensor_addr.hpp | include（kernel:50 / prefetch:37） | GM tensor-list 解引用 |
| op_kernel/arch22/utils/block_mmad_w4a4.hpp、block_mmad_preload_async_fixpipe_quant.hpp | include（kernel:36-37） | BlockMmad + GetL1EndOffset |
| op_kernel/arch22/utils/const_args.hpp | include（kernel:45） | 对齐/flag/上限常量 |
| catlass/arch/arch.hpp、resource.hpp（第三方） | include（kernel:27-34 / prefetch:38） | AtlasA2 UB/L1 常量、ubBuf/l1Buf |
| mega_moe_tiling_a2a3.h | 经 mega_moe_arch22.h 间接（:40） | tiling 字段来源 |
| op_host/op_tiling/arch22/mega_moe_tiling_arch22.cpp | 值域来源（非 include） | K/N 512 对齐校验、listLen/expertPerRank 赋值 |

**下游文件**:
| 文件路径/API | 关联方式 | 依据 |
|----------|----------|------|
| mega_moe_arch22.h:413（kernel(params)） | 被调用 | 唯一实例化点（MatmulKernel 别名 :295） |
| moe_permute_prologue.hpp / mc2_mega_moe_moe_token_unpermute.h | include + 调用 | DispatchAndCombine 内 prologue/unpermute |
| AscendC API（DataCopy/DataCopyPad/CrossCore*/SyncAll/PipeBarrier/SetFlag/WaitFlag/SetL2CacheHint） | API 依赖 | 见 API 调用索引 |

## 高性能设计（仅 Kernel 侧）

**流水线模式**: AIC-AIV 协同（MIX）+ Cube 5 级流水（BlockMmad MTE2 GM→L1 → MTE1 L1→L0 → MMAD → Fixpipe L0C→OUT）+ AIV Vector/MTE3 流水，chunk×round 两级软件流水。

**流水线设计**:
| 机制 | 状态 | 设计意图 |
|------|------|----------|
| EnQue/DeQue同步 | 无（本类不用 TPipe 队列） | 用 HardEvent Set/WaitFlag + PipeBarrier 手工重叠 |
| CrossCore C2V/V2C/syncgmm | 有 | AIC GMM 段 ↔ AIV swiglu 段握手；**W-c 预取刻意发射于 V2C wait 之前**（wait 窗口 = AIV 上一段 swiglu 的 gmC 读时间，哑拷贝与之间隔错峰） |
| L1 scratch 环写 | 有（本次新增） | 单 16KB chunk 环写于 GetL1EndOffset 之上，与 stage/S/F 零地址交集，无需排空同步 |
| L2 hint 联动 | 有（本次新增） | 被预取权重恒 NORMAL，未预取侧维持条件 DISABLE bypass |

**切分策略**（需给出具体数值）:
| 维度 | 切分方式 | 每核处理量 | 切分粒度 |
|------|----------|-----------|---------|
| 多核切分（AIC GMM） | blockScheduler M×N block 按 loopIdx mod coreNum 交错 | coreLoops/coreNum 个 block/专家 | L1TileShape M×N（w4a4/preload 变体按模板），groupIdx 顺序遍历 expertPerRank |
| 多核切分（W-c 预取） | (coreIdx, coreNum) mod 交错认领 16KB chunk | ≤capChunks(10)×16KB=160KB/专家/核 | 全核合计 ≤ coreNum×160KB/专家（24 核 int8 3.5MB/专家=精确全覆盖；注释 :85-88） |
| UB切分（窗口0，休眠） | AIV 洞 [179552, 179552+洞宽) 单缓冲 | 洞宽典型 EP=8/epr=16 时 15.5KB | chunk=洞宽（钳 16KB、32B 对齐、≥2048） |
| chunk/round | 发送 1024 token/chunk；接收 ceil(R/B)/轮 | — | PERMUTE_CHUNK=1024 |

**Buffer 规划**:
| Buffer | 类型 | 大小(B) | 用途 |
|--------|------|------|------|
| L1 W2 预取 scratch | LocalTensor<uint8_t>（l1Buf 子区） | 16384 | GMM2 段界 GM→L1 哑拷贝环写（偏移=AlignUp(GetL1EndOffset,16KB)） |
| L1 W1 冷启动 scratch（休眠） | 同上 | 16384 | 同法（kernel:286-292） |
| UB 窗口0 scratch（休眠） | LocalTensor<uint8_t>（ubBuf 子区） | 洞宽（≤16KB） | AIV GM→UB 哑拷贝 |
| UB 顶部常驻区 | rankMask + 2×routeCache | AlignUp(EP×4,32)+2×AlignUp(EP×epr×4,32) | initBuffer:378-399；AIV 洞上界=其下界（镜像算术 ComputeAivPrefetchHoleBytes） |
| 低地址 UB 临时区 | CopyGMToGM 双 32KB@0/96KB；swiglu epilogue ~146KB 等 | 各阶段 ≤179552 | 既有功能缓冲（预取不占用：routing 时序在窗口之前） |

## 跨文件关系（多文件）

| 关系类型 | 源文件 | 目标文件 | 内容 | 位置 |
|---------|--------|---------|------|------|
| include | mega_moe_kernel_a3.hpp | utils/weight_l2_prefetch.hpp | 预取机制库（开关/剂量/游标/AicRange） | kernel:51 → prefetch 全文件 |
| 函数调用（活） | mega_moe_kernel_a3.hpp GMM2 | weight_l2_prefetch.hpp WeightL2PrefetchAicRange | W-c 段界 W2 发射（int4 :1100 / 非 int4 :1112，互斥分支） | kernel:1094-1104, 1111-1117 → prefetch:278-306 |
| 函数调用（裁剪） | kernel operator()<AIC> / AllGatherTokenPerExpert / DispatchAndCombine | ColdStartW1 / BurstAll / Init+Retarget | W1 冷启动与窗口0（常量 false：:284 为运行时 if + 函数内 early-return 双守卫；:1244/:1706 为 if constexpr） | kernel:289, 1245, 1708-1713 → prefetch:318, 263, 206/180 |
| 共享常量 | weight_l2_prefetch.hpp | mega_moe_kernel_a3.hpp | ENABLE_* 三开关、CHUNK_BYTES、ARCH_RESERVED_BYTES、UB_OFFSET（kernel 的 hint 联动 kGmm1/kGmm2WeightPrefetched 直接引用开关） | prefetch:49-112 → kernel:284-288, 909-910, 1044-1046, 1069, 1094, 1111 |
| 镜像算术 | kernel ComputeAivPrefetchHoleBytes | kernel initBuffer | AIV 洞宽计算镜像 UB 顶部常驻区布局（改动 initBuffer 须同步，两处注释互指） | kernel:1404-1415 ↔ 378-399 |
| 公式共用 | kernel PrefetchW1BytesPerExpert | operator()<AIC>（冷启动）+ DispatchAndCombine（窗口0 Init） | W1 每专家字节单一来源，防两处漂移 | kernel:1374-1384 ← 291, 1710 |
| 段界同源 | kernel GMM2 sectionEnd 推进（IsSyncTask） | ProcessRecvRound nSyncSwiglu 分段（IsSyncTask） | 预取段区间与 AIV swiglu 分段（V2C 置位点）一致：段末 IsSyncTask(g) 为真、GMM2 段始于 IsSyncTask(g-1) | kernel:1089-1104 ↔ 1509-1512, 1567-1587 |
| 数据流（值域） | op_host tiling_arch22.cpp | 两文件 | K/N 512 对齐（:703/:953）→ 预取字节公式除法安全；listLen=expertPerRank（:1220）→ isPerExpertWeightTensor | host → kernel:1062/:1100, prefetch:154/292 |
| 布局合约 | weight_l2_prefetch.hpp 注释（UB_OFFSET 下界= swiglu BUFFER_SIZE+MIN_SHARED_TMP） | utils/block_epilogue_w4a8post_pertoken_swiglu.hpp（static_assert 检查该合约） | 修改任一布局须同步校验 179552 偏移 | prefetch:95-102 |
