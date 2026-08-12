# apace_quant_matmul_reduce_scatter_v2 算子开发计划

> 本文档在开发流程中持续更新。

---

## 1. 需求概述

| 项目 | 内容 |
|-----|------|
| 算子名称 | apace_quant_matmul_reduce_scatter_v2 |
| 数学公式 | C_r = Σ_{i=0}^{R-1} dequant(A_i[r*M/R:(r+1)*M/R]) × dequant(B_i)，R=4 |
| 输入 | A: shape=[M, K_local], dtype=fp8_e4m3fn_t/fp8_e5m2_t; ScaleA: MXFP8 scale; B: shape=[K, N], dtype=同A; ScaleB: MXFP8 scale; workspace: GM 地址 |
| 输出 | C: shape=[M/rankSize, N], dtype=bfloat16_t |
| 算子类别 | MC2 通算融合（Quant MatMul + AllToAll PUT 通信 + ReduceSum 流水线） |
| 需求类型 | 通用 |
| v1 基底 | apace_quant_matmul_reduce_scatter（分支 feat/apace-quant-matmul-reduce-scatter） |
| 参考实现 | ops-nn quant_matmul_reduce_sum（PIPELINE_NUM=4 流水线模式） |

### 核心改造点

1. **3 级流水**：mm(AIC)→aivComm(AIV)→reduceSum(AIV)，PIPELINE_NUM=4 多 buffer workspace 轮转
2. **workspace 入参**：mm 输出写 workspace，reduceSum 从 workspace 读取
3. **reduceSum 简化**：Blaze MX QuantMatmul 已输出 BF16，reduceSum = DataCopyPad + AtomicAdd
4. **AIV 时分复用**：先 AllToAll PUT 通信，后 reduceSum
5. **local 前置**：默认不启用（localMatmul=0），作为 tiling 可选参数

---

## 2. 文件清单

### [MODIFY] 算子层 kernel 文件（3 个）

| 文件 | 职责 | 状态 |
|------|------|------|
| `kernel/apace_quant_matmul_reduce_scatter_v2/apace_quant_matmul_reduce_scatter_v2_tiling_data.h` | tiling 结构体 + CommContext + 常量定义（PIPELINE_NUM, SYNC flags） | ✅ |
| `kernel/apace_quant_matmul_reduce_scatter_v2/apace_quant_matmul_reduce_scatter_v2_udma_impl.h` | Impl 类（Init/Run/RunAllToAll/RunReduceSum/RunMatmul/SetupParams）+ reduceSum 逻辑 | ✅ |
| `kernel/apace_quant_matmul_reduce_scatter_v2/quant_matmul_mx_kernel.h` | Blaze matmul 调度（改造：mm 输出到 workspace + 3 通道 flag 编排） | ✅ |

### [MODIFY] ST 测试工程

| 文件 | 职责 | 状态 |
|------|------|------|
| `st/apace_quant_matmul_reduce_scatter_v2/CMakeLists.txt` | 构建配置（target 名 + `--npu-arch=dav-3510` + include 路径） | ✅ |
| `st/apace_quant_matmul_reduce_scatter_v2/run.sh` | ST 驱动脚本（cases.csv 行选择 + `--perf` msprof 采集） | ✅ |
| `st/apace_quant_matmul_reduce_scatter_v2/cases.csv` | 测试用例（m,k,n,rank_num,head_m_size,dtype） | ✅ |
| `st/apace_quant_matmul_reduce_scatter_v2/src/main.cpp` | Host 入口（tiling 推导 + CommContext 建链 + workspace 分配 + yGm 清零 + launch） | ✅ |
| `st/apace_quant_matmul_reduce_scatter_v2/src/kernel_launcher.h` | 4 个 `__global__` 入口函数（8 参数含 workspace） | ✅ |
| `st/apace_quant_matmul_reduce_scatter_v2/scripts/gen_data.py` | 数据生成（FP8 量化 + MXFP8 scale + golden） | ✅ |
| `st/apace_quant_matmul_reduce_scatter_v2/scripts/verify_result.py` | 精度验证（rtol=atol=1e-2） | ✅ |
| `st/apace_quant_matmul_reduce_scatter_v2/scripts/parse_prof.py` | 性能数据解析 | ✅ |

### [REUSE] 共享层文件（禁止修改）

| 目录 | 来源 | 状态 |
|------|------|------|
| `block/aiv_comm/` | collective_comm_api.h, all_to_all_udma_put.h, barrier_ubmem.h 等 | ✅ REUSE |
| `block/blaze_ext/` | qmm_mx_block_mmad_fragment.h 等 | ✅ REUSE |
| `tiling/` | comm_tiling_data.h, quant_matmul_tiling_*.h | ✅ REUSE |
| `basic/fragment_tensor/` | fragment_tensor.h | ✅ REUSE |
| `utils/` | comm_channel_builder.h, constant.h | ✅ REUSE |
| `st/utils/root_info_exchanger.h` | TCP 交换 HcclRootInfo | ✅ REUSE |
| `st/CMakeLists.txt` | 顶层构建 | ✅ REUSE |

### 文档

| 文件 | 状态 |
|------|------|
| `docs/DESIGN.md` | ✅ 已完成 |
| `docs/PLAN.md` | ✅ 本文件 |
| `docs/environment.md` | ✅ 已完成 |

---

## 3. 测试计划

精度标准：bfloat16_t rtol=atol=1e-2（与 v1 一致，FP8 量化容差）

**Golden 计算**：在 `scripts/gen_data.py` 中使用 numpy 模拟 FP8 量化 matmul + ReduceScatter。

**环境约束**：
- 编译/运行前必须 `source /home/l00947526/Ascend806/cann/set_env.sh`
- 4 卡 NPU 排队执行（`qrun`）
- 编译参数：`--npu-arch=dav-3510`

**用例（T=可执行文件, 4 rank 并行）**：

测试用例必须可追溯到 DESIGN.md §0.2 用户原始需求。

| 编号 | 用例 | 需求追溯 | M | K | N | rankNum | dtype | 预期输出 |
|-----|------|---------|---|---|---|---------|-------|---------|
| T1 | 基础精度-E4M3 | §0.2 #1,#2,#6 | 2048 | 4096 | 2048 | 4 | e4m3e4m3 | C[512×2048] BF16 |
| T2 | 基础精度-E5M2 | §0.2 #1,#2 | 2048 | 4096 | 2048 | 4 | e5m2e5m2 | C[512×2048] BF16 |
| T3 | 混合 dtype | §0.2 #1 | 2048 | 4096 | 2048 | 4 | e4m3e5m2 | C[512×2048] BF16 |
| T4 | 大 shape | §0.2 #2 | 4096 | 8192 | 4096 | 4 | e4m3e4m3 | C[1024×4096] BF16 |
| T5 | 小 shape | §0.2 #2 | 512 | 1024 | 512 | 4 | e4m3e4m3 | C[128×512] BF16 |
| T6 | localMatmul=0 (默认) | §0.2 #4 | 2048 | 4096 | 2048 | 4 | e4m3e4m3 | 验证默认不启用 local |
| P1 | 性能-E4M3 | §0.2 #2 | 2048 | 4096 | 2048 | 4 | e4m3e4m3 | Task Duration + msprof |

> **输入校验约束**：M%4==0, K%4==0, ka=K/4, ka%64==0, tileCnt≤14

---

## 4. 开发进度

| 阶段 | 检查项 | 状态 |
|------|--------|------|
| 框架搭建 | 工程创建 + CMake + 共享层拷贝（从 v1 分支） + 空 Kernel 编译通过 | ✅ |
| tiling 结构体 | tiling_data.h（新增 pipelineNum/coreNum/ubCalSize/totalMmTiles(DEPRECATED)/workspaceSize） | ✅ |
| kernel 实现 | impl.h（RunAllToAll + RunReduceSum(BlockScheduler 驱动) + RunMatmul + flag 编排） + kernel.h（mm→workspace 改造） | ✅ |
| ST host | main.cpp（workspace 分配 + tiling 推导 + launch） + kernel_launcher.h（8 参数） | ✅ |
| 编译验证 | 4 dtype 变体编译通过 | ✅ |
| 精度验证 | T1-T6 全部通过（rtol=atol=1e-2） | ✅ |
| 性能采集 | P1 msprof 采集 + 数据归档 + 与 v1 对比 | ✅ |

### 调试进展记录

| 日期 | 调试内容 | 结果 |
|------|---------|------|
| 2026-08-11 | Step A 最小改名编译通过 | ✅ 14s |
| 2026-08-11 | Step B+C v2 核心改造编译通过（IDX_M_TILEIDX 歧义修复） | ✅ |
| 2026-08-11 | T1 基础用例运行（M=2048,K=4096,N=2048） | ❌ 超时死锁 |
| 2026-08-11 | 小 shape 测试（M=512,K=1024,N=512） | ❌ 超时死锁 |
| 2026-08-11 | 简化版 reduceSum（只做 flag 配对，不做 DataCopyPad） | ✅ 执行成功——flag 配对正确 |
| 2026-08-11 | 简化版 + DataCopyPad（不用 BlockScheduler，用 totalMmTiles 线性循环） | ✅ 执行成功——DataCopyPad 不死锁 |
| 2026-08-12 | **design_issue 1 修复**：RunReduceSum 重构为 BlockScheduler 驱动（§2.16 方案 b，6 项复刻要求逐项落实），取消 coreIdx==0 限制，GM→UB 改 2D 搬运 | ✅ 编译通过，死锁消除 |
| 2026-08-12 | **design_issue 2 修复**：totalMmTiles 标记 DEPRECATED（host 填 0，kernel 不消费），AIV 循环由 bsAiv.GetTileIdx() 隐式终止 | ✅ |
| 2026-08-12 | T5 首跑：无死锁，精度 FAIL（输出全 0 + 乱码） | 定位：DataCopyPad GM→UB(MTE2) 与 UB→GM(MTE3) 之间缺流水同步 |
| 2026-08-12 | 修复①：新增 MTE2→MTE3（RAW）/ MTE3→MTE2（WAR）event 同步（pipe_.FetchEventID + SetFlag/WaitFlag\<HardEvent\>） | 输出出现真实数据，但行错位（row0 正确、row1 全 0、row2+ 错位） |
| 2026-08-12 | 定位行错位：T5 baseN=48（rowBytes=96B，32B 对齐但非 64B 对齐），cpIn.dstStride=64B 对齐 pitch（=1）而 cpOut.srcStride 按 DESIGN 表格公式（AlignUp(,16)）=0，UB 读写行距不一致 | 根因确认 |
| 2026-08-12 | 修复②：cpOut.srcStride 与 cpIn.dstStride 统一为 64B 对齐 pitch 公式（CeilDiv(rowBytes,64)*2 - CeilDiv(rowBytes,32)） | ✅ T5 PASS（max_err=0 bit 级一致） |
| 2026-08-12 | cases.csv 全量 10 行（T1/T2/T3/T5 + 混合 dtype + M=1120 非对齐尾块） | ✅ 10/10 PASS |
| 2026-08-12 | T4 大 shape CLI（4096×8192×4096，tileCnt=2 多通信轮次 + mTailTile=3 尾块拆分） | ✅ PASS |
| 2026-08-12 | P1 性能采集（msprof task-based + L2 flush）；修复 parse_prof.py KERNEL_SUBSTR 沿用 v1 名导致解析失败 | ✅ 58.0 us，vs v1 143 us，≈2.47× |
| 2026-08-12 | 通信 flagId 映射 CommFlagIdV2 落地（避开 3/5/[11-14]），tileCnt=4 用例（16384×4096×2048）验证 + 全量回归 | ✅ PASS + 10/10 回归无劣化 |

---

## 5. 已知问题和决策记录

| 日期 | 问题/决策 | 说明 |
|------|----------|------|
| 2026-08-11 | flagId 冲突规避 | 通信就绪 flagId 使用映射函数避开 3 和 5（mm 完成=5，回压=3）；tile 总数 ≤ 10 约束 |
| 2026-08-11 | yGm 清零方式 | 不需要——reduceSum 使用普通 DataCopyPad 覆盖，非 AtomicAdd |
| 2026-08-11 | reduceSum 简化 | Blaze MX QuantMatmul 已输出 BF16，reduceSum 不需要 AscendDequant |
| 2026-08-11 | localMatmul 默认值 | v2 默认 0（不启用），v1 默认 1。local 作为可选优化项 |
| 2026-08-11 | AIV 时分复用 | 每个 AIV 核先完成所有通信轮次，后切换到 reduceSum |
| 2026-08-11 | **design_issue: AIV 侧 BlockScheduler 调度不一致** | AIV 使用 BlockScheduler 时死锁，根因是 MIX_AIC_1_1 核配比下 AIV/AIC 的 GetBlockIdx() 或 GetBlockNum() 行为差异导致调度不一致。DESIGN.md §1.4 伪代码中 AIV reduceSum 的 `DataCopyPad(UB → yGm[outOffset])` 的 outOffset 计算方式未明确说明，且未考虑 AIV 侧不用 BlockScheduler 时如何推导 mPos/nPos。已用 totalMmTiles 次线性循环临时替代，但 yGm 位置推导需要与 AIC 的 BlockScheduler 调度顺序一致。**需 Architect 重新评估 AIV 侧 reduceSum 的实现方式。** |
| 2026-08-12 | ✅ **design_issue 1 已解决（Architect 决策）** | **决策：方案 (b)——AIV 复用 BlockScheduler**（DESIGN.md §2.16）。官方文档证明 MIX_AIC_1_1 下 `GetBlockIdx()/GetBlockNum()/GetTaskRatio()` 在 AIC 与配对 AIV 上完全一致（GetBlockIdx.md / GetBlockNum.md / GetTaskRatio.md），原「行为差异」根因假设不成立；死锁真实根因为 AIV 复刻循环时调用序列与 AIC 不一致（UpdateTailTile / GetBlockShape 模板参数 / 零形状早退语义），导致 flag 配对失衡。设计要求 AIV 按 §2.16「精确复刻要求清单」6 项逐字复刻 AIC 调度器调用序列，mPos/nPos/curBaseM/curBaseN 由 `GetTileCoord`/`GetBlockShape` 推导，`outOffset = mPos * N + nPos`。方案 (a) workspace metadata 已拒绝（AIC 无 UB，标量 GM 写无法被 mode 0/1/2 的 pipe fence，可见性无保证）。同时取消 `coreIdx==0` 单核限制，恢复多核并行 reduceSum；修正 GM→UB 为 2D 搬运（slot 内行距=curBaseN 紧密排列 srcStride=0，UB 行距 32B 对齐，同 ops-nn DataCopyPad2D/DataCopyOut）。 |
| 2026-08-11 | **design_issue: totalMmTiles 计算与 BlockScheduler 调度不匹配** | 当前 `totalMmTiles = tileCnt * CeilDiv(N, baseN)`，可能不等于 BlockScheduler 实际调度的 tile 总数。AIV 的 reduceSum 循环次数需要 = AIC 的 SetFlag(5) 总次数（每核），但 totalMmTiles 的计算方式可能不正确。**需 Architect 确认 totalMmTiles 的正确计算公式。** |
| 2026-08-12 | ✅ **design_issue 2 已解决（Architect 决策）** | **决策：totalMmTiles 废除**（DESIGN.md §2.16）。旧公式双重错误：(1) `tileCnt` 是通信 tile 数（headMSize 粒度）而非 mm tile 数（baseM×baseN 粒度）；(2) 算出的是全局 tile 数而非每核 tile 数。每核 mm tile 数 = 本核 BlockScheduler 实际迭代数（roundNums_ 经 UpdateTailTile 调整 + 零形状早退截断），host 无法可靠预算。方案 (b) 下 AIV 循环由 `bsAiv.GetTileIdx()` 隐式终止，与 AIC SetFlag(5) 次数构造一致（flag 配对平衡证明见 DESIGN.md §2.16）。tiling 字段标记 DEPRECATED，host 填 0，kernel 不消费。 |

---

## 6. 测试结果

### 6.1 可执行文件通路

**状态**: ✅ 全部通过 | **脚本**: run.sh + scripts/verify_result.py

| 编号 | 用例 | 需求追溯 | 结果 | 备注 |
|-----|------|---------|------|----------|
| T1 | 基础精度-E4M3（2048×4096×2048） | §0.2 #1,#2,#6 | ✅ PASS | cases.csv row1 |
| T2 | 基础精度-E5M2（2048×4096×2048） | §0.2 #1,#2 | ✅ PASS | cases.csv row5 |
| T3 | 混合 dtype（2048×4096×2048） | §0.2 #1 | ✅ PASS | row7(e4m3e5m2) + row9(e5m2e4m3) |
| T4 | 大 shape（4096×8192×4096） | §0.2 #2 | ✅ PASS | CLI 模式；tileCnt=2 多通信轮次 + mTailTile=3 尾块拆分路径 |
| T5 | 小 shape（512×1024×512） | §0.2 #2 | ✅ PASS | row2；baseN=48 非 64B 对齐路径，max_err=0 |
| T6 | localMatmul=0 (默认) | §0.2 #4 | ✅ PASS | 全部用例均为 localMatmul=0（REMOTE-only） |
| 补充 | 非对齐 M=1120（mLocal=280 尾块）× 4 dtype | §0.2 #2 | ✅ PASS | row3/6/8/10 |
| 补充 | K=4608（ka=1152） | §0.2 #2 | ✅ PASS | row4 |
| 补充 | tileCnt=4 多通信轮次（16384×4096×2048） | §0.2 #2 | ✅ PASS | CLI；验证 CommFlagIdV2 映射（fid 0,1,2,4 避开 3） |

### 6.2 产物 & 执行状态

- [x] `st/build/apace_quant_matmul_reduce_scatter_v2/apace_quant_matmul_reduce_scatter_v2` 可执行文件存在
- [x] 4 rank 并行执行成功（10/10 cases + T4 CLI 全部通过）

| 通路 | 状态 | 运行时间 | 跳过原因 |
|------|------|---------|---------|
| 可执行文件 | ✅ | 全量 10 case ≈ 2min47s | - |

### 6.3 调试关键发现

1. **flag 配对正确**：简化版 reduceSum（只做 WaitFlag(5)→SetFlag(3)，不做 DataCopyPad）执行成功，确认 3 通道 flag 编排（通信就绪 + mm 完成 + 回压）配对正确。
2. **死锁根因修正（§2.16）**：AIV 使用 BlockScheduler 死锁的真实根因是 AIV 复刻循环的调用序列与 AIC 不一致（非 GetBlockIdx() 行为差异——MIX_AIC_1_1 下官方文档证明两侧一致）。按 6 项复刻要求逐项对齐后死锁消除。
3. **DataCopyPad 必须显式流水同步**：裸 TBuf 场景下 GM→UB(MTE2) 与 UB→GM(MTE3) 之间需 SetFlag/WaitFlag\<HardEvent::MTE2_MTE3\>（RAW）与 MTE3_MTE2（WAR）event 同步，否则 UB→GM 读到未完成搬运的数据（输出全 0/乱码）。ops-nn 通过 TQue EnQue/DeQue 隐式获得该同步。
4. **UB 行距 pitch 必须双向一致**：cpIn.dstStride（64B 对齐 pitch）与 cpOut.srcStride 必须使用同一公式；curBaseN 非 64B 对齐（如 baseN=48）时 DESIGN 表格中 cpOut 的 AlignUp(,16) 公式会与 cpIn 行距不匹配导致行错位。
5. **HCCL 建链环境问题**："CreateDeviceContext failed" 通过 run.sh 内置的 fuser 端口清理 + 重试机制规避，本轮未复现。

---

## 7. 性能验收

**状态**: ✅ 达标 | **数据**: docs/perf/round_001/（summary.txt + parse_output.txt + prof_raw/ + prof/（2026-08-12 10:47 复测轮，57.626us，精度 PASS））

| 指标 | 值 | 判定 |
|------|------|------|
| Task Duration (v2) | 58.003 us（msprof task-based，4 卡平均，P1: 2048×4096×2048 e4m3；复测轮 57.626 us） | ✅ |
| Task Duration (v1 基线) | 143 us (开发历史数据) | - |
| 加速比 | ≈ 2.47× | ✅ |
| Block Dim | 28（usedCoreNum，baseM=256 baseN=160 baseK=128） | - |
| 主导流水 | cube ≈ 91%（90.9%~92.0%） | mm 主导，通信已掩盖 |

**达标判定**: ✅ | **理由**: v2 的 3 级流水 + PIPELINE_NUM=4 workspace 轮转将 AIV 通信与 reduceSum 掩盖在 AIC mm 之下，Task Duration 58.0 us 显著优于 v1 串行模式 143 us（≈2.47×），cube 流水占比 ~91% 表明计算侧接近饱和，架构设计目标达成。

---

## 8. 汇总

| 通路 | 用例数 | 通过 | 失败 | 状态 |
|------|--------|------|------|------|
| 可执行文件 | 12（cases.csv 10 + T4 CLI + tileCnt=4 CLI） | 12 | 0 | ✅ |
| 性能 | 1 | 1 | 0 | ✅ |

### 开发成果总结

| 阶段 | 状态 | 说明 |
|------|------|------|
| 工程骨架 | ✅ | 从 v1 分支复制共享层 + ST 骨架 + apace 符号链接 |
| tiling 结构体 | ✅ | 新增 pipelineNum/coreNum/ubCalSize/workspaceSize；totalMmTiles DEPRECATED（§2.16 Issue 2） |
| impl.h | ✅ | 3 级流水（RunAllToAll + RunReduceSum + RunMatmul）+ 3 通道 flag 编排；RunReduceSum 按 §2.16 方案 (b) 重构为 BlockScheduler 驱动（6 项复刻要求逐项落实）+ MTE2/MTE3 event 流水同步 + 2D DataCopyPad（UB 行距 64B 对齐 pitch 双向一致） |
| kernel.h | ✅ | mm→workspace 改造（per-tile gmCWorkspace + PostCompute drain） |
| kernel_launcher.h | ✅ | 4 dtype 变体入口（8 参数含 workspace） |
| main.cpp | ✅ | workspace 分配 + tiling 填充 + localMatmul=0 + flagId 校验 + totalMmTiles=0（调试用 DUMP_WORKSPACE 开关已在精度达标后移除） |
| 编译验证 | ✅ | 4 dtype 变体编译通过 |
| 精度验证 | ✅ | T1-T6 + 补充用例全部通过（rtol=atol=1e-2，小 shape bit 级一致） |
| 性能采集 | ✅ | P1: 58.0 us vs v1 143 us（≈2.47×），数据归档 docs/perf/round_001/ |

### §2.16 设计决策落实核对（自我检查）

| 复刻要求 | 落实 | 代码位置 |
|---------|------|---------|
| 1. 同一 problemShape/schParams 构造调度器 | ✅ | impl.h RunReduceSum：`SetupParams(&tilingData_->tileQbmmTilingData, params, mode)` → `BlockScheduler bsAiv(params.problemShape, params.schParams)` |
| 2. UpdateTailTile 判定逐字复刻（restBatch=0, isTailRound=true） | ✅ | `if ((bsAiv.GetEndBlockIdx()+1) * mTailTile * nTailTile <= GetBlockNum()) bsAiv.UpdateTailTile(...)` |
| 3. 循环内调用顺序 GetTileIdx→GetBlockShape\<MX,MX,weightNz\>→零形状检查→GetTileCoord | ✅ | RunReduceSum while 循环 |
| 4. 零形状早退 break（对齐 AIC return） | ✅ | `if (M<=0 \|\| N<=0) break;` |
| 5. cubeNum 仅在实际处理 tile 上自增（SetFlag(3) 之后） | ✅ | 循环尾部 `cubeNum++` |
| 6. coreIdx = GetBlockIdx()（AIV，1:1 下与 AIC 相同） | ✅ | RunReduceSum 头部 |
| outOffset = mPos * N + nPos | ✅ | UB→yGm 2D 搬运 |
| 取消 coreIdx==0 单核限制（多核并行） | ✅ | 每个 AIV 核处理本核 tile 序列 |
| GM→UB 2D 搬运（srcStride=0，UB 行距对齐） | ✅ | cpIn/cpOut 2D 参数 |
| totalMmTiles 废除（host 填 0，kernel 不消费） | ✅ | main.cpp / tiling_data.h / impl.h |

### 与 DESIGN.md 的偏差说明（实现级修正，不改变设计框架）

1. **cpOut.srcStride 公式统一**：DESIGN.md §2.16 表格中 UB→yGm 的 srcStride=(AlignUp(curBaseN,16)-curBaseN)*2/32，在 curBaseN 为 32B 对齐但非 64B 对齐（T5 baseN=48）时与 cpIn.dstStride（64B 对齐 pitch）不一致导致行错位。实现统一为 cpIn 同款公式（CeilDiv(rowBytes,64)*2 - CeilDiv(rowBytes,32)），64B 对齐场景两者等价（均为 0），符合设计意图"与 GM→UB 建立的 UB 行对齐匹配"。
2. **新增 MTE2/MTE3 event 同步**：DESIGN §1.4 伪代码未显式写出 DataCopyPad 间的流水同步（ops-nn 由 TQue EnQue/DeQue 隐式提供）；v2 用裸 TBuf，需显式 SetFlag/WaitFlag\<HardEvent::MTE2_MTE3\>（RAW）+ MTE3_MTE2（WAR），否则 UB→GM 读到未完成数据。
3. **UB buffer 安全上界**：reduceSumBuf 大小 baseM × AlignUp(baseN,32) × 2B（容纳 64B 对齐 pitch，baseN 32 对齐时与设计值相同）。
4. **parse_prof.py 修正**：KERNEL_SUBSTR 从 v1 核名修正为 `ApaceQuantMatmulReduceScatterV2Kernel`（v2 工程 bug，非设计偏差）。
5. **CommFlagIdV2 映射公式修正**：DESIGN §2.6 公式 "(tid<3)?tid:(tid<5)?tid+1:tid+2" 在 tid=4 处映射到 5（与 mm 完成通道冲突）、tid=9 处映射到 11（SyncAll 占用区）。实现按 DESIGN 给出的可用集 {0,1,2,4,6,7,8,9,10,15} 修正映射，符合设计意图（避开 3/5/[11-14]）。

### 遗留事项

1. **HCCL 建链环境问题**："CreateDeviceContext failed" 本轮未复现（run.sh 内置 fuser 端口清理 + 重试机制）。如复现需清理 NPU 残留进程。
2. ~~通信 flagId 映射函数未启用~~ ✅ **已实现（2026-08-12）**：`CommFlagIdV2(tid)` 映射到可用集 {0,1,2,4,6,7,8,9,10,15}（避开 3/5/[11-14]），AIV Set（RunAllToAll）与 AIC Wait（UdmaCommWaitPolicy::WaitTile）两侧一致。已修正 DESIGN 表格公式的 off-by-one（tid=4 会映射到 5）。经 tileCnt=4 用例（M=16384×4096×2048，tid 0..3 → fid 0,1,2,4）验证通过，全量回归 10/10 PASS。

### 待解决问题（后续迭代）

见上方「遗留事项」。此前的 2 个 design_issue 均已解决并验证（见 §5 决策记录与上方「§2.16 设计决策落实核对」表）。

### 关键设计决策（实现中确认）

| 决策 | 说明 |
|------|------|
| 无 AtomicAdd | ✅ 已实现——reduceSum 使用普通 DataCopyPad（非 AtomicAdd），yGm 不清零 |
| Blaze mm per-tile 动态输出 | ✅ 已实现——while 循环内 per-tile 创建 gmCWorkspace Tensor |
| CrossCoreWaitFlag 模板参数 | ✅ 已实现——modeId=2 + PIPE_MTE2（AIV）/ PIPE_M（AIC） |
| 3 通道 flag 编排 | ✅ 已实现——通信就绪(per-tile) + mm 完成(5) + 回压(3) |
| AIV 时分复用 | ✅ 已实现——先 RunAllToAll 后 RunReduceSum |
| localMatmul 默认 0 | ✅ 已实现——REMOTE-only，splitKNum=rankSize |
