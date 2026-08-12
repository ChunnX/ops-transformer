# apace_quant_matmul_reduce_scatter_v2 审查报告

## Round 0 审查报告（Step 4 初审）

- **审查日期**：2026-08-12
- **判定**：**PASS**
- **总分**：**98 / 100**
- **审查人**：ascendc-kernel-reviewer（独立审查，全部结论基于独立构建/独立运行验证，未采用 Developer 自报结果）

---

### 0. 审查概要

| 审查项 | 方法 | 结果 |
|--------|------|------|
| 独立编译 | qrun 排队，全新 `build_review/` 目录，bisheng（CANN 9.2.0） | ✅ BUILD_RC=0，0 warning |
| 独立精度 | qrun 排队，Reviewer 自编译二进制跑 12 用例（cases.csv 全量 10 + T4 大 shape + tileCnt=4 多通信轮次） | ✅ 12/12 PASS |
| 独立性能 | qrun 排队，msprof task-based + L2 flush（Reviewer 自编译二进制） | ✅ 60.3 us，cube≈91%，与 Developer 58.0us 一致（偏差 <4.2%） |
| APACE 红线 R1-R8 | grep + 逐文件 diff | ✅ 8/8 通过 |
| v2 特有审查重点 | 逐项代码核对 | ✅ 全部落实 |

**独立精度验证备注**：首轮 10 个 case 全部报 `CreateDeviceContext failed`（rc=255），但同轮后 2 个大 shape case 通过；第二轮重跑 10/10 全部一次通过（重试次数=0）。判定为 HCCL 建链环境抖动（与 PLAN.md §5 已知问题一致），**非算子 bug**。精度实测：rel_err max ≈ 5.0e-3 ~ 6.4e-3（阈值 1e-2），小 shape（512×1024×512、1120×2048×1024 等）max_err=0 bit 级一致。

---

### 1. APACE 路线红线检查（R1-R8）

| # | 检查项 | 结果 | 证据 |
|---|--------|------|------|
| R1 | 无 `__schedmode__` / `core_ratio` | ✅ | grep 全工程（kernel/ st/ block/ tiling/ basic/ utils/）无匹配 |
| R2 | 有 `KERNEL_TYPE_MIX_AIC_1_1` | ✅ | kernel_launcher.h 4 个入口均含 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1)` |
| R3 | 入口变体与参考算子一致 | ✅ | PUT=4 dtype 变体入口于 `kernel_launcher.h`（E4M3E4M3/E5M2E5M2/E4M3E5M2/E5M2E4M3，8 参数含 workspaceGM + `__gm__ CommContext*`） |
| R4 | `block/` `tiling/` 未修改 | ✅ | 与 v1 参考分支（feat/apace-quant-matmul-reduce-scatter，取自官网仓）逐文件 diff：`block/` `tiling/` `basic/` `utils/` 全部 identical；`st/utils/root_info_exchanger.h` identical |
| R5 | CrossCore flag idx 配对 | ✅ | 通道1：AIV `SetFlag<0x2,MTE3>(CommFlagIdV2(tid))` ↔ AIC `WaitFlag<0x2,MTE2>(CommFlagIdV2(tileIdx))`（同一映射函数两侧一致）；通道2：AIC `SetFlag<2,FIX>(5)` ↔ AIV `WaitFlag<2,MTE2>(5)`；通道3：AIV `SetFlag<2,MTE3>(3)` ↔ AIC `WaitFlag<2,M>(3)` |
| R6 | CommContext 与引擎匹配 | ✅ | 4 个 UDMA 入口均含 `__gm__ CommContext*` 首参 |
| R7 | 无 asc-devkit matmul | ✅ | grep 无 `AscendC::Matmul`；计算走 Blaze `BlockMmad`+`BlockSchedulerQuantBatchMatmulV3`+`MatmulWithScaleMx` |
| R8 | 无 HCCL 高阶 API | ✅ | grep 无 `Hccl::`（host 侧 `HcclCommInitRootInfoConfig` 为建链 C API，非 kernel 级 HCCL 高阶 API，符合 apace 路线约定） |

### 2. v2 特有审查重点逐项核对

| # | 检查项 | 结果 | 证据 |
|---|--------|------|------|
| 1 | 3 级流水编排（mm→comm→reduceSum，PIPELINE_NUM=4） | ✅ | `Run()`：AIV `RunAllToAll()`→`RunReduceSum()` 时分复用，AIC `RunMatmul()`；workspace 4 slot 轮转；AIC 回压点 `cubeNum_ >= pipelineNum_`（kernel.h:404） |
| 2 | workspace 地址计算 `baseM×baseN×(coreIdx+(cubeNum%4)×coreNum)` | ✅ | AIC 侧 kernel.h:418-419 与 AIV 侧 impl.h:362-363 公式逐字一致；host 分配 `4×coreNum×baseM×baseN×2B`（main.cpp:199）上界正确（最大偏移 baseM×baseN×(4×coreNum−1) < 总量） |
| 3 | AIV 复用 BlockScheduler 6 项复刻要求（DESIGN §2.16） | ✅ | 逐项核对：① 同一 SetupParams 产物构造（impl.h:319-332）② UpdateTailTile 判定与 AIC 等价（restBatch=0 展开后逐字一致，impl.h:336-338 vs kernel.h:375-380）③ 循环内 GetTileIdx→GetBlockShape<MX,MX,weightNz>→零形状检查→GetTileCoord 顺序一致 ④ 零形状 `break` 对齐 AIC `return` ⑤ cubeNum 仅在 SetFlag(3) 后自增 ⑥ coreIdx=GetBlockIdx() |
| 4 | 3 通道 flag 编排配对 | ✅ | 配对计数平衡证明复核：设单核 tile 数 T，通道2 AIC Set T 次↔AIV Wait T 次（含零形状截断一致）；通道3 AIV Set T 次↔AIC Wait max(T−4,0)+min(T,4)=T 次；通道1 per-tid 1:1 |
| 5 | flagId 避开 [11-14] | ✅ | `CommFlagIdV2` 可用集 {0,1,2,4,6,7,8,9,10,15}，避开 3/5/[11-14]；修正了 DESIGN 表格公式 tid=4→5 的 off-by-one；host 侧校验 tileCnt≤10（main.cpp:210-213）；tileCnt=4 用例独立验证通过 |
| 6 | CrossCoreWaitFlag 模板参数（modeId=2 + PIPE_MTE2/PIPE_M） | ✅ | AIV 侧 `<SYNC_MODE2_V2, PIPE_MTE2>`（impl.h:359），AIC 回压 `<SYNC_MODE2_V2, PIPE_M>`（kernel.h:405,517），与 950PR 模式 2 模板参数生效约束一致 |
| 7 | reduceSum 无 AtomicAdd（WALKTHROUGH 问题 4 修正） | ✅ | `isAtomicAdd_=false`（kernel.h:293），无 `SetAtomicAdd`；reduceSum 为普通 2D DataCopyPad 覆盖写；host 无 yGm 清零（main.cpp:322）；`SetL2Cache` 中 `isAtomicAdd_` 分支恒不触发 |
| 8 | localMatmul 默认 0 | ✅ | tiling 结构体默认值 0 + main.cpp:190 显式置 0（REMOTE-only，splitKNum=rankSize） |
| 9 | MTE2/MTE3 流水同步（裸 TBuf 无 TQue 隐式同步） | ✅ | RAW：`SetFlag/WaitFlag<HardEvent::MTE2_MTE3>`（impl.h:381-382）；WAR：`WaitFlag<HardEvent::MTE3_MTE2>`（cubeNum>0 时，impl.h:376-378）+ 循环尾 `SetFlag<MTE3_MTE2>`；event 计数逐轮配对平衡 |
| 10 | UB 行距 pitch 双向一致 | ✅ | `cpOut.srcStride = cpIn.dstStride`（impl.h:392），64B 对齐 pitch 公式统一；baseN=48 非 64B 对齐路径（T5）独立验证 bit 级一致 |

### 3. 同步 API 逐项依赖分析（维度 4.4 强制项）

算子未使用 `PipeBarrier`（无冗余 barrier 分析问题）。全部同步原语逐项分析：

| 位置 | 前操作 | 后操作 | 依赖类型 | 判定 |
|------|--------|--------|---------|------|
| impl.h:359 `CrossCoreWaitFlag<2,MTE2>(5)` | AIC mm fixpipe 写 workspace（PIPE_FIX） | AIV DataCopyPad workspace→UB（MTE2） | 跨核 RAW | 必要 ✅ |
| impl.h:376-378 `WaitFlag<MTE3_MTE2>`（cubeNum>0） | 上一轮 UB→GM 读 UB（MTE3） | 本轮 GM→UB 写 UB（MTE2） | WAR | 必要 ✅ |
| impl.h:381-382 `SetFlag/WaitFlag<MTE2_MTE3>` | GM→UB 写 UB（MTE2） | UB→GM 读 UB（MTE3） | RAW 跨 pipe | 必要 ✅ |
| impl.h:400 `CrossCoreSetFlag<2,MTE3>(3)` | UB→yGm 完成（MTE3） | AIC 回压 Wait（PIPE_M） | 跨核（slot 释放） | 必要 ✅ |
| kernel.h:405 `CrossCoreWaitFlag<2,M>(3)` | AIV slot 释放 | AIC mm 覆写 workspace slot（M/FIX） | 跨核 WAR | 必要 ✅（仅 cubeNum≥4 时触发，非每 tile 冗余等待） |
| kernel.h:496 `CrossCoreSetFlag<2,FIX>(5)` | mm fixpipe 完成 | AIV 读 workspace | 跨核 RAW | 必要 ✅ |
| impl.h:299 `SyncAll<true>` | AllToAll PUT Commit/Wait | SetFlag 通知 AIC | AIV 块间同步（WriteNbi 对端可见性） | 必要 ✅（v1 已验证模式；flagId 已避开其占用区 [11-14]） |
| impl.h:301 `CrossCoreSetFlag<0x2,MTE3>(CommFlagIdV2(tid))` | 通信完成 | AIC 读 Win 区（MTE2 阻塞） | 跨核 RAW | 必要 ✅ |

**冗余率：0%**。所有同步点均落在真实跨 pipe/跨核数据依赖上，无过度同步。

### 4. 七维度评分

| 维度 | 检查项 | 得分 | 说明 |
|------|--------|------|------|
| 1 编译验证（10） | 1.1 独立编译成功 | 7/7 | qrun 独立构建 BUILD_RC=0 |
| | 1.2 无代码级警告 | 2/3 | 构建日志 0 warning，但 CMake `KERNEL_OPTS`/`HOST_OPTS` 带 `-w` 抑制警告，无法完全排除被抑制的代码级告警（见备注 N2） |
| 2 架构合规（15） | 2.1 TPipe/TQue 模式 | 3/3 | AIV 侧 TPipe+TBuf<VECCALC>；通信对象按 apace 参考结构自管理 |
| | 2.2 入口属性正确 | 3/3 | `__global__ __aicore__` + MIX_AIC_1_1 |
| | 2.3 定义顺序正确 | 3/3 | 声明→实现顺序规范 |
| | 2.4 内存管理配对 | 3/3 | host aclrtMalloc/Free 全配对（含 v2 新增 deviceWorkspace）；UB 经 TPipe 管理 |
| | 2.5 数据流完整 | 3/3 | GM→Win→L1→L0C→workspace→UB→yGm 全链路闭环 |
| 3 编码规范（15） | 3.1 矢量/计算 API | 4/4 | 搬运用 DataCopyPad；计算走 Blaze mmad（MC2 强制） |
| | 3.2 API 约束满足 | 4/4 | DataCopyPad 对齐约束经 baseN=48 边界用例实测验证 |
| | 3.3 数据对齐 | 4/4 | UB 行距 64B pitch 双向一致 |
| | 3.4 命名规范 | 3/3 | 一致 |
| 4 性能优化（20） | 4.1 动态硬件参数 | 4/4 | 核数/baseM/baseN 全部由 SWAT tiling 引擎运行时推导，launch 用 `usedCoreNum`；grep 无 `blockDim=/blockIdx=` 硬编码 |
| | 4.2 多核并行 | 4/4 | M×N tile 28 核切分 + 尾块拆分；AIV 多核并行 reduceSum（无 coreIdx==0 限制） |
| | 4.3 流水线/双缓冲 | 4/4 | PIPELINE_NUM=4 workspace 轮转 + L1 nBufferNum=2 + dbL0c |
| | 4.4 同步策略 | 4/4 | 逐项依赖分析（§3），冗余率 0%，flag 配对计数平衡证明复核通过 |
| | 4.5 计算效率与上板性能 | 4/4 | **独立采集**：60.3us（4 卡均值），cube≈91%，vs Developer 58.0us 偏差 4.2%（机器负载波动范围内），vs v1 基线 143us ≈2.37× |
| 5 测试覆盖（15） | 5.1 测试数据生成 | 4/4 | gen_data.py：FP8 量化 + MXFP8 scale + numpy golden，seed 固定 |
| | 5.2 结果验证脚本 | 4/4 | verify_result.py：per-rank allclose + 误差统计 + corner dump |
| | 5.3 Level 0 覆盖 | 4/4 | 小 shape 512×1024×512 起，覆盖基础功能 |
| | 5.4 精度标准明确 | 3/3 | rtol=atol=1e-2（BF16 输出 + FP8 量化），DESIGN §2.13 与 verify_result.py 一致 |
| 6 精度验证（10） | 6.1 FP32 | N/A→4/4 | 算子仅声明 FP8 输入/BF16 输出（DESIGN §0.1），FP32/FP16 不适用；按适用 dtype 全覆盖计分 |
| | 6.2 FP16 | N/A→3/3 | 同上 |
| | 6.3 BF16 全用例 PASS | 3/3 | **独立验证 12/12 PASS**（4 dtype 组合 × 常规/尾块/非对齐/大 shape/多通信轮次） |
| 7 文档（15） | 7.1 README 存在 | 3/3 | ✅ |
| | 7.2 数学公式 | 3/3 | README + DESIGN §1.1 |
| | 7.3 编译运行指南 | 3/3 | README「构建与测试」含 env source/qrun/run.sh 用法 |
| | 7.4 API 映射/约束 | 3/3 | DESIGN §1.2 API 映射表（含验证状态与文档出处） |
| | 7.5 已知限制 | 2/3 | README 有输入校验约束（tileCnt≤10 等），但无独立「已知限制」章节；HCCL 建链抖动、`-w` 警告抑制等仅散见于 PLAN.md（见备注 N1/N4） |

**总分：98 / 100**

### 5. 必须修复问题

无。（检查项 1.1/2.1/2.2/3.1/3.2/4.1/6.1 全部通过）

### 6. 备注（非阻塞，建议后续迭代处理）

| # | 类型 | 描述 | 建议 |
|---|------|------|------|
| N1 | 注释陈旧 | `main.cpp` 头注释仍写 "ReduceScatter via AllToAll PUT + **AtomicAdd**"、"localMatmul = 1"；`gen_data.py` docstring 同样残留 "AtomicAdd"；`verify_result.py:32` 注释残留 "localMatmul=1 模式下 BF16 AtomicAdd 精度损失" | 清理为 v2 语义（无 AtomicAdd、localMatmul 默认 0），避免误导后续维护者 |
| N2 | 构建配置 | CMake `KERNEL_OPTS`/`HOST_OPTS` 使用 `-w` 全量抑制警告 | 建议开发期去掉 `-w` 复验一次确认无代码级告警，发布前可加回 |
| N3 | 风格一致性 | 通道 1 用字面量 `0x2`，通道 2/3 用常量 `SYNC_MODE2_V2`（值相同） | 统一为常量，提升可读性 |
| N4 | 测试基建 | 首轮独立验证遭遇 `CreateDeviceContext failed` 环境抖动（10 连失败后自愈，第二轮 10/10 一次通过）。run.sh 重试条件仅匹配 "connect peers failed"，不覆盖该错误 | 建议 run.sh 将 `CreateDeviceContext failed` 纳入重试匹配条件（纯测试基建改进，非算子问题） |
| N5 | 验证脚本适配 | `verify_cmake_config.py` 对两级 CMake 结构误报（顶层含 `find_package(ASC REQUIRED)`+`LANGUAGES CXX ASC`，子目录含 `add_executable`+`--npu-arch=dav-3510`+`tiling_api`，合并后要素齐全） | 已人工核对确认 CMake 配置合规；脚本不支持 add_subdirectory 递归，属工具限制，不计入扣分 |
| N6 | 测试代码 | main.cpp perf 模式 `HEAVY_BLOCK_NUM=56` 硬编码（L2 flush 专用 heavy kernel 核数） | 测试 harness 范畴，不影响算子 kernel；如需适配其他 SKU 可改为运行时查询 |

### 7. 独立验证记录（审计追溯）

| 项目 | 命令/方式 | 结果 |
|------|----------|------|
| 独立编译 | qrun `z00949060-v2-review-build`，`rm -rf build_review && cmake -S st -B build_review && cmake --build` | ✅ 12s，BUILD_RC=0，0 warning |
| 独立精度（第 1 轮） | qrun `z00949060-v2-review-precision`，12 用例 | 2 PASS + 10 建链失败（环境抖动） |
| 独立精度（第 2 轮） | qrun `z00949060-v2-review-precision2`，10 用例重跑 | ✅ 10/10 PASS（零重试） |
| 独立精度汇总 | cases.csv 全量 10 + T4(4096×8192×4096) + tileCnt=4(16384×4096×2048) | ✅ **12/12 PASS**，rel_err max ≤ 6.4e-3 < 1e-2 |
| 独立性能 | qrun `z00949060-v2-review-perf`，msprof task-based + L2 flush，P1 用例 | ✅ 60.344us（4 卡均值），cube 90.9%~91.0% |
| 共享层一致性 | git show v1 分支 → 逐文件 diff `block/ tiling/ basic/ utils/` | ✅ 全部 identical |
| 审查现场清理 | 删除 `st/build_review/`、`st/.../input|output/` | ✅ 工作区无残留 |

### 8. 结论

**判定：PASS（98/100）**

v2 的 3 级流水架构（mm→comm→reduceSum，PIPELINE_NUM=4 workspace 轮转）实现与 DESIGN.md（含 §2.16 design_issue 决策）完全一致；APACE 路线 R1-R8 红线全部通过；WALKTHROUGH 问题 4 的 AtomicAdd 修正已正确落实（workspace 中已是 L0C 跨 rank 累加后的最终结果，reduceSum 仅普通 DataCopyPad 搬运，无 AtomicAdd、无 yGm 清零）；AIV 复用 BlockScheduler 的 6 项复刻要求逐项落实，flag 三通道配对计数平衡；独立精度 12/12 PASS、独立性能 60.3us（≈2.37× vs v1）复现 Developer 结论。

建议进入 Step 6 精度与性能验收。N1-N6 为可选改进项，不阻塞。
