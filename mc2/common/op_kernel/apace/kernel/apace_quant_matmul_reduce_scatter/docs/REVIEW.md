# apace_quant_matmul_reduce_scatter_v3 算子审查报告

## Round 0 审查报告（Step 4 初审）

- **审查日期**：2026-08-12
- **审查人**：Reviewer（独立审查，全部结论基于独立构建/独立上板实测，不信任 Developer 自报）
- **判定**：**PASS**
- **总分**：**98 / 100**

---

### 1. 审查范围与方法

| 项 | 内容 |
|----|------|
| 审查对象 | `operators/apace_quant_matmul_reduce_scatter_v3/`（v2 的外科手术式优化：取消 workspace 轮转回压） |
| 对照基底 | `operators/apace_quant_matmul_reduce_scatter_v2/`（已 PASS 98/100） |
| 独立验证环境 | 4 × Ascend950PR（dav-3510），CANN 9.2.0（`source /home/l00947526/Ascend806/cann/set_env.sh`） |
| NPU 排队 | 全部 NPU 任务经 qrun 提交（实际绑定工号为 **c00697382**，与 prompt 所述 z00949060 不一致，以 `qrun whoami` 实测为准，任务名前缀按实际工号） |
| 加载 Skill | ascendc-mc2-best-practice / ascendc-blaze-best-practice / ascendc-code-review / npu-arch / ascendc-docs-search / ops-precision-standard |

**审查方法**：v2↔v3 全量 diff 核对变更清单 → APACE 红线 R1-R8 grep → v3 特有审查重点逐项核对 → qrun 独立清洁编译 → qrun 独立精度（13 用例 + T13 双校验）→ qrun 独立性能（msprof，v2/v3 交错对照）。

---

### 2. 独立验证结果（qrun 实测）

#### 2.1 独立编译（任务 c00697382-v3-review-build）

- **方式**：`rm -rf st/build` 后全量重建（不依赖 Developer 构建产物），并运行主用例冒烟。
- **结果**：✅ 编译成功，主用例 4 卡精度 PASS（rel_err max ≈ 6.1e-3）。
- **备注**：`KERNEL_OPTS` 含 `-w` 全局警告抑制（继承 v2 已审查配置），编译输出无法用于确认代码级警告，见 §6 扣分说明。

#### 2.2 独立精度（任务 c00697382-v3-review-precision / t13 / t13b）

cases.csv 全量 13 用例独立复跑：**13/13 PASS**（rtol=atol=1e-2，BF16 输出）：

| Row | Shape (M,K,N) | dtype | 结果 | max rel_err |
|-----|---------------|-------|------|-------------|
| 1 | 2048,4096,2048 | e4m3e4m3 | ✅ PASS | ≤6.1e-3 |
| 2 | 512,1024,512 | e4m3e4m3 | ✅ PASS | ≤6.1e-3 |
| 3 | 1120,2048,1024 | e4m3e4m3 | ✅ PASS | ≤6.1e-3 |
| 4 | 2048,4608,2048 | e4m3e4m3 | ✅ PASS | ≤6.1e-3 |
| 5 | 2048,4096,2048 | e5m2e5m2 | ✅ PASS | ≤6.4e-3 |
| 6 | 1120,2048,1024 | e5m2e5m2 | ✅ PASS | ≤6.2e-3 |
| 7 | 2048,4096,2048 | e4m3e5m2 | ✅ PASS | ≤6.1e-3 |
| 8 | 1120,2048,1024 | e4m3e5m2 | ✅ PASS | ≤6.1e-3 |
| 9 | 2048,4096,2048 | e5m2e4m3 | ✅ PASS | ≤6.2e-3 |
| 10 | 1120,2048,1024 | e5m2e4m3 | ✅ PASS | 0（全等） |
| 11 (T11) | 16384,4096,2048 | e4m3e4m3 | ✅ PASS | ≤6.4e-3 |
| 12 (T12, tiles/core=10>4) | 16384,4096,4096 | e4m3e4m3 | ✅ PASS | ≤6.5e-3 |
| 13 (T14, 尾块拆分) | 2252,2048,1024 | e4m3e4m3 | ✅ PASS | ≤5.6e-3 |

T13 校验用例（CLI 独立验证）：

| 用例 | Shape | 预期 | 独立实测 |
|------|-------|------|----------|
| T13-a | 32768,4096,8192（T=37>15） | host 拒绝 | ✅ 正确拒绝，报错清晰：`maxTilesPerCore=37 > 15 (blockDimM=32 x blockDimN=32 = blockNums=1024, coreNum=28, ...)`，提示回退 v2 |
| T13-b | 10240,4096,10752（T=15 边界） | 恰好放行 | ✅ 放行（日志确认 `maxTilesPerCore=15`，workspace=55,050,240B），4 卡全 PASS，rel_err max 6.5e-3 |

> 首次 T13-b 运行曾被审查端 shell 超时中断（非算子失败），重跑确认 PASS。

#### 2.3 独立性能（任务 c00697382-v3-review-perf / perf-main / v2-review-perf-main / v3v2-interleave / v2-perf-t12）

msprof task-based 采集（L2 flush 由 heavy kernel 完成），独立实测并与 v2 同条件对照：

| 用例 | v3 独立实测 | v2 独立实测（同晚对照） | Developer 归档（v3 / v2） | 判定 |
|------|------------|------------------------|---------------------------|------|
| T12 大 shape 16384,4096,4096（T=10>4，核心验收） | **265.97 us**（cube 95.2-95.6%） | **288.48 us** | 263.56 / 291.45 us | ✅ **v3 较 v2 -7.8%（独立实测）/ -9.6%（归档）**，回压消除收益确认，无回归 |
| 主用例 2048,4096,2048（T=1） | 受环境干扰（见下） | 59.66 us（21:13）/ 101.82 us（21:16，退化态） | 57.56 / 57.63 us | ⚠️ 见下注 |
| T11 16384,4096,2048（T=5） | —（未独立复测） | — | 198.44 / 199.22 us | 归档持平微优（-0.4%） |

**主用例性能异常调查（重要）**：审查中独立采集主用例时出现 123.996 / 108.43 / 116.97 / 119.54 us，显著高于归档 57.6 us。逐项排查结论为**环境退化，非 v3 代码回归**：

1. **v2 同条件同步退化**：交错实验中 v2 亦从 59.66 us 退化至 101.82 us，且后续一次 v2 主用例运行直接超时失败（10min，退出码 1）——退化与算子版本无关；
2. **硬件证据**：`npu-smi` 显示 NPU 0/1 健康状态 Warning（ErrorCode 81AF8009，UB RAS bus error "probably caused by software"）；
3. **数据特征**：退化 run 中慢 rank 随机分布（case4 慢 rank=1/2/3，case6 慢 rank=0/1/2），快 rank 恒为 54.8-57.7 us（与归档基线一致）——非确定性代码路径差异特征；
4. **T12 不受影响**：v3 T12 各 rank 均匀 261-276 us，与归档 263.6 us 一致，且同条件 v2 T12=288.5 us 亦与归档 291.4 us 一致——大 shape（compute-bound）不受该环境退化影响；
5. **代码机理**：T=1 时 v3 相对 v2 严格少做事（删除 PostCompute drain、回压本不触发、槽位寻址 cubeNum=0 时与 v2 完全相同、workspace footprint 相同），不存在 v3 更慢的代码路径。

结论：主用例「与 v2 持平」由归档同条件数据（57.56 vs 57.63 us）+ 代码机理 + 本次快 rank 数据（54.8-57.7 us）三重支撑；**建议设备恢复健康后补测一次主用例 v3/v2 对照归档**（遗留跟进项 N1，非阻塞）。

---

### 3. v3 特有审查重点逐项核对

| # | 审查重点 | 结果 | 证据 |
|---|----------|------|------|
| 1 | 通道 3 回压完整删除（kernel.h Wait(3) + PostCompute drain + impl.h Set(3)） | ✅ | diff 确认 3 处全部删除（kernel.h 原 L403-405 Wait、L217+L507-519 PostCompute 调用与本体、udma_impl.h 原 L399-400 Set）；grep 全工程无 `SYNC_AIV_AIC_FLAG`/`PostCompute` 代码符号残留（仅注释） |
| 2 | 槽位寻址去取模两侧一致（`coreIdx + cubeNum×coreNum`） | ✅ | kernel.h L410-411 与 udma_impl.h L363-365 公式逐字一致；`baseM×baseN×(coreIdx + cubeNum×coreNum)` |
| 3 | maxTilesPerCore 收紧公式（CeilDiv 无 +1）+ host ≤15 校验 | ✅ | main.cpp：`CeilDiv(CeilDiv(m,baseM)×CeilDiv(n,baseN), coreNum)`，四输入全部取自 `tileQbmmTilingData` 最终输出（串讲问题 7 决议落实）；校验在 tiling 后、`aclrtMalloc` 前执行（顺序正确）；T13-a/T13-b 实测双向生效 |
| 4 | pipelineNum 字段全链删除 | ✅ | tiling_data.h 字段→main.cpp 赋值→InitBaseParams→baseParams_→SetupParams→LocalParams→kernel 成员 `pipelineNum_` 全链无残留（grep 仅注释命中）；新字段 `maxTilesPerCore{0}` 仅 host 消费，不向 kernel plumbing（与 C6 设计一致） |
| 5 | 通道 1/2 flag 配对不受删通道 3 影响 | ✅ | diff 确认通道 1（CommFlagIdV3 + drain 兜底）与通道 2（Set(5)/Wait(5)）代码除常量更名 _V3 外零变更；flagId 空间独立（1 通道用 {0,1,2,4,6,7,8,9,10,15}，2 通道用 5，无通道使用 3） |
| 6 | flagId=5 计数器超 15 风险防护 | ✅ | C8 host 校验 `maxTilesPerCore ≤ 15` 已生效（T13-a 实测拒绝 T=37；T13-b 实测 T=15 放行）；校验边界与硬件 0-15 计数器上限精确对齐（严格上界 CeilDiv 无 +1，无误拒） |
| 7 | v2 对照 diff 仅限 C1-C10 + 更名 | ✅ | 全量 diff：kernel 3 文件 = C1/C2/C3/C10 + C4/C5/C9（保持 v2 映射）+ C6；main.cpp = C7/C8 + 更名；kernel_launcher.h = 4 入口更名；cases.csv = +3 行（T11/T12/T14）；run.sh/CMakeLists/parse_prof.py/README = 更名；**共享层 block/tiling/basic/utils/third_party 零 diff** |

### 4. APACE 审查验收条件（R1-R8）

| # | 检查项 | 结果 | 证据 |
|---|--------|------|------|
| R1 | 无 `__schedmode__`/`core_ratio` | ✅ | grep kernel/st 零命中 |
| R2 | `KERNEL_TYPE_MIX_AIC_1_1` | ✅ | kernel_launcher.h 4 个入口均含 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1)` |
| R3 | 4 dtype 入口 | ✅ | E4M3E4M3 / E5M2E5M2 / E4M3E5M2 / E5M2E4M3 四入口 _V3 |
| R4 | block/tiling 未修改 | ✅ | 与 v2 零 diff（v2 已与官网仓对齐并通过审查） |
| R5 | CrossCore flag idx 配对 | ✅ | AIC `SetFlag<2,PIPE_FIX>(5)` ⇔ AIV `WaitFlag<2,PIPE_MTE2>(5)`；通道 1 两侧同用 `CommFlagIdV3` 映射 |
| R6 | UDMA 有 `__gm__ CommContext*` | ✅ | impl Init 与 4 入口均含 |
| R7 | 无 `AscendC::Matmul` | ✅ | grep 零命中（Blaze BlockMmad 路线） |
| R8 | 无 `Hccl::` | ✅ | grep 零命中（host 侧仅用 HcclCommInit/Destroy 建链，非 kernel 高阶 API） |

**通用规范补充检查**：无 `blockDim=N`/`blockIdx=N` 硬编码；无 `DataCopy`（非 Pad）/`SetValue`/`GetValue` 黑名单 API；kernel 层无 `std::`/`new`/`malloc`；核数经 `usedCoreNum`（SWAT tiling 运行时推导）动态获取；`--npu-arch=dav-3510` 与 Ascend950PR/DAV_3510 匹配（/npu-arch skill 确认）。

**CMake 校验说明**：`workflows/scripts/verify_cmake_config.py` 面向标准 add_custom 模板（要求 `find_package(ASC)`/`LANGUAGES ASC CXX`），对 APACE ST 工程构建体系（ascend.cmake + `-xasc --npu-arch=dav-3510` + tiling_api/hccl_fwk 链接，自 v2 继承并已通过审查）不适配，其报错项不适用于本工程；以独立清洁编译成功 + R1-R8 为实际门禁。记录在案，不计缺陷。

### 5. 同步策略逐项依赖分析（强制项）

| 同步点 | 方向/配对 | 依赖关系分析 | 冗余判定 |
|--------|-----------|--------------|----------|
| 通道 1：AIV `SetFlag<0x2,MTE3>(fid)` → AIC `WaitFlag<0x2,MTE2>(fid)` | per-tile + 去重 + 循环末尾 drain 兜底 | AIC 读 Win 区 A 数据（MTE2/mm）必须先于通信完成；Set 在 AllToAll 数据+scale 双通道 Finalize 前发出，MTE3 完成即数据可达 | 必要，无冗余（继承 v2 零变更） |
| 通道 2：AIC `SetFlag<2,PIPE_FIX>(5)` → AIV `WaitFlag<2,MTE2>(5)` | 每 tile 各 1 次 | AIV reduceSum DataCopyPad（MTE2）读 workspace 槽位 k 必须先于 AIC fixpipe 写槽位 k 完成；两侧同一 BlockScheduler 枚举序（复刻 6 项），Set/Wait 次数严格相等（串讲 V5 已证，含早退分支对称跳过） | 必要，无冗余；计数器峰值 ≤ T ≤ 15 由 host 校验保证 |
| 通道 3（v2 回压） | — | **已删除**：槽位独占（每 (coreIdx, cubeNum) 唯一槽位、写 1 次读 1 次）使「等槽位释放」依赖不复存在，删除有充分依据（DESIGN §4.6 无越界论证 + 实测 13/13 PASS） | 删除正确，无残留 |
| MTE2↔MTE3 event（RunReduceSum） | RAW（MTE2→MTE3）/ WAR（MTE3→MTE2）双 event 配对 | UB buffer 单缓冲流水：GM→UB 完成才能 UB→GM，UB→GM 完成才能下一轮 GM→UB 覆写 | 必要，无冗余（继承 v2 零变更） |
| SyncAll\<true\> / TeamBarrier / AllToAll Wait | AIV↔AIV / 跨卡 | 未动 | 无新增依赖环 |

### 6. 评分表

| 维度 | 满分 | 得分 | 说明 |
|------|------|------|------|
| 1. 编译验证 | 10 | **9** | 1.1 独立清洁编译成功 ✅（7/7）；1.2 `KERNEL_OPTS` 含 `-w` 全局警告抑制（继承 v2），无法从输出确认代码级警告（2/3） |
| 2. 架构合规 | 15 | **15** | TPipe/TBuf（AIV reduceSum）、`__global__ __aicore__` + MIX_AIC_1_1、定义顺序、Malloc/Free 配对（含 deviceWorkspace）、数据流完整（mm→workspace→reduceSum→yGm） |
| 3. 编码规范 | 15 | **15** | Blaze/DataCopyPad 合规 API、flagId 约束满足、对齐参数继承 v2 已审查实现、_V3 命名一致 |
| 4. 性能优化 | 20 | **19** | 4.1 动态硬件参数 ✅；4.2 多核切分+空闲核早退 ✅；4.3 3 级流水+L0C db+回压消除 ✅；4.4 同步逐项分析无冗余 ✅；4.5 T12 -7.8% 独立实测确认，主用例复测受环境退化干扰（3/4，见 §2.3 与跟进项 N1） |
| 5. 测试覆盖 | 15 | **15** | gen_data/verify 脚本齐全、13 用例（含 Level 0 小 shape 512 系列 + 大 shape + 尾块 + 双边界校验）、精度标准 1e-2 明确 |
| 6. 精度验证 | 10 | **10** | 独立复测 13/13 PASS + T13 双向校验通过；4 dtype 组合 × BF16 输出全覆盖，rel_err max 6.5e-3 ≪ 1e-2（本算子无 FP32/FP16 输出 dtype，按 BF16 标准应用） |
| 7. 文档 | 15 | **15** | README 含概述/数学公式/路线约束/编译运行/测试结果/已知限制；DESIGN/PLAN/WALKTHROUGH（含用户仲裁记录）完整 |
| **总分** | 100 | **98** | |

**必须修复项检查**：1.1 ✅ / 2.1 ✅ / 2.2 ✅ / 3.1 ✅ / 3.2 ✅ / 4.1 ✅ / 6.1 ✅ —— 全部通过。

### 7. 最终轮附加检查

**交付件（D1-D8，按 APACE 工程布局适配）**：✅ 算子层 3 头文件（kernel/）独立编译通过；构建文件（st/CMakeLists + ascend.cmake）依赖完整；gen_data.py 支持 4 dtype 组合；run.sh 可执行（独立复跑 13 用例 + CLI 校验）；README/DESIGN/PLAN 齐全且 PLAN 全阶段标记完成；本报告（D8）已写入。

**代码清洁（C1-C4）**：✅ kernel/host 无 printf 残留（仅 host 侧必要状态输出 `std::cout`，继承 v2）；无 TODO/FIXME/HACK；无大段注释代码块（v3 变更点注释为说明性，符合「外科手术式优化」可审计要求）；无调试硬编码（关键参数全部 tiling 推导）。

**精度全覆盖**：4 dtype 组合 × 13 shape 用例 + T13 双边界，全部独立实测通过，结果见 §2.2 表。

### 8. 判定与跟进项

**判定：PASS（98/100）**。C1-C10 变更与 DESIGN §1.1 逐项一致；共享层零改动；红线 R1-R8 全过；独立编译/精度/校验/性能（T12 收益）全部实测通过。

**跟进项（非阻塞）**：

| # | 事项 | 说明 |
|---|------|------|
| N1 | 主用例 v3/v2 性能对照补测 | 本次审查期间 NPU 0/1 出现 UB bus error 告警、主用例测量环境退化（v2 同步退化甚至超时），建议设备恢复健康后补测一次主用例（2048,4096,2048）v3/v2 对照并归档至 docs/perf/；当前持平结论由归档同条件数据 + 代码机理 + 快 rank 数据支撑 |
| N2 | `-w` 警告抑制（建议） | 继承 v2 配置；后续版本可考虑去除 `-w` 或改为选择性抑制，使代码级警告可见 |
| N3 | 遗留风险确认（记录） | T12 workspace 35.0MB（usedCoreNum=28 实测）已在 WALKTHROUGH 仲裁中由用户接受（32MB 口径标记遗留）；T13-b 边界用例 workspace 52.5MB 为一次性校验用途，README 已如实记录 |

---

*Round 0 审查完成。后续复审轮次请在本文件末尾追加，禁止覆盖本报告。*
