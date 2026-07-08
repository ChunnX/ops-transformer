# FfnWorkerBatching 开发日志（LOG）

> 本日志由 Ascend C 算子开发工作流维护，记录 FfnWorkerBatching 算子「新增 Ascend950 (arch35/A5) 支持」的完整开发过程。

## 一、算子基本信息

| 项 | 值 |
|----|----|
| 算子名称 | ffn_worker_batching |
| OpType | FfnWorkerBatching |
| 所属域 | ffn（MoE / FFN 分离部署） |
| 生产仓路径 | `/home/zl/ops-transformer/ffn/ffn_worker_batching/` |
| 当前已支持芯片 | Atlas A2（Ascend910B）、Atlas A3（Ascend910_93） |
| 本次新增芯片 | Ascend950（arch35，即 A5；SoC 实测 Ascend950DT_9584） |
| 功能概述 | Attention 与 FFN 分离部署场景下，在 FFN worker 上按专家维度聚合并重排 token，产出各专家连续 token 数据块（含 group_list、session/micro_batch/token ids、dynamic_scale 等） |
| 调用方式 | 图模式（GEIR）+ 单算子模式（aclnn） |

## 二、用户原始需求（全文）

> 为 ops-transformer 仓已有算子 FfnWorkerBatching 新增 Ascend950(arch35/A5) 支持，A5 的 kernel/tiling 是全新代次路径；支持范围（dtype/shape/输入输出/功能路径）对齐现有 A2/A3 实现，A2/A3 分支保留不动。

补充说明（来自任务下发）：
- 这是一个**已存在的算子**，当前已支持 Atlas A2/A3（Ascend910B / Ascend910_93）。
- 本次任务：**为该算子新增 Ascend950 (arch35, 即 A5) 支持**。A2/A3 现有实现原样保留不动，新增 A5 支持范围对齐 A2。
- 产出目录（LOG/issues/docs）落在该真实算子目录下，不另建 operators/ 目录。

## 三、当前算子现状（新增前基线）

算子已在生产仓实现并支持 A2/A3，关键现状如下：

- **芯片配置**（`op_host/ffn_worker_batching_def.cpp`）：`AICore().AddConfig("ascend910_93").AddConfig("ascend910b")`——仅 A3 + A2。
- **算子配置目录**（`op_host/config/`）：仅 `ascend910b/`、`ascend910_93/` 两套（各含 `ffn_worker_batching.json` + `ffn_worker_batching_simplified_key.ini`）。**无 ascend950 目录**。
- **README 产品支持情况**：`Ascend 950PR/Ascend 950DT` 标记为 `×`（不支持）；A2/A3 标记为 `√`。
- **输入/输出/属性**：输入 `schedule_context`(INT8)；输出 `y`(FP16/BF16/INT8)、`group_list`(INT64)、`session_ids/micro_batch_ids/token_ids/expert_offsets`(INT32)、`dynamic_scale`(FP32)、`actual_token_num`(INT64)；属性 `expert_num`/`max_out_shape`/`token_dtype`/`need_schedule`/`layer_num`。
- **Kernel 现状**（`op_kernel/`，16 个文件）：主入口 `ffn_worker_batching.cpp`，配套 `ffn_wb_common.h`、`ffn_wb_gather_out_all.h`、`ffn_wb_group_listing.h`、`ffn_wb_scan_*`、`ffn_wb_sort_*` 等——当前均面向 A2/A3 代次实现。
- **Host 现状**（`op_host/`）：`ffn_worker_batching_tiling.cpp/.h`、`ffn_worker_batching_infershape.cpp`、`ffn_worker_batching_def.cpp`。
- **测试现状**（`tests/`）：仅 `tests/ut/op_host/`（infershape + tiling 两个 UT）。暂无 st/ 目录。
- **图/示例**：`op_graph/ffn_worker_batching_proto.h` + `fusion_pass/`；`examples/test_geir_ffn_worker_batching.cpp`。

> ⚠️ 现状说明：本次为**已存在算子新增芯片支持**，非全新算子开发。A2/A3 分支（def 配置、config 目录、现有 kernel/tiling 逻辑）保留不动，A5 走全新代次路径。

## 四、本次目标（新增 A5 支持）

- 为 FfnWorkerBatching **新增 Ascend950（arch35/A5）支持**，A5 的 kernel/tiling 为**全新代次路径**（arch35 与 A2/A3 指令/资源模型不同，不能直接复用 A2/A3 kernel）。
- **支持范围对齐 A2**：dtype 矩阵、shape 约束、输入输出定义、功能路径（need_schedule 0/1、token_dtype 0/1/2）与现有 A2 实现一致。
- **A2/A3 分支保留不动**：`def.cpp` 现有 `AddConfig("ascend910_93")/("ascend910b")` 不删改，仅新增 A5 配置；`config/ascend910b`、`config/ascend910_93` 不改；现有 A2/A3 kernel/tiling 逻辑不改。
- 交付：A5 可编译、可在真实 Ascend950 NPU 上运行并通过精度验收；README 产品支持情况更新 Ascend950 为 √。

## 五、环境检查结果

**检查时间**：2026-07-07　**工具**：`ascendc-env-check` skill

| 检查项 | 结果 | 详情 |
|-------|------|------|
| **Ascend950 设备是否可用** | ✅ **可用** | 运行时检测 SoC = `Ascend950DT_9584`，NpuArch = `3510`（`dav-3510`，即 arch35/A5）；设备节点 `/dev/davinci0` + `/dev/davinci_manager` 存在；`halGetChipInfo` 查询成功 |
| NPU 架构（runtime） | dav-3510 (arch35) | `get_npu_arch.py` 经 `libascend_hal.so` 实测返回 |
| 驱动版本 | 7.0.t9.0.B827 | `/usr/local/Ascend/driver/version.info`（timestamp 20260513），ascendhal 7.35.23 |
| CANN 包版本 | 9.1.0 | Ascend-cann-toolkit 9.1.0（innerversion V100R001C11B134），含 Ascend950 platform_config（Ascend950DT/PR 多款 .ini，NpuArch=3510） |
| ASCEND_TOOLKIT_HOME / ASCEND_HOME_PATH | `/root/Ascend/cann-9.1.0` | 需 `source /root/Ascend/ascend-toolkit/set_env.sh` 后生效 |
| ASCEND_OPP_PATH | `/root/Ascend/cann-9.1.0/opp` | source 后生效 |
| ASC_DEVKIT_DIR | 未设置（空） | set_env.sh 未导出该变量；如后续步骤需要 devkit（API 文档/模板）须另行确认 |
| 编译工具链 | ✅ 可用 | gcc 13.3.0、cmake 3.28.3、python3 3.12.3；`ccec`/`opc` 位于 `/root/Ascend/cann-9.1.0/bin/` |
| npu-smi | ⚠️ 不可用 | 容器内无 `npu-smi` 命令，设备健康/占用无法用 npu-smi 查询；但 HAL 接口正常，芯片型号已确认 |

**关键结论**：
- ✅ **Ascend950 设备可用**：实测芯片 `Ascend950DT_9584`（arch35/A5），设备节点存在，HAL 查询成功——满足本次 A5 开发与 NPU 实机验证的前提。
- ⚠️ 默认新开 shell **未 source** CANN 环境；后续所有编译/运行步骤前须先 `source /root/Ascend/ascend-toolkit/set_env.sh`。
- ⚠️ `npu-smi` 不可用、`ASC_DEVKIT_DIR` 为空——不阻塞 1.1，但在需要设备健康检查或 devkit 资源的阶段需留意（届时如成为阻塞项再单独归档 issue）。

## 六、开发状态

| 阶段 | 任务 | 状态 | 完成时间 | 备注 |
|------|------|------|---------|------|
| 1.1 | 开发准备 | ✅完成 | 2026-07-07 | LOG.md + issues/ 已建；环境检查完成，Ascend950 可用 |
| 1.2 | 需求分析 | ✅完成 | 2026-07-07 | REQUIREMENTS.md + aclnn 接口文档已产出；A5 对外支持范围全对齐 A2，arch35 可行 |
| 1.2.5 | spec 生成 | ✅完成 | 2026-07-07 | spec.yaml 9-stage 全 PASS（stage8/9 非计算类合法 SKIP）；主 Agent 独立复跑确认 |
| 1.2.5R | spec 自审 | ✅完成 | 2026-07-07 | SPEC_REVIEW.md：13 条 SPEC-* 0 个 ❌（7✓+1⚠+5暂缓）；tag spec-approved |
| 1.3 | 方案设计 | ✅完成 | 2026-07-07 | DESIGN.md + PLAN.md；arch35 _apt.cpp 入口 + tiling 方案A隔离（分派器3行守卫）+ UB运行时取值 |
| 1.3R | 方案评审 | ✅完成 | 2026-07-07 | DESIGN_REVIEW.md：12 条全过，HIGH=0，无 spec 冲突；3MED+2LOW 非阻塞 |
| 1.4 | 测试设计 | ✅完成 | 2026-07-07 | TEST.md + L0/L1/L2 用例(5/515/13)；spec.yaml 测试映射齐备 |
| 1.4R | 测试设计评审 | ✅完成 | 2026-07-07 | TEST_REVIEW.md：7 条全过，HIGH=0，无 spec 冲突；2MED(标签订正)+2LOW 非阻塞 |
| 2-迭代一 | 骨架搭建（A5 单 TilingKey + 穿刺 + L0 ST） | ✅完成 | 2026-07-08 | A1-Main骨架✅+B✅(10/10)+4穿刺✅全NPU二进制一致+UT轨✅(10/10)；汇合验证✅(NPU 8路二进制一致连跑3次)；测试工程师验收✅(ST 100%)。环境切/home/zl/Ascend、验证转TTK+Kernel直调 |
| 2-迭代二 | 策略整合（多 TilingKey + 穿刺 + 多 shape ST） | 🔄进行中 | 2026-07-08 | A1-Main✅ NORM100+RECV101 双 TilingKey 整合(arch35 补9头共15+apt双分发, ascend910b A2零回归, UB216/248收敛issue✅)；A2 UT✅18/18(NORM/RECV分派+单多核阈值+RECV限核双边界+A2/A3守卫)；B✅543用例(NORM357+RECV184,3profile,含上界shape)；穿刺 int8+RECV/上界shape+UB不溢出/稀疏mask ✅ NPU二进制一致, k=0暴露group_listing缺陷已单行修复+验证+归档(迭代三整合)；汇合验证进行中(重build坐实ascend950二进制+NPU双路径ST) |
| 2-迭代三 | 全量覆盖（全功能 + 全覆盖 UT + 全量 ST） | ⬜待办 | - | - |
| 2-白盒 | 白盒测试生成与用例汇合 | ⬜待办 | - | - |
| 2-PyTorch ST | PyTorch ST 测试开发（L0+L1） | ⬜待办 | - | - |
| 3.1 | 最终精度验收 | ⬜待办 | - | 真实 Ascend950 NPU |
| 3.2 | 性能达标验收（可选） | ⬜待办 | - | 视需求是否含性能指标 |
| 4.1 | 文档与示例 | ⬜待办 | - | README 更新 A5 = √ |
| 4.2 | 代码检视 | ⬜待办 | - | 全量检视 + 设计一致性 |
| 4.3 | 开发总结 | ⬜待办 | - | - |

## 七、交付物清单

| 交付物 | 路径 | 状态 |
|-------|------|------|
| 开发日志 | `docs/LOG.md` | ✅已创建 |
| 问题记录目录 | `issues/` | ✅已创建（含 .gitkeep） |
| 需求分析文档 | `docs/REQUIREMENTS.md` | ✅已产出 |
| aclnn 接口文档 | `docs/aclnnFfnWorkerBatching.md` | ✅已产出 |
| L0 数学契约 | `docs/spec.yaml` | ✅已产出 |
| spec 自审报告 | `docs/SPEC_REVIEW.md` | ✅已产出 |
| 详细设计文档 | `docs/DESIGN.md` | ✅已产出 |
| 迭代执行计划 | `docs/PLAN.md` | ✅已产出 |
| 方案评审报告 | `docs/DESIGN_REVIEW.md` | ✅已产出 |
| 测试设计文档 | `docs/TEST.md` | ✅已产出 |
| 测试用例 | `tests/st/testcases/`（L0/L1/L2 CSV + 覆盖报告） | ✅已产出 |
| 测试设计评审报告 | `docs/TEST_REVIEW.md` | ✅已产出 |
| A5 算子代码 | `op_kernel/`（arch35/ 6头 + _apt.cpp 入口）+ `op_host/`（tiling_arch35 + def/tiling 守卫） | ✅已产出（骨架，新环境复验中） |
| A5 算子配置 | `op_host/config/ascend950/`（json + simplified_key.ini） | ✅已产出 |
| L0 C++ ST 工程 | `tests/st/`（test_aclnn + CMakeLists + run.sh） | ✅已产出（Mock 10/10） |
| 穿刺工程 | `probe/`（probe1~4 + PROBE_SUMMARY.md） | 🔄穿刺3/4✅NPU，1/2待TTK复核 |
| TTK 使用参考 | `docs/TTK_GUIDE.md` + skill `resources/ttk-usage.md` | 🔄产出中 |

## 八、开发记录

- **2026-07-07 [1.1 开发准备]**：执行环境检查——实测 Ascend950DT_9584（arch35/A5）设备可用，CANN 9.1.0 + 驱动 7.0.t9.0.B827，编译工具链就绪；创建 `docs/LOG.md` 与 `issues/`。确认现状：算子已支持 A2(ascend910b)/A3(ascend910_93)，本次新增 A5 走全新代次路径、支持范围对齐 A2、A2/A3 分支保留不动。无阻塞问题。
- **2026-07-07 [1.2 需求分析]**：产出 `docs/REQUIREMENTS.md` + `docs/aclnnFfnWorkerBatching.md`，支持范围全部追溯到 def.cpp/README/infershape/tiling 源码 + 两个样板算子。结论：A5 对外支持范围可完全对齐 A2（8 输入输出 / 3 profile dtype 矩阵 / 5 属性 / 6 条功能路径 TilingKey 100/101 / shape 约束 A≤1024·topK+1≤64·M≤64·expert_num≤8192·Y=A×BS×(topK+1) / 二进制一致精度全对齐），差异仅在内部实现代次（arch35 全新 kernel/tiling）与内部切分阈值（UB 248KB vs 192KB 重算），不改变对外可见支持范围；arch35 经 moe_init_routing_v3/moe_gating_top_k 样板验证可行、无阻塞风险。两处源码事实供下游：① kernel TilingKey 102(RECV_GATHER) 为已实现未启用保留分支，tiling 从不设置；② 本算子属非计算类，精度口径为二进制一致，spec.yaml tolerance 需对齐。无阻塞问题。
- **2026-07-07 [1.2.5 spec 生成 + 1.2.5R 自审]**：spec.yaml（320 行）9-stage 全 PASS（stage8 textual_only / stage9 oracle absent 为非计算类合法 SKIP），主 Agent 独立 --strict 复跑确认 overall PASS；tolerance 6 dtype 全 bitwise_equal。自审 SPEC_REVIEW.md 13 条 SPEC-* 0 个 ❌（7✓活跃 + 1⚠ FORMULA 合法降级 + 5 条 v1 暂缓）。tag spec-approved。
- **2026-07-07 [1.3 方案设计 + 1.3R 评审]**：DESIGN.md（441 行）+ PLAN.md。关键决策：arch35 走 ops-transformer 仓 _apt.cpp 约定（非 ops-nn add_kernel_sources）；tiling 双代隔离采方案A——读码校正「现有非2000档模板而是 monolithic IMPL_OP_OPTILING」，分派器加 3 行 IsRegbaseSocVersion 守卫分流 ascend950→1000档 Regbase 模板，arch22 类体零改；UB/核数全部运行时 GetCoreMemSize 取值无 arch 常量；主路线经典 SIMD/MemBase（gather 原样字节搬运无需 RegBase）。1.3R DESIGN_REVIEW.md 12 条全过 HIGH=0 无冲突，评审补做 11 个 API 官方文档演练确认 950 支持。
- **2026-07-07 [1.4 测试设计 + 1.4R 评审]**：TEST.md + L0/L1/L2 用例（5/515/13），按 ascendc-st-design 默认流程产出，含 spec.yaml 测试映射专章；精度 bitwise_equal（CPU numpy 精确重排 + np.array_equal）；boundary 10 条 / extreme 3 条逐条对齐，L2 错误码全 ⊆ spec.error_codes，TilingKey 102 正确排除。1.4R TEST_REVIEW.md 7 条全过 HIGH=0 无冲突。
- **2026-07-08 [2-迭代一 第一波]**：CP2 通过，并行发起 A1-Main + 4 穿刺 + B。①A1-Main✅：A5 arch35 骨架（100/NORM/fp16/单核 sort）编译通过，ascend950+ascend910b 双路径 kernel 二进制生成，A2/A3 零回归（opFile 隔离 + IsRegbaseSocVersion 守卫实测生效）；落 arch35/ 6 头 + _apt.cpp 入口 + tiling_arch35 + config/ascend950/。DESIGN §3.6.2 宏注册 `_1000` 在本仓 opc 下 kernel 无法解析类型，已按 §3.2 手写平铺 struct 落地（编译即过），意图完整保留。②B✅：C++ ST 工程落盘，Mock 编译 + CPU golden 自测 10/10 PASS（L0_002 need_schedule=1 RECV 路径正确 SKIP，非迭代一范围）。③穿刺3(int8/P2)✅、穿刺4(bf16/P1)✅：真实 Ascend950 NPU 8 路输出二进制一致 + 确定性通过；穿刺4 补齐 GatherMask arch35 实证，A2 经典 SIMD 全栈直编 dav-3510 正确，主路线无需 RegBase。穿刺1(RECV)/穿刺2(多核 MrgSort) 旧环境已跑通（待复核）。
- **2026-07-08 [环境/工具转向 TTK]**：用户决策——① CANN 环境统一改用 `/home/zl/Ascend`（不再用 /root/Ascend）；② kernel/aclnn/E2E 验证统一走 **TTK（ops-test-kit，`/home/zl/ops-test-kit`）**，CSV 用例驱动 + golden 插件，不再手搓 .asc Kernel 直调；③ 已 NPU 通过的穿刺3/4 及旧环境穿刺1/2 在新环境+TTK 下重跑复核；④ 确立「kernel 二进制优先」验证原则（kernel→aclnn→e2e）。旧环境在飞的 probe1 已 kill、A1-Main 已跑完保留代码。当前正搭 TTK 集成地基（`docs/TTK_GUIDE.md` + skill `resources/ttk-usage.md`），地基绿后铺开穿刺 TTK 复核 + A2 UT。观察项（非阻塞）：穿刺3 kernel 侧 `ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE`=216KB vs 穿刺4 host 侧 `GetCoreMemSize(UB)`=248KB 不一致，主线 tiling 走 248KB 分支，大 shape 下有 UB 溢出风险，迭代二整合前须收敛。
- **2026-07-08 [TTK 集成 + 迭代一收口]**：TTK 地基结论——环境切 `/home/zl/Ascend`、`ttk info` 认到 Ascend950、add.csv 冒烟 PASS、golden/input/CSV 资产就绪；但本算子唯一输入 `schedule_context` 内嵌 4 个 GM 设备绝对地址（二级指针，A2/A3 亦如此，见 `ffn_wb_get_schedule_context.h` 偏移 400/528/544/560），**TTK kernel 模式无地址注入能力、无法直驱**（已证据化，SOC_NOT_SUPPORT）→ 裁定：kernel 级验证用 **Kernel 直调**（host 组装 context），TTK 用于 add 冒烟 + 后续 aclnn/E2E 层。4 穿刺全部新环境 NPU 二进制一致（PROBE_SUMMARY 全 ✅）。**迭代一收口**：汇合验证 ✅（probe5 fp16 骨架真机 8 路二进制一致、连跑 3 次 SHA256 恒定，`iter1-integration-report.md`）；测试工程师验收 ✅（C++ ST 10/10、连跑 2 次确定，`iter1-acceptance-report.md`）；UT 轨 10/10、A2/A3 零回归。UB 216/248 不一致建 tracking issue `issues/issue_20260708_ub-size-host-kernel-mismatch_01.md`（迭代二整合前收敛）。进入迭代二。
- **2026-07-08 [工作流切换 + 完整开发收口 + 代码检视]**：用户改用 `changwei-op-dev` 手册主导（非 ops-registry-invoke-workflow），流水=开发→交付物→检视→出包→抽检→low→full泛化→性能→精度。①**完整开发收口**（dev subagent）：核对确认 arch35 是 A2 忠实全量移植（15 kernel 头逐字节一致、无桩），NORM100+RECV101 双 TilingKey + token_dtype 0/1/2 + 单多核 sort + 边界校验 + UB 运行时取值全接线；k=0 守卫已落主线 arch35 头。补 README(A5→√) + 6 条 arch35 边界负路径 UT。编译 ascend950 --opkernel EXIT0 + 3 binary、ascend910b EXIT0 零回归、op_host UT 24/24 PASS，全程未碰 A2/A3。②**代码检视 ✅**（内联，report `docs/arch35_code_review_summary.md`）：delta 无阻塞级逻辑缺陷；HIGH=1【H1 UB 大 shape 溢出风险未验：tiling_arch35.cpp:220 从 host 248KB 派生 sortLoopMaxElement，穿刺2 说 kernel 216KB 会溢、两文档矛盾，大 shape NPU 未实测→泛化必跑】；M2 多核 group_listing k=0 未 NPU 覆盖；其余 PASS（校验完整/分派隔离/UB运行时取值）。
- **⏸️ 2026-07-08 会话恢复锚点**：用户重启会话，已存盘。**下一步 = changwei step4 出包安装（check_arch35_config → build.sh --pkg --soc=ascend950 → .run 装）→ step5-8 泛化验证（边验边修，H1/M2 大shape+多核k=0 必覆盖）→ 性能/精度 → 上库**。恢复必读 `docs/SESSION_HANDOFF.md`。⚠️ **全部代码未 commit（git 停在阶段 1.4），再崩会丢，尽早 checkpoint**。⚠️ arch35 接线 _apt.cpp vs changwei add_kernel_sources 约定悬项（用户暂 deprioritize）。


- **2026-07-08 [changwei step4 出包安装 ✅]**：会话恢复后先补 checkpoint commit（`4e5fc1aed`，171 产物落地，`tests/reports/*.log` 不进仓）。出包前确认 `check_arch35_config.py` 属 ops-nn 仓约定（找 `scripts/kernel/binary_config/ascendc_config.json`），**ops-transformer 无此文件、天然不适用**——本仓 arch35 进包由 def.cpp `AddConfig("ascend950",regbaseCfg)+ExtendCfgInfo("opFile.value","ffn_worker_batching_apt")` 决定（已核实在位）+ config/ascend950/ 齐备。`bash build.sh --pkg --soc=ascend950 --ops=ffn_worker_batching --vendor_name=custom -j32` → 产 `cann-ops-transformer-custom_linux-x86_64.run`（17:27），包内 `kernel/ascend950/ffn_worker_batching_apt/` **3 个 .o（各 219KB）+ 3 json** 坐实。`./*.run --quiet` merge 安装 EXIT 0 → `opp/vendors/custom_transformer`。**新鲜度三查全绿**：部署 binary 时间戳 17:33=本次装；vendor 内 17 个 arch35 头 + apt.cpp 时间戳 17:33 新于 repo；repo↔vendor arch35 头逐个 diff 全一致、apt.cpp 一致 → 部署产物 = 当前 repo，可在此包验证。**下一步 = step5 whitebox 生成泛化用例 → step6 抽检（含 H1 大shape/M2 多核k=0）→ step7 low → step8 full（出功能 xlsx）**。注：schedule_context 内嵌 GM 绝对地址，TTK kernel 模式不可直驱 → kernel 级走 Kernel直调，aclnn/E2E 走 TTK/装好的包。

- **2026-07-08 [step5前 H1 UB 溢出修复]**：进入泛化前复核检视 H1，发现穿刺2(`probe_iter2_boundary_maxshape`)已**实测坐实**主线 UB 溢出（非"未验"）：主线 `tiling_arch35.cpp` 走 host `GetCoreMemSize(UB)=248KB` 派生 `sortLoopMaxElement=7936`，多核 sort 每核 4-buffer footprint=248KB **超物理 216KB 达 32KB**。issue #01 旧"收敛结论"（248KB 权威、用同一 ubSize 自洽）**被推翻**（其"数学保证"只算单核、漏多核）。**修复**：`GetPlatformInfo()` 中 `GetCoreMemSize(UB)` 后扣减 `UB_SYS_RESERVED_SIZE=32*1024`（对齐同仓 `mhc_pre_sinkhorn_backward` arch35），得 216KB → SLM=6912（穿刺2 实证 FITS）。查证 host `platform_ascendc` 无 vector-core UB 专用接口（`CoreMemType` 仅单一 UB），"标称值扣固定系统预留"是仓内标准做法；不违反「UB size no hardcode」。仅动 `_tiling_arch35.cpp`（+10/-2），A2 老路径 `ffn_worker_batching_tiling.cpp` 零改动。UT 期望值同步（子agent）：253952→221184/6912、262144→229376/7168，A2/A3 DAV_2201(261888/8160) 及注入常量(kArch35Ub248K/MakeArch35Attrs 末参)不动，Y 值无落 (6912,7936] 翻转区。issue #01 + 检视报告 H1 + INTEGRATION_REPORT 已订正。ascend950 --opkernel 重编 EXIT0+3 binary。**op_host UT 复验中**。
