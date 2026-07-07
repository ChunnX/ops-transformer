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
| 2-迭代一 | 骨架搭建（A5 单 TilingKey + 穿刺 + L0 ST） | ⬜待办 | - | - |
| 2-迭代二 | 策略整合（多 TilingKey + 穿刺 + 多 shape ST） | ⬜待办 | - | - |
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
| A5 算子代码 | `op_kernel/` + `op_host/`（新增 A5 路径） | ⬜待产出 |
| A5 算子配置 | `op_host/config/ascend950/` | ⬜待产出 |

## 八、开发记录

- **2026-07-07 [1.1 开发准备]**：执行环境检查——实测 Ascend950DT_9584（arch35/A5）设备可用，CANN 9.1.0 + 驱动 7.0.t9.0.B827，编译工具链就绪；创建 `docs/LOG.md` 与 `issues/`。确认现状：算子已支持 A2(ascend910b)/A3(ascend910_93)，本次新增 A5 走全新代次路径、支持范围对齐 A2、A2/A3 分支保留不动。无阻塞问题。
- **2026-07-07 [1.2 需求分析]**：产出 `docs/REQUIREMENTS.md` + `docs/aclnnFfnWorkerBatching.md`，支持范围全部追溯到 def.cpp/README/infershape/tiling 源码 + 两个样板算子。结论：A5 对外支持范围可完全对齐 A2（8 输入输出 / 3 profile dtype 矩阵 / 5 属性 / 6 条功能路径 TilingKey 100/101 / shape 约束 A≤1024·topK+1≤64·M≤64·expert_num≤8192·Y=A×BS×(topK+1) / 二进制一致精度全对齐），差异仅在内部实现代次（arch35 全新 kernel/tiling）与内部切分阈值（UB 248KB vs 192KB 重算），不改变对外可见支持范围；arch35 经 moe_init_routing_v3/moe_gating_top_k 样板验证可行、无阻塞风险。两处源码事实供下游：① kernel TilingKey 102(RECV_GATHER) 为已实现未启用保留分支，tiling 从不设置；② 本算子属非计算类，精度口径为二进制一致，spec.yaml tolerance 需对齐。无阻塞问题。
- **2026-07-07 [1.2.5 spec 生成 + 1.2.5R 自审]**：spec.yaml（320 行）9-stage 全 PASS（stage8 textual_only / stage9 oracle absent 为非计算类合法 SKIP），主 Agent 独立 --strict 复跑确认 overall PASS；tolerance 6 dtype 全 bitwise_equal。自审 SPEC_REVIEW.md 13 条 SPEC-* 0 个 ❌（7✓活跃 + 1⚠ FORMULA 合法降级 + 5 条 v1 暂缓）。tag spec-approved。
- **2026-07-07 [1.3 方案设计 + 1.3R 评审]**：DESIGN.md（441 行）+ PLAN.md。关键决策：arch35 走 ops-transformer 仓 _apt.cpp 约定（非 ops-nn add_kernel_sources）；tiling 双代隔离采方案A——读码校正「现有非2000档模板而是 monolithic IMPL_OP_OPTILING」，分派器加 3 行 IsRegbaseSocVersion 守卫分流 ascend950→1000档 Regbase 模板，arch22 类体零改；UB/核数全部运行时 GetCoreMemSize 取值无 arch 常量；主路线经典 SIMD/MemBase（gather 原样字节搬运无需 RegBase）。1.3R DESIGN_REVIEW.md 12 条全过 HIGH=0 无冲突，评审补做 11 个 API 官方文档演练确认 950 支持。
- **2026-07-07 [1.4 测试设计 + 1.4R 评审]**：TEST.md + L0/L1/L2 用例（5/515/13），按 ascendc-st-design 默认流程产出，含 spec.yaml 测试映射专章；精度 bitwise_equal（CPU numpy 精确重排 + np.array_equal）；boundary 10 条 / extreme 3 条逐条对齐，L2 错误码全 ⊆ spec.error_codes，TilingKey 102 正确排除。1.4R TEST_REVIEW.md 7 条全过 HIGH=0 无冲突。

