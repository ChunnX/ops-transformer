# 工作日志 - 2026-04-28

**日期**: 2026-04-28  
**主要工作**: 初始化项目盘点，明确当前要求、实现状态和下一步阻塞项

---

## 本次对话内容

### 项目要求确认

- 必须继续按日期维护文档：`docs/YYYY-MM-DD/WORK_LOG.md` 和 `docs/YYYY-MM-DD/PROGRESS_AND_ISSUES.md`。
- 每次对话结束前必须更新日志、进度文档，并在 `AGENTS.md` 的 `Debug History` 追加记录。
- 当前算子目标是完成 `chunk_gated_delta_rule_neumann` 的 CHUNK=16 + Neumann 求逆迁移。
- 已确定方案是将 `(I-L)` 初始矩阵生成从 AIV 移到 AIC，避免 `TASK_RATIO=2` 下 AIV 只处理 `16×8` 半块导致的 workspace 冲突。

### 当前实现状态核对

- `op_host/chunk_gated_delta_rule_neumann_tiling.cpp` 已设置 `chunkSize=16`，并为 Neumann 分配每 core 三个 `16×16` workspace 矩阵。
- `op_kernel/chunk_gated_delta_rule_neumann_stage1.h` 已设置 `INVERSE_SHAPE=16`。
- `tests/pytest/chunk_gated_delta_rule_neumann_golden.py` 默认 `chunk_size=16`。
- `ProcessOneChunk()` 仍保持旧流程：AIC 等待 AIV `InverseCompute()` 通过 flag `0x7` 产生 `INVWsGm_` / `L_powWsGm_`。
- `InverseCompute()` 仍在 AIV 分支中按 `halfChunkSize=8` 处理半块，尚未实现 AIC 生成完整 `16×16` 初始矩阵。

### 当前结论

- 文档中的方案和代码真实状态一致：方案已确定，关键实现未完成。
- 当前主要阻塞项仍是 Kernel 挂起风险，根因是 Neumann AIC 矩阵乘依赖完整 `16×16` 矩阵，但 AIV 初始化只写半块。

### 2. 实施计划与首轮实现

**用户要求**：编写 plan 和 todolist，启动 sub-agents，并开始实现。

**已执行**：
- 启动两个 sub-agents：
  - `Turing`：分析 Stage1 AIC/AIV 同步和最小改造方案，已完成。
  - `Anscombe`：查找相邻算子可复用模式，已完成。
- 新建实施计划：`docs/2026-04-28/AIC_INIT_INVERSE_IMPLEMENTATION_PLAN.md`。
- 新建 TODO：`docs/2026-04-28/TODOLIST.md`。
- 修改 `op_kernel/chunk_gated_delta_rule_neumann_stage1.h`：
  - 修复 `InverseCompute()` 中第二半块 `I-L` 对角线写入索引。
  - 限定 `NeumannAddCompute()` 只有 `subBlockIdx_ == 0` 执行完整 `16×16` 的 `INV += tmp` 写回。

**关键判断**：
- 子代理确认当前 AIC 分支没有初始化 vector UB/queue，不能直接复用 AIV 的 `DataCopyPad`、`Duplicate`、`Muls` 路径。
- 子代理确认完整 `L` 不能只来自 `kkWsGm_`，还必须包含 `beta`、`gamma/decay`、下三角 mask，数据源应来自 AIV `KKBetaCompute()` 后的 `attnUbFloat_`。
- 子代理建议保留 `0x7` 为 AIV→AIC 的“`L_powWsGm_` 就绪”语义，避免 AIC/AIV 互等。
- 因此本轮不强行实现纯 AIC 逐元素初始化，而是先修复确定错误并降低 workspace 竞态。

---

## 决策和原因

| 决策 | 结论 | 原因 |
|---|---|---|
| 保持 `TASK_RATIO=2` | 是 | 避免破坏现有 AIC/AIV 协同架构 |
| 保持 `chunkSize=16` | 是 | 已与 FlashKDA/Neumann 方案对齐 |
| 下一步优先实现 `InitInverseMatrix()` | 是 | 这是解除当前同步和 workspace 冲突的最短路径 |
| 本次不修改 kernel 实现 | 是 | 用户请求是初始化盘点，先明确要求和进度 |
| 首轮实现采用同步/半块修复优先 | 是 | AIC 侧缺少 vector buffer，直接迁移风险高 |

---

## 下一步建议

1. 在 `op_kernel/chunk_gated_delta_rule_neumann_stage1.h` 增加 AIC 版 `InitInverseMatrix()`。
2. 调整 `ProcessOneChunk()` 同步流：AIC 在 `KKBetaCompute()` 后生成 `INVWsGm_` / `L_powWsGm_`，不再等待 AIV `InverseCompute()`。
3. 保留 AIV 后续 `NeumannAddCompute()`、`GBKCompute()`、`VBetaCompute()`、`QPrimeCompute()` 分工，避免一次性重构过大。
4. 修改后执行编译和 pytest 单算子测试验证。
5. 当前首轮补丁后，下一步应先编译验证，再决定是否继续做 AIC vector buffer 初始化。

---

## 文件核对

| 文件 | 状态 |
|---|---|
| `op_host/chunk_gated_delta_rule_neumann_tiling.cpp` | `chunkSize=16` 和 Neumann workspace 已配置 |
| `op_kernel/chunk_gated_delta_rule_neumann_stage1.h` | `INVERSE_SHAPE=16` 已配置，AIC 初始化待实现 |
| `tests/pytest/chunk_gated_delta_rule_neumann_golden.py` | golden 默认 `chunk_size=16` |
| `docs/2026-04-28/AIC_INVERSE_GENERATION_PLAN.md` | 当前方案文档已存在 |
| `docs/2026-04-28/AIC_INIT_INVERSE_IMPLEMENTATION_PLAN.md` | 本轮新增实施计划 |
| `docs/2026-04-28/TODOLIST.md` | 本轮新增 TODO 跟踪 |
| `op_kernel/chunk_gated_delta_rule_neumann_stage1.h` | 本轮首批实现改动 |

---

**最后更新**: 2026-04-28

---

## 3. 单算子直调精度调试

**用户要求**：继续调试算子精度；不要 `torch_npu`，走单算子直调，项目内已有单算子直调精度对比脚本。

**执行内容**：
- 使用 `tests/test_aclnn_chunk_gated_delta_rule_neumann.cpp` / `tests/test_neumann_direct` 作为精度验证入口。
- 新增 NumPy-only 数据生成器 `tests/generate_aclnn_test_data_cpu.py`，不依赖 `torch` 或 `torch_npu`。
- 修复直调测试：优先读取算子本地 `test_data/`，精度失败时返回非 0。
- 使用 `ascend910_93` 构建并安装 custom op 包，匹配当前 `Ascend910` 设备。

**发现的问题**：
- `ascend910b` 包在当前设备上 `GetWorkspaceSize` 失败，错误码 `561103`，plog 指向动态 kernel 配置解析失败。
- `ascend910_93` 包首次运行进入 kernel 后报 `507015`，plog 指向 AIV UB 非对齐访问。
- 修复 UB 对齐后出现 kernel 挂起，定位为 AIC 侧错误执行 `coreIdx_ /= TASK_RATIO`，导致 AIC/AIV workspace 分片不一致。

**修复内容**：
- `op_kernel/chunk_gated_delta_rule_neumann_stage1.h`：
  - `betaUbBfloat16_` 后将 `buffOffset` 对齐到 `BLOCK_SIZE`，修复 AIV vector UB 非对齐。
  - 移除 AIC 侧 `coreIdx_ /= TASK_RATIO`，恢复 AIC/AIV workspace 映射一致性。
  - Round3 Neumann 同步 flag 从复用 `0x1/0x2` 改为独立 `0xa/0xb`。
  - Neumann `L_pow/tmp/INV` 使用独立 GM workspace，并通过 `attnWsGm_` / `L_powWsGm_` ping-pong。
- `op_host/chunk_gated_delta_rule_neumann_tiling.cpp`：
  - 将 Neumann 三个 `16×16` float workspace 计入 `stageWorkspaceSz`，与 `stageWsAddr_` 实际使用一致。

**验证结果**：
- `default`：`B=2,T=128,nk=4,nv=4,dk=64,dv=64`，PASS；Output max_abs_err `0.000488`，FinalState max_abs_err `0.000001`。
- `var_seqlens`：`seqlens=17,65`，PASS；Output max_abs_err `0.000488`。
- `short_tail`：`B=1,T=5`，PASS；FinalState max_abs_err `0.000031`。
- 额外 spot check：GQA `nk=2,nv=4` 和 `dk=dv=32` 均 PASS。

**方案文档**：
- 新增 `docs/2026-04-28/DIRECT_ACLNN_PRECISION_FIX_PLAN.md`。
