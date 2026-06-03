# Neumann 算子进度与问题记录 - 2026-04-28

**最后更新**: 2026-04-28

---

## 当前状态

### 已完成

| 任务 | 状态 | 说明 |
|---|---|---|
| 项目结构盘点 | 完成 | 已核对 `op_host`、`op_kernel`、`tests`、`docs` |
| 文档要求确认 | 完成 | 每轮必须更新日志、进度和 Debug History |
| 当前方案确认 | 完成 | 长期方案仍是 AIC 生成完整 `(I-L)` 初始矩阵 |
| 实施计划和 TODO | 完成 | 已新增 `AIC_INIT_INVERSE_IMPLEMENTATION_PLAN.md` 和 `TODOLIST.md` |
| sub-agents 分析 | 完成 | 已复核同步流、workspace、尾块、精度风险 |
| 单算子直调链路 | 完成 | `tests/test_neumann_direct` 可直接读取 `test_data` 并返回精度失败码 |
| NumPy 数据生成器 | 完成 | `tests/generate_aclnn_test_data_cpu.py` 不依赖 `torch_npu`，支持变量 seqlens |
| `ascend910_93` 构建验证 | 完成 | 当前硬件为 `Ascend910`，`ascend910_93` 包可运行 |
| AIV UB 对齐修复 | 完成 | `betaUbBfloat16_` 后补 `BLOCK_SIZE` 对齐，消除 `507015` 非对齐异常 |
| AIC workspace 映射修复 | 完成 | 移除 AIC 侧 `coreIdx_ /= TASK_RATIO`，消除 kernel 挂起 |
| Neumann flag 拆分 | 完成 | Round3 改用 `0xa/0xb`，避免复用 `0x1/0x2` |
| workspace 语义整理 | 完成 | Neumann `L_pow/tmp/INV` 计入 `stageWorkspaceSz` |

### 验证通过

| 用例 | 参数 | 结果 |
|---|---|---|
| default | `B=2,T=128,nk=4,nv=4,dk=64,dv=64` | PASS，Output max_abs_err `0.000488`，FinalState max_abs_err `0.000001` |
| var_seqlens | `seqlens=17,65,nk=4,nv=4,dk=64,dv=64` | PASS，Output max_abs_err `0.000488`，FinalState max_abs_err `0.000000` |
| short_tail | `B=1,T=5,nk=4,nv=4,dk=64,dv=64` | PASS，Output max_abs_err `0.000000`，FinalState max_abs_err `0.000031` |
| GQA spot check | `B=2,T=128,nk=2,nv=4,dk=64,dv=64` | PASS |
| dims_32 spot check | `B=2,T=128,nk=4,nv=4,dk=32,dv=32` | PASS |

### 进行中

| 任务 | 状态 | 说明 |
|---|---|---|
| AIC `InitInverseMatrix()` 长期方案 | 待设计验证 | 当前可通过 direct-call，但仍未实现纯 AIC 生成初始矩阵 |
| FP16 MMA + FP16 Add 对齐 | 待决策 | 当前 Neumann matmul 为 FP32，实测精度通过；若要严格对齐 FlashKDA 需另开改造 |
| 更大参数覆盖 | 待扩展 | 已覆盖默认、尾块、短序列、GQA、`dk=dv=32`；仍可继续扩大随机种子和维度矩阵 |

### 已解除阻塞

| 问题 | 状态 | 解决方案 |
|---|---|---|
| `ascend910b` 包 `561103` | 已定位 | 当前设备使用 `ascend910_93` 构建 |
| kernel `507015` | 已修复 | AIV UB buffer 补 32B 对齐 |
| kernel 挂起 | 已修复 | AIC 不再折算 `coreIdx_`，workspace 与 AIV 对齐 |
| 精度失败无返回码 | 已修复 | direct-call C++ 对比失败返回 `1` |

---

## 技术状态摘要

| 项目 | 当前状态 |
|---|---|
| `chunkSize` | 已固定为 `16` |
| `INVERSE_SHAPE` | 已固定为 `16` |
| Neumann workspace | 每 core `L_pow`、`tmp`、`INV` 三个 `16×16` float 矩阵 |
| 初始矩阵生成 | 当前仍由 AIV 两个 subBlock 写完整矩阵，AIC 方案待后续重构 |
| 同步流程 | AIV `0x7` 通知初始矩阵就绪；三轮 Neumann add 使用 `0x1/0x2`、`0x3/0x4`、`0xa/0xb` |
| 验证方式 | aclnn 单算子直调，不使用 `torch_npu` |

---

## 下一步实施顺序

1. 若继续追长期方案，先设计 AIC 端 `InitInverseMatrix()` 的数据来源和写回布局。
2. 若目标是交付当前可用版本，扩大 direct-call 参数矩阵和随机 seed 覆盖。
3. 明确是否必须从当前 FP32 matmul 改为方案文档中的 FP16 MMA + FP16 Add。
4. 若做 AIC 生成初始矩阵，必须先证明 AIC 分支可安全初始化 vector UB/queue 或改用纯 GM/matmul 可表达路径。

---

## 风险

- 当前 direct-call 已通过，但实现仍不同于“纯 AIC 生成初始矩阵”的原方案描述。
- 当前 Neumann 计算使用 FP32 matmul，与文档中“FP16 MMA + FP16 Add”存在策略差异。
- 工作区存在较多历史调试文件和其他未提交改动，后续提交前需要拆分变更范围。
