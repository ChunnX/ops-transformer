# PROJECT EXECUTION LOG - chunk_gated_delta_rule_neumann

完整记录项目从零开始到当前的每一步操作。

---

## Timeline Overview

| 阶段 | 时间 | 主要任务 |
|------|------|----------|
| Phase 1 | 2025-04-27 上午 | 算子克隆与重命名 |
| Phase 2 | 2025-04-27 下午 | 编译安装测试 |
| Phase 3 | 2025-04-27 下午 | 精度调试与 Bug 修复 |
| Phase 4 | 2025-04-27 晚上 | 目录整理与文档 |
| Phase 5 | 2025-04-27 晚上 | CHUNK=16 迁移计划 |

---

## Phase 1: Operator Clone (算子克隆)

### Step 1.1: 创建算子目录结构
**时间**: 2025-04-27 上午
**操作**: 从 `chunk_gated_delta_rule` 克隆新算子

```bash
# 源算子位置
/mnt/workspace/gitCode/cann/ops-transformer/attention/chunk_gated_delta_rule/

# 目标算子位置
/mnt/workspace/gitCode/cann/ops-transformer/attention/chunk_gated_delta_rule_neumann/
```

**文件变更**:
- 复制 `op_host/`, `op_kernel/`, `op_graph/` 目录
- 重命名所有文件名中的 `_def`, `_tiling`, `_stage1` 等后缀

### Step 1.2: 文件重命名清单
**修改文件** (批量替换 `chunk_gated_delta_rule` → `chunk_gated_delta_rule_neumann`):

| 原文件 | 新文件 |
|--------|--------|
| `op_host/chunk_gated_delta_rule_def.cpp` | `chunk_gated_delta_rule_neumann_def.cpp` |
| `op_host/chunk_gated_delta_rule_tiling.cpp` | `chunk_gated_delta_rule_neumann_tiling.cpp` |
| `op_kernel/chunk_gated_delta_rule.h` | `chunk_gated_delta_rule_neumann.h` |
| `op_kernel/chunk_gated_delta_rule_stage1.h` | `chunk_gated_delta_rule_neumann_stage1.h` |
| `op_kernel/chunk_gated_delta_rule_stage2.h` | `chunk_gated_delta_rule_neumann_stage2.h` |
| `op_kernel/chunk_gated_delta_rule_stage3.h` | `chunk_gated_delta_rule_neumann_stage3.h` |
| ... | ... |

### Step 1.3: 内容替换
**文件内部字符串替换**:
- 所有 `chunk_gated_delta_rule` → `chunk_gated_delta_rule_neumann`
- 包括函数名、类名、宏定义、字符串常量

---

## Phase 2: Build & Install (编译安装)

### Step 2.1: 编译算子
**时间**: 2025-04-27 下午
**命令**:
```bash
cd /mnt/workspace/gitCode/cann/ops-transformer
bash build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule_neumann -j16
```

**结果**: ✅ 编译成功
**输出位置**: `build/out/chunk_gated_delta_rule_neumann_*.run`

### Step 2.2: 安装算子
**命令**:
```bash
bash build.sh --install
```

**安装位置**:
```
${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/
```

### Step 2.3: 编写测试程序
**文件**: `tests/test_aclnn_chunk_gated_delta_rule_neumann.cpp`

**测试数据生成**: `tests/pytest/generate_test_inputs.py`

---

## Phase 3: Precision Debug (精度调试)

### Step 3.1: 单 Batch 测试
**测试**: B=1, seqlen=16
**结果**: ✅ PASS, max_abs_err ≈ 0.0001

### Step 3.2: 多 Batch 测试失败
**测试**: B=2, seqlen=16
**现象**:
- 第一 batch 结果正确
- 第二 batch 输出全零
- golden 输出正确，NPU 输出异常

### Step 3.3: Bug 定位 🔴
**关键发现**: `seqlens_data` 类型不匹配！

**问题代码** (`test_aclnn_*.cpp`):
```cpp
// 错误: 存储为 int64_t (8 bytes)
std::vector<int64_t> seqlens_data = {16, 16};
workspace.SetSeqlens(seqlens_data.data(), seqlens_data.size());

// 但内核期望 int32_t (4 bytes)
// 内存布局错误: [16, 16] → 读取成 [16, 0, 16, 0]
// 第二 batch seqlen=0 → 空 tensor → 输出全零
```

### Step 3.4: Bug 修复
**修改**: `int64_t` → `int32_t`

```cpp
// 修复后
std::vector<int32_t> seqlens_data = {16, 16};  // ← 改为 int32_t
```

**验证结果**: ✅ 所有测试 PASS
- B=1, seqlen=16: max_abs_err = 0.00012
- B=2, seqlen=16: max_abs_err = 0.00015
- B=1, seqlen=64: max_abs_err = 0.00021

### Step 3.5: 经验总结
**创建文档**: `SKILL/precision-debugging/SKILL.md`

**核心经验**:
> Host 存储类型必须与 Kernel 读取类型完全一致
> `int64_t` vs `int32_t` 内存布局差异导致多 Batch 失败

---

## Phase 4: Directory Cleanup (目录整理)

### Step 4.1: 清理临时脚本
**删除文件**:
- `tests/pytest/generate_multi_test.py` (临时)
- `tests/pytest/generate_simple_test.py` (临时)
- `tests/pytest/precision_comparison.py` (临时)

### Step 4.2: 移动 examples → tests
**操作**:
- 删除 `examples/` 目录
- 移动测试代码到 `tests/`

### Step 4.3: 统一数据输出路径
**修改**: `generate_test_inputs.py` 输出到 `build/test_data/`

### Step 4.4: 修改 build.sh
**操作**: 添加 `tests/` 作为 examples 搜索 fallback

---

## Phase 5: CHUNK=16 Migration Plan (迁移计划)

### Step 5.1: 用户提出需求
**需求**:
1. Stage1 chunkSize 改为 16
2. 求逆方法改为 Neumann 级数展开

### Step 5.2: 分析现有实现
**参考文档**: `/mnt/workspace/gitCode/refresh/KDA/FlashKDA_方案分析与950适配规划.md`

**关键发现**:
- 当前 chunkSize=64，求逆用前代法 (31步×2 + 2次Cube)
- Neumann 级数只需 3步倍增 (6次Cube MMA)
- CHUNK=16 无需分块，单 16×16 矩阵

### Step 5.3: 编写详细计划
**文件**: `docs/CHUNK16_NEUMANN_MIGRATION_PLAN.md`

**内容包括**:
- 修改目标与位置
- Neumann 数学推导
- 代码修改点 (Tiling + Stage1)
- Workspace 计算变更
- 精度策略 (FP16 MMA)

### Step 5.4: 创建进度跟踪
**文件**: `docs/TODOLIST.md`

**结构**: 18 个任务，5 个 Phase

### Step 5.5: 用户确认决策
| 决策 | 用户选择 |
|------|----------|
| chunkSize | 统一 16 (Stage1/2/3) |
| 精度策略 | FP16 MMA (与 FlashKDA 对齐) |
| 计划位置 | `docs/` 目录 |

---

## Agent Orchestration Rules (子 Agent 分配规则)

### 规则来源
**用户要求**: "你需要合理分配子agent来执行任务"

### 分配策略

| 任务类型 | 推荐 Agent | 使用场景 |
|----------|------------|----------|
| **代码搜索/探索** | `explore` | 查找现有模式、代码位置 |
| **外部文档/库查询** | `librarian` | FlashKDA 参考、Ascend API |
| **架构决策** | `oracle` | 复杂权衡、风险评估 |
| **精度调试** | `ascendc-operator-precision-debug` skill | 误差分析、定位根因 |
| **代码生成** | `quick` category | 简单文件修改、常量替换 |
| **复杂实现** | `deep` category | 多步骤实现、需要研究 |

### 当前迁移任务分配

| Phase | 任务复杂度 | 执行方式 |
|-------|------------|----------|
| Phase 1 (Tiling) | 低 - 常量替换 | **主会话直接执行** |
| Phase 2 (Constants) | 低 - 常量替换 | **主会话直接执行** |
| Phase 3 (Stage1 Neumann) | 高 - 新函数实现 | **分配 `deep` + 技能** |
| Phase 4 (Golden) | 中 - Python 修改 | **分配 `quick`** |
| Phase 5 (Build/Test) | 中 - 验证流程 | **主会话 + oracle 验证** |

### Parallel Execution Pattern
```typescript
// 独立任务并行执行
task(subagent_type="explore", run_in_background=true, prompt="查找 matmul API 用法")
task(subagent_type="librarian", run_in_background=true, prompt="查询 FlashKDA Neumann 实现")
// 等待结果后继续
```

---

## Phase 6: Migration Implementation (迁移实施) - 进行中

### Step 6.1: 更新 AGENTS.md 约束
**时间**: 2025-04-27 20:25
**操作**: 添加 CHUNK=16 迁移状态章节

**新增内容**:
- Confirmed Decisions 表格 (chunkSize=16, FP16 MMA)
- Key Modification Locations (具体行号)
- Progress Tracking 文件位置
- Neumann Inverse Algorithm 公式

### Step 6.2: 开始 Phase 1 Tiling 修改 ⏳
**时间**: 2025-04-27 20:25
**操作**: 修改 `op_host/*_tiling.cpp` Line 113

**目标**: `int64_t c = 64 → 16`

---

## Current Status (当前状态)

```
算子创建:      ✅ 完成
编译安装:      ✅ 完成
精度测试:      ✅ 完成 (Bug 已修复)
目录整理:      ✅ 完成
迁移计划:      ✅ 完成
AGENTS更新:    ✅ 完成
迁移实施:      ⏳ 进行中 (Phase 1.1)
```

---

## Key Files Summary

| 目的 | 文件路径 |
|------|----------|
| 算子定义 | `op_host/chunk_gated_delta_rule_neumann_def.cpp` |
| Tiling | `op_host/chunk_gated_delta_rule_neumann_tiling.cpp` |
| Stage1 Kernel | `op_kernel/chunk_gated_delta_rule_neumann_stage1.h` |
| 测试程序 | `tests/test_aclnn_chunk_gated_delta_rule_neumann.cpp` |
| 数据生成 | `tests/pytest/generate_test_inputs.py` |
| Golden 函数 | `tests/pytest/chunk_gated_delta_rule_neumann_golden.py` |
| 精度调试经验 | `SKILL/precision-debugging/SKILL.md` |
| 迁移计划 | `docs/CHUNK16_NEUMANN_MIGRATION_PLAN.md` |
| 进度跟踪 | `docs/TODOLIST.md` |
| **执行日志** | `docs/PROJECT_EXECUTION_LOG.md` ← 本文档 |

---

## Lessons Learned

### 1. 多 Batch 精度失败根因
> **类型不匹配**: Host `int64_t` → Kernel `int32_t`
> 内存布局错位导致第二个 batch 参数为 0

### 2. Ascend C 测试要点
> Workspace 数据类型必须严格匹配内核预期
> 特别是 `seqlens`, `batch_sizes` 等整型参数

### 3. 目录结构规范
> - `examples/` 可删除，统一到 `tests/`
> - 测试数据输出到 `build/test_data/`，不污染项目根
> - SKILL 文档放 `SKILL/`，计划文档放 `docs/`

---

## Next Steps

1. ⏳ 开始 Phase 1: Tiling Layer 修改
2. ⏳ 实施 Neumann 求逆函数
3. ⏳ 更新 Golden 函数 chunk_size=16
4. ⏳ 验证精度与性能

---

*最后更新: 2025-04-27 20:20*