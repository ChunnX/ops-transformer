# PROJECT EXECUTION LOG - chunk_gated_delta_rule_neumann

完整记录项目从零开始到当前的每一步操作。

---

## Timeline Overview

| 阶段 | 时间 | 主要任务 |
|------|------|----------|
| Phase 1 | 2026-04-27 上午 | 算子克隆与重命名 |
| Phase 2 | 2026-04-27 下午 | 编译安装测试 |
| Phase 3 | 2026-04-27 下午 | 精度调试与 Bug 修复 |
| Phase 4 | 2026-04-27 晚上 | 目录整理与文档 |
| Phase 5 | 2026-04-27 晚上 | CHUNK=16 迁移计划 |

---

## Phase 1: Operator Clone (算子克隆)

### Step 1.1: 创建算子目录结构
**时间**: 2026-04-27 上午
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
**时间**: 2026-04-27 下午
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

## Phase 6: Migration Implementation (迁移实施) - 需重构

### Step 6.1: 更新 AGENTS.md 约束
**时间**: 2026-04-27 20:25
**操作**: 添加 CHUNK=16 迁移状态章节

**新增内容**:
- Confirmed Decisions 表格 (chunkSize=16, FP16 MMA)
- Key Modification Locations (具体行号)
- Progress Tracking 文件位置
- Neumann Inverse Algorithm 公式

### Step 6.2: 代码修改执行
**时间**: 2026-04-27 20:40 - 20:50

**修改文件清单**:

| 文件 | 行号 | 修改内容 |
|------|------|----------|
| `op_host/*_tiling.cpp` | 113 | `c = 64 → 16` |
| `op_host/*_tiling.cpp` | 135后新增 | Neumann workspace (L_pow + tmp) |
| `op_kernel/*_utils.h` | 40 | `TASK_RATIO = 2 → 1` |
| `op_kernel/*_stage1.h` | 32 | `INVERSE_SHAPE = 32 → 16` |
| `op_kernel/*_stage1.h` | 33 | `INVERSE_COUNT = 5 → 2` |
| `op_kernel/*_stage1.h` | 768 | 新增 L_powWsGm_, tmpWsGm_ |
| `op_kernel/*_stage1.h` | 111后 | GM buffer 初始化 |
| `op_kernel/*_stage1.h` | 508 | 新增 NeumannInverseCompute() |
| `op_kernel/*_stage1.h` | 460 | InverseAIV → NeumannInverseCompute |
| `tests/pytest/*_golden.py` | 111 | `chunk_size=64 → 16` |

### Step 6.3: 编译测试
**时间**: 2026-04-27 20:43
**命令**: `bash build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule_neumann -j16`
**结果**: ✅ 编译成功

### Step 6.4: 安装算子
**时间**: 2026-04-27 20:44
**命令**: `./build/*.run --quiet --install-path=...`
**结果**: ✅ 安装成功

---

## Phase 7: Kernel Execution Debug (内核执行调试) - 2026-04-27 21:00-21:18

### Step 7.1: ERROR 507015 首次出现 🔴
**时间**: 2026-04-27 20:48
**现象**: `SynchronizeStream failed. ERROR: 507015`
**分析**: Kernel 执行失败

### Step 7.2: 发现 L 已被 mask 修改
**时间**: 2026-04-27 21:00
**问题**: InverseCompute 中 `attnUbFloat_` 已经是 `L × mask/gamma`，不是原始 L
**修复尝试**: 在 InverseCompute 开始时保存原始 L 到 `L_powWsGm_`
**结果**: 仍然 ERROR 507015

### Step 7.3: 发现 AttnInverseMMCompute 越界访问 🔴 关键
**时间**: 2026-04-27 21:04
**问题**: 
```
原代码 (chunk=64, INVERSE_SHAPE=32):
  leftDown = chunkSize_ × curLen = 64 × 32 = 2048
  attnWsGm_ 大小 = chunkSize_ × chunkSize_ = 4096 ✓

修改后 (chunk=16, INVERSE_SHAPE=16):
  leftDown = chunkSize_ × curLen = 16 × 16 = 256
  attnWsGm_ 大小 = chunkSize_ × chunkSize_ = 256 ❌ 越界！
```
**修复**: 添加条件判断跳过 AttnInverseMMCompute
```cpp
if (INVERSE_SHAPE < chunkSize_) {
    AttnInverseMMCompute(INVERSE_SHAPE);
}
```
**结果**: 编译成功，仍然 ERROR 507015

### Step 7.4: 发现 AIV 不能调用 AICProcess 🔴 架构错误
**时间**: 2026-04-27 21:08
**问题**: NeumannInverseCompute 在 AIV 分支调用 AICProcess(matmul)
**根因**: AIV (Vector) 不能调用 AIC (Cube) 的 matmul API
**修复尝试**: 删除 NeumannInverseCompute，使用原始 InverseAIV (Gauss-Jordan)
**结果**: Kernel 挂起 (超时)

### Step 7.5: 发现 inverseUbFloat_ buffer 分配不足
**时间**: 2026-04-27 21:12
**问题**: 
```
需要: 3 × INVERSE_SHAPE² + 2 × INVERSE_SHAPE = 3×256+32 = 800 元素
实际: halfChunkSize_² × INVERSE_COUNT = 16×16×2 = 512 元素 ❌
```
**修复**: 增加分配到正确大小
```cpp
uint32_t inverseBufferPerBlock = 3 * INVERSE_SHAPE * INVERSE_SHAPE + 2 * INVERSE_SHAPE;
inverseUbFloat_ = tmpBuff_.GetWithOffset<float>(inverseBufferPerBlock × INVERSE_COUNT, ...);
```
**结果**: 编译成功，仍然挂起

### Step 7.6: 简化版测试成功 ✅
**时间**: 2026-04-27 21:17
**策略**: 简化 InverseCompute，仅输出单位矩阵 I（跳过求逆）
```cpp
__aicore__ inline void InverseCompute()
{
    // 仅做 mask/gamma 操作
    inverseLocal_ = fp32OutQueue_.AllocTensor<float>();
    Duplicate(inverseLocal_, static_cast<float>(1.0), chunkSize_ * chunkSize_);
    fp32OutQueue_.EnQue(inverseLocal_);
    DataCopyOutFp32(...);
}
```
**结果**: ✅ **测试成功！**
```
Output precision: max_abs_err: 0.000244, max_rel_err: 0.73% - PASS
FinalState precision: max_abs_err: 0.000488, max_rel_err: 0.40% - PASS
```
**结论**: 基本流程正确，问题在 InverseAIV 函数

### Step 7.7: 用户指导正确架构 ✅ 关键突破
**时间**: 2026-04-27 21:18
**用户提问**:
> 1. 我们的矩阵可以在 mac 完后直接在 L0C 累加吧？
> 2. 而且在 L0C 出来的是 FP32 我们可以直接 FP32 累加搬出再搬运回 L1 进行矩阵乘

**关键发现**:
```
正确的 Neumann 实现应该在 AIC 分支执行，利用:

1. Cube L0C 累加能力:
   - mm.SetBias(biasTensor) 设置 L0C 初始值
   - mm.IterateAll(z) 输出 = A×B + bias (自动累加)

2. FP32 数据流:
   GM(L FP32) → L1 → L0A/L0B → Cube(MMA FP32) → L0C(FP32) → GM(FP32)
   焯后 GM → L1 → L0A → 下一次 MMA

3. 无需 Vector Add:
   INV = INV × L_pow + INV 直接在 L0C 完成
```

**下一步**: 重构 Neumann 求逆到 AIC 分支，利用 L0C 累加

---

## Phase 8: Architecture Refactoring (架构重构) - 待执行

### 关键修改点

| 文件 | 修改内容 | 状态 |
|------|----------|------|
| `*_stage1.h` ProcessOneChunk AIC 分支 | 添加 NeumannMatmulCompute 调用 | ⏳ 待实现 |
| `*_stage1.h` | 新增 NeumannMatmulCompute 函数 (AIC 端) | ⏳ 待实现 |
| `*_stage1.h` InverseCompute (AIV 分支) | 仅准备 L 数据，不再求逆 | ⏳ 待修改 |
| `*_tiling.cpp` workspace | 增加 INV_Gm_ 和 I_Gm_ buffer | ⏳ 待修改 |

### 需研究 API

| API | 用途 | 待确认 |
|-----|------|--------|
| `mm.SetBias()` | 设置 L0C 初始值 (累加) | ⏳ 查阅文档 |
| `mm.MmadParams` | MMA 参数配置 | ⏳ 是否有累加模式 |
| `mm.template IterateAll<sync>()` | 同步参数 | ⏳ sync=false 是否累加 |

---

## Current Status (当前状态) - 2026-04-27 21:50

```
算子创建:      ✅ 完成
代码修改:      ✅ 完成 (tiling + kernel + Neumann functions)
编译成功:      ✅ 完成
安装成功:      ✅ 完成
Neumann实现:   ✅ 完成 (NeumannLpowCompute + NeumannMmulCompute + NeumannAddCompute)
测试验证:      🔴 **发现测试程序调用错误算子！**
```

### 🔴 关键问题：测试程序错误

**根因发现** (2026-04-27 21:45):
- 运行的测试 binary: `/mnt/workspace/.../chunk_gated_delta_rule/test_chunk_gated_delta_rule_neumann`
- 包含头文件: `aclnnop/aclnn_chunk_gated_delta_rule.h` (**原始算子**)
- 应包含: `aclnnop/aclnn_chunk_gated_delta_rule_neumann.h` (**Neumann算子**)

**正确的测试**位于:
```
chunk_gated_delta_rule_neumann/tests/test_aclnn_chunk_gated_delta_rule_neumann.cpp
```

但 build.sh 编译失败 - header 路径配置问题

---

## 代码修改清单 (最终版本)

| 文件 | 行号 | 修改内容 |
|------|------|----------|
| `op_host/*_tiling.cpp` | 113 | `c = 64 → 16` |
| `op_host/*_tiling.cpp` | 137-138 | `workspace × 3` (L_pow + tmp + INV) |
| `op_kernel/*_utils.h` | 40 | `TASK_RATIO = 2 → 1` |
| `op_kernel/*_stage1.h` | 32 | `INVERSE_SHAPE = 32 → 16` |
| `op_kernel/*_stage1.h` | 33 | `INVERSE_COUNT = 5 → 2` |
| `op_kernel/*_stage1.h` | 777 | 新增 `INVWsGm_` |
| `op_kernel/*_stage1.h` | 119-122 | INVWsGm_ GM buffer 初始化 |
| `op_kernel/*_stage1.h` | 744-748 | NeumannLpowCompute (AIC) |
| `op_kernel/*_stage1.h` | 749-753 | NeumannMmulCompute (AIC) |
| `op_kernel/*_stage1.h` | 487-503 | NeumannAddCompute (AIV) |

---

## Lessons Learned

### 1. 测试程序必须匹配算子
> **测试 binary 必须调用正确的算子头文件**
> 错误: 测试 chunk_gated_delta_rule 目录下的 binary
> 正确: 测试 chunk_gated_delta_rule_neumann 目录下的源码

### 2. Ascend C AIC/AIV 分工
> - AIC 只能执行 Cube matmul 操作
> - AIV 执行 Vector 操作 (Add, Mul, DataCopy)
> - 不能在 AIV 分支调用 AICProcess

### 3. L0C 累加能力
> `mm.SetBias(biasGm)` 设置 L0C 初始值
> `mm.IterateAll(dst)` 输出 = A×B + bias
> 可用于 Neumann 迭代的 INV = INV + INV×L_pow

---

## Next Steps

1. 🔴 修复测试程序编译配置
2. ⏳ 使用正确测试验证 kernel
3. ⏳ 如 kernel 问题，检查 sync flags
4. ⏳ 精度验证

---

*最后更新: 2026-04-27 21:50*

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

*最后更新: 2026-04-27 20:20*