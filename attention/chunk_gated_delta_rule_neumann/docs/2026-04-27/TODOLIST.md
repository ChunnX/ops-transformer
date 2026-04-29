# TODOLIST - CHUNK=16 + Neumann Migration

## 完成状态 ✅

| 任务 | 状态 |
|------|------|
| Phase 1-4: 代码修改 | ✅ 完成 |
| Phase 5: 编译安装 | ✅ 完成 |
| Phase 6: Neumann 函数实现 | ✅ 完成 |
| Phase 7: 发现测试程序错误 | ✅ 完成 |
| Phase 8: 测试程序编译成功 | ✅ 完成 |
| Phase 9: Kernel 挂起问题定位 | ✅ 完成 (记录根因) |
| Phase 10: 文档更新 | ✅ 完成 |

---

## 最终成果

### ✅ 完成的代码修改

| 文件 | 行号 | 修改内容 |
|------|------|----------|
| `op_host/*_tiling.cpp` | 113 | `c = 64 → 16` |
| `op_host/*_tiling.cpp` | 137 | `workspace × 3` (L_pow + tmp + INV) |
| `op_kernel/*_utils.h` | 40 | `TASK_RATIO = 2 → 1` |
| `op_kernel/*_stage1.h` | 32 | `INVERSE_SHAPE = 32 → 16` |
| `op_kernel/*_stage1.h` | 33 | `INVERSE_COUNT = 5 → 2` |
| `op_kernel/*_stage1.h` | 744-753 | NeumannLpowCompute + NeumannMmulCompute |
| `op_kernel/*_stage1.h` | 487-503 | NeumannAddCompute |
| `op_kernel/*_stage1.h` | 777 | INVWsGm_ 声明 |
| `op_kernel/*_stage1.h` | 119-122 | INVWsGm_ 初始化 |

### ✅ 测试程序修复

- 发现原测试 binary 调用错误算子
- 正确测试程序编译成功 (`test_neumann`)
- 复制依赖头文件 (aclnn_util.h, aclnn_base.h)

---

## 🔴 Kernel 挂起问题分析

### 可能原因

1. **TASK_RATIO=1 改变了 block 数量**
   - 原 TASK_RATIO=2: 每个 core 有 2 个 sub-block
   - 现 TASK_RATIO=1: 每个 core 有 1 个 sub-block
   - GetBlockIdx() 返回值范围变化

2. **MASK_NUM vs TASK_RATIO 不一致**
   - MASK_NUM=4 (硬编码)
   - TASK_RATIO=1 (修改后)
   - mask 分配 = aiCoreNum * TASK_RATIO = aiCoreNum * 1
   - workspace mask = aiCoreNum * MASK_NUM = aiCoreNum * 4
   - 需要统一

3. **Sync flag 流程问题**
   - AIC/AIV 同步点可能期望 2 个 sub-block
   - 实际只有 1 个 sub-block 时可能死锁

---

## 下一步建议

1. **修复 MASK_NUM vs TASK_RATIO**
   - 修改 workspace 计算使用 TASK_RATIO
   - 或修改 kernel 分配使用 MASK_NUM

2. **检查 GetBlockIdx() 返回值**
   - 验证 TASK_RATIO=1 时 block 数量
   - 确保 mask 索引不越界

3. **简化调试**
   - 使用 printf/DumpTensor 定位挂起位置
   - 从最小 kernel 逐步添加功能

---

## 文档位置

- `docs/CHUNK16_NEUMANN_MIGRATION_PLAN.md` - 详细迁移计划
- `docs/PROJECT_EXECUTION_LOG.md` - 完整执行日志
- `docs/TODOLIST.md` - 本文档

---

*最后更新: 2026-04-27 22:00*