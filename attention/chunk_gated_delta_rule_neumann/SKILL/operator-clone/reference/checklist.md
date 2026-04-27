# 操作检查清单

## 前置检查

- [ ] 确认原算子目录存在且干净
- [ ] 创建备份分支：`git checkout -b backup`
- [ ] 确认新算子名称（如 `chunk_gated_delta_rule_neumann`）

## Phase 1: 复制目录

- [ ] 执行复制：`cp -r chunk_gated_delta_rule chunk_gated_delta_rule_neumann`
- [ ] 确认新目录存在：`ls chunk_gated_delta_rule_neumann/`
- [ ] 确认原目录未改变：`git diff chunk_gated_delta_rule/`

## Phase 2: 清理缓存

- [ ] 删除 SKILL 文件（可选）
- [ ] 删除 `__pycache__` 目录
- [ ] 删除测试数据文件（可选）

## Phase 3: 重命名文件

- [ ] **切换到新目录**：`cd chunk_gated_delta_rule_neumann`
- [ ] **确认当前位置**：`pwd`（必须显示 `_neumann`）
- [ ] 重命名 `chunk_gated_delta_rule*` 文件
- [ ] 重命名 `aclnn_chunk_gated_delta_rule*` 文件
- [ ] 重命名 `test_chunk_gated_delta_rule*` 文件
- [ ] 重命名 `test_aclnn_chunk_gated_delta_rule*` 文件
- [ ] **验证原目录**：`ls ../chunk_gated_delta_rule/op_host/`（应为原文件名）

## Phase 4: 替换内容

- [ ] 替换类名：`ChunkGatedDeltaRule` → `ChunkGatedDeltaRuleNeumann`
- [ ] 替换宏名：`CHUNK_GATED_DELTA_RULE` → `CHUNK_GATED_DELTA_RULE_NEUMANN`
- [ ] 替换命名空间：`namespace ChunkGatedDeltaRule` → `namespace ChunkGatedDeltaRuleNeumann`
- [ ] 替换 aclnn 函数名
- [ ] 修复双重替换问题
- [ ] 检查残留引用：`grep -r "ChunkGatedDeltaRule[^N]" .`

## Phase 5: 最终验证

- [ ] 原目录文件名正确：`ls ../chunk_gated_delta_rule/op_host/`
- [ ] 新目录文件名正确：`ls chunk_gated_delta_rule_neumann/op_host/`
- [ ] 原目录无删除：`git status`（无 `D` 状态）
- [ ] 新目录未跟踪：`git status`（显示 `??`）
- [ ] 编译测试：`bash build.sh --pkg --soc=<soc> --ops=<new_op> -j8`

## 关键检查点

| 检查项 | 原目录期望 | 新目录期望 |
|-------|----------|----------|
| Git status | 无 `D` 状态 | `??` 未跟踪 |
| op_host 文件名 | `chunk_gated_delta_rule_def.cpp` | `chunk_gated_delta_rule_neumann_def.cpp` |
| namespace | `ChunkGatedDeltaRule` | `ChunkGatedDeltaRuleNeumann` |
| aclnn 函数 | `aclnnChunkGatedDeltaRule` | `aclnnChunkGatedDeltaRuleNeumann` |

## 禁止操作

- ❌ 在原目录内执行 `find | rename`
- ❌ 在原目录内执行 `sed -i`
- ❌ 使用 `find` 替换整个路径
- ❌ 不检查当前目录就执行 rename

## 必须操作

- ✅ 复制后再操作
- ✅ 切换到新目录 `cd chunk_gated_delta_rule_neumann`
- ✅ 使用 `basename` 处理文件名
- ✅ 每步检查原目录状态
- ✅ 修复双重替换