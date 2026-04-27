# 从现有算子创建新算子 (Operator Clone)

## 概述

本 skill 描述如何正确地从现有 Ascend C 算子复制创建新算子，避免常见的错误操作。

## 适用场景

- 基于现有算子开发新变体
- 创建算子的实验版本
- 需要保留原算子并创建新版本

## 完整工作流

### Phase 1: 创建新算子目录

```bash
cd /mnt/workspace/gitCode/cann/ops-transformer/attention

# 复制原算子目录
cp -r chunk_gated_delta_rule chunk_gated_delta_rule_neumann
```

### Phase 2: 清理缓存文件

```bash
cd chunk_gated_delta_rule_neumann

# 删除不必要的文件
rm -rf SKILL                  # 原算子的 skill 文档（可选保留）
rm -rf tests/pytest/__pycache__
rm -rf tests/pytest/.pytest_cache
rm -f test_inputs.pt          # 测试输入文件（可选保留）
```

### Phase 3: 重命名文件

**关键原则：只在新目录内操作，不要在原目录执行任何 rename！**

```bash
cd chunk_gated_delta_rule_neumann

# 重命名文件（使用 basename，避免改变目录路径）
find . -depth -type f -name "chunk_gated_delta_rule*" | while read f; do
  dir=$(dirname "$f")
  oldname=$(basename "$f")
  newname=$(echo "$oldname" | sed 's/^chunk_gated_delta_rule/chunk_gated_delta_rule_neumann/')
  if [ "$oldname" != "$newname" ]; then
    mv "$f" "$dir/$newname"
  fi
done

# 重命名其他前缀的文件
find . -depth -type f -name "aclnn_chunk_gated_delta_rule*" | while read f; do
  dir=$(dirname "$f")
  oldname=$(basename "$f")
  newname=$(echo "$oldname" | sed 's/^aclnn_chunk_gated_delta_rule/aclnn_chunk_gated_delta_rule_neumann/')
  if [ "$oldname" != "$newname" ]; then
    mv "$f" "$dir/$newname"
  fi
done

find . -depth -type f -name "test_chunk_gated_delta_rule*" | while read f; do
  dir=$(dirname "$f")
  oldname=$(basename "$f")
  newname=$(echo "$oldname" | sed 's/^test_chunk_gated_delta_rule/test_chunk_gated_delta_rule_neumann/')
  if [ "$oldname" != "$newname" ]; then
    mv "$f" "$dir/$newname"
  fi
done

find . -depth -type f -name "test_aclnn_chunk_gated_delta_rule*" | while read f; do
  dir=$(dirname "$f")
  oldname=$(basename "$f")
  newname=$(echo "$oldname" | sed 's/^test_aclnn_chunk_gated_delta_rule/test_aclnn_chunk_gated_delta_rule_neumann/')
  if [ "$oldname" != "$newname" ]; then
    mv "$f" "$dir/$newname"
  fi
done
```

### Phase 4: 替换内容

```bash
cd chunk_gated_delta_rule_neumann

# 替换类名、宏名、命名空间
find . -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.py" -o -name "CMakeLists.txt" -o -name "*.sh" -o -name "*.md" \) | while read f; do
  # 类名
  sed -i 's/ChunkGatedDeltaRule/ChunkGatedDeltaRuleNeumann/g' "$f"
  
  # 宏名
  sed -i 's/CHUNK_GATED_DELTA_RULE/CHUNK_GATED_DELTA_RULE_NEUMANN/g' "$f"
  
  # aclnn 函数名
  sed -i 's/aclnnChunkGatedDeltaRule/aclnnChunkGatedDeltaRuleNeumann/g' "$f"
  
  # namespace
  sed -i 's/namespace ChunkGatedDeltaRule/namespace ChunkGatedDeltaRuleNeumann/g' "$f"
  sed -i 's/namespace CGDR/namespace CGDRN/g' "$f"
done

# 修复可能的双重替换（如 ChunkGatedDeltaRuleNeumannNeumann）
find . -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.py" -o -name "CMakeLists.txt" \) | while read f; do
  sed -i 's/ChunkGatedDeltaRuleNeumannNeumann/ChunkGatedDeltaRuleNeumann/g' "$f"
  sed -i 's/CHUNK_GATED_DELTA_RULE_NEUMANN_NEUMANN/CHUNK_GATED_DELTA_RULE_NEUMANN/g' "$f"
  sed -i 's/chunk_gated_delta_rule_neumann_neumann/chunk_gated_delta_rule_neumann/g' "$f"
done
```

### Phase 5: 验证结果

```bash
# 验证原目录保持不变
ls ../chunk_gated_delta_rule/op_host/
# 应显示：chunk_gated_delta_rule_def.cpp, chunk_gated_delta_rule_tiling.cpp 等

# 验证新目录文件正确命名
ls chunk_gated_delta_rule_neumann/op_host/
# 应显示：chunk_gated_delta_rule_neumann_def.cpp, chunk_gated_delta_rule_neumann_tiling.cpp 等

# 检查是否有残留的旧引用
grep -r "ChunkGatedDeltaRule[^N]" . --include="*.cpp" --include="*.h" | wc -l
# 应为 0
```

## 编译测试

```bash
cd /mnt/workspace/gitCode/cann/ops-transformer

bash build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule_neumann -j8
```

## Reference 文件

- `reference/common-mistakes.md` - 常见错误及解决方案
- `reference/checklist.md` - 操作检查清单

## 相关 SKILL

- `../aclnn-single-op-test/` - aclnn 单算子直调测试，用于验证新算子功能

## 关键要点

1. **只在新目录内操作** - 绝不要在原目录执行 rename
2. **使用 basename** - 避免改变文件路径
3. **分步替换** - 先 rename 文件，再替换内容
4. **修复双重替换** - 检查并修复可能的 `NeumannNeumann` 问题
5. **验证原目录** - 确保原算子保持不变