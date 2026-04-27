# 常见错误及解决方案

## 错误 1: 在原目录内执行 rename

### 错误现象

```bash
# 在原目录执行 rename 操作
cd chunk_gated_delta_rule
find . -name "*.cpp" | xargs rename 's/chunk_gated_delta_rule/chunk_gated_delta_rule_neumann/'
```

**结果：**
- 原目录的文件被删除（git 显示 `D`）
- `_neumann` 文件出现在原目录内（git 显示 `??`）

```
 D chunk_gated_delta_rule/op_host/chunk_gated_delta_rule_def.cpp
?? chunk_gated_delta_rule/op_host/chunk_gated_delta_rule_neumann_def.cpp
```

### 根本原因

`find` 命令会匹配整个路径，当目录名本身包含 `chunk_gated_delta_rule` 时，rename 会尝试改变目录名，导致：
1. 文件被移动到新路径（`chunk_gated_delta_rule_neumann/...`）
2. 但新目录不存在，文件留在原目录内
3. Git 记录为删除旧文件 + 新增新文件

### 解决方案

**方法 1: 使用 basename**

```bash
find . -depth -type f -name "chunk_gated_delta_rule*" | while read f; do
  dir=$(dirname "$f")
  oldname=$(basename "$f")
  newname=$(echo "$oldname" | sed 's/^chunk_gated_delta_rule/chunk_gated_delta_rule_neumann/')
  mv "$f" "$dir/$newname"
done
```

**方法 2: 只在新目录内操作**

```bash
# 1. 先复制到新目录
cp -r chunk_gated_delta_rule chunk_gated_delta_rule_neumann

# 2. 切换到新目录
cd chunk_gated_delta_rule_neumann

# 3. 在新目录内执行所有操作
# ... rename 和替换内容
```

---

## 错误 2: 双重替换导致错误命名

### 错误现象

```cpp
namespace ChunkGatedDeltaRuleNeumannNeumann {  // 错误：双重 Neumann
```

### 根本原因

替换顺序问题：先替换 `ChunkGatedDeltaRule` → `ChunkGatedDeltaRuleNeumann`，然后文件名已经是 `_neumann`，再次替换时又匹配到 `ChunkGatedDeltaRule` 部分。

### 解决方案

```bash
# 修复双重替换
find . -type f -name "*.cpp" -o -name "*.h" | while read f; do
  sed -i 's/ChunkGatedDeltaRuleNeumannNeumann/ChunkGatedDeltaRuleNeumann/g' "$f"
  sed -i 's/CHUNK_GATED_DELTA_RULE_NEUMANN_NEUMANN/CHUNK_GATED_DELTA_RULE_NEUMANN/g' "$f"
done
```

---

## 错误 3: 原目录被污染

### 错误现象

原目录 `chunk_gated_delta_rule` 内出现 `_neumann` 文件。

### 解决方案

```bash
# 1. 恢复原目录
git checkout -- attention/chunk_gated_delta_rule/

# 2. 清理错误的新目录
rm -rf chunk_gated_delta_rule_neumann

# 3. 重新正确创建
cp -r chunk_gated_delta_rule chunk_gated_delta_rule_neumann

# 4. 只在新目录内操作
cd chunk_gated_delta_rule_neumann
# ... 执行 rename 和替换
```

---

## 错误 4: Git 状态混乱

### 错误现象

```bash
git status
# 显示大量删除和新增文件
 D chunk_gated_delta_rule/...
?? chunk_gated_delta_rule_neumann/...
?? chunk_gated_delta_rule/..._neumann
```

### 解决方案

```bash
# 1. 恢复原目录所有删除
git checkout -- chunk_gated_delta_rule/

# 2. 删除错误位置的文件
cd chunk_gated_delta_rule
rm -rf *_neumann*  # 删除所有错误位置的 neumann 文件
git checkout -- .   # 再次恢复确保干净

# 3. 确认状态
git status
# 原目录应无删除，新目录应为未跟踪（??）
```

---

## 预防措施

1. **创建备份分支**
   ```bash
   git checkout -b backup-before-clone
   ```

2. **操作前检查**
   ```bash
   # 确认当前目录
   pwd
   # 应显示：.../attention/chunk_gated_delta_rule_neumann
   # 而不是：.../attention/chunk_gated_delta_rule
   ```

3. **分步验证**
   ```bash
   # 每个步骤后检查原目录
   ls ../chunk_gated_delta_rule/op_host/
   ```

4. **使用 git diff**
   ```bash
   git diff --stat chunk_gated_delta_rule/
   # 应无差异（除了新增文件）
   ```