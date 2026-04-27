# AscendC Operator Precision Debugging Skill

AscendC 算子精度问题调试与根因定位。当算子精度测试失败（allclose 不通过、结果偏差、输出全零/NaN、多 batch 处理异常等）时使用。

## Trigger Keywords

精度调试、precision debug、多 batch 全零、数据类型不匹配、seqlens 错误、第二 batch 异常

---

## Debugging Workflow

### Phase 0: Intent Classification

```
B=1 测试 → PASS？→ 基础逻辑正确
B=1 测试 → FAIL？→ 基础逻辑/数据问题

B=2 测试 → PASS？→ batch 处理正确
B=2 测试 → FAIL？→ batch 处理问题
```

**关键策略：从简单到复杂逐层验证**

---

### Phase 1: Data Type Mismatch Check (FIRST PRIORITY)

**最常见问题：Host 端数据类型 ≠ Kernel 端预期类型**

```cpp
// ✗ 错误：类型不匹配
std::vector<int64_t> seqlens_data;  // Host 存储 8 bytes
CreateAclTensor(..., ACL_INT32);     // Kernel 期望 4 bytes

// ✓ 正确：类型一致
std::vector<int32_t> seqlens_data;  // Host 存储 4 bytes
CreateAclTensor(..., ACL_INT32);    // Kernel 期望 4 bytes
```

**内存布局错误示例**：
```
seqlens = [16, 16] 存储为 int64_t:
内存: [0x10 0x00 0x00 0x00 | 0x00 0x00 0x00 0x00 | 0x10 0x00 ...]
Kernel 按 int32 读取: [16, 0, 16, 0] → 第二个 batch seqlen=0，输出全零！
```

**检查清单**：
```bash
# 1. Host 端存储类型
grep "seqlens_data\|actual_seq" test_aclnn_*.cpp

# 2. Tensor 创建类型
grep "CreateAclTensor.*ACL_" test_aclnn_*.cpp

# 3. Kernel 端读取类型
grep "int32_t\|int64_t" op_kernel/*.h

# 4. Python Golden 类型
grep "dtype.*int\|.to(torch.int" *_golden.py
```

---

### Phase 2: "Second Batch All Zero" Diagnosis

**现象特征**：
- 第一个 batch 输出正常
- 第二个 batch 输出全零（或接近初始值）
- 两个 batch 使用相同参数

**排查顺序**：
```
1. seqlens 传递 → 类型匹配、偏移计算
2. state tensor → highState_[bid * nv * dv * dk]
3. output tensor → out_[cg.startPos * nv * dv]
```

---

### Phase 3: Data Value Check

用户提示解读：
- `|g| < 1` → 检查 gating 参数合法性
- `chunk_size 补齐` → 检查数据是否按 chunk 对齐

```python
# 检查 g 参数
print(f"g range: [{g.min():.4f}, {g.max():.4f}]")
print(f"|g| < 1: {(g.abs() < 1).all()}")

# 检查 chunk 补齐
print(f"seqlen % chunkSize: {seqlen % chunk_size}")
```

---

### Phase 4: Workspace Memory Check

```cpp
// 检查 workspace 偏移计算
offset += sizeof(highType) * b * nv * dv * dk;  // highState
offset += sizeof(highType) * chunkSize * chunkSize * aiCoreNum;  // mask

// 验证 workspace 大小
uint64_t workspaceSize = interWorkspaceSz + stageWorkspaceSz;
```

---

## Prevention Code Patterns

### 1. Explicit Type Annotation

```cpp
// 函数声明中显式标注数据类型
bool read_test_inputs(..., 
    std::vector<int32_t>& seqlens_data);  // 必须与 ACL_INT32 匹配
```

### 2. Static Type Check

```cpp
template<typename T>
int CreateAclTensor(const std::vector<T>& hostData, ..., aclDataType dataType) {
    static_assert(
        (std::is_same<T, int32_t>::value && dataType == ACL_INT32) ||
        (std::is_same<T, uint16_t>::value && dataType == ACL_BF16) ||
        (std::is_same<T, float>::value && dataType == ACL_FLOAT),
        "Host data type must match tensor data type!"
    );
    ...
}
```

### 3. Python Unified Type

```python
def save_seqlens(seqlens, file):
    seqlens_int32 = seqlens.to(torch.int32)
    file.write(seqlens_int32.cpu().numpy().tobytes())
```

---

## Time Cost Analysis

| 阶段 | 耗时比例 | 说明 |
|---|---|---|
| 数据类型检查 | 20% | **第一优先级** |
| Golden 逻辑检查 | 30% | 算法问题 |
| Kernel 检查 | 40% | 实现问题 |
| Workspace 检查 | 10% | 内存问题 |

**教训**：优先检查数据传递（类型/布局），再检查算法逻辑。

---

## One-Line Summary

> **跨组件数据类型不一致 = 隐蔽的内存布局错误 = 第二 batch 全零**

---

## Related Skills

- `ascendc-operator-precision-debug` - 算子精度问题调试（更详细的插桩方案）
- `ascendc-operator-precision-eval` - 算子精度评估（生成测试用例）
- `ascendc-operator-performance-eval` - 算子性能评估