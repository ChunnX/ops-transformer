# FP8 Group Quantization NaN Bug Fix

## 问题现象

### Case
- **Shape**: `n=2, cols=786, k=1122`
- **Template**: `moe_v3_gather_fp8_group_quant`
- **Symptom**: 输出精度 97.70%，失败元素全部为 NaN

### 对比
- 相同 shape 下 `perblock` 模板精度正常（100%）
- 仅 `pergroup` 模板出现精度问题

---

## Root Cause

### 直接原因：`UpdateMask` 的引用递减特性

`UpdateMask<T>(var)` 按**引用传递**参数并**内部递减**（饱和到 0）。代码中同一个 `groupElemNum` 被两处 `UpdateMask` 调用：

```cpp
// ❌ Bug 代码结构

} else if (groupElemNum > 0) {
    maskLoop = UpdateMask<float>(groupElemNum);  // groupElemNum: 18 → 0
    LoadFp8GroupInput<T>(xLeftReg, xAddr, maskLoop, 0);
    Abs(absLeftReg, xLeftReg, maskLoop);
    ReduceMax(maxReg, absLeftReg, maskLoop);
}

// ... scale/invScale 计算（不依赖 groupElemNum）...

} else if (groupElemNum > 0) {  // ❌ groupElemNum 已经是 0，分支被跳过
    maskLoop = UpdateMask<float>(groupElemNum);
    Mul(quantLeftReg, xLeftReg, invScaleReg, maskLoop);
    Cast<U, float, CAST_TRAIT_F32_TO_FP8>(outLeftReg, quantLeftReg, maskLoop);
    DataCopy<U, StoreDist::DIST_PACK4_B32>(yAddr, outLeftReg, maskLoop);
}
```

### 执行流程

以 `cols = 786 = 6 × 128 + 18` 为例：

1. **Max 计算阶段**（Line 108-113）
   ```
   输入: groupElemNum = 18
   UpdateMask<float>(groupElemNum) → groupElemNum 从 18 递减到 0
   执行: Abs + ReduceMax ✓
   ```

2. **Scale/InvScale 计算阶段**（Line 115-131）
   ```
   不依赖 groupElemNum 变量
   正常执行: Maxs → Div → Muls → Add → Muls → DataCopy → Div ✓
   ```

3. **Quant 计算阶段**（Line 132-144）
   ```
   检查: groupElemNum > 0
   实际: groupElemNum = 0（被前面的 UpdateMask 耗尽）
   结果: 分支被跳过，量化计算不执行 ❌
   UB 输出位置未被写入 → 残留数据为 NaN
   ```

### 触发条件

只对**不完整 group** 触发（`groupElemNum ≤ 128` 且 `> 0`）：

| Group Index | groupElemNum | 分支选择 | 结果 |
|:-----------:|:------------:|:--------:|:----:|
| 0-5 | 128 | `> vfLen` | ✓ 正常 |
| 6 | 18 | `> 0` | ❌ NaN |

### 为什么 perblock 模板没问题

`moe_v3_gather_fp8_perblock_quant` 之前已经修复过同类问题，使用了分离的计数器：

```cpp
uint32_t inputNum = loopCols;
uint32_t outputNum = loopCols;

// Max 阶段
while (inputNum > 0) {
    maskLoop = UpdateMask<T>(inputNum);
    // ...
}

// Quant 阶段
while (outputNum > 0) {
    maskLoop = UpdateMask<T>(outputNum);
    // ...
}
```

---

## 修复方案

### 核心修改：引入分离计数器

```cpp
// ✅ 修复后代码结构

uint32_t inputElemNum = groupElemNum;   // max 阶段专用
uint32_t outputElemNum = groupElemNum;  // quant 阶段专用

} else if (inputElemNum > 0) {
    maskLoop = UpdateMask<float>(inputElemNum);  // inputElemNum: 18 → 0
    LoadFp8GroupInput<T>(xLeftReg, xAddr, maskLoop, 0);
    Abs(absLeftReg, xLeftReg, maskLoop);
    ReduceMax(maxReg, absLeftReg, maskLoop);
}

// ... scale/invScale 计算 ...

} else if (outputElemNum > 0) {  // ✅ outputElemNum 仍是 18，分支正常进入
    maskLoop = UpdateMask<float>(outputElemNum);
    Mul(quantLeftReg, xLeftReg, invScaleReg, maskLoop);
    Cast<U, float, CAST_TRAIT_F32_TO_FP8>(outLeftReg, quantLeftReg, maskLoop);
    DataCopy<U, StoreDist::DIST_PACK4_B32>(yAddr, outLeftReg, maskLoop);
}
```

### 代码修改

**文件**: `moe/moe_init_routing_v3/op_kernel/arch35/moe_v3_gather_fp8_group_quant.h`

```diff
@@ Line 98-113 (Max 计算阶段) @@
-        if (groupElemNum > 0) {
-            maskLoop = UpdateMask<float>(groupElemNum);
+        uint32_t inputElemNum = groupElemNum;
+        if (inputElemNum > 0) {
+            maskLoop = UpdateMask<float>(inputElemNum);
             LoadFp8GroupInput<T>(xLeftReg, xAddr, maskLoop, 0);

@@ Line 132-145 (Quant 计算阶段) @@
-        } else if (groupElemNum > 0) {
-            maskLoop = UpdateMask<float>(groupElemNum);
+        uint32_t outputElemNum = groupElemNum;
+        } else if (outputElemNum > 0) {
+            maskLoop = UpdateMask<float>(outputElemNum);
             Mul(quantLeftReg, xLeftReg, invScaleReg, maskLoop);
             Cast<U, float, CAST_TRAIT_F32_TO_FP8>(outLeftReg, quantLeftReg, maskLoop);
             DataCopy<U, StoreDist::DIST_PACK4_B32>(yAddr, outLeftReg, maskLoop);
         }
```

### 顺手修复的原始 Bug

审查过程中发现原始代码缺少 `absRightReg` 的 `RegTensor` 声明：

```cpp
__VEC_SCOPE__
{
    RegTensor<float> xLeftReg;
    RegTensor<float> xRightReg;
    RegTensor<float> absLeftReg;
+   RegTensor<float> absRightReg;  // 补充缺失的声明
    RegTensor<float> maxReg;
    // ...
}
```

这个 bug 在原始代码中就存在，但可能因编译器优化或 UB 初始化被掩盖，之前未暴露。

---

## 验证

### 修复前
```
Case: n=2, cols=786, k=1122
Template: moe_v3_gather_fp8_group_quant
Result: Precision 97.70%, NaN in last 18 elements
```

### 修复后
```
Case: n=2, cols=786, k=1122
Template: moe_v3_gather_fp8_group_quant
Result: Precision 100% ✓
```

---

## 经验总结

### 关键教训

1. **`UpdateMask` 的副作用**: 按引用递减参数，同一变量多次调用会导致状态丢失
2. **分离关注点**: 不同计算阶段使用独立的计数器，避免状态污染
3. **不完整 group 的边界处理**: `groupElemNum ≤ 128` 的分支容易踩坑

### 代码审查建议

在使用 `UpdateMask` 时，检查以下要点：

- [ ] 是否有多个 `UpdateMask` 调用使用同一变量？
- [ ] 后续的分支判断是否依赖该变量的原始值？
- [ ] 是否需要引入独立计数器分离不同计算阶段？

### 相关修复记录

类似问题在 `moe_v3_gather_fp8_perblock_quant.h` 中也修复过，修复方式相同：引入 `inputNum` / `outputNum` 分离计数器。这应该是 AscendC kernel 开发中使用 `UpdateMask` 的通用最佳实践。

---

## 参考资料

- [AscendC UpdateMask API 行为](../../../../../docs/ascendc_update_mask.md)（内部文档）
- 相关 commit: `a4ffb660b` (perblock 模板同类问题修复)
