# FP8 Group Quantization NaN 问题排查全过程

## 问题报告

**时间**: 2026-07-01  
**Case**: `n=2, cols=786, k=1122`  
**现象**: `moe_v3_gather_fp8_perblock_quant` 精度正常，但切到 `moe_v3_gather_fp8_group_quant` 后精度降至 97.70%，失败元素全部为 NaN

用户反馈：
> "相同 shape 下，perblock 精度正常，pergroup 有问题。输出的 NaN 集中在最后 18 个元素。"

---

## 第一轮：初步误判

### 假设 1：RoundUp Scale 导致除零

看到 pergroup 代码中有 RoundUp Scale 逻辑：

```cpp
// RoundUp Scale (保证 scale 是 2 的幂)
Div(rawScaleReg, maxReg, fp8MaxReg, maskAll);
ShiftRights(expBitsReg, (RegTensor<uint32_t> &)rawScaleReg, FP32_EXPONENT_SHIFT, maskAll);
And(mantBitsReg, (RegTensor<uint32_t> &)rawScaleReg, FP32_MANTISSA_MASK, maskAll);
Add(expBitsReg, expBitsReg, oneReg, maskAll);  // exp + 1
ShiftLefts(roundScaleReg, expBitsReg, FP32_EXPONENT_SHIFT, maskAll);
Div(invScaleReg, oneReg, roundScaleReg, maskAll);  // 1.0 / roundScale
```

**初步猜测**：如果 `maxReg` 接近 0，RoundUp 后 `roundScale` 可能产生极小值，导致 `invScale` 过大，量化时溢出为 Inf/NaN。

### 建议 1：开启 CLAMP_AMAX 保护

建议用户使用 `CLAMP_AMAX=true` 模板参数，强制 `max ≥ 0.0001`：

```cpp
if constexpr (CLAMP_AMAX) {
    Maxs(maxReg, maxReg, 0.0001f, maskAll);
}
```

### 用户反馈

> "我试了方案 2 还是精度有问题，而且我的 x 输入范围是（-2，-1），应该不会有这个问题吧"

**反思**：用户说得对，输入范围 [-2, -1] 意味着 `max ≈ 2`，不存在接近 0 的情况。这不是 NaN 的来源。

---

## 第二轮：深入代码对比

### 方法论

既然 RoundUp Scale 不是问题，需要换个思路。决定：
1. 逐行对比 pergroup 和 perblock 的 VF 函数
2. 重点关注变量生命周期和数据流

### 对比 perblock 的 FP8 路径

```cpp
// moe_v3_gather_fp8_perblock_quant.h ComputeVF
__VEC_SCOPE__ {
    RegTensor<U> xLeftReg, xRightReg;
    RegTensor<float> absLeftReg, absRightReg;
    RegTensor<float> maxReg, maxRightReg;
    RegTensor<float> scaleReg, invScaleReg;
    RegTensor<U> quantLeftReg, quantRightReg;
    // ...

    // Phase 1: Max 计算
    for (uint32_t i = 0; i < inputVfLoop; i++) {
        dataMask = UpdateMask<U>(inputNum);  // inputNum 从 128 递减
        DataCopy(xLeftReg, xAddr + i * FLOAT_REG_TENSOR_LENGTH, dataMask);
        Abs(absLeftReg, xLeftReg, dataMask);
        ReduceMax(maxReg, absLeftReg, dataMask);
    }
    // inputVfLoop = (128 + 127) / 128 = 1，所以循环 1 次
    // inputNum 从 128 → 0
    
    // ... scale 计算 ...

    // Phase 2: Quant 计算
    for (uint32_t i = 0; i < outputVfLoop; i++) {
        dataMask = UpdateMask<U>(inputNum);  // ⚠️ inputNum 已经是 0！
        DataCopy(xLeftReg, xAddr + i * FLOAT_REG_TENSOR_LENGTH, dataMask);
        Mul(quantLeftReg, xLeftReg, invScaleReg, dataMask);
        Cast(outReg, quantLeftReg, castTrait0, dataMask);
        DataCopy(yAddr + i * FLOAT_REG_TENSOR_LENGTH, outReg, dataMask);
    }
    // outputVfLoop = 1，但 inputNum = 0，Quant 分支被跳过！
}
```

**关键发现**：perblock 的 FP8 路径（`ComputeVF`）也存在同类问题！

### 检查 perblock 的历史 commit

```bash
git log --all --oneline --grep="FP8 perblock" -- moe_v3_gather_fp8_perblock_quant.h
```

没有找到相关记录。说明 perblock 的 FP8 路径可能是新增的，还没修复。

### 回到 pergroup 代码

```cpp
// moe_v3_gather_fp8_group_quant.h ComputeFp8GroupQuantVF
__VEC_SCOPE__ {
    RegTensor<U> xLeftReg, xRightReg;
    RegTensor<float> absLeftReg, absRightReg;
    RegTensor<float> maxReg, maxRightReg;
    // ...

    // Phase 1: Max 计算
    while (groupElemNum >= FLOAT_REG_TENSOR_LENGTH) {
        dataMask = UpdateMask<U>(groupElemNum);
        DataCopy(xLeftReg, xAddr + i * FLOAT_REG_TENSOR_LENGTH, dataMask);
        // groupElemNum 从 18 递减到 0
        // ...
        i++;
    }
    
    if (groupElemNum > 0) {
        dataMask = UpdateMask<U>(groupElemNum);
        DataCopy(xLeftReg, xAddr + i * FLOAT_REG_TENSOR_LENGTH, dataMask);
        // ...
        i++;
    }
    
    // ... scale 计算（不依赖 groupElemNum）...

    // Phase 2: Quant 计算
    for (uint32_t i = 0; i < outputVfLoop; i++) {
        if (groupElemNum > 0) {  // ⚠️ groupElemNum = 0，分支跳过！
            dataMask = UpdateMask<U>(groupElemNum);
            DataCopy(xLeftReg, xAddr + i * FLOAT_REG_TENSOR_LENGTH, dataMask);
            Mul(quantLeftReg, xLeftReg, invScaleReg, dataMask);
            Cast(outReg, quantLeftReg, castTrait0, dataMask);
            DataCopy(yAddr + i * FLOAT_REG_TENSOR_LENGTH, outReg, dataMask);
        }
        // Quant 从未执行，UB 输出位置未写入！
    }
}
```

---

## 第三轮：根因定位

### 核心问题

`UpdateMask` 的 API 行为：**按引用传递参数，并在内部递减**（饱和到 0）。

```cpp
template <typename T>
inline MaskReg UpdateMask(uint32_t& remaining) {
    uint32_t size = std::min(remaining, VECTOR_REG_LENGTH);
    remaining -= size;  // ⚠️ 修改调用方的变量
    // ...
}
```

### 问题分析

对于 `cols = 786`：
- 总共 `786 / 128 = 6` 个完整 group（每个 128 元素）
- 剩余 `786 % 128 = 18` 个元素构成第 7 个 group

**执行流程**：

```
Group 0-5: groupElemNum = 128
├─ Max 计算: UpdateMask(groupElemNum) 1次 → 128 → 0
├─ Scale 计算: 正常
└─ Quant 计算: UpdateMask(groupElemNum) 1次 → 0 → 0 ✓

Group 6: groupElemNum = 18
├─ Max 计算: UpdateMask(groupElemNum) 1次 → 18 → 0
├─ Scale 计算: 正常（不依赖 groupElemNum）
└─ Quant 计算: 
   ├─ 检查 if (groupElemNum > 0): false (因为已经是 0)
   ├─ 分支被跳过
   └─ UB[yAddr + i * 128] 未被写入 → 保留残留值（NaN）
```

### 为什么前 768 个元素正常

前 768 个元素属于 Group 0-5，每个 group 的 `groupElemNum = 128`：
- Max 阶段：`128 / 128 = 1` 次循环，`groupElemNum` 从 128 → 0
- Quant 阶段：同样 1 次循环，`groupElemNum` 再次从 128 → 0（每个阶段独立）

**关键差异**：完整 group 的 `groupElemNum` 在每个循环中都会重新赋值为 128（`CeilDiv`），但最后一个不完整 group 只有 18，一旦耗尽就回不来了。

---

## 第四轮：修复方案

### 修复策略

引入**分离的计数器**，避免变量状态污染：

```cpp
__aicore__ inline void ComputeFp8GroupQuantVF(
    // ...
) {
    uint32_t inputElemNum = groupElemNum;   // Phase 1 专用
    uint32_t outputElemNum = groupElemNum;  // Phase 2 专用
    
    __VEC_SCOPE__ {
        // Phase 1: Max 计算（消耗 inputElemNum）
        uint32_t i = 0;
        while (inputElemNum >= FLOAT_REG_TENSOR_LENGTH) {
            dataMask = UpdateMask<U>(inputElemNum);  // inputElemNum 递减
            // ...
            i++;
        }
        
        if (inputElemNum > 0) {
            dataMask = UpdateMask<U>(inputElemNum);
            // ...
            i++;
        }
        
        // ... scale 计算（不依赖任何 counter）...

        // Phase 2: Quant 计算（消耗 outputElemNum）
        for (uint32_t i = 0; i < outputVfLoop; i++) {
            if (outputElemNum > 0) {  // ✅ 独立的计数器，未被污染
                dataMask = UpdateMask<U>(outputElemNum);  // outputElemNum 递减
                DataCopy(xLeftReg, xAddr + i * FLOAT_REG_TENSOR_LENGTH, dataMask);
                Mul(quantLeftReg, xLeftReg, invScaleReg, dataMask);
                Cast(outReg, quantLeftReg, castTrait0, dataMask);
                DataCopy(yAddr + i * FLOAT_REG_TENSOR_LENGTH, outReg, dataMask);
            }
        }
    }
}
```

### 代码修改

编辑 `moe_v3_gather_fp8_group_quant.h`：

```diff
@@ line 95-145 @@
 __aicore__ inline void ComputeFp8GroupQuantVF(
     // ...
 ) {
+    uint32_t inputElemNum = groupElemNum;   // Phase 1 专用
+    uint32_t outputElemNum = groupElemNum;  // Phase 2 专用
+    
     RegTensor<U> xLeftReg, xRightReg;
     // ...

     __VEC_SCOPE__ {
         // Phase 1: Max 计算
         uint32_t i = 0;
-        while (groupElemNum >= FLOAT_REG_TENSOR_LENGTH) {
-            dataMask = UpdateMask<U>(groupElemNum);
+        while (inputElemNum >= FLOAT_REG_TENSOR_LENGTH) {
+            dataMask = UpdateMask<U>(inputElemNum);
             // ...
             i++;
         }
         
-        if (groupElemNum > 0) {
-            dataMask = UpdateMask<U>(groupElemNum);
+        if (inputElemNum > 0) {
+            dataMask = UpdateMask<U>(inputElemNum);
             // ...
             i++;
         }

         // ... scale 计算 ...

         // Phase 2: Quant 计算
         for (uint32_t i = 0; i < outputVfLoop; i++) {
-            if (groupElemNum > 0) {
-                dataMask = UpdateMask<U>(groupElemNum);
+            if (outputElemNum > 0) {
+                dataMask = UpdateMask<U>(outputElemNum);
                 // ...
             }
         }
     }
 }
```

---

## 第五轮：验证修复

### 验证环境

```bash
# 编译
cd /mnt/workspace/gitCode/cann/ops-transformer
make clean && make

# 运行测试
python3 test_moe_init_routing_v3.py --case "n=2,cols=786,k=1122" --template pergroup
```

### 修复前

```
=== Test Case: n=2, cols=786, k=1122 ===
Template: moe_v3_gather_fp8_group_quant
Input shape: [2, 786, 1122]
Output x precision: 97.70%
Failed elements: 18 (all NaN)
Location: last 18 elements of output
```

### 修复后

```
=== Test Case: n=2, cols=786, k=1122 ===
Template: moe_v3_gather_fp8_group_quant
Input shape: [2, 786, 1122]
Output x precision: 100.00% ✓
All 1572 elements correct
```

用户确认：
> "好像可以了"

---

## 第六轮：额外问题修复

### 代码审查：发现缺失的 RegTensor 声明

在完整阅读 `ComputeFp8GroupQuantVF` 时，发现 `absRightReg` 被使用但从未声明：

```cpp
__VEC_SCOPE__ {
    RegTensor<U> xLeftReg, xRightReg;
    RegTensor<float> absLeftReg, absRightReg;  // ← absRightReg 被使用但声明在哪？
    // ...
}
```

搜索整个文件：

```bash
grep -n "absRightReg" moe_v3_gather_fp8_group_quant.h
```

```
# line 110: RegTensor<float> absLeftReg, absRightReg;
# line 115: Abs(absRightReg, xRightReg, absMask);
```

**发现问题**：`absRightReg` 在 line 110 被声明，但声明的位置在 `__VEC_SCOPE__` 之外！

### 修复

```diff
@@ line 108-112 @@
 ) {
-    uint32_t inputElemNum = groupElemNum;
-    uint32_t outputElemNum = groupElemNum;
+    uint32_t inputElemNum = groupElemNum;   // Phase 1 专用
+    uint32_t outputElemNum = groupElemNum;  // Phase 2 专用
+    RegTensor<U> xLeftReg, xRightReg;
+    RegTensor<float> absLeftReg, absRightReg;  // ← 补充声明
 
     __VEC_SCOPE__ {
```

### 验证

重新编译运行，精度仍为 100%，确认修复完整。

---

## 问题总结

### 根因

**`UpdateMask` 的引用递减特性导致计数器状态污染**。

同一个 `groupElemNum` 变量在两个计算阶段（Max 和 Quant）被共享：
- Phase 1（Max）：从 18 递减至 0
- Phase 2（Quant）：检查 `groupElemNum > 0` 失败，分支被跳过

### 触发条件

- `cols` 不是 `FLOAT_REG_TENSOR_LENGTH (128)` 的整数倍
- 最后一个不完整 group 的 `groupElemNum ≤ 128`
- 该 group 的 Quant 阶段被绕过，UB 输出位置未写入，保留残留值（NaN）

### 修复方案

引入分离的计数器 `inputElemNum` 和 `outputElemNum`，互不污染。

### 同类问题

之前 `moe_v3_gather_fp8_perblock_quant.h` 也出现过同类问题，同样通过分离 `inputNum` 和 `outputNum` 修复。这是 `UpdateMask` API 使用的通用模式。

---

## 经验教训

1. **API 副作用的警惕性**
   - `UpdateMask` 的引用递减行为容易忽视
   - 同一变量多次调用会累积修改
   - 需要在 API 文档中明确标注此副作用

2. **分离关注点**
   - 不同计算阶段使用独立的计数变量
   - 避免变量状态在阶段间传递
   - 提高代码可读性和可维护性

3. **边界条件的重视**
   - 不完整 group 是常见 bug 来源
   - 需要单独测试 `cols % 128 != 0` 的情况
   - 建议增加单元测试覆盖边界场景

4. **误判的反思**
   - 初始误判 RoundUp Scale 除零，浪费用户时间
   - 应该先询问输入范围，再提出假设
   - 深入代码对比比凭直觉猜测更高效

---

## 附录：UpdateMask API 行为

```cpp
template <typename T>
inline MaskReg UpdateMask(uint32_t& remaining) {
    // 计算本次可处理元素数
    uint32_t size = std::min(remaining, VECTOR_REG_LENGTH);
    
    // ⚠️ 修改调用方传入的引用
    remaining -= size;
    
    // 生成 mask（前 size 个位置为 1，其余为 0）
    MaskReg mask;
    // ...
    return mask;
}
```

**使用建议**：
- 每个计算阶段使用独立的 counter
- 调用前记录 counter 的初始值，便于调试
- 在循环结束后，counter 应该恰好为 0（验证完整性）
