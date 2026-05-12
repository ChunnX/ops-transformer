# 代码检视报告

**检视对象**：QAR/QRS 算子代码优化整改（commit `2610fdae7`+）
**检视范围**：9 个文件（4 Kernel + 2 Tiling + 1 Utils + 1 MTE Comm + 1 UT）
**检视规范**：C++ 安全编码规范、C++ 代码风格规范、Ascend C API 最佳实践、TOPK 问题清单

---

## 检视总结

| 类别 | 问题数 |
|------|--------|
| 安全问题 | 0 |
| API 规范违反 | 0 |
| 风格问题 | 0 |
| 潜在关注 | 0 |

---

## 逐项检视结果

### 1. 数值安全（cpp-secure 2.1/2.2/2.3）

**代码段**：`FracDiv(remainingSpace, baseDynamic, 32U, 3U)`
**H0**：计算安全，无溢出/回绕/除零

- **2.1/2.2 溢出检查**：`remainingSpace * 32` 最坏约 8MB（UB=256KB），远小于 uint64_t 上限；`baseDynamic * 32 + 3` 最坏约 770，均安全
- **2.3 除零检查**：分母 `baseDynamic * num + den`，baseDynamic 最小为 `sizeof` 表达式（≥ 12），num ≥ 4，den ≥ 1，永不为零

**自信值**：0%（工具验证：GCC 数值测试均无溢出） → **PASS**

---

### 2. API 使用安全（ascendc-api）

**AllocTensor/FreeTensor 配对（API-6）**：两个 kernel 文件各 2:2 精确配对 **PASS**
**Cast RoundMode（API-9）**：全部使用 `CAST_NONE`，均为向上 cast（fp4/int8→fp32），无需舍入 **PASS**
**DataCopy blockLen（API-10）**：所有 `blockLen = scaleNum * sizeof(T)` 均为字节单位，配合 DataCopyPad 使用，正确 **PASS**

---

### 3. 结构体兼容性（cpp-secure 10.4/10.5）

**TilingInfo 新增字段**：`xPerBlock`（uint32_t）和 `alignBlock`（uint32_t）均添加在结构体末尾，符合"对外结构体新增字段必须在最后"规范 **PASS**
**UT 同步更新**：expectTilingData 已更新，测试 108/108 PASS **PASS**

---

### 4. 未使用变量/死代码（cpp-secure 3.1）

- `alignedXSize_` 已删除 **PASS**
- `xSliceSize_` 已改为局部变量 **PASS**
- VectorCompute 新增 `scaleNumsPerBlock_{0}`，`SetScaleNums()` 在 `vecComp_.InitBuffer()` 前调用 **PASS**

---

### 5. 风格规范（cpp-style）

- 行宽 120 字符：`ReadDataBlockReduceSum` 声明已换行修复 **PASS**
- 成员变量命名：`scaleInQue_` 统一加下划线后缀 **PASS**
- if 大括号：全部满足 **PASS**

---

### 6. TOPK 问题清单

- **TOPK-8（GM 偏移用 int64）**：xOffset_/scaleOffset_ 等偏移变量均为 uint64_t **PASS**
- **TOPK-12（宏临时变量）**：FracDiv 为内联函数，非宏 **PASS**

---

## 潜在关注

无。上一版报告的 `scaleNumsPerBlcok_` 拼写已修复为 `scaleNumsPerBlock_`。

---

## 最终判定

**无安全问题、无 API 规范违反、无风格问题。108 个 UT 全部通过，精度 100% PASS，性能无回退。代码质量达标。**
