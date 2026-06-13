# 代码检视报告 — PR #6142

## 检视概览
- **PR 链接**: https://gitcode.com/cann/ops-transformer/pull/6142
- **变更文件**: `attention/common/op_kernel/memcopy/copy_gm_to_l1.h`
- **代码侧别**: Kernel侧
- **变更范围**: `CopySingleMatrixNDToNZ` 函数 else 分支（行 46-68）
- **检视文档**: cpp-secure、ascendc-api、ascendc-perf、ascendc-topk
- **总条例数**: 46（适用条例，过滤后）
- **检视时间**: 2026-06-01

## 变更概要

PR #6142 的实质变更为 `copy_gm_to_l1.h` 中 `CopySingleMatrixNDToNZ` 函数的 else 分支重构（+8/-12行）。变更内容：

1. **新增** `BLOCK_ELEMENT_CNT` 编译期常量，按元素类型区分：`int4b_t` → 64，其他 → 16
2. **移除** 中间局部变量 `srcNdMatrixNum`、`srcNdMatrixStride`、`dstNzMatrixStride`，直接在循环中使用原始参数
3. **修正** L1 offset 步进：从硬编码 `16` 改为 `BLOCK_ELEMENT_CNT`，**修复了 int4b 类型场景下 L1 偏移计算错误**（原代码对所有类型均使用 16，对 int4b（0.5B/元素）仅产生 8B 偏移而非正确的 32B）
4. **消除** 函数参数覆写：旧代码中 `nValue=1; srcDValue=dValue;` 直接修改入参，新代码直接使用字面量

## 检视统计

| 状态 | 条例数 | 占比 |
|-----|--------|------|
| PASS | 46 | 100% |
| FAIL（发现问题） | 0 | 0% |
| SUSPICIOUS（需关注） | 0 | 0% |

## 发现问题（HIGH 置信度）

无。

## 需关注（MED 置信度）

无。

## 疑似（LOW 置信度）

无。

## 通过条例

### cpp-secure（16条）
- 1.1 保证静态类型安全
- 1.2 保证内存安全
- 1.3 禁止使用未定义行为
- 2.1 有符号整数运算不溢出
- 2.2 无符号整数运算不回绕
- 2.3 除法/余数运算除零保护
- 3.1 禁止使用未初始化的变量
- 3.3 数组索引校验
- 3.4 禁止 sizeof 指针
- 3.5 指针使用前判空
- 4.1 外部输入合法性校验
- 4.2 内存操作长度校验
- 9.1 禁止逐位操作非 trivially copyable 对象
- 10.3 敏感信息使用后清零
- 10.4 结构体字段末尾添加
- 10.5 接口变更考虑兼容性

### ascendc-api（10条）
- API-1 禁止使用 GlobalTensor::SetValue/GetValue
- API-2 禁止使用 std:: 计算函数
- API-3 DataCopy/DataCopyPad 对齐要求
- API-4 Compare API 256字节对齐要求
- API-6 AllocTensor/FreeTensor 必须配对使用
- API-7 禁止动态内存分配
- API-8 repeatTimes 限制（≤255）
- API-9 Cast RoundMode 正确性
- API-10 DataCopyParams vs DataCopyExtParams 单位差异
- API-12 CrossCoreSetFlag/WaitFlag 必须对称

### ascendc-perf（13条）
- PERF-1 循环内禁止逐元素操作
- PERF-2 禁止写死硬件参数
- PERF-3 Double Buffer 使用
- PERF-4 PipeBarrier 优化
- PERF-5 单次搬运量优化
- PERF-6 避免 GM 重复读取
- PERF-7 尾块处理正确性
- PREC-1 流水线同步正确性
- PREC-2 除零保护
- PREC-3 FP16 中间精度保护
- TIL-1 多核负载均衡
- TIL-2 片上缓存容量不溢出
- TIL-3 Buffer 规划合理性

### ascendc-topk（7条）
- 3 生命周期内使用局部变量指针，避免野指针
- 6 必须考虑nan/inf/+0/-0等特殊值和边界值处理
- 8 gm内存偏移或大小必须用int64表示
- 9 atomic累加需src(ub)与dst(gm)清零处理
- 10 可整数计算时不允许转浮点数计算
- 11 通信算子融合需核间同步
- 12 宏定义中临时变量命名不能和外部变量冲突

## 检视结论

PR #6142 对 `copy_gm_to_l1.h` 的变更是一个**高质量的代码重构改进**：

1. **修复了 int4b 类型的 L1 偏移 bug**：原代码对所有类型硬编码 `dstNzMatrixStride=16`，对 int4b_t（0.5 字节/元素）仅产生 8 字节步进步而非正确的 32 字节。新代码使用类型感知的 `BLOCK_ELEMENT_CNT`（int4b→64，其他→16），确保 NZ 格式的 32B 行对齐。
2. **代码简化**：移除了冗余中间变量（`srcNdMatrixNum`/`srcNdMatrixStride`），直接使用函数参数和编译期常量，提高可读性。
3. **消除副作用**：旧代码直接覆写值参数 `nValue` 和 `srcDValue`，新代码使用字面量，行为更清晰。

全量 46 条条例检视均通过，未发现任何编码规范违规。

---

## 被检视代码

> 本次检视的变更部分（`copy_gm_to_l1.h` 行 46-68）

```cpp
46:     } else {
47:         constexpr uint32_t BLOCK_ELEMENT_CNT = IsSameType<T, int4b_t>::value ? 64 : 16;
48:         uint64_t l1Offset = 0;
49:         uint64_t gmOffset = 0;
50:         for (uint32_t i = 0; i < nValue; i++) {
51:             nd2nzPara.ndNum = 1;
52:             nd2nzPara.nValue = 1;
53:             if constexpr (IsSameType<T, int4b_t>::value) {
54:                 constexpr uint32_t HALF_SIZE_DIVISOR = 2;
55:                 nd2nzPara.dValue = dValue / HALF_SIZE_DIVISOR;
56:                 nd2nzPara.srcDValue = dValue / HALF_SIZE_DIVISOR;
57:             } else {
58:                 nd2nzPara.dValue = dValue;
59:                 nd2nzPara.srcDValue = dValue;
60:             }
61:             nd2nzPara.dstNzC0Stride = dstNzC0Stride;
62:             nd2nzPara.dstNzNStride = 1;
63:             nd2nzPara.srcNdMatrixStride = 0;
64:             nd2nzPara.dstNzMatrixStride = 0;
65:             DataCopy(l1Tensor[l1Offset], gmTensor[gmOffset], nd2nzPara);
66: 
67:             gmOffset += srcDValue;
68:             l1Offset += BLOCK_ELEMENT_CNT;
69:         }
70:     }
```
