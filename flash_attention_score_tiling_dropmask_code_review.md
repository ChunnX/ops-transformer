# 代码检视报告：flash_attention_score_tiling_dropmask.cpp

> **检视文件**：`attention/flash_attention_score/op_host/arch35/flash_attention_score_tiling_dropmask.cpp`
> **检视日期**：2026-06-03
> **检视方法**：假设检验驱动
> **检视规范来源**：C++ 安全编码规范（cpp-secure.md）、通用编码规范（cpp-general.md）、代码风格规范（cpp-style.md）、TOPK 问题清单（ascendc-topk.md）

---

## 检视概要

| 类别 | 高风险 | 中风险 | 低风险 | 需关注 | 合计 |
|------|--------|--------|--------|--------|------|
| 数值安全 | 1 | 1 | 0 | 1 | 3 |
| 内存安全 | 0 | 1 | 0 | 0 | 1 |
| LOG API | 0 | 0 | 1 | 0 | 1 |
| 编码规范 | 0 | 1 | 0 | 0 | 1 |
| 逻辑缺陷 | 1 | 0 | 0 | 0 | 1 |
| **合计** | **2** | **3** | **1** | **1** | **7** |

---

## 风险点详情

### P1 [高/逻辑缺陷] DoOpTiling 成功路径返回错误码

**规范来源**：逻辑正确性（无具体规范编号，属于 SEC-1.3 禁止使用未定义行为的延伸）

**风险代码**（第 66 行）：
```cpp
return ge::GRAPH_PARAM_INVALID;
```

**假设检验过程**：

- **H0**：DoOpTiling 返回值正确
- **H1**：DoOpTiling 成功路径返回了错误码
- **自信值初始**：0%

**证据收集**：

| 证据类型 | 分析动作 | 分值 |
|---------|---------|------|
| 规范违反 | DoOpTiling 成功完成 tiling 计算后应返回 `ge::GRAPH_SUCCESS`，但实际返回 `ge::GRAPH_PARAM_INVALID` | +40% |
| 上下文对比 | 查阅同类 tiling 实现：`flash_attention_score_tiling_basic.cpp` DoOpTiling 返回 `GRAPH_SUCCESS`（第 32/218/224 行）；`flash_attention_score_tiling_varlen.cpp` DoOpTiling 返回 `GRAPH_SUCCESS`（第 34/265/780 行）——**所有正常 tiling 模板的成功路径均返回 `GRAPH_SUCCESS`** | +25% |
| 函数调用链风险 | `DoOpTiling()` 由框架 TilingBaseClass 流程调用，返回 `GRAPH_PARAM_INVALID` 会被框架视为"该模板不适用"，可能跳过已计算好的 tiling 数据 | +25% |

**自信值**：90% → **判定：存在风险**

**影响分析**：
- 第 33 行的 `GRAPH_PARAM_INVALID` 用于"不需要 DropMask 时跳过"是合理的
- 但第 66 行在成功计算完所有 tiling 数据（`set_shapeTotalSize`、`set_multiCoreFactorSize` 等）后仍返回 `GRAPH_PARAM_INVALID`，将导致框架误判此模板失败，可能丢弃已设置的 tiling 数据

**建议修复**：
```cpp
// 第 66 行：成功路径应返回 GRAPH_SUCCESS
return ge::GRAPH_SUCCESS;
```

---

### P2 [高/数值安全] shapeTotalSize 五维乘法可能溢出 int64_t

**规范来源**：SEC-2.1 确保有符号整数运算不溢出 `[适用: All]`

**风险代码**（第 36-38 行）：
```cpp
int64_t shapeTotalSize = inputParamsRegbase_->get_bSize() * inputParamsRegbase_->get_n2Size() *
                         inputParamsRegbase_->get_gSize() * inputParamsRegbase_->get_s1Size() *
                         inputParamsRegbase_->get_s2Size();
```

**假设检验过程**：

- **H0**：五维乘法不会溢出 int64_t
- **H1**：五维乘法在极端输入下可能溢出 int64_t
- **自信值初始**：0%

**证据收集**：

| 证据类型 | 分析动作 | 分值 |
|---------|---------|------|
| 工具验证 | 使用 GCC `__int128` 验证：当 b=128, n2=128, g=8, s1=INT32_MAX(2147483647), s2=INT32_MAX 时，真实值 > INT64_MAX(9.2×10^18)，计算结果溢出截断为负值 | +40% |
| 业务约束缺失 | 代码中 `AnalyzeLayout()` 仅校验 s1/s2 不超过 INT32_MAX，未对五维乘积做溢出保护 | +30% |
| 实际可行性 | 大模型场景下 b×n2×g×s1×s2 可能接近溢出边界：如 b=128, n2=128, g=8, s1=65536, s2=65536 → ≈5.5×10^13（安全），但更大组合可能溢出 | +25% |

**自信值**：95% → **判定：存在溢出风险**

**工具验证详细结果**：
- b=128, n2=128, g=8, s1=INT32_MAX, s2=INT32_MAX：int128 真实值高位=32767，溢出！int64_t 计算结果为负值
- 实际合理场景 b=128, n2=128, g=8, s1=32768, s2=32768：结果 ≈1.4×10^12，安全

**建议修复**：对乘法结果做溢出检查，或在乘法前对各维度做范围校验：
```cpp
// 方案1：分步乘法 + 溓出检查
int64_t shapeTotalSize = inputParamsRegbase_->get_bSize();
OP_CHECK_IF(shapeTotalSize > INT64_MAX / inputParamsRegbase_->get_n2Size(), ...);
shapeTotalSize *= inputParamsRegbase_->get_n2Size();
// ... 逐维度检查并乘法

// 方案2：在 AnalyzeLayout 中增加总元素数上限校验
```

---

### P3 [中/数值安全] int64_t 到 int32_t 强转无溢出保护

**规范来源**：SEC-2.1 确保有符号整数运算不溢出 `[适用: All]`

**风险代码**（第 62、64 行）：
```cpp
dropmaskParamsRegbase_->set_multiCoreFactorSize(static_cast<int32_t>(multiCoreFactorSize));
dropmaskParamsRegbase_->set_baseUbCalSize(static_cast<int32_t>(baseUbCalSize));
```

**假设检验过程**：

- **H0**：multiCoreFactorSize 和 baseUbCalSize 均在 int32_t 范围内
- **H1**：强转可能导致截断
- **自信值初始**：0%

**证据收集**：

| 证据类型 | 分析动作 | 分值 |
|---------|---------|------|
| 规范违反 | SEC-2.1：int64_t → int32_t 强转未做溢出检查 | +40% |
| 业务约束分析 | baseUbCalSize = AlignDown(ubSize/DROP_MASK_CAL_SIZE, 2048)。ubSize 最大约 200KB，200000/6 ≈ 33333 → AlignDown → 32256，在 int32_t 范围内。但若未来 ubSize 增大，可能超出 | +25% |
| multiCoreFactorSize 分析 | CeilDivision(shapeSingleCoreSize×2048, baseUbCalSize)，即每核循环次数，典型值 ≤ 数百，远在 int32_t 范围内 | +0%（排除） |

**自信值**：65% → **判定：存在风险**

**建议修复**：
```cpp
OP_CHECK_IF(multiCoreFactorSize > std::numeric_limits<int32_t>::max(),
           OPS_REPORT_VECTOR_INNER_ERR(opName, "multiCoreFactorSize overflow int32"), return ge::GRAPH_FAILED);
OP_CHECK_IF(baseUbCalSize > std::numeric_limits<int32_t>::max(),
           OPS_REPORT_VECTOR_INNER_ERR(opName, "baseUbCalSize overflow int32"), return ge::GRAPH_FAILED);
```

---

### P4 [中/内存安全] 构造函数中 memset_s 使用指针未做空检查

**规范来源**：SEC-3.5 指针使用前判空 `[适用: All]`；SEC-8.3 检查安全函数返回值 `[适用: Tiling]`

**风险代码**（第 23-25 行）：
```cpp
OP_CHECK_IF(memset_s(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity(), 0,
                    context_->GetRawTilingData()->GetCapacity()) != EOK,
           OPS_REPORT_VECTOR_INNER_ERR(opName, "Fail to memset tiling data"), return;);
```

**假设检验过程**：

- **H0**：构造函数中 GetRawTilingData() 返回有效指针
- **H1**：构造函数在 CheckContext() 之前执行，指针可能为空
- **自信值初始**：0%

**证据收集**：

| 证据类型 | 分析动作 | 分值 |
|---------|---------|------|
| 上下文防御缺失 | `CheckContext()`（第 46-47 行）校验了 `GetRawTilingData()` 和 `GetData()` 非空，但构造函数在 `CheckContext()` 之前执行 | +30% |
| 调用链风险 | TilingBaseClass 构造 → FlashAttentionScoreTilingRegbase 构造（Reset）→ 子类构造（memset_s）→ 后续才调用 CheckContext。构造阶段 context 数据指针有效性依赖框架保证 | +25% |
| 同类代码模式 | 父类 `CheckContext()` 已做指针校验，构造函数跳过该校验是本代码库的通用模式，实际由框架保证 context 有效 | +0%（排除，但仍有理论风险） |

**自信值**：55% → **判定：需关注**（低于 60% 阈值，但仍值得标注）

**说明**：这是现有代码库的通用模式，框架通常保证 context 在构造时有效。但严格按 SEC-3.5 规范，应在使用指针前判空。

---

### P5 [中/编码规范] REGISTER_TILING_TEMPLATE_WITH_ARCH 使用魔鬼数字 81

**规范来源**：GEN-4.2 禁止使用魔鬼数字 `[适用: All]`

**风险代码**（第 84 行）：
```cpp
REGISTER_TILING_TEMPLATE_WITH_ARCH(FlashAttentionScore, FlashAttentionScoreTilingDropMaskRegbase, static_cast<int32_t>(NpuArch::DAV_3510), 81);
```

**假设检验过程**：

- **H0**：81 是有明确含义的常量
- **H1**：81 是魔鬼数字，无命名解释
- **自信值初始**：0%

**证据收集**：

| 证据类型 | 分析动作 | 分值 |
|---------|---------|------|
| 规范违反 | GEN-4.2：数字 81 无命名常量定义，含义不明（可能是 tiling 优先级/排序权重） | +40% |
| 上下文搜索 | 同文件其他模板注册也使用数字优先级参数，但无统一命名 | +25% |

**自信值**：65% → **判定：存在风险**

**建议修复**：
```cpp
constexpr int32_t DROP_MASK_TILING_PRIORITY = 81;
REGISTER_TILING_TEMPLATE_WITH_ARCH(FlashAttentionScore, FlashAttentionScoreTilingDropMaskRegbase, 
    static_cast<int32_t>(NpuArch::DAV_3510), DROP_MASK_TILING_PRIORITY);
```

---

### P6 [低/LOG API] OP_LOGD 格式说明符 %ld 与 int64_t 类型不完全匹配

**规范来源**：SEC-11.3 LOG API 参数类型与格式化说明符必须匹配 `[适用: Tiling]`

**风险代码**（第 45 行）：
```cpp
OP_LOGD(context_, "shapeTotalSize %ld dropTotalSize %ld.", shapeTotalSize, dropTotalSize);
```

**假设检验过程**：

- **H0**：%ld 在当前平台正确打印 int64_t
- **H1**：%ld 不是 int64_t 的标准安全说明符
- **自信值初始**：0%

**证据收集**：

| 证据类型 | 分析动作 | 分值 |
|---------|---------|------|
| 规范违反 | SEC-11.3：int64_t 推荐使用 `%lld`。`%ld` 在 64 位 Linux 上 long=64bit 可用，但跨平台不安全 | +40% |
| 平台特定 | 当前目标平台（Ascend NPU / 64位 Linux）上 `long` = 64bit，%ld 实际可正确打印 int64_t | -20%（排除部分） |

**自信值**：20% → **判定：低风险**（当前平台安全，但不符合规范建议）

**建议修复**：
```cpp
OP_LOGD(context_, "shapeTotalSize %lld dropTotalSize %lld.", shapeTotalSize, dropTotalSize);
```

---

### P7 [需关注/数值安全] baseUbCalSize 可能为零时作为除数使用

**规范来源**：SEC-2.3 确保除法运算不会导致除以零 `[适用: All]`；TOPK-6 必须考虑边界值处理

**风险代码**（第 58、63 行）：
```cpp
int64_t multiCoreFactorSize = CeilDivision(shapeSingleCoreSize * ubCalFactor, baseUbCalSize); // 第 58 行
dropmaskParamsRegbase_->set_multiCoreTotalSize(CeilDivision(shapeSplitCoreSize * ubCalFactor, baseUbCalSize)); // 第 63 行
```

**假设检验过程**：

- **H0**：baseUbCalSize 始终大于 0
- **H1**：极端场景下 baseUbCalSize 可能为 0，导致 CeilDivision 静默返回 0
- **自信值初始**：0%

**证据收集**：

| 证据类型 | 分析动作 | 分值 |
|---------|---------|------|
| 边界值分析 | baseUbCalSize = AlignDown(CeilDivision(ubSize, 6), 2048)。若 ubSize < 6×2048=12288，则 CeilDivision(ubSize,6) < 2048，AlignDown 结果为 0 | +25% |
| 函数调用链 | CeilDivision(x, 0) 返回 0 而非报错，导致 multiCoreFactorSize=0、multiCoreTotalSize=0，tiling 数据静默错误 | +25% |
| 业务约束 | 实际 ubSize 通常 ≥ 128KB，远大于 12288 临界值 | +0%（排除） |

**自信值**：50% → **判定：需关注**（低于 60%，但在极端配置下有风险）

**建议修复**：在 baseUbCalSize 计算后增加零值检查：
```cpp
OP_CHECK_IF(baseUbCalSize == 0,
           OPS_REPORT_VECTOR_INNER_ERR(opName, "baseUbCalSize is zero, ubSize too small"),
           return ge::GRAPH_FAILED);
```

---

## 检视统计

| 规范编号 | 规范名称 | 风险数量 |
|---------|---------|---------|
| SEC-2.1 | 有符号整数运算不溢出 | 2 (P2, P3) |
| SEC-2.3 | 除法/余数除零保护 | 1 (P7) |
| SEC-3.5 | 指针使用前判空 | 1 (P4) |
| SEC-11.3 | LOG API 参数类型匹配 | 1 (P6) |
| GEN-4.2 | 禁止魔鬼数字 | 1 (P5) |
| 逻辑缺陷 | DoOpTiling 返回值错误 | 1 (P1) |

---

## 附录：代码段与检视范围映射

| 代码段 | 行号范围 | 检视结果 |
|--------|---------|---------|
| 构造函数 memset_s | 19-26 | P4 (需关注) |
| DoOpTiling 早期返回 | 30-34 | PASS (语义正确) |
| shapeTotalSize 乘法 | 36-38 | P2 (高风险) |
| TND layout 循环累积 | 40-46 | P2 衍生 + OP_LOGD P6 |
| CeilDivision 除法链 | 49-50 | PASS (CeilDivision 内含零保护) |
| baseUbCalSize 计算 | 54-56 | P7 (需关注) |
| multiCoreFactorSize 除法 | 58 | P7 衍生 |
| shapeTotalSize AlignUp | 59 | PASS |
| tiling 数据设置 + int32 强转 | 61-64 | P3 (中风险) |
| SetScheduleMode | 65 | PASS |
| DoOpTiling 返回值 | 66 | P1 (高风险-BUG) |
| GetTilingKey | 69-72 | PASS |
| GetWorkspaceSize | 74-76 | PASS |
| CalcS1S2BasicBlock/CalcDBasicBlock | 79-80 | PASS |
| REGISTER 宏魔鬼数字 | 84 | P5 (中风险) |