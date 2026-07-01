# 代码检视报告 — PR #7464（更新后复检）

## 检视概览

- PR 链接：https://gitcode.com/cann/ops-transformer/pull/7464
- 算子：`fused_infer_attention_score`
- 复检 merge 提交：`b65d3a2cf00298d884735c61ec5005913edf2451`（2026-06-25 15:03）
- 变更文件：
  - `attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_impl.cpp`
  - `attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_impl.h`
- 代码侧别：**Tiling 侧（Host 侧）**
- 变更性质：cleancode 重构（大函数拆分、条件提取、`UpdateTilingKeyConfig` 表驱动化、工具函数复用、**新增空指针防护**）
- 检视文档：cpp-secure（31）+ cpp-general（44）+ ascendc-topk（11）+ ascendc-perf Tiling 设计（5）；compile-secure 不适用
- 总适用条例：**91 条**（25 组、3 波并行全量覆盖）

## 检视统计

| 状态 | 条例数 | 占比 |
|-----|--------|------|
| PASS | 89 | 97.8% |
| FAIL（发现问题） | 2 | 2.2% |
| SUSPICIOUS | 0 | 0% |

## 与上轮检视对比（已修复项）

| 上轮发现 | 本轮状态 | 说明 |
|---------|---------|------|
| SEC-3.5 指针未判空（`SetAntiQuantInfo`/`SetMaskTilingData`） | ✅ **已修复** | 新增 `scaleTensor == nullptr`（行1867）、`attenMask.tensor == nullptr`（行1751前）显式判空 |
| SEC-1.3 `int64` 乘法理论溢出 | ✅ 已澄清为改进 | `static_cast<int64_t>(gSize)*s1Size` 是防回绕改进，PASS |
| SEC-10.1 `std::string(layOut)` 空指针 | ✅ 已澄清 | 上游 parser `CheckRequiredAttrExistence` 已校验 `layOut!=nullptr`，PASS |
| SEC-11.3 `bool` 用 `%u` | ✅ 降级 PASS | 预存搬移、64 位平台无错位风险 |
| **GEN-15.2 出参用 `T&`** | ❌ **仍存在** | 未修复 |
| **GEN-4.2 魔鬼数字** | ❌ **仍存在** | 未修复 |

---

## 发现问题（HIGH 置信度）

### [cpp-general 15.2] 函数传参传递，要求入参用 `const T &`，出参用 `T *`

- **问题描述**：静态工具函数 `SetSparseCompressMode` 的出参 `inputParams` 使用非 const 引用 `InputParamsRegbase &`（函数内调用 `set_attenMaskCompressMode` 写入），违反"出参用 `T *`"约定。
- **代码片段**（`fused_infer_attention_score_tiling_impl.cpp` 行 1706-1716）：
  ```cpp
  static void SetSparseCompressMode(const FiaTilingInfo &fiaInfo,
                                    InputParamsRegbase &inputParams)   // 出参用 T&，规范要求 T*
  {
      static const std::map<uint32_t, uint8_t> sparseCompressModeMap = {{SPARSE_MODE_NO_MASK, 0},
                                                                        {SPARSE_MODE_ALL_MASK, 0},
                                                                        {SPARSE_MODE_LEFT_UP, 1},
                                                                        {SPARSE_MODE_RIGHT_DOWN, 2},
                                                                        {SPARSE_MODE_BAND, 3}};
      inputParams.set_attenMaskCompressMode(
          GetMapValueOrDefault(sparseCompressModeMap, static_cast<uint32_t>(fiaInfo.sparseMode)));
  }
  ```
- **修复建议**：出参改指针，调用处（行 1726）取地址：
  ```cpp
  static void SetSparseCompressMode(const FiaTilingInfo &fiaInfo, InputParamsRegbase *inputParams);
  // 调用：SetSparseCompressMode(fiaInfo, &inputParams);
  ```
- **备注**：本文件既有 setter 普遍使用 `&` 引用风格，团队可结合文件一致性裁量是否豁免。

### [cpp-general 4.2] 禁止使用魔鬼数字 — `CheckSmallHeadOptimization` / `configRules` / `ComputeSparseType`

- **问题描述**：多处新增代码使用裸字面量表达 sparseMode / 对齐尺寸 / sparseType，而代码库已存在对应命名常量（`SPARSE_MODE_LEFT_UP=2`、`SPARSE_MODE_NO_MASK=0`、`SPARSE_MODE_BAND=4`、`SINNER_*`、`SOUTER_*`）。
- **代码片段 1**（`CheckSmallHeadOptimization`，行 330-336）：
  ```cpp
      if (fiaInfo.sparseMode == 0) {                              // 应为 SPARSE_MODE_NO_MASK
          preTokens = (preTokens > 0) ? 0 : preTokens;
      } else if (fiaInfo.sparseMode == 4) {                       // 应为 SPARSE_MODE_BAND
          nextTokens = (nextTokens > 0) ? 0 : nextTokens;
      }
      if (fiaInfo.sparseMode == 2 || preTokens + nextTokens <= 128) {  // 2 应为 SPARSE_MODE_LEFT_UP；128 应具名
          return false;
      }
  ```
- **代码片段 2**（`configRules` 表 antiQuant 条目，行 1306-1315，量化条目用字面量而非量化条目用 `SOUTER_*`/`SINNER_*`）：
  ```cpp
      static const std::vector<TilingConfigRule> configRules = {
          {true, 16, 1024, DSIZE_64, DSIZE_64, Config_S1Aligned16_S2Aligned1024_DAligned64_DVAligned64},
          {true, 16, 512,  DSIZE_64, DSIZE_64, Config_S1Aligned16_S2Aligned512_DAligned64_DVAligned64},
          {true, 32, 512,  DSIZE_64, DSIZE_64, Config_S1Aligned32_S2Aligned512_DAligned64_DVAligned64},
          // ... 16/32/128/256/512/1024 等裸字面量
  ```
- **代码片段 3**（`ComputeSparseType`，行 1773-1802）：`return 0/3/4;` 多处裸返回，建议定义 `SPARSE_TYPE_NONE=0`、`SPARSE_TYPE_CAUSAL=3`、`SPARSE_TYPE_WINDOW=4`。
- **修复建议**：sparseMode 比较统一替换为 `SPARSE_MODE_*` 命名常量；`configRules` 中 `32/128/256/512` 替换为 `SOUTER_32`/`SINNER_128/256/512`，`16/1024` 补充具名常量；sparseType 返回值定义具名 `constexpr`。

---

## 通过条例（89 条，节选关键安全/性能项）

- **cpp-secure（PASS 31，全部通过）**：1.1 1.2 1.3 2.1 2.2 2.3 3.1 3.2 **3.5** 3.3 3.4 3.6 4.1 4.2 5.1 5.2 5.3 5.4 8.1 8.2 8.3 9.1 **10.1** 10.2 10.3 10.4 10.5 11.1 11.2 **11.3** 11.4（**加粗为本轮修复后转 PASS**）
- **cpp-general（PASS 42）**：除 4.2、15.2 外全部通过
- **ascendc-topk（PASS 11，全部通过）**：1~8 10 12 13
- **ascendc-perf Tiling（PASS 5，全部通过）**：PERF-2 PERF-7 TIL-1 TIL-2 TIL-3（负载均衡/尾块/缓存容量重构后等价）

---

## 总体结论

更新后的 PR **已修复上轮全部安全红线相关项**（`SetAntiQuantInfo`/`SetMaskTilingData` 指针判空已补齐），安全维度全部通过，TilingKey 表驱动重构经逐条比对与旧 if-else 链语义等价，负载均衡/缓存容量等性能项均无回退。

**剩余 2 项均为 cpp-general 代码风格/可维护性条款（非安全缺陷）：**
1. `SetSparseCompressMode` 出参用 `T&`（建议改 `T*`，或按文件一致性豁免）。
2. 多处魔鬼数字（建议替换为已有 `SPARSE_MODE_*` / `SOUTER_*` / `SINNER_*` 命名常量）。

整体质量良好，建议合入前一并整改上述两处风格问题。
