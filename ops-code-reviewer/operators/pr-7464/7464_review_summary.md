# 代码检视报告 — PR #7464

## 检视概览

- PR 链接：https://gitcode.com/cann/ops-transformer/pull/7464
- 算子：`fused_infer_attention_score`
- 变更文件：
  - `attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_impl.cpp`
  - `attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_impl.h`
- 代码侧别：**Tiling 侧（Host 侧）**
- 变更性质：cleancode 重构（长函数拆分为职责单一小函数、重复模式抽取为静态工具函数、`UpdateTilingKeyConfig` 的 if-else 链改为数据驱动 `TilingConfigRule` 向量匹配、Early Return 简化控制流）
- 检视文档：cpp-secure（31）+ cpp-general（44）+ ascendc-topk（11）+ ascendc-perf Tiling 设计（5）；compile-secure 因无构建文件变更不适用
- 总适用条例：**91 条**（按 Tiling 侧 `[适用: All]`+`[适用: Tiling/Host]` 过滤）
- diff：`./operators/.pr_diff/7464.diff` | 完整源码：`./operators/.pr_repo/7464/`

## 检视统计

| 状态 | 条例数 | 占比 |
|-----|--------|------|
| PASS | 85 | 93.4% |
| FAIL（发现问题） | 1 | 1.1% |
| SUSPICIOUS（需关注 + 疑似） | 5 | 5.5% |

> 说明：本 PR 为纯重构，未引入新业务逻辑。多数"需关注/疑似"项为**重构搬移保留的预存模式**（旧代码 `-` 行存在同样写法），非本次新引入的缺陷。

---

## 发现问题（HIGH 置信度）

### [cpp-general 15.2] 函数传参传递，要求入参用 `const T &`，出参用 `T *`

- **问题描述**：新增静态工具函数 `SetSparseCompressMode` 的出参 `inputParams` 使用非 const 引用 `InputParamsRegbase &`，违反"出参用 `T *`（指针）"的约定。引用作为出参时，调用方无法从调用语法区分入参/出参，降低可读性。
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
- **修复建议**：出参改为指针，调用处（行 1724）相应取地址：
  ```cpp
  static void SetSparseCompressMode(const FiaTilingInfo &fiaInfo, InputParamsRegbase *inputParams);
  // 调用：SetSparseCompressMode(fiaInfo, &inputParams);
  ```
- **备注**：该规范属编码约定（非功能缺陷）；本文件既有 setter 普遍使用 `auto &inputParams` 引用风格，团队可据本文件一致性决定是否豁免。

---

## 需关注（MED 置信度）

### [cpp-secure 3.5] 指针使用前必须判空 — `SetAntiQuantInfo` / `SetMaskTilingData`

- **问题描述**：两处新函数在解引用 `gert::Tensor*` 前未判空。上游 checker（`system_prefix_checker.cpp`、`paged_attention_checker.cpp` 等）存在对应 tensor 为 `nullptr` 的分支，说明该指针在部分路径下可能为空。
- **重要校对结论**：diff 显示**旧代码（`-` 行）存在完全相同的无判空解引用**（diff 行 1082-1085、823-832），即此为**重构搬移保留的预存模式**，非本次新引入。重构是补齐判空的良好时机。
- **代码片段 1**（`SetAntiQuantInfo`，行 1858-1872，仅 ANTI_QUANT 模式解引用）：
  ```cpp
  void FusedInferAttentionScoreTilingImpl::SetAntiQuantInfo(const FiaTilingInfo &fiaInfo)
  {
      auto &inputParams = faRunTilingAdapter_.inputParamsRegbase;
      if (fiaInfo.quantMode != FiaQuantMode::ANTI_QUANT) {
          inputParams.set_antiquantPerTensorFlag(0);
          return;
      }
      auto scaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
      uint32_t scaleDimNum = scaleTensor->GetStorageShape().GetDimNum();   // 行1868 未判空
      uint32_t scaleShape = scaleTensor->GetStorageShape().GetDim(0);      // 行1869 未判空
      bool perTensor = (scaleDimNum == 1 && scaleShape == 1);
      inputParams.set_antiquantPerTensorFlag(perTensor ? 1 : 0);
  }
  ```
- **代码片段 2**（`SetMaskTilingData`，行 1742-1755，`attenMaskFlag` 为 true 不等价于 tensor 非空）：
  ```cpp
      if (!fiaInfo.attenMaskFlag) {
          ...
          return;
      }
      const auto &shape = fiaInfo.opParamInfo.attenMask.tensor->GetStorageShape();  // 行1751 未判空
      uint64_t maskDimNum = shape.GetDimNum();
  ```
- **修复建议**：解引用前增加判空（为空时走默认/早返回分支）：
  ```cpp
  auto scaleTensor = fiaInfo.opParamInfo.keyAntiquantScale.tensor;
  if (scaleTensor == nullptr) { inputParams.set_antiquantPerTensorFlag(0); return; }
  ```

### [cpp-general 4.2] 禁止使用魔鬼数字 — `CheckSmallHeadOptimization` / `ComputeSparseType`

- **问题描述**：两处新增函数使用裸字面量表达 sparseMode / sparseType，而代码库已存在对应命名常量（`SPARSE_MODE_NO_MASK`/`SPARSE_MODE_BAND` 等），同文件其他位置（如 `ComputeSparseType` 的 `switch` 分支）已使用命名常量，存在不一致。
- **代码片段 1**（`CheckSmallHeadOptimization`，行 330-336）：
  ```cpp
      if (fiaInfo.sparseMode == 0) {                              // 应为 SPARSE_MODE_NO_MASK
          preTokens = (preTokens > 0) ? 0 : preTokens;
      } else if (fiaInfo.sparseMode == 4) {                       // 应为对应命名常量
          nextTokens = (nextTokens > 0) ? 0 : nextTokens;
      }
      if (fiaInfo.sparseMode == 2 || preTokens + nextTokens <= 128) {  // 2、128 为魔鬼数字
          return false;
      }
  ```
- **代码片段 2**（`ComputeSparseType`，行 1765-1793）：`return 0/3/4;` 多处裸返回，建议定义 `SPARSE_TYPE_NONE=0`、`SPARSE_TYPE_CAUSAL=3`、`SPARSE_TYPE_WINDOW=4`。
- **修复建议**：sparseMode 比较统一替换为 `SPARSE_MODE_*` 命名常量；阈值 `128` 与 sparseType 返回值定义为具名 `constexpr`。

---

## 疑似（LOW 置信度）

### [cpp-secure 1.3 / 关联 topk-8] `CheckAntiQuantGSMerge` 有符号整数理论溢出
- 行 176：`int64_t totalSize = static_cast<int64_t>(fiaInfo.gSize) * fiaInfo.s1Size;`
- 理论上 `gSize * s1Size` 若超过 `INT64_MAX` 会有符号溢出（UB）；但 `gSize`(≤256)、`s1Size`(≤2M) 受业务约束，实际乘积 ≪ `INT64_MAX`，触发概率极低。相较旧 `uint32_t` 乘法，本写法实为**防回绕改进**。建议如需严谨可对 `gSize/s1Size` 加上限约束。

### [cpp-secure 10.1] `std::string(layOut)` 潜在空指针构造
- 行 181：`std::string(fiaInfo.opParamInfo.layOut)`（`SetTransposeLayout` 中亦有同模式）。`layOut` 声明默认 `nullptr`，若上游未赋值则 `std::string(nullptr)` 为 UB。该模式在代码库已广泛预存（如未变更的 `CheckTransposeLayout`），正常执行路径中 layOut 必被赋值。建议统一整改时加判空。

### [cpp-secure 11.3] `OP_LOGI` 中 `bool` 用 `%u`
- `ComputeLoopParams` 中 `OP_LOGI(..., "gsMergeFlag:%u ...", gsMergeFlag_, ...)`，`gsMergeFlag_` 为 `bool`，规范建议用 `%d`。属**纯代码搬移**（旧代码 diff `-` 行 219 写法相同），64 位平台 `bool` 提升为 `int` 读取无错位风险。建议改 `%d`。

---

## 通过条例（85 条）

- **cpp-secure（PASS 28）**：1.1 1.2 1.3 2.1 2.2 2.3 3.1 3.2 3.3 3.4 3.6 4.1 4.2 5.1 5.2 5.3 5.4 8.1 8.2 8.3 9.1 10.2 10.3 10.4 10.5 11.1 11.2 11.4（3.5 需关注、1.3/10.1/11.3 疑似）
- **cpp-general（PASS 42）**：1.1 1.2 1.3 2.1 2.2 2.3 2.4 2.5 2.6 3.1 3.2 4.1 4.3 5.1 5.2 5.3 6.1 6.2 7.1 8.1 9.1 10.2 10.3 10.4 10.5 10.6 10.7 11.1 12.1 13.1 13.2 13.3 14.1 14.2 14.3 14.4 15.1 15.3 15.4 15.5 15.6 15.7（15.2 发现问题、4.2 需关注）
- **ascendc-topk（PASS 11）**：1 2 3 4 5 6 7 8 10 12 13（全部通过）
- **ascendc-perf Tiling（PASS 5）**：PERF-2 PERF-7 TIL-1 TIL-2 TIL-3（全部通过；多核负载均衡/尾块/缓存容量逻辑重构后保持等价）

---

## 总体结论

本 PR 为 Tiling 侧 cleancode 重构，整体质量良好：函数提取、数据驱动表替换、Early Return 等手法运用得当，重构前后业务逻辑等价（缓存容量、负载均衡、尾块处理均无变化）。

- **1 项发现问题**：`SetSparseCompressMode` 出参用 `T&`（编码约定，可结合文件一致性裁量）。
- **2 项需关注**：`SetAntiQuantInfo`/`SetMaskTilingData` 指针未判空（预存模式，建议借重构补齐判空）；`CheckSmallHeadOptimization`/`ComputeSparseType` 魔鬼数字（建议替换为命名常量）。
- **3 项疑似**：均为预存/低风险模式，可后续统一整改。

无新引入的高危安全缺陷。建议优先处理两处指针判空（红线相关）与魔鬼数字。
