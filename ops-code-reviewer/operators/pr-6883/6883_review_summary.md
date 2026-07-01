# 代码检视报告

## 检视概览
- **PR**: [#6883](https://gitcode.com/cann/ops-transformer/pull/6883) — `kvcache` 分支合入 `master`
- **功能**: 引入 key/value/kRope 的 stride（bnStride/n2Stride/pageStride）支持，使 PA（Page Attention）场景下的 KV cache 支持非连续存储
- **代码侧别**: 混合（Kernel侧 + Tiling侧 + 测试）
- **变更文件**: 16 个（Kernel 12 / Tiling 3 / 测试 1）
- **检视文档**: cpp-secure, ascendc-api, ascendc-perf, ascendc-topk, simt-api-analysis, cpp-general
- **适用条例数**: ~120 条（100% 覆盖）
- **检视时间**: 2026-06-18

## 检视统计

| 状态 | 条例数 | 占比 |
|-----|--------|------|
| PASS | ~112 | ~93% |
| FAIL（发现问题） | 1 | ~1% |
| SUSPICIOUS（需关注） | 3 | ~3% |
| LOW（疑似） | 1 | ~1% |

## 发现问题（HIGH 置信度）

### [GEN-1.1 / SEC-3.5 / SEC-4.1] Tiling 侧 keyStrides/valueStrides 指针解引用前未判空

- **问题描述**: `fia_tiling_nonquant_gqa.cpp` 和 `fused_infer_attention_score_tiling_impl.cpp` 中，`keyStrides` 和 `valueStrides` 指针在解引用（`->GetStride()`）前未做 null 检查。而同代码块中的 `kRopeStrides` 已正确添加了 `!= nullptr` 检查。存在防御性编码不一致。

- **代码片段1** — `attention/common/op_host/arch35/fia_tiling_nonquant_gqa.cpp:967-976`:
  ```cpp
      if (!fiaInfo_->isTensorV1) {
          // ⚠️ keyStrides 未判空
          tilingData_.baseTiling.fiaBaseParams.keyStrides.bnStride = fiaInfo_->keyStrides->GetStride(0);
          tilingData_.baseTiling.fiaBaseParams.keyStrides.n2Stride = fiaInfo_->keyStrides->GetStride(1);
          // ⚠️ valueStrides 未判空
          tilingData_.baseTiling.fiaBaseParams.valueStrides.bnStride = fiaInfo_->valueStrides->GetStride(0);
          tilingData_.baseTiling.fiaBaseParams.valueStrides.n2Stride = fiaInfo_->valueStrides->GetStride(1);
          if (fiaInfo_->kRopeStrides != nullptr) {  // ✅ kRopeStrides 已判空
              tilingData_.baseTiling.fiaBaseParams.kRopeStrides.bnStride = fiaInfo_->kRopeStrides->GetStride(0);
              tilingData_.baseTiling.fiaBaseParams.kRopeStrides.n2Stride = fiaInfo_->kRopeStrides->GetStride(1);
          }
      }
  ```

- **代码片段2** — `attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_impl.cpp:1868-1877`（相同模式）:
  ```cpp
      if (!fiaInfo.isTensorV1) {
          inputParams.keyStrides.set_bnStride(fiaInfo.keyStrides->GetStride(0));   // ⚠️ 未判空
          inputParams.keyStrides.set_n2Stride(fiaInfo.keyStrides->GetStride(1));    // ⚠️ 未判空
          inputParams.valueStrides.set_bnStride(fiaInfo.valueStrides->GetStride(0)); // ⚠️ 未判空
          inputParams.valueStrides.set_n2Stride(fiaInfo.valueStrides->GetStride(1)); // ⚠️ 未判空
          if (fiaInfo.kRopeStrides != nullptr) {   // ✅ kRopeStrides 已判空
              // ...
          }
      }
  ```

- **证据链**:
  - `FiaTilingInfo` 定义 (`fia_tiling_info.h:234-237`): 三者均初始化为 `nullptr`
  - `keyStrides_` 来源: `GetDynamicInputStride()` 框架 API — 返回值可能为 `nullptr`
  - 全量 Grep 确认: `attention/` 目录下不存在任何 `keyStrides == nullptr` 或 `valueStrides == nullptr` 检查
  - 代码风格不一致: `kRopeStrides` 在同一代码块已添加 `!= nullptr` 检查

- **风险评级**: 实际风险受 `!isTensorV1` 前置条件保护（V2+ 格式下框架保证 stride 存在），但防御性编码不一致，若框架行为变更可能引入空指针解引用。

- **修复建议**: 添加显式 null 检查保持防御一致性:
  ```cpp
  if (!fiaInfo_->isTensorV1) {
      if (fiaInfo_->keyStrides != nullptr) {
          // ... GetStride()
      }
      if (fiaInfo_->valueStrides != nullptr) {
          // ... GetStride()
      }
  }
  ```

- **关联规范**: SEC-3.5（指针使用前判空）、GEN-1.1（外部数据合法性检查）、SEC-4.1/TOPK-7（外部输入校验）

---

## 需关注（MED 置信度）

### [SEC-10.4 / SEC-10.5] CVSharedParams\<true, true\> 结构体大小可能突破 CacheLine 约束

- **问题描述**: `util_regbase.h` 中 `CVSharedParams<true, true>` 新增 6 个 `uint64_t` stride 字段（共 48 字节），加上原有约 80-90 字节其他字段，总大小可能超出注释要求的 128 字节 CacheLine 约束。且新字段插在结构体中间（`paLayoutType` 和 `isActualSharedPrefixLenNull` 之间），而非最末尾。

- **代码片段** — `attention/common/op_kernel/arch35/util_regbase.h:462-543`:
  ```cpp
  /* CVSharedParams需要小于等于CacheLine的大小：128Bytes */    // ← 行462约束
  template<>
  struct CVSharedParams<true, true> {
      CV_SHARED_PARAMS;
      // ... bitfields ...
      uint32_t paLayoutType;                 // 原 PA 区域末尾 → ~80-90字节

      uint64_t keyBnStride;        // ← 新增 +48字节（6×8），插入在prefix字段之前
      uint64_t keyN2Stride;
      uint64_t valueBnStride;
      uint64_t valueN2Stride;
      uint64_t keyRopeBnStride;
      uint64_t keyRopeN2Stride;

      // prefix
      bool isActualSharedPrefixLenNull;
      int64_t kvPrefixSize;
      int64_t totalSize;
  };
  ```

- **风险分析**: 若结构体超出 128 字节可能引发跨 cache line 访问的性能退化或数据一致性问题。由于 bitfields 打包和对齐填充，精确大小需编译器确认。

- **修复建议**: 添加编译期断言验证：
  ```cpp
  static_assert(sizeof(CVSharedParams<true, true>) <= 128,
                "CVSharedParams must fit in one cache line");
  ```

---

### [TOPK-8] CopyInL1.h 中 uint32_t × uint32_t 乘法再 static_cast\<uint64_t\> 有溢出风险

- **问题描述**: `CopyInL1.h` 中 6 处新增的 stride 乘法将 `static_cast<uint64_t>` 放在 `uint32_t` 乘法之后，若两操作数乘积超过 `uint32_t` 范围则溢出后扩展为 `uint64_t` 无效。

- **代码片段** — `attention/common/op_kernel/CopyInL1.h`（6 处典型位置之一，行 255）:
  ```cpp
  // ⚠️ uint32_t × uint32_t 先溢出，static_cast<uint64_t> 后扩展
  255:  offset += static_cast<uint64_t>(startPos.n2Idx * shape.n2Stride) +
  256:      remainRowCnt * blockElementCnt + startPos.dIdx * shape.blockSize;
  ```
  其他 5 处同模式位于行 261, 285, 292, 338, 355。

- **证据**: `startPos.n2Idx` 类型为 `uint32_t`，`shape.n2Stride` 类型为 `uint32_t`，两 `uint32_t` 相乘结果仍为 `uint32_t`（可能溢出）。规范要求"表达式中有 2 个及以上维度相乘时，第一个操作数应显式转换为 int64_t"。

- **修复建议**: 将第一个操作数提前转换为 `uint64_t`:
  ```cpp
  offset += static_cast<uint64_t>(startPos.n2Idx) * shape.n2Stride + ...
  ```

---

### [TOPK-8 子发现] PAShape::pageStride/n2Stride uint32_t 与上游 uint64_t 类型不一致

- **问题描述**: 整个数据流链路（`StridesParams` → `constInfo` → `sharedParams`）全部使用 `uint64_t`，仅在最终赋值给 `PAShape` 时窄化为 `uint32_t`。

- **代码位置**: `CopyInL1.h:40-41`:
  ```cpp
  uint32_t pageStride = 0;   // ← uint32_t，上游为 uint64_t
  uint32_t n2Stride = 0;     // ← 同上
  ```
  赋值点（如 `flash_attention_noquant_block_cube.h:1022-1023`）:
  ```cpp
  shape.pageStride = constInfo.keyBnStride;   // uint64_t → uint32_t 隐式窄化
  shape.n2Stride = constInfo.keyN2Stride;     // 同上
  ```

- **风险**: 当前模型场景下 stride 值实际不会超过 uint32_t 范围，但类型不一致违反"GM 内存偏移或大小必须用 int64 表示"原则。

- **修复建议**: 统一为 `uint64_t` 或添加范围断言。

---

## 疑似（LOW 置信度）

### [GEN-6.1] CheckKVContiguous 中 PA 分支比较表达式文字列风格与非 PA 分支不一致

- **问题描述**: `common_checker.cpp` 中新增的 `CheckKVContiguous` 函数，PA 分支（BBND/BNBD/NZ）使用 `ge::GRAPH_SUCCESS != CheckTensorContiguous(...)`（常量在左），而非 PA 分支使用 `CheckTensorContiguous(...) != ge::GRAPH_SUCCESS`（函数结果在左）。

- **注**: 经完整源码交叉验证，`common_checker.cpp` 中已有大量同模式代码（如行 520, 527, 534, 541, 1011-1014 均使用 `ge::GRAPH_SUCCESS != CheckDtypeCommon(...)`），此为新代码与既有代码风格一致性的轻度不一致，非功能性缺陷。已在行号校对阶段确认两种写法共存于代码库。

---

## 通过条例（部分代表性）

API-1~4, API-6~10, API-12, SIMT-1~16, SEC-1.1, SEC-1.2, SEC-1.3, SEC-2.1~2.3, SEC-3.1, SEC-3.2, SEC-3.3, SEC-3.4, SEC-3.6, SEC-4.2, SEC-5.1~5.4, SEC-8.1~8.3, SEC-9.1, SEC-10.1~10.3, SEC-11.1~11.4, TOPK-1~6, TOPK-9~13, PERF-1~7, PREC-1, PREC-3, TIL-1~3, GEN-1.2~1.3, GEN-2.1~2.6, GEN-3.1~3.2, GEN-4.1~4.3, GEN-5.1~5.3, GEN-6.2, GEN-7.1, GEN-8.1, GEN-9.1, GEN-10.2~10.7, GEN-11.1, GEN-12.1, GEN-13.1~13.3, GEN-14.1~14.4, GEN-15.1~15.7

（以上约 100+ 条条例全部通过，变更代码以 stride 数据流传递为主，无违规）

---

## 被检视代码

> 本次检视的完整 diff（889 行，16 个文件）

### 变更文件列表

| # | 文件 | 侧别 |
|---|------|------|
| 1 | attention/common/op_host/arch35/fia_tiling_nonquant_gqa.cpp | Tiling |
| 2 | attention/fused_infer_attention_score/op_host/arch35/fused_infer_attention_score_tiling_impl.cpp | Tiling |
| 3 | attention/fused_infer_attention_score/op_host/checkers/common_checker.cpp | Tiling |
| 4 | attention/fused_infer_attention_score/op_host/checkers/common_checker.h | Tiling |
| 5 | attention/common/op_kernel/CopyInL1.h | Kernel |
| 6 | attention/common/op_kernel/arch35/fia_block_cube_noquant_gqa.h | Kernel |
| 7 | attention/common/op_kernel/arch35/fia_kernel_noquant_gqa.h | Kernel |
| 8 | attention/common/op_kernel/arch35/flash_attention_block_cube_noquant_mla.h | Kernel |
| 9 | attention/common/op_kernel/arch35/flash_attention_kernel_noquant_mla.h | Kernel |
| 10 | attention/common/op_kernel/arch35/flash_attention_noquant_block_cube.h | Kernel |
| 11 | attention/common/op_kernel/arch35/flash_attention_noquant_block_vec_infer.h | Kernel |
| 12 | attention/common/op_kernel/arch35/flash_attention_noquant_kernel_infer.h | Kernel |
| 13 | attention/common/op_kernel/arch35/flash_attention_score_tiling_regbase.h | Kernel |
| 14 | attention/common/op_kernel/arch35/util_regbase.h | Kernel |
| 15 | attention/common/op_kernel/fia_tiling_data_public.h | Kernel |
| 16 | attention/fused_infer_attention_score/tests/pytest/test_fia_pa_noncontiguous.py | 测试 |

### 检视概要

PR 功能为 PA 场景下 KV cache 非连续存储 stride 支持，变更本质为：
- Tiling 侧: 从 `gert::Stride` 读取 stride 值填充到 TilingData，新增 `CheckKVContiguous` 校验函数
- Kernel 侧: 数据结构扩展传递 stride（StridesParams → constInfo → PAShape），offset 计算使用 stride 条件分支
- 测试: 新增 PA 非连续 stride 传递测试

**检视结论**: 整体代码质量良好。主要关注点：Tiling 侧 keyStrides/valueStrides 指针使用前缺少判空保护（与 kRopeStrides 判空逻辑不一致），Kernel 侧 CopyInL1.h 有 6 处 uint32 乘法溢出风险和 CVSharedParams CacheLine 容量验证缺失。
