# A5 CountingSort 两模板全覆盖用例验证报告

> **测试日期**：2026-08-07
> **固定参数**：N=9000, actualExpertNum=256（expertNum=256, active_expert_range=[0,256]）
> **变化维度**：模板（FullLoad cols=128 / CutOrigin cols=7168）× dtype（fp16/bf16/fp32）× quant_mode（-1/0/1）× row_idx_type（0/1）× expert_tokens_num_type（0/1/2）× drop_pad_mode（0/1）
> **CSV 文件**：`transpose/moe_v3_test_cs_coverage.csv`（16 条新增用例）
> **生成脚本**：`transpose/gen_cs_coverage.py`

---

## 1. FullLoad 用例（cols=128，UB 预算足够 X 全载）

| ti | 用例名 | dtype | quant | row_idx | etnt | dropPad | TilingKey | DYN_PERF (µs) | 状态 |
|----|--------|-------|-------|---------|------|---------|-----------|---------------|------|
| 0 | FL_unquant_fp16_gather_count_drop | fp16 | -1 | 0(GATHER) | 1(COUNT) | 0 | **400000** | 65.899 | ✅ PASS |
| 1 | FL_unquant_bf16_gather_count_drop | bf16 | -1 | 0(GATHER) | 1(COUNT) | 0 | **400000** | 66.232 | ✅ PASS |
| 2 | FL_unquant_fp32_gather_count_drop | fp32 | -1 | 0(GATHER) | 1(COUNT) | 0 | **400000** | 66.116 | ✅ PASS |
| 3 | FL_unquant_fp16_scatter_count_drop | fp16 | -1 | 1(SCATTER) | 1(COUNT) | 0 | **401000** | 71.474 | ✅ PASS |
| 4 | FL_unquant_fp16_gather_cumsum_drop | fp16 | -1 | 0(GATHER) | 0(CUMSUM) | 0 | **400000** | 66.024 | ✅ PASS |
| 5 | FL_unquant_fp16_gather_kv_drop | fp16 | -1 | 0(GATHER) | 2(KEY_VALUE) | 0 | **400000** | 65.676 | ✅ PASS |
| 6 | FL_static_fp16_gather_count_drop | fp16 | 0 | 0(GATHER) | 1(COUNT) | 0 | **410000** | 70.996 | ✅ PASS |
| 7 | FL_static_fp16_scatter_count_drop | fp16 | 0 | 1(SCATTER) | 1(COUNT) | 0 | **411000** | 75.428 | ✅ PASS |
| 8 | FL_static_fp16_gather_cumsum_drop | fp16 | 0 | 0(GATHER) | 0(CUMSUM) | 0 | **410000** | 69.944 | ✅ PASS |
| 9 | FL_unquant_fp16_gather_count_pad | fp16 | -1 | 0(GATHER) | 1(COUNT) | 1 | **400100** | 62.488 | ✅ PASS |
| 10 | FL_dynamic_fp16_gather_count_drop | fp16 | 1 | 0(GATHER) | 1(COUNT) | 0 | **11020000**（CS 关闭回落） | — | ✅ PASS（回落旧路径） |

**关键验证点**：
- **ti=9（FullLoad + dropPad）首次验证 A5 CS FullLoad 支持 dropPad**：TilingKey=400100（FullLoad unquant GATHER **+dropPad**），DYN_PERF=62.488µs，精度 PASS。证明 A5 CS FullLoad 的 kernel（`moe_v3_counting_sort_full_load_base.h` L457-461 dropPad 分支）实际可用，与文档结论一致。
- **ti=10（dynamic）验证 CS 全局关闭回落**：TilingKey=11020000（旧分阶段路径），不走 CS 模板。

---

## 2. CutOrigin 用例（cols=7168，X 全载超 UB）

| ti | 用例名 | dtype | quant | row_idx | etnt | dropPad | TilingKey | DYN_PERF (µs) | 状态 |
|----|--------|-------|-------|---------|------|---------|-----------|---------------|------|
| 11 | CO_unquant_bf16_gather_count_drop | bf16 | -1 | 0(GATHER) | 1(COUNT) | 0 | **500000** | 112.711 | ✅ PASS |
| 12 | CO_unquant_fp32_gather_count_drop | fp32 | -1 | 0(GATHER) | 1(COUNT) | 0 | **500000** | 176.328 | ✅ PASS |
| 13 | CO_unquant_fp16_scatter_count_drop | fp16 | -1 | 1(SCATTER) | 1(COUNT) | 0 | **501000** | 135.21 | ✅ PASS |
| 14 | CO_static_fp16_scatter_count_drop | fp16 | 0 | 1(SCATTER) | 1(COUNT) | 0 | **511000** | 120.788 | ✅ PASS |
| 15 | CO_static_fp16_gather_cumsum_drop | fp16 | 0 | 0(GATHER) | 0(CUMSUM) | 0 | **510000** | 100.408 | ✅ PASS |

> 已有用例（`moe_v3_test.csv`）已覆盖：CO unquant fp16 gather count dropless (ti=1)、CO static gather count (ti=19)、CO cumsum (ti=40)、CO key_value (ti=41)、CO dropPad (ti=42)、CO dynamic (ti=20)。

---

## 3. TilingKey 编码验证

A5 TilingKey 编码公式：`base + rowIdxType × 1000 + dropPadMode × 100`

| 模板 + 量化 | base | +SCATTER(+1000) | +dropPad(+100) | 实测验证 |
|------------|------|-----------------|----------------|---------|
| FullLoad unquant | 400000 | 401000 ✅（ti=3） | 400100 ✅（ti=9） | 全部一致 |
| FullLoad static | 410000 | 411000 ✅（ti=7） | —（attrs 禁止 static+dropPad） | 一致 |
| CutOrigin unquant | 500000 | 501000 ✅（ti=13） | 500100 ✅（已有 ti=42） | 全部一致 |
| CutOrigin static | 510000 | 511000 ✅（ti=14） | —（attrs 禁止 static+dropPad） | 一致 |
| 动态量化（CS 关闭） | — | — | — | 11020000 ✅（ti=10） |

---

## 4. 覆盖维度汇总

| 维度 | 取值 | FullLoad 覆盖 | CutOrigin 覆盖 |
|------|------|--------------|---------------|
| **dtype** | fp16 / bf16 / fp32 | ti=0/1/2 ✅ | ti=11/12 + 已有 ti=1 ✅ |
| **quant_mode** | -1（unquant） | ti=0/1/2/3/4/5/9 ✅ | ti=11/12/13 + 已有 ✅ |
| | 0（static） | ti=6/7/8 ✅ | ti=14/15 + 已有 ti=19 ✅ |
| | 1（dynamic） | ti=10 ✅（回落） | 已有 ti=20 ✅（回落） |
| **row_idx_type** | 0（GATHER） | 全部 GATHER 用例 ✅ | 全部 GATHER 用例 ✅ |
| | 1（SCATTER） | ti=3/7 ✅ | ti=13/14 ✅ |
| **expert_tokens_num_type** | 0（CUMSUM） | ti=4/8 ✅ | ti=15 + 已有 ti=40 ✅ |
| | 1（COUNT） | ti=0/1/2/3/6/7/9 ✅ | ti=11/12/13/14 + 已有 ✅ |
| | 2（KEY_VALUE） | ti=5 ✅ | 已有 ti=41 ✅ |
| **drop_pad_mode** | 0（dropless） | 大部分用例 ✅ | 大部分用例 ✅ |
| | 1（dropPad） | **ti=9 ✅（首次验证 FullLoad+dropPad）** | 已有 ti=42 ✅ |

---

## 5. 关键结论

1. **A5 CS FullLoad 全面支持 dropPad / CUMSUM / KEY_VALUE**（ti=4/5/9 验证），与文档 §4 类别 C 修正一致，A5 比 A3 CS FullLoad（仅 unquant+dropless+COUNT）覆盖更广。
2. **A5 CS 两模板的 TilingKey 编码完全符合预期**，16 条用例全部进入正确分支。
3. **精度全部 PASS**（unquant 100% gold，static quant 99.97%+ gold，精度容差内）。
4. **动态量化正确回落旧路径**（TilingKey=11020000），不进 CS 模板。
5. **attrs 层硬约束生效**：dropPad 场景自动限定 quant=-1 + etnt=1 + row_idx=0，未生成违规组合。

---

**文档版本**：v1.0
**生成日期**：2026-08-07
**测试设备**：device 4 / 5（Ascend950DT）
**关联文档**：`cs_a3_a5_support_comparison.md`、`cs_template_support_matrix.md`
