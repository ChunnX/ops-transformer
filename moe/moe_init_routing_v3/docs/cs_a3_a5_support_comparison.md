# A3 / A5 CountingSort 两模板支持情况对比

> **文档目标**：对比 A3（`moeInitrouteV3`）与 A5（`ops-transformer0716`，arch35）两个 CountingSort 性能模板（FullLoad / CutOrigin）在数据类型、参数组合上的支持情况，梳理依赖关系，并分类记录「不支持」的具体原因（tiling 拦截 / attrs 校验 / kernel 未实现 / 性能差 / 模板设计精简）。
>
> **关联文档**：`cs_template_support_matrix.md`（详细支持矩阵 + 约束根因）、`cs_extend_dropPad_etnt_plan.md`（扩展方案）、`perf_baseline_vs_new_template.md` §7.2（动态量化关闭性能依据）

---

## 1. 数据类型支持（X / expandedX，A3 = A5 完全一致）

| 量化模式 | X（输入） | expandedX（输出） |
|---------|----------|------------------|
| UN_QUANT(-1) | bf16 / fp16 / fp32 | 与 X 相同 |
| STATIC_QUANT(0) | bf16 / fp16 / fp32 | int8 |
| DYNAMIC_QUANT(1) | bf16 / fp16 / fp32 | int8 |

> 静态/动态量化的 expandedX 为 int8（per-token 量化结果）；动态量化虽 kernel 支持但 tiling 全局关闭（见 §4）。

---

## 2. 参数支持矩阵（4 列：A3/A5 × FullLoad/CutOrigin）

| 维度 | A3 FullLoad | A3 CutOrigin | A5 FullLoad | A5 CutOrigin |
|------|-------------|--------------|-------------|--------------|
| **N 范围** | [CS_MIN_N, CS_MAX_N] | 同左 | **[256, 1M]** | **[256, 1M]** |
| **actualExpertNum 上限** | ≤1024（间接，无显式） | ≤1024 | **≤256**（显式 `CS_MAX_ACTUAL_EXPERT_NUM`） | **≤256** |
| LE < E（窄 range） | ✓ | ✓ | ✓（≤256） | ✓（≤256） |
| LE = E = 256（全通过） | ✓ | ✓ | ✓ | ✓ |
| **row_idx_type=0** (GATHER) | ✓ | ✓ | ✓ | ✓ |
| **row_idx_type=1** (SCATTER) | ✓（dropless） / ❌（dropPad，attrs 拦截） | ✓（dropless） / ❌（dropPad，attrs 拦截） | ✓（dropless） / ❌（dropPad，attrs 拦截） | ✓（dropless） / ❌（dropPad，attrs 拦截） |
| **quant_mode=-1** (unquant) | ✓ | ✓ | ✓ | ✓ |
| **quant_mode=0** (static) | ❌ tiling（L848 限定 UN_QUANT） | ✓ | ✓ | ✓ |
| **quant_mode=1** (dynamic) | ❌ tiling | ❌ tiling | ❌ tiling | ❌ tiling |
| **drop_pad_mode=0** (dropless) | ✓ | ✓ | ✓ | ✓ |
| **drop_pad_mode=1** (dropPad) | ❌ FullLoad 不支持 | **✓** | **✓**（kernel 支持，UB 足够时走 FullLoad） | **✓**（本次扩展验证） |
| **expert_tokens_num_type=0** (CUMSUM) | ❌ FullLoad 精简 | ✓ | **✓**（kernel 支持，UB 足够时走 FullLoad） | **✓**（本次扩展验证） |
| **expert_tokens_num_type=1** (COUNT) | ✓ | ✓ | ✓ | ✓ |
| **expert_tokens_num_type=2** (KEY_VALUE) | ❌ FullLoad 精简 | ✓ | **✓**（kernel 支持，UB 足够时走 FullLoad） | **✓**（本次扩展验证） |

> A5 CutOrigin 列加粗的 3 项（dropPad / CUMSUM / KEY_VALUE）由本次扩展放开，验证用例 ti=40/41/42 全 PASS（见 §6）。
>
> **A5 CS FullLoad 实际也支持 dropPad / CUMSUM / KEY_VALUE**（kernel `moe_v3_counting_sort_full_load_base.h` L436-469 完整三分支 + dropPad/dropless 双分支），只是现有测试用例（ti=0/17 等）都是小 cols 场景命中 FullLoad，未覆盖这些组合。FullLoad 与 CutOrigin 的分派仅由 UB 预算决定。

---

## 3. 依赖关系（谁能和谁组合）

### 3.1 A5 硬依赖（attrs/tiling 校验层强制）

```
dropPadMode=1  ─┬─→ 必须 quant_mode=-1       (ValidateQuantMode L895-900)
                ├─→ 必须 expert_tokens_num_type=1 (COUNT)  (L847)
                ├─→ 必须 row_idx_type=0 (GATHER)  (ValidateRowIdxType L920-924)
                └─→ 必须 expert_range=[0, expertNum]（不能 LE 窄 range）(L975-981)

quant_mode=1   ──→ 全局禁用 CS（IsAnyDynamicQuantCase L2274）
                    （含 int8/int4 动态量化）

CS FullLoad    ─┬─→ quant_mode ∈ {-1, 0}     (IsCountingSortApplicable 共用约束)
（countingSortMode_=1） ├─→ UB 预算足够（X 全载得下）
                        └─→ 无 FullLoad 特有的 dropPadMode / expertTokensNumType 约束
                            （kernel 完整支持三分支 + dropPad，见 §4 类别 C 修正）
```

> **重要修正**（2026-08-07）：早期版本曾误写「CS FullLoad 必须 dropPadMode=0 + 必须 expert_tokens_num_type=1」——这是 **A3 旧 sort-based FullLoad** 的限制（A3 `ComputeCountingSortTiling` L847-852 显式限定 dropless），**不适用 A5 CS FullLoad**。A5 CS FullLoad 的 kernel（`moe_v3_counting_sort_full_load_base.h` L436-469）完整实现了三分支 expertTokensNumType + dropPad/dropless 双分支，tiling 侧也无 FullLoad 特有的 dropPad/expertTokensNumType 拦截。两者唯一差异是 UB 预算（FullLoad 要求 X 全载得下，CutOrigin 流式按需载入）。

### 3.2 依赖反推（A5 实际可走 CS 的组合）

| quant_mode | 可配 drop_pad_mode | 可配 row_idx_type | 可配 expert_tokens_num_type | 可走模板 |
|-----------|-------------------|------------------|----------------------------|---------|
| -1（unquant） | 0 或 1 | 0/1（dropPad 时仅 0） | 0/1/2（dropPad 时仅 1） | FullLoad（UB 足够时）或 CutOrigin |
| 0（static） | **仅 0** | 0/1 | 0/1/2 | FullLoad（UB 足够时）或 CutOrigin |
| 1（dynamic） | —（CS 关闭） | — | — | 回退旧分阶段路径 TilingKey=11020000 |

> FullLoad vs CutOrigin 的分派**仅由 UB 预算决定**（X 全载得下走 FullLoad，否则 CutOrigin），与 dropPadMode / expertTokensNumType / rowIdxType 无关。
>
> **dropPad 的硬约束叠加**：dropPadMode=1 时强制要求 quant_mode=-1 + expert_tokens_num_type=1 + row_idx_type=0 + expert_range=[0, expertNum]（4 项 attrs 校验联动，见 §3.1）。

### 3.3 A3 硬依赖（参考）

A3 有两类独立的 FullLoad，约束不同：

**A3 旧 sort-based FullLoad**（`moe_v3_full_load_*.h`，`isFullload_=true`）：

```
dropPadMode=1                  ──→ 禁用（L777 dropPadMode_ == 1 return false）
totalLength > sortLoopMaxElement ──→ 禁用
quant_mode ∈ {-1, 0, 1}        ──→ 均可（按模式有不同 UB 预算公式）
走旧 sort-based 路径，不走 CS 三分支
```

**A3 CS FullLoad**（`moe_v3_full_load_cut_origin_t.h`，`countingSortFullLoad_=1` L847-852）：

```
dropPadMode=1                  ──→ 禁用（L849 dropPadMode_ == DROP_LESS）
expert_tokens_num_type ∈ {0,2} ──→ 禁用（L849 expertTokensNumType_ == COUNT）
quant_mode=0 (static)          ──→ 禁用（L848 quantMode_ == UN_QUANT，仅非量化）
quant_mode=1 (dynamic)         ──→ 禁用（L815 quantValid）
ep_ != 1                       ──→ 禁用（L848 ep_ == 1，必须 LE 模式）
k_ != PERFORMANCE_MODE_TOP_K   ──→ 禁用（L848）
dropless + static + n ≤ 2560   ──→ 禁用（excludeNonDropPadSmallN）
dropPad + n < 512              ──→ 禁用（excludeDropPadTinyN）
```

> **A3 CS FullLoad 极其严苛**：仅支持 `unquant + dropless + COUNT + LE + topK=固定值` 这一窄组合，连静态量化都不支持。Phase B kernel（L451-465）也是精简实现，无三分支 expertTokensNumType、无 dropPad 分支。

---

## 4. 不支持的理由分类（核心）

### 类别 A：tiling 本身拦截（结构性，不会到 kernel）

| 不支持项 | 拦截位置 | 理由 |
|---------|---------|------|
| **A5 actualExpertNum > 256** | `IsCountingSortApplicable` `CS_MAX_ACTUAL_EXPERT_NUM=256` | UB 预算（expertTokensLocal 按 256×2 int64 上界恒计，tiling L2414）、Phase A 过滤 UB 容量均按 256 上界设计。A3 无此约束（≤1024 间接）。 |
| **A5 N < 256 或 N > 1M** | `CS_MIN_N=256, CS_MAX_N=1M` | Phase A chunk 向量过滤 UB 容量与 N 边界耦合；过小 N 收益不足，过大 N UB 装不下。 |
| **A5/A3 quant_mode=1（动态量化）** | A5 `IsAnyDynamicQuantCase` L2274 / A3 `quantValid` L815 | **性能差（主动关闭）**：CutOrigin 动态量化 ti=20 实测 **700µs vs 原路径 178µs (+326%)**。根因是 per-token absmax 两趟 VF（smooth 预乘写回 + absmax、Div+Cast f32→f16→int8）形成**纯计算串行链**（Vector 流水瓶颈），双缓冲 ping-pong 只能隐藏 DMA、追不平 Vector 瓶颈。关闭后 ti=20/37 等 8 个动态量化用例回落 TilingKey=11020000（原路径），耗时 ~175-178µs（-0.3%）。A3 同样理由关闭。 |
| **A3 dropPad + n < 512** | `excludeDropPadTinyN` | 小 N 下 dropPad 的 expertCapacity 槽位填充率低，CS 相比旧路径收益不足。 |
| **A3 dropless + static + n ≤ 2560** | `excludeNonDropPadSmallN` | 小 N 静态量化下 CS 相比旧路径无收益。 |

### 类别 B：attrs 校验拦截（语义层，tiling 前置）

| 不支持项 | 拦截位置 | 理由 |
|---------|---------|------|
| **A5 dropPad + quant_mode=0/1（静态/动态量化）** | `ValidateQuantMode` L895-900 | **kernel 未实现**：A5 静态量化 CutOrigin 的 `GatherAndWrite` / `GatherAndWriteDoubleBuffer` 明确注释「静态量化非 dropPad」，无 slot 溢出判定、无 int8 `ZeroOutExpandedDropPad`。attrs 提前拦截避免走到未实现分支。**A3 支持此组合**（A3 `GatherAndWriteDroppad` L1195-1532 完整实现 unquant + static 双路径）。 |
| **A5 dropPad + expert_tokens_num_type=0/2** | L847 | **语义冲突**：dropPad 的 expertCapacity 固定槽位输出（outputRows = expertNum × expertCapacity）与 cumsum 前缀和（dropless 累加语义）/ KEY_VALUE 紧凑对（变长输出）不兼容。Phase B `ComputeGlobalOffset` 的 dropPad seed 计算依赖 COUNT 语义。 |
| **A5/A3 dropPad + row_idx_type=1 (SCATTER)** | A5 `ValidateRowIdxType` L920-924 / A3 L369-372 | **语义不可行**：SCATTER 语义是 `expandedRowIdx[newPos] = origFlatIdx`，被 dropPad 丢弃的行没有 newPos（slot 溢出），无法表达「被丢弃」信息。GATHER 模式下丢弃行可写 `expandedRowIdx[origFlatIdx] = -1`（原位置标记），SCATTER 无此能力。 |
| **A5/A3 dropPad + expert_range ≠ [0, expertNum]（LE 窄 range）** | A5 L975-981 / A3 L373-378 | **语义约束**：dropPad 的 outputRows = expertNum × expertCapacity 要求全专家映射，LE 窄 range 下未映射专家的 expertCapacity 槽位无 token 填充，与 dropPad 语义不兼容。 |

### 类别 C：A3 旧 sort-based FullLoad 模板设计精简（仅适用 A3，不适用 A5）

> **重要修正**：本类别仅描述 **A3 旧 sort-based FullLoad**（`moe_v3_full_load_cut_origin_t.h`）的限制。**A5 CS FullLoad**（`moe_v3_counting_sort_full_load_base.h`）已完整实现三分支 expertTokensNumType + dropPad/dropless 双分支，无此限制。

| 不支持项 | 理由 | 适用范围 |
|---------|------|---------|
| **FullLoad dropPad** | A3 旧 sort-based FullLoad 假设 X 全载入 UB 后整体重排，与 dropPad 的「逐 pair 判定 slot 溢出 + 丢弃行写 -1 + 预填零」语义不匹配。A3 `ComputeCountingSortTiling` L847-852 显式限定 FullLoad 仅 dropless。 | **仅 A3**。A5 CS FullLoad kernel L457-461 支持 dropPad/dropless 双分支，tiling 无 FullLoad 特有 dropPad 约束。 |
| **FullLoad CUMSUM / KEY_VALUE** | A3 旧 sort-based FullLoad Phase B 精简为直接写出 `actualExpertNum` 个 int64（依赖上层已按模式填好），无 cumsum 标量累加循环、无 KEY_VALUE 紧凑 (expert_idx, count) 对构造逻辑。 | **仅 A3**。A5 CS FullLoad kernel L436-452 完整实现 COUNT/CUMSUM/KEY_VALUE 三分支，L466-468 支持 KEY_VALUE 动态 `expertCountElements_` 变长输出。 |

**A5 CS FullLoad 不存在类别 C 的限制**：FullLoad 与 CutOrigin 的唯一差异是 Phase C X 数据载入策略（FullLoad 全载入 UB，CutOrigin 流式按需载入），Phase A/B 的三分支 expertTokensNumType + dropPad/dropless 双分支逻辑两者共用（继承自 `MoeV3CountingSortFullLoadBase<T>` 和 `MoeV3CountingSortCutOriginBase<T>` 的同名 `ComputeGlobalOffset` / `WriteExpertTokens`）。

---

## 5. A3 vs A5 关键差异总结

| 差异点 | A3 | A5 | 差异原因 |
|-------|----|----|---------|
| **actualExpertNum 上限** | ≤1024（间接，无显式） | **≤256**（显式 `CS_MAX_ACTUAL_EXPERT_NUM`） | A5 UB 容量按 256 上界预算设计；A5 聚焦 LE（Local Expert）性能优化场景 |
| **dropPad + 静态量化** | **支持**（kernel 完整实现） | **attrs 禁用**（kernel 未实现） | A5 迁移时聚焦 dropless 优化，未同步 dropPad + static 的 kernel 分支（int8 `ZeroOutExpandedDropPad` + slot 溢出） |
| **动态量化关闭依据** | A3 自身历史决策（`quantValid`） | **实测 +326% 劣化**（ti=20 700µs vs 178µs）后主动关闭 | A5 有量化数据支撑，记录于 `perf_baseline_vs_new_template.md` §7.2 |
| **Phase C 双缓冲** | A3 `GatherAndWriteChunked` L762 完整 ping-pong | A5 已重建（Phase C 双缓冲默认实现，宏 `MOE_CS_CUTORIGIN_DOUBLE_BUFFER` 已移除） | A5 迁移时丢失双缓冲，本次工作已修复（详见 plan `precious-mixing-micali.md`） |
| **Phase A 向量过滤** | A3 `FilterAndCountChunked` ep_==1 完整 | A5 仍为纯标量（待阶段二） | A5 迁移时未同步 |
| **CS FullLoad 的 Phase B 能力** | A3 CS FullLoad Phase B 极精简：仅 unquant + dropless + COUNT + LE + topK=固定值（L847-852），kernel（L451-478）无三分支 expertTokensNumType、无 dropPad 分支 | **A5 CS FullLoad Phase B 完整**（COUNT/CUMSUM/KEY_VALUE 三分支 + dropPad/dropless 双分支），支持 quant_mode ∈ {-1, 0} | A5 在迁移时把 CS FullLoad 的 Phase B 写得比 A3 完整得多，覆盖比 A3 显著更广；FullLoad 与 CutOrigin 共用 `ComputeGlobalOffset` 逻辑 |

---

## 6. 本次扩展验证结果（A5 CutOrigin 新增覆盖）

| 用例 ID | ti | 扩展点 | TilingKey | DYN_PERF (µs) | 状态 |
|---------|----|--------|-----------|---------------|------|
| `A5_cutorigin_cumsum` | 40 | expert_tokens_num_type=0 (CUMSUM) | 500000 | 114.323 | **PASS** |
| `A5_cutorigin_key_value` | 41 | expert_tokens_num_type=2 (KEY_VALUE) | 500000 | 114.782 | **PASS** |
| `A5_cutorigin_droppad` | 42 | drop_pad_mode=1 (dropPad) | **500100**（首次出现 dropPad CutOrigin） | 178.046 | **PASS** |

**结论**：A5 CountingSort 当前覆盖能力 = A3 减去 (dropPad + 静态量化) 组合。差距源自 A5 静态量化 kernel 未实现 dropPad 分支（attrs 校验前置拦截）。其余维度（数据类型、row_idx_type、expert_tokens_num_type 三模式、dropPad 非量化）A5 已与 A3 对齐。

---

## 7. 附录：关键代码位置速查

### A5 tiling（`op_host/moe_init_routing_v3_tiling_arch35.cpp`）

| 函数 | 行号 | 作用 |
|------|------|------|
| `IsCountingSortApplicable` | L2264-2304 | N / actualExpertNum / dropPadMode / 动态量化拦截 |
| `ValidateQuantMode` | L895-900 | dropPad 要求 quant_mode=-1 |
| L847 | — | dropPad 要求 expert_tokens_num_type=1 (COUNT) |
| `IsAnyDynamicQuantCase` | L2274 | 动态量化全局关闭 |
| `ComputeArch35CountingSortCutOriginTiling` | L2388-2454 | UB 预算（expertTokens 上界恒计 2 倍） |

### A5 kernel（`op_kernel/arch35/`）

| 文件 | 函数 | 行号 | 说明 |
|------|------|------|------|
| `moe_v3_counting_sort_cut_origin_base.h` | `ComputeGlobalOffset` | L607-680 | Phase B，支持三模式 expertTokens + dropPad seed |
| `moe_v3_counting_sort_cut_origin_base.h` | `WriteExpertTokens` | L683-694 | 动态 `expertCountElements_` 适配 KEY_VALUE 变长 |
| `moe_v3_counting_sort_cut_origin_unquantized.h` | `Init` | L59-73 | dropPad zeroOut + scale 清零（已实现） |
| `moe_v3_counting_sort_cut_origin_unquantized.h` | `GatherAndWriteDoubleBuffer` | L209-216 | dropPad skipArr + 批量 -1 写出（已实现） |
| `moe_v3_counting_sort_cut_origin_static_quant.h` | `GatherAndWrite` | L181-182 | ❌ 注释「静态量化非 dropPad」，未实现分支 |

### A3 参考实现（`moeInitrouteV3`，只读）

| 文件 | 函数 | 行号 | 说明 |
|------|------|------|------|
| `op_host/moe_init_routing_v3_tiling.cpp` | `ShouldEnableCountingSortByShape` | L809-819 | dropPad + dropless 双分支，动态量化关闭 |
| `op_host/moe_init_routing_v3_tiling.cpp` | `ComputeCountingSortTiling` | L847-852 | FullLoad 限定 dropless |
| `op_kernel/moe_v3_cut_origin_t.h` | `GatherAndWriteDroppad` | L1195-1532 | dropPad 完整实现（unquant + static 双路径） |
| `op_kernel/moe_v3_cut_origin_t.h` | `WriteExpertTokens` | L667-713 | 三模式 expertTokens 输出 |

---

**文档版本**：v1.0
**生成日期**：2026-08-07
**维护者**：ops-transformer0716 项目组
**关联文档**：
- `cs_template_support_matrix.md`（详细支持矩阵 + 约束根因）
- `cs_extend_dropPad_etnt_plan.md`（扩展实现方案）
- `perf_baseline_vs_new_template.md` §7.2（动态量化关闭性能依据）
