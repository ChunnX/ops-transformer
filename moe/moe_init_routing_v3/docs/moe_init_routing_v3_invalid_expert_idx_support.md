# MoeInitRoutingV3 支持 expertIdx = -1（无效专家）特性说明

## 1. 背景与目标

MoeInitRoutingV3 算子需要支持传入的 `expertIdx` 中包含 -1（无效专家）。根据 README 公式：

$$availableIdxNum = |\{x \in expertIdx \mid expert\_start \le x < expert\_end\}|$$

-1 不在 `[expertStart, expertEnd)` 范围内，应被过滤，不参与 expandedXOut 的 gather 和 expertTokensCountOrCumsum 的统计。本特性仅涉及 arch35（Ascend 950PR/Ascend 950DT）实现。

## 2. arch35 排序阶段对 -1 的处理机制

arch35 的 `FilterExpertIdx`（`moe_v3_full_load_base.h`、`moe_v3_sort_one_core.h`、`moe_v3_sort_multi_core.h`）**无条件**执行以下操作（不依赖 `ep_` 标志）：

1. `Cast(int32 → float)`：-1 → -1.0
2. `CompareScalar<LT>(expertStart_)`：-1 < expertStart（当 expertStart=0 时为 true）
3. `Muls(-1)`：将满足条件的元素取负
4. `Select`：将满足条件的元素替换为 `MIN_FP32`（-3.4e38）

`Sort<float, true>`（[Ascend C Sort API](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/ascendcopapi/docs/api/SIMD-API/%E9%AB%98%E9%98%B6API/%E6%8E%92%E5%BA%8F%E6%93%8D%E4%BD%9C/Sort.md)）执行**降序排序**（大到小）。MIN_FP32 是最小值，排到队列**末尾**。

排序后 `Muls(-1)` 恢复 + `Cast(float → int32)` 时，`MIN_FP32 * -1 = 3.4e38` 超出 int32 表示范围，Cast 发生正向溢出。FullLoad 路径未显式设置 overflow mode 为饱和模式，Cast 溢出结果不可假设为固定值（可能为 INT32_MAX 或其他值）。

## 3. 风险点与修复

### 3.1 SIMT 函数缺少 `< expertStart` 守卫

4 个 SIMT 函数以 `elementNum = totalLength_`（即 `n*k`，包含无效元素）为上界遍历排序后的 expertIdx，但仅有 `>= expertEnd → break` 守卫，缺少 `< expertStart` 守卫。当 Cast 溢出结果为负值时，`expertFirstIndexLocalAddr[负值 - expertStart]` 或 `expertCountOutLocalAddr[负值 - expertStart]` 会触发越界写。

**修复**：在 `>= expertEnd` break 之前加 `< expertStart → continue`，使无效元素被显式跳过，不依赖 Cast 溢出的具体结果。

| 文件 | 函数 | 修复内容 |
|------|------|---------|
| `arch35/moe_v3_full_load_base.h` | `FullLoadComputeExpertFirstIndexSimt` | 加 `if (currExpertId < expertStart) { continue; }` |
| `arch35/moe_v3_full_load_base.h` | `FullLoadComputeExpertCountOutSimt` | 同上 |
| `arch35/moe_v3_expert_tokens_count.h` | `ComputeExpertFirstIndexSimt` | 同上 |
| `arch35/moe_v3_expert_tokens_count.h` | `ComputeExpertCountOutSimt` | 同上 |

注：`FullLoadComputeActualIdxNumSimt` 已有双向守卫 `currExpertId >= expertStart && currExpertId < expertEnd`，无需修改。

### 3.2 CopyOutRowIdx else 分支未填充 -1

`MoeV3FullLoadBase::CopyOutRowIdx` 的非 epFullload、非 SCATTER 分支（else 分支）直接拷贝 `expandedRowIdx`，但 `ComputeGatherIdx` 的第二次 sort 产生的 `expandedRowIdx` 对无效位置（`i >= actualExpertIdxNum_`）可能包含脏数据。下游 `GatherOutXScatterLoop` 虽有 `outIndex >= 0` 守卫，但无效位置若为非 -1 的正值会导致错误 gather。

**修复**：在 DataCopyPad 之前，遍历 `[actualExpertIdxNum_, totalLength_)`，将对应的 `expandedRowIdx[sortedRowIdx[i]]` 显式置 -1，确保输出符合 README 约定的"rowIdxType 为 0 时无效数据由 -1 填充"。

```cpp
for (int64_t i = actualExpertIdxNum_; i < totalLength_; i++) {
    int32_t invalidRouteIdx = sortedRowIdx.GetValue(i);
    expandedRowIdx.SetValue(invalidRouteIdx, -1);
}
```

## 4. 各路径安全性确认

### 4.1 FullLoad 路径（3 模板）

| 模板 | TilingKey | sort mask | SIMT 修复 | CopyOutRowIdx 修复 | GatherOutX 守卫 | 结论 |
|------|-----------|-----------|-----------|--------------------|-----------------|------|
| `MoeV3FullLoadUnquantized` | 200000 | ✅ 无条件 | ✅ | ✅ | ✅ `outIndex >= 0 && < outputRows` | 已修复 |
| `MoeV3FullLoadDynamicQuant` | 220000 | ✅ 无条件 | ✅ | ✅ | ✅ `outIndex >= 0 && < outputRows` | 已修复 |
| `MoeV3FullLoadStaticQuant` | 210000 | ✅ 无条件 | ✅ | ✅ | ✅ `outIndex >= 0 && < outputRows` | 已修复 |

### 4.2 非 FullLoad 路径

| 阶段 | 模板 | 遍历上界 / 守卫 | 结论 |
|------|------|----------------|------|
| Sort | `MoeSortOneCore` / `MoeSortMultiCore` | 无条件 mask | ✅ 安全 |
| ExpertTokensCount | `ComputeExpertFirstIndexSimt` / `ComputeExpertCountOutSimt` | 已加 `< expertStart` 守卫 | ✅ 已修复 |
| RowIdxGather | `RowIdxGather` | 上界为 `expertTotalCount`（有效 token 数），不触及无效元素 | ✅ 安全 |
| RowIdxGather | `MoeV3RowIdxGatherDropPad` | 遍历 n*k，有 `expertIdx < 0` 守卫 | ✅ 安全 |
| Gather（13 模板） | `MoeGatherOut` 等 | 上界为 `expertTotalCount` 或遍历 n*k 但有 `outIndex < 0` 守卫 | ✅ 安全 |

### 4.3 Gather 阶段两种安全策略

- **expertTotalCount 上界策略**（`MoeGatherOut`、`MoeGatherStaticQuant`、`MoeGatherOutDynamicQuant` 等 9 个模板）：遍历范围由 `expertTotalCount`（有效 token 数）限制，不触及无效元素。
- **n*k 上界 + -1 守卫策略**（`MoeV3GatherOutDropPad`、`MoeV3GatherOutMxfp8` Gather 模式、`MoeV3GatherFP8GroupQuant` Gather 模式等 4 个模板）：遍历完整 n*k 空间，但有 `outIndex < 0` / `outIndex >= 0` 守卫跳过无效位置。

## 5. 修改文件清单

| 文件 | 修改内容 | 行数 |
|------|---------|------|
| `op_kernel/arch35/moe_v3_full_load_base.h` | `FullLoadComputeExpertFirstIndexSimt` 加 `< expertStart` continue | +3 |
| `op_kernel/arch35/moe_v3_full_load_base.h` | `FullLoadComputeExpertCountOutSimt` 加 `< expertStart` continue | +3 |
| `op_kernel/arch35/moe_v3_full_load_base.h` | `CopyOutRowIdx` else 分支加无效位置置 -1 | +4 |
| `op_kernel/arch35/moe_v3_expert_tokens_count.h` | `ComputeExpertFirstIndexSimt` 加 `< expertStart` continue | +3 |
| `op_kernel/arch35/moe_v3_expert_tokens_count.h` | `ComputeExpertCountOutSimt` 加 `< expertStart` continue | +3 |
| `tests/ut/op_kernel/moe_init_routing_v3_data/gen_data.py` | 支持可选第 15 参数 `invalid_ratio` 生成 -1 expertIdx | +5 |
| `tests/ut/op_kernel/moe_init_routing_v3_data/gen_tiling.py` | 添加 `case_invalid_1/2/3` tiling case | +7 |
| `tests/ut/op_kernel/test_moe_init_routing_v3.cpp` | 添加 3 个含 -1 expertIdx 的 UT 用例 | +165 |

## 6. UT 用例说明

新增 3 个 UT 用例，覆盖不同场景：

| 用例名 | 模板 | activeExpertRange | expertTokensNumType | invalid_ratio | 覆盖点 |
|--------|------|-------------------|---------------------|---------------|--------|
| `test_case_invalid_expert_idx_1` | FullLoad Unquantized | `[873,889]`（子集） | KEY_VALUE | 0.3 | epFullload=1 路径 + SIMT 函数 |
| `test_case_invalid_expert_idx_2` | FullLoad Unquantized | `[0,734]`（全范围） | KEY_VALUE | 0.3 | epFullload=0 路径 + CopyOutRowIdx else 分支 |
| `test_case_invalid_expert_idx_3` | FullLoad Unquantized | `[873,889]`（子集） | COUNT | 0.3 | COUNT 模式 + ExpertTokensCount SIMT 函数 |

golden 计算（`gen_data.py`）已正确处理 -1：将 `expertIdx < expert_start` 的元素设为 `INT32_MAX`，排序后排到末尾，不参与有效输出。
