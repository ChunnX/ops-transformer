# issue: 全 mask(k=0) 场景 group_list[0][0] 写入脏 expert_id（应为 0）

- **发现阶段**: 迭代二 穿刺3（probe_iter2_boundary_kzero_samebin，NORM/100/fp16/单核，Ascend950DT_9584/arch35）
- **严重度**: 中（边界正确性缺陷，稳定复现；迭代三 k=0 用例必现）
- **状态**: ✅ 已修复（主线 arch35 头单行守卫已落，单核 k=0 NPU 8/8 二进制一致）；多核 k=0（行动项3）经 dispatch 逻辑证明**不可达**（见下方「行动项 3 结论」），非真空点
- **影响文件（主线）**: `op_kernel/arch35/ffn_wb_group_listing.h`（arch22 `op_kernel/ffn_wb_group_listing.h` 同源，同缺陷，需一并核对）

## 现象

全部 token 被 mask（k=0，`validGatherIdxLength=0`，`actual_token_num=0` 正确）时：

| 输出 | NPU 实测 | golden(spec 契约) | 结论 |
|------|---------|-------------------|------|
| actual_token_num | 0 | 0 | ✅ |
| y / session_ids / micro_batch_ids / token_ids / expert_offsets / dynamic_scale | 全 0 | 全 0 | ✅ |
| **group_list[0]** | **[8, 0]** | **[0, 0]** | ❌ 差 1 个 int64（`group_list[0][0]=8`，非 0） |

`8 == expertNum`。多次运行确定性复现（脏值稳定 = expertNum），非随机。spec.boundary_conditions `k_zero` 契约要求 `group_list 全 [0,0]`。

> 说明：`8 = expertNum` 是该 shape 下的脏值表现；不同 expertNum 下脏值可能不同（本质是未初始化 UB 读值，不保证恒等于 expertNum）。

## 根因

`KernelFfnWBGroupListing`（NORM 单核 group_listing，`groupListingDealFlag==0` 分支，末核执行）：

1. `Process()` 调 `CopyInOneCore(validGatherIdxLength=0)` → `DataCopyPad` 搬 0 字节，`sortedExpertIdxInQueue` 里的 UB 是未初始化脏数据。
2. `ComputeOneCore(0)`：`if (lastExpertId == -1) lastExpertId = inLocal.GetValue(0);` —— **读 UB[0] 脏值**（0 字节搬运后 UB 未被写入）；`for (i<0)` 循环体不执行，`lastExpertId` 保持脏值。
3. 回到 `Process()`：`if (curExpertIdOffset < expertNum)` 恒成立 → `group_list[0] = [lastExpertId(脏), tokenCount(=0)]` 被 flush。

即：**缺少 `validGatherIdxLength==0` 守卫**，无有效 token 时仍 flush 了一个基于未初始化 UB 的伪专家条目。

对比：
- `KernelFfnWBGatherOutAll` 有 `needCoreNum_==0`（perCoreRows==0）守卫，k=0 时 `Process()` 直接 return，故 y/ids 正确。
- `KernelSortMaskOneCore` 有 `rsvdCnt==0` 守卫，正确写 `validGatherIdxLength=0` / `actual_token_num=0`。
- 唯独 group_listing 单核路径缺此守卫。

## 修复方案（穿刺内已验证）

`ffn_wb_group_listing.h` `Process()` 的 flush 处加 `validGatherIdxLength > 0` 守卫：

```cpp
// 原：
if (curExpertIdOffset < expertNum) {
    groupListOutLocal.SetValue(curExpertIdOffset * NUM_TWO, lastExpertId);
    groupListOutLocal.SetValue(curExpertIdOffset * NUM_TWO + 1, tokenCount);
    curExpertIdOffset += 1;
}
// 改：k=0 时跳过 lastExpertId flush，仅由下方 [0,0] 终止符块产出 group_list[0]=[0,0]
if (validGatherIdxLength > 0 && curExpertIdOffset < expertNum) {
    groupListOutLocal.SetValue(curExpertIdOffset * NUM_TWO, lastExpertId);
    groupListOutLocal.SetValue(curExpertIdOffset * NUM_TWO + 1, tokenCount);
    curExpertIdOffset += 1;
}
```

修复后 NPU 实测：kzero 8/8 二进制一致 PASS（group_list[0]=[0,0]），samebin 8/8 PASS（group_list[0]=[5,24]），两场景 3x bitwise reproducible。单行守卫不影响非零路径。

> 备选：也可在 `ComputeOneCore` 入口 `if (inputNum == 0) return;`，效果等价；但 flush 块仍需守卫（`lastExpertId` 初值 -1，`SetValue(-1, ...)` 越界更危险），故推荐在 flush 处加守卫，语义最清晰。

## 迭代三行动项

1. 主线 `op_kernel/arch35/ffn_wb_group_listing.h` 落地上述守卫。
2. 核对 arch22 `op_kernel/ffn_wb_group_listing.h` 是否同缺陷（同源代码）；若 A2 也存在，需评估是否属既有行为（A2 对拍基线本身是否已含该脏值）——若 A2 base 也产 [expertNum,0]，则 A5 "对齐 A2" 与 "对齐 spec [0,0]" 冲突，需回 Architect 裁决基线口径。
3. ~~多核 group_listing 路径在 k=0 时的行为需另行穿刺确认~~ → **✅ 已结论：dispatch 逻辑证明不可达，无需穿刺**。主线 `op_kernel/ffn_worker_batching_apt.cpp:83`（及 probe kernel 同逻辑）：`if (validGatherIdxLength > GROUP_LISTING_MULTI_CORE_LENGTH) { groupListingDealFlag = 1; }`，其中 `GROUP_LISTING_MULTI_CORE_LENGTH = 512`（`ffn_wb_common.h:58`，正值）。k=0 即 `validGatherIdxLength==0`，`0 > 512` 恒假 → **k=0 必然走单核 `dealFlag==0` 路径**（本 issue 已修守卫、已 NPU 8/8 验证的那条）。"多核 + k=0" 两条件互斥、状态不可构造，故非真空点。此为比"造用例穿刺"更强的 dispatch 级证明。
4. UT 增加 k=0 用例断言 `group_list == 全 0`。
