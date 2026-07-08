# 迭代二 穿刺3 (probe_iter2_boundary_kzero_samebin) 验证结果

## 状态: ⚠️ 部分成功（真实 Ascend950 NPU）

- **same_bin（expert_num=8192 全同专家）**: ✅ 8/8 二进制一致，边界可行，无需改动。
- **k=0（全 mask）**: 穿刺**暴露主线 group_listing 单核路径 k=0 缺陷**（`group_list[0][0]` 写脏 expert_id，应为 0）；根因已定位，单行守卫修复后 NPU 实测 8/8 二进制一致。缺陷归档 `issues/issue_20260708_group-listing-kzero-dirty-expertid_01.md`，待迭代三整合到主线 arch35 头。

> 穿刺目的即为迭代三难点降风险（PLAN 迭代二穿刺 §"验证迭代三全覆盖难点"）。本穿刺按预期在 k=0 边界暴露了一个真实缺陷并给出可验证修复，达成降风险目标。

## 工程信息

| 项 | 值 |
|----|----|
| Task Name | probe_iter2_boundary_kzero_samebin |
| 验证目标 | 迭代三难点边界：全 mask(k=0) + same_bin(expert_num=8192 全同专家) |
| TilingKey × 路径 | 100 NORM |
| Dtype | fp16 (P0) — token_dtype=0 |
| 排序模式 | 单核 sort（totalLength=24 ≤ sortLoopMaxElement） |
| 运行环境 | **NPU（Ascend950DT_9584 / arch35 / DAV_3510）** |
| 编译 arch | `--npu-arch=dav-3510`（bisheng, cann-9.1.0） |
| CANN 环境 | `/home/zl/Ascend`（cann-9.1.0，`source .../set_env.sh`） |
| 运行时平台参数 | aivNum=64，UB=253952 B（248.0 KB）——运行时查询，无 arch 常量写死 |
| 验证方式 | **Kernel 直调**（host 组装 ScheduleContextInfo 标量 + GM 指针；非 TTK kernel 模式，schedule_context 二级指针无法注入已证据化） |

复用 `probe5_norm_fp16_skeleton` 的 arch35 骨架头（`ffn_wb_common.h` / `ffn_wb_sort_base.h` / `ffn_wb_sort_one_core.h` / `ffn_wb_gather_out_all.h` / `ffn_wb_group_listing.h`）+ 通用 kernel `probe_ffn_wb_norm`，仅按场景参数化 `expert_ids` / `expert_num`。

## 穿刺形状（两场景共用 shape，仅 expert_ids / expert_num 不同）

`A=4, M=2, BS=2, K=3, H=16`，`totalLength=Y=24`，coreNum=2（core0 gather / core1 group_listing）。

| 场景 | expert_ids | expert_num | 期望（spec.boundary_conditions） |
|------|-----------|:---:|------|
| kzero | 全部 = MASK_VAL(1000000) | 8 | actual_token_num=0，group_list 全 [0,0]，y/ids/scale 全 0 |
| samebin | 全部 = 5（无 mask，全同专家） | 8192 | actual_token_num=24，group_list[0]=[5,24]、[1]=[0,0]、其余 0 |

## 验证结果

### samebin（未改动，直接 PASS）

| 输出 | 元素数 | 结果 |
|------|-------|------|
| y (fp16) | 384 | ✅ 二进制一致 |
| group_list (int64) | 16384 | ✅ 二进制一致（head=[5,24,0,0]，单专家非零，符合 same_bin 契约） |
| session_ids / micro_batch_ids / token_ids / expert_offsets (int32) | 各 24 | ✅ 二进制一致 |
| dynamic_scale (fp32) | 24 | ✅ 二进制一致（恒 0） |
| actual_token_num (int64) | 1 | ✅ 二进制一致（=24） |

**8/8 PASS**，连跑 3 次合并 SHA256 恒为 `4d87a9bd76b98a4d96394d82d1f0ee3b4527fc09219086191a2bf20b90209113`。

### kzero（穿刺暴露缺陷 → 修复后 PASS）

**修复前（首跑，暴露缺陷）**：7/8 PASS。actual_token_num=0 ✅、y/ids/scale 全 0 ✅，唯 `group_list[0]=[8,0]`（脏 expert_id，应为 [0,0]），差 1 个 int64。多次运行确定性复现（脏值稳定 =expertNum）。

**根因**：`KernelFfnWBGroupListing` 单核路径（`groupListingDealFlag==0`）缺 `validGatherIdxLength==0` 守卫——k=0 时 `ComputeOneCore(0)` 读 0 字节搬运后的未初始化 UB（`lastExpertId=inLocal.GetValue(0)`），并 flush `group_list[0]=[脏值,0]`。详见 issue。

**修复后（单行守卫 `validGatherIdxLength > 0`）NPU 实测**：

| 输出 | 元素数 | 结果 |
|------|-------|------|
| y (fp16) | 384 | ✅ 二进制一致（全 0） |
| group_list (int64) | 16 | ✅ 二进制一致（head=[0,0,0,0]，全 0） |
| session_ids / micro_batch_ids / token_ids / expert_offsets (int32) | 各 24 | ✅ 二进制一致（全 0） |
| dynamic_scale (fp32) | 24 | ✅ 二进制一致（全 0） |
| actual_token_num (int64) | 1 | ✅ 二进制一致（=0） |

**8/8 PASS**，连跑 3 次合并 SHA256 恒为 `e253c6a87bdd62e771c0ef1b9850dbc9523c51408ca282f994d3530dbbad9b11`。修复对 samebin 无影响（samebin 修复后仍 8/8 PASS）。

> 本工程内 `op_kernel/arch35/ffn_wb_group_listing.h` 已应用该守卫修复（标 `[PROBE-FIX 迭代二穿刺3]`），供迭代三主线整合参考。

## 关键结论（供迭代三）

1. **same_bin(expert_num=8192)** 边界在 arch35 完全可行：单核 sort→gather→单核 group_listing 三步链对"全落同一 bin + expert_num 上界"二进制一致，group_list 单专家非零 + 末尾 [0,0]，确定性成立。
2. **k=0(全 mask)** 暴露 group_listing 单核路径缺陷：需在迭代三主线 `ffn_wb_group_listing.h` 落地 `validGatherIdxLength>0` 守卫；并核对 arch22 同源代码 / A2 对拍基线口径是否一致（见 issue 行动项 2）；多核 group_listing 分支 k=0 未覆盖，需另穿刺（issue 行动项 3）。
3. 平台资源全部运行时查询（aivNum=64 / UB=248KB），无 arch 常量写死。

## 复现

```bash
source /home/zl/Ascend/ascend-toolkit/set_env.sh
cd /home/zl/ops-transformer/ffn/ffn_worker_batching/probe/probe_iter2_boundary_kzero_samebin
bash run.sh           # 编译 + kzero + samebin（各含精度 + 3x 确定性）
# 复现"修复前"缺陷：将 ffn_wb_group_listing.h 的 [PROBE-FIX] 守卫回退为
#   if (curExpertIdOffset < expertNum) { ... }  后重编运行 kzero，即见 group_list[0]=[8,0]。
```
