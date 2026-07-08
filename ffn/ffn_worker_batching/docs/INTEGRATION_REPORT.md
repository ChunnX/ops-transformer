# FfnWorkerBatching 迭代二 整合报告（NORM+RECV 双 TilingKey / arch35）

> 环境：CANN `/home/zl/Ascend`（cann-9.1.0，bisheng），目标 Ascend950DT_9584（arch35/DAV_3510）。
> 交付口径：**编译通过 + 双 TilingKey 整合 + UB 口径收敛**（NPU 精度对拍由穿刺/汇合验证负责）。
> 原则：A2/A3(ascend910b/ascend910_93) 逻辑零改、零回归；禁写死 arch UB/核数常量。

## 1. 整合目标达成对照（PLAN.md 迭代二整合目标）

| 目标 | 状态 | 说明 |
|------|------|------|
| ① NORM(100)+RECV(101) 双 TilingKey 完整整合，单/多核 sort 全路径 | ✅ | `ffn_worker_batching_apt.cpp` 分发 100/101；NORM 单/多核、RECV 三分支全打通 |
| ② RECV 三种 sort 分支（isNormalBsK 分流） | ✅ | OneCore / MultiCore / MultiCoreBsK 按 `tilingKey` + `isNormalBsK` 分流，照搬 A2 语义 |
| ③ 合并主线 + 穿刺1(RECV) + 穿刺2(多核 sort) 已验证逻辑 | ✅ | 复用 A2 主线 `ffn_wb_*.h` 移植到 arch35（穿刺已证 A2 SIMD 全栈直编 dav-3510 二进制一致） |
| ④ tiling SetBatchingTilingKey(100/101) + UB 派生阈值全运行时推导；RECV 限核阈值标定 | ✅ | `_tiling_arch35.cpp`：`GetTilingKey()` by need_schedule；`sortLoopMaxElement` 由运行时 ubSize 派生；TH_RECV_CORE_NUM=32/TH_RECV_Y_NUM=200000 沿用 A2 |
| ⑤ 收敛 UB 216/248 不一致 | ✅ | 见 §4；issue 已更新收敛结论 |
| ⑥ 单 dtype 保持 P0 fp16 | ✅ | kernel token_dtype 正交处理，profile 全覆盖留迭代三；本迭代验证以 fp16 为主 |

## 2. 代码整合内容

### 2.1 kernel 侧（op_kernel/）

**入口 `ffn_worker_batching_apt.cpp`（重写）**：迭代一仅 NORM 单核 → 迭代二整合为：
- `FfnWorkerBatchingProcess`（NORM/100）：`TilingSort` → 单核 `KernelSortMaskOneCore` / 多核 `KernelSortMaskMultiCore`（含 MrgSort 归并）→ `ValidGatherIdxLengthCompute` → gather/group_listing 核间分工（`GROUP_LISTING_AIV_NUM` / `GROUP_LISTING_MULTI_AIV_NUM`）。
- `FfnWorkerBatchingRecvProcess`（RECV/101）：`ScheduleContextInfoCompute<true>` → `KernelScanTokenInfo` 前置扫描 + `SyncAll` → `TilingScanSort` → 三分支（`KernelScanSortMaskOneCore` / `KernelScanSortMaskMultiCoreBsK`(isNormalBsK≠NORMAL_BSK) / `KernelScanSortMaskMultiCore`）→ `ValidGatherIdxLengthCompute` → gather(`KernelFfnWBGatherOutAll<true>`) / `KernelFfnWBScanGroupListingOneCore` 核间分工（保留 `GROUP_LISTING_SCAN_AIV_NUM=1` 核做 group_listing）。
- 分发 `TILING_KEY_IS(100/101)`；**102(RECV_GATHER) 不分发**（A2 保留分支，tiling 从不 SetTilingKey(102)，对齐 DESIGN §1.4）。

**arch35/ 头文件（9 个新增移植，补齐至 15 头 + 1 tiling_def）**：
迭代一已有 6 头（common / get_schedule_context / sort_base / sort_one_core / gather_out_all / group_listing）。本迭代补齐：
`ffn_wb_sort_multi_core.h`、`ffn_wb_sort_mrgsort.h`、`ffn_wb_sort_mrgsort_out.h`、`ffn_wb_scan_token_info.h`、`ffn_wb_scan_sort_one_core.h`、`ffn_wb_scan_sort_multi_core.h`、`ffn_wb_scan_sort_multi_core_bsk.h`、`ffn_wb_scan_group_listing_one_core.h`、`ffn_wb_scan_get_valid_experts.h`（scan_get_valid_experts 为 102 保留分支伴随件，随 SortMaskBase 一并移植，不产生 102 分发）。

**移植方式**：这 9 头与 A2 源同族、仅消费 `ScheduleContextInfo` / `SortCustomTilingDataKernel`（均定义于 `arch35/ffn_wb_common.h`），**零 `FfnWorkerBatchingTilingData` 直接引用**，故为近逐字移植——`#include "ffn_wb_*.h"` 在 arch35/ 内相对解析。穿刺已证 A2 经典 SIMD 全栈（Sort/MrgSort/Concat/Extract/GatherMask/DataCopyPad/SetAtomicAdd）直编 dav-3510 无需 RegBase。

**MrgSortOut 双重偏移规避（穿刺2 排障 #2）**：移植的 `ffn_wb_sort_mrgsort_out.h` + `ffn_wb_scan_sort_multi_core.h` 保持 A2 主线约定——`SetInput` 传共享 workspace 基址（`workspaceGms[srcWsIndex]`），仅由 `offsets[i]=GetSortOffset(perListElements*i)` 区分各 list，**不叠加 per-list base**（对齐 `moe_v3_mrgsort_out.h`）。该 double-offset bug 仅存在于穿刺2 独立 .asc，主线移植代码本就正确。

### 2.2 host 侧（op_host/）— 迭代一已落地，本迭代复核确认

- `_tiling_arch35.cpp`（1000 档 Regbase，`IsCapable()=IsRegbaseSocVersion`）：`GetTilingKey()` 仅由 need_schedule 决定 100/101；`sortLoopMaxElement = ubSize/(4*2*4)/32*32` 运行时派生；`sortNumWorkSpace=Y`；RECV 限核 `needSchedule==1 && Y<=200000 → min(coreNum,32)`；workspace 公式同 A2。
- `ffn_worker_batching_tiling.cpp`：分派器 `Tiling4FfnWorkerBatching` 顶部 `IsRegbaseSocVersion` 守卫（ascend950→DoTilingImpl，A2/A3→原 monolithic），A2 类体零改。
- `ffn_worker_batching_def.cpp`：`AddConfig("ascend950", regbaseCfg)` + `ExtendCfgInfo("opFile.value","ffn_worker_batching_apt")`；A2/A3 两行 AddConfig 不动。

## 3. 编译验证

| 路径 | 命令 | 结果 |
|------|------|------|
| ascend950（arch35 整合 kernel） | `bash build.sh --pkg --soc=ascend950 --ops=ffn_worker_batching` | ✅ EXIT 0，无 error，**3 个 kernel 二进制生成**（`build/binary/ascend950/bin/ffn_worker_batching_apt/*.o`），simplifiedKey 含 d=0(NORM)/d=1(RECV) |
| ascend910b（A2 零回归） | `bash build.sh --pkg --soc=ascend910b --ops=ffn_worker_batching` | ✅ EXIT 0，`Built target ffn_worker_batching`（arch22 opFile），无 error |
| 自定义算子包 | 两路径均生成 `.run` 包 | ✅ `build_out/cann-ops-transformer-customize_linux-x86_64.run` |

**命名冲突检查**：无。同名 `extern "C" ffn_worker_batching` 经 opFile 隔离（ascend950 取 `_apt.cpp`、A2/A3 取 `ffn_worker_batching.cpp`），各 soc 编译单元符号唯一。TilingData（`FfnWorkerBatchingArch35TilingData` 平铺 struct vs A2 `FfnWorkerBatchingTilingData` 注册类）互不冲突。

**A2/A3 零回归证据**：`git status` 显示 op_kernel 下仅 `arch35/` 与 `ffn_worker_batching_apt.cpp` 为新增（`??`），arch22 入口 `ffn_worker_batching.cpp` 与根目录 15 个 `ffn_wb_*.h` **无 Modified 标记**，逐字未动。

## 4. UB 216/248 口径收敛（issue_20260708_ub-size-host-kernel-mismatch_01）

**结论：统一以 `GetCoreMemSize(CoreMemType::UB)` 返回值（arch35=253952 B=248KB）为 host/kernel 唯一 UB 真值源。**

- **根因定性**：216KB 与 248KB 测的不是同一个量。`GetCoreMemSize(UB)`（host tiling 用）读 CANN `platform_config/Ascend950DT_*.ini` 的 `ub_size`——本环境实测全部 `ub_size=253952`，是算子编译/tiling 侧的**权威可用 UB**；`ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE`（穿刺1/3 Kernel 直调工装用）走 ACL runtime attr 返回 221184，是穿刺工装的保守 runtime 口径（穿刺1 还自扣 16KB），**非算子编译期上限**。两者服务对象不同，非真实硬件冲突。
- **落地**：kernel 侧 UB 派生只经 `ScheduleContextInfo.ubSize`（由 host TilingData 拷入），**无任何直接 UB 查询、无字节常量**（grep 验证 arch35/*.h 仅 tiling_def.h 注释提及）；host 侧只用 `GetCoreMemSize(UB)`。
- **不溢出数学保证**：单核 sort 4×InitBuffer = `32×sortNum` 字节，单核触发上界 `totalLength≤sortLoopMaxElement=ubSize/32`，故合计≈`ubSize`——host/kernel 用同一个 248KB，预算自洽贴合；超上界自动走多核（每核≤sortLoopMaxElement），同样不溢出。编译器按 248KB 接受 InitBuffer 无 overflow 报错。
- **遗留**：A=1024 上界大 shape 多核 NPU 实跑不溢出验证，交穿刺/汇合在真机确认（PLAN 迭代二穿刺2）。

issue 状态已更新为 ✅已收敛。

## 5. 交付件

| 交付件 | 路径 |
|-------|------|
| arch35 入口（整合双 TilingKey） | `op_kernel/ffn_worker_batching_apt.cpp` |
| arch35 kernel 头（15 头 + tiling_def） | `op_kernel/arch35/*.h` |
| arch35 tiling（1000 档 Regbase） | `op_host/ffn_worker_batching_tiling_arch35.cpp` |
| ascend950 config | `op_host/config/ascend950/` |
| 整合报告 | `docs/INTEGRATION_REPORT.md`（本文件） |
| UB 收敛结论 | `issues/issue_20260708_ub-size-host-kernel-mismatch_01.md` |

## 6. 后续（迭代三）

- 全 profile（P0 fp16 / P1 bf16 / P2 int8）× 双路径 6 组合全功能覆盖 + 全边界。
- NPU 实跑对拍 A2 二进制一致（本迭代穿刺列表：P2 int8 RECV / A=1024 上界 / 全 mask+same_bin / ~50% 稀疏 RECV）。
- 全 UT/ST 覆盖 + A2/A3 回归。
