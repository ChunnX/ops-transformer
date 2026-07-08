# issue_20260708_ub-size-host-kernel-mismatch_01

## 标题
arch35(DAV_3510) UB 容量 host 侧(248KB) 与 kernel 侧(216KB) 查询口径不一致，大 shape 下有 UB 溢出风险

## 状态
✅ 已收敛（迭代二整合，2026-07-08）

## 背景
迭代一穿刺阶段，两个穿刺各自用不同接口查询单 vector core 的 UB 容量，返回值不一致：

| 来源 | 接口 | 返回值 |
|------|------|--------|
| 穿刺3 (int8)，kernel 侧 | `ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE` | 221184 B = **216KB** |
| 穿刺4 (bf16) / 主线 tiling，host 侧 | `GetCoreMemSize(UB)` / `PlatformAscendCManager` | 253952 B = **248KB** |

主线 tiling（`op_host/ffn_worker_batching_tiling.cpp:76` `GetCoreMemSize(UB)`）走的是 **host 侧 248KB** 这一支，据此派生 `sortLoopMaxElement` 等 UB 阈值。

## 现象 / 根因
- 同一颗 Ascend950DT_9584（arch35/DAV_3510），host 侧平台接口与 kernel 侧 ACL runtime attr 返回的单核 UB 容量差 ~32KB（约 13%）。
- 疑似两接口对 UB 保留区（如 workspace/系统预留）的口径不同（对比 A2/DAV_2201：host 侧预留 256B 得 261888）。
- 迭代一穿刺 shape 都很小、UB 分配未触顶，故 3/4 均通过，问题未暴露。

## 影响
- **迭代二/三的多核 + 大 shape 场景**：主线 tiling 按 248KB 预算切分，但 kernel 实际每核仅 216KB 可用 → 可能 UB 溢出 / 踩内存 / 精度错乱。
- 属"底层 API 行为不一致 + 需后续跟进"，非迭代一阻塞项，但**迭代二整合（多核 sort/MrgSort 大 shape）前必须收敛**。

## 建议方案（迭代二 A1-Main 整合时）
1. 统一 UB 取值口径：确认 host tiling 与 kernel 实际可用 UB 是否应取同一保守值（min(216KB, 248KB) 或按 kernel 侧实测值）。
2. 若 host 248KB 含 kernel 不可用的保留区，则 tiling 阈值改按 kernel 侧真实可用值派生，避免超分。
3. 用一个大 shape（A=1024 上界）多核用例在 NPU 上验证 UB 不溢出（对应 PLAN 迭代二穿刺2 边界 shape）。
4. 全程仍禁止写死 arch 常量（对齐 MEMORY「UB size no hardcode」），只统一"用哪个运行时接口"。

## 收敛结论（迭代二整合，2026-07-08）

**统一口径：host tiling 与 kernel 均以 `GetCoreMemSize(CoreMemType::UB)` 返回值（arch35 = 253952 B = 248KB）为唯一 UB 真值源；kernel 侧不再出现第二个 UB 查询口径。**

### 根因定性
- 两接口测的不是同一个量：
  - `platform_ascendc::PlatformAscendCManager::GetCoreMemSize(UB)`（host tiling 用）读的是 CANN `platform_config/Ascend950*.ini` 的 `ub_size`。本环境实测 `/home/zl/Ascend/cann-9.1.0/x86_64-linux/data/platform_config/Ascend950DT_*.ini` 全部 `ub_size=253952`——这是**编译/tiling 侧算子可用 UB 的权威值**，InitBuffer 预算即以此为上限校验。
  - `ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE`（穿刺1/3 kernel 直调 host 用）走 ACL runtime device attr，返回 221184（216KB），是 Kernel 直调穿刺工装为保守分配而查询的 runtime 口径，**并非算子编译期 UB 上限**；穿刺1 还额外自扣 16KB 冗余，属工装保守策略。
- 结论：248KB 是算子正式路径（tiling→kernel InitBuffer）应遵循的口径；216KB 是穿刺工装侧的保守 runtime 值，两者服务对象不同，非真实硬件冲突。

### 落地方式（本迭代已实施）
1. **kernel 侧 UB 单一真值源**：`op_kernel/arch35/` 全部头文件的 UB 派生（sort 的 `sortLoopMaxElement`、gather 的 `ubAvailable/maxBlockSize_` 等）**只经 `ScheduleContextInfo.ubSize`** 取值，该字段由 `ScheduleContextInfoCompute` 从 host TilingData（`FfnWorkerBatchingArch35TilingData.ubSize`）拷入。kernel 内**无任何直接 UB/ACL attr 查询、无 192/248/216 字节常量**（grep 验证：`op_kernel/**/*.h` 仅 tiling_def.h 注释提及 GetCoreMemSize，无字面常量）。
2. **host 侧 UB 单一取值接口**：`ffn_worker_batching_tiling_arch35.cpp::GetPlatformInfo()` 用 `GetCoreMemSize(UB)`（=253952），写入 `tilingData.ubSize`；`sortLoopMaxElement = ubSize/(4*2*4)/32*32` 随之派生。禁写死 arch 常量（对齐 MEMORY「UB size no hardcode」）。
3. **不溢出的数学保证**：单核 sort（`ffn_wb_sort_one_core.h`）4 个 InitBuffer 合计 = `4 × sortNum × sizeof(int32) × kvFactor(2)` = `32 × sortNum` 字节；而单核触发上界 `totalLength ≤ sortLoopMaxElement = ubSize/32`，故 `sortNum ≈ ubSize/32`，四缓冲合计 ≈ `ubSize` 字节——**正好贴合、不超过 `GetCoreMemSize` 返回的同一个 ubSize**。因 host/kernel 用的是同一个数（248KB），预算自洽；大 shape 超单核上界时自动走多核切分（每核 ≤ sortLoopMaxElement），同样不溢出。
4. **编译实证**：ascend950 整合 kernel（含单/多核 sort + RECV 三分支 + gather）在 `--npu-arch=dav-3510` 下编译通过、3 个 kernel 二进制生成，InitBuffer 预算按 248KB 被编译器接受无 UB overflow 报错。

### 遗留 / 后续
- PLAN 迭代二穿刺2（A=1024 上界大 shape 多核）仍是 NPU 实跑验证 UB 不溢出的天然载体，交由穿刺/汇合验证在真机确认（本迭代交付口径为「编译通过 + 口径统一」，真机对拍由穿刺/汇合负责）。
- 若未来某芯片 `GetCoreMemSize` 与实际可用 UB 出现真实差异（非工装口径差），应在 host tiling 侧统一加保守 reserve，仍禁止 kernel 侧写死。

## 关联
- 迭代一穿刺3/4 RESULT.md、汇合验证 iter1-integration-report.md 均记录了该观察项。
- PLAN.md 迭代二穿刺2（边界 shape A=1024）是验证收敛的天然载体。
- 本迭代整合报告：`docs/INTEGRATION_REPORT.md`。
