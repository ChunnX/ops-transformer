# issue_20260708_ub-size-host-kernel-mismatch_01

## 标题
arch35(DAV_3510) UB 容量 host 侧(248KB) 与 kernel 侧(216KB) 查询口径不一致，大 shape 下有 UB 溢出风险

## 状态
✅ 已真正修复（2026-07-08 会话恢复后，changwei step4→step5 之间）
⚠️ 下方"收敛结论（迭代二整合）"一节的定性（"248KB 是权威值、216KB 仅工装口径、用同一 ubSize 即自洽不溢出"）**已被穿刺2 实测推翻，保留仅作过程记录，勿再据其结论**。真实修复见本节下方「## 最终修复」。

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

## 最终修复（2026-07-08，会话恢复后）

### 为何推翻上面的"收敛结论"
上面「收敛结论（迭代二整合）」把 248KB 定为权威值、认为用同一个 ubSize 就自洽不溢出。**该结论错误，被穿刺2（`probe/probe_iter2_boundary_maxshape`）同进程实测推翻**：

- 穿刺2 同一进程内同时查两口径坐实：kernel 侧 `ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE`=**221184(216KB)** 是 vector core 物理可用上界，host 侧 `GetCoreMemSize(UB)`=**253952(248KB)** 是平台标称值，差 32KB 是**真实系统预留**，非工装口径差。
- 上面「数学保证」只算了**单核** sort（单核 4-buffer footprint≈ubSize，贴合 248KB）；但**多核** sort 每核同样分配 4 个 buffer，每核每 loop tile 打到 `sortLoopMaxElement=ubSize/32=7936` 时，footprint=`4×7936×8=253952(248KB)` **超物理 216KB 达 32KB → 溢出**。多核这一支被漏算。
- 穿刺2 实测：按 kernel 216KB 派生的 SLM=6912→footprint 216KB **FITS，NPU 实跑 8/8 二进制一致无 507035**；按 host 248KB 派生的 SLM=7936→footprint 248KB **OVERFLOW**。三重证据（实测两口径 + footprint 记账 + 216KB 恰好装满）。

### 修复内容
`op_host/ffn_worker_batching_tiling_arch35.cpp::GetPlatformInfo()`：`GetCoreMemSize(UB, ubSize_)` 后**扣减 32KB 系统预留**再用于所有 UB 派生：
```cpp
constexpr uint64_t UB_SYS_RESERVED_SIZE = 32 * 1024;  // arch35 vector core 系统预留
...
ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
ubSize_ -= UB_SYS_RESERVED_SIZE;   // 得 vector core 实际可用 UB，与 kernel 侧真实口径对齐
```
效果：`ubSize_` = 253952-32768 = **221184(216KB)**，`sortLoopMaxElement`=221184/32=**6912**（与穿刺2 FITS 值一致）。kernel 侧全部 UB 派生（sort footprint、gather `ubAvailable`、group_listing `perLoopMaxRows`）经 `tilingData.ubSize` 统一拿到扣减后的 216KB，一致修正（之前按 248KB 全部偏大，只是小 shape 未触顶）。

### 修法依据（非自造安全系数）
- 对齐同仓 arch35 算子既有约定：`mhc/mhc_pre_sinkhorn_backward/op_host/op_tiling/mhc_pre_sinkhorn_backward_tiling.cpp:491` 的 `TilingMhcPreSinkhornBackwardArch35` 里同样 `ubSize = ubSize - 32*1024; SetDynUBufSize(ubSize)`。
- host tiling 侧 `platform_ascendc` API 无 vector-core UB 专用接口（`CoreMemType` 只有单一 `UB=5`，无 device attr），确认后采用"标称值扣固定系统预留"是仓内标准做法。
- **不违反 MEMORY「UB size no hardcode」**：UB 容量仍运行时 `GetCoreMemSize` 取值，只扣一个与容量无关的固定系统预留常量（32KB），非把 216/248 这种 arch UB 容量写死。

### 验证
- ascend950 `--opkernel` 重编 EXIT 0，3 binary 重新生成（tiling 逻辑变更编译通过）。
- op_host tiling UT 期望值同步更新到扣减后口径（253952→221184/6912；262144→229376/7168），A2/A3(DAV_2201) 对照用例口径不动。
- 待办：泛化阶段用大 shape（多核 sort 每核 tile 打到 SLM=6912）NPU 实跑，最终确认不溢出（穿刺2 已在等价构造下实证 FITS）。

## 关联
- 迭代一穿刺3/4 RESULT.md、汇合验证 iter1-integration-report.md 均记录了该观察项。
- **穿刺2 `probe/probe_iter2_boundary_maxshape/RESULT.md`：推翻旧收敛结论、坐实溢出的关键实测载体。**
- 代码检视报告 `docs/arch35_code_review_summary.md` H1：抓住 INTEGRATION_REPORT 与 PROBE_SUMMARY 结论矛盾并定为头号 HIGH。
- PLAN.md 迭代二穿刺2（边界 shape A=1024）。
- 本迭代整合报告：`docs/INTEGRATION_REPORT.md`（其 §4 的 248KB 唯一真值源结论同被本修复推翻）。
