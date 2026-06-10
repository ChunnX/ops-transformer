# MixedQuantSparseFlashMla Work Progress

## Current Session

- 2026-06-08：已按要求清空旧版 `work_progress.md`，后续工作进展从本文件重新维护。
- 2026-06-08：最终目标是为 QSMLA 算子实现 flash decode 功能。

## Flash Decode Progress

- 已完成 S2 跨核切分的完整链路：metadata 单位转换 → isS2Split 标记 → staging 暂存 → ProcessFlashDecode 归约。
- 详细实现文档见 `/home/panjie/workspace/aichat/mixed_quant_sparse_flash_mla/s2_split_fd_implementation.md`。

### 已完成的提交

| Commit | 描述 |
|--------|------|
| `23b6ceb57` | 修复 `ApplyS2MetadataRange` 单位不匹配（block 级→token 级），新增 `isS2Split`/`s2SplitIdx` 标记 |
| `dc17dc839` | S2 split 暂存区：workspace 分配、ProcessVec2 staging 写入、isS2Split 分支处理 |
| `239d0c2f4` | 实现 ProcessFlashDecode 归约逻辑（使用 common VF），重构 staging 布局为连续区域，修复 FD buffer 溢出 |

### 当前状态

- **metadata 解析**：已完成，`FdRunInfo` 字段齐全
- **S2 范围裁剪**：已完成，`ApplyS2MetadataRange` 按 token 级裁剪并计算 `isS2Split`
- **暂存区写入**：已完成，`ProcessVec2` 最后一轮 S2 循环按 `isS2Split` 分支写 output/sum/max 到 GM
- **FD 归约**：已完成，`ProcessFlashDecode` 使用 `ComputeScaleValue_8_FD` + `ReduceFinalRes_VF` 归约
- **编译验证**：尚未执行

## Open Questions

- `SyncAll()` 全局 barrier 是否需要改为 GM polling 机制
- LSE 输出（`lse = globalMax + log(globalSum)`）尚未实现
- sinks 初始化在 `isS2Split && s2LoopCount == 0` 路径的对齐
- 更大 `mSize` 下 `commonTBuf`/`fdLseExpBuf` 容量验证
- 2026-06-09：针对 `BSND/PA_BBND, B=1, S1=1, S2=512, N1=64, CFA, cmp_ratio=4` decode case 继续定位：当前 HEAD 曾临时禁用 FD 归约，且 staging 使用“每核局部基址 + splitIdx”与 metadata 的全局 `firstFdDataWorkspaceIdx/FD_WORKSPACE_IDX` 不一致；恢复 FD 调用链后，需要使用全局 staging 基址、全局 slot 区域边界，并按 subBlock/FD_M_START 对 output/max/sum 做 M 行偏移读写。
- 2026-06-09：处理 common `vf_flash_decode.h` 中 `StoreDist` 解析编译问题时，未修改 common；对齐 FIA 的包含方式，在 QSMLA vector 私有头中将 CANN 基础头和 matmul 头前置到 `vf_flash_decode.h` 之前，并移除临时 `using AscendC::Reg::StoreDist`。
- 2026-06-09：修复 FD 前 `SyncAll()` 卡死风险：`hasLoad == 0` 的空核原先在主循环前提前 return，导致有效任务少时 active 核在 FD 前屏障等待不到空核；已在空核 return 前补同一个 `SyncAll()`。
- 2026-06-09：定位 `cfa_decode_cmp128` 中 aiv-0/aiv-1 `isS2Split=false`：metadata 已正确给 AIC0 `[S2Start=0,S2End=1)`、AIC1 `[1,0)`，但 kernel 只用 `M_END != 0` 判断是否需要把 `BN2_END` 转成右开循环上界，漏掉了同 BN/M 的 S2-only split。已改为 `M_END != 0 || S2_END != 0`。
- 2026-06-10：用户定位 FD 编译失败根因是 common `vf_flash_decode.h` 内未限定 `StoreDist`；为避免修改 common，在 QSMLA 私有 `mixed_quant_sparse_flash_mla_scfa_block_vector.h` include `vf_flash_decode.h` 前补 `using AscendC::Reg::StoreDist;`。
