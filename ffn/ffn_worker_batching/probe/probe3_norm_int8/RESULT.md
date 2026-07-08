# probe3_norm_int8 验证结果

## 状态: ✅ 成功

## 任务信息
- Task Name: probe3_norm_int8（迭代一 穿刺3）
- TilingKey: 100 (NORM)
- Dtype: int8 (P2, dynamic_quant)
- 排序模式: 单核 sort
- Memory Strategy: 尾部 32B FP32 scale 连续 gather
- 穿刺验证目标: P2 dynamic_quant int8 —— `BLOCK_BYTES=32` scale 连续搬运 + `dynamic_scale` 输出在 arch35 的搬运语义可行性

## 运行环境
- **运行环境: NPU**（真实硬件，非 Mock/CPU 仿真）
- 芯片: **Ascend950DT_9584**（arch35 / DAV_3510）
- CANN: 9.1.0；driver 7.0.t9.0.B849
- 平台运行时查询值（禁写死 arch 常量）：`vectorCoreNum=64`、`UBUF_PER_VECTOR_CORE=221184 字节 (216KB)`
  - 说明：DESIGN §3.6.3 假设 UB=248KB，实测运行时查询为 **216KB**；因代码全部经 `ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE` 运行时取值，阈值自动适配，无需改代码（印证「禁写死 UB 常量」的必要性）。

## 验证方法
- Kernel 直调工程（`ascendc-direct-invoke-template` 改造），`--npu-arch=dav-3510` 编译。
- 直接复用 arch22 的 6 个功能头（`ffn_wb_common.h` / `ffn_wb_get_schedule_context.h` / `ffn_wb_sort_base.h` / `ffn_wb_sort_one_core.h` / `ffn_wb_gather_out_all.h` / `ffn_wb_group_listing.h`），arch35 仅重编译（对齐 DESIGN §3.1 经典 SIMD/MemBase 主路线）。
- 入口逻辑照搬 `ffn_worker_batching.cpp::FfnWorkerBatchingProcess`（NORM 路径），直调场景 workspace 直接作为 userWS。
- Host 侧构建 1024B `schedule_context`（内嵌 device 子 buffer 地址：token_data / session_ids / micro_batch_ids / expert_ids），完整跑通 **sort → gather → group_listing** 三步。
- CPU golden 按 spec.yaml `math_semantics` 三步语义 + arch22 参考实现独立复算，精度口径 = 二进制一致（`np.array_equal`, rtol=atol=0）。

## 测试用例
| 项 | 值 |
|----|----|
| A (session_num) | 8 |
| M (micro_batch_num) | 2 |
| BS (micro_batch_size) | 2 |
| K (topK+1) | 4 |
| H (hidden, int8) | 128 |
| HS (attn_to_ffn_token_size) | 160 = H(128) int8 + 尾部 32B FP32 scale 块 |
| expert_num | 16 |
| token_dtype | 2 (P2 int8) |
| Y = A·BS·K | 64 |
| 有效 token (valid_len) | 56（含约 12.5% mask → expert_id 置 2000000 排末尾剔除） |

## 验证摘要
| 输出 | dtype | 元素数 | 结果 |
|------|-------|-------|------|
| y | int8 | 8192 (64×128) | ✅ bitwise equal |
| group_list | int64 | 32 (16×2) | ✅ bitwise equal |
| session_ids | int32 | 64 | ✅ bitwise equal |
| micro_batch_ids | int32 | 64 | ✅ bitwise equal |
| token_ids | int32 | 64 | ✅ bitwise equal |
| expert_offsets | int32 | 64 | ✅ bitwise equal |
| **dynamic_scale** | float32 | 64 | ✅ bitwise equal（P2 关键路径） |
| actual_token_num | int64 | 1 (+pad) | ✅ bitwise equal（=56） |

- **8 路输出全部与 CPU golden 二进制一致**（`np.array_equal`）。
- **确定性**：相同 `schedule_context` 连续两次运行输出逐 bit 相同（spec `determinism.bitwise_reproducible` ✅）。

## 关键发现（arch35 int8 + 32B scale gather 语义）
1. **P2 尾部 32B FP32 scale 连续 gather 在 DAV_3510 可行**：`KernelFfnWBGatherOutAll<false>` 的 int8 hidden 字节 gather（`DataCopyPad`）+ 尾部 4B FP32 scale 读取（`tokenDataBufGm_[base+H]`）+ scratch 区 32B 步长 `ArithProgression`/`Gather` 收敛到连续 `dynamic_scale` 输出，全链路 arch35 重编译即通过，`dynamic_scale` 二进制一致，**无需 RegBase 反量化管线**（印证 DESIGN §3.1 结论：原样字节搬运无算术）。
2. **ScheduleContext 1024B GM 逐字节解析 + DCCI 语义在 arch35 正确**：`ScheduleContextInfoCompute` 指针解析、`ValidGatherIdxLengthCompute` 的 `DataCacheCleanAndInvalid` 跨核读 `validCnt` 均正确（该风险点 DESIGN §7.2 列为中等，穿刺实测通过）。
3. **单核 sort 稳定序 = 升序 (expert_id, 原始索引)**：arch35 `Sort<float,true>`（对 -expertId 降序）+ `GatherMask` 掩码剔除，对相同 expert_id 保持原始索引升序，与 spec「专家 ID + 稳定序」一致；CPU golden 用同一稳定序即二进制命中。
4. **UB 运行时取值链路正确**：`sortLoopMaxElement=ubSize/32/32*32=6912`（totalLength=64 → 单核 sort）、gather `maxBlockSize_` 由运行时 ubSize 派生，实测 216KB 下自动适配，无写死常量。

## 结论
probe3 穿刺目标达成：**P2 dynamic_quant int8（BLOCK_BYTES=32 scale 连续 gather + dynamic_scale 输出）在 arch35/DAV_3510 上功能与精度可行，8 路输出对 CPU golden 二进制一致，确定性满足**。该 gather 分支代码可直接纳入迭代二/三整合（P2 profile）。

## 复现
```bash
cd probe/probe3_norm_int8
bash run.sh              # 完整：编译(dav-3510) → 生成数据+golden → NPU 实跑 → 二进制校验
bash run.sh --skip-build # 复用已编译产物
```
