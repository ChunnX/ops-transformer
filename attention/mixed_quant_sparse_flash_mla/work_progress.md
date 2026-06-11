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
- 2026-06-10：定位当前编译失败：`--soc=ascend910b` 会因当前 QSMLA 只生成 `ascend950` ops-info 而在 `generate_ops_info_ascend910b` 阶段失败；改用 `--soc=ascend950` 后进入 kernel 编译，真实错误为 `ProcessFlashDecode()` 中 GM->UB `DataCopyPad` 缺少 `DataCopyPadExtParams` 第四参。已在 staging output 读取处补 `DataCopyPadExtParams<OUTPUT_T>{false, 0, 0, 0}`，并通过 `bash build.sh --pkg --soc=ascend950 --ops=mixed_quant_sparse_flash_mla -j4` 编译打包验证，日志未见实际 compile/install error。
- 2026-06-10：核对 FD 规约阶段拷入数据，发现 `ProcessFlashDecode()` 用 `stage1OutQue[0]` 承接 staging output 读入会越过该队列按 Vec1 P 矩阵配置的容量；同时 staging output 写出按 GM packed `dSizeV` 连续行布局，规约读回却按 `dTemplateAlign64` 做行偏移。已改为使用传入的空闲 `bmm2Buffers` UB 作为 `partialOOut`，读 GM offset 改为 `(fdMOffset + startRow) * constInfo.dSizeV`，并用 `DataCopyPadExtParams` 将 UB 行补齐到 `dTemplateAlign64` 后再 cast/reduce。`bash build.sh --pkg --soc=ascend950 --ops=mixed_quant_sparse_flash_mla -j4` 编译打包通过；当前 `bash test_run.sh single` 仍停在安装态问题：`libopapi.so` 缺少 metadata 接口，未进入 kernel 执行。

## 2026-06-11：507015 访存越界根因定位与修复（FD 规约对齐链路）

用 `bash /home/panjie/qsmlaGo.sh` 可正确编译安装。`bash test_run.sh single`（用例 `single_batch_2_cores`：BSND/PA_BBND, B=1,S1=1,S2=512,N1=64,CFA,cmp_ratio=4, SPLIT_G=0）复现 507015。

### 关键纠正
- 旧判断"507015=GM 访存越界"不准确。设备日志（`~/ascend/log/debug/plog/`）显示精确硬件故障，按修复推进依次为：
  1. `error code 80`（UB 地址未对齐）→ FD 规约把紧凑布局 max/sum 喂给按 8-float/行块布局写死的 common VF。
  2. `error code 263,82`（MTE 写地址越界）→ staging max/sum 写读布局不一致 + 非对齐 DataCopy。
  3. `error code 341`（UB access address overflow）→ 新增 FD buffer 导致 UB 总分配超 192KB 物理上限。

### 已完成修复（方案 B：保持 GM 紧凑布局，UB 内重排为块布局）
- FD 规约读回 max/sum 到每 split 32B(8 float) 对齐槽位（复用 `stage0InBuf[0]/[1]`），用 `Brcb` 广播成 common VF 期望的 8-float/行块布局（复用 `stage0OutBuf[0]/[1]`），`lseExp` 复用 `dequantScaleBuff`，`globalMax` 用 `softmaxMaxBuf[0]`。全部复用主计算空闲 buffer，避免 UB 超容量。
- staging max/sum 写出端（`ProcessVec2`）改为按全局 split 索引 `workspaceIdx*(FLOAT_BYTES/4)+stagingMOffset` 分槽，并用 `DataCopyPad` 替代非对齐 `DataCopy`，与读回端布局统一。
- 去掉末尾重复归一化 `LastDivNew`（`ReduceFinalRes` 用权重和为 1 的归一化权重加权求和，结果已归一化）。
- 关键实测值：`mSize=halfMRealSize=1`（非 8 倍数），`numSplits=2`，`GetBlockNum()=32`，`slotNum=32`，`OUTPUT_BYTES=65536`，`FLOAT_BYTES=256`。

### 当前状态
- **507015 访存越界已彻底消除**，FD 全流程跑通（读回→Brcb→ComputeScaleValue_8_FD→ReduceFinalRes→CopyOut）。
- 现转入**精度问题**：`single_batch_2_cores` 精度 0.00%，需排查 FD 规约数值正确性（staging 写读 split/M 偏移对应、权重计算、加权求和）。
- 所有调试 printf 已清理。

## 2026-06-11：FD 调用链精简重构

### 修改内容
1. **消除 `ParseFdRunInfo` 重复调用**：原先在 `ProcessMainLoop` 和 `ProcessFlashDecode` 内各调用一次。改为仅在 `ProcessMainLoop` 中解析一次，将解析好的 `fdRunInfo` 传参给 `ProcessFlashDecode`。
2. **消除 FD enable 重复判断**：原先在调用 `ProcessFlashDecode` 前判断 `coreEnable`，函数内部又判断一次。改为仅在调用前判断，函数内部不再判断。
3. **移除 subBlock M 维度对半切分**：FD 规约阶段两个 vec 核不绑定，各自独立。每个核直接使用 `fdRunInfo` 中的 `mIdx`（块 ID）、`mStartIdx`（token 粒度起始）、`mNum`（token 数量）。
4. **FD 规约路径去除 `RunInfo` 依赖**：`ProcessFlashDecode` 签名改为 `(FdRunInfo &fdRunInfo, int64_t attentionOutOffset, ConstInfo &constInfo, mm2ResPos &tempBuf)`，`mSize`/`vec2CalcSize` 直接用 `fdRunInfo.mNum`。仅在最终 `CopyOutAttentionOut` 调用处构造局部 `RunInfo`（`Bmm2DataCopyOut` 接口需要）。
5. **移除 `mNum == 0` 判断**：能进入 `ProcessFlashDecode` 的核 `mNum` 必定不为 0。

### 涉及文件
- `op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_kernel.h`：`ProcessFlashDecode` 声明、实现、调用
- `op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h`：`SCFABlockVec::ProcessFlashDecode` 声明、实现、dummy 签名

## 2026-06-11：FD 调用链进一步精简 + max/sum 预广播优化

### 修改内容
1. **删除 kernel.h 中的 `ProcessFlashDecode` 中间层**：主循环直接调用 `vecBlock.ProcessFlashDecode`，`attentionOutOffset` 计算移入 `SCFABlockVec::ProcessFlashDecode` 内部。
2. **`seqOffset` 计算改用 `if constexpr` 编译期分支**：TND 用 `cuSeqlensQGm`，其余用 `bn2Idx * s1Size`。
3. **修复 `attentionOutOffset` stride bug**：`mStartIdx * dSizeV` → `mStartIdx * n2GDv`（token 级偏移应乘以所有 head 的 V 元素数）。
4. **无 FA 负载的 AIV 核支持 FD 计算**：`hasLoad == 0` 的 AIV 核在 `SyncAll()` 后检查 `fdRunInfo.coreEnable`，有 FD 负载则执行 `ProcessFlashDecode`。
5. **传入 sink 给 `ComputeScaleValue_8_FD`**：`learnableSinkFlag` 从硬编码 `false` 改为 `this->isSinks`，`dummySinkUb` 重命名为 `sinkUb`。
6. **max/sum 预广播优化**：Vec2 写 staging 时先 Brcb 广播为 8-float/行块布局再写 GM，FD 读回时按 chunk 直接 DataCopy 到 `blockMaxUb`/`blockSumUb`，省掉 UB 内的 Brcb 和紧凑中间 buffer。`MAX_SUM_BYTES` 从 `64*4` 改为 `64*8*4`。
7. **`ProcessFlashDecode` 内部变量外提**：`maxGm`/`sumGm`/`stagingOutGm`/`partialOOut`/`partialOFp32` 提到循环外初始化一次，循环内用偏移访问。删除 `chunkAccumO` 中间变量，直接用 `accumulatedO[startRow * dTemplateAlign64]`。
8. **命名优化**：`S2_SPLIT_STAGING_OUTPUT_BYTES` → `S2_SPLIT_STAGING_ATTEN_OUT_BYTES`，`S2_SPLIT_STAGING_FLOAT_BYTES` → `S2_SPLIT_STAGING_MAX_SUM_BYTES`，局部变量 `OUTPUT_BYTES` → `ATTEN_OUT_BYTES`，`FLOAT_BYTES` → `MAX_SUM_BYTES`。
9. **注释完善**：`attentionOutOffset` 每行加行尾注释，`ProcessFlashDecode` 按阶段标注注释（内存准备→读 max/sum→调用 VF→reduce 累加→数据拷出），`MAX_SUM_BYTES`/`ATTEN_OUT_BYTES` 加行尾注释。
10. **staging partial output 改为 fp32**：Vec2 写端去掉 Cast（fp32→bf16），直接写 fp32 到 GM staging；FD 读端去掉 `partialOOut`（bf16）和 Cast（bf16→fp32），直接 DataCopyPad 到 `partialOFp32`（fp32）。`ATTEN_OUT_BYTES` 从 `64*512*2` 改为 `64*512*4`。省掉两次类型转换，避免中间精度损失。

### 涉及文件
- `op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_kernel.h`：主循环 FD 调用、`ProcessFlashDecode` 删除
- `op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h`：`ProcessFlashDecode` 重构、Vec2 staging 写入预广播、变量外提、命名优化、注释
- `op_kernel/arch35/util_regbase.h`：`MAX_SUM_BYTES` 大小和命名

### 工作流约定
- 每次修改完成后立即更新本文件，不再批量更新

## 2026-06-11：FD 超时修复 + V_MTE3 同步 + workspace 对齐

### 修改内容
1. **FD 超时根因修复**：`ProcessFlashDecode` step 4 中 `WaitFlag<HardEvent::V_MTE2>` 死锁。去掉 Cast 后，DataCopyPad 前没有 V pipe 操作，但遗留的 WaitFlag 仍在等待 V pipe 完成，导致事件永远不触发。修复：去掉 DataCopyPad 前的 `WaitFlag<HardEvent::V_MTE2>`，仅保留 DataCopyPad 后的 `SetFlag/WaitFlag<HardEvent::MTE2_V>` 同步 MTE2→V。
2. **Vec2 写端 V_MTE3 同步**：去掉 Cast 后补上 `SetFlag/WaitFlag<HardEvent::V_MTE3>(vToMte3Id)`，确保 V pipe（FlashUpdateNew）写完 `vec2ResUb` 后 MTE3 pipe（DataCopyPad）才开始读。
3. **Vec2 写端 srcStride 修正**：fp32 下 `(dTemplateAlign64 - dSizeV) >> 3`（而非 `>> 4`，后者是 bf16 的步长）。
4. **Host workspace 对齐**：`tiling.cpp` 中 `VEC_RES_ELEM_SIZE` 从 2(bf16) 改为 4(fp32)，新增 `FD_BLOCK_ELEM=8`，`s2SplitStagingPerCore` 计算与 kernel 端一致。

### 当前状态
- **超时已消除**，kernel 正常执行完成
- 精度 0.01%，需继续排查 FD reduce 数值正确性

### 涉及文件
- `op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h`：FD step 4 同步修复、Vec2 写端 V_MTE3 同步和 srcStride
- `op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_kernel.h`：printf 清理
- `op_host/mixed_quant_sparse_flash_mla_tiling.cpp`：workspace 大小对齐
