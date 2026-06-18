# MixedQuantSparseFlashMla Work Progress

## Current Session

- 2026-06-08：已按要求清空旧版 `work_progress.md`，后续工作进展从本文件重新维护。
- 2026-06-08：最终目标是为 QSMLA 算子实现 flash decode 功能。
- 2026-06-15：按新约定，工作进展和学习进展均维护在当前算子目录。
- 2026-06-15：实现 FD N=128 staging 适配：split-G 下 staging 逻辑 M 行数改为 128，FA 写 staging 时后 64 head 增加 64 行偏移，staging logical slot 数按 cube pair 减半以保持 workspace 总量不增加；FD 阶段不引入前/后 64 head 判断，仅按 metadata 读取逻辑 M 行。
- 2026-06-16：修复 `ComputeS1LoopInfo()` 最后 BN 的 `gs1LoopEndIdx` 判断：新增 `s2EndIdx` 入参，并将 `FA_M_END_INDEX` 局部命名改为 `gS1EndIdx`；只有 `gS1EndIdx == 0 && s2EndIdx == 0` 时才回退到当前 batch 真实 S1 结尾，避免 S2 切分场景 `M_END=0, S2_END!=0` 中“当前 batch 的 M=0”被误判为虚拟下一个 batch 的 0。
- 2026-06-16：检查 `ProcessFlashDecode` 偶现精度/点位不固定问题，补充 FD chunk 间 `blockMaxUb/blockSumUb` 复用同步：下一 chunk GM->UB 读 max/sum 前等待上一 chunk 最后一次 `ReduceFinalRes` 的 V pipe 完成，避免 MTE2 抢先覆写 `blockSumUb`。
- 2026-06-16：将 `ProcessFlashDecode` 中验证用的 `PipeBarrier<PIPE_ALL>()` 替换为精确 `PipeBarrier<PIPE_V>()`：`WaitFlag<MTE2_V>` 后进入 `ReduceFinalRes` 前同步 V pipe，`ReduceFinalRes` 后、`SetFlag<V_MTE2>` 前再次同步，确保 split 间复用 `partialOFp32`/`chunkAccumO` 时不会读写未完成。
- 2026-06-16：确认 `s2SplitIdxCounter` 当前只支持每个 FA staging 起点最多 2 个 S2 split；若一个被切开的 S2 在同一核/同一行对应超过 2 个 split，`firstFdDataWorkspaceIdx + s2SplitIdx` 写 staging 会复用或越过 slot，属于需要 host metadata/workspace 共同约束或扩展的风险点。
- 2026-06-17：收窄 `s2SplitIdxCounter` 推进条件：仅当 `gS1Index == runParam.gs1LoopEndIdx - 1` 且发生 S2 split 时递增，避免非当前核 S1 循环末端的任务推进 split slot。
- 2026-06-17：确认 `s2SplitIdx` 实际只在 split task 的末轮 S2 loop 参与 staging 地址计算；普通中间 S2 基本块虽携带该字段，但不会用它写 FD staging。
- 2026-06-18：适配 N<64 且 N 为 2 的幂次方的 FD staging：确认当前 `sparse_flash_mla_metadata` 在 Ascend950 下 `mBaseSize_=groupSize_`，kernel 非 split-G 的 FD staging 行数从固定 64 改为运行时 `gSize`，host workspace 分配同步改为按 `gSize` 计算，N=128 split-G 继续保持 128 行逻辑 M 布局；新增 `N32_single_batch_3_cores` 单例参数用于覆盖该路径。

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

## 2026-06-12：lseExpUb dump 大量 0 问题定位与尝试修复

### 问题分析
- 用户发现 `lseExpUb` dump 出大量 0，只有少数位置有非零值（如每隔 64 个元素才有一个有效值）
- 根因定位：`vf_flash_decode.h` 中 `ComputeScaleValue_8_VF_FD` 函数 stride 参数有 bug
  - `blockStride = 0x1`, `repeatStride = dealRowCount = 8`
  - 在 `DATA_BLOCK_COPY` + `POST_MODE_UPDATE` 模式下，`repeatStride` 以数据块为单位（每块 8 float）
  - 当 `actualCombineLoopSize > 1` 时，写入位置分散：
    - split 0: 位置 0, 64, 128, 192...
    - split 1: 位置 8, 72, 136, 200...
  - 期望布局应为连续：split 0 写入 0-63，split 1 写入 64-127...
- 参考代码 `flash_attention_score_antiquant_block_vec.h` 使用同样的 VF 函数（`ComputeScaleValue_VF` 底层调用 `ComputeScaleValue_8`，stride 参数相同），但能正常工作
- 关键差异：参考代码使用 `DataCopyPad` + `dstStride = 0` 连续读取 LSE，而非循环写入

### 尝试方案

#### 方案 1：修改 LSE 读取方式（参考 antiquant）
- 使用 `DataCopyPad` + `blockCount = workspaceNum` + `dstStride = 0` 连续读取
- 调用 `ComputeScaleValue_VF` 和 `ReduceFinalRes_const_VF`
- 结果：精度 0.01%，失败。`srcStride` 计算可能有误。

#### 方案 2：手动实现 FD 归约（不依赖 VF stride）
- 对每行计算：
  1. 全局 max = max(split_max[0..N])
  2. globalSum = sum(exp(split_max[i] - globalMax) * split_sum[i])
  3. scale[i] = exp(split_max[i] - globalMax) * split_sum[i] / globalSum
- 使用 `GetValue`/`SetValue` 读取/写入数据（效率低，不适合 Ascend C）
- 结果：精度 0.02%，微小改善但失败。

#### 方案 3：手动实现 + 使用 Duplicate
- 利用 staging 数据已 broadcast（每行 8 个 float 相同），只读取第一个元素
- 使用 `Duplicate` broadcast 写入 scale 值
- 结果：精度 0.02%，失败。

### 当前状态
- **精度未调通**：测试精度 0.02%，失败
- **FD 可能未触发**：printf 输出未显示，`fdRunInfo.coreEnable` 从 metadata 读取，可能 FD 未启用
- **大量 RealOut = 0**：测试输出中大量位置为 0，算子整体流程可能有问题

### 待解决问题
1. **确认 FD 是否被正确触发**：检查 metadata 中 FD 相关参数是否正确设置
2. **stride 参数修复**：需要修改 `vf_flash_decode.h` 中的 stride 参数（`repeatStride` 应为 `0x1`），但用户不允许修改 common 代码
3. **算子整体流程排查**：大量输出为 0，需排查 FA 主循环是否正确执行

### 涉及文件
- `op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h`：ProcessFlashDecode 多次尝试修改
- `attention/common/op_kernel/arch35/vf/vf_flash_decode.h`：stride 参数 bug（不允许修改）
- `attention/common/op_kernel/arch35/flash_attention_score_antiquant_block_vec.h`：参考实现

## 2026-06-12：FD 精度从 0.01% 调通至 PASS（完整定位链）

本轮以参考算子 `flash_attention_score_antiquant_block_vec.h` 为基准，逐项对齐 QSMLA 的 FD 实现，最终 `single_batch_2_cores` 精度 PASS。按发现顺序记录根因与修复。

### case 关键事实（先澄清，避免误判）
- `single_batch_2_cores`：BSND/PA_BBND, B=1, S1=1, S2=512, N1=64(gSize), N2=1, CFA, cmp_ratio=4, 有 sinks。
- S2=512 实际只算 256 token（ori 选 128 + cmp 选 128），均分到 2 个 cube 核 → 每核 vec1 **单轮 S2 循环**（`s2LoopLimit=0`），对应 FD 的 2 个 split。
- FD 阶段：64 行（gSize=64）规约任务被切到 **64 个 AIV 核**，每核 `mStartIdx=核号, mNum=1`，各算 1 行。

### 根因 1：`expf` 在 Ascend C 内核不可用
- 现象：`splitCorrection[i] = expf(...)` 编译报 `expf undeclared`。
- 处理：放弃标量 `expf`/手写归约，改用参考的 VF 微内核 `ComputeScaleValue_VF` + `ReduceFinalRes_const_VF`（`vf_flash_decode.h`），全程寄存器内 `MicroAPI::Exp/Max` 完成 online-softmax 合并。

### 根因 2：staging max/sum 的 `DataCopyExtParams` 参数写反
- 现象：FD 读回的 max/sum「每行 8 个广播值不一致」「sum 有的 1.0 有的不是」。
- 根因：`DataCopyExtParams` 构造顺序是 `{blockCount, blockLen, srcStride, dstStride, rsv}`，原代码把 blockCount 和 blockLen 写反，导致越界读 + 槽内布局被打乱。
- 修复：改为 `{blockCount=行数, blockLen=FD_BLOCK_ELEM*sizeof(float)}`，与 attentOut 同构连续写。

### 确认点：广播时机 = 拷出前（与参考一致）
- 参考 `ComputeLogSumExpAndCopyToGm`：先 `BroadcastMaxSum` 再 `DataCopy` 到 GM。所以 GM staging 存的就是已广播 `[split][row][8]` 布局，FD 读回直接用。

### 根因 3：max/sum 拷出时机错误（应在 vec1 而非 vec2）
- 参考在 **vec1**（softmax 算完即拷出），避免 `sumUb`/`maxUb` 被后续 task 复用覆盖。
- 修复：新增 `CopyMaxSumToStaging`，在 vec1 末轮（`isS2Split && s2LoopCount==s2LoopLimit`）调用。行数用 vec1 的 `halfMRealSize`；Brcb 暂存复用 vec1 空闲的 `dequantScaleBuff`（8192B，max/sum 各占 1024B 偏移分开）。vec2 只保留 attentOut 拷出。

### 根因 4：`DEAL_ROW_COUNT` 与 buffer 容量
- `DEAL_ROW_COUNT` 8→16 时 `fdPartialBuf` 只按 8 行分配会越界，已同步扩到 16 行（`16*dTemplateAlign64*sizeof(T)`）。`MAX_S2_SPLIT_NUM=2`，blockMax/SumUb、lseExpUb 容量校验通过。

### 关于 vec1 的 sum/max 更新机制（排查 sum 全 1.0 的误判）
- `ProcessVec1Vf` 的 Update 路径（`ProcessVec1UpdateImpl128`）**不直接写** `maxUb`/`sumUb`，只把局部 max/sum 写到 `apiTmpBuffer`。
- 真正合并写回 `maxUb`/`sumUb` 的是随后的 `SFAUpdateExpSumAndExpMax`（从 `apiTmpBuffer` 读局部值与 in_max/in_sum 合并）。
- 所以「`SFAUpdateExpSumAndExpMax` 之前 dump 到 sum=1.0」是 sinks 初值（`DuplicateSumWithR0`），属正常；update 之后的 dump 才是最终值（非 1.0）。vec1 拷出链路无误。
- 教训：误把照搬自旧 vec2 的 `if (s2LoopCount==0) Duplicate(sumUb,1.0f)` 带进 `CopyMaxSumToStaging`，会把单轮场景真实 sum 覆盖成 1.0，已删除。

### 根因 5（精度主因）：partialO 与 scale 的「除 sum」错配
- VF `ComputeScaleValue_VF` 算出的 scale = `correction_i·sum_i/globalSum`（带 `sum_i`），这要求 partialO **已除过 sum_i**（参考路线）。
- 而 QSMLA 原本 vec2 末轮用 `FlashUpdateNew`（不除 sum），单轮更是直接 `DataCopy(P@V)`（不除 sum），导致最终多乘一个 `sum_i`。
- 修复：对齐参考路线 A——vec2 的 `isS2Split` 分支统一除 sum：
  - 多轮末轮：`FlashUpdateNew` → `FlashUpdateLastNew`（除 sum）。
  - 单轮（`s2LoopCount==0`）：拷出前补 `LastDivNew` 除 sum。
- 这样 partialO 除过 sum_i，与 VF scale 的 `sum_i` 相乘抵消，得到 `Σ correction_i·(Σexp·V)/globalSum`。

### 根因 6（致命，0.01%→81%）：FD 输出 `attentionOutOffset` 行步长错误
- 现象：输出连续后几行全 0，第一行偶尔对。
- 根因：行偏移项写成 `mStartIdx * n2GDv`，而 BSND 下 m 方向相邻行（相邻 head）内存间隔是 `dSizeV`。`n2GDv = n2*gSize*dSizeV = 64*dSizeV`，导致核 k 把结果写到正确位置的 64 倍远处，除核 0 外全部写飞，真实行保持 InitOutput 的 0。
- 修复：`mStartIdx * n2GDv` → `mStartIdx * dSizeV`。精度 0.01% → 81%。
- 备注：`mIdx * s1BaseSize * n2GDv` 这项本 case `mIdx=0` 不暴露，多 m 基本块场景需再核对其步长。

### 根因 7（致命，81%→PASS）：FD split 累加循环的同步竞态
- 现象：剩余 19% 误差**非确定性**（多次运行有时分散、有时连续），是竞态而非逻辑错误的标志。
- 根因：split 循环每轮复用同一 `partialOFp32` buffer，缺少反向 `V_MTE2` 同步——split N 的 `ReduceFinalRes`（V 读）尚未读完，split N+1 的 `DataCopyPad`（MTE2 写）已覆写。
- 修复（方案 A，加同步）：在 split 循环内，`DataCopyPad` 前 `WaitFlag<V_MTE2>`，`ReduceFinalRes` 后 `SetFlag<V_MTE2>`，保证「读完才覆写」。事件 ID 复用 `vToMte2V0Id[0]`（init 时已 SetFlag 配平）。
- 结果：精度 PASS。

### 最终修复涉及文件
- `op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h`：
  - `ProcessFlashDecode` 改用 VF 微内核；`attentionOutOffset` 行步长 `n2GDv→dSizeV`；split 循环加 `V_MTE2` 反向同步。
  - 新增 `CopyMaxSumToStaging`（vec1 末轮拷出 max/sum，Brcb 暂存复用 `dequantScaleBuff`）。
  - vec2 `isS2Split` 分支统一除 sum（多轮 `FlashUpdateLastNew`，单轮补 `LastDivNew`）。
  - `DataCopyExtParams` 参数顺序修正；`fdPartialBuf` 扩到 16 行；`DEAL_ROW_COUNT=16`。
- `op_kernel/arch35/util_regbase.h`：`MAX_SUM_BYTES` 等常量。

### 调试方法论小结（供后续复用）
- DumpTensor 直接打屏，`bash test_run.sh single` 即可看到；dump_size=16 的小张量配合每行 8 广播值判断布局是否错位。
- 「广播值不一致」→ offset/参数错位；「连续整行为 0」→ 输出偏移/核覆盖问题；「结果非确定性波动」→ 同步竞态；「每次稳定偏差」→ 逻辑/数值错配。
- 所有 DumpTensor/printf 调试代码已清理。

## 2026-06-12：FD split 循环同步方案 A vs 方案 B（双缓冲）评估

修复根因 7 时有两个候选方案，最终选 A。此处记录评估结论，供后续优化参考。

### 两方案
- **方案 A（已采用，加同步）**：单块 `fdPartialBuf`，split 循环内 `DataCopyPad` 前 `WaitFlag<V_MTE2>`、`ReduceFinalRes` 后 `SetFlag<V_MTE2>`，强制「读完才覆写」。代价：MTE2 搬入与 V 计算串行化。
- **方案 B（双缓冲 ping-pong）**：两块 `fdPartialBuf` 交替，靠 buffer 隔离消除竞态，可让 split N+1 的 MTE2 搬入与 split N 的 V 计算重叠。代价：多占 32KB UB（32KB→64KB），逼近 192KB 物理上限。

### 决定性约束：`MAX_S2_SPLIT_NUM = 2`（util_regbase.h:26）
- staging 只分配 2 个 slot，所有 case 的 `workspaceNum`（每 FD 任务 split 数）≤ 2，FD 内层 split 循环最多 2 次 → 每 chunk 最多 1 次 load/compute 重叠机会。

### 全 paramset 的 FD 负载特征
- host `SplitFD`（metadata aicpu）按 `averageLoad` 把 `fdMSize`（=g×s1_block，单 batch 为 64）行尽量摊到多核。单 batch FD 总负载小（`splitNum=2 × mSize=64=128`），`averageLoad` 极小 → `curFDVectorNum` 拉满到 64 → **64 核每核 mNum=1**。
- 各 case：`single_batch_*`、`K_1024`、`N128` 等单 batch case 均为 split≤2、mNum=1、mChunks=1；`multi_batch_full_cores`(B=8)、`1_core_has_2_reduce`(B=33) 多 FD 头，但每核单任务仍是「2 split × 极少行」。

### 结论：所有 case 切方案 B 都无实际性能收益
1. split ≤ 2 → 每 chunk 最多 1 次重叠。
2. mNum 普遍=1 → 每 split 搬入仅 `1×512` fp32 ≈ 2KB，DMA 极快。
3. mChunks=1（mNum<DEAL_ROW_COUNT=16）→ 无 chunk 间流水。
4. FD 在 `SyncAll()` 之后收尾阶段单独跑，不与主 cube/vec 流水线重叠，省下的 cycle 不影响整体吞吐。
- 反而多占 32KB UB + 增加 ping-pong 复杂度。**维持方案 A**。

### 真要做 FD 性能优化的更优方向（非本轮）
- 提高 `MAX_S2_SPLIT_NUM`：让长序列切更多 split，增大 cube 阶段并行度（量级更大的改动）。
- 调 `SplitFD` 的 `averageLoad`：让每核 mNum 更大、用满 `DEAL_ROW_COUNT=16` 的 chunk 能力，减少 FD 启用核数与调度开销。

## 2026-06-12：FD InitFDBuffers — 仿参考算子 Reset+重新分配

### 修改内容

仿照参考算子 `flash_attention_score_antiquant_block_vec.h` 的 `InitFDBuffers` 模式，将 QSMLA 的 FD 规约阶段内存准备从「复用 FA 阶段空闲 buffer」改为「先 Reset 释放再重新分配」。

1. **新增 `InitFDBuffers` 函数**（`mixed_quant_sparse_flash_mla_scfa_block_vector.h`）：
   - `tPipe->Reset()` 释放 FA 阶段所有 UB buffer（V0 KV 输入/输出、softmax、vec1/vec2 等）
   - 重新分配 FD 专用 buffer：
     | Buffer | 大小 | 用途 |
     |--------|------|------|
     | `fdAccumOutBuf` | `FD_MAX_ROW × dTemplateAlign64 × sizeof(T)` | FD 累加输出 |
     | `fdBlockMaxBuf` | `MAX_S2_SPLIT_NUM × FD_MAX_ROW × BLOCK_ELEM × sizeof(float)` | max 读入 |
     | `fdBlockSumBuf` | 同 max 大小 | sum 读入 |
     | `fdLseExpBuf` | `FD_MAX_ROW × BLOCK_ELEM × sizeof(float)` | ComputeScaleValue 输出 |
     | `fdPartialBuf` | `FD_MAX_ROW × dTemplateAlign64 × sizeof(T)` | partial O 读入 |
   - `FD_MAX_ROW = 64`，满足每次规约最多读入 64 行的容量需求
   - **不分配 `sinksBuf`**：sink 已在 vec1 阶段通过 `SinkSubExpAddGSFusedVF` 融入 `sumUb`/`maxUb`，FD 阶段 `learnableSinkFlag=false`，无需重新读取

2. **ProcessFlashDecode 改用 FD 专用 buffer**：
   - `accumulatedO`: `stage2OutBuf` → `fdAccumOutBuf`
   - `lseExpUb`: `dequantScaleBuff` → `fdLseExpBuf`
   - `blockMaxUb`: `stage0OutBuf[0]` → `fdBlockMaxBuf`
   - `blockSumUb`: `stage0OutBuf[1]` → `fdBlockSumBuf`
   - `sinkUb`: 改为空 tensor（`LocalTensor<T> sinkUb;`），`learnableSinkFlag=false` 保护不访问

3. **InitLocalBuffer 移除 `fdPartialBuf`**：不再在 FA 阶段分配 FD 专用 buffer，FD 阶段有自己的独立分配。

4. **`InitFDBuffers` 在 `ProcessFlashDecode` 开头调用**：FD 内存准备内聚到 vec 层，kernel 两处调用点保持直接调 `ProcessFlashDecode`。

5. **新增成员变量**：`fdAccumOutBuf`, `fdBlockMaxBuf`, `fdBlockSumBuf`, `fdLseExpBuf`（4 个 `TBuf<>`）

6. **Dummy 类同步**：`SCFABlockVecDummy` 新增空 `InitFDBuffers` 方法。

### 涉及文件
- `op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_block_vector.h`：`InitFDBuffers` 新增、`ProcessFlashDecode` buffer 替换、`InitLocalBuffer` 移除 fdPartialBuf、成员变量新增、Dummy 类更新
- `op_kernel/arch35/mixed_quant_sparse_flash_mla_scfa_kernel.h`：两处 `InitFDBuffers` 调用插入

### 优势
- FA 与 FD 的 UB buffer 生命周期完全分离，不再存在「FA buffer 被 FD 误改」或「FD buffer 容量受 FA 阶段剩余空间限制」的风险
- Reset 释放大量 FA 阶段不再需要的 buffer（V0 KV 输入/输出约 40KB、softmax 约 1.5KB、vec1 约 16KB 等），FD 专用 buffer 总量远小于 FA 阶段，UB 压力大幅降低
