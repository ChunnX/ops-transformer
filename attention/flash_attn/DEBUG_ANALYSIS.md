# FlashAttention BF16 批量执行精度异常 — 交互式根因分析

## 背景

> **用户原文**：
> 现在遇到一个棘手的精度问题，使用 ops-transformer/attention/flash_attn/tests/pytests/test_flash_attn.py 执行 flash_attn 算子，会导致精度异常，出现在批量执行 ops-transformer/attention/flash_attn/tests/pytests/test_cases/functional_redline_infer.py 时，Prefill_Unquant_BF16_BF16_L0_1_5_5_1024_1024_BSND_000006 用例的精度问题，另外 000002 和 000010 也会有类似的问题，精度现象见 ops-transformer/attention/flash_attn/tests/pytests/a 文件的描述，且该现象批量执行会出现，单 case 执行不会出现，怀疑是代码里访问了什么 GM 的脏数据，或者同步存在问题导致，详细分析 ops-transformer/attention/flash_attn 的代码找出原因。

受影响用例及关键特征：

- `Prefill_Unquant_BF16_BF16_L0_1_5_5_1024_1024_BSND_000006` — FailRatio 1.63%, MaxAbsErr 0.0625
- `Prefill_Unquant_BF16_BF16_L0_1_8_1_1024_1024_BNSD_000002` — 类似精度异常
- `Prefill_Unquant_BF16_BF16_L0_2_5_1_1024_1024_TND_000010` — 类似精度异常

三点共同特征：BF16 精度、S1=S2=1024、D=128、casual mask（mask_mode=3）、return_softmax_lse=1。

**核心特征：单 case 执行精度正常，批量执行精度异常。**

批量执行的主循环如下（`test_flash_attn.py`）：

```python
for name in run_cases:
    if NPU_AVAILABLE:
        torch.npu.synchronize()
        torch.npu.empty_cache()     # 释放缓存 GM 但不清零
    passed = call_flash_attn(name, ...)
```

`empty_cache()` 将内存归还物理池，但不对 GM 做清零。下一个 case 的 workspace 可能复用同一块物理内存，携带前一个 kernel 的残留数据。这是"单 case 正常、批量异常"的底层环境前提。

---

## 第一轮：代码结构初探

> **助手**：先全面阅读 `flash_attn` 目录下的所有源文件（排除 tests/），梳理整体代码架构与核心数据流。

### 整体分层

```
op_api/                              → 对外 API 层（aclnn / L0 算子包装）
op_host/                             → CPU 侧：shape 推导、参数校验、tiling 策略
  ├── checkers/                      → 参数合法性检查（layout/dtype/shape/mask）
  └── arch35/flash_attn_tiling.cpp  → 主 tiling：切分策略、workspace 计算
op_kernel/                           → 设备侧（Ascend910B/950 AIC+AIV 混算）
  ├── flash_attn.cpp                 → kernel 入口（REGISTER_TILING_DEFAULT）
  ├── arch35/
  │   ├── flash_attn_entry_regbase.h → tiling key → 模板参数推导 & 实例化
  │   ├── flash_attn_kernel_base_gqa_template.h → 主 kernel 编排（Init/Process）
  │   ├── flash_attn_block_cube_noquant_gqa.h    → AIC Cube：BMM1(Q×K)、BMM2(P×V)
  │   ├── flash_attn_block_vec_noquant_gqa.h     → AIV Vector：softmax、update、输出
  │   └── flash_attention_noquant_block_vec_flashdecode.h → AIV FD：多 S2 分片规约
  └── utils/flash_attn_common_def.h  → 布局枚举、CostTable、模板 ConfigValue
flash_attn_metadata/                 → 独立算子：AICPU 上运行，预计算 FA/FD 分片
  └── op_kernel_aicpu/
      ├── flash_attn_metadata.h      → metadata 格式定义（FA/FD 元数据索引）
      └── flash_attn_metadata_aicpu.cpp → metadata 计算：任务切分 + 负载均衡
common/op_kernel/load_balance/       → 通用调度器（被 metadata 算子调用）
  └── section_stream_k/
      ├── section_stream_k.h         → 对外的统一入口
      ├── section_stream_k_impl.h    → ScheduleFa / ScheduleFd / RecordFDInfo
      ├── section_stream_k_def.h     → 结果数据结构
      └── base_info.h               → IBaseInfo 接口（GetSparseMode/GetNextTokenLeftUp）
```

### Kernel 执行流程

kernel 入口 `flash_attn.cpp` 通过 4 字段 tiling key（`inOutLayoutType`, `kvLayoutType`, `hasAttenMask`, `config`）选择对应的模板实例。`config` 决定 sOuter/sInner 块大小：config=0 → (64, 128)，config=1 → (32, 256)。

设备侧主类 `FlashAttentionNoQuantGqaKernel`：

```
Init()
  ├── InitConstInfo()               → 从 tiling data 读入所有常量（B/N/S/D/layout...）
  ├── InitMMResBuf(workspace)       → AIC：分配 BMM2 GM 三缓冲（BMM2_TOUB=false 时）
  ├── InitVecBlock(workspace)       → AIV：分配 FD workspace
  │     ├── vec2ResGm[3]           → splitD 时的 BMM2 中间结果三缓冲（GM）
  │     ├── accumOutGm             → FD 累加输出（GM workspace）
  │     ├── softmaxFDSumGm         → FD softmax sum（GM workspace）
  │     └── softmaxFDMaxGm         → FD softmax max（GM workspace）
  └── ClearOutput()                 → 只清零 attentionOutGm & softmaxLseGm
                                    ⚠️ accumOutGm / softmaxFDSumGm / softmaxFDMaxGm 未清零！

Process()
  for each section（由 metadata 分片决定节数）：
    FlashAttention(section)         → AIC: IterateBmm1 / IterateBmm2
                                    → AIV: ProcessVec1 (softmax scale/exp/sum/max)
                                            ProcessVec2 (FlashUpdate + CopyOut)
      └── 当 isFd && isS2SplitCore:
           CopyOutAttentionOut → Bmm2ResForFDCopyOut()
             → 写 accumOutGm（workspace）
           SoftmaxDataCopyOut → ComputeLogSumExpAndCopyToGm()
             → 写 softmaxFDSumGm / softmaxFDMaxGm（workspace）
    FlashDecode(section)            → AIV only
      ├── isFd? → 读 FA fase 写入的 workspace
      ├── CopyLseIn()              → 读 softmaxFDSumGm / softmaxFDMaxGm
      ├── CopyAccumOutIn()         → 读 accumOutGm
      ├── ComputeScaleValue_8_VF_FD()
      │     ├── 全局 max/sum 跨所有 S2 分片
      │     ├── 每分片 scale = sum_i * exp(max_i - globalMax) / globalSum
      │     └── 写 fdLseExpBuf
      ├── ReduceFinalRes_VF()      → 规约：Σ(scale_i × accum_i)
      │     ├── splitKVIndex=0: Duplicate(0) + Mul + Add
      │     └── splitKVIndex>0: Load(dst) + Mul + Add
      └── CopyFinalResOut()        → Cast(FP32→BF16) + 写 attentionOutGm
```

FA 和 FD 之间通过 `SyncAll()` 同步，确保所有 core 的 FA 任务完成后再执行 FD。

### 数据流关键点

1. **workspace 分配**：host 侧 `CalcWorkspaceSize()` 根据 `mBaseSize` 和 `dSizeV` 计算 workspace 总大小，通过 `SetWorkspaceSize()` 告知框架。kernel 侧通过 `GetUserWorkspace()` 跳过系统头部，然后按顺序切分：
   ```
   系统头部 → BMM2 GM 缓冲（三缓冲 × coreNum）→ FD 累加输出 → FD max → FD sum
   ```
   AIC 只消费 BMM2 部分（`InitMMResBuf` 中 `workspace` 指针前移），AIV 继续消费 FD 三块（`InitVecBlock` 中设置三个 GlobalTensor）。

2. **FA 写入 workspace 的条件**：只有当 `isFd == true` 且 `isS2SplitCore == true` 时，FA 才经由 `Bmm2ResForFDCopyOut` 和 `ComputeLogSumExpAndCopyToGm` 写 workspace。其余情况直接写 `attentionOutGm`。

3. **FD 读取 workspace 的条件**：metadata 中 `isFd = (usedVecNum > 0)`，且每个 AIV core 的 `fdCoreEnable = (fd.mLen > 0)` 才执行 `FlashDecode`。

4. **metadata 算子**：AICPU 侧独立执行，接收 Q/KV shape、mask 参数，调用 `SectionStreamK::Compute` 做负载均衡，生成 FA 元数据（每 AIC core 的 bN2/m/s2 起止范围、workspace idx）和 FD 元数据（每 AIV core 的 regrade 任务信息）。

---

## 第二轮：发现 `flashDecodeFlag_` 无条件设置为 true

> **助手**：在 `op_host/arch35/flash_attn_tiling.cpp:192-194` 发现：
>
> ```cpp
> void FlashAttnTilingImpl::SplitPolicy() {
>     ...
>     if (false) {                              // ← 死分支
>         SplitOutSeq();                        // TODO S1外切，待适配
>     } else {
>         CalcNumBlocks(platformInfo_.aicNum);
>         flashDecodeFlag_ = true;              // ← 始终为 true，含 Prefill
>     }
> }
> ```
>
> `flashDecodeFlag_` 一旦为 true，`CalcWorkspaceSize()` 就无条件计算 FD workspace 尺寸并填入 tiling data。对应的 kernel `InitVecBlock` 中 `if (this->isFd)` 分支根据 metadata 中的 `isFd` 决定是否把这些指针真正挂上 GM buffer。若 `isFd = true`，三个未清零的 GlobalTensor 就会指向 workspace 中可能包含前一个 kernel 残留的物理内存区域。

---

## 第三轮：发现 metadata 算子标志硬编码

> **助手**：进一步发现 `flash_attn_metadata/op_kernel_aicpu/flash_attn_metadata_aicpu.cpp:210-211`：
>
> ```cpp
> baseInfo.attenMaskFlag = true;   // 始终 true，忽略 mask_mode
> baseInfo.sparseMode = 0;         // 始终 0，忽略 mask_mode（0=NO_MASK, 3=CAUSAL）
> ```
>
> 这两个字段注入调度器的 `baseInfo`，经由虚函数接口被 `SectionStreamK` 使用：
>
> `base_info.h:133-142` — `GetSparseMode()`：
> ```cpp
> if (!attenMaskFlag) return SparseMode::BUTT;   // 无 mask → 全量 S2 直通
> if (sparseMode > (uint32_t)SparseMode::BUTT) return SparseMode::BUTT;
> return static_cast<SparseMode>(sparseMode);     // 当前始终返回 DEFAULT_MASK(0)
> ```
>
> `base_info.h:156-170` — `GetNextTokenLeftUp()`：
> ```cpp
> case SparseMode::RIGHT_DOWN_CAUSAL:   // mask_mode=3 应走这个分支
>     return kvSeq - querySeq;           // 实际因 sparseMode=0，走 default 返回 nextToken=INT_MAX
> ```
>
> 这导致调度器在 `CalcS2Range()` 中按全量 attention（所有 S2 位置对所有 S1 位置有效）计算 S2 起止范围，而 kernel 在 `CalcCurS2StartEndWithSparse()` 中按实际的 `sparseMode=3`（RIGHT_DOWN_CAUSAL）计算因果范围。**FA 分片和 FD 规划建立在错误的 S2 范围前提下。**

> **用户**：会修改哪个文件？

> **助手**：只改一个文件 — `attention/flash_attn_metadata/op_kernel_aicpu/flash_attn_metadata_aicpu.cpp:210-211`：
>
> ```cpp
> // 修改前
> baseInfo.attenMaskFlag = true;
> baseInfo.sparseMode = 0;
>
> // 修改后
> baseInfo.attenMaskFlag = (maskMode_ != 0);
> baseInfo.sparseMode = static_cast<uint32_t>(maskMode_);
> ```
>
> mask_mode=0 → `attenMaskFlag=false` → `GetSparseMode()` 返回 `BUTT` → 无 mask 路径；mask_mode=3 → `attenMaskFlag=true, sparseMode=3` → `GetSparseMode()` 返回 `RIGHT_DOWN_CAUSAL` → 调度器 `GetNextTokenLeftUp()` 与 kernel 一致的因果逻辑。

> **用户**：不用，我本地在另一个地方修改尝试下。

---

## 第四轮：修复 1 未解决，怀疑 FD 处理逻辑

> **用户原文**：
> 这个改动没能解决问题，并且我打印了一些数据出来看 `ops-transformer/attention/flash_attn/op_kernel/arch35/flash_attn_block_vec_noquant_gqa.h` 中 `BroadCastAndCopyOut` 函数里的 `softmaxFDSumGm` 和 `softmaxFDMaxGm` 都是对的，但最终结果 lse 和 attn_out 都存在精度问题，高度怀疑是 fd 的问题，帮我仔细看下 `ops-transformer/attention/flash_attn/op_kernel/arch35/flash_attention_noquant_block_vec_flashdecode.h` 和 `ops-transformer/attention/flash_attn/op_kernel/arch35/flash_attn_kernel_base_gqa_template.h`，看下可能是什么问题。

> **助手**：重点对比了 FA 写入 workspace 和 FD 读取 workspace 的偏移计算：
>
> **FA 写入**（`flash_attn_block_vec_noquant_gqa.h:948-966`）：
> ```cpp
> void Bmm2ResForFDCopyOut(const RunInfoX &runInfo, ...) {
>     uint64_t gmOffset = runInfo.faTmpOutWsPos * mBaseSize * constInfo.dSizeV
>                       + (runInfo.vecMbaseIdx + mStartVec) * constInfo.dSizeV;
>     DataCopyPad(accumOutGm[gmOffset], vec2ResUb, dataCopyParams);
> }
> ```
> `faTmpOutWsPos` 来自 metadata 的 `firstFdDataWorkspaceIdx[coreIdx]`，在当前 core 的任务中可能因 `headS2Split + tailS2Split` 而 +1。FA 最多写 1~2 个 workspace slot。
>
> **FD 读取**（`flash_attention_noquant_block_vec_flashdecode.h:256-273`）：
> ```cpp
> void CopyAccumOutIn(LocalTensor<T> &accumOutLocal, uint32_t splitKVIndex,
>                     uint32_t startRow, uint32_t dealRowCount) {
>     uint64_t offset = startRow * constInfo.dSizeV
>                     + splitKVIndex * s1BaseSize * constInfo.dSizeV;
>     DataCopyPad(accumOutLocal, accumOutGm[offset], copyInParams, copyInPadParams);
> }
> ```
> 调用处传入 `taskOffset + startRow` 作为 `startRow` 参数，`taskOffset = fdWorkspaceIdx * s1BaseSize`。
>
> FD 主循环（`FlashDecode:552`）按 `actualCombineLoopSize = fdS2SplitNum` 遍历所有 S2 分片：
> ```cpp
> for (uint32_t i = 0; i < taskInfo.actualCombineLoopSize; i++) {
>     CopyAccumOutIn(mm2Res, i, taskOffset + startRow, actualGSplitSize);
>     ReduceFinalRes(reduceOut, mm2Res, lseExp, i, actualGSplitSize);
> }
> ```
>
> 关键对比：
>
> | | FA 写 | FD 读 |
> |---|---|---|
> | workspace slot | `faTmpOutWsPos`（经 headS2Split/tailS2Split 可达 W 或 W+1） | `fdWorkspaceIdx + splitKVIndex`（i 从 0 到 `fdS2SplitNum-1`） |
> | row 偏移 | `vecMbaseIdx + mStartVec` | `taskOffset + startRow`（含 `fd.mStart`） |
>
> FA 最多写 2 个 slot，FD 按 `fdS2SplitNum` 个 slot 循环读取。**如果 `fdS2SplitNum > FA 实际写入的 slot 数，FD 读取从未写入的 workspace 区域。**

> **用户原文**：
> 建议在哪里加，另外提供一个信息，`CopyLseIn` 时 `taskInfo.actualCombineLoopSize` 是 2，但精度正常时第二轮的 `lseSumFdGm[combineLseOffset + i * combineLoopOffset]` 和 `max` 都是全 0 值。

> **助手**：这就确认了核心机制。**`fdS2SplitNum=2`，但 FA 只写了 1 个 slot 的数据。** 单 case 第二槽全零——FD 规约时 `max(global_max, 0) = global_max`，`global_sum += 0`，scale_i = 0，accum × 0 = 0，不贡献最终结果，输出正确。批量执行第二槽是前一个 kernel 残留下的非零值，`ComputeScaleValue` 和 `ReduceFinalRes` 混入错误数据。
>
> 建议在三个位置加 `AscendC::PRINTF` 来打印 FA 写 / FD 元数据 / FD 读的实际偏移。

> **用户**：加一下，不需要 pipe->，直接 PRINTF 就可以了。

> **助手**：已在以下三个位置添加打印：

| 位置 | 文件 | 打印内容 |
|------|------|---------|
| FD 入口 | `flash_attention_noquant_block_vec_flashdecode.h:495` | fdWorkspaceIdx, fdS2SplitNum, fd.mStart, fd.mLen |
| FA 写入 | `flash_attn_block_vec_noquant_gqa.h:959` | faTmpOutWsPos, vecMbaseIdx, mDealSize, gmOffset |
| FD 读取 | `flash_attention_noquant_block_vec_flashdecode.h:272` | splitKVIndex, startRow, offset |

---

## 第五轮：从打印输出确认根因

> **用户**：输出是这样的
>
> ```
> FA_WR: aiv=0 n2=0 wsPos=0 vecMbase=0 mStart=0 mDeal=64 gmOff=0
> FD: aiv=0 b=0 n2=0 mStart=0 mLen=64 wsIdx=0 s2SplitNum=2
> FD_RD: aiv=0 splitKV=0 startRow=0 offset=0
> FD_RD: aiv=0 splitKV=1 startRow=0 offset=16384
> FD_RD: aiv=0 splitKV=0 startRow=8 offset=1024
> FD_RD: aiv=0 splitKV=1 startRow=8 offset=17408
> FD_RD: aiv=0 splitKV=0 startRow=16 offset=2048
> FD_RD: aiv=0 splitKV=1 startRow=16 offset=18432
> FD_RD: aiv=0 splitKV=0 startRow=24 offset=3072
> FD_RD: aiv=0 splitKV=1 startRow=24 offset=19456
> FD_RD: aiv=0 splitKV=0 startRow=32 offset=4096
> FD_RD: aiv=0 splitKV=1 startRow=32 offset=20480
> FD_RD: aiv=0 splitKV=0 startRow=40 offset=5120
> FD_RD: aiv=0 splitKV=1 startRow=40 offset=21504
> FD_RD: aiv=0 splitKV=0 startRow=48 offset=6144
> FD_RD: aiv=0 splitKV=1 startRow=48 offset=22528
> FD_RD: aiv=0 splitKV=0 startRow=56 offset=7168
> FD_RD: aiv=0 splitKV=1 startRow=56 offset=23552
> ```

> **助手**：先确认 FA/FD 使用的块大小是否一致。通过 offset 反推：
>
> FD splitKV=1 时 `offset = startRow * dSizeV + splitKV * s1BaseSize * dSizeV`。当 startRow=0，dSizeV=128，offset=16384 → `s1BaseSize = 16384 / 128 = 128`。
>
> FA 侧 `gmOff = wsPos * mBaseSize * dSizeV + ...`，wsPos=0 → 确认 `mBaseSize = 128`。
>
> 两边都等于 128（来自模板 ConfigValue[0].s1=Aligned128），不存在两边块大小不一致的问题。问题就是 FD 多读了一个 slot：
>
> | | Slot 0（splitKV=0） | Slot 1（splitKV=1） |
> |---|---|---|
> | FA 写入 | `gmOff=0`，写 64 行（dSizeV=128，float32，mDeal=64）→ 覆盖 `accumOutGm[0..8191]` | **无任何 FA_WR** |
> | FD 读取 | offset=0→7168，8 块×8 行（正确） | offset=16384→23552（**读空**） |
> | 批量执行 | 读到正确 FA 数据 | 读到前一个 kernel 残留脏数据 |
> | 单 case | 读到正确 FA 数据 | 读到全 0（框架新分配内存） |

---

## 第六轮：追踪调度器中 `fdS2SplitNum` 的来源

> **助手**：`fdS2SplitNum` 来自 FD 元数据。在 `section_stream_k_impl.h:688`：
>
> ```cpp
> result.s2SplitNum[result.fdTaskNum] = assignContext.curKvSplitPart;
> ```
>
> `curKvSplitPart` 在 `ScheduleFa()` 主循环中累积（line 562-613）：
>
> ```cpp
> for (uint32_t i = 0; i < faConfig.coreNum; ++i) {
>     // 记录本 core 的 firstFdDataWorkspaceIdx
>     result.firstFdDataWorkspaceIdx[i] =
>         assignContext.preFdDataNum + assignContext.curKvSplitPart - 1U;
>
>     // 分配工作量（AssignByBatch / AssignByRow / AssignByBlock）
>     ...
>
>     // key line：curS2Idx 在有效范围则累加
>     if (assignContext.curS2Idx > assignContext.s1GCache.s2Start &&
>         assignContext.curS2Idx <= assignContext.s1GCache.s2End) {
>         assignContext.curKvSplitPart++;        // 跨核累积且不释放
>     }
>
>     // 只有触发 RecordFDInfo 时才重置
>     if (faConfig.FdOn && IsNeedRecordFDInfo(assignContext, result)) {
>         RecordFDInfo(computeContext, assignContext, result);
>         assignContext.preFdDataNum += assignContext.curKvSplitPart;
>         assignContext.curKvSplitPart = 1U;     // 重置
>     }
> }
> ```
>
> `IsNeedRecordFDInfo` 对 core 0 直接返回 false（`curCoreIdx == 0`）。因此 core 0 的累加值不会被重置，传递到 core 1。core 1 触发 RecordFDInfo 时，`s2SplitNum = curKvSplitPart` 拿到了跨核累积值。
>
> 对于 mask_mode=3（causal），当某个 S1G 块的 S2 有效范围只有 1 个 block 时（s2Start=0, s2End=1，表示索引 [0, 1) 即 1 个 block），core 0 完整处理后：
> - `curS2Idx = 1 = s2End`
> - `<=` 条件命中 → `curKvSplitPart` 从 1 变成 2
> - 但 core 0 实际上完整处理了这一块，**不存在 S2 跨核切分**，不应累加

---

## 第七轮：最终确认

> **用户原文**：
> 最终确认发现是 `section_stream_k_impl.h` 当中算 s2range 时多算了一块，这样在 mask mode 是 3 的情况下，比如 m s2 为 0，0 时实际不需要做 fd，但负载均衡里认为 0，1 也是要算的，这样就认为要做 fd。

> **助手**：完全对得上。修复是将 `<=` 改为 `<`：
>
> ```cpp
> // section_stream_k_impl.h:609-610
> if (assignContext.curS2Idx > assignContext.s1GCache.s2Start &&
>     assignContext.curS2Idx < assignContext.s1GCache.s2End) {    // ≤ → <
>     assignContext.curKvSplitPart++;
> }
> ```
>
> 语义：只有当 core 的 curS2Idx **严格在** s2Start 和 s2End 之间（即 S2 被跨核截断、未完整处理）时，才累加 curKvSplitPart。完整处理一整块时不触发。

> **用户**：不用修改。

---

## 完整因果链

```
第一层：metadata 硬编码 attenMaskFlag=true, sparseMode=0
  → 调度器 `GetSparseMode()` 返回 DEFAULT_MASK → 未走无 mask 直通路径
    → `GetNextTokenLeftUp()` 返回 INT_MAX（而非因果的 S2-S1）
      → `CalcS2Range()` 按全量 attention 计算 s2Start/s2End
        → causal mask 下 S2 有效范围只有 1 个 block 时，s2End=1

第二层：curKvSplitPart ≤ 判据缺陷
  → core 0 完整处理完这 1 个 S2 block 后 curS2Idx=1=s2End
    → ≤ 条件错误命中 → curKvSplitPart 从 1 变成 2
      → core 0 不触发 RecordFDInfo → curKvSplitPart 不重置
        → core 1 触发 RecordFDInfo 时 s2SplitNum=curKvSplitPart=2

第三层：FD 多读 workspace slot
  → FD 读取 slot 0（FA 写了数据 → 正确）和 slot 1（FA 未写）
    → 单 case：slot 1 全零 → ComputeScaleValue/ReduceFinalRes 不产生额外贡献
    → 批量执行：slot 1 残留前一个 kernel 数据 → 非零值污染 global_max/sum
      → LSE 和 attention_out 双双异常
```

## 涉及的修复点

| # | 修改 | 文件 | 行号 | 性质 |
|---|------|------|------|------|
| 1 | `attenMaskFlag`/`sparseMode` 根据 `maskMode_` 设置 | `flash_attn_metadata/op_kernel_aicpu/flash_attn_metadata_aicpu.cpp` | 210-211 | 同步调度器与 kernel 的 mask 语义 |
| 2 | `<=` 改为 `<` | `common/op_kernel/load_balance/section_stream_k/section_stream_k_impl.h` | 610 | 防止完整处理 S2 范围时 curKvSplitPart 错误累加 |

## 关键文件索引

| 文件 | 作用 |
|------|------|
| `flash_attn_metadata/op_kernel_aicpu/flash_attn_metadata_aicpu.cpp:210-211` | metadata 标志硬编码（修复 1） |
| `common/op_kernel/load_balance/section_stream_k/section_stream_k_impl.h:688` | `s2SplitNum` 从 `curKvSplitPart` 取值 |
| `common/op_kernel/load_balance/section_stream_k/section_stream_k_impl.h:609-612` | `curKvSplitPart` 累加与 `≤` 判据（修复 2） |
| `common/op_kernel/load_balance/section_stream_k/section_stream_k_impl.h:648-666` | `IsNeedRecordFDInfo()` → core 0 永远返回 false |
| `common/op_kernel/load_balance/base_info.h:133-170` | `GetSparseMode()` / `GetNextTokenLeftUp()` 接口逻辑 |
| `op_host/arch35/flash_attn_tiling.cpp:193` | `flashDecodeFlag_` 无条件 true |
| `op_kernel/arch35/flash_attn_block_vec_noquant_gqa.h:271-282` | `ClearOutput()` → FD workspace 未清零 |
| `op_kernel/arch35/flash_attn_block_vec_noquant_gqa.h:948-967` | FA `Bmm2ResForFDCopyOut` 写 workspace |
| `op_kernel/arch35/flash_attn_block_vec_noquant_gqa.h:801-813` | `CopyOutAttentionOut` → isFd 时选 FD 路径 |
| `op_kernel/arch35/flash_attn_block_vec_noquant_gqa.h:396-417` | `SoftmaxDataCopyOut` → isFd/isS2SplitCore 分派 |
| `op_kernel/arch35/flash_attention_noquant_block_vec_flashdecode.h:256-274` | FD `CopyAccumOutIn` 读 accumOutGm |
| `op_kernel/arch35/flash_attention_noquant_block_vec_flashdecode.h:472-567` | FD 主循环 `FlashDecode()` |
| `op_kernel/arch35/flash_attn_entry_regbase.h:44-104` | tiling key → 模板实例化 / ConfigValue |
