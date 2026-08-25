---
> 交付说明：本文件是"flash_mla_with_kvcache 端到端调用链教程"的落盘底稿（规划会话只能写 .omo/）。
> 目标正式路径：`attention/flash_mla_with_kvcache/docs/zh/flash_mla_with_kvcache_调用链与内存同步教程.md`
> 落盘动作请在执行会话执行（把本文件内容复制/移动到目标路径）。
---

# flash_mla_with_kvcache 端到端调用链教程

> 从 torch extension 调用 → host Tiling → kernel 模板分发 → AIC/AIV 双核流水执行，
> 重点讲解 **kernel 的内存分配** 与 **同步机制**。
> 适用对象：刚接触本算子实现、需要给他人讲解或做二次开发的工程师。
> 代码路径：`attention/flash_mla_with_kvcache/`（host：`op_host/`，kernel：`op_kernel/`）。

---

## 0. 一图总览

```
    用户 Python
    ┌────────────────────────────────────────────────────────────────┐
    │ torch.ops.npu.flash_mla_with_kvcache(q, k_cache, ...)          │
    │   （ops/flash_mla_with_kvcache.py  TorchScript schema + impl）  │
    └──────────────┬───────────────────────────────┬─────────────────┘
                   │ 内部两步                        │
        ┌──────────▼──────────┐          ┌─────────▼──────────────────┐
        │ flash_mla_with_kvcache_metadata（前置：AICPU 任务切分）        │
        │ 产出 metadata tensor：[16字头][FA区 36核×16字/section]        │
        │                        [FD区 72核×16字/section]             │
        └──────────┬──────────┘          └─────────┬──────────────────┘
                   └──────────┬────────────────────┘
                              ▼
        ┌──────────────────────────────────────────────────────────────┐
        │ torch csrc wrapper（flash_mla_with_kvcache.cpp）              │
        │  计算输出 shape（out==layout_q, 末维 head_dim_v=512）          │
        │  ACLNN_CMD(aclnnFlashMlaWithKvcache, ...)                    │
        └──────────────┬───────────────────────────────────────────────┘
                       ▼
        ├─ host Tiling（op_host/flash_mla_with_kvcache_tiling.cpp）────┤
        │  IsCapable 门 → SplitPolicy → CalcWorkspaceSize → GenTilingKey│
        │  SetTilingKey/TilingData/WorkspaceSize/NumBlocks             │
        │  (blockDim=72; KERNEL_TYPE_MIX_AIC_1_2 → 36 AIC + 72 AIV)    │
        └──────────────┬───────────────────────────────────────────────┘
                       ▼
        ├─ kernel 入口（op_kernel/flash_mla_with_kvcache.cpp）─────────┤
        │  tiling key → 编译期模板参数（布局/Config/掩码/PA/FD）          │
        │  __DAV_C310_CUBE__ → AIC 取 Cube 实块 / AIV 取 Vec+Fd 实块     │
        │  FlashAttentionNoQuantMlaKernel::Init → Process              │
        └──────────────┬───────────────────────────────────────────────┘
                       ▼
        ├─ arch35 主逻辑（kernel_noquant_mla.h）───────────────────────┤
        │  逐 section（AICPU metadata）→ FlashAttention (FA 流水)        │
        │  + FlashDecode (FD, 长 KV 归约)                              │
        │  AIC: cube.IterateBmm1/2   AIV: vec.ProcessVec1/2           │
        └──────────────┬───────────────────────────────────────────────┘
                       ▼
        ├─ block_cube（AIC）：L1 Q/KV + L0A/B/C + Fixpipe → UB         │
        ├─ block_vec （AIV）：在线 softmax、加权累加、LSE、输出写回 GM    │
        └─ block_vec_flashdecode（AIV，FD）：accum 重读 + 跨段归约       │
```

**一句话链路**：Python 调 torch op → C++ wrapper 算好多余的 shape → aclnn 后段 host tiling 只做"门 + 透传 + key/workspace" → kernel 按 key 实例化出 36 个 AIC + 72 个 AIV 的混合任务，AIC 算两个 MatMul、AIV 做 softmax 与累加，两边用共享内存 + 事件流水协作。

---

## 1. torch extension：两层接口与 metadata 前置

### 1.1 为什么是"两个算子"？

`flash_mla_with_kvcache` 把**任务切分**从主算子中拆出来做成前置算子 `flash_mla_with_kvcache_metadata`（AICPU 执行，产出 metadata tensor）。原因：

- 切分是纯 host/CPU 逻辑（batch、序列长、block 数），与 NPU kernel 解耦；
- 主算子的 host tiling 拿到 metadata 后**跳过全部切分计算**（`metadataPassthrough_`，见 `op_host/arch35/flash_mla_with_kvcache_tiling.cpp:43`），只做"门 + 透传 + tiling key + workspace"；
- kernel 侧逐 section 读 metadata 自我驱动（见 §4.3），一个 kernel 实例覆盖所有切换方案。

### 1.2 metadata 缓冲与容量公式

`torch_extension/cann_ops_transformer/ops/csrc/flash_mla_with_kvcache.cpp:41-51`：

```cpp
Mrwords = ((36 + 72) * batchSize * 1 + 1) * 16          // kv==1 上界
bytes   = align4096(Mrwords * 4)                          // "36+72: 16字/核"
```

- `36` = AIC 核数（FA 区），`72` = AIV 核数（FD 区），每核 **16 个 int32 字**；
- `kv==1` 是 MLA 硬约束（单 latent KV 头），公式**不能**乘 num_heads_kv>1（有 static_assert 与 TORCH_CHECK 双保险）。

### 1.3 主算子 C++ wrapper（csrc）

- 入参：`q, k_cache` + 可选 `block_table/cache_seqlens/cu_seqlens_q/seqused_q/attn_mask/metadata` + 属性 `head_dim_qk(576) / head_dim_v(512) / softmax_scale / mask_mode / layout_q/kv/out`；
- 输出 shape 推导（`:108-148`）：`out==layout_q`，末维 = `head_dim_v=512`（不是 q 的 576！）；
  - TND→`(T,N,512)`；BNSD→`(B,N,S,512)`；BSND→`(B,S,N,512)`；LSE：TND→`(N,T)`、其余→`(B,N,S)`；
- 最后 `ACLNN_CMD(aclnnFlashMlaWithKvcache, ...)` 一次性发出，属性顺序必须与 def 输入序一致（用错曾致 cache_seqlens 收到 NULL 槽 → GM 越界，见 kernel 入口注释）。

---

## 2. host Tiling：门、透传、key 与 workspace

入口：`op_host/arch35/flash_mla_with_kvcache_tiling.cpp`
主流程（`DoOpTiling`, `:161-186`）：

```
SetPlatMemoryInfo         平台能力：AIV/AIC 核数、UB/L1/L0A/L0B/L0C/L2 容量
memset(&tilingData_,0)   全零初始化（tiling data 逐字节可哈希，UT 依赖）
InitImplParam            按 shape/attr 推导 seq-len 存在性（不读 GM 值）
SplitPolicy              AdjustSinnerAndSouter + metadata 透传
FillTiling               填 base/attenMask/pageAttention/workspace 等参数段
CalcScheduleMode         BATCH_MODE
CalcWorkspaceSize        workspace 公式（见 §2.3）
GenTilingKey             config=9 + 布局/kv 键位（见 §2.2）
SetNumBlocks/SetTilingKey/SetWorkspaceSize/SetTilingData
```

### 2.1 能力门（IsCapable）

- 空 Tensor、非 fp16/bf16 → 拒（`IsCapableBasicCheckMla`）；
- 非 `PAGE_ATTENTION` 存储 → 拒（`IsCapableFeatureCheckMla`）；
- mask ∈ {0,3}、qk=512+rope64、vHead=512、N2==1、N1∈[1,128]、
  q∈{TND,BNSD,BSND}、out==q、kv∈{NZ/PA_NZ,PA_BNBD,PA_BBND}（连续 KV 拒）→ 布局门（`IsCapableSparseLayoutCheckMla`, `:70-114`）。

### 2.2 tiling key 与调度

- **Config 固定 = 9**：`Config_S1Aligned64_S2Aligned128_DAligned576_DVAligned512`
  → kernel 编译期常量：`mBaseSize=64, s2BaseSize=128, dBaseSize=576(=512+64), dVBaseSize=512`；
- key 位编码（`op_kernel/arch35/flash_mla_with_kvcache_template_tiling_key.h:238-249`）：
  `key = InOutLayoutType | Config<<8 | HasAttenMask<<18 | KvLayoutType<<19 | IsFd<<21`；
- **blockDim=72**（AIV 数），`KERNEL_TYPE_MIX_AIC_1_2`：36 AIC 配 72 AIV（c:v=1:2，`CV_RATIO=2`）；
- metadata 透传路径 `flashDecodeFlag_=true` 恒真（`:272`）：FD 使能由各 section 的 `mLen>0` 运行时判定，非 FD 几何下 FD section 为空（mLen=0）自动跳过，无副作用。

### 2.3 workspace 公式（`CalcMaxWorkspaceSize`, `:140-153`）

```
workspace = defaultSysWorkspace
          + aicNum * 2 * 64 * 512 * sizeof(float)    // FA 每核 2 块预载的 attn 暂存（M=64, DV=512）
          + 2 * aicNum * 2 * 64 * 8  * sizeof(float)  // ResLse 两份（sum + max）
```

即把 flash-attention 核间归约需要的"中间 attn 输出 + LSE 双份"预留在 GM workspace 中，供 FD 段重读归约。

---

## 3. kernel 入口与模板分发

文件：`op_kernel/flash_mla_with_kvcache.cpp`

### 3.1 编译期推导（`__global__` 模板）

```cpp
__global__ __aicore__ void flash_mla_with_kvcache(
    query, key, blockTable, cacheSeqlens, cuSeqlensQ, sequsedQ, attnMask,
    metadata, attnOut, softmaxLse, workspace, tiling)   // :229-232
{
    REGISTER_TILING_DEFAULT(...);
    __gm__ user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);  // 混合 AIC+AIV 调度
    ...
    run_..._noquant_kernel<INPUT_T, OUT_T, inOutLayoutType, config, hasAttenMask, KvLayoutType, isFd>(...)
    AscendC::PipeBarrier<PIPE_ALL>();
}
```

关键点：

1. **dtype 宏**：`ORIG_DTYPE_Q/K_CACHE/ATTN_OUT`（asc_opc 按输入名派生的宏）决定 `INPUT_T/OUT_T = half | bfloat16_t`——注意内部名是 Q/K_CACHE/ATTN_OUT，抄 FIA 的 QUERY/KEY/ATTENTION_OUT 会导致恒走 bf16 分支（坑）；
2. **布局推导 + 三重 static_assert 互锁**（`:49-142`）：`GetQueryLayoutMla/GetOutLayoutMla/IsPageAttentionMla/GetKvLayoutMla` 与 `InOutLayoutTypeValue` 表、与实际消费的 `CubBlockNormal::LAYOUT / PAGE_ATTENTION / KV_FORMAT` 三方互锁——防"推导与消费漂移"；
3. **AIC/AIV 分块选型**（`:186-194`）：

```cpp
#ifdef __DAV_C310_CUBE__     // 950 编译：AIC TU 定义、AIV TU 不定义
    using CubBlock = CubBlockNormal;      // AIC：cube 实块
    using VecFaBlock = VecFaBlockDummy;   //      vec 哑块
    using VecFdBlock = VecFaBlockDummy;
#else
    using CubBlock = CubBlockDummy;       // AIV：cube 哑块
    using VecFaBlock = VecFaBlockNormal;  //      vec 实块
    using VecFdBlock = VecFdBlockNormal;  //      FD 实块（仅 AIV 有 FD）
#endif
```

4. **k==v、rope 合并**：`value` 与 `key` 传同一 GM 指针（k_cache 的 nope 段即 V）；q/k_cache 是 576 宽单张量（nope 512 + rope 64 合并），无独立 rope 输入；
5. `GET_TILING_DATA_MEMBER`：动态图零拷贝强转 tiling buffer；静态图 `convert_from_bytes` 拷成栈局部结构体（静态图下 tiling 是编译期常量数组，不能直接指针访问）。

### 3.2 实例化面（SEL）

`template_tiling_key.h` 的 `ASCENDC_TPL_SEL` 列表：
`InOutLayoutType{0,1,2} × Config{9} × HasAttenMask{0,1} × KvLayoutType{1,2,3} × IsFd{0,1}` = **36 组合 × 2 dtype = 72 个 kernel 实例**（kv=0 连续 KV 不实例化）。

---

## 4. arch35 主逻辑：FlashAttentionNoQuantMlaKernel

文件：`op_kernel/arch35/flash_mla_with_kvcache_kernel_noquant_mla.h`

### 4.1 Init（`:122-173`）

1. 读 metadata 头：`sectionNum_ = metadata[0]`；
2. 绑 GM 区：
   - `faMetaDataGm_ = metadata + 16`（FA 区，`36核×16字×sectionNum`）
   - `fdMetaDataGm_ = metadata + 16 + 36*16*sectionNum`（FD 区，`72核×16字×sectionNum`）
3. `InitConstInfo()`：把 tilingData 全字段灌入 `constInfo_`（b/s1/s2/d/dV/rope=64、keyStrides、mask、PA、LSE、S1OutSplit…）；
4. `seqLensTool_`：cu_seqlens_q / cache_seqlens 的 ACTLEN 解析器初始化（INT32→uint32）；
5. `InitMMResBuf()`：**这是 kernel 侧内存规划的骨架**（见 §5）；
6. 分核初始化：`ASCEND_IS_AIV` → `vecFaBlock_.InitVecBlock` + `ClearOutput()`（刷无效行/无效 batch）；`ASCEND_IS_AIC` → `cubeBlock_.InitCubeBlock`；`FLASH_DECODE` → `vecFdBlock_.InitParams/InitGlobalTensor`。

### 4.2 任务调度器（FlashAttention, `:373-426`）

按 AICPU metadata 给出的本核区间 `(bN2Start..End, gS1OStart..End, s2OStart..End)` 做三维循环（bN2 × gS1 × s2）：

- `GetTaskDealMode`：CREATE_TASK / DEAL_ZERO（空 seq 刷输出）/ SKIP / NOT_START / S2_END；
- `taskRunInfo[2]` 环形缓冲（`PRELOAD_N=1` → **C1 C1 C2 流水**：当前轮铺 mm1，下一轮铺 mm2，AIV 侧 vec1/vec2 同理延迟一轮）；
- `ExecuteTask`（`:595-617`）：

```cpp
if (runInfo0.isValid)  AIC→ComputeMm1(runInfo0) | AIV→ComputeVec1(runInfo0)
if (loop >= PRELOAD_N && runInfoNegN.isValid)
                       AIC→ComputeMm2(runInfoNegN) | AIV→ComputeVec2(runInfoNegN)
```

即同一条任务流，AIC 与 AIV 各跑各的算子，靠**共享 L1/UB 缓冲 + 跨核事件**错开依赖（AIC 写 → AIV 读 P/score → AIC 再读 P 做 mm2）。

### 4.3 多 section 与 FD（Process, `:845-879`）

```cpp
CrossCoreBufferInit();                       // 预置跨核事件标记
for (sectionIdx = 0; sectionIdx < sectionNum_; sectionIdx++) {
    FlashAttention(sectionIdx);              // 所有核（37+72）都跑 FA 流水
    if constexpr (FLASH_DECODE) {
        if ASCEND_IS_AIV {
            FlashDecode(sectionIdx);         // 仅 AIV：FD 归约
            // GetFDSectionInfo: fdParams_.fdCoreEnable = mLen>0
            // vecFdBlock_.InitBuffers(&pipe_)   // pipe_->Reset() 重排 UB
            // ICachePreLoad(2); SyncAll(); vecFdBlock_.FlashDecode(fdParams_)
        }
    }
}
CrossCoreBufferUnInit();
```

- **FA**（prefill/常规）：每个 section 一次完整 flash-attention 两矩阵循环；
- **FD**（decode / 长 KV 被切成多段的归约）：AIV 核把 FA 阶段写到 workspace 的 attn 暂存按段读回、在线 softmax 归一、跨段归约后写最终输出（§6.3）。

---

## 5. kernel 内存分配（重点①）

### 5.1 总预算（硬件常量）

| 存储 | 大小 | 在本算子的用途 |
|---|---|---|
| GM | 由 aclnn/workspace 给 | 输入输出、block_table、seq-lens、metadata、workspace（FD 归约暂存） |
| L1 | 512 KB（每核） | Q 行块 + K/V/P 共享块（k==v 复用） |
| L0A / L0B | 64 KB each | MatMulK/N 的 A、B 分块（每块 32 KB ×2 双缓冲） |
| L0C | 256 KB | 累加器（每块 128 KB ×2 双缓冲） |
| UB | 192 KB（平台） | mm1 结果（score，16K×2）+ mm2 结果（64K×2）+ vec 区（stage/softmax/LSE） |

### 5.2 分配代码逐行（kernel_noquant_mla.h `InitMMResBuf`, `:175-186` + cube/vec `InitBuffers`）

```cpp
// —— kernel 级（主逻辑持有 TPipe）——
l1BufferManager_.Init(&pipe_, 524288);          // L1 总账：512 KB
l1KvPBuffers_.Init(l1BufferManager_, mm12RightSize);   // 3buff × 144KB = 432KB
    // mm12RightSize = s2BaseSize * 576 * sizeof(INPUT_T) = 128*576*2 = 147456 (144KB)
ubBufferManager_.Init(&pipe_, mm1ResultSize*2 + mm2ResultSize*2);   // 16K*2 + 64K*2 = 160KB
bmm2Buffers_.Init(ubBufferManager_, mm2ResultSize);   // 2 × 64KB（mm2 结果，double buffer）
bmm1Buffers_.Init(ubBufferManager_, mm1ResultSize);   // 2 × 16KB（mm1 结果）

// —— cube（额外从 L1 总账取 Q 行块）——
l1QBuffers.Init(*l1BufferManagerPtr, mBaseSize*576*sizeof(Q_T));    // 64*576*2 = 72KB
l0aBufferManager.Init(tPipe, 65536);   // 64KB；mmL0ABuffers.Init(…, 32*1024) → 2 × 32KB
l0bBufferManager.Init(tPipe, 65536);   // 同上
l0cBufferManager.Init(tPipe, 262144);  // 256KB；mmL0CBuffers.Init(…, 128*1024) → 2 × 128KB

// —— vec（UB 内再分）——
// stage1OutQue(32+1)×128×2 ≈ 8.4KB（P 缓存，+1 行解 bank 冲突）
// stage2OutBuf 32×512×4 = 64KB（mm2 结果重排）
// softmaxMaxBuf/SumBuf/ExpBuf[2] 各 256B；maxBrdcst/sumBrdcst 各 1KB；softmaxLseQueue 1KB
// attenMaskInQue[2] 各 4KB（hasAtten 时）
// FD 阶段：fdMm2ResBuf1/2 各 16KB（双缓冲）、fdReduceBuf/fdOutputBuf 16KB、LSE 区 6KB×4 + 256B×3
```

**L1 布局的关键复用**（kernel 头注释 ":67" + cube 代码）：

```
K  (128 行 × 576，nope512+rope64 双段写：CopyKeyAndRopeTile)
V  = K 的 nope 段（bmm2 的 B 直接取 buffer 起始）
P  (128 行 × 512) 写在 K_rope 区（offset = s2BaseSize*dVBaseSize = 128*512）
   —— vec softmax 后的 P 覆盖 K_rope，一块 L1 三用（K→[P]+V）
```

- `CopyQueryAndRopeTile` / `CopyKeyAndRopeTile`（cube `:207-266`）：L1 上 nope[0,512) 与 rope[512,576) 两段分别搬运（rope 段 L1 偏移 = `dSize*dstStride`、GM 偏移 `dIdx=512`），拼出 mm1 的 K=576 S 矩阵；
- `valueGm` 只初始化不再单独读取（k==v，bmm2 的 V 段 = L1 上的 nope 区）。

### 5.3 GM 地址与偏移（page attention 的"取数"）

- `FaGmTensor` 全都在 `InitCubeInput` 里 `SetGlobalBuffer` + `offsetCalculator.Init`（cube `:122-189`）：
  - PA 布局的 KV：`blockTableGm`、`bnStride/n2Stride`（来自 tiling 的 `keyStrides.valueStrides`，host 在 `GetKvIsContiguous` 按布局算好）传入 `OffsetCalculator`；
  - `GM_KV_PA_NZ` 分支按 `d0 = 32/sizeof(KV_T)`（fp16→16）、`d1 = headDim/d0`（576/16=36）五维 NZ 分块寻址；
- 取某一行 KV 块：`blockIdx = block_table[bIdx * maxBlockNumPerBatch + s2Idx/blockSize]` → `offset = blockIdx*bnStride + n2Idx*n2Stride + (d1Idx…)*…`（`memcopy/offset_calculator_v2.h` + `parser.h` 的 `BlockTableParser`）；
- `GmCoord/GmKvCoord`（bIdx/n2Idx/gS1Idx/s2Idx/dIdx/dealSize）是"要搬哪一块"的描述，交给 `CopyKvGmToL1`/`CopyQueryGmToL1`（`memcopy/copy_gm_to_l1.h`）按 GM 布局生成 DataCopy 参数（跨块边界时按页切分）。

### 5.4 workspace（GM）语义

- host 按 §2.3 公式申请；kernel 侧 `GetUserWorkspace(workspace)` 取用户区；
- 实际消费方是 **vec/FD**（vec 块 `InitVecInput`）：
  - `accumOutGm = workspace`（FD 重读 FA 的中间 attn 输出）
  - `softmaxFDSumGm = workspace + accumOutSize`
  - `softmaxFDMaxGm = workspace + accumOutSize + logSumExpSize`

---

## 6. 同步机制（重点②）

AscendC 为同核流水（MTE/Vector/Cube 单元）提供"事件 + 栅栏"，为**跨核**（AIC↔AIV）提供硬件 cross-core 事件。本算子三层都用。

### 6.1 核内流水事件（同核单元间）

| 事件 | 语义 | 出现位置 |
|---|---|---|
| `MTE1_MTE2` | GM→L1/UB 搬运（MTE2 写）之后 → 消费方（MTE1/matmul）可读 | l1QBuffers、l1KvPBuffers、q/k 搬运 |
| `MTE2_MTE1` | 生产完成信号（反向） | 同上 |
| `M_FIX ↔ FIX_M` | matmul→L0C 完成 / fixpipe 从 L0C 取数完成 | cube IterateBmm1/2（`Wait<FIX_M>`、`Set/Wait<M_FIX>`） |
| `MTE3_V ↔ V_MTE3` | 向量写 GM 前/后的握手 | vec：`mte3ToVId[]/vToMte3Id[]`（输出副本与 LSE） |
| `PipeBarrier<PIPE_ALL>` | 内核收尾全流水栅栏 | 入口 `flash_mla_with_kvcache.cpp:259` |

典型用法（cube `IterateBmm1`）：
```cpp
outputBuf.WaitCrossCore();                        // ① AIC：等 AIV 消费完上一轮 mm1 结果（UB）
mm1A = l1QBuffers.Get();  mm1A.Wait<MTE1_MTE2>(); // ② Q 块搬运完成
CopyQueryAndRopeTile(...); mm1A.Set<MTE2_MTE1>();
WaitFlag<MTE1_MTE2>(kvp.GetEventID<MTE1_MTE2>()); // ③ 等 K 可写 → CopyKeyAndRopeTile
SetFlag<MTE2_MTE1>(kvp.GetEventID<MTE2_MTE1>());  //    K 就绪（同核 MTE1→matmul 消费）
mm1ResL0C = mmL0CBuffers.Get(); mm1ResL0C.Wait<FIX_M>();
MatmulK<...,MK,KN>(A=Q, B=K, K=576) → L0C
mm1ResL0C.Set<M_FIX>(); mm1ResL0C.Wait<M_FIX>();
FixpipeMm1(...)                                 //  L0C→UB（dualDstCtl=1: M 拆半给 AIV0/1）
mm1ResL0C.Set<FIX_M>();
outputBuf.SetCrossCore();                         // ④ 通知 AIV 可读 mm1 结果
```

### 6.2 跨核同步（AIC ↔ AIV）

- 载体：`Buffer<UB, SyncType::CROSS_CORE_SYNC_BOTH>`（bmm1/bmm2 结果）与
  `Buffer<L1, SyncType::CROSS_CORE_SYNC_FORWARD>`（l1KvP）；
- 语义：`SetCrossCore()` = `CrossCoreSetFlag`（事件号 id、AIV 侧偏移 16），
  `WaitCrossCore()` = `CrossCoreWaitFlag`；
- **AIC 生产、AIV 消费**（FORWARD）与**双向互等**（BOTH）区分如下：

```
 AIC                                   AIV
 │ InitBuffers(L1/UB/L0)               │ InitBuffers(UB 各段)
 │ CrossCoreBufferInit():              │ CrossCoreBufferInit():
 │   l1KvP SetEventID ×3 + SetFlag     │   bmm1/bmm2 SetCrossCore ×2×2（预置"可开始"）
 │       │ MTE1_MTE2（K 可写）           │
 ▼ mm1: ───────────────────────────────▼
 AIC: CopyQ/K→L1 → MatmulK →           AIV: bmm1ResBuf.WaitCrossCore()      ← ②
      Fixpipe→ UB bmm1                 │     读 score，ProcessVec1Vf 在线 softmax
      bmm1.SetCrossCore() ────────────►│     P 写回 L1(K_rope 区)
      kvp.SetCrossCore()？（FORWARD）    │     bmm1ResBuf.SetCrossCore()       ← AIC 复用
 ▼ mm2: ───────────────────────────────▼
 AIC: kvp.WaitCrossCore()  ← P 就绪     │ AIV: bmm2ResBuf.WaitCrossCore()     ← ④
      MatmulN(A=P, B=V) → L0C           │     读 mm2 结果 → FlashUpdate/累加
      Fixpipe→ UB bmm2                 │     bmm2ResBuf.SetCrossCore()
      bmm2.SetCrossCore() ────────────►│     ∑P·V/LSE → 输出 GM（V_MTE3 握手）
                                       │
 会话级：CrossCoreBufferUnInit（AIC WaitFlag×3 + WaitCrossCore×4）
         FD 前：SyncAll()；入口收尾：PipeBarrier<PIPE_ALL>
```

要点：

- **score（mm1 结果）从 L0C → UB 的这一跳就是 AIC→AIV 的握手点**（Fixpipe 写 UB + SetCrossCore）；
- `dualDstCtl=1`（cube FixpipeMm1 `:279`）：L0C 的 M 维按一半拆给 `AIV0/AIV1` 两个 subcore（`vecMbaseIdx = subBlockIdx ? actVecMSize : 0`），每个 AIV 只算自己那半行——所以 V 核有 72 个（36×2）；
- L1 KV 块是 **3 缓冲**（`BuffersPolicy3buff`），UB mm1/mm2 各 **2 缓冲**，任务环 `taskRunInfo[2]`（PRELOAD_N=1）——三级缓冲共同掩盖 AIC/AIV 交叉依赖（C1 C1 C2 流水）。

### 6.3 vec 在线 softmax 与 FD 归约的同步

- `ClearOutput`（vec `:187-198`）：可能含无效行时先 `SetFlag<MTE3_V>(0)` → `matmul::InitOutput` 分片写 0 → `WaitFlag` → `SyncAll()`（**72 个 AIV 全部到齐才开始**）；
- `ProcessVec1`：`WaitCrossCore` 取 score → `ProcessVec1Vf`（`vf_mul_sel_softmaxflashv2_cast_nz`：`P = softmax(QKᵀ/√512)`，含 mask、exp、sum/max 累加）→ `DataCopy` P 回 L1 → `SetCrossCore`（AIC 才能读 P 做 mm2）；
- `ProcessVec2`：`WaitCrossCore` 取 mm2 → `FlashUpdateNew/FlashUpdateLastNew`（`out = out*expScale + P*V`，在线加权）→ 末块 `LastDivNew`（`out/=sum`）→ `CopyAttentionOut`（运行时按 `outputLayout` switch BSH/BNSD/TND/NTD/NBSD 用 `CopyAttenOutUbToGm` 写回 GM）；`V_MTE3/MTE3_V` 事件对保证"向量算完才搬、搬完才重算"；
- **FD**（`block_vec_flashdecode_mla.h`）：AIV-only，`pipe_->Reset()` 后在 UB 重建自己一套缓冲；`CopyAccumOutIn` 用 `fdMm2ResBuf1/2`（16KB×2，`preLoadNum=2`）从 workspace 双缓冲重读 FA 各段 attn 输出 → 按 sum/max 归一 → `ReduceFinalRes` 跨 `fdS2SplitNum` 段归约 → `CopyFinalResOut`（`V_MTE3→MTE3_V` + `Bmm2DataCopyOutTrans` 写 attnOut）→ `CopySoftmaxLseOutput` 写 LSE；8 个 flag：`SYNC_LSE_MAX_SUM_BUF1/2、SYNC_MM2RES_BUF1/2、SYNC_FDOUTPUT、SYNC_LSEOUTPUT、SYNC_SINK_BUF1/2`。

---

## 7. 数据流走查（prefill 一例）

设：B=2，N=8（N1），seq_q: [192,192]，seq_kv: [2048,2048]，block=64，fp16，TND/PA_NZ。

```
host:  metadata(AICPU) ──► 36 个 AIC 各自获得本核 FA 区间；长 KV 还可能产生 FD section
       blockDim=72, key = 2|9<<8|0<<18|3<<19|1<<21   (TND_TND, cfg9, no mask, PA_NZ, fd)
kernel: 实例 <TND,9,0,PA_NZ,fd> 被 asc_opc 装配为 AIC(36)/AIV(72) 两份 TU

AIC 每核：
  逐 section → FlashAttention:
    for bN2, gS1 (64 行块), s2 (128 行块):
      mm1:  Q[64×576](L1) × K[128×576](L1 3buf) → L0C → Fixpipe → UB bmm1(16K)  [SetCrossCore]
      mm2:  A=P[64×512](L1 K_rope 区) × B=V[128×512](L1 nope 区) → L0C → UB bmm2(64K) [SetCrossCore]
      （prev 轮次 mm2 与本轮 mm1 重叠：C1 C1 C2）
AIV 每核（subcore 各半 M 行）：
      vec1: 读 bmm1 → P=softmax(QKᵀ·scale) fp16 → P 写回 L1 （3buf 轮转）
      vec2: 读 bmm2 → out += P·V · expScale → /sum → 输出 GM（TND 布局 (T,N,512)）
      LSE:  TND→(N,T) 转置写回 softmaxLse
  FD section（若 mLen>0）: AIV 从 workspace 重读各段 attn → 归约 → 终写输出+LSE
```

两个数值得记住：**mm1 用 scale=√512（不是 √576）**；**rope 段（64 维）不参与 mm2**（V 只取 nope 512）。

---

## 8. 关键文件速查

| 层 | 文件 | 要点 |
|---|---|---|
| torch 接口 | `torch_extension/cann_ops_transformer/ops/flash_mla_with_kvcache.py` | schema 双 op、impl、metadata 容量 |
| torch C++ | `torch_extension/.../ops/csrc/flash_mla_with_kvcache.cpp` | shape 推导、ACLNN_CMD、metadata 容量检查 |
| metadata 算子 | `attention/flash_mla_with_kvcache_metadata/` | AICPU 切分（16 字/核 双区布局） |
| host tiling | `op_host/arch35/flash_mla_with_kvcache_tiling.cpp` | 门链、透传、key、workspace、blockDim |
| tiling key | `op_kernel/arch35/flash_mla_with_kvcache_template_tiling_key.h` | 位编码 + SEL 72 实例 |
| kernel 入口 | `op_kernel/flash_mla_with_kvcache.cpp` | 布局推导、AIC/AIV 分块、Init/Process |
| 主逻辑 | `op_kernel/arch35/flash_mla_with_kvcache_kernel_noquant_mla.h` | TPipe、内存分配、任务调度、FA/FD |
| cube | `op_kernel/arch35/flash_mla_with_kvcache_block_cube_noquant_mla.h` | L1/L0 分配、mm1/mm2、Fixpipe、事件 |
| vec | `op_kernel/arch35/flash_mla_with_kvcache_block_vec_noquant_mla.h` | 在线 softmax、累加、LSE、输出写回 |
| FD | `op_kernel/arch35/flash_mla_with_kvcache_block_vec_flashdecode_mla.h` | 双缓冲、跨段归约、8 flag |
| 缓冲策略 | `attention/common/op_kernel/buffers_policy.h` `buffer_manager.h` `buffer.h` | 单/双/三缓冲、Set/WaitCrossCore |
| GM 布局 | `attention/common/op_kernel/memcopy/{gm_layout,gm_format,offset_calculator_v2,gm_coord,fa_gm_tensor}` | PA 寻址、offset 计算、拷贝参数 |
| 拷贝 | `attention/common/op_kernel/memcopy/{copy_gm_to_l1,copy_ub_to_gm}.h` | GM↔L1/UB 数据搬运（按布局/页） |

---

## 附：一句话记忆

> **一个入口，两波核心，三块缓冲，四类事件。**
> 入口按 tiling key 实例化 72 个内核；36 个 AIC 算两个矩阵乘、72 个 AIV 做在线 softmax 与输出；
> L1 三缓冲轮流装 K/[P]+V、UB 双缓冲传 score 与 mm2 结果、任务环再叠一轮流水；
> 核内用 MTE/FIXPIPE 事件、跨核用 Set/WaitCrossCore、会话用 SyncAll/PipeBarrier 收束。