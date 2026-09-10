# CCU AllGather + MXFP8 Matmul 阶段四设计文档

## 1. 目标与范围

### 1.1 目标

1. **将 matmul 替换为 MXFP8 matmul**：A/B 从 FP16 改为 `fp8_e4m3fn_t`，新增 scale A/B（`fp8_e8m0_t`），C 保持 BF16。
2. **去掉 AIV 核**：当前 `KERNEL_TYPE_MIX_AIC_1_1` 中 AIV 空跑（`if ASCEND_IS_AIV { return; }`），阶段四改为纯 AIC kernel。
3. **MXFP8 通信分阶段**：scale 一次完成所有数据的 AllGather；x1（FP8 data）分 commTurn 次 AllGather + matmul，保持当前通算并行逻辑。

### 1.2 不变项

- CCU `<<<>>>` 直调机制不变（`streamCcu` + `streamAivAic` 双流）。
- CCU kernel 内部循环 commTurn 次的结构不变。
- AIC 直轮询 CKE Event 同步机制不变。
- Variable 参数桥 + Event 同步流程不变。
- Host 侧资源编排模式不变（Variable/Event 分配、channel 获取）。

### 1.3 关键约束

- **K 必须是 32 的倍数**（MXFP8 量化基本要求）。
- **ceil(K/64) 必须是偶数**（scale 布局对齐要求，`MXFP_MULTI_BASE_SIZE=2`）。
- A/B dtype = `fp8_e4m3fn_t`（1 byte），scale dtype = `fp8_e8m0_t`（1 byte），C dtype = `bfloat16_t`（2 bytes）。

## 2. MXFP8 数据格式与内存布局

### 2.1 MXFP8 量化格式

```
MXFP_DIVISOR_SIZE = 64    // 每 64 个 FP8 元素一组
MXFP_MULTI_BASE_SIZE = 2  // 每组 2 个 E8M0 scale 值

A: [M, K] fp8_e4m3fn_t     (1 byte per element)
scaleA: [M, ceil(K/64), 2] fp8_e8m0_t  (ND layout, ScaleANDLayoutPtn)
  → scaleKLen = ceil(K/64) * 2

B: [K, N] fp8_e4m3fn_t     (DN layout, col-major)
scaleB: [ceil(K/64), 2, N] fp8_e8m0_t  (ScaleBDNLayoutPtn)
  → scaleKLen = ceil(K/64) * 2

C: [rankSize*M, N] bfloat16_t  (ND layout)
```

### 2.2 每行字节计算

```
dataBytesPerMRow  = K * sizeof(fp8_e4m3fn_t) = K * 1
scaleBytesPerMRow = ceil(K/64) * 2 * sizeof(fp8_e8m0_t) = ceil(K/64) * 2 * 1
cBytesPerM        = N * sizeof(bfloat16_t) = N * 2
```

### 2.3 recvBuf（AllGather 结果缓冲）布局

当前（FP16）：
```
recvBuf: [rank0 tile0 data | rank1 tile0 data | ... | rank0 tile1 data | rank1 tile1 data | ...]
```

阶段四（MXFP8）：scale 和 data 分区存放，参考 apace UDMA 实现：
```
recvBuf (data 段):
  [rank0 M行 data | rank1 M行 data | ... | rankR-1 M行 data]
  → dataRegionBytes = rankSize * M * dataBytesPerMRow

recvScaleBuf (scale 段):
  [rank0 M行 scale | rank1 M行 scale | ... | rankR-1 M行 scale]
  → scaleRegionBytes = rankSize * M * scaleBytesPerMRow
```

**关键变化**：scale 的 AllGather 一次性完成所有 M 行（不分 tile），data 的 AllGather 仍按 tileM 分 commTurn 轮。

### 2.4 sendBuf / sendScaleBuf 布局

```
sendBuf:    本 rank 的 [M, K] FP8 data，M*dataBytesPerMRow 字节
sendScaleBuf: 本 rank 的 [M, ceil(K/64)*2] E8M0 scale，M*scaleBytesPerMRow 字节
```

## 3. 通信方案设计

### 3.1 两阶段通信

用户需求：scale 一次完成所有数据通信，x1 分两次（commTurn 轮）完成通信和计算。

```
阶段 4.1: Scale AllGather（一次性，全部 M 行）
  → CCU kernel 第 0 轮：AllGather 全部 scale 数据到 recvScaleBuf
  → 完成后 EventRecord 通知 AIC

阶段 4.2: Data AllGather + Matmul（commTurn 轮，通算并行）
  → CCU kernel 第 1~commTurn 轮：每轮 AllGather tileM 行 FP8 data 到 recvBuf
  → AIC 每轮轮询 done → MXFP8 matmul
  → round d matmul 与 CCU round d+1 通信并行 overlap
```

### 3.2 CCU kernel 结构变更

当前 CCU kernel 内循环 `commTurn` 次，每轮做 FP16 data AllGather。

阶段四 CCU kernel 变为两阶段：

```cpp
__ccu_host__ void CcuAllGatherMesh1DMem2MemKernel(CcuKernelArg arg)
{
    // ============ 阶段 4.1: Scale AllGather（1 轮，全量）============
    // EventWait(events[0])  // 等 AIC 唤醒
    // 读 scale round 的 Variable 组（sendScaleBuf, recvScaleBuf, scaleSliceSize, ...）
    // ExecuteAllGatherTransfer(scale 通信参数)
    // EventRecord(events[1])  // 通知 AIC scale done

    // ============ 阶段 4.2: Data AllGather（commTurn 轮）============
    // for d = 0 .. commTurn-1:
    //   读 data round d 的 Variable 组
    //   ExecuteAllGatherTransfer(data 通信参数)
    //   EventRecord(events[d+2])  // 通知 AIC data round d done
}
```

**总 Event 数**：1（wake）+ 1（scale done）+ commTurn（data done）= commTurn + 2

**总 Variable 组数**：1（scale 组，10 个 Variable）+ commTurn（data 组，每组 10 个 Variable）= (commTurn + 1) × HCOMM_HOST_TASK_ARG_NUM

### 3.3 CCU kernel 参数桥 Variable 布局

```
Variable 组 0 (scale round):
  [0] sendScaleBuf addr          [5] scaleSliceSize
  [1] recvScaleBuf addr          [6] goSize0
  [2] token                      [7] goSize1
  [3] inputScaleOffset (=0)      [8] goSize2
  [4] outputScaleOffset          [9] goSize3

Variable 组 1..commTurn (data round d, d=0..commTurn-1):
  base = (d+1) * HCOMM_HOST_TASK_ARG_NUM
  [base+0] sendBuf addr          [base+5] dataSliceSize
  [base+1] recvBuf addr          [base+6] goSize0
  [base+2] token                 [base+7] goSize1
  [base+3] inputDataOffset      [base+8] goSize2
  [base+4] outputDataOffset     [base+9] goSize3
```

### 3.4 Event 布局

```
eventVa[0]          = wake（AIC→CCU，仅 scale round 前）
eventVa[1]          = scale AllGather done（CCU→AIC）
eventVa[d+2]        = data round d AllGather done（CCU→AIC），d = 0..commTurn-1
```

### 3.5 通信参数计算

**Scale 通信**（一次性全量）：
```
scaleSliceSize = M * scaleBytesPerMRow  // 本 rank 全量 scale 通信字节
                 = M * ceil(K/64) * 2

inputScaleOffset  = 0  // sendScaleBuf 从头开始
outputScaleOffset = rankId * scaleSliceSize  // recvScaleBuf 中本 rank 的位置
```

**Data 通信**（按 tileM 分轮，与当前 FP16 逻辑相同，仅 perDataSize 从 2 改为 1）：
```
dataSliceSize = tileM * K * sizeof(fp8_e4m3fn_t) = tileM * K * 1

inputDataOffset  = d * dataSliceSize  // round d 在 sendBuf 中的偏移
outputDataOffset = d * rankSize * dataSliceSize + rankId * dataSliceSize
                   // round d 在 recvBuf 中的位置（rank-major + tile 分轮）
```

## 4. Matmul 替换为 MXFP8

### 4.1 Blaze 类型链变更

当前（FP16 → BF16）：
```cpp
using MmTypeA = half;
using MmTypeB = half;
using MmTypeC = bfloat16_t;
using DispatchPolicy = Blaze::Gemm::MatmulMultiBlockBasic<>;
using BlockMmad = Blaze::Gemm::Block::BlockMmad<DispatchPolicy, MmTypeA, MmLayoutA, ...>;
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<MmProblemShape>;
using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<MmProblemShape, BlockMmad, ...>;
```

阶段四（MXFP8 → BF16）：

**方案选择**：直接用 `GemmUniversal` + `MatmulWithScaleMx` dispatch policy + `BlockMmad` + `BlockSchedulerMatmulBasic`，不使用 apace UDMA 参考实现的 `FragmentTensor` + `BlockSchedulerQuantBatchMatmulV3` 路线。

**理由**：
1. 我们的 CCU 直调场景下，通信结果已经落在 recvBuf/recvScaleBuf 的连续 GM 区域中，每轮 matmul 的 A 数据地址可通过简单偏移计算得到，不需要 FragmentTensor 的多地址列表机制。
2. `MatmulMultiBlockBasic` + `GemmUniversal` 的简单组装方式已在阶段三验证通过，MXFP8 只需替换 dispatch policy 和 BlockMmad 的 scale 参数。
3. apace UDMA 实现的 FragmentTensor 路线是为 AIV/AIC 同核 MIX + win buffer 复杂布局设计的，CCU 直调场景下 recvBuf 布局更简单。

```cpp
using MmTypeA = fp8_e4m3fn_t;
using MmTypeB = fp8_e4m3fn_t;
using MmTypeC = bfloat16_t;
using MmScaleType = fp8_e8m0_t;
using MmLayoutA = AscendC::Te::NDExtLayoutPtn;
using MmLayoutB = AscendC::Te::DNExtLayoutPtn;   // B 为 DN layout（col-major）
using MmLayoutC = AscendC::Te::NDExtLayoutPtn;
using MmLayoutScaleA = AscendC::Te::ScaleANDLayoutPtn;
using MmLayoutScaleB = AscendC::Te::ScaleBDNLayoutPtn;

using DispatchPolicy = Blaze::Gemm::MatmulWithScaleMx<0, 0>;
using BlockMmad = Blaze::Gemm::Block::BlockMmad<
    DispatchPolicy, MmTypeA, MmLayoutA, MmTypeB, MmLayoutB,
    MmTypeC, MmLayoutC, float, MmLayoutC>;
using BlockEpilogue = Blaze::Gemm::Block::BlockEpilogueEmpty;
using BlockScheduler = Blaze::Gemm::Block::BlockSchedulerMatmulBasic<MmProblemShape>;
using MatmulKernel = Blaze::Gemm::Kernel::GemmUniversal<
    MmProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
```

### 4.2 MatmulTilingData 变更

```cpp
#pragma pack(push, 8)
struct alignas(8) MatmulTilingData {
    uint32_t m{0};          // = rankSize * tileM（单 round AllGather 后总行数）
    uint32_t n{0};          // = N
    uint32_t k{0};          // = K
    uint32_t baseM{0};      // = 256
    uint32_t baseN{0};      // = 256
    uint32_t baseK{0};      // = 64（MXFP8 仍可用 64，但需对齐 MXFP_DIVISOR_SIZE=64）
    uint32_t scaleKL1{0};   // L1 中一次 scale staging 的 K 长度（新增）
    uint32_t mL1{0};        // = baseM
    uint32_t nL1{0};        // = baseN
    uint32_t kL1{0};        // = baseK * 2
    uint32_t mTailCnt{1};
    uint32_t nTailCnt{1};
    uint32_t usedCoreNum{0};
    uint8_t  stepK{0};      // L1 rolling buffer 步进（新增）
    uint8_t  nBufferNum{0}; // L1 rolling buffer 数（新增）
    uint8_t  dbL0c{0};      // L0C double buffer（新增）
    uint8_t  isHf32{0};     // 保留字段
};
#pragma pack(pop)
```

### 4.3 MatmulProcess 变更

当前 `MatmulProcess` 只接收 `aGM, bGM, cGM`。

阶段四需增加 `aScaleGM, bScaleGM`，并在组装 Params 时传入 scale 地址：

```cpp
__aicore__ inline void MatmulProcess(
    __gm__ void *aGM, __gm__ void *aScaleGM,
    __gm__ void *bGM, __gm__ void *bScaleGM,
    __gm__ void *cGM, MatmulTilingData tilingData)
{
    // 组装 Blaze 类型链（MatmulWithScaleMx）
    // ...
    GM_ADDR aGmAddr = reinterpret_cast<GM_ADDR>(reinterpret_cast<uint64_t>(aGM));
    GM_ADDR aScaleGmAddr = reinterpret_cast<GM_ADDR>(reinterpret_cast<uint64_t>(aScaleGM));
    GM_ADDR bGmAddr = reinterpret_cast<GM_ADDR>(reinterpret_cast<uint64_t>(bGM));
    GM_ADDR bScaleGmAddr = reinterpret_cast<GM_ADDR>(reinterpret_cast<uint64_t>(bScaleGM));
    GM_ADDR cGmAddr = reinterpret_cast<GM_ADDR>(reinterpret_cast<uint64_t>(cGM));

    using Params = typename MatmulKernel::Params;
    Params params = {
        {m, n, k, 1},                    // ProblemShape
        {aGmAddr, bGmAddr, cGmAddr,      // BlockMmadParams（含 scale）
         aScaleGmAddr, bScaleGmAddr, nullptr,
         mL1, nL1, kL1, baseM, baseN, baseK,
         L1_BUFFER_NUM, L0C_DB},
        {},                               // BlockEpilogueParams (empty)
        {mL1, nL1, kL1, baseM, baseN, baseK,
         mTailCnt, nTailCnt, ...}         // BlockSchedulerParams
    };

    MatmulKernel kernel;
    kernel(params);
}
```

> **待验证**：`BlockMmad::Params`（即 `BlockMmadParams`）在 `MatmulWithScaleMx` dispatch policy 下是否包含 scaleA/scaleB 字段。需查阅 `block_mmad.h` 对应特化的 Params 结构。这是实现阶段的首要验证点。

### 4.4 BlockScheduler 问题

当前用 `BlockSchedulerMatmulBasic`。`MatmulWithScaleMx` 的 `ScheduleType` 是 `KernelMmadWithScaleMx`，需确认 `BlockSchedulerMatmulBasic` 是否兼容此 schedule type。

**备选方案**：如果 `BlockSchedulerMatmulBasic` 不兼容 MX，则参照 apace UDMA 实现使用 `BlockSchedulerQuantBatchMatmulV3`，但这需要 FragmentTensor 支持，增加复杂度。

**优先方案**：优先尝试 `BlockSchedulerMatmulBasic` + `GemmUniversal` 的简单组装。如果编译/运行失败，再退回到 FragmentTensor 路线。

## 5. 去掉 AIV 核

### 5.1 当前 AIV 空跑

```cpp
__global__ __aicore__ void all_gather_matmul_kernel(...)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);  // MIX 1:1
    if ASCEND_IS_AIV {
        return;  // AIV 空跑
    }
    if ASCEND_IS_AIC {
        AicDirectPollProcess(...);
    }
}
```

### 5.2 阶段四：纯 AIC

**方案**：将 kernel task type 从 `KERNEL_TYPE_MIX_AIC_1_1` 改为纯 AIC kernel。

```cpp
__global__ __aicore__ void all_gather_matmul_kernel(...)
{
    // 不再使用 KERNEL_TYPE_MIX_AIC_1_1
    // 纯 AIC kernel，默认 task type 即为 AIC
    AscendC::InitSocState();
    AicDirectPollProcess(...);
}
```

**验证点**：
1. `<<<>>>` 直调是否支持纯 AIC kernel（不加 `KERNEL_TASK_TYPE_DEFAULT`）？阶段一直调 CCU 时用的 `schd = {1, 0, 0x01}`，AIC kernel 的 block 数由 `usedCoreNum` 决定。
2. 纯 AIC kernel 下 `GetBlockIdx()` / `GetBlockNum()` 的语义是否与 MIX 1:1 下一致？
3. 需确认纯 AIC kernel 的 launch 参数：`<<<usedCoreNum, 0, streamAivAic>>>`，block 数 = AIC 核数（不再是对数）。

**备选方案**：如果纯 AIC 直调有困难（例如 `<<<>>>` 不支持纯 AIC），保留 `KERNEL_TYPE_MIX_AIC_1_1` 但 AIV 继续空跑。这不影响功能，只是浪费 AIV 核资源。

> **决策建议**：优先尝试纯 AIC。如果 `<<<>>>` 直调不支持或编译报错，退回 MIX_AIC_1_1 + AIV 空跑（功能等价，仅性能浪费）。

## 6. AIC 直轮询流程变更

### 6.1 当前流程（FP16，commTurn 轮）

```
block 0 预写 commTurn 组参数桥 → 写 wake(eventVa[0]=1)
所有核循环 commTurn 次：
  轮询 eventVa[d+1]==1 → matmul tile d
```

### 6.2 阶段四流程（MXFP8，1 轮 scale + commTurn 轮 data）

```
block 0 预写 1 组 scale 参数桥 + commTurn 组 data 参数桥 → 写 wake(eventVa[0]=1)
所有核：
  1. 轮询 eventVa[1]==1（scale AllGather done）
     → 所有 scale 数据已就绪，后续每轮 matmul 都可使用
  2. 循环 commTurn 次（d = 0..commTurn-1）：
     轮询 eventVa[d+2]==1（data round d done）
     → MXFP8 matmul tile d
     ←── CCU round d+1 通信并行 overlap
```

### 6.3 AIC 参数桥预写

block 0 一次性预写所有 round 的参数：

```cpp
if (GetBlockIdx() == 0) {
    // 预写 scale round 参数桥（startVa[0][0..9]）
    WriteGmByPassDCache(startVa[0][0], sendScaleBuf);
    WriteGmByPassDCache(startVa[0][1], recvScaleBuf);
    // ... 其余 8 个参数

    // 预写 commTurn 组 data round 参数桥（startVa[d+1][0..9]）
    for (d = 0..commTurn-1) {
        WriteGmByPassDCache(startVa[d+1][0], sendBuf);
        WriteGmByPassDCache(startVa[d+1][1], recvBuf);
        WriteGmByPassDCache(startVa[d+1][3], inputDataOffset[d]);
        WriteGmByPassDCache(startVa[d+1][4], outputDataOffset[d]);
        // ... 其余参数
    }

    // 写 wake
    WriteGmByPassDCache(eventVa[0], 1);
}
```

### 6.4 Matmul 调用参数

每轮 matmul 的 A/scaleA 地址计算：

```
round d matmul:
  A = recvBuf + d * rankSize * tileM * dataBytesPerMRow
      （recvBuf 中 round d 的 data 区域，包含所有 rank 的 tile d data）
  scaleA = recvScaleBuf
           （scale 一次性全量 AllGather，所有 rank 的所有 M 行 scale 都在此）
           → round d 用 scaleA[0 .. rankSize*tileM-1, 0..scaleKLen-1]
           → scaleA 偏移 = rankId_offset 不需要按 round 分（scale 已全量）
           
  实际上 scaleA 每轮 matmul 的偏移与 A 的行范围对应：
    A 的行范围 = [d * rankSize * tileM, (d+1) * rankSize * tileM)
    对应 scaleA 行范围 = 同上
    → scaleA 偏移 = d * rankSize * tileM * scaleBytesPerMRow
```

**关键点**：虽然 scale 是一次性 AllGather 的，但每轮 matmul 只消费 A 的 `rankSize * tileM` 行，对应的 scaleA 也只是该行范围的子集。因此 scaleA 的地址仍需按 round 偏移。

```
round d matmul:
  aGM        = recvBuf + d * rankSize * tileM * K * 1
  aScaleGM   = recvScaleBuf + d * rankSize * tileM * scaleBytesPerMRow
  bGM        = weightBuf（B 矩阵，K×N，全局共享）
  bScaleGM   = weightScaleBuf（B 的 scale，全局共享）
  cGM        = outputBuf + d * rankSize * tileM * N * 2
```

## 7. Host 侧编排变更

### 7.1 AllGatherMatmulContext 变更

```cpp
struct AllGatherMatmulContext {
    // ... 现有字段 ...

    // 新增：scale 通信参数
    uint64_t scaleSliceSize;        // = M * scaleBytesPerMRow（全量 scale 通信字节）
    uint64_t scaleInputSize;        // = M * scaleBytesPerMRow（每 rank scale 全量）
    // token 复用（scale 和 data 可用同一 token，或分别获取）

    // 新增：scale offset 数组（scale 只 1 轮，不需数组）
    uint64_t scaleInputOffset;      // = 0
    uint64_t scaleOutputOffset;     // = rankId * scaleSliceSize

    // 变更：Event 总数 = commTurn + 2
    // 变更：Variable 总数 = (commTurn + 1) * HCOMM_HOST_TASK_ARG_NUM

    // 新增：weightScaleBuf（B 的 scale，host 侧分配+填充）
    // weightBuf 本身从 FP16 → FP8
};
```

### 7.2 PrepareAllGatherMatmul 变更

```cpp
inline HcclResult PrepareAllGatherMatmul(
    void *sendBuf, void *recvBuf,
    void *sendScaleBuf, void *recvScaleBuf,   // 新增
    uint64_t sendCount, HcclDataType dataType, // dataType 改为 FP8
    HcclComm comm, uint32_t tileM, uint32_t K,
    AllGatherMatmulContext &ctx)
{
    // perDataSize = sizeof(fp8_e4m3fn_t) = 1
    // scaleBytesPerMRow = ceil(K/64) * 2 * 1

    // scale 通信参数
    ctx.scaleSliceSize = M * scaleBytesPerMRow;
    ctx.scaleInputOffset = 0;
    ctx.scaleOutputOffset = rankId * ctx.scaleSliceSize;

    // data 通信参数（同当前，perDataSize=1）
    ctx.sliceSize = tileM * K * 1;

    // Variable: (commTurn + 1) * HCOMM_HOST_TASK_ARG_NUM
    uint32_t totalVarNum = (ctx.commTurn + 1) * HCOMM_HOST_TASK_ARG_NUM;

    // Event: commTurn + 2
    uint32_t totalEventNum = ctx.commTurn + 2;

    // ... 其余编排同当前，仅总数变更 ...
}
```

### 7.3 Sample 入口变更

```cpp
int Sample(void *arg)
{
    // M=512, tileM=256, commTurn=2, K=256, N=256
    // K=256: ceil(256/64)=4, 4 是偶数 ✓

    // 数据类型变更：
    //   sendBuf: M*K*1 = 128KB（FP8，每 rank）
    //   recvBuf: rankSize*M*K*1 = 256KB（FP8 AllGather 结果）
    //   sendScaleBuf: M*ceil(K/64)*2*1 = 512*4*2 = 4KB（E8M0 scale，每 rank）
    //   recvScaleBuf: rankSize*M*ceil(K/64)*2*1 = 8KB
    //   weightBuf: K*N*1 = 64KB（FP8 B 矩阵）
    //   weightScaleBuf: ceil(K/64)*2*N*1 = 4*2*256 = 2KB（E8M0 scale B）
    //   outputBuf: rankSize*M*N*2 = 512KB（BF16 C 输出）

    // 数据填充：
    //   A: FP8 E4M3FN 随机数据 + 对应 E8M0 scale
    //   B: FP8 E4M3FN 全 1.0 + E8M0 scale = 2^0 = 1.0 (exp=127)
    //   简化验证：B scale 全 127(=1.0), B data 全 0x3C (=1.0 in FP8 E4M3FN)
    //     → C[i][j] = sum(A[i][0..K-1]) * scaleA * scaleB
    //     → 精确值需按 MXFP8 dequant 公式计算

    // 调用 PrepareAllGatherMatmul + <<<>>> launch
    // CCU kernel: 1 次 launch，内部 scale round + commTurn data rounds
    // AIC kernel: 1 次 launch，轮询 scale done + commTurn data done + matmul
}
```

## 8. 时序图

### 8.1 整体时序（commTurn=2）

```
streamCcu:
  ┌─────────────────────────────────────────────────────────────┐
  │ EventWait(ev[0])                                           │
  │  Scale AG (全量)                                            │
  │ EventRecord(ev[1])                                         │
  │ ┌── Data AG round 0 ──┐┌── Data AG round 1 ──┐             │
  │ │                     ││                     │             │
  │ └─────────────────────┘└─────────────────────┘             │
  │ EventRecord(ev[2])         EventRecord(ev[3])              │
  └─────────────────────────────────────────────────────────────┘

streamAivAic (AIC):
  ┌─────────────────────────────────────────────────────────────┐
  │ block0: 预写 (commTurn+1)组参数桥 → write wake(ev[0]=1)     │
  │                                                             │
  │ poll ev[1]==1? ──────────────┐                              │
  │   (scale done)                │                              │
  │ ┌── MXFP8 matmul tile 0 ──┐  │                              │
  │ │                         │  │  ← overlap with Data AG r1   │
  │ └─────────────────────────┘  │                              │
  │ poll ev[2]==1? ──────────────┘                              │
  │   (data round 0 done)                                       │
  │                                                            │
  │ poll ev[3]==1? ──────────────────────┐                      │
  │   (data round 1 done)                │                      │
  │ ┌── MXFP8 matmul tile 1 ──┐         │                      │
  │ │                         │         │                      │
  │ └─────────────────────────┘         │                      │
  └─────────────────────────────────────────────────────────────┘
```

> 注：scale AllGather 完成后，AIC 先轮询 data round 0 done，再做 matmul tile 0。matmul tile 0 与 data round 1 通信并行。这与当前 FP16 流水逻辑一致，只是多了一个前置的 scale 通信阶段。

### 8.2 关键时序约束

1. **scale AllGather 必须在所有 data matmul 之前完成**：scale 是 matmul 的输入，必须先就绪。CCU kernel 中 scale round 在 data rounds 之前串行执行，EventRecord(ev[1]) 在 data round 0 之前，因此 AIC 轮询 ev[1] 通过即可保证 scale 就绪。
2. **data round d done 必须在 matmul tile d 之前**：AIC 轮询 ev[d+2]==1 后才执行 matmul tile d。
3. **matmul tile d 与 data round d+1 并行**：AIC 执行 matmul tile d 时，CCU 正在执行 data round d+1 的 AllGather（streamCcu 与 streamAivAic 并行）。
4. **scale round 和 data round 0 之间无并行**：CCU kernel 内部串行执行 scale round → data round 0。AIC 在 scale round 期间空等（轮询 ev[1]）。这是不可避免的串行段，但 scale 数据量小（M * ceil(K/64) * 2 字节），通信时间可忽略。

## 9. 文件变更清单

| 文件 | 变更类型 | 主要变更 |
|------|----------|----------|
| `matmul_fusion_kernel.h` | 重写 | Blaze 类型链改 MXFP8（MatmulWithScaleMx），MatmulProcess 增加 scaleA/scaleB 参数，AicDirectPollProcess 增加 scale 轮询阶段，tilingData 增加 MX 字段 |
| `ccu_all_gather_matmul_kernel.asc` | 修改 | CCU kernel 增加 scale round（Variable 组 0），data round 偏移从 base=d*10 改为 (d+1)*10；MIX kernel 去 AIV 或保留空跑；kernel 入参增加 sendScaleBuf/recvScaleBuf/weightScaleBuf |
| `host_launcher.h` | 修改 | PrepareAllGatherMatmul 增加 scale 通信参数编排，Variable/Event 总数变更，mappedAddrs 扩展 scale 组 |
| `sample_main.cpp` | 重写 | 数据从 FP16 改为 FP8+E8M0，新增 sendScaleBuf/recvScaleBuf/weightScaleBuf 分配+填充，tilingData 构造变更 |
| `CMakeLists.txt` | 可能修改 | 如果纯 AIC kernel 需要不同编译选项 |

### 9.1 CcuMappedAddrs 变更

```cpp
struct CcuMappedAddrs {
    // (commTurn+1) 组 Variable：组 0 = scale, 组 1..commTurn = data
    uint64_t startVa[MAX_COMMTURN + 1][HCOMM_HOST_TASK_ARG_NUM];
    // commTurn+2 个 Event
    uint64_t eventVa[MAX_COMMTURN + 2];
};
```

## 10. 验证方案

### 10.1 精度验证

1. **Golden 生成**：参照 `tests/st/all_gather_quant_matmul/scripts/gen_data.py` 的 MXFP8 dequant + matmul 逻辑。
2. **简化 case**：M=512, tileM=256, commTurn=2, K=256, N=256, rankSize=2。
3. **数据设计**：
   - A: FP8 E4M3FN 全 1.0（0x3C），scaleA: E8M0 全 127（2^0=1.0）
   - B: FP8 E4M3FN 全 1.0（0x3C），scaleB: E8M0 全 127（2^0=1.0）
   - 期望：C[i][j] = sum(A[i][0..K-1]) * scaleA * scaleB = K * 1.0 * 1.0 = 256.0
   - tile0 行 C=256.0，tile1 行 C=256.0（A 全 1.0 时两 tile 结果相同）
4. **区分 tile 验证**：A 前 tileM 行 1.0、后 tileM 行 2.0（0x40），scaleA 全 127
   - tile0: C = 256.0，tile1: C = 512.0

### 10.2 验证步骤

1. **P4.0 编译验证**：MXFP8 Blaze 类型链编译通过（MatmulWithScaleMx + BlockMmad）。
2. **P4.1 纯 AIC 验证**：去掉 AIV 后 kernel launch 正常（如果纯 AIC 不可行则退回 MIX_AIC_1_1）。
3. **P4.2 Scale 通信验证**：CCU scale round AllGather 结果正确（recvScaleBuf 内容校验）。
4. **P4.3 Data 通信 + MXFP8 matmul 验证**：全流程端到端，outputBuf 精度校验。

## 11. 风险与待验证项

| # | 风险点 | 影响 | 缓解方案 |
|---|--------|------|----------|
| R1 | `BlockMmad::Params` 在 `MatmulWithScaleMx` 下是否含 scaleA/scaleB 字段 | 编译失败或运行时 scale 未传入 | 查阅 `block_mmad.h` 对应特化；如不含，改用 `BlockMmadQbmmMx` 或 FragmentTensor 路线 |
| R2 | `BlockSchedulerMatmulBasic` 是否兼容 `MatmulWithScaleMx` | 编译失败 | 退回 `BlockSchedulerQuantBatchMatmulV3` + FragmentTensor |
| R3 | `<<<>>>` 直调是否支持纯 AIC kernel | launch 失败 | 保留 `KERNEL_TYPE_MIX_AIC_1_1` + AIV 空跑 |
| R4 | scale 和 data 用同一 token 是否可行 | 通信错误 | 分别获取 token（sendBuf 和 sendScaleBuf 各一个） |
| R5 | CCU Variable/Event 资源配额是否足够（(commTurn+1)*10 个 Variable） | 资源分配失败 | 当前配额 400 个 Variable，commTurn=2 时需 30 个，充足 |
| R6 | Blaze MXFP8 matmul 的 BlockMmadParams 结构与 GemmUniversal 的 Params 结构是否匹配 | 编译失败 | 需查阅 `kernel_matmul_basic.h` 或 `kernel_qbmm_mx.h` 的 Params 定义 |

## 12. 实施路线

```
Step 1: 确认 Blaze MXFP8 类型链可行性
  → 查阅 block_mmad.h / kernel_qbmm_mx.h 的 Params 结构
  → 确认 BlockMmadParams 是否含 scaleA/scaleB
  → 确认 BlockSchedulerMatmulBasic 是否兼容 MatmulWithScaleMx
  → 如不可行，确定 FragmentTensor 路线的最小组装方式

Step 2: 修改 matmul_fusion_kernel.h
  → 替换 Blaze 类型链（MatmulWithScaleMx）
  → MatmulProcess 增加 scale 参数
  → AicDirectPollProcess 增加 scale 轮询阶段
  → MatmulTilingData 增加 MX 字段

Step 3: 修改 ccu_all_gather_matmul_kernel.asc
  → CCU kernel 增加 scale round
  → 尝试纯 AIC（去 MIX_AIC_1_1）
  → kernel 入参增加 scale buffer

Step 4: 修改 host_launcher.h
  → Variable/Event 总数变更
  → scale 通信参数编排

Step 5: 重写 sample_main.cpp
  → FP8 数据生成 + E8M0 scale 生成
  → 简化验证 case

Step 6: 编译 + 运行验证
  → P4.0 编译通过
  → P4.1 运行通过
  → P4.2 精度校验
```
