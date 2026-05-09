# GmmSwiGluQuant MxA8W4 Design Document

## 1. Overview

`GmmSwiGluQuant` for `MxA8W4` is a fused grouped matrix multiplication kernel that combines **SwiGLU activation** and **MX-style quantization** into a single pipeline on Ascend950 (arch35). It is designed for MoE (Mixture-of-Experts) inference where:
- **Activation (x)** is in `fp8_e4m3fn` format
- **Weight (w)** is in `fp4_e2m1` format (4-bit, stored as FRACTAL_NZ)
- **Scales** are in `fp8_e8m0` format (per-block MX scales)
- **Output (y)** is quantized to FP8 or FP4 with per-block MX scales

The kernel leverages **heterogeneous AIC (cube) + AIV (vector) cores** with explicit cross-core synchronization. The AIC performs matrix multiply-accumulate (MMA) while the AIV handles weight anti-quantization, SwiGLU activation, and MX quantization.

**Key Files Referenced:**
- Tiling: `ops-transformer/gmm/grouped_matmul_finalize_routing/op_host/op_tiling/arch35/grouped_matmul_finalize_routing_weight_quant_tiling.cpp`
- Kernel: `ops-transformer/gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block/`
- SwiGLU + MX Quant: `DiscussionForum/block_epilogue_swiglu_quant_mx_analysis.md`

---

## 2. Tiling Algorithm

### 2.1 Design Philosophy

The host-side tiling is **lightweight**. It validates constraints and passes full dimensions to the kernel. The kernel-side `ResplitController` performs dynamic M/N/K splitting based on actual L1/L0/UB capacity.

### 2.2 Host-Side Tiling (`DoOpTiling`)

```cpp
// Fake code: Host-side tiling for MxA8W4
void DoOpTiling(const InputParams& inputParams, TilingData& tilingData) {
    // --- Validation ---
    assert(inputParams.xType == DT_FLOAT8_E4M3FN);
    assert(inputParams.wType == DT_FLOAT4_E2M1);
    assert(inputParams.scaleType == DT_FLOAT8_E8M0);
    assert(inputParams.transposeW == true);     // W must be transposed
    assert(inputParams.transposeX == false);    // X must not be transposed
    assert(inputParams.K % 32 == 0);            // K aligned to 32
    assert(inputParams.N % 128 == 0);           // N aligned to 128
    assert(inputParams.K >= 32 && inputParams.N >= 128);

    // --- Pass full dimensions to kernel ---
    tilingData.kSize    = inputParams.kSize;
    tilingData.nSize    = inputParams.nSize;
    tilingData.coreNum  = compileInfo.aicNum;   // AIC core count
    tilingData.groupNum = inputParams.groupNum;
    tilingData.groupSize= MX_BLOCK_SIZE;        // 32 (MX block size)

    // --- Tiling key dispatch ---
    // ATRANS=false, BTRANS=true, HAS_BIAS=optional
    tilingKey = MakeTilingKey(ATRANS=false, BTRANS=true, hasBias);
}
```

### 2.3 Key Tiling Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `MX_BLOCK_SIZE` | 32 | MX quantization block size (elements) |
| `MX_SCALE_BLOCK_SIZE` | 64 | Scale K-dimension block size |
| `ALIGN_SIZE_KN` | 32 (K) / 128 (N) | K/N alignment requirement |
| `M_L1` | 256 | M-axis L1 tile size |
| `N_L1` | 256 | N-axis L1 tile size: 128(Swish) + 128(Gate) |
| `KB_L1` | 256 / 512 | K-axis L1 tile size (dynamic) |
| `L1_SIZE` | 512 KB | Total L1 (TSCM) capacity |

### 2.4 Dynamic K Tuning

The kernel chooses `KB_L1` based on M/N tile sizes to maximize L1 utilization:

```cpp
// Fake code: Dynamic K split size selection
uint64_t ChooseKbL1Size(uint64_t mL1Size, uint64_t nL1Size) {
    // Small tiles → larger K buffer (more K reuse)
    // Large tiles → smaller K buffer (more M/N buffer space)
    if (mL1Size <= MX_A8W4_L1_K_DYNAMIC_CONFIG_M_THRESHOLD
        && nL1Size <= MX_A8W4_L1_K_DYNAMIC_CONFIG_N_THRESHOLD) {
        return MX_A8W4_L1_K_CONFIG_512;  // 512
    } else {
        return MX_A8W4_L1_K_CONFIG_256;  // 256
    }
}
```

---

## 3. Buffer Layout

### 3.1 L1 (TSCM) Layout — 512 KB Total

The L1 is shared between AIC (cube) and AIV (vector) cores with **double buffering** for all major buffers.

**Scenario A: With Bias**

```
L1 Address Space (0 ~ 512 KB):
├─ [0x00000, 0x10000)  WeightL1_P0        64 KB
├─ [0x10000, 0x11000)  Bias_P0             4 KB
├─ [0x11000, 0x19000)  ScaleAL1_P0        32 KB
├─ [0x19000, 0x21000)  ScaleBL1_P0        32 KB
├─ [0x21000, 0x40000)  AL1_P0            124 KB
├─ [0x40000, 0x5F000)  AL1_P1            124 KB
├─ [0x5F000, 0x67000)  ScaleBL1_P1        32 KB
├─ [0x67000, 0x6F000)  ScaleAL1_P1        32 KB
├─ [0x6F000, 0x70000)  Bias_P1             4 KB
└─ [0x70000, 0x80000)  WeightL1_P1        64 KB
```

**Scenario B: Without Bias**

```
L1 Address Space (0 ~ 512 KB):
├─ [0x00000, 0x10000)  WeightL1_P0        64 KB
├─ [0x10000, 0x18000)  ScaleAL1_P0        32 KB
├─ [0x18000, 0x20000)  ScaleBL1_P0        32 KB
├─ [0x20000, 0x40000)  AL1_P0            128 KB
├─ [0x40000, 0x60000)  AL1_P1            128 KB
├─ [0x60000, 0x68000)  ScaleBL1_P1        32 KB
├─ [0x68000, 0x70000)  ScaleAL1_P1        32 KB
└─ [0x70000, 0x80000)  WeightL1_P1        64 KB
```

```cpp
// Fake code: L1 buffer allocation
void MxA8W4InitL1(uint64_t l1RemainSize, uint64_t l1StartSize,
                   uint64_t& biasL1DbOffset, GM_ADDR& biasL1) {
    uint64_t aL1Size;
    if constexpr (hasBias) {
        mxA8W4L1KDynamicConfigMThreshold_ = 240;
        aL1Size = 124 * KB;  // Less space for A due to bias buffer
    } else {
        mxA8W4L1KDynamicConfigMThreshold_ = 256;
        aL1Size = 128 * KB;
    }

    aL1DbOffset_ = l1RemainSize >> 1;  // Double-buffered AL1
    weightL1DbOffset_ = MXA8W4_WEIGHT_SIZE;  // 64KB per weight block
    scaleL1DbOffset_ = MX_SCALE_L1_SIZE;     // 32KB per scale block

    // Bias at the end of first half
    biasL1DbOffset = (l1RemainSize >> 1) - BIAS_L1_SIZE;
    biasL1 = l1StartSize + biasL1DbOffset;
}
```

### 3.2 L0 Layout

```cpp
// Fake code: L0 buffer allocation (Cube side)
void InitL0Buffers() {
    // L0A: A-matrix input (int8, from AL1)
    l0a_ = LocalTensor<L0DataType>(TPosition::A2, 0,
           64 * GetKBUnit<int8_t>() / sizeof(L0DataType));

    // L0B: B-matrix input (int8, from WeightL1)
    l0b_ = LocalTensor<L0DataType>(TPosition::B2, 0,
           64 * GetKBUnit<int8_t>() / sizeof(L0DataType));

    // L0C: Accumulation buffer (float)
    l0c_ = LocalTensor<float>(TPosition::CO1, 0,
           256 * GetKBUnit<int8_t>() / sizeof(float));

    // Bias table in C2
    biasTable_ = LocalTensor<float>(TPosition::C2, 0,
                 4 * GetKBUnit<int8_t>() / sizeof(float));
}
```

| Buffer | Position | Size | Purpose |
|--------|----------|------|---------|
| L0A | A2 | 64 KB | A-matrix (activation) tile |
| L0B | B2 | 64 KB | B-matrix (weight) tile |
| L0C | CO1 | 256×256 floats | MMA accumulation |
| BiasTable | C2 | 4 KB | Bias lookup table |

### 3.3 UB (LCM) Layout — Vector Side

```cpp
// Fake code: UB buffer allocation (Vector side)
void InitUbBuffers() {
    // Low-bit weight input (4-bit packed)
    weightLowBit_ = LocalTensor<int8_t>(TPosition::LCM, 0, 64 * KB);

    // Anti-quantized high-bit weight output (fp8)
    weightHighBit_ = LocalTensor<fp8_e4m3fn_t>(TPosition::LCM, 64 * KB, 128 * KB);

    // MMA fixpipe output (reuses weightHighBit_ space after anti-quant)
    yF32Buffer_ = LocalTensor<float>(TPosition::LCM, 64 * KB, 128 * KB);

    // Bias and rescaled bias
    bias_        = LocalTensor<half>(TPosition::LCM, 192 * KB, 1 * KB);
    biasRescale_ = LocalTensor<half>(TPosition::LCM, 193 * KB, 1 * KB);

    // MX quantization buffers
    maxB16Buffer_    = LocalTensor<bf16_t>(TPosition::LCM, 194 * KB, 2 * KB);
    scaleE8M0Buffer_ = LocalTensor<fp8_e8m0_t>(TPosition::LCM, 196 * KB, 16 * KB);
    yFp8Buffer_      = LocalTensor<int8_t>(TPosition::LCM, 212 * KB, 32 * KB);
}
```

| Buffer | Offset | Size | Purpose |
|--------|--------|------|---------|
| `weightLowBit_` | 0 | 64 KB | Packed 4-bit weight input |
| `weightHighBit_` | 64 KB | 128 KB | Anti-quantized fp8 weight |
| `yF32Buffer_` | 64 KB | 128 KB | MMA fixpipe output (reuses weightHighBit_ space) |
| `bias_` | 192 KB | 1 KB | Bias input |
| `biasRescale_` | 193 KB | 1 KB | Rescaled bias |
| `maxB16Buffer_` | 194 KB | 2 KB | Per-block max exponent (ComputeMaxExp) |
| `scaleE8M0Buffer_` | 196 KB | 16 KB | MX E8M0 scales |
| `yFp8Buffer_` | 212 KB | 32 KB | Final quantized FP8/FP4 output |


---

## 4. Kernel Execution Flow

### 4.1 ResplitController — Dynamic M/N/K Splitting

The `ResplitController` runs on each AIC/AIV core and dynamically splits work based on group_list and L1 capacity.

**Global N Split for SwiGLU:**

SwiGLU splits the entire weight tensor along N into two equal halves at the global level:
- **Swish branch**: `W_swish = W[0 : N/2, :]` (first half of global N)
- **Gate branch**: `W_gate = W[N/2 : N, :]` (second half of global N)

Each half is then independently tiled into 128-column L1 blocks. One basic block processes:
- 128 columns from Swish branch → `l0cOutUbFirst_`
- 128 columns from Gate branch → `l0cOutUbSecond_`
- Combined: `N_L1 = 128 + 128 = 256`

**Example (N = 8192):**

```
Global Memory: Weight Tensor W[N, K] (N = 8192, transpose_w = true)

N = 0                              N = 4096                           N = 8192
│                                  │                                  │
▼                                  ▼                                  ▼
├──────────────────────────────────┼──────────────────────────────────┤
│ W[0:4096, :]                     │ W[4096:8192, :]                  │
│ ┌────┐┌────┐┌────┐     ┌────┐  │ ┌────┐┌────┐┌────┐     ┌────┐  │
│ │128 ││128 ││128 │ ... │128 │  │ │128 ││128 ││128 │ ... │128 │  │
│ │col ││col ││col │     │col │  │ │col ││col ││col │     │col │  │
│ └──┬─┘└──┬─┘└──┬─┘     └──┬─┘  │ └──┬─┘└──┬─┘└──┬─┘     └──┬─┘  │
│    │     │     │          │    │    │     │     │          │    │
│    └─────┴─────┴────┬─────┘    │    └─────┴─────┴────┬─────┘    │
│                     ↓          │                     ↓          │
│              l0cOutUbFirst_    │              l0cOutUbSecond_   │
│              (Swish output)    │              (Gate output)     │
│                     │          │                     │          │
└─────────────────────┴──────────┴─────────────────────┴──────────┘
                      │                                   │
                      └──────────────┬────────────────────┘
                                     ▼
                              SwiGLU(x, gate)
                                     │
                                     ▼
                                gluRes_
```

```cpp
// Fake code: ResplitController dynamic splitting
void ResplitController::Process() {
    // --- M split by group_list ---
    for (uint32_t groupIdx = 0; groupIdx < groupNum; groupIdx++) {
        uint64_t mSize = GetSplitValueFromGroupList(groupIdx);

        // Sub-divide M into L1-sized blocks
        uint64_t mBlkNum = CeilDivide(mSize, M_L1);  // M_L1 = 256
        uint64_t mL1Size = CeilDivide(mSize, mBlkNum);

        // --- N split across multicore ---
        // Global N is split into two halves for SwiGLU:
        //   - Swish: W[0 : N/2, :]
        //   - Gate:  W[N/2 : N, :]
        // Each half is tiled into 128-col L1 blocks.
        // One basic block = 128(Swish) + 128(Gate) = 256 = N_L1
        uint64_t basicBlockCount = CeilDivide(nSize, N_L1);  // N_L1 = 256

        for (uint64_t bbIdx = curBasicBlockId;
             bbIdx < basicBlockCount;
             bbIdx += tiling_->coreNum) {

            // nL1Size = 256 for cube (both Swish+Gate share A matrix)
            // Per-branch nSize = 128 for vector operations
            uint64_t nL1Size = Min(N_L1, nSize - bbIdx * N_L1);  // N_L1 = 256

            // --- K dynamic split ---
            uint64_t kbL1Size = ChooseKbL1Size(mL1Size, nL1Size);
            uint64_t kaL1Size = ComputeKaL1Size(mL1Size, nL1Size, kbL1Size);

            BasicBlockOffsetParam param;
            param.mL1Size = mL1Size;
            param.nL1Size = nL1Size;
            param.kaL1Size = kaL1Size;
            param.kbL1Size = kbL1Size;
            param.mOffset = groupOffset;
            param.nOffset = bbIdx * N_L1;

            // --- Execute one basic block ---
            if (ASCEND_IS_AIC) {
                IterateNzNkWithKAic(param);
            } else {
                IterateNzNkWithAiv(param);
            }
        }
    }
}
```

### 4.2 Init Phase

```cpp
// Fake code: Init phase for AIC and AIV
void GmmSwiGluQuantKernel::Init() {
    if (ASCEND_IS_AIC) {
        // --- Cube side ---
        cubeCompute_.MxA8W4Init(l1RemainSize, l1StartSize,
                                biasL1DbOffset_, biasL1_);
        // Cross-core sync: AIC signals ready to AIV
        SetAicToAiv<PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
        SetAicToAiv<PIPE_MTE1>(SYNC_AIC_AIV_FLAG);  // Double signal for two AIVs

    } else {
        // --- Vector side ---
        vecCompute_.Init(sharedInputWeight, y);

        // Initialize pipe events so first waits succeed immediately
        for (uint16_t idx = 0; idx < MTE3_V_BUFFER_NUM; idx++) {
            SetFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V + idx);
        }
        for (uint16_t idx = 0; idx < V_MTE3_BUFFER_NUM; idx++) {
            SetFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3 + idx);
        }
    }
}
```

### 4.3 AIV Path — `IterateNzNkWithAiv`

```cpp
// Fake code: AIV execution path for SwiGLU + Quant
void IterateNzNkWithAiv(BasicBlockOffsetParam& curParam,
                        BasicBlockOffsetParam& lastParam) {
    uint64_t kMte2Offset = 0;
    uint64_t curCvLoopIdx = cvLoopIdx_;

    // ============================================================
    // Pass 1: Pre-load first K chunk(s) for AIC MMA
    // ============================================================
    uint64_t preLoadLimit = Min(curParam.kSize,
                                DOUBLE_BUFFER_NUM * curParam.kbL1Size);
    for (; kMte2Offset < preLoadLimit; kMte2Offset += curParam.kbL1Size, cvLoopIdx_++) {

        // 1) Wait for previous V work to finish
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID_V_TO_MTE2 + (ubMte2LoopIdx_ & V_MTE2_MASK));

        // 2) MTE2: Copy packed 4-bit weight from GM to UB
        DataCopyPad2D(
            weightLowBit_[(ubMte2LoopIdx_ & V_MTE2_MASK) * weightLowBitSingleBufferSize],
            wGlobal_[(kMte2Offset + kL1Offset) * nAlign + nOffset * C0_SIZE],
            kMte2Size / C0_SIZE, nAlign * C0_SIZE, ...);

        // 3) Wait for AIC to be ready
        WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);

        // 4) V: Anti-quantize 4-bit → fp8 in UB
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V + (ubComputeLoopIdx_ & MTE3_V_MASK));
        AntiQuantProcessNzMxA8W4(biasRealSize, kRealSize, kGmOffset, curParam);

        // Barrier: ensure V compute done before MTE3 copies result out
        SetFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);

        // 5) MTE3: Copy anti-quantized weight from UB → L1
        if (likely(kRealSize > 0)) {
            CopyWeightHighBitForAligned(curParam.nL1Size, kRealSize,
                weightL1_[(cvLoopIdx_ & 1) * weightL1DbOffset_]);
        }
        if constexpr (hasBias) {
            if (kGmOffset == 0 && biasRealSize != 0) {
                DataCopy(biasL1_[(cvLoopIdx_ & 1) * biasL1DbOffset_ + biasL1Offset],
                         biasRescale_[...], biasRealSize);
            }
        }

        // Mark MTE3→L1 done
        SetFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V + (ubComputeLoopIdx_ & MTE3_V_MASK));
        ubComputeLoopIdx_++;

        // 6) Signal AIC that anti-quantized weight is ready
        SetAivToAic<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);

        // 7) Mark V work done, allow next MTE2 copy
        SetFlag<HardEvent::V_MTE2>(EVENT_ID_V_TO_MTE2 + (ubMte2LoopIdx_ & V_MTE2_MASK));
        ubMte2LoopIdx_++;
    }

    // ============================================================
    // Phase B: Process previous basic block's SwiGLU + Quant output
    // ============================================================
    if (curCvLoopIdx > 0) {
        // 8) Tell AIC to fixpipe previous block's L0C → UB
        SetAivToAic<PIPE_MTE3>(SYNC_AIV_MTE3_AIC_FIX_FLAG);

        // 9) Wait for AIC fixpipe to finish
        WaitAicToAiv<PIPE_V>(SYNC_AIC_FIX_AIV_VF_FLAG);

        // 10) V: SwiGLU compute on previous block's output
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V + (computeLoopIdx & MTE3_V_MASK));
        VFDoSwiglu(gluRes_, l0cOutUbFirst_, l0cOutUbSecond_,
                   lastParam.mL1Size, 128);

        // 11) MX Quantization
        ComputeMaxExp(gluRes_, maxB16Buffer_, totalDataInUb, loopDataNum);
        ComputeScale(maxB16Buffer_, scaleE8M0Buffer_, halfScaleLocalAddr,
                     totalScaleInUb, loopScaleNum);
        ComputeDataForQuantTargetFp8(
            gluRes_, halfScaleLocalAddr,
            yFp8Buffer_, totalDataInUb, loopDataNum);

        // 12) MTE3: Write quantized output + scales to GM
        CopyOutputFromUb2Gm(lastParam.mSize, yOffset, yFp8Buffer_);
        CopyScaleFromUb2Gm(lastParam.mSize, yScaleOffset, scaleE8M0Buffer_);

        // Mark MTE3→GM done
        SetFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V + (computeLoopIdx & MTE3_V_MASK));
        computeLoopIdx++;
    }

    // ============================================================
    // Pass 2: Continue loading remaining K chunks
    // ============================================================
    for (; kMte2Offset < curParam.kSize; kMte2Offset += curParam.kbL1Size, cvLoopIdx_++) {

        // Same steps 1-7 as Pass 1:
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID_V_TO_MTE2 + (ubMte2LoopIdx_ & V_MTE2_MASK));
        DataCopyPad2D(
            weightLowBit_[(ubMte2LoopIdx_ & V_MTE2_MASK) * weightLowBitSingleBufferSize],
            wGlobal_[(kMte2Offset + kL1Offset) * nAlign + nOffset * C0_SIZE],
            kMte2Size / C0_SIZE, nAlign * C0_SIZE, ...);

        WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);

        WaitFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V + (ubComputeLoopIdx_ & MTE3_V_MASK));
        AntiQuantProcessNzMxA8W4(biasRealSize, kRealSize, kGmOffset, curParam);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID_V_TO_MTE3);

        CopyWeightHighBitForAligned(curParam.nL1Size, kRealSize,
            weightL1_[(cvLoopIdx_ & 1) * weightL1DbOffset_]);
        if constexpr (hasBias) {
            if (kGmOffset == 0 && biasRealSize != 0) {
                DataCopy(biasL1_[(cvLoopIdx_ & 1) * biasL1DbOffset_ + biasL1Offset],
                         biasRescale_[...], biasRealSize);
            }
        }

        SetFlag<HardEvent::MTE3_V>(EVENT_ID_MTE3_TO_V + (ubComputeLoopIdx_ & MTE3_V_MASK));
        ubComputeLoopIdx_++;
        SetAivToAic<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID_V_TO_MTE2 + (ubMte2LoopIdx_ & V_MTE2_MASK));
        ubMte2LoopIdx_++;
    }
}
```

### 4.4 AIC Path — `IterateNzNkWithKAic`

```cpp
// Fake code: AIC execution path
void IterateNzNkWithKAic(BasicBlockOffsetParam& param) {
    // Step 1: Wait for AIV anti-quantization
    WaitAivToAic<PIPE_MTE1>(SYNC_AIV_AIC_FLAG);

    // Step 2: Load ScaleA and ScaleB from GM → L1 (MTE1→MTE2)
    for (uint64_t kOffset = 0; kOffset < param.kSize; kOffset += param.kaL1Size) {
        CopyMxScaleGmToL1(scaleAGlobal_, scaleAL1_[scaleL1Idx], param);
        CopyMxScaleGmToL1(scaleBGlobal_, scaleBL1_[scaleL1Idx], param);

        // Step 3: Load A matrix from GM → L1
        CopyAGmToL1(aGlobal_, al1_[al1Idx], param);

        // Pad L1 if K is not 64-aligned
        PadL1IfNotAlign(al1_[al1Idx], param);

        // Step 4: L1 → L0 load with MX scaling
        LoadAAndScaleL1ToL0(al1_[al1Idx], scaleAL1_[scaleL1Idx], l0a_, param);
        LoadBAndScaleL1ToL0(weightL1_[weightL1Idx], scaleBL1_[scaleL1Idx], l0b_, param);

        // Step 5: MMA compute on Cube unit
        // Cube processes nL1 = 256 because both branches share the same A matrix.
        // W_swish and W_gate are loaded into L0B sequentially,
        // but MMA outputs both 128-col halves using the same L0A.
        //   First 128 cols → l0cOutUbFirst_  (Swish)
        //   Second 128 cols → l0cOutUbSecond_ (Gate)
        MmadCompute(l0c_, l0a_, l0b_, param);

        // Step 6: Signal AIV
        SetAicToAiv<PIPE_MTE1>(SYNC_AIC_AIV_FLAG);
    }

    // Step 7: Fixpipe L0C → UB (output to AIV for logits multiply)
    if (cvLoopIdx_ > 0) {
        WaitAivToAic<PIPE_FIX>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
        GetTensorC(yF32Buffer_, lastOffsetParam);
        SetAicToAiv<PIPE_FIX>(SYNC_AIC_FIX_AIV_VF_FLAG);
    }
}
```

### 4.5 End Phase

```cpp
// Fake code: End phase
void GmmSwiGluQuantKernel::End() {
    if (ASCEND_IS_AIC) {
        if (cvLoopIdx_ > 0) {
            WaitAivToAic<PIPE_FIX>(SYNC_AIV_MTE3_AIC_FIX_FLAG);
            cubeCompute_.GetTensorC(yF32Buffer_, lastOffsetParam);
            SetAicToAiv<PIPE_FIX>(SYNC_AIC_FIX_AIV_VF_FLAG);
        }
        cubeCompute_.EndSync();

    } else {
        // Final AIV: process last block if any
        if (cvLoopIdx_ > 0) {
            WaitAicToAiv<PIPE_V>(SYNC_AIC_FIX_AIV_VF_FLAG);
            // ... same SwiGLU + Quant as IterateNzNkWithAiv ...
        }

        // Drain both AIC→AIV sync flags (AIC init set two of them)
        WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);
        WaitAicToAiv<PIPE_MTE3>(SYNC_AIC_AIV_FLAG);

        vecCompute_.End();  // drain remaining pipe events
    }
}
```

### 4.6 Synchronization Summary

| Flag ID | Direction | Pipe | Purpose |
|---------|-----------|------|---------|
| `SYNC_AIC_AIV_FLAG = 9` | AIC → AIV | MTE1 | Weight ready in L1 |
| `SYNC_AIV_AIC_FLAG = 8` | AIV → AIC | MTE3 | Anti-quantized weight ready |
| `SYNC_AIV_MTE3_AIC_FIX_FLAG = 4` | AIV → AIC | MTE3 | Request fixpipe L0C→UB |
| `SYNC_AIC_FIX_AIV_VF_FLAG = 3` | AIC → AIV | V / FIX | UB output ready for VF |

---

## 5. SwiGLU Algorithm

### 5.1 Mathematical Formula

```
SwiGLU(x, gate) = Swish(x) ⊙ gate
Swish(x) = x ⊙ sigmoid(x) = x / (1 + exp(-x))
```

SwiGLU requires **two matrix multiplications**:
- `Y_left  = X · W_left`   (Swish branch)
- `Y_right = X · W_right`  (Gate branch)
- `SwiGLU(X) = Swish(Y_left) ⊙ Y_right`

### 5.2 Vector Operations (Fake Code)

```cpp
// Fake code: SwiGLU computation per element
void VFDoSwigluAndQuantForMX(int8_t* outputDst, uint16_t* scaleDst,
                              bf16_t* firstSrc, bf16_t* secondSrc,
                              uint16_t mSize, uint16_t nSize) {
    // nSize = 128 (one branch half)
    // l0cOutUbFirst_ and l0cOutUbSecond_ each contain 128 columns
    const float scalarOne = 1.0f;

    for (uint16_t mIdx = 0; mIdx < mSize; mIdx++) {
        for (uint16_t vfBlockIdx = 0; vfBlockIdx < OneRowRepeatTimes; vfBlockIdx++) {
            uint32_t l0cOutOffset = mIdx * nSrcUbAligned + vfBlockIdx * sizePerRepeat;

            // --- Swish branch ---
            RegTensor<float> swishInput;
            DataCopy(swishInput, firstSrc + l0cOutOffset);

            RegTensor<float> negX, expNegX, denom, swishOutput;
            Muls(negX, swishInput, -scalarOne, mask);      // -x
            Exp(expNegX, negX, mask);                       // exp(-x)
            Adds(denom, expNegX, scalarOne, mask);          // 1 + exp(-x)
            Div(swishOutput, swishInput, denom, mask);      // x / (1 + exp(-x))

            // --- Gate branch ---
            RegTensor<float> gateInput;
            DataCopy(gateInput, secondSrc + l0cOutOffset);

            // --- SwiGLU: Swish(x) * gate ---
            RegTensor<float> swigluResult;
            Mul(swigluResult, swishOutput, gateInput, mask);

            // --- Cast to BF16 ---
            RegTensor<bf16_t> swigluBF16;
            Cast(swigluBF16, swigluResult, mask);

            // --- Store to gluRes_ ---
            uint32_t dstUbOffset = mIdx * nDstUbAligned + vfBlockIdx * sizePerRepeat;
            DataCopy(gluResAddr + dstUbOffset, swigluBF16, mask);
        }
    }
}
```

### 5.3 Key Observations

- Operates in **FP32 register precision** for numerical stability
- Inputs are `bfloat16_t` (from L0C), expanded to FP32, then contracted back
- `Div` uses `DIV_MODE` with `MaskMergeMode::ZEROING` for safe division
- The two matmul halves come from `l0cOutUbFirst_` (Swish) and `l0cOutUbSecond_` (Gate)

---

## 6. MxQuant Algorithm

### 6.1 Three-Step Pipeline

After SwiGLU produces `gluRes_` (BF16), quantization proceeds in three steps:

```
gluRes_ (bf16)
  → ComputeMaxExp      : Extract max exponent per 32B block
  → ComputeScale       : Derive shared MX scale and half-scale
  → ComputeDataForQuant: Scale data and cast to FP8/FP4
```

### 6.2 Step 1: ComputeMaxExp (Fake Code)

```cpp
// Fake code: Compute max exponent per 32B block
void ComputeMaxExp(bf16_t* srcAddr, uint16_t* maxExpAddr,
                   uint32_t totalCountInUB, uint16_t loopNum) {
    RegTensor<uint16_t> expMaskBF16;
    Duplicate(expMaskBF16, MAX_EXP_FOR_BF16);  // 0x7F80

    for (uint16_t i = 0; i < loopNum; i++) {
        MaskReg mask = UpdateMask<bf16_t>(totalCountInUB);

        // Load two interleaved chunks
        RegTensor<bf16_t> vdExp0, vdExp1;
        DataCopy(vdExp0, vdExp1, srcAddr, DIST_DINTLV_B16);

        // Mask out mantissa: keep only exponent bits
        RegTensor<uint16_t> vdExpExtract0, vdExpExtract1;
        And(vdExpExtract0, (RegTensor<uint16_t>&)vdExp0, expMaskBF16, mask);
        And(vdExpExtract1, (RegTensor<uint16_t>&)vdExp1, expMaskBF16, mask);

        // Per-element max exponent
        RegTensor<uint16_t> vdMaxExp;
        Max(vdMaxExp, vdExpExtract0, vdExpExtract1, mask);

        // Reduce to single max per 32B block
        ReduceMaxWithDataBlock(vdMaxExp, vdMaxExp, mask);

        // Store to maxExp_ buffer
        DataCopyUnAlign(maxExpAddr, vdMaxExp);
    }
}
```

**Output:** `maxExp_` — one `uint16_t` per 32B block, containing the maximum BF16 exponent field in that block.

### 6.3 Step 2: ComputeScale (Fake Code)

```cpp
// Fake code: Compute MX scale and half-scale per block
void ComputeScale(uint16_t* maxExpAddr, uint16_t* mxScaleLocalAddr,
                  uint16_t* halfScaleLocalAddr,
                  uint32_t totalScaleInUB, uint16_t loopNumScale) {
    RegTensor<uint16_t> expMask, scaleBias, fp8NanRegTensor;
    Duplicate(expMask, MAX_EXP_FOR_BF16);      // 0x7F80
    Duplicate(scaleBias, BF16_EXP_BIAS);       // 0x7F00 = 254 << 7
    Duplicate(fp8NanRegTensor, MAX_EXP_FOR_FP8); // 0x00FF

    RegTensor<uint16_t> maxExpValue, zeroRegTensor, nanRegTensor;
    Duplicate(maxExpValue, fpEmax_);           // Target format max (e.g. 0x0400)
    Duplicate(zeroRegTensor, 0);
    Duplicate(nanRegTensor, NAN_CUSTOMIZATION); // 0x7F81

    for (uint16_t i = 0; i < loopNumScale; i++) {
        MaskReg mask = UpdateMask<uint16_t>(totalScaleInUB);

        RegTensor<uint16_t> vdMaxExp;
        DataCopy(vdMaxExp, maxExpAddr);

        // Detect INF/NAN and zero
        MaskReg cmpResult, zeroMask;
        Compare(cmpResult, vdMaxExp, expMask, CMPMODE_NE);  // INF/NAN
        Compare(zeroMask, vdMaxExp, zeroRegTensor, CMPMODE_NE); // zero

        // Clamp maxExp to target format max
        MaskReg invalidDataMask;
        Compare(invalidDataMask, vdMaxExp, maxExpValue, CMPMODE_LE);
        Select(vdMaxExp, maxExpValue, vdMaxExp, invalidDataMask);

        // sharedExp = maxExp - fpEmax_
        RegTensor<uint16_t> sharedExp;
        Sub(sharedExp, vdMaxExp, maxExpValue, mask);

        // mxScale = sharedExp >> 7
        RegTensor<uint16_t> scaleValue;
        ShiftRight(scaleValue, sharedExp, SHR_NUM_FOR_BF16, mask);  // >> 7
        Select(scaleValue, scaleValue, fp8NanRegTensor, cmpResult);  // Preserve NAN
        Select(scaleValue, scaleValue, zeroRegTensor, zeroMask);     // Preserve zero

        // Store mxScale (E8M0)
        DataCopy(mxScaleLocalAddr, scaleValue);

        // halfScale = 0x7F00 - sharedExp
        //            = (254 << 7) - (mxScale << 7)
        //            = (254 - mxScale) << 7
        // In standard BF16 this represents 2^(127 - mxScale)
        RegTensor<uint16_t> halfScale;
        Sub(halfScale, scaleBias, sharedExp, mask);
        Select(halfScale, halfScale, nanRegTensor, cmpResult);
        Select(halfScale, halfScale, zeroRegTensor, zeroMask);

        // Store halfScale
        DataCopy(halfScaleLocalAddr, halfScale);
    }
}
```

**Key Derivation:**
```
mxScale     = sharedExp >> 7                    (E8M0 block scale, 0-255)
halfScale   = 0x7F00 - sharedExp
            = (254 - mxScale) << 7
            → represents 2^(127 - mxScale) in standard BF16
```

This is the reciprocal multiplier for `÷ 2^(mxScale - 127)`.

### 6.4 Step 3: ComputeDataForQuantTargetFp8 (Fake Code)

```cpp
// Fake code: Quantize BF16 data to FP8 with MX scaling
void ComputeDataForQuantTargetFp8(bf16_t* srcAddr, uint16_t* halfScaleLocalAddr,
                                  int8_t* outLocalAddr,
                                  uint32_t totalCountInUB, uint16_t loopNum) {
    for (uint16_t i = 0; i < loopNum; i++) {
        MaskReg mask = UpdateMask<bf16_t>(totalCountInUB);

        // Load BF16 SwiGLU result
        RegTensor<bf16_t> vdExp0, vdExp1;
        DataCopy(vdExp0, vdExp1, srcAddr, DIST_DINTLV_B16);

        // Load half-scale (reciprocal multiplier)
        RegTensor<uint16_t> halfScaleForMul;
        DataCopy(halfScaleForMul, halfScaleLocalAddr, DIST_E2B_B16);

        // Normalize: data * 2^(127 - mxScale)
        Mul(vdExp0, vdExp0, (RegTensor<bf16_t>&)halfScaleForMul, mask);
        Mul(vdExp1, vdExp1, (RegTensor<bf16_t>&)halfScaleForMul, mask);

        // Reorder for casting
        Interleave(vdExp0, vdExp1, vdExp0, vdExp1);

        // BF16 → FP32 (two layout paths)
        RegTensor<float> vdExp0FP32Zero, vdExp0FP32One;
        Cast(vdExp0FP32Zero, vdExp0, castTraitZero);
        Cast(vdExp0FP32One, vdExp0, castTraitOne);
        Interleave(vdExp0FP32Zero, vdExp0FP32One, vdExp0FP32Zero, vdExp0FP32One);

        // FP32 → FP8 with SAT (saturate)
        RegTensor<fp8_e4m3fn_t> vdExp0FP8;
        Cast(vdExp0FP8, vdExp0FP32Zero, castTrait32to8);

        // Store packed FP8
        DataCopy(outLocalAddr, (RegTensor<int8_t>&)vdExp0FP8, DIST_PACK4_B32);
    }
}
```

### 6.5 Step 3b: ComputeDataForQuantTargetFp4 (Fake Code)

```cpp
// Fake code: Quantize BF16 data to FP4 with MX scaling
void ComputeDataForQuantTargetFp4(bf16_t* srcAddr, uint16_t* halfScaleLocalAddr,
                                  int8_t* outLocalAddr,
                                  uint32_t totalCountInUB, uint16_t loopNum) {
    for (uint16_t i = 0; i < loopNum; i++) {
        MaskReg mask = UpdateMask<bf16_t>(totalCountInUB);

        RegTensor<bf16_t> vdExp0, vdExp1;
        DataCopy(vdExp0, vdExp1, srcAddr, DIST_DINTLV_B16);

        RegTensor<uint16_t> halfScaleForMul;
        DataCopy(halfScaleForMul, halfScaleLocalAddr, DIST_E2B_B16);

        // Normalize
        Mul(vdExp0, vdExp0, (RegTensor<bf16_t>&)halfScaleForMul, mask);
        Mul(vdExp1, vdExp1, (RegTensor<bf16_t>&)halfScaleForMul, mask);

        Interleave(vdExp0, vdExp1, vdExp0, vdExp1);

        // BF16 → FP4 (direct, no FP32 intermediate)
        RegTensor<fp4x2_e2m1_t> vdExp0FP4;
        Cast(vdExp0FP4, vdExp0, castTrait);

        // Store packed FP4
        DataCopy(outLocalAddr, (RegTensor<int8_t>&)vdExp0FP4, DIST_PACK4_B32);
    }
}
```

### 6.6 Scale Layout Transpose (Fake Code)

```cpp
// Fake code: Transpose MX scales from flat to row-major
void TransMxScaleLayout(uint16_t mSize, uint16_t scaleBlockN) {
    for (uint16_t mIdx = 0; mIdx < mSize; ++mIdx) {
        MaskReg mask = UpdateMask<int8_t>(scaleBlockN);

        RegTensor<int8_t> vreg0;
        UnalignReg u0;

        // Load flat scale block
        auto srcUb = quantScaleOutputInUbAddr + mIdx * scaleBlockN;
        DataCopyUnAlignPre(u0, srcUb);
        DataCopyUnAlign(vreg0, u0, srcUb);

        // Store to row-major layout at stride 32 bytes per row
        auto dstUb = quantScaleBlockOutputInUbAddr + mIdx * ONE_BLK_SIZE;
        DataCopy(dstUb, vreg0, DIST_NORM_B8, mask);
    }
}
```

---

## 7. Integration: SwiGLU + Quant Fusion Pipeline

### 7.1 Overall Data Flow

```
L0C Output (fp32, from AICore MMA)
  ├── l0cOutUbFirst_  ──┐  (from W[0:N/2, :], Swish branch)
  └── l0cOutUbSecond_ ──┘  (from W[N/2:N, :], Gate branch)
         ↓
    VFDoSwigluAndQuantForMX
         ↓
    ┌─────────────────────────────────────┐
    │ 1. SwiGLU (per element, fp32 regs)  │
    │    Swish(x) * gate                  │
    │    → gluRes_ (bf16)                 │
    │                                     │
    │ 2. ComputeMaxExp                    │
    │    Per-32B-block max bf16 exponent  │
    │    → maxExp_                        │
    │                                     │
    │ 3. ComputeScale                     │
    │    sharedExp = maxExp - fpEmax      │
    │    mxScale = sharedExp >> 7         │
    │    halfScale = 0x7F00 - sharedExp   │
    │    → quantScaleOutput_ + halfScale_ │
    │                                     │
    │ 4. ComputeDataForQuantTargetFp8/4   │
    │    data * 2^(127 - mxScale)         │
    │    bf16 → fp32 → fp8/fp4 (SAT)      │
    │    → quantOutput_                   │
    └─────────────────────────────────────┘
         ↓
    TransMxScaleLayout
         ↓
    DataCopy (UB → GM)
         ↓
    y (fp8/fp4) + yScale (e8m0)
```

### 7.2 Fusion Entry Point (Fake Code)

```cpp
// Fake code: Main SwiGLU + Quant entry point
void SwiGluQuant::operator()(BlockShape blockShape, BlockCoord blockCoord) {
    // Extract tile dimensions
    singleM_ = Get<MNK_M>(blockShape);
    singleN_ = Get<MNK_N>(blockShape);

    // Split M between two AIV sub-blocks
    uint64_t halfSingleM = singleM_ / 2;

    // Each AIV sub-block processes its M slice
    for (uint32_t subBlockIdx = 0; subBlockIdx < 2; subBlockIdx++) {
        uint64_t mOffset = subBlockIdx * halfSingleM;
        uint64_t mSizeInVec = (subBlockIdx == 1)
                              ? singleM_ - halfSingleM
                              : halfSingleM;

        if (mSizeInVec == 0) continue;

        // Step 1: SwiGLU + Quant for this M slice
        VFDoSwigluAndQuantForMX(
            quantOutputInUbAddr + mOffset * nDstUbAligned,
            quantScaleOutputInUbAddr + mOffset * scaleBlockN,
            l0cOutUbFirstAddr + mOffset * nSrcUbAligned,
            l0cOutUbSecondAddr + mOffset * nSrcUbAligned,
            mSizeInVec, singleN_);
    }

    // Step 2: Transpose MX scales to row-major
    TransMxScaleLayout(singleM_, scaleBlockN);

    // Step 3: Sync before DMA
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);

    // Step 4: Write quantized output to GM
    CopyOutputFromUb2Gm(singleM_, yOffset, quantOutput_);

    // Step 5: Write MX scales to GM
    CopyScaleFromUb2Gm(singleM_, yScaleOffset, quantScaleBlockOutput_);

    // Step 6: Sync after DMA
    SetFlag<HardEvent::MTE3_V>(0);
    WaitFlag<HardEvent::MTE3_V>(0);
}
```

### 7.3 Buffer Reuse Strategy

| Stage | Input Buffer | Output Buffer | Reuse Strategy |
|-------|-------------|---------------|----------------|
| MMA | AL1, WeightL1 | L0C | L0C → UB fixpipe |
| SwiGLU | l0cOutUbFirst_, l0cOutUbSecond_ | gluRes_ | Separate UB allocation |
| ComputeMaxExp | gluRes_ | maxExp_ | maxExp_ is much smaller |
| ComputeScale | maxExp_ | halfScale_, quantScaleOutput_ | In-place scale computation |
| ComputeDataForQuant | gluRes_, halfScale_ | quantOutput_ | quantOutput_ overwrites gluRes_ region if needed |

---

## 8. Key Invariants and Constraints

### 8.1 Data Type Constraints

| Tensor | Data Type | Format | Shape |
|--------|-----------|--------|-------|
| x (activation) | `fp8_e4m3fn` | 2D | `[M, K]` |
| w (weight) | `fp4_e2m1` | FRACTAL_NZ | `[E, K/32, N/16, 16, 32]` |
| scale (weight) | `fp8_e8m0` | 4D | `[E, N, CeilDiv(K,64), 2]` |
| perTokenScale | `fp8_e8m0` | 3D | `[M, CeilDiv(K,64), 2]` |
| y (output) | `fp8_e4m3fn` or `fp4_e2m1` | 2D | `[M, N]` |
| yScale | `fp8_e8m0` | 2D | `[M, CeilDiv(N,32)]` |

### 8.2 Alignment Constraints

- `K % 32 == 0`
- `N % 128 == 0`
- `K >= 32`
- `N >= 128`
- Weight stored in `FRACTAL_NZ` or `FRACTAL_NZ_C0_32`

### 8.3 Transpose Constraints

- `transpose_x` must be `false`
- `transpose_w` must be `true`

### 8.4 Scale Constraints

- MX block size: 32 elements
- Scale stored per 32-element block along K
- E8M0 scale bias: 127
- `halfScale` pivot: `0x7F00 = 254 << 7`

### 8.5 L1 Capacity Constraints

- Total L1: 512 KB
- WeightL1: 64 KB × 2 (double buffered)
- ScaleL1: 32 KB × 4 (double buffered, A + B)
- AL1: 124/128 KB × 2 (depends on bias presence)
- BiasL1: 4 KB × 2 (optional)

### 8.6 Source Code References

| Component | File Path |
|-----------|-----------|
| Tiling | `ops-transformer/gmm/grouped_matmul_finalize_routing/op_host/op_tiling/arch35/grouped_matmul_finalize_routing_weight_quant_tiling.cpp` |
| Kernel Main | `ops-transformer/gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_vcv_basic_block.h` |
| Cube Compute | `ops-transformer/gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_cube_compute.h` |
| Vector Compute | `ops-transformer/gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block/gmm_fr_weight_quant_vec_compute.h` |
| VF Kernels | `ops-transformer/gmm/grouped_matmul_finalize_routing/op_kernel/arch35/weight_quant_basic_block/gmm_fr_mx_a8w4_vf.h` |
| SwiGLU + Quant | `ops-transformer/gmm/common/cgmct/epilogue/block_epilogue_swiglu_mx_quant.h` |
| SwiGLU Analysis | `DiscussionForum/block_epilogue_swiglu_quant_mx_analysis.md` |
## 9. Address Space Layout at Each Stage

This section documents how the data layout changes at each stage of the SwiGLU + Quant pipeline, with element-level addressing.

#### Step 1: After L0C → UB (AIC Fixpipe)

1024B contiguous FP32 — 256 elements, 4B each (N = 256 = 128 + 128).

```
Addr:  0x00      0x04      0x08      0x0C      ...  0x3FC
       ├─────────┼─────────┼─────────┼─────────┼────┤
       │y1y1y1y1 │y2y2y2y2 │y3y3y3y3 │y4y4y4y4 │ ...│
       └─────────┴─────────┴─────────┴─────────┴────┘
       yN = one FP32 element (4 bytes), all contiguous and valid
```

#### Step 2: After SwiGLU

256B valid BF16 result — 128 elements, 2B each, stored with padding.

```
Addr:  0x00    0x02    0x04    0x06    ...  0xFE   0x100      0x400
       ├───────┼───────┼───────┼───────┼────┤├──────────────────┤
       │y1y1   │y2y2   │y3y3   │y4y4   │ ...││xxxx...xxxxxxxxxx │
       └───────┴───────┴───────┴───────┴────┘└──────────────────┘
       yN = one BF16 element (2 bytes)
       x  = unused / padding byte
```

#### Step 3: After ComputeMaxExp (Step 1 of Quant)

`maxScaleBuffer` — contiguous `uint16_t`, one per 32B block.

```
Addr:  0x00    0x02    0x04    0x06    ...
       ├───────┼───────┼───────┼───────┤
       │s1s1   │s2s2   │s3s3   │s4s4   │ ...
       └───────┴───────┴───────┴───────┘
       sN = max exponent of block N (uint16_t)
```

#### Step 4: After ComputeScale (Step 2 of Quant)

E8M0 scales are co-located with the SwiGLU result in UB.

```
Addr:  0x00                  0x100               0x200
       ├──────────────────┬──────────────────┬──────────────────┤
       │y1y1 y2y2 ... yNyN│xxxx xxxx ... xxxx│s1 s2 s3 ... s256 │
       │   256B valid     │    256B unused   │    256B scales   │
       └──────────────────┴──────────────────┴──────────────────┘
       yN = one BF16 element (2 bytes)
       sN = one E8M0 scale value (1 byte)
       x  = unused / padding byte
```

#### Step 5: After ComputeDataForQuant (Step 3 of Quant)

Quantized output buffer — 256B contiguous packed FP8 (256 elements × 1B).

```
Addr:  0x00  0x01  0x02  0x03  ...  0xFF
       ├─────┼─────┼─────┼─────┼────┤
       │ q1  │ q2  │ q3  │ q4  │ ...│
       └─────┴─────┴─────┴─────┴────┘
       qN = one quantized FP8 byte
```

#### Step 6: After TransMxScaleLayout

`scaleE8M0Buffer` — row-major transposed scales.

```
Addr:  0x00  0x01  0x02  0x03  0x04      0x20
       ├─────┼─────┼─────┼─────┼───────────────────────────────┤
       │ s1  │ s2  │ s3  │ s4  │xxxx xxxx ... xxxxxxxxxxxxxxxx │
       └─────┴─────┴─────┴─────┴───────────────────────────────┘
       sN = one E8M0 scale byte (1 byte)
       x  = padding byte
```
