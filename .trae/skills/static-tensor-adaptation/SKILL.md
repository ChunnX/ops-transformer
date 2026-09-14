---
name: "static-tensor-adaptation"
description: "Adapts AscendC kernel from dynamic tensor (TPipe/TBuf/TQue) to static tensor programming (LocalTensor with constexpr offsets + Mutex/CrossCore sync). Invoke when user asks to convert kernel to static tensor programming or adapt flash_attn-style static patterns."
---

# Static Tensor Programming Adaptation

Convert AscendC kernel code from dynamic tensor programming (TPipe/TBuf/TQue/BufferManager) to static tensor programming (LocalTensor with constexpr offsets, Mutex for intra-core sync, CrossCoreSetFlag/WaitFlag for inter-core sync).

## When to Invoke

- User asks to "adapt static tensor programming" for a kernel
- User asks to convert TPipe/TBuf/TQue to static LocalTensor
- User asks to follow flash_attn arch35 static tensor pattern
- User wants to remove TPipe dependency from kernel code

## Reference Implementation

`/home/wangke293/ops-transformer_qfa_mla/attention/flash_attn/op_kernel/arch35/` — the gold standard for static tensor programming.

## Step-by-Step Adaptation Guide

### 1. Identify Buffer Categories

Classify all buffers into:

| Category | Static Method | Notes |
|----------|--------------|-------|
| UB (cross-core) | `LocalTensor<uint8_t>(TPosition::VECIN, 0, size)` | Shared between AIC and AIV at same offsets |
| UB (vec-only) | `LocalTensor<T>(TPosition::VECIN, OFFSET_OF_MEMBER(UbLayout, member), SIZE_OF_MEMBER(UbLayout, member))` | Use struct layout + OFFSET_OF_MEMBER macro |
| L1 (cross-core) | `LocalTensor<uint8_t>(TPosition::A1, offset, size)` | Shared between AIC and AIV |
| L1 (cube-only) | Same pattern with L1Layout struct | Q/KV buffers |
| L0C | `LocalTensor<uint8_t>(TPosition::CO1, 0, totalSize)` | Static, no BufferManager |
| L0A/L0B | Keep `BufferManager` + `BuffersPolicyDB` with EXTERNAL IDs | Matmul API requires this |
| GM | Keep `GlobalTensor` | No change needed |

### 2. Define Sync IDs

Must be consistent across cube block, vec block, and kernel:

```cpp
// Cross-core sync (CROSS_CORE_SYNC_MODE = 4, AIV0_AIV1_OFFSET = 16)
static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4U;
static constexpr uint32_t CC_MM_0 = 0U;   // bmm1 result slot 0
static constexpr uint32_t CC_MM_1 = 1U;   // bmm1 result slot 1
static constexpr uint32_t CC_MM_2 = 2U;   // bmm2 result slot
static constexpr uint32_t CC_L1P_0 = 3U;  // L1 P slots (AIV→AIC)
static constexpr uint32_t CC_L1P_1 = 4U;
static constexpr uint32_t CC_L1P_2 = 5U;

// AIC (cube) MUST use both id and id + AIV0_AIV1_OFFSET for all cross-core flags
// AIV (vec) only uses id (framework handles AIV0/AIV1 mapping)

// Vec internal Mutex IDs (0-27 shared with Event IDs)
static constexpr uint32_t UB_OUT_VEC2_RES_EVENT0 = 0;   // vec2 result V↔MTE3 (Mutex)
static constexpr uint32_t UB_OUT_VEC1_RES_EVENT0 = 2;   // vec1 result V↔MTE3 (Mutex)
static constexpr uint32_t UB_OUT_VEC1_RES_EVENT1 = 3;
static constexpr uint32_t UB_OUT_LSE_OUT_EVENT0 = 4;    // LSE output (Mutex)
static constexpr uint32_t UB_OUT_LSE_OUT_EVENT1 = 5;
static constexpr uint32_t UB_IN_MASK_EVENT0 = 6;        // mask input (Mutex)
static constexpr uint32_t UB_IN_MASK_EVENT1 = 7;
static constexpr uint32_t UB_IN_QSCALE_EVENT0 = 8;      // Q scale input (SetFlag/WaitFlag)
static constexpr uint32_t UB_IN_QSCALE_EVENT1 = 9;
```

**CRITICAL**: Mutex ID and SetFlag/WaitFlag Event ID share the same hardware pool (0-27). Mutex uses 0-7, SetFlag/WaitFlag uses 8-9. Do NOT mix Mutex and SetFlag/WaitFlag on the same ID.

### 3. Cube Block Adaptation

Remove:
- `TPipe *tPipe_`
- `BufferManager<L1> *l1BufferManagerPtr_`
- `BufferManager<L0C>` and L0C `BuffersPolicy`
- All `Buffer<>` type aliases for cross-core communication

Add:
- `LocalTensor<uint8_t> l1PBuffers_, l1QBuffers_, l1KvBuffers_` with `OFFSET_OF_MEMBER`/`SIZE_OF_MEMBER`
- `LocalTensor<uint8_t> l0CBuffers_` at `TPosition::CO1`
- `uint32_t qL1BufId_, kvL1BufId_, l0cBufId_, mmResUbBufId_` for manual rotation
- `InitCrossCoreSync()` (empty for cube) and `UnInitCrossCoreSync()` (wait all pending flags)

Keep:
- `BufferManager<L0A/L0B>` with `BuffersPolicyDB<..., IdSource::EXTERNAL>` — Matmul API requires TBuf-like interface
- GM tensors and copy helpers

### 4. Vec Block Adaptation

Remove:
- `TPipe *tPipe_`
- All `TBuf<>` and `TQue<>` members
- Dynamic `AllocEventID()`/`ReleaseEventID()`

Add:
- `struct UbLayout { ... }` with all vec-specific buffers as struct members
- `static_assert(sizeof(UbLayout) <= 192 * 1024, "UB buffer too large")`
- Static `LocalTensor` for each buffer using `OFFSET_OF_MEMBER`/`SIZE_OF_MEMBER`

**Mutex pattern for vec2 result (CRITICAL)**:
- FlashUpdate + LastDivNew + Cast must be in SEPARATE Mutex Lock/Unlock blocks (matching flash_attn pattern)
- Do NOT hold Mutex across CopyOut (CopyOut has its own Mutex Lock/Unlock)
- LastDivNew only runs when isFirstS2Loop && isLastS2Loop (single S2 block case)

**AllocEventID**: Pre-arm `SetFlag<MTE3_V>(0)` — this is REQUIRED because Mutex and Event Flag share hardware, and the initial Mutex::Lock<PIPE_V>(0) needs the "MTE3 done" state.

### 5. TBuf Dependency (FP8 inputs)

When ProcessVec1Vf API requires `TBuf<>*`:
- For half/bfloat16: pass `nullptr` (API skips Get<uint8_t>() for non-FP8 types)
- For FP8: use pad TBuf approach — allocate `TBuf<>` of `sizeof(UbLayout)` to advance TPipe's UB pointer, then allocate vselrIndexes after

### 6. Entry Point

Keep `TPipe tPipe` as local variable (needed for `GetTPipePtr()` global initialization):
```cpp
TPipe tPipe;
Kernel op;
op.Init(...);  // No TPipe param
op.Process();
```

### 7. Known Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Missing TPipe object | `Svm_vma_fault` at UB base address | Keep `TPipe tPipe` in entry point |
| UB address conflict (vselrIndexes) | `error code = 271: address out of bounds` | Use pad TBuf to advance TPipe pointer |
| AIV0_AIV1_OFFSET mismatch | Deadlock on AIC side | AIC must use `id + AIV0_AIV1_OFFSET`, AIV must NOT |
| Mutex/EventFlag ID conflict | All tests fail | Do NOT use same ID for both Mutex and SetFlag/WaitFlag |
| Missing SetFlag pre-arm for Mutex | B16+ tests fail | AllocEventID must SetFlag<MTE3_V>(0) to initialize Mutex state |
| LastDivNew Mutex wrapping | NZ B4 tests fail | Match flash_attn pattern exactly: separate Mutex blocks |
| B1/B2 low-batch precision (~81%) | Attn output PctRlt=81.7%, LSE correct | Under investigation — likely cube block K reuse or cross-core sync timing |
