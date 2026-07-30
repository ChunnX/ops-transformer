# BSA FP8 PV Rescale-Mad Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In Arch35 BSA `BlockMmadPV`, on the FP8 full_quant path only, after the first N-chunk mmad and before `deqScalar` fixpipe, insert an additive cycle probe: L1 InitConst (64×64 + 64×128 fp32) → MTE1 → 2× Mad(64,128,64).

**Architecture:** All probe code lives in `block_mmad_pv_arch35_ABf16_C_to_UB.hpp` under `#ifdef BSA_PV_RESCALE_PROBE_FP8`. The call site is further gated by `if constexpr (FULL_QUANT_FP8)`. Probe writes L0C front half; existing PV/fixpipe keep using L0C back half. RescaleO is unchanged (additive measurement).

**Tech Stack:** AscendC (`InitConstValue`, `LoadData`, `Mmad`), BSA TLA `LocalTensor` buffers, compile flag `-DBSA_PV_RESCALE_PROBE_FP8`.

**Spec:** `attention/docs/superpowers/specs/2026-07-30-bsa-fp8-rescale-mad-probe-design.md`

## Global Constraints

- Macro name: `BSA_PV_RESCALE_PROBE_FP8` only (never `BSA_PV_RESCALE_PROBE`)
- Additive only: do not bypass or modify `rescaleO` / `mm2ToReFlag`
- Hot path keeps InitConst every PV (`nL0Itr == 0`)
- Runtime gate: `if constexpr (FULL_QUANT_FP8)` (`ElementA == fp8_e4m3fn_t`)
- Default build must not define the macro (CMake add is opt-in for puncture builds)
- Probe shapes: A `64×64` fp32, B `64×128` fp32; Mad `m=64,n=128,k=64` twice with `cmatrixInitVal=true`
- Events: L0A `EVENT_ID 0`, L0B `EVENT_ID 2`

---

## File map

| File | Responsibility |
|------|----------------|
| `attention/block_sparse_attention/op_kernel/attn_infra/gemm/block/block_mmad_pv_arch35_ABf16_C_to_UB.hpp` | Probe constants, L1/L0C alloc, `RunRescaleMadProbe()`, call site |
| `attention/block_sparse_attention/op_host/CMakeLists.txt` | Optional `-DBSA_PV_RESCALE_PROBE_FP8` for puncture builds |

No changes to `block_sparse_attention_kernel_arch35_full_quant.h`, rescaleO, or QK Softmax.

---

### Task 1: Scaffold probe hook (ifdef + empty method + gated call)

**Files:**
- Modify: `attention/block_sparse_attention/op_kernel/attn_infra/gemm/block/block_mmad_pv_arch35_ABf16_C_to_UB.hpp`

**Interfaces:**
- Consumes: existing `FULL_QUANT_FP8`, `nL0Itr`, `operator()` N-loop
- Produces: `RunRescaleMadProbe()` empty stub; call site ready for Task 2/3

- [ ] **Step 1: Add probe constants after `FULL_QUANT_FP8`**

Insert immediately after `static constexpr bool FULL_QUANT_FP8 = ...;` (currently ~line 117):

```cpp
#ifdef BSA_PV_RESCALE_PROBE_FP8
    static constexpr uint32_t PROBE_M = 64;
    static constexpr uint32_t PROBE_N = 128;
    static constexpr uint32_t PROBE_K = 64;
    static constexpr uint32_t PROBE_L1A_BYTES = PROBE_M * PROBE_K * sizeof(float); // 16KB
    static constexpr uint32_t PROBE_L1B_BYTES = PROBE_K * PROBE_N * sizeof(float); // 32KB
    static constexpr uint32_t PROBE_L0A_EVENT_ID = 0;
    static constexpr uint32_t PROBE_L0B_EVENT_ID = 2; // match PV: B uses +2
#endif
```

- [ ] **Step 2: Add empty `RunRescaleMadProbe` before `SparseVBaseTileL1FullLoad`**

```cpp
#ifdef BSA_PV_RESCALE_PROBE_FP8
    __aicore__ inline void RunRescaleMadProbe()
    {
        // filled in Task 3
    }
#endif
```

- [ ] **Step 3: Insert gated call immediately before `// fixpipe`**

Locate the end of the m/k loops (~line 362) and the `// fixpipe` comment (~line 363). Insert:

```cpp
#ifdef BSA_PV_RESCALE_PROBE_FP8
            if constexpr (FULL_QUANT_FP8) {
                if (nL0Itr == 0) {
                    RunRescaleMadProbe();
                }
            }
#endif
            // fixpipe
```

Do **not** move or alter the existing `WaitCrossCoreSync` / fixpipe / `deqScalar` block.

- [ ] **Step 4: Sanity-check without the macro**

Confirm that with `BSA_PV_RESCALE_PROBE_FP8` undefined, a search of the translation unit after preprocessing would strip all new code. Locally: `rg BSA_PV_RESCALE_PROBE_FP8` shows only ifdef guards; default CMake has no `-D`.

- [ ] **Step 5: Commit**

```bash
git add attention/block_sparse_attention/op_kernel/attn_infra/gemm/block/block_mmad_pv_arch35_ABf16_C_to_UB.hpp
git commit -m "$(cat <<'EOF'
feat(bsa): scaffold FP8 PV rescale-mad probe hook

Add BSA_PV_RESCALE_PROBE_FP8 constants, empty RunRescaleMadProbe,
and FULL_QUANT_FP8-gated call before fixpipe.
EOF
)"
```

---

### Task 2: Allocate probe L1 + L0C scratch in constructor

**Files:**
- Modify: `attention/block_sparse_attention/op_kernel/attn_infra/gemm/block/block_mmad_pv_arch35_ABf16_C_to_UB.hpp`

**Interfaces:**
- Consumes: `l1BufAddrStart`, `l1ATileM/K`, `l1BTileK/N`, `l1ABufNum`, `l1BBufNum`, `PROBE_*_BYTES`
- Produces: members `probeL1A_`, `probeL1B_`, `l0CScratch_`

- [ ] **Step 1: Add member tensors in `protected:` under existing L0 members**

```cpp
#ifdef BSA_PV_RESCALE_PROBE_FP8
    AscendC::LocalTensor<float> probeL1A_;
    AscendC::LocalTensor<float> probeL1B_;
    AscendC::LocalTensor<float> l0CScratch_;
#endif
```

- [ ] **Step 2: Initialize in constructor after the L0C loop**

```cpp
#ifdef BSA_PV_RESCALE_PROBE_FP8
        {
            const uint32_t probeBase =
                l1BufAddrStart +
                l1ATileM * l1ATileK * sizeof(ElementA) * l1ABufNum +
                l1BTileK * l1BTileN * sizeof(ElementB) * l1BBufNum;
            // L1 tail must fit 48KB probe buffers
            ASCENDC_ASSERT((probeBase + PROBE_L1A_BYTES + PROBE_L1B_BYTES) <= ArchTag::L1_SIZE,
                { AscendC::printf("BSA FP8 probe L1 overflow\n"); });
            probeL1A_ = resource.l1Buf.template GetBufferByByte<float>(probeBase);
            probeL1B_ = resource.l1Buf.template GetBufferByByte<float>(
                probeBase + PROBE_L1A_BYTES);
            l0CScratch_ = resource.l0CBuf.template GetBufferByByte<float>(0); // front half
        }
#endif
```

If `ASCENDC_ASSERT` is unavailable in this tree, use a compile-time-friendly alternative already used nearby, or omit assert and keep a `static_assert`/`printf` comment — prefer matching existing BSA assert style (`rg ASCENDC_ASSERT` in `block_sparse_attention/op_kernel`).

- [ ] **Step 3: Commit**

```bash
git add attention/block_sparse_attention/op_kernel/attn_infra/gemm/block/block_mmad_pv_arch35_ABf16_C_to_UB.hpp
git commit -m "$(cat <<'EOF'
feat(bsa): allocate FP8 probe L1 and L0C scratch buffers

Map 64x64 and 64x128 fp32 L1 after P/V and L0C front-half scratch.
EOF
)"
```

---

### Task 3: Implement `RunRescaleMadProbe` body

**Files:**
- Modify: `attention/block_sparse_attention/op_kernel/attn_infra/gemm/block/block_mmad_pv_arch35_ABf16_C_to_UB.hpp`

**Interfaces:**
- Consumes: `probeL1A_`, `probeL1B_`, `l0CScratch_`, `l0ATensor[0]`, `l0BTensor[0]`, `BYTE_PER_C0`, `BYTE_PER_FRACTAL`, `C0_NUM_PER_FRACTAL` from `bsa_base_defs.hpp`
- Produces: working probe sequence InitConst → MTE1 → Mad×2

- [ ] **Step 1: Replace empty `RunRescaleMadProbe` with full body**

```cpp
#ifdef BSA_PV_RESCALE_PROBE_FP8
    __aicore__ inline void RunRescaleMadProbe()
    {
        // --- L1 InitConst (MTE2) ---
        constexpr uint32_t BLOCKS_A = PROBE_L1A_BYTES / 32; // 512
        constexpr uint32_t BLOCKS_B = PROBE_L1B_BYTES / 32; // 1024
        AscendC::InitConstValueParams<float> paramsA(1, BLOCKS_A, 0, 1.0f);
        AscendC::InitConstValueParams<float> paramsB(1, BLOCKS_B, 0, 1.0f);
        AscendC::InitConstValue(probeL1A_, paramsA);
        AscendC::InitConstValue(probeL1B_, paramsB);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(PROBE_L0A_EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(PROBE_L0A_EVENT_ID);

        // --- L0A/L0B views as float (reuse stage 0 physical buffer) ---
        AscendC::LocalTensor<float> l0AFp32 =
            l0ATensor[0].template ReinterpretCast<float>();
        AscendC::LocalTensor<float> l0BFp32 =
            l0BTensor[0].template ReinterpretCast<float>();

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(PROBE_L0A_EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(PROBE_L0B_EVENT_ID);

        constexpr uint32_t ELE_PER_C0 = BYTE_PER_C0 / sizeof(float);           // 8
        constexpr uint32_t ELE_PER_FRACTAL = BYTE_PER_FRACTAL / sizeof(float); // 128

        AscendC::LoadData2DParamsV2 loadA;
        loadA.mStartPosition = 0;
        loadA.kStartPosition = 0;
        loadA.mStep = PROBE_M / C0_NUM_PER_FRACTAL;              // 4
        loadA.kStep = PROBE_K / ELE_PER_C0;                      // 8
        loadA.srcStride = (PROBE_M * PROBE_K) / ELE_PER_FRACTAL; // 32
        loadA.dstStride = loadA.srcStride;
        loadA.ifTranspose = false;
        AscendC::LoadData(l0AFp32, probeL1A_, loadA);

        AscendC::LoadData2DParamsV2 loadB;
        loadB.mStartPosition = 0;
        loadB.kStartPosition = 0;
        loadB.mStep = PROBE_K / C0_NUM_PER_FRACTAL;              // 4
        loadB.kStep = PROBE_N / ELE_PER_C0;                      // 16
        loadB.srcStride = (PROBE_K * PROBE_N) / ELE_PER_FRACTAL; // 64
        loadB.dstStride = (PROBE_K * PROBE_N) / ELE_PER_FRACTAL;
        loadB.ifTranspose = true;
        AscendC::LoadData(l0BFp32, probeL1B_, loadB);

        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(PROBE_L0A_EVENT_ID);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(PROBE_L0B_EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(PROBE_L0A_EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(PROBE_L0B_EVENT_ID);

        AscendC::MmadParams mmadParams;
        mmadParams.m = PROBE_M;
        mmadParams.n = PROBE_N;
        mmadParams.k = PROBE_K;
        mmadParams.unitFlag = 0;
        mmadParams.cmatrixInitVal = true;
#if (__CCE_AICORE__ == 310)
        mmadParams.disableGemv = true;
#endif

        AscendC::Mmad(l0CScratch_, l0AFp32, l0BFp32, mmadParams); // mad 1
        AscendC::Mmad(l0CScratch_, l0AFp32, l0BFp32, mmadParams); // mad 2

        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(PROBE_L0A_EVENT_ID);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(PROBE_L0B_EVENT_ID);
    }
#endif
```

If `LoadData2DParamsV2` / `InitConstValueParams` signatures differ on the local toolchain, adjust to the AscendC headers in the SDK while keeping shapes and event order from the spec.

- [ ] **Step 2: Commit**

```bash
git add attention/block_sparse_attention/op_kernel/attn_infra/gemm/block/block_mmad_pv_arch35_ABf16_C_to_UB.hpp
git commit -m "$(cat <<'EOF'
feat(bsa): implement FP8 PV rescale-mad probe body

InitConst L1, MTE1 load to L0A/L0B, dual fp32 Mad into L0C scratch.
EOF
)"
```

---

### Task 4: Opt-in CMake flag for puncture builds

**Files:**
- Modify: `attention/block_sparse_attention/op_host/CMakeLists.txt`

**Interfaces:**
- Consumes: existing `BUILD_OPS_RTY_KERNEL` / `add_ops_compile_options` block
- Produces: optional `-DBSA_PV_RESCALE_PROBE_FP8`

- [ ] **Step 1: Add the define only under the RTY kernel options block used for puncture**

Current file (lines 18–25) has no probe define. For a puncture build, add:

```cmake
if (BUILD_OPS_RTY_KERNEL)
    add_ops_compile_options(
            OP_NAME BlockSparseAttention
            OPTIONS --cce-auto-sync=off
                    -Wno-deprecated-declarations
                    -Werror
                    -fpermissive
                    -DBSA_PV_RESCALE_PROBE_FP8
    )
else()
    add_modules_sources(OPTYPE block_sparse_attention ACLNNTYPE aclnn_exclude)
endif()
```

**Note:** Spec says default build should not open the macro. If this CMake path is the team's only experimental kernel build, keep the `-D` here while documenting that production/open builds use the `else()` branch without the define. If both branches need a clean default, wrap the `-D` in a CMake option:

```cmake
option(BSA_ENABLE_FP8_RESCALE_PROBE "Enable FP8 PV rescale mad probe" OFF)
...
$<$<BOOL:${BSA_ENABLE_FP8_RESCALE_PROBE}>:-DBSA_PV_RESCALE_PROBE_FP8>
```

Prefer the `option(...)` form when the RTY path is also used for non-puncture work.

- [ ] **Step 2: Commit**

```bash
git add attention/block_sparse_attention/op_host/CMakeLists.txt
git commit -m "$(cat <<'EOF'
build(bsa): add opt-in BSA_PV_RESCALE_PROBE_FP8 compile flag

Enable FP8 PV rescale-mad probe only for puncture kernel builds.
EOF
)"
```

---

### Task 5: Build / run / profiler verification

**Files:**
- No code changes (verification only)

- [ ] **Step 1: Build without probe macro**

Build BSA Arch35 (regular + full_quant) with the macro **off**. Expected: compile success; binary behavior unchanged vs baseline branch tip before probe commits.

- [ ] **Step 2: Build with `-DBSA_PV_RESCALE_PROBE_FP8`**

Rebuild with the flag from Task 4. Expected: compile success for FP8 full_quant instantiation.

- [ ] **Step 3: Run one FP8 BSA case**

Use an existing FP8 full_quant pytest / ACLNN case for BlockSparseAttention (team's usual `blockShapeX==128`, `blockShapeY % 256 == 0` constraints). Expected: kernel launches and finishes; numerical mismatch vs golden is acceptable.

- [ ] **Step 4: msprof compare on / off**

Capture Cube timeline for the same case with macro on vs off. Expected with macro on: visible InitConst / MTE1 / two Mmad bursts after PV mmad and before fixpipe on the first N-chunk. RescaleO Vec work still present (additive).

- [ ] **Step 5: Optional non-quant sanity**

If a non-quant Arch35 case is built with the same macro: confirm no probe Cube segment (gate is `FULL_QUANT_FP8`).

- [ ] **Step 6: Commit verification notes only if the team wants a short status in the plan checklist**

Do not commit logs/binaries. Mark Task 5 checkboxes done in the plan file when verification passes, in a follow-up docs commit if desired:

```bash
git add attention/docs/superpowers/plans/2026-07-30-bsa-fp8-rescale-mad-probe.md
git commit -m "docs(bsa): mark FP8 rescale-mad probe verification complete"
```

---

## Spec coverage self-check

| Spec requirement | Task |
|------------------|------|
| Additive, keep rescaleO | Task 1 call site; no full_quant.h edit |
| Macro `BSA_PV_RESCALE_PROBE_FP8` | Tasks 1–4 |
| `if constexpr (FULL_QUANT_FP8)` | Task 1 Step 3 |
| Per-PV InitConst + MTE1 + 2×Mad | Task 3 |
| L1 64×64 / 64×128 after P/V | Task 2 |
| L0C front-half scratch | Task 2 |
| Events 0 / 2 | Task 3 |
| Opt-in CMake | Task 4 |
| Profiler / run verification | Task 5 |

No TBD/TODO placeholders remain in task steps.
