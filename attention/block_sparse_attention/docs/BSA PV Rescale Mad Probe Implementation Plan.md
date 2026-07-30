# BSA PV Rescale Mad Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In Arch35 BSA `BlockMmadPV`, after the first N-chunk mmad and before fixpipe, insert a `#ifdef BSA_PV_RESCALE_PROBE` cycle probe: L1 InitConst (64×64 + 64×128 fp32) → MTE1 → 2× Mad(64,128,64).

**Architecture:** Inline probe in `block_mmad_pv_arch35_ABf16_C_to_UB.hpp` only. L1 tail after P/V; L0A/L0B stage 0 reused; L0C front half (`@0`) as scratch so PV L0C (back half) is untouched. fixpipe and rescaleO unchanged.

**Tech Stack:** AscendC (`InitConstValue`, `LoadData`, `Mmad`), BSA TLA LocalTensor buffers, compile flag `-DBSA_PV_RESCALE_PROBE`.

**Spec:** `attention/block_sparse_attention/docs/BSA PV 后 Rescale Mad 性能穿刺设计.md`

## Global Constraints

- Cycle-only probe; numerical correctness not required
- Do not bypass or modify rescaleO / `mm2ToReFlag`
- Probe runs once per PV `operator()`, only when `nL0Itr == 0`
- Both mads: `(M=64, N=128, K=64)`, `cmatrixInitVal=true` each time
- L1 layout sizes stay fp32 (16KB + 32KB); if L0 fp32 Load/Mmad fails to compile on platform, fall back per spec §5.3 but keep L1 byte layout
- Default build: macro OFF

---

## File Map

| File | Role |
|------|------|
| Modify: `attention/block_sparse_attention/op_kernel/attn_infra/gemm/block/block_mmad_pv_arch35_ABf16_C_to_UB.hpp` | All probe code under `#ifdef BSA_PV_RESCALE_PROBE` |
| Reference (read-only): `attention/common/op_kernel/arch35/flash_attention_score_block_cube_mla_fullquant.h` (~727–768) | `InitConstValue` + fp32 L1→L0A/L0B pattern |
| Reference (read-only): `.../tile_common/copy_l1_to_l0a_a5.hpp`, `copy_l1_to_l0b_a5.hpp` | `LoadData2DParamsV2` field meanings |
| Reference (read-only): `.../tile_common/bsa_tile_mmad.hpp` | `AscendC::Mmad` / `MmadParams` |

No new files. No host/tiling changes.

---

### Task 1: Scaffold probe constants, L1 base member, and empty hook

**Files:**
- Modify: `attention/block_sparse_attention/op_kernel/attn_infra/gemm/block/block_mmad_pv_arch35_ABf16_C_to_UB.hpp`

**Interfaces:**
- Consumes: existing `operator()` N-loop
- Produces: probe size/event constants; empty `RunRescaleMadProbe()`; call site gated by `nL0Itr == 0`

- [x] **Step 1: Add probe constants (inside `#ifdef BSA_PV_RESCALE_PROBE`)**

Near the other `static constexpr` members of `BlockMmadTla` (after `FULL_QUANT_FP8`), add:

```cpp
#ifdef BSA_PV_RESCALE_PROBE
    static constexpr uint32_t PROBE_M = 64;
    static constexpr uint32_t PROBE_N = 128;
    static constexpr uint32_t PROBE_K = 64;
    static constexpr uint32_t PROBE_L1A_BYTES = PROBE_M * PROBE_K * sizeof(float); // 16KB
    static constexpr uint32_t PROBE_L1B_BYTES = PROBE_K * PROBE_N * sizeof(float); // 32KB
    static constexpr uint32_t PROBE_L0A_EVENT_ID = 0;
    static constexpr uint32_t PROBE_L0B_EVENT_ID = 2; // match PV: B uses +2
#endif
```

- [x] **Step 2: Add empty `RunRescaleMadProbe` method declaration/definition**

Still inside the class, add:

```cpp
#ifdef BSA_PV_RESCALE_PROBE
    __aicore__ inline void RunRescaleMadProbe()
    {
        // filled in Task 2
    }
#endif
```

- [x] **Step 3: Wire call site before fixpipe**

In `operator()`, immediately after the `mL0Itr`/`kL0Itr` loops close and **before** the `// fixpipe` comment (current ~line 363), insert:

```cpp
#ifdef BSA_PV_RESCALE_PROBE
            if (nL0Itr == 0) {
                RunRescaleMadProbe();
            }
#endif
            // fixpipe
```

- [x] **Step 4: Sanity-check insertion without enabling the macro**

Confirm default build (no `-DBSA_PV_RESCALE_PROBE`) still compiles the BSA Arch35 path the same as before (ifdef strips all new code except nothing — actually all new code is ifdef'd, so zero behavior change).

- [x] **Step 5: Commit** (combined with Task 2 in one commit)

```bash
git add attention/block_sparse_attention/op_kernel/attn_infra/gemm/block/block_mmad_pv_arch35_ABf16_C_to_UB.hpp
git commit -m "$(cat <<'EOF'
feat(bsa): scaffold PV rescale-mad probe hook under BSA_PV_RESCALE_PROBE

EOF
)"
```

---

### Task 2: Implement L1 InitConst + MTE1 LoadData + 2× Mmad

**Files:**
- Modify: `attention/block_sparse_attention/op_kernel/attn_infra/gemm/block/block_mmad_pv_arch35_ABf16_C_to_UB.hpp` (`RunRescaleMadProbe` body)

**Interfaces:**
- Consumes: constructor `l1BufAddrStart` / `resource`; members `l1ATileM/K`, `l1BTileK/N`, `l1ABufNum`, `l1BBufNum`, `l0ATensor[0]`, `l0BTensor[0]`
- Produces: members `probeL1A_`, `probeL1B_`, `l0CScratch_`; working `RunRescaleMadProbe()`

**L0C scratch note:** Constructor maps `l0CTensor[i]` to `L0C_HALF_BUF_SIZE + ...` only. Front-half scratch must be `GetBufferByByte<float>(0)`, not a view of `l0CTensor[0]`.

- [x] **Step 1: Bind probe L1/L0C tensors in constructor**

After existing L1/L0 setup in the constructor:

```cpp
#ifdef BSA_PV_RESCALE_PROBE
        {
            const uint32_t probeBase =
                l1BufAddrStart +
                l1ATileM * l1ATileK * sizeof(ElementA) * l1ABufNum +
                l1BTileK * l1BTileN * sizeof(ElementB) * l1BBufNum;
            probeL1A_ = resource.l1Buf.template GetBufferByByte<float>(probeBase);
            probeL1B_ = resource.l1Buf.template GetBufferByByte<float>(
                probeBase + PROBE_L1A_BYTES);
            l0CScratch_ = resource.l0CBuf.template GetBufferByByte<float>(0);
        }
#endif
```

Members:

```cpp
#ifdef BSA_PV_RESCALE_PROBE
    AscendC::LocalTensor<float> probeL1A_;
    AscendC::LocalTensor<float> probeL1B_;
    AscendC::LocalTensor<float> l0CScratch_;
#endif
```

- [x] **Step 2: Fill `RunRescaleMadProbe` body**

```cpp
#ifdef BSA_PV_RESCALE_PROBE
    __aicore__ inline void RunRescaleMadProbe()
    {
        // --- L1 InitConst (MTE2) ---
        // block count = elements * sizeof(float) / 32
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

        // L1 zN → L0A zN (no transpose). Fractal math for fp32:
        // ELE_NUM_PER_C0 = 8, ELE_NUM_PER_FRACTAL = 128 (512B fractal / 4)
        constexpr uint32_t ELE_PER_C0 = 32 / sizeof(float);      // 8
        constexpr uint32_t ELE_PER_FRACTAL = 512 / sizeof(float); // 128
        constexpr uint32_t C0_PER_FRACTAL = 16;

        AscendC::LoadData2DParamsV2 loadA;
        loadA.mStartPosition = 0;
        loadA.kStartPosition = 0;
        loadA.mStep = PROBE_M / C0_PER_FRACTAL;                 // 4
        loadA.kStep = PROBE_K / ELE_PER_C0;                     // 8
        loadA.srcStride = (PROBE_M * PROBE_K) / ELE_PER_FRACTAL; // 32
        loadA.dstStride = loadA.srcStride;
        loadA.ifTranspose = false;
        AscendC::LoadData(l0AFp32, probeL1A_, loadA);

        AscendC::LoadData2DParamsV2 loadB;
        loadB.mStartPosition = 0;
        loadB.kStartPosition = 0;
        // L0B nZ with transpose: K×N in L1 zN → L0B
        loadB.mStep = PROBE_K / C0_PER_FRACTAL;                 // 4
        loadB.kStep = PROBE_N / ELE_PER_C0;                     // 16
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

**Implementation notes (do not skip):**

1. If `LoadData2DParamsV2` stride fields fail compile-time asserts or hang at runtime, copy the exact stride formulas from a working fp32 path in `flash_attention_score_block_cube_mla_fullquant.h` (`LoadDataToL0A` / `LoadDataToL0B`) and adapt sizes to 64/128/64.
2. If fp32 `Mmad` fails to compile on A5, keep L1 as float InitConst for cycle, but `ReinterpretCast` L0 views to `half`/`bfloat16` and mad in that type — document the fallback in a one-line comment above the mad. L1 byte sizes must remain 16KB+32KB.
3. `InitConstValue` may need `AscendC::` prefix or ADL — match whatever FA fullquant uses in this tree (`InitConstValue` without namespace in that file). Prefer `AscendC::InitConstValue` if both compile.

- [ ] **Step 3: Compile with probe ON** (deferred with Task 3 per user request)

Build BSA Arch35 kernel with `-DBSA_PV_RESCALE_PROBE` (add to the op kernel compile flags / local cmake cache the team already uses for experimental defines).  

Expected: compile success. If fail, fix API/types per notes above before proceeding.

- [x] **Step 4: Commit**

```bash
git add attention/block_sparse_attention/op_kernel/attn_infra/gemm/block/block_mmad_pv_arch35_ABf16_C_to_UB.hpp
git commit -m "$(cat <<'EOF'
feat(bsa): implement PV rescale-mad cycle probe (L1/MTE1/2x mad)

EOF
)"
```

---

### Task 3: Smoke run + msprof cycle check

**Files:**
- No code changes expected (only if smoke fails and needs event/API fix in the same hpp)

**Interfaces:**
- Consumes: Task 2 probe enabled binary
- Produces: short note in plan checklist / optional comment in spec status

- [ ] **Step 1: Smoke run one BSA Arch35 case with probe ON**

Run the team's usual single-op BSA case (same shape that exercises Arch35 PV).  

Expected: process completes (output values may be wrong — OK). No hang/deadlock on `M_MTE1` / `MTE2_MTE1` flags.

- [ ] **Step 2: msprof / pipeline compare**

Capture two profiles:

1. Baseline: macro OFF  
2. Probe: macro ON  

Confirm the AIC timeline gains a segment after first PV mmad and before fixpipe containing InitConst/MTE2, LoadData/MTE1, and two Mmad.

- [ ] **Step 3: If smoke/profile fails, fix events only**

Common fixes:

- Missing `Wait M_MTE1` before probe Load (PV just Set those flags)
- InitConst without `MTE2_MTE1` barrier before Load
- Probe using same event IDs as an in-flight L0C fixpipe wait — keep probe on event 0/2 only; do not touch `l0CEventId` (`bufId+2`)

Re-run Step 1–2 after fix.

- [ ] **Step 4: Commit any fixes (skip if none)**

```bash
git add attention/block_sparse_attention/op_kernel/attn_infra/gemm/block/block_mmad_pv_arch35_ABf16_C_to_UB.hpp
git commit -m "$(cat <<'EOF'
fix(bsa): stabilize PV rescale-mad probe sync for smoke/msprof

EOF
)"
```

- [ ] **Step 5: Mark spec status**

Update the design doc header status from `已批准（待实现）` to `已实现（穿刺可开关）` in:

`attention/block_sparse_attention/docs/BSA PV 后 Rescale Mad 性能穿刺设计.md`

```bash
git add "attention/block_sparse_attention/docs/BSA PV 后 Rescale Mad 性能穿刺设计.md"
git commit -m "$(cat <<'EOF'
docs: mark BSA PV rescale-mad probe design as implemented

EOF
)"
```

---

## Spec Coverage Self-Review

| Spec requirement | Task |
|------------------|------|
| Insert after PV mmad, before fixpipe | Task 1 Step 3 |
| Once per PV, `nL0Itr==0` only | Task 1 Step 3 |
| L1 64×64 + 64×128 fp32 after P/V | Task 2 constructor probeBase |
| InitConst | Task 2 RunRescaleMadProbe |
| MTE1 → L0A/L0B | Task 2 LoadData |
| Two mad (64,128,64), init each time | Task 2 Mmad ×2 |
| L0C front-half scratch | Task 2 `l0CScratch_` @0 |
| Keep fixpipe/rescaleO | Task 1 (call only adds before fixpipe) |
| `#ifdef BSA_PV_RESCALE_PROBE` | All tasks |
| Compile / smoke / msprof | Task 2 Step 3, Task 3 |

No placeholders remaining after Step 2b solidification. Types: `float` L1/L0C scratch; L0A/B `ReinterpretCast<float>` from existing `ElementA/B` buffers.
