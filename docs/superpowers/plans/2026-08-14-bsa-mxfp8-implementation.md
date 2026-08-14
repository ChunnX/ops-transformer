# BSA MXFP8 Full Quant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a BSA MXFP8 OCP full-quant path (Q/K/V/P = e4m3fn, scales = e8m0) alongside existing MXFP4, with `TILE_GROUP_N=4` and `PV_RE_DELAY_N=8`.

**Architecture:** Clone the MXFP4 full-quant stack (`BsaMXFP4FullQuantKernelArch35` + QK/PV/epilogue/VF), introduce parallel `MXFP8` / `MXFP8Kernel` namespaces and `*mxfp8*` files, keep operand roles K→L0A / Q→L0B, and k-split QK (D→2×64) + PV (S2→2×128) to fit L0.

**Tech Stack:** AscendC / Atlas A5, CANN ops-transformer BSA, tiling keys, MX LoadData + MMAD.

**Spec:** `docs/superpowers/specs/2026-08-14-bsa-mxfp8-design.md`

## Global Constraints

- OCP only (no CX); QuantMode `MXFP8_OCP_QUANT = 4`
- Tile geometry unchanged: qs=128, kvs=256, D=128
- `TILE_GROUP_N = 4`, `PV_RE_DELAY_N = 8`, `DELAY_PSCALE_N = 3`, `PRELOAD_TASK_CACHE_SIZE = 9`
- L1 P ring = 8 slots × 32KB; UB P = 2 × 16KB half-tiles
- Data C0: `FP8_C0_ELEMS = 32`; scale plane geometry unchanged vs MXFP4
- Seed: data `0x38` / scale `0x7F` (e4m3×e8m0 = 1.0)
- Do not break MXFP4 path

---

## File map

**Create (clone from mxfp4):**
- `op_kernel/arch35/block_sparse_attention_kernel_arch35_mxfp8_full_quant.h`
- `op_kernel/attn_infra/gemm/block/block_mmad_qk_arch35_mxfp8.hpp`
- `op_kernel/attn_infra/gemm/block/block_mmad_pv_arch35_mxfp8.hpp`
- `op_kernel/attn_infra/gemm/block/block_mmad_copy_global_max_l1_to_ub_arch35_mxfp8.hpp`
- `op_kernel/attn_infra/gemm/tile_common/copy_l1_to_l0_mx_fp8_a5.hpp`
- `op_kernel/attn_infra/epilogue/block/block_epilogue_online_softmax_arch35_reg_low_prec_fp16_mxfp8.hpp`
- `op_kernel/attn_infra/epilogue/block/block_epilogue_compute_pscale_arch35_mxfp8.hpp`
- `op_kernel/attn_infra/epilogue/block/block_epilogue_rescale_o_arch35_reg_high_prec_mxfp8.hpp`
- `op_kernel/attn_infra/epilogue/block/block_epilogue_copy_global_max_ub_to_l1_arch35_mxfp8.hpp`
- `op_kernel/attn_infra/epilogue/block/mxfp8_vf/*` (from `mxfp4_vf`, adapt cast to e4m3)

**Modify:**
- `arch35/kernel_utils.hpp` — `MXFP8Kernel` with `TILE_GROUP_N=4`
- `gemm/block/block_mmad_arch35_utils.hpp` — `namespace MXFP8` + static_asserts
- `epilogue/block/block_epilogue_arch35_utils.hpp` — `namespace MXFP8`
- `gemm/bsa_gemm_dispatch_policy.hpp`, `gemm/block/bsa_block_mmad.hpp`
- `epilogue/bsa_epilogue_dispatch_policy.hpp`, `epilogue/block/bsa_block_epilogue.hpp`
- `block_sparse_attention_tilingkey.h`, `block_sparse_attention.cpp`, `block_sparse_attention_kernel_interface.cpp`
- `op_host/block_sparse_attention_tiling.{h,cpp}`, `block_sparse_attention_def.cpp`

---

### Task 1: MXFP8 memory constant namespaces

**Files:**
- Modify: `block_mmad_arch35_utils.hpp`, `block_epilogue_arch35_utils.hpp`, `kernel_utils.hpp`

- [ ] **Step 1:** Add `namespace MXFP8` to gemm utils with L1/L0 tables from spec §4–§5 (`L1_P_BUF_CNT=8`, `L1_P_BUF_SIZE=32K`, offsets per spec, `FP8_C0_ELEMS=32`, seed `0x3838`/`0x7F7F`, `static_assert` L1≤512K, L0A/B≤64K).
- [ ] **Step 2:** Add `namespace MXFP8` to epilogue utils with UB table from spec §6 (`UB_P_BUF_SIZE=16K` half-tile, `UB_LOCAL_GROUP_MAX_CNT=8`, shifted localGlobalMax offsets).
- [ ] **Step 3:** Add `namespace MXFP8Kernel { TILE_GROUP_N = 4; }` in `kernel_utils.hpp`.
- [ ] **Step 4:** Commit when user requests (do not commit unless asked).

---

### Task 2: L1→L0 fp8 copy + QK/PV blocks with k-split

**Files:**
- Create: `copy_l1_to_l0_mx_fp8_a5.hpp`, `block_mmad_qk_arch35_mxfp8.hpp`, `block_mmad_pv_arch35_mxfp8.hpp`, `block_mmad_copy_global_max_l1_to_ub_arch35_mxfp8.hpp`

- [ ] **Step 1:** Clone `copy_l1_to_l0_mx_a5.hpp` → fp8 variant; use `MXFP8::FP8_C0_ELEMS` / `NZ_C0_ELEMS`; keep MX yStep as k/64.
- [ ] **Step 2:** Clone QK mxfp4 → mxfp8; Element = `fp8_e4m3fn_t`; for each 128-col QK half, loop k(D) in 2×64 with L0A/L0B 8KB slots; accumulate L0C (`initC` only on first k-segment of first half as existing semantics).
- [ ] **Step 3:** Clone PV mxfp4 → mxfp8; split k(S2) into 2×128; Vᵀ slot 18KB; seed fill e4m3/e8m0 1.0; assert `ElementA == fp8_e4m3fn_t`.
- [ ] **Step 4:** Clone gmax L1→UB block with MXFP8 buffer constants.

---

### Task 3: Epilogue + VF + main kernel

**Files:**
- Create mxfp8 epilogue headers + `mxfp8_vf/*` + `block_sparse_attention_kernel_arch35_mxfp8_full_quant.h`

- [ ] **Step 1:** Clone VF dir; rename guards/includes; change packed-fp4 cast/gather to e4m3fn (1B/elem); keep OCP path only where CX branches exist.
- [ ] **Step 2:** Clone softmax/pscale/rescale/gmax-ub→l1; switch `EpMXFP4`→`EpMXFP8`, `GmMXFP4`→`GmMXFP8`; P half-tile double buffer (2×16KB).
- [ ] **Step 3:** Clone main kernel; `PV_RE_DELAY_N=8`, `PRELOAD=9`, `using namespace MXFP8Kernel`, class `BsaMXFP8FullQuantKernelArch35`.

---

### Task 4: Dispatch + host + tiling keys

**Files:**
- Modify: dispatch policies, `bsa_block_mmad.hpp`, `bsa_block_epilogue.hpp`, tilingkey, cpp interface, host tiling, def

- [ ] **Step 1:** Add `MmadAtlasA5BsaPVMxfp8` and includes for mxfp8 blocks.
- [ ] **Step 2:** Add tiling keys `QMFP8OCP_*` (new nibble, OCP only) and host `MXFP8_OCP_QUANT=4`, dtype allow `DT_FLOAT8_E4M3FN`.
- [ ] **Step 3:** Wire `BsaInferInterfaceMXFP8FullQuant` in kernel interface + `block_sparse_attention.cpp` `#elif` branches.

---

### Task 5: Smoke / UT scaffolding

- [ ] **Step 1:** Extend host UT for new QuantMode if existing mxfp4 tiling UT pattern is clear.
- [ ] **Step 2:** Note device pytest (`tests/mxfp8_pytest`) as follow-up if full golden not feasible in this pass.

---

## Execution

User requested implementation immediately → **Inline Execution** in this session on branch `mxfp8_0814` (already the feature baseline; no extra worktree).
