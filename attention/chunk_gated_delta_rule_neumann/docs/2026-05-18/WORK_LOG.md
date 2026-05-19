# 2026-05-18 Work Log

## BF16 resident/RHS cleanup

- Scope: Stage1 C64 fused RHS dtype cleanup only. Did not restore GM `attnWsGm_` final RHS path and did not route inverse output through GM.
- Changed `arch32` and mirrored `arch35`:
  - `GBKCompute()` / `VBetaCompute()` now only write BF16 RHS workspaces: `gBKWsGm_` and `vBetaWsGm_`.
  - Removed FP16 RHS GM workspaces and casts: `gBKHalfWsGm_`, `vBetaHalfWsGm_`.
  - `FusedRhsAICProcess()` consumes BF16 RHS via raw MMAD from packed L1/L0, and final output still uses `CopyL0CMatrixToGmBf16()`.
  - Final resident inverse is BF16 L1/TSCM (`aicBf16L1ResidentInv_`); inverse compute scratch remains FP16.
  - Removed host workspace sizing for the deleted FP16 RHS buffers.
- Source hashes after edit:
  - `op_kernel/arch32/chunk_gated_delta_rule_neumann_stage1.h`: `e04f8a190e552bbc2991f160833ec7893fadb752478cc19af26a74662d9e76df`
  - `op_kernel/arch35/chunk_gated_delta_rule_neumann_stage1.h`: `e04f8a190e552bbc2991f160833ec7893fadb752478cc19af26a74662d9e76df`
  - `op_host/chunk_gated_delta_rule_neumann_tiling.cpp`: `65bae56548a7e1691a8fece175846d093708615a3317fef083e5cd1af20df80a`

## Verification

- Build PASS:
  - `bash ../../build.sh --pkg --soc=ascend910_93 --ops=chunk_gated_delta_rule,chunk_gated_delta_rule_neumann -j16`
- Install PASS:
  - `../../build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet --install-path=/home/developer/Ascend/cann-9.0.0/opp`
- Target smoke PASS:
  - `S=64`, seed 0, `B=1,nk=4,nv=4,dk=64,dv=64`
  - `S=256`, seed 0, `B=1,nk=4,nv=4,dk=64,dv=64`
  - `S=1024`, seed 0, `B=1,nk=4,nv=4,dk=64,dv=64`
- Tail smoke PASS:
  - `S=65`, seed 0, `B=1,nk=4,nv=4,dk=64,dv=64`
- Compare set PASS:
  - `CGDR_GOLDEN_CHUNK_SIZE=64 TORCH_EXTENSIONS_DIR=/tmp/cgdr_torch_ext ASCEND_CUSTOM_OPP_PATH=/home/developer/Ascend/cann-9.0.0/opp/vendors/custom_transformer bash tests/pytest/test_run.sh compare`
  - Result: `16 passed`.
