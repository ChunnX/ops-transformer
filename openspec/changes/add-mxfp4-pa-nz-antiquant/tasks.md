## 1. Contract and host routing

- [x] 1.1 Add FIA mode-6 PA-NZ checker declarations and implementation in `dequant_checker.{h,cpp}` for the exact `[P,N,G,Bs]` scale/offset contract, dense storage requirement, E8M0 scale, Q_T offset, `D % 32 == 0`, and K/V symmetry.
- [x] 1.2 Add FIA negative host checks that reject mode-6 PA BNBD/BBH, logical `[B,N,S,G]`, token-major `[P,N,Bs,G]`, flattened parameters, invalid offset dtype and non-divisible D before tiling.
- [x] 1.3 Update `fused_infer_attention_score_tiling_impl.cpp`, `fused_infer_attention_score_tiling.cpp` and `fused_infer_attention_score_template_tiling_key.h` so the exact target tuple emits a mode-6 K-per-token PA-NZ template key and all non-target tuples preserve their old key selection.
- [x] 1.4 Audit FIA operator definition dtype registration; only modify `fused_infer_attention_score_def.cpp` if the exact E8M0-scale/FP4-KV/Q_T-offset tuple is not already registered.

## 2. Shared PA_NZ parameter loading

- [x] 2.1 Add an internal page-segment iterator to the FIA antiquant processor. It must produce `(p,u,len,copied)` from `bIdx`, `s2BatchOffset`, `copyTotalS`, `maxBlockNumPerSeq`, `kvCacheBlockSize` and `blockTableGm` and must be reusable by a later TOKEN_MAJOR policy.
- [x] 2.2 Add a typed GROUP_MAJOR PA-NZ parameter-copy primitive in `flash_attention_score_antiquant_processor.h`. Its source coordinate is `[((p*N+n)*G+g)*Bs+u]`; for every group it uses contiguous `DataCopyPad(blockCount=1)` and writes the corresponding raw UB row `[G,SAlign]`, avoiding an unaligned destination-stride encoding.
- [x] 2.3 Instantiate the primitive for scale as `uint8_t`; retain byte-addressed E8M0 source offsets and do not divide or reinterpret them as Q_T elements.
- [x] 2.4 Instantiate the primitive for a non-null offset as Q_T; retain Q_T element addressing and do not reuse scale byte offsets or raw-scale buffers.
- [ ] 2.5 Define and allocate buffers from `E=AlignUp(G*AlignUp(copyTotalS,32),128)`, including raw scale, raw offset, expanded scale and expanded offset capacity; replace insufficient fixed-size assumptions when necessary.
- [ ] 2.6 Initialize rounded raw-tail slots to E8M0 `0x00` for scale and zero for offset, and add any required queue/event synchronization before VF reads them.
- [ ] 2.7 Verify target CANN `DataCopyPad` behavior for both `uint8_t` and Q_T against a `G>1`, partial-page segment test; confirm that the per-group `blockCount=1` copies preserve the required `[G,SAlign]` coordinates.

## 3. FIA kernel and VF integration

- [x] 3.1 In `LoadAntiquantParamsPerTokenGroup`, add the `isPageAttentionAntiquant && isKvCacheNz` branch before the old non-PA NZ branch; leave the old non-PA NZ code byte-for-byte behaviorally unchanged.
- [x] 3.2 Enqueue/dequeue raw PA-NZ scale, call `AntiqScalePerTokenGroupByVF`, and retain the expanded `[G,SAlign]` scale in a scale-only TBuf.
- [x] 3.3 When offset exists, enqueue/dequeue raw PA-NZ offset, call `AntiqOffsetByVFNz`, and retain expanded `[G,SAlign]` BF16 offset in a distinct TBuf.
- [x] 3.4 Audit `AntiquantVec` so its PA-NZ `copyLoop` scale and offset bases are both `copyLoopIdx * copySplitS`; do not multiply this slice base by group count, byte size or scale expansion factor.
- [x] 3.5 Audit/modify `vf_antiquant_w4.h` so FP16 offsets are converted to BF16 before Add, BF16 offsets are correctly consumed, and symmetric/asymmetric calls select separate `hasOffset` paths.
- [x] 3.6 Confirm `CopyKVPageAttention`, `CopyKVPaNz` and `CopyAntiquantResToL1BaseAPI` are reused unchanged except for passing the new parameter route; prove no non-NZ/BNBD/BBH branch is reached.

## 5. Tests and acceptance evidence

- [x] 5.1 Add a producer-side/reference helper that visibly executes `[B,N,S,G] -> [P,N,Bs,G] -> [P,N,G,Bs]` for scale and offset independently; unit-test its coordinates with a non-identity `blockTable`.
- [x] 5.2 Add an independent direct-example CPU oracle that reconstructs logical per-token-group values from final PA-NZ storage through `blockTable`, decodes both packed FP4 nibbles, applies BF16 operand arithmetic, and applies the symmetric/asymmetric formula.
- [x] 5.3 Add FIA symmetric FP16, asymmetric FP16 and asymmetric BF16 PA-NZ direct calls with `B>=2`, `N>=2`, `D>=64`, non-monotonic physical pages, cross-page `copyTotalS`, and a non-32-aligned tail.
- [x] 5.4 Use values distinct by `(p,n,g,u)` for K scale, V scale, K offset and V offset; include negative offsets so a wrong page/group, a missing Add or K/V alias is observable.
- [x] 5.5 Add FIA negative and regression tests: BNBD/BBH rejection, logical/token-major/flattened parameter shape rejection, bad offset dtype, `D % 32 != 0`, and unchanged non-PA mode 6 symmetric behavior.
- [x] 5.6 Add a one-command validation wrapper that builds the Arch35 kernel and host UT, then invokes the direct ACLNN example against an explicitly supplied custom package. It must fail rather than silently test an installed, stale operator package.
- [ ] 5.7 Compile the direct ACLNN example in the target CANN environment and execute the full interface/correctness matrix. Record build logs, test commands, exit codes and numerical tolerances before declaring completion.
