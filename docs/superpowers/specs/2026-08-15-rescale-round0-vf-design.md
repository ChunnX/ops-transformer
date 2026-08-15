# RescaleRound0 FP8 E4M3 VF Design

## Goal

Move `QBSABlockVec::RescaleRound0` from the high-level Ascend C vector API into a dedicated SIMD VF in `vf_mul_sel_softmaxflashv2_cast_nz_dn.h`, supporting only `fp8_e4m3fn_t` while preserving the existing tensor layout and numerical behavior.

## Scope

- Add a dedicated `RescaleRound0VF` operating on UB pointers.
- Replace the class member call in `FinalizeVec1` with the new VF call.
- Remove the obsolete `QBSABlockVec::RescaleRound0` declaration and definition.
- Remove this operation's dependency on `lseTmpBuff` as an FP32 cast scratch tensor.
- Do not add support for `fp8_e5m2_t`, `hifloat8_t`, or other input types.
- Do not change the surrounding softmax, alpha accumulation, or L1 copy behavior.

## Data Layout and Processing

The VF preserves the current traversal exactly:

- `realK` is divided into groups of at most 64 columns.
- Each group contains two forms.
- Each column contains 32 contiguous FP8 E4M3 elements.
- Form 0 multiplies by `alphaTensor[0:32]`; form 1 multiplies by `alphaTensor[32:64]`.
- The existing strides remain 65 blocks per form and 130 blocks per group, where one block is 32 elements.
- The final group processes only `realK % 64` columns when `realK` is not 64-aligned.

For each 32-element column block, the VF performs:

1. Load FP8 E4M3 data into a register.
2. Convert FP8 E4M3 to FP32.
3. Multiply elementwise by the selected 32-element FP32 alpha vector.
4. Convert FP32 back to FP8 E4M3.
5. Store the 32 converted elements in place.

## Cast Semantics

The implementation follows the Ascend C Reg `Cast` support table:

- `fp8_e4m3fn_t -> float`: `RegLayout::ZERO`, `SatMode::UNKNOWN`, `MaskMergeMode::ZEROING`, `RoundMode::UNKNOWN`.
- `float -> fp8_e4m3fn_t`: `RegLayout::ZERO`, `SatMode::SAT`, `MaskMergeMode::ZEROING`, `RoundMode::CAST_RINT`.

The mask covers exactly 32 logical FP32 elements so no padded column is modified.

Reference: https://www.hiascend.com/document/detail/zh/canncommercial/latest/API/ascendcopapi/atlasascendc_api_07_0398.html

## Interface and Integration

The VF is declared in namespace `FaVectorApi` with UB pointer parameters equivalent to:

```cpp
__simd_vf__ inline void RescaleRound0VF(
    __ubuf__ fp8_e4m3fn_t *pTensor,
    __ubuf__ float *alphaTensor,
    uint32_t realK);
```

`FinalizeVec1` obtains the physical UB addresses from its `LocalTensor` values and invokes this function only for the confirmed FP8 E4M3 specialization. The existing multiplication of `combinedAlpha` by `alpha1` remains after the VF call.

## Verification

- Add or extend the narrowest available source/compile test so it fails before the VF exists and passes after integration.
- Build the affected quant block sparse attention kernel for the arch-920r1 target when the repository provides a runnable target.
- Check that the old declaration, implementation, and scratch-buffer use are absent.
- Check source-level behavior for `realK` values `1`, `63`, `64`, and `65`, covering partial and multi-group traversal.
- Review the resulting diff for unrelated changes.

## Non-Goals

- Packing multiple columns into one register operation.
- Changing the FP8 rounding policy.
- Refactoring other VF functions or shared cast traits.
- Performance tuning beyond replacing the high-level API loop with the dedicated VF.
