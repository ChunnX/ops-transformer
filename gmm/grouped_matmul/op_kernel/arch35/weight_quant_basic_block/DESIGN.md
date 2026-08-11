# QuantGroupedMatmul MXA8W4 Blaze migration

## Scope

This path migrates all pseudo-quantized MX A8W4 data flows currently accepted by the DAV_3510 host tiling from
CMCT to the Blaze tensor API:

- single-X/single-weight/single-Y, where the leading E dimension of weight, weight scale, and optional bias stores
  all groups contiguously;
- single-X/multi-weight/single-Y, where weight, weight scale, and optional bias contain one TensorList entry per
  group.

Older devkit builds keep the existing CMCT controller as a compatibility fallback.

## Kernel selection

The implementation reuses the same delivered Blaze path as QuantBatchMatmul:

- `MatmulWithWeightQuantMx` for MX scale-aware MMAD;
- `BlockSchedulerMatmulSwatWithTailSplit` for M/N tile scheduling;
- `KernelMatmulMixWeightPrologue` for the AIV FP4-to-FP8 weight conversion and AIC/AIV synchronization;
- four L1 buffers, matching the existing grouped MXA8W4 tiling (`baseK=64`).

No full-load mode is selected. Grouped workloads have variable M values and reuse the existing SWAT scheduler per
group; a full-load policy would add L1 residency constraints without a stable cross-group reuse opportunity.

## Group orchestration

The grouped wrapper reads `groupList` in cumulative-offset or count mode. X, per-token scale, and Y are contiguous
single tensors, so their offsets advance by `M*K`, `M*ceil(K/32)`, and `M*N`. For multi-weight input, weight, weight
scale, and optional bias are resolved from the corresponding TensorList entry for every group. For single-weight
input, only TensorList entry zero is decoded; group addresses then advance by `K*N/2` packed-FP4 bytes,
`N*ceil(K/32)` E8M0 scale elements, and `N` bias elements. Empty groups are skipped without changing the logical
group index or the offset derived from `groupList`.

The wrapper creates a fresh Blaze kernel invocation for each non-empty group. All AIC/AIV blocks traverse groups in
the same order, preserving the weight-prologue cross-core flag protocol.

## Tiling mapping

`baseM/baseN/baseK` come from the existing `TCubeTiling`. The conservative L1 K tile is `baseK`; scale K uses the
same extent. For NZ weights, `kBubSize` keeps one complete, 64-aligned L1 K tile in the prologue UB. Tail-resplit
parameters remain neutral; the scheduler still distributes the regular M/N tile grid and handles shape tails.

## Compatibility

Blaze is enabled for devkit versions newer than 9.0. The implementation is limited to DAV_3510 because the selected
Blaze block specialization and MX tensor layouts are architecture-specific.
