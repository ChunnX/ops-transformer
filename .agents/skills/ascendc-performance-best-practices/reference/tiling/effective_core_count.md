# Effective Core Count

Use when profiling shows small or boundary shapes launching far more AI Core tasks than useful independent work.

## Pattern

Do not blindly use physical core count as host `blockDim` or vector core count. If the kernel's useful work is bounded by a smaller independent task count, cap the host launch count by a conservative upper bound:

```text
effective_core_count = min(platform_core_count, shape_implied_task_upper_bound)
```

The upper bound must never be lower than the work implied by static shape and attrs. Runtime tensor values such as actual sequence lengths may reduce work further inside the kernel, but host tiling must not depend on unavailable runtime values.

## Applies When

- The kernel has one independent task per batch/head/block/state unit.
- Host tiling currently launches platform cores or TSch mix cores unconditionally.
- Profiling shows small, boundary, or sparse cases with many idle cores.
- The kernel already guards out-of-range cores or re-splits by actual work.

## Does Not Apply When

- Extra cores participate in required cooperative work, synchronization, or reductions.
- The host cannot derive a safe static upper bound.
- The cap would alter dtype/format support, tiling key semantics, or output shape.
- Large target shapes are compute or memory bound inside active cores rather than launch/scheduling bound.

## Required Evidence

- Host tiling UT that preserves large-shape platform core count and proves small-shape effective count.
- Package/load evidence for updated `op_host` tiling.
- Accuracy gate after install.
- Same baseline/candidate performance matrix with at least 3 valid samples per case.
- BlockDim or `vectorCoreNum` evidence in profiling logs.

## ops-transformer Validation

Validated on 2026-06-17:

- `attention/lightning_indexer`: cap host `blockDim` by shape-implied base-block upper bound. ST matrix improved 22.72% to 51.59%; README PA large was flat at +0.39%.
- `attention/recurrent_gated_delta_rule`: cap `vectorCoreNum` by `b * nv`. ST matrix improved 14.33% to 34.82%.

Ledger:
- `agent_ops/lightning_indexer/`
- `agent_ops/recurrent_gated_delta_rule/`

## Failure Modes

- Underestimating work from host shape when runtime tensors can increase it.
- Treating sub-1% noise as a meaningful gain.
- Reusing the pattern on kernels where idle-looking cores perform required synchronization.
- Running only small favorable cases and missing large-shape regressions.
