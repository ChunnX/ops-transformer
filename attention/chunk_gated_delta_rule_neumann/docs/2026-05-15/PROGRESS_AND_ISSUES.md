# Progress And Issues - 2026-05-15

## Accepted Active Source

- A131 is accepted as the active correctness-safe continuation.
- Active Stage1 path after cleanup: `op_kernel/chunk_gated_delta_rule_neumann_stage1.h`.
- Pre-cleanup A131 source SHA256: `966465538bf97e8a6f3966b1f59d341195d948f177197bc9a57b89fbad8f4c0f`.
- Official Stage1 SHA256 after cleanup: `8d4932568da50f3118e3e496dd2508c79cd2d10951d7af53fed6165e12440984`.
- Duplicate baseline headers are retired; stage evidence stays in dated docs and SKILL references.

## Result Summary

- Tail padding scalar loop is removed from `ApplyTailPaddingToAttn(validLen)`.
- Explicit padding remains, implemented by `Duplicate`:
  - whole invalid half-block,
  - row-start masked invalid columns,
  - contiguous invalid rows.
- Precision gate passed:
  - tail/boundary smokes `S=5/16/31/32/33/34/48/49/63/65/129`,
  - aligned smokes `S=64/256/1024`,
  - full compare `30 passed in 6.79s`.
- Profiling output: `tests/pytest/prof_out/packed64_phaseA131_tail_padding_duplicate_20260515_8shapes/performance_compare.csv`.
- Neumann/normal ratios `S=64/65/512/513/2048/2049/10240/10241 = 0.811 / 0.812 / 0.793 / 0.793 / 0.836 / 0.832 / 0.840 / 0.845`.
- Cleanup rebuild package SHA256: `f36da5e37df82acf4e1da9c0c05172265e0933bd9c7ca70c462fd1f69afded0a`.

## Issues And Boundaries

- The build directory had stale C128 cache before validation. It was isolated before the successful rebuild. Do not trust packages from before `20260515_110201` for A131.
- The OPP install tree still contains nested old `vendors/custom_transformer` layers. Use the top-level installed kernel path for verification:
  - `/home/developer/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_impl/ai_core/tbe/kernel/ascend910_93/chunk_gated_delta_rule_neumann/`.
- A131 does not prove the upstream zero-padding contract is sufficient to delete `ApplyTailPaddingToAttn(validLen)`. It only proves the explicit padding can be vectorized safely.
- Keep follow-up scope separated from A131. Next optimization must start from the official Stage1 A131 source and pass the same tail/aligned/full compare gate.
