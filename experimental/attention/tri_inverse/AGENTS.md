# AGENTS.md - tri_inverse

## Scope

This directory contains the experimental `tri_inverse` operator: a standalone
64x64 strict-lower triangular inverse extracted from `chunk_gated_delta_rule`
Stage1.

## Current Kernel Layout

- `op_kernel/tri_inverse.h`: passing baseline `TriInverseOp`; keep it as the
  reference implementation.
- `op_kernel/tri_inverse_mix.h`: active `MixInverseOp`; do real MixInverse
  work here.
- `op_kernel/tri_inverse.cpp`: compile-time include switch only:
  `TRI_INVERSE_USE_MIX_INVERSE=1` selects `tri_inverse_mix.h`,
  `TRI_INVERSE_USE_MIX_INVERSE=0` selects `tri_inverse.h`.

## Hard Boundaries

- For MixInverse implementation, edit only `op_kernel/tri_inverse_mix.h` and
  necessary docs.
- Do not modify `op_host/`, `op_host/op_api/`,
  `op_kernel/tri_inverse_tiling_data.h`, pytest golden, PyTorch wrapper, or
  Python call interface.
- Do not optimize or rewrite `op_kernel/tri_inverse.h`; it is the baseline.
- Do not use `PTO`, `Pto`, or `MixInverse2` names.
- Do not copy upper-triangular `LX/RX` formulas into this lower-triangular
  path.
- Do not use GM dump or tensor roundtrip debugging for correctness localization.

## Math Contract

```text
cc_in  : BF16, ND, [tile_num, 64, 64]
cc_out : BF16, ND, [tile_num, 64, 64]
```

`cc_in` is positive strict-lower `A`, not raw `KKT` and not `-A`.
The existing flow first applies `Muls(-1.0)`.

Lower-left compose order is fixed:

```text
outLower = bottomInv @ lowerWork @ topInv
lowerWork = -LOW
```

## Optimization Discipline

- Use the simplest viable implementation first. It is acceptable to land a
  precision-passing version before removing redundant movement or compute.
- Solve the active problem directly. Do not replace a direct path with a
  circuitous workaround and present that workaround as the main plan.
- The final MixInverse performance must beat the reference/baseline
  performance, not merely match correctness.
- Change one dataflow at a time.
- Run precision before starting the next optimization.
- Keep L0C accumulation and DataCopy/movement merge as later optimization
  phases only; do not mix them into Phase 2/3/4 correctness work.
- L0C accumulation may use
  `/mnt/workspace/gitCode/cann/ops-transformer/attention/chunk_gated_delta_rule_neumann`
  as a local reference for the accumulation technique only.
- Do not fully copy the Neumann inverse implementation into MixInverse. That
  version is known to work but has poor performance, so treat it as a partial
  reference, not a template.
- Profile fields must keep CANN original names.

Precision gate:

```text
shape=64 seed=0/21
shape=256 seed=0/21
shape=1024 seed=0/21
full 30 cases
```

Performance comparison shapes:

```text
64, 65, 1024, 1025, 10240, 10241
```

## Commands

Build and install:

```bash
bash build.sh --pkg --experimental --soc=ascend910_93 --ops=tri_inverse -j16
bash build/cann-ops-transformer-custom_linux-aarch64.run --quiet
source ${ASCEND_HOME_PATH}/opp/vendors/custom_transformer/bin/set_env.bash
```

Precision:

```bash
cd experimental/attention/tri_inverse/tests/pytest
bash test_run.sh single
```

## Docs

Start from:

```text
docs/2026-05-11/MIX_INVERSE_NEXT_WINDOW_HANDOFF.md
```

Background snapshot:

```text
docs/2026-05-11/TRI_INVERSE_STAGE_SUMMARY.md
```
