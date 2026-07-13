# MhcPre Low-API AIC With Original AIV Design

## Goal

Keep the Arch35 N=4 low-level AIC matmul and HF32 mode, while restoring the
original `MhcPreKernelSplitBS` AIV implementation for X preprocessing, invRms,
output postprocessing, and hIn. This intermediate version prioritizes correct
results and removes the scalar AIV regression before a later direct-UB design.

## Data Flow

1. The original AIV V0 path reads X, applies cast and optional gamma, accumulates
   RMS, and writes the FP32 matmul input to the existing workspace.
2. AIC reads the workspace and phi with the low-level L1/L0 `Mmad` service.
   HF32 mode is enabled and disabled only on AIC.
3. AIC uses Fixpipe to write the row-major raw hMix result to GM. When `outFlag`
   is false, hMix uses the existing workspace allocation; when true, it uses the
   optional output tensor.
4. The original AIV V0 finalizes and writes invRms.
5. The original AIV V1 reads hMix from GM, uses the existing Gather and vector
   functions for hPre, hPost, hRes, and hIn, and preserves existing output
   semantics.

## Scope

- Retain the low-level Arch35 Cube implementation and AIC-only HF32 lifecycle.
- Replace only the old `MatmulImpl` call inside the Split-BS execution model.
- Restore original queues, workspace offsets, synchronization, vector kernels,
  Gather behavior, and output handling.
- Keep the old non-low-API keys as fallback for N=6/8 and unsupported tilings.
- Remove the temporary low-API AIV scalar implementation from the selected fast
  path; unused helpers may remain only if another compiled path needs them.

## Later Direct-UB Phase

The next phase will replace the hMix GM handoff with L0C-to-UB Fixpipe and adapt
the original vector postprocessing to consume aligned row-major UB data. That
work is deliberately separate so its precision and performance can be measured
against this GM-backed low-level AIC baseline.

## Verification

- Add or update host tests so N=4 Split-BS still selects the low-API key.
- Build the package with `--soc=ascend950 --pkg`.
- Run the four N=4 BF16 precision cases for 10 iterations with HF32 golden.
- Require hIn, hPost, hRes, invRms, hMix, and hPre to pass existing tolerances.
- Profile the first case and compare AIC/AIV pipeline times with the previous
  high-level baseline; AIV scalar time must return near the original level.
