# Arch22 shared experts after the master chunk-pipeline migration

## Scheduling and ownership

The Ascend910_93 implementation keeps shared experts dense and batch-local.
Shared GMM1, activation and GMM2 execute once per invocation, not once per
received routing round. Master's routed chunk/receive-round loops, rank mask,
round tables and route-cache refreshes remain intact.

- AIC runs shared GMM1 before routed GMM1 preparation is consumed.
- In the first chunk's first receive round, AIV publishes the shared activation
  after dispatch; AIC consumes it after routed GMM1 and runs shared GMM2.
- All AIVs consume shared GMM2 completion once before the first Combine.
  Quantized shared epilogues can overlap routed GMM2. No all-AIV barrier is
  introduced inside the half-AIV Unpermute branch.
- Shared output is `[shared_expert, full_batch_token, hidden]`. Each chunk's
  Unpermute uses the full-batch expert stride and its own token base. Routed
  output, probabilities and permutation indices retain master's chunk layout.

Flags 11/12/13 belong to the shared pipeline. Input-ready and activation-ready
reuse V2C flag 12 in ordered phases. AIV-only SyncAll owns flag 14. Introducing
mixed-core SyncAll while shared flags are live would conflict with 11–13.

## Precision and memory

Shared GMM1/GMM2 intermediates reuse a separate workspace region appended after
master's complete routing allocation, including per-chunk receive-round tables.
The shared activation must be published before its GMM1 buffer can be reused.
INT8/INT4 GMM2 epilogues publish BF16 to a separate result buffer; this preserves
the rounding point and avoids W4A8 high/low-row in-place aliasing. Unpermute adds
routed weighted outputs and shared outputs in FP32, then casts once to output.

Token-active masks apply to routed assignments; shared experts still consume
the original dense X. Rank masks keep master's early-return and communication
semantics. Zero shared experts bypass the extra computation and handshakes.
Ascend910B shared experts remain explicitly rejected by tiling.

## Trade-off

Keeping the dense batch lifetime is a conservative migration, not a claim of
optimal large-batch performance. Its workspace scales with shared-expert count
and full batch size, and the first routed chunk can wait for full-batch shared
work. Chunk-local shared execution is a separate optimization requiring matched
input offsets, scale strides, buffer lifetimes and per-chunk synchronization.
Moving shared GMM2 behind routed GMM2 to overlap Combine also requires a separate
schedule and profiling; Unpermute's final shared addition cannot consume an
unfinished shared result.

## Validation

The migration adds tiling cases for M=1025, a small receive budget and a rank mask.
Validation must use freshly generated kernel scripts and matching master Python
bindings: old bindings lack the mask-buffer argument and cause an ABI mismatch
before kernel execution. Kernel compilation and host/API smoke tests do not
replace device precision tests, especially for multi-chunk and INT4 paths.
