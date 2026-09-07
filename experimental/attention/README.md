# FMM016 optimized streaming router

This directory contains the CANN upstream contribution used by the optimized
FMM016 GE path.  The router keeps the top-k state in on-chip tiles while it
scans source leaves in blocks; it emits fixed-shape `selected_index` and
`selected_valid` tensors so the following attention graph does not need a
Python-side padding or transpose copy.

Two static variants are provided:

- `fmm016_streaming_top_k_blocked_v2`: short/medium sequences, selected by the
  TorchAir converter when the padded leaf count is at most 1024.
- `fmm016_streaming_top_k_blocked_v4`: long sequences, selected when the
  padded leaf count is greater than 1024.  It uses the same host contract and
  tiled kernel with a larger source-leaf traversal budget.

Both operators use the same five-input contract:

`q_index [B,H,L,G,I]`, `k_index_transpose [B,H,I,L_pad]`,
`index_weight [B,H,L,G]`, and the optional `top_k` attribute.  The current
kernel masks the fixed five-leaf near window used by the FMM016 reference
path; a future radius-generalized variant must change both the host contract
and the tiling key.  Outputs are
`selected_index [B,H,L,K]` (`int32`) and `selected_valid [B,H,L,K]`
(`uint8`).  Invalid slots are zero-filled and marked false.  The host tiling
rejects unsupported ranks, data types, dimensions, and `K > 16` before launching
the kernel.

Build from the repository root with the normal CANN ops-transformer flow,
or set `FMM016_OPS` to these two operator names when running the focused
package build.  Runtime numerical coverage is exercised through the
`optimized_candidate/torch_npu` GE tests; the installed package is kept
separate from the legacy scalar FMM016 draft.
