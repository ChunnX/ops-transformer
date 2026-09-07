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

Both operators use the same three-input contract plus an optional attribute:

`q_index [B,H,L,G,I]`, `k_index_transpose [B,H,I,L_pad]`,
`index_weight [B,H,L,G]`, and optional `top_k` attribute.  The current
kernel masks the fixed five-leaf near window used by the FMM016 reference
path; a future radius-generalized variant must change both the host contract
and the tiling key.  Outputs are
`selected_index [B,H,L,K]` (`int32`) and `selected_valid [B,H,L,K]`
(`uint8`).  Invalid slots are zero-filled and marked false.  The operator schema
restricts inputs to FP32 and the host tiling rejects unsupported ranks, dimensions,
shape mismatches, padding, and `K > 16` before launching the kernel.

Build from the repository root with the normal CANN ops-transformer flow,
or set `FMM016_OPS` to these two operator names when running the focused
package build.

## Tests

Host tiling tests are under each operator's `tests/ut/op_host` directory and
are run by the normal `--ophost_test` target.  They cover short and long
shapes, 1024-leaf tile crossings, B/H boundary scheduling, and invalid shape
contracts.  The dependency-free CPU golden in
`fmm016_streaming_top_k_blocked_v2/tests/pytest` checks the radius-2 near window
and stable tie ordering; run it with `python3 test_fmm016_streaming_top_k_blocked_reference.py`.

NPU numerical and GE eager-vs-compiled tests require the generated ACLNN
wrapper and a matching `torch_npu` runtime; the deployment-specific wrapper
is kept outside this repository in the migration workspace.
