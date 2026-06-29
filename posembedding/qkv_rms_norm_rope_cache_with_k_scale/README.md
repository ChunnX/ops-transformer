# QkvRmsNormRopeCacheWithKScale

This directory is the implementation workspace for the arch35
`QkvRmsNormRopeCacheWithKScale` task plan.

Current state:

1. T00 host op definition, infer shape, tiling class, and host UT sources are
   present. Host-side validation and tiling math are owned by
   `QkvRmsNormRopeCacheWithKScaleBaseTiling`; it is not device kernel code.
2. T01 kernel controller and basic block skeleton are present under
   `op_kernel`. The kernel entry file only selects the tiling key, unpacks
   tiling, and dispatches the controller directly.
3. The kernel uses a POD tiling layout under `op_kernel` that host tiling writes
   directly. Host tiling computes `coreTokenTile` with ceil division and writes
   `coreGroupNum`; AIC and paired AIV cores consume those values to derive the
   same contiguous cube-group token range. Host tiling writes cache stride
   triplets, requiring `k_cache` and `v_cache` to share their non-innermost
   strides so the kernel can reuse the `slot_mapping`-derived cache offsets.
   The controller creates the contiguous core token range, expands it into
   token-size tiles, and derives per-tile tail parameters before calling the
   D=128 fast-path basic block.
4. T03 adds static GM/L1/UB tensor binding. `rotationL1Nz_` is the AIC-only
   resident rotation slot, and `a_rot_l1_nz` is the Cube AL1 data slot shared by
   the AIV->AIC protocol. `BasicBlock`
   coordinates the shared `a_rot_l1_nz` pingpong use id so AIV writes and AIC
   reads stay on the same slot; local-only UB/L0C buffer pools maintain their
   own counters rather than carrying a tile-level `bufferId`. The raw
   `outputDbPoolUb_` slots are fixed by purpose: slot 0 is the AIV preprocess
   staging buffer and slot 1 is the AIC FIXPIPE Q/K output buffer. The raw
   `GlobalTensors` bundle is only an entry argument; Cube and Vec bind their own
   typed GM tensors and neither controller nor basic block retains a shared
   bound-GM bundle. The host/API and kernel entry expose one `rotation`; Cube
   copies it once into resident L1, loads one L0B view, and reuses it for both Q
   and K rows.
5. T04/T11 add fixed CrossCore flag ids and wrappers for `A_READY`,
   `A_CONSUMED`, `C_READY`, and `C_CONSUMED`. `A_CONSUMED` is the A-side
   double-buffer reuse credit; T13 completed the final static protocol audit.
   CrossCore flag ids are semantic constants and are not derived from pingpong
   buffer ids or loop generation.
6. T05 adds fixed intra-core event constants and Init/End pairing for reusable
   AIV/AIC buffers. Active event ids stay within `0..5`; `a_rot_l1_nz` remains
   CrossCore-protected AL1 and does not consume AIV intra-core event ids.
7. After T04 the D=128 fast-path code is split into `common`, `controller`,
   `basic_block`, `cube`, and `vec` headers instead of keeping those larger
   modules inside one file. Core-type dispatch is centralized in the
   controller AIC/AIV entry split; lower layers do not inspect `g_coreType`.
8. Real Q/K/V compute, rotation matmul, Fixpipe distribution, rolling pipeline,
   tail/empty-slice handling, concrete synchronization call sites, and the
   synchronization audit are implemented through T13/H03. T14 kernel UT/ST
   closure remains open.
9. Implementation work must follow `TASKS.md`.
10. After each task is coded and verified, update `TASKS.md` and commit the
   code plus the task status change together.

## Rotation and layout contract

The public host/API contract adds `actual_seq_lens` for per-batch history
lengths and keeps a single optional BF16 ND `rotation` tensor that is required
by the D=128 fast-path host contract:

| Index | Input | Shape |
|---:|---|---|
| 9 | `actual_seq_lens` | `[batch]` |
| 10 | `rotation` | `[headDim, headDim]` |
| 11 | `v_scale` | `[numVHeads]` |

`cu_seq_len` is the prefix sum for the tokens in the current invocation.
`actual_seq_lens` stores each batch sequence length after the current tokens are
appended. The kernel derives each batch history offset as
`max(actual_seq_lens[batch] - (cu_seq_len[batch + 1] - cu_seq_len[batch]), 0)`
and adds it to the local token position before indexing the full `cos_sin`
table. For example, `cu_seq_len=[0,17,18]` and
`actual_seq_lens=[8209,1]` means batch0 uses RoPE positions `8192..8208`,
while batch1 starts from position `0`.

The required `qkv_layout` attribute is case-sensitive and only accepts `NTD`
or `TND`. `qkv`, `q_out`, and `q_scale` keep the same N/T layout: `NTD` uses
`qkv=[Nq+Nk+Nv,T,D]`, `q_out=[Nq,T,D]`, `q_scale=[Nq,T]`; `TND` uses
`qkv=[T,Nq+Nk+Nv,D]`, `q_out=[T,Nq,D]`, `q_scale=[T,Nq]`.

`rotation` must be present and BF16; missing input reports
`missing_rotation`. The kernel entry ABI and Cube internals carry one
`rotation` GM tensor. Cube copies it into `rotationL1Nz_`, loads one resident
L0B view, and uses the same matrix for Q rows and K rows.
