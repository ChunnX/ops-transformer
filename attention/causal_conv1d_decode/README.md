# CausalConv1dDecode

Decode-only causal depthwise conv1d for MTP chunks.

This operator intentionally keeps only the decode path. It borrows public parameter naming from
`attention/fused_causal_conv1d`, but removes prefill, mixed prefill/decode, and APC-specific fields.

## Scope

- BF16 first.
- Fixed kernel width `K=4`.
- `seq_len` in `[1, 4]`.
- `x`: `[batch, seq_len, dim]`.
- `weight`: `[4, dim]`.
- `conv_states`: `[num_cache_slots, state_len, dim]`, updated in-place.
- `cache_indices`: `[batch]`.
- Optional `bias`: `[dim]`.
- Optional `num_accepted_tokens`: `[batch]`.
- Output `y`: `[batch, seq_len, dim]`.

## Not Supported In v1

- Prefill.
- Mixed prefill/decode.
- APC block mapping.
- `run_mode`.
- Variable kernel width.

## Documents

- English API: `docs/aclnnCausalConv1dDecode.md`
- Chinese API: `docs/aclnnCausalConv1dDecode.zh.md`
- Trace record: `reports/work_log.md`

