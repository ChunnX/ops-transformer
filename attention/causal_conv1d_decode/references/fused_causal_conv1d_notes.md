# Reference Notes: ops-transformer fused_causal_conv1d

Source inspected:

- `https://gitcode.com/cann/ops-transformer/tree/master/attention/fused_causal_conv1d`
- `https://gitcode.com/cann/ops-transformer/blob/master/attention/fused_causal_conv1d/docs/aclnnFusedCausalConv1d.md`

Only public README/docs/example interface text was used. Kernel implementation files were not cloned or copied.

## Borrowed Interface Concepts

The upstream op exposes these concepts:

- `x`
- `weight`
- `conv_states`
- `query_start_loc`
- `cache_indices`
- `initial_state_mode`
- `bias`
- `num_accepted_tokens`
- optional APC-related inputs:
  - `num_computed_tokens`
  - `block_idx_first_scheduled_token`
  - `block_idx_last_scheduled_token`
  - `initial_state_idx`
- attributes:
  - `activation_mode`
  - `pad_slot_id`
  - `run_mode`
  - `max_query_len`
  - `residual_connection`
  - `block_size`
  - `conv_mode`
- output:
  - `y`

The upstream docs state `run_mode` is a legacy field and is not supported.

## Decode Constraints From Upstream Docs

Decode variable-length case:

- `x` supports `[cu_seq_len, dim]`.
- `weight` is `[K, dim]`, with upstream `K=3`.
- `conv_states` is `[..., K - 1 + m, dim]`.
- `cache_indices` is `[batch]` or `[batch, maxNumBlocks]`.
- `cu_seq_len` range is `[batch, batch * 8]`.
- Each batch has token count `[1, 8]`.
- `dim` range is `[64, 16384]` and must be a multiple of 16.
- `batch` range is `[1, 256]`.

Decode fixed-batch case:

- `x` supports `[batch, m + 1, dim]`.
- `m` range is `[0, 7]`.
- Other constraints are similar.

## Choices For This New Operator

This project intentionally narrows scope:

- Decode only.
- No prefill path.
- No mixed prefill/decode path.
- No APC path in v1.
- No `run_mode`.
- Width `K=4`, matching the CUDA decode operator we developed earlier.
- BF16 first.
- `num_accepted_tokens` supports MTP acceptance in `[1, seq_len]`.
- `cache_indices` is kept so runtime can map batch rows to KV/cache slots.

