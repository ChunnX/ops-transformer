# Quant Block Sparse Attention Block Parameters Design

## Scope

Add explicit `block_num` and `max_block_per_batch` values to every `_case`
entry in:

- `experimental/attention/quant_block_sparse_attn/tests/pytest/quant_block_sparse_attn_paramset.py`
- `experimental/attention/quant_block_sparse_attn/tests/pytest/quant_block_sparse_attn_paramset_generalized.py`

No golden-generation behavior or CSV data will be changed.

## Parameter Sources

Match Python cases to their corresponding CSV rows by exact case name.

- All 304 cases in `quant_block_sparse_attn_paramset_generalized.py` match
  `batch/cases_generalized.csv`. Copy each row's integer `block_num` and
  `max_block_per_batch` values without recalculation.
- None of the four cases in `quant_block_sparse_attn_paramset.py` match a row
  in `batch/cases.csv`. Calculate their values using the rules below.

CSV values take precedence over calculated constraints. They have been checked
against the current golden block-table requirements: enough physical blocks for
dedicated batch tail blocks and enough logical table columns for every KV
sequence.

## Rules for Cases Not Found in CSV

For each unmatched case, calculate the logical KV block count as:

```text
logical_blocks = ceil(max(seqused_kv_value) / 128)
```

Choose `block_num` as:

```text
block_num = max(
    logical_blocks,
    B + 1 if logical_blocks > 1 else B,
)
```

This satisfies both the requested `block_num` lower bound and the current
golden requirement for dedicated physical tail blocks.

Use the upper value for approximately 80% of unmatched cases:

```text
max_block_per_batch = B * block_num
```

Use a reduced value for approximately 20%. Since there are four unmatched
cases, select one case (25%, the nearest possible whole-case ratio):
`tnd_random_actlen`. Its value is:

```text
max_block_per_batch = ceil(0.75 * B * block_num)
```

The other three unmatched cases use the upper value.

## Verification

Automated checks will verify that:

1. Every case in both Python parameter files explicitly contains both fields.
2. All exact CSV matches contain the same integer values as their CSV rows.
3. The four unmatched cases use the formulas documented above.
4. The reduced unmatched case remains within
   `[B * block_num / 2, B * block_num]`.
5. Both Python files compile successfully.

