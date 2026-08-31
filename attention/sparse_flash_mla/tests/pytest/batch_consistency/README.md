# Batch consistency integration

The shared integration is used by SparseFlashMla, QuantSparseFlashMla, and
MixedQuantSparseFlashMla single-case, PT-save, eager batch, and graph batch
pytest paths.

Use `test_run.sh ... --batch-consistency <auto|on|off>`:

- `auto` enables the check when the paramset or loaded PT contains an enabled
  flag or any batch-consistency parameter.
- `on` forces the check and generates every missing parameter from a persisted
  random seed.
- `off` forces the ordinary execution path.

The paramset/Excel fields are flat:

- `batch_consistency`
- `batch_consistency_seed`
- `batch_consistency_order`
- `batch_consistency_batch_split`
- `batch_consistency_mode_batch`
- `batch_consistency_token_split`
- `batch_consistency_shape_change`

When PT saving is enabled, the resolved fields and
`npu_deterministic_level=3` are saved in `params`. Execution reuses saved
values; missing values are generated and logged. Enabled execution keeps level
3 active for metadata and operator calls and requires baseline precision,
derived-case precision, and bitwise equality for every applicable relation.
