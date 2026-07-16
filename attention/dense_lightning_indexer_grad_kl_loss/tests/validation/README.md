# DenseLightningIndexerGradKLLoss Migration Validation

This directory owns the A2/A5 functional baseline described in the migration
plan. Inputs are generated randomly on every new run. The random seed and SHA256
of every binary are recorded in `case_manifest.json`, so a failed input can be
replayed unchanged on A2 and A5.

## Golden changes

`golden_a2_a5.py` is a clean derivative of
`D:\Code\SYNC_FILE\DSA_LOSS\golden.py`; the original is unchanged.

- Removes the unnecessary TensorFlow dependency for BF16 binary I/O.
- Fixes TND multi-batch cumulative-length generation.
- Allows repeated cumulative lengths for partial zero-length batches.
- Computes all four max/sum tensors from the actual Q/K and index-score inputs.
- Keeps `d_weight` in FP32 when the weight input is FP32.
- Performs FP32 reference math and checks every generated tensor for NaN/Inf.
- Records random seed, shapes, dtypes, cumulative lengths, and input checksums.

## Suites

- `cases_smoke.yaml`: 4 dispatch and basic precision cases.
- `cases_typical.yaml`: 12 representative workload cases.
- `cases_full.yaml`: 48 fixed cases: 32 pairwise, 8 TND, 4 boundary, 4 risk.

The full ranges intentionally describe production-size shapes. Generating all
goldens can require substantial host memory and time; run them sequentially.

## Commands

```bash
python3 check_case_coverage.py
python3 generate_case.py --suite smoke --output-root /path/to/stage/inputs
python3 run_case.py /path/to/stage/inputs/SMK-01 --device-id 3
```

Or run a complete suite:

```bash
DATA_ROOT=/path/to/stage/cases DEVICE_ID=3 bash run_cases.sh smoke
```

The comparison rule is element-wise:

- `abs(golden) < 1`: absolute error must be at most `1e-3`.
- `abs(golden) >= 1`: relative error must be at most `1e-3`.
