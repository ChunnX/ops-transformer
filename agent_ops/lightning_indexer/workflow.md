# LightningIndexer Workflow

Reproduce package and accuracy:

```bash
bash build.sh --pkg --soc=ascend910b --ops=lightning_indexer,recurrent_gated_delta_rule --vendor_name=custom -j16 -O3
bash build_out/cann-ops-transformer-custom_linux-aarch64.run --quiet
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}
python3 scripts/st_accuracy/run_st_accuracy.py run --ops lightning_indexer,recurrent_gated_delta_rule --soc ascend910b --out build/st_accuracy/<tag>
```

Run focused host tests:

```bash
bash build.sh -u --ophost_test --soc=ascend910b --ops=lightning_indexer --disable_asan -j16
```

Profiling matrix:
- Use `artifacts/perf_baseline/default_matrix/batches/lightning_indexer_case_*.tsv`.
- Include `artifacts/perf_baseline/default_matrix/li_readme_pa_large.tsv`.
- Use single-case output directories and at least 3 valid samples per case.
- If a median is within about 1% of baseline, resample before deciding.

Retention:
- Keep summary and hashes in this ledger.
- Keep raw profiling under `artifacts/` only while it remains useful for the next candidate.
