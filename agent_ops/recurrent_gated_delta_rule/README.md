# RecurrentGatedDeltaRule Performance Ledger

Status: `optimized_kept`

Operator:
- domain: `attention`
- API: `RecurrentGatedDeltaRule` / `aclnn_recurrent_gated_delta_rule`
- target SoC: `ascend910b`
- arch: `arch22`
- vendor: `custom_transformer`

Optimization kept on 2026-06-17:
- Host tiling now caps `vectorCoreNum` by useful independent work, `b * nv`.
- Previous host tiling launched the platform AIV count even when the kernel had fewer batch-head tasks, producing idle vector cores.

Evidence:
- candidate summary: `artifacts/perf_candidate/li_rgdr_blockdim_20260617_0442/SUMMARY.md`
- summary sha256: `a2717a9985e14a45eb0c0fda7afa45f514414d79e2735c984a1eb3449c5a3fa9`
- package sha256: `9cadc00a497ed7acc41cbb7112eb3f529fbcf139d6813041879f4384c36e0f07`
- installed op_api sha256: `96bf2f2af0840292e7ceb3ecd149af57021ac0f44af12bd0bbe3820e8b1cd3ab`

Migration verification on `br_tang_dev` (2026-06-22):
- package sha256: `4d8f2567a0f2e9b8422e1d098db12f8b8dc9b0c50bba7a2a79d97af5742eca0c`
- installed op_api sha256: `7da7964e25974ba1af76afdd0cbdcd4836e34c157c3a16d29878f0266b2b0f30`
- op_host UT: `build.sh -u --ophost_test --soc=ascend910b --ops=lightning_indexer,recurrent_gated_delta_rule -j16`, 16 passed
- ST accuracy: `scripts/st_accuracy/run_st_accuracy.py run --ops lightning_indexer,recurrent_gated_delta_rule --out build/st_accuracy/migration_device`, 18 passed, 0 failed, 0 error, 0 unsupported

Forced-package ST validation (2026-06-23):
- command: `python3 scripts/st_accuracy/run_st_accuracy.py run --ops lightning_indexer,recurrent_gated_delta_rule --vendor custom --soc ascend910b --device 0 --out build/st_accuracy/force_pkg_device`
- package sha256: `1230df03307d0aa62611ad054860434f206f4f4ac2d02fb3b1c82d00e321da5a`
- installed/build op_api sha256: `7da7964e25974ba1af76afdd0cbdcd4836e34c157c3a16d29878f0266b2b0f30`
- result: 18 passed, 0 failed, 0 error, 0 unsupported; `prepare_pkg_ms=104151.404`

Outcome:
- ST matrix improved by 14.33% to 34.82% median `Task Duration(us)`.
- No stable representative-matrix regression was observed.

Follow-up:
- Case 2 has the smallest gain and should be the next RGDR target.
- Next hypotheses should inspect variable-length/state-index path, memory traffic for state update, and any serial or scalar-heavy loops around accepted-token handling.
