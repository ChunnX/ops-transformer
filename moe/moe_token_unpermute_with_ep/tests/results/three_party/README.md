# Step 4 NPU / CPU / GPU cross-check

Status: **BLOCKED** (external GPU/XPU endpoint unavailable). This is not a
PASS and no three-party precision or performance ratio is reported.

The operator has no same-name Torch or TensorFlow implementation. Its
independent CPU/GPU reference is therefore a composition of Torch
`index_select`/mask/multiply/sum primitives, registered as cross-check L0 in
`tests/assets/golden.py`. The Golden and third-party implementations are
separate and neither invokes the tested NPU operator.

## Runnable assets

- Regular CSV (105 cases):
  `tests/whitebox/ttk_design_doc_cases.csv`, SHA-256
  `d4c45a8eb16c777aabb9913d1cab978ece707b9a3770785b3882f5eea22c1d42`.
- Oversized CSV (1 case):
  `tests/whitebox/ttk_design_doc_cases_oversized.csv`, SHA-256
  `0c18815f643c4c68fc2a6369723d1723b806272cc769635a52b90be7e2e1ed4f`.
- Golden/plugin: `tests/assets/golden.py`, SHA-256
  `35d5cefaacfe39ffd6ba14f20173320c122085b97b0d6dbf78d9178932192c2d`.

After sourcing `/home/z50058944/.config/glacier/env.sh` and the installed
custom-package environment, run from `/home/z50058944/ops-test-kit`:

```bash
export ASCEND_CUSTOM_OPP_PATH=/home/z50058944/code/ops-transformer_moe/moe/moe_token_unpermute_with_ep/tests/ttk_bridge:/home/z50058944/Ascend/moe_token_unpermute_with_ep_validation_isolation_289067d40/vendors/custom_transformer
export LD_LIBRARY_PATH=/home/z50058944/Ascend/moe_token_unpermute_with_ep_validation_isolation_289067d40/vendors/custom_transformer/op_api/lib:${LD_LIBRARY_PATH}
export TTK_DISABLE_BUILTIN=0
```

```bash
/home/z50058944/venv/bin/python -m ttk kernel \
  -i /home/z50058944/code/ops-transformer_moe/moe/moe_token_unpermute_with_ep/tests/whitebox/ttk_design_doc_cases.csv \
  -o /home/z50058944/code/ops-transformer_moe/moe/moe_token_unpermute_with_ep/tests/results/three_party/result \
  -d false -b release -c false \
  --plugin /home/z50058944/code/ops-transformer_moe/moe/moe_token_unpermute_with_ep/tests/assets/golden.py \
  --compare cross_check --golden-mode Promote --seed 42 --pc 1 \
  --xpu-perf --provider torch
```

Then run the isolated oversized row with the same parameters:

```bash
/home/z50058944/venv/bin/python -m ttk kernel \
  -i /home/z50058944/code/ops-transformer_moe/moe/moe_token_unpermute_with_ep/tests/whitebox/ttk_design_doc_cases_oversized.csv \
  -o /home/z50058944/code/ops-transformer_moe/moe/moe_token_unpermute_with_ep/tests/results/three_party/result_oversized \
  -d false -b release -c false \
  --plugin /home/z50058944/code/ops-transformer_moe/moe/moe_token_unpermute_with_ep/tests/assets/golden.py \
  --compare cross_check --golden-mode Promote --seed 42 --pc 1 \
  --xpu-perf --provider torch
```

## Exact blocker evidence

After platform isolation was completed and commit `289067d40` had passed the
950 and 910B build gates, the regular command above was rerun unfiltered with
seed 42 and process count 1 on 2026-08-25. It exits with status 1 before case
preparation with:

```text
RuntimeError: --xpu-perf requires remote XPU config (ttk.conf.yaml or --config), but none is configured.
```

The external raw blocker log is
`/home/z50058944/moe_step4_three_party_blocker_isolation_289067d40.log`, SHA-256
`7af4ab157ba859457c56b90acf878f0ff6ca6e3cf66ff38c71b65d4310c278b4`.
It is intentionally not stored in Git.

No `nvidia-smi` executable, `/dev/nvidia*` device, user
`~/.config/ttk/ttk.conf.yaml`, or TTK-root `ttk.conf.yaml` is available. TTK's
`--xpu-perf` implementation requires `remote.endpoints`; a local CPU execution
cannot satisfy or replace that contract. Supplying a reachable remote endpoint
and rerunning the complete command is the remaining Step 4 gate.
