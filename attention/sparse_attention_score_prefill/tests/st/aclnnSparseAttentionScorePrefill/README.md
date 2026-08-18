# ATK smoke cases

The runner assumes the `rrh` environment and the custom OPP have already been
activated:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rrh
source /opt/rrh/zhengbao/0708/ascend-toolkit/set_env.sh
export ASCEND_CUSTOM_OPP_PATH=/opt/rrh/zhengbao/0708/cann/opp
```

Run the three paged-KV cases (LSE enabled, `innerPrecise` 0/1/4):

```bash
bash run_atk_sparse_attention_score_prefill.sh
```

Run the equivalent concatenated-KV (`blockSize=0`, no block table) cases:

```bash
bash run_atk_sparse_attention_score_prefill.sh --continuous-kv
```

Use `--inner-precise 4` to run one selected case. The final ATK summary line
`acc_pass_result:Pass` is the correctness result; the per-case
`input_bytes/output_bytes` columns may remain `None` because ATK does not emit
byte counters for this executor.
