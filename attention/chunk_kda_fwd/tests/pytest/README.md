# ChunkKdaFwd tests

The functional tests cover the public `cann_ops_transformer.ops.chunk_kda_fwd`
entry, all four input layouts, optional outputs, BF16 gate parameter promotion,
varlen tails, `state_v_first`, and binary determinism.

```bash
pytest -q attention/chunk_kda_fwd/tests/pytest/test_chunk_kda_fwd.py
```

The A5 performance gate uses the key BF16 case: B=1, T=16384, H=96,
K=V=128, chunk size 64, BSND, safe gate, gate-in-kernel, and final state.
Performance is read from `msopprof` `OpBasicInfo` records. The maximum valid
samples for the `ChunkKdaFwd` kernel and the BSND beta-layout transpose are
conservatively summed. This end-to-end device time, rather than Python wall
time or only the mean, must not exceed 11 ms.

```bash
mkdir -m 700 -p output/chunk_kda_fwd_profiles

msopprof \
  --application="python attention/chunk_kda_fwd/tests/pytest/profile_a5_h96_t16k.py --repeats 8" \
  --output=output/chunk_kda_fwd_profiles/a5_h96_t16k \
  --aic-metrics=BasicInfo \
  --kernel-name=ChunkKdaFwd \
  --launch-count=5 \
  --warm-up=3 \
  --kill=off

msopprof \
  --application="python attention/chunk_kda_fwd/tests/pytest/profile_a5_h96_t16k.py --repeats 8" \
  --output=output/chunk_kda_fwd_profiles/a5_h96_t16k_beta \
  --aic-metrics=BasicInfo \
  --kernel-name=Transpose_float16_int64_high_performance_10001 \
  --launch-count=5 \
  --warm-up=3 \
  --kill=off

python attention/chunk_kda_fwd/tests/pytest/check_msopprof.py \
  output/chunk_kda_fwd_profiles/a5_h96_t16k \
  --component Transpose_float16_int64_high_performance_10001 \
    output/chunk_kda_fwd_profiles/a5_h96_t16k_beta \
  --max-ms 11
```

Run the same profile entry once with
`--check-determinism --determinism-repeats 50` before profiling to verify 50-run
binary equality of `attn_out`, `final_state`, `Aqk`, and `Akk`.
