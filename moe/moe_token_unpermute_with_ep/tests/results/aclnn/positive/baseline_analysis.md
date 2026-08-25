# ACLNN positive baseline analysis (`3951ec004`)

## Immutable evidence

- Input CSV SHA-256: `3d6dc35ee09e0bbd7c189364e6e5a841ee4aa2aaba9a8465ee8cadae4508e714`
- Result CSV SHA-256: `fb4477af5c9414b1e7135417b2d1a2678b53c95af79581dfc3be3715ed86508f`
- Golden SHA-256: `5ace87305ea0d0b71c88261a55d6d3397a2f89500d098bb9105dc665faf72b05`
- Executed/generated/manifest count: `648/648/648`
- Raw outcome: `640 PASS`, `8 FAIL`, `0 SKIP`

The first-run result is preserved as `result_baseline_3951ec004.csv`. Failure dumps and the independent precision-debug reports are retained outside Git under
`/home/z50058944/moe_token_unpermute_with_ep_precision_debug/`.

## Failure classification

Five FP32 cases were false failures caused by the generated CSV setting
`absolute_precision=0` while the project baseline is `rtol=1e-4, atol=1e-4`:

- `aclnn_moe_token_unpermute_with_ep_l1_generated_59463`
- `aclnn_moe_token_unpermute_with_ep_l1_generated_52572`
- `aclnn_moe_token_unpermute_with_ep_l1_generated_69612`
- `aclnn_moe_token_unpermute_with_ep_l1_generated_71501`
- `aclnn_moe_token_unpermute_with_ep_l1_generated_65613`

The independent `ascendc-precision-debug` batch checker reports `5/5 PASS`
with the declared project tolerance. The generator was corrected to emit
`absolute_precision=0.0001`; no tolerance was weakened.

Three failures are real non-contiguous-output defects:

- `aclnn_moe_token_unpermute_with_ep_l0_float16_noncontiguous`
- `aclnn_moe_token_unpermute_with_ep_l0_bfloat16_noncontiguous`
- `aclnn_moe_token_unpermute_with_ep_l0_float32_noncontiguous`

TTK reports only `13.23529411764706%` matching elements for all three. The
independent checker also rejects FP16 and FP32 with the more permissive
`rtol=atol=1e-3` (`59/68` and `57/68` elements outside tolerance,
respectively). The BF16 NPY encoding is unsupported by that checker, so its
independent recheck is not claimed; the original TTK result remains the
evidence.

The observed output pattern is the backing-storage layout rather than the
requested strided view (`actual[k] == golden[2*k]` until the populated half,
then zeros). Generated ACLNN code had input contiguous conversion but no output
ViewCopy. The minimal implementation correction computes into a contiguous
temporary output and uses `ViewCopy` to populate the caller's requested output
view. The targeted three-dtype regression is preserved in
`../targeted_fix/noncontiguous_close.csv`; the final full-suite result is
recorded separately and is the release gate.
