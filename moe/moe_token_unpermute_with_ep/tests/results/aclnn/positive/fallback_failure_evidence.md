# Empty-range fallback failure evidence

This is the preserved pre-fix result from the first 648-case ACLNN positive
full run against the assertion-enabled package. It is retained so the
implementation correction is traceable rather than hidden by the final rerun.

- Input CSV SHA-256: `dbde1e29afcb0e4932cf4570cc5dbd9f8d8cc682143bfd63f83aaf41775913ac`
- Result: `result_prefallback_fix.csv`
- Result SHA-256: `f7e6c833e440091078fe6ad9a011f43014a3b9d98bbf5ff05cc3f395c484a1f6`
- Generated/manifest/executed: `648/648/648`
- Outcome: `573 PASS`, `75 FAIL`, `0 SKIP`
- All 75 failures: `rangeOptional=None` and `precision=NO_OUTPUT`
- Non-empty-range failures: `0`

The ACLNN optional array is represented at the tiling boundary as a non-null
`ContinuousVector` of length zero. `Init` rejected every such case with
`the size of range only support 2`, so no kernel output was produced. This is
an implementation defect because the documented empty-range contract falls
back to ordinary unpermute semantics.

The minimal fix is in implementation commit `f04eecc40`: both a missing range
attribute and a zero-length range vector select fallback, while non-empty
ranges still require exactly two elements. The regression UT
`test_tiling_empty_range_uses_fallback` reproduces the zero-length encoding;
all 30 host tests pass. Final ACLNN full-suite evidence is written to
`result_final.csv` only after rebuilding and installing the corrected package.

Raw plog evidence remains outside Git at
`/home/z50058944/ascend/log/debug/plog/plog-3970099_20260825053358228.log`.
