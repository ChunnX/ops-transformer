# CSV validation evidence

All commands were run after sourcing
`/home/z50058944/.config/glacier/env.sh`.

The final ACLNN files were checked by the `ascendc-950-testcase` validator,
loaded by `ttk list`, and passed TTK's route-specific validation:

```text
python -m ttk aclnn -i tests/ttk/aclnn_moe_token_unpermute_with_ep_positive.csv --validate
python -m ttk aclnn -i tests/ttk/aclnn_moe_token_unpermute_with_ep_negative.csv --validate
```

- Positive: 648/648 `VALID` (`aclnn_positive_validate.csv`).
- L2 negative: 35/35 `VALID` (`aclnn_negative_validate.csv`).
- Independent generated/manifest/merged audit: PASS
  (`aclnn_case_audit.json`).

The generated Kernel whitebox files were also loaded by `ttk list` and passed
Kernel validation under the isolated Ascend 950 custom package:

```text
python -m ttk kernel -i tests/whitebox/ttk_design_doc_cases.csv --validate
python -m ttk kernel -i tests/whitebox/ttk_design_doc_cases_oversized.csv --validate
```

- Regular: 105/105 `VALID` (`kernel_whitebox_regular_validate.csv`).
- Oversized: 1/1 `VALID` (`kernel_whitebox_oversized_validate.csv`).

Raw `ttk list` terminal logs are intentionally excluded from Git; the source
CSVs, validator result CSVs, hashes and independent count audits are retained.
