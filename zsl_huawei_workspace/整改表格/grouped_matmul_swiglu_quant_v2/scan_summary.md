# grouped_matmul_swiglu_quant_v2 error log rectification summary

- Base/branch: current workspace on `master`
- Scope: `gmm/grouped_matmul_swiglu_quant_v2/op_api/grouped_matmul_swiglu_quant_v2_utils.h`, focused on the user-specified line 1195 and nearby `OP_LOGE(ACLNN_ERR_PARAM_INVALID, ...)` candidates in the same file
- Rectified: 1
- Excluded: 5
- Changed source files: `gmm/grouped_matmul_swiglu_quant_v2/op_api/grouped_matmul_swiglu_quant_v2_utils.h`
- Rectification table: `zsl_huawei_workspace/整改表格/grouped_matmul_swiglu_quant_v2/grouped_matmul_swiglu_quant_v2_op_loge_rectification.csv`
- Exclusion table: `zsl_huawei_workspace/整改表格/grouped_matmul_swiglu_quant_v2/grouped_matmul_swiglu_quant_v2_excluded_error_logs.csv`

## Notes

- The user-specified log was converted from raw `OP_LOGE(ACLNN_ERR_PARAM_INVALID, ...)` to `OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON`.
- The original condition, control flow, return value, and data calculations were not changed.
- Excluded candidates are `CheckXXX failed` aggregation logs, which the skill classifies as internal positioning logs.

## Verification

- Passed: `git diff --check`
- Checked: reason phrase grep on `grouped_matmul_swiglu_quant_v2_utils.h`; matches are pre-existing lines outside the line 1195 change
- Passed: `Import-Csv -Encoding UTF8` for both CSV tables
