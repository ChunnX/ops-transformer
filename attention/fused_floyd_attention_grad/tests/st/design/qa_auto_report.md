# QA 自动校验报告（C-Auto）

> 由 qa_verify_oracle.py 自动生成

## 总览

| 状态 | 数量 |
|------|------|
| PASS | 16 |
| FAIL | 0 |
| SKIP | 15 |

## 逐项结果

| 检查项 | 状态 | 说明 |
|--------|------|------|
| Q1_product_support | SKIP |  |
| Q2_trace_coverage | PASS | 1 条记录 |
| Q3_output_indexes | PASS | 不一致: 0/62 |
| Q4_yaml_domain | PASS |  |
| Q5_existence | SKIP |  |
| Q6_param_value_domain | SKIP |  |
| Q7_param_coverage | PASS | 全部必选入参已覆盖 |
| Q7_et_completeness | SKIP |  |
| Q7b_et_completeness_enhanced | SKIP |  |
| Q8_et_var_exist | SKIP |  |
| Q9_bd_boundary | SKIP |  |
| Q9b_bd_boundary_yaml | PASS |  |
| Q10_ex_validity | PASS |  |
| Q11_ex_expected_error | PASS |  |
| Q12_scenario_id_format | PASS |  |
| Q13_et_case_coverage | SKIP |  |
| Q14_bd_case_coverage | PASS | 14 条记录 |
| Q15_ex_case_coverage | SKIP |  |
| Q16_rn_ex_coverage | SKIP |  |
| Q17_locked_consistency | PASS |  |
| Q18_et_constraint | SKIP |  |
| Q20_output_shape_fidelity | SKIP |  |
| Q13b_et_e2e | PASS |  |
| Q14b_bd_e2e | PASS | 14 条记录 |
| Q15b_ex_e2e | PASS | 16 条记录 |
| Q1_product_support_L1 | SKIP |  |
| Q2_trace_coverage_L1 | PASS | 1 条记录 |
| Q3_output_indexes_L1 | PASS | 不一致: 0/651 |
| Q5_existence_L1 | SKIP |  |
| Q6_param_value_domain_L1 | SKIP |  |
| Q7_param_coverage_L1 | PASS | 全部必选入参已覆盖 |

## 详细结果

### Q1_product_support

*跳过（缺少必要输入）*

### Q2_trace_coverage

- {'param': 'scaleValue', 'unique': 54, 'total': 62, 'is_scalar': True, 'type': 'scalar', 'min': -1.7976931348623157e+308, 'max': 1.7976931348623157e+308}

### Q3_output_indexes

**total:** 62
**mismatches:** 0
**samples:** `[]`

### Q4_yaml_domain


### Q5_existence

*跳过（缺少必要输入）*

### Q6_param_value_domain

*跳过（缺少必要输入）*

### Q7_param_coverage

**missing:** `[]`
**optional_absent:** `[]`
**coverage:** `{'scaleValue': '100%(62/62)'}`

### Q7_et_completeness

*跳过（缺少必要输入）*

### Q7b_et_completeness_enhanced

*跳过（缺少必要输入）*

### Q8_et_var_exist

*跳过（缺少必要输入）*

### Q9_bd_boundary

*跳过（缺少必要输入）*

### Q9b_bd_boundary_yaml


### Q10_ex_validity


### Q11_ex_expected_error


### Q12_scenario_id_format


### Q13_et_case_coverage

*跳过（缺少必要输入）*

### Q14_bd_case_coverage

- {'sid': 'BD-S1', 'found': True, 'cells': ['BD-S1', '_b.value_range=[1,1]', '01 R1 "B：取值范围为1~2K"', 'B 下界（B=1）']}
- {'sid': 'BD-S2', 'found': True, 'cells': ['BD-S2', '_b.value_range=[2048,2048], _h.value=1, _d.value=32, _n.value_range=[16,16], _m.value_range=[128,128], _k.value_range=[128,128]', '01 R1 "B：取值范围为1~2...
- {'sid': 'BD-S3', 'found': True, 'cells': ['BD-S3', '_h.value=1', '01 R2 "H：取值范围为1~256"', 'H 下界（H=1）']}
- {'sid': 'BD-S4', 'found': True, 'cells': ['BD-S4', '_h.value=256, _b.value_range=[1,1], _d.value=32, _n.value_range=[16,16], _m.value_range=[128,128], _k.value_range=[128,128]', '01 R2 "H：取值范围为1~256"'...
- {'sid': 'BD-S5', 'found': True, 'cells': ['BD-S5', '_n.value_range=[16,16]', '01 R3 "N：取值范围为16~1M且N%16==0"', 'N 下界（N=16，最小 16 对齐值）']}
- {'sid': 'BD-S6', 'found': True, 'cells': ['BD-S6', '_m.value_range=[128,128]', '01 R4 "M：取值范围为128~1M且M%128==0"', 'M 下界（M=128，最小 128 对齐值）']}
- {'sid': 'BD-S7', 'found': True, 'cells': ['BD-S7', '_k.value_range=[128,128]', '01 R5 "K：取值范围为128~1M且K%128==0"', 'K 下界（K=128，最小 128 对齐值）']}
- {'sid': 'BD-S8', 'found': True, 'cells': ['BD-S8', '_d.value=32', '01 R6 "D：取值范围为32/64/128"', 'D 枚举值 32']}
- {'sid': 'BD-S9', 'found': True, 'cells': ['BD-S9', '_d.value=64', '01 R6 "D：取值范围为32/64/128"', 'D 枚举值 64']}
- {'sid': 'BD-S10', 'found': True, 'cells': ['BD-S10', '_d.value=128', '01 R6 "D：取值范围为32/64/128"', 'D 枚举值 128']}
- {'sid': 'BD-S11', 'found': True, 'cells': ['BD-S11', '_d.value=128, _m.value_range=[512,512], _b.value_range=[1,1], _h.value=1, _n.value_range=[16,16], _k.value_range=[128,128]', '01 约束说明 "当M*D>=65536...
- {'sid': 'BD-S12', 'found': True, 'cells': ['BD-S12', '_d.value=128, _m.value_range=[256,256], _b.value_range=[1,1], _h.value=1, _n.value_range=[16,16], _k.value_range=[128,128]', '01 约束说明（同上）', '性能语义边...
- {'sid': 'BD-S13', 'found': True, 'cells': ['BD-S13', '_d.value=128, _k.value_range=[512,512], _b.value_range=[1,1], _h.value=1, _n.value_range=[16,16], _m.value_range=[128,128]', '01 约束说明 "当...K*D>=65...
- {'sid': 'BD-S14', 'found': True, 'cells': ['BD-S14', '_d.value=128, _k.value_range=[256,256], _b.value_range=[1,1], _h.value=1, _n.value_range=[16,16], _m.value_range=[128,128]', '01 约束说明（同上）', '性能语义边...

### Q15_ex_case_coverage

*跳过（缺少必要输入）*

### Q16_rn_ex_coverage

*跳过（缺少必要输入）*

### Q17_locked_consistency


### Q18_et_constraint

*跳过（缺少必要输入）*

### Q20_output_shape_fidelity

*跳过（缺少必要输入）*

### Q13b_et_e2e


### Q14b_bd_e2e

- {'sid': 'BD-S1', 'found': True}
- {'sid': 'BD-S2', 'found': True}
- {'sid': 'BD-S3', 'found': True}
- {'sid': 'BD-S4', 'found': True}
- {'sid': 'BD-S5', 'found': True}
- {'sid': 'BD-S6', 'found': True}
- {'sid': 'BD-S7', 'found': True}
- {'sid': 'BD-S8', 'found': True}
- {'sid': 'BD-S9', 'found': True}
- {'sid': 'BD-S10', 'found': True}
- {'sid': 'BD-S11', 'found': True}
- {'sid': 'BD-S12', 'found': True}
- {'sid': 'BD-S13', 'found': True}
- {'sid': 'BD-S14', 'found': True}

### Q15b_ex_e2e

- {'sid': 'EX-S1', 'found': True}
- {'sid': 'EX-S2', 'found': True}
- {'sid': 'EX-S3', 'found': True}
- {'sid': 'EX-S4', 'found': True}
- {'sid': 'EX-S5', 'found': True}
- {'sid': 'EX-S6', 'found': True}
- {'sid': 'EX-S7', 'found': True}
- {'sid': 'EX-S8', 'found': True}
- {'sid': 'EX-S9', 'found': True}
- {'sid': 'EX-S10', 'found': True}
- {'sid': 'EX-S11', 'found': True}
- {'sid': 'EX-S12', 'found': True}
- {'sid': 'EX-S13', 'found': True}
- {'sid': 'EX-S14', 'found': True}
- {'sid': 'EX-S15', 'found': True}
- {'sid': 'EX-S16', 'found': True}

### Q1_product_support_L1

*跳过（缺少必要输入）*

### Q2_trace_coverage_L1

- {'param': 'scaleValue', 'unique': 450, 'total': 651, 'is_scalar': True, 'type': 'scalar', 'min': -1.7976931348623157e+308, 'max': 1.7976931348623157e+308}

### Q3_output_indexes_L1

**total:** 651
**mismatches:** 0
**samples:** `[]`

### Q5_existence_L1

*跳过（缺少必要输入）*

### Q6_param_value_domain_L1

*跳过（缺少必要输入）*

### Q7_param_coverage_L1

**missing:** `[]`
**optional_absent:** `[]`
**coverage:** `{'scaleValue': '100%(651/651)'}`
