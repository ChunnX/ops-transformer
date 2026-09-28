# QA 自动校验报告（C-Auto）

> 由 qa_verify_oracle.py 自动生成

## 总览

| 状态 | 数量 |
|------|------|
| PASS | 21 |
| FAIL | 0 |
| SKIP | 10 |

## 逐项结果

| 检查项 | 状态 | 说明 |
|--------|------|------|
| Q1_product_support | SKIP |  |
| Q2_trace_coverage | PASS | 3 条记录 |
| Q3_output_indexes | PASS | 不一致: 0/72 |
| Q4_yaml_domain | PASS |  |
| Q5_existence | SKIP |  |
| Q6_param_value_domain | SKIP |  |
| Q7_param_coverage | PASS | 全部必选入参已覆盖 |
| Q7_et_completeness | PASS |  |
| Q7b_et_completeness_enhanced | PASS |  |
| Q8_et_var_exist | PASS |  |
| Q9_bd_boundary | SKIP |  |
| Q9b_bd_boundary_yaml | PASS |  |
| Q10_ex_validity | PASS |  |
| Q11_ex_expected_error | PASS |  |
| Q12_scenario_id_format | PASS |  |
| Q13_et_case_coverage | PASS | 5 条记录 |
| Q14_bd_case_coverage | PASS | 6 条记录 |
| Q15_ex_case_coverage | SKIP |  |
| Q16_rn_ex_coverage | SKIP |  |
| Q17_locked_consistency | PASS |  |
| Q18_et_constraint | PASS |  |
| Q20_output_shape_fidelity | SKIP |  |
| Q13b_et_e2e | PASS | 5 条记录 |
| Q14b_bd_e2e | PASS | 6 条记录 |
| Q15b_ex_e2e | PASS | 4 条记录 |
| Q1_product_support_L1 | SKIP |  |
| Q2_trace_coverage_L1 | PASS | 3 条记录 |
| Q3_output_indexes_L1 | PASS | 不一致: 0/628 |
| Q5_existence_L1 | SKIP |  |
| Q6_param_value_domain_L1 | SKIP |  |
| Q7_param_coverage_L1 | PASS | 全部必选入参已覆盖 |

## 详细结果

### Q1_product_support

*跳过（缺少必要输入）*

### Q2_trace_coverage

- {'param': 'alignCorners', 'unique': 2, 'total': 72, 'is_scalar': True, 'type': 'scalar', 'min': False, 'max': True}
- {'param': 'interpolationMode', 'unique': 2, 'total': 72, 'is_scalar': True, 'type': 'scalar', 'min': 0, 'max': 1}
- {'param': 'paddingMode', 'unique': 3, 'total': 72, 'is_scalar': True, 'type': 'scalar', 'min': 0, 'max': 2}

### Q3_output_indexes

**total:** 72
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
**coverage:** `{'interpolationMode': '100%(72/72)', 'paddingMode': '100%(72/72)', 'alignCorners': '100%(72/72)'}`

### Q7_et_completeness


### Q7b_et_completeness_enhanced


### Q8_et_var_exist


### Q9_bd_boundary

*跳过（缺少必要输入）*

### Q9b_bd_boundary_yaml


### Q10_ex_validity


### Q11_ex_expected_error


### Q12_scenario_id_format


### Q13_et_case_coverage

- {'sid': 'ET-S1', 'found': True, 'cells': ['ET-S1', '_n.value=0', 'out 的 N 轴为 0，input/grid/out 全空', 'batch 维空 tensor，空到空']}
- {'sid': 'ET-S2', 'found': True, 'cells': ['ET-S2', '_c.value=0', 'out 的 C 轴为 0，input 与 out 通道维空', '通道维空 tensor']}
- {'sid': 'ET-S3', 'found': True, 'cells': ['ET-S3', '_out_d.value=0', 'out 的输出深度为 0，grid 输出深度同步为空', '输出深度为空（空输出）']}
- {'sid': 'ET-S4', 'found': True, 'cells': ['ET-S4', '_out_h.value=0', 'out 的输出高为 0，grid 输出高同步为空', '输出高为空（空输出）']}
- {'sid': 'ET-S5', 'found': True, 'cells': ['ET-S5', '_out_w.value=0', 'out 的输出宽为 0，grid 输出宽同步为空', '输出宽为空（空输出）']}

### Q14_bd_case_coverage

- {'sid': 'BD-S1', 'found': True, 'cells': ['BD-S1', '_d_in.value=1,_h_in.value=1,_w_in.value=1', '01 R8 空间三维下边界 1（推断）', '输入空间维下边界']}
- {'sid': 'BD-S2', 'found': True, 'cells': ['BD-S2', '_d_in.value=1290,_h_in.value=1290,_w_in.value=1290,_n.value=1,_c.value=1', '01 R7 D×H×W 上界（1290×1290×1290 逼近 INT32 最大值）', 'D×H×W 上边界']}
- {'sid': 'BD-S3', 'found': True, 'cells': ['BD-S3', '_out_d.value=1,_out_h.value=1,_out_w.value=1', '01 R8 输出维度下边界（取最小非空值 1）', '最小输出尺寸']}
- {'sid': 'BD-S4', 'found': True, 'cells': ['BD-S4', 'grid.value_range=[-1.0,-1.0]', '01 R10 归一化坐标下边界（取 -1）', '全部坐标取下边界']}
- {'sid': 'BD-S5', 'found': True, 'cells': ['BD-S5', 'grid.value_range=[1.0,1.0]', '01 R10 归一化坐标上边界（取 +1）', '全部坐标取上边界']}
- {'sid': 'BD-S6', 'found': True, 'cells': ['BD-S6', 'input.format=NDHWC,_c.value=4', '01 R9 C 轴 4 的倍数边界（最小正倍数）', 'NDHWC 条件格式的 C 对齐下边界']}

### Q15_ex_case_coverage

*跳过（缺少必要输入）*

### Q16_rn_ex_coverage

*跳过（缺少必要输入）*

### Q17_locked_consistency


### Q18_et_constraint


### Q20_output_shape_fidelity

*跳过（缺少必要输入）*

### Q13b_et_e2e

- {'sid': 'ET-S1', 'found': True}
- {'sid': 'ET-S2', 'found': True}
- {'sid': 'ET-S3', 'found': True}
- {'sid': 'ET-S4', 'found': True}
- {'sid': 'ET-S5', 'found': True}

### Q14b_bd_e2e

- {'sid': 'BD-S1', 'found': True}
- {'sid': 'BD-S2', 'found': True}
- {'sid': 'BD-S3', 'found': True}
- {'sid': 'BD-S4', 'found': True}
- {'sid': 'BD-S5', 'found': True}
- {'sid': 'BD-S6', 'found': True}

### Q15b_ex_e2e

- {'sid': 'EX-S1', 'found': True}
- {'sid': 'EX-S2', 'found': True}
- {'sid': 'EX-S3', 'found': True}
- {'sid': 'EX-S4', 'found': True}

### Q1_product_support_L1

*跳过（缺少必要输入）*

### Q2_trace_coverage_L1

- {'param': 'alignCorners', 'unique': 2, 'total': 628, 'is_scalar': True, 'type': 'scalar', 'min': False, 'max': True}
- {'param': 'interpolationMode', 'unique': 2, 'total': 628, 'is_scalar': True, 'type': 'scalar', 'min': 0, 'max': 1}
- {'param': 'paddingMode', 'unique': 3, 'total': 628, 'is_scalar': True, 'type': 'scalar', 'min': 0, 'max': 2}

### Q3_output_indexes_L1

**total:** 628
**mismatches:** 0
**samples:** `[]`

### Q5_existence_L1

*跳过（缺少必要输入）*

### Q6_param_value_domain_L1

*跳过（缺少必要输入）*

### Q7_param_coverage_L1

**missing:** `[]`
**optional_absent:** `[]`
**coverage:** `{'interpolationMode': '100%(628/628)', 'paddingMode': '100%(628/628)', 'alignCorners': '100%(628/628)'}`
