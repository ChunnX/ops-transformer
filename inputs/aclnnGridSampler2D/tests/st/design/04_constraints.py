# -*- coding: utf-8 -*-
"""aclnnGridSampler2D 约束模块

来源: inputs/aclnnGridSampler2D/tests/st/design/01_parameter_description.md 约束关系 R1~R9
"""
from solver import solves, Candidates, SKIP, NOT_APPLICABLE
from solver.tags import tag

INT32_MAX = 2147483647            # 来源: R7 "INT32的最大值"
MAX_SHAPE_PRODUCT = 2147483648    # 引擎 shape 乘积上限（MAX_SHAPE_PRODUCT）

# ===== 约束追溯表 =====
# R1 (grid数据类型与input一致)  条件因子: 无 → 模式1:类型等值 → solve_grid_dtype
# R2 (out数据类型与input一致)  条件因子: 无 → 模式1:类型等值 → solve_out_dtype
# R3 (grid.shape=(N,H_out,W_out,2)，N与input一致、末维固定2；input/grid均为4维)  条件因子: 无 → 模式3:Shape公式 → solve_input_shape, solve_grid_shape
# R4 (out.shape=(N,C,H_out,W_out)，N/C取input、H_out/W_out取grid)  条件因子: 无 → 模式3:Shape公式 → solve_out_shape
# R5 (interpolationMode∈{0,1,2}；paddingMode∈{0,1,2})  条件因子: 无 → factor-domain（YAML .value枚举）→ 无（validate校验）
# R6 (bicubic(2)时dtype须为FLOAT32/FLOAT16)  条件因子: interpolationMode.value → 模式8B:条件过滤 → solve_input_dtype [条件过滤]
#   被约束参数: input.dtype
#   条件因子: interpolationMode.value
#   条件分支: mode==2时返回Candidates(['float32','float16'])，否则返回全dtype域
# R7 (input的H_in×W_in < INT32最大值)  条件因子: 无 → solve_input_shape 内 assert → solve_input_shape
# R8 (input最后两维≥1；N/C可为0；grid/out各维≥0)  条件因子: 无 → solve_input_shape/solve_grid_shape 构造（_h_in/_w_in值域下界1）+ ET-S1~S4 → solve_input_shape, solve_grid_shape
# R9 (grid归一化坐标[-1,1]，越界按paddingMode处理)  条件因子: paddingMode.value → YAML grid.value_range 分段（场景A/B）→ 无@solves（IMPL-RANGE内置推导；validate校验值域包络）
# ===== 追溯表结束 =====


@solves('input.dtype', sources=['interpolationMode.value'])
def solve_input_dtype(interpolation_mode):
    """R6: interpolationMode为bicubic(2)时，input/grid/out的数据类型须为FLOAT32或FLOAT16。"""
    if interpolation_mode == 2:
        # 场景C5-S2: bicubic → dtype 收窄为 float32/float16
        tag('interp_mode_scenario', 'C5-S2')
        return Candidates(['float32', 'float16'])
    # 场景C5-S1: 非 bicubic → dtype 全域
    tag('interp_mode_scenario', 'C5-S1')
    return Candidates(['float16', 'float32', 'float64', 'bfloat16'])


@solves('grid.dtype', sources=['input.dtype'])
def solve_grid_dtype(input_dtype):
    """R1: grid的数据类型与入参input的数据类型一致。"""
    return input_dtype


@solves('out.dtype', sources=['input.dtype'])
def solve_out_dtype(input_dtype):
    """R2: out的数据类型与input的数据类型一致。"""
    return input_dtype


@solves('input.shape', sources=['input.dimensions', '_n.value', '_c.value', '_h_in.value', '_w_in.value'])
def solve_input_shape(dimensions, n, c, h_in, w_in):
    """R3/R7/R8: input.shape=(N, C, H_in, W_in)，最后两维≥1，H×W<INT32最大值。"""
    assert dimensions == 4, f"input must be 4-D, got dimensions={dimensions}"
    # R8: input 的 shape 最后两维的维度值不能为 0
    assert h_in >= 1 and w_in >= 1, f"H_in/W_in must be >= 1, got {h_in}/{w_in}"
    assert n >= 0 and c >= 0, f"N/C must be >= 0, got {n}/{c}"
    # R7: H_in × W_in < INT32 最大值
    assert h_in * w_in < INT32_MAX, f"H*W={h_in * w_in} must be < {INT32_MAX}"
    # 引擎 shape 乘积上限
    assert n * c * h_in * w_in <= MAX_SHAPE_PRODUCT, \
        f"tensor product={n * c * h_in * w_in} exceeds {MAX_SHAPE_PRODUCT}"
    return [n, c, h_in, w_in]


@solves('grid.shape', sources=['grid.dimensions', 'input.shape', '_out_h.value', '_out_w.value'])
def solve_grid_shape(dimensions, input_shape, out_h, out_w):
    """R3: grid.shape=(N, H_out, W_out, 2)，N与input.shape[0]一致，最后一维固定为2。"""
    assert dimensions == 4, f"grid must be 4-D, got dimensions={dimensions}"
    n = input_shape[0]
    assert n >= 0 and out_h >= 0 and out_w >= 0
    assert n * out_h * out_w * 2 <= MAX_SHAPE_PRODUCT
    return [n, out_h, out_w, 2]


@solves('out.shape', sources=['input.shape', 'grid.shape'])
def solve_out_shape(input_shape, grid_shape):
    """R4: out.shape=(N, C, H_out, W_out)，N、C取自input，H_out、W_out取自grid。"""
    out_shape = [input_shape[0], input_shape[1], grid_shape[1], grid_shape[2]]
    # C3 子场景标记（03_scenario_enumeration.md）
    if out_shape[0] == 0:
        tag('shape_scenario', 'C3-S2')  # batch 维空
    elif out_shape[1] == 0:
        tag('shape_scenario', 'C3-S3')  # 通道维空
    elif out_shape[2] == 0 or out_shape[3] == 0:
        tag('shape_scenario', 'C3-S4')  # 输出空
    else:
        tag('shape_scenario', 'C3-S1')  # 常规全维 >= 1
    assert out_shape[0] == input_shape[0] and out_shape[1] == input_shape[1]
    assert out_shape[2] == grid_shape[1] and out_shape[3] == grid_shape[2]
    return out_shape


# ===== validate_constraints（循环 dispatch，杜绝漏调）=====
def validate_constraints(case):
    violations = []
    for r in ('R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9'):
        violations.extend(globals()[f'_validate_{r.lower()}'](case))
    return violations


def _validate_r1(case):
    """R1: grid的数据类型与入参input的数据类型一致。"""
    violations = []
    input_dtype = case.get('input.dtype')
    grid_dtype = case.get('grid.dtype')
    if input_dtype is not None and grid_dtype is not None and input_dtype != grid_dtype:
        violations.append('R1_grid_dtype_mismatch')
    return violations


def _validate_r2(case):
    """R2: out的数据类型与input的数据类型一致。"""
    violations = []
    input_dtype = case.get('input.dtype')
    out_dtype = case.get('out.dtype')
    if input_dtype is not None and out_dtype is not None and input_dtype != out_dtype:
        violations.append('R2_out_dtype_mismatch')
    return violations


def _validate_r3(case):
    """R3: input/grid均为4维；grid.shape=(N,H_out,W_out,2)且N与input一致、末维固定为2。"""
    violations = []
    input_shape = case.get('input.shape')
    grid_shape = case.get('grid.shape')
    if grid_shape is None:
        return violations
    if len(grid_shape) != 4:
        violations.append('R3_grid_not_4d')
        return violations
    if grid_shape[3] != 2:
        violations.append('R3_grid_last_dim_not_2')
    if input_shape is not None and len(input_shape) == 4:
        if grid_shape[0] != input_shape[0]:
            violations.append('R3_batch_mismatch')
    return violations


def _validate_r4(case):
    """R4: out.shape=(N,C,H_out,W_out)，N、C与input一致，H_out、W_out与grid一致。"""
    violations = []
    input_shape = case.get('input.shape')
    grid_shape = case.get('grid.shape')
    out_shape = case.get('out.shape')
    if input_shape is None or grid_shape is None or out_shape is None:
        return violations
    if len(out_shape) != 4:
        violations.append('R4_out_not_4d')
        return violations
    if out_shape[0] != input_shape[0] or out_shape[1] != input_shape[1]:
        violations.append('R4_out_nc_mismatch')
    if out_shape[2] != grid_shape[1] or out_shape[3] != grid_shape[2]:
        violations.append('R4_out_hw_mismatch')
    return violations


def _validate_r5(case):
    """R5: interpolationMode∈{0,1,2}；paddingMode∈{0,1,2}。"""
    violations = []
    mode = case.get('interpolationMode.value')
    padding = case.get('paddingMode.value')
    if mode is not None and mode not in (0, 1, 2):
        violations.append('R5_interpolation_mode_out_of_range')
    if padding is not None and padding not in (0, 1, 2):
        violations.append('R5_padding_mode_out_of_range')
    return violations


def _validate_r6(case):
    """R6: interpolationMode为bicubic时，input、grid、out的数据类型须为FLOAT32或FLOAT16。"""
    violations = []
    mode = case.get('interpolationMode.value')
    if mode is None or mode != 2:
        return violations
    for key in ('input.dtype', 'grid.dtype', 'out.dtype'):
        dt = case.get(key)
        if dt is not None and dt not in ('float32', 'float16'):
            violations.append('R6_bicubic_dtype_not_supported')
            break
    return violations


def _validate_r7(case):
    """R7: input的H_in×W_in < INT32的最大值（2147483647）。"""
    violations = []
    input_shape = case.get('input.shape')
    if input_shape is None or len(input_shape) != 4:
        return violations
    if input_shape[2] * input_shape[3] >= INT32_MAX:
        violations.append('R7_hw_exceeds_int32_max')
    return violations


def _validate_r8(case):
    """R8: input最后两维（H_in、W_in）≥1；N、C≥0；grid/out各维≥0（支持空Tensor）。"""
    violations = []
    input_shape = case.get('input.shape')
    grid_shape = case.get('grid.shape')
    out_shape = case.get('out.shape')
    if input_shape is not None and len(input_shape) == 4:
        if input_shape[2] < 1 or input_shape[3] < 1:
            violations.append('R8_input_last_two_dims_zero')
        if input_shape[0] < 0 or input_shape[1] < 0:
            violations.append('R8_input_nc_negative')
    if grid_shape is not None and any(d < 0 for d in grid_shape):
        violations.append('R8_grid_negative_dim')
    if out_shape is not None and any(d < 0 for d in out_shape):
        violations.append('R8_out_negative_dim')
    return violations


# 引擎 rangebound_highpre 边界覆盖目标允许的扩展段（dtype极值/inf/nan/±0），来源: generate_test_cases.py _FLOAT_DTYPE_EXTREME_VALUES
_R9_DTYPE_EXTREMES = {
    65504.0, -65504.0,                      # float16
    3.4028235e38, -3.4028235e38,            # float32
    1.7976931348623157e308, -1.7976931348623157e308,  # float64
    3.3895313892515355e38, -3.3895313892515355e38,    # bfloat16
}
_R9_SPECIAL_STRINGS = {'inf', '-inf', 'nan', '-0', '+0'}


def _validate_r9(case):
    """R9: grid元素为归一化坐标，基准区间[-1,1]，越界按paddingMode处理（值域分段：场景A [-1,1] + 场景B 轻度越界 [-3,3]）。"""
    violations = []
    vr = case.get('grid.value_range')
    if vr is None:
        return violations
    segments = vr if (isinstance(vr, list) and vr and isinstance(vr[0], (list, tuple))) else [vr]
    for seg in segments:
        if not (isinstance(seg, (list, tuple)) and len(seg) == 2):
            violations.append('R9_grid_range_malformed')
            break
        for v in seg:
            if isinstance(v, str):
                if v.lower() not in _R9_SPECIAL_STRINGS:
                    violations.append('R9_grid_range_out_of_domain')
            elif isinstance(v, (int, float)):
                fv = float(v)
                if fv not in _R9_DTYPE_EXTREMES and not (-3.0 <= fv <= 3.0):
                    violations.append('R9_grid_range_out_of_domain')
            else:
                violations.append('R9_grid_range_malformed')
    return violations
