# -*- coding: utf-8 -*-
"""aclnnGridSampler3D 约束模块

来源: inputs/aclnnGridSampler3D/tests/st/design/01_parameter_description.md 约束关系 R1~R10
"""
from solver import solves, Candidates, SKIP, NOT_APPLICABLE
from solver.tags import tag

INT32_MAX = 2147483647            # 来源: R7 "INT32的最大值"
MAX_SHAPE_PRODUCT = 2147483648    # 引擎 shape 乘积上限（MAX_SHAPE_PRODUCT）

# ===== 约束追溯表 =====
# R1 (grid数据类型与input一致)  条件因子: 无 → 模式1:类型等值 → solve_grid_dtype
# R2 (out数据类型与input保持一致)  条件因子: 无 → 模式1:类型等值 → solve_out_dtype
# R3 (out数据格式与input保持一致)  条件因子: 无 → 模式1:类型等值 → solve_out_format
# R4 (input/grid均为5维，ND按NCDHW处理；grid.shape=(N,D_out,H_out,W_out,3)，N与input一致、末维固定3)  条件因子: 无 → 模式3:Shape公式 + 模式13:format→dimensions → solve_input_dimensions, solve_input_shape, solve_grid_shape
# R5 (out.shape=(N,C,D_out,H_out,W_out)，N/C取input、D/H/W取grid)  条件因子: 无 → 模式3:Shape公式 → solve_out_shape
# R6 (interpolationMode∈{0,1}；paddingMode∈{0,1,2})  条件因子: 无 → factor-domain（YAML .value枚举）→ 无（validate校验）
# R7 (input的D_in×H_in×W_in < INT32最大值)  条件因子: 无 → solve_input_shape 内 assert → solve_input_shape
# R8 (input空间三维≥1；N/C可为0；grid/out各维≥0)  条件因子: 无 → solve_input_shape/solve_grid_shape 构造（_d_in/_h_in/_w_in值域下界1）+ ET-S1~S5 → solve_input_shape, solve_grid_shape
# R9 (NDHWC条件格式：f32+mode=0+C%4==0)  条件因子: input.format → 模式8B:条件过滤 → solve_input_dtype [条件过滤], solve_interpolation_mode [条件过滤], solve_input_shape [条件过滤]
#   被约束参数: input.dtype, interpolationMode.value, input.shape(C轴)
#   条件因子: input.format
#   条件分支: format==NDHWC时 dtype=float32、mode=0、C向上对齐4的倍数；否则dtype全域、mode∈{0,1}、C任意
# R10 (grid归一化坐标[-1,1]，越界按paddingMode处理)  条件因子: paddingMode.value → YAML grid.value_range 分段（场景A/B）→ 无@solves（IMPL-RANGE内置推导；validate校验值域包络）
# ===== 追溯表结束 =====


@solves('input.dtype', sources=['input.format'])
def solve_input_dtype(input_format):
    """R9: NDHWC 仅当数据类型为 FLOAT32 时支持。"""
    if input_format == 'NDHWC':
        # 场景C8-S2: NDHWC → dtype 固定 float32
        tag('format_scenario', 'C8-S2')
        return 'float32'
    # 场景C8-S1: NCDHW/ND → dtype 全域
    tag('format_scenario', 'C8-S1')
    return Candidates(['float16', 'float32', 'float64', 'bfloat16'])


@solves('interpolationMode.value', sources=['input.format'])
def solve_interpolation_mode(input_format):
    """R9: NDHWC 仅当 interpolationMode 为 0（bilinear）时支持；R6: 3D 不支持 bicubic。"""
    if input_format == 'NDHWC':
        return 0
    return Candidates([0, 1])


@solves('grid.dtype', sources=['input.dtype'])
def solve_grid_dtype(input_dtype):
    """R1: grid的数据类型与入参input的数据类型一致。"""
    return input_dtype


@solves('out.dtype', sources=['input.dtype'])
def solve_out_dtype(input_dtype):
    """R2: out的数据类型与input保持一致。"""
    return input_dtype


@solves('out.format', sources=['input.format'])
def solve_out_format(input_format):
    """R3: out的数据格式与input保持一致。"""
    return input_format


@solves('input.dimensions', sources=['input.format'])
def solve_input_dimensions(input_format):
    """R4: input 为 5 维（NCDHW/NDHWC/ND 均为 5 维，ND 按 NCDHW 处理）。"""
    _FORMAT_DIMS = {'NCDHW': 5, 'NDHWC': 5, 'ND': 5}
    dims = _FORMAT_DIMS.get(input_format)
    assert dims is not None, f"unknown format: {input_format}"
    return dims


@solves('input.shape', sources=['input.dimensions', 'input.format', '_n.value', '_c.value',
                                '_d_in.value', '_h_in.value', '_w_in.value'])
def solve_input_shape(dimensions, input_format, n, c, d_in, h_in, w_in):
    """R4/R7/R8/R9: input.shape=(N, C, D_in, H_in, W_in)，空间三维≥1，D×H×W<INT32最大值；NDHWC 时 C 对齐 4 的倍数。"""
    assert dimensions == 5, f"input must be 5-D, got dimensions={dimensions}"
    # R8: input 空间三维（D_in、H_in、W_in）不能为 0（推断约束，见 01 待确认项）
    assert d_in >= 1 and h_in >= 1 and w_in >= 1, f"D_in/H_in/W_in must be >= 1, got {d_in}/{h_in}/{w_in}"
    assert n >= 0 and c >= 0, f"N/C must be >= 0, got {n}/{c}"
    # R9: NDHWC 时 C 轴为 4 的倍数（向上对齐；0 亦为 4 的倍数）
    if input_format == 'NDHWC':
        c = 4 * ((c + 3) // 4)
    # R7: D_in × H_in × W_in < INT32 最大值
    assert d_in * h_in * w_in < INT32_MAX, f"D*H*W={d_in * h_in * w_in} must be < {INT32_MAX}"
    # 引擎 shape 乘积上限
    assert n * c * d_in * h_in * w_in <= MAX_SHAPE_PRODUCT, \
        f"tensor product={n * c * d_in * h_in * w_in} exceeds {MAX_SHAPE_PRODUCT}"
    return [n, c, d_in, h_in, w_in]


@solves('grid.shape', sources=['grid.dimensions', 'input.shape', '_out_d.value', '_out_h.value', '_out_w.value'])
def solve_grid_shape(dimensions, input_shape, out_d, out_h, out_w):
    """R4: grid.shape=(N, D_out, H_out, W_out, 3)，N与input.shape[0]一致，最后一维固定为3。"""
    assert dimensions == 5, f"grid must be 5-D, got dimensions={dimensions}"
    n = input_shape[0]
    assert n >= 0 and out_d >= 0 and out_h >= 0 and out_w >= 0
    assert n * out_d * out_h * out_w * 3 <= MAX_SHAPE_PRODUCT
    return [n, out_d, out_h, out_w, 3]


@solves('out.shape', sources=['input.shape', 'grid.shape'])
def solve_out_shape(input_shape, grid_shape):
    """R5: out.shape=(N, C, D_out, H_out, W_out)，N、C取自input，D、H、W取自grid。"""
    out_shape = [input_shape[0], input_shape[1], grid_shape[1], grid_shape[2], grid_shape[3]]
    # C4 子场景标记（03_scenario_enumeration.md）
    if out_shape[0] == 0:
        tag('shape_scenario', 'C4-S2')  # batch 维空
    elif out_shape[1] == 0:
        tag('shape_scenario', 'C4-S3')  # 通道维空
    elif out_shape[2] == 0 or out_shape[3] == 0 or out_shape[4] == 0:
        tag('shape_scenario', 'C4-S4')  # 输出空
    else:
        tag('shape_scenario', 'C4-S1')  # 常规全维 >= 1
    product = out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3] * out_shape[4]
    assert product <= MAX_SHAPE_PRODUCT, f"out product={product} exceeds {MAX_SHAPE_PRODUCT}"
    assert out_shape[0] == input_shape[0] and out_shape[1] == input_shape[1]
    assert out_shape[2] == grid_shape[1] and out_shape[3] == grid_shape[2] and out_shape[4] == grid_shape[3]
    return out_shape


# ===== validate_constraints（循环 dispatch，杜绝漏调）=====
def validate_constraints(case):
    violations = []
    for r in ('R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10'):
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
    """R2: out的数据类型与input保持一致。"""
    violations = []
    input_dtype = case.get('input.dtype')
    out_dtype = case.get('out.dtype')
    if input_dtype is not None and out_dtype is not None and input_dtype != out_dtype:
        violations.append('R2_out_dtype_mismatch')
    return violations


def _validate_r3(case):
    """R3: out的数据格式与input保持一致。"""
    violations = []
    input_format = case.get('input.format')
    out_format = case.get('out.format')
    if input_format is not None and out_format is not None and input_format != out_format:
        violations.append('R3_out_format_mismatch')
    return violations


def _validate_r4(case):
    """R4: input/grid均为5维；grid.shape=(N,D_out,H_out,W_out,3)且N与input一致、末维固定为3。"""
    violations = []
    input_shape = case.get('input.shape')
    grid_shape = case.get('grid.shape')
    if grid_shape is None:
        return violations
    if len(grid_shape) != 5:
        violations.append('R4_grid_not_5d')
        return violations
    if grid_shape[4] != 3:
        violations.append('R4_grid_last_dim_not_3')
    if input_shape is not None and len(input_shape) == 5:
        if grid_shape[0] != input_shape[0]:
            violations.append('R4_batch_mismatch')
    return violations


def _validate_r5(case):
    """R5: out.shape=(N,C,D_out,H_out,W_out)，N、C与input一致，D、H、W与grid一致。"""
    violations = []
    input_shape = case.get('input.shape')
    grid_shape = case.get('grid.shape')
    out_shape = case.get('out.shape')
    if input_shape is None or grid_shape is None or out_shape is None:
        return violations
    if len(out_shape) != 5:
        violations.append('R5_out_not_5d')
        return violations
    if out_shape[0] != input_shape[0] or out_shape[1] != input_shape[1]:
        violations.append('R5_out_nc_mismatch')
    if out_shape[2] != grid_shape[1] or out_shape[3] != grid_shape[2] or out_shape[4] != grid_shape[3]:
        violations.append('R5_out_dhw_mismatch')
    return violations


def _validate_r6(case):
    """R6: interpolationMode∈{0,1}（3D不支持bicubic）；paddingMode∈{0,1,2}。"""
    violations = []
    mode = case.get('interpolationMode.value')
    padding = case.get('paddingMode.value')
    if mode is not None and mode not in (0, 1):
        violations.append('R6_interpolation_mode_out_of_range')
    if padding is not None and padding not in (0, 1, 2):
        violations.append('R6_padding_mode_out_of_range')
    return violations


def _validate_r7(case):
    """R7: input的D_in×H_in×W_in < INT32的最大值（2147483647）。"""
    violations = []
    input_shape = case.get('input.shape')
    if input_shape is None or len(input_shape) != 5:
        return violations
    if input_shape[2] * input_shape[3] * input_shape[4] >= INT32_MAX:
        violations.append('R7_dhw_exceeds_int32_max')
    return violations


def _validate_r8(case):
    """R8: input空间三维（D_in、H_in、W_in）≥1；N、C≥0；grid/out各维≥0（支持空Tensor）。"""
    violations = []
    input_shape = case.get('input.shape')
    grid_shape = case.get('grid.shape')
    out_shape = case.get('out.shape')
    if input_shape is not None and len(input_shape) == 5:
        if input_shape[2] < 1 or input_shape[3] < 1 or input_shape[4] < 1:
            violations.append('R8_input_spatial_dims_zero')
        if input_shape[0] < 0 or input_shape[1] < 0:
            violations.append('R8_input_nc_negative')
    if grid_shape is not None and any(d < 0 for d in grid_shape):
        violations.append('R8_grid_negative_dim')
    if out_shape is not None and any(d < 0 for d in out_shape):
        violations.append('R8_out_negative_dim')
    return violations


def _validate_r9(case):
    """R9: input.format=NDHWC时须同时满足dtype=float32、interpolationMode=0、C轴为4的倍数。"""
    violations = []
    input_format = case.get('input.format')
    if input_format is None or input_format != 'NDHWC':
        return violations
    input_dtype = case.get('input.dtype')
    mode = case.get('interpolationMode.value')
    input_shape = case.get('input.shape')
    if input_dtype is not None and input_dtype != 'float32':
        violations.append('R9_ndhwc_dtype_not_float32')
    if mode is not None and mode != 0:
        violations.append('R9_ndhwc_mode_not_bilinear')
    if input_shape is not None and len(input_shape) == 5 and input_shape[1] % 4 != 0:
        violations.append('R9_ndhwc_c_not_multiple_of_4')
    return violations


# 引擎 rangebound_highpre 边界覆盖目标允许的扩展段（dtype极值/inf/nan/±0），来源: generate_test_cases.py _FLOAT_DTYPE_EXTREME_VALUES
_R10_DTYPE_EXTREMES = {
    65504.0, -65504.0,                      # float16
    3.4028235e38, -3.4028235e38,            # float32
    1.7976931348623157e308, -1.7976931348623157e308,  # float64
    3.3895313892515355e38, -3.3895313892515355e38,    # bfloat16
}
_R10_SPECIAL_STRINGS = {'inf', '-inf', 'nan', '-0', '+0'}


def _validate_r10(case):
    """R10: grid元素为归一化坐标，基准区间[-1,1]，越界按paddingMode处理（值域分段：场景A [-1,1] + 场景B 轻度越界 [-3,3]）。"""
    violations = []
    vr = case.get('grid.value_range')
    if vr is None:
        return violations
    segments = vr if (isinstance(vr, list) and vr and isinstance(vr[0], (list, tuple))) else [vr]
    for seg in segments:
        if not (isinstance(seg, (list, tuple)) and len(seg) == 2):
            violations.append('R10_grid_range_malformed')
            break
        for v in seg:
            if isinstance(v, str):
                if v.lower() not in _R10_SPECIAL_STRINGS:
                    violations.append('R10_grid_range_out_of_domain')
            elif isinstance(v, (int, float)):
                fv = float(v)
                if fv not in _R10_DTYPE_EXTREMES and not (-3.0 <= fv <= 3.0):
                    violations.append('R10_grid_range_out_of_domain')
            else:
                violations.append('R10_grid_range_malformed')
    return violations
