"""aclnnFusedFloydAttentionGrad 约束模块

预算分配链（constraint-writing-guide §3.3.2 模式B）：
  锚点: _h.value_range / _d.value / _b,_n,_m,_k.value_range（YAML 声明文档完整范围）
  推导: _h,_d → _b.value → _n.value → _m.value → _k.value（逐级 2G 预算收窄，
        保证 query/key1/key2 三个 5 维 tensor 的元素乘积均 ≤ MAX_SHAPE_PRODUCT）
  形状: 11 个 tensor 的 .shape 全部由中间因子推导（@solved）
  类型: query.dtype 为锚点，其余 dtype 等值推导
"""

import random

from solver import solves, Candidates, SKIP, NOT_APPLICABLE, tag

MAX_SHAPE_PRODUCT = 2147483648  # 2G，shape 元素乘积预算上限（与 utils.MAX_SHAPE_PRODUCT 一致）

# ===== 约束追溯表 =====
# R1 (B轴取值范围1~2K)  条件因子: 无 → @solves → solve_b_value（预算收窄: B*H*128*128*D ≤ 2G，K/M 下界 128）
# R2 (H轴取值范围1~256)  条件因子: 无 → factor-domain(_h.value_range 5段覆盖[1,256]) → validate _validate_r2
# R3 (N轴取值范围16~1M且N%16==0)  条件因子: 无 → @solves → solve_n_value（16对齐采样+预算收窄: B*H*N*128*D ≤ 2G）
# R4 (M轴取值范围128~1M且M%128==0)  条件因子: 无 → @solves → solve_m_value（128对齐采样+预算收窄: B*H*N*M*D ≤ 2G）
# R5 (K轴取值范围128~1M且K%128==0)  条件因子: 无 → @solves → solve_k_value（128对齐采样+预算收窄: min(key1,key2) 预算）
# R6 (D轴枚举{32,64,128})  条件因子: 无 → factor-domain(_d.value 显式枚举) → validate _validate_r6
# R7 (query与key1的第0/2/4轴需相同)  条件因子: 无 → @solves → solve_key1_shape（[B,H,N,K,D] 与 query 共享 _b/_n/_d 锚点）
# R8 (key1与value1 shape需相同)  条件因子: 无 → @solves → solve_value1_shape
# R9 (key2与value2 shape需相同)  条件因子: 无 → @solves → solve_value2_shape
# R10 (query与dy shape需相同)  条件因子: 无 → @solves → solve_dy_shape
# R11 (query与attentionIn shape需相同)  条件因子: 无 → @solves → solve_attentionIn_shape
# R12 (softmaxMax与softmaxSum shape需相同)  条件因子: 无 → @solves → solve_softmaxSum_shape
# R13 (attenMask shape为[B,1,N,1,K])  条件因子: attenMaskOptional.exist → @solves → solve_attenMask_shape（始终返回模板 shape，exist 语义由 CSV 渲染层处理，见函数 docstring）
# R14 (softmaxMax/softmaxSum shape为[B,H,N,M,8])  条件因子: 无 → @solves → solve_softmaxMax_shape（尾轴固定 8）
# R15 (五输入dtype一致)  条件因子: 无 → @solves → solve_key1_dtype / solve_value1_dtype / solve_key2_dtype / solve_value2_dtype（等值）
# R16 (attentionIn dtype与query一致)  条件因子: 无 → @solves → solve_attentionIn_dtype（等值）
# R17 (dy dtype与query一致)  条件因子: 无 → @solves → solve_dy_dtype（等值）
# R18 (输出dtype与query一致)  条件因子: 无 → @solves → solve_dqOut_dtype / solve_dk1Out_dtype / solve_dv1Out_dtype / solve_dk2Out_dtype / solve_dv2Out_dtype（等值）
# R19 (输出shape与对应输入相同)  条件因子: 无 → @solves → solve_dqOut_shape / solve_dk1Out_shape / solve_dv1Out_shape / solve_dk2Out_shape / solve_dv2Out_shape
# R20 (attenMask元素值域{0,1})  条件因子: 无 → factor-domain(attenMaskOptional.value_range_bool/uint8 = [[0,0],[1,1],[0,1]]) → validate _validate_r20
# ===== 追溯表结束 =====


# ============================================================
# 中间因子推导（预算分配链）
# ============================================================

@solves('_b.value', sources=['_b.value_range', '_h.value', '_d.value'])
def solve_b_value(b_range, h_val, d_val):
    """R1: B ∈ [1,2048]；预算收窄 B ≤ 2G//(H*D*128*128)（key2: B*H*K*M*D ≤ 2G，K/M 下界 128）。
    采样段整体超预算时收窄至 [1, cap]（场景收窄，不丢弃组合；cap ≥ 1 恒成立，因 H*D ≤ 256*128）"""
    lo, hi = b_range
    if hi <= 0:
        return 0  # EX 场景锁定 B=0（空轴异常，由 R1 标记违规）
    denom = h_val * d_val * 128 * 128
    cap = MAX_SHAPE_PRODUCT // denom if denom > 0 else MAX_SHAPE_PRODUCT
    hi2 = min(hi, cap)
    lo2 = max(lo, 1)
    if lo2 > hi2:
        lo2 = 1  # 段整体超预算：收窄至文档下界
    assert lo2 <= hi2, f"no feasible B in [{lo},{hi}] under budget cap={cap}"
    return random.randint(lo2, hi2)


@solves('_n.value', sources=['_n.value_range', '_b.value', '_h.value', '_d.value'])
def solve_n_value(n_range, b_val, h_val, d_val):
    """R3: N ∈ [16,1048576] 且 N%16==0；预算收窄 N ≤ 2G//(B*H*D*128)（query/key1: M/K 下界 128）。
    cap ≥ 16 恒成立（B ≤ 2G//(H*D*16384) → B*H*D ≤ 131072 → cap ≥ 128）；段超预算时收窄至 [16, cap]"""
    lo, hi = n_range
    if hi <= 0:
        return 0  # EX 场景锁定 N=0
    if hi < 16:
        return max(lo, 0)  # EX 场景锁定区间整体低于文档下界 16（如 [15,15]），返回原值由 R3 标记违规
    denom = b_val * h_val * d_val * 128
    cap = MAX_SHAPE_PRODUCT // denom if denom > 0 else MAX_SHAPE_PRODUCT
    hi2 = min(hi, cap)
    lo2 = max(lo, 16)
    if lo2 > hi2:
        lo2 = 16  # 段整体超预算：收窄至文档下界
    assert hi2 >= 16, f"no feasible N in [{lo},{hi}] under budget cap={cap}"
    start = ((lo2 + 15) // 16) * 16
    if start > hi2:
        return lo2  # EX 场景锁定区间内无 16 对齐值（如 [17,17]），返回原值由 R3 标记对齐违规
    return random.choice(range(start, hi2 + 1, 16))


@solves('_m.value', sources=['_m.value_range', '_b.value', '_h.value', '_n.value', '_d.value'])
def solve_m_value(m_range, b_val, h_val, n_val, d_val):
    """R4: M ∈ [128,1048576] 且 M%128==0；
    预算收窄 M ≤ min(2G//(B*H*N*D), 2G//(B*H*128*D))（query 预算 + key2 预留 K 下界 128 预算）。
    cap ≥ 128 恒成立（N ≤ 2G//(B*H*D*128) → cap_q ≥ 128；B*H*D ≤ 131072 → cap_k ≥ 128）；
    段超预算时收窄至 [128, cap]"""
    lo, hi = m_range
    if hi <= 0:
        return 0  # EX 场景锁定 M=0
    if hi < 128:
        return max(lo, 0)  # EX 场景锁定区间整体低于文档下界 128
    denom_q = b_val * h_val * n_val * d_val
    denom_k = b_val * h_val * 128 * d_val
    cap_q = MAX_SHAPE_PRODUCT // denom_q if denom_q > 0 else MAX_SHAPE_PRODUCT
    cap_k = MAX_SHAPE_PRODUCT // denom_k if denom_k > 0 else MAX_SHAPE_PRODUCT
    cap = min(cap_q, cap_k)
    hi2 = min(hi, cap)
    lo2 = max(lo, 128)
    if lo2 > hi2:
        lo2 = 128  # 段整体超预算：收窄至文档下界
    assert hi2 >= 128, f"no feasible M in [{lo},{hi}] under budget cap={cap}"
    start = ((lo2 + 127) // 128) * 128
    if start > hi2:
        return lo2  # EX 场景锁定区间内无 128 对齐值（如 [129,129]），返回原值由 R4 标记对齐违规
    return random.choice(range(start, hi2 + 1, 128))


@solves('_k.value', sources=['_k.value_range', '_b.value', '_h.value', '_n.value', '_m.value', '_d.value'])
def solve_k_value(k_range, b_val, h_val, n_val, m_val, d_val):
    """R5: K ∈ [128,1048576] 且 K%128==0；
    预算收窄 K ≤ min(2G//(B*H*N*D), 2G//(B*H*M*D))（key1: B*H*N*K*D；key2: B*H*K*M*D）。
    cap ≥ 128 恒成立（N/M 均被上级预算约束 → 两个分母均 ≤ 2G/128）；段超预算时收窄至 [128, cap]"""
    lo, hi = k_range
    if hi <= 0:
        return 0  # EX 场景锁定 K=0
    if hi < 128:
        return max(lo, 0)  # EX 场景锁定区间整体低于文档下界 128（如 [100,100]）
    denom1 = b_val * h_val * n_val * d_val
    denom2 = b_val * h_val * m_val * d_val
    cap1 = MAX_SHAPE_PRODUCT // denom1 if denom1 > 0 else MAX_SHAPE_PRODUCT
    cap2 = MAX_SHAPE_PRODUCT // denom2 if denom2 > 0 else MAX_SHAPE_PRODUCT
    cap = min(cap1, cap2)
    hi2 = min(hi, cap)
    lo2 = max(lo, 128)
    if lo2 > hi2:
        lo2 = 128  # 段整体超预算：收窄至文档下界
    assert hi2 >= 128, f"no feasible K in [{lo},{hi}] under budget cap={cap}"
    start = ((lo2 + 127) // 128) * 128
    if start > hi2:
        return lo2  # EX 场景锁定区间内无 128 对齐值，返回原值由 R5 标记对齐违规
    return random.choice(range(start, hi2 + 1, 128))


# ============================================================
# Shape 推导
# ============================================================

@solves('query.shape', sources=['_b.value', '_h.value', '_n.value', '_m.value', '_d.value'])
def solve_query_shape(b, h, n, m, d):
    """R1~R6: query.shape = [B,H,N,M,D]"""
    shape = [int(b), int(h), int(n), int(m), int(d)]
    assert len(shape) == 5, f"query.shape len {len(shape)} != 5"
    assert all(v >= 0 for v in shape), f"negative axis in query.shape: {shape}"
    assert b * h * n * m * d <= MAX_SHAPE_PRODUCT, f"query product exceeds 2G: {shape}"
    return shape


@solves('key1.shape', sources=['_b.value', '_h.value', '_n.value', '_k.value', '_d.value'])
def solve_key1_shape(b, h, n, k, d):
    """R7: key1.shape = [B,H,N,K,D]，第 0/2/4 轴与 query 相同（共享 _b/_n/_d）"""
    shape = [int(b), int(h), int(n), int(k), int(d)]
    assert len(shape) == 5, f"key1.shape len {len(shape)} != 5"
    assert all(v >= 0 for v in shape), f"negative axis in key1.shape: {shape}"
    assert b * h * n * k * d <= MAX_SHAPE_PRODUCT, f"key1 product exceeds 2G: {shape}"
    return shape


@solves('value1.shape', sources=['key1.shape'])
def solve_value1_shape(key1_shape):
    """R8: value1.shape 与 key1.shape 相同"""
    assert key1_shape is not None, "key1.shape is None"
    return list(key1_shape)


@solves('key2.shape', sources=['_b.value', '_h.value', '_k.value', '_m.value', '_d.value'])
def solve_key2_shape(b, h, k, m, d):
    """key2.shape = [B,H,K,M,D]（B/H 轴与 query 一致，K/M 独立）"""
    shape = [int(b), int(h), int(k), int(m), int(d)]
    assert len(shape) == 5, f"key2.shape len {len(shape)} != 5"
    assert all(v >= 0 for v in shape), f"negative axis in key2.shape: {shape}"
    assert b * h * k * m * d <= MAX_SHAPE_PRODUCT, f"key2 product exceeds 2G: {shape}"
    return shape


@solves('value2.shape', sources=['key2.shape'])
def solve_value2_shape(key2_shape):
    """R9: value2.shape 与 key2.shape 相同"""
    assert key2_shape is not None, "key2.shape is None"
    return list(key2_shape)


@solves('dy.shape', sources=['query.shape'])
def solve_dy_shape(query_shape):
    """R10: dy.shape 与 query.shape 相同"""
    assert query_shape is not None, "query.shape is None"
    return list(query_shape)


@solves('attentionIn.shape', sources=['query.shape'])
def solve_attentionIn_shape(query_shape):
    """R11: attentionIn.shape 与 query.shape 相同"""
    assert query_shape is not None, "query.shape is None"
    return list(query_shape)


@solves('softmaxMax.shape', sources=['query.shape'])
def solve_softmaxMax_shape(query_shape):
    """R14: softmaxMax.shape = [B,H,N,M,8]（B/H/N/M 轴与 query 一致，尾轴固定 8）"""
    assert query_shape is not None and len(query_shape) == 5, f"invalid query.shape: {query_shape}"
    shape = [int(query_shape[0]), int(query_shape[1]), int(query_shape[2]), int(query_shape[3]), 8]
    assert b_prod(shape) <= MAX_SHAPE_PRODUCT or 0 in shape, f"softmaxMax product exceeds 2G: {shape}"
    return shape


@solves('softmaxSum.shape', sources=['softmaxMax.shape'])
def solve_softmaxSum_shape(softmaxMax_shape):
    """R12: softmaxSum.shape 与 softmaxMax.shape 相同"""
    assert softmaxMax_shape is not None, "softmaxMax.shape is None"
    return list(softmaxMax_shape)


@solves('attenMaskOptional.shape',
        sources=['attenMaskOptional.exist', '_b.value', '_n.value', '_k.value'])
def solve_attenMask_shape(mask_exist, b, n, k):
    """R13: attenMaskOptional.shape = [B,1,N,1,K]。
    始终返回模板 shape（不按 exist 返回 NOT_APPLICABLE）：L2 回退填充时 exist 因子会被重新采样，
    若按 exist 返回 NOT_APPLICABLE 会与 case 中已记录的 exist 不一致（exist=True 但 shape=None），
    导致转换层 has_required_none_shape 丢行、行数错位。CSV 渲染由 exist 列决定 absent 参数输出 None，
    语义不受影响（R13 校验在 validate 中按 exist 跳过）。"""
    if mask_exist is None:
        return NOT_APPLICABLE  # exist 因子未求解（SKIP）时的防御分支
    shape = [int(b), 1, int(n), 1, int(k)]
    assert len(shape) == 5, f"attenMask.shape len {len(shape)} != 5"
    assert all(v >= 0 for v in shape), f"negative axis in attenMask.shape: {shape}"
    return shape


@solves('dqOut.shape', sources=['query.shape'])
def solve_dqOut_shape(query_shape):
    """R19: dqOut.shape 与 query.shape 相同"""
    assert query_shape is not None, "query.shape is None"
    return list(query_shape)


@solves('dk1Out.shape', sources=['key1.shape'])
def solve_dk1Out_shape(key1_shape):
    """R19: dk1Out.shape 与 key1.shape 相同"""
    assert key1_shape is not None, "key1.shape is None"
    return list(key1_shape)


@solves('dv1Out.shape', sources=['value1.shape'])
def solve_dv1Out_shape(value1_shape):
    """R19: dv1Out.shape 与 value1.shape 相同"""
    assert value1_shape is not None, "value1.shape is None"
    return list(value1_shape)


@solves('dk2Out.shape', sources=['key2.shape'])
def solve_dk2Out_shape(key2_shape):
    """R19: dk2Out.shape 与 key2.shape 相同"""
    assert key2_shape is not None, "key2.shape is None"
    return list(key2_shape)


@solves('dv2Out.shape', sources=['value2.shape'])
def solve_dv2Out_shape(value2_shape):
    """R19: dv2Out.shape 与 value2.shape 相同"""
    assert value2_shape is not None, "value2.shape is None"
    return list(value2_shape)


def b_prod(shape):
    """shape 元素乘积（空/含 None 安全）"""
    result = 1
    for v in shape:
        result *= int(v)
    return result


# ============================================================
# dtype 推导（等值链，query.dtype 为锚点）
# ============================================================

@solves('key1.dtype', sources=['query.dtype'])
def solve_key1_dtype(query_dtype):
    """R15: key1.dtype 与 query.dtype 一致"""
    return query_dtype


@solves('value1.dtype', sources=['query.dtype'])
def solve_value1_dtype(query_dtype):
    """R15: value1.dtype 与 query.dtype 一致"""
    return query_dtype


@solves('key2.dtype', sources=['query.dtype'])
def solve_key2_dtype(query_dtype):
    """R15: key2.dtype 与 query.dtype 一致"""
    return query_dtype


@solves('value2.dtype', sources=['query.dtype'])
def solve_value2_dtype(query_dtype):
    """R15: value2.dtype 与 query.dtype 一致"""
    return query_dtype


@solves('dy.dtype', sources=['query.dtype'])
def solve_dy_dtype(query_dtype):
    """R17: dy.dtype 与 query.dtype 一致（推断自：dy 为正向输出 Y 的梯度）"""
    return query_dtype


@solves('attentionIn.dtype', sources=['query.dtype'])
def solve_attentionIn_dtype(query_dtype):
    """R16: attentionIn.dtype 与 query.dtype 一致"""
    return query_dtype


@solves('dqOut.dtype', sources=['query.dtype'])
def solve_dqOut_dtype(query_dtype):
    """R18: dqOut.dtype 与 query.dtype 一致"""
    return query_dtype


@solves('dk1Out.dtype', sources=['query.dtype'])
def solve_dk1Out_dtype(query_dtype):
    """R18: dk1Out.dtype 与 query.dtype 一致"""
    return query_dtype


@solves('dv1Out.dtype', sources=['query.dtype'])
def solve_dv1Out_dtype(query_dtype):
    """R18: dv1Out.dtype 与 query.dtype 一致"""
    return query_dtype


@solves('dk2Out.dtype', sources=['query.dtype'])
def solve_dk2Out_dtype(query_dtype):
    """R18: dk2Out.dtype 与 query.dtype 一致"""
    return query_dtype


@solves('dv2Out.dtype', sources=['query.dtype'])
def solve_dv2Out_dtype(query_dtype):
    """R18: dv2Out.dtype 与 query.dtype 一致"""
    return query_dtype


# ============================================================
# validate_constraints（R1~R20 反向校验，从 01_parameter_description.md 独立转录）
# ============================================================

def validate_constraints(case):
    violations = []
    for r in ('R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10',
              'R11', 'R12', 'R13', 'R14', 'R15', 'R16', 'R17', 'R18', 'R19', 'R20'):
        violations.extend(globals()[f'_validate_{r.lower()}'](case))
    return violations


def _validate_r1(case):
    """R1: B：取值范围为1~2K（B ∈ [1, 2048]）"""
    violations = []
    b = case.get('_b.value')
    if b is not None and not (1 <= b <= 2048):
        violations.append('R1_b_out_of_range')
    return violations


def _validate_r2(case):
    """R2: H：取值范围为1~256（H ∈ [1, 256]）"""
    violations = []
    h = case.get('_h.value')
    if h is not None and not (1 <= h <= 256):
        violations.append('R2_h_out_of_range')
    return violations


def _validate_r3(case):
    """R3: N：取值范围为16~1M且N%16==0"""
    violations = []
    n = case.get('_n.value')
    if n is not None:
        if not (16 <= n <= 1048576):
            violations.append('R3_n_out_of_range')
        elif n % 16 != 0:
            violations.append('R3_n_not_16_aligned')
    return violations


def _validate_r4(case):
    """R4: M：取值范围为128~1M且M%128==0"""
    violations = []
    m = case.get('_m.value')
    if m is not None:
        if not (128 <= m <= 1048576):
            violations.append('R4_m_out_of_range')
        elif m % 128 != 0:
            violations.append('R4_m_not_128_aligned')
    return violations


def _validate_r5(case):
    """R5: K：取值范围为128~1M且K%128==0"""
    violations = []
    k = case.get('_k.value')
    if k is not None:
        if not (128 <= k <= 1048576):
            violations.append('R5_k_out_of_range')
        elif k % 128 != 0:
            violations.append('R5_k_not_128_aligned')
    return violations


def _validate_r6(case):
    """R6: D：取值范围为32/64/128（D ∈ {32, 64, 128}）"""
    violations = []
    d = case.get('_d.value')
    if d is not None and d not in (32, 64, 128):
        violations.append('R6_d_invalid')
    return violations


def _validate_r7(case):
    """R7: query与key1的第0/2/4轴需相同"""
    violations = []
    q = case.get('query.shape')
    k1 = case.get('key1.shape')
    if q is not None and k1 is not None and len(q) == 5 and len(k1) == 5:
        if q[0] != k1[0] or q[2] != k1[2] or q[4] != k1[4]:
            violations.append('R7_query_key1_axis_mismatch')
    return violations


def _validate_r8(case):
    """R8: key1与value1 shape需相同"""
    violations = []
    k1 = case.get('key1.shape')
    v1 = case.get('value1.shape')
    if k1 is not None and v1 is not None and list(k1) != list(v1):
        violations.append('R8_key1_value1_shape_mismatch')
    return violations


def _validate_r9(case):
    """R9: key2与value2 shape需相同"""
    violations = []
    k2 = case.get('key2.shape')
    v2 = case.get('value2.shape')
    if k2 is not None and v2 is not None and list(k2) != list(v2):
        violations.append('R9_key2_value2_shape_mismatch')
    return violations


def _validate_r10(case):
    """R10: query与dy shape需相同"""
    violations = []
    q = case.get('query.shape')
    dy = case.get('dy.shape')
    if q is not None and dy is not None and list(q) != list(dy):
        violations.append('R10_query_dy_shape_mismatch')
    return violations


def _validate_r11(case):
    """R11: query与attentionIn shape需相同"""
    violations = []
    q = case.get('query.shape')
    ai = case.get('attentionIn.shape')
    if q is not None and ai is not None and list(q) != list(ai):
        violations.append('R11_query_attentionIn_shape_mismatch')
    return violations


def _validate_r12(case):
    """R12: softmaxMax与softmaxSum shape需相同"""
    violations = []
    sm = case.get('softmaxMax.shape')
    ss = case.get('softmaxSum.shape')
    if sm is not None and ss is not None and list(sm) != list(ss):
        violations.append('R12_softmax_shape_mismatch')
    return violations


def _validate_r13(case):
    """R13: attenMaskOptional 存在时 shape 为 [B,1,N,1,K]（B/N 轴与 query 一致，K 轴与 key1 一致）"""
    violations = []
    mask_exist = case.get('attenMaskOptional.exist')
    if not mask_exist:
        return violations
    mask_shape = case.get('attenMaskOptional.shape')
    q = case.get('query.shape')
    k1 = case.get('key1.shape')
    if mask_shape is not None and q is not None and k1 is not None and len(q) == 5 and len(k1) == 5:
        expected = [q[0], 1, q[2], 1, k1[3]]
        if list(mask_shape) != expected:
            violations.append('R13_atten_mask_shape_mismatch')
    return violations


def _validate_r14(case):
    """R14: softmaxMax/softmaxSum shape 为 [B,H,N,M,8]（B/H/N/M 轴与 query 一致，尾轴固定 8）"""
    violations = []
    q = case.get('query.shape')
    sm = case.get('softmaxMax.shape')
    ss = case.get('softmaxSum.shape')
    if q is not None and len(q) == 5:
        expected = [q[0], q[1], q[2], q[3], 8]
        if sm is not None and list(sm) != expected:
            violations.append('R14_softmaxMax_shape_mismatch')
        if ss is not None and list(ss) != expected:
            violations.append('R14_softmaxSum_shape_mismatch')
    return violations


def _validate_r15(case):
    """R15: query、key1、value1、key2、value2 数据类型一致"""
    violations = []
    dtypes = [case.get(f'{p}.dtype') for p in ('query', 'key1', 'value1', 'key2', 'value2')]
    anchor = dtypes[0]
    if anchor is None:
        return violations
    for p, dt in zip(('key1', 'value1', 'key2', 'value2'), dtypes[1:]):
        if dt is not None and dt != anchor:
            violations.append(f'R15_{p}_dtype_mismatch')
    return violations


def _validate_r16(case):
    """R16: attentionIn 数据类型和 shape 类型与 query 保持一致（dtype 维度）"""
    violations = []
    q = case.get('query.dtype')
    ai = case.get('attentionIn.dtype')
    if q is not None and ai is not None and q != ai:
        violations.append('R16_attentionIn_dtype_mismatch')
    return violations


def _validate_r17(case):
    """R17: dy 数据类型与 query 一致（推断自：dy 为正向输出 Y 的梯度，Y dtype 与 query 一致）"""
    violations = []
    q = case.get('query.dtype')
    dy = case.get('dy.dtype')
    if q is not None and dy is not None and q != dy:
        violations.append('R17_dy_dtype_mismatch')
    return violations


def _validate_r18(case):
    """R18: 输出数据类型与 query 一致（infershape: SetOutputDataType(0..4, GetInputDataType(0))）"""
    violations = []
    q = case.get('query.dtype')
    if q is None:
        return violations
    for p in ('dqOut', 'dk1Out', 'dv1Out', 'dk2Out', 'dv2Out'):
        dt = case.get(f'{p}.dtype')
        if dt is not None and dt != q:
            violations.append(f'R18_{p}_dtype_mismatch')
    return violations


def _validate_r19(case):
    """R19: 输出 shape 与对应输入相同（dqOut=query; dk1Out=key1; dv1Out=value1; dk2Out=key2; dv2Out=value2）"""
    violations = []
    pairs = [('dqOut', 'query'), ('dk1Out', 'key1'), ('dv1Out', 'value1'),
             ('dk2Out', 'key2'), ('dv2Out', 'value2')]
    for out_p, in_p in pairs:
        out_shape = case.get(f'{out_p}.shape')
        in_shape = case.get(f'{in_p}.shape')
        if out_shape is not None and in_shape is not None and list(out_shape) != list(in_shape):
            violations.append(f'R19_{out_p}_shape_mismatch')
    return violations


def _validate_r20(case):
    """R20: attenMask 元素值域 {0,1}（取值为1代表该位不参与计算，为0代表该位参与计算）"""
    violations = []
    mask_exist = case.get('attenMaskOptional.exist')
    if not mask_exist:
        return violations
    for vr_key in ('attenMaskOptional.value_range_bool', 'attenMaskOptional.value_range_uint8'):
        vr = case.get(vr_key)
        if vr is None:
            continue
        segments = vr if isinstance(vr, list) and vr and isinstance(vr[0], (list, tuple)) else [vr]
        for seg in segments:
            if isinstance(seg, (list, tuple)) and len(seg) == 2:
                lo, hi = seg
                if not (0 <= lo <= hi <= 1):
                    violations.append(f'R20_mask_value_out_of_range')
                    break
    return violations
