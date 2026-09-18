#!/usr/bin/env python3
"""QkvRmsNormRopeCache TTK 用例生成器(arch35 / Ascend950)。

产出:
  tests/st/arch35/ttk_kernel_qkv_rms_norm_rope_cache_st.csv   —— kernel 通路
  tests/st/arch35/ttk_aclnn_qkv_rms_norm_rope_cache_st.csv    —— aclnn 通路

用例维度(覆盖 01 §3 的支持面):
  - dtype 组合:k_cache / v_cache 各自 ∈ {同 qkv dtype, int8}(8 档)
  - Nq/Nk/Nv:Nk==Nv、Nq%Nk==0 的多组取值
  - BlockSize:32 / 64 / 128 / 256(BlockSize % 32 == 0 的边界与内点)
  - token 数:单核 / 多核 / 末核余数 0 与非 0
  - 量化:不量化 / 对称 / 非对称(k_offset / v_offset)
  - is_output_qkv:True / False
  - index:全有效 / 含 -1

用法:
  python3 tests/assets/make_cases.py
"""

import os

D = 128
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(HERE), "st", "arch35")

# (tag, qkv_dtype, k_cache_dtype, v_cache_dtype)
DTYPE_COMBOS = [
    ("fp16", "float16", "float16", "float16"),
    ("bf16", "bfloat16", "bfloat16", "bfloat16"),
    ("fp16_kq_vq", "float16", "int8", "int8"),
    ("bf16_kq_vq", "bfloat16", "int8", "int8"),
    ("fp16_kq", "float16", "int8", "float16"),
    ("bf16_kq", "bfloat16", "int8", "bfloat16"),
    ("fp16_vq", "float16", "float16", "int8"),
    ("bf16_vq", "bfloat16", "bfloat16", "int8"),
]


def cache_shape(block_num, head_num, block_size, elem_bytes):
    d0 = 32 // elem_bytes
    return [block_num, head_num * D // d0, block_size, d0]


def elem_bytes(dtype):
    return 1 if dtype == "int8" else 2


def make_case(name, b, s, nq, nk, nv, dtype_tag, qkv_dt, k_dt, v_dt, block_size, offset, is_output_qkv,
              with_neg_index, block_pad=0, omit_optional=False):
    """block_size 必须整除 s(blockNum*BlockSize == B*S):kernel 通路的 in-place cache 输出
    与输入是两个 buffer,TTK 比较的是输出 buffer;只有每个 slot 都被写过,输出 buffer 的
    内容才完全由算子决定(未写 slot 保留的是 buffer 初值,与 golden 的 input clone 不同)。"""
    n_all = nq + nk + nv
    total = b * s
    block_num = -(-s // block_size) * b + block_pad

    in_shapes = [
        [total, n_all * D],                 # qkv
        [D],                                # q_gamma
        [D],                                # k_gamma
        [total, D],                         # cos
        [total, D],                         # sin
        [total],                            # index
        [total, nq * D],                    # q_out (in-place)
        cache_shape(block_num, nk, block_size, elem_bytes(k_dt)),   # k_cache (in-place)
        cache_shape(block_num, nv, block_size, elem_bytes(v_dt)),   # v_cache (in-place)
        [nk, D] if k_dt == "int8" else None,                        # k_scale
        [nv, D] if v_dt == "int8" else None,                        # v_scale
        [nk, D] if (k_dt == "int8" and offset) else None,           # k_offset
        [nv, D] if (v_dt == "int8" and offset) else None,           # v_offset
    ]
    in_dtypes = [qkv_dt, qkv_dt, qkv_dt, qkv_dt, qkv_dt, "int64", qkv_dt, k_dt, v_dt]
    for dt, shp in zip(("float32",) * 4, in_shapes[9:]):
        in_dtypes.append(dt if shp is not None else "float32")
    out_shapes = [[total, nq * D], cache_shape(block_num, nk, block_size, elem_bytes(k_dt)),
                  cache_shape(block_num, nv, block_size, elem_bytes(v_dt)), [total, nq * D], [total, nk * D],
                  [total, nv * D]]
    out_dtypes = [qkv_dt, k_dt, v_dt, qkv_dt, qkv_dt, qkv_dt]
    ranges = [(-1, 1)] * 5 + [(0, 0)] + [(-1, 1), (0, 0), (0, 0)]
    ranges += [(0, 1)] * 4
    # omit_optional:不带 cache_mode / epsilon,走 def 的缺省值(PA_NZ / 1e-6)。
    # R2 要求"不传"与"传了某值"同权覆盖 —— 用户最常走的恰恰是缺省。
    # is_output_qkv 不能省:省了默认 false,3 个 proto 输出不写,kernel 通路无法比对。
    attrs = {
        "qkv_size": [b, s, n_all, D],
        "head_nums": [nq, nk, nv],
        "epsilon": 1e-6,
        "cache_mode": "PA_NZ",
        "is_output_qkv": is_output_qkv,
    }
    if omit_optional:
        attrs.pop("epsilon")
        attrs.pop("cache_mode")
    return {
        "name": name,
        "dtype_tag": dtype_tag,
        "in_shapes": in_shapes,
        "in_dtypes": in_dtypes,
        "out_shapes": out_shapes,
        "out_dtypes": out_dtypes,
        "ranges": ranges,
        "attrs": attrs,
        "index_has_neg": with_neg_index,
        # kernel 通路可用:6 个输出都写,且 cache 容量恰好等于 token 数
        # (index 取遍所有 slot,每个 slot 都被写 -> 输出 buffer 内容完全由算子决定)
        #
        # 【bf16 已纳入 kernel 通路】(先前曾因误判根因被剔除,2026-09-18 修复后补回)
        # 曾经的症状:bf16 的原地输出(q_out/k_cache/v_cache)恒报元素数 2 倍
        # (262144 vs 131072),而独立输出(*_before_quant)正常。
        # 真实根因【不在 TTK,在本文件同目录的 golden.py】:
        #   kernel 侧 customize_inputs 复用了 _prepare->_to_tensor,而 _to_tensor 因为
        #   torch.from_numpy 不认 ml_dtypes bf16,把输入 astype 成了 float32;
        #   TTK 会把 customize_inputs 的返回值【整体替换】context.input_arrays
        #   (ttk input_generation.py:274-284),于是"声明 bf16、实为 fp32"的缓冲
        #   进了内核。TTK 的原地输出本就直接复用输入缓冲(profiling_structure.py:66-68),
        #   4 字节/元素的数组再按 bf16(2 字节)比较 -> 元素数 2 倍。
        #   fp16 在 torch 安全 dtype 集内、往返无损,故只有 bf16 中招。
        # 修法:customize_inputs 改走 _prepare(for_npu=True) —— 全程 numpy,
        #   只重建 index,其余输入原对象返回,dtype 结构上不可能漂移。
        # 实测:bf16 原地用例 0/1 -> 1/1;kernel 32/32、aclnn 43/43 无回归。
        "kernel_ok": is_output_qkv and (block_num * block_size == total),
    }


def build_cases():
    cases = []
    # ── 1. dtype 8 档 × 中等 shape(blockSize 64 整除 S=64) ──
    for tag, qkv_dt, k_dt, v_dt in DTYPE_COMBOS:
        cases.append(make_case("dtype_%s_b2_s64" % tag, 2, 64, 8, 1, 1, tag, qkv_dt, k_dt, v_dt, 64, False,
                               True, False))
    # ── 2. head 组合(Nk == Nv,Nq % Nk == 0) ──
    head_combos = [(1, 1, 1), (8, 1, 1), (16, 1, 1), (8, 2, 2), (16, 4, 4), (4, 4, 4), (2, 2, 2), (32, 2, 2)]
    for nq, nk, nv in head_combos:
        cases.append(make_case("head_q%d_k%d_v%d" % (nq, nk, nv), 1, 32, nq, nk, nv, "fp16", "float16", "float16",
                               "float16", 32, False, True, False))
    # ── 3. BlockSize 扫描(32B 对齐的边界与内点;S 取 BlockSize 的整数倍) ──
    for bs in (32, 64, 96, 128, 256, 512):
        cases.append(make_case("bs%d" % bs, 1, bs * 2, 8, 1, 1, "fp16", "float16", "float16", "float16", bs, False,
                               True, False))
    # ── 4. token 数:单核 / 多核 / 末核余数(BlockSize 整除 S,cache 恰好写满) ──
    for b, s, bs in [(1, 32, 32), (1, 64, 64), (1, 128, 128), (1, 512, 512), (4, 32, 32), (2, 128, 128),
                     (8, 64, 64), (1, 256, 256)]:
        cases.append(make_case("tok_b%d_s%d_bs%d" % (b, s, bs), b, s, 4, 1, 1, "fp16", "float16", "float16",
                               "float16", bs, False, True, False))
    # ── 4b. 极小 / 非对齐 token 数(BlockSize 无法整除 S,cache 必有余量 -> 只进 aclnn 集) ──
    for b, s in [(1, 1), (1, 7), (1, 63), (1, 65), (1, 96)]:
        cases.append(make_case("tok_small_b%d_s%d" % (b, s), b, s, 4, 1, 1, "fp16", "float16", "float16", "float16",
                               32, False, True, False))
    # ── 5. 量化:对称 / 非对称 / k 量化 v 不量化 / v 量化 k 不量化 ──
    cases.append(make_case("quant_sym_kq_vq", 2, 64, 8, 1, 1, "fp16", "float16", "int8", "int8", 64, False, True,
                           False))
    cases.append(make_case("quant_asym_kq_vq", 2, 64, 8, 1, 1, "fp16", "float16", "int8", "int8", 64, True, True,
                           False))
    cases.append(make_case("quant_asym_kq_only", 2, 64, 8, 1, 1, "bf16", "bfloat16", "int8", "bfloat16", 64, True,
                           True, False))
    cases.append(make_case("quant_asym_vq_only", 2, 64, 8, 1, 1, "fp16", "float16", "float16", "int8", 64, True,
                           True, False))
    cases.append(make_case("quant_asym_kq_bf16", 1, 96, 16, 2, 2, "bf16", "bfloat16", "int8", "bfloat16", 96, True,
                           True, False))
    # ── 6. index 含 -1(部分 token 跳过 cache 更新;cache 留有余量时才可能出现) ──
    cases.append(make_case("idx_neg_pad", 1, 128, 8, 1, 1, "fp16", "float16", "float16", "float16", 64, False, True,
                           True, block_pad=2))
    cases.append(make_case("idx_neg_int8_pad", 1, 128, 8, 1, 1, "fp16", "float16", "int8", "int8", 64, False, True,
                           True, block_pad=2))
    # ── 7. blockNum 有余量(kernel 通路下未写 slot 的内容由 buffer 初值决定,
    #      故只进 aclnn 用例集 —— 那里 in-place cache 就是输入本身,未写 slot 保持初值) ──
    cases.append(make_case("blknum_pad", 1, 128, 8, 1, 1, "fp16", "float16", "float16", "float16", 64, False, True,
                           False, block_pad=3))
    # ── 8. is_output_qkv = False(proto 输出不写,只进 aclnn 用例集) ──
    cases.append(make_case("no_proto_b2_s64", 2, 64, 8, 1, 1, "fp16", "float16", "float16", "float16", 64, False,
                           False, False))
    # ── 9. 可选属性缺省:不带 cache_mode / epsilon,走 def 的默认值(PA_NZ / 1e-6) ──
    cases.append(make_case("defaults_omitted", 2, 64, 8, 1, 1, "fp16", "float16", "float16", "float16", 64, False,
                           True, False, omit_optional=True))
    cases.append(make_case("defaults_omitted_int8", 1, 64, 8, 1, 1, "fp16", "float16", "int8", "int8", 64, False,
                           True, False, omit_optional=True))
    return cases


# 逐 dtype 的 (rtol, ptol)。与项目判据一致:fp16 双千分之一 / bf16 双千分之四 / fp32 双万分之一。
# 必须显式下发 —— TTK 的 close 判据对 bf16 用的是 1e-3(≈ 1/4 个 bf16 ULP = 2^-8),
# 比 1 个 ULP 还紧,会把正常的末位舍入判成失败;这里按 dtype 给出真正的判据。
# int8 输出由 Spec 的 quant 判据(绝对误差 <= 1 LSB)接管,rtol/ptol 不参与。
TOL_BY_DTYPE = {
    "float16": (0.001, 0.001),
    "bfloat16": (0.004, 0.001),
    "float32": (0.0001, 0.001),
    "int8": (1.0, 1.0),
}


def fmt(v):
    if v is None:
        return "None"
    if isinstance(v, bool):
        return "True" if v else "False"
    return repr(v)


def write_kernel_csv(cases, path):
    # output_inplace_indexes:逐输出写明它别名到哪个【输入】下标(None = 独立输出)。
    # 本算子 q_out/k_cache/v_cache 三对输入输出原地复用,必须声明 —— 否则 TTK 会把它们
    # 当独立 buffer 另行分配并与输入脱钩(同族 with_k_scale / rotary_position_embedding
    # 等有原地输出的算子都带这一列)。
    header = ["testcase_name", "op_name", "input_shapes", "output_shapes", "output_inplace_indexes", "attributes",
              "input_data_ranges", "input_dtypes", "output_dtypes", "input_formats", "output_formats",
              "precision_tolerances", "dtype_tag"]
    lines = [",".join(header)]
    for c in cases:
        if not c["kernel_ok"]:
            continue  # 见 make_case docstring:未写 slot 的内容不属于算子输出
        row = [
            c["name"],
            "qkv_rms_norm_rope_cache",
            '"%s"' % fmt(c["in_shapes"]).replace("None", "None"),
            '"%s"' % fmt(c["out_shapes"]),
            '"(6, 7, 8, None, None, None)"',
            '"%s"' % fmt(c["attrs"]),
            '"%s"' % fmt(c["ranges"]),
            '"%s"' % fmt(c["in_dtypes"]),
            '"%s"' % fmt(c["out_dtypes"]),
            '"%s"' % fmt(["ND"] * len(c["in_dtypes"])),
            '"%s"' % fmt(["ND"] * len(c["out_dtypes"])),
            '"%s"' % fmt([TOL_BY_DTYPE[d] for d in c["out_dtypes"]]),
            c["dtype_tag"],
        ]
        lines.append(",".join(row))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_aclnn_csv(cases, path):
    """aclnn 张量序 = aclnnQkvRmsNormRopeCacheGetWorkspaceSize 的张量形参序:
    qkv, qGamma, kGamma, cos, sin, index, qOut, kCache, vCache,
    kScale, vScale, kOffset, vOffset, qOutBeforeQuant, kOutBeforeQuant, vOutBeforeQuant
    输出索引 0..5;原地输出(0/1/2)对应张量位置 6/7/8。"""
    header = ["testcase_name", "api_name", "tensor_view_shapes", "tensor_dtypes", "tensor_formats", "attributes",
              "output_tensor_indexes", "output_inplace_indexes", "input_data_ranges", "precision_tolerances"]
    lines = [",".join(header)]
    for c in cases:
        with_proto = c["attrs"]["is_output_qkv"]
        shapes = list(c["in_shapes"])
        dtypes = list(c["in_dtypes"])
        ranges = list(c["ranges"])
        out_idx = "(6, 7, 8)"
        if with_proto:
            shapes += [c["out_shapes"][3], c["out_shapes"][4], c["out_shapes"][5]]
            dtypes += [c["out_dtypes"][3], c["out_dtypes"][4], c["out_dtypes"][5]]
            ranges += [(-1, 1), (-1, 1), (-1, 1)]
            out_idx = "(6, 7, 8, 13, 14, 15)"
        attrs = {
            "qkvSize": c["attrs"]["qkv_size"],
            "headNums": c["attrs"]["head_nums"],
        }
        # 缺省档:attrs 里没有该键 = 不传,aclnn 侧走 def 的默认值
        if "epsilon" in c["attrs"]:
            attrs["epsilon"] = c["attrs"]["epsilon"]
        if "cache_mode" in c["attrs"]:
            attrs["cacheModeOptional"] = c["attrs"]["cache_mode"]
        row = [
            "QkvRmsNormRopeCache_" + c["name"],
            "aclnnQkvRmsNormRopeCache",
            '"%s"' % fmt(shapes),
            '"%s"' % fmt(dtypes),
            '"%s"' % fmt(["ND"] * len(shapes)),
            '"%s"' % fmt(attrs),
            '"%s"' % out_idx,      # 含逗号,必须加引号,否则 CSV 会被拆成多列
            '"(6, 7, 8)"',
            '"%s"' % fmt(ranges),
            '"%s"' % fmt([TOL_BY_DTYPE[c["out_dtypes"][i]] for i in range(6)]),
        ]
        lines.append(",".join(row))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cases = build_cases()
    write_kernel_csv(cases, os.path.join(OUT_DIR, "ttk_kernel_qkv_rms_norm_rope_cache_st.csv"))
    write_aclnn_csv(cases, os.path.join(OUT_DIR, "ttk_aclnn_qkv_rms_norm_rope_cache_st.csv"))
    print("kernel cases: %d, aclnn cases: %d" % (len(cases), sum(1 for c in cases if c["attrs"]["is_output_qkv"])))


if __name__ == "__main__":
    main()
