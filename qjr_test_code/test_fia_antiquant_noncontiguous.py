#!/usr/bin/env python3
"""
FIA Antiquant PA 场景 0/1 轴非连续冒烟测试

不依赖 pytest,纯 __main__ 运行:
  python test_fia_antiquant_noncontiguous.py --case SMOKE_ALL
  python test_fia_antiquant_noncontiguous.py --case BBH-DIM0

测试策略:
  1. 构造连续 KV(int8) + scale(fp32),跑一次得到 reference 输出(npu golden)
  2. 构造非连续 KV + scale(同数据,不同 stride),跑一次得到 actual 输出
  3. 对比 reference vs actual,连续与非连续应产出相同结果(因为数据相同,只是内存排布不同)
  4. 对 checker 拦截的 case(如 BBH dim1、非 PA 非连续),期望抛异常

反量化溢出保护:
  - KV 为 int8, 范围 [-128, 127]
  - 反量化公式: dequant = kv_int8 * scale + offset
  - scale 范围 [0.001, 0.01], 最大反量化值 = 127 * 0.01 = 1.27, 远小于 fp16 上限 65504
  - offset 范围 [-0.1, 0.1], 不影响溢出
  - query 范围 [-1, 1], 保证 QK^T 经 softmax 后不溢出
"""
import argparse, json, logging, math, os, random, sys, traceback
import torch, torch_npu

logging.basicConfig(level=logging.INFO, format='%(message)s', force=True)
logger = logging.getLogger(__name__)

DEVICE_ID = 0
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# dtype 与数值范围常量
# ============================================================================

KV_DTYPE_MAP = {
    "int8":          torch.int8,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "hifloat8":      torch.float8_e5m2,
    "int4":          torch.int32,
    "fp4_e2m1":      torch.float4_e2m1fn_x2,
}

SCALE_DTYPE_MAP = {
    0: None,
    1: torch.float32,
    2: None,
    3: torch.float32,
    4: torch.float32,
    5: torch.float32,
    6: torch.float8_e8m0fnu,
}

DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

KV_DATA_RANGE_INT8    = (-5, 5)
KV_DATA_RANGE_INT4    = (-3, 3)
KV_DATA_RANGE_FP8     = (-2.0, 2.0)
KV_DATA_RANGE_HIF8    = (-2.0, 2.0)
KV_DATA_RANGE_FP4     = (0, 15)
SCALE_DATA_RANGE = (0.001, 0.01)
OFFSET_DATA_RANGE = (-0.1, 0.1)
Q_DATA_RANGE = (-1.0, 1.0)


def _get_kv_dtype(kv_dtype_str):
    return KV_DTYPE_MAP.get(kv_dtype_str, torch.int8)


def _get_kv_data_range(kv_dtype_str):
    if kv_dtype_str == "int8":   return KV_DATA_RANGE_INT8
    if kv_dtype_str == "int4":   return KV_DATA_RANGE_INT4
    if kv_dtype_str in ("float8_e4m3fn", "hifloat8"): return KV_DATA_RANGE_FP8
    if kv_dtype_str == "fp4_e2m1": return KV_DATA_RANGE_FP4
    return KV_DATA_RANGE_INT8


def _get_scale_dtype(antiq_mode_num, query_dtype):
    sd = SCALE_DTYPE_MAP.get(antiq_mode_num, torch.float32)
    return sd if sd is not None else query_dtype

# ============================================================================
# JSON 用例加载
# ============================================================================

def load_cases():
    cases_path = os.path.join(SCRIPT_DIR, "cases.json")
    with open(cases_path, "r") as f:
        cfg = json.load(f)
    defaults = cfg["global_defaults"]
    normals = cfg["normal_cases"]
    errors = cfg["error_cases"]
    all_cases = {}
    for c in normals:
        merged = {**defaults, **c}
        merged["expect_pass"] = True
        all_cases[merged["id"]] = merged
    for c in errors:
        merged = {**defaults, **c}
        merged["expect_pass"] = False
        all_cases[merged["id"]] = merged
    return all_cases

# ============================================================================
# 辅助函数
# ============================================================================

def _block_table(batch, seqs_kv, bs):
    vn = [math.ceil(s / bs) + 1 for s in seqs_kv]
    total, mx = sum(vn), max(vn)
    ids = list(range(total)); random.shuffle(ids)
    tbl = -torch.ones(batch, mx, dtype=torch.int32)
    off = 0
    for b in range(batch):
        tbl[b, :vn[b]] = torch.tensor(ids[off:off+vn[b]], dtype=torch.int32); off += vn[b]
    return tbl, total


def _kv_shape(fmt, total_blocks, bs, d, n_kv, b, kv_seq, q_layout, act_kv):
    if fmt == "BnNBsD":  return (total_blocks, n_kv, bs, d)
    if fmt == "BnBsH":   return (total_blocks, bs, n_kv * d)
    if fmt == "NZ":      return (total_blocks, n_kv, d // 16, bs, 16)
    if fmt == "none":
        if q_layout == "BSND": return (b, kv_seq, n_kv, d)
        if q_layout == "TND":  return (sum(act_kv), n_kv, d)
        return (b, n_kv, kv_seq, d)
    raise ValueError(fmt)


def _make_nc(tensor, nc):
    if nc == "none": return tensor
    if nc == "dim01":
        tensor = _make_nc(tensor, "dim0")
        return _make_nc(tensor, "dim1")
    pad_dim = 1 if nc == "dim0" else 2
    Bv = tensor.shape[pad_dim]
    shp = list(tensor.shape); shp[pad_dim] = Bv + 1
    p = torch.zeros(shp, dtype=tensor.dtype, device=tensor.device)
    ws = [slice(None)] * len(tensor.shape); ws[pad_dim] = slice(None, Bv)
    p[tuple(ws)] = tensor
    return p[tuple(ws)]


def _apply_nc(tensor, axes):
    """把逻辑轴列表翻译成 _make_nc 可识别的 nc 串后调用(_make_nc 本身不改动)。
    语义: 声明非连续的逻辑轴 ax -> 使 ax 轴 stride 变大 -> _make_nc 在 ax+1 维 pad。
    dim0 -> 轴0(Bn) 非连续; dim1 -> 轴1(N2) 非连续; [0,1] -> dim01。
    可 pad 维不存在时(如 2D scale [totalBlocks,bs] 的 ax=1)该轴自动跳过。
    """
    axes = [a for a in (axes or []) if a <= 1]
    has0 = 0 in axes
    has1 = 1 in axes
    nc = "none"
    if has0 and len(tensor.shape) > 1:          # dim0: pad 轴1
        nc = "dim0"
    if has1 and len(tensor.shape) > 2:          # dim1: pad 轴2
        nc = "dim01" if nc == "dim0" else "dim1"
    return _make_nc(tensor, nc)


def _cstrides(shape):
    s = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1): s[i] = s[i+1] * shape[i+1]
    return tuple(s)


def _accum(seqs):
    out, acc = [], 0
    for s in seqs:
        acc += s
        out.append(acc)
    return out


def _rand_kv(shape, kv_dtype_str, seed):
    """根据 kv dtype 生成安全范围的 KV tensor"""
    torch.manual_seed(seed)
    lo, hi = _get_kv_data_range(kv_dtype_str)
    if kv_dtype_str == "int8":
        t = torch.randint(lo, hi + 1, shape, dtype=torch.int32)
        return t.to(_get_kv_dtype(kv_dtype_str))
    elif kv_dtype_str == "int4":
        # INT4 存储为 INT32: 8 个有符号 4-bit nibble 打包成 1 个 int32。
        # shape 末维已是 d//8 (存储D), 需在此维度展开 8 个 nibble 独立取值再打包。
        # 直接对 int32 randint 会让高 7 个 nibble 退化为符号位(0 或 -1), 掩盖精度问题。
        shp = list(shape)
        nibble_shape = shp[:-1] + [shp[-1] * 8]  # 还原逻辑 D 维
        vals = torch.randint(lo, hi + 1, nibble_shape, dtype=torch.int32)  # 每个逻辑 int4 值 [-3,3]
        vals = vals.reshape(shp[:-1] + [shp[-1], 8])
        # 有符号 nibble 打包: 低 nibble 在低位, 高 nibble 在高位 (与算子解析一致)
        packed = torch.zeros(shp, dtype=torch.int32)
        for i in range(8):
            nibble = vals[..., i] & 0xF  # 取低 4 位 (负数转无符号 nibble: -3 -> 0xD)
            packed = packed | (nibble << (i * 4))
        return packed.to(_get_kv_dtype(kv_dtype_str))
    elif kv_dtype_str in ("float8_e4m3fn", "hifloat8"):
        t = torch.empty(shape, dtype=torch.float32)
        t.uniform_(lo, hi)
        return t.to(_get_kv_dtype(kv_dtype_str))
    elif kv_dtype_str == "fp4_e2m1":
        t = torch.randint(lo, hi + 1, shape, dtype=torch.uint8)
        shp = list(t.shape)
        t = t.reshape(shp[:-1] + [shp[-1] // 2, 2])
        packed = (t[..., 1] << 4) | t[..., 0]
        return packed.view(torch.float4_e2m1fn_x2)
    else:
        t = torch.randint(lo, hi + 1, shape, dtype=torch.int32)
        return t.to(_get_kv_dtype(kv_dtype_str))


def _rand_float(shape, data_range, dtype, seed):
    """生成安全范围的 float tensor"""
    torch.manual_seed(seed)
    lo, hi = data_range
    t = torch.empty(shape, dtype=dtype)
    t.uniform_(lo, hi)
    return t


def _make_scale(total_blocks, n_kv, bs, antiq_mode_num, scale_dtype, seed, pa=True, d=128, max_block_num_per_batch=0):
    """根据 antiquant mode 和 scale dtype 构造 scale tensor
    PA 场景:
      per_channel(0):       [max_block_num_per_batch * bs] 或 [1]  (与 query 同 dtype)
      per_token(1):         [B, max_block_num_per_batch * bs]       (FLOAT32)
      per_token_head(3):    [B, n_kv, max_block_num_per_batch * bs] (FLOAT32)
      per_token_pa(4):      [total_blocks, bs]                      (FLOAT32)
      per_token_head_pa(5): [total_blocks, n_kv, bs]               (FLOAT32)
      per_token_group(6):   [total_blocks, n_kv, bs//32, d//32, 32] (e8m0)
    非 PA 场景:
      per_channel(0):       [n_kv, d]   (与 query 同 dtype)
      per_token(1):         [B, kv_seq] (FLOAT32)
      per_token_head(3):    [B, n_kv, kv_seq] (FLOAT32)
    """
    if pa:
        if antiq_mode_num == 5:
            return _rand_float((total_blocks, n_kv, bs), SCALE_DATA_RANGE, scale_dtype, seed)
        elif antiq_mode_num == 4:
            return _rand_float((total_blocks, bs), SCALE_DATA_RANGE, scale_dtype, seed)
        elif antiq_mode_num == 3:
            seq_len = max_block_num_per_batch * bs
            return _rand_float((1, n_kv, seq_len) if n_kv > 1 else (1, seq_len), SCALE_DATA_RANGE, scale_dtype, seed)
        elif antiq_mode_num == 1:
            seq_len = max_block_num_per_batch * bs
            return _rand_float((1, seq_len), SCALE_DATA_RANGE, scale_dtype, seed)
        elif antiq_mode_num == 0:
            return _rand_float((n_kv, d), SCALE_DATA_RANGE, scale_dtype, seed)
        elif antiq_mode_num == 2:
            return _rand_float((n_kv,), SCALE_DATA_RANGE, scale_dtype, seed)  # 1D [N], per-tensor-head scale
        elif antiq_mode_num == 6:
            grp_size = 32
            seq_len = max_block_num_per_batch * bs
            b_size = total_blocks // max_block_num_per_batch if max_block_num_per_batch > 0 else 1
            raw = torch.randint(120, 140, (1, b_size, n_kv, seq_len, d // grp_size),
                                dtype=torch.uint8)
            return raw.view(torch.float8_e8m0fnu).reshape(1, b_size, n_kv, seq_len, d // grp_size)
    else:
        if antiq_mode_num == 0:
            return _rand_float((n_kv, d), SCALE_DATA_RANGE, scale_dtype, seed)
        elif antiq_mode_num == 1:
            return _rand_float((1, bs), SCALE_DATA_RANGE, scale_dtype, seed)
        elif antiq_mode_num == 3:
            return _rand_float((1, n_kv, bs), SCALE_DATA_RANGE, scale_dtype, seed)
        elif antiq_mode_num == 6:
            grp_size = 32
            raw = torch.randint(120, 140, (1, n_kv, bs // grp_size, d // grp_size, grp_size),
                                dtype=torch.uint8)
            return raw.view(torch.float8_e8m0fnu).reshape(1, n_kv, bs // grp_size, d // grp_size, grp_size)
    raise ValueError(f"unknown antiquant mode num: {antiq_mode_num} pa={pa}")


# ============================================================================
# NPU 调用
# ============================================================================

def _run_npu(q_npu, k_npu, v_npu, k_scale_npu, v_scale_npu,
             pa, q_layout, act_q, act_kv, bt_npu, bs, scale_val,
             q_n, kv_n, d, antiq_mode_num):
    torch_npu.npu.set_device(DEVICE_ID); torch.npu.synchronize()
    if q_layout == "TND":
        aq = torch.tensor(_accum(act_q), dtype=torch.int64)
        akv = torch.tensor(act_kv if pa else _accum(act_kv), dtype=torch.int64)
    else:
        aq = torch.tensor(act_q, dtype=torch.int64)
        akv = torch.tensor(act_kv, dtype=torch.int64)

    r, _ = torch_npu.npu_fused_infer_attention_score(
        q_npu, k_npu, v_npu,
        actual_seq_lengths=aq, actual_seq_lengths_kv=akv,
        key_antiquant_scale=k_scale_npu,
        value_antiquant_scale=v_scale_npu,
        num_heads=q_n, num_key_value_heads=kv_n,
        input_layout=q_layout, scale=scale_val,
        pre_tokens=65535, next_tokens=65535,
        block_table=bt_npu if pa else None,
        block_size=bs if pa else 0,
        key_antiquant_mode=antiq_mode_num,
        value_antiquant_mode=antiq_mode_num,
        sparse_mode=0)
    torch.npu.synchronize()
    return r


def _check(ref, out, tag, atol=5e-3, rtol=5e-3, max_fail_ratio=5e-3):
    """逐位双千分之五精度比对。
    逐元素通过条件: |actual - golden| <= atol + rtol * |golden|, 其中 atol=rtol=5e-3(双千分之五)。
    整体通过条件: 超差元素个数占比(差异>容差的数据个数) < 5e-3(千分之5)。
    """
    a = out.cpu().float()
    b = ref.cpu().float()
    d = (a - b).abs()
    tol = atol + rtol * b.abs()
    bad = (d > tol) | a.isnan() | b.isnan()
    bad_ratio = bad.sum().item() / max(d.numel(), 1)
    mx = d.max().item()
    status = "PASS" if bad_ratio < max_fail_ratio else "FAIL"
    logger.info(f"  [{tag}] {status} max_err={mx:.6f} bad_ratio={bad_ratio:.5f} "
                f"(must<{max_fail_ratio:.1e}) atol={atol} rtol={rtol}")
    return status == "PASS"


# ============================================================================
# 单 case 执行
# ============================================================================

def _run_one(cfg):
    cid = cfg["id"]
    expect_pass = cfg["expect_pass"]
    b = cfg["B"]; n_q = cfg["N_q"]; n_kv = cfg["N_kv"]
    pa = cfg.get("pa", True)
    fmt = cfg["fmt"]; ql = cfg["ql"]
    k_axes = cfg.get("k_uncontiguous", [])
    v_axes = cfg.get("v_uncontiguous", [])
    k_scale_axes = cfg.get("k_antiquantScale_uncontiguous", [])
    v_scale_axes = cfg.get("v_antiquantScale_uncontiguous", [])
    kv_seq = cfg["kv_seq"]; bs = cfg["bs"]; q_seq = cfg["q_seq"]; d = cfg["d"]
    act_q = cfg.get("act_q", [q_seq] * b)
    act_kv = cfg.get("act_kv", [kv_seq] * b)
    # q_seq 在 [1,16] 之间随机泛化（若 case 未指定固定 q_seq）
    if "q_seq" not in cfg:
        q_seq = random.randint(1, 16)
        act_q = [q_seq] * b
    antiq_mode = cfg.get("antiquant_mode_num", 5)
    out_dtype_str = cfg.get("out_dtype", "fp16")
    kv_dtype_str = cfg.get("kv_dtype", "int8")
    q_dtype = DTYPE_MAP[out_dtype_str]
    kv_dtype = _get_kv_dtype(kv_dtype_str)
    scale_dtype = _get_scale_dtype(antiq_mode, q_dtype)
    scale_val = 1.0 / math.sqrt(d)

    bt, total_b = (_block_table(b, act_kv, bs) if pa else (None, b))
    bt_npu = bt.npu() if bt is not None else None
    logger.info(f"  kv_dtype={kv_dtype_str}, antiq_mode={antiq_mode}, scale_dtype={scale_dtype}, "
                f"out_dtype={out_dtype_str}, B={b}, totalBlocks={total_b}, N_kv={n_kv}, d={d}")

    # 构造 query (fp16/bf16, 安全范围)
    q_bnsd = _rand_float((b, n_q, q_seq, d), Q_DATA_RANGE, q_dtype, seed=42)
    if ql == "BSND": q_bnsd = q_bnsd.permute(0, 2, 1, 3).contiguous()
    elif ql == "TND":
        t = sum(act_q); q_tnd = torch.zeros(t, n_q, d, dtype=q_dtype); off = 0
        for bi, s in enumerate(act_q):
            q_tnd[off:off+s] = q_bnsd[bi, :, :s, :].transpose(0, 1); off += s
        q_bnsd = q_tnd
    q_npu = q_bnsd.npu()

    # 构造 KV (按 kv_dtype 生成安全范围)
    # INT4 存储为 INT32: kv tensor 的 D 维 = 逻辑D / 8 (8个int4打包成1个int32)
    kv_d = d
    if kv_dtype_str == "int4":
        kv_d = d // 8
    kv_shape = _kv_shape(fmt, total_b, bs, kv_d, n_kv, b, kv_seq, ql, act_kv) if pa \
        else _kv_shape("none", total_b, kv_seq, kv_d, n_kv, b, kv_seq, ql, act_kv)
    k_ref = _rand_kv(kv_shape, kv_dtype_str, seed=100)
    v_ref = _rand_kv(kv_shape, kv_dtype_str, seed=200)

    # 构造 scale (按 antiq_mode 和 scale_dtype)
    max_block_num_per_batch = bt.shape[1] if bt is not None else 0
    scale_bs = bs if pa else kv_seq
    k_scale_ref = _make_scale(total_b, n_kv, scale_bs, antiq_mode, scale_dtype, seed=300,
                              pa=pa, d=d, max_block_num_per_batch=max_block_num_per_batch)
    v_scale_ref = _make_scale(total_b, n_kv, scale_bs, antiq_mode, scale_dtype, seed=400,
                              pa=pa, d=d, max_block_num_per_batch=max_block_num_per_batch)

    # 1) 连续 reference (npu golden)
    try:
        ref_out = _run_npu(q_npu, k_ref.npu(), v_ref.npu(),
                           k_scale_ref.npu(), v_scale_ref.npu(),
                           pa, ql, act_q, act_kv, bt_npu, bs, scale_val,
                           n_q, n_kv, d, antiq_mode)
    except Exception as e:
        if cfg.get("skip_if_error", False):
            logger.info(f"  [{cid}] SKIP (CANN包不支持, skip_if_error=true): {str(e)[:500]}")
            return True
        logger.info(f"  [{cid}] REF ERROR: {e}")
        traceback.print_exc()
        return False

    is_nc = any([k_axes, v_axes, k_scale_axes, v_scale_axes])
    if not is_nc:
        # 连续 case: reference 即结果, 无需对比
        logger.info(f"  [{cid}] PASS (continuous baseline)")
        return True

    # 2) 非连续 actual (同数据, 改 stride) —— _make_nc 不改, 调用侧逐 tensor 翻译轴列表
    k_nc = _apply_nc(k_ref.npu(), k_axes)
    v_nc = _apply_nc(v_ref.npu(), v_axes)
    k_scale_nc = _apply_nc(k_scale_ref.npu(), k_scale_axes)
    v_scale_nc = _apply_nc(v_scale_ref.npu(), v_scale_axes)

    # 验证非连续性: 对每个声明非连续的逻辑轴 ax, 校验其 stride 相对连续值变大
    # (_make_nc: dim0 -> pad轴1 -> stride(0) 变; dim1 -> pad轴2 -> stride(1) 变)
    def _verify_nc(tname, tensor, axes):
        # axes=[] 表示该 tensor 不要求非连续(部分非连续场景: KV非连续+scale连续 等),
        # 直接视为校验通过, 不拦截整个 case。
        if not axes:
            return True
        cs = _cstrides(tensor.shape)
        ok = 0
        for ax in sorted(axes):
            if ax + 1 >= len(tensor.shape):   # 该轴无可 pad 维, 跳过
                continue
            if tensor.stride(ax) == cs[ax]:
                logger.info(f"  [{cid}] WARN: {tname} axis {ax} stride not changed, skipping")
                return False
            ok += 1
        if ok == 0:
            logger.info(f"  [{cid}] WARN: {tname} no pad-able axis from {axes}, skipping")
        return ok > 0

    if not (_verify_nc("k", k_nc, k_axes) and _verify_nc("v", v_nc, v_axes) and
            _verify_nc("k_scale", k_scale_nc, k_scale_axes) and
            _verify_nc("v_scale", v_scale_nc, v_scale_axes)):
        return False
    logger.info(f"  k contig={k_nc.is_contiguous()} stride={k_nc.stride()}")
    logger.info(f"  k_scale contig={k_scale_nc.is_contiguous()} stride={k_scale_nc.stride()}")

    if not expect_pass:
        # 期望 checker 拦截
        try:
            _run_npu(q_npu, k_nc, v_nc, k_scale_nc, v_scale_nc,
                     pa, ql, act_q, act_kv, bt_npu, bs, scale_val,
                     n_q, n_kv, d, antiq_mode)
            logger.info(f"  [{cid}] FAIL: expected error but succeeded")
            return False
        except Exception as e:
            logger.info(f"  [{cid}] PASS (expected error): {str(e)[:100]}")
            return True

    # 3) 对比
    try:
        out = _run_npu(q_npu, k_nc, v_nc, k_scale_nc, v_scale_nc,
                       pa, ql, act_q, act_kv, bt_npu, bs, scale_val,
                       n_q, n_kv, d, antiq_mode)
    except Exception as e:
        logger.info(f"  [{cid}] FAIL: runtime error: {e}")
        traceback.print_exc()
        return False

    return _check(ref_out, out, cid)


# ============================================================================
# main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="SMOKE_ALL", help="case id or SMOKE_ALL")
    parser.add_argument("--contiguous-only", action="store_true",
                        help="只跑连续模式(强制 nc=none), 不跑非连续对比")
    args = parser.parse_args()

    all_cases = load_cases()

    if args.case == "SMOKE_ALL":
        cases_to_run = list(all_cases.items())
    else:
        case_ids = [c.strip() for c in args.case.split(",")]
        cases_to_run = []
        for cid in case_ids:
            if cid not in all_cases:
                logger.info(f"unknown case: {cid}")
                logger.info(f"available: {list(all_cases.keys())}")
                sys.exit(1)
            cases_to_run.append((cid, all_cases[cid]))

    if args.contiguous_only:
        cases_to_run = [(cid,
                         {**cfg,
                          "k_uncontiguous": [], "v_uncontiguous": [],
                          "k_antiquantScale_uncontiguous": [], "v_antiquantScale_uncontiguous": [],
                          "expect_pass": True})
                        for cid, cfg in cases_to_run if cfg.get("expect_pass", True)]

    logger.info(f"\n{'='*60}")
    logger.info(f"FIA Antiquant PA 非连续冒烟测试 ({len(cases_to_run)} cases)")
    logger.info(f"  KV dtype: INT8 / FLOAT8_E4M3FN / HIFLOAT8 / INT4 / FP4_E2M1 泛化")
    logger.info(f"  Output dtype: fp16 + bf16 泛化")
    logger.info(f"  Scale dtype: 随 mode 自动选择 (fp32 / query dtype / e8m0)")
    logger.info(f"{'='*60}\n")

    passed, failed = 0, 0
    for cid, cfg in cases_to_run:
        logger.info(f"\n[RUN] {cid}: {cfg.get('desc', '')}")
        try:
            ok = _run_one(cfg)
            if ok:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.info(f"  [{cid}] EXCEPTION: {e}")
            traceback.print_exc()
            failed += 1

    logger.info(f"\n{'='*60}")
    logger.info(f"Summary: {passed} passed, {failed} failed, {len(cases_to_run)} total")
    logger.info(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
