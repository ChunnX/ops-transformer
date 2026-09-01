"""适配层 Host 逻辑自检（CPU fallback，仅流程验证）。"""
import sys, warnings
sys.path.insert(0, "F:/sageAttention2算子/experimental/attention/sageattention2/python")
import torch
from sageattention2 import sageattn, sageattn_qk_int8_pv_fp8_asc

ok = []
def check(name, cond, extra=""):
    ok.append(bool(cond))
    print(("PASS" if cond else "FAIL"), name, extra)

def mk(b, hq, hkv, sq, sk, d, layout="HND", dtype=torch.float16, causal=False):
    g = torch.Generator().manual_seed(0)
    if layout == "HND":
        q = torch.randn(b, hq, sq, d, generator=g).to(dtype)
        k = torch.randn(b, hkv, sk, d, generator=g).to(dtype)
        v = torch.randn(b, hkv, sk, d, generator=g).to(dtype)
    else:
        q = torch.randn(b, sq, hq, d, generator=g).to(dtype)
        k = torch.randn(b, sk, hkv, d, generator=g).to(dtype)
        v = torch.randn(b, sk, hkv, d, generator=g).to(dtype)
    return q, k, v

# 1. HND 基本
q, k, v = mk(1, 8, 8, 128, 128, 64)
o = sageattn_qk_int8_pv_fp8_asc(q, k, v)
check("HND basic", o.shape == (1, 8, 128, 64) and o.dtype == torch.float16, f"{o.shape}")

# 2. return_lse
o, lse = sageattn_qk_int8_pv_fp8_asc(q, k, v, return_lse=True)
check("return_lse", o.shape == (1, 8, 128, 64) and lse.shape == (1, 8, 128) and torch.isfinite(lse).all(), f"lse={lse.shape}")

# 3. head_dim pad
q48, k48, v48 = mk(1, 8, 8, 128, 128, 48)
o = sageattn_qk_int8_pv_fp8_asc(q48, k48, v48)
check("pad 48->64", o.shape == (1, 8, 128, 48), f"{o.shape}")
q96, k96, v96 = mk(1, 8, 8, 128, 128, 96)
o = sageattn_qk_int8_pv_fp8_asc(q96, k96, v96)
check("pad 96->128", o.shape == (1, 8, 128, 96), f"{o.shape}")

# 4. NHD + per_warp + fp32
qn, kn, vn = mk(1, 8, 8, 128, 128, 64, layout="NHD")
o = sageattn_qk_int8_pv_fp8_asc(qn, kn, vn, tensor_layout="NHD", qk_quant_gran="per_warp", pv_accum_dtype="fp32")
check("NHD+per_warp+fp32", o.shape == (1, 128, 8, 64), f"{o.shape}")

# 5. causal
qc, kc, vc = mk(1, 8, 8, 128, 128, 64, causal=True)
o = sageattn_qk_int8_pv_fp8_asc(qc, kc, vc, is_causal=True)
check("causal", o.shape == (1, 8, 128, 64))

# 6. GQA Sq != Sk + smooth_v warning
qg, kg, vg = mk(1, 8, 4, 256, 512, 64)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    o = sageattn_qk_int8_pv_fp8_asc(qg, kg, vg, smooth_v=True)
check("GQA+sq!=sk", o.shape == (1, 8, 256, 64) and any("smooth_v" in str(x.message) for x in w), f"warn={len(w)}")

# 7. sageattn 分发
o = sageattn(q, k, v)
check("sageattn dispatch", o.shape == (1, 8, 128, 64))

# 8. 负向校验
def expect_raise(name, fn, exc=Exception):
    try:
        fn()
        check(name, False, "no raise")
    except exc as e:
        check(name, True, f"{type(e).__name__}: {str(e)[:40]}")
    except Exception as e:
        check(name, False, f"wrong exc {type(e).__name__}")

q256, k256, v256 = mk(1, 8, 8, 128, 128, 256)
expect_raise("head_dim>128", lambda: sageattn_qk_int8_pv_fp8_asc(q256, k256, v256))
expect_raise("attn_mask", lambda: sageattn_qk_int8_pv_fp8_asc(q, k, v, attn_mask=torch.zeros(1)))
expect_raise("bad gran", lambda: sageattn_qk_int8_pv_fp8_asc(q, k, v, qk_quant_gran="nope"))
expect_raise("bad accum", lambda: sageattn_qk_int8_pv_fp8_asc(q, k, v, pv_accum_dtype="fp16"))

print(f"\n{'ALL PASS' if all(ok) else 'SOME FAILED'}: {sum(ok)}/{len(ok)}")
sys.exit(0 if all(ok) else 1)
