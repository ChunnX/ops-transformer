#!/usr/bin/python3
from quant_flash_attn_paramset_common import expand_paramset_to_cases

TEST_PARAMS = {
    "BNSD_B1_QS256_KVS256_Nq1_Nkv1_D128_qm3": {
        "B": [1], "N_q": [1], "N_kv": [1], "D": [128],
        "cu_seqlens_q": [None], "cu_seqlens_kv": [None],
        "seqused_q": [[256]], "seqused_kv": [[256]],
        "enable_pa": [False], "kv_cache_layout": [None], "block_size": [None],
        "mask_mode": [0], "q_scale_layout": ["BNSD"], "p_scale": [1.0], "enable_lse": [False],
    },
}

CASES = expand_paramset_to_cases(TEST_PARAMS)

# ===== 辅助函数 =====
def _sq(s, n):   return [s]*n
def _sq0(n):     return [0]*n
def _sq_zf(s, n): return [0] + [s]*(n-1)            # 首 batch=0
def _sq_zl(s, n): return [s]*(n-1) + [0]            # 末 batch=0
def _sq_alt(s, n): return [s if i%2==0 else 0 for i in range(n)]  # 交替 0
def _sq_step(n):  return [i*64+1 for i in range(n)] # 递增不定长
def _sq_mix(s, n): return [s]*n if n<=1 else [1]+[s]*(n-2)+[1]  # 两头窄中间宽

def _c(name, b, nq, sq_q, sq_kv):
    return {"name": name, "B": b, "N_q": nq, "N_kv": 1, "D": 128,
            "cu_seqlens_q": None, "cu_seqlens_kv": None,
            "seqused_q": sq_q, "seqused_kv": sq_kv,
            "enable_pa": False, "kv_cache_layout": None, "block_size": None,
            "mask_mode": 0, "q_scale_layout": "BNSD", "p_scale": 1.0, "enable_lse": False}

# ===== 边值集 =====
_S1 = [1, 2, 31, 32, 33, 63, 64, 65, 95, 96, 97,
       127, 128, 129, 159, 160, 161, 191, 192, 193,
       223, 224, 225, 255, 256, 257, 319, 320, 383,
       384, 385, 447, 448, 511, 512]
_S2 = [1, 2, 31, 32, 33, 63, 64, 65, 95, 96, 97,
       127, 128, 129, 159, 160, 191, 192, 193, 223,
       224, 255, 256, 257, 319, 320, 383, 384, 385,
       447, 448, 511, 512]
_S2_L = [256, 512, 767, 768, 769, 1023, 1024, 1025, 2047, 2048, 2049, 4095, 4096]
_B  = [1, 2, 3, 4, 8, 16, 32]
_G  = [1, 2, 3, 4, 8, 16, 32]

# ===== 1. S1 独立扫描 =====
for s1 in _S1:
    CASES.append(_c(f"S1_{s1}_B1_G1",    1, 1, [s1], [256]))

# ===== 2. S2 独立扫描 =====
for s2 in _S2 + _S2_L:
    CASES.append(_c(f"S2_{s2}_B1_G1",    1, 1, [256], [s2]))

# ===== 3. G 独立扫描 =====
for g in _G:
    CASES.append(_c(f"G_{g}_B1",         1, g, [256], [256]))

# ===== 4. B 独立扫描(全等长) =====
for b in _B:
    CASES.append(_c(f"B_{b}_uniform",    b, 1, _sq(256,b), _sq(256,b)))

# ===== 5. seq_used 零长模式 =====
for b in _B:
    if b < 2: continue
    CASES.append(_c(f"B{b}_Q0first",     b, 1, _sq_zf(256,b), _sq(256,b)))
    CASES.append(_c(f"B{b}_K0first",     b, 1, _sq(256,b), _sq_zf(256,b)))
    CASES.append(_c(f"B{b}_Q0last",      b, 1, _sq_zl(256,b), _sq(256,b)))
    CASES.append(_c(f"B{b}_K0last",      b, 1, _sq(256,b), _sq_zl(256,b)))
    CASES.append(_c(f"B{b}_Q0alt",       b, 1, _sq_alt(256,b), _sq(256,b)))
    CASES.append(_c(f"B{b}_K0alt",       b, 1, _sq(256,b), _sq_alt(256,b)))
    CASES.append(_c(f"B{b}_Qstep",       b, 1, _sq_step(b), _sq(256,b)))
    CASES.append(_c(f"B{b}_Kstep",       b, 1, _sq(256,b), _sq_step(b)))
    CASES.append(_c(f"B{b}_Qmix",        b, 1, _sq_mix(256,b), _sq(256,b)))
    CASES.append(_c(f"B{b}_Kmix",        b, 1, _sq(256,b), _sq_mix(256,b)))

# ===== 6. S1×S2 关键边值交叉（仅边值子集，含零长）=====
_S1_X = [1, 32, 64, 128, 192, 256, 320, 384, 512]
_S2_X = [1, 64, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096]
for s1 in _S1_X:
    for s2 in _S2_X:
        CASES.append(_c(f"S1{s1}_S2{s2}_X",  1, 1, [s1], [s2]))

# ===== 7. S1×G 交叉 =====
for s1 in [1, 32, 64, 128, 192, 256, 384, 512]:
    for g in [1, 2, 4, 8, 32]:
        CASES.append(_c(f"S1{s1}_G{g}_X",    1, g, [s1], [256]))

# ===== 8. S2×G 交叉 =====
for s2 in [1, 64, 128, 192, 256, 384, 512, 2048, 4096]:
    for g in [1, 2, 4, 8, 32]:
        CASES.append(_c(f"S2{s2}_G{g}_X",    1, g, [256], [s2]))

# ===== 9. B×G 交叉 =====
for b in _B:
    for g in _G:
        CASES.append(_c(f"B{b}_G{g}_X",      b, g, _sq(256,b), _sq(256,b)))

# ===== 10. B×S1 交叉(含零长) =====
for b in _B:
    for s1 in [1, 64, 128, 256, 384, 512]:
        CASES.append(_c(f"B{b}_S1{s1}_X",     b, 1, _sq(s1,b), _sq(256,b)))
        if b >= 2:
            CASES.append(_c(f"B{b}_S1{s1}_0f_X",  b, 1, _sq_zf(s1,b), _sq(256,b)))

# ===== 11. B×S2 交叉(含零长) =====
for b in _B:
    for s2 in [1, 64, 128, 256, 384, 512, 2048]:
        CASES.append(_c(f"B{b}_S2{s2}_X",     b, 1, _sq(256,b), _sq(s2,b)))
        if b >= 2:
            CASES.append(_c(f"B{b}_S2{s2}_0f_X",  b, 1, _sq(256,b), _sq_zf(s2,b)))

print(f"Total debug+brute cases: {len(CASES)}")
