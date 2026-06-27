# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Regression repro for KvRmsNormRopeCache regbase quant-rope GM offset overflow.
#
# Root cause: in op_kernel/arch35/kv_rms_norm_rope_cache_regbase_recompute.h the
# quantized rope paths declared the GM element offsets (cosSinOffset / kScaleOffset /
# kOffset / kOutOffset) as uint32_t. When rowIdx*dk (or dk) exceeds 2^32 the offset
# wraps, producing wrong addresses -> AI Core "VEC access UB out of bounds" (err 341).
#
# Requires: Ascend NPU with >= ~60 GiB free HBM, torch_npu + op-plugin.
# Before fix: crashes (507035 / aivec 341). After fix (offsets int64): slot 2 == 2.0.
import torch, torch_npu, traceback
dev = 'npu:0'; torch.npu.set_device(0)
B, N, S = 3, 1, 1
DK = 2147483648            # 2^31 ; row2 cosSinOffset1 = 2*dk = 2^32 -> uint32 wrap
DV, KVD = 1, 2147483649
ROWVAL = [0.5, 1.0, 2.0]   # per-row cos constant (fp8-exact) to make the wrap observable
try:
    kv    = torch.ones (B, N, S, KVD, dtype=torch.bfloat16, device=dev)
    gamma = torch.ones (DV,          dtype=torch.bfloat16, device=dev)
    cos   = torch.empty(B, N, S, DK,  dtype=torch.bfloat16, device=dev)
    for b in range(B): cos[b].fill_(ROWVAL[b])
    sin   = torch.zeros(B, N, S, DK,  dtype=torch.bfloat16, device=dev)
    index = torch.zeros(B * S,        dtype=torch.int64,    device=dev)
    k_cache   = torch.zeros(B, N, S, DK, dtype=torch.float8_e4m3fn, device=dev)
    ckv_cache = torch.zeros(B, N, S, DV, dtype=torch.float8_e4m3fn, device=dev)
    ks = torch.ones(1, dtype=torch.float32, device=dev)
    cs = torch.ones(1, dtype=torch.float32, device=dev)
    torch_npu.npu_kv_rmsnorm_rope_cache_v2(
        kv, gamma, cos, sin, index, k_cache, ckv_cache,
        k_rope_scale=ks, c_kv_scale=cs, v=None,
        epsilon=1e-5, cache_mode='Norm', is_output_kv=False)
    torch.npu.synchronize()
    ok = True
    for r in range(B):
        v = k_cache[r, 0, 0, 0].float().item()
        good = abs(v - ROWVAL[r]) < 0.2
        ok &= good
        print(f"k_cache slot{r} = {v:.3f} (expect {ROWVAL[r]}) {'OK' if good else 'WRONG(wrap)'}")
    print("RESULT:", "PASS (no uint32 wrap)" if ok else "FAIL (uint32 offset wrap)")
except Exception:
    print("RESULT: CRASH (uint32 offset wrap -> VEC OOB)"); traceback.print_exc()
