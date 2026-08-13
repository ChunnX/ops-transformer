#!/usr/bin/env python3
"""Stall/死锁测试 for dropmask (§9.4)

7 个场景验证 lseOutBuffers 复用不导致 stall。
每个场景设 10s 超时，NPU 调用超时即判定 stall。
"""
import sys
import time
import signal
import torch
import torch_npu
import cann_ops_transformer

torch_npu.npu.set_device(0)

def timeout_handler(signum, frame):
    raise TimeoutError("NPU call timed out - possible stall")

def run_test(name, params, timeout_sec=30):
    """运行单个 stall 测试，返回 (pass, elapsed_sec, detail)"""
    B = params["B"]; S1 = params["S1"]; S2 = params["S2"]
    N1 = params["N1"]; N2 = params["N2"]; D = params["D"]
    dtype = params.get("dtype", "fp16")
    layout = params.get("layout", "BSND")
    keep_prob = params.get("keep_prob", 0.9)
    seed = params.get("seed", 2)
    offset = params.get("offset", 0)
    mask_mode = params.get("mask_mode", 0)
    return_lse = params.get("return_softmax_lse", False)
    loops = params.get("loops", 1)

    torch_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16

    shape_q = (B, S1, N1, D) if layout == "BSND" else (B, N1, S1, D)
    shape_kv = (B, S2, N2, D) if layout == "BSND" else (B, N2, S2, D)

    q = torch.randn(*shape_q, dtype=torch_dtype, device="npu")
    k = torch.randn(*shape_kv, dtype=torch_dtype, device="npu")
    v = torch.randn(*shape_kv, dtype=torch_dtype, device="npu")
    scale = 1.0 / (D ** 0.5)

    attn_mask = None
    if mask_mode != 0:
        attn_mask = torch.triu(torch.ones(2048, 2048), diagonal=1).to(dtype=torch.int8, device="npu")

    meta = cann_ops_transformer.ops.flash_attn_metadata(
        N1, N2, D, batch_size=B, max_seqlen_q=S1, max_seqlen_kv=S2,
        mask_mode=mask_mode, layout_q=layout, layout_kv=layout, layout_out=layout)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  config: B={B} S1={S1} S2={S2} N1={N1} N2={N2} D={D} "
          f"layout={layout} dtype={dtype} keepProb={keep_prob} "
          f"mask_mode={mask_mode} return_lse={return_lse} loops={loops}")
    print(f"{'='*60}")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_sec)

    try:
        t0 = time.time()
        for i in range(loops):
            out, lse = cann_ops_transformer.ops.flash_attn(
                q, k, v, metadata=meta, softmax_scale=scale, mask_mode=mask_mode,
                attn_mask=attn_mask, layout_q=layout, layout_kv=layout, layout_out=layout,
                keep_prob=keep_prob, seed=seed, offset=offset,
                return_softmax_lse=return_lse)
            torch.npu.synchronize()
        elapsed = time.time() - t0
        signal.alarm(0)

        has_nan = torch.isnan(out).any().item()
        lse_ok = True
        if return_lse and lse is not None:
            lse_ok = not torch.isnan(lse).any().item() if lse.numel() > 0 else True

        passed = not has_nan and lse_ok
        detail = f"elapsed={elapsed:.3f}s nan={has_nan} lse_ok={lse_ok} out_shape={out.shape}"
        if passed:
            print(f"  ✓ PASS — {detail}")
        else:
            print(f"  ✗ FAIL — {detail}")
        return passed, elapsed, detail

    except TimeoutError:
        signal.alarm(0)
        print(f"  ✗ STALL — timed out after {timeout_sec}s")
        return False, timeout_sec, "STALL"
    except Exception as e:
        signal.alarm(0)
        print(f"  ✗ ERROR — {type(e).__name__}: {e}")
        return False, 0, f"ERROR: {e}"


tests = [
    ("1. 基本 dropmask", {
        "B": 2, "S1": 128, "S2": 128, "N1": 8, "N2": 2, "D": 128,
        "layout": "BSND", "dtype": "fp16", "keep_prob": 0.9,
    }),
    ("2. 多 s2 tile (S2=512)", {
        "B": 1, "S1": 128, "S2": 512, "N1": 8, "N2": 2, "D": 128,
        "layout": "BSND", "dtype": "fp16", "keep_prob": 0.9,
    }),
    ("3. isLastS2Loop 交接 (return_lse=True)", {
        "B": 2, "S1": 128, "S2": 128, "N1": 8, "N2": 2, "D": 128,
        "layout": "BSND", "dtype": "fp16", "keep_prob": 0.9,
        "return_softmax_lse": True,
    }),
    ("4. 多 batch (B=4)", {
        "B": 4, "S1": 128, "S2": 128, "N1": 8, "N2": 2, "D": 128,
        "layout": "BSND", "dtype": "fp16", "keep_prob": 0.9,
    }),
    ("5. PRELOAD_N=2 跨任务 MTE3 (S2=512 + return_lse)", {
        "B": 2, "S1": 128, "S2": 512, "N1": 8, "N2": 2, "D": 128,
        "layout": "BSND", "dtype": "fp16", "keep_prob": 0.9,
        "return_softmax_lse": True,
    }),
    ("6. causal mask + dropmask", {
        "B": 2, "S1": 128, "S2": 128, "N1": 8, "N2": 2, "D": 128,
        "layout": "BNSD", "dtype": "fp16", "keep_prob": 0.9,
        "mask_mode": 3,
    }),
    ("7. 长时间运行 (100 次循环)", {
        "B": 2, "S1": 128, "S2": 128, "N1": 8, "N2": 2, "D": 128,
        "layout": "BSND", "dtype": "fp16", "keep_prob": 0.9,
        "loops": 100,
    }),
]

results = []
for name, params in tests:
    passed, elapsed, detail = run_test(name, params)
    results.append((name, passed, elapsed, detail))

print(f"\n{'='*60}")
print("  STALL TEST SUMMARY")
print(f"{'='*60}")
all_pass = True
for name, passed, elapsed, detail in results:
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}  {name:40s}  {elapsed:.1f}s")
    if not passed:
        all_pass = False
print(f"{'='*60}")
print(f"  {'ALL PASS' if all_pass else 'SOME FAILED'}")
sys.exit(0 if all_pass else 1)
