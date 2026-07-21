"""FusedQKV vs torch.matmul performance comparison -- subprocess isolation"""

import subprocess
import sys
import os
import time

CASES = [
    (1, 256, 128, 64, 32, 32),
    (1, 512, 128, 64, 32, 32),
    (1, 1024, 128, 64, 32, 32),
    (1, 2048, 128, 64, 32, 32),
]

FUSED_CODE = r"""
import ctypes, os, time, numpy as np
RTLD_GLOBAL, RTLD_LAZY = 0x100, 0x1
OP = os.environ.get("ASCEND_OPP_PATH", "")
try:
    ctypes.CDLL(f"{OP}/built-in/op_impl/ai_core/tbe/op_host/lib/linux/aarch64/libophost_transformer.so",
                mode=RTLD_GLOBAL | RTLD_LAZY)
except Exception:
    pass
import torch, torch_npu
# Use device 3 to avoid torch_npu ACL contamination on device 0
torch.npu.set_device(3)
AH = os.environ.get("ASCEND_HOME_PATH", "")
lib_n = ctypes.cdll.LoadLibrary(f"{AH}/lib64/libnnopbase.so")
lib_c = ctypes.cdll.LoadLibrary(f"{OP}/vendors/custom_transformer/op_api/lib/libcust_opapi.so")
lib_c.aclnnFusedQkvProjectionGetWorkspaceSize.restype = ctypes.c_int32
lib_c.aclnnFusedQkvProjectionGetWorkspaceSize.argtypes = (
    [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 3 +
    [ctypes.c_void_p] * 3 +
    [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)])
lib_c.aclnnFusedQkvProjection.restype = ctypes.c_int32
lib_c.aclnnFusedQkvProjection.argtypes = [
    ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p]
lib_n.aclCreateTensor.restype = ctypes.c_void_p
lib_n.aclCreateTensor.argtypes = [
    ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64, ctypes.c_void_p]
def mk_t(t):
    s = list(t.shape); n = len(t.shape); st = [1] * n
    for i in range(n - 2, -1, -1):
        st[i] = t.shape[i + 1] * st[i + 1]
    return lib_n.aclCreateTensor(
        (ctypes.c_int64 * n)(*s), n, 0,
        (ctypes.c_int64 * n)(*st), 0, 0,
        (ctypes.c_int64 * n)(*s), n,
        ctypes.c_void_p(t.data_ptr()))
CASES = __CASES__
results = []
for b,s,h,q,k,v in CASES:
    np.random.seed(42)
    hs_t = torch.from_numpy(np.random.randn(b, s, h).astype(np.float32)).to("npu:3")
    wt_t = torch.from_numpy(np.random.randn(h, q + k + v).astype(np.float32)).to("npu:3")
    q_t = torch.zeros((b, s, q), device="npu:3")
    k_t = torch.zeros((b, s, k), device="npu:3")
    v_t = torch.zeros((b, s, v), device="npu:3")
    a = [mk_t(t) for t in [hs_t, wt_t, q_t, k_t, v_t]]
    ws = ctypes.c_uint64(0)
    ex = ctypes.c_void_p()
    lib_c.aclnnFusedQkvProjectionGetWorkspaceSize(
        a[0], a[1], None, q, k, v, a[2], a[3], a[4],
        ctypes.byref(ws), ctypes.byref(ex))
    wsp = ctypes.c_void_p(0)
    if ws.value > 0:
        wsd = torch.empty(ws.value, dtype=torch.uint8, device="npu:3")
        wsp = ctypes.c_void_p(wsd.data_ptr())
    st = ctypes.c_void_p(torch_npu.npu.current_stream().npu_stream)
    for _ in range(5):
        lib_c.aclnnFusedQkvProjection(wsp, ws, ex, st)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        lib_c.aclnnFusedQkvProjection(wsp, ws, ex, st)
    torch.npu.synchronize()
    ms = (time.perf_counter() - t0) / 20 * 1000
    results.append(ms)
print("RESULT:" + ",".join(f"{r:.4f}" for r in results))
sys.stdout.flush()
"""

TORCH_CODE = r"""
import torch, torch_npu, time, numpy as np, sys
CASES = __CASES__
results = []
for b,s,h,q,k,v in CASES:
    np.random.seed(42)
    hs = torch.from_numpy(np.random.randn(b*s,h).astype(np.float32)).to("npu:0")
    wt = torch.from_numpy(np.random.randn(h,q+k+v).astype(np.float32)).to("npu:0")
    wq,wk,wv = wt[:,:q], wt[:,q:q+k], wt[:,q+k:]
    for _ in range(10): _ = hs @ wq; _ = hs @ wk; _ = hs @ wv
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(100): _ = hs @ wq; _ = hs @ wk; _ = hs @ wv
    torch.npu.synchronize()
    ms = (time.perf_counter() - t0) / 100 * 1000
    results.append(ms)
print("RESULT:" + ",".join(f"{r:.4f}" for r in results))
sys.stdout.flush()
"""


def run_bench(desc, code, timeout=600):
    """Run benchmark in subprocess and return list of latencies (ms)."""
    cases_src = "CASES = [\n" + "".join(f"    {c},\n" for c in CASES) + "]"
    src = code.replace("__CASES__", cases_src)
    try:
        tc = subprocess.run(
            [sys.executable, "-c", src],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "ASCEND_OPP_PATH": os.environ.get("ASCEND_OPP_PATH", "")}
        )
        for line in tc.stdout.splitlines():
            if line.startswith("RESULT:"):
                return [float(x) for x in line[7:].split(",")]
        print(f"  [{desc}] no RESULT. stderr: {tc.stderr[:200]}")
        return None
    except subprocess.TimeoutExpired:
        print(f"  [{desc}] TIMEOUT")
        return None
    except Exception as e:
        print(f"  [{desc}] ERR: {e}")
        return None


def main():
    header = (
        f"{'Case':<25} {'torch(ms)':<12} {'fused(ms)':<10} {'speedup':<10}\n"
        + "-" * 59
    )
    print(header)

    # Run fused benchmarks
    print("\nRunning fused benchmarks (device 3)...")
    fused_results = run_bench("fused", FUSED_CODE)

    # Run torch benchmarks
    print("\nRunning torch benchmarks (device 0)...")
    torch_results = run_bench("torch", TORCH_CODE)

    print()
    if fused_results is None and torch_results is None:
        print("Both benchmarks failed!")
        return
    if fused_results is None:
        print("Fused benchmark failed entirely.")
        return
    if torch_results is None:
        print("Torch benchmark failed entirely.")
        return

    for i, (b, s, h, q, k, v) in enumerate(CASES):
        M = b * s
        tm = torch_results[i]
        fm = fused_results[i]
        sp = tm / fm if fm > 0 else float("inf")
        desc = f"M={M} H={h} q={q} k={k} v={v}"
        speedup_str = f"{sp:.2f}x" if sp > 0 else "FAIL"
        print(
            f"{desc:<25} {tm:<12.4f} {fm:<10.2f} {speedup_str:<10}"
        )


if __name__ == "__main__":
    main()
