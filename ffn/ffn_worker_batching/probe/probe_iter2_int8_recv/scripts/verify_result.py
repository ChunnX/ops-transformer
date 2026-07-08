#!/usr/bin/env python3
# ============================================================================
# 迭代二 穿刺1 (probe_iter2_int8_recv) 精度验证：
#   8 路输出 vs CPU golden，np.array_equal 二进制一致(rtol=atol=0)。
#   P2 int8 + RECV：y=int8，dynamic_scale=有效 FP32(须逐 bit 相等)。
# ============================================================================
import sys
import numpy as np

# shape 与 op_host/ffn_wb_recv_probe.asc + scripts/gen_data.py 一致
A, M, BS, K, H = 8, 2, 2, 3, 128
EXPERT_NUM = 16
Y = A * BS * K   # 48

SPECS = {
    "y":                (np.int8,    Y * H),
    "group_list":       (np.int64,   EXPERT_NUM * 2),
    "session_ids":      (np.int32,   Y),
    "micro_batch_ids":  (np.int32,   Y),
    "token_ids":        (np.int32,   Y),
    "expert_offsets":   (np.int32,   Y),
    "dynamic_scale":    (np.float32, Y),
    "actual_token_num": (np.int64,   1),
}


def load(path, dtype, count):
    return np.fromfile(path, dtype=dtype, count=count)


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "output"
    all_ok = True
    print("=== 8 路输出 二进制一致比对 (np.array_equal, rtol=atol=0) ===")
    for name, (dtype, cnt) in SPECS.items():
        act = load(f"{outdir}/{name}.bin", dtype, cnt)
        gold = load(f"{outdir}/{name}_golden.bin", dtype, cnt)
        ok = (act.shape == gold.shape) and np.array_equal(act, gold)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name:<18} dtype={np.dtype(dtype).name:<8} n={cnt}")
        if not ok:
            all_ok = False
            n = min(len(act), len(gold))
            diff = np.where(act[:n] != gold[:n])[0] if n else []
            print(f"        actual(head)={act[:min(n, 32)].tolist()}")
            print(f"        golden(head)={gold[:min(n, 32)].tolist()}")
            if len(diff):
                print(f"        first diff idx={int(diff[0])} ({diff.size} total)")
    print("=== 结论:", "全部 PASS (二进制一致)" if all_ok else "存在 FAIL", "===")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
