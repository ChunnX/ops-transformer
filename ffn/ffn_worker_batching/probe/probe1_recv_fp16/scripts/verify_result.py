#!/usr/bin/env python3
# ============================================================================
# 穿刺1 精度验证：8 路输出 vs CPU golden，np.array_equal 二进制一致(rtol=atol=0)
# ============================================================================
import sys
import numpy as np

# name -> (dtype, count)  count=-1 表示整段
A, M, BS, K, H = 4, 2, 2, 2, 8
EXPERT_NUM = 8
Y = A * BS * K

SPECS = {
    "y":                (np.float16, Y * H),
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
    print("=== 8 路输出 二进制一致比对 (np.array_equal) ===")
    for name, (dtype, cnt) in SPECS.items():
        act = load(f"{outdir}/{name}.bin", dtype, cnt)
        gold = load(f"{outdir}/{name}_golden.bin", dtype, cnt)
        ok = (act.shape == gold.shape) and np.array_equal(act, gold)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name:<18} dtype={np.dtype(dtype).name:<8} n={cnt}")
        if not ok:
            all_ok = False
            # 打印首处差异
            n = min(len(act), len(gold))
            diff = np.where(act[:n] != gold[:n])[0] if n else []
            print(f"        actual={act.tolist()}")
            print(f"        golden={gold.tolist()}")
            if len(diff):
                print(f"        first diff idx={int(diff[0])}")
    print("=== 结论:", "全部 PASS (二进制一致)" if all_ok else "存在 FAIL", "===")
    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()
