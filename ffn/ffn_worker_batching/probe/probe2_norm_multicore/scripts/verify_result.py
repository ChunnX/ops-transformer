# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd. — CANN Open Software License Agreement Version 2.0.
# ----------------------------------------------------------------------------------------------------------
# 迭代一 穿刺2 (probe2_norm_multicore) — 8 路输出二进制一致校验 (np.array_equal, rtol=atol=0)
#   通过标准：全部 8 路输出与 CPU golden 逐 bit 相同。
#   y 以 uint16 视图逐 bit 比对 (原样字节 gather，非重算)。
# ----------------------------------------------------------------------------------------------------------
import numpy as np
import sys

# (name, dtype_for_bitwise_compare)
OUTPUTS = [
    ("sorted_expert_ids", np.int32),
    ("gather_idx", np.int32),
    ("y", np.uint16),                 # fp16 -> uint16 逐 bit
    ("session_ids", np.int32),
    ("micro_batch_ids", np.int32),
    ("token_ids", np.int32),
    ("group_count", np.int32),
    ("actual_token_num", np.int32),
]


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "output"
    all_pass = True
    print("=== probe2 二进制一致校验 (np.array_equal) ===")
    for name, dt in OUTPUTS:
        gpath = f"{out_dir}/golden_{name}.bin"
        opath = f"{out_dir}/{name}.bin"
        golden = np.fromfile(gpath, dtype=dt)
        out = np.fromfile(opath, dtype=dt)
        if out.shape != golden.shape:
            print(f"[FAIL] {name:18s} shape mismatch: npu {out.shape} vs golden {golden.shape}")
            all_pass = False
            continue
        if np.array_equal(out, golden):
            print(f"[PASS] {name:18s} bitwise-equal ({out.size} elems, {dt.__name__})")
        else:
            n_diff = int(np.sum(out != golden))
            first = int(np.argmax(out != golden))
            print(f"[FAIL] {name:18s} {n_diff}/{out.size} differ; first@{first} npu={out[first]} golden={golden[first]}")
            all_pass = False

    print("=== 结果:", "全部 PASS (8/8 二进制一致)" if all_pass else "存在 FAIL", "===")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
