#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe3_norm_int8 精度验证：8 路输出与 CPU golden 二进制一致 (np.array_equal, rtol=atol=0)。
任一路不一致即退出码 1。
"""
import os
import sys
import numpy as np

OUT_DIR = "./output"

# (名称, dtype, 是否 P2 关键路径)
SPECS = [
    ("y", np.int8),
    ("group_list", np.int64),
    ("session_ids", np.int32),
    ("micro_batch_ids", np.int32),
    ("token_ids", np.int32),
    ("expert_offsets", np.int32),
    ("dynamic_scale", np.float32),
    ("actual_token_num", np.int64),
]


def load(name, dtype):
    return np.fromfile(os.path.join(OUT_DIR, name), dtype=dtype)


def main():
    all_ok = True
    print("=== probe3_norm_int8 二进制一致校验 (np.array_equal) ===")
    for name, dtype in SPECS:
        out_p = os.path.join(OUT_DIR, f"{name}.bin")
        gold_p = os.path.join(OUT_DIR, f"golden_{name}.bin")
        if not os.path.exists(out_p):
            print(f"[FAIL] {name}: 输出 {out_p} 不存在")
            all_ok = False
            continue
        out = np.fromfile(out_p, dtype=dtype)
        gold = np.fromfile(gold_p, dtype=dtype)
        if out.shape != gold.shape:
            print(f"[FAIL] {name}: shape 不一致 out={out.shape} gold={gold.shape}")
            all_ok = False
            continue
        # 对 float32 也用严格 array_equal（原样字节搬运，无算术，须逐 bit 相等）
        equal = np.array_equal(out, gold)
        if equal:
            print(f"[ OK ] {name}: bitwise equal ({out.size} elems, {dtype.__name__})")
        else:
            all_ok = False
            diff = np.where(out != gold)[0]
            print(f"[FAIL] {name}: {diff.size}/{out.size} 元素不一致")
            show = diff[:12]
            for idx in show:
                print(f"        idx={idx} out={out[idx]} gold={gold[idx]}")
    print("=== 结果:", "全部通过 ✅" if all_ok else "存在不一致 ❌", "===")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
