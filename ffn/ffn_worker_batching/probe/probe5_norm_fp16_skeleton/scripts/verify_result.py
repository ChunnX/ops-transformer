"""
probe5_norm_fp16_skeleton —— 精度验证：8 路输出与 golden 二进制一致 (np.array_equal, rtol=atol=0)。
运行目录 = build/。逐路对比 output/<name>.bin vs output/golden/<name>.bin。
任一路不一致 -> 退出码 1。y 以 uint16 视图逐 bit 比对（fp16 原样字节 gather）。
"""
import sys
import numpy as np

# name -> dtype
OUTPUTS = [
    ("y", np.uint16),                 # bf16 raw bytes（byte-exact）
    ("group_list", np.int64),
    ("session_ids", np.int32),
    ("micro_batch_ids", np.int32),
    ("token_ids", np.int32),
    ("expert_offsets", np.int32),
    ("dynamic_scale", np.float32),
    ("actual_token_num", np.int64),
]


def main():
    all_ok = True
    print("=== probe4_norm_bf16 精度验证 (二进制一致 np.array_equal) ===")
    for name, dt in OUTPUTS:
        out = np.fromfile(f"output/{name}.bin", dtype=dt)
        gold = np.fromfile(f"output/golden/{name}.bin", dtype=dt)
        if out.shape != gold.shape:
            print(f"  [FAIL] {name:18s} shape {out.shape} != golden {gold.shape}")
            all_ok = False
            continue
        # dynamic_scale 是 fp32 但恒 0（bf16 不用），用 bit 级比较避免 -0/NaN 干扰
        if name == "dynamic_scale":
            eq = np.array_equal(out.view(np.uint32), gold.view(np.uint32))
        else:
            eq = np.array_equal(out, gold)
        if eq:
            print(f"  [PASS] {name:18s} ({out.size} elems) 二进制一致")
        else:
            n_diff = int(np.sum(out.view(np.uint8) != gold.view(np.uint8)))
            print(f"  [FAIL] {name:18s} 不一致，差异字节数={n_diff}")
            print(f"         out ={out.reshape(-1)[:32].tolist()}")
            print(f"         gold={gold.reshape(-1)[:32].tolist()}")
            all_ok = False

    if all_ok:
        print("=== 全部 8 路输出二进制一致：PASS ===")
        sys.exit(0)
    else:
        print("=== 存在不一致：FAIL ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
