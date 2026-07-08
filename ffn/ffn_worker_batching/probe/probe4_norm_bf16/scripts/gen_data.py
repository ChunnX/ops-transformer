"""
probe4_norm_bf16 —— 生成输入 bin + golden bin。
运行目录 = build/（run.sh 中 cd build 后执行）。
输入写 ./input/，golden 写 ./output/golden/，kernel 输出写 ./output/（互不覆盖）。
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import golden as G


def main():
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("output/golden", exist_ok=True)

    expert_ids, token_data_u16, session_tbl, microbatch_tbl = G.gen_inputs()

    # ---- 写输入 ----
    expert_ids.astype(np.int32).tofile("input/expert_ids.bin")
    token_data_u16.astype(np.uint16).tofile("input/token_data.bin")
    session_tbl.astype(np.int32).tofile("input/session_tbl.bin")
    microbatch_tbl.astype(np.int32).tofile("input/microbatch_tbl.bin")

    gold = G.compute_golden(expert_ids, token_data_u16, session_tbl, microbatch_tbl)

    # ---- 写 golden ----
    gold["y"].astype(np.uint16).tofile("output/golden/y.bin")
    gold["group_list"].astype(np.int64).tofile("output/golden/group_list.bin")
    gold["session_ids"].astype(np.int32).tofile("output/golden/session_ids.bin")
    gold["micro_batch_ids"].astype(np.int32).tofile("output/golden/micro_batch_ids.bin")
    gold["token_ids"].astype(np.int32).tofile("output/golden/token_ids.bin")
    gold["expert_offsets"].astype(np.int32).tofile("output/golden/expert_offsets.bin")
    gold["dynamic_scale"].astype(np.float32).tofile("output/golden/dynamic_scale.bin")
    gold["actual_token_num"].astype(np.int64).tofile("output/golden/actual_token_num.bin")

    print("[gen_data] inputs + golden generated.")
    print("  valid_len (actual_token_num) =", int(gold["actual_token_num"][0]))
    print("  expert_ids =", expert_ids.tolist())
    print("  group_list =", gold["group_list"].reshape(-1).tolist())


if __name__ == "__main__":
    main()
