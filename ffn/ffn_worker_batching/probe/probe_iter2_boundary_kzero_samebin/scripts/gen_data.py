"""
probe_iter2_boundary_kzero_samebin —— 生成输入 bin + golden bin（两场景 kzero/samebin）。
运行目录 = build/（run.sh 中 cd build 后执行）。
用法：python3 ../scripts/gen_data.py <scenario>   scenario in {kzero, samebin}
输入写 ./input/<scenario>/，golden 写 ./output/<scenario>/golden/，kernel 输出写 ./output/<scenario>/。
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import golden as G


def main():
    scenario = sys.argv[1] if len(sys.argv) > 1 else "kzero"
    if scenario not in ("kzero", "samebin"):
        print(f"[gen_data] unknown scenario {scenario}")
        sys.exit(1)

    in_dir = f"input/{scenario}"
    out_dir = f"output/{scenario}"
    gold_dir = f"{out_dir}/golden"
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(gold_dir, exist_ok=True)

    expert_ids, token_data_u16, session_tbl, microbatch_tbl = G.gen_inputs(scenario)

    # ---- 写输入 ----
    expert_ids.astype(np.int32).tofile(f"{in_dir}/expert_ids.bin")
    token_data_u16.astype(np.uint16).tofile(f"{in_dir}/token_data.bin")
    session_tbl.astype(np.int32).tofile(f"{in_dir}/session_tbl.bin")
    microbatch_tbl.astype(np.int32).tofile(f"{in_dir}/microbatch_tbl.bin")

    gold = G.compute_golden(scenario, expert_ids, token_data_u16, session_tbl, microbatch_tbl)

    # ---- 写 golden ----
    gold["y"].astype(np.uint16).tofile(f"{gold_dir}/y.bin")
    gold["group_list"].astype(np.int64).tofile(f"{gold_dir}/group_list.bin")
    gold["session_ids"].astype(np.int32).tofile(f"{gold_dir}/session_ids.bin")
    gold["micro_batch_ids"].astype(np.int32).tofile(f"{gold_dir}/micro_batch_ids.bin")
    gold["token_ids"].astype(np.int32).tofile(f"{gold_dir}/token_ids.bin")
    gold["expert_offsets"].astype(np.int32).tofile(f"{gold_dir}/expert_offsets.bin")
    gold["dynamic_scale"].astype(np.float32).tofile(f"{gold_dir}/dynamic_scale.bin")
    gold["actual_token_num"].astype(np.int64).tofile(f"{gold_dir}/actual_token_num.bin")

    print(f"[gen_data] scenario={scenario} inputs + golden generated.")
    print("  expert_num          =", G.expert_num_of(scenario))
    print("  actual_token_num    =", int(gold["actual_token_num"][0]))
    print("  group_list[:4] rows =", gold["group_list"][:4].reshape(-1).tolist())


if __name__ == "__main__":
    main()
