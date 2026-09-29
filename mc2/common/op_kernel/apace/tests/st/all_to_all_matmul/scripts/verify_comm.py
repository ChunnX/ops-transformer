#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import os
import sys

import numpy as np

# 按原始 2 字节位模式比较（fp16/bf16 通用；位模式相等即字节相等，NaN 也安全）
DATA_TYPE = np.uint16


def verify_comm(m, k, rank_num, base_dir="./output"):
    all_pass = True
    print(f"\n{'=' * 60}")
    print(f"Verifying comm buffers for {rank_num} ranks (comm_buffer.bin vs expected_comm.bin)")
    print(f"Layout: window[r] = concat_s a_local[s][r*m:(r+1)*m, :], size={m * k} elems")
    print(f"{'=' * 60}")

    for rank_id in range(rank_num):
        comm_path = os.path.join(base_dir, str(rank_id), "comm_buffer.bin")
        golden_path = os.path.join(base_dir, str(rank_id), "expected_comm.bin")

        if not os.path.exists(comm_path):
            print(f"[FAIL] Rank {rank_id}: comm_buffer.bin not found: {comm_path}")
            all_pass = False
            continue
        if not os.path.exists(golden_path):
            print(f"[FAIL] Rank {rank_id}: expected_comm.bin not found: {golden_path}")
            all_pass = False
            continue

        comm = np.fromfile(comm_path, dtype=DATA_TYPE)
        golden = np.fromfile(golden_path, dtype=DATA_TYPE)

        expected_size = m * k
        if comm.size != expected_size:
            print(f"[FAIL] Rank {rank_id}: comm size {comm.size} != expected {expected_size}")
            all_pass = False
            continue
        if golden.size != expected_size:
            print(f"[FAIL] Rank {rank_id}: golden size {golden.size} != expected {expected_size}")
            all_pass = False
            continue

        if np.array_equal(comm, golden):
            print(f"[PASS] Rank {rank_id}: comm buffer matches expected")
            continue

        diff = np.nonzero(comm != golden)[0]
        n_mismatch = diff.size
        print(f"[FAIL] Rank {rank_id}: {n_mismatch}/{expected_size} elements mismatch")
        if n_mismatch > 0:
            idx = int(diff[0])
            rank_slots = rank_num
            slot = idx // (m * k // rank_slots) if rank_slots else 0
            print(f"       first mismatch at flat idx={idx} (slot from rank ~{slot})")
            print(f"       expected={golden[idx]}, got={comm[idx]}")
            print(f"       expected file: {golden_path}")
            print(f"       got      file: {comm_path}")
        all_pass = False

    print(f"\n{'=' * 60}")
    print("Comm Verification Summary: PASS" if all_pass else "Comm Verification Summary: FAIL")
    print(f"{'=' * 60}")
    return all_pass


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 verify_comm.py m k rank_num [base_dir]")
        print("  m: matrix M dimension (per-rank output rows)")
        print("  k: total K dimension")
        print("  rank_num: number of ranks")
        print("  base_dir: output directory (default: ./output)")
        print("\nExpected file structure:")
        print("  {base_dir}/0/comm_buffer.bin     - dumped comm window from main.cpp")
        print("  {base_dir}/0/expected_comm.bin   - CPU golden comm window from gen_data.py")
        print("  ...")
        sys.exit(1)

    m = int(sys.argv[1])
    k = int(sys.argv[2])
    rank_num = int(sys.argv[3])
    base_dir = sys.argv[4] if len(sys.argv) > 4 else "./output"

    try:
        all_pass = verify_comm(m, k, rank_num, base_dir)
        if not all_pass:
            raise ValueError("[ERROR] Some comm buffers differ from expected.\n")
        print(f"\n[ALL PASS] All {rank_num} ranks passed comm verification!\n")
    except Exception as e:
        print(e)
        sys.exit(1)
