#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import os
import numpy as np

def generate_golden():
    bs = 16
    hidden_size = 8192
    ranksize = 4
    seed = 1185

    np.random.seed(seed)

    output_dir = "./golden"
    os.makedirs(output_dir, exist_ok=True)

    x_data_list = []
    for rank in range(ranksize):
        x_data = np.random.randint(-128, 127, size=(bs, hidden_size), dtype=np.int8)
        x_file = os.path.join(output_dir, f"quantallreduce_input_x_{rank}.bin")
        x_data.tofile(x_file)
        x_data_list.append(x_data)
        print(f"Rank {rank}: generated input x -> {x_file}")

    allgather_output = np.concatenate(x_data_list, axis=0)
    for rank in range(ranksize):
        output_file = os.path.join(output_dir, f"quantallreduce_output_cpu_{rank}.bin")
        allgather_output.tofile(output_file)
        print(f"Rank {rank}: generated AllGather golden output -> {output_file}")

    print(f"\nGolden data generation complete!")
    print(f"Directory: {output_dir}")

if __name__ == '__main__':
    generate_golden()
