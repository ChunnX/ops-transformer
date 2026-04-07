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
    ranksize = 2
    seed = 1185
    
    np.random.seed(seed)
    
    output_dir = "./golden/quantallreduce_"
    os.makedirs(output_dir, exist_ok=True)
    
    for rank in range(ranksize):
        x_data = np.random.randint(-128, 127, size=(bs, hidden_size), dtype=np.int8)
        x_file = os.path.join(output_dir, f"input_x_{rank}.bin")
        x_data.tofile(x_file)
        print(f"Rank {rank}: 生成输入x数据 -> {x_file}")
    
    scale_size = hidden_size // 128
    for rank in range(ranksize):
        scale_data = np.random.uniform(-3.0, 3.0, size=(bs, scale_size)).astype(np.float32)
        scale_data = scale_data.view(np.int32)
        scale_file = os.path.join(output_dir, f"input_scale_{rank}.bin")
        scale_data.tofile(scale_file)
        print(f"Rank {rank}: 生成scale数据 -> {scale_file}")
    
    print(f"\nGolden数据生成完成！")
    print(f"目录：{output_dir}")

if __name__ == '__main__':
    generate_golden()
