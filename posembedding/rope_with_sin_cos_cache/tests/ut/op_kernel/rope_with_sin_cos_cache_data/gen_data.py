#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2025 Huawei Technologies Co., Ltd
import sys
import numpy as np
import torch
bf16 = tf.bfloat16.as_numpy_dtype
np.random.seed(0)

def gen_golden_data_simple(numTokens, numQHeads, numKHeads, headSize, rotaryDim, dtype):
    d_type_dict = {
        "fp32": np.float32,
        "fp16": np.float16,
        "bf16": bf16
    }

    dtype = d_type_dict.get(dtype)

    data_position = np.random.uniform(1, int(numTokens), [int(numTokens)]).astype(dtype)
    data_query = np.random.uniform(-2, 2, [int(numTokens), int(numQHeads * headSize)]).astype(dtype)
    data_key = np.random.uniform(-2, 2, [int(numTokens), int(numKHeads * headSize)]).astype(dtype)
    data_cosSinCache = np.random.uniform(-2, 2, [int(numTokens), int(rotaryDim)]).astype(dtype)

    data_position.tofile("./input_position.bin")
    data_query.tofile("./input_query.bin")
    data_key.tofile("./input_key.bin")
    data_cosSinCache.tofile("./input_cosSinCache.bin")

    data_queryOut = data_query
    data_keyOut = data_key

    data_queryOut.tofile("./output_queryOut.bin")
    data_keyOut.tofile("./output_keyOut.bin")


if __name__ == "__main__":
    os.system("rm -rf *.bin")
    gen_golden_data_simple(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
