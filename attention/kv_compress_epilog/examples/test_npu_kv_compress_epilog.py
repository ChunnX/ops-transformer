# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import torch_npu
import numpy as np
import math

from torch_npu.testing.testcase import TestCase, run_tests
from common.examples import aclnn_example_runner
from .kv_compress_epilog_golden import DATA_TYPE_INT_TO_STR, hifp8_block_quant, requantize_compare

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))

class TestNpuKvCompressEpilog(TestCase):

    def test_kv_compress_epilog_with_cpu_benchmark(self):
        """测试函数，包含CPU标杆验证"""
        # 设置测试参数（与原始测试相同）
        num_tokens = 4096
        head_dim = 512
        quant_group_size = 128
        quant_mode = 1
        round_scale = True
        
        # 生成输入数据
        np.random.seed(0)
        x = torch.tensor(np.random.uniform(-10, 10, (num_tokens, head_dim))).to(torch.bfloat16)
        
        # 生成slot_mapping
        slot_mapping = torch.tensor([i for i in range(num_tokens)], dtype=torch.int32)
        
        scale_num = math.ceil((head_dim - 64) / 128)
        scale_size = 1
        if quant_mode == 1:
            scale_size = 4
        y_size = head_dim - 64 + 2 * 64 + scale_num * scale_size
        
        kv_len = (128 - y_size % 128) % 128 + y_size
        
        # 初始化kv_compress_cache
        kv_compress_cache = torch.zeros(num_tokens, kv_len, dtype=torch.float8_e5m2)
        
        # 1. 运行NPU操作
        print(f'======================== NPU Eager BEGIN ========================')
        x_npu = x.to("npu:%s" % DEVICE_ID)
        slot_mapping_npu = slot_mapping.to("npu:%s" % DEVICE_ID)
        kv_cache_npu = kv_compress_cache.to("npu:%s" % DEVICE_ID)
        
        aclnn_example_runner.kv_compress_epilog(
            kv_cache_npu,
            x_npu,
            slot_mapping_npu,
            quant_group_size=quant_group_size,
            quant_mode=quant_mode
        )
        print(f'======================== NPU Eager FINISH ========================')
        
        return kv_cache_npu

    def test_kv_compress_epilog_hifloat8_eager(self):
        num_tokens = 4096
        head_dim = 512
        quant_group_size = 128
        quant_mode = 3
        round_scale = True
        scale_val = 0.5
        dst_type = 34

        np.random.seed(0)
        x_np = np.random.uniform(2, 2, (num_tokens, head_dim)).astype(np.float32)

        slot_mapping = np.arange(num_tokens) # 生成下标序列
        np.random.shuffle(slot_mapping) # 随机打乱顺序

        kv_compress_cache_np = np.zeros((num_tokens, head_dim), dtype=np.float16)
        kv_compress_cache_golden = hifp8_block_quant(kv_compress_cache_np.copy().astype(np.int8), x_np.copy(),
                                                     slot_mapping.copy().astype(np.int32), scale_val)

        print(f'======================== Eager BEGIN ========================')
        x_npu = torch.tensor(x_np).to(torch.bfloat16).to("npu:%s" % DEVICE_ID)
        slot_mapping_npu = torch.tensor(slot_mapping).to(torch.int32).to("npu:%s" % DEVICE_ID)
        kv_cache_npu = torch.tensor(kv_compress_cache_np).to("npu:%s" % DEVICE_ID)
        kv_cache_npu = torch_npu.npu_dtype_cast(kv_cache_npu, torch_npu.hifloat8)

        aclnn_example_runner.kv_compress_epilog(
            kv_cache_npu,
            x_npu,
            slot_mapping_npu,
            quant_group_size=quant_group_size,
            quant_mode=quant_mode,
            round_scale_flag=round_scale,
            scale=scale_val,
            kv_cache_dtype=4
        )

        kv_cache_cpu = kv_cache_npu.cpu()

        dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
        cache_close = requantize_compare(torch.from_numpy(kv_compress_cache_golden.view(np.int8)),
                                         kv_cache_cpu, dst_type_str)

        self.assertTrue(cache_close, f"kv_compress_cache precision compare fail")
        print(f'======================== Eager FINISH ========================')



if __name__ == "__main__":
    run_tests()
