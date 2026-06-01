# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import torch_npu
import numpy as np
from torch_npu.testing.testcase import TestCase, run_tests
from common.examples import aclnn_example_runner
from .indexer_compress_epilog_golden import (
    DATA_TYPE_INT_TO_STR,
    hifp8_block_quant,
    indexer_compress_epilog_golden,
    requantize_compare,
)

np.random.seed(121)
np.set_printoptions(suppress=True)

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))

class TestCustomIndexerCompressEpilog(TestCase):
    def test_indexer_compress_epilog_normal_quant(self):
        torch_npu.npu.set_device(int(DEVICE_ID))
        b = 1
        s = 8192
        d = 4096
        dst_type = 35
        scale_factor = int((d + 127) / 128)
        quant_mode = 1
        round_scale = True

        np.random.seed(42)

        x = torch.tensor(np.random.uniform(-2, 2, (b * s, d))).to(torch.float16)
        # 构造slot_mapping
        slot_mapping = np.arange(b * s) # 生成下标序列
        np.random.shuffle(slot_mapping) # 随机打乱顺序
        slot_mapping = torch.tensor(slot_mapping).to(torch.int32)

        # 1: 'float16',
        # 27: 'bfloat16',
        # 35: 'float8_e5m2',
        # 36: 'float8_e4m3fn',
        dst_type = 35
        indexer_compress_cache = torch.tensor(np.random.uniform(-2, 2, (b * s, d))).to(torch.float8_e5m2)
        indexer_compress_cache_scale = torch.tensor(np.random.uniform(-2, 2, (b * s, scale_factor))).to(torch.float32)

        indexer_compress_cache_golden, indexer_compress_cache_scale_golden = indexer_compress_epilog_golden(indexer_compress_cache.view(torch.int8).numpy().copy(), indexer_compress_cache_scale.numpy().copy().astype(np.float32), \
            x.numpy().copy().astype(np.float16), slot_mapping.numpy().copy().astype(np.int32), dst_type, quant_mode, round_scale)

        x_npu = x.to("npu:%s" % DEVICE_ID)
        slot_mapping_npu = slot_mapping.to("npu:%s" % DEVICE_ID)
        indexer_compress_cache_npu = indexer_compress_cache.to("npu:%s" % DEVICE_ID)
        indexer_compress_cache_scale_npu = indexer_compress_cache_scale.to("npu:%s" % DEVICE_ID)

        aclnn_example_runner.indexer_compress_epilog(indexer_compress_cache_npu, indexer_compress_cache_scale_npu,
            x_npu, slot_mapping_npu, quant_mode=quant_mode, round_scale=round_scale)

        indexer_compress_cache_cpu = indexer_compress_cache_npu.cpu()
        indexer_compress_cache_scale_cpu = indexer_compress_cache_scale_npu.cpu().float().numpy()
        dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
        cache_close = requantize_compare(torch.from_numpy(indexer_compress_cache_golden.view(np.int8)), indexer_compress_cache_cpu, dst_type_str)
        cache_scale_close = np.allclose(indexer_compress_cache_scale_golden.reshape(-1), indexer_compress_cache_scale_cpu.reshape(-1), rtol=0.00001, atol=0.00001, equal_nan=True)

        self.assertTrue(cache_close, f"indexer_compress_cache precision compare fail")
        self.assertTrue(cache_scale_close, f"indexer_compress_cache_scale precision compare fail")

    def test_indexer_compress_epilog_mx_fp8_quant(self):
        torch_npu.npu.set_device(int(DEVICE_ID))
        b = 1
        s = 8192
        d = 4096
        dst_type = 35
        scale_factor = int((d + 127) / 128)
        quant_mode = 0
        round_scale = True

        np.random.seed(42)

        x = torch.tensor(np.random.uniform(-2, 2, (b * s, d))).to(torch.float16)
        # 构造slot_mapping
        slot_mapping = np.arange(b * s) # 生成下标序列
        np.random.shuffle(slot_mapping) # 随机打乱顺序
        slot_mapping = torch.tensor(slot_mapping).to(torch.int32)

        # 1: 'float16',
        # 27: 'bfloat16',
        # 35: 'float8_e5m2',
        # 36: 'float8_e4m3fn',
        dst_type = 35
        indexer_compress_cache = torch.tensor(np.random.uniform(-2, 2, (b * s, d))).to(torch.float8_e5m2)
        indexer_compress_cache_scale = torch.tensor(np.random.uniform(-2, 2, (b * s, scale_factor))).to(torch.float8_e8m0fnu)

        indexer_compress_cache_golden, indexer_compress_cache_scale_golden = indexer_compress_epilog_golden(indexer_compress_cache.view(torch.int8).numpy().copy(), \
            indexer_compress_cache_scale.view(torch.int8).numpy().copy().astype(np.int8), x.numpy().copy().astype(np.float16), slot_mapping.numpy().copy().astype(np.int32), dst_type, quant_mode, round_scale)

        x_npu = x.to("npu:%s" % DEVICE_ID)
        slot_mapping_npu = slot_mapping.to("npu:%s" % DEVICE_ID)
        indexer_compress_cache_npu = indexer_compress_cache.to("npu:%s" % DEVICE_ID)
        indexer_compress_cache_scale_npu = indexer_compress_cache_scale.to("npu:%s" % DEVICE_ID)

        aclnn_example_runner.indexer_compress_epilog(indexer_compress_cache_npu, indexer_compress_cache_scale_npu,
            x_npu, slot_mapping_npu, quant_mode=quant_mode, round_scale=round_scale, cache_scale_dtype=37)

        indexer_compress_cache_cpu = indexer_compress_cache_npu.cpu()
        indexer_compress_cache_scale_cpu = indexer_compress_cache_scale_npu.cpu().view(torch.int8).numpy().astype(np.int8)
        dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
        cache_close = requantize_compare(torch.from_numpy(indexer_compress_cache_golden.view(np.int8)), indexer_compress_cache_cpu, dst_type_str)
        cache_scale_close = np.allclose(indexer_compress_cache_scale_golden.reshape(-1), indexer_compress_cache_scale_cpu.reshape(-1), rtol=0.00001, atol=0.00001, equal_nan=True)

        self.assertTrue(cache_close, f"indexer_compress_cache precision compare fail")
        self.assertTrue(cache_scale_close, f"indexer_compress_cache_scale precision compare fail")

    def test_indexer_compress_epilog_hifloat8_quant(self):
        torch_npu.npu.set_device(int(DEVICE_ID))

        b = 1
        s = 8192
        d = 4096
        dst_type = 35
        scale_factor = int((d + 127) / 128)
        quant_mode = 2
        round_scale = True
        scale_val = 0.5

        np.random.seed(42)

        x_np = np.random.uniform(-2, 2, (b * s, d)).astype(np.float16)
        #x = torch.tensor(np.random.uniform(-2, 2, (b * s, d))).to(torch.float16)
        # 构造slot_mapping
        slot_mapping = np.arange(b * s) # 生成下标序列
        np.random.shuffle(slot_mapping) # 随机打乱顺序
        #slot_mapping = torch.tensor(slot_mapping).to(torch.int32)

        # 1: 'float16',
        # 27: 'bfloat16',
        # 34: 'hifloat8'
        # 35: 'float8_e5m2',
        # 36: 'float8_e4m3fn',
        dst_type = 34
        indexer_compress_cache_np = np.random.uniform(-2, 2, (b * s, d)).astype(np.float16)
        indexer_compress_cache_scale = torch.tensor(np.random.uniform(0, 0, (b * s, scale_factor))).to(torch.float32)
        indexer_compress_cache_golden = hifp8_block_quant(indexer_compress_cache_np.copy().astype(np.int8),
                                                          x_np.copy(), slot_mapping.copy().astype(np.int32), scale_val)

        x_npu = torch.tensor(x_np).to(torch.float16).to("npu:%s" % DEVICE_ID)
        slot_mapping_npu = torch.tensor(slot_mapping).to(torch.int32).to("npu:%s" % DEVICE_ID)
        indexer_compress_cache_npu = torch.tensor(indexer_compress_cache_np).to("npu:%s" % DEVICE_ID)
        indexer_compress_cache_npu = torch_npu.npu_dtype_cast(indexer_compress_cache_npu, torch_npu.hifloat8)
        indexer_compress_cache_scale_npu = indexer_compress_cache_scale.to("npu:%s" % DEVICE_ID)

        aclnn_example_runner.indexer_compress_epilog(indexer_compress_cache_npu, indexer_compress_cache_scale_npu,
            x_npu, slot_mapping_npu, quant_mode=quant_mode, round_scale=round_scale, scale=scale_val,
            cache_dtype=4)

        indexer_compress_cache_cpu = indexer_compress_cache_npu.cpu()
        dst_type_str = DATA_TYPE_INT_TO_STR[dst_type]
        cache_close = requantize_compare(torch.from_numpy(indexer_compress_cache_golden.view(np.int8)),
                                         indexer_compress_cache_cpu, dst_type_str)

        self.assertTrue(cache_close, f"indexer_compress_cache precision compare fail")

if __name__ == "__main__":
    run_tests()
