# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import torch_npu
import numpy as np
from torch_npu.testing.testcase import TestCase, run_tests
from common.examples import aclnn_example_runner
from .moe_gating_top_k_hash_golden import moe_gating_top_k_hash_golden

np.random.seed(21)  # 固定随机种子
np.set_printoptions(suppress=True)

DEVICE_ID = 0
torch_npu.npu.set_device(int(DEVICE_ID))

class TestCustomMoeGatingTopK(TestCase):
    def test_moe_gating_top_k_different_dtypes(self):
        """测试不同数据类型的MOE gating top_k算子"""
        torch_npu.npu.set_device(int(DEVICE_ID))

        # 设置参数
        batch_size = 16
        expert_count = 256
        k = 6
        kGroup = 1
        groupCount = 1
        routedScalingFactor = 1.0
        eps = 1e-6
        groupSelectMode = 0
        renorm = 0
        outFlag = False
        N = 100

        # 测试不同数据类型
        dtypes = [torch.float16, torch.bfloat16, torch.float32]
        normTypes = [1, 2] # 测试sigmoid及softplus

        for dtype in dtypes:
            for normType in normTypes:
                print(f"Testing dtype: {dtype}, norm_type: {normType}")

                # 创建输入数据
                np.random.seed(42)
                x = torch.tensor(np.random.uniform(-2, 2, (batch_size, expert_count))).to(dtype)
                input_ids = torch.tensor(np.random.uniform(0, N, (batch_size, ))).to(torch.int64)
                tid2eid = torch.tensor(np.random.uniform(0, expert_count, (N, k))).to(torch.int32)
                
                # CPU参考结果
                cpu_yOut, cpu_expertIdxOut = moe_gating_top_k_hash_golden(
                    x, input_ids.numpy(), tid2eid.numpy(), None, k=k, k_group=kGroup, group_count=groupCount,
                    routed_scaling_factor=routedScalingFactor, eps=eps,
                    group_select_mode=groupSelectMode, renorm=renorm, norm_type=normType, y2_flag=outFlag
                )

                # 转换到NPU
                x_npu = x.to("npu:%s" % DEVICE_ID)
                input_ids_npu = input_ids.to("npu:%s" % DEVICE_ID)
                tid2eid_npu = tid2eid.to("npu:%s" % DEVICE_ID)

                # NPU计算
                npu_yOut, npu_expertIdxOut, _ = aclnn_example_runner.moe_gating_top_k_hash(
                    x_npu, k, bias=None, input_ids=input_ids_npu, tid2eid=tid2eid_npu, k_group=kGroup, group_count=groupCount,
                    routed_scaling_factor=routedScalingFactor, eps=eps,
                    group_select_mode=groupSelectMode, renorm=renorm, norm_type=normType, out_flag=outFlag
                )

                # # 转换回CPU进行比较
                npu_yOut_cpu = npu_yOut.cpu().float().numpy()
                npu_expertIdxOut_cpu = npu_expertIdxOut.cpu().int().numpy()

                cpu_yOut = cpu_yOut.float().numpy()

                # # 验证结果
                yOut_close = np.allclose(npu_yOut_cpu, cpu_yOut, rtol=0.01, atol=0.001, equal_nan=True)
                expertIdxOut_equal = np.array_equal(npu_expertIdxOut_cpu, cpu_expertIdxOut)

                print(f"  yOut close: {yOut_close}, expertIdxOut equal: {expertIdxOut_equal}")

                self.assertTrue(yOut_close, f"yOut precision compare fail for dtype {dtype}")
                self.assertTrue(expertIdxOut_equal, f"expertIdxOut compare fail for dtype {dtype}")

    def test_moe_gating_top_k_384_experts_topk6(self):
        """测试 384 专家 + topk 6 场景"""
        torch_npu.npu.set_device(int(DEVICE_ID))

        # 设置参数 - 384专家 + topk 6
        batch_size = 16
        expert_count = 384
        k = 6
        kGroup = 1
        groupCount = 1
        routedScalingFactor = 1.0
        eps = 1e-6
        groupSelectMode = 0
        renorm = 0
        outFlag = False
        N = 100

        # 测试不同数据类型
        dtypes = [torch.float16, torch.bfloat16]
        normTypes = [1, 2]

        for dtype in dtypes:
            for normType in normTypes:
                print(f"Testing 384 experts: dtype: {dtype}, norm_type: {normType}")

                # 创建输入数据
                np.random.seed(42)
                x = torch.tensor(np.random.uniform(-2, 2, (batch_size, expert_count))).to(dtype)
                
                # CPU参考结果
                cpu_yOut, cpu_expertIdxOut = moe_gating_top_k_hash_golden(
                    x, None, None, None, k=k, k_group=kGroup, group_count=groupCount,
                    routed_scaling_factor=routedScalingFactor, eps=eps,
                    group_select_mode=groupSelectMode, renorm=renorm, norm_type=normType, y2_flag=outFlag
                )

                # 转换到NPU
                x_npu = x.to("npu:%s" % DEVICE_ID)

                # NPU计算
                npu_yOut, npu_expertIdxOut, _ = aclnn_example_runner.moe_gating_top_k_hash(
                    x_npu, k, bias=None, input_ids=None, tid2eid=None, k_group=kGroup, group_count=groupCount,
                    routed_scaling_factor=routedScalingFactor, eps=eps,
                    group_select_mode=groupSelectMode, renorm=renorm, norm_type=normType, out_flag=outFlag
                )

                # 转换回CPU进行比较
                npu_yOut_cpu = npu_yOut.cpu().float().numpy()
                npu_expertIdxOut_cpu = npu_expertIdxOut.cpu().int().numpy()
                cpu_yOut = cpu_yOut.float().numpy()

                # 验证结果
                yOut_close = np.allclose(npu_yOut_cpu, cpu_yOut, rtol=0.01, atol=0.001, equal_nan=True)
                expertIdxOut_equal = np.array_equal(npu_expertIdxOut_cpu, cpu_expertIdxOut)

                print(f"  yOut close: {yOut_close}, expertIdxOut equal: {expertIdxOut_equal}")

                self.assertTrue(yOut_close, f"yOut precision compare fail for 384 experts dtype {dtype}")
                self.assertTrue(expertIdxOut_equal, f"expertIdxOut compare fail for 384 experts dtype {dtype}")

if __name__ == "__main__":
    run_tests()
