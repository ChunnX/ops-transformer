#!/usr/bin/python
# -*- coding: utf-8 -*-
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
import torch 

def _compare_res(golden_res, npu_res, ratio_thress_hold=0.999):
    golden_res = golden_res.cpu()
    npu_res = npu_res.cpu()
    npu_res, _ = torch.sort(npu_res)
    npu_res = npu_res.reshape(-1) 
    golden_res, _ = torch.sort(golden_res)
    golden_res = golden_res.reshape(-1)
    total_res_num = golden_res.numel()
    diff_res = npu_res - golden_res
    match_ratio = (diff_res == 0).sum().float() / total_res_num
    result = "pass"
    if match_ratio >= ratio_thress_hold:
        print(f"Compare npu cpu res success! Match ratio is {match_ratio:.4%}")
        return result, match_ratio
    else:
        print(f"Match ratio {match_ratio:.4%} is under thress_hold {ratio_thress_hold} ",
              "Please Check!")
        non_zero_index = torch.nonzero(diff_res, as_tuple=False).squeeze(1)
        npu_index = npu_res[non_zero_index]
        golden_index = golden_res[non_zero_index]
        for i in non_zero_index:
            print(f"mismatch idx: {i}, golden and npu res is {golden_res[i]} || {npu_res[i]}")
        result = "failed"
        return result, match_ratio