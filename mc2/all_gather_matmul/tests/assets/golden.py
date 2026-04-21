#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
__golden__ = {
    "kernel": {
        "all_gather_matmul": "all_gather_matmul_golden"
    }
}

import torch
import torch.distributed as dist

def all_gather_matmul_golden(x1, x2, bias=None, is_trans_a=False, is_trans_b=False,
                             gather_index: int = 0, comm_turn: int = 0, rank_size: int = 1,
                             is_gather_out: bool = True, **kwargs):
    """
    AllGather + Matmul 组合计算
    标杆以 CPU 运行结果为准
    计算逻辑来源于 MC2_test 的 get_cpu() 方法
    
    Args:
        x1: 输入张量 1 (REQUIRED) - 来自 Input("x1")
        x2: 输入张量 2 (REQUIRED) - 来自 Input("x2")
        bias: 偏置 (OPTIONAL) - 来自 Input("bias")
        is_trans_a: 是否转置 x1 - 来自 Attr("is_trans_a").Bool(false)
        is_trans_b: 是否转置 x2 - 来自 Attr("is_trans_b").Bool(false)
        gather_index: Gather 索引 - 来自 Attr("gather_index").Int(0)
        comm_turn: 通信轮次 - 来自 Attr("comm_turn").Int(0)
        rank_size: Rank 数量 - 来自 Attr("rank_size").Int(0)
        is_gather_out: 是否输出 gather 结果 - 来自 Attr("is_gather_out").Bool(true)
        **kwargs: 扩展参数
    """
    # CPU 计算逻辑（优先）
    # 步骤 1: 升精度处理（来自 get_cpu）
    x1 = x1.to(torch.float32)
    x2 = x2.to(torch.float32)
    
    # 步骤 2: 参数调整（transpose）
    if is_trans_a:
        x1 = x1.transpose()
    if is_trans_b:
        x2 = x2.transpose()
    
    # 步骤 3: AllGather 操作（CPU 实现，来自 get_cpu）
    try:
        # 分布式环境：使用 dist.all_gather
        all_gather_list = [torch.zeros_like(x1) for _ in range(rank_size)]
        dist.all_gather(all_gather_list, x1)
        gather_output = torch.cat(all_gather_list, dim=0)
    except:
        # 单机测试环境：模拟 all_gather
        gather_output = torch.cat([x1] * rank_size, dim=0)
    
    # 步骤 4: Matmul 计算（CPU 实现，来自 get_cpu）
    output = torch.matmul(gather_output, x2)
    
    # 步骤 5: Bias 处理（补充逻辑）
    if bias is not None:
        bias = bias.to(torch.float32)
        output = output + bias
    
    # 步骤 6: 根据 is_gather_out 返回不同结果
    if is_gather_out:
        return output, gather_output
    else:
        return output