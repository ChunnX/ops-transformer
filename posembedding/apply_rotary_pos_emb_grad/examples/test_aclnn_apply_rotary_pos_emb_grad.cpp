/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

/*!
 * \file test_aclnn_apply_rotary_pos_emb_grad.cpp
 * \brief aclnn 调用示例
 *
 * torch API:
 *   torch.ops.custom.npu_apply_rotary_pos_emb_backward(
 *     grad_query_embed, grad_key_embed, cos, sin,
 *     *, query=None, key=None, rotary_mode="half", layout=1)
 */
#include <iostream>

int main()
{
    std::cout << "npu_apply_rotary_pos_emb_backward — 双路 Q+K rope 反向梯度" << std::endl;
    std::cout << "输入: grad_query_embed, grad_key_embed, cos, sin, [query], [key]" << std::endl;
    std::cout << "输出: grad_query, grad_key, grad_cos, grad_sin" << std::endl;
    std::cout << "属性: rotary_mode=\"half\", layout={1=BSND,2=SBND,3=BNSD,4=TND} default=1" << std::endl;
    std::cout << "仅 Ascend 950 支持" << std::endl;

    // 示例调用 (Python 侧):
    //   import torch
    //   grad_q_out, grad_k_out, grad_cos, grad_sin = torch.ops.custom.npu_apply_rotary_pos_emb_backward(
    //       grad_query_embed, grad_key_embed, cos, sin,
    //       query=query, key=key, rotary_mode="half", layout=1)

    return 0;
}
