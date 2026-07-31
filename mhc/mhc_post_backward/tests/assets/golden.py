# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""mhc_post_backward TestSpec — kernel (numpy) + aclnn (torch) golden.

Ported from tbetoolkits golden (xrun aclnnMhcPostBackward CPU reference).

4 outputs (order matches def.cpp / aclnn*.h):
  grad_x      = h_res empty ? grad_output
                : sum(h_res.unsqueeze(-1) * grad_output.unsqueeze(-3), dim=-2)  -> x.dtype
  grad_h_res  = matmul(x, grad_output.transpose(-1, -2))                        -> fp32
  grad_h_out  = sum(grad_output * h_post.unsqueeze(-1), dim=-2)                 -> h_out.dtype
  grad_h_post = sum(grad_output * h_out.unsqueeze(-2), dim=-1)                  -> fp32

All ops are dim-relative (-1/-2/-3), so 4D ([B,S,N,D]) and 3D ([T,N,D])
share identical core logic.
"""

import numpy
import torch

__spec__ = {
    "mhc_post_backward": "MhcPostBackwardTestSpec",
    "aclnnMhcPostBackward": "AclnnMhcPostBackwardTestSpec",
}


class MhcPostBackwardTestSpec:
    """mhc_post_backward kernel golden (numpy in; params match def.cpp).
    def.cpp inputs : grad_y, x, h_res, h_out, h_post
    def.cpp outputs: grad_x, grad_h_res, grad_h_out, grad_h_post
    """

    def golden(grad_y, x, h_res, h_out, h_post, **kwargs):
        grad_output = grad_y
        h_res_empty = (h_res is None) or (h_res.size == 0)

        orig_x_dtype = x.dtype
        orig_h_out_dtype = h_out.dtype

        if str(grad_output.dtype) in ("float16", "bfloat16"):
            grad_output = grad_output.astype(numpy.float32)
            x = x.astype(numpy.float32)
            if not h_res_empty:
                h_res = h_res.astype(numpy.float32)
            h_out = h_out.astype(numpy.float32)
            h_post = h_post.astype(numpy.float32)

        grad_output_t = torch.from_numpy(grad_output)
        x_t = torch.from_numpy(x)
        h_res_t = torch.from_numpy(h_res) if not h_res_empty else None
        h_out_t = torch.from_numpy(h_out)
        h_post_t = torch.from_numpy(h_post)

        ndim = grad_output_t.ndim
        if ndim not in (3, 4):
            raise ValueError(
                f"[mhc_post_backward] golden error: unexpected ndim: {ndim}"
            )

        if h_res_empty:
            grad_x_fp32 = grad_output_t.clone()
        else:
            grad_x_fp32 = torch.sum(
                h_res_t.unsqueeze(-1) * grad_output_t.unsqueeze(-3), dim=-2
            )
        grad_h_out_fp32 = torch.sum(grad_output_t * h_post_t.unsqueeze(-1), dim=-2)
        grad_h_post_fp32 = torch.sum(grad_output_t * h_out_t.unsqueeze(-2), dim=-1)
        grad_h_res_fp32 = torch.matmul(x_t, grad_output_t.transpose(-1, -2))

        def to_numpy(tensor, target_dtype):
            if str(target_dtype) == "bfloat16":
                return tensor.numpy().astype(target_dtype, copy=False)
            return tensor.numpy().astype(target_dtype)

        grad_x = to_numpy(grad_x_fp32, orig_x_dtype)
        grad_h_out = to_numpy(grad_h_out_fp32, orig_h_out_dtype)
        grad_h_res = grad_h_res_fp32.numpy().astype(numpy.float32)
        grad_h_post = grad_h_post_fp32.numpy().astype(numpy.float32)

        return [grad_x, grad_h_res, grad_h_out, grad_h_post]


class AclnnMhcPostBackwardTestSpec:
    """aclnnMhcPostBackward aclnn golden (torch.Tensor in; params match aclnn*.h).
    header inputs : gradOutput, x, hRes, hOut, hPost
    header outputs: gradX, gradHRes, gradHOut, gradHPost
    """

    def golden(
        gradOutput, x, hRes, hOut, hPost, gradX, gradHRes, gradHOut, gradHPost, **kwargs
    ):
        h_res_empty = (hRes is None) or (hRes.numel() == 0)

        orig_x_dtype = x.dtype
        orig_h_out_dtype = hOut.dtype

        if gradOutput.dtype in (torch.float16, torch.bfloat16):
            grad_output = gradOutput.to(torch.float32)
            x_t = x.to(torch.float32)
            h_res_t = hRes.to(torch.float32) if not h_res_empty else None
            h_out_t = hOut.to(torch.float32)
            h_post_t = hPost.to(torch.float32)
        else:
            grad_output = gradOutput
            x_t = x
            h_res_t = hRes if not h_res_empty else None
            h_out_t = hOut
            h_post_t = hPost

        ndim = grad_output.ndim
        if ndim not in (3, 4):
            raise ValueError(
                f"[aclnnMhcPostBackward] golden error: unexpected ndim: {ndim}"
            )

        if h_res_empty:
            grad_x_fp32 = grad_output.clone()
        else:
            grad_x_fp32 = torch.sum(
                h_res_t.unsqueeze(-1) * grad_output.unsqueeze(-3), dim=-2
            )
        grad_h_out_fp32 = torch.sum(grad_output * h_post_t.unsqueeze(-1), dim=-2)
        grad_h_post_fp32 = torch.sum(grad_output * h_out_t.unsqueeze(-2), dim=-1)
        grad_h_res_fp32 = torch.matmul(x_t, grad_output.transpose(-1, -2))

        grad_x = grad_x_fp32.to(orig_x_dtype)
        grad_h_out = grad_h_out_fp32.to(orig_h_out_dtype)
        grad_h_res = grad_h_res_fp32.to(torch.float32)
        grad_h_post = grad_h_post_fp32.to(torch.float32)

        return [grad_x, grad_h_res, grad_h_out, grad_h_post]
