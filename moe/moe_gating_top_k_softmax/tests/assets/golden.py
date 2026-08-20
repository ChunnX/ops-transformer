#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


import numpy
import torch

__spec__ = {
    "moe_gating_top_k_softmax": "MoeGatingTopKSoftmaxTestSpec",
    "aclnnMoeGatingTopKSoftmax": "AclnnMoeGatingTopKSoftmaxTestSpec",
    "aclnnMoeGatingTopKV2": "AclnnMoeGatingTopKV2TestSpec",
}


def _softmax_numpy(x, axis=-1):
    if "float16" in x.dtype.name:
        x = x.astype(numpy.float32)
    x_max = x.max(axis=axis, keepdims=True)
    x_sub = x - x_max
    y = numpy.exp(x_sub)
    x_sum = y.sum(axis=axis, keepdims=True)
    return y / x_sum


def _softmax_torch(x, axis=-1):
    if x.dtype == torch.float16:
        x = x.to(torch.float32)
    x_max = x.max(dim=axis, keepdim=True).values
    x_sub = x - x_max
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=axis, keepdim=True)
    return y / x_sum


class AclnnMoeGatingTopKSoftmaxTestSpec:
    @staticmethod
    def golden(
        x,
        finishedOptional=None,
        k=1,
        yOut=None,
        expertIdxOut=None,
        rowIdxOut=None,
        *args,
        **kwargs,
    ):
        ori_dtype = x.dtype
        x_f32 = x.to(torch.float32)

        num_expert = x_f32.shape[-1]
        softmax = _softmax_torch(x_f32, -1)

        indices = torch.argsort(x_f32, dim=-1, descending=True, stable=True)
        indices = indices[..., :k].to(torch.int64)
        y = torch.gather(softmax, -1, indices)

        if finishedOptional is not None:
            finished = finishedOptional.to(torch.bool)
            if finished.dim() == 1:
                finished = finished.reshape(-1, 1).expand(-1, k)
            indices = torch.where(
                finished, torch.full_like(indices, num_expert), indices
            )

        out_rows = y.shape[0] * y.shape[1]
        row_idx = (
            torch.arange(out_rows, dtype=torch.int32)
            .reshape(y.shape[1], y.shape[0])
            .t()
            .contiguous()
        )

        return [y.to(ori_dtype), indices.to(torch.int32), row_idx]

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }


class MoeGatingTopKSoftmaxTestSpec:
    @staticmethod
    def golden(x, finished, *, k, **kwargs):
        ori_dtype = x.dtype
        x_f32 = x.astype(numpy.float32)

        num_expert = x_f32.shape[-1]
        softmax = _softmax_numpy(x_f32, -1)

        indices = numpy.argsort(-softmax, axis=-1, kind="stable")
        indices = indices[..., :k]
        y = numpy.take_along_axis(softmax, indices, axis=-1)

        if finished is not None:
            finished_2d = finished.reshape(finished.shape[0], 1)
            finished_2d = numpy.tile(finished_2d, (1, k))
            indices = numpy.where(finished_2d, num_expert, indices)

        out_rows = y.shape[0] * y.shape[1]
        row_idx = (
            numpy.arange(out_rows, dtype=numpy.int32)
            .reshape([y.shape[1], y.shape[0]])
            .transpose(1, 0)
        )

        return [y.astype(ori_dtype), indices.astype(numpy.int32), row_idx]

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }


class AclnnMoeGatingTopKV2TestSpec:
    @staticmethod
    def golden(
        x,
        biasOptional,
        inputIdsOptional=None,
        tid2eidOptional=None,
        k=1,
        kGroup=1,
        groupCount=1,
        groupSelectMode=0,
        renorm=0,
        normType=0,
        outFlag=False,
        routedScalingFactor=1.0,
        eps=1e-20,
        yOut=None,
        expertIdxOut=None,
        outOut=None,
        *args,
        **kwargs,
    ):
        ori_dtype = x.dtype
        x = x.to(torch.float32)
        if biasOptional is not None:
            biasOptional = biasOptional.to(torch.float32)

        if normType == 0:
            x = _softmax_torch(x, -1)
        elif normType == 1:
            x = 1 / (1 + numpy.exp(-x.numpy()))
            x = torch.from_numpy(x).to(torch.float32)
        elif normType == 2:
            x = torch.sqrt(torch.nn.functional.softplus(x))

        original_x = x
        if biasOptional is not None:
            x = x + biasOptional

        hashFlag = inputIdsOptional is not None and tid2eidOptional is not None
        if hashFlag:
            indices = tid2eidOptional[inputIdsOptional].to(torch.int64)
        else:
            if groupCount > 1:
                x_reshaped = x.reshape(x.shape[0], groupCount, -1)
                if groupSelectMode == 0:
                    group_x = torch.amax(x_reshaped, dim=-1)
                else:
                    top2 = torch.topk(x_reshaped, 2, dim=-1).values
                    group_x = top2.sum(dim=-1)
                _, group_indices = torch.sort(
                    group_x, dim=-1, descending=True, stable=True
                )
                group_indices = group_indices[:, :kGroup]

                mask = torch.ones((x_reshaped.shape[0], groupCount), dtype=torch.bool)
                mask[torch.arange(x_reshaped.shape[0])[:, None], group_indices] = False
                x = torch.where(mask.unsqueeze(-1), float("-inf"), x_reshaped)
                x = x.reshape(x.shape[0], -1)

            _, indices = torch.sort(x, dim=-1, stable=True, descending=True)
            indices = indices[:, :k].to(torch.int64)

        y = torch.gather(original_x, 1, indices)

        if normType != 0 or renorm != 0:
            y = y / (y.sum(dim=-1, keepdim=True) + eps)
        y = y * routedScalingFactor

        if outFlag:
            out = original_x
        else:
            out = None

        return [y.to(ori_dtype), None, out]

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
    }
