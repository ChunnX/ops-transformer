#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
#
# 对比数据生成 plugin（golden）: aclnnGridSampler3D
# 适配框架: TTK 测试框架（TestSpec 规范, aclnn 流程）
# 规格来源: inputs/aclnnGridSampler3D/docs/aclnnGridSampler3D.md

__spec__ = {"aclnnGridSampler3D": "AclnnGridSampler3DTestSpec"}

import torch
import torch.nn.functional as F

# aclnn 枚举值 → PyTorch grid_sample 字符串参数映射
# 来源: aclnnGridSampler3D.md 参数说明
#   interpolationMode: 0=bilinear(双线性) 1=nearest(最邻近)（3D 不支持 bicubic）
#   paddingMode:       0=zeros 1=border 2=reflection
_INTERP_MODE_MAP = {0: "bilinear", 1: "nearest"}
_PADDING_MODE_MAP = {0: "zeros", 1: "border", 2: "reflection"}


class AclnnGridSampler3DTestSpec:
    """aclnnGridSampler3D 算子测试规范（aclnn 流程）

    输入: input (N, C, D_in, H_in, W_in)，grid (N, D_out, H_out, W_out, 3)
    输出: out   (N, C, D_out, H_out, W_out)
    枚举属性: interpolationMode/paddingMode/alignCorners
    """

    def golden(input, grid, interpolationMode=0, paddingMode=0,
               alignCorners=False, out=None, **kwargs):
        """
        Golden reference implementation for aclnnGridSampler3D.
        根据 grid 归一化坐标对 input 做三维网格采样（支持 bilinear/nearest
        插值与 zeros/border/reflection 填充），直接映射到 PyTorch 原生接口
        torch.nn.functional.grid_sample（5D 输入自动走 3D 采样路径）。
        置信度: ⭐⭐⭐⭐⭐ (使用PyTorch原生接口 F.grid_sample)

        Note:
            - aclnn 反归一化公式与 F.grid_sample 逐字一致:
              align_corners=true:  x' = (grid_x + 1) / 2 * (W_in - 1)
              align_corners=false: x' = ((grid_x + 1) * W_in - 1) / 2
              本算子即 torch_npu F.grid_sample 的 NPU 后端实现，语义与 PyTorch 对齐。
            - 3D 场景不支持 bicubic（F.grid_sample 对 5D 输入同样仅支持
              bilinear/nearest，与规格一致）。
            - 空tensor（N/C/D_out/H_out/W_out 轴为0）由 F.grid_sample 原生支持。
            - 参数按 C header 顺序以位置参数传入，不能用 * 分隔符;
              out 为输出参数（保留占位，框架回填）。
            - dtype防护: 部分 PyTorch 版本 CPU grid_sample 不支持
              float16/bfloat16，失败时升精度到 float32 计算后回转原 dtype
              （计算可用性转换，非精度增强；文档未声明精度要求，默认原生计算）。
        """
        mode = _INTERP_MODE_MAP[int(interpolationMode)]
        padding = _PADDING_MODE_MAP[int(paddingMode)]

        try:
            out = F.grid_sample(input, grid, mode=mode,
                                padding_mode=padding,
                                align_corners=alignCorners)
        except (RuntimeError, TypeError):
            in_dtype = input.dtype
            out = F.grid_sample(input.to(torch.float32),
                                grid.to(torch.float32),
                                mode=mode, padding_mode=padding,
                                align_corners=alignCorners).to(in_dtype)
        return [out]
