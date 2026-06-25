#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import os
import numpy as np
import torch


def save_tensor_to_txt(tensor, filepath):
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    arr = tensor.detach().cpu().float().numpy().flatten()
    shape_str = "x".join(str(s) for s in tensor.shape)
    with open(filepath, "w") as fh:
        fh.write(f"# shape: {shape_str}  total: {arr.size}\n")
        for v in arr:
            fh.write(f"{v:.8f}\n")


def load_tensor_from_txt(filepath, target_dtype=torch.float32, target_device='cpu'):
    with open(filepath, "r") as fh:
        header = fh.readline().strip()
        if not header.startswith("# shape:"):
            raise ValueError(f"文件 {filepath} 格式不正确，缺少 shape 注释。")
        shape_str = header.split()[2]
        shape = tuple(int(s) for s in shape_str.split('x'))
        arr = np.loadtxt(fh, dtype=np.float32)
    expected = 1
    for s in shape:
        expected *= s
    if arr.size != expected:
        raise ValueError(f"形状 {shape} 与数值数 {arr.size} 不匹配。")
    tensor = torch.from_numpy(arr).reshape(shape).to(dtype=target_dtype, device=target_device)
    return tensor
