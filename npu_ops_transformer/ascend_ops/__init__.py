#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import glob
import os
from torch.utils.cpp_extension import load
import torch_npu

PACKAGE_DIR = os.path.dirname(__file__)
_cann_path = os.environ["ASCEND_HOME_PATH"]
_torch_npu_path = os.path.dirname(os.path.abspath(torch_npu.__file__))


def get_include_paths():
    paths = [
        os.path.join(_torch_npu_path, 'include'),
        os.path.join(_torch_npu_path, 'include/third_party/hccl/inc'),
        os.path.join(_torch_npu_path, 'include/third_party/acl/inc'),
        os.path.join(_cann_path, 'include'),
        os.path.join(PACKAGE_DIR, 'ops/inc'),
    ]
    return paths


def get_absolute_paths(paths):
    return [os.path.join(PACKAGE_DIR, path) for path in paths]


def get_cxx_args():
    args = ['-fstack-protector-all', '-Wl,-z,relro,-z,now,-z,noexecstack', '-fPIC', '-pie',
            '-s', '-fvisibility=hidden', '-D_FORTIFY_SOURCE=2', '-O3', '-march=native', '-w']
    return args


def get_ld_flags():
    flags = [
        '-L' + os.path.join(_cann_path, 'lib64'), '-lascendcl',
        '-L' + os.path.join(_torch_npu_path, 'lib'), '-ltorch_npu'
    ]
    return flags


def load_extension():
    cpp_files = glob.glob(os.path.join(PACKAGE_DIR, "**", "*.cpp"), recursive=True)
    print("cpp_files:", cpp_files)
    include_paths = get_absolute_paths(get_include_paths())
    cflags = get_cxx_args()
    ldflags = get_ld_flags()

    return load(
        name="ascend_ops",
        sources=cpp_files,
        extra_include_paths=include_paths,
        extra_cflags=cflags,
        extra_ldflags=ldflags,
        verbose=True
    )


ext_module = load_extension()
