/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file npu_ops_def.cpp
 * \brief
 */

#include <Python.h>
#include <torch/all.h>

extern "C" {
PyObject* PyInit__C(void)
{
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT, "_C", NULL, -1, NULL,
    };
    return PyModule_Create(&module_def);
}
}

namespace ascend_ops {
TORCH_LIBRARY(ascend_ops, m)
{
    m.def(R"(flash_attn_simplified_example(Tensor q,
             Tensor k,
             Tensor v,
             float softmaxScale = 0) -> Tensor)");
    m.def(R"(flash_attn_example(Tensor q,
             Tensor k,
             Tensor v,
             Tensor attnMask,
             float softmaxScale = 0,
             bool isCausal = True) -> Tensor)");
}
}