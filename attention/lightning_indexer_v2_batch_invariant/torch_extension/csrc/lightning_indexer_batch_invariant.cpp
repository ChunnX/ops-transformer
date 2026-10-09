/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer_batch_invariant.cpp
 * \brief
 */

#include <torch/extension.h>
#include "lightning_indexer_common.h"

namespace op_api {
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("lightning_indexer_batch_invariant", &LightningIndexer<true>, "lightning_indexer_batch_invariant");
}
} // namespace op_api
