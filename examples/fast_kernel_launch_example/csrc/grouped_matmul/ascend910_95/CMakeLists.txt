# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

set(GMM_TILING_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../gmm/grouped_matmul/op_host/op_tiling/arch35")
file(GLOB GMM_TILING_SOURCES ${GMM_TILING_DIR}/*.cpp)

# TODO: compile tiling
set(GMM_TILING_SOURCES "")

# TODO: arch
add_sources("--npu-arch=dav-3510 --cce-aicore-input-parameter-size=4096" "${GMM_TILING_SOURCES}")