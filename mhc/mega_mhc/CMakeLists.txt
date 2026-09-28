# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_host/CMakeLists.txt")
    add_subdirectory(op_host)
endif()

# MegaMhc's generated dynamic wrapper searches the snake-case OpType.
if(BUILD_OPEN_PROJECT AND DEFINED IMPL_INSTALL_DIR)
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/"
            DESTINATION "${IMPL_INSTALL_DIR}/ascendc/mega_mhc/op_kernel"
            FILES_MATCHING PATTERN "*.cpp" PATTERN "*.h")
endif()
