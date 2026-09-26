# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# -----------------------------------------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_host/CMakeLists.txt")
    add_subdirectory(op_host)
endif()

# AiInfraMegaMhc's generated dynamic wrapper searches the snake-case OpType.
if(BUILD_OPEN_PROJECT AND DEFINED IMPL_INSTALL_DIR)
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/"
            DESTINATION "${IMPL_INSTALL_DIR}/ascendc/ai_infra_mega_mhc/op_kernel"
            FILES_MATCHING PATTERN "*.cpp" PATTERN "*.h")
endif()
