# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

set(_Target ${UTest_NamePrefix}_Utils)
aux_source_directory(src _Sources)

add_library(${_Target} STATIC)
target_sources(${_Target} PRIVATE ${_Sources})
target_include_directories(${_Target}
        PUBLIC
            inc
        PRIVATE
            src
)
target_compile_definitions(${_Target}
        PUBLIC
            LOG_CPP
            PROCESS_LOG
)
target_link_libraries(${_Target}
        PUBLIC
            register
            graph_base
            ascendalog
            ascendcl
            nnopbase
            ascend_hal
            dl
            tikicpulib::${OPS_ADV_UTEST_OPS_TEST_ASCEND_PRODUCT_TYPE}
        PRIVATE
            $<BUILD_INTERFACE:intf_pub_utest>
            $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:alog_headers>>
            $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:dlog_headers>>
)
target_compile_options(${_Target}
        PUBLIC
            $<$<COMPILE_LANGUAGE:CXX>:-std=gnu++1z>
)
