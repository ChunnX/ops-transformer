# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# makeself_custom.cmake - 自定义 makeself 打包脚本

set(MAKESELF_NAME "makeself")
set(MAKESELF_PATH ${CANN_3RD_LIB_PATH}/makeself)
set(MAKESELF_VERSION_PKG makeself-release-2.5.0-patch1.tar.gz)

message(STATUS "[DEBUGGING] Inside makeself_custom.cmake, CANN_3RD_LIB_PATH: ${CANN_3RD_LIB_PATH}")


if (POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# 默认配置的makeself还是不存在则下载
if (NOT EXISTS "${MAKESELF_PATH}/makeself-header.sh" OR NOT EXISTS "${MAKESELF_PATH}/makeself.sh")
    # 优先查找本地压缩包
    if(EXISTS "${CANN_3RD_LIB_PATH}/pkg/${MAKESELF_VERSION_PKG}")
        set(REQ_URL "file://${CANN_3RD_LIB_PATH}/pkg/${MAKESELF_VERSION_PKG}")
        message(STATUS "[ThirdPartyLib][makeself] found in ${REQ_URL}.")
    elseif(EXISTS "${CANN_3RD_LIB_PATH}/${MAKESELF_VERSION_PKG}")
        set(REQ_URL "file://${CANN_3RD_LIB_PATH}/${MAKESELF_VERSION_PKG}")
        message(STATUS "[ThirdPartyLib][makeself] found in ${REQ_URL}.")
    else()
        set(REQ_URL "https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz")
        message(STATUS "[ThirdPartyLib][makeself] ${REQ_URL} not found, need download.")
    endif()

    include(FetchContent)
    FetchContent_Declare(
        ${MAKESELF_NAME}
        URL ${REQ_URL}
        URL_HASH SHA256=bfa730a5763cdb267904a130e02b2e48e464986909c0733ff1c96495f620369a
        SOURCE_DIR "${MAKESELF_PATH}"  # 直接解压到此目录
    )
    FetchContent_MakeAvailable(${MAKESELF_NAME})
    execute_process(
        COMMAND chmod 700 "${CMAKE_BINARY_DIR}/makeself/makeself.sh"
        COMMAND chmod 700 "${CMAKE_BINARY_DIR}/makeself/makeself-header.sh"
        -E env
        CMAKE_TLS_VERIFY=0
        RESULT_VARIABLE CHMOD_RESULT
        ERROR_VARIABLE CHMOD_ERROR
    )
endif()

# 设置 makeself 路径
# set(MAKESELF_EXE ${CPACK_MAKESELF_PATH}/makeself.sh)
# set(MAKESELF_HEADER_EXE ${CPACK_MAKESELF_PATH}/makeself-header.sh)

set(MAKESELF_EXE ${CMAKE_BINARY_DIR}/makeself/makeself.sh)
set(MAKESELF_HEADER_EXE ${CMAKE_BINARY_DIR}/makeself/makeself-header.sh)
message(STATUS "[DEBUGGING] Inside makeself_custom.cmake, CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")

message(STATUS "[DEBUGGING] Inside makeself_custom.cmake, CPACK_MAKESELF_PATH: ${CPACK_MAKESELF_PATH}")
message(STATUS "[DEBUGGING] Inside makeself_custom.cmake, CPACK_EXTERNAL_BUILT_PACKAGES: ${CPACK_EXTERNAL_BUILT_PACKAGES}")
message(STATUS "[DEBUGGING] Inside makeself_custom.cmake, CPACK_TEMPORARY_DIRECTORY: ${CPACK_TEMPORARY_DIRECTORY}")
message(STATUS "[DEBUGGING] Inside makeself_custom.cmake, CPACK_INSTALL_PREFIX: ${CPACK_INSTALL_PREFIX}")
message(STATUS "[DEBUGGING] Inside makeself_custom.cmake, CPACK_PACKAGE_FILE_NAME: ${CPACK_PACKAGE_FILE_NAME}")

# if (NOT EXISTS MAKESELF_EXE OR NOT EXISTS MAKESELF_EXE)
#     include(cmake/third_party/makeself-fetch.cmake)
# endif()

execute_process(COMMAND bash ${MAKESELF_EXE}
                        --header ${MAKESELF_HEADER_EXE}
                        --help-header ./help.info --tar-format posix
                        --gzip --complevel 4 --nomd5 --sha256
                        ./ ${CPACK_PACKAGE_FILE_NAME} "version:1.0" ./install.sh
                WORKING_DIRECTORY ${CPACK_TEMPORARY_DIRECTORY}
                RESULT_VARIABLE EXEC_RESULT
                ERROR_VARIABLE  EXEC_ERROR
)

if (NOT "${EXEC_RESULT}x" STREQUAL "0x")
  message(FATAL_ERROR "CPack Command error: ${EXEC_RESULT}\n${EXEC_ERROR}")
endif()

execute_process(COMMAND cp ${CPACK_EXTERNAL_BUILT_PACKAGES} ${CPACK_PACKAGE_DIRECTORY}/
        COMMAND echo "Copy ${CPACK_EXTERNAL_BUILT_PACKAGES} to ${CPACK_PACKAGE_DIRECTORY}/"
        WORKING_DIRECTORY ${CPACK_TEMPORARY_DIRECTORY}
    )

if (NOT "${CPACK_PACKAGE_DIRECTORY}x" STREQUAL "${CPACK_INSTALL_PREFIX}x")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CPACK_INSTALL_PREFIX}
        WORKING_DIRECTORY ${CPACK_TEMPORARY_DIRECTORY}
    )

    execute_process(
        COMMAND cp ${CPACK_EXTERNAL_BUILT_PACKAGES} ${CPACK_INSTALL_PREFIX}/
        COMMAND echo "Copy ${CPACK_EXTERNAL_BUILT_PACKAGES} to ${CPACK_INSTALL_PREFIX}/"
        WORKING_DIRECTORY ${CPACK_TEMPORARY_DIRECTORY}
    )
endif()


