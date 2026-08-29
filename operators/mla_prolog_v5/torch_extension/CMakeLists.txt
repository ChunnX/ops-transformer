# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# torch_extension 构建配置：pybind 封装 aclnnMlaPrologV5WeightNz（两段式，运行时经
# ASCEND_CUSTOM_OPP_PATH 定位 libcust_opapi.so，无需静态链接）
cmake_minimum_required(VERSION 3.10)
project(mla_prolog_v5_torch_ext)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Torch REQUIRED)
message(STATUS "PyTorch version: ${TORCH_VERSION}")
message(STATUS "PyTorch path: ${TORCH_INSTALL_PREFIX}")

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH environment variable is not set!")
endif()
set(CANN_ROOT $ENV{ASCEND_HOME_PATH})
message(STATUS "CANN_ROOT: ${CANN_ROOT}")

# torch_npu
execute_process(
    COMMAND python3 -c "import torch_npu, os; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_VARIABLE TORCH_NPU_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT TORCH_NPU_ROOT)
    message(FATAL_ERROR "torch_npu not found.")
endif()
set(TORCH_NPU_INCLUDE_DIR "${TORCH_NPU_ROOT}/include")
find_library(TORCH_NPU_LIBRARY torch_npu PATHS ${TORCH_NPU_ROOT}/lib PATH_SUFFIXES lib)
if(NOT TORCH_NPU_LIBRARY)
    message(FATAL_ERROR "torch_npu library not found.")
endif()
message(STATUS "torch_npu: ${TORCH_NPU_LIBRARY}")

add_library(mla_prolog_v5_ext SHARED
    csrc/mla_prolog_v5.cpp
)

target_include_directories(mla_prolog_v5_ext PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_DIR}
    ${CANN_ROOT}/include
    ${CANN_ROOT}/aarch64-linux/include
    ${CANN_ROOT}/aarch64-linux/include/op_common
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
)

target_compile_definitions(mla_prolog_v5_ext PRIVATE TORCH_EXTENSION_NAME=mla_prolog_v5_ext)
target_compile_options(mla_prolog_v5_ext PRIVATE -D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI} -Wno-deprecated-declarations)

target_link_libraries(mla_prolog_v5_ext PRIVATE
    ${TORCH_LIBRARIES}
    ${CANN_ROOT}/lib64/libascendcl.so
    ${CANN_ROOT}/lib64/libopapi.so
    ${CANN_ROOT}/lib64/libnnopbase.so
    ${CANN_ROOT}/lib64/libruntime.so
    ${TORCH_NPU_LIBRARY}
)

get_filename_component(TORCH_NPU_LIB_DIR ${TORCH_NPU_LIBRARY} DIRECTORY)
target_link_options(mla_prolog_v5_ext PRIVATE
    "-Wl,-rpath,${CANN_ROOT}/lib64"
    "-Wl,-rpath,${TORCH_NPU_LIB_DIR}"
)

set_target_properties(mla_prolog_v5_ext PROPERTIES PREFIX "lib" OUTPUT_NAME "mla_prolog_v5_ext")

install(TARGETS mla_prolog_v5_ext DESTINATION lib)
