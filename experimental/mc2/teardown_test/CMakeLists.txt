# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.  Licensed under Apache-2.0.
# -----------------------------------------------------------------------------------------------------------
# teardown_test：dispatch_teardown / combine_teardown 的单卡 aclnn 测试驱动（standalone 工程）。
#   产 host exe，调用已装的 cust aclnn 算子，纯单卡无 MPI/通信。
#   注册构建(build.sh --pkg)上下文里 BUILD_OPEN_PROJECT 守卫跳过，避免被顶层 GLOB 误纳入包构建。
if (BUILD_OPEN_PROJECT)
    return()
endif()

cmake_minimum_required(VERSION 3.16)
project(teardown_test CXX)

if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH})
    else()
        set(ASCEND_CANN_PACKAGE_PATH "/home/0625/env/ascend-toolkit/latest")
    endif()
endif()
message(STATUS "ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}")

# CANN 公共头（acl/acl.h, aclnn/aclnn_base.h）
find_path(CANN_PUB_INC NAMES acl/acl.h
    PATHS ${ASCEND_CANN_PACKAGE_PATH}
    PATH_SUFFIXES x86_64-linux/include aarch64-linux/include include
    NO_DEFAULT_PATH)
if(NOT CANN_PUB_INC)
    message(FATAL_ERROR "CANN_PUB_INC(acl/acl.h) 未定位到，可 -DCANN_PUB_INC=..")
endif()

# 库
foreach(_lib ascendcl nnopbase)
    find_library(LIB_${_lib} ${_lib}
        PATHS ${ASCEND_CANN_PACKAGE_PATH}
        PATH_SUFFIXES lib64 x86_64-linux/lib64 x86_64-linux/devlib lib
        NO_DEFAULT_PATH)
    if(NOT LIB_${_lib})
        message(WARNING "lib${_lib} 未定位到，回退按名链接。")
        set(LIB_${_lib} ${_lib})
    endif()
endforeach()

# cust 算子对外 opapi 库（build.sh --pkg 装包后才有）
find_library(LIB_cust_opapi NAMES cust_opapi cust_op_api
    PATHS $ENV{ASCEND_OPP_PATH} ${ASCEND_CANN_PACKAGE_PATH}
    PATH_SUFFIXES vendors/customize/op_api/lib vendors/custom_transformer/op_api/lib vendors/mc2/op_api/lib op_api/lib lib64
    NO_DEFAULT_PATH)
if(NOT LIB_cust_opapi)
    message(WARNING "cust_opapi(提供 aclnnDispatchTeardown/aclnnCombineTeardown) 未找到：先 build.sh --pkg 装算子，再 cmake。")
endif()

function(add_teardown_test exe src api_dir)
    add_executable(${exe} ${src})
    target_compile_options(${exe} PRIVATE -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall)
    target_include_directories(${exe} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/${api_dir}
        ${CANN_PUB_INC})
    target_link_libraries(${exe} PRIVATE ${LIB_ascendcl} ${LIB_nnopbase} pthread)
    if(LIB_cust_opapi)
        target_link_libraries(${exe} PRIVATE ${LIB_cust_opapi})
    endif()
    target_link_options(${exe} PRIVATE -Wl,--copy-dt-needed-entries)
endfunction()

add_teardown_test(test_dispatch_teardown ${CMAKE_CURRENT_SOURCE_DIR}/test_dispatch_teardown.cpp ../dispatch_teardown/op_api)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/test_combine_teardown.cpp)
    add_teardown_test(test_combine_teardown ${CMAKE_CURRENT_SOURCE_DIR}/test_combine_teardown.cpp ../combine_teardown/op_api)
endif()
