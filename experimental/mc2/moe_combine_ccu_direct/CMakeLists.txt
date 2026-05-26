# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0.
# -----------------------------------------------------------------------------------------------------------
# A6 CCU MoeDistributeCombine 算子工程（v16 ★ 目录整理：物理上无 dispatch 文件复制）
#
# 工程定位：
#   - 本目录只放 combine 自己的代码 + 完整链路 driver
#   - dispatch 源 / headers 通过相对路径引用 ../moe_dispatch_ccu_direct/（不物理复制）
#   - host driver test_moe_distribute_combine_ccu 跑完整 dispatch+combine 链路 + 数据闭环校验
#
# 编译路径分裂：
#   - *_setup.cpp 是 hcomm CCU kernel，用 host g++ 编进静态库
#   - *_teardown.cpp 是 ascendc kernel（__global__ __aicore__），需 ascendc 工具链单独编译
#   - test_*.cpp 含 `<<<>>>` 直调语法，也需 ascendc 工具链
#   - A5 纯 host gcc 环境不能整体编译——目标 A6 ascendc 仿真/真机
# -----------------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.16)
project(moe_distribute_combine_ccu)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(ASCEND_CANN_PACKAGE_PATH "/home/0513/env/cann-9.1.0"
    CACHE STRING "ASCEND CANN package installation directory")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" FORCE)
endif()

# v16 ★ 跨目录引用 dispatch 源（避免文件复制）
set(DISPATCH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../moe_dispatch_ccu_direct)

# setup 静态库：dispatch_setup 从 dispatch 目录读 .cpp，combine_setup 在本目录
#   dispatch_setup + combine_setup 都是 CcuKernel 子类，用 host g++ 编进静态库
add_library(moe_distribute_ccu_setups STATIC
    ${DISPATCH_DIR}/moe_distribute_dispatch_ccu_setup.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moe_distribute_combine_ccu_setup.cpp
)
target_compile_options(moe_distribute_ccu_setups PRIVATE
    -O2 -Wall -D_GLIBCXX_USE_CXX11_ABI=0
)
target_include_directories(moe_distribute_ccu_setups PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${DISPATCH_DIR}
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/asc/include/adv_api/hccl/internal/hcomm
)

# host driver：跑完整 dispatch+combine 链路 + 数据闭环校验
add_executable(test_moe_distribute_combine_ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_moe_distribute_combine_ccu.cpp
)
target_compile_options(test_moe_distribute_combine_ccu PRIVATE
    -O2 -Wall -D_GLIBCXX_USE_CXX11_ABI=0
)
target_include_directories(test_moe_distribute_combine_ccu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${DISPATCH_DIR}
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/asc/include/adv_api/hccl/internal/hcomm
)
target_link_libraries(test_moe_distribute_combine_ccu PRIVATE
    moe_distribute_ccu_setups
    hcomm
    hccl
    ascendcl
    pthread
)

install(TARGETS test_moe_distribute_combine_ccu
    RUNTIME DESTINATION bin
)
