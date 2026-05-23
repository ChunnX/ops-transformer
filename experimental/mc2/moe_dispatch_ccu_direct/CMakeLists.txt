# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0.
# -----------------------------------------------------------------------------------------------------------
# A6 CCU MoeDistributeDispatchCcu 算子工程（setup + teardown）
#   - device 层走 hcomm::CcuKernel + HcclCcuKernelRegister/Launch，跑在 CCU 引擎
#   - 0.5RTT 上层 API (RecordExpectCount / WriteNbWithArriveNotify /
#       WaitAllPeersArriveNotify / ClearAllPeersArriveNotify) 当前 hcomm 仓未实现，
#       链接期由 SE 提供实现后即可解析
#   - A5 远程跑不动 A6 0.5RTT，本工程目标为 A6 真机/仿真
# -----------------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.16)
project(moe_distribute_dispatch_ccu)

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

# 两个 CCU kernel 子类编译为 host 端目标（CCU kernel 不是 ascendc kernel，跑在 CCU 引擎、
# 由 hcomm 运行时调度，不需要 ascendc 编译器）
add_library(moe_dispatch_ccu STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/moe_distribute_dispatch_ccu_setup.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moe_distribute_dispatch_ccu_teardown.cpp
)
target_compile_options(moe_dispatch_ccu PRIVATE
    -O2 -Wall -D_GLIBCXX_USE_CXX11_ABI=0
)
target_include_directories(moe_dispatch_ccu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/asc/include/adv_api/hccl/internal/hcomm
)

# host 驱动可执行
add_executable(test_moe_distribute_dispatch_ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_moe_distribute_dispatch_ccu.cpp
)
target_compile_options(test_moe_distribute_dispatch_ccu PRIVATE
    -O2 -Wall -D_GLIBCXX_USE_CXX11_ABI=0
)
target_include_directories(test_moe_distribute_dispatch_ccu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/asc/include/adv_api/hccl/internal/hcomm
)

# 链接：moe_dispatch_ccu 静态库 + hcomm（含 CcuKernel 基类、HcclCcuKernel*、0.5RTT API 实现 by SE）
# + hccl（HcclComm*, HcclChannel*, ThreadHandle 等）+ acl runtime
target_link_libraries(test_moe_distribute_dispatch_ccu PRIVATE
    moe_dispatch_ccu
    hcomm
    hccl
    ascendcl
    pthread
)

install(TARGETS test_moe_distribute_dispatch_ccu
    RUNTIME DESTINATION bin
)
