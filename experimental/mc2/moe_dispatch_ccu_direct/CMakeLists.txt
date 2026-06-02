# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0.
# -----------------------------------------------------------------------------------------------------------
# A6 CCU MoeDistributeDispatchCcuSetup 独立验证工程（v21：纯 setup 0.5RTT 通信验证）
#
#   - 只编 dispatch_setup（CCU CcuKernel 子类，跑 CCU 引擎）+ host driver。
#   - 无 AIV teardown、无 <<<>>>，故【纯 host g++ 即可编译】（与 hcomm-test 同形态，不需要 ascendc 工具链）。
#   - 链接 / 运行需 A6：0.5RTT 上层 API（RecordExpectCount / WriteNbWithArriveNotify /
#     WaitallpeersarriveNotify / clearallpeersarriveNotify）由 A6 hcomm 提供；A5 上链接会缺这些符号。
#   - 多进程用 mpirun（见 run.sh）。
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

# setup 静态库：dispatch_setup 是 hcomm CcuKernel 子类（CCU DSL 在 host 侧构造 SQE），普通 g++ 编进静态库。
add_library(moe_dispatch_ccu STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/moe_distribute_dispatch_ccu_setup.cpp
)
target_compile_options(moe_dispatch_ccu PRIVATE
    -O2 -Wall -D_GLIBCXX_USE_CXX11_ABI=0
)
target_include_directories(moe_dispatch_ccu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/include
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/pkg_inc/hcomm/ccu
)

# A6 调用范式（对齐 hcomm-test/ccu_kernel_test + oam-tools/hccl_test）：MPI 多进程 + RankGraph channel + WithStream。
find_package(MPI REQUIRED)

# host driver：MPI 多进程 → 注册并 Launch dispatch_setup（CCU 0.5RTT）→ host 校验 commWindow。
add_executable(test_moe_distribute_dispatch_ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_moe_distribute_dispatch_ccu.cpp
)
target_compile_options(test_moe_distribute_dispatch_ccu PRIVATE
    -O2 -Wall -D_GLIBCXX_USE_CXX11_ABI=0
)
target_include_directories(test_moe_distribute_dispatch_ccu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/include
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/pkg_inc/hcomm/ccu
    ${MPI_CXX_INCLUDE_DIRS}
)
# 链接：setup 静态库 + hcomm（CcuKernel 基类 + 0.5RTT API by A6）+ hccl（comm/channel/thread）+ acl + MPI。
target_link_libraries(test_moe_distribute_dispatch_ccu PRIVATE
    moe_dispatch_ccu
    hcomm
    hccl
    ascendcl
    pthread
    MPI::MPI_CXX
)

install(TARGETS test_moe_distribute_dispatch_ccu
    RUNTIME DESTINATION bin
)
