# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.  Licensed under Apache-2.0.
# -----------------------------------------------------------------------------------------------------------
# dispatch_setup：CCU kernel + 集成驱动。独立 CMake 工程（对齐 quant_all_reduce 的独立构建形态），
# 产出一个 host 可执行 dispatch_min_driver：编入 CCU kernel（普通 C++ + hcomm），
# 调用已装的 teardown aclnn 算子，单 die 两条流靠状态位串行同步。
#
# 注意：本目录【不】走 ops-transformer 注册框架（BUILD_OPEN_PROJECT）——它产的是 host 驱动 exe，
#   不是 aclnn 算子包。故顶层 experimental/mc2 GLOB add_subdirectory 到本目录时，
#   这份 standalone CMake 在注册构建上下文里应被 BUILD_OPEN_PROJECT 守卫跳过（见下）。
if (BUILD_OPEN_PROJECT)
    return()   # 注册构建（build.sh --pkg）上下文里跳过：驱动单独用本目录 cmake 构建（run.sh）。
endif()

cmake_minimum_required(VERSION 3.16)
project(dispatch_min_driver CXX)

if(NOT DEFINED SOC_VERSION)
    set(SOC_VERSION "ascend950pr_9599")   # [T8] a6 目标 SoC 串，确认是否真值
endif()

if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH})
    else()
        set(ASCEND_CANN_PACKAGE_PATH "/home/0617/env/ascend-toolkit/latest")  # a5 CANN 参考路径
    endif()
endif()
message(STATUS "ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}  SOC_VERSION=${SOC_VERSION}")

# ---- MPI（2-die bootstrap：root 取 rootInfo → 2 rank MPI_Bcast 广播；镜像 hcomm-test init_hcclComm）----
#   依赖 MPI dev 包 / module（与 hcomm-test 相同前提）。仅用 C MPI API（MPI_Init/Comm_rank/Bcast/Finalize），
#   故链 MPI::MPI_C 足矣；如 a6 环境 find_package 失败，可 -DMPI_HOME=.. 或加 CMAKE_PREFIX_PATH 指到 MPI。
# MPI（仅 C API）：★ 不用 find_package(MPI)——它靠问 MPI wrapper(mpicc/mpicxx) 拿路径并 try_compile，
#   a6 上 MPICH/OpenMPI 的 wrapper 常自报一个写死的、当前机上不存在的 include（如 /mnt/mpich/include：
#   安装位与挂载位不一致），直接把 find_package 搞挂（Imported target MPI::MPI_CXX includes non-existent path）。
#   改为【直接从 MPI_HOME 定位 mpi.h + libmpi(C 库)】，绕开 wrapper 脏路径，用 MPI_HOME 作唯一真值来源。
#   （本驱动只用 C MPI API；C++ bindings 已由 OMPI_SKIP_MPICXX/MPICH_SKIP_MPICXX 关掉，故不碰 libmpi_cxx。）
find_path(MPI_INC_DIR NAMES mpi.h
    HINTS ${MPI_HOME} $ENV{MPI_HOME}
    PATHS /usr/lib/x86_64-linux-gnu/openmpi /usr/include /usr/local/include
    PATH_SUFFIXES include)
find_library(MPI_C_LIB NAMES mpi
    HINTS ${MPI_HOME} $ENV{MPI_HOME}
    PATHS /usr/lib/x86_64-linux-gnu/openmpi /usr/lib /usr/lib64 /usr/local/lib
    PATH_SUFFIXES lib lib64)
if(NOT MPI_INC_DIR OR NOT MPI_C_LIB)
    message(FATAL_ERROR "MPI(C) 未定位到：mpi.h=[${MPI_INC_DIR}] libmpi=[${MPI_C_LIB}]。"
        " 请 -DMPI_HOME=<mpi 根，其下应有 include/mpi.h 与 lib/libmpi.so>")
endif()
set(MPI_LINK_TARGET ${MPI_C_LIB})
message(STATUS "MPI(C-only): inc=${MPI_INC_DIR}  lib=${MPI_C_LIB}")

# a6-only CCU 路径开关：关(默认)= a5 只编驱动本体（不编 CCU kernel —— 它继承 hcomm::CcuKernel，
#   而 CcuKernel 基类/HcclCcuKernel* 符号 nm 实测不在 a5 libhcomm，属 a6-only、a5 链接无解）；
#   开 = 编入 dispatch_setup_ccu.cpp + 启用 CCU 拉起与段10 LocalCopyNb 真实写，需 a6 libhcomm。
option(DISPATCH_CCU_A6 "Enable a6-only CCU launch + LocalCopyNb write (needs a6 libhcomm)" OFF)

set(DRIVER_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/dispatch_min_driver.cpp)
if(DISPATCH_CCU_A6)
    list(APPEND DRIVER_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/dispatch_setup_ccu.cpp)
endif()
add_executable(dispatch_min_driver ${DRIVER_SRCS})

target_compile_options(dispatch_min_driver PRIVATE
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall
)
target_compile_definitions(dispatch_min_driver PRIVATE SOC_VERSION="${SOC_VERSION}"
    OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)   # ★ 关掉 mpi.h 拖进的已废弃 C++ bindings，只走 C MPI API

# a6 链接：CCU 头(hcomm/ccu)内联用了安全 C 函数 memset_s（属 libc_sec）；ld 默认不传递解析 DT_NEEDED，
#   故 (1) 上面 foreach 已把 c_sec 纳入显式链接，(2) 加 --copy-dt-needed-entries 让 CANN 库的传递符号也能解析
#   （对齐 hcomm-test 基线用法；a5 骨架 CCU 关、不引用 memset_s，故此依赖仅 a6 暴露）。
target_link_options(dispatch_min_driver PRIVATE -Wl,--copy-dt-needed-entries)

if(DISPATCH_CCU_A6)
    target_compile_definitions(dispatch_min_driver PRIVATE DISPATCH_CCU_A6)
endif()

# ---- 自动定位 hcomm/hccl 头根目录（[T6✓] 头已坐实；a5/a6 布局不同用多候选后缀兜底）----
#   实测 a5 路径：<pkg>/x86_64-linux/asc/include/adv_api/hccl/internal/hcomm[/pkg_inc]
find_path(HCOMM_PKG_INC NAMES hcomm/ccu/ccu_kernel.h
    PATHS ${ASCEND_CANN_PACKAGE_PATH}
    PATH_SUFFIXES
        x86_64-linux/asc/include/adv_api/hccl/internal/hcomm/pkg_inc
        asc/include/adv_api/hccl/internal/hcomm/pkg_inc
        include/adv_api/hccl/internal/hcomm/pkg_inc
        pkg_inc
    NO_DEFAULT_PATH)
find_path(HCOMM_INT_INC NAMES hcomm_res_defs.h
    PATHS ${ASCEND_CANN_PACKAGE_PATH}
    PATH_SUFFIXES
        x86_64-linux/asc/include/adv_api/hccl/internal/hcomm
        asc/include/adv_api/hccl/internal/hcomm
        include/adv_api/hccl/internal/hcomm
    NO_DEFAULT_PATH)
# CANN 公共头根（acl/acl.h, hccl/hccl.h, aclnn/aclnn_base.h；实测在 x86_64-linux/include 而非 /include）
find_path(CANN_PUB_INC NAMES acl/acl.h
    PATHS ${ASCEND_CANN_PACKAGE_PATH}
    PATH_SUFFIXES x86_64-linux/include aarch64-linux/include include
    NO_DEFAULT_PATH)
if(NOT HCOMM_PKG_INC OR NOT HCOMM_INT_INC OR NOT CANN_PUB_INC)
    message(WARNING "头根未全部自动定位（PKG=${HCOMM_PKG_INC} INT=${HCOMM_INT_INC} PUB=${CANN_PUB_INC}）；a6 路径可能不同，可 -DHCOMM_PKG_INC=.. -DHCOMM_INT_INC=.. -DCANN_PUB_INC=.. 手动指定。")
endif()

target_include_directories(dispatch_min_driver PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../dispatch_teardown/op_api          # aclnn_dispatch_teardown.h
    ${CANN_PUB_INC}                                                  # acl/acl.h, hccl/hccl.h, aclnn/*, hccl/hccl_types.h
    ${HCOMM_PKG_INC}                                                 # hcomm/ccu/{ccu_kernel,hccl_ccu_res,ccu_assist_pub}.h
    ${HCOMM_PKG_INC}/hcomm/ccu                                       # plain "ccu_kernel.h"（dispatch_setup_ccu.h 用）
    ${HCOMM_INT_INC}                                                 # hccl/hccl_res.h, hcomm_res_defs.h, hcomm_primitives.h
    ${HCOMM_INT_INC}/hccl                                            # plain "hccl_types.h"（ccu_kernel.h 内部 plain include）
    ${MPI_INC_DIR}                                                   # mpi.h（直接从 MPI_HOME 定位，不走 find_package）
)

# ---- 自动定位库（lib64 布局 a5/a6 不同）----
foreach(_lib ascendcl nnopbase hccl hcomm c_sec)
    find_library(LIB_${_lib} ${_lib}
        PATHS ${ASCEND_CANN_PACKAGE_PATH}
        PATH_SUFFIXES lib64 x86_64-linux/lib64 x86_64-linux/devlib lib
        NO_DEFAULT_PATH)
    if(NOT LIB_${_lib})
        message(WARNING "lib${_lib} 未定位到，回退按名链接（依赖系统库路径）。")
        set(LIB_${_lib} ${_lib})
    endif()
endforeach()

# [T9] teardown 自定义算子包对外 opapi 库（build.sh --pkg 装包后才有；CANN 惯例 libcust_opapi.so）。
find_library(LIB_cust_opapi NAMES cust_opapi cust_op_api
    PATHS $ENV{ASCEND_OPP_PATH} ${ASCEND_CANN_PACKAGE_PATH}
    PATH_SUFFIXES vendors/customize/op_api/lib vendors/mc2/op_api/lib op_api/lib lib64
    NO_DEFAULT_PATH)

target_link_libraries(dispatch_min_driver PRIVATE
    ${LIB_ascendcl} ${LIB_nnopbase} ${LIB_hccl} ${LIB_hcomm} ${LIB_c_sec} ${MPI_LINK_TARGET} pthread
)
if(LIB_cust_opapi)
    target_link_libraries(dispatch_min_driver PRIVATE ${LIB_cust_opapi})
else()
    message(WARNING "[T9] cust opapi 库（提供 aclnnDispatchTeardown）未找到：先 build.sh --pkg 装 dispatch_teardown，再重跑 cmake。")
endif()
