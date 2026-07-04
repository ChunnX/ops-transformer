# ==============================================================================================
#  chain_e2e/CMakeLists.txt —— a6 端到端 4 算子链驱动（standalone host exe，非 aclnn 算子包）。
#  产物：chain_e2e_driver。CHAIN_CCU_A6=OFF(默认)=a5 shell（不编 CCU kernel、只验驱动本体编译/链接）；
#         CHAIN_CCU_A6=ON=a6 全功能（编入 dispatch_setup_ccu.cpp + combine_setup_ccu.cpp + CCU 真跨卡写）。
#  裁剪自 dispatch_setup/CMakeLists.txt，扩成：两个 CCU kernel 源 + 两个 teardown op_api include。
# ==============================================================================================
if (BUILD_OPEN_PROJECT)
    return()   # 注册构建(build.sh --pkg)上下文跳过：本工程产 host exe、非 aclnn 包
endif()

cmake_minimum_required(VERSION 3.16)
project(chain_e2e_driver CXX)

if(NOT DEFINED SOC_VERSION)
    set(SOC_VERSION "ascend950pr_9599")    # [T8] a6 待核实真值
endif()
if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH})
    else()
        set(ASCEND_CANN_PACKAGE_PATH "/home/0617/env/ascend-toolkit/latest")
    endif()
endif()

# ---- MPI 直定位（不用 find_package，绕 wrapper 脏路径）----
find_path(MPI_INC_DIR NAMES mpi.h HINTS ${MPI_HOME} $ENV{MPI_HOME}
    PATHS /usr/lib/x86_64-linux-gnu/openmpi /usr/include /usr/local/include PATH_SUFFIXES include)
find_library(MPI_C_LIB NAMES mpi HINTS ${MPI_HOME} $ENV{MPI_HOME}
    PATHS /usr/lib/x86_64-linux-gnu/openmpi /usr/lib /usr/lib64 /usr/local/lib PATH_SUFFIXES lib lib64)
if(NOT MPI_INC_DIR OR NOT MPI_C_LIB)
    message(FATAL_ERROR "MPI 未找到：请 -DMPI_HOME=<根，其下 include/mpi.h + lib/libmpi.so>")
endif()
set(MPI_LINK_TARGET ${MPI_C_LIB})

# ---- CCU 门控：默认 OFF(a5 shell)。ON 时编入两个 CCU kernel 源 ----
option(CHAIN_CCU_A6 "Enable a6-only CCU kernels + real cross-card write" OFF)
set(DRIVER_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/chain_e2e_driver.cpp)
if(CHAIN_CCU_A6)
    list(APPEND DRIVER_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/../dispatch_setup/dispatch_setup_ccu.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../combine_setup/combine_setup_ccu.cpp)
endif()
add_executable(chain_e2e_driver ${DRIVER_SRCS})

target_compile_options(chain_e2e_driver PRIVATE -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall)
target_compile_definitions(chain_e2e_driver PRIVATE SOC_VERSION="${SOC_VERSION}" OMPI_SKIP_MPICXX MPICH_SKIP_MPICXX)
target_link_options(chain_e2e_driver PRIVATE -Wl,--copy-dt-needed-entries -Wl,--allow-shlib-undefined)
# --allow-shlib-undefined：CANN 库的传递 .so 依赖（libhccl_alg/libruntime/libmmpa…）运行时经 LD_LIBRARY_PATH
#   解析，链接期不强求（a5 实测：加此 flag 后 shell 整链接零 undefined、产出可执行 40KB）。
if(CHAIN_CCU_A6)
    # ★根因修复：链同时编入 dispatch_setup_ccu.cpp(#ifdef DISPATCH_CCU_A6) + combine_setup_ccu.cpp(#ifdef
    #   COMBINE_CCU_A6)，必须【一并定义这两个宏】，否则两个 kernel 的真 Algorithm 被 #ifdef 掉、只剩 a5 桩
    #   (return HCCL_SUCCESS 不建微码) → 驱动 launch 空 kernel → HcclCcuKernelLaunch ret=4(HCCL_E_INTERNAL)。
    #   旧版只传 CHAIN_CCU_A6，漏了这俩——正是 ret=4 根因。
    target_compile_definitions(chain_e2e_driver PRIVATE CHAIN_CCU_A6 DISPATCH_CCU_A6 COMBINE_CCU_A6)
endif()

# ---- CANN / hcomm 头根自动定位（多候选 PATH_SUFFIXES 兜 a5/a6 布局）----
find_path(HCOMM_PKG_INC NAMES hcomm/ccu/ccu_kernel.h PATHS ${ASCEND_CANN_PACKAGE_PATH} PATH_SUFFIXES
    x86_64-linux/asc/include/adv_api/hccl/internal/hcomm/pkg_inc asc/include/adv_api/hccl/internal/hcomm/pkg_inc
    include/adv_api/hccl/internal/hcomm/pkg_inc pkg_inc NO_DEFAULT_PATH)
find_path(HCOMM_INT_INC NAMES hcomm_res_defs.h PATHS ${ASCEND_CANN_PACKAGE_PATH} PATH_SUFFIXES
    x86_64-linux/asc/include/adv_api/hccl/internal/hcomm asc/include/adv_api/hccl/internal/hcomm
    include/adv_api/hccl/internal/hcomm NO_DEFAULT_PATH)
find_path(CANN_PUB_INC NAMES acl/acl.h PATHS ${ASCEND_CANN_PACKAGE_PATH} PATH_SUFFIXES
    x86_64-linux/include aarch64-linux/include include NO_DEFAULT_PATH)

# ★ 头根未定位守卫（照父工程 dispatch_setup/CMakeLists.txt；a6 布局不同 [T8] 时给明确根因，
#   否则 find_path 返回 <VAR>-NOTFOUND → 拼成无效 -I → CCU 段报 'hcomm_res_defs.h: No such file' 无根因提示）。
if(NOT CANN_PUB_INC)
    message(FATAL_ERROR "CANN 公共头(acl/acl.h)未定位：-DCANN_PUB_INC=<其下含 acl/acl.h 的目录> 或核对 ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}")
endif()
if(CHAIN_CCU_A6 AND (NOT HCOMM_PKG_INC OR NOT HCOMM_INT_INC))
    message(FATAL_ERROR "CHAIN_CCU_A6=ON 但 HCOMM 头根未全部定位（PKG=${HCOMM_PKG_INC} INT=${HCOMM_INT_INC}）——a6 布局若不同，用 -DHCOMM_PKG_INC=<含 hcomm/ccu/ccu_kernel.h> -DHCOMM_INT_INC=<含 hcomm_res_defs.h> 手动指定")
endif()

target_include_directories(chain_e2e_driver PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../dispatch_setup          # dispatch_setup_ccu.h + _api.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../combine_setup           # combine_setup_ccu.h + _api.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/inc              # setup_host_precompute.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../dispatch_teardown/op_api # aclnn_dispatch_teardown.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../combine_teardown/op_api  # aclnn_combine_teardown.h
    ${CANN_PUB_INC} ${HCOMM_PKG_INC} ${HCOMM_PKG_INC}/hcomm/ccu ${HCOMM_INT_INC} ${HCOMM_INT_INC}/hccl
    ${MPI_INC_DIR})

# ---- CANN 库 foreach（找不到回退按名链接）----
foreach(_lib ascendcl nnopbase hccl hcomm c_sec)
    find_library(LIB_${_lib} ${_lib} PATHS ${ASCEND_CANN_PACKAGE_PATH}
        PATH_SUFFIXES lib64 x86_64-linux/lib64 x86_64-linux/devlib lib NO_DEFAULT_PATH)
    if(NOT LIB_${_lib})
        set(LIB_${_lib} ${_lib})
    endif()
endforeach()

# ---- teardown 算子包对外库（同时提供 aclnnDispatchTeardown + aclnnCombineTeardown）----
find_library(LIB_cust_opapi NAMES cust_opapi cust_op_api PATHS $ENV{ASCEND_OPP_PATH} ${ASCEND_CANN_PACKAGE_PATH}
    PATH_SUFFIXES vendors/customize/op_api/lib vendors/mc2/op_api/lib vendors/custom_transformer/op_api/lib op_api/lib lib64 NO_DEFAULT_PATH)

target_link_libraries(chain_e2e_driver PRIVATE
    ${LIB_ascendcl} ${LIB_nnopbase} ${LIB_hccl} ${LIB_hcomm} ${LIB_c_sec} ${MPI_LINK_TARGET} pthread)
if(LIB_cust_opapi)
    target_link_libraries(chain_e2e_driver PRIVATE ${LIB_cust_opapi})
else()
    message(WARNING "cust_opapi 未找到（aclnnDispatchTeardown/aclnnCombineTeardown）——先 build.sh --pkg --experimental 装两个 teardown 再重跑 cmake")
endif()
