# SageAttention2 公共依赖解析（op_kernel / op_host 两个 CMakeLists 共享，唯一定义点）
#
# 官方 CATLASS（cann/catlass 仓）：sageattn_fp8/catlass/sage_tile.hpp 无条件
# include <catlass/catlass.hpp> 并从官方常量（BYTE_PER_C0 / C0_NUM_PER_FRACTAL /
# BYTE_PER_FRACTAL）派生分形粒度，因此本依赖为硬性前置，缺失即 fatal（与仓内
# CATLASS 样例工程的 CATLASS_ROOT 约定一致）。
#
# 解析链：CATLASS_ROOT（CMake 变量 → 环境变量）→ <repo>/../catlass → <repo>/third_party/catlass
# 产出变量（调用方直接消费，勿自行重解析）：
#   SAGE_ATTN_CATLASS_INCLUDE_DIR  官方 CATLASS include 目录

if(NOT DEFINED CATLASS_ROOT)
    set(CATLASS_ROOT "$ENV{CATLASS_ROOT}")
endif()

# CMAKE_CURRENT_LIST_DIR = <算子目录>/cmake；仓根为其上三级
get_filename_component(SAGE_ATTN_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

if(NOT CATLASS_ROOT)
    if(EXISTS "${SAGE_ATTN_REPO_ROOT}/../catlass/include/catlass/catlass.hpp")
        set(CATLASS_ROOT "${SAGE_ATTN_REPO_ROOT}/../catlass")
    elseif(EXISTS "${SAGE_ATTN_REPO_ROOT}/third_party/catlass/include/catlass/catlass.hpp")
        set(CATLASS_ROOT "${SAGE_ATTN_REPO_ROOT}/third_party/catlass")
    endif()
endif()

if(NOT CATLASS_ROOT OR NOT EXISTS "${CATLASS_ROOT}/include/catlass/catlass.hpp")
    message(FATAL_ERROR
        "SageAttention2: set CATLASS_ROOT to an official cann/catlass checkout "
        "with Ascend950 support (checked: \$CATLASS_ROOT env, "
        "${SAGE_ATTN_REPO_ROOT}/../catlass, ${SAGE_ATTN_REPO_ROOT}/third_party/catlass)")
endif()

set(SAGE_ATTN_CATLASS_INCLUDE_DIR "${CATLASS_ROOT}/include")
