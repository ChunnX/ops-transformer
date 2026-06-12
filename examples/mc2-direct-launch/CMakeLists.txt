cmake_minimum_required(VERSION 3.16)

# 在 project() 之前设置 bisheng 编译器
# 编译前必须设置环境变量 ASCEND_HOME_PATH（指向 CANN 安装目录）
if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH environment variable is not set. Please source CANN set_env.sh first.")
endif()
set(ASCEND_HOME_PRE "$ENV{ASCEND_HOME_PATH}")
execute_process(COMMAND uname -m OUTPUT_VARIABLE ARCH_PRE OUTPUT_STRIP_TRAILING_WHITESPACE)
set(BISHENG_PRE "${ASCEND_HOME_PRE}/${ARCH_PRE}-linux/ccec_compiler/bin/bisheng")
set(CMAKE_C_COMPILER ${BISHENG_PRE})
set(CMAKE_CXX_COMPILER ${BISHENG_PRE})

set(PKG_NAME mc2_direct_launch)
project(${PKG_NAME} LANGUAGES CXX VERSION 1.0.0)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(cmake/ascend.cmake)
include(cmake/python.cmake)
include(cmake/torch.cmake)
include(cmake/torch_npu.cmake)
include(cmake/func.cmake)

set(NPU_ARCH "ascend950" CACHE STRING "NPU architecture")
set(EXTENSION_MODULE_NAME "mc2_direct_launch" CACHE STRING "Extension module name")

if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME "$ENV{ASCEND_HOME_PATH}" CACHE PATH "Ascend path" FORCE)
else()
    set(ASCEND_HOME "/usr/local/Ascend/latest" CACHE PATH "Ascend path")
endif()

# 自动识别架构
execute_process(COMMAND uname -m OUTPUT_VARIABLE HOST_ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
set(SYSTEM_PREFIX "${HOST_ARCH}-linux")

set(INCLUDE_DIRECTORIES
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_PATH}
    ${ASCEND_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/op_kernel
)

add_subdirectory(csrc)
