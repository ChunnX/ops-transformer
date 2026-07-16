cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH not set. Please source CANN set_env.sh first.")
endif()

set(ASCEND_HOME_PRE "$ENV{ASCEND_HOME_PATH}")
execute_process(COMMAND uname -m OUTPUT_VARIABLE ARCH_PRE OUTPUT_STRIP_TRAILING_WHITESPACE)
set(BISHENG_PRE "${ASCEND_HOME_PRE}/${ARCH_PRE}-linux/ccec_compiler/bin/bisheng")

if(NOT EXISTS ${BISHENG_PRE})
    message(FATAL_ERROR "BiSheng compiler not found at ${BISHENG_PRE}")
endif()

set(CMAKE_C_COMPILER ${BISHENG_PRE})
set(CMAKE_CXX_COMPILER ${BISHENG_PRE})

project(mc2_direct_launch LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(cmake/ascend.cmake)
include(cmake/python.cmake)
include(cmake/torch.cmake)
include(cmake/torch_npu.cmake)

add_subdirectory(csrc)
