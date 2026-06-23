cmake_minimum_required(VERSION 3.14)
project(qli_v2_metadata_cpu_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define RUN_TEST so that log.h uses printf-based macros
add_definitions(-DRUN_TEST)

# ---- Include paths ----
# stubs/ contains our mock CANN SDK headers (cpu_kernel.h, cpu_context.h, etc.)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/stubs)
# kernel/ contains the 3 kernel files
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/kernel)
# Project root for main.cpp includes
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# ---- Source files ----
set(KERNEL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel/quant_lightning_indexer_v2_metadata_aicpu.cpp
)

set(MAIN_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

# ---- Build executable ----
add_executable(qli_v2_metadata_test ${MAIN_SRC} ${KERNEL_SRC})

target_include_directories(qli_v2_metadata_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/stubs
    ${CMAKE_CURRENT_SOURCE_DIR}/kernel
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Compiler warnings
target_compile_options(qli_v2_metadata_test PRIVATE -Wall -Wextra -Wno-unused-parameter)
