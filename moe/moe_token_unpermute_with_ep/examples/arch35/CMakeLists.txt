cmake_minimum_required(VERSION 3.14)
project(moe_token_unpermute_with_ep_geir_validation LANGUAGES CXX)

if("$ENV{ASCEND_HOME_PATH}" STREQUAL "")
    message(FATAL_ERROR "ASCEND_HOME_PATH is required")
endif()
if("$ENV{ASCEND_CUSTOM_OPP_PATH}" STREQUAL "")
    message(FATAL_ERROR "ASCEND_CUSTOM_OPP_PATH is required")
endif()

set(CANN_ROOT "$ENV{ASCEND_HOME_PATH}")
set(CANN_ARCH_ROOT "${CANN_ROOT}/x86_64-linux")
string(REPLACE ":" ";" CUSTOM_OPP_PATHS "$ENV{ASCEND_CUSTOM_OPP_PATH}")
list(GET CUSTOM_OPP_PATHS 0 CUSTOM_OPP_ROOT)

find_library(GRAPH_LIBRARY graph REQUIRED PATHS "${CANN_ARCH_ROOT}/lib64" NO_DEFAULT_PATH)
find_library(GE_RUNNER_LIBRARY ge_runner REQUIRED PATHS "${CANN_ARCH_ROOT}/lib64" NO_DEFAULT_PATH)
find_library(GRAPH_BASE_LIBRARY graph_base REQUIRED PATHS "${CANN_ARCH_ROOT}/lib64" NO_DEFAULT_PATH)
find_library(GE_COMPILER_LIBRARY ge_compiler REQUIRED PATHS "${CANN_ARCH_ROOT}/lib64" NO_DEFAULT_PATH)

set(GEIR_INCLUDE_DIRS
    "${CANN_ARCH_ROOT}/include"
    "${CANN_ARCH_ROOT}/include/ge"
    "${CANN_ARCH_ROOT}/include/graph"
    "${CANN_ROOT}/opp/built-in/op_graph/inc"
    "${CUSTOM_OPP_ROOT}/op_proto/inc"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
set(GEIR_LIBRARIES
    "${GRAPH_LIBRARY}"
    "${GE_RUNNER_LIBRARY}"
    "${GRAPH_BASE_LIBRARY}"
    "${GE_COMPILER_LIBRARY}"
)

foreach(target IN ITEMS
        test_geir_moe_token_unpermute_with_ep
        test_geir_moe_token_unpermute_with_ep_dynamic)
    add_executable(${target} "${target}.cpp")
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_include_directories(${target} PRIVATE ${GEIR_INCLUDE_DIRS})
    target_link_libraries(${target} PRIVATE ${GEIR_LIBRARIES})
endforeach()
