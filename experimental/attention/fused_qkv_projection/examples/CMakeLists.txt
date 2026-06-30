cmake_minimum_required(VERSION 3.14)
project(ACLNN_FUSED_QKV_PROJECTION_EXAMPLE)

add_compile_options(-std=c++11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "./bin")

set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

if(NOT DEFINED ENV{LD_LIBRARY_PATH})
    message(FATAL_ERROR "LD_LIBRARY_PATH environment variable is not set")
endif()
set(LD_LIB_PATH "$ENV{LD_LIBRARY_PATH}")

string(REPLACE ":" ";" LD_LIB_LIST "${LD_LIB_PATH}")
set(TARGET_PATH "")
foreach(path ${LD_LIB_LIST})
    if(path MATCHES "/vendors/")
        set(TARGET_PATH "${path}")
        break()
    endif()
endforeach()
if(NOT TARGET_PATH)
    message(FATAL_ERROR "LD_LIBRARY_PATH not found /vendors/")
endif()
if(TARGET_PATH MATCHES "/vendors/([^/]+)")
    set(TARGET_SUBDIR "${CMAKE_MATCH_1}")
else()
    message(FATAL_ERROR "subdir not found in ${TARGET_PATH}")
endif()

add_executable(test_aclnn_fused_qkv_projection
    test_aclnn_fused_qkv_projection.cpp)

if(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
    set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
else()
    set(ASCEND_PATH "/ddhome/qkv_xpb/Ascend/cann-9.1.0-beta.1")
endif()

set(INCLUDE_BASE_DIR "${ASCEND_PATH}/include")
include_directories(
    ${INCLUDE_BASE_DIR}
    ${ASCEND_PATH}/opp/vendors/${TARGET_SUBDIR}/op_api/include
)

target_link_libraries(test_aclnn_fused_qkv_projection PRIVATE
    ${ASCEND_PATH}/lib64/libascendcl.so
    ${ASCEND_PATH}/lib64/libnnopbase.so
    ${ASCEND_PATH}/opp/vendors/${TARGET_SUBDIR}/op_api/lib/libcust_opapi.so
)

install(TARGETS test_aclnn_fused_qkv_projection DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
