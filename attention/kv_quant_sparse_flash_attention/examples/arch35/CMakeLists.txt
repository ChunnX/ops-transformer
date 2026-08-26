cmake_minimum_required(VERSION 3.18.4)
project(QSFA_MXFP4_ACLNN_EXAMPLE LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
    set(ASCEND_PATH "$ENV{ASCEND_HOME_PATH}")
else()
    set(ASCEND_PATH "/usr/local/Ascend/cann")
endif()

if(NOT "$ENV{QSFA_VENDOR_NAME}" STREQUAL "")
    set(QSFA_VENDOR_NAME "$ENV{QSFA_VENDOR_NAME}")
else()
    set(QSFA_VENDOR_NAME "custom")
endif()

if(NOT "$ENV{QSFA_OPAPI_LIBRARY}" STREQUAL "")
    set(QSFA_OPAPI_LIBRARY "$ENV{QSFA_OPAPI_LIBRARY}")
else()
    set(QSFA_OPAPI_LIBRARY "${ASCEND_PATH}/lib64/libopapi_transformer.so")
endif()

add_executable(test_aclnn_kv_quant_sparse_flash_attention_mxfp4
    test_aclnn_kv_quant_sparse_flash_attention_mxfp4.cpp)

target_include_directories(test_aclnn_kv_quant_sparse_flash_attention_mxfp4 PRIVATE
    "${ASCEND_PATH}/include"
    "${ASCEND_PATH}/include/aclnn"
    "${ASCEND_PATH}/include/aclnnop"
    "${ASCEND_PATH}/opp/vendors/${QSFA_VENDOR_NAME}_transformer/op_api/include")

target_link_libraries(test_aclnn_kv_quant_sparse_flash_attention_mxfp4 PRIVATE
    "${ASCEND_PATH}/lib64/libascendcl.so"
    "${ASCEND_PATH}/lib64/libnnopbase.so"
    "${ASCEND_PATH}/lib64/libopapi_math.so"
    "${QSFA_OPAPI_LIBRARY}")
