cmake_minimum_required(VERSION 3.14)
project(ACLNN_V5_EXAMPLE)

add_compile_options(-std=c++11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "./bin")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

add_executable(opapi_test
               test_aclnn_flash_attention_varlen_score_v5.cpp)

if(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
    set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
else()
    set(ASCEND_PATH "/usr/local/Ascend/cann")
endif()
set(INCLUDE_BASE_DIR "${ASCEND_PATH}/include")
set(CUSTOM_OPP_PATH "${ASCEND_PATH}/opp/vendors/custom_transformer")
include_directories(
    ${CUSTOM_OPP_PATH}/op_api/include
    ${INCLUDE_BASE_DIR}
    ${INCLUDE_BASE_DIR}/aclnn
    ${INCLUDE_BASE_DIR}/aclnnop
)

target_link_libraries(opapi_test PRIVATE
                      ${ASCEND_PATH}/lib64/libascendcl.so
                      ${ASCEND_PATH}/lib64/libnnopbase.so
                      ${ASCEND_PATH}/lib64/libopapi_math.so
                      ${ASCEND_PATH}/lib64/libopapi_transformer.so
                      ${CUSTOM_OPP_PATH}/op_api/lib/libcust_opapi.so
                      )

install(TARGETS opapi_test DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
