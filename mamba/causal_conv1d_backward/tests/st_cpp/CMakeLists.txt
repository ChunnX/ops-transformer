cmake_minimum_required(VERSION 3.14)
project(test_causal_conv1d_backward)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项：USE_MOCK_ACLNN（Mock/Real 模式切换）
option(USE_MOCK "Use mock mode without NPU" OFF)

if(USE_MOCK)
    add_definitions(-DUSE_MOCK_ACLNN)
    message(STATUS "Building in Mock mode (CPU golden only)")
else()
    message(STATUS "Building in Real mode (NPU execution)")
endif()

# 测试可执行文件
add_executable(test_causal_conv1d_backward test_causal_conv1d_backward.cpp)

if(NOT USE_MOCK)
    # Real 模式：链接 ACL 运行时和算子库
    find_path(ACL_INC_DIR NAMES acl/acl.h PATHS $ENV{ASCEND_TOOLKIT_HOME}/include $ENV{ASCEND_HOME_PATH}/include /usr/local/Ascend/ascend-toolkit/latest/include)
    find_path(OP_API_INC_DIR NAMES aclnn_causal_conv1d_backward.h PATHS
        $ENV{ASCEND_HOME_PATH}/opp/vendors/custom_transformer/op_api/include
        $ENV{ASCEND_OPP_PATH}/vendors/custom_transformer/op_api/include
        $ENV{ASCEND_CUSTOM_OPP_PATH}/op_api/include
        /usr/local/Ascend/latest/include/aclnn)

    find_library(ACL_LIB NAMES ascendcl PATHS $ENV{ASCEND_TOOLKIT_HOME}/lib64 $ENV{ASCEND_HOME_PATH}/lib64 /usr/local/Ascend/ascend-toolkit/latest/lib64)
    find_library(NNOPBASE_LIB NAMES nnopbase PATHS $ENV{ASCEND_TOOLKIT_HOME}/lib64 $ENV{ASCEND_HOME_PATH}/lib64 /usr/local/Ascend/ascend-toolkit/latest/lib64)
    # libopapi.so: 提供 l0op::Contiguous / l0op::ViewCopy 等符号（cust_opapi L0 层依赖）
    find_library(OPAPI_LIB NAMES opapi PATHS $ENV{ASCEND_TOOLKIT_HOME}/lib64 $ENV{ASCEND_HOME_PATH}/lib64 /usr/local/Ascend/ascend-toolkit/latest/lib64)
    find_library(OP_API_LIB NAMES cust_opapi PATHS
        $ENV{ASCEND_HOME_PATH}/opp/vendors/custom_transformer/op_api/lib
        $ENV{ASCEND_OPP_PATH}/vendors/custom_transformer/op_api/lib
        $ENV{ASCEND_CUSTOM_OPP_PATH}/op_api/lib
        /usr/local/Ascend/latest/lib64)

    if(NOT ACL_INC_DIR)
        message(FATAL_ERROR "Cannot find ACL include directory. Set ASCEND_TOOLKIT_HOME environment variable.")
    endif()

    if(NOT OP_API_INC_DIR)
        message(WARNING "Cannot find causal_conv1d_backward op_api headers. Real mode may fail.")
    endif()

    if(NOT ACL_LIB)
        message(FATAL_ERROR "Cannot find libascendcl.so. Check ASCEND_TOOLKIT_HOME.")
    endif()

    if(NOT OP_API_LIB)
        message(WARNING "Cannot find libcust_opapi.so. Install the operator package first.")
    endif()

    target_include_directories(test_causal_conv1d_backward PRIVATE ${ACL_INC_DIR})
    if(OP_API_INC_DIR)
        target_include_directories(test_causal_conv1d_backward PRIVATE ${OP_API_INC_DIR})
    endif()

    target_link_libraries(test_causal_conv1d_backward ${ACL_LIB})
    if(NNOPBASE_LIB)
        target_link_libraries(test_causal_conv1d_backward ${NNOPBASE_LIB})
    endif()
    if(OPAPI_LIB)
        target_link_libraries(test_causal_conv1d_backward ${OPAPI_LIB})
    else()
        message(FATAL_ERROR "Cannot find libopapi.so (l0op symbols). Check ASCEND_TOOLKIT_HOME.")
    endif()
    if(OP_API_LIB)
        target_link_libraries(test_causal_conv1d_backward ${OP_API_LIB})
    endif()

    message(STATUS "ACL_INC_DIR: ${ACL_INC_DIR}")
    message(STATUS "ACL_LIB: ${ACL_LIB}")
    message(STATUS "OPAPI_LIB: ${OPAPI_LIB}")
    message(STATUS "OP_API_INC_DIR: ${OP_API_INC_DIR}")
    message(STATUS "OP_API_LIB: ${OP_API_LIB}")
endif()

# 编译选项
target_compile_options(test_causal_conv1d_backward PRIVATE -Wall -Wextra)
