cmake_minimum_required(VERSION 3.14)
# 设置工程名
project(ACLNN_EXAMPLE)

set(CMAKE_BUILD_TYPE "Debug")

# 设置C++编译标准
add_compile_options(-std=c++17)
add_compile_options(-std=c++17 -g -O0)

# 设置编译输出目录为当前目录下的bin文件夹
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./bin")

# 设置调试和发布模式的编译选项
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

# 获取LD_LIBRARY_PATH环境变量
if(NOT DEFINED ENV{LD_LIBRARY_PATH})
    message(FATAL_ERROR "LD_LIBRARY_PATH environment variable is not set")
endif()
set(LD_LIB_PATH "$ENV{LD_LIBRARY_PATH}")

# 分割路径列表并找到包含/vendors/的路径（仅自定义算子需要）  
string(REPLACE ":" ";" LD_LIB_LIST "${LD_LIB_PATH}")
set(TARGET_PATH "")
foreach(path ${LD_LIB_LIST})
    # 匹配包含/vendors/的路径（不区分位置）
    if(path MATCHES "/vendors/")
        set(TARGET_PATH "${path}")
        break()
    endif()
endforeach()
if(NOT TARGET_PATH)
    message(FATAL_ERROR "未在LD_LIBRARY_PATH中找到包含/vendors/的路径")
endif()
if(TARGET_PATH MATCHES "/vendors/([^/]+)")
    set(TARGET_SUBDIR "${CMAKE_MATCH_1}")
else()
    message(FATAL_ERROR "在路径${TARGET_PATH}中未找到/vendors/的直接子目录")
endif()

# 添加可执行文件（请替换为实际算子可执行文件），指定算子调用的*.cpp文件
add_executable(test_aclnn_mega_moe              
                test_aclnn_mega_moe.cpp)

# ASCEND_PATH（CANN软件包目录，请根据实际路径修改）
if(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")      
    set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
else()
    set(ASCEND_PATH "/usr/local/Ascend/cann")
endif()

# 设置头文件路径
set(INCLUDE_BASE_DIR "${ASCEND_PATH}/include")
include_directories(
    ${INCLUDE_BASE_DIR}
    ${ASCEND_PATH}/opp/vendors/${TARGET_SUBDIR}/op_api/include    # 仅自定义算子需要
    # ${INCLUDE_BASE_DIR}/aclnn                                   # 仅内置算子需要
)

# 1. 查找线程库
find_package(Threads REQUIRED)

# 2. 假设你的目标名称是 test_aclnn_mega_moe
# 将线程库链接到你的目标上
target_link_libraries(test_aclnn_mega_moe PRIVATE Threads::Threads stdc++fs)

# 链接所需的动态库
target_link_libraries(test_aclnn_mega_moe PRIVATE             # 替换实际算子可执行文件
    ${ASCEND_PATH}/lib64/libascendcl.so
    ${ASCEND_PATH}/lib64/libnnopbase.so
    ${ASCEND_PATH}/lib64/libhccl.so
    ${CMAKE_CURRENT_SOURCE_DIR}/mega_moe/vendors/MEGA_transformer/op_api/lib/libcust_opapi.so   # 仅自定义算子需要
    # ${ASCEND_PATH}/lib64/libopapi_transformer.so    # 仅内置算子需要
)

# 安装目标文件到bin目录  
install(TARGETS test_aclnn_mega_moe DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})