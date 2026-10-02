# benchmarks 构建脚本：aclnn C++ 性能对比基准测试可执行程序构建

if(BUILD_OPEN_PROJECT OR ENABLE_TEST OR BENCHMARK)
    add_executable(benchmark_sage_attention2_aclnn benchmark_aclnn_fia.cpp)
    target_include_directories(benchmark_sage_attention2_aclnn PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_host
    )
    target_link_libraries(benchmark_sage_attention2_aclnn PRIVATE
        intf_pub
        ascendcl
        cust_opapi
    )
endif()
