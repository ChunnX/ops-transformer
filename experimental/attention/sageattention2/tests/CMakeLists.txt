# tests 构建脚本：aclnn C++ 测试可执行程序构建

if(BUILD_OPEN_PROJECT OR ENABLE_TEST)
    add_executable(test_sage_attention2_aclnn test_aclnn.cpp)
    target_include_directories(test_sage_attention2_aclnn PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
    )
    target_link_libraries(test_sage_attention2_aclnn PRIVATE
        intf_pub
        ascendcl
        cust_opapi
    )

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../benchmarks/benchmark_aclnn_fia.cpp")
        add_executable(benchmark_sage_attention2_aclnn ../benchmarks/benchmark_aclnn_fia.cpp)
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
endif()
