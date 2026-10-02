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
endif()
