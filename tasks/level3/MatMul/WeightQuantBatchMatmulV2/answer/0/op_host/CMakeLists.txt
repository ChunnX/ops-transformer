file(GLOB_RECURSE ops_srcs "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
file(GLOB_RECURSE DEF_CPP "${CMAKE_CURRENT_SOURCE_DIR}/*def.cpp")
file(GLOB_RECURSE TILING_SRC "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
file(GLOB_RECURSE ACLNN_SRC "${CMAKE_CURRENT_SOURCE_DIR}/aclnn*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/aclnn*.h")
list(REMOVE_ITEM TILING_SRC ${DEF_CPP})

opbuild(OPS_SRC ${DEF_CPP}
        OUT_DIR ${ASCEND_AUTOGEN_PATH}
)

file(GLOB group_proto_src ${ASCEND_AUTOGEN_PATH}/group_proto/*.cc)

add_library(cust_op_proto SHARED
    $<$<TARGET_EXISTS:group_proto_src>:${group_proto_src}>
    ${ASCEND_AUTOGEN_PATH}/op_proto.cc
)
target_include_directories(cust_op_proto PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn
        ${ASCEND_CANN_PACKAGE_PATH}/include/external
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
)
target_compile_definitions(cust_op_proto PRIVATE OP_PROTO_LIB)
target_compile_options(cust_op_proto PRIVATE
        -fvisibility=hidden
)
if(ENABLE_CROSS_COMPILE)
    target_link_directories(cust_op_proto PRIVATE
                            ${CMAKE_COMPILE_COMPILER_LIBRARY}
                            ${CMAKE_COMPILE_RUNTIME_LIBRARY}
    )
endif()
target_link_libraries(cust_op_proto PRIVATE
        intf_pub
        exe_graph
        register
        tiling_api
        -Wl,--whole-archive
        rt2_registry
        -Wl,--no-whole-archive
)
set_target_properties(cust_op_proto PROPERTIES OUTPUT_NAME
                      cust_opsproto_rt2.0
)

file(GLOB fallback_src ${ASCEND_AUTOGEN_PATH}/fallback_*.cpp)
add_library(cust_optiling SHARED ${TILING_SRC})
target_include_directories(cust_optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/op_common
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/op_common/op_host
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn
        ${ASCEND_CANN_PACKAGE_PATH}/include/external
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
)
if (${fallback_src})
    target_sources(cust_optiling PRIVATE ${fallback_src})
endif()
target_compile_definitions(cust_optiling PRIVATE OP_TILING_LIB)
target_compile_options(cust_optiling PRIVATE
        -fvisibility=hidden
)
if(ENABLE_CROSS_COMPILE)
    target_link_directories(cust_optiling PRIVATE
                            ${CMAKE_COMPILE_COMPILER_LIBRARY}
                            ${CMAKE_COMPILE_RUNTIME_LIBRARY}
    )
endif()
target_link_libraries(cust_optiling PRIVATE
        nnopbase
        ops_base
        intf_pub
        exe_graph
        register
        tiling_api
        -Wl,--whole-archive
        rt2_registry
        -Wl,--no-whole-archive
)
set_target_properties(cust_optiling PROPERTIES OUTPUT_NAME
                      cust_opmaster_rt2.0
)

add_custom_target(optiling_compat ALL
        COMMAND ln -sf lib/linux/${CMAKE_SYSTEM_PROCESSOR}/$<TARGET_FILE_NAME:cust_optiling>
                ${CMAKE_CURRENT_BINARY_DIR}/liboptiling.so
)

# Build cust_opapi library (required by validation layer)
file(GLOB aclnn_src ${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp)
file(GLOB aclnn_inc ${ASCEND_AUTOGEN_PATH}/aclnn_*.h)
if(aclnn_src)
    add_library(cust_opapi SHARED ${aclnn_src})
else()
    add_library(cust_opapi SHARED ${ASCEND_AUTOGEN_PATH}/op_proto.cc)
    target_compile_definitions(cust_opapi PRIVATE ACLNN_WITH_BINARY)
endif()
target_include_directories(cust_opapi PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn
        ${ASCEND_CANN_PACKAGE_PATH}/include/external
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/platform
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/metadef
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/runtime
        ${ASCEND_CANN_PACKAGE_PATH}/include/experiment/msprof
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_link_libraries(cust_opapi PRIVATE
        intf_pub
        ascendcl
        nnopbase
)
set_target_properties(cust_opapi PROPERTIES OUTPUT_NAME
                      cust_opapi
)

install(TARGETS cust_op_proto
        LIBRARY DESTINATION packages/vendors/${vendor_name}/op_proto/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
install(FILES ${ASCEND_AUTOGEN_PATH}/op_proto.h
        DESTINATION packages/vendors/${vendor_name}/op_proto/inc)
file(GLOB GROUP_PROTO_HEADERS ${ASCEND_AUTOGEN_PATH}/group_proto/*.h)
if (GROUP_PROTO_HEADERS)
    install(FILES ${GROUP_PROTO_HEADERS}
            DESTINATION packages/vendors/${vendor_name}/op_proto/inc)
endif()
install(TARGETS cust_optiling
        LIBRARY DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_tiling/lib/linux/${CMAKE_SYSTEM_PROCESSOR})
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/liboptiling.so
        DESTINATION packages/vendors/${vendor_name}/op_impl/ai_core/tbe/op_tiling)
install(TARGETS cust_opapi
        LIBRARY DESTINATION packages/vendors/${vendor_name}/op_api/lib)
install(FILES ${aclnn_inc}
        DESTINATION packages/vendors/${vendor_name}/op_api/include)
