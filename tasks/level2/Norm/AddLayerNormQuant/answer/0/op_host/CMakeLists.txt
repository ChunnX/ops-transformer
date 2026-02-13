# Add CANN aclnn/opdev include for hand-written aclnn (op_api_def, CREATE_EXECUTOR, L2_DFX)
set(CANN_ACLNN_OPDEV_INC ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include/aclnn)
set(CANN_OP_API_OPDEV_INC ${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} ops_srcs)

# opbuild only needs def cpp to generate autogen; aclnn/l0 sources need opdev and must not be passed to opbuild
file(GLOB opbuild_def_srcs "${CMAKE_CURRENT_SOURCE_DIR}/*def.cpp")
if(NOT opbuild_def_srcs)
    set(opbuild_def_srcs ${ops_srcs})
endif()
opbuild(OPS_SRC ${opbuild_def_srcs}
        OUT_DIR ${ASCEND_AUTOGEN_PATH}
)

file(GLOB group_proto_src ${ASCEND_AUTOGEN_PATH}/group_proto/*.cc)
 
add_library(cust_op_proto SHARED
    $<$<TARGET_EXISTS:group_proto_src>:${group_proto_src}>
    ${ops_srcs}
    ${ASCEND_AUTOGEN_PATH}/op_proto.cc
)
target_include_directories(cust_op_proto PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CANN_OP_API_OPDEV_INC}
        ${CANN_ACLNN_OPDEV_INC}
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
# Tiling lib is loaded by opc in isolation: must not include aclnn/executor (l0op::Contiguous etc.)
set(optiling_srcs ${ops_srcs})
list(FILTER optiling_srcs EXCLUDE REGEX "aclnn_.*\\.cpp$")
list(REMOVE_ITEM optiling_srcs add_layer_norm_quant.cpp)
add_library(cust_optiling SHARED ${optiling_srcs})
target_include_directories(cust_optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CANN_OP_API_OPDEV_INC}
        ${CANN_ACLNN_OPDEV_INC}
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

# Prefer hand-written aclnn in op_host if present (for two-phase API with executor)
file(GLOB LOCAL_ACLNN_SRC "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_*.cpp")
file(GLOB LOCAL_ACLNN_INC "${CMAKE_CURRENT_SOURCE_DIR}/aclnn_*.h")
if(LOCAL_ACLNN_SRC AND LOCAL_ACLNN_INC)
    set(aclnn_src ${LOCAL_ACLNN_SRC})
    set(aclnn_inc ${LOCAL_ACLNN_INC})
else()
    file(GLOB aclnn_src ${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp)
    file(GLOB aclnn_inc ${ASCEND_AUTOGEN_PATH}/aclnn_*.h)
endif()
if(NOT ASCEND_PACK_SHARED_LIBRARY)
    add_library(cust_opapi SHARED ${aclnn_src} ${CMAKE_CURRENT_SOURCE_DIR}/add_layer_norm_quant.cpp)
else()
    file(GLOB op_registry ${ASCEND_AUTOGEN_PATH}/custom_op_registry.cpp)
    add_library(cust_opapi SHARED ${aclnn_src} ${op_registry})
    target_compile_definitions(cust_opapi PRIVATE ACLNN_WITH_BINARY)
endif()
if(ENABLE_CROSS_COMPILE)
    target_link_directories(cust_opapi PRIVATE
                            ${CMAKE_COMPILE_COMPILER_LIBRARY}
                            ${CMAKE_COMPILE_RUNTIME_LIBRARY}
    )
endif()
target_include_directories(cust_opapi PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CANN_OP_API_OPDEV_INC}
        ${CANN_ACLNN_OPDEV_INC}
        ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
        ${ASCEND_CANN_PACKAGE_PATH}/include
)
if(NOT ASCEND_PACK_SHARED_LIBRARY)
    target_link_libraries(cust_opapi PRIVATE intf_pub ascendcl nnopbase opapi)
else()
    add_library(cust_op_proto_obj OBJECT
                $<$<TARGET_EXISTS:group_proto_src>:${group_proto_src}>
                ${ops_srcs}
                ${ASCEND_AUTOGEN_PATH}/op_proto.cc
    )
    target_compile_definitions(cust_op_proto_obj PRIVATE OP_PROTO_LIB)
    target_compile_options(cust_op_proto_obj PRIVATE
                           -fvisibility=hidden
    )
    target_include_directories(cust_op_proto_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CANN_OP_API_OPDEV_INC}
        ${CANN_ACLNN_OPDEV_INC}
    )
    if(ENABLE_CROSS_COMPILE)
        target_link_directories(cust_op_proto_obj PRIVATE
                                ${CMAKE_COMPILE_COMPILER_LIBRARY}
                                ${CMAKE_COMPILE_RUNTIME_LIBRARY}
        )
    endif()
    target_link_libraries(cust_op_proto_obj PRIVATE
                          intf_pub
                          exe_graph
                          register
                          tiling_api
                          -Wl,--whole-archive
                          rt2_registry
                          -Wl,--no-whole-archive
    )
    add_library(cust_optiling_obj OBJECT ${ops_srcs})
    target_compile_definitions(cust_optiling_obj PRIVATE OP_TILING_LIB)
    target_compile_options(cust_optiling_obj PRIVATE
                           -fvisibility=hidden
    )
    target_include_directories(cust_optiling_obj PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CANN_OP_API_OPDEV_INC}
        ${CANN_ACLNN_OPDEV_INC}
    )
    if(ENABLE_CROSS_COMPILE)
        target_link_directories(cust_optiling_obj PRIVATE
                                ${CMAKE_COMPILE_COMPILER_LIBRARY}
                                ${CMAKE_COMPILE_RUNTIME_LIBRARY}
        )
    endif()
    target_link_libraries(cust_optiling_obj PRIVATE
                          intf_pub
                          exe_graph
                          register
                          tiling_api
                          -Wl,--whole-archive
                          rt2_registry
                          -Wl,--no-whole-archive
    )
    target_compile_options(cust_opapi PRIVATE -DLOG_CPP)
    target_include_directories(cust_opapi INTERFACE ${CMAKE_SOURCE_DIR}/build_out/library/)
    target_link_libraries(cust_opapi PRIVATE intf_pub ascendcl nnopbase cust_optiling_obj cust_op_proto_obj ascend_opregistry ascend_kernels)
    add_dependencies(cust_opapi ascend_opregistry)
endif()

add_custom_target(optiling_compat ALL
                  COMMAND ln -sf lib/linux/${CMAKE_SYSTEM_PROCESSOR}/$<TARGET_FILE_NAME:cust_optiling>
                          ${CMAKE_CURRENT_BINARY_DIR}/liboptiling.so
)
if(NOT ASCEND_PACK_SHARED_LIBRARY)
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
else()
        file(GLOB group_inc ${ASCEND_AUTOGEN_PATH}/group_proto/*.h)
        install(TARGETS cust_opapi
                LIBRARY DESTINATION op_api/lib)
        install(FILES ${ASCEND_AUTOGEN_PATH}/op_proto.h
                DESTINATION op_api/include)
        install(FILES ${group_inc}
                DESTINATION op_api/include)
        install(FILES ${aclnn_inc}
                DESTINATION op_api/include)
endif()
