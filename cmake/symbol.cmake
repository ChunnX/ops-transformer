# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

function(gen_common_symbol)
  add_library(${COMMON_NAME} SHARED
    $<$<TARGET_EXISTS:${COMMON_NAME}_obj>:$<TARGET_OBJECTS:${COMMON_NAME}_obj>>
  )

  target_link_libraries(${COMMON_NAME}
    PRIVATE
    c_sec
    -Wl,--no-as-needed
    register
    -Wl,--as-needed
    exe_graph
    tiling_api
  )

  install(TARGETS ${COMMON_NAME}
    LIBRARY DESTINATION ${COMMON_LIB_INSTALL_DIR}
  )
  install(DIRECTORY ${OPS_TRANSFORMER_COMMON_INC_HEADERS}
    DESTINATION ${COMMON_INC_INSTALL_DIR}
  )
endfunction()

# ophost shared
function(gen_ophost_symbol)
  add_library(${OPHOST_NAME} SHARED
    $<$<TARGET_EXISTS:${OPHOST_NAME}_infer_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_infer_obj>>
    $<$<TARGET_EXISTS:${OPHOST_NAME}_tiling_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_tiling_obj>>
    $<$<TARGET_EXISTS:${OPHOST_NAME}_aicpu_objs>:$<TARGET_OBJECTS:${OPHOST_NAME}_aicpu_objs>>
    $<$<TARGET_EXISTS:${COMMON_NAME}_obj>:$<TARGET_OBJECTS:${COMMON_NAME}_obj>>
    $<$<TARGET_EXISTS:${OPHOST_NAME}_opmaster_ct_gentask_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_opmaster_ct_gentask_obj>>
  )

  target_link_libraries(
    ${OPHOST_NAME}
    PRIVATE $<BUILD_INTERFACE:intf_pub_cxx17>
            c_sec
            -Wl,--no-as-needed
            register
            $<$<TARGET_EXISTS:opsbase>:opsbase>
            -Wl,--as-needed
            -Wl,--whole-archive
            rt2_registry_static
            -Wl,--no-whole-archive
            tiling_api
    )

  target_link_directories(${OPHOST_NAME}
    PRIVATE
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64
  )

  install(TARGETS ${OPHOST_NAME}
    LIBRARY DESTINATION ${OPHOST_LIB_INSTALL_PATH}
  )
endfunction()

# graph_plugin shared
function(gen_opgraph_symbol)
  add_library(${OPGRAPH_NAME} SHARED
    $<$<TARGET_EXISTS:${GRAPH_PLUGIN_NAME}_obj>:$<TARGET_OBJECTS:${GRAPH_PLUGIN_NAME}_obj>>
  )

  target_link_libraries(
    ${OPGRAPH_NAME}
    PRIVATE $<BUILD_INTERFACE:intf_pub_cxx17>
            c_sec
            -Wl,--no-as-needed
            register
            $<$<TARGET_EXISTS:opsbase>:opsbase>
            -Wl,--as-needed
            -Wl,--whole-archive
            rt2_registry_static
            -Wl,--no-whole-archive
    )

  target_link_directories(${OPGRAPH_NAME}
    PRIVATE
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64
  )

  set_target_properties(${OPGRAPH_NAME} PROPERTIES OUTPUT_NAME "opgraph_transformer")
  install(TARGETS ${OPGRAPH_NAME}
    LIBRARY DESTINATION ${OPGRAPH_LIB_INSTALL_DIR}
  )
  install(FILES ${ASCEND_GRAPH_CONF_DST}/ops_proto_transformer.h
    DESTINATION ${OPGRAPH_INC_INSTALL_DIR} OPTIONAL
  )
endfunction()

function(gen_opapi_symbol)
  # opapi shared	
  add_library(${OPAPI_NAME} SHARED	
    $<$<TARGET_EXISTS:${OPHOST_NAME}_opapi_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_opapi_obj>>
    $<$<TARGET_EXISTS:opbuild_gen_aclnn_all>:$<TARGET_OBJECTS:opbuild_gen_aclnn_all>>
  )

  target_link_libraries(${OPAPI_NAME}
    PUBLIC
    $<BUILD_INTERFACE:intf_pub>
    -Wl,--whole-archive
    ops_aclnn
    -Wl,--no-whole-archive
    nnopbase
    profapi
    ge_common_base
    ascend_dump
    ascendalog
    dl
  )
endfunction()

function(gen_built_in_opapi_symbol)
  install(TARGETS ${OPAPI_NAME}
    LIBRARY DESTINATION ${ACLNN_LIB_INSTALL_DIR}
  )
endfunction()

function(gen_cust_opapi_symbol)
  #op_api
  set_target_properties(${OPAPI_NAME} PROPERTIES OUTPUT_NAME "cust_opapi")

  install(TARGETS ${OPAPI_NAME}
    LIBRARY DESTINATION ${ACLNN_LIB_INSTALL_DIR}
  )
endfunction()

function(gen_cust_optiling_symbol)
  # op_tiling
  if(NOT TARGET ${OPHOST_NAME}_tiling_obj)
    return()
  endif()
  add_library(cust_opmaster SHARED
    $<$<TARGET_EXISTS:${OPHOST_NAME}_tiling_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_tiling_obj>
    $<$<TARGET_EXISTS:${COMMON_NAME}_obj>:$<TARGET_OBJECTS:${COMMON_NAME}_obj>>>
  )
  target_link_libraries(cust_opmaster
    PRIVATE
    c_sec
    tiling_api
    -Wl,--no-as-needed
    register
    $<$<TARGET_EXISTS:opsbase>:opsbase>
    -Wl,--as-needed
    -Wl,--whole-archive
    rt2_registry_static
    -Wl,--no-whole-archive
  )
  target_link_directories(cust_opmaster
    PRIVATE
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64
  )
  set_target_properties(cust_opmaster PROPERTIES OUTPUT_NAME "cust_opmaster_rt2.0")

  install(TARGETS cust_opmaster
    LIBRARY DESTINATION ${OPTILING_LIB_INSTALL_DIR}
  )
  add_custom_target(optiling_compat ALL
    COMMAND ln -sf lib/linux/${CMAKE_SYSTEM_PROCESSOR}/$<TARGET_FILE_NAME:cust_opmaster>
            ${CMAKE_BINARY_DIR}/liboptiling.so
  )
  install(FILES ${CMAKE_BINARY_DIR}/liboptiling.so
    DESTINATION ${OPTILING_INSTALL_DIR}
  )
endfunction()

function(gen_cust_proto_symbol)
  # op_proto
  if(NOT TARGET ${OPHOST_NAME}_infer_obj)
    return()
  endif()
  add_library(cust_proto SHARED
    $<$<TARGET_EXISTS:${OPHOST_NAME}_infer_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_infer_obj>>
    $<$<TARGET_EXISTS:${GRAPH_PLUGIN_NAME}_obj>:$<TARGET_OBJECTS:${GRAPH_PLUGIN_NAME}_obj>>
  )
  target_link_libraries(cust_proto
    PRIVATE
    c_sec
    -Wl,--no-as-needed
    register
    $<$<TARGET_EXISTS:opsbase>:opsbase>
    -Wl,--as-needed
    -Wl,--whole-archive
    rt2_registry_static
    -Wl,--no-whole-archive
  )
  target_link_directories(cust_proto
    PRIVATE
    ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64
  )
  set_target_properties(cust_proto PROPERTIES OUTPUT_NAME "cust_opsproto_rt2.0")

  install(TARGETS cust_proto
    LIBRARY DESTINATION ${OPPROTO_LIB_INSTALL_DIR}
  )
  file(GLOB_RECURSE proto_headers
    ${ASCEND_AUTOGEN_PATH}/*_proto.h
  )
  install(FILES ${proto_headers}
    DESTINATION ${OPPROTO_INC_INSTALL_DIR} OPTIONAL
  )
endfunction()

function(gen_norm_symbol)
  gen_common_symbol()

  if (ENABLE_OPS_HOST)
    gen_ophost_symbol()
    gen_opapi_symbol()
    gen_built_in_opapi_symbol()
  endif()

  gen_opgraph_symbol()

endfunction()

function(gen_cust_symbol)
  gen_opapi_symbol()

  gen_cust_opapi_symbol()

  gen_cust_optiling_symbol()

  gen_cust_proto_symbol()
endfunction()