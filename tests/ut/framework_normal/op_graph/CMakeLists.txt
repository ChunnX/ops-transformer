# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

if(UT_TEST_ALL OR OP_GRAPH_UT)
  set(OP_GRAPH_UT_EXE ${PKG_NAME}_op_graph_ut)

  if(NOT TARGET ${OP_GRAPH_MODULE_NAME}_cases_obj)
    add_op_graph_ut_modules(${OP_GRAPH_MODULE_NAME})
  endif()

  add_executable(${OP_GRAPH_UT_EXE}
      test_op_graph_main.cpp
      $<TARGET_OBJECTS:${OP_GRAPH_MODULE_NAME}_cases_obj>
      $<$<TARGET_EXISTS:${GRAPH_PLUGIN_NAME}_obj>:$<TARGET_OBJECTS:${GRAPH_PLUGIN_NAME}_obj>>
  )
  target_compile_definitions(${OP_GRAPH_UT_EXE} PRIVATE LOG_CPP)
  target_compile_options(${OP_GRAPH_UT_EXE} PUBLIC -fPIE -fno-access-control)
  target_include_directories(${OP_GRAPH_UT_EXE} PRIVATE
      ${METADEF_INCLUDE_DIRS}
      $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/include>>
      $<$<BOOL:${BUILD_OPEN_PROJECT}>:$<BUILD_INTERFACE:${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/include/external>>
  )
  target_link_directories(${OP_GRAPH_UT_EXE} PRIVATE ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64)
  target_link_libraries(${OP_GRAPH_UT_EXE} PRIVATE
      $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
      $<$<BOOL:${alog_FOUND}>:$<BUILD_INTERFACE:alog_headers>>
      -Wl,--no-as-needed
      metadef
      graph
      graph_base
      ge_common
      ge_common_base
      exe_graph
      register
      fusion_pass
      eager_style_graph_builder_base
      platform
      runtime
      acl_rt
      ascendalog
      gtest
      c_sec
      dl
      -Wl,--as-needed
  )

  if(ENABLE_UT_EXEC)
    add_custom_command(
      TARGET ${OP_GRAPH_UT_EXE}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E env LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH} ./${OP_GRAPH_UT_EXE}
      COMMENT "Run ops op_graph utest"
    )
  endif()
endif()
