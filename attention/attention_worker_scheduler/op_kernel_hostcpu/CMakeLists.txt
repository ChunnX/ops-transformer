# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for the details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------------------------------------

# Host-side constant folding: compile aicpu source as x86 OBJECT.
# Source files live in ../op_kernel_aicpu/ alongside this folder.
if (BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG AND NOT ENABLE_CUSTOM)
  file(GLOB AICPU_SRC ${CMAKE_SOURCE_DIR}/attention/attention_worker_scheduler/op_kernel_aicpu/*_aicpu.cpp)

  set(OBJ_NAME attention_worker_scheduler_host_const_obj)
  add_aicpu_host_kernel_modules(${OBJ_NAME})
  target_sources(${OBJ_NAME} PRIVATE ${AICPU_SRC})
endif()