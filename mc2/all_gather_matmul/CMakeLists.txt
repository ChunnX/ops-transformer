# ----------------------------------------------------------------------------
 # Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 # This file is a part of the CANN Open Software.
 # Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 # Please refer to the License for details. You may not use this file except in compliance with the License.
 # THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 # INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 # See LICENSE in the root of the software repository for the full text of the License.
 # ----------------------------------------------------------------------------

 file(GLOB SUBDIRECTORIES LIST_DIRECTORIES true RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
 # 遍历子目录
 foreach(SUBDIR ${SUBDIRECTORIES})
   # 检查子目录中是否存在 CMakeLists.txt
   if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${SUBDIR}/CMakeLists.txt)
     add_subdirectory(${SUBDIR})
   endif()
 endforeach()

 if (BUILD_OPEN_PROJECT)
    get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    set(${CURRENT_DIR_NAME}_depends mc2/common mc2/3rd PARENT_SCOPE)
endif()