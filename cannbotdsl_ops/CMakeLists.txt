# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
cmake_minimum_required(VERSION 3.20)

# This directory is also a small standalone parent OPP for binary acceptance.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(FlashKdaFrozenOperator LANGUAGES CXX)
endif()

set(FLASH_KDA_FROZEN_STAGE "" CACHE PATH "Verified cannbot_framework FlashKda stage")
set(CANNBotDSL_NPU_ARCH "dav-3510" CACHE STRING "Architecture of the frozen bundle")
set(NNOPBASE_LIBRARY "" CACHE FILEPATH "External Task opbase shared library")
set(NNOPBASE_INCLUDE_DIR "" CACHE PATH "Directory containing opdev/external_op_task.h")

if(NOT EXISTS "${FLASH_KDA_FROZEN_STAGE}/stage.json")
  message(FATAL_ERROR "FLASH_KDA_FROZEN_STAGE must contain a verified frozen stage")
endif()
if(NOT TARGET nnopbase)
  if(NOT EXISTS "${NNOPBASE_LIBRARY}" OR
     NOT EXISTS "${NNOPBASE_INCLUDE_DIR}/opdev/external_op_task.h")
    message(FATAL_ERROR "Supply External Task NNOPBASE_LIBRARY and NNOPBASE_INCLUDE_DIR")
  endif()
  add_library(nnopbase SHARED IMPORTED)
  set_target_properties(nnopbase PROPERTIES
    IMPORTED_LOCATION "${NNOPBASE_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${NNOPBASE_INCLUDE_DIR};$ENV{ASCEND_HOME_PATH}/include;$ENV{ASCEND_HOME_PATH}/include/aclnn;$ENV{ASCEND_HOME_PATH}/include/experiment")
endif()

find_package(CANNBotFramework CONFIG REQUIRED)
cannbot_framework_add_frozen_bundle(flash_kda_frozen DIRECTORY "${FLASH_KDA_FROZEN_STAGE}")
if(NOT TARGET cust_opapi)
  add_library(cust_opapi SHARED)
  set_target_properties(cust_opapi PROPERTIES INSTALL_RPATH "$ORIGIN")
  install(TARGETS cust_opapi LIBRARY DESTINATION packages/vendors/custom_transformer/op_api/lib)
  install(FILES "${NNOPBASE_LIBRARY}" DESTINATION packages/vendors/custom_transformer/op_api/lib)
  install(PROGRAMS "${CMAKE_CURRENT_LIST_DIR}/install.sh" DESTINATION .)
endif()
target_link_libraries(cust_opapi PRIVATE flash_kda_frozen)

option(FLASH_KDA_BUILD_ACLNN_VALIDATOR "Build the frozen ACLNN binary validator" ON)
if(FLASH_KDA_BUILD_ACLNN_VALIDATOR)
  add_executable(flash_kda_aclnn_validate tests/flash_kda_frozen_validate.cpp)
  target_compile_features(flash_kda_aclnn_validate PRIVATE cxx_std_17)
  target_include_directories(flash_kda_aclnn_validate PRIVATE
    "${FLASH_KDA_FROZEN_STAGE}/generated"
    "$ENV{ASCEND_HOME_PATH}/include")
  find_library(FLASH_KDA_ASCENDCL ascendcl PATHS "$ENV{ASCEND_HOME_PATH}/lib64" REQUIRED NO_DEFAULT_PATH)
  target_link_libraries(flash_kda_aclnn_validate PRIVATE cust_opapi nnopbase "${FLASH_KDA_ASCENDCL}")
endif()
