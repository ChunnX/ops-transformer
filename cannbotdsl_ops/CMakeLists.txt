# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
cmake_minimum_required(VERSION 3.20)

# The standard ops-transformer parent owns targets, package layout and installer.
if(NOT TARGET cust_opapi OR NOT TARGET nnopbase)
  message(FATAL_ERROR "Configure the ops-transformer repository root")
endif()

set(FLASH_KDA_FROZEN_STAGE "" CACHE PATH "Verified cannbot_framework FlashKda stage")
set(CANNBotDSL_NPU_ARCH "dav-3510" CACHE STRING "Architecture of the frozen bundle")

if(NOT EXISTS "${FLASH_KDA_FROZEN_STAGE}/stage.json")
  message(FATAL_ERROR "FLASH_KDA_FROZEN_STAGE must contain a verified frozen stage")
endif()
set(_flash_stage "${FLASH_KDA_FROZEN_STAGE}")
if(ENABLE_BUILT_IN)
  # stage.json describes parent install destinations; the schema-2 bundle and
  # all generated ACLNN sources are copied unchanged from the verified stage.
  set(_flash_stage "${CMAKE_CURRENT_BINARY_DIR}/builtin-stage")
  file(COPY "${FLASH_KDA_FROZEN_STAGE}/" DESTINATION "${_flash_stage}")
  file(READ "${_flash_stage}/stage.json" _flash_stage_json)
  string(JSON _flash_stage_json SET "${_flash_stage_json}" artifact_install_destination
    "\"opp/built-in/op_impl/cannbotdsl/FlashKda\"")
  string(JSON _flash_stage_json SET "${_flash_stage_json}" api_header_install_destination
    "\"${ACLNN_INC_INSTALL_DIR}\"")
  file(WRITE "${_flash_stage}/stage.json" "${_flash_stage_json}\n")
endif()

find_package(CANNBotFramework CONFIG REQUIRED)
cannbot_framework_add_frozen_bundle(flash_kda_frozen DIRECTORY "${_flash_stage}")
if(ENABLE_BUILT_IN)
  # gen_norm_symbol() creates and installs this official target after this file.
  set(_flash_aclnn_library "${OPAPI_NAME}")
else()
  set(_flash_aclnn_library cust_opapi)
  target_link_libraries(cust_opapi PRIVATE flash_kda_frozen)
endif()

option(FLASH_KDA_BUILD_ACLNN_VALIDATOR "Build the frozen ACLNN binary validator" ON)
if(FLASH_KDA_BUILD_ACLNN_VALIDATOR)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(_flash_runner "${CMAKE_CURRENT_BINARY_DIR}/flash_kda_aclnn_runner.cpp")
  add_custom_command(OUTPUT "${_flash_runner}"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_flash_kda.py"
      --emit-aclnn-runner --output "${_flash_runner}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_flash_kda.py"
    VERBATIM)
  add_executable(flash_kda_aclnn_validate "${_flash_runner}")
  target_compile_features(flash_kda_aclnn_validate PRIVATE cxx_std_17)
  target_include_directories(flash_kda_aclnn_validate PRIVATE
    "${FLASH_KDA_FROZEN_STAGE}/generated"
    "$ENV{ASCEND_HOME_PATH}/include")
  find_library(FLASH_KDA_ASCENDCL ascendcl PATHS "$ENV{ASCEND_HOME_PATH}/lib64" REQUIRED NO_DEFAULT_PATH)
  target_link_libraries(flash_kda_aclnn_validate PRIVATE "${_flash_aclnn_library}" nnopbase "${FLASH_KDA_ASCENDCL}")
endif()
