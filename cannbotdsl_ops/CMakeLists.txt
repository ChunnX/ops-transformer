if(NOT CANNBOTDSL_SOURCE_DIR)
  get_filename_component(_cannbotdsl_sibling
    "${CMAKE_SOURCE_DIR}/../cannbot-dsl" ABSOLUTE)
  if(EXISTS "${_cannbotdsl_sibling}/cmake/CANNBotDSLOperatorPackage.cmake")
    set(CANNBOTDSL_SOURCE_DIR "${_cannbotdsl_sibling}")
  endif()
endif()
if(NOT EXISTS "${CANNBOTDSL_SOURCE_DIR}/cmake/CANNBotDSLOperatorPackage.cmake")
  message(FATAL_ERROR
    "ENABLE_CANNBOTDSL_PROVIDER_ACLNN requires CANNBOTDSL_SOURCE_DIR")
endif()
if(NOT CANNBOTDSL_RUNTIME_ROOT)
  set(CANNBOTDSL_RUNTIME_ROOT
    "${CANNBOTDSL_SOURCE_DIR}/build/python_packages")
endif()
if(NOT EXISTS "${CANNBOTDSL_RUNTIME_ROOT}/cannbotdsl/package/cli.py")
  message(FATAL_ERROR
    "CANNBOTDSL_RUNTIME_ROOT must contain cannbotdsl/package/cli.py: "
    "${CANNBOTDSL_RUNTIME_ROOT}")
endif()
if(NOT "${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend950")
  message(FATAL_ERROR
    "FlashKDA Provider package currently requires ASCEND_COMPUTE_UNIT=ascend950")
endif()

include("${CANNBOTDSL_SOURCE_DIR}/cmake/CANNBotDSLOperatorPackage.cmake")

set(_cannbotdsl_stage_dir "${CMAKE_BINARY_DIR}/cannbotdsl_component")
set(_cannbotdsl_registry_dir "${CMAKE_BINARY_DIR}/cannbotdsl_provider_aclnn")
cannbotdsl_stage_operator_packages(
  TARGET cannbotdsl_operator_packages
  RUNTIME_ROOT "${CANNBOTDSL_RUNTIME_ROOT}"
  OUTPUT_DIR "${_cannbotdsl_stage_dir}"
  INSTALL_DESTINATION
    "packages/vendors/${VENDOR_NAME}_transformer/op_impl/cannbotdsl"
  SPECS
    "${CMAKE_CURRENT_LIST_DIR}/attention/flash_kda/package.toml")

cannbotdsl_generate_provider_aclnn_registry(
  TARGET cannbotdsl_provider_aclnn_registry
  STAGE_TARGET cannbotdsl_operator_packages
  STAGE_DIR "${_cannbotdsl_stage_dir}"
  OUTPUT_DIR "${_cannbotdsl_registry_dir}"
  RUNTIME_ROOT "${CANNBOTDSL_RUNTIME_ROOT}")

set(_flash_kda_policy_dir
  "${CMAKE_CURRENT_LIST_DIR}/attention/flash_kda")
set(_flash_kda_policy_source
  "${_flash_kda_policy_dir}/flash_kda_variant_policy.py")
set(_flash_kda_policy_generator
  "${_flash_kda_policy_dir}/generate_variant_policy.py")
set(_flash_kda_manifest
  "${_cannbotdsl_stage_dir}/packages/ops-transformer-flash-kda/0.1.0/manifest.json")
set(_flash_kda_policy_header
  "${_cannbotdsl_registry_dir}/flash_kda_variant_policy.h")
add_custom_command(
  OUTPUT "${_flash_kda_policy_header}"
  COMMAND "${CMAKE_COMMAND}" -E env
    "PYTHONPATH=${_flash_kda_policy_dir}"
    python3 "${_flash_kda_policy_generator}"
    --manifest "${_flash_kda_manifest}"
    --output "${_flash_kda_policy_header}"
  DEPENDS
    "${CANNBOTDSL_PROVIDER_ACLNN_REGISTRY_HEADER}"
    "${_flash_kda_policy_source}"
    "${_flash_kda_policy_generator}"
  COMMENT "Validating FlashKDA variant coverage and generating L0 policy"
  VERBATIM)
add_custom_target(cannbotdsl_flash_kda_variant_policy
  DEPENDS "${_flash_kda_policy_header}")
add_dependencies(cannbotdsl_flash_kda_variant_policy
  cannbotdsl_provider_aclnn_registry)

set(CANNBOTDSL_PROVIDER_OPBASE_SOURCE_DIR "" CACHE PATH
  "opbase source tree containing opdev/provider_kernel.h")
if(NOT CANNBOTDSL_PROVIDER_OPBASE_SOURCE_DIR AND
   EXISTS "${CMAKE_SOURCE_DIR}/../opbase/include/nnopbase/opdev/provider_kernel.h")
  get_filename_component(CANNBOTDSL_PROVIDER_OPBASE_SOURCE_DIR
    "${CMAKE_SOURCE_DIR}/../opbase" ABSOLUTE)
endif()
if(NOT EXISTS
   "${CANNBOTDSL_PROVIDER_OPBASE_SOURCE_DIR}/include/nnopbase/opdev/provider_kernel.h")
  message(FATAL_ERROR
    "Provider ACLNN requires an opbase tree with opdev/provider_kernel.h")
endif()

if(NOT CANNBOTDSL_PROVIDER_OPBASE_LIBRARY)
  set(_cannbotdsl_installed_nnopbase
    "${ASCEND_CANN_PACKAGE_PATH}/lib64/libnnopbase.so")
  if(EXISTS "${_cannbotdsl_installed_nnopbase}")
    set(CANNBOTDSL_PROVIDER_OPBASE_LIBRARY
      "${_cannbotdsl_installed_nnopbase}")
  endif()
endif()
if(NOT EXISTS "${CANNBOTDSL_PROVIDER_OPBASE_LIBRARY}")
  message(FATAL_ERROR
    "Provider ACLNN requires CANNBOTDSL_PROVIDER_OPBASE_LIBRARY to name "
    "a Provider ABI v1 libnnopbase.so")
endif()
get_filename_component(CANNBOTDSL_PROVIDER_OPBASE_LIBRARY
  "${CANNBOTDSL_PROVIDER_OPBASE_LIBRARY}" REALPATH)

set(_cannbotdsl_abi_probe
  "${CMAKE_BINARY_DIR}/cannbotdsl_provider_abi_probe.cpp")
file(WRITE "${_cannbotdsl_abi_probe}"
  "namespace op { void RequireProviderAclnnRuntimeAbiV1(); }\n"
  "int main() { op::RequireProviderAclnnRuntimeAbiV1(); return 0; }\n")
try_compile(_cannbotdsl_has_provider_abi_v1
  "${CMAKE_BINARY_DIR}/cannbotdsl_provider_abi_probe"
  SOURCES "${_cannbotdsl_abi_probe}"
  LINK_LIBRARIES "${CANNBOTDSL_PROVIDER_OPBASE_LIBRARY}"
  OUTPUT_VARIABLE _cannbotdsl_abi_probe_log)
if(NOT _cannbotdsl_has_provider_abi_v1)
  message(FATAL_ERROR
    "libnnopbase.so does not provide Provider ACLNN runtime ABI v1: "
    "${CANNBOTDSL_PROVIDER_OPBASE_LIBRARY}\n"
    "${_cannbotdsl_abi_probe_log}")
endif()
message(STATUS "Validated Provider ACLNN runtime ABI v1: "
  "${CANNBOTDSL_PROVIDER_OPBASE_LIBRARY}")

if(NOT TARGET ${OPHOST_NAME}_opapi_obj)
  add_opapi_modules()
endif()
set_source_files_properties(
  "${CANNBOTDSL_PROVIDER_ACLNN_REGISTRY_SOURCE}"
  "${CANNBOTDSL_PROVIDER_ACLNN_ADAPTER_SOURCE}"
  TARGET_DIRECTORY ${OPHOST_NAME}_opapi_obj
  PROPERTIES GENERATED TRUE)
target_sources(${OPHOST_NAME}_opapi_obj PRIVATE
  "${CANNBOTDSL_PROVIDER_ACLNN_REGISTRY_SOURCE}"
  "${CANNBOTDSL_PROVIDER_ACLNN_ADAPTER_SOURCE}"
  "${CMAKE_CURRENT_LIST_DIR}/custom_flash_kda.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/flash_kda_l0.cpp")
add_dependencies(${OPHOST_NAME}_opapi_obj
  cannbotdsl_provider_aclnn_registry
  cannbotdsl_flash_kda_variant_policy)
target_include_directories(${OPHOST_NAME}_opapi_obj BEFORE PRIVATE
  "${CANNBOTDSL_PROVIDER_OPBASE_SOURCE_DIR}/include/nnopbase"
  "${_cannbotdsl_registry_dir}"
  "${CMAKE_CURRENT_LIST_DIR}")
target_link_libraries(cust_opapi PRIVATE
  "${CANNBOTDSL_PROVIDER_OPBASE_LIBRARY}")

install(FILES
  "${CMAKE_CURRENT_LIST_DIR}/aclnn_custom_flash_kda.h"
  "${CMAKE_CURRENT_LIST_DIR}/aclnn_flash_kda.h"
  DESTINATION "${ACLNN_INC_INSTALL_DIR}")

option(CANNBOTDSL_BUILD_FLASH_KDA_ACLNN_SMOKE
  "Build the FlashKDA ACLNN device smoke executable" OFF)
if(CANNBOTDSL_BUILD_FLASH_KDA_ACLNN_SMOKE)
  find_library(_cannbotdsl_ascendcl_library ascendcl
    PATHS
      "${ASCEND_CANN_PACKAGE_PATH}/lib64"
      "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/lib64"
    NO_DEFAULT_PATH REQUIRED)
  add_executable(cannbotdsl_flash_kda_aclnn_smoke
    "${CMAKE_CURRENT_LIST_DIR}/tests/flash_kda_aclnn_smoke.cpp")
  add_dependencies(cannbotdsl_flash_kda_aclnn_smoke cust_opapi)
  target_compile_features(cannbotdsl_flash_kda_aclnn_smoke PRIVATE cxx_std_17)
  target_include_directories(cannbotdsl_flash_kda_aclnn_smoke PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}"
    "${_cannbotdsl_registry_dir}"
    "${CANNBOTDSL_PROVIDER_OPBASE_SOURCE_DIR}/include/aclnnop"
    "${ASCEND_CANN_PACKAGE_PATH}/include"
    "${ASCEND_CANN_PACKAGE_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include")
  target_link_libraries(cannbotdsl_flash_kda_aclnn_smoke PRIVATE
    cust_opapi
    opapi_math
    "${CANNBOTDSL_PROVIDER_OPBASE_LIBRARY}"
    "${_cannbotdsl_ascendcl_library}")

  set(CANNBOTDSL_FLASH_KDA_ACCEPTANCE_RUN "" CACHE FILEPATH
    "Final mixed .run package consumed by the FlashKDA device acceptance target")
  set(CANNBOTDSL_FLASH_KDA_ACCEPTANCE_REPORT
    "${CMAKE_BINARY_DIR}/flash_kda_acceptance.json" CACHE FILEPATH
    "Machine-readable FlashKDA device acceptance report")
  if(CANNBOTDSL_FLASH_KDA_ACCEPTANCE_RUN)
    if(NOT IS_ABSOLUTE "${CANNBOTDSL_FLASH_KDA_ACCEPTANCE_RUN}" OR
       NOT IS_ABSOLUTE "${CANNBOTDSL_FLASH_KDA_ACCEPTANCE_REPORT}")
      message(FATAL_ERROR
        "FlashKDA acceptance .run and report paths must be absolute")
    endif()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_custom_target(cannbotdsl_flash_kda_acceptance
      COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_LIST_DIR}/tests/run_flash_kda_acceptance.py"
        --run "${CANNBOTDSL_FLASH_KDA_ACCEPTANCE_RUN}"
        --smoke "$<TARGET_FILE:cannbotdsl_flash_kda_aclnn_smoke>"
        --opbase-library "${CANNBOTDSL_PROVIDER_OPBASE_LIBRARY}"
        --cannbotdsl-source "${CANNBOTDSL_SOURCE_DIR}"
        --runtime-root "${CANNBOTDSL_RUNTIME_ROOT}"
        --library-dir "$<TARGET_FILE_DIR:opapi_math>"
        --report "${CANNBOTDSL_FLASH_KDA_ACCEPTANCE_REPORT}"
        --expected-traditional-kernel-count 8
      DEPENDS cannbotdsl_flash_kda_aclnn_smoke
      COMMENT "Running final FlashKDA mixed-package device acceptance"
      USES_TERMINAL
      VERBATIM)
  endif()
endif()
