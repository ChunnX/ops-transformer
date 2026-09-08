# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#### CPACK to package run #####

function(pack_custom)
    add_cann_third_party(makeself-fetch)
    add_custom_target(modify_vendor ALL
            DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/scripts/install.sh ${CMAKE_CURRENT_BINARY_DIR}/scripts/upgrade.sh
    )

    # modify VENDOR_NAME in install.sh and upgrade.sh
    if (EXISTS ${ASCEND_PROJECT_DIR}/fwk_modules/scripts AND NOT NATIVE_ONLY)
        set(ASCEND_PROJECT_DIR_SCRIPTS_PATH ${ASCEND_PROJECT_DIR}/fwk_modules/scripts)
    else()
        set(ASCEND_PROJECT_DIR_SCRIPTS_PATH ${CMAKE_SOURCE_DIR}/cmake/scripts/custom)
    endif()
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/scripts/install.sh ${CMAKE_CURRENT_BINARY_DIR}/scripts/upgrade.sh
            COMMAND mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/scripts
            COMMAND cp -r ${ASCEND_PROJECT_DIR_SCRIPTS_PATH}/* ${CMAKE_CURRENT_BINARY_DIR}/scripts/
            COMMAND chmod +w ${CMAKE_CURRENT_BINARY_DIR}/scripts/*
            COMMAND sed -i "s/vendor_name=customize/vendor_name=${VENDOR_NAME}_transformer/g" ${CMAKE_CURRENT_BINARY_DIR}/scripts/*
    )

    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/scripts/
            DESTINATION . FILE_PERMISSIONS OWNER_EXECUTE OWNER_READ GROUP_READ
    )

    # gen version.info
    set(version_info_dir  ${CMAKE_CURRENT_BINARY_DIR})
    set(version_info_file ${version_info_dir}/version.info)
    add_custom_target(gen_version_info ALL
            DEPENDS ${version_info_file}
    )

    add_custom_command(OUTPUT ${version_info_file}
            COMMAND bash ${ASCENDC_CMAKE_UTIL_DIR}/gen_version_info.sh ${ASCEND_CANN_PACKAGE_PATH} ${version_info_dir}
    )

    install(FILES ${version_info_file}
            DESTINATION packages/vendors/${VENDOR_NAME}_transformer/
    )

    if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
        message(STATUS "Detected architecture: x86_64")
        set(ARCH x86_64)
    elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|arm")
        message(STATUS "Detected architecture: ARM64")
        set(ARCH aarch64)
    else ()
        message(WARNING "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    endif ()

    # CPack config
    set(CPACK_PACKAGE_NAME ${CMAKE_PROJECT_NAME})
    set(CPACK_PACKAGE_VERSION ${CMAKE_PROJECT_VERSION})
    set(CPACK_PACKAGE_DESCRIPTION "CPack ops project")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "CPack ops project")
    set(CPACK_PACKAGE_DIRECTORY ${CMAKE_BINARY_DIR})
    set(CPACK_PACKAGE_FILE_NAME "cann-ops-transformer-${VENDOR_NAME}_linux-${ARCH}.run")
    set(CPACK_GENERATOR External)
    set(CPACK_CMAKE_GENERATOR "Unix Makefiles")
    set(CPACK_EXTERNAL_ENABLE_STAGING TRUE)

    if (ENABLE_BUILD_PKG)
        set(CPACK_MAKESELF_PATH ${OPS_TRANSFORMER_DIR}/third_party/makeself)
        set(CPACK_EXTERNAL_PACKAGE_SCRIPT ${ASCEND_PROJECT_DIR}/fwk_modules/makeself.cmake)
    endif()
    set(CPACK_EXTERNAL_BUILT_PACKAGES ${CPACK_PACKAGE_DIRECTORY}/_CPack_Packages/Linux/External/${CPACK_PACKAGE_FILE_NAME}/${CPACK_PACKAGE_FILE_NAME})
    include(CPack)
endfunction()

function(pack_built_in)
  # mc2_matmul KB install
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/runtimeKB.cmake)
  #### built-in package ####
  message(STATUS "System processor: ${CMAKE_SYSTEM_PROCESSOR}")
  if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
      message(STATUS "Detected architecture: x86_64")
      set(ARCH x86_64)
  elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|arm")
      message(STATUS "Detected architecture: ARM64")
      set(ARCH aarch64)
  else ()
      message(WARNING "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
  endif ()

  set(script_prefix ${CMAKE_SOURCE_DIR}/scripts/package/ops_transformer/scripts)
  install(DIRECTORY ${script_prefix}/
      DESTINATION share/info/ops_transformer/script
      FILE_PERMISSIONS
      OWNER_READ OWNER_WRITE OWNER_EXECUTE  # 文件权限
      GROUP_READ GROUP_EXECUTE
      WORLD_READ WORLD_EXECUTE
      DIRECTORY_PERMISSIONS
      OWNER_READ OWNER_WRITE OWNER_EXECUTE  # 目录权限
      GROUP_READ GROUP_EXECUTE
      WORLD_READ WORLD_EXECUTE
      REGEX "(setenv|prereq_check)\\.(bash|fish|csh)" EXCLUDE
  )

  set(SCRIPTS_FILES
      ${CANN_CMAKE_DIR}/scripts/install/check_version_required.awk
      ${CANN_CMAKE_DIR}/scripts/install/common_func.inc
      ${CANN_CMAKE_DIR}/scripts/install/common_interface.sh
      ${CANN_CMAKE_DIR}/scripts/install/common_interface.csh
      ${CANN_CMAKE_DIR}/scripts/install/common_interface.fish
      ${CANN_CMAKE_DIR}/scripts/install/version_compatiable.inc
      ${CANN_CMAKE_DIR}/scripts/package/merge_binary_info_config.py
  )

  install(FILES ${SCRIPTS_FILES}
      DESTINATION share/info/ops_transformer/script
  )
  set(COMMON_FILES
      ${CANN_CMAKE_DIR}/scripts/install/install_common_parser.sh
      ${CANN_CMAKE_DIR}/scripts/install/common_func_v2.inc
      ${CANN_CMAKE_DIR}/scripts/install/common_installer.inc
      ${CANN_CMAKE_DIR}/scripts/install/script_operator.inc
      ${CANN_CMAKE_DIR}/scripts/install/version_cfg.inc
  )

  set(PACKAGE_FILES
      ${COMMON_FILES}
      ${CANN_CMAKE_DIR}/scripts/install/multi_version.inc
  )
  set(LATEST_MANGER_FILES
      ${COMMON_FILES}
      ${CANN_CMAKE_DIR}/scripts/install/common_func.inc
      ${CANN_CMAKE_DIR}/scripts/install/version_compatiable.inc
      ${CANN_CMAKE_DIR}/scripts/install/check_version_required.awk
  )
  set(CONF_FILES
      ${CANN_CMAKE_DIR}/scripts/package/cfg/path.cfg
  )
  foreach(components ${CANN_VERSION_PACKAGES})
      install(FILES ${CMAKE_BINARY_DIR}/version.${components}.info
          DESTINATION share/info/ops_transformer
          RENAME version.info
      )
  endforeach()
  install(FILES ${CONF_FILES}
      DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/conf
  )
  install(FILES ${PACKAGE_FILES}
      DESTINATION share/info/ops_transformer/script
  )

  if(NOT NATIVE_ONLY)
    # 打包 cann_ops_transformer whl 文件
    set(WHL_SOURCE_DIR "${CMAKE_SOURCE_DIR}/torch_extension/dist")
    file(GLOB WHL_FILES "${WHL_SOURCE_DIR}/cann_ops_transformer-*.whl")
    if(WHL_FILES)
        install(FILES ${WHL_FILES}
            DESTINATION ${WHL_INSTALL_DIR}/es_packages/whl
        )
        message(STATUS "Including whl package: ${WHL_FILES}")
    else()
        message(WARNING "Whl package not found in ${WHL_SOURCE_DIR}")
    endif()
  endif()

  string(FIND "${ASCEND_COMPUTE_UNIT}" ";" SEMICOLON_INDEX)
  if (SEMICOLON_INDEX GREATER -1)
      # 截取分号前的字串
      math(EXPR SUBSTRING_LENGTH "${SEMICOLON_INDEX}")
      string(SUBSTRING "${ASCEND_COMPUTE_UNIT}" 0 "${SUBSTRING_LENGTH}" compute_unit)
  else()
      # 没有分号取全部内容
      set(compute_unit "${ASCEND_COMPUTE_UNIT}")
  endif()

  message(STATUS "current compute_unit is: ${compute_unit}")

  # ============= CPack =============
  set_cann_cpack_config(ops-transformer COMPUTE_UNIT ${compute_unit} SHARE_INFO_NAME ops_transformer PACKAGE_TYPE "${PACKAGE_TYPE}")
endfunction()
