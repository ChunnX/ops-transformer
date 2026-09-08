# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.

if (NOT ENABLE_BUILT_IN AND BUILD_OPEN_PROJECT)
    add_custom_target(modify_vendor ALL
            DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/scripts/install.sh ${CMAKE_CURRENT_BINARY_DIR}/scripts/upgrade.sh
    )

    # modify VENDOR_NAME in install.sh and upgrade.sh
    if (EXISTS ${ASCEND_PROJECT_DIR}/fwk_modules/scripts AND NOT TARGET ops_transformer_native)
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
endif ()
