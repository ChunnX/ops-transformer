#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# Align deb/rpm postinst with .run opp_install.sh (仓内特性权限 & cleanup).

sourcedir="${INSTALL_PATH}"
WHL_INSTALL_DIR_PATH="${sourcedir}/python/site-packages"
TARGET_OPP_BUILT_IN="${sourcedir}/opp/built-in"
unset PYTHONPATH
export PIP_BREAK_SYSTEM_PACKAGES=1

run_pip() { python3 -m pip "$@" || pip3 "$@"; }

# Same as opp_install.sh / gen_postinst root elevation: 550→555, 440→444, 750→755
if [ "${EUID:-0}" -eq 0 ]; then
    CUSTOM_PERM="755"
    BUILTIN_PERM="555"
    ONLYREAD_PERM="444"
else
    CUSTOM_PERM="750"
    BUILTIN_PERM="550"
    ONLYREAD_PERM="440"
fi

elevate_mod() {
    local mod="$1"
    if [ "${EUID:-0}" -eq 0 ]; then
        case "${mod}" in
            550) echo "555" ;;
            440) echo "444" ;;
            750) echo "755" ;;
            *) echo "${mod}" ;;
        esac
    else
        echo "${mod}"
    fi
}

# Mirror run install_common_parser: recursive chmod for every copy_entity in filelist
# (deb generated postinst only chmods the entity path itself / script files via find -type f).
apply_copy_entity_perms_from_filelist() {
    local filelist share_info_dir
    filelist=$(find "${sourcedir}/share/info" -path '*/script/filelist.csv' 2>/dev/null | head -1)
    [ -f "${filelist}" ] || return 0
    share_info_dir=$(dirname "$(dirname "${filelist}")")

    # CSV: module,operation,relative_path_in_pkg,relative_install_path,...,permission,...
    while IFS=',' read -r _ op _ relpath _ perm _; do
        [ "${op}" = "copy_entity" ] || continue
        [ -n "${relpath}" ] && [ "${relpath}" != "NA" ] || continue
        [ -e "${sourcedir}/${relpath}" ] || continue
        [ -n "${perm}" ] && [ "${perm}" != "NA" ] || continue
        chmod -R "$(elevate_mod "${perm}")" "${sourcedir}/${relpath}" 2>/dev/null || true
    done < <(tail -n +2 "${filelist}")

    # Match run opp_install: scene.info / version.info → ONLYREAD (same share/info package dir)
    chmod "${ONLYREAD_PERM}" "${share_info_dir}/scene.info" 2>/dev/null || true
    chmod "${ONLYREAD_PERM}" "${share_info_dir}/version.info" 2>/dev/null || true
}

# install es_* / cann_ops_* whl (same as install_es_whl_package + install_whl_package)
whl_dir="${sourcedir}/ops_transformer/es_packages/whl"
if [ -d "${whl_dir}" ]; then
    chmod u+w "${sourcedir}/python" 2>/dev/null || true
    chmod u+w "${WHL_INSTALL_DIR_PATH}" 2>/dev/null || true
    for whl in "${whl_dir}"/*.whl; do
        if [ -f "${whl}" ]; then
            echo "[ops-transformer] installing ${whl}"
            run_pip install --disable-pip-version-check --upgrade --no-deps --force-reinstall -t "${WHL_INSTALL_DIR_PATH}" "${whl}" \
                && rm -f "${whl}" || true
        fi
    done
fi

# Match run: CUSTOM_PERM on installed python packages (names follow pip layout, not fixed paths)
chmod -R "${CUSTOM_PERM}" "${WHL_INSTALL_DIR_PATH}"/es_transformer 2>/dev/null || true
chmod -R "${CUSTOM_PERM}" "${WHL_INSTALL_DIR_PATH}"/es_transformer-*.dist-info 2>/dev/null || true
chmod -R "${CUSTOM_PERM}" "${WHL_INSTALL_DIR_PATH}"/cann_ops_transformer 2>/dev/null || true
chmod -R "${CUSTOM_PERM}" "${WHL_INSTALL_DIR_PATH}"/cann_ops_transformer-*.dist-info 2>/dev/null || true

# Match run: remove ops_transformer delivery tree after whl install
if [ -d "${sourcedir}/ops_transformer" ]; then
    rm -rf "${sourcedir}/ops_transformer"
fi

# Match run add_init_py
built_in_impl_path="${TARGET_OPP_BUILT_IN}/op_impl/ai_core/tbe/impl/ops_transformer"
if [ -d "${built_in_impl_path}" ]; then
    opp_builtin_mod=$(stat -c %a "${built_in_impl_path}" 2>/dev/null || true)
    if [ "$(id -u)" != "0" ] && [ ! -w "${built_in_impl_path}" ]; then
        chmod u+w -R "${built_in_impl_path}" 2>/dev/null || true
    fi
    touch "${built_in_impl_path}/__init__.py" 2>/dev/null || true
    [ -d "${built_in_impl_path}/dynamic" ] && touch "${built_in_impl_path}/dynamic/__init__.py" 2>/dev/null || true
    if [ -n "${opp_builtin_mod}" ]; then
        chmod "${opp_builtin_mod}" -R "${built_in_impl_path}" 2>/dev/null || true
    fi
fi

apply_copy_entity_perms_from_filelist

# Match run:仓内特性 — root 刷 CUSTOM_PERM，非 root 刷 BUILTIN_PERM
if [ -d "${TARGET_OPP_BUILT_IN}" ]; then
    if [ "$(id -u)" = "0" ]; then
        chmod "${CUSTOM_PERM}" -R "${TARGET_OPP_BUILT_IN}" 2>/dev/null || true
    else
        chmod "${BUILTIN_PERM}" -R "${TARGET_OPP_BUILT_IN}" 2>/dev/null || true
    fi
fi
