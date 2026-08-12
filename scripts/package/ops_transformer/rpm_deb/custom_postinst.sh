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

# Same as opp_install.sh: root → 755/555, else 750/550
if [ "${EUID:-0}" -eq 0 ]; then
    CUSTOM_PERM="755"
    BUILTIN_PERM="555"
    ENTITY_PERM="555"
else
    CUSTOM_PERM="750"
    BUILTIN_PERM="550"
    ENTITY_PERM="550"
fi

# install es_transformer whl (same as install_es_whl_package)
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
    chmod -R "${CUSTOM_PERM}" "${WHL_INSTALL_DIR_PATH}"/es_transformer 2>/dev/null || true
    chmod -R "${CUSTOM_PERM}" "${WHL_INSTALL_DIR_PATH}"/es_transformer-*.dist-info 2>/dev/null || true
fi

# Match run install_whl_package: cann_ops_transformer CUSTOM_PERM
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

# Match run copy_entity recursive chmod (XML install_mod=550)
es_entity_dir="${sourcedir}/${PKG_ARCH_NAME}-linux/include/es/es_transformer"
if [ -d "${es_entity_dir}" ]; then
    chmod -R "${ENTITY_PERM}" "${es_entity_dir}" 2>/dev/null || true
fi

# Match run:仓内特性 — root 刷 CUSTOM_PERM，非 root 刷 BUILTIN_PERM
if [ -d "${TARGET_OPP_BUILT_IN}" ]; then
    if [ "$(id -u)" = "0" ]; then
        chmod "${CUSTOM_PERM}" -R "${TARGET_OPP_BUILT_IN}" 2>/dev/null || true
    else
        chmod "${BUILTIN_PERM}" -R "${TARGET_OPP_BUILT_IN}" 2>/dev/null || true
    fi
fi

# Match run: scene.info / version.info → ONLYREAD
if [ "${EUID:-0}" -eq 0 ]; then
    ONLYREAD_PERM="444"
else
    ONLYREAD_PERM="440"
fi
chmod "${ONLYREAD_PERM}" "${sourcedir}/share/info/ops_transformer/scene.info" 2>/dev/null || true
chmod "${ONLYREAD_PERM}" "${sourcedir}/share/info/ops_transformer/version.info" 2>/dev/null || true
