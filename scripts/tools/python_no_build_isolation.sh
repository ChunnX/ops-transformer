#!/usr/bin/env bash
# Wrapper used by CMake's Python3_EXECUTABLE. It keeps normal Python calls
# unchanged, but prevents local ES wheel builds from downloading build deps.

set -e

REAL_PYTHON="${OPS_REAL_PYTHON:-}"
if [ -z "${REAL_PYTHON}" ]; then
    if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python3" ]; then
        REAL_PYTHON="${VIRTUAL_ENV}/bin/python3"
    else
        REAL_PYTHON="$(command -v python3)"
    fi
fi

if [ "$#" -ge 3 ] && [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "wheel" ]; then
    shift 3
    HAS_NO_BUILD_ISOLATION="false"
    for arg in "$@"; do
        if [ "${arg}" = "--no-build-isolation" ]; then
            HAS_NO_BUILD_ISOLATION="true"
            break
        fi
    done

    if [ "${HAS_NO_BUILD_ISOLATION}" = "false" ]; then
        exec "${REAL_PYTHON}" -m pip wheel --no-build-isolation "$@"
    fi
    exec "${REAL_PYTHON}" -m pip wheel "$@"
fi

exec "${REAL_PYTHON}" "$@"
