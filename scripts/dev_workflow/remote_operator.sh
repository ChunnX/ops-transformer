#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

readonly SCRIPT_NAME="$(basename "$0")"
readonly DEFAULT_CANN_ENV="/usr/local/Ascend/cann/set_env.sh"

usage()
{
    cat <<EOF
Usage:
  $SCRIPT_NAME check-env --output-dir DIR [--cann-env FILE]
  $SCRIPT_NAME verify --artifact-dir DIR
  $SCRIPT_NAME install --artifact-dir DIR [--package FILE] [--install-root DIR] [--cann-env FILE]
  $SCRIPT_NAME build --source-dir DIR --artifact-dir DIR [--custom-env FILE] [--cann-env FILE] -- COMMAND [ARG ...]
  $SCRIPT_NAME run --artifact-dir DIR --result-root DIR [--custom-env FILE] [--cann-env FILE] -- COMMAND [ARG ...]

Commands:
  check-env  Collect remote architecture, NPU, CANN and toolchain information.
  verify     Verify the checksums recorded in SHA256SUMS.
  install    Verify and install exactly one operator .run package.
  build      Build a runner on the remote server and save the log.
  run        Run a runner and save metadata, stdout, stderr and exit code.
EOF
}

die()
{
    echo "[ERROR] $*" >&2
    exit 1
}

need_value()
{
    [[ -n "${2:-}" ]] || die "$1 requires a value"
}

load_env()
{
    local cann_env="$1"
    local custom_env="$2"
    [[ -f "$cann_env" ]] || die "CANN environment script not found: $cann_env"
    # shellcheck disable=SC1090
    source "$cann_env"
    if [[ -n "$custom_env" ]]; then
        [[ -f "$custom_env" ]] || die "custom environment script not found: $custom_env"
        # shellcheck disable=SC1090
        source "$custom_env"
    fi
}

verify_artifact()
{
    local artifact_dir="$1"
    [[ -d "$artifact_dir" ]] || die "artifact directory not found: $artifact_dir"
    [[ -f "$artifact_dir/SHA256SUMS" ]] || die "SHA256SUMS not found in: $artifact_dir"
    (cd "$artifact_dir" && sha256sum -c SHA256SUMS)
}

select_package()
{
    local artifact_dir="$1"
    local requested="$2"
    local -a packages=()
    if [[ -n "$requested" ]]; then
        [[ "$requested" = /* ]] || requested="$artifact_dir/$requested"
        [[ -f "$requested" ]] || die "package not found: $requested"
        printf '%s\n' "$requested"
        return
    fi
    mapfile -t packages < <(find "$artifact_dir" -maxdepth 1 -type f -name '*.run' -print | sort)
    [[ "${#packages[@]}" -gt 0 ]] || die "no .run package found in: $artifact_dir"
    [[ "${#packages[@]}" -eq 1 ]] || die "multiple .run packages found; use --package"
    printf '%s\n' "${packages[0]}"
}

check_env()
{
    local output_dir="" cann_env="$DEFAULT_CANN_ENV"
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            --output-dir) need_value "$1" "${2:-}"; output_dir="$2"; shift 2 ;;
            --cann-env) need_value "$1" "${2:-}"; cann_env="$2"; shift 2 ;;
            *) die "unknown check-env option: $1" ;;
        esac
    done
    [[ -n "$output_dir" ]] || die "check-env requires --output-dir"
    mkdir -p "$output_dir"
    if [[ -f "$cann_env" ]]; then
        # shellcheck disable=SC1090
        source "$cann_env"
    else
        echo "[WARNING] CANN environment script not found: $cann_env" >&2
    fi
    local report="$output_dir/remote_environment.txt"
    {
        echo "===== DATE ====="; date --iso-8601=seconds
        echo "===== ARCHITECTURE ====="; uname -a
        echo "===== NPU ====="
        command -v npu-smi >/dev/null 2>&1 && npu-smi info || echo "npu-smi not found"
        echo "===== CANN ====="
        echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-NOT_SET}"
        env | sort | grep -E '^(ASCEND|CANN|LD_LIBRARY_PATH)=' || true
        echo "===== TOOLCHAIN ====="
        command -v gcc >/dev/null 2>&1 && gcc --version | head -1 || echo "gcc not found"
        command -v g++ >/dev/null 2>&1 && g++ --version | head -1 || echo "g++ not found"
        command -v cmake >/dev/null 2>&1 && cmake --version | head -1 || echo "cmake not found"
        command -v python3 >/dev/null 2>&1 && python3 --version || echo "python3 not found"
        echo "===== GIT ====="
        if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
            git rev-parse HEAD
            git status --short --branch
        else
            echo "current directory is not a Git worktree"
        fi
    } 2>&1 | tee "$report"
    echo "[OK] environment report: $report"
}

verify_cmd()
{
    local artifact_dir=""
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            --artifact-dir) need_value "$1" "${2:-}"; artifact_dir="$2"; shift 2 ;;
            *) die "unknown verify option: $1" ;;
        esac
    done
    [[ -n "$artifact_dir" ]] || die "verify requires --artifact-dir"
    verify_artifact "$artifact_dir"
    echo "[OK] artifact verification passed"
}

install_cmd()
{
    local artifact_dir="" package="" install_root="" cann_env="$DEFAULT_CANN_ENV"
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            --artifact-dir) need_value "$1" "${2:-}"; artifact_dir="$2"; shift 2 ;;
            --package) need_value "$1" "${2:-}"; package="$2"; shift 2 ;;
            --install-root) need_value "$1" "${2:-}"; install_root="$2"; shift 2 ;;
            --cann-env) need_value "$1" "${2:-}"; cann_env="$2"; shift 2 ;;
            *) die "unknown install option: $1" ;;
        esac
    done
    [[ -n "$artifact_dir" ]] || die "install requires --artifact-dir"
    load_env "$cann_env" ""
    verify_artifact "$artifact_dir"
    local package_path log_file="$artifact_dir/install.log"
    package_path="$(select_package "$artifact_dir" "$package")"
    chmod u+x "$package_path"
    if [[ -n "$install_root" ]]; then
        mkdir -p "$install_root"
        "$package_path" --install-path="$install_root" 2>&1 | tee "$log_file"
    else
        "$package_path" 2>&1 | tee "$log_file"
    fi
    echo "[OK] install log: $log_file"
}

build_cmd()
{
    local source_dir="" artifact_dir="" cann_env="$DEFAULT_CANN_ENV" custom_env=""
    local -a command=()
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            --source-dir) need_value "$1" "${2:-}"; source_dir="$2"; shift 2 ;;
            --artifact-dir) need_value "$1" "${2:-}"; artifact_dir="$2"; shift 2 ;;
            --cann-env) need_value "$1" "${2:-}"; cann_env="$2"; shift 2 ;;
            --custom-env) need_value "$1" "${2:-}"; custom_env="$2"; shift 2 ;;
            --) shift; command=("$@"); break ;;
            *) die "unknown build option: $1" ;;
        esac
    done
    [[ -d "$source_dir" ]] || die "source directory not found: $source_dir"
    [[ -d "$artifact_dir" ]] || die "artifact directory not found: $artifact_dir"
    [[ "${#command[@]}" -gt 0 ]] || die "build requires a command after --"
    load_env "$cann_env" "$custom_env"
    printf '%q ' "${command[@]}" > "$artifact_dir/runner-build-command.txt"
    printf '\n' >> "$artifact_dir/runner-build-command.txt"
    (cd "$source_dir" && "${command[@]}") 2>&1 | tee "$artifact_dir/runner-build.log"
}

run_cmd()
{
    local artifact_dir="" result_root="" cann_env="$DEFAULT_CANN_ENV" custom_env=""
    local -a command=()
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            --artifact-dir) need_value "$1" "${2:-}"; artifact_dir="$2"; shift 2 ;;
            --result-root) need_value "$1" "${2:-}"; result_root="$2"; shift 2 ;;
            --cann-env) need_value "$1" "${2:-}"; cann_env="$2"; shift 2 ;;
            --custom-env) need_value "$1" "${2:-}"; custom_env="$2"; shift 2 ;;
            --) shift; command=("$@"); break ;;
            *) die "unknown run option: $1" ;;
        esac
    done
    [[ -d "$artifact_dir" ]] || die "artifact directory not found: $artifact_dir"
    [[ -n "$result_root" ]] || die "run requires --result-root"
    [[ "${#command[@]}" -gt 0 ]] || die "run requires a command after --"
    load_env "$cann_env" "$custom_env"
    local commit="unknown"
    [[ ! -f "$artifact_dir/git-commit.txt" ]] || commit="$(tr -d '[:space:]' < "$artifact_dir/git-commit.txt")"
    local result_dir="$result_root/$(date +%Y%m%d-%H%M%S)-${commit:0:12}"
    mkdir -p "$result_dir"
    local name
    for name in git-commit.txt git-branch.txt build-info.txt build-architecture.txt SHA256SUMS; do
        [[ ! -f "$artifact_dir/$name" ]] || cp "$artifact_dir/$name" "$result_dir/$name"
    done
    printf '%q ' "${command[@]}" > "$result_dir/run-command.txt"; printf '\n' >> "$result_dir/run-command.txt"
    uname -m > "$result_dir/remote-architecture.txt"
    command -v npu-smi >/dev/null 2>&1 && npu-smi info > "$result_dir/npu-info.txt" 2>&1 || true
    env | sort | grep -E '^(ASCEND|CANN|LD_LIBRARY_PATH)=' > "$result_dir/runtime-env.txt" || true
    set +e
    "${command[@]}" > "$result_dir/stdout.log" 2> "$result_dir/stderr.log"
    local exit_code=$?
    set -e
    printf '%s\n' "$exit_code" > "$result_dir/exit-code.txt"
    echo "[INFO] result directory: $result_dir"
    echo "[INFO] command exit code: $exit_code"
    return "$exit_code"
}

main()
{
    [[ "$#" -gt 0 ]] || { usage; exit 1; }
    local command="$1"; shift
    case "$command" in
        check-env) check_env "$@" ;;
        verify) verify_cmd "$@" ;;
        install) install_cmd "$@" ;;
        build) build_cmd "$@" ;;
        run) run_cmd "$@" ;;
        help | --help | -h) usage ;;
        *) usage >&2; die "unknown command: $command" ;;
    esac
}

main "$@"
