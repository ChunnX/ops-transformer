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
readonly KEEP_RUNS=5

usage()
{
    cat <<EOF
Usage:
  $SCRIPT_NAME --commit FULL_SHA --task TASK_FILE [--repo REPO] [--run-root DIR]
  $SCRIPT_NAME --commit FULL_SHA --task TASK_FILE --resume-run-dir DIR [--repo REPO] [--run-root DIR]

Fetches and checks out an exact task commit, sources TASK_FILE from that commit,
builds and installs the operator, optionally runs its runner, archives logs,
then restores the original checkout and stashed local/untracked changes.

With --resume-run-dir, reuses that run installation and executes only the
runner while retaining the same checkout, logging and restore safeguards.
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

run_logged()
{
    local phase="$1"
    shift
    local log_file="$RUN_DIR/$phase.log"
    echo "[INFO] starting phase: $phase" | tee -a "$RUN_DIR/summary.log"
    set +e
    "$@" 2>&1 | tee "$log_file"
    local status=${PIPESTATUS[0]}
    set -e
    printf '%s\n' "$status" > "$RUN_DIR/$phase-exit-code.txt"
    echo "[INFO] phase $phase exit code: $status" | tee -a "$RUN_DIR/summary.log"
    return "$status"
}

restore_checkout()
{
    local task_status=$?
    trap - EXIT INT TERM
    set +e

    if [[ -n "${REPO:-}" && -d "$REPO/.git" ]]; then
        cd "$REPO" || true
        if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then
            git stash push -u -m "server-b-task-generated:${RUN_ID:-unknown}" >/dev/null
            echo "[WARNING] task-generated worktree changes were preserved in a separate stash" | tee -a "${RUN_DIR:-/dev/null}/summary.log"
        fi

        if [[ -n "${ORIGINAL_BRANCH:-}" ]]; then
            git switch "$ORIGINAL_BRANCH" >/dev/null 2>&1
        elif [[ -n "${ORIGINAL_HEAD:-}" ]]; then
            git switch --detach "$ORIGINAL_HEAD" >/dev/null 2>&1
        fi

        if [[ "${USER_STASHED:-0}" == "1" ]]; then
            if git stash apply "$USER_STASH_COMMIT"; then
                local user_stash_selector
                user_stash_selector="$(git stash list --format='%gd %H' | awk -v commit="$USER_STASH_COMMIT" '$2 == commit { print $1; exit }')"
                if [[ -n "$user_stash_selector" ]]; then
                    git stash drop "$user_stash_selector" >/dev/null
                fi
                echo "[OK] original local and untracked changes restored" | tee -a "${RUN_DIR:-/dev/null}/summary.log"
            else
                echo "[ERROR] stash restore conflicted; the stash was kept for manual recovery" | tee -a "${RUN_DIR:-/dev/null}/summary.log"
                task_status=1
            fi
        fi
    fi

    if [[ -n "${RUN_DIR:-}" && -d "$RUN_DIR" ]]; then
        printf '%s\n' "$task_status" > "$RUN_DIR/final-exit-code.txt"
        tar --exclude="$(basename "$RUN_DIR")/install" \
            -czf "$RUN_DIR.tar.gz" \
            -C "$(dirname "$RUN_DIR")" "$(basename "$RUN_DIR")"
        echo "[RESULT] log directory: $RUN_DIR"
        echo "[RESULT] diagnostic archive: $RUN_DIR.tar.gz"
        echo "[RESULT] show summary: cat '$RUN_DIR/summary.log'"
    fi

    prune_old_runs
    exit "$task_status"
}

prune_old_runs()
{
    [[ -n "${RUN_ROOT:-}" && -d "$RUN_ROOT" ]] || return 0
    local -a runs=()
    mapfile -t runs < <(find "$RUN_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'run-*' -printf '%T@ %p\n' | sort -rn | cut -d' ' -f2-)
    local index old_run
    for ((index = KEEP_RUNS; index < ${#runs[@]}; index++)); do
        old_run="${runs[$index]}"
        [[ "$old_run" == "$RUN_ROOT"/run-* ]] || continue
        rm -rf -- "$old_run"
        [[ ! -f "$old_run.tar.gz" ]] || rm -f -- "$old_run.tar.gz"
    done
}

main()
{
    local requested_commit=""
    local task_file=""
    local requested_repo=""
    local requested_run_root=""
    local requested_resume_run_dir=""

    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            --commit) need_value "$1" "${2:-}"; requested_commit="$2"; shift 2 ;;
            --task) need_value "$1" "${2:-}"; task_file="$2"; shift 2 ;;
            --repo) need_value "$1" "${2:-}"; requested_repo="$2"; shift 2 ;;
            --run-root) need_value "$1" "${2:-}"; requested_run_root="$2"; shift 2 ;;
            --resume-run-dir) need_value "$1" "${2:-}"; requested_resume_run_dir="$2"; shift 2 ;;
            -h | --help) usage; exit 0 ;;
            *) die "unknown option: $1" ;;
        esac
    done

    [[ "$requested_commit" =~ ^[0-9a-fA-F]{40}$ ]] || die "--commit must be a full 40-character commit ID"
    [[ -n "$task_file" ]] || die "--task is required"
    [[ "$task_file" != /* && "$task_file" != *".."* ]] || die "--task must be a safe repository-relative path"

    if [[ -n "$requested_repo" ]]; then
        REPO="$(cd "$requested_repo" && git rev-parse --show-toplevel)"
    else
        REPO="$(git rev-parse --show-toplevel 2>/dev/null || true)"
    fi
    [[ -d "$REPO/.git" ]] || die "ops-transformer repository not found"

    local repo_parent
    repo_parent="$(dirname "$REPO")"
    RUN_ROOT="${requested_run_root:-$repo_parent/task-runs}"
    [[ "$RUN_ROOT" = /* ]] || die "run root must be an absolute path"
    mkdir -p "$RUN_ROOT"

    local resume_install_root=""
    if [[ -n "$requested_resume_run_dir" ]]; then
        [[ "$requested_resume_run_dir" = /* ]] || die "resume run directory must be an absolute path"
        requested_resume_run_dir="$(realpath -e "$requested_resume_run_dir")"
        case "$requested_resume_run_dir" in
            "$RUN_ROOT"/run-*) ;;
            *) die "resume run directory must be an existing run under: $RUN_ROOT" ;;
        esac
        resume_install_root="$requested_resume_run_dir/install"
        [[ -d "$resume_install_root" ]] || die "resume install directory not found: $resume_install_root"
    fi

    cd "$REPO"
    ORIGINAL_HEAD="$(git rev-parse HEAD)"
    ORIGINAL_BRANCH="$(git branch --show-current)"
    USER_STASHED=0
    USER_STASH_COMMIT=""
    RUN_ID="run-$(date +%Y%m%d-%H%M%S)-${requested_commit:0:12}"
    RUN_DIR="$RUN_ROOT/$RUN_ID"
    mkdir -p "$RUN_DIR"
    trap restore_checkout EXIT INT TERM

    {
        echo "requested_commit=$requested_commit"
        echo "task_file=$task_file"
        echo "repository=$REPO"
        echo "original_head=$ORIGINAL_HEAD"
        echo "original_branch=${ORIGINAL_BRANCH:-DETACHED}"
        echo "started_at=$(date --iso-8601=seconds)"
        echo "architecture=$(uname -m)"
        echo "resume_run_dir=${requested_resume_run_dir:-NONE}"
    } > "$RUN_DIR/metadata.txt"
    git remote -v > "$RUN_DIR/git-remotes.txt"
    git status --short --branch > "$RUN_DIR/git-status-before.txt"

    if [[ -n "$(git status --porcelain)" ]]; then
        git stash push -u -m "server-b-task-user:$RUN_ID" | tee "$RUN_DIR/stash.log"
        USER_STASHED=1
        USER_STASH_COMMIT="$(git rev-parse refs/stash)"
    fi

    git fetch origin --prune 2>&1 | tee "$RUN_DIR/fetch.log"
    git cat-file -e "$requested_commit^{commit}" || die "commit not found after fetch: $requested_commit"
    git switch --detach "$requested_commit" 2>&1 | tee "$RUN_DIR/checkout.log"
    [[ "$(git rev-parse HEAD)" == "$requested_commit" ]] || die "checked out commit does not match request"
    [[ -f "$REPO/$task_file" ]] || die "task file not found in requested commit: $task_file"
    cp "$REPO/$task_file" "$RUN_DIR/task-file.sh"

    unset TASK_NAME OPERATOR SOC CANN_ENV TASK_HAS_RUNNER
    unset -f task_build task_select_package task_install task_run 2>/dev/null || true
    # The task file is trusted code committed by server A in the requested revision.
    # shellcheck disable=SC1090
    source "$REPO/$task_file"

    [[ -n "${TASK_NAME:-}" ]] || die "task file must set TASK_NAME"
    [[ -n "${OPERATOR:-}" ]] || die "task file must set OPERATOR"
    [[ -n "${SOC:-}" ]] || die "task file must set SOC"
    [[ -n "${CANN_ENV:-}" ]] || die "task file must set CANN_ENV"
    [[ "$SOC" == "ascend950" ]] || die "server B is Ascend950PR; task SOC must be ascend950"
    case "$CANN_ENV" in
        "/usr/local/Ascend/cann-9.2.0/set_env.sh" | \
        "/home/l00803537/CANN90/cann-9.0.0/set_env.sh") ;;
        *) die "task CANN environment is not allowlisted for server B: $CANN_ENV" ;;
    esac
    declare -F task_build >/dev/null || die "task file must define task_build"
    declare -F task_select_package >/dev/null || die "task file must define task_select_package"
    declare -F task_install >/dev/null || die "task file must define task_install"
    [[ "${TASK_HAS_RUNNER:-0}" =~ ^[01]$ ]] || die "TASK_HAS_RUNNER must be 0 or 1"
    if [[ "$TASK_HAS_RUNNER" == "1" ]]; then
        declare -F task_run >/dev/null || die "TASK_HAS_RUNNER=1 requires task_run"
    fi

    {
        echo "task_name=$TASK_NAME"
        echo "operator=$OPERATOR"
        echo "soc=$SOC"
        echo "cann_env=$CANN_ENV"
        echo "task_has_runner=$TASK_HAS_RUNNER"
    } >> "$RUN_DIR/metadata.txt"

    [[ -f "$CANN_ENV" ]] || die "CANN environment script not found: $CANN_ENV"
    # shellcheck disable=SC1090
    source "$CANN_ENV"
    echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-NOT_SET}" > "$RUN_DIR/cann-environment.txt"
    command -v npu-smi >/dev/null 2>&1 && npu-smi info > "$RUN_DIR/npu-info.txt" 2>&1 || true

    local install_root
    if [[ -n "$resume_install_root" ]]; then
        install_root="$resume_install_root"
        echo "[INFO] resuming runner with install directory: $install_root" | tee -a "$RUN_DIR/summary.log"
    else
        run_logged build task_build
        local package_path
        package_path="$(task_select_package)"
        [[ -f "$package_path" ]] || die "task_select_package did not return a package file: $package_path"
        sha256sum "$package_path" > "$RUN_DIR/package-sha256.txt"
        printf '%s\n' "$package_path" > "$RUN_DIR/package-path.txt"

        install_root="$RUN_DIR/install"
        mkdir -p "$install_root"
        run_logged install task_install "$package_path" "$install_root"
    fi

    if [[ "$TASK_HAS_RUNNER" == "1" ]]; then
        run_logged run task_run "$install_root"
        echo "[OK] build, install and runner completed" | tee -a "$RUN_DIR/summary.log"
    else
        echo "[NO_RUNNER] build and install succeeded; this task has no runnable example" | tee -a "$RUN_DIR/summary.log"
    fi
}

main "$@"
