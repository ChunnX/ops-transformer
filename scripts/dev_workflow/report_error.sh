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

usage()
{
    cat <<'EOF'
Usage:
  report_error.sh --results-repo DIR [--source-repo DIR] [--server-id ID] -- COMMAND [ARG ...]

Runs COMMAND, creates a sanitized diagnostic incident, commits it to the
from-server-b branch of the private results repository, and pushes it.
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

main()
{
    local results_repo=""
    local source_repo=""
    local server_id="server-b"
    local -a command=()

    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            --results-repo) need_value "$1" "${2:-}"; results_repo="$2"; shift 2 ;;
            --source-repo) need_value "$1" "${2:-}"; source_repo="$2"; shift 2 ;;
            --server-id) need_value "$1" "${2:-}"; server_id="$2"; shift 2 ;;
            --) shift; command=("$@"); break ;;
            -h | --help) usage; exit 0 ;;
            *) die "unknown option: $1" ;;
        esac
    done

    [[ -d "$results_repo/.git" ]] || die "results repository not found: $results_repo"
    [[ "${#command[@]}" -gt 0 ]] || die "a command is required after --"
    if [[ -z "$source_repo" ]]; then
        source_repo="$(git rev-parse --show-toplevel 2>/dev/null || true)"
    fi
    [[ -d "$source_repo/.git" ]] || die "source repository not found: $source_repo"

    local results_branch
    results_branch="$(git -C "$results_repo" branch --show-current)"
    [[ "$results_branch" == "from-server-b" ]] || die "results repository must be on from-server-b, current: $results_branch"
    [[ -z "$(git -C "$results_repo" status --porcelain)" ]] || die "results repository has uncommitted changes"
    git -C "$results_repo" pull --ff-only origin from-server-b

    local source_commit source_branch incident_id incident_dir
    source_commit="$(git -C "$source_repo" rev-parse HEAD)"
    source_branch="$(git -C "$source_repo" branch --show-current)"
    [[ -n "$source_branch" ]] || source_branch="DETACHED"
    incident_id="$(date +%Y%m%d-%H%M%S)-${source_commit:0:12}"
    incident_dir="$results_repo/incidents/$server_id/$incident_id"
    mkdir -p "$incident_dir"

    printf '%q ' "${command[@]}" > "$incident_dir/command.txt"
    printf '\n' >> "$incident_dir/command.txt"
    {
        echo "incident_id=$incident_id"
        echo "server_id=$server_id"
        echo "created_at=$(date --iso-8601=seconds)"
        echo "source_commit=$source_commit"
        echo "source_branch=$source_branch"
        echo "source_repo=$source_repo"
        echo "working_directory=$PWD"
        echo "architecture=$(uname -m)"
        echo "ascend_home_path=${ASCEND_HOME_PATH:-NOT_SET}"
    } > "$incident_dir/metadata.txt"
    git -C "$source_repo" status --short --branch > "$incident_dir/git-status.txt"
    git -C "$source_repo" diff --stat > "$incident_dir/git-diff-stat.txt"
    uname -a > "$incident_dir/uname.txt"
    if command -v npu-smi >/dev/null 2>&1; then
        npu-smi info > "$incident_dir/npu-info.txt" 2>&1 || true
    fi

    set +e
    "${command[@]}" > "$incident_dir/stdout.log" 2> "$incident_dir/stderr.log"
    local command_status=$?
    set -e
    printf '%s\n' "$command_status" > "$incident_dir/exit-code.txt"

    git -C "$source_repo" status --short --branch > "$incident_dir/git-status-after.txt"
    git -C "$source_repo" diff --stat > "$incident_dir/git-diff-stat-after.txt"
    git -C "$results_repo" add -- "incidents/$server_id/$incident_id"
    git -C "$results_repo" \
        -c user.name="$server_id" \
        -c user.email="$server_id@noreply.gitcode.com" \
        commit -m "incident: $server_id $incident_id"
    git -C "$results_repo" push origin from-server-b

    echo "[OK] incident published: $incident_id"
    echo "[INFO] command exit code: $command_status"
    echo "[INFO] local report: $incident_dir"
    return "$command_status"
}

main "$@"
