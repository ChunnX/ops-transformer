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
  sync_fix.sh --inbox-repo DIR [--incident ID]

Pulls the from-server-a branch and displays the newest response, or the
response matching the supplied incident ID. It does not modify source code.
EOF
}

die()
{
    echo "[ERROR] $*" >&2
    exit 1
}

main()
{
    local inbox_repo=""
    local incident=""
    while [[ "$#" -gt 0 ]]; do
        case "$1" in
            --inbox-repo) [[ -n "${2:-}" ]] || die "$1 requires a value"; inbox_repo="$2"; shift 2 ;;
            --incident) [[ -n "${2:-}" ]] || die "$1 requires a value"; incident="$2"; shift 2 ;;
            -h | --help) usage; exit 0 ;;
            *) die "unknown option: $1" ;;
        esac
    done
    [[ -d "$inbox_repo/.git" ]] || die "inbox repository not found: $inbox_repo"
    [[ "$(git -C "$inbox_repo" branch --show-current)" == "from-server-a" ]] || die "inbox repository must be on from-server-a"
    [[ -z "$(git -C "$inbox_repo" status --porcelain)" ]] || die "inbox repository has uncommitted changes"
    git -C "$inbox_repo" pull --ff-only origin from-server-a

    local response=""
    if [[ -n "$incident" ]]; then
        response="$(find "$inbox_repo/responses/server-a" -type f -name "*$incident*.md" -print 2>/dev/null | sort | tail -1)"
    else
        response="$(find "$inbox_repo/responses/server-a" -type f -name '*.md' -print 2>/dev/null | sort | tail -1)"
    fi
    [[ -n "$response" ]] || die "no matching response found"
    echo "[INFO] response: $response"
    cat "$response"
}

main "$@"
