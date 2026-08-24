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

# Incremental clang-format: only format changed lines vs HEAD.
# Used by pre-commit to avoid reformatting untouched legacy code.

set -euo pipefail

CLANG_FORMAT="${CLANG_FORMAT:-clang-format}"

if ! command -v "$CLANG_FORMAT" &>/dev/null; then
    echo "Error: $CLANG_FORMAT not found. Install clang-format or set CLANG_FORMAT env var." >&2
    exit 1
fi

changed=0

for f in "$@"; do
    [[ -f "$f" ]] || continue

    if ! git cat-file -e "HEAD:$f" 2>/dev/null; then
        # Newly added file: format the whole file.
        "$CLANG_FORMAT" --style=file -i "$f" 2>/dev/null || true
    else
        # Modified file: format only changed line ranges.
        mapfile -t RANGES < <(
            git diff HEAD -U0 -- "$f" 2>/dev/null |
            sed -nE 's/^@@ -[0-9]+(,[0-9]+)? \+([0-9]+)(,([0-9]+))? @@.*/\2 \4/p'
        )

        lines_args=()
        for r in "${RANGES[@]}"; do
            start=${r%% *}
            count=${r##* }
            [[ -z "$start" ]] && continue
            [[ -z "$count" ]] && count=1
            if [[ "$count" -gt 0 ]]; then
                end=$((start + count - 1))
                lines_args+=( --lines="${start}:${end}" )
            fi
        done

        if [[ ${#lines_args[@]} -gt 0 ]]; then
            "$CLANG_FORMAT" --style=file "${lines_args[@]}" -i "$f" 2>/dev/null || true
        fi
    fi

    if ! git diff --quiet -- "$f" 2>/dev/null; then
        echo "clang-format modified: $f"
        changed=1
    fi
done

exit $changed
