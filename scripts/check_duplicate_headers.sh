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
# 拦截「新增(A)/重命名(R)」头文件的 basename 重名（修改已有文件不查）。
#
# 文件收集（参考 oat_check.sh，按优先级）：
#   1) 命令行入参：pre-commit --files / pass_filenames 传入的路径
#      - 仅保留 .h/.hpp
#      - 相对比对基线不存在的路径视为新增/重命名目标：
#        * merge 且父对象存在：HEAD^1
#        * 已有 origin/<trunk>：merge-base
#        * 浅克隆无父提交：自动 fetch 目标分支后再判（仅此场景联网）
#        * 本地 commit hook：基线 = HEAD（提交尚未生成）
#   2) 回退：git diff --cached --diff-filter=AR（本地暂存区）
#
# 判定：find <仓库根> -name <basename>，非白名单命中数 > 1 则拦截
# 退出码：0 = 通过；1 = 发现重名
#
# 白名单：仓库相对路径前缀（不要加 './'）。命中则跳过；find 命中白名单不计数。
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

# 预留白名单路径前缀（相对仓库根）。示例：
WHITELIST_PATH_PREFIXES=(
  # "third_party/"
  # "3rdparty/"
  # "experimental/"
)

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "$ROOT" ]]; then
  echo "[dup-header] Not inside a git repository; skipping."
  exit 0
fi
cd "$ROOT"

is_whitelisted_path() {
  local rel="${1#./}"
  local prefix
  if [[ ${#WHITELIST_PATH_PREFIXES[@]} -eq 0 ]]; then
    return 1
  fi
  for prefix in "${WHITELIST_PATH_PREFIXES[@]}"; do
    [[ -z "$prefix" ]] && continue
    prefix="${prefix#./}"
    if [[ "$rel" == "$prefix"* ]]; then
      return 0
    fi
  done
  return 1
}

is_header_file() {
  local f="$1"
  [[ "$f" == *.h || "$f" == *.hpp ]]
}

# 比对基线（用于判断路径是新增还是修改），不做网络访问：
#   1) merge 且父树存在 -> HEAD^1
#   2) 与已有远端主干 / TARGET_BRANCH 的 merge-base
#   3) HEAD（本地 commit hook：提交尚未产生，新文件相对 HEAD 不存在）
resolve_compare_base() {
  local head mb b remote_b
  head="$(git rev-parse HEAD 2>/dev/null || true)"

  if git rev-parse -q --verify HEAD^2 >/dev/null 2>&1 \
     && git cat-file -e "HEAD^1^{tree}" 2>/dev/null; then
    printf '%s\n' "HEAD^1"
    return 0
  fi

  for b in \
      "${PRE_COMMIT_FROM_REF:-}" \
      "${TARGET_BRANCH:-}" \
      "${CI_MERGE_REQUEST_TARGET_BRANCH_NAME:-}" \
      "${GITHUB_BASE_REF:-}" \
      master main develop dev
  do
    [[ -z "$b" ]] && continue
    b="${b#refs/heads/}"
    b="${b#origin/}"
    for remote_b in "origin/${b}" "$b"; do
      if git rev-parse -q --verify "$remote_b" >/dev/null 2>&1; then
        mb="$(git merge-base HEAD "$remote_b" 2>/dev/null || true)"
        if [[ -n "$mb" && "$mb" != "$head" ]]; then
          printf '%s\n' "$mb"
          return 0
        fi
      fi
    done
  done

  printf '%s\n' "HEAD"
}

# 浅克隆兜底：拉取目标分支作为比对基线
try_fetch_compare_base() {
  local head mb fetch_ref
  head="$(git rev-parse HEAD 2>/dev/null || true)"
  fetch_ref="${TARGET_BRANCH:-${CI_MERGE_REQUEST_TARGET_BRANCH_NAME:-${GITHUB_BASE_REF:-master}}}"
  fetch_ref="${fetch_ref#refs/heads/}"
  fetch_ref="${fetch_ref#origin/}"

  # 诊断必须打到 stderr，避免被 $() 吃进 base ref
  echo "[dup-header] Shallow checkout has no usable parent/base; fetching origin/${fetch_ref} ..." >&2
  if ! git fetch --depth=1 origin "$fetch_ref" 2>/dev/null; then
    echo "[dup-header] WARNING: git fetch origin/${fetch_ref} failed; cannot distinguish Added vs Modified." >&2
    return 1
  fi
  if ! git rev-parse -q --verify FETCH_HEAD >/dev/null 2>&1; then
    return 1
  fi
  mb="$(git merge-base HEAD FETCH_HEAD 2>/dev/null || true)"
  if [[ -n "$mb" && "$mb" != "$head" ]]; then
    printf '%s\n' "$mb"
    return 0
  fi
  if git cat-file -e "FETCH_HEAD^{tree}" 2>/dev/null; then
    printf '%s\n' "FETCH_HEAD"
    return 0
  fi
  return 1
}

# 路径在 base 中是否已存在（已存在 => 视为修改，非 A/R）
path_exists_in_ref() {
  local ref="$1"
  local rel="$2"
  git cat-file -e "${ref}:${rel}" 2>/dev/null
}

basename_exists() {
  local name="$1"
  local hit
  hit="$(
    find "$ROOT" \( -path '*/.git' -o -path '*/.git/*' \) -prune -o \
      -type f -name "$name" -print -quit 2>/dev/null
  )"
  [[ -n "$hit" ]]
}

# 推荐唯一 basename：按路径段由浅到深加前缀
# 例：attention/common/op_kernel/buffer.h -> attention_buffer.h，
# 再试 attention_common_buffer.h、attention_common_op_kernel_buffer.h 等
suggest_unique_basename() {
  local rel="$1"
  local base stem ext dir IFS
  local -a parts=()
  local candidate prefix i

  base="$(basename "$rel")"
  if [[ "$base" == *.hpp ]]; then
    stem="${base%.hpp}"
    ext=".hpp"
  else
    stem="${base%.h}"
    ext=".h"
  fi

  dir="$(dirname "$rel")"
  if [[ "$dir" != "." ]]; then
    IFS='/' read -r -a parts <<< "$dir"
  fi

  # 从仓库相对路径顶层目录起逐步加长前缀
  prefix=""
  for ((i = 0; i < ${#parts[@]}; i++)); do
    if [[ -z "$prefix" ]]; then
      prefix="${parts[i]}"
    else
      prefix="${prefix}_${parts[i]}"
    fi
    candidate="${prefix}_${stem}${ext}"
    if ! basename_exists "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  # 回退：整段路径下划线拼接，仍冲突则加 _vN 后缀
  if [[ ${#parts[@]} -gt 0 ]]; then
    candidate="$(IFS=_; echo "${parts[*]}")_${stem}${ext}"
  else
    candidate="${stem}_unique${ext}"
  fi
  if ! basename_exists "$candidate"; then
    printf '%s\n' "$candidate"
    return 0
  fi

  i=2
  while basename_exists "${stem}_v${i}${ext}"; do
    i=$((i + 1))
  done
  printf '%s\n' "${stem}_v${i}${ext}"
}

# ---------------------------------------------------------------------------
# 收集待检文件（参考 oat_check.sh）
# ---------------------------------------------------------------------------
COMPARE_BASE="$(resolve_compare_base)"
COLLECT_MODE=""
CANDIDATES=()
ARGV_HEADERS=()

collect_from_argv_against_base() {
  local base="$1"
  local _f _rel
  CANDIDATES=()
  for _f in "${ARGV_HEADERS[@]}"; do
    _rel="${_f#./}"
    if path_exists_in_ref "$base" "$_rel"; then
      continue
    fi
    CANDIDATES+=("$_rel")
  done
}

if [[ $# -gt 0 ]]; then
  # 1) pre-commit --files / pass_filenames 入参（线上主路径；本地 commit 也会传）
  for _f in "$@"; do
    _rel="${_f#./}"
    case "$_rel" in
      /*) _rel="${_rel#"${ROOT}/"}" ;;
    esac
    is_header_file "$_rel" || continue
    ARGV_HEADERS+=("$_rel")
  done

  collect_from_argv_against_base "$COMPARE_BASE"

  # 浅克隆：入参文件都已在 HEAD 中，base=HEAD 无法区分 A/R -> 拉目标分支再判一次
  if [[ ${#CANDIDATES[@]} -eq 0 && ${#ARGV_HEADERS[@]} -gt 0 && "$COMPARE_BASE" == "HEAD" ]]; then
    _all_in_head=1
    for _rel in "${ARGV_HEADERS[@]}"; do
      if ! path_exists_in_ref "HEAD" "$_rel"; then
        _all_in_head=0
        break
      fi
    done
    if [[ "$_all_in_head" -eq 1 ]]; then
      _fetched_base="$(try_fetch_compare_base || true)"
      # 只保留最后一行有效 ref，防止异常输出污染
      _fetched_base="$(printf '%s\n' "${_fetched_base}" | sed '/^$/d' | tail -n 1)"
      if [[ -n "${_fetched_base}" ]]; then
        COMPARE_BASE="$_fetched_base"
        collect_from_argv_against_base "$COMPARE_BASE"
      fi
    fi
  fi

  COLLECT_MODE="argv (pre-commit --files / pass_filenames), base=${COMPARE_BASE}"
else
  # 2) 无入参时回退暂存区 AR（本地手动执行脚本）
  COLLECT_MODE="cached AR (git diff --cached)"
  mapfile -t CANDIDATES < <(
    git diff --cached --diff-filter=AR --name-only --relative -- '*.h' '*.hpp' \
      | sed '/^$/d' | sort -u
  )
fi

# 去重
if [[ ${#CANDIDATES[@]} -gt 0 ]]; then
  mapfile -t CANDIDATES < <(printf '%s\n' "${CANDIDATES[@]}" | sed '/^$/d' | sort -u)
fi

echo "[dup-header] Collect mode: ${COLLECT_MODE}"

if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
  echo "[dup-header] No Added/Renamed .h/.hpp files to check; skipping."
  exit 0
fi

FAIL=0
echo "[dup-header] Checking ${#CANDIDATES[@]} Added/Renamed header(s) for duplicate basenames..."
if [[ ${#WHITELIST_PATH_PREFIXES[@]} -gt 0 ]]; then
  echo "[dup-header] Whitelist prefixes: ${WHITELIST_PATH_PREFIXES[*]}"
fi

for rel in "${CANDIDATES[@]}"; do
  rel="${rel#./}"
  if is_whitelisted_path "$rel"; then
    echo "[dup-header] SKIP (whitelisted): ${rel}"
    continue
  fi

  base="$(basename "$rel")"
  # 排除 .git；白名单路径不计入重名集合
  mapfile -t HITS < <(
    find "$ROOT" \( -path '*/.git' -o -path '*/.git/*' \) -prune -o \
      -type f -name "$base" -print 2>/dev/null \
      | sed "s|^${ROOT}/||" \
      | sed 's|^\./||' \
      | sort -u \
      | while IFS= read -r p; do
          is_whitelisted_path "$p" && continue
          printf '%s\n' "$p"
        done
  )

  count="${#HITS[@]}"
  if [[ "$count" -gt 1 ]]; then
    suggested="$(suggest_unique_basename "$rel")"
    parent="$(dirname "$rel")"
    if [[ "$parent" == "." ]]; then
      new_path="${suggested}"
    else
      new_path="${parent}/${suggested}"
    fi
    echo "[dup-header] FAIL: duplicate header basename '${base}'"
    echo "[dup-header] Reason: Added/Renamed file '${rel}' introduces basename '${base}',"
    echo "[dup-header]         but find '${base}' returned ${count} non-whitelist paths:"
    printf '[dup-header]   - %s\n' "${HITS[@]}"
    echo "[dup-header] How to fix:"
    echo "[dup-header]   Naming guidance: prefer an operator/component name as prefix"
    echo "[dup-header]     (e.g. moe_init_routing_buffer.h), or an architecture as suffix"
    echo "[dup-header]     (e.g. buffer_arch35.h / buffer_arch22.h)."
    echo "[dup-header]   Recommended name: ${suggested}"
    echo "[dup-header]   Rename: ${rel}  =>  ${new_path}"
    echo "[dup-header]   Steps:"
    echo "[dup-header]     1. git mv '${rel}' '${new_path}'"
    echo "[dup-header]     2. Update #include and build references to use '${suggested}'"
    echo "[dup-header]     3. git add -u && bash scripts/check_duplicate_headers.sh"
    FAIL=1
  else
    echo "[dup-header] OK: ${rel}"
  fi
done

if [[ "$FAIL" -ne 0 ]]; then
  echo "[dup-header] Commit blocked: duplicate header basename(s) are not allowed."
  exit 1
fi

echo "[dup-header] Passed."
exit 0
