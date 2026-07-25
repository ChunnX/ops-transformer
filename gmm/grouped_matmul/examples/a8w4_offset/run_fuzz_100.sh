#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
RUNNER="${SCRIPT_DIR}/build_install_run_custom.sh"
RESULT_ROOT="${A8W4_FUZZ_RESULT_DIR:-${REPO_ROOT}/build_out/a8w4_fuzz_100}"
MANIFEST="${RESULT_ROOT}/manifest.tsv"
MODE="${1:-all}"
ROUNDS="${A8W4_FUZZ_ROUNDS:-3}"
CASE_TIMEOUT="${A8W4_FUZZ_TIMEOUT:-300}"
GOLDEN_SAMPLES="${A8W4_GOLDEN_SAMPLES:-1024}"
CARDS_TEXT="${A8W4_FUZZ_CARDS:-0 1 2 3 4 5 6 7}"
CASE_FILTER="${A8W4_CASE_FILTER:-.*}"

case "${MODE}" in
  generate|run|all|summary) ;;
  *) echo "usage: $0 [generate|run|all|summary]" >&2; exit 2 ;;
esac

mkdir -p "${RESULT_ROOT}"

case_id=0
emit_case() {
  local bucket="$1" api="$2" format="$3" e="$4" m="$5" n="$6" k="$7"
  local data_mode="$8" scale_profile="$9" group_mode="${10}" seed="${11}"
  local carrier="${12:-INT4}" special_nz="${13:-0}"
  printf '%03d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${case_id}" "${bucket}" "${api}" "${format}" "${e}" "${m}" "${n}" "${k}" \
    "${data_mode}" "${scale_profile}" "${group_mode}" "${seed}" "${carrier}" "${special_nz}" >> "${MANIFEST}"
  case_id=$((case_id + 1))
}

generate_manifest() {
  : > "${MANIFEST}"
  printf 'case_id\tbucket\tapi\tformat\tE\tM\tN\tK\tdata_mode\tscale_profile\tgroup_mode\tseed\tcarrier\tspecial_nz\n' \
    >> "${MANIFEST}"

  # Exact network shapes. NZ shapes run through both public entries.
  emit_case net V5 ND       16  512 4096 7168 random varying balanced    2026072501
  emit_case net V5 ND       16  768 4096 7168 random tiers   alternating 2026072502
  emit_case net V5 ND       16 1024 7168 2048 random varying one-heavy  2026072503
  emit_case net V5 ND       17  544 4096 7168 random tiers   balanced    2026072504
  emit_case net V5 ND       17  816 4096 7168 row-varying varying random 2026072505
  emit_case net V5 ND       17 1088 7168 2048 random varying zipf        2026072506
  emit_case net V5       NZ 16 1024 4096 7168 random tiers   balanced    2026072507
  emit_case net WeightNz NZ 16 1024 4096 7168 random tiers   balanced    2026072507
  emit_case net V5       NZ 16 1024 7168 2048 random varying zero-expert 2026072508
  emit_case net WeightNz NZ 16 1024 7168 2048 random varying zero-expert 2026072508

  local m_values=(1 7 8 9 15 16 17 19 20 64 128 255 256 257 1024 4096 65536)
  local k_nd=(8 16 128 256 1024 2048 4096 7160 7168)
  local n_nd=(8 16 128 256 1024 2048 4096 7160 7168)
  local e_values=(1 2 4 8 16 17 64 128)
  local data_modes=(random pattern row-varying extreme)
  local scale_profiles=(varying tiers one)
  local group_modes=(balanced random one-heavy last-heavy zipf zero-expert alternating)

  # 22 ND pairwise-style boundary anchors.
  local i
  for ((i = 0; i < 17; ++i)); do
    emit_case boundary V5 ND \
      "${e_values[i % ${#e_values[@]}]}" "${m_values[i]}" "${n_nd[i % 4]}" \
      "${k_nd[i % ${#k_nd[@]}]}" "${data_modes[i % ${#data_modes[@]}]}" \
      "${scale_profiles[i % ${#scale_profiles[@]}]}" "${group_modes[i % ${#group_modes[@]}]}" \
      "$((2026072600 + i))"
  done
  for ((i = 0; i < 5; ++i)); do
    emit_case boundary V5 ND "${e_values[i]}" "${m_values[9 + i]}" "${n_nd[4 + i]}" \
      "${k_nd[i % 3]}" random "${scale_profiles[i % 3]}" "${group_modes[(i + 2) % ${#group_modes[@]}]}" \
      "$((2026072700 + i))"
  done

  # 16 NZ boundary anchors: every legal K/N appears, split across V5/WeightNz.
  local k_nz=(16 128 256 1024 2048 4096 7168)
  local n_nz=(64 128 256 1024 2048 4096 7168)
  for ((i = 0; i < 7; ++i)); do
    local api="V5"
    ((i % 2 == 1)) && api="WeightNz"
    emit_case boundary "${api}" NZ "${e_values[i]}" "${m_values[i + 1]}" 64 "${k_nz[i]}" \
      "${data_modes[i % 4]}" "${scale_profiles[i % 3]}" "${group_modes[i % ${#group_modes[@]}]}" \
      "$((2026072800 + i))"
  done
  for ((i = 0; i < 7; ++i)); do
    local api="WeightNz"
    ((i % 2 == 1)) && api="V5"
    emit_case boundary "${api}" NZ "${e_values[(i + 3) % ${#e_values[@]}]}" \
      "${m_values[i + 8]}" "${n_nz[i]}" 128 "${data_modes[(i + 1) % 4]}" \
      "${scale_profiles[(i + 1) % 3]}" "${group_modes[(i + 3) % ${#group_modes[@]}]}" \
      "$((2026072900 + i))"
  done
  emit_case boundary V5       NZ 17  4096 64 16 pattern varying alternating 2026072910
  emit_case boundary WeightNz NZ 128 65536 64 16 random  tiers   zero-expert 2026072911

  # Eight value anchors from ACC_P1_VAL_001..008.
  emit_case value V5       ND 2 8 16 512 zero        varying balanced 2026073001
  emit_case value V5       ND 2 8 16 512 one         varying balanced 2026073002
  emit_case value V5       ND 4 17 256 1024 random   varying random   2026073003
  emit_case value V5       ND 4 19 256 7168 extreme  varying alternating 2026073004
  emit_case value V5       NZ 8 64 256 2048 random   zero    zero-expert 2026073005
  emit_case value WeightNz NZ 8 64 256 2048 random   one     one-heavy 2026073006
  emit_case value V5       NZ 16 257 1024 4096 random tiers zipf      2026073007
  emit_case value WeightNz NZ 17 257 1024 4096 row-varying varying last-heavy 2026073008

  # 44 constrained large random cases.
  local large_m=(257 512 544 768 816 1024 1088 2048 4096)
  local large_e=(8 16 17 64 128)
  local large_k_nd=(1024 2048 4096 7160 7168)
  local large_n_nd=(256 1024 2048 4096 7160 7168)
  local large_k_nz=(1024 2048 4096 7168)
  local large_n_nz=(256 1024 2048 4096 7168)
  local fuzz_state=20260725
  local fuzz_value=0
  next_fuzz() {
    fuzz_state=$(((fuzz_state * 1103515245 + 12345) & 0x7fffffff))
    fuzz_value=$(((fuzz_state ^ (fuzz_state >> 16)) & 0x7fffffff))
  }
  declare -A seen_large=()
  add_large_cases() {
    local api="$1" format="$2" wanted="$3" added=0
    while ((added < wanted)); do
      next_fuzz; local e="${large_e[fuzz_value % ${#large_e[@]}]}"
      next_fuzz; local m="${large_m[fuzz_value % ${#large_m[@]}]}"
      local k n
      if [[ "${format}" == "NZ" ]]; then
        next_fuzz; k="${large_k_nz[fuzz_value % ${#large_k_nz[@]}]}"
        next_fuzz; n="${large_n_nz[fuzz_value % ${#large_n_nz[@]}]}"
      else
        next_fuzz; k="${large_k_nd[fuzz_value % ${#large_k_nd[@]}]}"
        next_fuzz; n="${large_n_nd[fuzz_value % ${#large_n_nd[@]}]}"
      fi
      local weight_bytes=$((e * k * n))
      local x_bytes=$((m * k))
      local out_bytes=$((m * n * 2))
      if ((weight_bytes > 536870912 || x_bytes > 536870912 || out_bytes > 268435456)); then
        continue
      fi
      local signature="${api}_${format}_${e}_${m}_${n}_${k}"
      [[ -n "${seen_large[${signature}]:-}" ]] && continue
      seen_large["${signature}"]=1
      next_fuzz; local data="${data_modes[fuzz_value % ${#data_modes[@]}]}"
      next_fuzz; local scale="${scale_profiles[fuzz_value % ${#scale_profiles[@]}]}"
      next_fuzz; local group="${group_modes[fuzz_value % ${#group_modes[@]}]}"
      emit_case large "${api}" "${format}" "${e}" "${m}" "${n}" "${k}" "${data}" "${scale}" "${group}" \
        "$((2026074000 + case_id))"
      added=$((added + 1))
    done
  }
  add_large_cases V5 ND 20
  add_large_cases V5 NZ 12
  add_large_cases WeightNz NZ 12

  if ((case_id != 100)); then
    echo "internal error: generated ${case_id} cases, expected 100" >&2
    exit 1
  fi
  # Overlay carrier/layout variants without changing the 100 logical configs.
  local variant_manifest
  variant_manifest="$(mktemp)"
  awk -F'\t' 'BEGIN{OFS="\t"}
    NR==1 { print; next }
    {
      if ($1 ~ /^(010|017|025|048|056|069|035|043|053|082)$/) $13="packed-INT32"
      if ($1 ~ /^(054|055|081|091)$/) $14=1
      print
    }' "${MANIFEST}" > "${variant_manifest}"
  mv "${variant_manifest}" "${MANIFEST}"
  echo "manifest=${MANIFEST}"
  echo "cases=${case_id}"
}

write_summary() {
  local summary="${RESULT_ROOT}/final_summary.tsv"
  local coverage="${RESULT_ROOT}/coverage.tsv"
  printf 'round\tcase_id\tbucket\tapi\tformat\tcarrier\tspecial_nz\tE\tM\tN\tK\tseed\tstatus\texit\tmax_abs_error\tnon_finite\tchecksum\tlog\tgolden\n' \
    > "${summary}"
  find "${RESULT_ROOT}" -mindepth 3 -maxdepth 6 -type f -name '*.status.tsv' -print0 2>/dev/null |
    sort -z | xargs -0 -r cat >> "${summary}"
  local total pass fail
  total="$(tail -n +2 "${summary}" | wc -l)"
  pass="$(awk -F'\t' 'NR>1 && $13=="PASS"{n++} END{print n+0}' "${summary}")"
  fail=$((total - pass))
  local determinism_mismatches
  determinism_mismatches="$(awk -F'\t' '
    NR>1 && $1==1 { seed[$2]=$12; checksum[$2]=$17 }
    NR>1 && $1==3 && (seed[$2]!=$12 || checksum[$2]!=$17) { mismatch++ }
    END { print mismatch+0 }' "${summary}")"
  printf 'dimension\tvalue\tcount\n' > "${coverage}"
  awk -F'\t' '
    NR>1 {
      count["bucket",$2]++
      count["route",$3 "/" $4]++
      count["E",$5]++
      count["M",$6]++
      count["N",$7]++
      count["K",$8]++
      count["data_mode",$9]++
      count["scale_profile",$10]++
      count["group_mode",$11]++
      count["carrier",$13]++
      count["special_nz",$14]++
    }
    END {
      for (key in count) {
        split(key, parts, SUBSEP)
        print parts[1] "\t" parts[2] "\t" count[key]
      }
    }' "${MANIFEST}" | sort -t$'\t' -k1,1 -k2,2V >> "${coverage}"
  echo "summary=${summary}"
  echo "coverage=${coverage}"
  echo "completed=${total}"
  echo "passed=${pass}"
  echo "failed=${fail}"
  echo "determinism_mismatches=${determinism_mismatches}"
}

run_worker() {
  local card="$1" shard="$2" shard_count="$3"
  mapfile -t lines < <(tail -n +2 "${MANIFEST}")
  local round
  for ((round = 1; round <= ROUNDS; ++round)); do
    local round_dir="${RESULT_ROOT}/round_${round}/card_${card}"
    mkdir -p "${round_dir}/logs" "${round_dir}/golden" "${round_dir}/status"
    local start=0 end="${#lines[@]}" step=1
    if ((round == 3)); then
      start=$((${#lines[@]} - 1))
      end=-1
      step=-1
    fi
    local index
    for ((index = start; index != end; index += step)); do
      IFS=$'\t' read -r id bucket api format e m n k data_mode scale_profile group_mode base_seed \
        carrier special_nz \
        <<< "${lines[index]}"
      ((10#${id} % shard_count == shard)) || continue
      [[ "${id}_${bucket}_${api}_${format}" =~ ${CASE_FILTER} ]] || continue
      local status_file="${round_dir}/status/${id}.status.tsv"
      [[ -s "${status_file}" ]] && continue
      local run_seed="${base_seed}"
      ((round == 2)) && run_seed=$((base_seed + 1000003))
      local log_path="${round_dir}/logs/${id}_${bucket}_${api}_${format}.log"
      local golden_path="${round_dir}/golden/${id}_${bucket}_${api}_${format}.csv"
      local weight_nz=0
      [[ "${format}" == "NZ" ]] && weight_nz=1
      echo "[RUN] round=${round} card=${card} case=${id} ${bucket} ${api}/${format} E=${e} M=${m} N=${n} K=${k}"
      set +e
      timeout --signal=INT --kill-after=5 "${CASE_TIMEOUT}s" env \
        ASCEND_RT_VISIBLE_DEVICES="${card}" \
        A8W4_API="${api}" A8W4_WEIGHT_NZ="${weight_nz}" \
        A8W4_RANDOM_DATA=1 A8W4_RANDOM_SEED="${run_seed}" \
        A8W4_E="${e}" A8W4_M="${m}" A8W4_N="${n}" A8W4_K="${k}" \
        A8W4_DATA_MODE="${data_mode}" A8W4_SCALE_PROFILE="${scale_profile}" \
        A8W4_GROUP_MODE="${group_mode}" A8W4_GOLDEN_SAMPLES="${GOLDEN_SAMPLES}" \
        A8W4_PACKED_INT32="$([[ "${carrier}" == "packed-INT32" ]] && echo 1 || echo 0)" \
        A8W4_SPECIAL_NZ="${special_nz}" \
        A8W4_GOLDEN_PATH="${golden_path}" A8W4_FAST_EXIT=1 \
        bash "${RUNNER}" run-only > "${log_path}" 2>&1
      local rc=$?
      set -e
      local status="FAIL"
      grep -qx 'PASS' "${log_path}" && [[ ${rc} -eq 0 ]] && status="PASS"
      local max_error non_finite checksum
      max_error="$(awk -F'max_abs_error=' '/ACLNN offset=non-null max_abs_error=/{print $2}' "${log_path}" | tail -n 1)"
      non_finite="$(awk -F'non_finite=' '/sample_count=/{split($2,a," "); print a[1]}' "${log_path}" | tail -n 1)"
      checksum="$(awk -F'output_checksum=' '/sample_count=/{print $2}' "${log_path}" | tail -n 1)"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${round}" "${id}" "${bucket}" "${api}" "${format}" "${carrier}" "${special_nz}" \
        "${e}" "${m}" "${n}" "${k}" "${run_seed}" \
        "${status}" "${rc}" "${max_error:--}" "${non_finite:--}" "${checksum:--}" "${log_path}" "${golden_path}" \
        > "${status_file}"
      echo "[${status}] round=${round} card=${card} case=${id} max_abs=${max_error:--} exit=${rc}"
    done
  done
}

run_all() {
  [[ -s "${MANIFEST}" ]] || generate_manifest
  bash "${RUNNER}" test-build
  read -r -a cards <<< "${CARDS_TEXT}"
  ((${#cards[@]} > 0)) || { echo "A8W4_FUZZ_CARDS is empty" >&2; exit 2; }
  local pids=()
  local shard
  for ((shard = 0; shard < ${#cards[@]}; ++shard)); do
    run_worker "${cards[shard]}" "${shard}" "${#cards[@]}" \
      > "${RESULT_ROOT}/worker_card_${cards[shard]}.log" 2>&1 &
    pids+=("$!")
  done
  local failed_workers=0 pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed_workers=$((failed_workers + 1))
  done
  write_summary
  if ((failed_workers > 0)); then
    echo "failed_workers=${failed_workers}" >&2
    return 1
  fi
}

case "${MODE}" in
  generate) generate_manifest ;;
  run) run_all ;;
  all) generate_manifest; run_all ;;
  summary) write_summary ;;
esac
