#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
RUNNER="${SCRIPT_DIR}/build_install_run_custom.sh"
WEIGHT_FORMAT="${A8W4_WEIGHT_FORMAT:-ND}"
API="${A8W4_API:-V5}"
case "${API}" in
  V5|WeightNz) ;;
  *) echo "A8W4_API must be V5 or WeightNz, got: ${API}" >&2; exit 2 ;;
esac
DEQUANT_MODE="${A8W4_DEQUANT_MODE:-perchannel}"
case "${DEQUANT_MODE}" in
  perchannel|pergroup) ;;
  *) echo "A8W4_DEQUANT_MODE must be perchannel or pergroup, got: ${DEQUANT_MODE}" >&2; exit 2 ;;
esac
PERGROUP_K_MIN="${A8W4_PERGROUP_K_MIN:-256}"
PERGROUP_K_MAX="${A8W4_PERGROUP_K_MAX:-18432}"
if [[ "${DEQUANT_MODE}" == "pergroup" ]]; then
  if ((PERGROUP_K_MIN <= 0 || PERGROUP_K_MAX > 18432 ||
       PERGROUP_K_MIN > PERGROUP_K_MAX ||
       PERGROUP_K_MIN % 256 != 0 || PERGROUP_K_MAX % 256 != 0)); then
    echo "Per-group K range must satisfy 0 < MIN <= MAX <= 18432 and both bounds must be multiples of 256; got MIN=${PERGROUP_K_MIN}, MAX=${PERGROUP_K_MAX}" >&2
    exit 2
  fi
fi
case "${WEIGHT_FORMAT}" in
  ND) WEIGHT_NZ=0 ;;
  NZ) WEIGHT_NZ=1 ;;
  *) echo "A8W4_WEIGHT_FORMAT must be ND or NZ, got: ${WEIGHT_FORMAT}" >&2; exit 2 ;;
esac
if [[ "${API}" == "WeightNz" && "${WEIGHT_NZ}" != "1" ]]; then
  echo "A8W4_API=WeightNz requires A8W4_WEIGHT_FORMAT=NZ" >&2
  exit 2
fi
RESULT_ROOT="${A8W4_MATRIX_RESULT_DIR:-${REPO_ROOT}/build_out/a8w4_case_matrix_${API}_${WEIGHT_FORMAT}_${DEQUANT_MODE}}"
CASE_TIMEOUT="${A8W4_CASE_TIMEOUT:-45}"
SAVE_GOLDEN="${A8W4_SAVE_GOLDEN:-1}"
STRICT_EXIT="${A8W4_STRICT_EXIT:-0}"
FAST_EXIT="${A8W4_MATRIX_FAST_EXIT:-1}"
CASE_RETRIES="${A8W4_CASE_RETRIES:-1}"
CASE_FILTER="${A8W4_CASE_FILTER:-.*}"
RANDOM_ROUNDS="${A8W4_RANDOM_ROUNDS:-20}"
MATRIX_SEED="${A8W4_MATRIX_SEED:-20260723}"
INCLUDE_FIXED_CASES="${A8W4_INCLUDE_FIXED_CASES:-1}"
DRY_RUN="${A8W4_DRY_RUN:-0}"

# name E M N K scale-mode seed
CASES=()
if [[ "${INCLUDE_FIXED_CASES}" == "1" ]]; then
  if [[ "${WEIGHT_FORMAT}" == "NZ" ]]; then
    # ACLNN V5 requires INT4 NZ logical K/N to be aligned to 16/64.
    CASES+=(
      "baseline_e2_m8_n64_k512 2 8 64 512 varying 20260723"
      "m_tail_e3_m9_n64_k512 3 9 64 512 varying 20260724"
      "m_multitile_e5_m257_n64_k512 5 257 64 512 varying 20260725"
      "n128_e7_m14_n128_k512 7 14 128 512 varying 20260726"
      "n192_e4_m17_n192_k512 4 17 192 512 varying 20260727"
      "n_multitile_e8_m33_n320_k512 8 33 320 512 varying 20260728"
      "k_tail_e3_m13_n64_k1008 3 13 64 1008 varying 20260729"
      "k_tail_e12_m24_n64_k1040 12 24 64 1040 varying 20260730"
      "max_tile_e16_m256_n256_k512 16 256 256 512 varying 20260731"
      "fixed_scale_e5_m15_n64_k128 5 15 64 128 fixed-one 20260732"
    )
  else
    CASES+=(
      "baseline_e2_m8_n16_k512 2 8 16 512 varying 20260723"
      "m_tail_e3_m9_n16_k512 3 9 16 512 varying 20260724"
      "m_multitile_e5_m257_n16_k512 5 257 16 512 varying 20260725"
      "n_min_e7_m14_n8_k512 7 14 8 512 varying 20260726"
      "n_tail_e4_m17_n24_k512 4 17 24 512 varying 20260727"
      "n_tail_e8_m33_n248_k512 8 33 248 512 varying 20260728"
      "k_tail_e3_m13_n32_k1001 3 13 32 1001 varying 20260729"
      "k_tail_e12_m24_n64_k2049 12 24 64 2049 varying 20260730"
      "max_tile_e16_m256_n256_k512 16 256 256 512 varying 20260731"
      "fixed_scale_e5_m15_n32_k128 5 15 32 128 fixed-one 20260732"
    )
  fi
fi

expert_options=(2 3 4 5 7 8 12 16)
if [[ "${WEIGHT_FORMAT}" == "NZ" ]]; then
  n_options=(64 128 192 256 320 384 448 512)
else
  n_options=(8 16 24 32 40 64 72 120 128 136 248 256)
fi
if [[ "${DEQUANT_MODE}" != "pergroup" ]]; then
  if [[ "${WEIGHT_FORMAT}" == "NZ" ]]; then
    k_options=(128 144 240 256 272 496 512 528 768 1008 1024 1040 2048)
  else
    k_options=(127 128 255 256 257 511 512 513 768 1001 1023 1024 2049)
  fi
fi
random_state=$((MATRIX_SEED & 0x7fffffff))
next_random() {
  random_state=$(((random_state * 1103515245 + 12345) & 0x7fffffff))
  # LCG low bits have short periods. Mix high and low halves before selecting
  # dimensions so consecutive rounds do not repeat the same N/E pattern.
  random_value=$(((random_state ^ (random_state >> 16)) & 0x7fffffff))
}

for ((round = 0; round < RANDOM_ROUNDS; ++round)); do
  next_random
  e=${expert_options[random_value % ${#expert_options[@]}]}
  next_random
  if ((round % 7 == 0)); then
    m=$((257 + random_value % 48))
  else
    m=$((e + random_value % 97))
  fi
  next_random
  n=${n_options[random_value % ${#n_options[@]}]}
  next_random
  if [[ "${DEQUANT_MODE}" == "pergroup" ]]; then
    min_group_count=$((PERGROUP_K_MIN / 256))
    max_group_count=$((PERGROUP_K_MAX / 256))
    quant_group_count=$((min_group_count + random_value % (max_group_count - min_group_count + 1)))
    k=$((quant_group_count * 256))
  else
    k=${k_options[random_value % ${#k_options[@]}]}
  fi
  seed=$((MATRIX_SEED + 1000 + round))
  printf -v round_id '%03d' "${round}"
  CASES+=("random_r${round_id}_e${e}_m${m}_n${n}_k${k} ${e} ${m} ${n} ${k} varying ${seed}")
done

if [[ "${DEQUANT_MODE}" == "pergroup" ]]; then
  valid_cases=()
  for case_spec in "${CASES[@]}"; do
    read -r _ _ _ _ case_k _ _ <<< "${case_spec}"
    if ((case_k % 256 == 0 && case_k >= PERGROUP_K_MIN && case_k <= PERGROUP_K_MAX)); then
      valid_cases+=("${case_spec}")
    fi
  done
  CASES=("${valid_cases[@]}")
fi

mkdir -p "${RESULT_ROOT}/logs" "${RESULT_ROOT}/golden"
bash "${RUNNER}" test-build

CASE_MANIFEST="${RESULT_ROOT}/cases.tsv"
printf 'case\tapi\tformat\tE\tM\tN\tK\tscale\tseed\n' > "${CASE_MANIFEST}"
for case_spec in "${CASES[@]}"; do
  read -r name e m n k scale_mode seed <<< "${case_spec}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${name}" "${API}" "${WEIGHT_FORMAT}" "${e}" "${m}" "${n}" "${k}" "${scale_mode}" "${seed}" \
    >> "${CASE_MANIFEST}"
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "case_manifest=${CASE_MANIFEST}"
  exit 0
fi

SUMMARY="${RESULT_ROOT}/summary.tsv"
printf 'case\tapi\tformat\tE\tM\tN\tK\tscale\tseed\taccuracy\tmax_abs_error\tprocess_exit\tlog\tgolden\n' > "${SUMMARY}"

accuracy_failures=0
teardown_failures=0

for case_spec in "${CASES[@]}"; do
  read -r name e m n k scale_mode seed <<< "${case_spec}"
  if [[ ! "${name}" =~ ${CASE_FILTER} ]]; then
    continue
  fi
  log_path="${RESULT_ROOT}/logs/${name}.log"
  golden_path="${RESULT_ROOT}/golden/${name}.csv"
  fixed_scale=0
  [[ "${scale_mode}" == "fixed-one" ]] && fixed_scale=1

  case_env=(
    "A8W4_DEQUANT_MODE=${DEQUANT_MODE}"
    "A8W4_RANDOM_DATA=1"
    "A8W4_RANDOM_SEED=${seed}"
    "A8W4_E=${e}"
    "A8W4_M=${m}"
    "A8W4_N=${n}"
    "A8W4_K=${k}"
    "A8W4_WEIGHT_NZ=${WEIGHT_NZ}"
    "A8W4_API=${API}"
    "A8W4_FIXED_SCALE_ONE=${fixed_scale}"
    "A8W4_FAST_EXIT=${FAST_EXIT}"
  )
  if [[ "${SAVE_GOLDEN}" == "1" ]]; then
    case_env+=("A8W4_GOLDEN_PATH=${golden_path}")
  else
    golden_path="-"
  fi

  accuracy="FAIL"
  rc=1
  max_error=""
  for ((attempt = 1; attempt <= CASE_RETRIES + 1; ++attempt)); do
    echo "[RUN] ${name}: mode=${DEQUANT_MODE} api=${API} format=${WEIGHT_FORMAT} E=${e} M=${m} N=${n} K=${k} scale=${scale_mode} seed=${seed} attempt=${attempt}"
    set +e
    timeout --signal=INT --kill-after=3 "${CASE_TIMEOUT}s" \
      env "${case_env[@]}" bash "${RUNNER}" run-only > "${log_path}" 2>&1
    rc=$?
    set -e

    max_error="$(awk -F'max_abs_error=' '/ACLNN offset=(null|non-null) max_abs_error=/{print $2}' "${log_path}" | tail -n 1)"
    if grep -qx 'PASS' "${log_path}" && [[ -n "${max_error}" ]]; then
      accuracy="PASS"
      break
    fi
  done
  if [[ "${accuracy}" != "PASS" ]]; then
    accuracy_failures=$((accuracy_failures + 1))
  fi

  process_exit="clean"
  if [[ ${rc} -ne 0 ]]; then
    process_exit="exit-${rc}"
    teardown_failures=$((teardown_failures + 1))
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${name}" "${API}" "${WEIGHT_FORMAT}" "${e}" "${m}" "${n}" "${k}" "${scale_mode}" "${seed}" \
    "${accuracy}" "${max_error:--}" "${process_exit}" "${log_path}" "${golden_path}" >> "${SUMMARY}"
  echo "[${accuracy}] ${name}: max_abs_error=${max_error:--}, process=${process_exit}"
done

echo
echo "summary=${SUMMARY}"
echo "accuracy_failures=${accuracy_failures}"
echo "non_clean_process_exits=${teardown_failures}"

if [[ ${accuracy_failures} -ne 0 ]]; then
  exit 1
fi
if [[ "${STRICT_EXIT}" == "1" && ${teardown_failures} -ne 0 ]]; then
  exit 2
fi
