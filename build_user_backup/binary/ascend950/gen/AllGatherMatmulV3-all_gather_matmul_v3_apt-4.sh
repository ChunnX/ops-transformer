#!/bin/bash
echo "[ascend950] Generating AllGatherMatmulV3_127e133767c9499ed539712be3e1f3e0 ..."

PID_FILE="/home/developer/codes/ops-transformer/build/binary/ascend950/gen/compile_pids.txt"
STOP_FILE="/home/developer/codes/ops-transformer/build/binary/ascend950/gen/compile_stop.flag"

add_pid() {
    (
        flock 9
        echo $$ >> "$PID_FILE"
    ) 9>"${PID_FILE}.lock"
}

remove_pid() {
    (
        flock 9
        grep -v "^$$$" "$PID_FILE" > "${PID_FILE}.tmp" && mv "${PID_FILE}.tmp" "$PID_FILE"
    ) 9>"${PID_FILE}.lock"
}

check_stop() {
    if [ -f "$STOP_FILE" ]; then
        echo "Stop flag detected, exiting..."
        exit 1
    fi
}

on_failure() {
    touch "$STOP_FILE"

    # PIDPID
    (
        flock 9
        while read pid; do
            if [ "$pid" -ne $$ ]; then
                kill -9 "$pid" 2>/dev/null || true
            fi
        done < "$PID_FILE"
    ) 9>"${PID_FILE}.lock"
}

check_stop
add_pid
check_stop
start_time=$(date +%s.%N)
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=1

while true; do
  case "$1" in
    --kernel-src=*)
      export BUILD_KERNEL_SRC=$(echo "$1" | cut -d"=" -f2-)
      shift
      ;;
    -*)
      shift
      ;;
    *)
      break
      ;;
  esac
done
res=$(asc_opc $1 --main_func=all_gather_matmul_v3 --input_param=/home/developer/codes/ops-transformer/build/binary/ascend950/gen/AllGatherMatmulV3_127e133767c9499ed539712be3e1f3e0_param.json --soc_version=Ascend950PR_9599                  --output=$2 --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic )


    __ret=$?
    if [ $__ret -eq 0 ]; then
        echo "${res}"
        remove_pid
        
if ! test -f $2/AllGatherMatmulV3_127e133767c9499ed539712be3e1f3e0.json ; then
  echo "$2/AllGatherMatmulV3_127e133767c9499ed539712be3e1f3e0.json not generated!"
  exit 1
fi

        
if ! test -f $2/AllGatherMatmulV3_127e133767c9499ed539712be3e1f3e0.o ; then
  echo "$2/AllGatherMatmulV3_127e133767c9499ed539712be3e1f3e0.o not generated!"
  exit 1
fi

    else
        echo "${res}"
        on_failure
        exit 1
    fi
    end_time=$(date +%s.%N)
duration=$(awk "BEGIN { print $end_time - $start_time }")
kernel_path=$(find "/home/developer/codes/ops-transformer/build/binary/ascend950" -type f -name "AllGatherMatmulV3_127e133767c9499ed539712be3e1f3e0.o" -print | head -n 1)
kernel_size=$(stat -c%s "$kernel_path")
echo "Build [ascend950] op_name [all_gather_matmul_v3_apt] index [4] bin_file [AllGatherMatmulV3_127e133767c9499ed539712be3e1f3e0] duration [$duration] size [$kernel_size] start_time [$start_time] end_time [$end_time]" >> all_gather_matmul_v3_apt-4.txt
echo "[ascend950] Generating AllGatherMatmulV3_127e133767c9499ed539712be3e1f3e0 Done"
