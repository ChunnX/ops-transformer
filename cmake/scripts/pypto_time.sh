#!/bin/bash
# Wrapper: run a command with timing, append result to timing file.
# Usage:
#   pypto_time.sh <op_file> <timing_file> <cmd...>   — time a command
#   pypto_time.sh summary <compute_unit> <timing_file> — print summary
if [ "$1" = "summary" ]; then
    compute_unit="$2"
    timing_file="$3"
    echo "=== PyPTO Timing Summary ($compute_unit) ==="
    if [ -f "$timing_file" ]; then
        cat "$timing_file"
    else
        echo "(no timing data)"
    fi
    echo "============================================="
    exit 0
fi

op_file="$1"
timing_file="$2"
shift 2
T0=$(date +%s%3N)
"$@"
RC=$?
T1=$(date +%s%3N)
printf "%s\tcompile.o\t%dms\n" "$op_file" $(( T1 - T0 )) >> "$timing_file"
exit $RC
