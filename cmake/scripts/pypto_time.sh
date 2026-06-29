#!/bin/bash
# Wrapper: run a command with timing, print result to stdout.
# Usage: pypto_time.sh <op_file> <cmd...>
op_file="$1"
shift 1
T0=$(date +%s%3N)
"$@"
RC=$?
T1=$(date +%s%3N)
echo "[pypto-timing] ${op_file} compile.o: $(( T1 - T0 ))ms"
exit $RC
