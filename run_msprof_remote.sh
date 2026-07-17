#!/bin/bash
source /home/tjj/setup_env.sh
LABEL=$1
if [ -z "$LABEL" ]; then LABEL="default"; fi
OUTDIR=/home/tjj/prof_paramset_$LABEL
mkdir -p $OUTDIR
rm -rf $OUTDIR/*
/home/tjj/0623/cann-9.0.0/tools/profiler/bin/msprof --output=$OUTDIR --aic-metrics=PipeUtilization --export=on --summary-format=csv --type=text --ai-core=on /usr/bin/python3 /home/tjj/run_bench_paramset.py 2>&1 | tail -3
echo "---"
python3 /home/tjj/parse_csv.py $OUTDIR
