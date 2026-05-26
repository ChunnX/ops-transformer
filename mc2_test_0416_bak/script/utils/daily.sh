#!/bin/bash
xlsx_name=$1
sheet=$2
start=$3
end=$4

echo "连接device串口"
screen -r device0 -X quit
screen -r device1 -X quit
screen -r device2 -X quit
screen -r device3 -X quit
screen -S device0 -d -m telnet localhost 5001
screen -S device1 -d -m telnet localhost 5009
screen -S device2 -d -m telnet localhost 5017
screen -S device3 -d -m telnet localhost 5025
if [ -z $TESTCASE_SET_PATH ]; then
  TESTCASE_SET_PATH=/home/david610/mnt/mc2_test/bin_path
fi
echo "用例路径为："$TESTCASE_SET_PATH

function is_test_proc_exit(){
/usr/bin/expect <<-EOF
  set timeout 60
  spawn bash -c "telnet localhost 5000 > hosttmp.log"
  sleep 5
  send "\r"
  send "ps -ef | grep aarch | grep -v grep\r"
  expect "#"
EOF
  result=`grep aarch hosttmp.log | grep -v grep`
  if [[ -z "$result" ]]; then
    return 1 # not exit
  else
    return 0
  fi
}

function is_daily_sh_running(){
  sh_name=$0
  sh_name=${sh_name##*/}
  echo `ps -ef | grep "$sh_name"`
  result=`ps -ef | grep "$sh_name" | grep -v grep | grep -v $$ | grep -v $PPID | wc -l`
  echo $PPID
  echo $result
  if [[ "$result" -eq 0 ]]; then
    return 1
  else
    return 0
  fi
}

function is_has_output(){
  case_name=$1
  num=${case_name##*rank}
  num=${num%%_*}
  j=0
  output=0
  while [[ $j -lt $num ]]; do
    if [[ -f $TESTCASE_SET_PATH/$1/output/$j/output_npu.bin ]]; then
      let output++
    else
      break
    fi
    let j++
  done
  if [[ "$output" -eq $num ]]; then
    return 0  # test true
  else
    return 1
  fi
}

function wait_result(){
  k=0
  cnt=0
  echo `date`": start wait "$1
  while true; do
    let k++
    # if is_has_output $1; then
    #   echo `date`": has output file, finish wait "$1
    #   break
    # fi
    sleep 120
    if is_test_proc_exit ; then
      cnt=0
    else
      let cnt++
      echo `date`": no test program detected, cnt="$cnt
      if [[ $cnt -ge 15 ]]; then
        echo `date`": test proceess exit, finish wait "$1
        break
      fi
    fi
    let res=$k%10
    if [[ $res -eq 0 ]]; then
      echo `date`": waiting "$1
    fi
  done
}

function wait_test_proc_exit(){
  cnt=0
  echo `date`": start wait test proc exit, "$1
  while true; do
    if is_test_proc_exit ; then
      cnt=0
    else
      let cnt++
      if [[ $cnt -ge 15 ]]; then
        echo `date`": test proceess exit,finish wait "$1
        break
      fi
    fi
    sleep 20
  done
}

function remote_execute(){
case_name=$1
/usr/bin/expect <<-EOF
  set timeout 20
  spawn bash -c "telnet localhost 5000"
  sleep 5
  send "\r"
  expect "#"
  send "cd /mnt/mc2_test/bin_path\r"
  expect "#"
  send "./autotest.sh ${case_name}\r"
  expect "fastmodel case is end"
EOF
}

function clear_result(){
  case_name=$1
  if [[ ${case_name} == *"rank4"* ]];then
    rm $TESTCASE_SET_PATH/${case_name}/output/0/output_npu.bin
    rm $TESTCASE_SET_PATH/${case_name}/output/1/output_npu.bin
    rm $TESTCASE_SET_PATH/${case_name}/output/2/output_npu.bin
    rm $TESTCASE_SET_PATH/${case_name}/output/3/output_npu.bin
  else
    rm $TESTCASE_SET_PATH/${case_name}/output/0/output_npu.bin
    rm $TESTCASE_SET_PATH/${case_name}/output/1/output_npu.bin
  fi
}

function judge_result(){
  xlsx_name=$1
  sheet=$2
  index=$3
  echo > test.txt
  echo "check result of case number"$index
  # python3 ./data_compare/data_compare.py $TESTCASE_SET_PATH/${case_name} | tee -a test.txt
  python3 mc2_test.py --xlsx $xlsx_name --sheet $sheet --num $index --scene compare --soc 910D| tee -a test.txt
}

function run_case(){
# for case_name in `cat caselist.csv`
xlsx_name=$1
sheet=$2
start=$3
end=$4
for i in $(seq $start $end)
do
  case_name=$(python3 get_case_name.py $xlsx_name $sheet $i )
  if is_test_proc_exit ; then
    echo `date`": test process is running, skip "$case_name
    continue
  fi
  echo "清理用例历史output："$case_name
  clear_result $case_name
  echo "执行用例："$case_name
  remote_execute $case_name
  echo "等待用例执行结果："$case_name
  wait_result $case_name
  echo "精度比对："$case_name
  judge_result $xlsx_name $sheet $i
  echo "等待用例执行结束："$case_name
  wait_test_proc_exit $case_name
done
}

run_case $xlsx_name $sheet $start $end
echo `date`": "$0" exit"
