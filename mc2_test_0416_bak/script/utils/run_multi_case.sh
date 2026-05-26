export ASCEND_GLOBAL_LOG_LEVEL=3;
export ASCEND_SLOG_PRINT_TO_STDOUT=1;
#rm -rf result.csv
#rm -rf result_1.csv
xlsx_name=$1
sheet=$2
start=$3
end=$4

for i in $(seq $start $end)
do
    case_name=$(python3 get_case_name.py $xlsx_name $sheet $i )
    echo "当前运行用例：" $case_name
    python3 mc2_test.py --xlsx $xlsx_name --sheet $sheet --num $i --scene cpu> cpu_$case_name.log 2>&1
    python3 mc2_test.py --xlsx $xlsx_name --sheet $sheet --num $i --scene npu> npu_$case_name.log 2>&1
    if [ $? -ne 0 ];then
    echo npu_$case_name.log |tee -a  break.txt
    ps -ef|grep python| awk '{print $2}'|xargs kill -9
    sleep 5
    else
    if [[ `grep "ERROR" cpu_$case_name.log` ]];then
    echo "---save log -------"
    else
    rm -rf  cpu_$case_name.log
    fi
    if [[ `grep -i "ERROR" npu_$case_name.log` ]];then
    echo "---save log -------"
    else
    rm -rf npu_$case_name.log
    fi
    fi

done
