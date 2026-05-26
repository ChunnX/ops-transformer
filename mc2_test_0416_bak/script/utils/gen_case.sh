xlsx_name=$1
sheet=$2
start=$3
end=$4

for i in $(seq $start $end)
do
	echo "生成cpu真值"
	python3 mc2_test.py --xlsx $xlsx_name --sheet $sheet --num $i --scene cpu
	#echo "生成输入及aclnn golden"
        #python3 mc2_test.py --xlsx $xlsx_name --sheet $sheet --num $i --scene aclnn_golden
	echo "生成输入及aclnn执行shell"
	python3 mc2_test.py --xlsx $xlsx_name --sheet $sheet --num $i --scene aclnn
done
