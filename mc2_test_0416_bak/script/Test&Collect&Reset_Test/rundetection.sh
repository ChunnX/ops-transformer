source /usr/local/Ascend/latest/bin/setenv.bash
source /home/anaconda3/etc/profile.d/conda.sh
# conda activate deepseek_py38
conda activate py38
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_DETERMINISTIC=True
export HCCL_RDMA_TC=100
export HCCL_RDMA_SL=3
export HCCL_IF_BASE_PORT=65100
export HCCL_EXEC_TIMEOUT=1800
export CPU_AFFINITY_CONF=1

# 激活conda
# __conda_setup="$('/home/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)";
# eval "$__conda_setup";
# conda activate py38

# 配置HCCL
export HCCL_WHITELIST_DISABLE=1

LOCAL_HOST=`hostname -I|awk -F " " '{print$1}'`       # 获取本节点IP
export HCCL_IF_IP=$LOCAL_HOST
echo "hccl_if _ip:"$HCCL_IF_IP

IPs=('100.102.180.164'
'100.102.180.168'
  )

MA_NUM_HOSTS=${#IPs[@]}                           # 节点数量
export MASTER_ADDR=${IPs[0]}                      # 主节点IP
export MASTER_PORT=6038                           # 主节点port
VC_TASK_INDEX=0                                   # 本节点index

# 获取每个节点的index
for i in "${!IPs[@]}";
do
    echo "LOCAL_HOST=${LOCAL_HOST}, IPs[${i}]=${IPs[$i]}"
    if [ "$LOCAL_HOST" == "${IPs[$i]}" ]; then
        echo "Node Rank : ${i}"
        VC_TASK_INDEX=$i
        break
    fi
done

server_num=$1
case_file=$2
script_py=$3
begin=$4
end=$5

#将待验证的脚本放到case.txt中
rm -rf case.txt
if [[ -n $begin ]] && [[ -n $end ]];then
echo "run case between:"$begin $end 
case_file_1=$(sed -n "$begin,$end p" $case_file|tee case.txt)
else
echo "run all case in:"$case_file
case_file_1=$(cp $case_file case.txt)
fi


sed -i "s/^server_index = .*/server_index = ${VC_TASK_INDEX}/" $script_py
sed -i "s/^server_num = .*/server_num = ${server_num}/" $script_py
sed -i "s/^master_ip = .*/master_ip = '${MASTER_ADDR}'/" $script_py

pkill -9 python
while IFS= read -r line; do
    echo "$line"
    echo "$line" | awk 'BEGIN{OFS=",";FPAT="([^,]+)|(\"[^\"]+\")"}{}'
		case_name=$(echo "$line" | awk 'BEGIN{OFS=",";FPAT="([^,]+)|(\"[^\"]+\")"}{print $1}')
		case_name="\"$case_name\""
		numbers=$(echo "$line" | awk 'BEGIN{OFS=",";FPAT="([^,]+)|(\"[^\"]+\")"}{print $5}')
		numbers="${numbers//\"/}"


    echo "numbers: "$numbers


    escaped_x_shape=$(printf '%s\n' "$numbers" | sed -e 's/[\/&]/\\&/g')
    sed -i "s/^numbers = .*/shape = $escaped_x_shape/" $script_py
	sed -i "s/^casename = .*/casename = ${escaped_case_name}/" $script_py
    

done < case.txt
