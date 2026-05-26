#!/bin/bash
case_file=$1
script_py=$2
begin=$3
end=$4
rm -rf case.txt
if [[ -n $begin ]] && [[ -n $end ]];then
echo "run case between:"$begin $end 
case_file_1=$(sed -n "$begin,$end p" $case_file|tee case.txt)
else
echo "run all case in:"$case_file
case_file_1=$(cp $case_file case.txt)
fi

for line in $(cat case.txt); do
      echo "---------line   "$line 
      IFS=',' read -r case_name bs att_worker_num ffn_worker_num micro_batch_num shared_expert_rank_num moe_expert_rank_num moe_expert_per_rank Y H k shared_expert_num expert_num_per_token x_dtype quant_mode graph_type is_ffn2attn worldsize is_performance is_fullcore core_num <<< "$line"
		  echo "case_name: "$case_name "bs: "$bs "att_worker_num: "$att_worker_num "ffn_worker_num: "$ffn_worker_num "micro_batch_num: "$micro_batch_num
      echo "shared_expert_rank_num: "$shared_expert_rank_num "moe_expert_rank_num: "$moe_expert_rank_num "moe_expert_per_rank:" $moe_expert_per_rank "Y: "$Y
      echo "h: "$H "k: "$k "shared_expert_num: "$shared_expert_num echo  "expert_num_per_token: "$expert_num_per_token "x_dtype: "$x_dtype
      echo "quant_mode: "$quant_mode "graph_type: "$graph_type "is_ffn2attn: "$is_ffn2attn "worldsize: "$worldsize
      echo "is_performance: "$is_performance "is_fullcore: "$is_fullcore "core_num: "$core_num



      sed -i "s/^current_case_name = .*/current_case_name = '${case_name}'/" $script_py
      sed -i "s/^bs = .*/bs = ${bs}/" $script_py

      sed -i "s/^shared_expert_num = .*/shared_expert_num = ${shared_expert_num}/" $script_py
      sed -i "s/^shared_expert_rank_num_ffn = .*/shared_expert_rank_num_ffn = ${shared_expert_rank_num}/" $script_py
      sed -i "s/^moe_rank_ffn_num = .*/moe_rank_ffn_num = ${moe_expert_rank_num}/" $script_py
      sed -i "s/^experts_per_rank = .*/experts_per_rank = ${moe_expert_per_rank}/" $script_py


      sed -i "s/^h = .*/h = ${H}/" $script_py
      sed -i "s/^k = .*/k = ${k}/" $script_py
      sed -i "s/^out_dtype = .*/out_dtype = ${x_dtype}/" $script_py
      sed -i "s/^is_ffn2attn = .*/is_ffn2attn = ${is_ffn2attn}/" $script_py
      sed -i "s/^graph_type = .*/graph_type = ${graph_type}/" $script_py
      sed -i "s/^quant_mode = .*/quant_mode = ${quant_mode}/" $script_py




      sed -i "s/^is_performance = .*/is_performance = ${is_performance}/" $script_py
      sed -i "s/^is_full_core = .*/is_full_core = ${is_fullcore}/" $script_py



      rm -rf /root/atc_data/*
      export HCCL_BUFFSIZE=6000
	    mkdir -p 16p_log

      timeout 240 python $script_py 2>&1 | tee 16p_log/${case_name}.log
      wait
      sleep 10
      pkill -9 python
    done
