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
      IFS=',' read -r case_name c_bs c_h c_k c_x_dtype c_quant c_graph scale c_is_continue  tp localmoe sharednum moeExpertNum fullcore core_num seed is_mask activeMask_Dim is_combine_x comm_mode prof isvb shared_num is_elastic share_broken_card_num moe_broken_card_num zero_expert_num copy_expert_num const_expert_num <<< "$line"
		echo "case_name: "$case_name "bs: "$c_bs "h: "$c_h "k: "$c_k "x_dtype: "$c_x_dtype "quant_mode: "$c_quant 
		echo  "graph_type: "$c_graph "is_scale: "$scale "is_continue":$c_is_continue 
      echo "tp: "$tp "localmoe: "$localmoe "shared_expert_rank_num: "$sharednum "moeExpertNum: "$moeExpertNum
      echo "fullcore: "$fullcore "core_num: "$core_num  "random_seed: "$seed "activeMask_Dim:" $activeMask_Dim
      echo "is_mask: "$is_mask "is_shared_expert_x: "$is_combine_x "comm_quant_mode: "$comm_mode "is_performance: "$prof
      echo "is_elastic:"$is_elastic  "share_broken_card_num:"$share_broken_card_num  "moe_broken_card_num:"$moe_broken_card_num
      echo "zero_expert_num:"$zero_expert_num "copy_expert_num:"$copy_expert_num "const_expert_num:"$const_expert_num
		# change value
     
     if [ "$c_is_continue" -eq 1 ]; then
         echo "is_continue true------"
         sed -i "s/^is_dispatch = .*/is_dispatch = 0/" $script_py
         sed -i "s/^is_combine = .*/is_combine = 0/" $script_py
         sed -i "s/^is_cascade = .*/is_cascade = 1/" $script_py
      elif [ "$c_is_continue" -eq 0 ]; then
			echo "is_continue false------"
         sed -i "s/^is_dispatch = .*/is_dispatch = 1/" $script_py
         sed -i "s/^is_combine = .*/is_combine = 0/" $script_py
         sed -i "s/^is_cascade = .*/is_cascade = 0/" $script_py
      else
         echo "wrong is_continue"
      fi

      sed -i "s/^bs = .*/bs = ${c_bs}/" $script_py
      sed -i "s/^h = .*/h = ${c_h}/" $script_py
      sed -i "s/^k = .*/k = ${c_k}/" $script_py
      sed -i "s/^quant_mode = .*/quant_mode = ${c_quant}/" $script_py
      sed -i "s/^out_dtype = .*/out_dtype = ${c_x_dtype}/" $script_py
      sed -i "s/^graph_type = .*/graph_type = ${c_graph}/" $script_py
      sed -i "s/^is_dispatch_scales = .*/is_dispatch_scales = ${scale}/" $script_py
      sed -i "s/^is_cascade = .*/is_cascade = ${c_is_continue}/" $script_py
      sed -i "s/^current_case_name = .*/current_case_name = '${case_name}'/" $script_py
      sed -i "s/^tp_world_size = .*/tp_world_size = ${tp}/" $script_py
      sed -i "s/^shared_expert_rank_num = .*/shared_expert_rank_num = ${sharednum}/" $script_py
      sed -i "s/^local_moe_expert_num = .*/local_moe_expert_num = ${localmoe}/" $script_py
      # sed -i "s/^server_num = .*/server_num = ${server_num}/" $script_py
      sed -i "s/^is_full_core = .*/is_full_core = ${fullcore}/" $script_py
      sed -i "s/^core_num = .*/core_num = ${core_num}/" $script_py
      sed -i "s/^random_seed = .*/random_seed = ${seed}/" $script_py
  #    sed -i "s/^is_mask = .*/is_mask = ${is_mask}/" $script_py
      sed -i "s/^is_shared_expert_x = .*/is_shared_expert_x = ${is_combine_x}/" $script_py
      sed -i "s/^comm_quant_mode = .*/comm_quant_mode = ${comm_mode}/" $script_py
      sed -i "s/^is_performance = .*/is_performance = ${prof}/" $script_py
      sed -i "s/^is_variable_bs = .*/is_variable_bs = ${isvb}/" $script_py
      sed -i "s/^shared_expert_num = .*/shared_expert_num = ${shared_num}/" $script_py

      # 新增缩容/特殊专家入参配置
      sed -i "s/^x_active_mask_dim = .*/x_active_mask_dim = ${activeMask_Dim}/" $script_py
      sed -i "s/^is_elastic = .*/is_elastic = ${is_elastic}/" $script_py
      sed -i "s/^elastic_shared_ranks = .*/elastic_shared_ranks = ${share_broken_card_num}/" $script_py
      sed -i "s/^elastic_moe_ranks = .*/elastic_moe_ranks = ${moe_broken_card_num}/" $script_py
      sed -i "s/^zero_expert_num = .*/zero_expert_num = ${zero_expert_num}/" $script_py
      sed -i "s/^copy_expert_num = .*/copy_expert_num = ${copy_expert_num}/" $script_py
      sed -i "s/^const_expert_num = .*/const_expert_num = ${const_expert_num}/" $script_py

      rm -rf /root/atc_data/*
      export HCCL_BUFFSIZE=6000

      timeout 240 python $script_py 2>&1 | tee 16p_log/${case_name}.log
      wait
      sleep 10
      pkill -9 python
    done
