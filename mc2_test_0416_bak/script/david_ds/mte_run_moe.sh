#!/bin/bash
script_py=$1
case_file=$2
sheet_name=$3
begin=$4
end=$5
mkdir log/
rm -rf case.txt
rm -rf csv_result
python3 read_excel.py $case_file $sheet_name $begin $end

for line in $(cat case.txt); do
      echo "---------line   "$line
      IFS=',' read -r case_name	bs h k	x_dtype	expandx_dtype	quant_mode	graph_type is_quant_scales	is_static_shared_scales	is_continue	tp	localmoe	share_num	moe_num	is_full corenum comm_mode is_perf shared_expertx mask_dim  is_variable_bs shared_expert_num token_type commalg seed  <<< "$line"
      echo "case_name: "$case_name
      echo "bs: "$bs "k: "$k  "h: " $h "x_dtype: "$x_dtype "expandx_dtype: " $expandx_dtype
      echo "quant_mode: "$quant_mode "graph_type: "$graph_type  "is_quant_scales: "$is_quant_scales "is_static_shared_scales":$is_static_shared_scales
      echo "is_continue: "$is_continue "tp: "$tp "localmoe: "$localmoe "share_num: "$share_num
      echo "moe_num: "$moe_num  "seed: "$seed  "is_variable_bs: "$is_variable_bs "is_mask: "$is_mask 
      echo "commalg: "$commalg

      if [ "$is_continue" -eq 1 ]; then
          echo "is_continue true------"
          sed -i "s/^is_dispatch = .*/is_dispatch = 0/" $script_py
          sed -i "s/^is_combine = .*/is_combine = 0/" $script_py
          sed -i "s/^is_continue = .*/is_continue = 1/" $script_py
      elif [ "$is_continue" -eq 0 ]; then
          echo "is_continue false------"
          sed -i "s/^is_dispatch = .*/is_dispatch = 1/" $script_py
          sed -i "s/^is_combine = .*/is_combine = 1/" $script_py
          sed -i "s/^is_continue = .*/is_continue = 0/" $script_py
      else
          echo "wrong is_continue"
      fi

      sed -i "s/^bs = .*/bs = ${bs}/" $script_py
      sed -i "s/^k = .*/k = ${k}/" $script_py
      sed -i "s/^h = .*/h = ${h}/" $script_py
      sed -i "s/^comm_alg =.*/comm_alg =${commalg}/" $script_py
			sed -i "s/^x_dtype = .*/x_dtype = ${x_dtype}/" $script_py
      sed -i "s/^expandx_dtype = .*/expandx_dtype = ${expandx_dtype}/" $script_py
      sed -i "s/^quant_mode = .*/quant_mode = ${quant_mode}/" $script_py
			sed -i "s/^graph_type = .*/graph_type = ${graph_type}/" $script_py
      sed -i "s/^is_quant_scales = .*/is_quant_scales = ${is_quant_scales}/" $script_py
      sed -i "s/^if_full_core = .*/is_full_core = ${is_full}/" $script_py
			sed -i "s/^is_static_shared_scales = .*/is_static_shared_scales = ${is_static_shared_scales}/" $script_py
      sed -i "s/^current_case_name = .*/current_case_name = ${case_name}/" $script_py
      sed -i "s/^is_shared_expert_x = .*/is_shared_expert_x = ${shared_expertx}/" $script_py
      sed -i "s/^x_active_mask_dim = .*/x_active_mask_dim = ${mask_dim}/" $script_py
      sed -i "s/^is_variable_bs = .*/is_variable_bs = ${is_variable_bs}/" $script_py
      sed -i "s/^local_moe_expert_num = .*/local_moe_expert_num = ${localmoe}/" $script_py
      sed -i "s/^shared_expert_rank_num = .*/shared_expert_rank_num = ${share_num}/" $script_py
      sed -i "s/^shared_expert_num = .*/shared_expert_num = ${shared_expert_num}/" $script_py
      sed -i "s/^expert_token_nums_type = .*/expert_token_nums_type = ${token_type}/" $script_py
      sed -i "s/^random_seed = .*/random_seed = ${seed}/" $script_py

      new_case_name="${case_name#\'}"
      new_case_name="${new_case_name%\'}"
      timeout 240 python $script_py 2>&1 | tee log/${new_case_name}.log
      wait
      sleep 10
      #pkill -9 python
    done
#python3 parse_result.py dispatch_combine_result.csv
