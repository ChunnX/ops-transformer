import itertools
import csv
# from allpairspy import AllPairs
import argparse
import random



# 返回一个能被world_size整除的包含0的list
def find_factors(world_size):
    factors = []
    for i in range(1, world_size):
        if world_size % i == 0:
            factors.append(i)
    factors.append(0)
    return factors


def cal_buff_size(_bs, h, k, ep_world_size, _local_moe_expert_num, shared_expert_num):
    # 计算需要的buffsize
    win_addr_align = 512
    scale_expand_idx_buffer = 44
    double_data_buffer = 2
    max_out_dtype_size = 2
    ub_align = 32
    token_actual_len = ((h * max_out_dtype_size + ub_align - 1) // ub_align) * ub_align + scale_expand_idx_buffer
    token_dispatch_need = ((token_actual_len + win_addr_align - 1) // win_addr_align) * win_addr_align
    token_combine_need = ((h * max_out_dtype_size + win_addr_align - 1) // win_addr_align) * win_addr_align
    actual_size = ((_bs * token_dispatch_need * ep_world_size * _local_moe_expert_num) + (
            _bs * token_combine_need * (k + shared_expert_num))) * double_data_buffer
    final_buffer_size = int(actual_size / 1024 / 1024) + 1
    return final_buffer_size


def gen_case(mode, server_num, loop, dev_num, op_type, is_cloud):
    world_size = server_num * dev_num
    #hccl_buffsize = 3 * 1024 * 1024 / 2 / world_size - 300  # HCCL_BUFFSIZE上限=3TB/(2*rank_size)
    csv_name = f"{mode}_{server_num}server_case_{op_type}.csv"
    combinations = []
    # 测试因子
    if mode == "ac":
        bs = [8, 12, 32]
    elif mode == "bc":
        bs = [8, 12, 16, 24, 32, 48, 96, 128, 256, 512]

    elif mode == "AttToFFN":
        count = 0
        while count < loop:
            token_X = 1
            bs = random.choice(list(range(1, 513)))
            H = random.choice(list(range(1024, 8193)))
            k = random.choice(list(range(1, 17)))
            x_dtype = random.choice([0, 1])  # 0:bf16 1:fp16
            is_attn2ffn = 1
            graph_type = random.choice([0, 1])
            quant_mode = random.choice([0, 2])
            is_mask = random.choice([0, 1])
            sync_flag = random.choice([0, 1])
            is_scale = random.choice([0, 1])
            world_size = server_num*dev_num

            rank_num_per_shared_expert = random.choice([1, 2, 3, 4])
            shared_expert_num = random.choice([0,1,2,3,4])
            moe_expert_rank_num = random.choice(list(range(1, world_size)))

            ffn_worker_num = moe_expert_rank_num + shared_expert_num*rank_num_per_shared_expert
            att_worker_num = world_size - ffn_worker_num

            if att_worker_num < 1 or att_worker_num >= world_size or ffn_worker_num < 1 or ffn_worker_num >= world_size:
                continue

            moe_expert_per_rank = random.choice(list(range(1, 129)))

            moe_expert_num = moe_expert_per_rank * moe_expert_rank_num

            if moe_expert_num < 1 or moe_expert_num > 1024:
                continue


            is_performance = random.choice([0, 1])
            if graph_type == 1:
                is_performance = 0

            is_fullcore = 1
            core_num = 24

            combinations.append([token_X,bs,H,k,x_dtype,is_attn2ffn,graph_type,quant_mode,is_mask,sync_flag,is_scale,world_size,att_worker_num,moe_expert_num,ffn_worker_num,moe_expert_per_rank,moe_expert_rank_num,
                                shared_expert_num,rank_num_per_shared_expert,is_performance,is_fullcore,core_num])

            title = ["case_name", "X","bs","h","k","x_dtype","is_attn2ffn","graph_type","quant_mode","is_mask","sync_flag","is_scale","world_size","att_worker_num","moe_expert_num","ffn_worker_num","moe_expert_per_rank",
                     "moe_expert_rank_num","shared_expert_num","rank_num_per_shared_expert","is_performance","is_fullcore","core_num"]

            count = count + 1




    else:
        print("wrong type ")
        exit()
    # 生成的用例写入csv中
    print(combinations)

    with open(csv_name, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(title)  # 写入表头
        for i, combination in enumerate(combinations, start=1):
            case_name = f"AttToFNN_fanhua_case_ID{str(i).zfill(4)}"
            writer.writerow([case_name] + list(combination))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", default="AttToFFN", help="AttToFFN/FFNtoAtt")
    parser.add_argument("-n", "--server_num", default=1, type=int, help="910C机器数")
    parser.add_argument("-l", "--loop", default=2000, type=int, help="生成泛化用例数")
    parser.add_argument("-d", "--dev_num", default=16, type=int, help="每台server使用的die数")
    parser.add_argument("-t", "--type", default="v1", type=str, help="支持 v1/v2/sdma")
    parser.add_argument("-c", "--cloud", default=0, type=int, help="0:本地批跑格式, 1:云道批跑格式")

    args = parser.parse_args()

    gen_case(args.mode, args.server_num, args.loop, args.dev_num, args.type, args.cloud)
