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

    csv_name = f"Reset{op_type}_{mode}_caselist_{loop}.csv"

    elastic = [0,1]
    timeout_flag = [0, 1]
    valid_list = [(-1.0, 1.0), (-5.0, 5.0), (-3.0, 3.0), (-10.0, 10.0), (-20.0, 20.0), (0.001, 0.01),
                  (-0.001, 0),
                  (-0.01, -0.001), (-100, 100), (-1000, 1000)]
    dtpye_list = ['torch.float32','torch.float16','torch.bfloat16','torch.int32','torch.int16','torch.int64','torch.int8','torch.uint8','torch.bool']
    graph_list =[0,1]
    performance_list=[0,1]
    broken_card_list=[0,1,2,3,4,5,6,7]
    numbers=[]
    combinations = []
    if mode =="fanhua":
        count = 1
        while count < loop+1:
            shape_dim = random.choice([x for x in range(16) if x != 0])
            x_shape = []
            product = 1
            for i in range(shape_dim):
                if count < 51:
                    random_shape = random.choice(list(range(1, 10)))
                elif count <101:
                    random_shape = random.choice(list(range(1, 20)))
                elif count <151:
                    random_shape = random.choice(list(range(1, 100)))
                elif count <201:
                    random_shape = random.choice(list(range(1, 1000)))
                else:
                    random_shape = random.choice(list(range(1, 2000)))

                product *= random_shape
                x_shape.append(random_shape)

            if product > 1000000:  # 防止shape值过大，环境内存不足/单用例执行耗时久/x无法生成
                continue

            #server_num = server_num
            #rank_per_dev = 16
            #world_size = server_num*rank_per_dev

            case_name = f"Reset_iter1_redLine_ID{count}"
            x_range = random.choice(valid_list)
            x_dtype = random.choice(dtpye_list)
            #is_elastic = random.choice(elastic)
            # broken_card = random.choice(broken_card_list)
            #is_timeout = random.choice(timeout_flag)
            #timeout = random.choice(range(70000,1000000)) if is_timeout else 0
            #graph_type = random.choice(graph_list)
            #is_performance =random.choice(performance_list)
            key =random.choices(range(1,16))
            print(key)
            numbers = random.sample(range(16), k=key[0])




            print(f"case_name:{case_name} x_shape:{x_shape} x_range:{x_range} x_dtype:{x_dtype}  numbers:{numbers} ")
            #print(f"x_shape:{x_shape} x_range:{x_range} x_dtype:{x_dtype} server_num:{server_num} is_elastic:{is_elastic} is_timeout:{is_timeout} timeout:{timeout} world_size:{world_size} graph_type:{graph_type} is_performance:{is_performance} broken_card:{broken_card}")

            combinations.append([case_name,x_shape,x_range, x_dtype,numbers])
            count += 1

        title = ["case_name","x_shape","x_range","x_dtype", "numbers"]

        with open(csv_name, 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(title)  # 写入表头
            for combination in combinations:
                writer.writerow(combination)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", default="fanhua", help="fanhua")
    parser.add_argument("-n", "--server_num", default=1, type=int, help="默认单机通信，如果需要双机通信，赋值2")
    parser.add_argument("-l", "--loop", default=2000, type=int, help="生成泛化用例数")
    parser.add_argument("-d", "--dev_num", default=16, type=int, help="每台server使用的die数")
    parser.add_argument("-t", "--type", default="V2", type=str, help="支持 v1/v2/sdma")
    parser.add_argument("-c", "--cloud", default=0, type=int, help="0:本地批跑格式, 1:云道批跑格式")

    args = parser.parse_args()

    gen_case(args.mode, args.server_num, args.loop, args.dev_num, args.type, args.cloud)
