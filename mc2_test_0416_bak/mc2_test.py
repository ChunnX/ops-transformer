import sys
import os
import time
import argparse
import random
import hashlib
import csv
from torch.multiprocessing import Process, Lock, Value
import pprint
from openpyxl import load_workbook

import torch
import torch_npu

from script.func import run_errcode
from script import func
from op_class.registry import *

if "errorcode" in ' '.join(sys.argv[0:]):
    flag = 1
else:
    flag = 0

ACLNN_INDEX_MAP = {
    0: "output_hccl",
    1: "gatherout_hccl",
    2: "bmmout_hccl",
    3: "mm_out_hccl",
    4: "permute_out_hccl",
    5: "alltoall_output_hccl"
}


def get_op_class(case, rank, dist=None, hcomm_info=None):
    op_name = case['op_name']
    cls = class_registry_func.get(op_name, None)
    if cls is None:
        print('[ERROR] op func is None, check op_name')
        exit(-1)
    else:
        mc2_op = cls(case, rank, dist, hcomm_info)

    return mc2_op


def run_aclnn_npu(cases, scene, rank=0):
    if isinstance(cases, dict):
        cases = [cases]
    from torch.multiprocessing import Process
    for case in cases:
        mc2_op = get_op_class(case, rank=rank)
        mc2_op.process_aclnn()
        p_list = []
        for rank in range(case['world_size']):
            p = Process(target=mc2_op.compare, args=(rank,))
            p_list.append(p)
        for p in p_list:
            p.start()
        for p in p_list:
            p.join()

# 不区分量化或非量化，单算子或图模式
def run_mc2_npu(rank, case, args):
    mc2_op = get_op_class(case, rank)
    mc2_op.get_hcomm()
    mc2_op.get_input_weight(args.weight_same)
    mc2_op.get_tmp_tensor()
    if args.scene == 'aclnn_golden':
        output = mc2_op.get_hccl_mm()
        if not isinstance(output, tuple):
            output = (output,)
        for index, value in enumerate(output):
            if value is not None:
                print(f"value:{ACLNN_INDEX_MAP[index]} is saved")
                func.save_file(f"bin_path/{case['case_name']}/output/{rank}", value, value.dtype,
                               ACLNN_INDEX_MAP[index], "None", rank)
            else:
                print(f"value:{ACLNN_INDEX_MAP[index]} is None, Not save")
        return

    if args.scene == 'npu':
        print('[INFO] device_{} 构造output_golden数据 '.format(rank))
        mc2_op.get_hccl_mm()
        print('[INFO] device_{} 构造output_npu数据'.format(rank))
        mc2_op.get_npu()  # MC2单算子模式
    elif args.scene == 'graph':
        print('[INFO] device_{} 构造入图output_graph_golden数据'.format(rank))
        mc2_op.get_graph("hccl_mm")
        print('[INFO] device_{} 构造入图output_graph_npu数据'.format(rank))
        mc2_op.get_graph()
    mc2_op.compare(check_md5=args.weight_same)


def gen_input(rank, case, scene='910B'):
    mc2_op = get_op_class(case, rank)
    mc2_op.get_input_weight()
    mc2_op.get_tmp_tensor()


def run_mc2_dfx(rank, case, scene):
    plog_error_message = ''
    log_path = os.path.join(os.getcwd(), 'log')
    os.system(f'rm -rf {log_path}')
    os.environ['ASCEND_PROCESS_LOG_PATH'] = log_path
    mc2_op = get_op_class(case, rank)
    try:
        mc2_op.get_hcomm()
        mc2_op.get_input_weight()
        mc2_op.get_tmp_tensor()
        if scene == 'dfx_graph':
            mc2_op.get_graph()
        else:
            mc2_op.get_npu()
    except Exception as e:
        print(f"用例 {case['case_name']} 执行异常 \n {e}")
        plog_error_message = str(e)
    if rank == case['save_rank']:
        plog_error_message1, device_log_error_message = mc2_op.check_log(log_path)
        dfx_csv = f"dfx_{case['op_name']}_result.csv"
        func.write_csv(dfx_csv, [case['case_name'], plog_error_message + plog_error_message1, device_log_error_message])


def run_mc2_cpu(rank, case, port):
    mc2_op = get_op_class(case, rank)
    mc2_op.get_dist_gloo(port)  # CPU通信链路建立
    mc2_op.get_input_weight()
    mc2_op.get_tmp_tensor()
    mc2_op.get_cpu()


def run_mc2_performance(rank, case, args):
    mc2_op = get_op_class(case, rank)
    mc2_op.get_hcomm()
    mc2_op.get_input_weight(args.weight_same)
    mc2_op.get_tmp_tensor()
    print('[INFO] device_{} 构造output_golden数据 '.format(rank))
    mc2_op.get_performace('golden', args.soc)  # 小算子性能
    print('[INFO] device_{} 构造output_npu数据'.format(rank))
    mc2_op.get_performace('npu', args.soc)  # MC2性能
    mc2_op.compare(check_md5=args.weight_same)
    if rank == case['save_rank']:
        func.save_performance_result_csv(case)


def errorcode_consume(flag):
    def wrapper(func):
        def decorate(*args, **kw):
            func(*args, **kw)
            if int(flag):
                run_errcode()
                print("errrcode判断结束")

        return decorate

    return wrapper


def init_csv(file):
    item = ['testcase_name', 'bin_name', 'output_dtypes', 'stc_outputs', 'input_data_ranges']
    if os.path.exists(file):
        os.remove(file)
    with open(file, mode="a", encoding='utf-8-sig') as f:
        write = csv.writer(f)
        write.writerow(item)


def run_mc2_performance_onehccl(rank, cases, args, port, shared_counters=None, lock=None):
    if 'determin' in args.scene:
        print('[INFO] 设置确定性计算的环境变量')
        torch.use_deterministic_algorithms(True)
    if args.scene == 'one_hccl_cpu':
        hcomm_info_dist = {'default_pg_info': None, 'ep_hcomm_info': None, 'group_ep': None, 'tp_hcomm_info': None,
                           'group_tp': None}
        import torch.distributed as dist
        dist.init_process_group(backend='gloo', rank=rank, world_size=args.world_size, init_method=f'tcp://127.0.0.1:{port}')
        if args.ep and args.tp:
            hcomm_info_dist['group_ep'], hcomm_info_dist['group_tp'] = func.setup_ep_tp(rank, args.tp, args.ep, "gloo")
        for case in cases:
            mc2_op = get_op_class(case, rank, dist=dist, hcomm_info=hcomm_info_dist)
            mc2_op.get_input_weight()
            mc2_op.get_tmp_tensor()
            mc2_op.get_cpu()
    elif args.scene == 'one_hccl_dfx':
        log_path = os.path.join(os.getcwd(), 'log')
        os.system(f'rm -rf {log_path}')
        os.environ['ASCEND_PROCESS_LOG_PATH'] = log_path
        torch_npu.npu.set_device(rank)
        print("current device:", torch_npu.npu.current_device())
        print('[INFO] device_{} 创建HCCL通信链路 '.format(rank))
        dist, hcomm_info = func.get_hcomm_info(args.world_size, rank, ep_size=args.ep, tp_size=args.tp, master_ip=args.master_ip,
                                               master_port=args.master_port)
        for case in cases:
            plog_error_message = ''
            mc2_op = get_op_class(case, rank, dist, hcomm_info)
            try:
                mc2_op.get_input_weight()
                mc2_op.get_tmp_tensor()
                mc2_op.get_npu()
            except Exception as e:
                print(f"用例 {case['case_name']} 执行异常 \n {e}")
                plog_error_message = str(e)
            if rank == 0:
                plog_error_message1, device_log_error_message = mc2_op.check_log(log_path)
                dfx_csv = f"dfx_{case['op_name']}_result.csv"
                if not os.path.exists(dfx_csv):
                    func.write_csv(dfx_csv, ["case_name", "plog_error", "device_error"])
                else:
                    func.write_csv(dfx_csv, [case['case_name'], plog_error_message + plog_error_message1,
                                             device_log_error_message])
    else:
        torch_npu.npu.set_device(rank)
        print("current device:", torch_npu.npu.current_device())
        print('[INFO] device_{} 创建HCCL通信链路 '.format(rank))
        dist, hcomm_info = func.get_hcomm_info(args.world_size, rank, ep_size=args.ep, tp_size=args.tp, master_ip=args.master_ip,
                                               master_port=args.master_port)
        tmp = 0
        for case in cases:
            mc2_op = get_op_class(case, rank, dist, hcomm_info)
            mc2_op.get_input_weight(args.weight_same)
            mc2_op.get_tmp_tensor()
            if args.scene == 'one_hccl_npu':
                mc2_op.get_hccl_mm()
                mc2_op.get_npu()
                mc2_op.compare(check_md5=args.weight_same)
            elif args.scene == 'one_hccl_graph':
                mc2_op.get_graph("hccl_mm")
                mc2_op.get_graph()
                mc2_op.compare(check_md5=args.weight_same)
            elif args.scene == 'one_hccl':
                mc2_op.get_performace('golden', args.soc)  # 小算子性能
                mc2_op.get_performace('npu', args.soc)  # MC2性能
                if rank == 0:
                    func.save_performance_result_csv(case)
                mc2_op.compare(check_md5=args.weight_same)
            elif args.scene == 'one_hccl_aclnn_golden':
                output_golden = mc2_op.get_hccl_mm()  # 小算子级联
                if output_golden is not None:
                    func.save_file(f"bin_path/{case['case_name']}/output/{rank}", output_golden, output_golden.dtype,
                                "output_hccl", "None", rank)
            elif 'one_hccl_mc2_save' in args.scene:
                if "npu" in args.scene:
                    output_golden = mc2_op.get_hccl_mm()
                    output = mc2_op.get_npu()
                elif "graph" in args.scene:
                    output_golden = mc2_op.get_graph("hccl_mm")
                    output = mc2_op.get_graph()
                func.save_file(f"bin_path/{case['case_name']}/output/{rank}", output_golden, output_golden.dtype,
                               "output_hccl", "None", rank)
                func.save_file(f"bin_path/{case['case_name']}/output/{rank}", output, output.dtype,
                               "output", "None", rank)
            elif 'determin' in args.scene:
                check_determin(rank, case, mc2_op)
            else:
                print("Wrong one_hccl scene!")
            with lock:
                shared_counters[tmp].value += 1
                print(f"[rank{rank}]{tmp} counter is {shared_counters[tmp].value}")
            start_time = time.time()
            while True:
                with lock:
                    if shared_counters[tmp].value == args.world_size:
                        print(f"[rank{rank}]{tmp} end time:{func.get_time()}")
                        break
                current_time = time.time()
                elapsed_time = current_time - start_time
                if elapsed_time >= 500:
                    print(f"[ERROR] rank{rank} timeout")
                    break
            tmp += 1


def run_mc2_determin(rank, case):
    print('[INFO] 设置确定性计算的环境变量')
    torch.use_deterministic_algorithms(True)
    mc2_op = get_op_class(case, rank)
    mc2_op.get_hcomm()
    mc2_op.get_input_weight()
    mc2_op.get_tmp_tensor()
    check_determin(rank, case, mc2_op)


def check_determin(rank, case, mc2_op):
    print('[INFO] 测试确定性计算场景的精度')
    md5_list = []
    result_list = [case['case_name'], rank]
    for i in range(2):
        res = mc2_op.get_graph() if "graph" in case['scene'] else mc2_op.get_npu()
        if not torch.any(res):
            print("Output npu Result is empty tensor!!!")
            break
        m = hashlib.md5()
        m.update(res.cpu().to(torch.float32).numpy())
        print(f'第{i}次测试 device_{rank} ，md5值为：{m.hexdigest()}')
        md5_list.append(m.hexdigest())
    result = all(x == md5_list[0] for x in md5_list) if md5_list else "False"
    result_list.append(result)
    print("rank:{}  result:{}".format(rank, result))
    func.write_csv("{}_determin.csv".format(case['op_name']), result_list)


def check_md5():
    with open('allgather_md5.csv', 'r') as f:
        md5_reader = csv.reader(f)
        md5_res = {}
        for row in md5_reader:
            if row[0] in md5_res:
                md5_res[row[0]].append(row[2])
            else:
                res = {row[0]: [row[2]]}
                md5_res.update(res)
        for res in md5_res:
            value = md5_res[res]
            result = all(x == value[0] for x in value) if value else "False"
            if not result:
                print(f"[ERROR] check case {res} md5 res FAIL!!!!!!!")
            func.write_csv("md5_res.csv", [res, result])
    os.system(f'rm -rf allgather_md5.csv')


@errorcode_consume(flag)
def main():
    cmdline = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cmdline.add_argument('--xlsx', required=True, nargs='+', help="xlsx name")
    cmdline.add_argument('--sheet', required=True, help="测试表格sheet页")
    cmdline.add_argument('--num', nargs='+', type=int, required=True,
                         help="若num=x，则测试表格中的第x个用例；若num=i j（仅支持--scene为one_hccl/one_hccl_cpu模式），则测试表格中第i个到第j个的所有用例")
    cmdline.add_argument('--scene', default='npu', required=True,
                         help="仅支持cpu/npu/graph/performance/one_hccl/determin/one_hccl_cpu/dfx/one_hccl_dfx/one_hccl_determin/one_hccl_determin_graph")
    cmdline.add_argument('--world_size', type=int, default=4, help="当scene=one_hccl时，需要此参数")
    cmdline.add_argument('--soc', default='910B', help="区分910B/910C/910D")
    cmdline.add_argument('--weight_same', type=int, default=0,
                         help="控制各device上weight是否一致，一致时allgather场景校验各device上md5值")
    cmdline.add_argument('--ep', default='0', type=int, help="当scene=one_hccl时，且为双通信域场景时，需要此参数")
    cmdline.add_argument('--tp', default='0', type=int, help="当scene=one_hccl时，且为双通信域场景时，需要此参数")
    cmdline.add_argument('--master_ip', default='127.0.0.1', help="主节点ip，单机场景不需要，默认为回环ip")
    cmdline.add_argument('--master_port', default='50001', help="跨server场景需要")
    cmdline.add_argument('--server_num', type=int, default=1, help="跨server场景需要，通信节点数量")
    cmdline.add_argument('--server_index', type=int, default=0, help="跨server场景需要，节点序号")
    cmdline.add_argument('--dev_num', type=int, default=0, help="跨server场景需要，每个节点上使用的device数量")
    cmdline.add_argument('--mix', default=0, help="使能不同算子混跑，默认是非混跑")

    flags = cmdline.parse_args()
    soc_version = func.get_soc_version()
    if "910_93" in soc_version:
        flags.soc = "910C"
    elif "Ascend950" in soc_version:
        flags.soc = "910D"
    elif "910B" in soc_version:
        flags.soc = "910B"
    else:
        flags.soc = "310P"
    if len(flags.xlsx) > 1:
        flags.mix = 1
    if not flags.mix:
        flags.xlsx = (flags.xlsx)[0]
        print(flags.xlsx)
        flags.op_name = flags.xlsx.split("/")[-1].split(".")[0]
    print("xlsx             :%s" % flags.xlsx)
    print("sheet            :%s" % flags.sheet)
    print("num              :%s" % flags.num)
    print("scene            :%s" % flags.scene)
    print("world_size       :%s" % flags.world_size)
    print("soc              :%s" % flags.soc)
    print("weight_same      :%s" % flags.weight_same)
    print("ep               :%s" % flags.ep)
    print("tp               :%s" % flags.tp)
    print("master_ip        :%s" % flags.master_ip)
    print("master_port      :%s" % flags.master_port)
    print("server_num       :%s" % flags.server_num)
    print("server_index     :%s" % flags.server_index)
    print("dev_num          :%s" % flags.dev_num)
    print("mix              :%s" % flags.mix)

    func.remove_profiling_result()  # 环境清理
    if flags.mix:
        data = []
        for cur_xlsx in flags.xlsx:
            wb = load_workbook(os.path.join(os.getcwd(), cur_xlsx))
            flags.xlsx = cur_xlsx
            flags.op_name = flags.xlsx.split("/")[-1].split(".")[0]
            data.extend(func.get_case(flags))
        value = random.randint(1, 1000)
        print("随机数种子值：", value)
        random.seed(value)
        random.shuffle(data)
    else:
        data = func.get_case(flags)

    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(data)
    single_list=[]
    if "one_hccl_mc2_save" in flags.scene:
        for index, s in enumerate(data):
            # case_list.append([s['case_name'],])
            tmp_list=func.get_case_info(s['case_name'],s['world_size'],s['bin_name'],s['input_tensor_range'])
            single_list.append(tmp_list)
        if os.path.exists("MC2.csv"):
            os.remove("MC2.csv")
        if not os.path.exists("MC2.csv"):
            func.init_csv("MC2.csv")
        for case in single_list:
            func.write_csv("MC2.csv", case)

    if flags.weight_same == 1:
        os.system(f'rm -rf allgather_md5.csv')  # 清理md5值表格
    case_list=[]
    if "npu" in flags.scene or "graph" in flags.scene or "aclnn" in flags.scene:
        if "one_hccl" in flags.scene:
            for index,s  in enumerate(data):
                case_list.append([s['case_name']])               
            for case in case_list:
                func.write_csv("case_name.csv",case)
        else:
            case_list.append([data['case_name']])
            for case in case_list:
                func.write_csv("case_name.csv", case)

    if flags.scene in ['aclnn', 'one_hccl_aclnn']:
        run_aclnn_npu(data, flags.scene)
    else:
        # 多进程锁 避免批跑场景进程间精度比对时间差导致aic timeout
        if "one_hccl" in flags.scene:
            torch.multiprocessing.set_start_method("forkserver", force=True)
            shared_counters = [Value('i', 0) for _ in range(len(data))]
            lock = Lock()
        p_list = []
        port = 29500 + random.randint(0, 10000)
        if flags.server_num > 1:
            print("multi_server scene!!!!! only support running case one by one!!!!!!!!!!!!!")
            index = data.get('server_index')
            rank_list = data.get(f'rank_list{index}') if 'one_hccl' not in flags.scene else list(
                range(flags.server_num * data.get('dev_num'), (flags.server_num + 1) * data.get('dev_num')))
        else:
            print("single_server scene!!!!!")
            rank_list = list(range(data.get('world_size'))) if 'one_hccl' not in flags.scene else list(
                range(flags.world_size))
        print("rank list is: ", rank_list)

        # 低bit通信cpu计算场景, 需要共享内存获取各进程量化系数
        if flags.scene == "cpu" and flags.op_name in ("aclnnQuantMatmulAllReduceV3",):
            torch.multiprocessing.set_start_method("fork", force=True)
        else:
            torch.multiprocessing.set_start_method("forkserver", force=True)
        for rank in rank_list:
            if flags.scene in ['npu', 'graph', 'aclnn_golden']:
                p = Process(target=run_mc2_npu, args=(rank, data, flags))
            elif flags.scene == 'cpu':
                p = Process(target=run_mc2_cpu, args=(rank, data, port))
            elif flags.scene == 'performance':
                p = Process(target=run_mc2_performance, args=(rank, data, flags))
            elif 'one_hccl' in flags.scene:
                p = Process(target=run_mc2_performance_onehccl,
                            args=(rank, data, flags, port, shared_counters, lock))
            elif flags.scene in ['determin', 'determin_graph']:
                p = Process(target=run_mc2_determin, args=(rank, data))
            elif flags.scene in ['dfx', 'dfx_graph']:
                p = Process(target=run_mc2_dfx, args=(rank, data, flags.scene))
            p.start()
            p_list.append(p)
        for p in p_list:
            p.join()
        if flags.op_name == "aclnnAllGatherMatmul" and flags.weight_same == 1:
            check_md5()


if __name__ == "__main__":
    print(f'[TEST] {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}  测试程序开始')
    main()
    print(f'[TEST] {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}  测试程序结束')
