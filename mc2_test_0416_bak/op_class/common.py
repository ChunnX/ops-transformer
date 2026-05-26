import os
import time

import torch
import torch_npu
import torch.distributed as dist
import subprocess
import abc
import ml_dtypes
from ml_dtypes import bfloat16
import torchair

from en_dtypes import float8_e8m0, float4_e2m1, float4_e1m2, hifloat8
from script import func
from script.func import *

from script.new_compare import *
from script.generate_float_data import (
    new_trans_np_fp4_e1m2_tensor_to_bfloat16,
    unpacked_float4e2m1_to_float32
)

INF_NAN_NUM = 10
MAIN_PATH = os.path.abspath(os.getcwd())


class MC2_OP(object, metaclass=abc.ABCMeta):
    def __init__(self, case, rank, dist=None, hcomm_info=None):
        # MC2_OP公共参数
        self.dist = dist
        self.hcomm_info_dist = hcomm_info
        try:  # one_hccl场景初始化通信链路信息
            self.hcomm_info = self.hcomm_info_dist['default_pg_info']
            self.ep_hcomm_info = self.hcomm_info_dist['ep_hcomm_info']
            self.tp_hcomm_info = self.hcomm_info_dist['tp_hcomm_info']
            self.group_ep = self.hcomm_info_dist['group_ep']
            self.group_tp = self.hcomm_info_dist['group_tp']
        except:
            pass
        self.case = case
        self.case_name = case['case_name']
        self.world_size = case['world_size']
        self.bin_name = case['bin_name']
        self.sheet_name = case['sheet_name']
        self.op_name = case['op_name']
        self.quant_mode = case.get("quant_mode", "")
        try:
            self.case_dtype = list(map(func.get_dtype, case['input_tensor_dtype']))
            self.input_shape = case['input_tensor_shape'][0]
            self.input_dtype = self.case_dtype[0]
            self.weight_dtype = self.case_dtype[1]
            self.weight_shape = case['input_tensor_shape'][1]
            self.input_range = case['input_tensor_range'][0]
            self.weight_range = case['input_tensor_range'][1]
        except:
            print('新表格已将输入拆分成x1/x2/bias字段')
        self.golden_type = case['golden_type'] if 'golden_type' in case else None
        self.rank = rank
        self.scene = case['scene']
        self.graph_type = case["graph_type"] if "graph_type" in case else 0
        if "npu" in self.scene:
            self.graph_type = 0
        if self.graph_type == 1:
            print(f"{self.case_name} graph_type=1, 静态图模式")
        elif self.graph_type == 2:
            print(f"{self.case_name} graph_type=2, 动态图模式")
        elif self.graph_type == 0:
            print(f"{self.case_name} graph_type=0, 单算子模式")
        self.ep_size = case['ep_size'] if 'ep_size' in case else None
        self.tp_size = case['tp_size'] if 'tp_size' in case else None
        self.ep_ranks_list = case['ep_ranks_list'] if 'ep_ranks_list' in case else None
        self.tp_ranks_list = case['tp_ranks_list'] if 'tp_ranks_list' in case else None
        self.server_index = case['server_index'] if 'server_index' in case else 0
        self.server_num = case['server_num']
        self.dev_num = case['dev_num'] if 'dev_num' in case else 0
        self.master_ip = case['master_ip'] if 'master_ip' in case else None
        self.master_port = case['master_port'] if 'master_port' in case else None
        self.soc = case['soc']
        self.seed_num = int(case['seed']) if 'seed' in case else 1
        self.is_trans = int(case['is_trans']) if 'is_trans' in case else 0
        self.is_trans_b = int(case['is_trans_b']) if 'is_trans_b' in case else 0  # 待删除
        self.weight_same = case["weight_same"]
        self.save_path = 'bin_path/{}'.format(self.case_name)
        self.bin_path = os.path.join(MAIN_PATH, "bin_path", self.case_name)
        self.input_path = os.path.join("bin_path", self.case_name, "input", str(rank))
        self.output_path = os.path.join("bin_path", self.case_name, "output", str(rank))
        for path in [self.input_path, self.output_path]:
            if not os.path.exists(path):
                os.system(f"mkdir -p {path}")
        self.save_flag = True if "aclnn" in self.scene else False
        self.device = torch.device('cpu') if "cpu" in self.scene else torch.device('npu')
        self.loop = 10
        self.output_npu_name = "output_npu.bin"
        self.output_hccl_name = "output_hccl.bin"
        self.output_cpu_name = "output_cpu.bin"

    def get_hcomm(self):
        torch_npu.npu.set_device(self.rank - self.server_index * self.dev_num)
        print("current device:", torch_npu.npu.current_device())
        print('[INFO] device_{} 创建HCCL通信链路 '.format(self.rank))
        self.dist, self.hcomm_info_dist = func.get_hcomm_info(self.world_size, self.rank, self.ep_size, self.tp_size,
                                                              self.ep_ranks_list, self.tp_ranks_list, self.master_ip,
                                                              self.master_port)
        self.hcomm_info = self.hcomm_info_dist['default_pg_info']
        self.ep_hcomm_info = self.hcomm_info_dist['ep_hcomm_info']
        self.tp_hcomm_info = self.hcomm_info_dist['tp_hcomm_info']
        self.group_ep = self.hcomm_info_dist['group_ep']
        self.group_tp = self.hcomm_info_dist['group_tp']

    def get_dist_gloo(self, port=29500):
        hcomm_info_dist = {'default_pg_info': None, 'ep_hcomm_info': None, 'ep_group': None, 'tp_hcomm_info': None,
                           'tp_group': None}
        # torch_npu.npu.set_device(self.rank) cpu场景不需要
        dist.init_process_group(backend='gloo', rank=self.rank, world_size=self.world_size,
                                init_method=f'tcp://127.0.0.1:{port}')
        if self.ep_size and self.tp_size:
            self.group_ep, self.group_tp = func.setup_ep_tp(self.rank, self.tp_size, self.ep_size, "gloo",
                                                            self.ep_ranks_list, self.tp_ranks_list)

    @abc.abstractmethod
    def get_input_weight(self, weight_same=0):
        ...

    @abc.abstractmethod
    def get_tmp_tensor(self):
        ...

    @abc.abstractmethod
    def get_npu(self):
        ...

    @abc.abstractmethod
    def get_hccl_mm(self):
        ...

    @abc.abstractmethod
    def process_aclnn(self):
        ...

    def get_tmp_tensor_memory(self):
        return 0

    def compare(self, rank=0, check_md5=0):
        op = self.case['op_name']
        if self.case['scene'] in ["compare", "aclnn"]:  # aclnn场景，golden和npu从已保存文件中读取
            output_npu_path = f"{self.save_path}/output/{rank}/output_npu.bin"
            print(f"output_npu_path ,{output_npu_path}")
            output_golden_path = f"{self.save_path}/output/{rank}/output_hccl.bin"
            if self.output_dtype == 'fp16':
                np_dtype = np.float16
            elif self.output_dtype == 'bf16':
                np_dtype = bfloat16
            else:
                np_dtype = np.float32
            ## bf16场景 单独处理
            if self.output_dtype == 'bf16':
                self.output_npu = torch.tensor(np.fromfile(output_npu_path, np_dtype).astype(np.float32)).to(
                    torch.bfloat16)
                self.output_golden = torch.tensor(np.fromfile(output_golden_path, np_dtype).astype(np.float32)).to(
                    torch.bfloat16)
            else:
                self.output_npu = torch.tensor(np.fromfile(output_npu_path, np_dtype))
                self.output_golden = torch.tensor(np.fromfile(output_golden_path, np_dtype))
            cpu_input_path = os.path.join('bin_path', self.case_name,
                                          '{}_cpu_output_{}.bin'.format(self.bin_name, rank))
        else:
            output_npu = self.output_npu.cpu().view(-1)
            output_golden = self.output_golden.cpu().view(-1)
            cpu_input_path = os.path.join('bin_path', self.case_name,
                                          '{}_cpu_output_{}.bin'.format(self.bin_name, self.rank))
        output_cpu = torch.tensor(np.fromfile(cpu_input_path, np.float32))
        output_npu = self.output_npu.cpu().view(-1)
        output_golden = self.output_golden.cpu().view(-1)
        rst_npu = checkResult(output_npu, output_cpu, "{}_dq_npu".format(self.bin_name))
        rst_npu.print_result()
        rst_gpu = checkResult(output_golden, output_cpu, "{}_dq_gpu".format(self.bin_name))
        rst_gpu.print_result()
        str1, str2 = rst_npu.check_result(rst_gpu)
        if 'error' in str1:
            res = 'FAIL'
        else:
            res = 'PASS'
        compare_rank = rank if self.case['scene'] in ["compare", "aclnn"] else self.rank
        print('[INFO] device_{} 精度结果为：{}'.format(compare_rank, res))
        print('[INFO] device_{} 精度计算结束：{} '.format(compare_rank, time.strftime('%H:%M:%S', time.localtime())))
        with open('{}_result.csv'.format(op), 'a') as f:
            f.write(self.case_name + "," + str(compare_rank) + ',' + str(res) + '\n')
        return res

    def get_performace(self, perf_type='npu', soc='910B'):
        tensor = torch.ones(10000, 10000).to(torch.float16).npu() ## 重型算子计算需要
        if perf_type == 'npu':
            path_tmp = 'mc2_'
        elif perf_type == 'golden':
            path_tmp = 'hccl_mm_'
        experimental_config, profiling_result = func.get_profiling_config(self.case['case_name'], path_tmp)
        with torch_npu.profiler.profile(
                schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_result),
                profile_memory=True,
                with_stack=False,
                with_flops=False,
                with_modules=False,
                experimental_config=experimental_config,
        ) as prof:
            for i in range(10):
                if i == 1: ## 防止host bound现象，插入重型算子
                    for _ in range(50):
                        torch.exp(tensor)
                if perf_type == 'npu':
                    self.get_npu()
                elif perf_type == 'golden':
                    self.get_hccl_mm()
        if self.rank == self.case['save_rank']:
            op = self.case['op_name']
            csv_file = f'{path_tmp}{op}_pro_result.csv'
            ## soc为910D时，或是两个双通信域moe算子时，需要计算性能掩盖率
            cal_mask = True if soc == '910D' or op in ['aclnnAlltoAllAllGatherBatchMatMul', 'aclnnBatchMatMulReduceScatterAlltoAll'] else False
            result_list, memory, performance = func.get_memory_performance(self.case, path_tmp, cal_mask, perf_type, soc)
            if perf_type == "golden":
                diff = func.get_memory_diff(self.case)
                tmp_tensor_memory = self.get_tmp_tensor_memory()
                print(
                    f'Profiling采集的内存为：{memory}；MC2内存计算需要累加值为： {diff}; tmp_tensor_memory需要累加值为：{tmp_tensor_memory}')
                golden_memory = memory + diff + tmp_tensor_memory
                if cal_mask:
                    hccl_perf = result_list[0]
                    mm_perf = result_list[1]
                    func.write_csv(csv_file, [self.case_name, self.rank, hccl_perf, mm_perf, golden_memory,
                                              performance])
                    print(f'[INFO] 小算子级联 内存为：{golden_memory} M, 通信性能为：{hccl_perf} us, 计算性能为：{mm_perf} us, 总性能： {performance}us \n')
                else:
                    func.write_csv(csv_file, [self.case_name, self.rank, golden_memory, performance])
                    print(f'[INFO] 小算子hccl+mm级联 内存为：{golden_memory} M，性能为：{performance} us \n')
            else:
                func.write_csv(csv_file, [self.case_name, self.rank, memory, performance])
                print(f'[INFO] MC2融合算子 内存为：{memory} M，性能为：{performance} us \n')

    @staticmethod
    def define_model(model, graph_type):
        if graph_type == 1:  # 传统入图模式，静态shape+在线编译场景
            print('graph_type=1，测试场景：传统入图模式，静态shape+二进制场景')
            npu_backend = torchair.get_npu_backend(compiler_config=None)
            model = torch.compile(model, backend=npu_backend, dynamic=False)
        elif graph_type == 2:  # ACLNN入图模式，动态shape+二进制
            print('graph_type=2，测试场景：ACLNN入图模式，动态shape+二进制')
            npu_backend = torchair.get_npu_backend(compiler_config=None)
            model = torch.compile(model, backend=npu_backend, dynamic=True)
        else:
            print('用例未设置graph_type，测试场景：单算子模式')
        return model

    def check_log(self, log_path):
        import subprocess
        debug_error_message = ''
        run_error_message = ''
        # debug plog收集
        debug_log_path = os.path.join(log_path, "debug", "plog")
        if os.path.exists(debug_log_path):
            debug_error_message = subprocess.check_output(f"grep -rn ERROR {debug_log_path} | head -n 3", shell=True,
                                                          stderr=subprocess.STDOUT).decode()
        # run plog收集
        run_log_path = os.path.join(log_path, "run", "plog")
        if os.path.join(run_log_path):
            run_error_message = subprocess.check_output(f"grep -rn ERROR {run_log_path} | head -n 3", shell=True,
                                                        stderr=subprocess.STDOUT).decode()
        plog_error_message = debug_error_message + run_error_message
        if plog_error_message:
            print(f'plog_error_message :\n{plog_error_message}')
        # device log收集
        all_device_debug_error_message = ""
        all_device_run_error_message = ""
        for device_id in range(self.world_size):
            device_debug_log_path = os.path.join(log_path, "debug", f"device-{device_id}")
            run_debug_log_path = os.path.join(log_path, "run", f"device-{device_id}")
            if os.path.exists(device_debug_log_path):
                device_debug_error_message = subprocess.check_output(
                    f"grep -rn ERROR {device_debug_log_path} | head -n 3", shell=True,
                    stderr=subprocess.STDOUT).decode()
                all_device_debug_error_message += device_debug_error_message
            if os.path.exists(run_debug_log_path):
                device_run_error_message = subprocess.check_output(f"grep -rn ERROR {run_debug_log_path} | head -n 3",
                                                                   shell=True,
                                                                   stderr=subprocess.STDOUT).decode()
                all_device_run_error_message += device_run_error_message
        device_log_error_message = all_device_debug_error_message + all_device_run_error_message
        if device_log_error_message:
            print(f'device_log_error_message: \n{device_log_error_message}')
        return plog_error_message, device_log_error_message

    def create_array(self, name, shape, data_range, dtype, path, seed=0):
        if dtype in ["fp4_e2m1", "fp4_e1m2"]:
            print(f"[ERROR] dtype={dtype} is not unsupported")
            sys.exit(1)
        np.random.seed(seed)
        rank = self.rank % 2
        if "int" in dtype:
            array = np.random.uniform(*data_range, shape).astype(NUMPY_DTYPE_MAP[dtype])
            if dtype == "int4":
                packed_array = pack_int4(np.array(array))
                packed_array.tofile(os.path.join(path, "packed_" + name + ".bin"))
        else:
            if data_range[rank] in ["nan", "inf", "-inf"]:
                if dtype == "hif8":
                    array = np.random.uniform(*data_range, shape).astype(NUMPY_DTYPE_MAP[dtype])
                    # todo hif8里面有inf/nan需要处理
                    # array = random_inf_nan(array, shape, INF_NAN_NUM, data_range[rank], self.server_num)
                else:
                    array = np.random.uniform(-1, 1, shape).astype(NUMPY_DTYPE_MAP[dtype])
                    array = random_inf_nan(array, shape, INF_NAN_NUM, data_range[rank], self.server_num)
            else:
                array = np.random.uniform(*data_range, shape).astype(NUMPY_DTYPE_MAP[dtype])
        if self.save_flag:
            array.tofile(os.path.join(path, name + ".bin"))
        return array

    def tensor(self, src):
        if isinstance(src, torch.Tensor):
            tensor = src
        else:
            array = src
            if array.dtype in ["float8_e4m3fn", "float8_e5m2", "bfloat16"]:
                if not hasattr(torch, str(array.dtype)):
                    print(f"{array.dtype} is not supported in pytorch {torch.__version__}")
                    sys.exit(1)
                tensor = torch.from_numpy(array.astype(np.float32)).to(getattr(torch, str(array.dtype))).to(self.device)
            elif array.dtype in ["float8_e8m0", "float4_e2m1", "float4_e1m2", "hifloat8"]:
                if array.dtype in ["float8_e8m0", "hifloat8"]:
                    if self.device.type == "npu":
                        tensor = torch.from_numpy(array.view(np.uint8)).to(self.device)
                    else:
                        tensor = torch.from_numpy(array.astype(np.float32)).to(self.device)
                else:
                    # 通过pack函数转换成uint8的array,两个float4数据占8位
                    tensor = torch.from_numpy(array).to(self.device)
            else:
                tensor = torch.from_numpy(array).to(self.device)
        return tensor

    @staticmethod
    def array(src):
        if isinstance(src, np.ndarray):
            array = src
        else:
            tensor = src
            dtype = str(tensor.dtype).split(".")[-1]
            if dtype in ["float8_e4m3fn", "float8_e5m2", "bfloat16"]:
                dtype_map = {
                    "float8_e5m2": "fp8_e5m2",
                    "float8_e4m3fn": "fp8_e4m3fn",
                    "bfloat16": "bfloat16",
                }
                array = tensor.to(torch.float32).numpy().astype(NUMPY_DTYPE_MAP[dtype_map[dtype]])
            elif dtype in ["float8_e8m0", "float4_e2m1", "float4_e1m2"]:
                print(f"{dtype} is not supported")
                sys.exit(1)
            else:
                array = tensor.numpy()
        return array

    @staticmethod
    def save(src, path, name):
        if isinstance(src, np.ndarray):
            array = src
        else:
            tensor = src
            dtype = str(tensor.dtype).split(".")[-1]
            if dtype in ["float8_e4m3fn", "float8_e5m2", "bfloat16"]:
                array = torch.Tensor.cpu(tensor).to(torch.float32).numpy().astype(getattr(ml_dtypes, dtype))
            else:
                array = torch.Tensor.cpu(tensor).numpy()
        bin_path = os.path.join(path, name)
        array.tofile(bin_path)

    @staticmethod
    def unpack_group_size(group_size):
        if group_size == -1:
            return 0, 0, 0
        group_size_m = (group_size >> 32) & 0xFFFF
        group_size_n = (group_size >> 16) & 0xFFFF
        group_size_k = group_size & 0xFFFF
        return group_size_m, group_size_n, group_size_k

    @staticmethod
    def trans_to_float32(array, dtype):
        """
        将解包后的 fp4 数组转换为 float32 数组
        - fp4_e2m1: 使用向量化实现（性能优化）
        - fp4_e1m2: 保持原有 for 循环实现
        """
        if dtype == "fp4_e1m2":
            # 保持原有实现：for 循环转换
            fp32_array = new_trans_np_fp4_e1m2_tensor_to_bfloat16(array).astype(np.float32)
        elif dtype == "fp4_e2m1":
            # 优化实现：向量化转换
            fp32_array = unpacked_float4e2m1_to_float32(array).astype(np.float32)
        else:
            fp32_array = array.astype(np.float32)
        return fp32_array

    @staticmethod
    def mxfp_cpu_compute(x1, x2, x1scale, x2scale):
        for array in [x1, x2, x1scale, x2scale]:
            if array.ndim != 2:
                print(f"[ERROR] array.ndim must be 2")
        if x1.shape[0] != x1scale.shape[0]:
            print(f"[ERROR] x1.shape[0] != x1scale.shape[0],x1.shape[0]={x1.shape[0]}, x1scale.shape[0]={x1scale.shape[0]}")
        if x2.shape[1] != x2scale.shape[1]:
            print(f"[ERROR] x2.shape[1] != x2scale.shape[1],x2.shape[1]={x2.shape[1]}, x1scale.shape[1]={x2scale.shape[1]}")
        if x1.shape[1] != x2.shape[0]:
            print(f"[ERROR] x1.shape[1] != x2.shape[0],x1.shape[1]={x1.shape[1]}, x2.shape[0]={x2.shape[0]}")
        if x1scale.shape[1] != x2scale.shape[0]:
            print(f"[ERROR] x1scale.shape[1] != x2scale.shape[0], x1scale.shape[1]={x1scale.shape[1]}, x2scale.shape[0]={x2scale.shape[0]}")
        repeated_x1scale = np.repeat(x1scale, 32, axis=-1)
        repeated_x2scale = np.repeat(x2scale, 32, axis=-2)
        x1_pad_length = repeated_x1scale.shape[-1] - x1.shape[-1]
        x2_pad_len = repeated_x2scale.shape[-2] - x2.shape[-2]
        x1_pad_tuple = [(0, 0)] * (len(x1.shape) - 1) + [(0, x1_pad_length)]
        x2_pad_tuple = [(0, 0)] * (len(x2.shape) - 2) + [(0, x2_pad_len)] + [(0, 0)]
        padded_x1 = np.pad(x1, x1_pad_tuple, mode='constant', constant_values=0)
        padded_x2 = np.pad(x2, x2_pad_tuple, mode='constant', constant_values=0)

        padded_x1 = torch.from_numpy(padded_x1)
        padded_x2 = torch.from_numpy(padded_x2)
        repeated_x1scale = torch.from_numpy(repeated_x1scale)
        repeated_x2scale = torch.from_numpy(repeated_x2scale)
        output = torch.matmul(padded_x1 * repeated_x1scale, padded_x2 * repeated_x2scale)
        return output
