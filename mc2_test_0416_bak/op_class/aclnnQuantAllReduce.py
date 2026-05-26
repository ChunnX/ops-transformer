import torch
import torch.distributed as dist
from torch.distributed import ReduceOp
from .common import *
from .registry import class_registry
from script import generate_mx_data
from script import toolss
from script.func import *


class QuantAllReduceGraphModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    # 图模式当前没有诉求，先预留
    def forward(self, x, scale, hcomm_info, world_size, reduce_op="sum", output_dtype=None, x_dtype=None,
                scales_dtype=None, hcom_type="mc2"):
        if hcom_type == "mc2":
            return torch_npu.npu_quant_all_reduce(
                x=x,
                scales=scale,
                hcom=hcomm_info,
                world_size=world_size,
                reduce_op=reduce_op,
                output_dtype=output_dtype,
                x_dtype=x_dtype,
                scales_dtype=scales_dtype
            )


@class_registry("aclnnQuantAllReduce")
class QuantAllReduce(MC2_OP):
    def __init__(self, case, rank, dist=None, hcomm_info=None):
        super().__init__(case, rank, dist, hcomm_info)
        self.x_shape = case["input_tensor_shape"]
        self.x_dtype = case["input_tensor_dtype"]
        self.x_range = case["input_tensor_range"]
        self.scale_shape = case["scales_shape"]
        self.scale_dtype = case["scales_dtype"]
        self.scale_range = case["scales_range"]
        self.output_shape = self.x_shape
        self.output_dtype = case.get("outdtype", "fp16")
        self.output_format = "ND"
        self.reduce_op = case.get("reduce_op", "sum")
        self.seed = case["seed"]
        self.is_mxFp = self.scale_dtype == "fp8_e8m0"
        if self.weight_same:
            self.weight_seed = self.seed + self.rank * 2
        else:
            self.weight_seed = self.seed
        self.x = None
        self.scale = None
        self.output_npu = None
        # self.output_golden = None
        self.diff_thd = 0.001
        self.pct_thd = 0.001
        self.max_diff_hd = 0

    def generate_mx_fp_data(self):
        dtype_map = {
            "fp8_e5m2": "float8_e5m2",
            "fp8_e4m3fn": "float8_e4m3fn",
            "fp4_e2m1": "float4_e2m1",
            "fp4_e1m2": "float4_e1m2",
        }
        x_fp32 = self.create_array("x", self.x_shape, self.x_range, "fp32", self.input_path, self.seed)
        x, scale = generate_mx_data.mx_quantize(x_fp32, dtype_map[self.x_dtype], axis=-1, block_size=32,
                                                round_mode="rint")
        # 根据表示，判断是否是aclnn通路保存输入的bin文件
        if self.save_flag:
            x.tofile(os.path.join(self.input_path, "x.bin"))
            scale.tofile(os.path.join(self.input_path, "scales.bin"))
        self.x = self.tensor(x)
        # 将scale的前2维保存，合并后面维度
        if len(self.scale_shape) == 3:
            self.scale = (
                self.tensor(scale).reshape(self.scale_shape[0], -1) if "cpu" in self.scene else self.tensor(scale))
        elif len(self.scale_shape) == 4:
            self.scale = (self.tensor(scale).reshape(self.scale_shape[0], self.scale_shape[1],
                                                     -1) if "cpu" in self.scene else self.tensor(scale))
        else:
            print("scales的shape维度不支持, 当前shape为:", self.scale_shape)

    def get_input_weight(self, weight_same=0):
        if self.is_mxFp:  # mx量化
            self.generate_mx_fp_data()
        else:  # pertoken-pergroup量化
            x = self.create_array("x", self.x_shape, self.x_range, self.x_dtype, self.input_path, self.seed)
            self.x = self.tensor(x)

    def get_tmp_tensor(self):
        if not self.is_mxFp:
            scale = self.create_array("scales", self.scale_shape, self.scale_range, self.scale_dtype, self.input_path,
                                      self.weight_seed)
            self.scale = self.tensor(scale)
        if self.scene in ["performance", "one_hccl"]:
            self.output_hccl = torch.zeros(self.output_shape, dtype=TORCH_DTYPE_MAP[self.output_dtype]).to(self.device)

    def cpu_dequant(self, x, scale, group_size):
        repeated_scale = np.repeat(scale, group_size, axis=-1)
        x = torch.from_numpy(x)
        repeated_scale = torch.from_numpy(repeated_scale)
        return x * repeated_scale

    def get_cpu(self):
        reduce_op_map = {
            sum: dist.ReduceOp.SUM,
            max: dist.ReduceOp.MAX,
            min: dist.ReduceOp.MIN,
        }
        op = reduce_op_map.get(self.reduce_op, dist.ReduceOp.SUM)  # 默认sum
        x = self.x.to(torch.float32)
        scale = self.scale.to(torch.float32)
        x = x.numpy()
        scale = scale.numpy()
        output = self.cpu_dequant(x, scale, 32 if self.is_mxFp else 128)

        dist.all_reduce(output, op)
        if os.path.exists(self.output_path + '/output_cpu.bin'):
            os.remove(self.output_path + '/output_cpu.bin')
            print(f"旧文件已删除: {self.output_path + '/output_cpu.bin'}")
        self.save(output, self.output_path, self.output_cpu_name)

    ## 精度比对：与cpu golden比较，按照双千分之一标准，不涉及npu小算子级联（get_hccl_mm）
    def get_hccl_mm(self):
        if self.scene in ["performance", "one_hccl"]:
            dist.all_reduce(self.output_hccl, op=ReduceOp.SUM)

    def get_npu(self):
        self.output_npu = torch_npu.npu_quant_all_reduce(
            self.x,
            self.scale,
            self.hcomm_info,
            self.world_size,
            reduce_op="sum",
            output_dtype=OUT_DTYPE_ENUM[self.output_dtype],
            x_dtype=DTYPE_ENUM[self.x_dtype],
            scales_dtype=DTYPE_ENUM[self.scale_dtype],
        )
        return self.output_npu

    def get_graph(self, hcom_type="mc2"):
        if hcom_type not in ["mc2", "hccl_mm"]:
            print(f"[ERROR] unsupported hcom_type {hcom_type}")
            sys.exit(1)
        model = QuantAllReduceGraphModel()
        model = self.define_model(model, self.graph_type)

        out_graph = model(
            self.x,
            self.scale,
            self.hcomm_info,
            self.world_size,
            reduce_op="sum",
            output_dtype=OUT_DTYPE_ENUM[self.output_dtype],
            x_dtype=DTYPE_ENUM[self.x_dtype],
            scales_dtype=DTYPE_ENUM[self.scale_dtype],
            hcom_type=hcom_type
        )
        if hcom_type == "mc2":
            self.output_npu = out_graph
            return self.output_npu

    def get_command(self, binary, rankid):
        command = [f"{self.root_path}/bin_path/{binary}"]
        command.append(f"--x_shape {','.join(map(str, self.x_shape))}")
        command.append(f"--scales_shape {','.join(map(str, self.scale_shape))}")
        command.append(f"--x_dtype {self.x_dtype}")
        command.append(f"--scales_dtype {self.scale_dtype}")
        command.append(f"--output_shape {','.join(map(str, self.output_shape))}")
        command.append(f"--output_dtype {self.output_dtype}")
        command.append(f"--n_dev {self.world_size}")
        command.append(f"--bin_path {self.bin_path}")
        command.append(f"--loop_cnt {self.loop}")
        command.append(f"--rankId {rankid}")
        command.append(f"--is_mxfp {1 if self.is_mxFp else 0} \&")
        return " ".join(command)

    def process_aclnn(self):
        self.root_path = os.path.abspath(os.getcwd())
        self.case_format = ['ND', 'ND']
        self.bin_path = self.root_path + "/bin_path/" + self.case_name
        print(f"bin_path: {self.bin_path}")
        result = subprocess.check_output(["uname", "-a"]).decode("utf-8")
        arch = "aarch" if "aarch64" in result.lower() else "x86"
        exe_binary = self.op_name + "_" + arch
        self.loop = 1
        if self.case['scene'] == 'aclnn':
            mc2_sh = self.case_name + '_mc2'
            # 移除已有shell文件
            mv_command = f"rm -rf bin_path/{self.case_name}/{self.case_name}*.sh"
            ret = subprocess.run(mv_command, shell=True)
            # 拼二进制执行命令
            for i in range(self.world_size):
                aclnn_command = self.get_command(exe_binary, i)
                print(f"[INFO] rank:{i}, aclnn command:{aclnn_command}")
                # 命令写入shell
                ret = subprocess.check_output(f"echo {aclnn_command} >> {self.bin_path}/{mc2_sh}.sh", shell=True)
            if ret:
                print(f"Generate shell command excute result: {ret}")
            sh_command = f"bash {self.bin_path}/{mc2_sh}.sh"
            print(f"[INFO] sh command:{sh_command}")
            pre_pids = get_pids(exe_binary)
            try:
                ret = subprocess.run(sh_command, shell=True)
            except subprocess.CalledProcessError:
                print(f"Aclnn command excute result: {ret}")
            # 获取当前进程的pid
            current_pids = get_pids(exe_binary)
            target_pid = [pid for pid in current_pids if pid not in pre_pids]
            print(f"进程名 {exe_binary} 对应的PID: {target_pid}")
            # 等待aaa进程结束
            print("等待进程结束...")
            while True:
                # 检查PID是否存活
                alive_pids = [pid for pid in target_pid if psutil.pid_exists(pid)]
                if not alive_pids:
                    break
                print(f"进程 {exe_binary} (PID:{alive_pids}) 仍在运行...", end="\r")
                time.sleep(1)

    def compare(self, rank=0, diffThd=0.005, pctThd=0.005, check_md5=0):
        op = self.case['op_name']
        np_dtype = NUMPY_DTYPE_MAP[self.output_dtype]
        if self.case['scene'] in ["aclnn"]:  # aclnn场景，npu从已保存文件中读取
            output_path = f"{self.save_path}/output/{rank}/output_npu.bin"
            print(f"output_hccl_path ,{output_path}")
            # #
            # # # 确保所有进程都完成写入
            # torch.distributed.barrier()
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"Output file not found: {output_path}")
            ## bf16场景 单独处理
            if self.output_dtype == 'bf16':
                self.output_npu = torch.tensor(np.fromfile(output_path, np_dtype).astype(np.float32)).to(
                    torch.bfloat16)
            else:
                self.output_npu = torch.tensor(np.fromfile(output_path, np_dtype))
        compare_rank = rank if "aclnn" in self.scene else self.rank
        cpu_input_path = f"{self.save_path}/output/{rank}/output_cpu.bin"
        np_output_cpu = np.fromfile(cpu_input_path, np.float32).astype(np_dtype)
        np_output_npu = self.output_npu.cpu().to(torch.float32).numpy()
        res_output = toolss.data_compare_np(
            np_output_npu,
            np_output_cpu,
            diff_thd=self.diff_thd,
            pct_thd=self.pct_thd,
            max_diff_hd=self.max_diff_hd
        )
        print('[INFO] device_{} output测试的精度结果为：{}'.format(compare_rank, res_output))
        func.write_csv('aclnnQuantAllreduce_result.csv', [self.case_name, compare_rank, res_output])

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
            ## quantAllreduce, 与16bit的allreduce进行性能比较，不需要统计掩盖率
            cal_mask = False
            result_list, memory, performance = func.get_memory_performance(self.case, path_tmp, cal_mask, perf_type)
            if perf_type == "golden":
                print('小算子性能为16bit allreduce性能，不统计内存')
                golden_memory = 0
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
