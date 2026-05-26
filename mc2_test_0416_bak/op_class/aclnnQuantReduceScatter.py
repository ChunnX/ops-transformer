from .common import *
import torch
from script import toolss
from torch.distributed import ReduceOp
from .registry import class_registry
from script import generate_mx_data
import numpy


class QuantReduceScatterGraphModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, scale, hcomm_info, world_size, reduce_op="sum", output_dtype=None, x_dtype=None,scales_dtype=None, hcom_type="mc2"):
        # quant_reducescatter算子是和cpu比，没有小算子级联
        if hcom_type == "mc2":
            output = torch_npu.npu_quant_reduce_scatter(
                x=x,
                scales=scale,
                hcom=hcomm_info,
                world_size=world_size,
                reduce_op=reduce_op,
                output_dtype=output_dtype,
                x_dtype=x_dtype,
                scales_dtype=scales_dtype
            )
        else:
            output = None
        return output


@class_registry("aclnnQuantReduceScatter")
class QuantReduceScatter(MC2_OP):
    def __init__(self, case, rank, dist=None, hcomm_info=None):
        super().__init__(case, rank, dist, hcomm_info)
        self.x_shape = case["input_tensor_shape"]
        self.x_dtype = case["input_tensor_dtype"]
        self.x_range = case["input_tensor_range"]
        self.scale_shape = case["scales_shape"]
        self.scale_dtype = case["scales_dtype"]
        self.scale_range = case["scales_range"]
        self.output_shape = [self.x_shape[0] * self.x_shape[1] // self.world_size if len(self.x_shape) == 3
            else self.x_shape[0] // self.world_size,self.x_shape[-1]]
        self.output_dtype = case["outdtype"]
        self.output_format = "ND"
        self.reduce_op = case["reduce_op"]
        self.seed = case["seed"]
        self.is_mxFp = self.scale_dtype == "fp8_e8m0"
        if self.weight_same:
            self.weight_seed = self.seed + self.rank * 2
        else:
            self.weight_seed = self.seed
        self.x = None
        self.scale = None
        self.output_npu = None
        self.output_golden = None

    def generate_mx_fp_data(self):
        dtype_map = {
            "fp8_e5m2": "float8_e5m2",
            "fp8_e4m3fn": "float8_e4m3fn",
            "fp8_e8m0": "float8_e8m0",
        }
        x_fp32 = self.create_array("x", self.x_shape, self.x_range, "fp32", self.input_path, self.seed)
        x, scale = generate_mx_data.mx_quantize(x_fp32, dtype_map[self.x_dtype], axis=-1, block_size=32, round_mode="rint")
        #根据表示，判断是否是aclnn通路保存输入的bin文件
        if self.save_flag:
            x.tofile(os.path.join(self.input_path, "x.bin"))
            scale.tofile(os.path.join(self.input_path, "scales.bin"))
        self.x = self.tensor(x)
        # 将scale的前2维保存，合并后面维度
        if len(self.scale_shape) == 3:
            self.scale = (self.tensor(scale).reshape(self.scale_shape[0], -1) if "cpu" in self.scene else self.tensor(scale))
        elif len(self.scale_shape) == 4:
            self.scale = (self.tensor(scale).reshape(self.scale_shape[0], self.scale_shape[1], -1) if "cpu" in self.scene else self.tensor(scale))
        else:
            print("scales的shape维度不支持, 当前shape为:", self.scale_shape)

    def get_input_weight(self, weight_same=0):
        if self.is_mxFp:  # mxFp量化
            self.generate_mx_fp_data()
        else:  # pergroup量化
            x = self.create_array("x", self.x_shape, self.x_range, self.x_dtype, self.input_path, self.seed)
            if self.save_flag:
                x.tofile(os.path.join(self.input_path, "x.bin"))
            self.x = self.tensor(x)

    def get_tmp_tensor(self):
        if not self.is_mxFp:
            scale = self.create_array("scales", self.scale_shape, self.scale_range, self.scale_dtype, self.input_path, self.weight_seed)
            if self.save_flag:
                scale.tofile(os.path.join(self.input_path, "scales.bin"))
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
        x = self.x.to(torch.float32)
        scale = self.scale.to(torch.float32)
        x = x.numpy()
        scale = scale.numpy()
        output = self.cpu_dequant(x, scale, 32 if self.is_mxFp else 128)

        dist.all_reduce(output, op=reduce_op_map[self.reduce_op])
        flat = output.reshape(-1)  # 1D view
        scatter_shape_m = flat.numel() // self.world_size  # 按卡数均分
        scatter_output = flat.narrow(0, self.rank * scatter_shape_m, scatter_shape_m).to(OUT_DTYPE_ENUM[self.output_dtype])
        self.save(scatter_output, self.output_path, self.output_cpu_name)

    def get_hccl_mm(self):
        # 只有性能采集的时候有小算子级联
        if self.scene in ["performance", "one_hccl"]:
            dist._reduce_scatter_base(self.output_hccl, self.x_hccl, op=ReduceOp.SUM)

    def get_npu(self):
        self.output_npu= torch_npu.npu_quant_reduce_scatter(
            self.x,
            self.scale,
            self.hcomm_info,
            self.world_size,
            reduce_op="sum",
            output_dtype=OUT_DTYPE_ENUM[self.output_dtype],
            x_dtype=DTYPE_ENUM[self.x_dtype],
            scales_dtype=DTYPE_ENUM[self.scale_dtype],
        )
        if self.save_flag:
            self.save(self.output_npu, self.output_path, self.output_npu_name)
        return

    def get_graph(self, hcom_type="mc2"):
        if hcom_type not in ["mc2", "hccl_mm"]:
            print(f"[ERROR] unsupported hcom_type {hcom_type}")
            sys.exit(1)
        model = QuantReduceScatterGraphModel()
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
        else:  # 没有小算子级联，直接返回空
            return None


    def get_command(self, binary, rankid):
        command = [f"{self.root_path}/cpp/bin/{binary}"]
        command.append(f"--x_shape {','.join(map(str, self.x_shape))}")
        command.append(f"--scales_shape {','.join(map(str, self.scale_shape))}")
        command.append(f"--input_tensor_dtype {self.x_dtype}")
        command.append(f"--input_tensor_format {','.join(map(str, self.case_format))}")
        command.append(f"--output_shape {','.join(map(str, self.output_shape))}")
        command.append(f"--output_dtype {self.output_dtype}")
        command.append(f"--output_format {self.output_format}")
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



    def compare(self, rank=0, diffThd=0.001, pctThd=0.001, check_md5=0):
        np_dtype = NUMPY_DTYPE_MAP[self.output_dtype]
        if self.scene in ["aclnn", "compare"]:  # aclnn场景，golden和npu从已保存文件中读取
            if not os.path.exists(self.output_path):
                raise FileNotFoundError(f"Output file not found: {self.output_path}")
            output_npu_path = os.path.join(self.output_path, self.output_npu_name)
            print(f"output_npu_path ,{output_npu_path}")
            output_cpu_path = os.path.join(self.output_path, self.output_cpu_name)
            if self.output_dtype == 'bf16':
                self.output_npu = torch.tensor(np.fromfile(output_npu_path, np_dtype).astype(np.float32)).to(torch.bfloat16)
            else:
                self.output_npu = torch.tensor(np.fromfile(output_npu_path, np_dtype))
            output_cpu = torch.tensor(np.fromfile(output_cpu_path, np_dtype).astype(np.float32)).to(torch.bfloat16).cpu().to(torch.float32).numpy()
        output_cpu_path = os.path.join(self.output_path, self.output_cpu_name)
        if self.output_dtype == 'fp16':
            output_cpu = torch.tensor(np.fromfile(output_cpu_path, np_dtype).astype(np.float32)).to(torch.float32).numpy()
        elif self.output_dtype == 'bf16':
            output_cpu = torch.tensor(np.fromfile(output_cpu_path, np_dtype).astype(np.float32)).to(torch.bfloat16).cpu().to(torch.float32).numpy()
        else:
            output_cpu = torch.tensor(np.fromfile(output_cpu_path, np_dtype).astype(np.float32)).cpu().to(torch.float32).numpy()
        np_output_npu = self.output_npu.cpu().to(torch.float32).numpy()
        compare_rank = rank if "aclnn" in self.scene else self.rank
        res_output = toolss.data_compare_np(
            np_output_npu,
            output_cpu,
            diff_thd=diffThd,
            pct_thd=pctThd,
            max_diff_hd=0
        )
        print('[INFO] device_{} output测试的精度结果为：{}'.format(compare_rank, res_output))
        func.write_csv('aclnnQuantReducescatter_result.csv', [self.case_name, compare_rank, res_output])

    def get_performace(self, perf_type='npu', soc='910B'):
        tensor = torch.ones(10000, 10000).to(torch.float16).npu()  ## 重型算子计算需要
        if perf_type == 'npu':
            path_tmp = 'mc2_'
        elif perf_type == 'golden':
            path_tmp = 'hccl_mm_'
            # self.x_hccl = self.x.to(TORCH_DTYPE_MAP[self.output_dtype])
            self.x_hccl = torch.zeros_like(self.x, dtype=TORCH_DTYPE_MAP[self.output_dtype])
            self.scatter_output = torch.zeros(self.output_shape, dtype=TORCH_DTYPE_MAP[self.output_dtype]).to(
                self.device)
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
                if i == 1:  ## 防止host bound现象，插入重型算子
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
            result_list, memory, performance = func.get_memory_performance(self.case, path_tmp, cal_mask, perf_type,soc)
            if perf_type == "golden":
                #没有内存标准
                golden_memory = None
                func.write_csv(csv_file, [self.case_name, self.rank, golden_memory, performance])
                print(f'[INFO] 小算子hccl+mm级联 内存为：{golden_memory} M，性能为：{performance} us \n')
            else:
                func.write_csv(csv_file, [self.case_name, self.rank, memory, performance])
                print(f'[INFO] MC2融合算子 内存为：{memory} M，性能为：{performance} us \n')
