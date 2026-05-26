import os

from .common import *
import torch
from torch.distributed import ReduceOp
from .registry import class_registry


class MatmulReduceScatterGraphModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, hcomm_info, world_size, tensor_scatter, is_trans_b, hcom_type):
        if is_trans_b:
            x2 = x2.t()
        if hcom_type == "mc2":
            output = torch_npu.npu_mm_reduce_scatter_base(x1, x2, hcomm_info, world_size, reduce_op="sum")
        else:
            output_golden = torch.matmul(x1, x2)
            dist._reduce_scatter_base(tensor_scatter, output_golden, op=ReduceOp.SUM)  # [512,1920]
            output = tensor_scatter
        return output


@class_registry("aclnnMatmulReduceScatter")
class MatmulReduceScatter(MC2_OP):
    def __init__(self, case, rank, dist=None, hcomm_info=None):
        super().__init__(case, rank, dist, hcomm_info)
        self.is_trans_b = case["is_trans_b"]
        self.world_size = case["world_size"]
        self.seed = case["seed"]
        self.x1_shape, self.x2_shape, *_ = case["input_tensor_shape"]
        self.output_shape = [self.x1_shape[0] // self.world_size, self.x2_shape[0]] if self.is_trans_b \
            else [self.x1_shape[0] // self.world_size, self.x2_shape[1]]
        self.x1_dtype, self.x2_dtype, *_ = case["input_tensor_dtype"]
        self.output_dtype = self.x1_dtype
        self.output_format = "ND"
        self.x1_range, self.x2_range, *_ = case["input_tensor_range"]
        if self.weight_same:
            self.weight_seed = self.seed + self.rank * 2
        else:
            self.weight_seed = self.seed
        self.x1 = None
        self.x2 = None
        self.output_npu = None
        self.output_golden = None
        self.tensor_scatter = None

    def get_tmp_tensor(self):
        self.tensor_scatter = torch.zeros(self.output_shape, dtype=TORCH_DTYPE_MAP[self.x1_dtype]).to(self.device)

    def get_input_weight(self, weight_same=0):
        x1 = self.create_array("x1", self.x1_shape, self.x1_range, self.x1_dtype, self.input_path, self.seed)
        x2 = self.create_array("x2", self.x2_shape, self.x2_range, self.x2_dtype, self.input_path, self.weight_seed)
        self.x1 = self.tensor(x1)
        self.x2 = self.tensor(x2)
        if self.is_trans_b and "graph" not in self.scene:
            self.x2 = self.x2.t()

    def get_hccl_mm(self):
        output = torch.matmul(self.x1, self.x2)
        dist._reduce_scatter_base(self.tensor_scatter, output, op=ReduceOp.SUM)
        self.output_golden = self.tensor_scatter
        return self.output_golden

    def get_npu(self):
        self.output_npu = torch_npu.npu_mm_reduce_scatter_base(
            self.x1,
            self.x2,
            self.hcomm_info,
            self.world_size,
            reduce_op="sum"
        )
        return self.output_npu

    def get_graph(self, hcom_type="mc2"):
        if hcom_type not in ["mc2", "hccl_mm"]:
            print(f"[ERROR] unsupported hcom_type {hcom_type}")
            sys.exit(1)
        model = MatmulReduceScatterGraphModel()
        model = self.define_model(model, self.graph_type)
        output = model(
            self.x1,
            self.x2,
            self.hcomm_info,
            self.world_size,
            tensor_scatter=self.tensor_scatter,
            is_trans_b = self.is_trans_b,
            hcom_type=hcom_type
        )
        if hcom_type == "mc2":
            self.output_npu = output
        else:
            self.output_golden = output
        return output

    def get_cpu(self):
        print('[INFO] device_{} 构造output_cpu数据'.format(self.rank))
        x1 = self.x1.to(torch.float32)
        x2 = self.x2.to(torch.float32)
        output = torch.matmul(x1, x2)
        dist.all_reduce(output, op=dist.ReduceOp.SUM)
        scatter_shape_m = self.input_shape[0] // self.world_size
        scatter_output = output.narrow(0, self.rank * scatter_shape_m, scatter_shape_m)
        self.save(scatter_output, self.output_path, self.output_cpu_name)

    def get_tmp_tensor_memory(self):
        if self.x1_dtype in ["fp16", "bf16"]:
            return self.output_shape[0] * self.output_shape[1] * 2 / 1024 / 1024
        elif self.x1_dtype in ["int8","fp8_e5m2","fp8_e4m3fn","hif8"]:
            return self.output_shape[0] * self.output_shape[1] * 1 / 1024 / 1024

    def get_command(self, binary, rank=0):
        x2_shape = [self.x2_shape[1], self.x2_shape[0]] if self.is_trans_b else self.x2_shape
        if self.case['soc'] == "910D":
            command = [f"nohup /mnt/dmc2_test/bin_path/{binary}"]
            command.append(f"--bin_path {self.bin_path_tmp}")
        else:
            command = [f"{MAIN_PATH}/cpp/bin/{binary}"]
        command.append(f"--is_quant UQUANT")
        command.append(f"--x1_shape {','.join(map(str, self.input_shape))}")
        command.append(f"--x2_shape {','.join(map(str, x2_shape))}")
        command.append(f"--transpose_b_flag {self.is_trans_b}")
        command.append(f"--input_tensor_dtype {','.join(map(str, self.case['input_tensor_dtype']))}")
        command.append(f"--input_tensor_format {','.join(map(str, ['ND', 'ND', 'ND']))}")
        command.append(f"--output_shape {','.join(map(str, self.output_shape))}")
        command.append(f"--output_dtype {self.output_dtype}")
        command.append(f"--output_format {self.output_format}")
        command.append(f"--n_dev {self.world_size}")
        command.append(f"--bin_path {self.bin_path}")
        command.append(f"--loop_cnt {self.loop}")
        if self.case['soc'] == "910D":
            command.append(f"--rankId {rank} \&")
        aclnn_command = " ".join(command)
        return aclnn_command

    def process_aclnn(self):
        self.bin_path_tmp = "/mnt/mc2_test/bin_path/" + self.case_name
        result = subprocess.check_output(["uname", "-a"]).decode("utf-8")
        arch = "aarch" if "aarch64" in result.lower() else "x86"
        exe_binary = self.op_name + "_" + arch
        if self.case['soc'] == "910D":
            if self.case['scene'] == 'aclnn':
                # 根据架构aarch/x86获取待执行二进制名称              
                ranktable_cmd = f"cp -f /mnt/mc2_test/bin_path/{self.world_size}P1K/*.json ./"
                ret = subprocess.check_output(f"echo {ranktable_cmd} >> {self.bin_path}/{self.case_name}.sh", shell=True)
                # 拼二进制执行命令
                for i in range(self.world_size):
                    aclnn_command = self.get_command(exe_binary, i)
                    print(f"[INFO] rank:{i}, aclnn command:{aclnn_command}")
                    # 命令写入shell
                    ret = subprocess.check_output(f"echo {aclnn_command} >> {self.bin_path}/{self.case_name}.sh", shell=True)
                if ret:
                    print(f"Generate shell command excute result: {ret}")
                sh_command = f"bash bin_path/{self.case_name}/{self.case_name}.sh"
                print(f"[INFO] sh command:{sh_command}")
        else:
            aclnn_command = self.get_command(exe_binary)
            print(f"[INFO] aclnn command:{aclnn_command}")
            # 命令写入shell
            ret = subprocess.check_output(f"echo {aclnn_command} > {self.bin_path}/{self.case_name}.sh", shell=True)
            if ret:
                print(f"Generate shell command excute result: {ret}")
            sh_command = f"bash bin_path/{self.case_name}/{self.case_name}.sh"
            print(f"[INFO] sh command:{sh_command}")
            try:
                ret = subprocess.run(sh_command, shell=True)
            except subprocess.CalledProcessError:
                print(f"aclnn command excute result: {ret}")

    def compare(self, rank=0, check_md5=0):
        op = self.case['op_name']
        cpu_output_path = os.path.join(self.output_path, self.output_cpu_name)
        if self.case['scene'] in ["compare", "aclnn"]:  # aclnn场景，golden和npu从已保存文件中读取
            output_npu_path = os.path.join(self.output_path, self.output_npu_name)
            output_golden_path = os.path.join(self.output_path, self.output_hccl_name)
            np_dtype = NUMPY_DTYPE_MAP[self.output_dtype]
            # bf16场景 单独处理
            if self.output_dtype == 'bf16':
                self.output_npu = torch.tensor(np.fromfile(output_npu_path, np_dtype).astype(np.float32)).to(torch.bfloat16)
                self.output_golden = torch.tensor(np.fromfile(output_golden_path, np_dtype).astype(np.float32)).to(torch.bfloat16)
            else:
                self.output_npu = torch.tensor(np.fromfile(output_npu_path, np_dtype))
                self.output_golden = torch.tensor(np.fromfile(output_golden_path, np_dtype))
        output_cpu = torch.from_numpy(np.fromfile(cpu_output_path, np.float32))
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
