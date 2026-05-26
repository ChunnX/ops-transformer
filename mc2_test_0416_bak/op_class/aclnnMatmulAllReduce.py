import os.path

from .common import *
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp
from .registry import class_registry
import os


class MatmulAllReduceGraphModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, hcom, bias, is_bias, is_trans_b, hcom_type):
        if is_trans_b:
            x2 = x2.t()
        if hcom_type == "mc2":
            output = torch_npu.npu_mm_all_reduce_base(
                x1=x1,
                x2=x2,
                hcom=hcom,
                reduce_op="sum",
                bias=bias,
            )
        else:
            output = torch.matmul(x1, x2)
            if is_bias:
                output = torch.add(output, bias)
            dist.all_reduce(output, op=ReduceOp.SUM)
        return output


@class_registry("aclnnMatmulAllReduce")
class MatmulAllReduce(MC2_OP):
    def __init__(self, case, rank, dist=None, hcomm_info=None):
        super().__init__(case, rank, dist, hcomm_info)
        self.is_bias = case["is_bias"]
        self.is_trans_b = case["is_trans_b"]
        self.world_size = case["world_size"]
        self.seed = case["seed"] if "seed" in case else self.rank
        self.x1_shape, self.x2_shape, self.bias_shape = case["input_tensor_shape"]
        self.x1_range, self.x2_range, self.bias_range = case["input_tensor_range"]
        self.x1_dtype, self.x2_dtype, self.bias_dtype = case["input_tensor_dtype"]
        self.output_shape = (self.x1_shape[:-1] + self.x2_shape[:1]) if self.is_trans_b else (self.x1_shape[:-1] + self.x2_shape[-1:])
        self.output_dtype = self.x1_dtype
        self.output_format = "ND"
        self.x1 = None
        self.x2 = None
        self.bias = None

    def get_input_weight(self, weight_same=0):
        x1 = self.create_array("x1", self.x1_shape, self.x1_range, self.x1_dtype, self.input_path, self.seed)
        x2 = self.create_array("x2", self.x2_shape, self.x2_range, self.x2_dtype, self.input_path, self.seed)
        self.x1 = self.tensor(x1)
        self.x2 = self.tensor(x2)
        # 图模式转置要放在图内
        if self.is_trans_b and "graph" not in self.scene:
            self.x2 = self.x2.t()

    def get_tmp_tensor(self):
        if self.is_bias:
            bias = self.create_array("bias", self.bias_shape, self.bias_range, self.bias_dtype, self.input_path, self.rank * 3)
            self.bias = self.tensor(bias)

    def get_hccl_mm(self):
        output = torch.matmul(self.x1, self.x2)
        if self.is_bias:
            output += self.bias
        dist.all_reduce(output, op=ReduceOp.SUM)
        self.output_golden = output
        return self.output_golden

    def get_npu(self):
        self.output_npu = torch_npu.npu_mm_all_reduce_base(
            x1=self.x1,
            x2=self.x2,
            hcom=self.hcomm_info,
            reduce_op="sum",
            bias=self.bias,
        )
        return self.output_npu

    def get_graph(self, hcom_type="mc2"):
        if hcom_type not in ["mc2", "hccl_mm"]:
            print(f"[ERROR] unsupported hcom_type {hcom_type}")
            sys.exit(1)
        model = MatmulAllReduceGraphModel()
        model = self.define_model(model, self.graph_type)
        output = model(
            self.x1,
            self.x2,
            self.hcomm_info,
            self.bias,
            self.is_bias,
            self.is_trans_b,
            hcom_type
        )
        if hcom_type == "mc2":
            self.output_npu = output
        else:
            self.output_golden = output
        return output

    def get_cpu(self):
        x1 = self.x1.to(torch.float32)
        x2 = self.x2.to(torch.float32)
        output = torch.matmul(x1, x2)
        if self.is_bias:
            bias = self.bias.to(torch.float32)
            output += bias
        dist.all_reduce(output, op=dist.ReduceOp.SUM)
        self.save(output, self.output_path, self.output_cpu_name)

    def get_command(self, binary, rank=0, is_david=0):
        input_tensor_dtype = ','.join(map(str, self.case['input_tensor_dtype']))
        input_tensor_format = ','.join(map(str, ['ND', 'ND', 'ND']))
        bin_path = self.bin_path
        is_quant = "UQUANT"
        command = [f"{MAIN_PATH}/bin_path/{binary}"] if is_david else [f"{MAIN_PATH}/cpp/bin/{binary}"]
        command.append(f"--is_quant {is_quant}")
        command.append(f"--x1_shape {','.join(map(str, self.x1_shape))}")
        tmp_x2_shape = [self.x2_shape[1], self.x2_shape[0]] if self.is_trans_b else self.x2_shape
        command.append(f"--x2_shape {','.join(map(str, tmp_x2_shape))}")
        command.append(f"--transpose_b_flag {int(self.is_trans_b)}")
        command.append(f"--input_tensor_dtype {input_tensor_dtype}")
        command.append(f"--input_tensor_format {input_tensor_format}")
        command.append(f"--output_shape {','.join(map(str, self.output_shape))}")
        command.append(f"--output_dtype {self.output_dtype}")
        command.append(f"--output_format {self.output_format}")
        command.append(f"--n_dev {self.world_size}")
        command.append(f"--bin_path {bin_path}")
        command.append(f"--loop_cnt {self.loop}")
        command.append(f"--bias_flag {self.is_bias}")
        command.append(f"--bias_shape {self.output_shape[-1]}")
        if is_david:
            command.append(f"--rankId {rank} \&")
        aclnn_command = " ".join(command)
        return aclnn_command

    def process_aclnn(self):
        # 根据架构aarch/x86获取待执行二进制名称
        result = subprocess.check_output(["uname", "-a"]).decode("utf-8")
        arch = "aarch" if "aarch64" in result.lower() else "x86"
        if self.scene in ["aclnn", "one_hccl_aclnn"]:
            exe_binary = self.op_name + "_" + arch
        else:
            exe_binary = f'hccl_allreduce_{arch}'
        if self.case['soc'] == "910D":  
            if self.case['scene'] == 'aclnn':
                # 移除已有shell文件
                mv_command = f"rm -rf bin_path/{self.case_name}/{self.case_name}*.sh"
                ret = subprocess.run(mv_command, shell=True)
                # 拼二进制执行命令
                for i in range(self.world_size):
                    aclnn_command = self.get_command(exe_binary, i, 1)
                    print(f"[INFO] rank:{i}, aclnn command:{aclnn_command}")
                    # 命令写入shell
                    ret = subprocess.check_output(f"echo {aclnn_command} >> {self.bin_path}/{self.case_name}.sh", shell=True)
                if ret:
                    print(f"Generate shell command excute result: {ret}")
                sh_command = f"bash bin_path/{self.case_name}/{self.case_name}.sh"
                print(f"[INFO] aclnn sh command:{sh_command}")
        else:
            aclnn_command = self.get_command(exe_binary)
            print(f"[INFO] , aclnn command:{aclnn_command}")
            # 命令写入shell
            ret = subprocess.check_output(f"echo {aclnn_command} > {self.bin_path}/{self.case_name}.sh", shell=True)
            if ret:
                print(f"Generate shell command excute result: {ret}")
            sh_command = f"bash bin_path/{self.case_name}/{self.case_name}.sh"
            print(f"[INFO] sh command:{sh_command}")
        pre_pids = get_pids(exe_binary)
        try:
            ret = subprocess.run(sh_command, shell=True)
        except subprocess.CalledProcessError:
            print(f"Aclnn command excute result: {ret}")
        if self.case['soc'] == "910D": ## david场景等待进程结束
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

    def compare(self, rank=0, check_md5=0):
        op = self.case['op_name']
        cpu_output_path = os.path.join(self.output_path, self.output_cpu_name)
        if "aclnn" in self.scene:  # aclnn场景，golden和npu从已保存文件中读取
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
