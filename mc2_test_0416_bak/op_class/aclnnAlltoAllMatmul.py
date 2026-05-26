import os
from .common import *
import torch
import torch_npu
from script import toolss
from script.func import *
from .registry import class_registry


class AlltoAllMatmulGraphModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    # 暂不支持
    def forward(self, x1, x2, hcomm_info, world_size, bias, all2all_axes, is_alltoall_output, hcom_type):
        if hcom_type == "mc2":
            output, alltoall_output = torch_npu.npu_all_to_all_matmul(x1, x2, hcom=hcomm_info, world_size=world_size,
                                                                      bias=bias, all2all_axes=all2all_axes,
                                                                      all2all_out_flag=is_alltoall_output)
        else:
            pass
        return output, alltoall_output


ACLNN_INDEX_MAP_ALLTOALL = {
    0: "output",
    1: "alltoall_output",
}


@class_registry("aclnnAlltoAllMatmul")
class AlltoAllMatmul(MC2_OP):
    def __init__(self, case, rank, dist=None, hcomm_info=None):
        super().__init__(case, rank, dist, hcomm_info)
        self.x1_shape, self.x2_shape, self.bias_shape, *_ = case["input_tensor_shape"] + [[None]] * 3
        self.x1_range, self.x2_range, self.bias_range, *_ = case["input_tensor_range"] + [[None]] * 3
        self.input_tensor_dtype = case["input_tensor_dtype"] + [[None]] * 3
        self.x1_dtype, self.x2_dtype, self.bias_dtype, *_ = self.input_tensor_dtype
        self.output_dtype = case["output_dtype"]
        self.alltoall_dtype = "fp32" if "cpu" in self.scene else self.x1_dtype
        self.transposeX1 = case["x1_transpose"]
        self.transposeX2 = case["x2_transpose"]
        self.is_bias = case["is_bias"]
        self.is_alltoall_output = case['alltoall_out_flag']
        self.world_size = case["world_size"]
        self.seed = case["seed"]
        temp_shape = [0, 0, 0]
        temp_shape[0] = self.world_size
        temp_shape[1] = self.x1_shape[1] // self.world_size if self.transposeX1 else self.x1_shape[0] // self.world_size
        temp_shape[2] = self.x1_shape[0] if self.transposeX1 else self.x1_shape[1]
        self.temp_shape = temp_shape
        self.alltoall_shape = [temp_shape[1], temp_shape[0] * temp_shape[2]]
        output_shape = [0, 0]
        output_shape[0] = self.alltoall_shape[0]
        output_shape[1] = self.x2_shape[0] if self.transposeX2 else self.x2_shape[1]
        self.output_shape = output_shape
        self.output_format = "ND"
        if self.weight_same:
            self.weight_seed = self.seed + self.rank * 2
        else:
            self.weight_seed = self.seed
        self.output_npu_name = "output_npu.bin"
        self.alltoall_output_npu_name = "alltoall_out_npu.bin"
        self.output_hccl_name = "output_hccl.bin"
        self.alltoall_output_hccl_name = "alltoall_output_hccl.bin"
        self.output_cpu_name = "output_cpu.bin"
        self.alltoall_output_cpu_name = "alltoall_output_cpu.bin"
        self.output_npu = None
        self.output_hccl = None
        self.output_cpu = None
        self.alltoall_output_npu = None
        self.alltoall_output_hccl = None
        self.alltoall_output_cpu = None
        self.x1 = None
        self.x2 = None
        self.bias = None  # 初始化未定义变量
        self.alltoall_out = None

    def get_input_weight(self, weight_same=0):
        x1 = self.create_array("x1", self.x1_shape, self.x1_range, self.x1_dtype, self.input_path, self.seed)
        x2 = self.create_array("x2", self.x2_shape, self.x2_range, self.x2_dtype, self.input_path, self.weight_seed)
        self.x1 = self.tensor(x1)
        self.x2 = self.tensor(x2)
        if self.is_bias:
            bias = self.create_array("bias", self.bias_shape, self.bias_range, self.bias_dtype, self.input_path,
                                     self.weight_seed)
            self.bias = self.tensor(bias)
        self.input = self.x1.t() if self.transposeX1 else self.x1
        self.weight = self.x2.t() if self.transposeX2 else self.x2
        self.input_npu = self.input.contiguous()
        self.weight_npu = self.weight.contiguous()

    def get_tmp_tensor(self):
        self.alltoall_out = torch.zeros(self.temp_shape, dtype=TORCH_DTYPE_MAP[self.alltoall_dtype]).to(self.device)

    def get_tmp_tensor_memory(self):
        if self.alltoall_dtype == 'fp4_e2m1' or self.alltoall_dtype == 'fp4_e1m2':
            return (self.temp_shape[0] * self.temp_shape[1] * self.temp_shape[2] * DTYPE_BYTE_MAP["int8"] + 1 ) / 2 / 1024 / 1024
        return self.temp_shape[0] * self.temp_shape[1] * self.temp_shape[2] * DTYPE_BYTE_MAP[
            self.alltoall_dtype] / 1024 / 1024

    def get_cpu(self):
        input_re = self.input.to(torch.float32).reshape(self.temp_shape)
        alltoall_out = self.alltoall_out
        dist.all_to_all_single(alltoall_out, input_re)
        torch.cpu.synchronize()
        alltoall_out = alltoall_out.permute(1, 0, 2).reshape(self.alltoall_shape)
        weight = self.weight.to(torch.float32)

        if self.is_bias:
            cpu_mm_out = torch.matmul(alltoall_out, weight) + self.bias.to(torch.float32)
        else:
            cpu_mm_out = torch.matmul(alltoall_out, weight)

        self.output_cpu = cpu_mm_out
        self.alltoall_output_cpu = alltoall_out if self.is_alltoall_output else None
        self.save(self.output_cpu, self.output_path, self.output_cpu_name)
        if self.is_alltoall_output:
            self.save(self.alltoall_output_cpu, self.output_path, self.alltoall_output_cpu_name)

    def get_hccl_mm(self):
        # npu 小算子拼接
        input_re = self.input.reshape(self.temp_shape)
        dist.all_to_all_single(self.alltoall_out, input_re)
        alltoall_out = self.alltoall_out.permute(1, 0, 2).reshape(self.alltoall_shape)
        
        # matmul会升精度，再累加，即使是 bf16/bf16/bf16 场景，精度结果也会提高
        if self.is_bias:
            if self.bias_dtype == 'fp32':
                hccl_mm_out = torch.matmul(alltoall_out, self.weight).to(torch.float32) + self.bias
            else:
                hccl_mm_out = torch.matmul(alltoall_out, self.weight) + self.bias
        else:
            hccl_mm_out = torch.matmul(alltoall_out, self.weight)

        self.alltoall_output_hccl = alltoall_out if self.is_alltoall_output else None
        self.output_hccl = hccl_mm_out.to(TORCH_DTYPE_MAP[self.output_dtype]) if self.bias_dtype == 'fp32' else hccl_mm_out
        return self.output_hccl, None, None, None, None, self.alltoall_output_hccl

    def get_npu(self):
        output = torch_npu.npu_all_to_all_matmul(self.input_npu,
                                                 self.weight_npu,
                                                 hcom=self.hcomm_info,
                                                 world_size=self.world_size,
                                                 bias=self.bias,
                                                 all2all_axes=None,
                                                 all2all_out_flag=self.is_alltoall_output)
        self.output_npu, self.alltoall_output_npu = output
        return self.output_npu

    def get_graph(self, hcom_type="mc2"):
        if hcom_type == "mc2":
            model = AlltoAllMatmulGraphModel()
            model = self.define_model(model, self.graph_type)
            out_graph = model(self.input_npu, self.weight_npu, self.hcomm_info, self.world_size, self.bias,
                              all2all_axes=[-2, -1], is_alltoall_output=self.is_alltoall_output, hcom_type="mc2")
            self.output_npu, self.alltoall_output_npu = out_graph
            return self.output_npu
        elif hcom_type == "hccl_mm":
            self.get_hccl_mm()
            return self.output_hccl
        else:
            print("In alltoall get_graph,wrong type..")

    def compare(self, rank=0, check_md5=0):
        outDtype = NUMPY_DTYPE_MAP[self.output_dtype]
        allDtype = NUMPY_DTYPE_MAP[self.alltoall_dtype]
        if self.scene in ["aclnn"]:  # aclnn场景，hccl和npu从已保存文件中读取
            output_npu_path = os.path.join(self.output_path, self.output_npu_name)
            output_hccl_path = os.path.join(self.output_path, self.output_hccl_name)
            if not os.path.exists(output_npu_path):
                raise FileNotFoundError(f"Output file not found: {output_npu_path}")
            self.output_npu = torch.from_numpy(np.fromfile(output_npu_path, outDtype).astype(np.float32))
            self.output_hccl = torch.from_numpy(np.fromfile(output_hccl_path, outDtype).astype(np.float32))
            if self.is_alltoall_output:
                alltoall_out_npu_path = os.path.join(self.output_path, self.alltoall_output_npu_name)
                alltoall_out_hccl_path = os.path.join(self.output_path, self.alltoall_output_hccl_name)
                self.alltoall_output_npu = torch.from_numpy(
                    np.fromfile(alltoall_out_npu_path, allDtype).astype(np.float32))
                self.alltoall_output_hccl = torch.from_numpy(
                    np.fromfile(alltoall_out_hccl_path, allDtype).astype(np.float32))

        output_cpu_path = os.path.join(self.output_path, self.output_cpu_name)
        self.output_cpu = torch.from_numpy(np.fromfile(output_cpu_path, np.float32))
        if self.is_alltoall_output:
            alltoall_out_cpu_path = os.path.join(self.output_path, self.alltoall_output_cpu_name)
            self.alltoall_output_cpu = torch.from_numpy(np.fromfile(alltoall_out_cpu_path, np.float32))

        compare_rank = rank if "aclnn" in self.scene else self.rank
        output_npus = (self.output_npu, self.alltoall_output_npu)
        output_cpus = (self.output_cpu, self.alltoall_output_cpu)
        output_hccls = (self.output_hccl, self.alltoall_output_hccl)
        for index, value in enumerate(output_npus):
            if value is not None:
                output_cpu = output_cpus[index].view(-1).to(torch.float32)
                output_npu = output_npus[index].cpu().view(-1).to(torch.float32)
                output_hccl = output_hccls[index].cpu().contiguous().view(-1).to(torch.float32)
                res = "FAIL"
                rst_npu_cpu = checkResult(output_npu, output_cpu, "{}_npu_cpu".format(ACLNN_INDEX_MAP_ALLTOALL[index]))
                rst_npu_cpu.print_result()
                rst_hccl_cpu = checkResult(output_hccl, output_cpu,
                                           "{}_hccl_cpu".format(ACLNN_INDEX_MAP_ALLTOALL[index]))
                rst_hccl_cpu.print_result()
                str1, str2 = rst_npu_cpu.check_result(rst_hccl_cpu)
                if 'error' in str1:
                    res = 'FAIL'
                else:
                    res = 'PASS'
                print(
                    '[INFO] device_{} OUT {} 精度结果为：{}'.format(compare_rank, ACLNN_INDEX_MAP_ALLTOALL[index], res))
                print('[INFO] device_{} OUT {} 精度计算结束：{} '.format(compare_rank, ACLNN_INDEX_MAP_ALLTOALL[index],
                                                                        time.strftime('%H:%M:%S', time.localtime())))
                with open('{}_result.csv'.format(self.op_name), 'a') as f:
                    f.write(self.case_name + "," + str(compare_rank) + ',' + str(res) + ',' + str(
                        ACLNN_INDEX_MAP_ALLTOALL[index]) + '\n')
        return res

    def get_command(self, binary, rankid):
        input_tensor_format = ["ND", "ND", "ND"]
        command = [f"nohup {self.root_path}/bin_path/{binary}"]
        command.append(f"--bin_path {self.bin_path}")
        command.append(f"--x1_shape {','.join(map(str, self.x1_shape))}")
        command.append(f"--x2_shape {','.join(map(str, self.x2_shape))}")
        command.append(f"--bias_shape {','.join(map(str, self.bias_shape))}")
        command.append(f"--transpose_x1_flag {self.transposeX1}")
        command.append(f"--transpose_x2_flag {self.transposeX2}")
        command.append(f"--input_tensor_dtype {','.join(map(str, self.input_tensor_dtype))}")
        command.append(f"--input_tensor_format {','.join(map(str, input_tensor_format))}")
        command.append(f"--output_shape {','.join(map(str, self.output_shape))}")
        command.append(f"--output_dtype {self.output_dtype}")
        command.append(f"--alltoall_dtype {self.alltoall_dtype}")
        command.append(f"--output_format {self.output_format}")
        command.append(f"--n_dev {self.world_size}")
        command.append(f"--loop_cnt {self.loop}")
        command.append(f"--alltoall_shape {','.join(map(str, self.alltoall_shape))}")
        command.append(f"--bias_flag {self.is_bias}")
        command.append(f"--alltoall_flag {self.is_alltoall_output}")
        command.append(f"--rankId {rankid} \&")
        return " ".join(command)

    def process_aclnn(self):
        self.root_path = os.path.abspath(os.getcwd())
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
                    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print(f"\n[{current_time}] 进程 {exe_binary} 已结束...", end="\n")
                    break
                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f"[{current_time}] 进程 {exe_binary} (PID:{alive_pids}) 仍在运行...", end="\r")
                time.sleep(1)
