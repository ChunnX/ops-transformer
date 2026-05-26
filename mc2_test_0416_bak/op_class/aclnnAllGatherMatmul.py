import os

from .common import *
import torch
import hashlib
from script import toolss
from script.func import *
from .registry import class_registry


class AllGatherMatmulGraphModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, hcomm_info, world_size, is_gather_output, tensor_allgather, is_trans_b, hcom_type):
        if is_trans_b:
            x2 = x2.t()
        if hcom_type == "mc2":
            output, gather_output = torch_npu.npu_all_gather_base_mm(
                x1, x2, hcomm_info, world_size, gather_output=is_gather_output
            )
        else:
            dist._all_gather_base(tensor_allgather, x1)
            output = torch.matmul(tensor_allgather, x2)
            gather_output = tensor_allgather if is_gather_output else None
        return output, gather_output


@class_registry("aclnnAllGatherMatmul")
class AllGatherMatmul(MC2_OP):
    def __init__(self, case, rank, dist=None, hcomm_info=None):
        super().__init__(case, rank, dist, hcomm_info)
        # allgather涉及
        self.is_gather_output = case['gather_output']
        self.is_trans_b = case["is_trans_b"]
        self.world_size = case["world_size"]
        self.seed = case["seed"]
        self.x1_shape, self.x2_shape, *_ = case["input_tensor_shape"]
        self.gather_out_shape = [self.x1_shape[0] * self.world_size] + self.x1_shape[1:]
        self.output_shape = [self.gather_out_shape[0], self.x2_shape[0]] if self.is_trans_b \
            else [self.gather_out_shape[0], self.x2_shape[1]]
        self.x1_dtype, self.x2_dtype, *_ = case["input_tensor_dtype"]
        self.output_dtype = self.x1_dtype
        self.output_format = "ND"
        self.x1_range, self.x2_range, *_ = case["input_tensor_range"]
        if self.weight_same:
            self.weight_seed = self.seed + self.rank * 2
        else:
            self.weight_seed = self.seed
        self.gather_output_npu_name = "gatherout_npu.bin"
        self.gather_output_hccl_name = "gather_output_hccl.bin"
        self.gather_output_cpu_name = "gather_output_cpu.bin"
        self.diff_thd = 0.005
        self.pct_thd = 0.005
        self.max_diff_hd = 2
        self.gather_diff_thd = 0.000001
        self.gather_pct_thd = 0.000001
        self.gather_max_diff_hd = 0
        self.output_npu = None
        self.gather_output_npu = None
        self.output_golden = None
        self.gather_output_golden = None
        self.x1 = None
        self.x2 = None
        self.tensor_allgather = None

    def get_tmp_tensor(self):
        self.tensor_allgather = torch.zeros(self.gather_out_shape, dtype=TORCH_DTYPE_MAP[self.x1_dtype]).to(self.device)

    def get_input_weight(self, weight_same=0):
        x1 = self.create_array("x1", self.x1_shape, self.x1_range, self.x1_dtype, self.input_path, self.seed)
        x2 = self.create_array("x2", self.x2_shape, self.x2_range, self.x2_dtype, self.input_path, self.weight_seed)
        self.x1 = self.tensor(x1)
        self.x2 = self.tensor(x2)
        if self.is_trans_b and "graph" not in self.scene:
            self.x2 = self.x2.t()
        self.x1 = self.x1 if "cpu" in self.scene else self.x1.npu()
        self.x2 = self.x2 if "cpu" in self.scene else self.x2.npu()

    def get_hccl_mm(self):
        dist._all_gather_base(self.tensor_allgather, self.x1)
        self.output_golden = torch.matmul(self.tensor_allgather, self.x2)
        self.gather_output_golden = self.tensor_allgather
        if self.scene == "aclnn_golden":
            self.save(self.output_golden, self.output_path, self.output_hccl_name)
            if self.is_gather_output:
                self.save(self.gather_output_golden, self.output_path, self.gather_output_hccl_name)

    def get_npu(self):
        self.output_npu, self.gather_output_npu = torch_npu.npu_all_gather_base_mm(
            self.x1,
            self.x2,
            self.hcomm_info,
            self.world_size,
            gather_output=self.is_gather_output
        )
        return self.output_npu

    def get_graph(self, hcom_type="mc2"):
        if hcom_type not in ["mc2", "hccl_mm"]:
            print(f"[ERROR] unsupported hcom_type {hcom_type}")
            sys.exit(1)
        model = AllGatherMatmulGraphModel()
        model = self.define_model(model, self.graph_type)
        out_graph = model(
            self.x1,
            self.x2,
            self.hcomm_info,
            self.world_size,
            is_gather_output=self.is_gather_output,
            tensor_allgather=self.tensor_allgather,
            is_trans_b = self.is_trans_b,
            hcom_type=hcom_type
        )
        if hcom_type == "mc2":
            self.output_npu, self.gather_output_npu = out_graph
            return self.output_npu
        else:
            self.output_golden, self.gather_output_golden = out_graph

    def get_cpu(self):
        x1 = self.x1.to(torch.float32)
        x2 = self.x2.to(torch.float32)
        all_gather_list = [torch.zeros_like(x1) for _ in range(self.world_size)]
        dist.all_gather(all_gather_list, x1)
        all_gather_out = torch.cat(all_gather_list, dim=0)
        cpu_output = torch.matmul(all_gather_out, x2)
        self.save(cpu_output, self.output_path, self.output_cpu_name)

    def compare(self, rank=0, diffThd=0.005, pctThd=0.005, check_md5=0):
        if self.scene in ["aclnn", "compare"]:  # aclnn场景，golden和npu从已保存文件中读取
            output_npu_path = os.path.join(self.output_path, self.output_npu_name)
            output_hccl_path = os.path.join(self.output_path, self.output_hccl_name)
            gatherout_npu_path = os.path.join(self.output_path, self.gather_output_npu_name)
            gatherout_hccl_path = os.path.join(self.output_path, self.gather_output_hccl_name)
            dtype = NUMPY_DTYPE_MAP[self.x1_dtype]
            self.output_npu = torch.from_numpy(np.fromfile(output_npu_path, dtype).astype(np.float32))
            self.output_golden = torch.from_numpy(np.fromfile(output_hccl_path, dtype).astype(np.float32))
            if self.is_gather_output:
                self.gather_output_npu = torch.tensor(np.fromfile(gatherout_npu_path, dtype).astype(np.float32))
                self.gather_output_golden = torch.tensor(np.fromfile(gatherout_hccl_path, dtype).astype(np.float32))
        np_output_npu = self.output_npu.cpu().to(torch.float32).numpy()
        np_output_golden = self.output_golden.cpu().to(torch.float32).numpy()
        compare_rank = rank if "aclnn" in self.scene else self.rank
        # inf/nan计数
        func.cal_inf_nan(rank, np_output_npu, 'npu')
        func.cal_inf_nan(rank, np_output_golden, 'golden')
        if self.weight_same:
            m = hashlib.md5()  # 创建一个hashlib.md5()对象
            m.update(np_output_npu)
            md5 = m.hexdigest()
            print(f"device {compare_rank} out md5 is: {md5}")
            func.write_csv('allgather_md5.csv', [self.case_name, compare_rank, md5])
        if self.is_gather_output:
            np_gather_output_npu = self.gather_output_npu.to(torch.float32).cpu().numpy()
            np_gather_output_golden = self.gather_output_golden.cpu().to(torch.float32).numpy()
            res_gather = toolss.data_compare_np(
                np_gather_output_npu,
                np_gather_output_golden,
                diff_thd=self.gather_diff_thd,
                pct_thd=self.gather_pct_thd,
                max_diff_hd=self.gather_max_diff_hd
            )
            print('[INFO] device_{} gather测试的精度结果为：{}'.format(compare_rank, res_gather))
            func.write_csv('aclnnAllGatherMatmul_result.csv', [self.case_name, compare_rank, res_gather, 'gather_out'])
        res_output = toolss.data_compare_np(
            np_output_npu,
            np_output_golden,
            diff_thd=self.diff_thd,
            pct_thd=self.pct_thd,
            max_diff_hd=self.max_diff_hd
        )
        print('[INFO] device_{} output测试的精度结果为：{}'.format(compare_rank, res_output))
        func.write_csv('aclnnAllGatherMatmul_result.csv', [self.case_name, compare_rank, res_output])

    def get_tmp_tensor_memory(self):
        if self.x1_dtype in ["fp16", "bf16"]:
            return self.x1_shape[0] * self.x1_shape[1] * 2 / 1024 / 1024
        elif self.x1_dtype in ["int8","fp8_e4m3fn","fp8_e5m2","hif8"]:
            return self.x1_shape[0] * self.x1_shape[1] * 1 / 1024 / 1024

    def process_aclnn(self):
        input_tensor_format = ["ND", "ND", "ND"]
        bin_path_tmp = "/mnt/mc2_test/bin_path/" + self.case_name
        result = subprocess.check_output(["uname", "-a"]).decode("utf-8")
        arch = "aarch" if "aarch64" in result.lower() else "x86"
        exe_binary = self.op_name + "_" + arch
        if self.case['soc'] == "910D":
            command = [f"nohup /mnt/mc2_test/bin_path/{exe_binary}"]
            command.append(f"--bin_path {bin_path_tmp}")
        else:
            command = [f"{MAIN_PATH}/cpp/bin/{exe_binary}"]
            command.append(f"--bin_path {self.bin_path}")
        x2_shape = self.x2_shape if not self.is_trans_b else [self.x2_shape[1], self.x2_shape[0]]
        command.append(f"--x1_shape {','.join(map(str, self.x1_shape))}")
        command.append(f"--x2_shape {','.join(map(str, x2_shape))}")
        command.append(f"--transpose_b_flag {self.is_trans_b}")
        command.append(f"--input_tensor_dtype {','.join(map(str, self.case['input_tensor_dtype']))}")
        command.append(f"--input_tensor_format {','.join(map(str, input_tensor_format))}")
        command.append(f"--output_shape {','.join(map(str, self.output_shape))}")
        command.append(f"--output_dtype {self.output_dtype}")
        command.append(f"--output_format {self.output_format}")
        command.append(f"--n_dev {self.world_size}")
        command.append(f"--loop_cnt {self.loop}")
        command.append(f"--gatherout_shape {','.join(map(str, self.gather_out_shape))}")

        if self.case['soc'] == "910D":
            if self.scene == 'aclnn':
                # 根据架构aarch/x86获取待执行二进制名称
                ranktable_cmd = f"cp -f /mnt/mc2_test/bin_path/{self.world_size}P1K/*.json ./"
                env_cmd = f"export HCCL_TOPO_FILE_PATH=/mnt/mc2_test/bin_path/{self.case_name}/topo.json"
                subprocess.check_output(f"echo {ranktable_cmd} >> {self.bin_path}/{self.case_name}_mc2.sh", shell=True)
                subprocess.check_output(f"echo {env_cmd} >> {self.bin_path}/{self.case_name}_mc2.sh", shell=True)
                # 拼二进制执行命令
                for i in range(self.world_size):
                    tmp_command = command.copy()
                    tmp_command.append(f"--rankId {i} \&")
                    aclnn_command = " ".join(tmp_command)
                    print(f"[INFO] rank:{i}, aclnn command:{aclnn_command}")
                    # 命令写入shell
                    ret = subprocess.check_output(f"echo {aclnn_command} >> {self.bin_path}/{self.case_name}_mc2.sh", shell=True)
                if ret:
                    print(f"Generate shell command excute result: {ret}")
                sh_command = f"bash bin_path/{self.case_name}/{self.case_name}_mc2.sh"
                print(f"[INFO] sh command:{sh_command}")
        else:
            aclnn_command = " ".join(command)
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
