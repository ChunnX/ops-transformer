from .common import *
import torch
from torch.distributed import ReduceOp
from .registry import class_registry
from script import generate_mx_data
from .aclnnMatmulReduceScatter import MatmulReduceScatter


class MatmulReduceScatterV2GraphModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, hcomm_info, world_size, bias, x1scale, x2scale,
                out_dtype, x1_dtype, x2_dtype, x1scale_dtype, x2scale_dtype, block_size, group_sizes,
                tensor_scatter, is_quant, is_bias, is_trans_b, is_per_block, is_mxfp, hcom_type):
        if is_trans_b:
            x2 = x2.t()
            if is_mxfp or is_per_block:
                x2scale = torch.transpose(x2scale, 0, 1)
        if hcom_type == "mc2":
            output, _ = torch_npu.npu_quant_mm_reduce_scatter(
                x1,
                x2,
                hcomm_info,
                world_size,
                bias=bias,
                reduce_op="sum",
                x1_scale=x1scale,
                x2_scale=x2scale,
                quant_scale=None,
                block_size=block_size,
                comm_turn=0,
                group_sizes=group_sizes,
                amax_output=False,
                y_dtype=OUT_DTYPE_ENUM[out_dtype],
                x1_dtype=DTYPE_ENUM[x1_dtype],
                x2_dtype=DTYPE_ENUM[x2_dtype],
                x1_scale_dtype=DTYPE_ENUM[x1scale_dtype],
                x2_scale_dtype=DTYPE_ENUM[x2scale_dtype],
            )
        else:
            if not is_quant:
                output = torch.matmul(x1, x2)
                if is_bias:
                    output += bias
            else:
                output = torch_npu.npu_quant_matmul(
                    x1=x1,
                    x2=x2,
                    pertoken_scale=x1scale,
                    scale=x2scale,
                    offset=None,
                    bias=bias,
                    output_dtype=OUT_DTYPE_ENUM[out_dtype],
                    x1_dtype=DTYPE_ENUM[x1_dtype],
                    x2_dtype=DTYPE_ENUM[x2_dtype],
                    pertoken_scale_dtype=DTYPE_ENUM[x1scale_dtype],
                    scale_dtype=DTYPE_ENUM[x2scale_dtype],
                    group_sizes=group_sizes,
                    y_scale=None,
                )
            dist._reduce_scatter_base(tensor_scatter, output, op=ReduceOp.SUM)
            output = tensor_scatter
        return output


@class_registry("aclnnMatmulReduceScatterV2")
class MatmulReduceScatterV2(MatmulReduceScatter):
    def __init__(self, case, rank, dist=None, hcomm_info=None):
        super().__init__(case, rank, dist, hcomm_info)
        self.per_block_flag = case['per_block_flag']
        self.block_size = case['block_size'] if 'block_size' in case else 128
        self.group_size = case["group_size"]
        self.group_sizes = self.unpack_group_size(self.group_size)
        self.is_bias = case['is_bias']
        if self.is_bias:
            self.bias_shape = case["input_tensor_shape"][2]
            self.bias_dtype = case["input_tensor_dtype"][2]
            self.bias_range = case["input_tensor_range"][2]
        self.x1scale_shape = case['x1scale_shape']
        self.x1scale_dtype = case["x1scale_dtype"]
        self.x1scale_range = case["x1scale_range"]
        self.x2scale_shape = case['x2scale_shape']
        self.x2scale_dtype = case['x2scale_dtype']
        self.x2scale_range = case['x2scale_range']
        self.output_dtype = case['outdtype']
        self.is_quant = 0 if self.x1_dtype in ["fp16", "bf16"] else 1
        self.is_mxFp = True if self.case['x1scale_dtype'] == "fp8_e8m0" else False
        self.x1scale = None
        self.x2scale = None
        self.bias = None

    def generate_mx_fp_data(self):
        dtype_map = {
            "fp8_e5m2": "float8_e5m2",
            "fp8_e4m3fn": "float8_e4m3fn",
            "fp8_e8m0": "float8_e8m0",
            "fp4_e2m1": "float4_e2m1",
            "fp4_e1m2": "float4_e1m2",
        }
        x1_fp32 = self.create_array("x1", self.x1_shape, self.x1_range, "fp32", self.input_path, self.seed)
        x2_fp32 = self.create_array("x2", self.x2_shape, self.x2_range, "fp32", self.input_path, self.weight_seed)
        x1, x1scale = generate_mx_data.mx_quantize(x1_fp32, dtype_map[self.x1_dtype], axis=-1, block_size=32, round_mode="rint")
        x2, x2scale = generate_mx_data.mx_quantize(x2_fp32, dtype_map[self.x2_dtype], axis=-1, block_size=32, round_mode="rint")
        if self.is_trans_b and "graph" not in self.scene:
            x2 = x2.transpose()
            x2scale = x2scale.transpose(1, 0, 2)
        # x2 [n,k] x2_scale [n, ceil(k,32)]
        x1 = func.pack_int4(x1) if self.x1_dtype in ['fp4_e2m1', 'fp4_e1m2'] else x1
        x2 = func.pack_int4(x2) if self.x2_dtype in ['fp4_e2m1', 'fp4_e1m2'] else x2
        #self.save(x1, self.input_path, "x1.bin")
        #self.save(x2, self.input_path, "x2.bin")
        #self.save(x1scale, self.input_path, "x1scale.bin")
        #self.save(x2scale, self.input_path, "x2scale.bin")
        self.x1 = self.tensor(x1)
        self.x2 = self.tensor(x2)
        self.x1scale = self.tensor(x1scale).reshape(self.x1scale_shape[0], -1) if "cpu" in self.scene else self.tensor(x1scale)
        self.x2scale = self.tensor(x2scale).reshape(-1,self.x2scale_shape[0]) if "cpu" in self.scene else self.tensor(x2scale)

    def get_input_weight(self, weight_same=0):
        if self.is_mxFp:
            self.generate_mx_fp_data()
        else:
            super().get_input_weight(self.weight_same)

    def get_tmp_tensor(self):
        if self.is_bias:
            bias = self.create_array("bias", self.bias_shape, self.bias_range, self.bias_dtype, self.input_path, self.rank * 3)
            self.bias = self.tensor(bias)
        if "cpu" not in self.scene:
            self.tensor_scatter = torch.zeros(self.output_shape, dtype=TORCH_DTYPE_MAP[self.output_dtype]).to(self.device)
        if not self.is_mxFp and self.is_quant:
            x1scale = self.create_array("x1scale", self.x1scale_shape, self.x1scale_range,
                                        self.x1scale_dtype, self.input_path, self.rank * 11)
            x2scale = self.create_array("x2scale", self.x2scale_shape, self.x2scale_range,
                                        self.x2scale_dtype, self.input_path, self.rank * 11)
            self.x1scale = self.tensor(x1scale)
            self.x2scale = self.tensor(x2scale)
            if self.is_trans_b and self.per_block_flag and "graph" not in self.scene:
                self.x2scale = self.x2scale.t()

    def get_hccl_mm(self):
        if not self.is_quant:
            output = torch.matmul(self.x1, self.x2)
            if self.is_bias:
                output += self.bias
        else:
            output = torch_npu.npu_quant_matmul(
                x1=self.x1,
                x2=self.x2,
                pertoken_scale=self.x1scale,
                scale=self.x2scale,
                offset=None,
                bias=self.bias,
                output_dtype=OUT_DTYPE_ENUM[self.output_dtype],
                x1_dtype=DTYPE_ENUM[self.x1_dtype],
                x2_dtype=DTYPE_ENUM[self.x2_dtype],
                pertoken_scale_dtype=DTYPE_ENUM[self.x1scale_dtype],
                scale_dtype=DTYPE_ENUM[self.x2scale_dtype],
                group_sizes=self.group_sizes,
                y_scale=None,
            )
        dist._reduce_scatter_base(self.tensor_scatter, output, op=ReduceOp.SUM)
        self.output_golden = self.tensor_scatter
        return self.output_golden

    def get_cpu(self):
        x1 = self.x1.to(torch.float32)
        x2 = self.x2.to(torch.float32)
        bias = self.bias.to(torch.float32) if self.is_bias else None
        if not self.is_quant:  # 情形1:非量化
            output = torch.matmul(x1, x2)
            if self.is_bias:
                output += bias
        else:
            x1scale = self.x1scale.to(torch.float32)
            x2scale = self.x2scale.to(torch.float32)
            if self.per_block_flag:  # 情形4：per_block量化
                output = func.per_block_cpu_compute(self.group_size, x1, x2, x1scale, x2scale)
            elif self.is_mxFp:  # 情形5:mxfp量化
                x1 = x1.numpy()
                x2 = x2.numpy()
                x1scale = x1scale.numpy()
                x2scale = x2scale.numpy()
                output = self.mxfp_cpu_compute(x1, x2, x1scale, x2scale)
                if self.is_bias:
                    output += bias
            else:  # 情形2/3 per_tensor
                output = torch.matmul(x1, x2)
                if self.is_bias:
                    output += bias
                double_scale = func.scale_generate(x1scale.numpy() * x2scale.numpy())
                double_scale_tensor = torch.unsqueeze(torch.from_numpy(double_scale), dim=1).to(torch.float32)
                output *= double_scale_tensor
        dist.all_reduce(output, op=dist.ReduceOp.SUM)
        scatter_shape_m = self.x1_shape[0] // self.world_size
        scatter_output = output.narrow(0, self.rank * scatter_shape_m, scatter_shape_m)
        self.save(scatter_output, self.output_path, self.output_cpu_name)

    def get_npu(self):
        self.output_npu, _ = torch_npu.npu_quant_mm_reduce_scatter(
            self.x1,
            self.x2,
            self.hcomm_info,
            self.world_size,
            bias=self.bias,
            reduce_op="sum",
            x1_scale=self.x1scale,
            x2_scale=self.x2scale,
            quant_scale=None,
            block_size=self.block_size,
            comm_turn=0,
            group_sizes=self.group_sizes,
            amax_output=False,
            y_dtype=OUT_DTYPE_ENUM[self.output_dtype],
            x1_dtype=DTYPE_ENUM[self.x1_dtype],
            x2_dtype=DTYPE_ENUM[self.x2_dtype],
            x1_scale_dtype=DTYPE_ENUM[self.x1scale_dtype],
            x2_scale_dtype=DTYPE_ENUM[self.x2scale_dtype],
        )
        return

    def get_graph(self, hcom_type="mc2"):
        if hcom_type not in ["mc2", "hccl_mm"]:
            print(f"[ERROR] unsupported hcom_type {hcom_type}")
            sys.exit(1)
        model = MatmulReduceScatterV2GraphModel()
        model = self.define_model(model, self.graph_type)
        out_graph = model(
            self.x1,
            self.x2,
            self.hcomm_info,
            self.world_size,
            self.bias,
            self.x1scale,
            self.x2scale,
            self.output_dtype,
            self.x1_dtype,
            self.x2_dtype,
            self.x1scale_dtype,
            self.x2scale_dtype,
            self.block_size,
            self.group_sizes,
            self.tensor_scatter,
            self.is_quant,
            self.is_bias,
            self.is_trans_b,
            self.per_block_flag,
            self.is_mxFp,
            hcom_type,
        )
        if hcom_type == "mc2":
            self.output_npu = out_graph
        else:
            self.output_golden = out_graph

    def get_command(self, binary, rankid):
        command = [f"nohup /mnt/mc2_test/bin_path/{binary}"]
        command.append(f"--x1_shape {','.join(map(str, self.input_shape))}")
        command.append(f"--x2_shape {','.join(map(str, self.weight_shape))}")
        command.append(f"--bias_shape {self.weight_shape[-1]}")
        command.append(f"--quant_flag {self.is_quant}")
        command.append(f"--transpose_a_flag {self.is_trans_a}")
        command.append(f"--transpose_b_flag {self.is_trans_b}")
        command.append(f"--block_size {str(self.block_size)}")
        command.append(f"--block_mode {str(self.block_mode)}")
        command.append(f"--per_block_flag {str(self.per_block_flag).lower()}")
        command.append(f"--scale1_shape {','.join(map(str, self.case['scaleInv1_shape']))}")
        command.append(f"--scale2_shape {','.join(map(str, self.case['scaleInv2_shape']))}")
        command.append(f"--bias_flag {self.is_bias}")
        command.append(f"--input_tensor_dtype {','.join(map(str, self.case['input_tensor_dtype']))}")
        command.append(f"--input_tensor_format {','.join(map(str, self.case_format))}")
        command.append(f"--output_shape {','.join(map(str, self.output_shape))}")
        command.append(f"--output_dtype {self.output_dtype}")
        command.append(f"--output_format {self.output_format}")
        command.append(f"--n_dev {self.world_size}")
        command.append(f"--bin_path {self.bin_path_tmp}")
        command.append(f"--loop_cnt {self.loop}")
        command.append(f"--mmout_shape {','.join(map(str, self.matmul_shape))}")
        if self.is_quant:
            command.append("--is_hccl_mm 1")
        if self.is_mxFp:
            command.append("--is_mxfp 1")
        command.append(f"--rankId {rankid} \&")
        aclnn_command = " ".join(command)
        return aclnn_command

    def process_aclnn(self):
        self.root_path = os.path.abspath(os.getcwd())
        self.case_format = ['ND', 'ND', 'ND']
        self.output_shape = [self.input_shape[0] // self.world_size, self.weight_shape[1]]
        self.matmul_shape = [self.input_shape[0], self.weight_shape[1]]
        self.output_dtype = self.outdtype
        self.output_format = self.case_format[0]
        self.bin_path = self.root_path + "/bin_path/" + self.case_name
        self.bin_path_tmp = "/mnt/mc2_test/bin_path/" + self.case_name
        print(f"bin_path: {self.bin_path}")
        result = subprocess.check_output(["uname", "-a"]).decode("utf-8")
        arch = "aarch" if "aarch64" in result.lower() else "x86"
        exe_binary = self.op_name + "_" + arch
        self.loop = 1
        if self.case['scene'] == 'aclnn':
            # else: # fp16/bf16场景，仅计算mc2结果，小算子级联结果取1971结果，执行二进制reducescatter_v2_using_malloc_aarch_4p1k
            #     exe_binary = "mm_reducescatter_v2_aarch_4p1k"
            exe_binary_golden = "hccl_reducescatter_v2_using_malloc_aarch_4p1k"
            ranktable_cmd = f"cp -f /mnt/mc2_test/bin_path/{self.world_size}P1K/*.json ./"
            env_cmd = f"export HCCL_TOPO_FILE_PATH=/mnt/mc2_test/bin_path/{self.case_name}/topo.json"
            hccl_mm_sh = self.case_name + '_hccl_mm'  # 仅生成golden场景
            mc2_sh = self.case_name + '_mc2'
            # 移除已有shell文件
            mv_command = f"rm -rf bin_path/{self.case_name}/{self.case_name}*.sh"
            ret = subprocess.run(mv_command, shell=True)
            # golden生成命令
            ret = subprocess.check_output(f"echo {ranktable_cmd} >> {self.bin_path}/{hccl_mm_sh}.sh", shell=True)
            ret = subprocess.check_output(f"echo {env_cmd} >> {self.bin_path}/{hccl_mm_sh}.sh", shell=True)
            # 拼二进制执行命令
            for i in range(self.world_size):
                aclnn_command_golden = self.get_command(exe_binary_golden, i)
                print(f"[INFO] rank:{i}, aclnn command:{aclnn_command_golden}")
                # 命令写入shell
                ret = subprocess.check_output(f"echo {aclnn_command_golden} >> {self.bin_path}/{hccl_mm_sh}.sh", shell=True)
            if ret:
                print(f"Generate shell command excute result: {ret}")
            # mc2 结果生成命令
            ret = subprocess.check_output(f"echo {ranktable_cmd} >> {self.bin_path}/{mc2_sh}.sh", shell=True)
            ret = subprocess.check_output(f"echo {env_cmd} >> {self.bin_path}/{mc2_sh}.sh", shell=True)
            # 拼二进制执行命令
            for i in range(self.world_size):
                aclnn_command = self.get_command(exe_binary, i)
                print(f"[INFO] rank:{i}, aclnn command:{aclnn_command}")
                # 命令写入shell
                ret = subprocess.check_output(f"echo {aclnn_command} >> {self.bin_path}/{mc2_sh}.sh", shell=True)
            if ret:
                print(f"Generate shell command excute result: {ret}")
            sh_command = f"bash bin_path/{self.case_name}/{self.case_name}.sh"
            print(f"[INFO] sh command:{sh_command}")
