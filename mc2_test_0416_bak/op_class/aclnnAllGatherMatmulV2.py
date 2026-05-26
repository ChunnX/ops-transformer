from .common import *
import torch
import hashlib
from script import toolss
from .registry import class_registry
from ml_dtypes import float8_e4m3fn, float8_e5m2
from script import generate_mx_data
from script.generate_float_data import new_trans_np_fp4_e1m2_tensor_to_bfloat16, new_trans_np_fp4_e2m1_tensor_to_bfloat16
from .aclnnAllGatherMatmul import AllGatherMatmul


class AllGatherMatmulV2GraphModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, hcomm_info, world_size, bias, x1scale, x2scale, is_gather_output,
                out_dtype, x1_dtype, x2_dtype, x1scale_dtype, x2scale_dtype, block_size, group_sizes,
                tensor_allgather, tensor_allgather_x1scale, is_mxfp, is_quant, is_bias, is_trans_b,hcom_type, per_block_flag):
        if is_trans_b:
            x2 = x2.t()
            if is_mxfp:
                x2scale = torch.transpose(x2scale, 0, 1)
            if per_block_flag:
                x2scale = x2scale.t()
        if hcom_type == "mc2":
            output, gather_output, _ = torch_npu.npu_all_gather_quant_mm(
                x1,
                x2,
                hcomm_info,
                world_size,
                bias=bias,
                x1_scale=x1scale,
                x2_scale=x2scale,
                quant_scale=None,
                block_size=block_size,
                gather_index=0,
                gather_output=is_gather_output,
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
            dist._all_gather_base(tensor_allgather, x1)
            if is_mxfp or per_block_flag:
                dist._all_gather_base(tensor_allgather_x1scale, x1scale)
                x1scale = tensor_allgather_x1scale
            if not is_quant:
                output = torch.matmul(tensor_allgather, x2)
                if is_bias:
                    output += bias
            else:
                output = torch_npu.npu_quant_matmul(
                    x1=tensor_allgather,
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
            gather_output = tensor_allgather if is_gather_output else None
        return output, gather_output


@class_registry("aclnnAllGatherMatmulV2")
class AllGatherMatmulV2(AllGatherMatmul):
    def __init__(self, case, rank, dist=None, hcomm_info=None):
        super().__init__(case, rank, dist, hcomm_info)
        self.per_block_flag = case['per_block_flag']
        self.block_size = case['block_size']
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
        self.is_mxFp = True if self.x1scale_dtype == "fp8_e8m0" else False
        self.x1scale = None
        self.x2scale = None
        self.bias = None
        self.tensor_allgather_x1scale = None

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
            if self.x1_dtype in UNSUPPORTED_DTYPES:
                self.tensor_allgather = torch.zeros(
                    self.gather_out_shape,
                    dtype=torch.float32
                ).to(TORCH_DTYPE_MAP[self.x1_dtype]).to(self.device)
            else:
                self.tensor_allgather = torch.zeros(
                    self.gather_out_shape,
                    dtype=TORCH_DTYPE_MAP[self.x1_dtype]
                ).to(self.device)
            if self.is_mxFp or self.per_block_flag:
                allgather_x1scale_shape = [self.x1scale_shape[0] * self.world_size, self.x1scale_shape[1], 2] if self.is_mxFp else [self.x1scale_shape[0] * self.world_size, self.x1scale_shape[1]]
                if self.x1scale_dtype in UNSUPPORTED_DTYPES:
                    self.tensor_allgather_x1scale = torch.zeros(
                        allgather_x1scale_shape, dtype=torch.float32
                    ).to(TORCH_DTYPE_MAP[self.x1scale_dtype]).to(self.device)
                else:
                    self.tensor_allgather_x1scale = torch.zeros(
                        allgather_x1scale_shape,
                        dtype=TORCH_DTYPE_MAP[self.x1scale_dtype]
                    ).to(self.device)
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
        dist._all_gather_base(self.tensor_allgather, self.x1)
        if self.is_mxFp or self.per_block_flag:
            dist._all_gather_base(self.tensor_allgather_x1scale, self.x1scale)
            x1scale = self.tensor_allgather_x1scale
        else:
            x1scale = self.x1scale
        if not self.is_quant:
            self.output_golden = torch.matmul(self.tensor_allgather, self.x2)
            if self.is_bias:
                self.output_golden += self.bias
        else:
            self.output_golden = torch_npu.npu_quant_matmul(
                x1=self.tensor_allgather,
                x2=self.x2,
                pertoken_scale=x1scale,
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
            #self.save(self.output_golden, self.output_path, self.output_hccl_name)
        self.gather_output_golden = self.tensor_allgather
        if self.scene == "aclnn_golden":
            self.save(self.output_golden, self.output_path, self.output_hccl_name)
            if self.is_gather_output:
                self.save(self.gather_output_golden, self.output_path, self.gather_output_hccl_name)

    def get_cpu(self):
        x1 = self.x1.to(torch.float32)
        all_gather_list = [torch.zeros_like(x1) for _ in range(self.world_size)]
        dist.all_gather(all_gather_list, x1)
        all_gather_out = torch.cat(all_gather_list, dim=0)
        if self.is_gather_output:
            self.save(all_gather_out.numpy(), self.output_path, self.gather_output_cpu_name)
        x2 = self.x2.to(torch.float32)
        bias = self.bias.to(torch.float32) if self.is_bias else None
        if not self.is_quant:  # 情形1:非量化
            output = torch.matmul(all_gather_out, x2)
            if self.is_bias:
                output += bias
        else:
            x1scale = self.x1scale.to(torch.float32)
            x2scale = self.x2scale.to(torch.float32)
            if self.per_block_flag:  # 情形4：per_block量化
                all_gather_scale_list = [torch.zeros_like(x1scale) for _ in range(self.world_size)]
                dist.all_gather(all_gather_scale_list, x1scale)
                all_gather_scale_out = torch.cat(all_gather_scale_list, dim=0)
                output = func.per_block_cpu_compute(self.group_size, all_gather_out, x2, all_gather_scale_out, x2scale)
                #output = func.per_block_cpu_compute(self.group_size, all_gather_out, x2, x1scale, x2scale)
            elif self.is_mxFp:  # 情形5:mxfp量化
                all_gather_scale_list = [torch.zeros_like(x1scale) for _ in range(self.world_size)]
                dist.all_gather(all_gather_scale_list, x1scale)
                all_gather_scale_out = torch.cat(all_gather_scale_list, dim=0)

                all_gather_out = all_gather_out.numpy()
                x2 = x2.numpy()
                all_gather_scale_out = all_gather_scale_out.numpy()
                x2scale = x2scale.numpy()

                output = self.mxfp_cpu_compute(all_gather_out, x2, all_gather_scale_out, x2scale)
                if self.is_bias:
                    output += bias
            else:  # 情形2/3 per_tensor
                output = torch.matmul(all_gather_out, x2)
                if self.is_bias:
                    output += bias
                double_scale = func.scale_generate(x1scale.numpy() * x2scale.numpy())
                double_scale_tensor = torch.unsqueeze(torch.from_numpy(double_scale), dim=1).to(torch.float32)
                output *= double_scale_tensor
        self.save(output, self.output_path, self.output_cpu_name)

    def get_npu(self):
        self.output_npu, self.gather_output_npu, _ = torch_npu.npu_all_gather_quant_mm(
            self.x1,
            self.x2,
            self.hcomm_info,
            self.world_size,
            bias=self.bias,
            x1_scale=self.x1scale,
            x2_scale=self.x2scale,
            quant_scale=None,
            block_size=self.block_size,
            gather_index=0,
            gather_output=self.is_gather_output,
            comm_turn=0,
            group_sizes=self.group_sizes,
            amax_output=False,
            y_dtype=OUT_DTYPE_ENUM[self.output_dtype],
            x1_dtype=DTYPE_ENUM[self.x1_dtype],
            x2_dtype=DTYPE_ENUM[self.x2_dtype],
            x1_scale_dtype=DTYPE_ENUM[self.x1scale_dtype],
            x2_scale_dtype=DTYPE_ENUM[self.x2scale_dtype],
        )
        #self.save(self.output_npu, self.output_path, self.output_npu_name)
        return self.output_npu

    def get_graph(self, hcom_type="mc2"):
        if hcom_type not in ["mc2", "hccl_mm"]:
            print(f"[ERROR] unsupported hcom_type {hcom_type}")
            sys.exit(1)
        model = AllGatherMatmulV2GraphModel()
        model = self.define_model(model, self.graph_type)
        out_graph = model(
            self.x1,
            self.x2,
            self.hcomm_info,
            self.world_size,
            self.bias,
            self.x1scale,
            self.x2scale,
            self.is_gather_output,
            self.output_dtype,
            self.x1_dtype,
            self.x2_dtype,
            self.x1scale_dtype,
            self.x2scale_dtype,
            self.block_size,
            self.group_sizes,
            self.tensor_allgather,
            self.tensor_allgather_x1scale,
            self.is_mxFp,
            self.is_quant,
            self.is_bias,
            self.is_trans_b,
            hcom_type,
            self.per_block_flag
        )
        if hcom_type == "mc2":
            self.output_npu, self.gather_output_npu = out_graph
            return self.output_npu
        else:
            self.output_golden, self.gather_output_golden = out_graph

    def compare(self, rank=0, check_md5=0):
        op = self.case['op_name']
        cpu_input_path = os.path.join(self.output_path, self.output_cpu_name)
        if self.is_gather_output:
            gather_cpu_input_path = os.path.join(self.output_path, self.gather_output_cpu_name)
        if self.case['scene'] in ["compare", "aclnn"]:  # aclnn场景，golden和npu从已保存文件中读取
            output_npu_path = os.path.join(self.output_path, self.output_npu_name)
            output_golden_path = os.path.join(self.output_path, self.output_hccl_name)
            np_dtype = NUMPY_DTYPE_MAP[self.output_dtype]
            # bf16场景 单独处理
            if self.output_dtype == 'bf16':
                self.output_npu = torch.tensor(np.fromfile(output_npu_path, np_dtype).astype(np.float32)).to(
                    torch.bfloat16)
                self.output_golden = torch.tensor(np.fromfile(output_golden_path, np_dtype).astype(np.float32)).to(
                    torch.bfloat16)
            else:
                self.output_npu = torch.tensor(np.fromfile(output_npu_path, np_dtype))
                self.output_golden = torch.tensor(np.fromfile(output_golden_path, np_dtype))
            if self.is_gather_output:
                gather_output_npu_path = os.path.join(self.output_path, self.gather_output_npu_name)
                gather_output_golden_path = os.path.join(self.output_path, self.gather_output_hccl_name)
                if self.x1_dtype == 'bf16':
                    self.gather_output_npu = torch.tensor(
                        np.fromfile(gather_output_npu_path, bfloat16).astype(np.float32)).to(torch.bfloat16)
                    self.gather_output_golden = torch.tensor(
                        np.fromfile(gather_output_golden_path, bfloat16).astype(np.float32)).to(torch.bfloat16)
                elif self.x1_dtype == 'fp16':
                    self.gather_output_npu = torch.tensor(np.fromfile(gather_output_npu_path, np.float16))
                    self.gather_output_golden = torch.tensor(np.fromfile(gather_output_golden_path, np.float16))
                elif self.x1_dtype == "hif8":
                    self.gather_output_npu = torch.tensor(
                        np.fromfile(gather_output_npu_path, np.uint8).astype(np.float32)).to(torch.float32)
                    self.gather_output_golden = torch.tensor(
                        np.fromfile(gather_output_golden_path, np.uint8).astype(np.float32)).to(torch.float32)
                else:
                    self.gather_output_npu = torch.tensor(
                        np.fromfile(gather_output_npu_path, self.x1_dtype).astype(np.float32)).to(torch.float32)
                    self.gather_output_golden = torch.tensor(
                        np.fromfile(gather_output_golden_path, self.x1_dtype).astype(np.float32)).to(torch.float32)
        else:
            output_npu = self.output_npu.cpu().view(-1)
            output_golden = self.output_golden.cpu().view(-1)
            if self.is_gather_output:
                if self.x1_dtype in ["fp16", "bf16"]:
                    gather_output_npu = self.gather_output_npu.cpu().view(-1)
                    gather_output_golden = self.gather_output_golden.cpu().view(-1)
                else:
                    gather_output_npu = self.gather_output_npu.to(torch.float32).cpu().view(-1)
                    gather_output_golden = self.gather_output_golden.to(torch.float32).cpu().view(-1)
        output_cpu = torch.from_numpy(np.fromfile(cpu_input_path, np.float32))
        if self.is_gather_output:
            gather_output_cpu = torch.from_numpy(np.fromfile(gather_cpu_input_path, np.float32))
        rst_npu = checkResult(output_npu, output_cpu, "{}_dq_npu".format(self.bin_name))
        rst_npu.print_result()
        rst_gpu = checkResult(output_golden, output_cpu, "{}_dq_gpu".format(self.bin_name))
        rst_gpu.print_result()
        str1, str2 = rst_npu.check_result(rst_gpu)
        if 'error' in str1:
            res = 'FAIL'
        else:
            res = 'PASS'
        if self.is_gather_output:
            rst_gather_npu = checkResult(gather_output_npu, gather_output_cpu, "{}_dq_npu".format(self.bin_name))
            rst_gather_npu.print_result()
            rst_gather_gpu = checkResult(gather_output_golden, gather_output_cpu, "{}_dq_gpu".format(self.bin_name))
            rst_gather_gpu.print_result()
            str3, str4 = rst_gather_npu.check_result(rst_gather_gpu)
            if 'error' in str3:
                res = 'FAIL'
        compare_rank = rank if self.case['scene'] in ["compare", "aclnn"] else self.rank
        print('[INFO] device_{} 精度结果为：{}'.format(compare_rank, res))
        print('[INFO] device_{} 精度计算结束：{} '.format(compare_rank, time.strftime('%H:%M:%S', time.localtime())))
        with open('{}_result.csv'.format(op), 'a') as f:
            f.write(self.case_name + "," + str(compare_rank) + ',' + str(res) + '\n')
        return res

    def get_command(self, binary, rankid):
        command = [f"nohup /mnt/mc2_test/bin_path/{binary}"]
        # command.append(f"--is_quant {self.is_quant}")
        command.append(f"--x1_shape {','.join(map(str, self.x1_shape))}")
        command.append(f"--x2_shape {','.join(map(str, self.x2_shape))}")
        command.append(f"--bias_shape {self.x2_shape[-1]}")
        command.append(f"--quant_flag {self.is_quant}")
        if self.is_quant:
            command.append("--is_hccl_mm  1")
        if self.is_mxFp:
            command.append("--is_mxFp  1")
            command.append("--gatherx1scale_shape {','.join(map(str, self.case['scaleInv1_shape']))}")
        command.append(f"--transpose_b_flag {self.is_trans_b}")
        command.append(f"--bias_flag {self.is_bias}")
        command.append(f"--input_tensor_dtype {','.join(map(str, self.case['input_tensor_dtype']))}")
        command.append(f"--input_tensor_format {','.join(map(str, self.case_format))}")
        command.append(f"--output_shape {','.join(map(str, self.output_shape))}")
        command.append(f"--block_size {str(self.block_size)}")
        command.append(f"--per_block_flag {str(self.per_block_flag).lower()}")
        command.append(f"--scale1_shape {','.join(map(str, self.case['scaleInv1_shape']))}")
        command.append(f"--scale2_shape {','.join(map(str, self.case['scaleInv2_shape']))}")
        command.append(f"--output_dtype {self.output_dtype}")
        command.append(f"--output_format {self.output_format}")
        command.append(f"--n_dev {self.world_size}")
        command.append(f"--bin_path {self.bin_path_tmp}")
        command.append(f"--loop_cnt {self.loop}")
        command.append(f"--gatherout_shape {','.join(map(str, self.gaout_shape))}")
        command.append(f"--rankId {rankid} \&")
        aclnn_command = " ".join(command)
        return aclnn_command

    def process_aclnn(self):
        self.root_path = os.path.abspath(os.getcwd())
        self.case_format = ['ND', 'ND', 'ND']
        self.gaout_shape = self.input_shape
        input_shape = [dd // self.world_size for i, dd in enumerate(self.input_shape) if i == 0] + self.input_shape[1:]
        self.input_shape = input_shape
        self.output_shape = [self.gaout_shape[0], self.weight_shape[1]]
        print(self.gaout_shape, self.input_shape, self.output_shape)
        self.output_format = self.case_format[0]
        self.bin_path = self.root_path + "/bin_path/" + self.case_name
        self.bin_path_tmp = "/mnt/mc2_test/bin_path/" + self.case_name
        print(f"bin_path: {self.bin_path}")
        result = subprocess.check_output(["uname", "-a"]).decode("utf-8")
        arch = "aarch" if "aarch64" in result.lower() else "x86"
        exe_binary = self.op_name + "_" + arch
        if self.case['scene'] == 'aclnn' and self.case['soc'] == '910D':
            ranktable_cmd = f"cp -f /mnt/mc2_test/bin_path/{self.world_size}P1K/*.json ./"
            mc2_sh = self.case_name + '_mc2'
            # 移除已有shell文件
            mv_command = f"rm -rf bin_path/{self.case_name}/{self.case_name}*.sh"
            ret = subprocess.run(mv_command, shell=True)
            # mc2 结果生成命令
            ret = subprocess.check_output(f"echo {ranktable_cmd} >> {self.bin_path}/{mc2_sh}.sh", shell=True)
            env_cmd = f"export HCCL_TOPO_FILE_PATH=/mnt/mc2_test/bin_path/{self.case_name}/topo.json"
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
