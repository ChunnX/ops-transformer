from .common import *
import torch
import torch_npu
import hashlib
from script import toolss
from ml_dtypes import bfloat16
from .registry import class_registry


ALLGATHER_QBMM_MAP = {
    0: "output",
    1: "gather_out",
    2: "gather_pertoken_out",
}


class ALLGATHER_QUANT_BMM_GRAPH_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, weight, dequant_scale, hcomm_info, world_size, pertoken_scale, bias, output_dtype,
                gather_axis, gather_output, tensor_allgather, input_golden, tensor_allgather_pertoken,
                pertoken_scale_golden, is_pertoken, is_nz, is_trans, type):
        if is_nz:
            weight = torch_npu.npu_format_cast(weight, 29)
        if is_trans:
            weight = weight.t()
        if type == "mc2":
            output, gather_x1_output, gather_pertoken_output = torch_npu.npu_all_gather_quant_bmm(input, weight,
                                                                                                  dequant_scale,
                                                                                                  hcomm_info,
                                                                                                  world_size,
                                                                                                  pertoken_scale=pertoken_scale,
                                                                                                  bias=bias,
                                                                                                  output_dtype=output_dtype,
                                                                                                  gather_axis=gather_axis,
                                                                                                  gather_output=gather_output,
                                                                                                  comm_turn=0)
        elif type == "hccl_mm":
            dist._all_gather_base(tensor_allgather, input_golden)
            if is_pertoken:
                dist._all_gather_base(tensor_allgather_pertoken, pertoken_scale_golden)

            output = torch_npu.npu_quant_matmul(x1=tensor_allgather, x2=weight,
                                                scale=dequant_scale, bias=bias,
                                                pertoken_scale=tensor_allgather_pertoken,
                                                output_dtype=output_dtype)
            gather_x1_output = tensor_allgather if gather_output else None
            gather_pertoken_output = tensor_allgather_pertoken if (gather_x1_output and is_pertoken) else None
        else:
            print("In allgather get_graph,wrong type.")
        return output, gather_x1_output, gather_pertoken_output


@class_registry("aclnnAllGatherQuantBatchMatMul")
class ALLGATHER_QUANT_BMM(MC2_OP):
    def __init__(self, case, rank, dist=None, hcomm_info=None):
        super().__init__(case, rank, dist, hcomm_info)
        # allgather涉及
        self.gather_output = case['gather_output'] if 'gather_output' in case else False
        self.is_bias = case['is_bias'] if 'is_bias' in case else 0
        self.is_dequant_scale = case['is_dequant_scale'] if 'is_dequant_scale' in case else 0
        self.is_nz = case['is_nz'] if 'is_nz' in case else 0
        self.is_pertoken = case['is_pertoken_scale'] if 'is_pertoken_scale' in case else 0
        self.gather_axis = case['gather_axis'] if 'gather_axis' in case else 0
        self.output_dtype = eval(case['output_dtype']) if 'output_dtype' in case else torch.float16
        self.pertoken_shape = case['pertoken_scale_shape'] if 'pertoken_scale_shape' in case else None
        self.is_trans = case['is_trans'] if 'is_trans'  in case else 0

    # B*S*N*datasize+16 ps:datasize=sizeof(output_dtype),fp16/bf16=2
    def get_tmp_tensor_memory(self):
        final_memory = self.input_shape[0] * self.input_shape[1] * 2
        if self.is_bias:
            final_memory = final_memory *self.bias.shape[0]
        return final_memory

    def get_tmp_tensor(self):
        self.bias = func.get_tensor_bais(self.case, self.rank) if self.is_bias else None

        # ====================== 处理转置场景weight =======================
        self.weight_graph = self.weight
        if self.is_trans:
            self.weight = self.weight.t()

        # ====================== 处理input_golden =======================
        # quant_bmm的input reshape成[1,B*S,K],因交付算子内部实际将[B,S,K]-->[1,B*S,K]送入quant_bmm
        tmp_input = self.input
        self.input_golden = tmp_input.reshape(1, -1, self.input_shape[2])
        tensor_allgather_shape = [self.input_golden.shape[0], self.input_golden.shape[1] * self.world_size,
                                  self.input_golden.shape[2]]
        tmp_tensor_allgather = torch.zeros(tensor_allgather_shape, dtype=self.input_dtype)
        self.tensor_allgather = tmp_tensor_allgather if "cpu" in self.scene else tmp_tensor_allgather.npu()

        # ====================== 处理dequant_scale =======================
        self.dequant_scale = func.get_tensor_bais(self.case, self.rank, name='dequant_scale')
        self.dequant_scale_cpu = self.dequant_scale.to(torch.float32)
        # 非cpu场景, 输出=fp16, pertoken=None  ------> scale 需为int64/uint64
        if not self.is_pertoken and self.output_dtype in [torch.float16, None] and "cpu" not in self.case["scene"]:
            print("scale : fp32 ------> int64/uint64")
            self.dequant_scale = torch_npu.npu_trans_quant_param(self.dequant_scale.npu())

        # ====================== 处理pertoken_scale =======================
        self.pertoken_scale = func.get_tensor_bais(self.case, self.rank,
                                                   name='pertoken_scale') if self.is_pertoken else None
        self.tensor_allgather_pertoken = None
        self.pertoken_scale_golden = None
        if self.is_pertoken:
            # quant_bmm的pertoken仅支持1维
            tmp_pertoken = self.pertoken_scale
            self.pertoken_scale_golden = tmp_pertoken.reshape(-1)
            # 预先创建allgather_pertoken的接收tensor
            allgather_pertoken_shape = [self.pertoken_scale_golden.shape[0] * self.world_size]
            tmp_allgather_pertoken = torch.zeros(allgather_pertoken_shape, dtype=torch.float32)
            self.tensor_allgather_pertoken = tmp_allgather_pertoken if "cpu" in self.scene else tmp_allgather_pertoken.npu()

    # allgather + quant_bmm
    def get_hccl_mm(self):
        dist._all_gather_base(self.tensor_allgather, self.input_golden)
        if self.is_pertoken:
            dist._all_gather_base(self.tensor_allgather_pertoken, self.pertoken_scale_golden)

        self.output_golden = torch_npu.npu_quant_matmul(x1=self.tensor_allgather, x2=self.weight,
                                                        scale=self.dequant_scale, bias=self.bias,
                                                        pertoken_scale=self.tensor_allgather_pertoken,
                                                        output_dtype=self.output_dtype)
        self.gather_output_golden = self.tensor_allgather if self.gather_output else None
        self.gather_pertoken_output_golden = self.tensor_allgather_pertoken if (
                self.gather_output and self.is_pertoken) else None
        return self.output_golden, self.gather_output_golden, self.gather_pertoken_output_golden

    def get_npu(self):
        out = torch_npu.npu_all_gather_quant_bmm(self.input, self.weight, self.dequant_scale, self.hcomm_info,
                                                 self.world_size, pertoken_scale=self.pertoken_scale, bias=self.bias,
                                                 output_dtype=self.output_dtype, gather_axis=self.gather_axis,
                                                 gather_output=self.gather_output, comm_turn=0)
        self.output_npu = out[0]
        self.gather_output_npu = out[1] if self.gather_output else None
        self.gather_pertoken_output_npu = out[2] if (self.gather_output and self.is_pertoken) else None

        return self.output_npu

    def get_graph(self, type="mc2"):
        model = ALLGATHER_QUANT_BMM_GRAPH_Model()
        model = self.define_model(model, self.graph_type)
        out_graph = model(self.input, self.weight_graph, self.dequant_scale, self.hcomm_info, self.world_size,
                          self.pertoken_scale, self.bias, self.output_dtype, self.gather_axis, self.gather_output,
                          self.tensor_allgather, self.input_golden, self.tensor_allgather_pertoken,
                          self.pertoken_scale_golden, self.is_pertoken, self.is_nz, self.is_trans, type)

        self.gather_output_npu = None
        self.gather_pertoken_output_npu = None
        self.gather_output_golden = None
        self.gather_pertoken_output_golden = None
        if type == "mc2":
            self.output_npu = out_graph[0]
            if self.gather_output:
                self.gather_output_npu = out_graph[1]
                if self.is_pertoken:
                    self.gather_pertoken_output_npu = out_graph[2]
            return self.output_npu
        elif type == "hccl_mm":
            self.output_golden = out_graph[0]
            if self.gather_output:
                self.gather_output_golden = out_graph[1]
                if self.is_pertoken:
                    self.gather_pertoken_output_golden = out_graph[2]
        else:
            print("In allgather_quant_bmm get_graph,wrong type..")

    # y = dequantScale∗pertokenScaleOptional*( allgather(x1)@x2+biasoptional)
    # gather_out = allgather(x1)
    # gather_pertoken_scale_output = allgather(pertoken)
    def get_cpu(self):
        print('[INFO] device_{} 构造output_cpu数据'.format(self.rank))
        # step1: 执行allgather
        all_gather_list = [torch.zeros_like(self.input) for _ in range(self.world_size)]
        dist.all_gather(all_gather_list, self.input)
        all_gather_out = torch.cat(all_gather_list, dim=self.gather_axis)
        print(f"cpu gather x1 shape:{all_gather_out.shape}")

        # 保存all_gather_output输出
        if self.gather_output:
            print("Gather_output:TRUE, 保存gather x1输出")
            func.save_file(self.save_path, all_gather_out, torch.int8, "gather_out", self.bin_name, self.rank)

        # step2：x1/x2升精度后做matmul
        input = all_gather_out.to(torch.int32)
        weight = self.weight.to(torch.int32)
        output = torch.matmul(input, weight)  # torch.int32
        # step3: 根据bias/pertoken是否需要执行相应运算
        if self.is_bias:
            print("BIAS: TRUE")
            bias = self.bias if self.bias.dtype == torch.int32 else self.bias.to(torch.float32)
            output = torch.add(output, bias)
        if self.is_pertoken:
            print("PER-TOKEN: TRUE")
            # pertoken=1维,gather_axis=1时会索引越界
            pertoken_gather_axis = 0 if (
                    self.gather_axis == 1 and len(self.pertoken_scale.shape) == 1) else self.gather_axis
            pertoken_all_gather_list = [torch.zeros_like(self.pertoken_scale) for _ in range(self.world_size)]
            dist.all_gather(pertoken_all_gather_list, self.pertoken_scale)
            pertoken_all_gather_out = torch.cat(pertoken_all_gather_list, dim=pertoken_gather_axis)
            print(f"cpu gather pertoken shape:{pertoken_all_gather_out.shape}")

            # 保存pertoken_all_gather输出
            if self.gather_output and self.is_pertoken:
                print("保存gather pertoken输出")
                func.save_file(self.save_path, pertoken_all_gather_out, torch.float32, "gather_pertoken_output",
                               self.bin_name, self.rank)

            output_origial_shape = output.shape
            if len(output_origial_shape) == 3:
                output = output.reshape(-1, output.shape[-1])
            pertoken_scale = pertoken_all_gather_out.reshape(-1)
            pertoekn_scale = torch.unsqueeze(pertoken_scale, dim=1)
            output = (output * pertoekn_scale).reshape(output_origial_shape)

        cpu_output = torch.mul(output, self.dequant_scale_cpu)
        func.save_file(self.save_path, cpu_output, 'bf16', "cpu_output", self.bin_name, self.rank)

    def compare(self, rank=0, check_md5=0):
        if "aclnn" in self.case["scene"]:  # aclnn场景，golden和npu从已保存文件中读取
            output_npu_path = f"{self.save_path}/output/{rank}/output_npu.bin"
            output_golden_path = f"{self.save_path}/output/{rank}/output_hccl.bin"
            if self.output_dtype == 'fp16':
                np_dtype = np.float16
            else:
                np_dtype = bfloat16
            self.output_npu = torch.tensor(np.fromfile(output_npu_path, np_dtype).astype(np.float32))
            self.output_golden = torch.tensor(np.fromfile(output_golden_path, np_dtype).astype(np.float32))
            self.gather_output_npu = None
            self.gather_pertoken_output_npu = None
            self.gather_output_golden = None
            self.gather_pertoken_output_golden = None
            if self.gather_output:
                gatherout_npu_path = f"{self.save_path}/output/{rank}/gatherout_npu.bin"
                gatherout_golden_path = f"{self.save_path}/output/{rank}/gatherout_hccl.bin"
                self.gather_output_npu = torch.tensor(np.fromfile(gatherout_npu_path, np_dtype).astype(np.float32))
                self.gather_output_golden = torch.tensor(
                    np.fromfile(gatherout_golden_path, np_dtype).astype(np.float32))
                if self.is_pertoken:
                    gather_pertoken_npu_path = f"{self.save_path}/output/{rank}/gather_pertokenout_npu.bin"
                    gather_pertoken_golden_path = f"{self.save_path}/output/{rank}/gather_pertokenout_hccl.bin"
                    self.gather_pertoken_npu = torch.tensor(np.fromfile(gather_pertoken_npu_path, np.float32))
                    self.gather_pertoken_golden = torch.tensor(np.fromfile(gather_pertoken_golden_path, np.float32))
        output_npu = (self.output_npu, self.gather_output_npu, self.gather_pertoken_output_npu)
        for index, value in enumerate(output_npu):
            if value is not None:
                compare_rank = rank if "aclnn" in self.case['scene'] else self.rank
                cpu_input_path = os.path.join('bin_path', self.case_name,
                                              '{}_cpu_{}_{}.bin'.format(self.bin_name, ALLGATHER_QBMM_MAP[index],
                                                                        compare_rank))
                output_cpu = torch.tensor(np.fromfile(cpu_input_path, np.float32))
                if index == 0:
                    output_npu = self.output_npu.cpu().view(-1)
                    output_golden = self.output_golden.cpu().view(-1)
                    # 迭代一不交付gather_output=true,暂不比较gatherout & gather_pertoken_out---2024/11/21
                    # elif index == 1:
                    #     output_npu = self.gather_output_npu.cpu().view(-1)
                    #     output_golden = self.gather_output_golden.cpu().view(-1)
                    # elif index == 2:
                    #     output_npu = self.bmm_output_npu.cpu().view(-1)
                    #     output_golden = self.bmm_out_golden.cpu().view(-1)
                    rst_npu = checkResult(output_npu, output_cpu, "{}_dq_npu".format(self.bin_name))
                    rst_npu.print_result()
                    rst_gpu = checkResult(output_golden, output_cpu, "{}_dq_gpu".format(self.bin_name))
                    rst_gpu.print_result()
                    str1, str2 = rst_npu.check_result(rst_gpu)
                    if 'error' in str1:
                        res = 'FAIL'
                    else:
                        res = 'PASS'
                    print('[INFO] device_{} OUT {} 精度结果为：{}'.format(compare_rank, ALLGATHER_QBMM_MAP[index], res))
                    print('[INFO] device_{} OUT {} 精度计算结束：{} '.format(compare_rank, ALLGATHER_QBMM_MAP[index],
                                                                      time.strftime('%H:%M:%S', time.localtime())))
                    with open('allgather_quant_bmm_result.csv', 'a') as f:
                        f.write(self.case_name + "," + str(compare_rank) + ',' + str(res) + ',' + str(
                            ALLGATHER_QBMM_MAP[index]) + '\n')
        return res

    def process_aclnn(self):
        print("-----------in allgather_quant_bmm aclnn")
        self.root_path = os.path.abspath(os.getcwd())
        self.case_format = ['ND', 'ND', 'ND'] if self.is_nz == 0 else ['ND', 'NZ', 'ND']
        self.is_trans = 1 if self.case['is_trans'] else 0
        x2_shape = [self.weight_shape[1],self.weight_shape[0]] if self.is_trans else self.weight_shape
        self.output_shape = (self.input_shape[:-1] + self.weight_shape[:1])
        self.output_dtype = 'fp16' if self.case['output_dtype'] == "torch.float16" else 'bf16'
        self.output_format = 'ND'
        self.bin_path = self.root_path + "/bin_path/" + self.case_name
        print(f"bin_path: {self.bin_path}")
        self.loop = 1
        # 根据架构aarch/x86获取待执行二进制名称
        result = subprocess.check_output(["uname", "-a"]).decode("utf-8")
        arch = "aarch" if "aarch64" in result.lower() else "x86"
        exe_binary = self.op_name + "_" + arch
        # 获取初始的input_tensor_dtype/ input_tensor_format
        input_tensor_dtype = ','.join(map(str, self.case['input_tensor_dtype'])) + ',' + self.case[
            'dequant_scale_dtype']
        input_tensor_format = ','.join(map(str, self.case_format)) + ',ND'
        # 根据pertoken可选增加input_tensor_dtype/format
        if self.is_pertoken:
            input_tensor_dtype = input_tensor_dtype + ',' + self.case['pertoken_scale_dtype']
            input_tensor_format += ',ND'

        gather_pertoken_shape = [self.pertoken_shape[0] * self.world_size]
        if self.gather_axis == 0:
            self.output_shape = [self.input_shape[0] * self.world_size, self.input_shape[1], self.weight_shape[1]]
            gather_output_shape = [self.input_shape[0] * self.world_size, self.input_shape[1], self.input_shape[2]]
            if len(self.pertoken_shape) == 2:
                gather_pertoken_shape = [self.pertoken_shape[0] * self.world_size, self.pertoken_shape[1]]
        else:
            self.output_shape = [self.input_shape[0], self.input_shape[1] * self.world_size, self.weight_shape[1]]
            gather_output_shape = [self.input_shape[0], self.input_shape[1] * self.world_size, self.input_shape[2]]
            if len(self.pertoken_shape) == 2:
                gather_pertoken_shape = [self.pertoken_shape[0], self.pertoken_shape[1] * self.world_size]

        gather_out_flag = 1 if self.gather_output else 0
        # 拼二进制执行命令
        command = [f"{self.root_path}/cpp/bin/{exe_binary}"]
        command.append(f"--x1_shape {','.join(map(str, self.input_shape))}")
        command.append(f"--x2_shape {','.join(map(str, x2_shape))}")
        command.append(f"--transpose_b_flag {self.is_trans}")
        command.append(f"--input_tensor_dtype {input_tensor_dtype}")
        command.append(f"--input_tensor_format {input_tensor_format}")
        command.append(f"--output_shape {','.join(map(str, self.output_shape))}")
        command.append(f"--output_dtype {self.output_dtype}")
        command.append(f"--output_format {self.output_format}")
        command.append(f"--n_dev {self.world_size}")
        command.append(f"--bin_path {self.bin_path}")
        command.append(f"--loop_cnt {self.loop}")
        command.append(f"--bias_shape {self.output_shape[-1]}")
        command.append(f"--bias_flag {self.is_bias}")
        command.append(f"--dequant_scale_shape {','.join(map(str, self.case['dequant_scale_shape']))}")
        command.append(f"--gatherout_shape {','.join(map(str, gather_output_shape))}")
        command.append(f"--gather_out_flag {gather_out_flag}")
        command.append(f"--gather_axis {self.gather_axis}")
        if self.is_pertoken:
            command.append(f"--pertoken_flag {self.is_pertoken}")
            command.append(f"--pertoken_shape {','.join(map(str, self.case['pertoken_scale_shape']))}")
            if self.gather_output:
                command.append(f"--gather_perout_shape {','.join(map(str, gather_pertoken_shape))}")
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
            print(f"Aclnn command excute result: {ret}")
