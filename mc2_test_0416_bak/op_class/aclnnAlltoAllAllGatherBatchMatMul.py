from .common import *
import torch
from mindspeed.ops.npu_all_to_all_all_gather_bmm import npu_alltoall_allgather_bmm
from script import toolss
from ml_dtypes import bfloat16
from .registry import class_registry


def activate(act_type, input, scene):
    output = input
    if not act_type:
        output = input
    elif act_type.upper() == 'GeLu'.upper():
        gelu = torch.nn.GELU(approximate="tanh")
        output = gelu(input)
    elif act_type.upper() == 'SiLu'.upper():
        if scene == 'cpu':
            output = input / (1 + torch.exp(-input))
        else:
            output = torch_npu.npu_silu(input)
    elif act_type.upper() == 'ReLu'.upper():
        relu = torch.nn.ReLU()
        output = relu(input)
    elif act_type.upper() == 'FastGeLu'.upper():
        if scene == 'cpu':
            output = input / (1 + torch.exp(-1.702 * input))
        else:
            output = torch_npu.npu_fast_gelu(input)
    else:
        print(
            f"in alltoall_allgather_bmm, get wrong act_type {act_type}!!! act_type only support none/gelu/silu/relu/fastgelu!!!")
    return output


ACLNN_INDEX_MAP_ALLTOALL = {
    0: "output",
    1: "gatherout",
    2: "bmmout",
}


class ALLTOALL_ALLGATHER_BMM_GRAPH_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, weight, bias, ep_hcomm_info, tp_hcomm_info, ep_size, tp_size, shard_type, act,
                need_allgather_out, need_activation_feature,
                all_to_all_out, tensor_allgather, group_ep, group_tp, reshape_1_shape, reshape_2_shape, reshape_3_shape,
                is_bias, type):
        if type == "mc2":
            output = npu_alltoall_allgather_bmm(x=input,
                                                weight=weight,
                                                bias=bias,
                                                group_ep=ep_hcomm_info,
                                                group_tp=tp_hcomm_info,
                                                group_ep_worldsize=ep_size,
                                                group_tp_worldsize=tp_size,
                                                shard_type=shard_type,
                                                act_type=act,
                                                need_allgather_out=need_allgather_out,
                                                need_activation_feature=need_activation_feature)
        elif type == "hccl_mm":
            output = []
            dist.all_to_all_single(all_to_all_out, input, group=group_ep)
            all_to_all_out = all_to_all_out.reshape(reshape_1_shape).permute(1, 0, 2, 3).contiguous()
            dist._all_gather_base(tensor_allgather, all_to_all_out, group=group_tp)
            all_gather_out = tensor_allgather.reshape(reshape_2_shape)
            if shard_type == 0:
                all_gather_out = all_gather_out.permute(1, 2, 3, 0, 4)
            else:
                all_gather_out = all_gather_out.permute(1, 2, 0, 3, 4)
            allgather_out_golden = all_gather_out.reshape(reshape_3_shape)
            bmm_out_golden = torch.bmm(allgather_out_golden, weight)
            if is_bias:
                # bias 为2维场景，需要reshape为3维再做add
                bias = bias.reshape([bias.shape[0], 1, bias.shape[1]]) if len(bias.shape) == 2 else bias
                bmm_out_golden += bias
            out = activate(act, bmm_out_golden, 'npu')
            output.append(out)
            output.append(all_gather_out)
            output.append(bmm_out_golden)
        else:
            print("In alltoall_allgather_bmm get_graph, wrong type.")
        return output

@class_registry("aclnnAlltoAllAllGatherBatchMatMul")
class ALLTOALL_ALLGATHER_BMM(MC2_OP):
    def __init__(self, case, rank, dist=None, hcomm_info=None):
        super().__init__(case, rank, dist, hcomm_info)
        self.is_bias = case['is_bias'] if 'is_bias' in case else None
        self.is_trans = case['is_trans'] if 'is_trans' in case else None
        self.act_type = case['act_type'] if 'act_type' in case else None
        self.shard_type = int(case['shard_type']) if 'shard_type' in case else None
        self.need_allgather_out = case['need_allgather_out'] if 'need_allgather_out' in case else False
        self.need_activation_feature = case['need_activation_feature'] if 'need_activation_feature' in case else False
    
    def get_input_weight(self, weight_same=0):
        self.input = func.random_input(func.TYPE_DICT.get(self.input_dtype), self.input_range, self.input_shape,
                                       self.seed_num + self.rank, "tensor")
        weight_seed = self.seed_num + self.rank * 2 if weight_same == 0 else self.seed_num
        tmp_weight_shape = [self.weight_shape[0], self.weight_shape[2], self.weight_shape[1]] if self.is_trans else self.weight_shape
        self.weight = func.random_input(func.TYPE_DICT.get(self.weight_dtype), self.weight_range, tmp_weight_shape,
                                        weight_seed, "tensor")
        if self.input_dtype == torch.bfloat16:
           self.input = self.input.to(torch.bfloat16)
        if self.weight_dtype == torch.bfloat16:
           self.weight = self.weight.to(torch.bfloat16)

        if "aclnn" in self.scene and '910D' in self.case['soc'] or "aclnn_golden" in self.scene:
            func.save_file_new(self.input, f"bin_path/{self.case_name}/input/{self.rank}/x1.bin")
            func.save_file_new(self.weight, f"bin_path/{self.case_name}/input/{self.rank}/x2.bin")
            try:
                os.makedirs(f"bin_path/{self.case['case_name']}/output/{self.rank}", self.rank)
            except:
                print("output dir alreday exits!!")
        if self.is_trans:
            self.weight = self.weight.permute(0,2,1)
        self.input = self.input if "cpu" in self.scene else self.input.npu()
        self.weight = self.weight if "cpu" in self.scene else self.weight.npu()

    def get_tmp_tensor(self):
        self.E = self.input_shape[0]
        self.M = self.weight_shape[2] * self.tp_size
        self.bias = func.get_tensor_bais(self.case, self.rank) if self.is_bias else None
        tensor_alltoall_shape = self.input_shape
        if self.shard_type == 0:
            self.C = self.input_shape[1]
            self.H = self.input_shape[2] * self.tp_size
            self.tensor_allgather_shape = [int(self.tp_size * self.E / self.ep_size), self.ep_size, self.C,
                                           int(self.H / self.tp_size)]
            self.reshape_1_shape = [self.ep_size, int(self.E / self.ep_size), self.C, int(self.H / self.tp_size)]
            self.reshape_2_shape = [self.tp_size, int(self.E / self.ep_size), self.ep_size,
                                    self.C, int(self.H / self.tp_size)]
        elif self.shard_type == 1:
            self.C = self.input_shape[1] * self.tp_size
            self.H = self.input_shape[2]
            self.reshape_1_shape = [self.ep_size, int(self.E / self.ep_size), int(self.C / self.tp_size), self.H]
            self.tensor_allgather_shape = [int(self.tp_size * self.E / self.ep_size), self.ep_size,
                                           int(self.C / self.tp_size), self.H]
            self.reshape_2_shape = [self.tp_size, int(self.E / self.ep_size), self.ep_size, int(self.C / self.tp_size),
                                    self.H]
        else:
            print("shard_type only support 0 and 1!!!!!!")
        self.reshape_3_shape = [int(self.E / self.ep_size), int(self.ep_size * self.C), self.H]
        # 预先创建alltoall/allgather接收所需的tensor
        tmp_tensor_alltoall = torch.zeros(tensor_alltoall_shape, dtype=self.input_dtype)
        self.all_to_all_out = tmp_tensor_alltoall if "cpu" in self.scene else tmp_tensor_alltoall.npu()
        tmp_tensor_allgather = torch.zeros(self.tensor_allgather_shape, dtype=self.input_dtype)
        self.tensor_allgather = tmp_tensor_allgather if "cpu" in self.scene else tmp_tensor_allgather.npu()

    def get_tmp_tensor_memory(self):
        # 预先创建的通信接收tensor内存
        memory_alltoall = self.input_shape[0] * self.input_shape[1] * self.input_shape[2] * 2 / 1024 / 1024
        memory_allgather = self.tensor_allgather_shape[0] * self.tensor_allgather_shape[1] * \
                           self.tensor_allgather_shape[2] * 2 / 1024 / 1024
        # 2024.11.25迭代二性能优化新增
        iter2_memory = self.E * self.C / self.tp_size * self.H * 2 / 1024 / 1024
        # 2024.12.10 支持shard_h迭代内存计算
        if self.shard_type == 0:
            if not self.need_activation_feature:  # 无激活劣化E * C * H + (ep-1) * E/ep * C * M/tp + 16M
                final_memory = (self.E * self.C * self.H + (
                        self.ep_size - 1) * self.E / self.ep_size * self.C * self.M / self.tp_size) * 2 / 1024 / 1024 + 16
            else:  # 有激活劣化E * C * H - E / ep * C * M / tp + 16M
                final_memory = (self.E * self.C * self.H - self.E / self.ep_size * self.C * self.M / self.tp_size) * 2 \
                               / 1024 / 1024 + 16
            final_memory = final_memory + memory_alltoall + memory_allgather + iter2_memory
        else:  # shard_type=1
            if self.is_bias:  # 迭代一 :有bias劣化16M
                iter1_memory = 16
            else:  # 无bias劣化E*C/tp*H+16M
                iter1_memory = self.E * self.C / self.tp_size * self.H * 2 / 1024 / 1024 + 16
            final_memory = memory_alltoall + memory_allgather + iter1_memory + iter2_memory
        if self.need_allgather_out:
            final_memory = final_memory + self.E * self.C * self.H * self.tp_size * 2 * 2 / 1024 / 1024
        if self.need_activation_feature:
            final_memory = final_memory + self.E * self.C * self.M / self.tp_size * 2 * 2 / 1024 / 1024
        return final_memory

    def get_hccl_mm(self):
        # alltoall --- shape同input_shape
        dist.all_to_all_single(self.all_to_all_out, self.input, group=self.group_ep)
        #  shard_type=0:reshape(ep, E/ep, C, H/tp) + transpose(E/ep, ep, C, H/tp)
        #  shard_type=1:reshape(ep, E/ep, C/tp, H) + transpose(E/ep, ep, C/tp, H)
        all_to_all_out = self.all_to_all_out.reshape(self.reshape_1_shape).permute(1, 0, 2, 3).contiguous()
        # allgather --- shard_type=0:(tp*E/ep, ep, C, H/tp)、shard_type=1:(tp*E/ep, ep, C/tp, H)
        dist._all_gather_base(self.tensor_allgather, self.all_to_all_out, group=self.group_tp)
        # reshape --- shard_type=0:(tp, E/ep, ep, C, H/tp)、shard_type=1:(tp, E/ep, ep, C/tp, H)
        all_gather_out = self.tensor_allgather.reshape(self.reshape_2_shape)
        if self.shard_type == 0:
            # transpose:(E/ep, ep, C, tp, H/tp)
            all_gather_out = all_gather_out.permute(1, 2, 3, 0, 4)
        else:  # shard_type=1
            # transpose:(E/ep, ep, tp, C/tp, H)
            all_gather_out = all_gather_out.permute(1, 2, 0, 3, 4)
        # reshape --- (E/ep, ep*C, H)
        self.gather_output_golden = all_gather_out.reshape(self.reshape_3_shape)
        # bmm
        self.bmm_out_golden = torch.bmm(self.gather_output_golden, self.weight)
        # add
        if self.is_bias:
            # bias为2维场景，需要reshape为3维再做add
            bias = self.bias.reshape([self.bias.shape[0], 1, self.bias.shape[1]]) if len(
                self.bias.shape) == 2 else self.bias
            self.bmm_out_golden += bias
        self.output_golden = activate(self.act_type, self.bmm_out_golden, 'npu')
        return self.output_golden, self.gather_output_golden, self.bmm_out_golden

    def get_npu(self):
        self.act_type=str(self.act_type)
        out = npu_alltoall_allgather_bmm(x=self.input,
                                         weight=self.weight,
                                         bias=self.bias,
                                         group_ep=self.ep_hcomm_info,
                                         group_ep_worldsize=self.ep_size,
                                         group_tp=self.tp_hcomm_info,
                                         group_tp_worldsize=self.tp_size,
                                         shard_type=self.shard_type,
                                         act_type=self.act_type,
                                         need_allgather_out=self.need_allgather_out,
                                         need_activation_feature=self.need_activation_feature)
        self.output_npu = out[0]
        self.gather_output_npu = out[1] if self.need_allgather_out else None
        self.bmm_output_npu = out[2] if self.need_activation_feature else None
        return self.output_npu

    def get_cpu(self):  # 三方比对需要，当前采用双千分之二标准对比，暂不需要
        input = self.input.cpu().to(torch.float32)
        weight = self.weight.cpu().to(torch.float32)
        alltoall_out = self.all_to_all_out.cpu().to(torch.float32)
        # alltoall
        dist.all_to_all_single(alltoall_out, input, group=self.group_ep)
        alltoall_out = alltoall_out.reshape(self.reshape_1_shape).permute(1, 0, 2, 3).contiguous()
        # allgather 输出(m/rank, k) --- 输出allgather_list：rank个 m/rank,k
        all_gather_list = [torch.zeros_like(alltoall_out) for _ in range(self.tp_size)]
        dist.all_gather(all_gather_list, alltoall_out, group=self.group_tp)
        all_gather_out = torch.cat(all_gather_list, dim=0)  # ----allgather_out:(m,k)
        # reshape
        all_gather_out = all_gather_out.reshape(self.reshape_2_shape)
        if self.shard_type == 0:
            # transpose
            all_gather_out = all_gather_out.permute(1, 2, 3, 0, 4)
        else:
            # transpose
            all_gather_out = all_gather_out.permute(1, 2, 0, 3, 4)
        # reshape
        all_gather_out = all_gather_out.reshape(self.reshape_3_shape)
        # bmm
        bmm_out = torch.bmm(all_gather_out, weight)
        # add
        if self.is_bias:
            bias = self.bias.cpu().to(torch.float32)
            bias = bias.reshape([bias.shape[0], 1, bias.shape[1]]) if len(bias.shape) == 2 else bias
            bmm_out += bias
        # activate
        output_cpu = activate(self.act_type, bmm_out, 'cpu')
        func.save_file(self.save_path, output_cpu, 'bf16', "cpu_output", self.bin_name, self.rank)
        if self.need_allgather_out:
            func.save_file(self.save_path, all_gather_out, 'bf16', "cpu_gatherout", self.bin_name, self.rank)
        if self.need_activation_feature:
            func.save_file(self.save_path, bmm_out, 'bf16', "cpu_bmmout", self.bin_name, self.rank)

    def get_graph(self, type="mc2"):
        model = ALLTOALL_ALLGATHER_BMM_GRAPH_Model()
        model = self.define_model(model, self.graph_type)
        output = model(input=self.input,
                       weight=self.weight,
                       bias=self.bias,
                       ep_hcomm_info=self.ep_hcomm_info,
                       tp_hcomm_info=self.tp_hcomm_info,
                       ep_size=self.ep_size,
                       tp_size=self.tp_size,
                       shard_type=self.shard_type,
                       act=self.act_type,
                       need_allgather_out=self.need_allgather_out,
                       need_activation_feature=self.need_activation_feature,
                       all_to_all_out=self.all_to_all_out,
                       tensor_allgather=self.tensor_allgather,
                       group_ep=self.group_ep,
                       group_tp=self.group_tp,
                       reshape_1_shape=self.reshape_1_shape,
                       reshape_2_shape=self.reshape_2_shape,
                       reshape_3_shape=self.reshape_3_shape,
                       is_bias=self.is_bias,
                       type=type)
        if type == "mc2":
            self.output_npu = output[0]
            self.gather_output_npu = output[1] if self.need_allgather_out else None
            self.bmm_output_npu = output[2] if self.need_activation_feature else None
        elif type == "hccl_mm":
            self.output_golden = output[0]
            self.gather_output_golden = output[1] if self.need_allgather_out else None
            self.bmm_out_golden = output[2] if self.need_activation_feature else None
        else:
            print("In alltoall_allgather_bmm get_graph,wrong type.")
        return output


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
            self.gather_output_npu, self.gather_output_golden, self.bmm_output_npu, self.bmm_out_golden = None, None, None, None
            if self.need_allgather_out:
                gatherout_npu_path = f"{self.save_path}/output/{rank}/gatherout_npu.bin"
                gatherout_golden_path = f"{self.save_path}/output/{rank}/gatherout_hccl.bin"
                self.gather_output_npu = torch.tensor(np.fromfile(gatherout_npu_path, np_dtype).astype(np.float32))
                self.gather_output_golden = torch.tensor(
                    np.fromfile(gatherout_golden_path, np_dtype).astype(np.float32))
            if self.need_activation_feature:
                bmmout_npu_path = f"{self.save_path}/output/{rank}/bmmout_npu.bin"
                bmmout_golden_path = f"{self.save_path}/output/{rank}/bmmout_hccl.bin"
                self.bmm_output_npu = torch.tensor(np.fromfile(bmmout_npu_path, np_dtype).astype(np.float32))
                self.bmm_out_golden = torch.tensor(np.fromfile(bmmout_golden_path, np_dtype).astype(np.float32))
        output_npu = (self.output_npu, self.gather_output_npu, self.bmm_output_npu)
        for index, value in enumerate(output_npu):
            if value is not None:
                compare_rank = rank if "aclnn" in self.case['scene'] else self.rank
                cpu_input_path = os.path.join('bin_path', self.case_name,
                                              '{}_cpu_{}_{}.bin'.format(self.bin_name, ACLNN_INDEX_MAP_ALLTOALL[index],
                                                                        compare_rank))
                output_cpu = torch.tensor(np.fromfile(cpu_input_path, np.float32))
                if index == 0:
                    output_npu = self.output_npu.cpu().view(-1)
                    output_golden = self.output_golden.cpu().view(-1)
                    rst_npu = checkResult(output_npu, output_cpu, "{}_dq_npu".format(self.bin_name))
                    rst_npu.print_result()
                    rst_gpu = checkResult(output_golden, output_cpu, "{}_dq_gpu".format(self.bin_name))
                    rst_gpu.print_result()
                    str1, str2 = rst_npu.check_result_moe(rst_gpu)
                    if 'error' in str1:
                        res = 'FAIL'
                    else:
                        res = 'PASS'
                    print('[INFO] device_{} OUT {} 精度结果为：{}'.format(compare_rank, ACLNN_INDEX_MAP_ALLTOALL[index], res))
                    print('[INFO] device_{} OUT {} 精度计算结束：{} '.format(compare_rank, ACLNN_INDEX_MAP_ALLTOALL[index],
                                                                      time.strftime('%H:%M:%S', time.localtime())))
                    with open('alltoall_allgather_bmm_result.csv', 'a') as f:
                        f.write(self.case_name + "," + str(compare_rank) + ',' + str(res) + ',' + str(
                            ACLNN_INDEX_MAP_ALLTOALL[index]) + '\n')
        return res

    def process_aclnn(self):
        self.root_path = os.path.abspath(os.getcwd())
        self.case_format = ['ND', 'ND', 'ND']
        if self.shard_type == 0:
            self.output_shape = [int(self.input_shape[0] / self.ep_size), self.ep_size * self.input_shape[1],
                                 self.weight_shape[2]]
            self.tensor_allgather_shape = [int(self.input_shape[0] / self.ep_size), self.ep_size * self.input_shape[1],
                                           self.input_shape[-1] * self.tp_size]
        else:  # shard_type=1
            self.output_shape = [int(self.input_shape[0] / self.ep_size),
                                 self.ep_size * self.tp_size * self.input_shape[1],
                                 self.weight_shape[2]]
            self.tensor_allgather_shape = [int(self.input_shape[0] / self.ep_size),
                                           (self.ep_size * self.input_shape[1] * self.tp_size), self.input_shape[2]]
        self.output_dtype = self.case['input_tensor_dtype'][0]
        self.output_format = self.case_format[0]
        self.bin_path = self.root_path + "/bin_path/" + self.case_name
        self.bias_shape = self.case['input_tensor_shape'][2]
        print(f"bin_path: {self.bin_path}")
        self.loop = 1
        act_int = func.get_act_type(self.act_type)
        # 根据架构aarch/x86获取待执行二进制名称
        result = subprocess.check_output(["uname", "-a"]).decode("utf-8")
        arch = "aarch" if "aarch64" in result.lower() else "x86"
        exe_binary = "cpp/bin/" + self.op_name + "_" + arch
        is_quant = "UQUANT"
        if self.is_trans:
            weight_shape = [self.weight_shape[0], self.weight_shape[2], self.weight_shape[1]]
        else:
            weight_shape = self.weight_shape
        need_allgather_out = 1 if self.need_allgather_out else 0
        need_activation_feature = 1 if self.need_activation_feature else 0
        # 拼二进制执行命令
        command = [f"{self.root_path}/{exe_binary}"]
        command.append(f"--is_quant {is_quant}")
        command.append(f"--x1_shape {','.join(map(str, self.input_shape))}")
        command.append(f"--x2_shape {','.join(map(str, weight_shape))}")
        command.append(f"--bias_flag {self.is_bias}")
        command.append(f"--bias_shape {','.join(map(str, self.bias_shape))}")
        command.append(f"--transpose_b_flag {self.is_trans}")
        command.append(f"--ep_size {self.ep_size}")
        command.append(f"--tp_size {self.tp_size}")
        command.append(f"--shard_type {self.shard_type}")
        command.append(f"--act_type {act_int}")
        command.append(f"--input_tensor_dtype {','.join(map(str, self.case['input_tensor_dtype']))}")
        command.append(f"--input_tensor_format {','.join(map(str, self.case_format))}")
        command.append(f"--output_shape {','.join(map(str, self.output_shape))}")
        command.append(f"--output_dtype {self.output_dtype}")
        command.append(f"--output_format {self.output_format}")
        command.append(f"--n_dev {self.world_size}")
        command.append(f"--bin_path {self.bin_path}")
        command.append(f"--loop_cnt {self.loop}")
        command.append(f"--gatherout_shape {','.join(map(str, self.tensor_allgather_shape))}")
        command.append(f"--bmmout_shape {','.join(map(str, self.output_shape))}")
        command.append(f"--need_allgather_out {need_allgather_out}")
        command.append(f"--need_activation_feature {need_activation_feature}")

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
