from .common import *
import torch
from torch.distributed import ReduceOp
from mindspeed.ops.npu_bmm_reduce_scatter_all_to_all import npu_bmm_reducescatter_alltoall
from script import toolss
from ml_dtypes import bfloat16
from .registry import class_registry


class BMM_REDUCESCATTER_ALLTOALL_GRAPH_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, weight, bias, ep_hcomm_info, tp_hcomm_info, ep_size, tp_size, shard_type,
                reduce_scatter_out, all_to_all_out,
                group_ep, group_tp, reshape_1_shape, reshape_2_shape, reshape_3_shape, reshape_4_shape, is_bias, type):
        if type == "mc2":
            output = npu_bmm_reducescatter_alltoall(x=input,
                                                    weight=weight,
                                                    bias=bias,
                                                    group_ep=ep_hcomm_info,
                                                    group_tp=tp_hcomm_info,
                                                    group_ep_worldsize=ep_size,
                                                    group_tp_worldsize=tp_size,
                                                    shard_type=shard_type)
        elif type == "hccl_mm":
            bmm_out = torch.bmm(input, weight)
            bmm_out = bmm_out.reshape(reshape_1_shape)
            if shard_type == 0:
                bmm_out = bmm_out.permute(2, 0, 1, 3)
            else:
                bmm_out = bmm_out.permute(2, 0, 1, 3, 4)
            bmm_out = bmm_out.reshape(reshape_2_shape)
            bmm_out = bmm_out.contiguous()
            dist._reduce_scatter_base(reduce_scatter_out, bmm_out, op=ReduceOp.SUM, group=group_tp)
            if is_bias:
                # bias为二维场景，需要reshape成3维再做add
                bias = bias.reshape([bias.shape[0], 1, bias.shape[1]]) if len(bias.shape) == 2 else bias
                reduce_scatter_out += bias
            reduce_scatter_out = reduce_scatter_out.reshape(reshape_3_shape)
            reduce_scatter_out = reduce_scatter_out.permute(1, 0, 2, 3).contiguous()
            dist.all_to_all_single(all_to_all_out, reduce_scatter_out, group=group_ep)
            output = all_to_all_out.reshape(reshape_4_shape)
        else:
            print("In bmm_reducescatter_alltoall get_graph, wrong type.")
        return output


@class_registry("aclnnBatchMatMulReduceScatterAlltoAll")
class BMM_REDUCESCATTER_ALLTOALL(MC2_OP):
    def __init__(self, case, rank, dist=None, hcomm_info=None):
        super().__init__(case, rank, dist, hcomm_info)
        self.is_bias = case['is_bias'] if 'is_bias' in case else None
        self.is_trans = case['is_trans'] if 'is_trans' in case else None
        self.shard_type = int(case['shard_type']) if 'shard_type' in case else None
     
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
            if self.trans_b:
                self.weight = self.weight.permute(0,2,1)
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
        self.E = self.input_shape[0] * self.ep_size
        self.M = self.input_shape[2] * self.tp_size
        self.C = int(self.input_shape[1] / self.ep_size)
        self.H = self.weight_shape[2]
        self.bias = func.get_tensor_bais(self.case, self.rank) if self.is_bias else None
        if self.shard_type == 0:
            self.reshape_1_shape = [int(self.E / self.ep_size), int(self.ep_size * self.C), self.tp_size,
                                    int(self.H / self.tp_size)]
            self.reshape_2_shape = [int(self.tp_size * self.E / self.ep_size),
                                    self.ep_size * self.C, int(self.H / self.tp_size)]
            self.reshape_3_shape = [int(self.E / self.ep_size), self.ep_size, self.C,
                                    int(self.H / self.tp_size)]
            self.scatter_shape_m = int(self.E / self.ep_size)
            self.tensor_scatter_shape = [int(self.E / self.ep_size), int(self.ep_size * self.C),
                                         int(self.H / self.tp_size)]
            self.alltoall_shape = [self.ep_size, int(self.E / self.ep_size), self.C, int(self.H / self.tp_size)]
            self.reshape_4_shape = [self.E, self.C, int(self.H / self.tp_size)]
        elif self.shard_type == 1:
            self.reshape_1_shape = [int(self.E / self.ep_size), self.ep_size, self.tp_size, int(self.C / self.tp_size),
                                    self.H]
            self.reshape_2_shape = [int(self.tp_size * self.E / self.ep_size),
                                    int(self.ep_size * self.C / self.tp_size), self.H]
            self.scatter_shape_m = int(self.E / self.ep_size)
            self.tensor_scatter_shape = [int(self.E / self.ep_size), int(self.ep_size * self.C / self.tp_size), self.H]
            self.reshape_3_shape = [int(self.E / self.ep_size), self.ep_size, int(self.C / self.tp_size), self.H]
            self.alltoall_shape = [self.ep_size, int(self.E / self.ep_size), int(self.C / self.tp_size), self.H]
            self.reshape_4_shape = [self.E, int(self.C / self.tp_size), self.H]
        else:
            print("shard_type only support 0 and 1!!!!!!")
        tmp_rs_out = torch.zeros(self.tensor_scatter_shape, dtype=self.input_dtype)
        self.reduce_scatter_out = tmp_rs_out if "cpu" in self.scene else tmp_rs_out.npu()
        tmp_alltoall_out = torch.zeros(self.alltoall_shape, dtype=self.input_dtype)
        self.all_to_all_out = tmp_alltoall_out if "cpu" in self.scene else tmp_alltoall_out.npu()

    def get_hccl_mm(self):
        bmm_out = torch.bmm(self.input, self.weight)
        # reshape -- shard_type=0(E/ep, ep*C, tp, H/tp)  shard_type=1(E/ep, ep, tp, C/tp, H)
        bmm_out = bmm_out.reshape(self.reshape_1_shape)
        if self.shard_type == 0:
            # transpose --  (tp, E/ep, ep*C, H/tp)
            bmm_out = bmm_out.permute(2, 0, 1, 3)
        else:  # sahrd_type=1
            # transpose --  (tp, E/ep, ep, C/tp, H)
            bmm_out = bmm_out.permute(2, 0, 1, 3, 4)
            ## reshape -- (tp*E/ep, ep*C/tp, H)
        bmm_out = bmm_out.reshape(self.reshape_2_shape)
        bmm_out = bmm_out.contiguous()
        # reducescatter
        dist._reduce_scatter_base(self.reduce_scatter_out, bmm_out, op=ReduceOp.SUM, group=self.group_tp)
        if self.is_bias:
            # bias为2维场景，需要reshape为3维再做add
            bias = self.bias.reshape([self.bias.shape[0], 1, self.bias.shape[1]]) if len(
                self.bias.shape) == 2 else self.bias
            self.reduce_scatter_out += bias
        ## reshape -- shard_type=0(E/ep, ep, C, H/tp)  shard_type=1(E/ep, ep, C/tp, H)
        reduce_scatter_out = self.reduce_scatter_out.reshape(self.reshape_3_shape)
        ## transpose -- shard_type=0(ep, E/ep,C, H/tp)  shard_type=1(ep, E/ep, C/tp, H)
        reduce_scatter_out = reduce_scatter_out.permute(1, 0, 2, 3).contiguous()
        ## alltoall
        dist.all_to_all_single(self.all_to_all_out, reduce_scatter_out, group=self.group_ep)
        ## reshape -- shard_type=0(E, C, H/tp)  shard_type=1(E, C/tp, H)
        self.output_golden = self.all_to_all_out.reshape(self.reshape_4_shape)
        return self.output_golden

    def get_tmp_tensor_memory(self):
        # 预先创建的通信接收tensor内存
        memory_alltoall = self.alltoall_shape[0] * self.alltoall_shape[1] * self.alltoall_shape[2] * 2 / 1024 / 1024
        memory_scatter = self.tensor_scatter_shape[0] * self.tensor_scatter_shape[1] * self.tensor_scatter_shape[
            2] * 2 / 1024 / 1024
        # 2024.11.25--迭代二性能优化新增 [31*M*(ep-1) + 31*tp*H*(ep-1)]
        iter2_memory = 31 * (self.ep_size - 1) * (self.M + self.tp_size * self.H) * 2 / 1024 / 1024
        if self.shard_type == 0:
            if self.is_bias:  # 有bias劣化:E/ep*(ep-1)*C*(H+M/tp)+16
                final_memory = self.E / self.ep_size * (self.ep_size - 1) * self.C * (
                        self.H + self.M / self.tp_size) * 2 / 1024 / 1024 + 16
            else:  # 无bias劣化:E*C*H/tp+E/ep*(ep-1)*C*(H+M/tp)+16
                final_memory = (self.E * self.C * self.H / self.tp_size) * 2 / 1024 / 1024 + self.E / self.ep_size * (
                        self.ep_size - 1) * self.C * (self.H + self.M / self.tp_size) * 2 / 1024 / 1024 + 16
            final_memory = final_memory + memory_alltoall + memory_scatter + iter2_memory
        else:  # shard_type=1
            if self.is_bias:  # 迭代一 :有bias劣化16M
                iter1_memory = 16
            else:  # 迭代一 :无bias劣化E*C/tp*H+16M
                iter1_memory = self.E * self.C / self.tp_size * self.H * 2 / 1024 / 1024 + 16
            final_memory = memory_alltoall + memory_scatter + iter1_memory + iter2_memory
        return final_memory

    def get_npu(self):
        self.output_npu = npu_bmm_reducescatter_alltoall(x=self.input,
                                                         weight=self.weight,
                                                         bias=self.bias,
                                                         group_ep=self.ep_hcomm_info,
                                                         group_ep_worldsize=self.ep_size,
                                                         group_tp=self.tp_hcomm_info,
                                                         group_tp_worldsize=self.tp_size,
                                                         shard_type=self.shard_type)
        return self.output_npu

    def get_cpu(self):  # 三方比对需要
        input = self.input.cpu().to(torch.float32)
        weight = self.weight.cpu().to(torch.float32)
        alltoall_out = self.all_to_all_out.cpu().to(torch.float32)
        # bmm
        bmm_out = torch.bmm(input, weight)
        # reshape 
        bmm_out = bmm_out.reshape(self.reshape_1_shape)
        if self.shard_type == 0:
            # transpose
            bmm_out = bmm_out.permute(2, 0, 1, 3)
        else:
            # transpose
            bmm_out = bmm_out.permute(2, 0, 1, 3, 4)
        # reshpe
        bmm_out = bmm_out.reshape(self.reshape_2_shape)
        # reducescatter =allreduce+narrow
        bmm_out = bmm_out.contiguous()
        dist.all_reduce(bmm_out, op=dist.ReduceOp.SUM, group=self.group_tp)
        reduce_scatter_out = bmm_out.narrow(0, int(self.rank / self.ep_size) * self.scatter_shape_m,
                                            self.scatter_shape_m)
        # add
        if self.is_bias:
            bias = self.bias.cpu().to(torch.float32)
            bias = bias.reshape([bias.shape[0], 1, bias.shape[1]]) if len(bias.shape) == 2 else bias
            reduce_scatter_out += bias
        # reshape 
        reduce_scatter_out = reduce_scatter_out.reshape(self.reshape_3_shape)
        # transpose
        reduce_scatter_out = reduce_scatter_out.permute(1, 0, 2, 3).contiguous()
        # alltoall
        dist.all_to_all_single(alltoall_out, reduce_scatter_out, group=self.group_ep)
        # reshape
        output_cpu = alltoall_out.reshape(self.reshape_4_shape)
        func.save_file(self.save_path, output_cpu, 'bf16', "cpu_output", self.bin_name, self.rank)

    def get_graph(self, type="mc2"):
        model = BMM_REDUCESCATTER_ALLTOALL_GRAPH_Model()
        model = self.define_model(model, self.graph_type)
        output = model(input=self.input,
                       weight=self.weight,
                       bias=self.bias,
                       ep_hcomm_info=self.ep_hcomm_info,
                       tp_hcomm_info=self.tp_hcomm_info,
                       ep_size=self.ep_size,
                       tp_size=self.tp_size,
                       shard_type=self.shard_type,
                       reduce_scatter_out=self.reduce_scatter_out,
                       all_to_all_out=self.all_to_all_out,
                       group_ep=self.group_ep,
                       group_tp=self.group_tp,
                       reshape_1_shape=self.reshape_1_shape,
                       reshape_2_shape=self.reshape_2_shape,
                       reshape_3_shape=self.reshape_3_shape,
                       reshape_4_shape=self.reshape_4_shape,
                       is_bias=self.is_bias,
                       type=type)
        if type == "mc2":
            self.output_npu = output
        elif type == "hccl_mm":
            self.output_golden = output
        else:
            print("In bmm_reducescatter_alltoall get_graph,wrong type.")
        return output

    def compare(self, rank=0, check_md5=0):
        op = self.case['op_name']
        if "aclnn" in self.case['scene']:  # aclnn场景，golden和npu从已保存文件中读取
            output_npu_path = f"{self.save_path}/output/{rank}/output_npu.bin"
            print(f"output_npu_path ,{output_npu_path}")
            output_golden_path = f"{self.save_path}/output/{rank}/output_hccl.bin"
            if self.output_dtype == 'fp16':
                np_dtype = np.float16
            else:
                np_dtype = bfloat16
            self.output_npu = torch.tensor(np.fromfile(output_npu_path, np_dtype).astype(np.float32))
            self.output_golden = torch.tensor(np.fromfile(output_golden_path, np_dtype).astype(np.float32))
            cpu_input_path = os.path.join('bin_path', self.case_name,
                                          '{}_cpu_output_{}.bin'.format(self.bin_name, rank))
        else:
            output_npu = self.output_npu.cpu().view(-1)
            output_golden = self.output_golden.cpu().view(-1)
            cpu_input_path = os.path.join('bin_path', self.case_name,
                                          '{}_cpu_output_{}.bin'.format(self.bin_name, self.rank))
        output_cpu = torch.tensor(np.fromfile(cpu_input_path, np.float32))
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
        compare_rank = rank if "aclnn" in self.case['scene'] else self.rank
        print('[INFO] device_{} 精度结果为：{}'.format(compare_rank, res))
        print('[INFO] device_{} 精度计算结束：{} '.format(compare_rank, time.strftime('%H:%M:%S', time.localtime())))
        with open('{}_result.csv'.format(op), 'a') as f:
            f.write(self.case_name + "," + str(compare_rank) + ',' + str(res) + '\n')
        return res

    def process_aclnn(self):
        self.root_path = os.path.abspath(os.getcwd())
        self.case_format = ['ND', 'ND', 'ND']
        self.output_shape = [int(self.input_shape[0] * self.ep_size),
                             int(self.input_shape[1] / self.ep_size / self.tp_size),
                             self.weight_shape[2]]
        self.output_dtype = self.case['input_tensor_dtype'][0]
        self.output_format = self.case_format[0]
        self.bin_path = self.root_path + "/bin_path/" + self.case_name
        self.bias_shape = self.case['input_tensor_shape'][2]
        print(f"bin_path: {self.bin_path}")
        self.loop = 1
        # 根据架构aarch/x86获取待执行二进制名称
        result = subprocess.check_output(["uname", "-a"]).decode("utf-8")
        arch = "aarch" if "aarch64" in result.lower() else "x86"
        exe_binary = "cpp/bin/" + self.op_name + "_" + arch
        is_quant = "UQUANT"
        if self.is_trans:
            weight_shape = [self.weight_shape[0], self.weight_shape[2], self.weight_shape[1]]
        else:
            weight_shape = self.weight_shape
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
        command.append(f"--input_tensor_dtype {','.join(map(str, self.case['input_tensor_dtype']))}")
        command.append(f"--input_tensor_format {','.join(map(str, self.case_format))}")
        command.append(f"--output_shape {','.join(map(str, self.output_shape))}")
        command.append(f"--output_dtype {self.output_dtype}")
        command.append(f"--output_format {self.output_format}")
        command.append(f"--n_dev {self.world_size}")
        command.append(f"--bin_path {self.bin_path}")
        command.append(f"--loop_cnt {self.loop}")

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
