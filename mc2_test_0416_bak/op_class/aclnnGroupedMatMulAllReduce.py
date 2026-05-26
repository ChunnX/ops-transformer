from .common import *
import torch
import torch_npu
from torch.distributed import ReduceOp
from mindspeed.ops.npu_grouped_mat_mul_all_reduce import npu_grouped_mat_mul_all_reduce
from .registry import class_registry


class GMM_ALLREDUCE_GRAPH_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, hcom, bias, group_list, split_item, comm_turn=0, reduce_op='sum'):
        output_npu = npu_grouped_mat_mul_all_reduce(x=x1,
                                                    weight=x2,
                                                    hcomm=hcom,
                                                    reduce_op=reduce_op,
                                                    bias=bias,
                                                    group_list=group_list,
                                                    split_item=split_item,
                                                    comm_turn=comm_turn
                                                    )
        return output_npu

@class_registry("aclnnGroupedMatMulAllReduce")
class GMM_ALLREDUCE(MC2_OP):
    def __init__(self, case, rank, dist=None, hcomm_info=None):
        super().__init__(case, rank, dist, hcomm_info)
        # allreduce涉及的所有参数进行初始化
        self.is_bias = case['is_bias'] if 'is_bias' in case else None
        self.group_list = case['group_list'] if 'group_list' in case else None
        self.split_item = case['split_item'] if 'split_item' in case else 0

    def get_tmp_tensor(self):
        self.x1 = func.get_tensor_list(self.case, self.rank, name='x1')
        self.x2 = func.get_tensor_list(self.case, self.rank, name='x2')
        self.bias = func.get_tensor_list(self.case, self.rank, name='bias') if self.is_bias else None

        self.x1_npu = func.get_x_split(self.x1, 1, self.world_size)

        if self.split_item == 1 or self.split_item == 3:
            self.x1_cpu = []
            for rank in range(self.world_size):
                x_split = []
                offset = 0
                for i in self.group_list:
                    x_split.append(self.x1_npu[self.rank][0][offset:i, :])
                    offset = i
                self.x1_cpu.append(x_split)
        else:
            self.x1_cpu = self.x1_npu
        self.x2_cpu = func.get_x_split(self.x2, 0, self.world_size)

        bias_dtype = func.get_dtype(self.case['bias_dtype'][0])  # 默认所有bias dtype相同
        if self.is_bias:
            self.bias_cpu = []
            for tensor in self.bias:
                a = tensor.to(torch.float32) / self.world_size  # bias进行升精度除法
                b = a.to(bias_dtype) if "cpu" in self.scene else a.to(bias_dtype).npu()  # 恢复成原精度模式
                self.bias_cpu.append(b)
        else:
            self.bias_cpu = None if torch.__version__ > '2.0.1' else []

        self.scale = None if torch.__version__ > '2.0.1' else []
        self.offset = None if torch.__version__ > '2.0.1' else []
        self.antiquant_scale = None if torch.__version__ > '2.0.1' else []
        self.antiquant_offset = None if torch.__version__ > '2.0.1' else []

    def get_hccl_mm(self):
        self.output_golden = torch_npu.npu_grouped_matmul(self.x1_npu[self.rank], self.x2_cpu[self.rank],
                                                          bias=self.bias_cpu, scale=self.scale,
                                                          offset=self.offset,
                                                          antiquant_scale=self.antiquant_scale,
                                                          antiquant_offset=self.antiquant_offset,
                                                          group_list=self.group_list, split_item=self.split_item)
        for tensor in self.output_golden:
            dist.all_reduce(tensor, op=ReduceOp.SUM)
        return self.output_golden

    def get_npu(self):
        print('[INFO] device_{} 构造output_npu数据'.format(self.rank))
        self.output_npu = npu_grouped_mat_mul_all_reduce(x=self.x1_npu[self.rank],
                                                         weight=self.x2_cpu[self.rank],
                                                         hcomm=self.hcomm_info,
                                                         reduce_op='sum',
                                                         bias=self.bias_cpu,
                                                         group_list=self.group_list,
                                                         split_item=self.split_item,
                                                         comm_turn=0
                                                         )
        return self.output_npu

    def get_graph(self, type="mc2"):
        model = GMM_ALLREDUCE_GRAPH_Model()
        model = self.define_model(model, self.graph_type)
        output = model(x=self.x1_npu[self.rank],
                       weight=self.x2_cpu[self.rank],
                       hcomm=self.hcomm_info,
                       reduce_op='sum',
                       bias=self.bias,
                       group_list=self.group_list,
                       split_item=self.split_item,
                       comm_turn=0
                       )
        return self.output_npu


    def get_cpu(self):
        print('[INFO] device_{} 构造output_cpu数据'.format(self.rank))
        res_cpu = []

        for i in range(len(self.x1_cpu[self.rank])):
            x1_tmp = self.x1_cpu[self.rank][i].cpu().to(torch.float32)
            x2_tmp = self.x2_cpu[self.rank][i].cpu().to(torch.float32)
            mm_tmp = torch.matmul(x1_tmp, x2_tmp)
            if self.is_bias:
                bias_tmp = self.bias_cpu[i].cpu().to(torch.float32)
                gmm_output_cpu = torch.add(mm_tmp, bias_tmp)
            else:
                gmm_output_cpu = mm_tmp
            dist.all_reduce(gmm_output_cpu, op=dist.ReduceOp.SUM)
            res_cpu.append(gmm_output_cpu)

        output_cpu = func.change_to_one_dim(res_cpu)
        func.save_file(self.save_path, output_cpu, 'bf16', "cpu_output", self.bin_name, self.rank)


    def compare(self, check_md5=0):
        op = self.case['op_name']
        cpu_input_path = os.path.join('bin_path', self.case_name,
                                      '{}_cpu_output_{}.bin'.format(self.bin_name, self.rank))
        output_cpu = torch.tensor(np.fromfile(cpu_input_path, np.float32))
        output_npu = func.change_to_one_dim(self.output_npu).cpu()  # output_npu tensor_list转成1维tensor进行比较
        output_golden = func.change_to_one_dim(self.output_golden).cpu()  # 转成1维tensor
        rst_npu = checkResult(output_npu, output_cpu, "{}_dq_npu".format(self.bin_name))
        if rst_npu != 999:
            rst_npu.print_result()
        rst_gpu = checkResult(output_golden, output_cpu, "{}_dq_gpu".format(self.bin_name))
        if rst_gpu != 999:
            rst_gpu.print_result()
        if rst_gpu == 999 and rst_npu == 999:
            res = 'PASS'
        else:
            str1, str2 = rst_npu.check_result(rst_gpu)
            if 'error' in str1:
                res = 'FAIL'
            else:
                res = 'PASS'
        print('[INFO] device_{} 精度结果为：{}'.format(self.rank, res))
        print('[INFO] device_{} 精度计算结束：{} '.format(self.rank, time.strftime('%H:%M:%S', time.localtime())))
        with open('{}_result.csv'.format(op), 'a') as f:
            f.write(self.case_name + "," + str(self.rank) + ',' + str(res) + '\n')
        return res
