from .common import *
import torch.distributed as dist
from torch.distributed import ReduceOp
from .registry import class_registry
from op_class.aclnnMatmulAllReduce import MatmulAllReduce


class MatmulAllReduceV2GraphModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, hcom, bias, x3, is_bias, is_x3, is_trans_b, hcom_type):
        if is_trans_b:
            x2 = x2.t()
        if hcom_type == "mc2":
            output = torch_npu.npu_mm_all_reduce_base(
                x1=x1,
                x2=x2,
                hcom=hcom,
                reduce_op="sum",
                bias=bias,
                x3=x3,
            )
        else:
            output = torch.matmul(x1, x2)
            if is_bias:
                output += bias
            if is_x3:
                output += x3
            dist.all_reduce(output, op=ReduceOp.SUM)
        return output


@class_registry("aclnnMatmulAllReduceV2")
class MatmulAllReduceV2(MatmulAllReduce):
    def __init__(self, case, rank, dist=None, hcomm_info=None):
        super().__init__(case, rank, dist, hcomm_info)
        self.is_x3 = case['is_x3']
        self.x3_shape = case["x3_shape"]
        self.x3_range = case["x3_range"]
        self.x3_dtype = case["x3_dtype"]
        self.x3 = None

    def get_tmp_tensor(self):
        MatmulAllReduce.get_tmp_tensor(self)
        if self.is_x3:
            x3 = self.create_array("x3", self.x3_shape, self.x3_range, self.x3_dtype, self.input_path, self.rank * 6)
            self.x3 = self.tensor(x3)

    def get_input_weight(self, weight_same=0):
        MatmulAllReduce.get_input_weight(self, weight_same)

    def get_hccl_mm(self):
        output = torch.matmul(self.x1, self.x2)
        if self.is_bias:
            output += self.bias
        if self.is_x3:
            output += self.x3
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
            x3=self.x3,
        )
        return self.output_npu

    def get_graph(self, hcom_type="mc2"):
        if hcom_type not in ["mc2", "hccl_mm"]:
            print(f"[ERROR] unsupported hcom_type {hcom_type}")
            sys.exit(1)
        model = MatmulAllReduceV2GraphModel()
        model = self.define_model(model, self.graph_type)
        output = model(
            self.x1,
            self.x2,
            self.hcomm_info,
            self.bias,
            self.x3,
            self.is_bias,
            self.is_x3,
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
        if self.is_x3:
            x3 = self.x3.to(torch.float32)
            output += x3
        dist.all_reduce(output, op=dist.ReduceOp.SUM)
        self.save(output, self.output_path, self.output_cpu_name)

    def get_command(self, binary, rank=0, is_david=0):
        # 拼二进制执行命令 [k,n]
        input_tensor_dtype = ','.join(map(str, self.case['input_tensor_dtype']))
        input_tensor_format = ','.join(map(str, ['ND', 'ND', 'ND']))
        input_tensor_dtype = input_tensor_dtype + ',' + self.case['x3_dtype']
        input_tensor_format = input_tensor_format + ',ND'
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
        command.append(f"--x3_flag {self.is_x3}")
        if self.is_x3:
            command.append(f"--x3_shape {','.join(map(str, self.case['x3_shape']))}")
        if is_david:
            command.append(f"--rankId {rank} \&")
        aclnn_command = " ".join(command)
        return aclnn_command

    def process_aclnn(self):
        MatmulAllReduce.process_aclnn(self)
