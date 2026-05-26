import os
import torch
from .common import *
from op_class.aclnnMatmulAllReduceV2 import MatmulAllReduceV2
from torch.distributed import ReduceOp
from script.func import *
from .registry import class_registry


class QuantMatmulAllReduceGraphModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, hcom, bias, x3, dequant_scale, x1_golden, dequant_scale_golden, is_nz, is_x3, is_trans_b, hcom_type):
        if is_nz:
            x2 = torch_npu.npu_format_cast(x2, 29)
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
                dequant_scale=dequant_scale,
            )
        else:
            if dequant_scale.dtype in [torch.int64, torch.float32]:
                mm_output_dtype = torch.float16
            else:
                mm_output_dtype = torch.bfloat16
            output = torch_npu.npu_quant_matmul(
                x1=x1_golden,
                x2=x2,
                scale=dequant_scale_golden,
                bias=bias,
                output_dtype=mm_output_dtype
            )
            if is_x3:
                output = output.reshape(x3.shape)
                output = torch.add(output, x3)
            dist.all_reduce(output, op=ReduceOp.SUM)
        return output


@class_registry("aclnnQuantMatmulAllReduce")
class QuantMatmulAllReduce(MatmulAllReduceV2):
    def __init__(self, case, rank, dist=None, hcomm_info=None):
        super().__init__(case, rank, dist, hcomm_info)
        self.is_nz = case['is_nz']
        self.is_dequant_scale = case['is_dequant_scale']
        self.dequant_scale_shape = case["dequant_scale_shape"]
        self.dequant_scale_range = case["dequant_scale_range"]
        self.dequant_scale_dtype = case["dequant_scale_dtype"]
        self.dequant_scale = None
        self.dequant_scale_golden = None
        self.output_dtype = "bf16" if self.dequant_scale_dtype == "bf16" else "fp16"
        self.is_pertoken = False
        self.quant_type = "v1"

    def get_input_weight(self, weight_same=0):
        MatmulAllReduceV2.get_input_weight(self, weight_same)

    def get_tmp_tensor(self):
        MatmulAllReduceV2.get_tmp_tensor(self)
        if len(self.x1_shape) == 3 and "fp4" not in self.x1_dtype:
            self.x1_golden = self.x1.reshape(-1, self.x1_shape[-1])
        else:
            self.x1_golden = self.x1
        if self.is_dequant_scale and self.quant_type != "mxfp":
            dequant_scale = self.create_array(
                "dequant_scale", self.dequant_scale_shape, self.dequant_scale_range,
                self.dequant_scale_dtype, self.input_path, self.rank * 7
            )
            self.dequant_scale = self.tensor(dequant_scale)
            # fp32需要转换为int64
            if self.dequant_scale_dtype == "fp32" and "cpu" not in self.scene:
                if not self.is_pertoken:
                    self.dequant_scale = torch_npu.npu_trans_quant_param(self.dequant_scale)
            # per channel场景，dequant_scale shape 为[1,n]时，matmul的scale当前仅支持1维（即[n]）
            if len(self.dequant_scale_shape) == 2 and self.dequant_scale_shape[0] == 1:
                self.dequant_scale_golden = self.dequant_scale.reshape(self.dequant_scale_shape[-1])
            else:
                self.dequant_scale_golden = self.dequant_scale
        ## quantmatmulallreduceV4 perblock量化场景，非图模式下，对dequantscale做转置
        if self.is_trans_b and self.quant_type == "per_block" and "graph" not in self.scene:
            self.dequant_scale = self.dequant_scale.t()
            self.dequant_scale_golden = self.dequant_scale_golden.t()

    def get_npu(self):
        self.output_npu = torch_npu.npu_mm_all_reduce_base(
            x1=self.x1,
            x2=self.x2,
            hcom=self.hcomm_info,
            reduce_op="sum",
            bias=self.bias,
            x3=self.x3,
            dequant_scale=self.dequant_scale
        )
        return self.output_npu

    def get_hccl_mm(self):
        if self.dequant_scale_golden.dtype in [torch.int64, torch.float32]:
            mm_output_dtype = torch.float16
        else:
            mm_output_dtype = torch.bfloat16
        mm_output = torch_npu.npu_quant_matmul(
            x1=self.x1_golden,
            x2=self.x2,
            scale=self.dequant_scale_golden,
            bias=self.bias,
            output_dtype=mm_output_dtype
        )
        if self.is_x3:
            mm_output = mm_output.reshape(self.x3.shape)
            mm_output += self.x3
        dist.all_reduce(mm_output, op=ReduceOp.SUM)
        self.output_golden = mm_output
        return self.output_golden

    # cpu golden：allreduce(dequant_scale*pertoken_scale*(x1@x2+bias)+x3)
    def get_cpu(self):
        x1 = self.x1.to(torch.int32)
        x2 = self.x2.to(torch.int32)
        dequant_scale = self.dequant_scale.to(torch.float32)
        output = torch.matmul(x1, x2)  # torch.int32
        if self.is_bias:
            output += self.bias
        output = torch.mul(output, dequant_scale)
        if self.is_x3:
            output = output.reshape(self.x3.shape)
            x3 = self.x3.to(torch.float32)
            output += x3
        # 通信部分
        dist.all_reduce(output, op=dist.ReduceOp.SUM)
        self.save(output, self.output_path, self.output_cpu_name)

    def get_graph(self, hcom_type="mc2"):
        model = QuantMatmulAllReduceGraphModel()
        model = self.define_model(model, self.graph_type)
        output = model(
            self.x1,
            self.x2,
            self.hcomm_info,
            self.bias,
            self.x3,
            self.dequant_scale,
            self.x1_golden,
            self.dequant_scale_golden,
            self.is_nz,
            self.is_x3,
            self.is_trans_b,
            hcom_type,
        )
        if hcom_type == "mc2":
            self.output_npu = output
        else:
            self.output_golden = output
        return output

    def get_command(self, binary, rank=0, is_david=0):
        case_format = ['ND', 'ND', 'ND'] if not self.is_nz else ['ND', 'NZ', 'ND']
        input_tensor_dtype = ','.join(map(str, self.case['input_tensor_dtype']))
        input_tensor_format = ','.join(map(str, case_format))
        input_tensor_dtype = input_tensor_dtype + ',' + self.dequant_scale_dtype
        input_tensor_format = input_tensor_format + ',ND'
        if self.is_x3:
            input_tensor_dtype += ',' + self.case['x3_dtype']
            input_tensor_format += ',ND'
        bin_path = self.bin_path
        command = [f"{MAIN_PATH}/bin_path/{binary}"] if is_david else [f"{MAIN_PATH}/cpp/bin/{binary}"]
        is_quant = "A8W8"
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
        command.append(f"--loop_cnt {self.loop}")
        command.append(f"--bias_flag {self.is_bias}")
        command.append(f"--bias_shape {self.output_shape[-1]}")
        command.append(f"--dequant_scale_shape {','.join(map(str, self.dequant_scale_shape))}")
        if self.is_x3:
            command.append(f"--x3_flag {int(self.is_x3)}")
            command.append(f"--x3_shape {','.join(map(str, self.x3_shape))}")
        command.append(f"--bin_path {bin_path}")
        command.append(f"--is_quant {is_quant}")
        if is_david:
            command.append(f"--rankId {rank} \&")
        aclnn_command = " ".join(command)
        return aclnn_command

    def process_aclnn(self):
        MatmulAllReduceV2.process_aclnn(self)
