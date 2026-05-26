from .common import *
from op_class.aclnnMatmulAllReduceV2 import MatmulAllReduceV2
import torch
from torch.distributed import ReduceOp
from .registry import class_registry
import os


# 此函数仅A16W8支持
class WeightQuantMatmulAllReduceGraphModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, hcom, bias, antiquant_scale, antiquant_offset, x3, antiquant_group_size,
                x1_golden, bias_golden, is_nz, is_x3, is_trans_b, x2_dtype, hcom_type, is_antiquant_offset, soc):
        if is_trans_b:
            x2 = x2.t()
        if len(antiquant_scale.shape) == 2 and is_trans_b:
            antiquant_scale = antiquant_scale.t()
            if is_antiquant_offset:
                antiquant_offset = antiquant_offset.t()
        if is_nz:
            x2 = torch_npu.npu_format_cast(x2, 29)
        if hcom_type == "mc2":
            if soc == "910D":
                output = torch_npu.npu_mm_all_reduce_base(
                    x1=x1,
                    x2=x2,
                    hcom=hcom,
                    reduce_op="sum",
                    bias=bias,
                    antiquant_scale=antiquant_scale,
                    antiquant_offset=antiquant_offset,
                    x3=x3,
                    antiquant_group_size=antiquant_group_size,
                    x2_dtype=DTYPE_ENUM[x2_dtype],
                )
            else:
                output = torch_npu.npu_mm_all_reduce_base(
                    x1=x1,
                    x2=x2,
                    hcom=hcom,
                    reduce_op="sum",
                    bias=bias,
                    antiquant_scale=antiquant_scale,
                    antiquant_offset=antiquant_offset,
                    x3=x3,
                    antiquant_group_size=antiquant_group_size,
                )
        elif hcom_type == "hccl_mm":
            if soc == "910D":
                output = torch_npu.npu_weight_quant_batchmatmul(
                    x=x1_golden,
                    weight=x2,
                    antiquant_scale=antiquant_scale,
                    antiquant_offset=antiquant_offset,
                    bias=bias_golden,
                    antiquant_group_size=antiquant_group_size,
                    weight_dtype=DTYPE_ENUM[x2_dtype],
                )
            else:
                output = torch_npu.npu_weight_quant_batchmatmul(
                    x=x1_golden,
                    weight=x2,
                    antiquant_scale=antiquant_scale,
                    antiquant_offset=antiquant_offset,
                    bias=bias_golden,
                    antiquant_group_size=antiquant_group_size,
                )
            if is_x3:
                output = output.reshape(x3.shape)
                output = torch.add(output, x3)
            dist.all_reduce(output, op=ReduceOp.SUM)  # torch.float16
        else:
            print("In allreduce a16w8 get_graph,wrong type..")
        return output


@class_registry("aclnnWeightQuantMatmulAllReduce")
class WeightQuantMatmulAllReduce(MatmulAllReduceV2):
    def __init__(self, case, rank, dist=None, hcomm_info=None):
        super().__init__(case, rank, dist, hcomm_info)
        self.is_nz = case['is_nz']
        self.is_antiquant_scale = case['is_antiquant_scale']
        self.antiquant_scale_shape = case['antiquant_scale_shape']
        self.antiquant_scale_range = case['antiquant_scale_range']
        self.antiquant_scale_dtype = case['antiquant_scale_dtype']
        self.is_antiquant_offset = case['is_antiquant_offset']
        self.antiquant_offset_shape = case['antiquant_offset_shape']
        self.antiquant_offset_range = case['antiquant_offset_range']
        self.antiquant_offset_dtype = case['antiquant_offset_dtype']
        self.is_group_size = case['is_group_size']
        self.antiquant_group_size = case['antiquant_group_size'] if self.is_group_size else 0
        self.quant_mode = case["quant_mode"]
        self.antiquant_scale = None
        self.antiquant_offset = None
        self.output_dtype = self.x1_dtype
        self.bias_golden = None

    def get_input_weight(self, weight_same=0):
        # cpu场景用array, npu场景为tensor
        if "cpu" in self.scene or "aclnn" in self.scene:
            self.x1 = self.create_array("x1", self.x1_shape, self.x1_range, self.x1_dtype, self.input_path, self.seed)
            self.x2 = self.create_array("x2", self.x2_shape, self.x2_range, self.x2_dtype, self.input_path, self.seed)
            if self.is_trans_b:
                self.x2 = self.x2.T
            # npu_weight_quant_batchmatmul接口，x1仅支持2维
            if len(self.x1_shape) == 3:
                self.x1_golden = self.x1.reshape(-1, self.x1_shape[-1])
            else:
                self.x1_golden = self.x1
            self.save(self.x1_golden, self.input_path, "x1_golden.bin")
        else:
            x1 = self.create_array("x1", self.x1_shape, self.x1_range, self.x1_dtype, self.input_path, self.seed)
            x2 = self.create_array("x2", self.x2_shape, self.x2_range, self.x2_dtype, self.input_path, self.seed)
            self.x1 = self.tensor(x1)
            self.x2 = self.tensor(x2)
            # 图模式转置要放在图内
            if self.is_trans_b and "graph" not in self.scene:
                self.x2 = self.x2.t()
            # npu_weight_quant_batchmatmul接口，x1仅支持2维
            if len(self.x1_shape) == 3:
                self.x1_golden = self.x1.reshape(-1, self.x1_shape[-1])
            else:
                self.x1_golden = self.x1

    def get_tmp_tensor(self):
        # cpu场景用array, npu场景为tensor
        if "cpu" in self.scene or "aclnn" in self.scene:
            self.antiquant_scale = self.create_array("antiquant_scale", self.antiquant_scale_shape, self.antiquant_scale_range,
                                                     self.antiquant_scale_dtype, self.input_path, self.seed)
            if len(self.antiquant_scale.shape) == 2 and self.is_trans_b:
                self.antiquant_scale = self.antiquant_scale.T
            if self.is_antiquant_offset:
                self.antiquant_offset = self.create_array("antiquant_offset", self.antiquant_offset_shape, self.antiquant_offset_range,
                                                          self.antiquant_offset_dtype, self.input_path, self.seed)
                if len(self.antiquant_offset.shape) == 2 and self.is_trans_b:
                    self.antiquant_offset = self.antiquant_offset.T
            # npu_weight_quant_batchmatmul接口，当output=bf16时,bias=fp32
            if self.is_bias:
                self.bias = self.create_array("bias", self.bias_shape, self.bias_range, self.bias_dtype, self.input_path, self.rank * 3)
                self.bias_golden = self.bias
                self.save(self.bias_golden, self.input_path, "bias_fp32.bin")
            if self.is_x3:
                self.x3 = self.create_array("x3", self.x3_shape, self.x3_range, self.x3_dtype, self.input_path, self.rank * 6)
        else:
            antiquant_scale = self.create_array("antiquant_scale", self.antiquant_scale_shape, self.antiquant_scale_range,
                                                self.antiquant_scale_dtype, self.input_path, self.seed)
            self.antiquant_scale = self.tensor(antiquant_scale)
            if len(self.antiquant_scale.shape) == 2 and self.is_trans_b and "graph" not in self.scene:
                self.antiquant_scale = self.antiquant_scale.t()
                print(f"antiquant scale trans!!!, shape after trans is {self.antiquant_scale.shape}")
            if self.is_antiquant_offset:
                antiquant_offset = self.create_array("antiquant_offset", self.antiquant_offset_shape, self.antiquant_offset_range,
                                                     self.antiquant_offset_dtype, self.input_path, self.seed)
                self.antiquant_offset = self.tensor(antiquant_offset)
                if len(self.antiquant_offset.shape) == 2 and self.is_trans_b and "graph" not in self.scene:
                    self.antiquant_offset = self.antiquant_offset.t()
            if self.is_bias:
                bias = self.create_array("bias", self.bias_shape, self.bias_range, self.bias_dtype, self.input_path, self.rank * 3)
                self.bias = self.tensor(bias)
                # npu_weight_quant_batchmatmul接口，当output=bf16时,bias=fp32
                self.bias_golden = self.bias
            if self.is_x3:
                x3 = self.create_array("x3", self.x3_shape, self.x3_range, self.x3_dtype, self.input_path, self.rank * 6)
                self.x3 = self.tensor(x3)

    # 此接口不支持weight_dtype=int4（此接口仅支持A16W8）
    def get_hccl_mm(self):
        if self.quant_mode == "a16w8":
            if self.soc == "910D":
                mm_output = torch_npu.npu_weight_quant_batchmatmul(
                    x=self.x1_golden,
                    weight=self.x2,
                    antiquant_scale=self.antiquant_scale,
                    antiquant_offset=self.antiquant_offset,
                    bias=self.bias_golden,
                    antiquant_group_size=self.antiquant_group_size,
                    weight_dtype=DTYPE_ENUM[self.x2_dtype],
                )
            else:
                mm_output = torch_npu.npu_weight_quant_batchmatmul(
                    x=self.x1_golden,
                    weight=self.x2,
                    antiquant_scale=self.antiquant_scale,
                    antiquant_offset=self.antiquant_offset,
                    bias=self.bias_golden,
                    antiquant_group_size=self.antiquant_group_size,
                )
            if self.is_x3:
                mm_output = mm_output.reshape(self.x3.shape)
                mm_output = torch.add(mm_output, self.x3)
            dist.all_reduce(mm_output, op=ReduceOp.SUM)  # torch.float16
            self.output_golden = mm_output
        else: ## a16w4场景，仅生成输入，golden输出在--scene aclnn内与mc2一并执行
            try:
                dist.destroy_process_group()
            except:
                pass
            self.output_golden = None

        return self.output_golden

    # 此接口不支持weight_dtype=int4（此接口仅支持A16W8）
    def get_npu(self):
        if self.soc == "910D":
            self.output_npu = torch_npu.npu_mm_all_reduce_base(
                x1=self.x1,
                x2=self.x2,
                hcom=self.hcomm_info,
                reduce_op="sum",
                bias=self.bias,
                antiquant_scale=self.antiquant_scale,
                antiquant_offset=self.antiquant_offset,
                x3=self.x3,
                antiquant_group_size=self.antiquant_group_size,
                x2_dtype=DTYPE_ENUM[self.x2_dtype],
            )
        else:
            self.output_npu = torch_npu.npu_mm_all_reduce_base(
                x1=self.x1,
                x2=self.x2,
                hcom=self.hcomm_info,
                reduce_op="sum",
                bias=self.bias,
                antiquant_scale=self.antiquant_scale,
                antiquant_offset=self.antiquant_offset,
                x3=self.x3,
                antiquant_group_size=self.antiquant_group_size,
            )
        return self.output_npu

    # 此函数不支持weight_dtype=int4（此函数仅支持A16W8）
    def get_graph(self, hcom_type="mc2"):
        model = WeightQuantMatmulAllReduceGraphModel()
        model = self.define_model(model, self.graph_type)
        output = model(
            x1=self.x1,
            x2=self.x2,
            hcom=self.hcomm_info,
            bias=self.bias,
            antiquant_scale=self.antiquant_scale,
            antiquant_offset=self.antiquant_offset,
            x3=self.x3,
            antiquant_group_size=self.antiquant_group_size,
            x1_golden=self.x1_golden,
            bias_golden=self.bias_golden,
            is_nz=self.is_nz,
            is_x3=self.is_x3,
            is_trans_b=self.is_trans_b,
            x2_dtype=self.x2_dtype,
            hcom_type=hcom_type,
            is_antiquant_offset=self.is_antiquant_offset,
            soc=self.soc
        )
        if hcom_type == "mc2":
            self.output_npu = output
        elif hcom_type == "hccl_mm":
            self.output_golden = output
        else:
            print("In allreduce a16w8 get_graph,wrong type..")
        return output

    # cpu-golden: ( (weight + offset)*antiquant_scale ) * input + bias --> x3 --->allreduce
    def get_cpu(self):
        # x1/x2均为numpy.array
        x1 = self.x1.astype(np.float32)
        x2 = self.x2.astype(np.float32)
        antiquant_scale = self.antiquant_scale.astype(np.float32)
        if self.is_group_size:
            for i in range(self.x1_shape[-1]):
                group_size_dim = i // self.antiquant_group_size
                if self.is_antiquant_offset:
                    antiquant_offset = self.antiquant_offset.astype(np.float32)
                    x2[i, :] = (x2[i, :] + antiquant_offset[group_size_dim, :]) * antiquant_scale[
                                                                                  group_size_dim, :]
                else:
                    x2[i, :] = x2[i, :] * antiquant_scale[group_size_dim, :]
        else:
            if self.is_antiquant_offset:
                antiquant_offset = self.antiquant_offset.astype(np.float32)
                x2 = x2 + antiquant_offset
            x2 = x2 * antiquant_scale
        x2 = x2.astype(np.float32)
        mm = np.matmul(x1, x2)
        if self.is_bias:
            bias = self.bias
            mm += bias
        if self.is_x3:
            x3 = self.x3.astype(np.float32)
            mm += x3
        mm = torch.from_numpy(mm)
        dist.all_reduce(mm, op=ReduceOp.SUM)
        self.save(mm, self.output_path, self.output_cpu_name)

    def get_command(self, binary, rank=0, is_david=0):
        case_format = ['ND', 'ND', 'ND'] if not self.is_nz else ['ND', 'NZ', 'ND']
        input_tensor_dtype = ','.join(map(str, self.case['input_tensor_dtype'])) + ',' + self.case['antiquant_scale_dtype'] + ',' + \
                             self.case['antiquant_offset_dtype']
        input_tensor_format = ','.join(map(str, case_format)) + ',ND' + ',ND'
        if self.is_x3:
            input_tensor_dtype = input_tensor_dtype + ',' + self.case['x3_dtype']
            input_tensor_format = input_tensor_format + ',ND'
        bin_path = self.bin_path
        is_quant = self.quant_mode.upper()
        command = [f"{MAIN_PATH}/bin_path/{binary}"] if is_david else [f"{MAIN_PATH}/cpp/bin/{binary}"]
        command.append(f"--is_quant {is_quant}")
        command.append(f"--x1_shape {','.join(map(str, self.x1_shape))}")
        command.append(f"--x2_shape {','.join(map(str, self.x2_shape))}")
        command.append(f"--transpose_b_flag {int(self.is_trans_b)}")
        command.append(f"--input_tensor_dtype {input_tensor_dtype}")
        command.append(f"--input_tensor_format {input_tensor_format}")
        command.append(f"--output_shape {','.join(map(str, self.output_shape))}")
        command.append(f"--output_dtype {self.output_dtype}")
        command.append(f"--output_format {self.output_format}")
        command.append(f"--n_dev {self.world_size}")
        command.append(f"--bin_path {bin_path}")
        command.append(f"--loop_cnt {self.loop}")
        command.append(f"--bias_shape {self.output_shape[-1]}")
        command.append(f"--bias_flag {self.is_bias}")
        command.append(f"--x3_flag {self.is_x3}")
        if self.is_x3:
            command.append(f"--x3_shape {','.join(map(str, self.x3_shape))}")
        if self.is_antiquant_offset:
            command.append(f"--antiquant_offset_flag {self.is_antiquant_offset}")
        command.append(f"--antiquant_scale_shape {','.join(map(str, self.antiquant_scale_shape))}")
        command.append(f"--antiquant_offset_shape {','.join(map(str, self.antiquant_offset_shape))}")
        command.append(f"--antiquant_group_size {self.antiquant_group_size}")
        if self.quant_mode == "a16w4":
            command.append(f"--is_hccl_mm 1")
        if is_david:
            command.append(f"--rankId {rank} \&")
        aclnn_command = " ".join(command)
        return aclnn_command

    def process_aclnn(self):
        MatmulAllReduceV2.process_aclnn(self)
