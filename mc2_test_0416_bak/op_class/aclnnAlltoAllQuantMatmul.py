import os
from .common import *
import torch
import torch_npu
from script import toolss
from script.func import *
from script.david_ds.quantize import *
from .registry import class_registry
from script import generate_mx_data
from op_class.aclnnAlltoAllMatmul import AlltoAllMatmul


DATA_TYPE_QUANTIZE_TO_TORCH = {
    'float32': 'fp32',
    'float16': 'fp16',
    'bfloat16': 'bf16',
    'hifloat8': 'hif8',
    'float8_e5m2': 'fp8_e5m2',
    'float8_e4m3fn': 'fp8_e4m3fn',
    'float8_e8m0': 'fp8_e8m0'
}

def get_group_scale(m, n, gm, gn, scale):
    M_block = m // gm
    N_block = n // gn
    assert scale.shape == (M_block, N_block), \
        f"scale形状应为({M_block}, {N_block})，当前为{scale.shape}（与分块数量不匹配）"
    broadcasted_subblocks = []
    for i in range(M_block):
        row_subblocks = []
        for j in range(N_block):
            subblock_scale = scale[i, j]
            subblock_broadcasted = subblock_scale.expand(gm, gn)
            row_subblocks.append(subblock_broadcasted)
        row_broadcasted = torch.cat(row_subblocks, dik=1)
        broadcasted_subblocks.append(row_broadcasted)
    scale = torch.cat(broadcasted_subblocks, dik=0)
    return scale.to(torch.float32)


def QuantByToken(input, dtype, max_value, smoothScales):
    """
    dtype (str): "fp8_e4m3fn"或"fp8_e5m2"
    """
    if len(input.shape) != 2:
        raise ValueError("输入必须是二维张量！")
    input_tensor = input * smoothScales
    abs_input = torch.abs(input_tensor)
    row_max_abs = torch.amax(abs_input, dim=1, keepdim=True)  # 按行取最大值，dim=1对应行
    scaleOut = row_max_abs / max_value
    zero_row_mask = (row_max_abs == 0.0)
    scaleOut[zero_row_mask] = 1.0
    yOut = input_tensor / scaleOut
    quant_output = yOut.to(TORCH_DTYPE_MAP[dtype])
    return quant_output, scaleOut


def QuantByChannel(input, dtype, max_value):
    """
    dtype (str): "fp8_e4m3fn"或"fp8_e5m2"
    """
    m, n = input.shape
    quant_output = torch.zeros_like(input)
    scales = []
    for j in range(n):
        col = input[:, j]
        abs_max = torch.max(torch.abs(col))
        if abs_max == 0:
            scale = 1.0
        else:
            scale = max_value / abs_max
        scales.append(scale)
        quant_output[:, j] = col * scale

    scales = torch.tensor(scales)  # 形状：[n]
    quant_output = quant_output.to(TORCH_DTYPE_MAP[dtype])

    return quant_output, scales


class AlltoAllQuantMatmulGraphModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, hcom, world_size, all2all_out_flag=True, bias=None, x1_scale=None, x2_scale=None,
                common_scale=None, x1_offset=None, x2_offset=None, x1_quant_mode=7, x2_quant_mode=2,
                common_quant_mode=0, group_sizes=[0, 0, 0], all2all_axes=None, comm_quant_dtype=-1,
                x1_quant_dtype=None, x1_dtype=None, x2_dtype=None, x1_scale_dtype=None, x2_scale_dtype=None,
                output_scale_dtype=None, comm_scale_dtype=None, y_dtype=None, transpose_x2=True, hcom_type="mc2"):
        '''
        :param transpose_x2: 测试专用，图模式入参：构造x2转置节点
        '''
        if transpose_x2:
            x2 = x2.t()

        if hcom_type == "mc2":
            output, alltoall_output = torch_npu.npu_all_to_all_quant_matmul(x1=x1,
                                                                            x2=x2,
                                                                            hcom=hcom,
                                                                            world_size=world_size,
                                                                            all2all_out_flag=all2all_out_flag,
                                                                            bias=bias,
                                                                            x1_scale=x1_scale,
                                                                            x2_scale=x2_scale,
                                                                            common_scale=common_scale,
                                                                            x1_offset=x1_offset,
                                                                            x2_offset=x2_offset,
                                                                            x1_quant_mode=x1_quant_mode,
                                                                            x2_quant_mode=x2_quant_mode,
                                                                            common_quant_mode=common_quant_mode,
                                                                            group_sizes=group_sizes,
                                                                            all2all_axes=all2all_axes,
                                                                            comm_quant_dtype=comm_quant_dtype,
                                                                            x1_quant_dtype=x1_quant_dtype,
                                                                            x1_dtype=x1_dtype,
                                                                            x2_dtype=x2_dtype,
                                                                            x1_scale_dtype=x1_scale_dtype,
                                                                            x2_scale_dtype=x2_scale_dtype,
                                                                            output_scale_dtype=output_scale_dtype,
                                                                            comm_scale_dtype=comm_scale_dtype,
                                                                            y_dtype=y_dtype)

        else:
            pass
        return output, alltoall_output


@class_registry("aclnnAlltoAllQuantMatmul")
class AlltoAllQuantMatmul(AlltoAllMatmul):
    def __init__(self, case, rank, dist=None, hcomm_info=None):
        super().__init__(case, rank, dist, hcomm_info)
        self.x1Scale = None  # 初始化未定义变量
        self.x2Scale = None  # 初始化未定义变量
        self.x1Scale_shape = case["x1_scale_shape"]
        self.x1Scale_range = case["x1_scale_range"]
        self.x1Scale_dtype = case["x1_scale_dtype"]
        self.is_x1_scale = case["is_x1_scale"]
        
        self.x2Scale_shape = case["x2_scale_shape"]
        self.x2Scale_range = case["x2_scale_range"]
        self.x2Scale_dtype = case["x2_scale_dtype"]
        self.x1QuantMode = case["x1_quant_mode"]  # perToken-dynamic 7
        self.x2QuantMode = case["x2_quant_mode"]  # perChannel 2
        self.x1QuantDtype = case["x1_quant_dtype"]  # 35 if 'e5m2' in self.x2_dtype else 36
        self.x1QuantDtype_mapping = DATA_TYPE_INT_TO_STR[self.x1QuantDtype]
        self.groupSize = case["group_size"]
        self.groupSizeDec = self.unpack_group_size(self.groupSize)
        # 固定参数，预留接口，直接写死
        self.commQuantMode = 0
        self.commQuantDtype = -1
        self.is_mxfp = 1 if self.x1QuantMode == 6 and self.x2QuantMode == 6 else 0
        #mxfp4场景，左右矩阵使用uint8存储fp4，为了保证数据量一致，对应shape最内轴减半
        if "aclnn" != self.scene and (self.alltoall_dtype == "fp4_e2m1" or self.alltoall_dtype == "fp4_e1m2"):
            self.alltoall_compare_shape = self.alltoall_shape[:]
            self.temp_shape[-1] = int(self.temp_shape[-1] / 2)
            self.alltoall_shape[-1] = int(self.alltoall_shape[-1] / 2)

    def get_input_weight_kc(self, weight_same=0):
        x1 = self.create_array("x1", self.x1_shape, self.x1_range, self.x1_dtype, self.input_path, self.seed)
        x2 = self.create_array("x2", self.x2_shape, self.x2_range, self.x2_dtype, self.input_path, self.weight_seed)
        x2Scale = self.create_array("x2Scale", self.x2Scale_shape, self.x2Scale_range, self.x2Scale_dtype,
                                    self.input_path, self.weight_seed)
        self.x1 = self.tensor(x1)
        self.x2 = self.tensor(x2)
        self.x2Scale = self.tensor(x2Scale)
        if self.is_x1_scale:
            x1Scale = self.create_array("x1Scale", self.x1Scale_shape, self.x1Scale_range, self.x1Scale_dtype,
                        self.input_path, self.weight_seed)
            self.x1Scale = self.tensor(x1Scale)
            # self.smoothScale = self.x1Scale.clone() if self.x1QuantMode == 7 else 1

        if self.is_bias:
            bias = self.create_array("bias", self.bias_shape, self.bias_range, self.bias_dtype, self.input_path,
                                     self.weight_seed)
            self.bias = self.tensor(bias)
        self.input = self.x1.t() if self.transposeX1 else self.x1
        self.weight = self.x2.t() if self.transposeX2 else self.x2
        self.input_npu = self.input.contiguous()
        self.weight_npu = self.weight
        self.x2Scale_npu = self.tensor(x2Scale)
        # 图模式
        self.x1_graph = self.tensor(x1)
        self.x2_graph = self.tensor(x2)
        self.x1_scale_graph = None
        self.x2_scale_graph = self.tensor(x2Scale)

    def mx_generate_data(self):
        dtype_map = {
            "fp8_e5m2": "float8_e5m2",
            "fp8_e4m3fn": "float8_e4m3fn",
            "fp8_e8m0": "float8_e8m0",
            "fp4_e2m1": "float4_e2m1",
            "fp4_e1m2": "float4_e1m2",
        } 
        x1_fp32 = self.create_array("x1", self.x1_shape, self.x1_range, "fp32", self.input_path, self.seed)
        x2_fp32 = self.create_array("x2", self.x2_shape, self.x2_range, "fp32", self.input_path, self.weight_seed)

        # 生成x1量化后的数据
        x1, x1Scale = generate_mx_data.mx_quantize(x1_fp32, dtype_map[self.x1_dtype], axis=-1, block_size=32, round_mode="rint")
        if "cpu" in self.scene and self.x1_dtype in ['fp4_e2m1', 'fp4_e1m2']:
            x1 = self.trans_to_float32(self.array(x1), self.x1_dtype)
        if self.x1_dtype in ['fp4_e2m1', 'fp4_e1m2'] and "cpu" not in self.scene:
            x1 = func.pack_int4(x1)
            x1 = x1.reshape(self.x1_shape[0], self.x1_shape[1] >> 1)

        # 生成x2量化后的数据
        if self.transposeX2:
            x2, x2Scale = generate_mx_data.mx_quantize(x2_fp32, dtype_map[self.x2_dtype], axis=-1, block_size=32, round_mode="rint")
        else:
            x2, x2Scale = generate_mx_data.mx_quantize(x2_fp32, dtype_map[self.x2_dtype], axis=0, block_size=32, round_mode="rint")
        if "cpu" in self.scene and self.x2_dtype in ['fp4_e2m1', 'fp4_e1m2']:
            x2 = self.trans_to_float32(self.array(x2), self.x2_dtype)
        if self.x2_dtype in ['fp4_e2m1', 'fp4_e1m2'] and "cpu" not in self.scene:
            x2 = func.pack_int4(x2)
            x2 = x2.reshape(self.x2_shape[0], self.x2_shape[1] >> 1)
        if not self.transposeX2:
            x2 = np.ascontiguousarray(x2)
            x2Scale = np.ascontiguousarray(x2Scale)

        if self.save_flag:
            os.remove(os.path.join(self.input_path, 'x1.bin'))
            os.remove(os.path.join(self.input_path, 'x2.bin'))
            x1.tofile(os.path.join(self.input_path, 'x1.bin'))
            x2.tofile(os.path.join(self.input_path, 'x2.bin'))
            x1Scale.tofile(os.path.join(self.input_path, 'x1Scale.bin'))
            x2Scale.tofile(os.path.join(self.input_path, 'x2Scale.bin'))
        self.x2 = x2
        self.tempX1 = self.tensor(x1)
        self.tempX2 = self.tensor(x2)
        self.tempX1Scale = self.tensor(x1Scale)
        self.tempX2Scale = self.tensor(x2Scale)

        self.tempAll2AllOut = torch.zeros(self.temp_shape, dtype=TORCH_DTYPE_MAP[self.alltoall_dtype]).to(self.device)
        alltoall_x1Scale_dtype = TORCH_DTYPE_MAP[self.x1Scale_dtype] if "cpu" not in self.scene else torch.float32
        temp_x1Scale_shape = [self.world_size, self.tempX1Scale.shape[0] // self.world_size, self.tempX1Scale.shape[1] * self.tempX1Scale.shape[2]]
        self.tempAll2AllX1ScaleOut = torch.zeros(temp_x1Scale_shape, dtype=alltoall_x1Scale_dtype).to(self.device)
        
        self.input = self.tempX1.t() if self.transposeX1 else self.tempX1
        # pta matmul输入
        self.input_npu = self.input.contiguous()
        self.weight_npu = self.tempX2.t() if self.transposeX2 else self.tempX2
        self.x1Scale_npu = self.tempX1Scale
        self.x2Scale_npu = self.tempX2Scale
        # 图模式
        self.x1_graph = self.tensor(x1)
        self.x2_graph = self.tensor(x2)
        self.x1_scale_graph = self.tensor(x1Scale)
        self.x2_scale_graph = self.tensor(x2Scale)

    def all2allX1AndX1Scale(self, runType):
        input_re = self.input.to(torch.float32).reshape(self.temp_shape) if "cpu" in self.scene else self.input.reshape(self.temp_shape)
        dist.all_to_all_single(self.tempAll2AllOut, input_re)
        self.alltoall_out_mx = self.tempAll2AllOut.permute(1, 0, 2).reshape(self.alltoall_shape)
        self.alltoall_out_mx = self.alltoall_out_mx.contiguous()
        # torch.cpu.synchronize()
        temp_x1Scale_shape = [self.world_size, self.tempX1Scale.shape[0] // self.world_size, self.tempX1Scale.shape[1] * self.tempX1Scale.shape[2]]
        x1Scale = self.tempX1Scale.reshape(temp_x1Scale_shape)
        dist.all_to_all_single(self.tempAll2AllX1ScaleOut, x1Scale)

        self.alltoall_x1Scale_out = self.tempAll2AllX1ScaleOut.permute(1, 0, 2)
        temp_x1Scale_shape = [self.alltoall_x1Scale_out.shape[0], self.alltoall_x1Scale_out.shape[1] * self.alltoall_x1Scale_out.shape[2] // 2, 2]
        self.alltoall_x1Scale_out = self.alltoall_x1Scale_out.reshape(temp_x1Scale_shape)
        self.alltoall_x1Scale_out = self.alltoall_x1Scale_out.contiguous()
        
        # 小算子级联matmul输入
        if runType == "hccl":
            self.weight = self.tempX2.t() if self.transposeX2 else self.tempX2
            self.x1Scale = self.alltoall_x1Scale_out
            self.x2Scale = self.tempX2Scale.transpose(0,1) if self.transposeX2 else self.tempX2Scale
        # cpu matmul数据
        if runType == "cpu":
            self.x1 = self.alltoall_out_mx
            self.x1Scale = self.alltoall_x1Scale_out
            self.x2Scale_numpy = self.tempX2Scale.numpy()

    def get_input_weight(self, weight_same=0):
        self.x1QuantDtypeNpu = TORCH_DTYPE_MAP[DATA_TYPE_QUANTIZE_TO_TORCH[self.x1QuantDtype_mapping]]
        if self.is_mxfp:
            self.x1_scale_dtype=DTYPE_ENUM[self.x1Scale_dtype]
            self.x2_scale_dtype=DTYPE_ENUM[self.x2Scale_dtype]
            if self.is_bias:
                bias = self.create_array("bias", self.bias_shape, self.bias_range, self.bias_dtype, self.input_path,
                                        self.weight_seed)
                self.bias = self.tensor(bias)
            self.mx_generate_data()
        else:
            self.x1_scale_dtype=None
            self.x2_scale_dtype=None
            self.x1Scale_npu = None
            self.get_input_weight_kc()

    def scale_broadcast(self, input, dtype):
        '''
        0: 不量化
        1: pertensor
        2: perchanenl
        3: pertoken
        4: pergroup
        5: perblock
        6: mx量化 fpe8m0 + 每行 + 每32/128列
        7: pertoken动态量化
        '''

        x1Scale = self.x1Scale
        inputQuant = input
        if self.x1QuantMode == 0:
            x1Scale = 1
        elif self.x1QuantMode == 1:
            x1Scale = self.x1Scale.to(torch.float32)
        elif self.x1QuantMode == 2:
            x1Scale = self.x1Scale.unsqueeze(0).to(torch.float32)
        elif self.x1QuantMode == 3:
            x1Scale = self.x1Scale.unsqueeze(1).to(torch.float32)
        elif self.x1QuantMode == 4:
            m, k = self.input.shape
            G_m, G_n, G_k = self.groupSizeDec
            x1Scale = get_group_scale(m, k, G_m, G_k, x1Scale)
        elif self.x1QuantMode == 5:
            pass
        elif self.x1QuantMode == 6:
            pass
        elif self.x1QuantMode == 7:
            max_map = {
                "fp8_e4m3fn": 448.0,
                "fp8_e5m2": 57344.0,
            }
            max_value = max_map[dtype]
            x1_smooth_scale = self.x1Scale.unsqueeze(0) if self.x1Scale else 1
            inputQuant, scales = QuantByToken(input, dtype, max_value, x1_smooth_scale)
            inputQuant = inputQuant.to(torch.float32)
            x1Scale = scales.to(torch.float32)
        else:
            pass

        x2Scale = self.x2Scale
        if self.x2QuantMode == 0:
            x2Scale = 1
        elif self.x2QuantMode == 1:
            x2Scale = self.x2Scale.to(torch.float32)
        elif self.x2QuantMode == 2:
            x2Scale = self.x2Scale.unsqueeze(0).to(torch.float32)
        elif self.x2QuantMode == 3:
            x2Scale = self.x2Scale.unsqueeze(1).to(torch.float32)
        elif self.x2QuantMode == 4:
            k, n = self.weight.shape
            G_m, G_n, G_k = self.groupSizeDec
            x2Scale = get_group_scale(k, n, G_k, G_n, x2Scale)
        elif self.x2QuantMode == 5:
            pass
        elif self.x2QuantMode == 6:
            pass
        elif self.x2QuantMode == 7:
            pass
        else:
            pass

        return inputQuant, x1Scale, x2Scale

    def cpu_mx_compute(self):
        x1 = self.x1.numpy().astype(np.float32)
        x2 = self.x2.astype(np.float32)
        pertoken_scale_mx = self.x1Scale.numpy().astype(np.float32)
        deq_scale_mx = self.x2Scale_numpy.astype(np.float32)
        pertoken_scale_mx = pertoken_scale_mx.reshape(pertoken_scale_mx.shape[0], pertoken_scale_mx.shape[1] * pertoken_scale_mx.shape[2])
        if self.transposeX2:
            x2 = np.swapaxes(x2, -1, -2)
            deq_scale_mx = deq_scale_mx.reshape(deq_scale_mx.shape[0], deq_scale_mx.shape[1] * deq_scale_mx.shape[2])
            deq_scale_mx = np.swapaxes(deq_scale_mx, -1, -2)
        else:
            deq_scale_mx = np.swapaxes(deq_scale_mx, -1, -2)
            deq_scale_mx = deq_scale_mx.reshape(deq_scale_mx.shape[0] * deq_scale_mx.shape[1], deq_scale_mx.shape[2])
        output = self.mxfp_cpu_compute(x1, x2, pertoken_scale_mx, deq_scale_mx)
        if self.is_bias:
            output += self.bias.to(torch.float32)
        return output

    def get_cpu(self):
        if self.is_mxfp:
            self.all2allX1AndX1Scale("cpu")
            cpu_mm_dequant_out=self.cpu_mx_compute()
            self.output_cpu = cpu_mm_dequant_out
            self.alltoall_output_cpu = self.alltoall_out_mx if self.is_alltoall_output else None
        else:
            input_re = self.input.to(torch.float32).reshape(self.temp_shape)
            alltoall_out = self.alltoall_out
            dist.all_to_all_single(alltoall_out, input_re)
            torch.cpu.synchronize()
            alltoall_out = alltoall_out.permute(1, 0, 2).reshape(self.alltoall_shape)
            quantAlltoAllOut, x1Scale, x2Scale = self.scale_broadcast(
                alltoall_out, DATA_TYPE_QUANTIZE_TO_TORCH[self.x1QuantDtype_mapping])
            weight = self.weight.to(torch.float32)
            if self.is_bias:
                cpu_mm_out = torch.matmul(quantAlltoAllOut, weight) + self.bias.to(torch.float32)
            else:
                cpu_mm_out = torch.matmul(quantAlltoAllOut, weight)
            cpu_mm_dequant_out = cpu_mm_out * x1Scale * x2Scale
            self.output_cpu = cpu_mm_dequant_out
            self.alltoall_output_cpu = alltoall_out if self.is_alltoall_output else None

        self.save(self.output_cpu, self.output_path, self.output_cpu_name)
        if self.is_alltoall_output:
            self.save(self.alltoall_output_cpu, self.output_path, self.alltoall_output_cpu_name)

    def get_hccl_mm(self):
        # npu 小算子拼接
        alltoall_output_hccl_save = None
        if self.is_mxfp:
            self.all2allX1AndX1Scale("hccl")
            quantAlltoAllOut=self.alltoall_out_mx
            x1Scale=self.alltoall_x1Scale_out
            pertoken_scale_dtype=DTYPE_ENUM[self.x1Scale_dtype]
            # torch.npu.synchronize()
            self.alltoall_output_hccl = self.alltoall_out_mx if self.is_alltoall_output else None
            alltoall_output_hccl_save = self.alltoall_output_hccl.view(torch.uint8) if self.is_alltoall_output else None
        else:
            input_re = self.input.reshape(self.temp_shape)
            dist.all_to_all_single(self.alltoall_out, input_re)
            alltoall_out = self.alltoall_out.permute(1, 0, 2).reshape(self.alltoall_shape)

            quantAlltoAllOut, x1Scale = torch_npu.npu_dynamic_quant(alltoall_out, smooth_scales=self.x1Scale,
                dst_type=DTYPE_TO_INT_VALUE_MAP[self.x1QuantDtype_mapping], quant_mode="pertoken")
            pertoken_scale_dtype=None
            self.alltoall_output_hccl = alltoall_out if self.is_alltoall_output else None
            alltoall_output_hccl_save = self.alltoall_output_hccl

        mm_res = torch_npu.npu_quant_matmul(
            x1=quantAlltoAllOut,
            x2=self.weight,
            pertoken_scale=x1Scale,
            scale=self.x2Scale,
            offset=None,
            bias=self.bias,
            output_dtype=TORCH_DTYPE_MAP[self.output_dtype],
            x1_dtype=DTYPE_ENUM[self.x1_dtype],
            x2_dtype=DTYPE_ENUM[self.x2_dtype],
            pertoken_scale_dtype=pertoken_scale_dtype,
            scale_dtype=DTYPE_ENUM[self.x2Scale_dtype],
            group_sizes=self.groupSizeDec,
            y_scale=None,
        )
        self.output_hccl = mm_res
        # self.save(self.output_hccl, self.output_path, self.output_hccl_name)
        return self.output_hccl, None, None, None, None, alltoall_output_hccl_save

    def get_npu(self):
        output = torch_npu.npu_all_to_all_quant_matmul(self.input_npu, self.weight_npu,
                                                       self.hcomm_info, self.world_size,
                                                       bias=self.bias, x1_scale=self.x1Scale_npu, x2_scale=self.x2Scale_npu,
                                                       common_scale=None,
                                                       x1_offset=None,
                                                       x2_offset=None,
                                                       x1_quant_mode=self.x1QuantMode,
                                                       x2_quant_mode=self.x2QuantMode,
                                                       x1_quant_dtype=self.x1QuantDtypeNpu,
                                                       common_quant_mode=self.commQuantMode,
                                                       group_sizes=self.groupSizeDec,
                                                       all2all_axes=None,
                                                       comm_quant_dtype=self.commQuantDtype,
                                                       x1_dtype=DTYPE_ENUM[self.x1_dtype],
                                                       x2_dtype=DTYPE_ENUM[self.x2_dtype],
                                                       x1_scale_dtype=self.x1_scale_dtype,
                                                       x2_scale_dtype=self.x2_scale_dtype,
                                                       output_scale_dtype=None,
                                                       comm_scale_dtype=None,
                                                       y_dtype=TORCH_DTYPE_MAP[self.output_dtype],
                                                       all2all_out_flag=self.is_alltoall_output)

        self.output_npu, self.alltoall_output_npu = output
        # self.save(self.output_npu, self.output_path, self.output_npu_name)
        return self.output_npu

    def get_graph(self, hcom_type="mc2"):
        if hcom_type == "mc2":
            model = AlltoAllQuantMatmulGraphModel()
            model = self.define_model(model, self.graph_type)

            out_graph = model(x1=self.x1_graph,  # self.input_npu,
                              x2=self.x2_graph,  # self.weight_npu,
                              hcom=self.hcomm_info,
                              world_size=self.world_size,
                              all2all_out_flag=self.is_alltoall_output,
                              bias=self.bias,
                              x1_scale=self.x1_scale_graph,  # self.x1Scale_npu,
                              x2_scale=self.x2_scale_graph,  # self.x2Scale_npu,
                              common_scale=None,
                              x1_offset=None,
                              x2_offset=None,
                              x1_quant_mode=self.x1QuantMode,
                              x2_quant_mode=self.x2QuantMode,
                              x1_quant_dtype=self.x1QuantDtypeNpu,
                              common_quant_mode=self.commQuantMode,
                              group_sizes=self.groupSizeDec,
                              all2all_axes=None,
                              comm_quant_dtype=self.commQuantDtype,
                              x1_dtype=DTYPE_ENUM[self.x1_dtype],
                              x2_dtype=DTYPE_ENUM[self.x2_dtype],
                              x1_scale_dtype=self.x1_scale_dtype,
                              x2_scale_dtype=self.x2_scale_dtype,
                              output_scale_dtype=None,
                              comm_scale_dtype=None,
                              y_dtype=TORCH_DTYPE_MAP[self.output_dtype],
                              transpose_x2=self.transposeX2,
                              hcom_type=hcom_type)
            self.output_npu, self.alltoall_output_npu = out_graph
            return self.output_npu
        elif hcom_type == "hccl_mm":
            self.get_hccl_mm()
            return self.output_hccl
        else:
            print("In alltoall get_graph,wrong type..")

    def compare(self, rank=0, check_md5=0):
        if self.alltoall_dtype == "fp4_e2m1" or self.alltoall_dtype == "fp4_e1m2":
            print("unpack fp4 alltoall out")
            if self.is_alltoall_output and self.scene not in ["aclnn"]:
                # 优化：减少中间内存分配，一次性完成 unpack + trans + reshape
                alltoall_hccl_unpacked = self.unpack_int4(self.alltoall_output_hccl.cpu().numpy())
                alltoall_npu_unpacked = self.unpack_int4(self.alltoall_output_npu.cpu().numpy())
                # trans_to_float32 已是向量化实现，直接生成 float32 并 reshape
                self.alltoall_output_hccl = torch.from_numpy(
                    self.trans_to_float32(alltoall_hccl_unpacked, self.alltoall_dtype)
                    .reshape(self.alltoall_compare_shape)
                )
                self.alltoall_output_npu = torch.from_numpy(
                    self.trans_to_float32(alltoall_npu_unpacked, self.alltoall_dtype)
                    .reshape(self.alltoall_compare_shape)
                )
            self.compare_mxfp4(rank, check_md5)
        else:
            super().compare(rank, check_md5)
    
    def compare_mxfp4(self, rank=0, check_md5=0):
        ACLNN_INDEX_MAP_ALLTOALL = {
            0: "output",
            1: "alltoall_output",
        }
        outDtype = NUMPY_DTYPE_MAP[self.output_dtype]
        allDtype = np.uint8
        if self.scene in ["aclnn"]:  # aclnn场景，hccl和npu从已保存文件中读取
            output_npu_path = os.path.join(self.output_path, self.output_npu_name)
            output_hccl_path = os.path.join(self.output_path, self.output_hccl_name)
            if not os.path.exists(output_npu_path):
                raise FileNotFoundError(f"Output file not found: {output_npu_path}")
            self.output_npu = torch.from_numpy(np.fromfile(output_npu_path, outDtype).astype(np.float32))
            self.output_hccl = torch.from_numpy(np.fromfile(output_hccl_path, outDtype).astype(np.float32))
            if self.is_alltoall_output:
                alltoall_out_npu_path = os.path.join(self.output_path, self.alltoall_output_npu_name)
                alltoall_out_hccl_path = os.path.join(self.output_path, self.alltoall_output_hccl_name)
                # 优化：trans_to_float32 已返回 float32，无需再转
                alltoall_output_npu_temp = self.trans_to_float32(
                    self.unpack_int4(np.fromfile(alltoall_out_npu_path, allDtype)),
                    self.alltoall_dtype
                ).reshape(self.alltoall_shape)
                alltoall_output_hccl_temp = self.trans_to_float32(
                    self.unpack_int4(np.fromfile(alltoall_out_hccl_path, allDtype)),
                    self.alltoall_dtype
                ).reshape(self.alltoall_shape)
                self.alltoall_output_npu = torch.from_numpy(alltoall_output_npu_temp)
                self.alltoall_output_hccl = torch.from_numpy(alltoall_output_hccl_temp)

        output_cpu_path = os.path.join(self.output_path, self.output_cpu_name)
        self.output_cpu = torch.from_numpy(np.fromfile(output_cpu_path, np.float32))
        if self.is_alltoall_output:
            alltoall_out_cpu_path = os.path.join(self.output_path, self.alltoall_output_cpu_name)
            self.alltoall_output_cpu = torch.from_numpy(np.fromfile(alltoall_out_cpu_path, np.float32))

        compare_rank = rank if "aclnn" in self.scene else self.rank
        output_npus = (self.output_npu, self.alltoall_output_npu)
        output_cpus = (self.output_cpu, self.alltoall_output_cpu)
        output_hccls = (self.output_hccl, self.alltoall_output_hccl)
        for index, value in enumerate(output_npus):
            if value is not None:
                output_cpu = output_cpus[index].view(-1).to(torch.float32)
                output_npu = output_npus[index].cpu().view(-1).to(torch.float32)
                output_hccl = output_hccls[index].cpu().contiguous().view(-1).to(torch.float32)
                res = "FAIL"
                rst_npu_cpu = checkResult(output_npu, output_cpu, "{}_npu_cpu".format(ACLNN_INDEX_MAP_ALLTOALL[index]))
                rst_npu_cpu.print_result()
                rst_hccl_cpu = checkResult(output_hccl, output_cpu,
                                           "{}_hccl_cpu".format(ACLNN_INDEX_MAP_ALLTOALL[index]))
                rst_hccl_cpu.print_result()
                str1, str2 = rst_npu_cpu.check_result(rst_hccl_cpu)
                if 'error' in str1:
                    res = 'FAIL'
                else:
                    res = 'PASS'
                print(
                    '[INFO] device_{} OUT {} 精度结果为：{}'.format(compare_rank, ACLNN_INDEX_MAP_ALLTOALL[index], res))
                print('[INFO] device_{} OUT {} 精度计算结束：{} '.format(compare_rank, ACLNN_INDEX_MAP_ALLTOALL[index],
                                                                        time.strftime('%H:%M:%S', time.localtime())))
                with open('{}_result.csv'.format(self.op_name), 'a') as f:
                    f.write(self.case_name + "," + str(compare_rank) + ',' + str(res) + ',' + str(
                        ACLNN_INDEX_MAP_ALLTOALL[index]) + '\n')
        return res

    def get_command(self, binary, rankid):
        input_tensor_format = ["ND", "ND", "ND"]
        command = [f"nohup {self.root_path}/bin_path/{binary}"]
        command.append(f"--bin_path {self.bin_path}")
        command.append(f"--x1_shape {','.join(map(str, self.x1_shape))}")
        command.append(f"--x2_shape {','.join(map(str, self.x2_shape))}")
        command.append(f"--bias_shape {','.join(map(str, self.bias_shape))}")
        command.append(f"--transpose_x1_flag {self.transposeX1}")
        command.append(f"--transpose_x2_flag {self.transposeX2}")
        command.append(f"--input_tensor_dtype {','.join(map(str, self.input_tensor_dtype))}")
        command.append(f"--input_tensor_format {','.join(map(str, input_tensor_format))}")
        command.append(f"--output_shape {','.join(map(str, self.output_shape))}")
        command.append(f"--output_dtype {self.output_dtype}")
        command.append(f"--alltoall_dtype {self.alltoall_dtype}")
        command.append(f"--output_format {self.output_format}")
        command.append(f"--n_dev {self.world_size}")
        command.append(f"--loop_cnt {self.loop}")
        command.append(f"--alltoall_shape {','.join(map(str, self.alltoall_shape))}")
        command.append(f"--bias_flag {self.is_bias}")
        command.append(f"--alltoall_flag {self.is_alltoall_output}")
        command.append(f"--x2Scale_shape {','.join(map(str, self.x2Scale_shape))}")
        command.append(f"--x1QuantMode {self.x1QuantMode}")
        command.append(f"--x2QuantMode {self.x2QuantMode}")
        command.append(f"--commQuantMode {self.commQuantMode}")
        command.append(f"--commQuantDtype {self.commQuantDtype}")
        command.append(f"--x1QuantDtype {self.x1QuantDtype}")
        command.append(f"--groupSize {self.groupSize}")
        command.append(f"--x2Scale_dtype {self.x2Scale_dtype}")
        command.append(f"--x1Scale_shape {','.join(map(str, self.x1Scale_shape))}")
        command.append(f"--x1Scale_dtype {self.x1Scale_dtype}")
        command.append(f"--x1Scale_flag {self.is_x1_scale}")
        command.append(f"--rankId {rankid} \&")
        return " ".join(command)

    def unpack_int4(self, src):
        """
        Unpack uint8 array (each byte contains two int4 values) into a uint8 array 
        where each element represents one int4 value (in low 4 bits, value 0-15).
        
        Args:
            src: Packed uint8 array, shape (N,)
            
        Returns:
            unpacked: uint8 array of shape (2*N,), each element in [0, 15]
        """
        if src.dtype != np.uint8:
            raise RuntimeError(f"Input dtype must be uint8, got {src.dtype}")
        
        src = src.flatten()
        # Extract low 4 bits and high 4 bits
        low_nibbles = src & 0x0F          # lower 4 bits
        high_nibbles = (src >> 4) & 0x0F  # upper 4 bits
        
        # Stack and flatten in correct order
        unpacked = np.empty((src.size * 2,), dtype=np.uint8)
        unpacked[0::2] = low_nibbles   # even indices: original first element
        unpacked[1::2] = high_nibbles  # odd indices: original second element
        
        return unpacked
