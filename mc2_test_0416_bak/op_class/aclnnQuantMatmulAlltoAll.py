import os
from .common import *
import torch
import torch_npu
from script import toolss
from script.func import *
from .registry import class_registry
from op_class.aclnnMatmulAlltoAll import MatmulAlltoAll
from script import generate_mx_data

class QuantMatmulAlltoAllGraphModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, hcom, world_size, bias=None, x1_scale=None, x2_scale=None,
                common_scale=None, x1_offset=None, x2_offset=None, x1_quant_mode=3, x2_quant_mode=2,
                common_quant_mode=0, group_sizes=[0, 0, 0], all2all_axes=None, comm_quant_dtype=28,
                x1_dtype=None, x2_dtype=None, x1_scale_dtype=None, x2_scale_dtype=None,
                output_scale_dtype=None, comm_scale_dtype=None, y_dtype=6, transpose_x2=True, hcom_type="mc2"):
        '''
        :param transpose_x2: 测试专用，图模式入参：构造x2转置节点
        '''
        if transpose_x2:
            x2 = x2.t()

        if hcom_type == "mc2":
            output = torch_npu.npu_quant_matmul_all_to_all(x1=x1,
                                                           x2=x2,
                                                           hcom=hcom,
                                                           world_size=world_size,
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
                                                           x1_dtype=x1_dtype,
                                                           x2_dtype=x2_dtype,
                                                           x1_scale_dtype=x1_scale_dtype,
                                                           x2_scale_dtype=x2_scale_dtype,
                                                           output_scale_dtype=output_scale_dtype,
                                                           comm_scale_dtype=comm_scale_dtype,
                                                           y_dtype=y_dtype)
        else:
            pass
        return output


@class_registry("aclnnQuantMatmulAlltoAll")
class QuantMatmulAlltoAll(MatmulAlltoAll):
    def __init__(self, case, rank, dist=None, hcomm_info=None):
        super().__init__(case, rank, dist, hcomm_info)
        self.x1Scale = None  # 初始化未定义变量
        self.x2Scale = None  # 初始化未定义变量
        self.x1Scale_shape = case["x1_scale_shape"]
        self.x2Scale_shape = case["x2_scale_shape"]
        self.x1Scale_range = case["x1_scale_range"]
        self.x2Scale_range = case["x2_scale_range"]
        self.x1Scale_dtype = case["x1_scale_dtype"]
        self.x2Scale_dtype = case["x2_scale_dtype"]
        self.x1QuantMode = case["x1_quant_mode"]  # perToken
        self.x2QuantMode = case["x2_quant_mode"]  # perChannel
        self.groupSize = case["group_size"]  # 用于pergroup分组
        self.groupSizeDec = self.unpack_group_size(self.groupSize)
        # 固定参数，预留接口，直接写死
        self.commQuantMode = 0
        self.commQuantDtype = -1
        self.is_mxfp = True if self.x1QuantMode == 6 and self.x2QuantMode == 6 else False

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
        x1, x1Scale = generate_mx_data.mx_quantize(x1_fp32, dtype_map[self.x1_dtype], axis=-1, block_size=32, round_mode="rint")
        if self.transposeX2:
            x2, x2Scale = generate_mx_data.mx_quantize(x2_fp32, dtype_map[self.x2_dtype], axis=-1, block_size=32, round_mode="rint") 
        else:
            x2, x2Scale = generate_mx_data.mx_quantize(x2_fp32, dtype_map[self.x2_dtype], axis=0, block_size=32, round_mode="rint")
            x2 = np.ascontiguousarray(x2)
            x2Scale = np.ascontiguousarray(x2Scale)
        if "cpu" not in self.scene:
            x1 = func.pack_int4(x1) if self.x1_dtype in ['fp4_e2m1', 'fp4_e1m2'] else x1
            x2 = func.pack_int4(x2) if self.x2_dtype in ['fp4_e2m1', 'fp4_e1m2'] else x2
            if self.x1_dtype in ["fp4_e2m1", "fp4_e1m2"]:
                x1 = x1.reshape(self.x1_shape[0], self.x1_shape[1] >> 1)
            if self.x2_dtype in ["fp4_e2m1", "fp4_e1m2"]:
                x2 = x2.reshape(self.x2_shape[0], self.x2_shape[1] >> 1)
            
        if self.save_flag:
            x1.tofile(os.path.join(self.input_path, 'x1.bin'))
            x2.tofile(os.path.join(self.input_path, 'x2.bin'))
            x1Scale.tofile(os.path.join(self.input_path, 'x1Scale.bin'))
            x2Scale.tofile(os.path.join(self.input_path, 'x2Scale.bin'))
        return x1, x2, x1Scale, x2Scale

    def cpu_mx_compute(self):
        x1 = self.trans_to_float32(self.array(self.x1), self.x1_dtype)
        x2 = self.trans_to_float32(self.array(self.x2), self.x2_dtype)
        pertoken_scale_mx = self.x1Scale.numpy().astype(np.float32)
        deq_scale_mx = self.x2Scale.numpy().astype(np.float32)
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

    def get_input_weight_old(self, weight_same=0):
        x1 = self.create_array("x1", self.x1_shape, self.x1_range, self.x1_dtype, self.input_path, self.seed)
        x2 = self.create_array("x2", self.x2_shape, self.x2_range, self.x2_dtype, self.input_path, self.weight_seed)
        x1Scale = self.create_array("x1Scale", self.x1Scale_shape, self.x1Scale_range, self.x1Scale_dtype,
                                    self.input_path, self.weight_seed)
        x2Scale = self.create_array("x2Scale", self.x2Scale_shape, self.x2Scale_range, self.x2Scale_dtype,
                                    self.input_path, self.weight_seed)
        return x1, x2, x1Scale, x2Scale

    def get_input_weight(self, weight_same=0):
        if self.is_bias:
            bias = self.create_array("bias", self.bias_shape, self.bias_range, self.bias_dtype, self.input_path,
                                    self.weight_seed)
            self.bias = self.tensor(bias)

        if self.is_mxfp:
            x1, x2, x1Scale, x2Scale = self.mx_generate_data()
        else:
            x1, x2, x1Scale, x2Scale = self.get_input_weight_old(weight_same)

        self.x1 = self.tensor(x1)
        self.x2 = self.tensor(x2)
        self.x1Scale = self.tensor(x1Scale)
        self.x2Scale = self.tensor(x2Scale)

        self.input = self.x1.t() if self.transposeX1 else self.x1
        self.weight = self.x2.t() if self.transposeX2 else self.x2
        if self.is_mxfp: # 小算子级联输入x2Scale shape 为 (k, n, 2)
            self.x2Scale_golden = self.x2Scale.transpose(0,1) if self.transposeX2 else self.x2Scale
        else:
            self.x2Scale_golden = self.x2Scale
        
        self.input_npu = self.input.contiguous()
        self.weight_npu = self.weight
        self.x2Scale_npu = self.x2Scale

        self.input_graph = self.x1
        self.weight_graph = self.x2
        self.x1Scale_graph = self.x1Scale
        self.x2Scale_graph = self.x2Scale

    def scale_broadcast(self):
        '''
        0: 不量化
        1: pertensor
        2: perchanenl
        3: pertoken
        4: pergroup
        5: perblock
        6: mx量化
        7: pertoken动态量化
        '''

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

        x1Scale = self.x1Scale
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

        return x1Scale, x2Scale

    def get_cpu(self):
        input = self.input.to(torch.float32)
        weight = self.weight.to(torch.float32)
        if self.is_mxfp:
            deQuant_mm_out = self.cpu_mx_compute()
        else:
            if self.is_bias:
                cpu_mm_out = torch.matmul(input, weight) + self.bias.to(torch.float32)
            else:
                cpu_mm_out = torch.matmul(input, weight)
            x1Scale, x2Scale = self.scale_broadcast()
            deQuant_mm_out = cpu_mm_out * x1Scale * x2Scale

        perOut = deQuant_mm_out.view(self.temp_shape).permute(1, 0, 2).contiguous()
        alltoall_out = self.alltoall_out
        dist.all_to_all_single(alltoall_out, perOut)
        torch.cpu.synchronize()

        self.output_cpu = alltoall_out.reshape(self.output_shape)
        self.save(self.output_cpu, self.output_path, self.output_cpu_name)

    def get_hccl_mm(self):
        # npu 小算子拼接
        mm_res = torch_npu.npu_quant_matmul(
            x1=self.input,
            x2=self.weight,
            pertoken_scale=self.x1Scale,
            scale=self.x2Scale_golden,
            offset=None,
            bias=self.bias,
            output_dtype=TORCH_DTYPE_MAP[self.output_dtype],
            x1_dtype=DTYPE_ENUM[self.x1_dtype],
            x2_dtype=DTYPE_ENUM[self.x2_dtype],
            pertoken_scale_dtype=DTYPE_ENUM[self.x1Scale_dtype],
            scale_dtype=DTYPE_ENUM[self.x2Scale_dtype],
            group_sizes=self.groupSizeDec,
            y_scale=None,
        )
        perOut = mm_res.view(self.temp_shape).permute(1, 0, 2).contiguous()
        dist.all_to_all_single(self.alltoall_out, perOut)
        self.output_hccl = self.alltoall_out.reshape(self.output_shape)
        return self.output_hccl, None, None, None, None

    def get_npu(self):
        self.output_npu = torch_npu.npu_quant_matmul_all_to_all(self.input_npu, self.weight_npu,
                                                                self.hcomm_info, self.world_size,
                                                                self.bias, self.x1Scale, self.x2Scale_npu,
                                                                common_scale=None,
                                                                x1_offset=None,
                                                                x2_offset=None,
                                                                x1_quant_mode=self.x1QuantMode,
                                                                x2_quant_mode=self.x2QuantMode,
                                                                common_quant_mode=self.commQuantMode,
                                                                group_sizes=self.groupSizeDec,
                                                                all2all_axes=None,
                                                                comm_quant_dtype=self.commQuantDtype,
                                                                x1_dtype=DTYPE_ENUM[self.x1_dtype],
                                                                x2_dtype=DTYPE_ENUM[self.x2_dtype],
                                                                x1_scale_dtype=DTYPE_ENUM[self.x1Scale_dtype],
                                                                x2_scale_dtype=DTYPE_ENUM[self.x2Scale_dtype],
                                                                output_scale_dtype=None,
                                                                comm_scale_dtype=None,
                                                                y_dtype=TORCH_DTYPE_MAP[self.output_dtype])
        return self.output_npu

    def get_graph(self, hcom_type="mc2"):
        if hcom_type == "mc2":
            model = QuantMatmulAlltoAllGraphModel()
            model = self.define_model(model, self.graph_type)
            
            out_graph = model(
                x1=self.input_graph,
                x2=self.weight_graph,
                hcom=self.hcomm_info,
                world_size=self.world_size,
                bias=self.bias,
                x1_scale=self.x1Scale_graph,
                x2_scale=self.x2Scale_graph,
                common_scale=None,
                x1_offset=None,
                x2_offset=None,
                x1_quant_mode=self.x1QuantMode,
                x2_quant_mode=self.x2QuantMode,
                common_quant_mode=self.commQuantMode,
                group_sizes=self.groupSizeDec,
                all2all_axes=None,
                comm_quant_dtype=self.commQuantDtype,
                x1_dtype=DTYPE_ENUM[self.x1_dtype],
                x2_dtype=DTYPE_ENUM[self.x2_dtype],
                x1_scale_dtype=DTYPE_ENUM[self.x1Scale_dtype],
                x2_scale_dtype=DTYPE_ENUM[self.x2Scale_dtype],
                output_scale_dtype=None,
                comm_scale_dtype=None,
                y_dtype=TORCH_DTYPE_MAP[self.output_dtype],
                transpose_x2=self.transposeX2,
                hcom_type=hcom_type
            )

            self.output_npu = out_graph
            return self.output_npu
        elif hcom_type == "hccl_mm":
            self.get_hccl_mm()
            return self.output_hccl
        else:
            print("In alltoall get_graph,wrong type..")

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
        command.append(f"--output_format {self.output_format}")
        command.append(f"--n_dev {self.world_size}")
        command.append(f"--loop_cnt {self.loop}")
        command.append(f"--bias_flag {self.is_bias}")
        command.append(f"--x1Scale_shape {','.join(map(str, self.x1Scale_shape))}")
        command.append(f"--x2Scale_shape {','.join(map(str, self.x2Scale_shape))}")
        command.append(f"--x1QuantMode {self.x1QuantMode}")
        command.append(f"--x2QuantMode {self.x2QuantMode}")
        command.append(f"--commQuantMode {self.commQuantMode}")
        command.append(f"--commQuantDtype {self.commQuantDtype}")
        command.append(f"--groupSize {self.groupSize}")
        command.append(f"--x1Scale_dtype {self.x1Scale_dtype}")
        command.append(f"--x2Scale_dtype {self.x2Scale_dtype}")
        command.append(f"--rankId {rankid} \&")
        return " ".join(command)
