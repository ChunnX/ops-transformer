import os
import torch
from .common import *
from op_class.aclnnQuantMatmulAllReduceV3 import QuantMatmulAllReduceV3
from torch.distributed import ReduceOp
from script.func import *
import math
from .registry import class_registry
from script import generate_mx_data


class QuantMatmulAllReduceV4GraphModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, hcom, bias, x3, dequant_scale, pertoken_scale, x1_golden, dequant_scale_golden,
                comm_quant_scale1, comm_quant_scale2, is_nz, is_x3, is_commquant, is_trans_b,
                x1_dtype, x2_dtype, x1scale_dtype, x2scale_dtype, output_dtype, group_sizes,
                diff_tensor, all_to_all_out, all_gather_out, world_size, use_alltoall, hcom_type, quant_type, comm_quant_mode):
        if is_nz:
            x2 = torch_npu.npu_format_cast(x2, 29)
        if is_trans_b: ## 图模式，转置场景(mxfp和perblock场景)，x2和dequantscale需要在图内转置
            x2 = x2.t()
            if quant_type in ["mxfp", "per_block"]: ## 规避mxfp场景的转置问题单
                dequant_scale = torch.transpose(dequant_scale, 0, 1)
                dequant_scale_golden = torch.transpose(dequant_scale_golden, 0, 1)
        if hcom_type == "mc2":
            output = torch_npu.npu_mm_all_reduce_base(
                x1=x1,
                x2=x2,
                hcom=hcom,
                reduce_op="sum",
                bias=bias,
                x3=x3,
                dequant_scale=dequant_scale,
                pertoken_scale=pertoken_scale,
                comm_quant_scale_1=comm_quant_scale1,
                comm_quant_scale_2=comm_quant_scale2,
                y_dtype=OUT_DTYPE_ENUM[output_dtype],
                x1_dtype=DTYPE_ENUM[x1_dtype],
                x2_dtype=DTYPE_ENUM[x2_dtype],
                pertoken_scale_dtype=DTYPE_ENUM[x1scale_dtype],
                dequant_scale_dtype=DTYPE_ENUM[x2scale_dtype],
                comm_quant_mode=comm_quant_mode,
                group_sizes=group_sizes
            )
        elif hcom_type == "hccl_mm":
            output = torch_npu.npu_quant_matmul(
                x1=x1_golden,
                x2=x2,
                scale=dequant_scale_golden,
                bias=bias,
                pertoken_scale=pertoken_scale,
                output_dtype=OUT_DTYPE_ENUM[output_dtype],
                x1_dtype=DTYPE_ENUM[x1_dtype],
                x2_dtype=DTYPE_ENUM[x2_dtype],
                pertoken_scale_dtype=DTYPE_ENUM[x1scale_dtype],
                scale_dtype=DTYPE_ENUM[x2scale_dtype],
                group_sizes=group_sizes,
                y_scale=None,
            )
            if is_x3:
                output = output.reshape(x3.shape)
                output = torch.add(output, x3)
            # 通信部分
            if not is_commquant:
                dist.all_reduce(output, op=ReduceOp.SUM)
            else:
                output = func.hcom_quant(output, comm_quant_scale1, comm_quant_scale2, diff_tensor, all_to_all_out,
                                         all_gather_out, world_size, use_alltoall)
        else:
            print("In allreduce  a8w8 get_graph,wrong type..")
        return output


@class_registry("aclnnQuantMatmulAllReduceV4")
class QuantMatmulAllReduceV4(QuantMatmulAllReduceV3):
    def __init__(self, case, rank, dist=None, hcomm_info=None):
        super().__init__(case, rank, dist, hcomm_info)
        self.quant_type = case["quant_type"]
        self.output_dtype = self.x3_dtype
        self.group_size = case["group_size"]
        self.group_sizes = self.unpack_group_size(self.group_size)
        self.comm_quant_mode = 1 if self.quant_type == "per_tile" else 0

    def generate_mx_fp_data(self):
        dtype_map = {
            "fp8_e5m2": "float8_e5m2",
            "fp8_e4m3fn": "float8_e4m3fn",
            "fp8_e8m0": "float8_e8m0",
            "fp4_e2m1": "float4_e2m1",
            "fp4_e1m2": "float4_e1m2",
        }
        x1_fp32 = self.create_array("x1", self.x1_shape, self.x1_range, "fp32", self.input_path, self.seed)
        x2_fp32 = self.create_array("x2", self.x2_shape, self.x2_range, "fp32", self.input_path, self.seed)
        x1, x1scale = generate_mx_data.mx_quantize(x1_fp32, dtype_map[self.x1_dtype], axis=-1, block_size=32, round_mode="rint")
        x2, x2scale = generate_mx_data.mx_quantize(x2_fp32, dtype_map[self.x2_dtype], axis=-1, block_size=32, round_mode="rint")
        if len(self.x1_shape) == 3: # x1 shape为3维时，取首batch轴的scale作为x1scale
            x1scale = x1scale[0].reshape(self.pertoken_scale_shape)
        if "cpu" in self.scene:
            x2scale = x2scale.reshape(self.dequant_scale_shape[0], -1)
            if self.is_trans_b:
                x2 = x2.transpose()
                x2scale = x2scale.transpose()
            self.x1 = x1.reshape(-1, self.x1_shape[-1])
            self.x2 = x2
            if len(self.x1_shape) == 3: # x1 shape为3维时，进行cpu计算时，需要对x1 scale进行扩维
                x1scale_cpu = np.broadcast_to(x1scale[np.newaxis, :, :], (self.x1_shape[0], *x1scale.shape))
                x1scale_cpu = x1scale_cpu.reshape(-1, *x1scale_cpu.shape[2:]) ## [B*S, ceilK, 2]
                # 对后两轴合轴
                self.pertoken_scale = self.tensor(x1scale_cpu).reshape(x1scale_cpu.shape[0], x1scale_cpu.shape[1]*x1scale_cpu.shape[2]) ## [B*S, ceilK*2] 
            else:
                self.pertoken_scale = self.tensor(x1scale).reshape(self.pertoken_scale_shape[0], -1)
            self.dequant_scale = self.tensor(x2scale)
        else:
            x1 = func.pack_int4(x1) if self.x1_dtype in ['fp4_e2m1', 'fp4_e1m2'] else x1
            x2 = func.pack_int4(x2) if self.x2_dtype in ['fp4_e2m1', 'fp4_e1m2'] else x2
            # self.save(x1, self.input_path, "x1.bin")
            # self.save(x2, self.input_path, "x2.bin")
            # self.save(x1scale, self.input_path, "x1scale.bin")
            # self.save(x2scale, self.input_path, "x2scale.bin")
            self.x1 = self.tensor(x1)
            self.x2 = self.tensor(x2)
            # fp4只支持x2转置， 要求K轴被2整除
            if self.x1_dtype in ["fp4_e2m1", "fp4_e1m2"]:
                if len(self.x1_shape) == 2:
                    self.x1 = self.x1.reshape(self.x1_shape[0], self.x1_shape[1] >> 1)
                else:
                    self.x1 = self.x1.reshape(self.x1_shape[0], self.x1_shape[1], self.x1_shape[2] >> 1)
            if self.x2_dtype in ["fp4_e2m1", "fp4_e1m2"]:
                self.x2 = self.x2.reshape(self.x2_shape[0], self.x2_shape[1] >> 1)
            self.pertoken_scale = self.tensor(x1scale).reshape(self.pertoken_scale_shape)
            self.dequant_scale = self.tensor(x2scale).reshape(self.dequant_scale_shape)
            if self.is_trans_b and "graph" not in self.scene:
                self.x2 = self.x2.t()
                """以下为修复代码"""
                if len(self.dequant_scale_shape) == 3:
                    self.dequant_scale = torch.transpose(self.dequant_scale, 0, 1)
                else:
                    self.dequant_scale = self.dequant_scale.t()
            self.dequant_scale_golden = self.dequant_scale

    def get_tmp_tensor(self):
        QuantMatmulAllReduceV3.get_tmp_tensor(self)
        if self.quant_type in ["per_block", "mxfp"]:
            self.x1_golden = self.x1
            self.dequant_scale_golden = self.dequant_scale
        self.x2_golden = self.x2

    def get_input_weight(self, weight_same=0):
        if self.quant_type == "mxfp":
            self.generate_mx_fp_data()
        else:
            QuantMatmulAllReduceV3.get_input_weight(self, weight_same)

    def get_tmp_tensor_memory(self):
        return 0

    # npu 小算子级联
    def get_hccl_mm(self):
        mm_output = torch_npu.npu_quant_matmul(
            x1=self.x1_golden,
            x2=self.x2_golden,
            scale=self.dequant_scale_golden,
            bias=self.bias,
            pertoken_scale=self.pertoken_scale,
            output_dtype=OUT_DTYPE_ENUM[self.output_dtype],
            x1_dtype=DTYPE_ENUM[self.x1_dtype],
            x2_dtype=DTYPE_ENUM[self.x2_dtype],
            pertoken_scale_dtype=DTYPE_ENUM[self.pertoken_scale_dtype],
            scale_dtype=DTYPE_ENUM[self.dequant_scale_dtype],
            group_sizes=self.group_sizes,
            y_scale=None,
        )
        if self.is_x3:
            mm_output = mm_output.reshape(self.x3.shape)
            mm_output = torch.add(mm_output, self.x3)
        # 通信部分
        if self.comm_quant_mode:
            # dist.all_reduce(mm_output, op=ReduceOp.SUM)
            mm_output = self.hcom_quant_per_tile(mm_output)
        elif not self.is_commquant:
            dist.all_reduce(mm_output, op=ReduceOp.SUM)
        else:
            mm_output = func.hcom_quant(mm_output, self.comm_quant_scale1, self.comm_quant_scale2, self.diff_tensor,
                                        self.all_to_all_out, self.all_gather_out, self.world_size, self.use_alltoall)
        self.output_golden = mm_output
        #self.save(self.output_golden, self.output_path, "output_golden")
        return self.output_golden

    def get_npu(self):
        self.output_npu = torch_npu.npu_mm_all_reduce_base(
            x1=self.x1,
            x2=self.x2,
            hcom=self.hcomm_info,
            reduce_op="sum",
            bias=self.bias,
            x3=self.x3,
            dequant_scale=self.dequant_scale,
            pertoken_scale=self.pertoken_scale,
            comm_quant_scale_1=self.comm_quant_scale1,
            comm_quant_scale_2=self.comm_quant_scale2,
            y_dtype=OUT_DTYPE_ENUM[self.output_dtype],
            x1_dtype=DTYPE_ENUM[self.x1_dtype],
            x2_dtype=DTYPE_ENUM[self.x2_dtype],
            pertoken_scale_dtype=DTYPE_ENUM[self.pertoken_scale_dtype],
            dequant_scale_dtype=DTYPE_ENUM[self.dequant_scale_dtype],
            comm_quant_mode=self.comm_quant_mode,
            group_sizes=self.group_sizes,
        )
        #self.save(self.output_npu, self.output_path, "output_npu")
        return self.output_npu

    def get_graph(self, hcom_type="mc2"):
        model = QuantMatmulAllReduceV4GraphModel()
        model = self.define_model(model, self.graph_type)
        if self.quant_type == "per_tile" and hcom_type == "hccl_mm":
            print(f"per tile量化场景，小算子级联走单算子模式")
            if self.is_trans_b:  ## 图模式，转置场景(mxfp和perblock场景)，x2和dequantscale需要在图内转置
                self.x2_golden = self.x2_golden.t()
            output = self.get_hccl_mm()
            self.output_golden = output
        else:
            output = model(
                x1=self.x1,
                x2=self.x2,
                hcom=self.hcomm_info,
                bias=self.bias,
                x3=self.x3,
                dequant_scale=self.dequant_scale,
                pertoken_scale=self.pertoken_scale,
                x1_golden=self.x1_golden,
                dequant_scale_golden=self.dequant_scale_golden,
                comm_quant_scale1=self.comm_quant_scale1,
                comm_quant_scale2=self.comm_quant_scale2,
                is_nz=self.is_nz,
                is_x3=self.is_x3,
                is_commquant=self.is_commquant,
                is_trans_b=self.is_trans_b,
                x1_dtype=self.x1_dtype,
                x2_dtype=self.x2_dtype,
                x1scale_dtype=self.pertoken_scale_dtype,
                x2scale_dtype=self.dequant_scale_dtype,
                output_dtype=self.output_dtype,
                group_sizes=self.group_sizes,
                diff_tensor=self.diff_tensor,
                all_to_all_out=self.all_to_all_out,
                all_gather_out=self.all_gather_out,
                world_size=self.world_size,
                use_alltoall=self.use_alltoall,
                hcom_type=hcom_type,
                quant_type = self.quant_type,
                comm_quant_mode = self.comm_quant_mode
            )
            if hcom_type == "mc2":
                self.output_npu = output
            elif hcom_type == "hccl_mm":
                self.output_golden = output
            else:
                print("In allreduce a8w8 get_graph,wrong type..")
        return output

    def get_cpu(self):
        # 场景1/2/3 继承v1/v2/v3
        if self.x1_dtype == "int8":
            x1 = self.x1.to(torch.int32)
            x2 = self.x2.to(torch.int32)
            dequant_scale = self.dequant_scale.to(torch.float32)
            output = torch.matmul(x1, x2)  # torch.int32
            if self.is_bias:
                output += self.bias
            if self.is_pertoken:
                output_origial_shape = output.shape
                if len(output_origial_shape) == 3:
                    output = output.reshape(-1, output.shape[-1])
                pertoekn_scale = torch.unsqueeze(self.pertoken_scale, dim=1)
                output = (output * pertoekn_scale).reshape(output_origial_shape)
            output = torch.mul(output, dequant_scale)
        else:
            if self.quant_type != "mxfp":
                x1 = self.x1.to(torch.float32)
                x2 = self.x2.to(torch.float32)
                # x1 = torch.from_numpy(self.trans_to_float32(self.array(self.x1), self.x1_dtype))
                # x2 = torch.from_numpy(self.trans_to_float32(self.array(self.x2), self.x2_dtype))
            if self.quant_type == "per_block":  # 场景5 per_block
                x1scale = self.pertoken_scale.to(torch.float32)
                x2scale = self.dequant_scale.to(torch.float32)
                output = func.per_block_cpu_compute(self.group_size, x1, x2, x1scale, x2scale)
            elif self.quant_type == "mxfp":  # 场景6 mxfp
                x1 = self.trans_to_float32(self.array(self.x1), self.x1_dtype)
                x2 = self.trans_to_float32(self.array(self.x2), self.x2_dtype)
                x1scale = self.pertoken_scale.to(torch.float32).numpy()
                x2scale = self.dequant_scale.to(torch.float32).numpy()
                output = self.mxfp_cpu_compute(x1, x2, x1scale, x2scale)
                if self.is_bias:
                    bias = self.bias.to(torch.float32)
                    output += bias
            elif self.quant_type == "per_tile":
                dequant_scale = self.dequant_scale.to(torch.float32)
                output = torch.matmul(x1, x2)
                if self.is_bias:
                    output += self.bias
                if self.is_pertoken:
                    output_origial_shape = output.shape
                    if len(output_origial_shape) == 3:
                        output = output.reshape(-1, output.shape[-1])
                    pertoekn_scale = torch.unsqueeze(self.pertoken_scale, dim=1)
                    output = (output * pertoekn_scale).reshape(output_origial_shape)
                output = torch.mul(output, dequant_scale)
            else:  # 场景4 v2接口支持新数据类型
                dequant_scale = self.dequant_scale.to(torch.float32)
                output = torch.matmul(x1, x2)
                if self.is_bias:
                    output += self.bias
                if self.is_pertoken:
                    output_origial_shape = output.shape
                    if len(output_origial_shape) == 3:
                        output = output.reshape(-1, output.shape[-1])
                    pertoekn_scale = torch.unsqueeze(self.pertoken_scale, dim=1)
                    output = (output * pertoekn_scale).reshape(output_origial_shape)
                output = torch.mul(output, dequant_scale)
        if self.is_x3:
            output = output.reshape(self.x3.shape)
            x3 = self.x3.to(torch.float32)
            output += x3
        if not self.is_commquant and self.quant_type != "per_tile":
            dist.all_reduce(output, op=ReduceOp.SUM)
        elif self.quant_type == "per_tile":
            output = self.hcom_quant_per_tile(output)
        else:
            output = self.hcom_quant_cpu(output)
        self.save(output, self.output_path, self.output_cpu_name)

    def quant_cpu(self, mm_output, dtype):
        device = torch.device("cpu") if "cpu" in self.scene else torch.device("npu")
        m = mm_output.shape[0]
        n = mm_output.shape[1]
        max_map = {
            "fp8_e4m3fn": 448.0,
            "fp8_e5m2": 57344.0,
        }
        max_value = max_map[dtype]
        abs_out = torch.abs(mm_output).to(device)
        out_tiles = abs_out.reshape(-1, 128)
        abs_max = torch.max(out_tiles, dim=1).values
        abs_max = abs_max.masked_fill(abs_max == 0, 1)
        max_val_tensor = torch.full(abs_max.shape, max_value).to(device)
        scales = max_val_tensor / abs_max
        scales = scales.masked_fill(scales == 0, 1).reshape(m, -1)
        broadcast_scales = scales.repeat_interleave(128, dim=-1).reshape(m, n)
        quant_out = mm_output * broadcast_scales
        quant_out = quant_out.to(TORCH_DTYPE_MAP[dtype])
        return quant_out, scales

    def hcom_quant_per_tile(self, mm_output):
        device = torch.device("cpu") if "cpu" in self.scene else torch.device("npu")
        dst_type_map = {
            "fp8_e4m3fn": torch.float8_e4m3fn,
            "fp8_e5m2": torch.float8_e5m2,
        }
        mm_output_shape = mm_output.shape
        mm_output = mm_output.reshape(-1, mm_output_shape[-1])
        m = mm_output.shape[0]
        n = mm_output.shape[1]
        pad_n = torch.zeros([m, math.ceil(n / 128) * 128 - n], dtype=torch.float32).to(device)
        padded_output = torch.concat([mm_output, pad_n], dim=1)
        if "cpu" in self.scene:
            quant_output, scale = self.quant_cpu(padded_output, self.x1_dtype)  # 量化过程
        else:
            quant_output, scale = torch_npu.npu_dynamic_block_quant(
                x=padded_output.to(torch.float16),
                round_mode="rint",
                dst_type=dst_type_map[self.x1_dtype]
            )
        quant_output = quant_output[0:m, 0:n]
        quant_output = torch.concat([quant_output.to(torch.float32), scale], dim=1)  # 拼接量化结果与scale
        n_with_scale = quant_output.shape[-1]

        pad_m = torch.zeros([math.ceil(m / self.world_size) * self.world_size - m, n_with_scale], dtype=torch.float32).to(device)
        if pad_m.shape[0] != 0:
            padded_output = torch.concat([quant_output, pad_m], dim=0).reshape(self.world_size, -1, n_with_scale)
        else:
            padded_output = quant_output.reshape(self.world_size, -1, n_with_scale)

        all_to_all_out = torch.zeros_like(padded_output).to(device)
        dist.all_to_all_single(all_to_all_out, padded_output)

        dequant_output = all_to_all_out.to(torch.float32)[:, :, :n]

        scales = all_to_all_out[:, :, n:]
        broadcast_scales = scales.repeat_interleave(128, dim=-1)[:, :, :n]
        if "cpu" in self.scene:
            dequant_output /= broadcast_scales  # 反量化过程
        else: # npu 小算子级联
            dequant_output *= broadcast_scales

        reduce_sum_out = torch.sum(dequant_output, dim=0, keepdim=True)[0]

        pad_n = torch.zeros([reduce_sum_out.shape[0], math.ceil(n / 128) * 128 - n], dtype=torch.float32).to(device)
        padded_output2 = torch.concat([reduce_sum_out, pad_n], dim=1)
        if "cpu" in self.scene:
            quant_output2, scale2 = self.quant_cpu(padded_output2, self.x1_dtype)  # 量化过程
        else:
            quant_output2, scale2 = torch_npu.npu_dynamic_block_quant(
                x=padded_output2.to(torch.float16),
                round_mode="rint",
                dst_type=dst_type_map[self.x1_dtype]
            )
        quant_output2 = quant_output2[0:m, 0:n]
        padded_output2 = torch.concat([quant_output2.to(torch.float32), scale2], dim=1)

        all_gather_list = [torch.zeros_like(padded_output2).to(device) for _ in range(self.world_size)]
        dist.all_gather(all_gather_list, padded_output2)
        all_gather_out = torch.cat(all_gather_list, dim=0).reshape(-1, n_with_scale)

        dequant_output2 = all_gather_out[:, :n]
        scale2 = all_gather_out[:, n:]

        broadcast_scale2 = scale2.repeat_interleave(128, dim=-1)[:, :n]
        dequant_output2 /= broadcast_scale2  # 反量化过程

        output = dequant_output2.reshape(-1, dequant_output2.shape[-1])[0:m, 0:n].reshape(mm_output_shape)
        if "cpu" in self.scene:
            output = output.to(torch.float32)
        else:
            output = output.to(TORCH_DTYPE_MAP[self.output_dtype])
        return output

    def process_aclnn(self):
        pass

    def get_memory_performance(self, type, cal_mask, perf_type, soc="910B"):
        case_name = self.case['case_name']
        case_path = os.path.join(os.getcwd(), 'Profiling_Result_' + type + case_name)
        performance = 0.0
        golden_hccl_max = 0.0
        golden_mm_max = 0.0
        result_list = []
        memory = 0
        for dirpath, _, filenames in os.walk(case_path):
            for filename in filenames:
                if filename.startswith("memory_record") and "ASCEND_PROFILER_OUTPUT" in dirpath:
                    memory_file = filename
                    with open(os.path.join(dirpath, memory_file)) as f1:
                        memory_list = []
                        csvreader = csv.reader(f1)
                        for row in list(csvreader)[1:]:
                            memory_list.append(float(row[2]))
                        memory_use = max(memory_list) - min(memory_list)
                    if memory < memory_use:
                        memory = memory_use
                if filename.startswith("op_summary"):
                    summary_file = filename
                    with open(os.path.join(dirpath, summary_file)) as f2:
                        csvreader = csv.reader(f2)
                        headers = next(csvreader)
                        t = headers.index('Task Duration(us)')
                        task_type = headers.index('Task Type')
                        op_type = headers.index('OP Type')
                        hccl_list, matmul_list, mc2_list, aicpu_list = [], [], [], []
                        for row in list(csvreader)[:]:
                            if row[op_type] == "Exp":  # 跳过插入的算子
                                continue
                            if cal_mask:
                                if perf_type == 'golden':
                                    if row[task_type] in ["HCCL", "COMMUNICATION"]:
                                        hccl_list.append(float(row[t]))
                                    # 910D 大数据量下，需要统计hccl通信算子的AICPU编排耗时
                                    elif row[task_type] == "AI_CPU" and soc == '910D':
                                        aicpu_list.append(float(row[t]))
                                    elif row[task_type] in ['AI_VECTOR_CORE', 'AI_CORE', 'MIX_AIC']:
                                        matmul_list.append(float(row[t]))
                                    else: # matmulAllreduce pertile场景，perf_type为FUSION
                                        hccl_list.append(float(row[t]))
                                else:
                                    if row[task_type] != 'AI_CPU':
                                        mc2_list.append(float(row[t]))
                            else:
                                if row[task_type] != 'AI_CPU':
                                    mc2_list.append(float(row[t]))
                    if all([hccl_list, aicpu_list]):
                        # 合并aicpu + hccl,910D大数据量场景下
                        hccl_list = [i + j for i, j in zip(hccl_list, aicpu_list)]
                    if type == "hccl_mm_" and cal_mask:
                        # 写入hccl耗时/计算耗时
                        avg_hccl = get_avg_perf(hccl_list)
                        avg_matmul = get_avg_perf(matmul_list)
                        if avg_hccl + avg_matmul > performance:
                            performance = avg_hccl + avg_matmul
                            golden_hccl_max = avg_hccl
                            golden_mm_max = avg_matmul
                    else:
                        avg_mc2 = get_avg_perf(mc2_list)
                        if avg_mc2 > performance:
                            performance = avg_mc2
        result_list.extend([[golden_hccl_max, golden_mm_max], memory, performance])
        return result_list

    def get_performace(self, perf_type='npu', soc='910B'):
        if self.quant_type == "per_tile": # pertile场景，需要先计算出小算子级联精度结果，性能与16bit allreduce比较
            self.get_hccl_mm()
        tensor = torch.ones(10000, 10000).to(torch.float16).npu() ## 重型算子计算需要
        if perf_type == 'npu':
            path_tmp = 'mc2_'
        elif perf_type == 'golden':
            path_tmp = 'hccl_mm_'
        experimental_config, profiling_result = func.get_profiling_config(self.case['case_name'], path_tmp)
        with torch_npu.profiler.profile(
                schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_result),
                profile_memory=True,
                with_stack=False,
                with_flops=False,
                with_modules=False,
                experimental_config=experimental_config,
        ) as prof:
            for i in range(10):
                if i == 1: ## 防止host bound现象，插入重型算子
                    for _ in range(50):
                        torch.exp(tensor)
                if perf_type == 'npu':
                    self.get_npu()
                elif perf_type == 'golden':
                    if self.quant_type == "per_tile": # 与非低bit通信的mmAllreduce相比
                        out = torch_npu.npu_mm_all_reduce_base(
                            x1=self.x1,
                            x2=self.x2,
                            hcom=self.hcomm_info,
                            reduce_op="sum",
                            bias=self.bias,
                            x3=self.x3,
                            dequant_scale=self.dequant_scale,
                            pertoken_scale=self.pertoken_scale,
                            comm_quant_scale_1=self.comm_quant_scale1,
                            comm_quant_scale_2=self.comm_quant_scale2,
                            y_dtype=OUT_DTYPE_ENUM[self.output_dtype],
                            x1_dtype=DTYPE_ENUM[self.x1_dtype],
                            x2_dtype=DTYPE_ENUM[self.x2_dtype],
                            pertoken_scale_dtype=DTYPE_ENUM[self.pertoken_scale_dtype],
                            dequant_scale_dtype=DTYPE_ENUM[self.dequant_scale_dtype],
                            comm_quant_mode=0, # 非低bit mmallreduce
                            group_sizes=self.group_sizes,
                        )
                    else:
                        self.get_hccl_mm()
        if self.rank == self.case['save_rank']:
            op = self.case['op_name']
            csv_file = f'{path_tmp}{op}_pro_result.csv'
            ## soc为910D时，或是两个双通信域moe算子时，需要计算性能掩盖率
            cal_mask = True
            result_list, memory, performance = self.get_memory_performance(path_tmp, cal_mask, perf_type, soc)
            if perf_type == "golden":
                diff = func.get_memory_diff(self.case)
                tmp_tensor_memory = self.get_tmp_tensor_memory()
                print(
                    f'Profiling采集的内存为：{memory}；MC2内存计算需要累加值为： {diff}; tmp_tensor_memory需要累加值为：{tmp_tensor_memory}')
                golden_memory = memory + diff + tmp_tensor_memory
                if cal_mask:
                    hccl_perf = result_list[0]
                    mm_perf = result_list[1]
                    func.write_csv(csv_file, [self.case_name, self.rank, hccl_perf, mm_perf, golden_memory,
                                              performance])
                    print(f'[INFO] 小算子级联 内存为：{golden_memory} M, 通信性能为：{hccl_perf} us, 计算性能为：{mm_perf} us, 总性能： {performance}us \n')
                else:
                    func.write_csv(csv_file, [self.case_name, self.rank, golden_memory, performance])
                    print(f'[INFO] 小算子hccl+mm级联 内存为：{golden_memory} M，性能为：{performance} us \n')
            else:
                func.write_csv(csv_file, [self.case_name, self.rank, memory, performance])
                print(f'[INFO] MC2融合算子 内存为：{memory} M，性能为：{performance} us \n')
