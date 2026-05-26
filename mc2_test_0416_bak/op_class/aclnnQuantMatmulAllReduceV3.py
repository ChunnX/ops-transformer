import os
import torch
from .common import *
from op_class.aclnnQuantMatmulAllReduceV2 import QuantMatmulAllReduceV2
from torch.distributed import ReduceOp
from script.func import *
import math
from .registry import class_registry


class QuantMatmulAllReduceV3GraphModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2, hcom, bias, x3, dequant_scale, pertoken_scale, x1_golden, dequant_scale_golden, is_nz, is_x3,
                is_trans_b, is_commquant, comm_quant_scale_1, comm_quant_scale_2, diff_tensor, all_to_all_out, all_gather_out,
                world_size, use_alltoall, hcom_type):
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
                pertoken_scale=pertoken_scale,
                comm_quant_scale_1=comm_quant_scale_1,
                comm_quant_scale_2=comm_quant_scale_2
            )
        elif hcom_type == "hccl_mm":
            if dequant_scale.dtype in [torch.int64, torch.float32]:
                mm_output_dtype = torch.float16
            else:
                mm_output_dtype = torch.bfloat16
            output = torch_npu.npu_quant_matmul(
                x1=x1_golden,
                x2=x2,
                scale=dequant_scale_golden,
                bias=bias,
                pertoken_scale=pertoken_scale,
                output_dtype=mm_output_dtype
            )
            if is_x3:
                output = output.reshape(x3.shape)
                output = torch.add(output, x3)
            # 通信部分
            if not is_commquant:
                dist.all_reduce(output, op=ReduceOp.SUM)
            else:
                output = func.hcom_quant(output, comm_quant_scale_1, comm_quant_scale_2, diff_tensor, all_to_all_out,
                                         all_gather_out, world_size, use_alltoall)
        else:
            print("In allreduce  a8w8 get_graph,wrong type..")
        return output


@class_registry("aclnnQuantMatmulAllReduceV3")
class QuantMatmulAllReduceV3(QuantMatmulAllReduceV2):
    def __init__(self, case, rank, dist=None, hcomm_info=None):
        super().__init__(case, rank, dist, hcomm_info)
        self.is_commquant = case["is_commquant"]
        if self.is_commquant:
            self.comm_quant_scale1_dtype = case['commquant_scale1_dtype']
            self.comm_quant_scale2_dtype = case['commquant_scale2_dtype']
            self.comm_quant_scale1_shape = case['commquant_scale1_shape']
            self.comm_quant_scale2_shape = case['commquant_scale2_shape']
            self.output_dtype = self.comm_quant_scale1_dtype
        self.use_alltoall = 1
        self.comm_quant_scale1 = None
        self.comm_quant_scale2 = None
        self.comm_quant_scale1_high_name = "fp32_comm_quant_scale_1_0.bin"
        self.comm_quant_scale1_low_name = "comm_quant_scale_1.bin"
        self.comm_quant_scale2_high_name = "fp32_comm_quant_scale_2_0.bin"
        self.comm_quant_scale2_low_name = "comm_quant_scale_2.bin"
        # 小算子级联需要的tensor
        self.m = self.x1_shape[0] * self.x1_shape[1] if len(self.x1_shape) == 3 else self.x1_shape[0]
        self.n = self.bias_shape[0]
        final_m = math.ceil(self.m / self.world_size) * self.world_size
        self.diff_tensor_shape = [int(final_m - self.m), int(self.n)]
        self.all_to_all_out_shape = [int(self.world_size), int(final_m / self.world_size), int(self.n)]
        self.all_gather_out_shape = self.all_to_all_out_shape
        self.diff_tensor = None
        self.all_to_all_out = None
        self.all_gather_out = None
        self.quant_type = "v3"

    def get_input_weight(self, weight_same=0):
        QuantMatmulAllReduceV2.get_input_weight(self, weight_same)

    def get_tmp_tensor(self):
        QuantMatmulAllReduceV2.get_tmp_tensor(self)
        # 获取comm_quant_scale (由cpu预先计算得出)
        comm_quant_scale1_path = os.path.join(self.save_path, self.comm_quant_scale1_high_name)
        comm_quant_scale2_path = os.path.join(self.save_path, self.comm_quant_scale2_high_name)
        if self.is_commquant:
            if os.path.exists(comm_quant_scale1_path):
                comm_scale1_dtype = TORCH_DTYPE_MAP[self.comm_quant_scale1_dtype]
                comm_quant_scale1 = np.fromfile(comm_quant_scale1_path, dtype=np.float32).reshape(self.comm_quant_scale1_shape)
                self.comm_quant_scale1 = torch.from_numpy(comm_quant_scale1).to(comm_scale1_dtype).to(self.device)
            if os.path.exists(comm_quant_scale2_path):
                comm_scale2_dtype = TORCH_DTYPE_MAP[self.comm_quant_scale2_dtype]
                comm_quant_scale2 = np.fromfile(comm_quant_scale2_path, dtype=np.float32).reshape(self.comm_quant_scale2_shape)
                self.comm_quant_scale2 = torch.from_numpy(comm_quant_scale2).to(comm_scale2_dtype).to(self.device)

            self.diff_tensor = torch.zeros(self.diff_tensor_shape, dtype=torch.int8).to(self.device)
            self.all_to_all_out = torch.zeros(self.all_to_all_out_shape, dtype=torch.int8).to(self.device)
            self.all_gather_out = torch.zeros(self.all_gather_out_shape, dtype=torch.int8).to(self.device)

    def get_tmp_tensor_memory(self):
        diff_tensor_size = self.diff_tensor_shape[0] * self.diff_tensor_shape[1] * 2 / 1024 / 1024
        alltoall_allgather_size = self.all_gather_out_shape[0] * self.all_gather_out_shape[1] * \
                                  self.all_gather_out_shape[2] * 2 / 1024 / 1024
        return diff_tensor_size + 2 * alltoall_allgather_size

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
            comm_quant_scale_2=self.comm_quant_scale2
        )

    def get_hccl_mm(self):
        # 计算部分
        if self.dequant_scale_golden.dtype in [torch.int64, torch.float32]:
            mm_output_dtype = torch.float16
        else:
            mm_output_dtype = torch.bfloat16
        mm_output = torch_npu.npu_quant_matmul(
            x1=self.x1_golden,
            x2=self.x2,
            scale=self.dequant_scale_golden,
            bias=self.bias,
            pertoken_scale=self.pertoken_scale,
            output_dtype=mm_output_dtype
        )
        if self.is_x3:
            mm_output = mm_output.reshape(self.x3.shape)
            mm_output = torch.add(mm_output, self.x3)
        # 通信部分
        if not self.is_commquant:
            dist.all_reduce(mm_output, op=ReduceOp.SUM)
        else:
            mm_output = func.hcom_quant(mm_output, self.comm_quant_scale1, self.comm_quant_scale2, self.diff_tensor,
                                        self.all_to_all_out, self.all_gather_out, self.world_size, self.use_alltoall)
        self.output_golden = mm_output
        return self.output_golden

    # cpu golden：allreduce(dequant_scale*pertoken_scale*(x1@x2+bias)+x3)
    def get_cpu(self):
        x1 = self.x1.to(torch.int32)
        x2 = self.x2.to(torch.int32)
        dequant_scale = self.dequant_scale.to(torch.float32)
        output = torch.matmul(x1, x2)
        if self.is_bias:
            output = torch.add(output, self.bias)
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
            output = torch.add(output, x3)
        if not self.is_commquant:
            dist.all_reduce(output, op=ReduceOp.SUM)
        else:
            output = self.hcom_quant_cpu(output)
        self.save(output, self.output_path, self.output_cpu_name)

    def get_graph(self, hcom_type="mc2"):
        model = QuantMatmulAllReduceV3GraphModel()
        model = self.define_model(model, self.graph_type)
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
            is_nz=self.is_nz,
            is_x3=self.is_x3,
            is_trans_b=self.is_trans_b,
            is_commquant=self.is_commquant,
            comm_quant_scale_1=self.comm_quant_scale1,
            comm_quant_scale_2=self.comm_quant_scale2,
            diff_tensor=self.diff_tensor,
            all_to_all_out=self.all_to_all_out,
            all_gather_out=self.all_gather_out,
            world_size=self.world_size,
            use_alltoall=self.use_alltoall,
            hcom_type=hcom_type,
        )
        if hcom_type == "mc2":
            self.output_npu = output
        elif hcom_type == "hccl_mm":
            self.output_golden = output
        else:
            print("In allreduce a8w8 get_graph,wrong type..")
        return output

    def process_aclnn(self):
        QuantMatmulAllReduceV2.process_aclnn(self)

    def get_command(self, binary, rank=0, is_david=0):
        case_format = ['ND', 'ND', 'ND'] if not self.is_nz else ['ND', 'NZ', 'ND']
        input_tensor_dtype = ','.join(map(str, self.case['input_tensor_dtype']))
        input_tensor_format = ','.join(map(str, case_format))
        input_tensor_dtype = input_tensor_dtype + ',' + self.dequant_scale_dtype
        input_tensor_format = input_tensor_format + ',ND'
        input_tensor_format = input_tensor_format + ',ND'
        input_tensor_dtype = input_tensor_dtype + ',' + self.pertoken_scale_dtype
        if self.is_x3:
            input_tensor_dtype += ',' + self.case['x3_dtype']
            input_tensor_format += ',ND'
        if self.is_commquant:
            input_tensor_dtype = input_tensor_dtype + ',' + self.case['commquant_scale1_dtype']
            input_tensor_dtype = input_tensor_dtype + ',' + self.case['commquant_scale2_dtype']
            input_tensor_format = input_tensor_format + ',ND,ND'
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
        command.append(f"--pertoken_flag {self.is_pertoken}")
        command.append(f"--pertoken_shape {','.join(map(str, self.pertoken_scale_shape))}")
        command.append(f"--commquant_flag {self.is_commquant}")
        command.append(f"--commquant_1_shape {','.join(map(str, self.case['commquant_scale1_shape']))}")
        command.append(f"--commquant_2_shape {','.join(map(str, self.case['commquant_scale2_shape']))}")
        if self.is_x3:
            command.append(f"--x3_flag {int(self.is_x3)}")
            command.append(f"--x3_shape {','.join(map(str, self.x3_shape))}")
        command.append(f"--bin_path {bin_path}")
        command.append(f"--is_quant {is_quant}")
        if is_david:
            command.append(f"--rankId {rank} \&")
        aclnn_command = " ".join(command)
        return aclnn_command

    def gen_comm_quant_scale(self, tensor, scale_name):
        tensor = torch.abs(tensor)
        local_scale = torch.div(torch.max(tensor, dim=0)[0], 127).to(torch.float32)
        if torch.all(local_scale == 0):
            local_scale = torch.ones_like(local_scale)
        print(f"rank:{self.rank},当前卡计算出系数:{local_scale}")
        all_gather_list = [torch.zeros_like(local_scale) for _ in range(self.world_size)]
        dist.all_gather(all_gather_list, local_scale)
        comm_quant_scale = torch.cat(all_gather_list, dim=0).reshape(self.world_size, -1)
        comm_quant_scale = comm_quant_scale.max(dim=0)[0]
        if scale_name == "scale1":
            self.save(comm_quant_scale, self.save_path, self.comm_quant_scale1_high_name)
            self.save(comm_quant_scale.to(TORCH_DTYPE_MAP[self.output_dtype]), self.save_path, self.comm_quant_scale1_low_name)
        else:
            self.save(comm_quant_scale, self.save_path, self.comm_quant_scale2_high_name)
            self.save(comm_quant_scale.to(TORCH_DTYPE_MAP[self.output_dtype]), self.save_path, self.comm_quant_scale2_low_name)
        return comm_quant_scale

    def hcom_quant_cpu(self, output):
        ori_shape = output.shape
        n = self.n
        m = self.m
        output = output.reshape(m, n)
        comm_quant_scale1 = self.comm_quant_scale1.to(torch.float32) if self.comm_quant_scale1 is not None else None
        comm_quant_scale2 = self.comm_quant_scale2.to(torch.float32) if self.comm_quant_scale2 is not None else None
        if comm_quant_scale1 is None or comm_quant_scale2 is None:
            comm_quant_scale1 = self.gen_comm_quant_scale(output, "scale1")
        # step1:quant操作 output / comm_quant_scale1
        output = output / comm_quant_scale1
        # step2：计算需padding的tensor，并concat拼接成可被all_to_all整除的tensor
        diff_tensor_shape = [math.ceil(m / self.world_size) * self.world_size - m, n]
        diff_tensor = torch.zeros(diff_tensor_shape)
        new_output = torch.concat([output, diff_tensor], dim=0)
        # step3: 拼接完的tensor reshape成3维：[world_size, -1, N]
        quant_1_output = new_output.reshape(self.world_size, -1, n)
        # 转int8
        quant_1_int8_output = torch.clamp(torch.round(quant_1_output), -128, 127).to(torch.int8)
        if self.use_alltoall:
            # step4: 调用all_to_all接口,
            all_to_all_out = torch.zeros_like(quant_1_int8_output)
            dist.all_to_all_single(all_to_all_out, quant_1_int8_output)
            # step5: all_to_all结果做reduce sum,shape：[1,-1,N], 转fp32
            sum_out = torch.sum(all_to_all_out, dim=0, keepdim=True).to(torch.float32)
        else:
            sum_out = torch.sum(quant_1_int8_output, dim=0, keepdim=True)
        # step6: sum结果做comm_quant_1反量化
        sum_dequant_1_out = sum_out * comm_quant_scale1
        sum_dequant_1_out = sum_dequant_1_out.reshape(-1, sum_dequant_1_out.shape[-1])
        # 获取comm_quant_scale2
        if comm_quant_scale2 is None:
            comm_quant_scale2 = self.gen_comm_quant_scale(sum_dequant_1_out, "scale2")
        # step7: sum后对1反量化的结果,做comm_quant_2量化，并转int8送进allgather
        quant_2_out = sum_dequant_1_out / comm_quant_scale2
        quant_2_int8_out = torch.clamp(torch.round(quant_2_out), -128, 127).to(torch.int8)

        all_gather_list = [torch.zeros_like(quant_2_int8_out) for _ in range(self.world_size)]
        dist.all_gather(all_gather_list, quant_2_int8_out)
        all_gather_out = torch.cat(all_gather_list, dim=0)
        # step 8:all_gather结果做comm_quant_2反量化，并转fp32
        dequant_2_out = (all_gather_out * comm_quant_scale2).to(torch.float32)
        # step 9:最终结果删除脏数据，并转回3维
        output = dequant_2_out.reshape(-1, dequant_2_out.shape[-1])[0:m, 0:n]
        final_output = output.reshape(ori_shape)
        return final_output

    @staticmethod
    def hcom_quant(output, comm_quant_scale1, comm_quant_scale2, diff_tensor, all_to_all_out, all_gather_out,
                   world_size, use_alltoall):
        ori_shape = output.shape
        n = output.shape[-1]
        if len(output.shape) == 3:
            m = output.shape[0] * output.shape[1]
            output = output.reshape(-1, output.shape[-1])
        else:
            m = output.shape[0]
        # step1:quant操作 mm_output / comm_quant_scale1
        output = output / comm_quant_scale1
        # step2：拼接脏数据使得tensor可被all_to_all整除
        new_output = torch.concat([output, diff_tensor], dim=0)
        # step3: 拼接完的tensor reshape成3维：[world_size, -1, N]
        quant_1_output = new_output.reshape(world_size, -1, n)
        # 转int8
        quant_1_int8_output = torch.clamp(torch.round(quant_1_output), -128, 127).to(torch.int8)
        all_to_all_out = all_to_all_out
        if use_alltoall:
            # step4: 调用all_to_all接口, all_to_all结果转fp32
            dist.all_to_all_single(all_to_all_out, quant_1_int8_output)
            # step5: all_to_all结果做reduce sum,shape：[1,-1,N]
            sum_out = torch.sum(all_to_all_out, dim=0, keepdim=True).to(torch.float32)
        else:
            sum_out = torch.sum(quant_1_int8_output, dim=0, keepdim=True)
        # step6: sum结果做comm_quant_1反量化
        sum_dequant_1_out = sum_out * comm_quant_scale1
        # step7: sum后对1反量化的结果,做comm_quant_2量化，并转int8送进allgather
        quant_2_out = sum_dequant_1_out / comm_quant_scale2
        quant_2_int8_out = torch.clamp(torch.round(quant_2_out), -128, 127).to(torch.int8)
        all_gather_out = all_gather_out
        dist._all_gather_base(all_gather_out, quant_2_int8_out)
        # step 8:all_gather结果做comm_quant_2反量化
        dequant_2_out = (all_gather_out * comm_quant_scale2)
        # step 9:最终结果删除脏数据，并转回3维
        output = dequant_2_out.reshape(-1, dequant_2_out.shape[-1])[0:m, 0:n]
        final_output = output.reshape(ori_shape)
        return final_output
