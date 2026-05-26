from .common import *
import math
import torch
import random
from mindspeed.ops import gmm
import torch_npu
from script import toolss
from ml_dtypes import bfloat16
from en_dtypes import hifloat8
from .registry import class_registry
from .aclnnGroupedMatMulAlltoAllv import *
import numpy as np
import ast
from script import generate_mx_data

ACLNN_INDEX_MAP_ALLTOALL = {
    0: "out",
    1: "mm_out",
}

TORCH_DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "hif8": torch.float32,
}

# MX 量化数据类型映射（新增）
MX_DTYPE_MAP = {
    "fp8_e5m2": "float8_e5m2",
    "fp8_e4m3fn": "float8_e4m3fn",
    "fp8_e8m0": "float8_e8m0fnu",
    "fp4_e2m1": "float4_e2m1",
    "fp4_e1m2": "float4_e1m2",
}

def fp32_to_hif8(fp32_tensor):
    """输入: torch.float32 → 输出: torch.uint8 (hif8 二进制)"""
    np_arr = fp32_tensor.cpu().numpy().astype(hifloat8)
    return torch.from_numpy(np_arr.view(np.uint8)).to(fp32_tensor.device)


def unpermute_fun(tokens,exp_per_card,ep_world_size,rankId,expTokenNums,device="cpu"):
    emptyArr = np.zeros((ep_world_size, exp_per_card),dtype=np.int64)
    for i in range(ep_world_size):
        for j in range(exp_per_card):
            emptyArr[i][j] = int(expTokenNums[i][rankId * exp_per_card + j])
    tmp1 = emptyArr.transpose()
    sum_list1 = np.sum(tmp1, axis=1)
    sum_list2= np.cumsum(sum_list1,axis=0)
    offsets = [0]
    for s in sum_list2[0:-1]:
        offsets.append(int(s))
    sum_list = np.cumsum(tmp1, axis=1)
    indices_list = []
    for i in range(exp_per_card):
        tmp = []
        for j in range(ep_world_size):
            if j == 0:
                tmp.append(list(map(lambda x: x + offsets[i],  list(range(0, sum_list[i][j])))))
            else:
                tmp.append(list(map(lambda x: x + offsets[i], list(range(sum_list[i][j - 1], sum_list[i][j])))))
        indices_list.append(tmp)
    selected = []
    for i in range(ep_world_size):
        for j in range(exp_per_card):
            indices = torch.tensor(indices_list[j][i], dtype=torch.long).to(device)
            selected_rows = tokens.to(device).index_select(dim=0, index=indices)
            selected.append(selected_rows)
    return torch.cat(selected, dim=0).to(tokens.dtype)


class GMM_ALLTOALLV_GRAPH_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, _gmm_x, _gmm_weight, _gmm_x_scale, _gmm_weight_scale, _hcom, _ep_world_size,
                _send_counts, _recv_counts, _gmm_y_dtype, _mm_y_dtype, _mm_x, _mm_weight, _mm_x_scale,
                _mm_weight_scale, _comm_quant_scale, _gmm_x_quant_mode, _gmm_weight_quant_mode,
                _mm_x_quant_mode, _mm_weight_quant_mode, _comm_quant_mode, _gmm_x_dtype, _gmm_weight_dtype,
                _mm_x_dtype, _mm_weight_dtype, type, _output=None, _output_split_sizes=None,
                _input_split_sizes=None, _scale_list=None, _per_token_scale_list=None, _group_list=None,
                _gmm_weight_scale_dtype=None, _gmm_x_scale_dtype=None, _mm_x_scale_dtype=None,
                _mm_weight_scale_dtype=None, _mm_out_flag=False, _rank=None, _device=None,
                _expTokenNums=None, _trans_gmm_weight=False, _trans_mm_weight=False, _ranks=None, 
                _exp_per_card=None):
        if _trans_gmm_weight:
            _gmm_weight = torch.transpose(_gmm_weight, -2, -1)
        if _trans_mm_weight and _mm_weight is not None:
            _mm_weight = _mm_weight.t()
        if type == "mc2":
            gmm_y, mm_y = torch_npu.npu_quant_gmm_alltoallv(
                _gmm_x,
                _gmm_weight,
                _gmm_x_scale,
                _gmm_weight_scale,
                _hcom,
                _ep_world_size,
                _send_counts,
                _recv_counts,
                _gmm_y_dtype,
                send_counts_tensor=None,
                recv_counts_tensor=None,
                mm_x=_mm_x,
                mm_weight=_mm_weight,
                mm_x_scale=_mm_x_scale,
                mm_weight_scale=_mm_weight_scale,
                comm_quant_scale=_comm_quant_scale,
                gmm_x_quant_mode=_gmm_x_quant_mode,
                gmm_weight_quant_mode=_gmm_weight_quant_mode,
                mm_x_quant_mode=_mm_x_quant_mode,
                mm_weight_quant_mode=_mm_weight_quant_mode,
                comm_quant_mode=_comm_quant_mode,
                group_size=None,
                gmm_x_dtype=_gmm_x_dtype,
                gmm_weight_dtype=_gmm_weight_dtype,
                gmm_x_scale_dtype=_gmm_x_scale_dtype,
                gmm_weight_scale_dtype=_gmm_weight_scale_dtype,
                mm_x_dtype=_mm_x_dtype,
                mm_weight_dtype=_mm_weight_dtype,
                mm_x_scale_dtype=_mm_x_scale_dtype,
                mm_weight_scale_dtype=_mm_weight_scale_dtype,
                comm_quant_dtype=None,
                mm_y_dtype=_mm_y_dtype
            )
            return gmm_y, mm_y

        else:
            output_list = torch_npu.npu_grouped_matmul(
                x=[_gmm_x],
                weight=[_gmm_weight],
                bias=None,
                scale=_scale_list,
                offset=None,
                antiquant_scale=None,
                antiquant_offset=None,
                per_token_scale=_per_token_scale_list,
                group_list=_group_list,
                activation_input=None,
                activation_quant_scale=None,
                activation_quant_offset=None,
                split_item=2,
                group_type=0,
                group_list_type=0,
                act_type=0,
                output_dtype=_gmm_y_dtype,
                x_dtype=_gmm_x_dtype,
                weight_dtype=_gmm_weight_dtype,
                scale_dtype=_gmm_weight_scale_dtype,
                per_token_scale_dtype=_gmm_x_scale_dtype,
            )
            output_gmm = output_list[0]

            unpermuted_out = func.unpermute_func(output_gmm, _ranks, _rank, _ep_world_size, _expTokenNums, _exp_per_card, device="npu")
            hccl_out = _output.to(_gmm_y_dtype)
            dist.all_to_all_single(hccl_out, input=unpermuted_out, output_split_sizes=_output_split_sizes, input_split_sizes=_input_split_sizes)

            # 共享专家
            if _mm_out_flag:
                mm_out = torch_npu.npu_quant_matmul(
                    x1=_mm_x,
                    x2=_mm_weight,
                    pertoken_scale=_mm_x_scale,
                    scale=_mm_weight_scale,
                    offset=None,
                    bias=None,
                    output_dtype=_mm_y_dtype,
                    x1_dtype=_mm_x_dtype,
                    x2_dtype=_mm_weight_dtype,
                    pertoken_scale_dtype=_mm_x_scale_dtype,
                    scale_dtype=_mm_weight_scale_dtype,
                    group_sizes=None,
                    y_scale=None,
                )
                hccl_mm_out = mm_out.to(_mm_y_dtype)
            else:
                hccl_mm_out = None
            return hccl_out, hccl_mm_out

@class_registry("aclnnQuantGroupedMatMulAlltoAllv")
class QUANT_GMM_ALLTOALLV(GMM_ALLTOALLV):

    def __init__(self, case, rank, dist=None, hcomm_info=None):
        super().__init__(case, rank, dist, hcomm_info)
        # 量化特有字段
        (self.gmm_x_scale_shape, self.gmm_weight_scale_shape, self.mm_x_scale_shape, self.mm_weight_scale_shape, self.comm_quant_scale_shape) = case["input_scales_shape"]
        (self.gmm_x_scale_dtype, self.gmm_weight_scale_dtype, self.mm_x_scale_dtype, self.mm_weight_scale_dtype, self.comm_quant_scale_dtype) = case["input_scales_dtype"]
        (self.gmm_x_scale_range, self.gmm_weight_scale_range, self.mm_x_scale_range, self.mm_weight_scale_range, self.comm_quant_scale_range) = case["input_scales_range"]
        self.input_scales_dtype = case['input_scales_dtype'] if 'input_scales_dtype' in case else None

        # 量化模式: 0=不量化, 1=TT(张量级), 6=MX ###error1
        self.gmm_x_quant_mode = case['gmmx_quant_mode'] if 'gmmx_quant_mode' in case else None   #None
        self.gmm_weight_quant_mode = case['gmmweight_quant_mode'] if 'gmmweight_quant_mode' in case else None
        self.mm_x_quant_mode = case['mm_quant_mode'] if 'mm_quant_mode' in case else None
        self.mm_weight_quant_mode = case['mmweight_quant_mode'] if 'mmweight_quant_mode' in case else None
        self.comm_quant_mode = case['comm_quant_mode'] if 'comm_quant_mode' in case else None

        self.is_mxfp = (self.gmm_x_quant_mode == 6 and self.gmm_weight_quant_mode == 6)
        self.groupSize = case["group_size"]
        self.groupSizeDec = self.unpack_group_size(self.groupSize)

        ##不用区分场景
        self.exe_bin_name = "aclnnQuantGroupedMatMulAlltoAllv"
        
    def mx_generate_data(self):
        """MX 量化数据生成（每 64 个元素共享一个 scale）"""
        # MX 量化数据缓存 (用于 CPU 计算)
        self.scale_list = None 
        self.per_token_scale_list = None
        self.scale_dtype = None
        self.per_token_scale_dtype = None
        # 生成 FP32 原始数据
        gmm_x_fp32 = self.create_array("gmm_x", self.gmm_x_shape, self.gmm_x_range, "fp32", self.input_path, self.seed)
        gmm_weight_fp32 = self.create_array("gmm_weight", self.gmm_weight_shape, self.gmm_weight_range, "fp32", self.input_path, self.seed)

        # GMM 输入 MX 量化 (axis=-1, 每行每 64 元素共享 scale)
        gmm_x, gmm_x_scale = generate_mx_data.mx_quantize(gmm_x_fp32, MX_DTYPE_MAP[self.gmm_x_dtype], axis=-1, block_size=32, round_mode="rint" )
        # GMM 权重 MX 量化 (根据 trans_gmm_weight_flag 选择 axis)
        if self.trans_gmm_weight_flag:
            gmm_weight, gmm_weight_scale = generate_mx_data.mx_quantize(gmm_weight_fp32, MX_DTYPE_MAP[self.gmm_weight_dtype], axis=-1, block_size=32, round_mode="rint")
        else:
            gmm_weight, gmm_weight_scale = generate_mx_data.mx_quantize(gmm_weight_fp32, MX_DTYPE_MAP[self.gmm_weight_dtype], axis=1, block_size=32, round_mode="rint")
            gmm_weight = np.ascontiguousarray(gmm_weight)
            gmm_weight_scale = np.ascontiguousarray(gmm_weight_scale)

        # 小算子级联输入
        self.gmm_x = self.tensor(gmm_x)
        self.gmm_weight = self.tensor(gmm_weight).transpose(1,2) if self.trans_gmm_weight_flag else self.tensor(gmm_weight)
        self.gmm_x_scale = self.tensor(gmm_x_scale)
        self.gmm_weight_scale = self.tensor(gmm_weight_scale).transpose(1,2) if self.trans_gmm_weight_flag else self.tensor(gmm_weight_scale)
        if "cpu" not in self.scene:
            self.scale_list = [self.gmm_weight_scale]
            self.per_token_scale_list = [self.gmm_x_scale]
            self.scale_dtype = DTYPE_ENUM[self.gmm_weight_scale_dtype]
            self.per_token_scale_dtype = DTYPE_ENUM[self.gmm_x_scale_dtype]

        # 保存 FP32 数据供 CPU 计算
        self.gmm_x_fp32 = torch.from_numpy(gmm_x_fp32).to(torch.float32)
        self.gmm_weight_fp32 = torch.from_numpy(gmm_weight_fp32).to(torch.float32).transpose(1,2) if self.trans_gmm_weight_flag else torch.from_numpy(gmm_weight_fp32).to(torch.float32)

        # 共享专家 MX 量化
        if self.mm_out_flag:
            mm_x_fp32 = self.create_array("mm_x", self.mm_x_shape, self.mm_x_range,"fp32", self.input_path, self.seed)
            mm_weight_fp32 = self.create_array("mm_weight", self.mm_weight_shape, self.mm_weight_range, "fp32", self.input_path, self.seed)

            mm_x, mm_x_scale = generate_mx_data.mx_quantize(mm_x_fp32, MX_DTYPE_MAP[self.mm_x_dtype], axis=-1, block_size=32, round_mode="rint")

            if self.trans_mm_weight_flag:
                mm_weight, mm_weight_scale = generate_mx_data.mx_quantize(mm_weight_fp32, MX_DTYPE_MAP[self.mm_weight_dtype], axis=-1, block_size=32, round_mode="rint")
            else:
                mm_weight, mm_weight_scale = generate_mx_data.mx_quantize(mm_weight_fp32, MX_DTYPE_MAP[self.mm_weight_dtype], axis=0, block_size=32, round_mode="rint"
                )
                mm_weight = np.ascontiguousarray(mm_weight)
                mm_weight_scale = np.ascontiguousarray(mm_weight_scale)

            self.mm_x = self.tensor(mm_x)
            self.mm_weight = self.tensor(mm_weight).transpose(0,1) if self.trans_mm_weight_flag else self.tensor(mm_weight)
            self.mm_x_scale = self.tensor(mm_x_scale)
            self.mm_weight_scale = self.tensor(mm_weight_scale).transpose(0,1) if self.trans_mm_weight_flag else self.tensor(mm_weight_scale)

            # 保存 FP32 数据供 CPU 计算
            self.mm_x_fp32 = torch.from_numpy(mm_x_fp32).to(torch.float32)
            self.mm_weight_fp32 = torch.from_numpy(mm_weight_fp32).to(torch.float32).t() if self.trans_mm_weight_flag else torch.from_numpy(mm_weight_fp32).to(torch.float32)
        else:
            self.mm_x = None
            self.mm_weight = None
            self.mm_x_scale = None
            self.mm_weight_scale = None


        # 保存量化后的输入 bin
        # NOTE: bin 文件存原始 layout（不做物理转置）。
        #   trans_xxx_weight_flag 只作为 API 参数传递，tiling 代码自行处理转置。
        #   C++ 二进制通过 CreateTensor 的 isTransB 参数设置转置 stride。
        if self.save_flag:
            gmm_x.tofile(os.path.join(self.input_path, 'gmm_x.bin'))
            gmm_weight.tofile(os.path.join(self.input_path, 'gmm_weight.bin'))
            gmm_weight_scale.tofile(os.path.join(self.input_path, 'gmm_weight_scale.bin'))
            gmm_x_scale.tofile(os.path.join(self.input_path, 'gmm_x_scale.bin'))

        if self.mm_out_flag and self.save_flag:
            mm_x.tofile(os.path.join(self.input_path, 'mm_x.bin'))
            mm_x_scale.tofile(os.path.join(self.input_path, 'mm_x_scale.bin'))
            mm_weight.tofile(os.path.join(self.input_path, 'mm_weight.bin'))
            mm_weight_scale.tofile(os.path.join(self.input_path, 'mm_weight_scale.bin'))

    def get_input_weight(self, weight_same=0):
        """获取输入权重，支持 TT 和 MX 量化"""
        if self.is_mxfp:
            # MX 量化模式：生成 MX 数据
            # 先调用父类生成 expTokenNums 等基础结构
            super().get_input_weight(weight_same)
            # 然后生成 MX 量化数据覆盖原始数据
            self.mx_generate_data()
        else :
            # TT 量化或非量化模式：使用父类逻辑
            super().get_input_weight(weight_same)
            self.generate_scale()

    def generate_scale(self):
        """TT（张量级）量化 scale 生成"""
        self.scale_list = None
        self.per_token_scale_list = None
        self.scale_dtype = None
        self.per_token_scale_dtype = None

        # 生成 scale tensor（使用父类的 create_array 方法）
        if self.gmm_x_quant_mode == 1:
            self.gmm_x_scale = self.tensor(self.create_array("gmm_x_scale", self.gmm_x_scale_shape, self.gmm_x_scale_range, self.gmm_x_scale_dtype, self.input_path, self.seed))
            if self.gmm_x_scale.numel() == 1:
                scale_val = self.gmm_x_scale.view(-1)[0]
                per_token_scale_tensor = torch.full(
                    (self.exp_per_card,),
                    scale_val,
                    dtype=self.gmm_x_scale.dtype,
                    device=self.device,
                )
            else:
                per_token_scale_tensor = self.gmm_x_scale
            self.per_token_scale_list = [per_token_scale_tensor]
        else:
            self.gmm_x_scale = None

        if self.gmm_weight_quant_mode == 1:
            self.gmm_weight_scale = self.tensor(self.create_array("gmm_weight_scale", self.gmm_weight_scale_shape, self.gmm_weight_scale_range, self.gmm_weight_scale_dtype, self.input_path, self.seed))
            if self.gmm_weight_scale.numel() == 1:
                # 标量扩展为 [e] 张量
                scale_val = self.gmm_weight_scale.view(-1)[0]
                scale_tensor = torch.full(
                    (self.exp_per_card,),
                    scale_val,
                    dtype=self.gmm_weight_scale.dtype,
                    device=self.device,
                )
            else:
                scale_tensor = self.gmm_weight_scale  # 假设已是 [e] 或适当形状
            self.scale_list = [scale_tensor]
        else:
            self.gmm_weight_scale = None

        if self.mm_x_quant_mode == 1 and self.mm_out_flag:
            self.mm_x_scale = self.tensor(self.create_array("mm_x_scale", self.mm_x_scale_shape, self.mm_x_scale_range, self.mm_x_scale_dtype, self.input_path, self.seed))
            
        else:
            self.mm_x_scale = None

        if self.mm_weight_quant_mode == 1 and self.mm_out_flag:
            self.mm_weight_scale = self.tensor(self.create_array("mm_weight_scale", self.mm_weight_scale_shape, self.mm_weight_scale_range, self.mm_weight_scale_dtype, self.input_path, self.seed))
        else:
            self.mm_weight_scale = None
        
        self.scale_dtype = TORCH_DTYPE_MAP[self.gmm_weight_scale_dtype]
        self.per_token_scale_dtype = TORCH_DTYPE_MAP[self.gmm_x_scale_dtype]

        if self.save_flag:
            if self.gmm_x_scale is not None:
                    self.save(self.gmm_x_scale, self.input_path, "gmm_x_scale.bin")
            if self.gmm_weight_scale is not None:
                self.save(self.gmm_weight_scale, self.input_path, "gmm_weight_scale.bin")
            if self.mm_x_scale is not None:
                self.save(self.mm_x_scale, self.input_path, "mm_x_scale.bin")
            if self.mm_weight_scale is not None:
                self.save(self.mm_weight_scale, self.input_path, "mm_weight_scale.bin")

    def get_recv_counts(self):
        send_counts = []
        expTokenNum = self.expTokenNums[self.rank]
        for i in range(self.ranks):
            tmp = expTokenNum[i*self.exp_per_card:(i+1)*self.exp_per_card]
            for j in range(len(tmp)):
                # tmp[j]=16
                send_counts.append(tmp[j])
        return  send_counts

    def get_send_counts(self):
        recv_counts = []
        for i in range(self.ranks):
            tmp = self.expTokenNums[i][ self.rank*self.exp_per_card:(self.rank+1)*self.exp_per_card]
            for j in range(len(tmp)):
                # tmp[j]=16
                recv_counts.append(tmp[j])
        return recv_counts

    def gen_input_splits(self):
        input_splits = []
        expTokenNum = self.expTokenNums[self.rank]
        for i in range(self.ranks):
            input_splits.append(sum(expTokenNum[i*self.exp_per_card:(i+1)*self.exp_per_card]))
        return input_splits

    def gen_output_splits(self):
        recv_counts = []
        for i in range(self.ranks):
            recv_counts.append(sum(self.expTokenNums[i][ self.rank*self.exp_per_card:(self.rank+1)*self.exp_per_card]))
        return recv_counts

    def unpermute(self,tokens,device="cpu"):
        emptyArr = np.zeros((self.ranks, self.exp_per_card),dtype=np.int64)
        for i in range(self.world_size):
            tmp = self.expTokenNums[i][self.rank * self.exp_per_card:(self.rank + 1) * self.exp_per_card]
            emptyArr[i] = tmp
        tmp1 = emptyArr.transpose()
        sum_list1 = np.sum(tmp1, axis=1)
        sum_list2= np.cumsum(sum_list1,axis=0)
        offsets = sum_list2[0:-1].tolist()
        offsets = [0] + offsets
        sum_list = np.cumsum(tmp1, axis=1)
        indices_list = []
        for i in range(self.exp_per_card):
            tmp = []
            for j in range(self.ranks):
                if j == 0:
                    tmp.append(list(map(lambda x: x + offsets[i],  list(range(0, sum_list[i][j])))))
                else:
                    tmp.append(list(map(lambda x: x + offsets[i], list(range(sum_list[i][j - 1], sum_list[i][j])))))
            indices_list.append(tmp)
        selected = []
        for i in range(self.ranks):
            for j in range(self.exp_per_card):
                indices = torch.tensor(indices_list[j][i], dtype=torch.long).to(device)
                selected_rows = tokens.to(device).index_select(dim=0, index=indices)
                selected.append(selected_rows)
        return torch.cat(selected, dim=0).to(tokens.dtype)

    def unpermute_npu(self,tokens):
        return self.unpermute(tokens,"npu")

    def get_tmp_tensor(self):
        if self.mm_out_flag ==False:
            self.mm_weight =None
            self.mm_x = None
        if "cpu" not in self.case['scene']:
            from itertools import accumulate
            self.gmm_x = self.gmm_x.npu()
            self.gmm_weight = self.gmm_weight.npu()
            self.mm_x = self.mm_x.npu() if self.mm_out_flag else None
            self.mm_weight = self.mm_weight.npu() if self.mm_out_flag else None
            self.expert_token_num = self.expert_token_num.npu()
            self.group_list = torch.cumsum(self.group_list,dim=0).npu()
            self.output =self.output.npu()
            if self.save_flag:
                func.save_file(self.input_path, self.send_tensor, self.send_tensor.dtype, "send_counts", "None", self.rank)
                func.save_file(self.input_path, self.recv_tensor, self.recv_tensor.dtype, "recv_counts", "None", self.rank)
            self.send_tensor = self.send_tensor.npu()
            self.recv_tensor = self.recv_tensor.npu()
            self.gmm_weight_before_trans = self.gmm_weight_before_trans.npu()
            # Move scale tensors to NPU
            self.gmm_x_scale = self.gmm_x_scale.npu() if self.gmm_x_scale is not None else None
            self.gmm_weight_scale = self.gmm_weight_scale.npu() if self.gmm_weight_scale is not None else None
            self.mm_x_scale = self.mm_x_scale.npu() if self.mm_x_scale is not None else None
            self.mm_weight_scale = self.mm_weight_scale.npu() if self.mm_weight_scale is not None else None
            if self.mm_out_flag:
                self.mm_weight_before_trans = self.mm_weight_before_trans.npu()

    def get_tmp_tensor_memory(self):
        return 0

    def get_group_list(self):
        recv = torch.zeros(self.ranks,self.exp_per_card).to(torch.int64)
        for i in range(self.world_size):
           tmp = self.expTokenNums[i][self.rank*self.exp_per_card:(self.rank+1)*self.exp_per_card]
           recv[i:] = torch.tensor(tmp)
        return torch.sum(recv,dim=0)

    def get_hccl_mm(self):
        """HCCL 小算子拼接计算"""
        output_list = torch_npu.npu_grouped_matmul(
            x=[self.gmm_x],
            weight=[self.gmm_weight],
            bias=None,
            scale=self.scale_list,
            offset=None,
            antiquant_scale=None,
            antiquant_offset=None,
            per_token_scale=self.per_token_scale_list,
            group_list=self.group_list,
            activation_input=None,
            activation_quant_scale=None,
            activation_quant_offset=None,
            split_item=2,
            group_type=0,
            group_list_type=0,
            act_type=0,
            output_dtype=TORCH_DTYPE_MAP[self.gmm_y_dtype],
            x_dtype=DTYPE_ENUM[self.gmm_x_dtype],
            weight_dtype=DTYPE_ENUM[self.gmm_weight_dtype],
            scale_dtype=self.scale_dtype,
            per_token_scale_dtype=self.per_token_scale_dtype,
        )

        unpermuted_out = self.unpermute_npu(output_list[0])
        self.npu_hccl_out = self.output.to(TORCH_DTYPE_MAP[self.gmm_y_dtype])
        dist.all_to_all_single(self.npu_hccl_out, input=unpermuted_out,
                                output_split_sizes=self.output_splits, input_split_sizes=self.input_splits)

        # 共享专家
        if self.mm_out_flag:
            mm_out = torch_npu.npu_quant_matmul(
                x1=self.mm_x,
                x2=self.mm_weight,
                pertoken_scale=self.mm_x_scale,
                scale=self.mm_weight_scale,
                offset=None,
                bias=None,
                output_dtype=OUT_DTYPE_ENUM[self.mm_y_dtype],
                x1_dtype=DTYPE_ENUM[self.mm_x_dtype],
                x2_dtype=DTYPE_ENUM[self.mm_weight_dtype],
                pertoken_scale_dtype=DTYPE_ENUM[self.mm_x_scale_dtype],
                scale_dtype=DTYPE_ENUM[self.mm_weight_scale_dtype],
                group_sizes=self.groupSizeDec,
                y_scale=None,
            )
            self.npu_hccl_mm_out = mm_out.to(OUT_DTYPE_ENUM[self.mm_y_dtype])
        else:
            self.npu_hccl_mm_out = None

        # 保存 hccl 输出
        if self.save_flag:
            self.save(self.npu_hccl_out, self.output_path, "output_hccl.bin")
            if self.mm_out_flag:
                self.save(self.npu_hccl_mm_out, self.output_path, "mm_out_hccl.bin")
        return (self.npu_hccl_out, None, None, self.npu_hccl_mm_out)

    def get_npu(self):
        # MC2接口使用转置前的tensor。因为可以通过标志区分
        gmm_out, mm_out = torch_npu.npu_quant_gmm_alltoallv(
            self.gmm_x.to("npu"),
            self.gmm_weight.to("npu"),
            self.gmm_x_scale.to("npu"),
            self.gmm_weight_scale.to("npu"),
            self.hcomm_info,
            self.world_size,
            self.get_send_counts(),
            self.get_recv_counts(),
            gmm_y_dtype=OUT_DTYPE_ENUM[self.gmm_y_dtype],
            send_counts_tensor=None,
            recv_counts_tensor=None,
            mm_x=self.mm_x,
            mm_weight=self.mm_weight,
            mm_x_scale=self.mm_x_scale,
            mm_weight_scale=self.mm_weight_scale,
            comm_quant_scale=None,
            gmm_x_quant_mode=self.gmm_x_quant_mode,
            gmm_weight_quant_mode=self.gmm_weight_quant_mode,
            mm_x_quant_mode=self.mm_x_quant_mode if self.mm_out_flag else None,
            mm_weight_quant_mode=self.mm_weight_quant_mode if self.mm_out_flag else None,
            comm_quant_mode=self.comm_quant_mode,
            group_size=self.groupSizeDec,
            gmm_x_dtype=DTYPE_ENUM[self.gmm_x_dtype],
            gmm_weight_dtype=DTYPE_ENUM[self.gmm_weight_dtype],
            gmm_x_scale_dtype=DTYPE_ENUM[self.gmm_x_scale_dtype],
            gmm_weight_scale_dtype=DTYPE_ENUM[self.gmm_weight_scale_dtype],
            mm_x_dtype=DTYPE_ENUM[self.mm_x_dtype] if self.mm_out_flag else None,
            mm_weight_dtype=DTYPE_ENUM[self.mm_weight_dtype] if self.mm_out_flag else None,
            mm_x_scale_dtype=DTYPE_ENUM[self.mm_x_scale_dtype] if self.mm_out_flag else None,
            mm_weight_scale_dtype=DTYPE_ENUM[self.mm_weight_scale_dtype] if self.mm_out_flag else None,
            comm_quant_dtype=None,
            mm_y_dtype=OUT_DTYPE_ENUM[self.mm_y_dtype],
        )
        self.npu_out = gmm_out
        self.npu_mm_out = mm_out if self.mm_out_flag else None
        return self.npu_out

    def grouped_matmul_cpu(self,gmm_x, gmm_weight, group_list):
        # 按行分割左矩阵
        # print(f"[MX_] grouped_matmul_cpu: gmm_x={gmm_x.shape}, gmm_weight={gmm_weight.shape}, group_list={group_list}")
        B_list = list(torch.unbind(gmm_weight, dim=0))
        A_groups = torch.split(gmm_x, group_list, dim=0)
        # print(f"[MX_] grouped_matmul_cpu: num_experts={len(B_list)}, num_groups={len(A_groups)}")
        # 逐组相乘并拼接结果
        results = []
        for i in range(len(group_list)):
            a = A_groups[i].numpy()
            b = B_list[i].numpy()
            r = np.matmul(a,b)
            # print(f"[MX_] grouped_matmul_cpu: expert {i}: A[{a.shape}] x B[{b.shape}] = [{r.shape}], absmax={np.abs(r).max():.6f}")
            results.append(torch.from_numpy(r))
        #因为torch2.1有bug 会报 could not create a primitive descriptor for a matmul primitive所以转成np处理
        #results = [torch.mm(a,b) for a, b in zip(A_groups, B_list)]
        return torch.cat(results, dim=0)

    def get_cpu(self):
        """CPU 端计算"""
        if self.is_mxfp:
            # MX 量化计算
            self.get_cpu_mx()

        else :
            # TT 量化或非量化计算
            cpu_gmm_out = self.grouped_matmul_cpu(self.gmm_x.to(torch.float),self.gmm_weight.to(torch.float),self.group_list.tolist())
            unpermute_out =  self.unpermute(cpu_gmm_out)
            unpermute_out_float = unpermute_out.to(torch.float)
            output = self.output.to(torch.float)
            # 路由专家 TT 量化
            if self.gmm_x_quant_mode == 1 or self.gmm_weight_quant_mode == 1:
                unpermute_out_float = unpermute_out_float * (self.gmm_x_scale * self.gmm_weight_scale)
            dist.all_to_all_single(output, input=unpermute_out_float, output_split_sizes=self.output_splits,input_split_sizes=self.input_splits)
            torch.cpu.synchronize()

            self.cpu_alltoallv_out = output.to(TORCH_DTYPE_MAP.get(self.gmm_x_dtype, ""))
            cpu_gmm_out = cpu_gmm_out.to(TORCH_DTYPE_MAP.get(self.gmm_x_dtype, ""))

            mm_out = None
            if self.mm_out_flag:
                mm_out = torch.mm(self.mm_x.to(torch.float),self.mm_weight.to(torch.float)).to(TORCH_DTYPE_MAP.get(self.gmm_x_dtype, ""))
                # 共享专家 TT 量化
                if self.mm_x_quant_mode == 1 or self.mm_weight_quant_mode == 1:
                    mm_out = mm_out * (self.mm_x_scale * self.mm_weight_scale)

            output_path = f"{self.save_path}/output/{self.rank}/"

            func.save_file(output_path, self.cpu_alltoallv_out, self.cpu_alltoallv_out.dtype, "cpu_out", "None", self.rank)
            if self.mm_out_flag:
                func.save_file(output_path, mm_out, mm_out.dtype, "cpu_mm_out", "None", self.rank)
            self.cpu_golden_out = self.cpu_alltoallv_out
            self.cpu_golden_mm_out = mm_out

    def _ensure_scale_float(self, scale_np):
        """确保 scale 是正确的 float32 值（2^(e-127)），处理 NPU 场景的 uint8 raw bytes"""
        scale_f32 = scale_np.astype(np.float32)
        if scale_f32.max() > 10.0:
            return np.power(2.0, scale_f32.astype(np.float64) - 127).astype(np.float32)
        else:
            return scale_f32

    def get_cpu_mx(self):
        """MX 量化 CPU 计算 - 使用 mxfp_cpu_compute 正确模拟 MX block-scaled matmul"""
        group_list_vals = self.group_list.tolist()

        # 获取 scale 并确保是正确的 float 值
        x_scale_raw = self.gmm_x_scale.cpu().numpy() if hasattr(self.gmm_x_scale, 'cpu') else self.gmm_x_scale.numpy()
        w_scale_raw = self.gmm_weight_scale.cpu().numpy() if hasattr(self.gmm_weight_scale, 'cpu') else self.gmm_weight_scale.numpy()

        x_scale = self._ensure_scale_float(x_scale_raw)
        w_scale = self._ensure_scale_float(w_scale_raw)

        # 处理 pair 维度：[M, ceil(K/64), 2] → [M, ceil(K/64)*2] = [M, ceil(K/32)]
        # MX scale 的 pair=2 表示每个 64-group 包含 2 个 32-block 的 scale
        x_scale = x_scale.reshape(x_scale.shape[0], -1)   # [M, ceil(K/64)*2]
        # [ep, ceil(K/64), N, 2] → [ep, ceil(K/64)*2, N]
        ep, k_groups, N, pair = w_scale.shape
        w_scale = w_scale.transpose(0, 1, 3, 2).reshape(ep, k_groups * pair, N)

        # 逐专家计算
        results = []
        offset = 0
        for i, gl in enumerate(group_list_vals):
            if gl <= 0:
                continue
            x_chunk = self.gmm_x_fp32[offset:offset+gl]        # [gl, K]
            xs_chunk = x_scale[offset:offset+gl]           # [gl, ceil(K/32)]
            w_expert = self.gmm_weight_fp32[i]                   # [K, N]
            ws_expert = w_scale[i]                          # [ceil(K/32), N]

            result = MC2_OP.mxfp_cpu_compute(x_chunk, w_expert, xs_chunk, ws_expert)
            results.append(result)
            offset += gl

        cpu_gmm_out = torch.cat(results, dim=0).to(torch.float32)
        unpermute_out = self.unpermute(cpu_gmm_out)
        unpermute_out_float = unpermute_out.to(torch.float)
        output = self.output.to(torch.float)

        dist.all_to_all_single(output, input=unpermute_out_float, output_split_sizes=self.output_splits, input_split_sizes=self.input_splits)
        torch.cpu.synchronize()

        self.cpu_alltoallv_out = output.to(TORCH_DTYPE_MAP.get(self.gmm_x_dtype, torch.float32))
        mm_out = None
        if self.mm_out_flag:
            # 共享专家 MX 计算
            mm_xs_raw = self.mm_x_scale.cpu().numpy() if hasattr(self.mm_x_scale, 'cpu') else self.mm_x_scale.numpy()
            mm_ws_raw = self.mm_weight_scale.cpu().numpy() if hasattr(self.mm_weight_scale, 'cpu') else self.mm_weight_scale.numpy()
            mm_xs = self._ensure_scale_float(mm_xs_raw)
            mm_ws = self._ensure_scale_float(mm_ws_raw)
            # 处理 pair 维度：[M, ceil(K/64), 2] → [M, ceil(K/64)*2] = [M, ceil(K/32)]
            mm_xs = mm_xs.reshape(mm_xs.shape[0], -1)
            # [ceil(K/64), N, 2] → [ceil(K/64)*2, N]
            k_groups, N, pair = mm_ws.shape
            mm_ws = mm_ws.transpose(0, 2, 1).reshape(k_groups * pair, N)
            mm_result = MC2_OP.mxfp_cpu_compute(self.mm_x_fp32, self.mm_weight_fp32, mm_xs, mm_ws)
            mm_out = mm_result.to(TORCH_DTYPE_MAP.get(self.gmm_x_dtype, torch.float32))

        output_path = f"{self.save_path}/output/{self.rank}/"

        func.save_file(output_path, self.cpu_alltoallv_out, self.cpu_alltoallv_out.dtype,
                      "cpu_out", "None", self.rank)
        if self.mm_out_flag:
            func.save_file(output_path, mm_out, mm_out.dtype, "cpu_mm_out", "None", self.rank)
        self.cpu_golden_out = self.cpu_alltoallv_out
        self.cpu_golden_mm_out = mm_out

    def get_graph(self, type="mc2"):
        model = GMM_ALLTOALLV_GRAPH_Model()
        alltoallv_output= self.output
        _send_counts = [int(x) for x in self.get_send_counts()]
        _recv_counts = [int(y) for y in self.get_recv_counts()]
        model = self.define_model(model, self.graph_type)
        tmp_mm_weight = self.mm_weight_before_trans if self.mm_out_flag else None

        if type == "mc2":
            out_graph = model(
                _gmm_x=self.gmm_x,
                _gmm_weight=self.gmm_weight_before_trans,
                _gmm_x_scale=self.gmm_x_scale,
                _gmm_weight_scale=self.gmm_weight_scale,
                _hcom=self.hcomm_info,
                _ep_world_size=self.world_size,
                _send_counts=_send_counts,
                _recv_counts=_recv_counts,
                _gmm_y_dtype=OUT_DTYPE_ENUM[self.gmm_y_dtype],
                _mm_y_dtype=OUT_DTYPE_ENUM[self.mm_y_dtype] if self.mm_out_flag else None,
                _mm_x=self.mm_x if self.mm_out_flag else None,
                _mm_weight=self.mm_weight_before_trans if self.mm_out_flag else None,
                _mm_x_scale=self.mm_x_scale if self.mm_x_quant_mode == 1 else None,
                _mm_weight_scale=self.mm_weight_scale if self.mm_weight_quant_mode == 1 else None,
                _comm_quant_scale=self.comm_quant_scale if self.comm_quant_mode == 1 else None,
                _gmm_x_quant_mode=self.gmm_x_quant_mode,
                _gmm_weight_quant_mode=self.gmm_weight_quant_mode,
                _mm_x_quant_mode=self.mm_x_quant_mode if self.mm_out_flag else None,
                _mm_weight_quant_mode=self.mm_weight_quant_mode if self.mm_out_flag else None,
                _comm_quant_mode=self.comm_quant_mode,
                _gmm_x_dtype=DTYPE_ENUM[self.gmm_x_dtype],
                _gmm_weight_dtype=DTYPE_ENUM[self.gmm_weight_dtype],
                _mm_x_dtype=DTYPE_ENUM[self.mm_x_dtype] if self.mm_out_flag else None,
                _mm_weight_dtype=(DTYPE_ENUM[self.mm_weight_dtype]) if self.mm_out_flag else None,
                type=type,
                _output=self.output,
                _output_split_sizes=self.output_splits,
                _input_split_sizes=self.input_splits,
                _scale_list=self.scale_list,
                _per_token_scale_list=self.per_token_scale_list,
                _group_list=self.group_list,
                _gmm_weight_scale_dtype=DTYPE_ENUM[self.gmm_weight_scale_dtype],
                _gmm_x_scale_dtype=DTYPE_ENUM[self.gmm_x_scale_dtype],
                _mm_x_scale_dtype=DTYPE_ENUM[self.mm_x_scale_dtype] if self.mm_out_flag else None,
                _mm_weight_scale_dtype=DTYPE_ENUM[self.mm_weight_scale_dtype] if self.mm_out_flag else None,
                _mm_out_flag=self.mm_out_flag,
                _rank=self.rank,
                _device=self.device,
                _expTokenNums=self.expTokenNums,
                _trans_gmm_weight=self.trans_gmm_weight_flag,
                _trans_mm_weight=self.trans_mm_weight_flag,
                _ranks=self.ranks,
                _exp_per_card=self.exp_per_card)
            self.npu_out, self.npu_mm_out = out_graph
        elif type == "hccl_mm":
            self.npu_hccl_out, _, _, self.npu_hccl_mm_out = self.get_hccl_mm()
        else:
            print("In QuantGroupedMatmulAlltoAllv get_graph,wrong type..")

    def compare(self, rank=0, check_md5=0):
        return super().compare(rank, check_md5)

    def process_aclnn(self):

        self.root_path = os.path.abspath(os.getcwd())
        self.case_format = ['ND', 'ND', 'ND']
        self.output_dtype = self.case['output_tensor_dtype'][0]
        self.output_format = self.case_format[0]
        self.bin_path = self.root_path + "/bin_path/" + self.case_name
        self.input_tensor_dtype = str(self.input_tensor_dtype).replace("[", "").replace("]", "").replace("'", "").replace(" ", "")
        self.output_tensor_dtype = str(self.output_tensor_dtype).replace("[", "").replace("]", "").replace("'", "").replace(" ", "")
        self.output_tensor_format = str(self.output_tensor_format).replace("[", "").replace("]", "").replace("'", "").replace(" ", "")
        self.loop = 1
        # 根据架构aarch/x86获取待执行二进制名称
        result = subprocess.check_output(["uname", "-a"]).decode("utf-8")
        arch = "aarch" if "aarch64" in result.lower() else "x86"
        exe_binary = "bin_path/" + self.exe_bin_name + "_" + arch
        # 拼二进制执行命令
        if self.case['soc'] == "910D":
            tmp_binary = self.exe_bin_name + arch
            command = [f"nohup {self.root_path}/{exe_binary}"]
            command.append(f"--bin_path {self.bin_path}")
        else:
            exe_binary = "cpp/bin/" + self.exe_bin_name + "_" + arch
            command = [f"{self.root_path}/{exe_binary}"]
            command.append(f"--bin_path {self.bin_path}")
        alltoallv_shape = [self.bsk, self.n]
        mm_out_shape = [self.bs, self.n2]
        gmm_w_shape = self.gmm_weight_shape
        command.append(f"--gmm_weight_shape {','.join(map(str, gmm_w_shape))}")
        gmm_ws_shape = self.gmm_weight_scale_shape
        command.append(f"--gmm_weight_scale_shape {','.join(map(str, gmm_ws_shape))}")
        command.append(f"--send_counts_shape {','.join(map(str, self.send_counts_shape))}")
        command.append(f"--recv_counts_shape {','.join(map(str, self.recv_counts_shape))}")
        command.append(f"--mm_x_shape {','.join(map(str, self.mm_x_shape))}")
        command.append(f"--mm_x_scale_shape {','.join(map(str, self.mm_x_scale_shape))}")
        mm_ws_shape = self.mm_weight_scale_shape
        command.append(f"--mm_weight_scale_shape {','.join(map(str, mm_ws_shape))}")
        if self.mm_out_flag:
            mm_w_shape = self.mm_weight_shape
            command.append(f"--mm_weight_shape {','.join(map(str, mm_w_shape))}")
            command.append(f"--mm_output_shape {','.join(map(str, mm_out_shape))}")
        else:
            self.trans_mm_weight_flag = False
            command.append(f"--mm_weight_shape None")
            command.append(f"--mm_output_shape None")
        command.append(f"--input_tensor_dtype {self.input_tensor_dtype}")
        command.append(f"--input_tensor_format {self.input_tensor_format}")
        command.append(f"--input_scales_dtype {self._format_dtype_list(self.input_scales_dtype)}")
        command.append(f"--output_tensor_dtype {self.output_tensor_dtype}")
        command.append(f"--output_tensor_format {self.output_tensor_format}")
        command.append(f"--ep_world_size {self.world_size}")
        command.append(f"--trans_gmm_weight_flag {str(self.trans_gmm_weight_flag).lower()}")
        command.append(f"--trans_mm_weight_flag {str(self.trans_mm_weight_flag).lower()}")
        command.append(f"--n_dev {self.ranks}")
        command.append(f"--loop_cnt {self.loop}")
        command.append(f"--mm_out_flag {str(self.mm_out_flag).lower()}")
        command.append(f"--alltoallv_out_shape {','.join(map(str, alltoallv_shape))}")
        command.append(f"--gmm_x_quant_mode {str(self.gmm_x_quant_mode).lower()}")
        command.append(f"--gmm_weight_quant_mode {str(self.gmm_weight_quant_mode).lower()}")
        command.append(f"--mm_x_quant_mode {str(self.mm_x_quant_mode).lower()}")
        command.append(f"--mm_weight_quant_mode {str(self.mm_weight_quant_mode).lower()}")
        command.append(f"--comm_quant_mode {str(self.comm_quant_mode).lower()}")
        command.append(f"--groupSize {self.groupSize}")
        if self.case['soc'] == "910D":
           if self.case['scene'] == 'aclnn':
                mc2_sh = self.case_name + '_mc2'
                # 移除已有shell文件
                mv_command = f"rm -rf bin_path/{self.case_name}/{self.case_name}*.sh"
                ret = subprocess.run(mv_command, shell=True)
                # 拼二进制执行命令
                for i in range(self.ranks):
                    tmp_command = command.copy()
                    tmp_num = self.gmm_x_A_array[i].tolist()
                    tmp = np.array([tmp_num, self.k])
                    ceil_k_div_64 = self.k//64
                    tmp_scale = np.array([int(tmp_num), int(ceil_k_div_64), 2])
                    if self.gmm_x_quant_mode == 6:
                        tmp_command.append(f"--gmm_x_scale_shape {','.join(map(str, tmp_scale))}")
                    else :
                        tmp_command.append(f"--gmm_x_scale_shape {','.join(map(str, self.gmm_x_scale_shape))}")
                    tmp_command.append(f"--gmm_x_shape {','.join(map(str, tmp))}")
                    tmp_command.append(f"--rankId {i} &")
                    aclnn_command = " ".join(tmp_command)
                    print(f"[INFO] rank:{i}, aclnn command:{aclnn_command}")
                    with open(f"{self.bin_path}/{mc2_sh}.sh", "a") as f:
                        f.write(aclnn_command + "\n")
                sh_command = "bash " + self.bin_path + "/" + self.case_name + "_mc2.sh"
                print(f"[INFO] sh command:{sh_command}")
                exe_binary = self.exe_bin_name + "_" + arch
                pre_pids = func.get_pids(exe_binary)
                try:
                    ret = subprocess.run(sh_command, shell=True)
                except subprocess.CalledProcessError:
                    print(f"Aclnn command excute result: {ret}")
                # 等待后台进程完成
                import psutil
                current_pids = func.get_pids(exe_binary)
                target_pid = [pid for pid in current_pids if pid not in pre_pids]
                print(f"进程名 {exe_binary} 对应的PID: {target_pid}")
                print("等待进程结束...")
                while True:
                    alive_pids = [pid for pid in target_pid if psutil.pid_exists(pid)]
                    if not alive_pids:
                        break
                    print(f"进程 {exe_binary} (PID:{alive_pids}) 仍在运行...", end="\r")
                    time.sleep(1)
                # [DEBUG] verify output files after C++ binary finishes
                for rank_id in range(self.ranks):
                    for fname in ["npu_out.bin", "npu_mm_out.bin"]:
                        fpath = f"{self.bin_path}/output/{rank_id}/{fname}"
                        if os.path.exists(fpath):
                            fsize = os.path.getsize(fpath)
                            fmtime = time.ctime(os.path.getmtime(fpath))
                            data = np.fromfile(fpath, dtype=np.float16)
                            nan_cnt = np.sum(np.isnan(data))
                            inf_cnt = np.sum(np.isinf(data))
                            print(f"[OUTPUT_CHECK] {fpath}: size={fsize}, mtime={fmtime}, "
                                  f"shape={data.shape}, nan={nan_cnt}, inf={inf_cnt}, "
                                  f"min={np.nanmin(data):.4f}, max={np.nanmax(data):.4f}")
                        else:
                            print(f"[OUTPUT_CHECK] MISSING: {fpath}")
        else:
            tmp = self.gmm_x_A_array.tolist()
            tmp_scale = self.gmm_x_A_array.tolist()
            tmp.append(self.k)
            ceil_k_div_64 = self.k//64
            tmp_scale_shape = np.array([int(tmp_num), int(ceil_k_div_64), int(2)])
            print(f"===tmp_scale_shape is {tmp_scale_shape}===")
            if self.gmm_x_quant_mode == 6:
                command.append(f"--gmm_x_scale_shape {','.join(map(str, tmp_scale_shape))}")
            else:
                command.append(f"--gmm_x_scale_shape {','.join(map(str, self.gmm_x_scale_shape))}")
            command.append(f"--gmm_x_shape {','.join(map(str, tmp))}")
            aclnn_command = " ".join(command)
            print(f"[INFO] aclnn command:{aclnn_command}")
            with open(f"{self.bin_path}/{self.case_name}.sh", "w") as f:
                f.write(aclnn_command + "\n")
            sh_command = f"bash bin_path/{self.case_name}/{self.case_name}.sh"
            print(f"[INFO] sh command:{sh_command}")
            try:
                ret = subprocess.run(sh_command, shell=True)
            except subprocess.CalledProcessError:
                print(f"Aclnn command excute result: {ret}")

    def _format_dtype_list(self, dtype_list):
        if isinstance(dtype_list, (list, tuple)):
            return ",".join(str(d).replace("'", "").replace(" ", "") for d in dtype_list)
        return str(dtype_list).replace("'", "").replace(" ", "")