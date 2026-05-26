from en_dtypes import hifloat8
from script import generate_mx_data
import ast
# 公共引入在量化脚本中
from .aclnnAlltoAllvGroupedMatMul import *

ACLNN_INDEX_MAP_ALLTOALL = {0: "out", 1: "mm_out", 2: "permute_out"}

# MX量化dtype映射，抽取到外层方便其他量化方法复用
MX_DTYPE_MAP = {
    "fp8_e5m2": "float8_e5m2",
    "fp8_e4m3fn": "float8_e4m3fn",
    "fp8_e8m0": "float8_e8m0",
    "fp4_e2m1": "float4_e2m1",
    "fp4_e1m2": "float4_e1m2",
}

class AlltoAllvQuantGroupedMatmulGraphModel(torch.nn.Module):
    """
    量化的图模型，继承自非量化图模型并添加scale参数
    """

    def __init__(self):
        super().__init__()

    def forward(self, _gmm_x, _gmm_weight, _gmm_x_scale, _gmm_weight_scale, _hcom, _ep_world_size,
        _send_counts, _recv_counts, _gmm_y_dtype, _mm_y_dtype, _mm_x, _mm_weight, _mm_x_scale,
        _mm_weight_scale, _permute_out_flag, _gmm_x_quant_mode, _gmm_weight_quant_mode,
        _mm_x_quant_mode, _mm_weight_quant_mode, _gmm_x_dtype, _gmm_weight_dtype, _mm_x_dtype,
        _mm_weight_dtype, type, _alltoallv_output=None, _output_split_sizes=None,
        _input_split_sizes=None, _scale_list=None, _per_token_scale_list=None, _group_list=None,
        _gmm_weight_scale_dtype=None, _gmm_x_scale_dtype=None, _mm_x_scale_dtype=None,
        _mm_weight_scale_dtype=None, _mm_out_flag=False, _e=None, _rank=None, _device=None,
        _expTokenNums=None, _trans_gmm_weight=False, _trans_mm_weight=False, _group_size=None):
        if _trans_gmm_weight:
            _gmm_weight = torch.transpose(_gmm_weight, -2, -1)
        if _trans_mm_weight and _mm_weight is not None:
            _mm_weight = _mm_weight.t()
        if type == "mc2":
            gmm_y, mm_y, permute_out = torch_npu.npu_alltoallv_quant_gmm(
                gmm_x=_gmm_x,
                gmm_weight=_gmm_weight,
                gmm_x_scale=_gmm_x_scale,
                gmm_weight_scale=_gmm_weight_scale,
                hcom=_hcom,
                ep_world_size=_ep_world_size,
                send_counts=_send_counts,
                recv_counts=_recv_counts,
                gmm_y_dtype=_gmm_y_dtype,
                mm_y_dtype=_mm_y_dtype,
                mm_x=_mm_x,
                mm_weight=_mm_weight,
                mm_x_scale=_mm_x_scale,
                mm_weight_scale=_mm_weight_scale,
                permute_out_flag=_permute_out_flag,
                gmm_x_quant_mode=_gmm_x_quant_mode,
                gmm_weight_quant_mode=_gmm_weight_quant_mode,
                mm_x_quant_mode=_mm_x_quant_mode,
                mm_weight_quant_mode=_mm_weight_quant_mode,
                gmm_x_dtype=_gmm_x_dtype,
                gmm_weight_dtype=_gmm_weight_dtype,
                mm_x_dtype=_mm_x_dtype,
                mm_weight_dtype=_mm_weight_dtype,
                gmm_x_scale_dtype=_gmm_x_scale_dtype,
                gmm_weight_scale_dtype=_gmm_weight_scale_dtype,
                mm_x_scale_dtype=_mm_x_scale_dtype,
                mm_weight_scale_dtype=_mm_weight_scale_dtype,
                group_size=_group_size
            )
            return gmm_y, mm_y, permute_out
        else:
            # alltoallv
            dist.all_to_all_single(
                _alltoallv_output,
                input=_gmm_x,
                output_split_sizes=_output_split_sizes,
                input_split_sizes=_input_split_sizes,
            )
            # permute
            _hccl_permute_out = permute_func(_alltoallv_output, _e, _ep_world_size, _rank, _device, _expTokenNums)
            # 路由专家 grouped matmul

            output_list = torch_npu.npu_grouped_matmul(
                x=[_hccl_permute_out],
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
            gmm_y = output_gmm

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
                mm_y = mm_out.to(_mm_y_dtype)
            else:
                mm_y = None
            if _permute_out_flag:
                permute_out = _hccl_permute_out
            else:
                permute_out=None
            return gmm_y, mm_y, permute_out


@class_registry("aclnnAlltoAllvQuantGroupedMatMul")
class AlltoAllvQuantGroupedMatMul(AlltoAllvGroupedMatMul):
    """
    继承自V1版本，增加 scale 相关处理
    """

    def __init__(self, case, rank, dist=None, hcomm_info=None):
        # 先调用父类初始化，加载公共字段
        super().__init__(case, rank, dist, hcomm_info)
        # 量化特有字段
        (self.gmm_x_scale_shape, self.gmm_weight_scale_shape, self.mm_x_scale_shape, self.mm_weight_scale_shape) = case["input_scales_shape"]
        (self.gmm_x_scale_dtype, self.gmm_weight_scale_dtype, self.mm_x_scale_dtype, self.mm_weight_scale_dtype) = case["input_scales_dtype"]
        (self.gmm_x_scale_range, self.gmm_weight_scale_range, self.mm_x_scale_range, self.mm_weight_scale_range) = case["input_scales_range"]
        # 量化模式
        self.gmmx_quant_mode = case.get("gmmx_quant_mode")
        self.gmm_weight_quant_mode = case.get("gmm_weight_quant_mode")
        self.mmx_quant_mode = case.get("mmx_quant_mode")
        self.mm_weight_quant_mode = case.get("mm_weight_quant_mode")
        self.group_size = case["group_size"]
        self.group_size_dec = self.unpack_group_size(self.group_size)
        # 判断是否为MX量化：四个quant_mode入参都是6
        self.is_mxfp = (self.gmmx_quant_mode == 6 and self.gmm_weight_quant_mode == 6 and
                       self.mmx_quant_mode == 6 and self.mm_weight_quant_mode == 6)
        self.exe_bin_name = "aclnnAlltoAllvQuantGroupedMatMul"

    def mx_generate_data(self):
        """MX量化特殊的输入生成方法"""
        # MX量化需要先生成fp32数据，再进行量化处理
        gmm_x_fp32 = self.create_array("gmm_x", self.gmm_x_shape, self.gmm_x_range, "fp32", self.input_path, self.seed)
        gmm_weight_fp32 = self.create_array("gmm_weight", self.gmm_weight_shape, self.gmm_weight_range, "fp32", self.input_path, self.seed)

        # gmm_x量化
        self.gmm_x, self.gmm_x_scale = generate_mx_data.mx_quantize(gmm_x_fp32, MX_DTYPE_MAP[self.gmm_x_dtype], axis=-1, block_size=32, round_mode="rint")

        # gmm_weight量化，根据trans_gmm_weight_flag选择量化axis
        if self.trans_gmm_weight_flag:
            self.gmm_weight, self.gmm_weight_scale = generate_mx_data.mx_quantize(gmm_weight_fp32, MX_DTYPE_MAP[self.gmm_weight_dtype], axis=-1, block_size=32, round_mode="rint")
        else:
            self.gmm_weight, self.gmm_weight_scale = generate_mx_data.mx_quantize(gmm_weight_fp32, MX_DTYPE_MAP[self.gmm_weight_dtype], axis=1, block_size=32, round_mode="rint")
            self.gmm_weight = np.ascontiguousarray(self.gmm_weight)
            self.gmm_weight_scale = np.ascontiguousarray(self.gmm_weight_scale)
        if "cpu" not in self.scene:
            self.gmm_x = func.pack_int4(self.gmm_x) if self.gmm_x_dtype in ['fp4_e2m1', 'fp4_e1m2'] else self.gmm_x
            self.gmm_weight = func.pack_int4(self.gmm_weight) if self.gmm_weight_dtype in ['fp4_e2m1', 'fp4_e1m2'] else self.gmm_weight
            if self.gmm_x_dtype in ["fp4_e2m1", "fp4_e1m2"]:
                self.gmm_x = self.gmm_x.reshape(self.gmm_x_shape[0], self.gmm_x_shape[1] >> 1)
            if self.gmm_weight_dtype in ["fp4_e2m1", "fp4_e1m2"]:
                self.gmm_weight = self.gmm_weight.reshape(self.gmm_weight_shape[0], self.gmm_weight_shape[1], self.gmm_weight_shape[2] >> 1)
        else:
            if self.gmm_x_dtype in ["fp4_e2m1", "fp4_e1m2"]:
                self.gmm_x = self.trans_to_float32(self.gmm_x, self.gmm_x_dtype)
            if self.gmm_weight_dtype in ["fp4_e2m1", "fp4_e1m2"]:
                self.gmm_weight = self.trans_to_float32(self.gmm_weight, self.gmm_weight_dtype)

        if self.save_flag:
            self.gmm_x.tofile(os.path.join(self.input_path, "gmm_x.bin"))
            self.gmm_x_scale.tofile(os.path.join(self.input_path, "gmm_x_scale.bin"))
            self.gmm_weight.tofile(os.path.join(self.input_path, "gmm_weight.bin"))
            self.gmm_weight_scale.tofile(os.path.join(self.input_path, "gmm_weight_scale.bin"))

        self.gmm_x = self.tensor(self.gmm_x)
        self.gmm_weight = self.tensor(self.gmm_weight)
        self.gmm_weight = self.gmm_weight.transpose(1, 2) if self.trans_gmm_weight_flag else self.gmm_weight
        self.gmm_x_scale = self.tensor(self.gmm_x_scale)
        self.gmm_weight_scale = self.tensor(self.gmm_weight_scale)
        
        # mm_x和mm_weight量化
        if self.mm_out_flag:
            mm_x_fp32 = self.create_array("mm_x", self.mm_x_shape, self.mm_x_range, "fp32", self.input_path, self.seed)
            mm_weight_fp32 = self.create_array("mm_weight", self.mm_weight_shape, self.mm_weight_range, "fp32", self.input_path, self.seed)

            self.mm_x, self.mm_x_scale = generate_mx_data.mx_quantize(mm_x_fp32, MX_DTYPE_MAP[self.mm_x_dtype], axis=-1, block_size=32, round_mode="rint")

            if self.trans_mm_weight_flag:
                self.mm_weight, self.mm_weight_scale = generate_mx_data.mx_quantize(mm_weight_fp32, MX_DTYPE_MAP[self.mm_weight_dtype], axis=-1, block_size=32, round_mode="rint")
            else:
                self.mm_weight, self.mm_weight_scale = generate_mx_data.mx_quantize(mm_weight_fp32, MX_DTYPE_MAP[self.mm_weight_dtype], axis=0, block_size=32, round_mode="rint")
                self.mm_weight = np.ascontiguousarray(self.mm_weight)
                self.mm_weight_scale = np.ascontiguousarray(self.mm_weight_scale)
            if "cpu" not in self.scene:
                if self.mm_x_dtype in ['fp4_e2m1', 'fp4_e1m2']:
                    self.mm_x = func.pack_int4(self.mm_x)
                    self.mm_x = self.mm_x.reshape(self.mm_x_shape[0], self.mm_x_shape[1] >> 1)
                if self.mm_weight_dtype in ['fp4_e2m1', 'fp4_e1m2']:
                    self.mm_weight = func.pack_int4(self.mm_weight)
                    self.mm_weight = self.mm_weight.reshape(self.mm_weight_shape[0], self.mm_weight_shape[1] >> 1)
            else:
                if self.mm_x_dtype in ['fp4_e2m1', 'fp4_e1m2']:
                    self.mm_x = self.trans_to_float32(self.mm_x, self.mm_x_dtype)
                if self.mm_weight_dtype in ['fp4_e2m1', 'fp4_e1m2']:
                    self.mm_weight = self.trans_to_float32(self.mm_weight, self.mm_weight_dtype)
            
            if self.save_flag:
                self.mm_x.tofile(os.path.join(self.input_path, "mm_x.bin"))
                self.mm_x_scale.tofile(os.path.join(self.input_path, "mm_x_scale.bin"))
                self.mm_weight.tofile(os.path.join(self.input_path, "mm_weight.bin"))
                self.mm_weight_scale.tofile(os.path.join(self.input_path, "mm_weight_scale.bin"))

            self.mm_x = self.tensor(self.mm_x)
            self.mm_weight = self.tensor(self.mm_weight)
            self.mm_weight = self.mm_weight.transpose(0, 1) if self.trans_mm_weight_flag else self.mm_weight
            self.mm_x_scale = self.tensor(self.mm_x_scale)
            self.mm_weight_scale = self.tensor(self.mm_weight_scale)

    def get_input_weight(self, weight_same=0):
        if self.is_mxfp:
            # MX量化使用专门的生成方法
            self.mx_generate_data()
            self.input_split_sizes = self.gen_input_split_sizes()
            self.output_split_sizes = self.gen_output_split_sizes()
            self.group_list = self.get_group_list()
            # 保存输入bin
            if self.save_flag:
                self.save(torch.tensor(self.get_send_counts()).to(TORCH_DTYPE_MAP[self.send_counts_dtype]), self.input_path, "send_counts.bin")
                self.save(torch.tensor(self.get_recv_counts()).to(TORCH_DTYPE_MAP[self.recv_counts_dtype]), self.input_path, "recv_counts.bin")
        else:
            # 调用父类生成基础 tensor（gmm_x, gmm_weight, mm_x, mm_weight, expTokenNums 等）
            super().get_input_weight(weight_same)
            # 非MX量化（TT量化），使用父类的create_array方法生成scale
            if self.gmmx_quant_mode == 1:
                self.gmm_x_scale = self.tensor(
                    self.create_array("gmm_x_scale", self.gmm_x_scale_shape, self.gmm_x_scale_range, self.gmm_x_scale_dtype, self.input_path, self.seed)
                )

            if self.gmm_weight_quant_mode == 1:
                self.gmm_weight_scale = self.tensor(
                    self.create_array("gmm_weight_scale", self.gmm_weight_scale_shape, self.gmm_weight_scale_range, self.gmm_weight_scale_dtype, self.input_path, self.seed)
                )

            if self.mmx_quant_mode == 1 and self.mm_out_flag:
                self.mm_x_scale = self.tensor(
                    self.create_array("mm_x_scale", self.mm_x_scale_shape, self.mm_x_scale_range, self.mm_x_scale_dtype, self.input_path, self.seed)
                )

            if self.mm_weight_quant_mode == 1 and self.mm_out_flag:
                self.mm_weight_scale = self.tensor(
                    self.create_array("mm_weight_scale", self.mm_weight_scale_shape, self.mm_weight_scale_range, self.mm_weight_scale_dtype, self.input_path, self.seed)
                )

            # 保存scale到输入bin
            if self.save_flag:
                if self.gmmx_quant_mode == 1 and self.gmm_x_scale is not None:
                    self.save(self.gmm_x_scale, self.input_path, "gmm_x_scale.bin")
                if self.gmm_weight_quant_mode == 1 and self.gmm_weight_scale is not None:
                    self.save(self.gmm_weight_scale, self.input_path, "gmm_weight_scale.bin")
                if self.mmx_quant_mode == 1 and self.mm_out_flag and self.mm_x_scale is not None:
                    self.save(self.mm_x_scale, self.input_path, "mm_x_scale.bin")
                if self.mm_weight_quant_mode == 1 and self.mm_out_flag and self.mm_weight_scale is not None:
                    self.save(self.mm_weight_scale, self.input_path, "mm_weight_scale.bin")

    def mx_get_tmp_tensor(self):
        """MX量化：预计算scale处理和耗时操作"""
        # 耗时操作：提前reshape和contiguous
        self.gmm_x_scale_reshaped = self.gmm_x_scale.reshape(self.bsk, -1).contiguous()
        if self.mm_out_flag:
            self.mm_x_scale_reshaped = self.mm_x_scale.reshape(self.bs, -1).contiguous()
        # 构造scale列表（权重量化）
        if self.gmm_weight_scale is not None:
            if self.gmm_weight_scale.numel() == 1:
                scale_val = self.gmm_weight_scale.view(-1)[0]
                scale_tensor = torch.full((self.e,), scale_val, dtype=self.gmm_weight_scale.dtype, device=self.device)
            else:
                scale_tensor = self.gmm_weight_scale
            self.scale_list = [scale_tensor]
            self.hccl_scale_list = [self.scale_list[0].transpose(1, 2)] if self.trans_gmm_weight_flag else [self.scale_list[0].contiguous()]
        else:
            self.scale_list = None
            self.hccl_scale_list = None

        # 构造per_token_scale列表（输入量化）
        if self.gmm_x_scale is not None:
            if self.gmm_x_scale.numel() == 1:
                scale_val = self.gmm_x_scale.view(-1)[0]
                per_token_scale_tensor = torch.full((self.e,), scale_val, dtype=self.gmm_x_scale.dtype, device=self.device)
            else:
                per_token_scale_tensor = self.gmm_x_scale
            self.per_token_scale_list = [per_token_scale_tensor]
        else:
            self.per_token_scale_list = None

        if self.mm_out_flag:
            self.hccl_mm_weight_scale = self.mm_weight_scale.transpose(0, 1) if self.trans_mm_weight_flag else self.mm_weight_scale.contiguous()
        self.alltoallv_scale_output = torch.zeros(
            [self.permute_out_shape_list[self.rank][0], math.ceil(self.h1 / 64), 2],
            dtype=TORCH_DTYPE_MAP[self.gmm_x_scale_dtype],
        ).to(self.device)

    def get_tmp_tensor(self):
        """预计算 scale 乘积"""
        super().get_tmp_tensor()

        if self.is_mxfp:
            # MX量化分支
            self.mx_get_tmp_tensor()
        else:
            # 非MX量化：处理TT量化的scale
            if self.gmm_weight_scale is not None:
                if self.gmm_weight_scale.numel() == 1:
                    scale_val = self.gmm_weight_scale.view(-1)[0]
                    scale_tensor = torch.full((self.e,), scale_val, dtype=self.gmm_weight_scale.dtype, device=self.device)
                else:
                    scale_tensor = self.gmm_weight_scale
                self.scale_list = [scale_tensor]
                self.hccl_scale_list = self.scale_list
            else:
                self.scale_list = None
                self.hccl_scale_list = None

            # 构造per_token_scale列表（输入量化）
            if self.gmm_x_scale is not None:
                if self.gmm_x_scale.numel() == 1:
                    scale_val = self.gmm_x_scale.view(-1)[0]
                    per_token_scale_tensor = torch.full((self.e,), scale_val, dtype=self.gmm_x_scale.dtype, device=self.device)
                else:
                    per_token_scale_tensor = self.gmm_x_scale
                self.per_token_scale_list = [per_token_scale_tensor]
            else:
                self.per_token_scale_list = None

            if self.mm_out_flag:
                self.hccl_mm_weight_scale = self.mm_weight_scale

            if self.gmmx_quant_mode == 1 and self.gmm_weight_quant_mode == 1:
                if self.device.type == "cpu" or self.device.type == "npu":
                    self.gmm_x_scale.cpu().numpy()
                    self.gmm_scale = (torch.unsqueeze(torch.from_numpy(func.scale_generate(self.gmm_x_scale.cpu().numpy() * self.gmm_weight_scale.cpu().numpy())), dim=1).to(torch.float32).to(self.device))
                else:
                    self.gmm_scale = None

            if self.mmx_quant_mode == 1 and self.mm_weight_quant_mode == 1:
                if self.mm_out_flag:
                    self.mm_scale = (torch.unsqueeze(torch.from_numpy(func.scale_generate(self.mm_x_scale.cpu().numpy() * self.mm_weight_scale.cpu().numpy())), dim=1).to(torch.float32).to(self.device))
                else:
                    self.mm_scale = None
        if "cpu" != self.scene and (self.permute_out_dtype == "fp4_e2m1" or self.permute_out_dtype == "fp4_e1m2"):
            self.alltoallv_output = torch.zeros(
                [self.permute_out_shape_list[self.rank][0], self.h1 >> 1],
                dtype=TORCH_DTYPE_MAP[self.gmm_x_dtype],
            ).to(self.device)
        else:
            self.alltoallv_output = torch.zeros(
                [self.permute_out_shape_list[self.rank][0], self.h1],
                dtype=TORCH_DTYPE_MAP[self.gmm_x_dtype],
            ).to(self.device)

    def get_npu(self):
        gmm_out, mm_out, permute_out = torch_npu.npu_alltoallv_quant_gmm(
            gmm_x=self.gmm_x,
            gmm_weight=self.gmm_weight,
            gmm_x_scale=self.gmm_x_scale,
            gmm_weight_scale=self.gmm_weight_scale,
            hcom=self.hcomm_info,
            ep_world_size=self.world_size,
            send_counts=self.get_send_counts(),
            recv_counts=self.get_recv_counts(),
            gmm_y_dtype=OUT_DTYPE_ENUM[self.gmm_y_dtype],
            mm_y_dtype=OUT_DTYPE_ENUM[self.mm_y_dtype] if self.mm_out_flag else None,
            mm_x=self.mm_x if self.mm_out_flag else None,
            mm_weight=self.mm_weight if self.mm_out_flag else None,
            mm_x_scale=self.mm_x_scale if self.mm_out_flag else None,
            mm_weight_scale=self.mm_weight_scale if self.mm_out_flag else None,
            permute_out_flag=self.permute_out_flag,
            gmm_x_quant_mode=self.gmmx_quant_mode,
            gmm_weight_quant_mode=self.gmm_weight_quant_mode,
            mm_x_quant_mode=self.mmx_quant_mode if self.mm_out_flag else None,
            mm_weight_quant_mode=self.mm_weight_quant_mode if self.mm_out_flag else None,
            gmm_x_dtype=DTYPE_ENUM[self.gmm_x_dtype],
            gmm_weight_dtype=DTYPE_ENUM[self.gmm_weight_dtype],
            mm_x_dtype=DTYPE_ENUM[self.mm_x_dtype] if self.mm_out_flag else None,
            mm_weight_dtype=DTYPE_ENUM[self.mm_weight_dtype] if self.mm_out_flag else None,
            gmm_x_scale_dtype=DTYPE_ENUM[self.gmm_x_scale_dtype],
            gmm_weight_scale_dtype=DTYPE_ENUM[self.gmm_weight_scale_dtype],
            mm_x_scale_dtype=DTYPE_ENUM[self.mm_x_scale_dtype] if self.mm_out_flag else None,
            mm_weight_scale_dtype=DTYPE_ENUM[self.mm_weight_scale_dtype] if self.mm_out_flag else None,
            group_size=self.group_size_dec,
        )
        self.npu_gmm_y = gmm_out
        self.npu_mm_y = mm_out if self.mm_out_flag else None
        self.npu_permute_out = permute_out if self.permute_out_flag else None

        return self.npu_gmm_y

    def get_hccl_mm(self):
        """小算子级联实现（带量化）"""
        # alltoallv
        dist.all_to_all_single(
            self.alltoallv_output,
            input=self.gmm_x,
            output_split_sizes=self.output_split_sizes,
            input_split_sizes=self.input_split_sizes,
        )
        # permute
        self.hccl_permute_out = self.permute(self.alltoallv_output)        
        if self.is_mxfp:
            # MX量化分支
            # gmm_x_scale的alltoall通信
            dist.all_to_all_single(
                self.alltoallv_scale_output,
                input=self.gmm_x_scale_reshaped,
                output_split_sizes=self.output_split_sizes,
                input_split_sizes=self.input_split_sizes,
            )
            self.per_token_scale_list = [self.permute(self.alltoallv_scale_output)]

        self.hccl_gmm_y = torch_npu.npu_grouped_matmul(
            x=[self.hccl_permute_out],
            weight=[self.gmm_weight],
            bias=None,
            scale=self.hccl_scale_list,
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
            scale_dtype=DTYPE_ENUM[self.gmm_weight_scale_dtype],
            per_token_scale_dtype=DTYPE_ENUM[self.gmm_x_scale_dtype],
        )[0]

        # 共享专家
        if self.mm_out_flag:
            self.hccl_mm_y = torch_npu.npu_quant_matmul(
                x1=self.mm_x,
                x2=self.mm_weight,
                pertoken_scale=self.mm_x_scale,
                scale=self.hccl_mm_weight_scale,
                offset=None,
                bias=None,
                output_dtype=OUT_DTYPE_ENUM[self.mm_y_dtype],
                x1_dtype=DTYPE_ENUM[self.mm_x_dtype],
                x2_dtype=DTYPE_ENUM[self.mm_weight_dtype],
                pertoken_scale_dtype=DTYPE_ENUM[self.mm_x_scale_dtype],
                scale_dtype=DTYPE_ENUM[self.mm_weight_scale_dtype],
                group_sizes=self.group_size_dec,
                y_scale=None,
            )
        else:
            self.hccl_mm_y = None

        
        if not self.permute_out_flag:
            self.hccl_permute_out = None
            hccl_permute_out_save = self.hccl_permute_out
        else:
            hccl_permute_out_save = self.hccl_permute_out.view(torch.uint8) if self.is_mxfp else self.hccl_permute_out
        return (
            self.hccl_gmm_y,
            None,
            None,
            self.hccl_mm_y,
            hccl_permute_out_save,
        )

    def mx_get_cpu(self):
        self.alltoallv_scale_output = self.alltoallv_scale_output.to(torch.float32)
        dist.all_to_all_single(
            self.alltoallv_scale_output,
            input=self.gmm_x_scale.reshape(self.bsk, -1),
            output_split_sizes=self.output_split_sizes,
            input_split_sizes=self.input_split_sizes,
        )
        # 对 scale 执行 permute，得到每个专家对应的 scale
        if self.permute_out_shape_list[self.rank][0] == 0:
            self.cpu_permute_scale_out = self.permute(
                torch.empty(0, 0)
            )
        else:
            self.cpu_permute_scale_out = self.permute(
                self.alltoallv_scale_output.reshape(self.permute_out_shape_list[self.rank][0], -1)
            )
        self.cpu_permute_out = self.permute(self.alltoallv_output)
        cpu_gmm_y_tmp = []
        if self.trans_gmm_weight_flag:
            self.gmm_weight_scale = self.gmm_weight_scale.permute(0, 2, 1, 3)
        for e, expert_tokens in enumerate(self.cpu_permute_out):
            if self.permute_out_shape_list[self.rank][0] == 0:
                out = torch.empty(0, 0)
                cpu_gmm_y_tmp.append(out)
                continue
            expert_scale = self.cpu_permute_scale_out[e]
            weight_scale = self.gmm_weight_scale[e].permute(0, 2, 1).reshape(-1, self.n1)
            out = self.mxfp_cpu_compute(
                expert_tokens.cpu().numpy(),
                self.gmm_weight[e].cpu().numpy(),
                expert_scale.cpu().numpy(),
                weight_scale.cpu().numpy()
            )
            cpu_gmm_y_tmp.append(out)
        self.cpu_gmm_y = torch.cat(cpu_gmm_y_tmp, dim=0)
        self.cpu_permute_out = torch.cat(self.cpu_permute_out, dim=0)

        if self.mm_out_flag:
            x_scale = self.mm_x_scale.reshape(self.bs, -1)
            if self.trans_mm_weight_flag:
                weight_scale = self.mm_weight_scale.reshape(self.n2, -1).transpose(0, 1)
            else:
                weight_scale = self.mm_weight_scale.permute(0, 2, 1).contiguous().reshape(-1, self.n2)
            mm_out = self.mxfp_cpu_compute(
                self.mm_x.cpu().numpy(),
                self.mm_weight.cpu().numpy(),
                x_scale.cpu().numpy(),
                weight_scale.cpu().numpy()
            )
            self.cpu_mm_y = mm_out
        else:
            self.cpu_mm_y = None
        

    def get_cpu(self):
        """CPU 标杆计算（支持 per-tensor 和 MX 量化）"""
        # 升精度到 float32
        self.gmm_x = self.gmm_x.to(torch.float32)
        self.gmm_weight = self.gmm_weight.to(torch.float32)
        if self.mm_out_flag:
            self.mm_x = self.mm_x.to(torch.float32)
            self.mm_weight = self.mm_weight.to(torch.float32)
        self.alltoallv_output = self.alltoallv_output.to(torch.float32)

        # ---------- alltoallv 通信 ----------
        dist.all_to_all_single(
            self.alltoallv_output,
            input=self.gmm_x,
            output_split_sizes=self.output_split_sizes,
            input_split_sizes=self.input_split_sizes,
        )

        if self.is_mxfp:
            # MX量化分支
            self.mx_get_cpu()
        else:
            # 非MX量化分支
            # ---------- 对 token 执行 permute ----------
            self.cpu_permute_out = self.permute(self.alltoallv_output)

            # ---------- 路由专家计算 ----------
            cpu_gmm_y_tmp = []
            for e, expert_tokens in enumerate(self.cpu_permute_out):
                out = torch.matmul(expert_tokens, self.gmm_weight[e])
                cpu_gmm_y_tmp.append(out)

            self.cpu_gmm_y = torch.cat(cpu_gmm_y_tmp, dim=0)
            self.cpu_gmm_y = self.cpu_gmm_y * self.gmm_scale
            self.cpu_permute_out = torch.cat(self.cpu_permute_out, dim=0)

            # ---------- 共享专家计算 ----------
            if self.mm_out_flag:
                mm_out = torch.mm(self.mm_x, self.mm_weight)
                mm_out = mm_out * self.mm_scale
                self.cpu_mm_y = mm_out
            else:
                self.cpu_mm_y = None

        # ---------- 保存输出 ----------
        self.save(self.cpu_gmm_y, self.output_path, "cpu_out.bin")
        if self.mm_out_flag:
            self.save(self.cpu_mm_y, self.output_path, "cpu_mm_out.bin")
        if self.permute_out_flag:
            self.save(self.cpu_permute_out, self.output_path, "cpu_permute_out.bin")

    def get_graph(self, type="mc2"):
        model = AlltoAllvQuantGroupedMatmulGraphModel()
        model = self.define_model(model, self.graph_type)

        __send_counts = [int(x) for x in self.get_send_counts()]
        __recv_counts = [int(x) for x in self.get_recv_counts()]

        if type == "mc2":
            out_graph = model(
                _gmm_x=self.gmm_x,
                _gmm_weight=self.gmm_weight_before_trans,
                _gmm_x_scale=self.gmm_x_scale,
                _gmm_weight_scale=self.gmm_weight_scale,
                _hcom=self.hcomm_info,
                _ep_world_size=self.world_size,
                _send_counts=__send_counts,
                _recv_counts=__recv_counts,
                _gmm_y_dtype=OUT_DTYPE_ENUM[self.gmm_y_dtype],
                _mm_y_dtype=OUT_DTYPE_ENUM[self.mm_y_dtype] if self.mm_out_flag else None,
                _mm_x=self.mm_x if self.mm_out_flag else None,
                _mm_weight=self.mm_weight_before_trans if self.mm_out_flag else None,
                _mm_x_scale=self.mm_x_scale if self.mm_out_flag else None,
                _mm_weight_scale=self.mm_weight_scale if self.mm_out_flag else None,
                _permute_out_flag=self.permute_out_flag,
                _gmm_x_quant_mode=self.gmmx_quant_mode,
                _gmm_weight_quant_mode=self.gmm_weight_quant_mode,
                _mm_x_quant_mode=self.mmx_quant_mode if self.mm_out_flag else None,
                _mm_weight_quant_mode=self.mm_weight_quant_mode if self.mm_out_flag else None,
                _gmm_x_dtype=DTYPE_ENUM[self.gmm_x_dtype],
                _gmm_weight_dtype=DTYPE_ENUM[self.gmm_weight_dtype],
                _mm_x_dtype=DTYPE_ENUM[self.mm_x_dtype] if self.mm_out_flag else None,
                _mm_weight_dtype=(DTYPE_ENUM[self.mm_weight_dtype]) if self.mm_out_flag else None,
                type=type,
                _alltoallv_output=self.alltoallv_output,
                _output_split_sizes=self.output_split_sizes,
                _input_split_sizes=self.input_split_sizes,
                _scale_list=self.scale_list,
                _per_token_scale_list=self.per_token_scale_list,
                _group_list=self.group_list,
                _gmm_weight_scale_dtype=DTYPE_ENUM[self.gmm_weight_scale_dtype],
                _gmm_x_scale_dtype=DTYPE_ENUM[self.gmm_x_scale_dtype],
                _mm_x_scale_dtype=DTYPE_ENUM[self.mm_x_scale_dtype] if self.mm_out_flag else None,
                _mm_weight_scale_dtype=DTYPE_ENUM[self.mm_weight_scale_dtype] if self.mm_out_flag else None,
                _mm_out_flag=self.mm_out_flag,
                _e=self.e,
                _rank=self.rank,
                _device=self.device,
                _expTokenNums=self.expTokenNums,
                _trans_gmm_weight=self.trans_gmm_weight_flag,
                _trans_mm_weight=self.trans_mm_weight_flag,
                _group_size=self.group_size_dec)
            self.npu_gmm_y, self.npu_mm_y, self.npu_permute_out = out_graph
        elif type == "hccl_mm":
            self.hccl_gmm_y, _, _, self.hccl_mm_y, self.hccl_permute_out = self.get_hccl_mm()
        else:
            print("In AlltoAllvQuantGroupedMatmul get_graph, wrong type.")

    def compare(self, rank=0, check_md5=0):
        if len(self.recv_count_zero_rank_list) > 0:
            torch.distributed.barrier()
        if self.permute_out_flag and self.scene not in ["aclnn"] and (self.permute_out_dtype == "fp4_e2m1" or self.permute_out_dtype == "fp4_e1m2"):
            if self.npu_permute_out.numel() == 0:
                self.hccl_permute_out = torch.empty(0, dtype=torch.float32)
                self.npu_permute_out = torch.empty(0, dtype=torch.float32)
            else:
                # 优化：减少中间内存分配，一次性完成 unpack + trans + reshape
                hccl_permute_out_unpacked = self.unpack_int4(self.hccl_permute_out.contiguous().cpu().numpy())
                npu_permute_out_unpacked = self.unpack_int4(self.npu_permute_out.cpu().numpy())
                self.hccl_permute_out = torch.from_numpy(
                    self.trans_to_float32(hccl_permute_out_unpacked, self.permute_out_dtype))
                self.npu_permute_out = torch.from_numpy(
                    self.trans_to_float32(npu_permute_out_unpacked, self.permute_out_dtype))
        """比对复用父类逻辑"""
        return super().compare(rank, check_md5)

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

    def process_aclnn(self):
        self.root_path = os.path.abspath(os.getcwd())
        self.input_tensor_format = "ND,ND,ND,ND,ND,ND"
        self.loop = 1
        def fmt_dtype(dtype_list):
            if isinstance(dtype_list, (list, tuple)):
                return ",".join(str(d).replace("'", "").replace(" ", "") for d in dtype_list)
            return str(dtype_list).replace("'", "").replace(" ", "")
        self.input_tensor_dtype = fmt_dtype(self.case["input_tensor_dtype"])
        self.output_tensor_dtype = fmt_dtype(self.case["output_tensor_dtype"])
        self.input_scales_dtype = fmt_dtype(self.case["input_scales_dtype"])
        result = subprocess.check_output(["uname", "-a"]).decode("utf-8")
        arch = "aarch" if "aarch64" in result.lower() else "x86"
        exe_binary = "cpp/bin/" + self.exe_bin_name + "_" + arch
        tmp_binary = self.exe_bin_name + f"_{arch}"
        command = [f"{self.root_path}/bin_path/{tmp_binary}"]
        command.append(f"--bin_path {self.bin_path}")
        command.append(f"--gmm_x_shape {','.join(map(str, self.gmm_x_shape))}")
        command.append(f"--gmm_weight_shape {','.join(map(str, self.gmm_weight_shape))}")
        command.append(f"--gmm_x_scale_shape {','.join(map(str, self.gmm_x_scale_shape))}")
        command.append(f"--gmm_weight_scale_shape {','.join(map(str, self.gmm_weight_scale_shape))}")
        command.append(f"--send_counts_shape {str(len(self.get_send_counts()))}")
        command.append(f"--recv_counts_shape {str(len(self.get_recv_counts()))}")
        command.append(f"--mm_x_shape {','.join(map(str, self.mm_x_shape))}")
        command.append(f"--mm_weight_shape {','.join(map(str, self.mm_weight_shape))}")
        command.append(f"--mm_x_scale_shape {','.join(map(str, self.mm_x_scale_shape))}")
        command.append(f"--mm_weight_scale_shape {','.join(map(str, self.mm_weight_scale_shape))}")
        command.append(f"--input_tensor_dtype {self.input_tensor_dtype}")
        command.append(f"--input_tensor_format {self.input_tensor_format}")
        command.append(f"--input_scales_dtype {self.input_scales_dtype}")
        command.append(f"--input_scales_format ND,ND,ND,ND")
        command.append(f"--output_tensor_dtype {self.output_tensor_dtype}")
        command.append(f"--output_tensor_format ND,ND,ND")
        command.append(f"--ep_world_size {self.world_size}")
        command.append(f"--trans_gmm_weight_flag {str(self.trans_gmm_weight_flag).lower()}")
        command.append(f"--trans_mm_weight_flag {str(self.trans_mm_weight_flag).lower()}")
        command.append(f"--n_dev {self.world_size}")
        command.append(f"--loop_cnt {self.loop}")
        command.append(f"--mm_out_flag {str(self.mm_out_flag).lower()}")
        command.append(f"--permute_out_flag {str(self.permute_out_flag).lower()}")
        command.append(f"--gmm_x_quant_mode {self.gmmx_quant_mode}")
        command.append(f"--gmm_weight_quant_mode {self.gmm_weight_quant_mode}")
        command.append(f"--mm_x_quant_mode {self.mmx_quant_mode}")
        command.append(f"--mm_weight_quant_mode {self.mm_weight_quant_mode}")
        processes = []
        for i in range(self.world_size):
            tmp_command = command.copy()
            tmp_command.append(f"--mm_y_shape {','.join(map(str, self.mm_y_shape))}")
            tmp_command.append(f"--gmm_y_shape {','.join(map(str, self.gmm_y_shape_list[i]))}")
            tmp_command.append(f"--permute_out_shape {','.join(map(str, self.permute_out_shape_list[i]))}")
            tmp_command.append(f"--rankId {i}")
            aclnn_command = " ".join(tmp_command)
            proc = subprocess.Popen(aclnn_command, shell=True)
            processes.append(proc)
        for i, proc in enumerate(processes):
            proc.wait()