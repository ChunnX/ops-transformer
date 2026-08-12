# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
import torch
import torch.nn.functional as F
import numpy as np
import torch_npu
import os
import sys
import hashlib
from atk.common.connection import RemoteManager
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
import ctypes
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat, AclTensorStruct
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
import random
from atk.common.log import Logger
import json

# import operation_test
# import golden_cpu

from enum import Enum
from atk.configs.results_config import AccuracyConfig
from atk.tasks.post_process.base_compare import BaseAccuracyCompare
from atk.tasks.post_process import ACCURACY_REGISTRY
from atk.tasks.post_process.double_benchmark_compare import DoubleBenchmarkAccuracyCompare

ATB_HOME_PATH = os.environ.get("ATB_HOME_PATH")
if ATB_HOME_PATH is None:
    raise RuntimeError(
        "env ATB_HOME_PATH not exist, source set_env.sh")

sys.path.append(os.path.join(os.path.dirname(__file__), "./pythontools"))

# import tensor_file  # NOQA:E402

LIB_PATH = os.path.join(ATB_HOME_PATH, "lib/libatb_test_framework.so")
torch.classes.load_library(LIB_PATH)


logging = Logger().get_logger()

# torch.set_printoptions(
#     precision=4,           # 小数点精度
#     threshold=float('inf'),  # 不限制打印元素数量
#     linewidth=120,         # 每行最大字符数，避免换行混乱
#     sci_mode=False         # 关闭科学计数法（可选）
# )


try:
    import torch_npu
    _HAS_TORCH_NPU = True
except ImportError:
    _HAS_TORCH_NPU = False


QUANTMAX = 127
QUANTMIN = -128
BLOCK_NUM = 192
BLOCK_SIZE = 128

# common_file_save_path="/home/z00939187/atk_save_data/mlapo/case168"

def round_up(val: int, align: int) -> int:
    if align == 0:
        return 0
    return -(val // -align) * align

def transdata_nd2nz(x: torch.Tensor) -> torch.Tensor:
    align = 32 if x.dtype == torch.int8 else 16
    x = x if x.dim() == 3 else x.unsqueeze(0)

    batch, row, col = x.shape[0], round_up(x.shape[1], 16), round_up(x.shape[2], align)
    pad_row = row - x.shape[1]
    pad_col = col - x.shape[2]

    ret = F.pad(x, ((0, pad_row, 0, pad_col)))
    ret = torch.reshape(ret, (batch, row // 16, 16, col // align, align))
    ret = torch.permute(ret, [0, 3, 1, 2, 4])
    ret = torch.reshape(ret, (ret.shape[0], ret.shape[1], ret.shape[2] * ret.shape[3], ret.shape[4]))
    return ret

def reshape_and_cache_cpu(
    key: torch.Tensor,
    key_rope: torch.Tensor,
    key_cache_in: torch.Tensor,
    key_cache_rope_in: torch.Tensor,
    slot_mapping: torch.Tensor,
    cache_mode: int,
) -> None:
    key_cache_in, key_cache_rope_in = key_cache_in.clone(), key_cache_rope_in.clone()
    valid_indicies = torch.where(slot_mapping >= 0)[0]
    valid_slots = slot_mapping[valid_indicies]

    key_nope_valid = key[valid_indicies]
    key_rope_valid = key_rope[valid_indicies]

    block_idx = valid_slots // BLOCK_SIZE
    block_offset = valid_slots % BLOCK_SIZE

    if cache_mode == 0:
        key_cache_in[block_idx, block_offset] = torch.unsqueeze(
            torch.cat((key_nope_valid, key_rope_valid), dim=-1), dim=1
        )
        return key_cache_in, torch.zeros_like(key_cache_in)
    key_cache_in[block_idx, block_offset] = torch.unsqueeze(key_nope_valid, dim=1)
    key_cache_rope_in[block_idx, block_offset] = torch.unsqueeze(key_rope_valid, dim=1)
    return key_cache_in, key_cache_rope_in

def rms_norm_cpu(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor = None,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    square_sum = torch.sum(torch.square(x.float()), axis=-1, keepdims=True)
    rms_factor = 1.0 / torch.sqrt(square_sum / x.shape[-1] + epsilon)
    ret = x * rms_factor * gamma.float()
    if beta != None:
        ret += beta.float()
    return ret

def rms_norm_quant_cpu(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    quant_scale: torch.Tensor = None,
    quant_offset: torch.Tensor = None,
):
    # RmsNorm
    x_norm = rms_norm_cpu(x, gamma, beta)

    # quant
    dynamic_quant = quant_scale == None and quant_offset == None
    if dynamic_quant:
        max_abs_val, _ = torch.max(torch.abs(x_norm), dim=-1, keepdim=True)
        scale = 127.0 / max_abs_val
        ret = torch.clamp(torch.round(x_norm * scale).half(), min=-128, max=127)
        return ret, 1 / scale
    else:
        quant_scale = 1.0 / quant_scale.float().item()
        quant_offset = quant_offset.float().item()
        ret = x_norm * quant_scale + quant_offset
        ret = torch.clamp(torch.round(ret).half(), min=-128, max=127).to(torch.int8)
    return ret

def mm_w8a8_cpu(
    x: torch.Tensor,
    weight: torch.Tensor,
    deq_scale: torch.Tensor,
    per_token_scale: torch.Tensor = None,
    deq_offset: torch.Tensor = None,
) -> torch.Tensor:
    ret = torch.matmul(x.to(torch.int32), weight.transpose(0, 1).to(torch.int32))

    deq_scale = deq_scale.reshape((1, -1))
    deq_offset = deq_offset.reshape((1, -1))
    if per_token_scale != None:
        ret = ret.to(torch.float32) * deq_scale * per_token_scale
    else:
        ret += deq_offset
        ret = ret.to(torch.float32) * deq_scale
    return ret

def quant_cpu(
    x: torch.Tensor,
    scale: torch.Tensor,
    offset: torch.Tensor = None,
) -> torch.Tensor:
    x = x.to(torch.float32)
    scale = 1.0 / scale.to(torch.float32)

    ret = torch.round(x * scale)
    if offset != None:
        ret += offset
    ret = torch.clamp(torch.round(ret).to(torch.int32), min=-128, max=127)
    return ret.to(torch.int8)

def mm_cpu(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    x = x.to(torch.float32)
    weight = weight.transpose(1, 0).to(torch.float32)
    return torch.matmul(x, weight)

def mm_einsum_cpu(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    x = torch.permute(x, (1, 0, 2)).to(torch.float32)
    weight = weight.to(torch.float32)
    return torch.permute(torch.matmul(x, weight), (1, 0, 2))

def rope_cpu(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:

    def rotate_half(x):
        x0, x1 = x.chunk(2, -1)
        return torch.cat((-x1, x0), dim=x0.ndim - 1)

    x = x.to(torch.float32)
    sin = sin.to(torch.float32)
    cos = cos.to(torch.float32)

    ntoken, head_num, head_size = x.shape
    batch = ntoken
    x = x.view(batch, 1, head_num, head_size)
    cos = cos.view(batch, 1, head_size).unsqueeze(2)
    sin = sin.view(batch, 1, head_size).unsqueeze(2)
    return ((x * cos) + (rotate_half(x) * sin)).view(ntoken, head_num, head_size)

@register("function_aclnn_mla_preprocess")
class FunctionMlaPreprocessApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionMlaPreprocessApi, self).__init__(task_result)
        self.set_seeds(2025)


    def set_seeds(self, seed: int):
        """设置所有随机源的种子"""
        # 1. Python built-in random
        random.seed(seed)
        
        # 2. NumPy
        np.random.seed(seed)
        
        # 3. PyTorch
        torch.manual_seed(seed)
        
        # 4. 如果使用 CUDA/NPU，也设置 GPU 种子
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # 多 GPU
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        # 5. Ascend NPU（如果使用华为 NPU）
        if hasattr(torch, 'npu') and torch.npu.is_available():
            torch.npu.manual_seed(seed)
            torch.npu.manual_seed_all(seed)


    def process_deq_scale(self, deq_scale: torch.Tensor) -> torch.Tensor:
        ret = torch.frombuffer(deq_scale.numpy().tobytes(), dtype=torch.int32).to(torch.int64)
        return ret
        
    def ein_sum_out_quant_golden(self, input, scale):
        scale = scale[None, : ,None]
        if (scale.dtype == torch.bfloat16):
            input = input.float()
            scale = scale.float()
        quant = input * scale
        output = torch.round(quant.float()).half()
        output = torch.min(output, torch.tensor(QUANTMAX, dtype=torch.half))
        output = torch.max(output, torch.tensor(QUANTMIN, dtype=torch.half))
        return output.to(torch.int8)

    def __to_npu(self, tensor_list):
        return [x.contiguous().npu() for x in tensor_list]

    def __to_cpu(self, tensor_list):
        return [x.cpu() for x in tensor_list]

    def __npu_execute(self, op_name, op_params, in_tensors, out_tensors):
        in_tensors_npu = self.__to_npu(in_tensors)
        out_tensors_npu = [in_tensors_npu[x] if isinstance(x, int) else x.npu() for x in out_tensors]
        op = torch.classes.OperationTorch.OperationTorch(op_name)
        op.set_param(op_params)
        op.execute_out(in_tensors_npu, out_tensors_npu)
        return self.__to_cpu(out_tensors_npu)


    def quant_npu(
        self,
        x: torch.Tensor,
        quant_scale: torch.Tensor,
        quant_offset: torch.Tensor = None,
    ):
        if quant_offset == None:
            quant_offset = torch.Tensor([]).to(torch.int8)

        out_tensors = self.__npu_execute(
            op_name="ElewiseOperation",
            op_params=json.dumps({"elewiseType": 17}),
            in_tensors=[x, quant_scale, quant_offset],
            out_tensors=[torch.zeros_like(x, dtype=torch.int8)],
        )
        return out_tensors[0]

    def einsum_quant_npu(
        self,
        input: torch.Tensor,
        quant_scale: torch.Tensor,
    ):
        quant_scale = quant_scale[None, : ,None].cpu()

        if (quant_scale.dtype == torch.bfloat16):
            input = input.float()
            quant_scale = quant_scale.float()

        ret = input * quant_scale
        ret = torch.round(ret).half()
        ret = torch.min(ret, torch.tensor(127, dtype=torch.half))
        ret = torch.max(ret, torch.tensor(-128, dtype=torch.half))
        ret = ret.cpu()

        return ret.to(torch.int8)

    def rms_norm_quant_npu(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        quant_scale: torch.Tensor,
        quant_offset: torch.Tensor,
    ):
        out_tensors = self.__npu_execute(
            op_name="RmsNormOperation",
            op_params=json.dumps({"layerType": 1, "normParam": {"quantType": 2, "epsilon": 1e-6}}),
            in_tensors=[x, gamma, beta, quant_scale, quant_offset],
            out_tensors=[torch.zeros_like(x, dtype=torch.int8)],
        )
        return out_tensors[0]

    def mm_npu(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
    ):
        m, n = mat_a.shape[0], mat_b.shape[0]
        out_tensors = self.__npu_execute(
            op_name="LinearOperation",
            op_params=json.dumps(
                {"transposeA": False, "transposeB": True, "hasBias": False, "enAccum": False, "outDataType": -1}
            ),
            in_tensors=[mat_a, mat_b],
            out_tensors=[torch.zeros(size=(m, n), dtype=mat_a.dtype)],
        )
        return out_tensors[0]

    def mm_w8a8_npu(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        deq_scale: torch.Tensor,
        deq_offset: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        m, n = mat_a.shape[0], mat_b.shape[0]
        dtype_to_int = lambda out_dtype: 27 if out_dtype == torch.bfloat16 else 1

        if out_dtype == torch.float16:
            deq_scale = self.process_deq_scale(deq_scale.cpu()).npu()

        out_tensors = self.__npu_execute(
            op_name="LinearOperation",
            op_params=json.dumps(
                {"transposeA": False, "transposeB": True, "hasBias": True, "outDataType": dtype_to_int(out_dtype)}
            ),
            in_tensors=[mat_a, mat_b, deq_offset, deq_scale],
            out_tensors=[torch.zeros(size=(m, n), dtype=out_dtype)],
        )
        return out_tensors[0]

    def mm_einsum_npu(
        self,
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
    ):
        batch, m, n = mat_a.shape[1], mat_a.shape[0], mat_b.shape[2]
        out_tensors = self.__npu_execute(
            op_name="LinearOperation",
            op_params=json.dumps(
                {"matmulType": 1, "transposeA": False, "transposeB": False, "hasBias": False, "outDataType": -1}
            ),
            in_tensors=[mat_a, mat_b],
            out_tensors=[torch.zeros(size=(m, batch, n), dtype=mat_a.dtype)],
        )
        return out_tensors[0]

    def rope_q_concat_npu(self, query: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, einsum_out: torch.Tensor):
        out_shape = (query.shape[0], query.shape[1], query.shape[2] + einsum_out.shape[2])
        out_tensors = self.__npu_execute(
            op_name="RopeQConcatOperation",
            op_params=json.dumps({}),
            in_tensors=[query.reshape((query.shape[0], -1)), cos, sin, einsum_out],
            out_tensors=[torch.zeros(size=out_shape, dtype=query.dtype)],
        )
        return out_tensors[0]

    def rope_npu(
        self,
        query: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        seqlen = torch.ones(size=(query.shape[0],), dtype=torch.int32)
        out_tensors = self.__npu_execute(
            op_name="RopeOperation",
            op_params=json.dumps({"rotaryCoeff": 2}),
            in_tensors=[query, torch.zeros_like(query, dtype=query.dtype), cos, sin, seqlen],
            out_tensors=[torch.zeros_like(query), torch.zeros_like(query)],
        )
        return out_tensors[0]

    def rms_norm_npu(
        self,
        x: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        out_tensors = self.__npu_execute(
            op_name="RmsNormOperation",
            op_params=json.dumps({"layerType": 1, "normParam": {"epsilon": 1e-6}}),
            in_tensors=[x, gamma],
            out_tensors=[torch.zeros_like(x)],
        )
        return out_tensors[0]

    def reshape_and_cache_npu(self, key: torch.Tensor, key_cache: torch.Tensor, slot_mapping: torch.Tensor):
        key_cache = key_cache.clone()
        out_tensors = self.__npu_execute(
            op_name="ReshapeAndCacheOperation",
            op_params=json.dumps({"kvCacheCfg": 1}),
            in_tensors=[key, key_cache, slot_mapping],
            out_tensors=[1],
        )
        return out_tensors[0]     

    def transdata(self, nd_mat, block_size: tuple = (16, 16)):
        r = round_up(nd_mat.shape[0], block_size[0])
        c = round_up(nd_mat.shape[1], block_size[1])
        r_pad = r - nd_mat.shape[0]
        c_pad = c - nd_mat.shape[1]
        nd_mat = F.pad(nd_mat, (0, r_pad, 0, c_pad))
        nz_mat = torch.permute(
            torch.reshape(nd_mat, (r // block_size[0], block_size[0], c // block_size[1], block_size[1])), [2, 0, 1, 3]
        )
        nz_mat = torch.reshape(nz_mat, (nz_mat.shape[0], nz_mat.shape[1] * nz_mat.shape[2], nz_mat.shape[3]))
        return nz_mat


    def init_by_input_data(self, input_data: InputDataset):
        """
        该接口可实现部门场景下api的初始化需要依赖于当前的输入数据，且不希望计入耗时，
        可以在此接口实现
        :param input_data:
        :return:
        """
        input_token_num = input_data.kwargs["input"].shape[0]
        # print("intput_token_num = ",input_token_num)
        #第一次执行需要生成slotmapping
        input_data.kwargs["slotmapping"] = torch.from_numpy(np.random.choice(192 * 128, input_token_num, replace=False).astype(np.int32)).to(torch.int32)
        # input_data.kwargs["slotmapping"] = torch.arange(input_token_num).to(torch.int32)
       
        CACHEMODE = input_data.kwargs["cacheMode"]
        
        if self.device == "npu":

            input_data.kwargs["input"] = input_data.kwargs["input"].npu()
            input_data.kwargs["gamma0"] = input_data.kwargs["gamma0"].npu()
            input_data.kwargs["beta0"] = input_data.kwargs["beta0"].npu()
            input_data.kwargs["quantScale0"] = input_data.kwargs["quantScale0"].npu()
            input_data.kwargs["quantOffset0"] = input_data.kwargs["quantOffset0"].npu()
            input_data.kwargs["wdqkv"]= input_data.kwargs["wdqkv"].npu()
            input_data.kwargs["deScale0"] = input_data.kwargs["deScale0"].npu()
            input_data.kwargs["bias0"] = input_data.kwargs["bias0"].npu()
            input_data.kwargs["gamma1"] = input_data.kwargs["gamma1"].npu()
            input_data.kwargs["beta1"] = input_data.kwargs["beta1"].npu()
            input_data.kwargs["quantScale1"] = input_data.kwargs["quantScale1"].npu()
            input_data.kwargs["quantOffset1"] = input_data.kwargs["quantOffset1"].npu()
            input_data.kwargs["wuq"]= input_data.kwargs["wuq"].npu()
            input_data.kwargs["deScale1"] = input_data.kwargs["deScale1"].npu()
            input_data.kwargs["bias1"] = input_data.kwargs["bias1"].npu()
            input_data.kwargs["gamma2"] = input_data.kwargs["gamma2"].npu()
            input_data.kwargs["cos"] = input_data.kwargs["cos"].npu()
            input_data.kwargs["sin"] = input_data.kwargs["sin"].npu()
            input_data.kwargs["wuk"] = input_data.kwargs["wuk"].npu()
            input_data.kwargs["kvCache"] = input_data.kwargs["kvCache"].npu()
            input_data.kwargs["kvCacheRope"] = input_data.kwargs["kvCacheRope"].npu()
            input_data.kwargs["slotmapping"] = input_data.kwargs["slotmapping"].npu()
            input_data.kwargs["ctkvScale"] = input_data.kwargs["ctkvScale"].npu()
            input_data.kwargs["qNopeScale"] = input_data.kwargs["qNopeScale"].npu()

        if self.device == "pyaclnn":
            # wuq = input_data.kwargs["wuq"].cpu().clone()
            # wuq = self.transdata(wuq, (16, 32)).contiguous()
            # wdqkv = input_data.kwargs["wdqkv"].cpu().clone()
            npu_cast_to_nz = lambda tensor: torch_npu.npu_format_cast(transdata_nd2nz(tensor).contiguous().npu(), 29)
            if input_data.kwargs["input"].dtype == torch.float16 :
                deScale0 = self.process_deq_scale(input_data.kwargs["deScale0"].cpu()).npu()
                deScale1 = self.process_deq_scale(input_data.kwargs["deScale1"].cpu()).npu()
            else:
                deScale0 = input_data.kwargs["deScale0"]
                deScale1 = input_data.kwargs["deScale1"]

            input_data.kwargs["input"] = input_data.kwargs["input"].npu()
            input_data.kwargs["gamma0"] = input_data.kwargs["gamma0"].npu()
            input_data.kwargs["beta0"] = input_data.kwargs["beta0"].npu()
            input_data.kwargs["quantScale0"] = input_data.kwargs["quantScale0"].npu()
            input_data.kwargs["quantOffset0"] = input_data.kwargs["quantOffset0"].npu()
            #转NZ格式
            # input_data.kwargs["wdqkv"]= torch_npu.npu_format_cast(wdqkv.npu(), 29).npu()
            input_data.kwargs["wdqkv"]= npu_cast_to_nz(input_data.kwargs["wdqkv"])

            # if input_data.kwargs["input"].dtype == torch.float16 :
            #     input_data.kwargs["deScale0"] = deScale0.npu()
            # else:
            #     input_data.kwargs["deScale0"] = input_data.kwargs["deScale0"].npu()
            input_data.kwargs["deScale0"] = deScale0.npu()

            input_data.kwargs["bias0"] = input_data.kwargs["bias0"].npu()
            input_data.kwargs["gamma1"] = input_data.kwargs["gamma1"].npu()
            input_data.kwargs["beta1"] = input_data.kwargs["beta1"].npu()
            input_data.kwargs["quantScale1"] = input_data.kwargs["quantScale1"].npu()
            input_data.kwargs["quantOffset1"] = input_data.kwargs["quantOffset1"].npu()
            # input_data.kwargs["wuq"]= torch_npu.npu_format_cast(wuq.npu(), 29).npu()
            input_data.kwargs["wuq"] = npu_cast_to_nz(input_data.kwargs["wuq"])
            # if input_data.kwargs["input"].dtype == torch.float16 :
            #     input_data.kwargs["deScale1"] = deScale1.npu()
            # else:
            #     input_data.kwargs["deScale1"] = input_data.kwargs["deScale1"].npu()
                        
            input_data.kwargs["deScale1"] = deScale1.npu()
 
            input_data.kwargs["bias1"] = input_data.kwargs["bias1"].npu()
            input_data.kwargs["gamma2"] = input_data.kwargs["gamma2"].npu()
            input_data.kwargs["cos"] = input_data.kwargs["cos"].npu()
            input_data.kwargs["sin"] = input_data.kwargs["sin"].npu()
            input_data.kwargs["wuk"] = input_data.kwargs["wuk"].npu()
            input_data.kwargs["kvCache"] = input_data.kwargs["kvCache"].npu()
            input_data.kwargs["kvCacheRope"] = input_data.kwargs["kvCacheRope"].npu()

            input_data.kwargs["slotmapping"] = input_data.kwargs["slotmapping"].npu()
            input_data.kwargs["ctkvScale"] = input_data.kwargs["ctkvScale"].npu()
            input_data.kwargs["qNopeScale"] = input_data.kwargs["qNopeScale"].npu()





    def __call__(self, input_data: InputDataset, with_output: bool = False):
        
        post_node = self.task_result.nodes.get_main_node()
        remote_nodes = []
        for node in self.task_result.nodes.nodes:
            if node != post_node:
                remote_nodes.append(node)
        remote_manager = RemoteManager(self.task_result.case_config, post_node,remote_nodes[0] )
        local_dir, remote_dir = remote_manager.get_local_and_remote_dir()
        # self.lo
        self.local_dir = local_dir
        self.remote_dir = remote_dir

        self.cacheMode = input_data.kwargs["cacheMode"]
        self.input1 = input_data.kwargs["input"]
        self.gamma0 = input_data.kwargs["gamma0"]
        self.beta0 = input_data.kwargs["beta0"]
        self.quantScale0 = input_data.kwargs["quantScale0"]
        self.quantOffset0 = input_data.kwargs["quantOffset0"]
        self.wdqkv = input_data.kwargs["wdqkv"]
        self.deScale0 = input_data.kwargs["deScale0"]
        self.bias0 = input_data.kwargs["bias0"]
        self.gamma1 = input_data.kwargs["gamma1"]
        self.beta1 = input_data.kwargs["beta1"]
        self.quantScale1 = input_data.kwargs["quantScale1"]
        self.quantOffset1 = input_data.kwargs["quantOffset1"]
        self.wuq = input_data.kwargs["wuq"]
        self.deScale1 = input_data.kwargs["deScale1"]
        self.bias1 = input_data.kwargs["bias1"]
        self.gamma2 = input_data.kwargs["gamma2"]
        self.cos = input_data.kwargs["cos"]
        self.sin = input_data.kwargs["sin"]
        self.wuk = input_data.kwargs["wuk"]

        self.slotMapping = input_data.kwargs["slotmapping"]
        self.ctkvScale = input_data.kwargs["ctkvScale"]
        self.qNopeScale = input_data.kwargs["qNopeScale"]

        self.kvCache = input_data.kwargs["kvCache"]
        self.kvCacheRope = input_data.kwargs["kvCacheRope"]
        input_token_num = self.input1.shape[0]
        self.INPUT_HIDDEN_SIZE = self.input1.shape[1]
        head_num = self.wuk.shape[0]
        self.data_type = self.input1.dtype

        quant_mode = input_data.kwargs["quantMode"]
        self.input_token_num = input_token_num
        self.epsilon = 1e-6
        self.headNum = head_num

        # do_rms_norm = not (input_data.kwargs["wdqkv"].dtype == torch.bfloat16 and input_data.kwargs["wuq"].dtype == torch.bfloat16)
        do_rms_norm = input_data.kwargs["doRmsNorm"]
        if input_data.kwargs["wdqkv"].dtype == torch.int8 :
            en_w8a8 = True
        else:
            en_w8a8 = False

        # CPU标杆
        if self.device == "cpu":
            # print("-------------------enter CPU-----------------------")
            # print("-------------------------cpu slotmapping-----------------------")

            if en_w8a8:
                if do_rms_norm:
                    mm1_in = rms_norm_quant_cpu(self.input1, self.gamma0, self.beta0, self.quantScale0, self.quantOffset0)
                else:
                    mm1_in = quant_cpu(self.input1, self.quantScale0, self.quantOffset0)
                mm1_out = mm_w8a8_cpu(mm1_in, self.wdqkv, self.deScale0, None, self.bias0).to(self.data_type)
            else:
                # mm1_in = rms_norm(self.input, self.gamma1)
                mm1_out = mm_cpu(mm1_in, self.wdqkv).to(self.data_type)

            split1_out = [mm1_out[...,:512], mm1_out[..., 512:576], mm1_out[...,576:]]

            if en_w8a8:
                mm2_in = rms_norm_quant_cpu(split1_out[2], self.gamma1, self.beta1, self.quantScale1, self.quantOffset1)
                mm2_out = mm_w8a8_cpu(mm2_in, self.wuq, self.deScale1, None, self.bias1).to(self.data_type)
            else:
                mm2_in = rms_norm_cpu(split1_out[2], self.gamma1)
                mm2_out = mm_cpu(mm2_in, self.wuq).to(self.data_type)

            nope_dim = self.wuk.shape[1]
            rope_dim = 64
            mm2_out = mm2_out.reshape(self.input_token_num, self.headNum, nope_dim + rope_dim)
            split2_out = [mm2_out[..., :nope_dim], mm2_out[..., nope_dim:]]

            einsum_out = mm_einsum_cpu(split2_out[0], self.wuk).to(self.data_type)

            q_out = quant_cpu(einsum_out, self.qNopeScale[None, : ,None].cpu()) if self.cacheMode == 2 else einsum_out

            q_rope_out = rope_cpu(split2_out[1], self.cos, self.sin).to(self.data_type)

            rms3_out = rms_norm_cpu(split1_out[0], self.gamma2).to(self.data_type)

            if self.cacheMode == 2:
                key_out = quant_cpu(rms3_out, self.ctkvScale)
            else:
                key_out = rms3_out
            

            # rope (key)
            key_rope_out = rope_cpu(split1_out[1].unsqueeze(dim=1), self.cos, self.sin).to(self.data_type).squeeze(dim=1)


            self.kv_cache_out_cpu, self.kv_cache_rope_out_cpu = reshape_and_cache_cpu(
                key_out, key_rope_out, self.kvCache, self.kvCacheRope, self.slotMapping, self.cacheMode
            )

            if self.cacheMode > 1:
                self.kv_cache_out_cpu = transdata_nd2nz(self.kv_cache_out_cpu.reshape((BLOCK_NUM, BLOCK_SIZE, -1)))
                self.kv_cache_rope_out_cpu = transdata_nd2nz(self.kv_cache_rope_out_cpu.reshape((BLOCK_NUM, BLOCK_SIZE, -1)))

            if self.cacheMode == 0:
                self.q_out_cpu = torch.cat((q_out, q_rope_out), dim=-1)
                self.q_rope_out_cpu = torch.zeros_like(self.q_out_cpu)
            elif self.cacheMode == 2:
                self.q_out_cpu = self.ein_sum_out_quant_golden(q_out, self.qNopeScale)
                self.q_rope_out_cpu = q_rope_out
            else: # cache_mode == 1, 3
                self.q_out_cpu = q_out
                self.q_rope_out_cpu = q_rope_out
            
            res0 = self.q_out_cpu
            res1 = self.kv_cache_out_cpu
            res2 = self.q_rope_out_cpu
            res3 = self.kv_cache_rope_out_cpu
            
            result = [res0, res1, res2, res3]
            torch.save(self.cacheMode,f"{remote_dir}/cacheMode.bin")
            torch.save(result, f"{remote_dir}/result_cpu.bin")
            # torch.save(self.cacheMode,f"{common_file_save_path}/cacheMode.bin")
            # torch.save(result, f"{common_file_save_path}/result_cpu.bin")
            
            # print("-----------------cpu end---------------")
            return res0, res1, res2, res3

        # 小算子拼接标杆
        if self.device == "npu":
            # print("-------------------------enter npu-----------------------")
            # print("-------------------------npu slotmapping-----------------------")

            res0, res1, res2, res3 = self.__npu_golden_calc_impl(en_w8a8, do_rms_norm)
            # print("res0:", res0)
            result = [res0.cpu(), res1.cpu(), res2.cpu(), res3.cpu()]
            torch.save(result, f"{remote_dir}/result_npu.bin")
            # torch.save(result, f"{common_file_save_path}/result_npu.bin")

            # print("-------------------------npu end-----------------------")
            return res0, res1, res2, res3



    def __npu_golden_calc_impl(self, en_w8a8, do_rms_norm):
        if en_w8a8:
            if do_rms_norm:
                mm1_in = self.rms_norm_quant_npu(self.input1, self.gamma0, self.beta0, self.quantScale0, self.quantOffset0)
            else:
                mm1_in = self.quant_npu(self.input1, self.quantScale0, self.quantOffset0)


            mm1_out = self.mm_w8a8_npu(mm1_in, self.wdqkv, self.deScale0, self.bias0, self.data_type)
        else:
            mm1_out = self.mm_npu(self.input1, self.wdqkv)

        split1_out = [mm1_out[...,:512], mm1_out[..., 512:576], mm1_out[...,576:]]

        if en_w8a8:
            mm2_in = self.rms_norm_quant_npu(split1_out[2], self.gamma1, self.beta1, self.quantScale1, self.quantOffset1)
            mm2_out = self.mm_w8a8_npu(mm2_in, self.wuq, self.deScale1, self.bias1, self.data_type)
        else:
            mm2_in = self.rms_norm_npu(split1_out[2], self.gamma1)
            mm2_out = self.mm_npu(mm2_in, self.wuq)

        nope_dim = self.wuk.shape[1]
        rope_dim = 64
        mm2_out = mm2_out.reshape(self.input_token_num, self.headNum, nope_dim + rope_dim)
        split2_out = [mm2_out[..., :nope_dim], mm2_out[..., nope_dim:]]

        q_out = self.mm_einsum_npu(split2_out[0], self.wuk)

        q_rope_out = self.rope_npu(split2_out[1].reshape(split2_out[1].shape[0], -1), self.cos, self.sin).reshape(split2_out[1].shape)
        if self.cacheMode == 0:
            self.q_out_npu = torch.concat((q_out, q_rope_out), dim=-1)
            self.q_rope_out_npu = torch.zeros_like(self.q_out_npu)
        elif self.cacheMode == 2:
            self.q_out_npu = self.einsum_quant_npu(q_out, self.qNopeScale)
            self.q_rope_out_npu = q_rope_out
        else: # cache_mode == 1, 3
            self.q_out_npu = q_out
            self.q_rope_out_npu = q_rope_out

        key_rope_out = self.rope_npu(split1_out[1], self.cos, self.sin)

        key_out = self.rms_norm_npu(split1_out[0], self.gamma2)
        if self.cacheMode == 2:
            key_out = self.quant_npu(key_out, self.ctkvScale)


        self.kv_cache_out_npu, self.kv_cache_rope_out_npu = reshape_and_cache_cpu(
            key_out.npu(), key_rope_out.npu(), self.kvCache, self.kvCacheRope, self.slotMapping, self.cacheMode
        )

        if self.cacheMode > 1:
            self.kv_cache_out_npu = transdata_nd2nz(self.kv_cache_out_npu.reshape((BLOCK_NUM, BLOCK_SIZE, -1)))
            self.kv_cache_rope_out_npu = transdata_nd2nz(self.kv_cache_rope_out_npu.reshape((BLOCK_NUM, BLOCK_SIZE, -1)))
        return self.q_out_npu, self.kv_cache_out_npu, self.q_rope_out_npu, self.kv_cache_rope_out_npu



@register("function_pyaclnn_mla_preprocess")
class aclnnMlaPreprocessApi(AclnnBaseApi):
    def __init__(self, task_result: TaskResult, backend):
        super(aclnnMlaPreprocessApi, self).__init__(task_result, backend)


    def get_format(self, input_data: InputDataset, index=None, name=None):
        """
        :param input_data: 参数列表
        :param index: 参数位置
        :param name: 参数名字
        :return:
        format at this index or name
        """
        if input_data.kwargs["cacheMode"] in [2,3] :
            if name in ["wdqkv","wuq",'kvCache','kvCacheRope'] : # 5为输出或输出的位置，此时处理第五5位置的入参，可能为输入或输出
                return AclFormat.ACL_FORMAT_FRACTAL_NZ
            if index== 1 or index== 3: # 5为输出或输出的位置，此时处理第五5位置的入参，可能为输入或输出
                return AclFormat.ACL_FORMAT_FRACTAL_NZ  
        else:
            if name in ["wdqkv","wuq"]: # 5为输出或输出的位置，此时处理第五5位置的入参，可能为输入或输出
                return AclFormat.ACL_FORMAT_FRACTAL_NZ

        return AclFormat.ACL_FORMAT_ND

    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        #获取第一个tensor的size
        self.input_args,tmp= super().init_by_input_data(input_data)
        self_tensor = self.acl_tensor_to_torch(self.input_args[0])
        self.self_shape = self_tensor.size()
        cacheMode = input_data.kwargs["cacheMode"]

        npu_cast_to_nz = lambda tensor: torch_npu.npu_format_cast(transdata_nd2nz(tensor).contiguous().npu(), 29)
        # print("-------------------enter aclnn---------------")
        # print("-------------------------aclnn slotmapping-----------------------")

        if cacheMode > 1:
            input_data.kwargs["kvCache"] = npu_cast_to_nz(input_data.kwargs["kvCache"].reshape((BLOCK_NUM, BLOCK_SIZE, -1)))
            input_data.kwargs["kvCacheRope"] = npu_cast_to_nz(input_data.kwargs["kvCacheRope"].reshape((BLOCK_NUM, BLOCK_SIZE, -1)))


        return input_args, output_packages   

    def get_cpp_func_signature_type(self):
        # 校验函数
        return "aclnnStatus aclnnMlaPreprocessGetWorkspaceSize(const aclTensor *input,  \
        const aclTensor *gamma0, \
        const aclTensor *beta0, \
        const aclTensor *quantScale0, \
        const aclTensor *quantOffset0,\
        const aclTensor *wdqkv, \
        const aclTensor *deScale0, \
        const aclTensor *bias0, \
        const aclTensor *gamma1, \
        const aclTensor *beta1, \
        const aclTensor *quantScale1, \
        const aclTensor *quantOffset1, \
        const aclTensor *wuq, \
        const aclTensor *deScale1, \
        const aclTensor *bias1, \
        const aclTensor *gamma2, \
        const aclTensor *cos, \
        const aclTensor *sin, \
        const aclTensor *wuk, \
        const aclTensor *kvCache, \
        const aclTensor *kvCacheRope, \
        const aclTensor *slotmapping, \
        const aclTensor *ctkvScale, \
        const aclTensor *qNopeScale, \
        int64_t wdqDim, \
        int64_t qRopeDim, \
        int64_t kRopeDim, \
        double epsilon, \
        int64_t qRotaryCoeff, \
        int64_t kRotaryCoeff, \
        bool transposeWdq, \
        bool transposeWuq, \
        bool transposeWuk, \
        int64_t cacheMode, \
        int64_t quantMode, \
        bool doRmsNorm, \
        int64_t wdkvSplitCount, \
        aclTensor *qOut, \
        aclTensor *kvCacheOut, \
        aclTensor *qRopeOut, \
        aclTensor *krCacheOut, \
        uint64_t *workspaceSize, \
        aclOpExecutor **executor)"

    def after_call(self, output_packages):
        aclnn0, aclnn1,aclnn2,aclnn3 = super().after_call(output_packages)
        from atk.common.connection import RemoteManager
        post_node = self.task_result.nodes.get_main_node()
        remote_nodes = []
        for node in self.task_result.nodes.nodes:
            if node != post_node:
                remote_nodes.append(node)
        remote_manager = RemoteManager(self.task_result.case_config, post_node,remote_nodes[0] )
        local_dir, remote_dir = remote_manager.get_local_and_remote_dir()
        result = [aclnn0.cpu(), aclnn1.cpu(),aclnn2.cpu(),aclnn3.cpu()]
        tmp_path = os.path.join(local_dir, 'aclnn_tmp') 
        os.makedirs(tmp_path, exist_ok=True)
        torch.save(result, f"{tmp_path}/result.bin")
        # torch.save(result, f"{common_file_save_path}/result.bin")
        
        return aclnn0,aclnn1,aclnn2,aclnn3

MIN_ERR = 1e-7
class OpTypes(Enum):
    NA = 0 # new standard is not available
    MOVE = 1
    RAND = 2
    CAST = 3
    COMPUTE_INTEGER = 4
    COMPUTE_QUANT = 5
    COMPUTE_FLOAT = 6
    COMPUTE_FLOAT_HIGH_PRECISION = 7
    VECTOR_FUSION = 8
    CV_FUSION = 9

dtype_dict = {"float": torch.float32, "float16": torch.float16, "int8": torch.int8, "int32": torch.int32, "uint8": torch.uint8,
              "int16": torch.int16, "uint16": torch.int16, "uint32": torch.int32, "int64": torch.int64, "uint64": torch.int64,
              "double": torch.double, "bool": torch.bool, "complex64": torch.complex64, "complex128": torch.complex128, "bf16": torch.bfloat16}

def compare_data(tensor1, tensor2):
    out = tensor1.flatten()
    out_len = out.shape[0]
    golden = tensor2.flatten()
    diff = torch.abs(golden - out)
    max_diff = diff.max().item()
    # print(f"maxDiff {max_diff}")
    golden = golden.to(torch.float32)
    out = out.to(torch.float32)
    limit_error = torch.maximum(
        torch.abs(golden * 0.001), torch.tensor(0.001))
    strict_limit_error = torch.maximum(
        torch.abs(golden * 0.003), torch.tensor(0.003))
    error_count = torch.gt(diff, limit_error).sum().item()
    strict_error_count = torch.gt(
        diff, strict_limit_error).sum().item()

    # print("1/1000 Accuracy is %f",
    #                     1 - float(error_count) / out_len)
    # print("3/1000 Accuracy is %f",  1 -
    #                     float(strict_error_count) / out_len)
    # print("accuracy is correct: %r", (float(strict_error_count) / out_len) <= 0.001)
    return (float(strict_error_count) / out_len) <= 0.001

def get_eb_threshold(dtype:torch.dtype):
    eb_threshold = 0
    if dtype in [torch.bfloat16]:
        eb_threshold = 2**(-7)
    if dtype in [torch.float16]:
        eb_threshold = 2**(-10)
    if dtype in [torch.float32]:
        eb_threshold = 2**(-14)
    return eb_threshold

def get_err_threshold(op_type:OpTypes, dtype:torch.dtype):
    err_threshold = 0
    if op_type in [OpTypes.MOVE, OpTypes.RAND, OpTypes.CAST, OpTypes.COMPUTE_INTEGER]:
        pass
    if op_type in [OpTypes.COMPUTE_QUANT, OpTypes.COMPUTE_FLOAT]:
        if dtype in [torch.bfloat16]:
            err_threshold = 2**(-7)
        if dtype in [torch.float16]:
            err_threshold = 2**(-8)
        if dtype in [torch.float32]:
            err_threshold = 2**(-11)
    if op_type in [OpTypes.CV_FUSION]:
        if dtype in [torch.bfloat16]:
            err_threshold = 2**(-8)
        if dtype in [torch.float16]:
            err_threshold = 2**(-11)
        if dtype in [torch.float32]:
            err_threshold = 2**(-14)
    return err_threshold

def compare_cv(golden:torch.Tensor, gpu:torch.Tensor, actual:torch.Tensor):
    op_type = OpTypes.CV_FUSION
    judge_threshold = 522
    eb_threshold = get_eb_threshold(actual.dtype)
    err_threshold = get_err_threshold(op_type, actual.dtype)
    # logging.info(f"err_threshold:{err_threshold} eb_threshold:{eb_threshold}")
    mare_npu = get_mare(golden, actual)
    mare_gpu = get_mare(golden, gpu)
    mere_npu = get_mere(golden, actual)
    mere_gpu = get_mere(golden, gpu)
    rmse_npu = get_rmse(golden, actual)
    rmse_gpu = get_rmse(golden, gpu)
    mare_rate = mare_npu / max(mare_gpu, err_threshold)
    # logging.info(f"mare_rate:{mare_rate}")
    mere_rate = mere_npu / max(mere_gpu, err_threshold)
    # logging.info(f"mere_rate:{mere_rate}")
    rmse_rate = rmse_npu / max(rmse_gpu, err_threshold)
    # logging.info(f"rmse_rate:{rmse_rate}")
    EB = get_eb(gpu, actual)
    result = (mare_rate < 10) and (mere_rate < 2) and (rmse_rate < 2) and (EB < eb_threshold)

    return result

#误差均衡性（EB）
def get_eb(golden:torch.Tensor, actual:torch.Tensor):
    golden = golden.to(torch.float32)
    golden_nmax = torch.clamp(torch.abs(golden), min = 1)
    actual_error = actual.to(torch.float32) - golden
    EB = torch.mean(actual_error / golden_nmax)
    return EB

#最大相对误差：max relative error，MARE
def get_mare(golden:torch.Tensor, actual:torch.Tensor):
    golden = golden.to(torch.float32)
    abs_error = torch.abs(actual.to(torch.float32) - golden) / (torch.abs(golden) + MIN_ERR)
    mare = torch.max(abs_error.flatten())
    return mare

#平均相对误差：mean relative error，MERE
def get_mere(golden:torch.Tensor, actual:torch.Tensor):
    golden = golden.to(torch.float32)
    abs_error = torch.abs(actual.to(torch.float32) - golden) / (torch.abs(golden) + MIN_ERR)
    mere = torch.mean(abs_error)
    return mere

#均方根误差:Root Mean Squared Error，RMSE
def get_rmse(golden:torch.Tensor, actual:torch.Tensor):
    golden = golden.to(torch.float32)
    sqr_err = torch.pow((actual.to(torch.float32) - golden), 2)
    rmse = torch.sqrt(torch.mean(sqr_err))
    return rmse

@ACCURACY_REGISTRY.register("cv_fused_double_benchmark_aclnn_mla_preprocess")
class CvFusedNpuNsaCompressAttentionInferAccuracyCompare(DoubleBenchmarkAccuracyCompare):
    def compute_accuracy_result(self, local_output: torch.Tensor, remote_output: torch.Tensor, data_file):
        """
        cpu_result:  cpu真值 
        npu_result:  aclnn输出,实际输出
        atb_result:  atb小算子输出 
        """
        local_dir, remote_dir = self.remote_manager.get_local_and_remote_dir()
        cpu_result = torch.load(f"{remote_dir}/result_cpu.bin")
        npu_result = torch.load(f"{remote_dir}/result_npu.bin")
        aclnn_result = torch.load(f"{local_dir}/aclnn_tmp/result.bin")
        # aclnn_result = torch.load(f"{local_dir}/result.bin")

        # cpu_result = torch.load(f"{common_file_save_path}/result_cpu.bin")
        # npu_result = torch.load(f"{common_file_save_path}/result_npu.bin")
        # aclnn_result = torch.load(f"{common_file_save_path}/result.bin")

        result = False

        cacheMode = torch.load(f"{remote_dir}/cacheMode.bin")
        # cacheMode = torch.load(f"{common_file_save_path}/cacheMode.bin")
        print(cacheMode)

        if cacheMode == 0:
            res_0 = compare_cv(cpu_result[0].cpu(), npu_result[0].cpu(), aclnn_result[0].cpu())
            res_1 = compare_cv(cpu_result[1].cpu(), npu_result[1].cpu(), aclnn_result[1].cpu())
            # print("----------res_0--------------")
            # print(res_0)
            # print("----------res_1--------------")
            # print(res_1)
            result = res_0 and res_1
        elif cacheMode == 1:
            res_0 = compare_cv(cpu_result[0], npu_result[0], aclnn_result[0])
            res_1 = compare_cv(cpu_result[1], npu_result[1], aclnn_result[1])
            res_2 = compare_cv(cpu_result[2], npu_result[2], aclnn_result[2])
            res_3 = compare_cv(cpu_result[3], npu_result[3], aclnn_result[3])
            # print("----------res_0--------------")
            # print(res_0)
            # print("----------res_1--------------")
            # print(res_1)
            # print("----------res_2--------------")
            # print(res_2)
            # print("----------res_3--------------")
            # print(res_3)
            result = res_0 and res_1 and res_2 and res_3
        elif cacheMode == 2:
            res_0 = torch.max(torch.abs(npu_result[0].flatten() - aclnn_result[0].flatten())) <= 1
            res_1 = torch.max(torch.abs(npu_result[1].flatten() - aclnn_result[1].flatten())) <= 1
            res_2 = compare_cv(cpu_result[2], npu_result[2], aclnn_result[2])
            res_3 = compare_cv(cpu_result[3], npu_result[3], aclnn_result[3])
            result = res_0 and res_1 and res_2 and res_3
        elif cacheMode == 3:
            res_0 = compare_cv(cpu_result[0], npu_result[0] ,aclnn_result[0])
            # res_1 = compare_data(npu_result[1].flatten() ,aclnn_result[1].flatten())
            res_1 = compare_cv(cpu_result[1], npu_result[1] ,aclnn_result[1])
            res_2 = compare_cv(cpu_result[2], npu_result[2], aclnn_result[2])
            res_3 = compare_cv(cpu_result[3], npu_result[3], aclnn_result[3])
            # res_3 = compare_data(npu_result[3].flatten(), aclnn_result[3].flatten())

            # print("----------res_0--------------")
            # print(res_0)
            # print("----------res_1--------------")
            # print(res_1)
            # print("----------res_2--------------")
            # print(res_2)
            # print("----------res_3--------------")
            # print(res_3)
            result = res_0 and res_1 and res_2 and res_3

        if result:
            ret = AccuracyConfig(
                result=True,
                error_info="Accuracy pass"
            )
        else:
            ret = AccuracyConfig(
                result=False,
                error_info="Accuracy failed"
            )
        return ret