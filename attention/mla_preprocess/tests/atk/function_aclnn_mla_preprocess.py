# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
import torch
import torch.nn.functional as F
import numpy as np
import os
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

logging = Logger().get_logger()

try:
    import torch_npu
    _HAS_TORCH_NPU = True
except ImportError:
    _HAS_TORCH_NPU = False

def get_file_md5(filepath):
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:  # 必须以二进制模式读取
        # 分块读取，避免大文件占用过多内存
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

QUANTMAX = 127
QUANTMIN = -128
# block_size = 256

def reshapeAndCacheNz(self, input, keycache, slotMapping,num,fenxin,loop):
    keycache = keycache.flatten()
    input = input.reshape(-1,num)
    for i in range(len(slotMapping)):
        slot_idx = slotMapping[i]
        outer_idx = (int)(slot_idx / 128)
        inner_idx = slot_idx % 128

        stride = 128*fenxin
        for j in range(loop):
            startIdx = inner_idx*fenxin + j*stride + outer_idx * 128 * num
            keycache[startIdx: startIdx + fenxin] = input[i][j*fenxin : (j+1)*fenxin]

    return keycache

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

    def round_up(self, val: int, align: int) -> int:
        if align == 0:
            return 0
        return -(val // -align) * align

    def transdata(self, nd_mat, block_size: tuple = (16, 16)):
        r = self.round_up(nd_mat.shape[0], block_size[0])
        c = self.round_up(nd_mat.shape[1], block_size[1])
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
        #第一次执行需要生成slotmapping
        # input_data.kwargs["slotmapping"] = torch.from_numpy(np.random.choice(192 * 128, input_token_num, replace=False).astype(np.int32)).to(torch.int32)
        
        wuq = input_data.kwargs["wuq"].cpu().clone()
        wuq = self.transdata(wuq, (16, 32)).contiguous()
        wdqkv = input_data.kwargs["wdqkv"].cpu().clone()

        # case_expected_error_msg = self.task_result.case_config.expected_error_msg
        # if  case_expected_error_msg not in ["161002","561002"]:
        wdqkv = self.transdata(wdqkv, (16, 32)).contiguous()
        deScale0 = self.process_deq_scale(deq_scale=input_data.kwargs["deScale0"].cpu().clone())
        deScale1 = self.process_deq_scale(deq_scale=input_data.kwargs["deScale1"].cpu().clone())
        
        case_id = self.task_result.case_config.id
        
        import numpy as np
        _slot_rng = np.random.RandomState(2025)
        input_token_num = input_data.kwargs["input"].shape[0]
        input_data.kwargs["slotmapping"] = torch.from_numpy(
            _slot_rng.choice(192 * 128, input_token_num, replace=False).astype(np.int32)
        ).to(torch.int32)
        if self.device == "pyaclnn":
            input_data.kwargs["input"] = input_data.kwargs["input"].npu()
            input_data.kwargs["gamma0"] = input_data.kwargs["gamma0"].npu()
            input_data.kwargs["beta0"] = input_data.kwargs["beta0"].npu()
            input_data.kwargs["quantScale0"] = input_data.kwargs["quantScale0"].npu()
            input_data.kwargs["quantOffset0"] = input_data.kwargs["quantOffset0"].npu()
            #转NZ格式
            input_data.kwargs["wdqkv"]= torch_npu.npu_format_cast(wdqkv.npu(), 29).npu()
            if input_data.kwargs["input"].dtype == torch.float16 :
                input_data.kwargs["deScale0"] = deScale0.npu()
            else:
                input_data.kwargs["deScale0"] = input_data.kwargs["deScale0"].npu()
            input_data.kwargs["bias0"] = input_data.kwargs["bias0"].npu()
            input_data.kwargs["gamma1"] = input_data.kwargs["gamma1"].npu()
            input_data.kwargs["beta1"] = input_data.kwargs["beta1"].npu()
            input_data.kwargs["quantScale1"] = input_data.kwargs["quantScale1"].npu()
            input_data.kwargs["quantOffset1"] = input_data.kwargs["quantOffset1"].npu()
            input_data.kwargs["wuq"]= torch_npu.npu_format_cast(wuq.npu(), 29).npu()
            if input_data.kwargs["input"].dtype == torch.float16 :
                input_data.kwargs["deScale1"] = deScale1.npu()
            else:
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

    def process_deq_scale(self, deq_scale: torch.Tensor) -> torch.Tensor:
        # 确保是 float32
        assert deq_scale.dtype == torch.float32, "Only support float32"
        # reinterpret float32 bits as int32
        int32_tensor = deq_scale.view(torch.int32)  # ← 更高效、更清晰！
        return int32_tensor.to(torch.int64)

    def padHeadNum(self, headdim_sin_cos):
        return torch.tile(headdim_sin_cos, (1, self.headNum))

    def rotateHalfX(self, q_temp):
        q_splits = torch.chunk(q_temp, self.headNum, dim=1)
        processed_q_splits = []
        for q_split in q_splits:
            first_half, second_half = torch.chunk(q_split, 2, dim=1)
            processed_q_split = torch.cat((-second_half, first_half), dim=1)
            processed_q_splits.append(processed_q_split)

        return torch.cat(processed_q_splits, dim=1)

    def rotateHalf2(self, q_temp):
        q_splits = np.split(q_temp, self.headNum, axis=1)
        processed_q_splits = []
        for q_split in q_splits:
            first_half, second_half = np.split(q_split, 2, axis=1)
            processed_q_split = np.concatenate((second_half, first_half), axis=1)
            processed_q_splits.append(processed_q_split)

        return np.concatenate(processed_q_splits, axis=1)

    def RopeConcatGolden(self, q, sin, cos, concatInput):
        pad_sin = self.padHeadNum(sin)
        pad_cos = self.padHeadNum(cos)
        rope_res = q * pad_cos + self.rotateHalfX(q) * pad_sin
        rope_res = rope_res.reshape(self.input_token_num, self.headNum, self.rope_hidden_size)
        rope_res = rope_res.to(self.dtype)

        return torch.cat((concatInput.to(self.dtype), rope_res), dim=2)

    def rmsNormGolden(self, x, gamma):
        x_float32 = x.to(torch.float32)
        square_sum = torch.sum(torch.square(x_float32), axis=-1, keepdims=True)
        rms = 1.0 / torch.sqrt(square_sum /self.rms_hidden_size + self.epsilon)  
        gamma_float32 = gamma.to(torch.float32)
        rmsNorm = rms * x_float32 * gamma_float32
        result = rmsNorm.to(self.dtype)
        np.set_printoptions(suppress=True, formatter={'float_kind':'{:.15f}'.format})
        return result

    def rotateHalf(self, k_temp):
        first_half, second_half = torch.chunk(k_temp, 2, dim=1)
        processed_k_split = torch.cat((-second_half, first_half), dim=1)
        return processed_k_split

    def RopeGolden(self, keyRope, sin, cos):
        RopeGolden = keyRope * cos + self.rotateHalf(keyRope) * sin
        return RopeGolden

    def RACGolden(self, keyRAC, slotMapping, keycacheout_golden):
        for i, slot in enumerate(slotMapping):
            if slot < 0:
                continue
            block_index = slot // self.block_size
            block_offset = slot % self.block_size
            token_key = keyRAC[i]
            keycacheout_golden[block_index][block_offset] = token_key
        return keycacheout_golden

    def RmsNormAndRopeAndReshapeAndCacheGolden(self, x, gamma, keyRope, cos, sin, slotMapping, keycachein):
        rmsNormOutput = self.rmsNormGolden(x, gamma)
        ropeOutput = self.RopeGolden(keyRope, sin, cos)
        ropeReshape = ropeOutput.reshape(self.input_token_num, 1, self.rope_hidden_size)
        keyRAC = torch.cat((rmsNormOutput, ropeReshape), axis=-1)

        return self.RACGolden(keyRAC, slotMapping, keycachein)

    def rms_norm_quant_calc(self, input, gamma, beta, quantScale, quantOffset, data_type, is_norm):
        if not is_norm:
            return input.to(data_type)
        out_shape = input.shape
        scale = 1.0 / quantScale.item()
        offset = quantOffset.item()
        inputScale = torch.tensor(scale, dtype=torch.float)
        inputOffset = torch.tensor(offset, dtype=torch.float)
        input0 = torch.tensor(input).float()
        input1 = torch.tensor(gamma).float()
        square_sum = torch.sum(torch.square(input0), axis=-1, keepdims=True)
        factor = 1.0 / torch.sqrt(square_sum / out_shape[-1] + self.epsilon)
        output = input0 * factor * input1
        output = (output + beta.float()) * inputScale + inputOffset
        self.xxTest = output.clone()
        output = torch.round(output)
        output = torch.tensor(output).to(torch.float16)
        output = torch.min(output, torch.tensor(QUANTMAX, dtype=torch.half))
        output = torch.max(output, torch.tensor(QUANTMIN, dtype=torch.half))
        return torch.tensor(output).to(torch.int8)

    def calc_vec_mm_data(self, N, headNum, data_type, is_norm):
        # cpu标杆计算流程
        if is_norm:
            mm1In = self.rms_norm_quant_calc(self.input1, self.gamma0, self.beta0, self.quantScale0, self.quantOffset0, data_type, is_norm)
            self.rmsquantOut1 = mm1In.clone()
            print(f"[DBG-FFA] 101 mm1_in first128={mm1In.flatten()[:128].tolist()}")
            mm1Out = torch.matmul(mm1In.to(torch.int32), self.wdqkv.transpose(0,1).to(torch.int32))
            mm1Out = mm1Out + self.bias0
            print(f"[DBG-FFA] 201 mm1_out first128={(mm1Out.to(torch.float32) * self.deScale0).to(data_type).flatten()[:128].tolist()}")
            mm1Out = (mm1Out.to(torch.float32) * self.deScale0).to(data_type)
        else:
            mm1Out = torch.matmul(self.input1.to(torch.float32), self.wdqkv.transpose(0, 1).to(torch.float32))
        
        self.mm1Out1 = mm1Out
        if data_type == torch.float16:
            self.deScale0 = self.process_deq_scale(deq_scale=self.deScale0)
        mm1OutSplit1, mm1OutSplit2 = torch.split(mm1Out, [576, 1536], dim=1)
        rms2Out = self.rms_norm_quant_calc(mm1OutSplit2, self.gamma1, self.beta1, self.quantScale1, self.quantOffset1, data_type, is_norm)
        self.rmsquantOut2 = rms2Out.clone()
        print(f"[DBG-FFA] 102 mm2_in first128={rms2Out.flatten()[:128].tolist()}")
        mm2Out = torch.matmul(rms2Out.to(torch.int32), self.wuq.transpose(0, 1).to(torch.int32))
        
        if is_norm:
            mm2Out = mm2Out + self.bias1
            print(f"[DBG-FFA] 202 mm2_out first128={(mm2Out.to(torch.float32) * self.deScale1).to(data_type).flatten()[:128].tolist()}")
            mm2Out = (mm2Out.to(torch.float32) * self.deScale1).to(data_type)
        else:
            mm2Out.to(data_type)
        self.mm2Out2 = mm2Out
        if data_type == torch.float16:
            self.deScale1 = self.process_deq_scale(deq_scale=self.deScale1)

        mm11OutSplit1, mm12OutSplit1 = torch.split(mm1OutSplit1, [512, 64], dim=1)
        mm11OutSplit1 = mm11OutSplit1.reshape(N, 1, 512)
        self.keyOutTensor = self.keyCache.clone()
        self.keyOut1 = self.RmsNormAndRopeAndReshapeAndCacheGolden(mm11OutSplit1, self.gamma2, mm12OutSplit1, self.cos1, self.sin1, self.slotMapping, self.keyCache)
        mm2Out = mm2Out.reshape(N, headNum, 192)
        mm2OutSplit1, mm2OutSplit2 = torch.split(mm2Out, [128, 64], dim=2)
        self.bmmOut= torch.permute(
            torch.matmul(torch.permute(mm2Out[:, :, :128], (1, 0, 2)).float(), self.wuk.float()),
            (1, 0, 2),
        )
        print(f"[DBG-FFA] 203 einsum_out first128={self.bmmOut.flatten()[:128].tolist()}")
        self.gg = mm2OutSplit2.clone()
        qOut = self.RopeConcatGolden(mm2OutSplit2.reshape(N,  headNum * 64), self.sin2, self.cos2, self.bmmOut)
        self.qOut = qOut.to(data_type)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        
        post_node = self.task_result.nodes.get_main_node()
        remote_nodes = []
        for node in self.task_result.nodes.nodes:
            if node != post_node:
                remote_nodes.append(node)
        remote_manager = RemoteManager(self.task_result.case_config, post_node,remote_nodes[0] )
        local_dir, remote_dir = remote_manager.get_local_and_remote_dir()
        # logging.info(f"local_dir{local_dir}, remote_dir{remote_dir}")

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
        self.cos1 = input_data.kwargs["cos"]
        self.sin1 = input_data.kwargs["sin"]
        self.cos2 = self.cos1
        self.sin2 = self.sin1
        self.wuk = input_data.kwargs["wuk"]
        self.keyCache_tmp = input_data.kwargs["kvCache"]
        self.keyCache = torch.zeros((192, 128, 1, 576), dtype=self.input1.dtype)
        self.slotMapping = input_data.kwargs["slotmapping"]
        self.ctkvScale = input_data.kwargs["ctkvScale"]
        self.qNopeScale = input_data.kwargs["qNopeScale"]
        
        input_token_num = self.input1.shape[0]
        head_num = self.wuk.shape[0]
        data_type = self.input1.dtype
        #[blockNum, headNum*64 / 16 ,blockSize, 16]
        blockNum = self.keyCache_tmp.shape[0]
        blockSize = self.keyCache_tmp.shape[2]

        # 异常用例执行
        case_id = self.task_result.case_config.id
        case_expected_error_msg = self.task_result.case_config.expected_error_msg
        if  case_expected_error_msg in ["161002","561002"]:
            qOut_shape = [input_token_num,head_num,576]
            kvCacheOut_shape = [blockNum,blockSize,576] 

            qOut = torch.zeros(qOut_shape, dtype=data_type)    
            kvCacheOut = torch.zeros(kvCacheOut_shape, dtype=data_type)
            res2 = torch.zeros((0, head_num, 512),dtype=data_type)
            res3 = torch.zeros((0, head_num, 512), dtype=data_type)    

            return qOut,kvCacheOut,res2,res3

        #data_type = 'flot16'
        quant_mode = input_data.kwargs["quantMode"]
        quant_mode = 0
        self.input_token_num = input_token_num
        self.epsilon = 1e-6
        self.rms_hidden_size = 512
        self.rope_hidden_size = 64
        self.dtype = data_type
        self.block_size = self.keyCache.shape[1]
        self.headNum = head_num

        is_norm = not (input_data.kwargs["wdqkv"].dtype == torch.bfloat16 and input_data.kwargs["wuq"].dtype == torch.bfloat16)
        self.calc_vec_mm_data(input_token_num, head_num, data_type, is_norm) 



        if self.cacheMode == 0:
            res0 = self.qOut
            res1 = self.keyOut1
            res2 = torch.zeros((0, head_num, 512), dtype=data_type)
            res3 = torch.zeros((0, head_num, 512), dtype=data_type)
        elif self.cacheMode == 1:
            res0 = self.qOut[..., 0:512]
            res1 = self.keyOut1[..., 0:512]
            res2 = self.qOut[..., 512:576]
            res3 = self.keyOut1[..., 512:576]
        elif self.cacheMode == 2:
            res0 = self.qOut[..., 0:512].to(torch.int8)
            res1 = self.keyCache_tmp.to(torch.int8)
            res2 = self.qOut[..., 512:576]
            res3 = torch.zeros([blockNum, head_num*64 // 16 ,blockSize, 16],dtype=data_type)
        elif self.cacheMode == 3:
            res0 = self.qOut[..., 0:512]
            res1 = self.keyCache_tmp
            res2 = self.qOut[..., 512:576]
            res3 = torch.zeros([blockNum, head_num*64 // 16 ,blockSize, 16],dtype=data_type)

        result = [res0.cpu(),res1.cpu(), res2.cpu(), res3.cpu()]
        torch.save(result, f"{remote_dir}/result.bin")

        return res0, res1, res2, res3


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
        # output_packages[1] = input_args[19] 
        # output_packages[3] = input_args[20]

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
        float epsilon, \
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
        print(f"[DBG-NPU] qOut first128={result[0].flatten()[:128].tolist()}")
        print(f"[DBG-NPU] kvCacheOut first4={result[1].flatten()[:4].tolist()}")
        print(f"[DBG-NPU] qRopeOut first4={result[2].flatten()[:4].tolist()}")
        print(f"[DBG-NPU] krCacheOut first4={result[3].flatten()[:4].tolist()}")
        tmp_path = os.path.join(local_dir, 'aclnn_tmp') 
        os.makedirs(tmp_path, exist_ok=True)
        torch.save(result, f"{tmp_path}/result.bin")
        
        return aclnn0,aclnn1,aclnn2,aclnn3
 