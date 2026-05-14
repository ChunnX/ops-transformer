import numpy as np
import torch
import unittest
import torch_npu
from torch_npu.testing.testcase import TestCase,run_tests
from torch_npu.testing.common_utils import SupportedDevices
import torch.nn as nn
import torchair as tng
import logging
from torchair.ge_concrete_graph import ge_apis as ge
from torchair.configs.compiler_config import CompilerConfig
from scipy.special import softmax
from torchair import logger
import random
import time
import turing_ascend_cloud_ops_nn

relative_tol = 1e-3
absolute_tol = 1e-3
error_tol = 1e-3

def compare_data(golden_file, output_file, d_type):
    if d_type == "float16":
        precision = 1/1000
    else:
        precision = 1/10000
    d_type_dict = {
        "float32":np.float32,
        "float16":np.float16
    }
    dtype = d_type_dict.get(d_type)

    golden = np.fromfile(golden_file, dtype)
    output = np.fromfile(output_file, dtype)

    diff_res = np.isclose(output, golden, precision, 0 , True)
    diff_idx = np.where(diff_res != True )[0]
    diff_ratio = len(diff_idx) / golden.size
    if len(diff_idx) == 0 or diff_ratio <precision:
        print("PASS!");
    else:
        print("FAILED!")
        print(f"Diff ratio:{len(diff_idx) / golden.size}")
        for idx in diff_idx[:5]:
            print(f"index:{idx},output:{output[idx]},golden:{golden[idx]}")
    return

def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch_npu.npu.manual_seed(seed)
    torch_npu.npu.manual_seed_all(seed)

def non_quant_golden(x, weight, scale, perTokenScale, groupList, bias, quantGroupNum, *, offset=None):
    groupNum, k, n = weight.shape
    # quantGroupNum = scale.shape[1]
    index = np.cumsum(groupList)
    xSplit = np.split(x,index * 2, axis=0)
    perTokenScaleSplit = np.split(perTokenScale,index,axis = 0)
    weightGroup = weight.reshape(groupNum, quantGroupNum, k //quantGroupNum, n).astype(np.int32)
    mmOuts = []
    atomic = np.float16

    print("x  cpu:",x)
    # print("weight_in  cpu:",weight)
    # print("groupList_in  cpu:",groupList)
    # print("scale_in  cpu:",scale)
    # print("bias_in  cpu:",bias)
    # print("perTokenScale_in  cpu:",perTokenScaleSplit)

    for i in range(groupNum):
        xi = xSplit[i].reshape(-1, quantGroupNum, k // quantGroupNum).astype(np.int32)
        mmi = np.zeros([xi.shape[0], n], dtype=atomic)
        for j in range(quantGroupNum):
            mm = np.matmul(xi[:, j, :],weightGroup[i, j, ...])
            # mm = mm.astype(np.float32) * scale[i, j].reshape(1, -1)
            mm = mm.astype(np.float32) * scale[i].reshape(1, -1)  # 从scale[i, j]改为scale[i]
            mmi = (mmi.astype(atomic)+mm.astype(atomic)).astype(atomic)
        mmi = mmi.reshape(-1, 2, n).astype(np.float32)
        mmi = mmi[:, 0, :] * 16 +mmi[:, 1, :]+bias[i].reshape(1,n)
        mmi_xo = np.zeros(mmi.shape, dtype= np.float32)
        if offset is not None:
            xi_o = xSplit[i].astype(np.int32).reshape(-1, 2, k)
            xi_o = xi_o[:, 0, :] * 16 + xi_o[:, 1, :] + 8
            xi_o = xi_o.astype(np.float16).reshape(-1, quantGroupNum, k // quantGroupNum)
            for j in range(quantGroupNum):
                mm = xi_o[:, j, :].sum(axis=1,keepdims=True).astype(np.float32)
                # mm = np.matmul(mm, offset[i, j].reshape(1, -1))
                mm = np.matmul(mm, offset[i].reshape(1, -1))  # 从offset[i, j]改为offset[i]
                mmi_xo += mm.astype(np.float32)
        mmi += mmi_xo
        xxx = mmi.astype(np.float32) + mmi_xo.astype(np.float32)
        mmi = mmi * perTokenScaleSplit[i]
        mmOuts.append(mmi)
    golden = np.concatenate(mmOuts, axis=0)
    golden_tensor = torch.from_numpy(golden)
    return golden_tensor.to(torch.float16)

def compute_gmm_a8w4_golden(x_in, weight_in, bias_in, scale_in, groupList_in, perTokenScale_in, quantGroupNum, *, offset_in = None):
    #  todo 待确认
    weightNz = weight_in.astype(np.int8)
    # weightNz = weight_in
    groupNum = groupList_in.shape[0]
    m = x_in.shape[0]
    k = x_in.shape[1]
    n = scale_in.shape[-1]

    weight=weightNz.reshape(groupNum, k ,n)
    # 将8位激活值拆分为两个4位部分：高4位和低4位
    # x_in.reshape(m, 1, k)//16 获取高4位（右移4位）
    # (x_in.reshape(m, 1, k)&0x0F)-8 获取低4位并减去偏移（有符号转换）
    xC12 = np.concatenate([x_in.reshape(m, 1, k)//16,(x_in.reshape(m, 1, k)&0x0F)-8],axis =1).reshape(m * 2, k)
    xInt4 = pack_4bit_to_bytes(xC12)
    xInt4.dtype = np.int8
    # scaleUint32 = scale_in.astype(np.uint32)
    # scaleUint32.dtype = np.float32
    scaleUint32 = scale_in.astype(np.uint64)
    scaleUint32.dtype = np.int64

    offset = None
    if offset_in is not None:
        assert scale_in.shape == offset_in.shape
        offset = offset_in.astype(np.float32)

    out = non_quant_golden(xC12, weight, scaleUint32, perTokenScale_in, groupList_in, bias_in, quantGroupNum, offset = offset)
    return out

def pack_4bit_to_bytes(data):
    # 打包操作：将相邻两个4位值合并为一个字节
    # data[..., 1::2] << 4：取奇数索引数据左移4位（放置到高4位）
    # data[..., ::2] & 0x0F：取偶数索引数据取低4位（放置到低4位）
    # 通过位或运算合并
    data = data.astype(np.int8)
    data[data<0] += 16
    packed = (data[..., 1::2] << 4) | (data[..., ::2]& 0x0F)
    return packed

def verify_result(output,golden):
    output = output.reshape(-1)
    golden = golden.reshape(-1)
    different_element_results = np.isclose(output,golden,rtol = relative_tol,atol = absolute_tol,equal_nan = True)
    different_element_indexes = np.where(different_element_results == False)[0]

    for index in range(len(different_element_indexes)):
        real_index = different_element_indexes[index]
        golden_data = golden[real_index]
        output_data = output[real_index]
        print(real_index, golden_data, output_data, abs(output_data - golden_data) / golden_data)
        print("data index:%06d, expected: %-.9f,actual: %-.9f,rdiff: %-.6f" %(real_index, golden_data, output_data, abs(output_data - golden_data) / golden_data))

        if index == 10:
            break
        error_ratio = float(different_element_indexes.size) / (golden.size()[0])
        print("error ratio: % .6f, tolrence: %.6f" % (error_ratio, error_tol))
        return error_ratio <= error_tol
def gen_input(E, M, K, N, quantGroupSize, batch):
    print("f:E={E}, M={M}, K={K}, N={N},batch = {batch}")
    x = torch.randint(-4, 4, (M, K), device="npu").to(torch.int8)
    x_copy = x.clone()
    x_in = x.clone().cpu().numpy()
    # x2 = x.clone()
    weight = torch.randint(-5, 5, (E, K, N), dtype=torch.int32, device="npu")
    weight_quant = torch_npu.npu_quantize(weight.to(torch.float32), torch.tensor([1.]).npu(), None, torch.quint4x2, -1,
                                          False)
    weight_quant = torch_npu.npu_format_cast(weight_quant.npu(), 29).to(weight_quant.dtype)
    weight_in = weight.cpu().numpy()
    bias = torch.zeros((E, N), dtype=torch.float32, device="npu")
    bias_in = bias.cpu().numpy()
    # scale_np = np.random.normal(0, 0.01, (E, 1, N)).astype(np.float16).astype(np.float32)
    # perGroupScale = np.ones([E, K // quantGroupSize, N], dtype=np.float32)
    # scaleUint32 = (scale_np * perGroupScale).astype(np.float16).astype(np.float32)
    # scaleUint32.dtype = np.uint32
    # scaleUint64 = np.zeros((E, K // quantGroupSize, N * 2), dtype=np.uint32)
    # scaleUint64.dtype = np.uint64
    # scale = torch.from_numpy(scaleUint64).npu()
    # scale_in = scaleUint64

    scale_np = np.random.normal(0, 0.01, (E, N)).astype(np.float16).astype(np.float32)  # 从(E,1,N)改为(E,N)
    perGroupScale = np.ones([E, N], dtype=np.float32)  # 从[E, K//quantGroupSize, N]改为[E, N]
    scaleUint32 = (scale_np * perGroupScale).astype(np.float16).astype(np.float32)
    scaleUint32.dtype = np.uint32
    scaleUint64 = np.zeros((E, N * 2), dtype=np.uint32)  # 从(E, K//quantGroupSize, N*2)改为(E, N*2)
    # scaleUint64.dtype = np.uint64
    # scale = torch.from_numpy(scaleUint64).npu()
    scaleUint64.dtype = np.int64
    scale = torch.from_numpy(scaleUint64).npu()
    scale_in = scaleUint64


    groupList = torch.zeros((E,), dtype=torch.int64, device="npu").fill_(batch)
    groupList_in = groupList.cpu().numpy()
    perTokenScale = torch.zeros((M, 1), dtype=torch.float32, device="npu").uniform_()
    # perTokenScale_new = perGroupScale.reshape(M)
    perTokenScale_in = perTokenScale.cpu().numpy()
    # offset = torch.ones((E, K // quantGroupSize, N), dtype=torch.float32, device="npu") * 0.05
    # offset_np_ = offset.cpu().numpy()
    # offset_np = offset_np_
    # offset = torch.from_numpy(offset_np).npu()
    # offset_in = offset_np

    offset = torch.ones((E, N), dtype=torch.float32, device="npu") * 0.05  # 从(E, K//quantGroupSize, N)改为(E, N)
    offset_np_ = offset.cpu().numpy()
    offset_np = offset_np_
    offset = torch.from_numpy(offset_np).npu()
    offset_in = offset_np

    return x,weight_quant,bias,scale,offset,perTokenScale,groupList,x_in,weight_in,bias_in,scale_in,groupList_in,offset_in,perTokenScale_in



def test_gmm_a8w4_offset(E, M, K, N, quantGroupSize, batch):
    print("f:E={E}, M={M}, K={K}, N={N},quantGroupSize ={quantGroupSize},batch = {batch}")
    (x, weight_quant, bias, scale, offset, perTokenScale, groupList, x_in, weight_in, bias_in, scale_in, groupList_in,offset_in, perTokenScale_in) = gen_input(E, M, K, N, quantGroupSize, batch)
    out = torch.ops.turing_ascend_cloud_ops_nn.cloud_grouped_matmul(x=[x],weight=[weight_quant], group_list=groupList,scale=[scale],
                                        # offset=[offset],
                                        bias=[bias],
                                        per_token_scale=[perTokenScale],
                                        split_item = 3,
                                        group_type=0,
                                        tuning_config=[M//E, 1],
                                        group_list_type =1,
                                        act_type=0,output_dtype=torch.float16
                                       )
    print("x  npu:",x)
    # print("weight_quant  npu:",weight_quant)
    # print("groupList  npu:",groupList)
    # print("scale  npu:",scale)
    # print("bias  npu:",bias)
    # print("perTokenScale  npu:",perTokenScale)

    out_golden = compute_gmm_a8w4_golden(x_in=x_in,weight_in=weight_in,bias_in=bias_in,scale_in=scale_in,
                                 groupList_in=groupList_in,perTokenScale_in=perTokenScale_in,quantGroupNum=K//quantGroupSize,offset_in=offset_in)

    out_shape = out[0].shape
    golden_shape = out_golden.shape
    out_dim1 = out_shape[1]
    golden_dim0 = golden_shape[0]
    golden_dim1 = golden_shape[1]
    out_npu = out[0][:golden_dim0, :]
    out_cpu = out[0].cpu().detach().numpy()
    golden_cpu = out_golden
    # golden_np = golden_cpu.npu()
    test_result = verify_result(out_cpu.reshape(-1),golden_cpu.reshape(-1))
    print(f'####{K}:test_result={test_result}')

# class TestGroupMatmul(TestCase):
#     @SupportedDevices(['Ascend910B'])
#     def test_npu_grouped_matmul_A8W4_ND_SingleOp(self):
#         torch.npu.set_device(7)
#         E=8
#         m_list = [8,16,32,64,128,256,1024,2048]
#         for M in m_list:
#             K = 7168
#             N = 4096
#             quantGroupSize = K
#             batch =  M // E
#             test_gmm_a8w4_offset(E,M,K,N,quantGroupSize,batch)
#             K = 2048
#             N = 7168
#             quantGroupSize = K
#             test_gmm_a8w4_offset(E,M,K,N,quantGroupSize,batch)


if __name__ == "__main__":
    # set_seed(42)
    # run_tests()
    torch.npu.set_device(7)
    E=8
    M = 8
    K = 7168
    N = 4096
    quantGroupSize = K
    batch =  M // E
    test_gmm_a8w4_offset(E,M,K,N,quantGroupSize,batch)
    # m_list = [8,16,32,64,128,256,1024,2048]
        # for M in m_list:
        #     K = 7168
        #     N = 4096
        #     quantGroupSize = K
        #     batch =  M // E
        #     test_gmm_a8w4_offset(E,M,K,N,quantGroupSize,batch)
        #     K = 2048
        #     N = 7168
        #     quantGroupSize = K
        #     test_gmm_a8w4_offset(E,M,K,N,quantGroupSize,batch)