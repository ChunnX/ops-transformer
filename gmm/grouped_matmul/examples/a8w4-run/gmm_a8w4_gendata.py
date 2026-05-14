import math
import unittest
import copy
import struct
from struct import pack, unpack
import numpy as np
import torch

#import torch_cpu
import sys
# import tensorflow as tf
import bfloat16 
print("import ok")
# bf16 = tf.bfloat16.as_numpy_dtype
bf16 = bfloat16.bfloat16
#from torch_cpu.testing.testcase import TestCase, run_tests
#from torch_cpu.testing.common_utils import SupportedDevices

DEBUG = False

def non_quant_golden_fakequant(x, weight, scale, perTokenScale, groupList, bias, isBF16, isNew = False):
    groupNum, k, n = weight.shape
    quantGroupNum = scale.shape[1]
    index = np.cumsum(groupList)
    xSplit = np.split(x, index, axis=0)
    perTokenScaleSplit = np.split(perTokenScale, index, axis=0)
    weightGroup = weight.reshape(groupNum, quantGroupNum, k // quantGroupNum, n).astype(np.int32)
    mmOuts = []
    atomic = np.float16
    if isNew:
        atomic = np.float32  # using A8W8
    for i in range(groupNum):
        xi = xSplit[i].reshape(-1, quantGroupNum, k // quantGroupNum).astype(np.int32)
        mmi = np.zeros([xi.shape[0], n], dtype=atomic)
        for j in range(quantGroupNum):
            mm = np.matmul(xi[:, j, :].astype(np.float64), weightGroup[i, j, ...].astype(np.float64)).astype(np.float32)
            mm = mm.astype(np.float32) * scale[i, j].reshape(1, -1)
            mmi = (mmi.astype(atomic) + mm.astype(atomic)).astype(atomic)

        mmi = mmi.astype(np.float32)
        mmi = mmi * perTokenScaleSplit[i]
        mmOuts.append(mmi)
    golden = np.concatenate(mmOuts, axis=0)
    golden_tensor = torch.from_numpy(golden)
    # out_dtype = torch.bfloat16 if isBF16 == True else torch.float16
    return golden_tensor

# pylint:disable = huawei-too-many-arguments
def gmm_a8w4_golden_fakequant(x_in, weight_in, bias_in, scale_in, groupList_in, perTokenScale_in, isBF16):
    weightNz = weight_in.astype(np.int8)
    groupNum = groupList_in.shape[0]
    m = x_in.shape[0]
    k = x_in.shape[1]
    n = scale_in.shape[2]
    isNew = scale_in.shape[1] == 1
    weight = weightNz.reshape(groupNum, k, n)
    scaleUint32 = scale_in.astype(np.uint32)
    scaleUint32.dtype = np.float32
    out = non_quant_golden_fakequant(x_in, weight, scaleUint32, perTokenScale_in, groupList_in, bias_in, isBF16, isNew)
    return out
 



def pack_4bit_to_bytes(data):
    data = data.astype(np.int8)
    data[data < 0] += 16
    packed = (data[..., 1::2] << 4) | (data[..., ::2] & 0x0F)
    return packed


def non_quant_golden(x, weight, scale, perTokenScale, groupList, bias, isBF16, isNew = False, offset = None):
    groupNum, k, n = weight.shape
    quantGroupNum = scale.shape[1]
    index = np.cumsum(groupList)
    xSplit = np.split(x, index * 2, axis=0)
    perTokenScaleSplit = np.split(perTokenScale, index, axis=0)
    weightGroup = weight.reshape(groupNum, quantGroupNum, k // quantGroupNum, n).astype(np.int32)
    mmOuts = []
    atomic = np.float16
    if isNew:
        atomic = np.float32
    # for i in range(groupNum):
    #     xi = xSplit[i].reshape(-1, quantGroupNum, k // quantGroupNum).astype(np.int32)
    #     mmi = np.zeros([xi.shape[0], n], dtype=atomic)
    #     for j in range(quantGroupNum):
    #         mm = np.matmul(xi[:, j, :].astype(np.float64), weightGroup[i, j, ...].astype(np.float64)).astype(np.float32)
    #         mm = mm.astype(np.float32) * scale[i, j].reshape(1, -1)
    #         mmi = (mmi.astype(atomic) + mm.astype(atomic)).astype(atomic)
    #         #print("mmi",mmi)

    #     mmi = mmi.reshape(-1, 2, n).astype(np.float32)
    #     #print("mmi 1",mmi)
    #     mmi = mmi[:, 0, :] * 16 + mmi[:, 1, :] + bias[i].reshape(1, n)
    #     #print("mmi 2",mmi)
    #     mmi = mmi * perTokenScaleSplit[i]
    #     mmOuts.append(mmi)
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
    return golden_tensor
    # if isBF16:
    #     return golden_tensor.to(torch.bfloat16)
    # else:
    #     return golden_tensor.to(torch.float16)


# pylint:disable = huawei-too-many-arguments
def gmm_a8w4_golden(x_in, weight_in, bias_in, scale_in, groupList_in, perTokenScale_in, isBF16, tuningConfig = 0, offset=None):
    weightNz = weight_in.astype(np.int8)
    groupNum = groupList_in.shape[0]
    m = x_in.shape[0]
    k = x_in.shape[1]
    n = scale_in.shape[2]
    # const int is_in_a8w4_white_list = ((tuningConfig !=0 &&tuningConfig <= 64) || (A8W4_PRETILING_WHITE_LIST.count(mKNList)))
    #         && quantGroupNum != 0 && k / quantGroupNum == 256 && k % quantGroupNum == 0
    #         && withOffset == 0 && k%64==0 && n%64==0;
    isNew = ((tuningConfig !=0 and tuningConfig <= 64) or (groupNum == 256 and m == 1024 and k == 512 and n == 32768)) and k // scale_in.shape[1] == 256 and k % scale_in.shape[1] == 0 and k%64==0 and n%64==0
    if offset is not None:
        isNew = False
    print("[gmm_a8w4_golden] isNew:",isNew, tuningConfig, groupNum, m, k, n)

    weight = weightNz.reshape(groupNum, k, n)
    #print("11")
    xC12 = np.concatenate([x_in.reshape(m, 1, k) // 16, (x_in.reshape(m, 1, k) & 0x0F) - 8], axis=1).reshape(m * 2, k)
    #print("22")
    # pack_4bit_to_bytes(xC12).tofile("data/a8w4_pre_process_golden_i4.bin")
    # xC12.tofile("data/a8w4_pre_process_golden_i8.bin")
    scaleUint32 = scale_in.astype(np.uint32)
    scaleUint32.dtype = np.float32
    #print("33")
    out = non_quant_golden(xC12, weight, scaleUint32, perTokenScale_in, groupList_in, bias_in, isBF16, isNew, offset)
    return out

def run_test(E, M, K, N, isBF16, G = 0, G_num = 0, quantGroupSize = 256, tuningConfig = 0, device=0, data_path="", withOffset=False):
    FIX_VALUES = True
    DISPLAY_VARS = False
    torch.manual_seed(42)  # 42 可以是任意整数
    x = torch.randint(-5,5, (M, K), device="cpu").to(torch.int8)
    # x = torch.full((M, K), 25, device="cpu").to(torch.int8)
    # for i in range(0,M):
    #     x[i] = (sss+i*2)[0]#torch.arange(i,K+i ).to(torch.int8)
    # x[0] = torch.tensor([(9) % 100] * (K) ).to(torch.int8)
    # x[1] = torch.tensor([(8+16) % 100] * (K) ).to(torch.int8)
    # x[2] = torch.tensor([(8+16+1) % 100] * (K) ).to(torch.int8)
    # A8W4 will inplace change the value of x.
    x_copy = x.clone()
    weight = torch.randint(-5, 5, (E, K, N), dtype=torch.int32, device="cpu")
    # weight = torch.ones(E, K, N, dtype=torch.int32, device="cpu")
    # weight[0][0][0]=0
    # weight[0][1024][0]=1
    # weight[0][1024*2][0]=2
    # weight[0][1024*3][0]=4
    # weight[0][0][512]=-1
    # weight[0][1024][512]=-2
    # weight[0][1024*2][512]=-4
    # weight[0][1024*3][512]=-8
    # weight[0][0][0+1024]=0
    # weight[0][1024][0+1024]=1
    # weight[0][1024*2][0+1024]=2
    # weight[0][1024*3][0+1024]=4
    # weight[0][0][512+1024]=-1
    # weight[0][1024][512+1024]=-2
    # weight[0][1024*2][512+1024]=-4
    # weight[0][1024*3][512+1024]=-8
    # for i in range(0,K):
    #     weight[0][i] = torch.tensor(range(0,8) * (K//8))#torch.tensor([4] * (4 ) + [i % 3] * (N - 4) )
    
    
    # weight_quant = torch_cpu.cpu_quantize(weight.to(torch.float32), torch.tensor([1.]).cpu(), None, torch.quint4x2, -1, False)

    print(weight.shape)
    #print(weight_quant.shape)

    scale_np = np.random.normal(0, 0.01, (E, 1, N)).astype(np.float32)
    # if FIX_VALUES == True:
    # scale_np = np.full([E, 1, N], 1).astype(np.float32)
    #print(scale_np)
    #scale_np[0][0]  =  np.arange(0,N) / 100
    perGroupScale = np.ones([E, K//quantGroupSize, N]).astype(np.float32)
    scaleUint32 = (scale_np * perGroupScale).astype(bf16).astype(np.float32)

    weightHigh = weight.numpy().astype(np.float32).reshape([E, K//quantGroupSize, -1, N]) * perGroupScale.reshape([E, K//quantGroupSize, 1, N])
    weightHigh = weightHigh.reshape([E, K, N])
    bias = torch.tensor(8 * (weightHigh.astype(np.float32) * scale_np).sum(axis=1))
    # bias = torch.full(bias.shape, 0, device="cpu").to(bias.dtype)
    
    scaleUint32.dtype = np.uint32
    scaleUint64 = np.zeros((E, K//quantGroupSize, N*2), dtype=np.uint32)
    scaleUint64[...,::2] = scaleUint32
    scaleUint64.dtype = np.int64
    scale = torch.from_numpy(scaleUint64).cpu()
    if G == 0:
        groupList = torch.tensor([M//E]*E).cpu()
    else:
        if G_num == 0:
            groupList = torch.tensor([G]*E).cpu()
        else:
            groupList = torch.tensor([G]*G_num+[0]*(E-G_num)).cpu()
    #groupList = torch.tensor([15, 44, 43, 42, 35, 1, 41, 8, 1, 44, 5, 40, 33, 42, 44, 35, 10, 27, 10, 26, 26, 31, 34, 12, 13, 14, 43, 7, 17, 27, 26, 32, 11, 37, 20, 8, 37, 41, 37, 1, 20, 16, 17, 28, 25, 20, 44, 13, 34, 35, 45, 8, 20, 6, 42, 24, 9, 37, 43, 7, 34, 10, 39, 5, 13, 40, 15, 33, 32, 23, 37, 33, 20, 13, 44, 4, 4, 31, 43, 21, 6, 2, 26, 7, 21, 23, 20, 5, 31, 5, 17, 6, 31])
    print("[msd.py] groupList",groupList)
    perTokenScale = torch.zeros((M,1), dtype=torch.float32, device="cpu").uniform_()
    # perTokenScale = torch.ones((M,1), dtype=torch.float32, device="cpu")
    #perTokenScale = torch.tensor(np.arange(1,M+1).reshape(M,1) / 10, dtype=torch.float32)
    if K//quantGroupSize != 1 and withOffset == True:      # 只有perchannel支持offset
        print("[WARNING] Only perchannel MSD scenario supports offset! withOffset = False")
        withOffset = False
    if withOffset:
        offset = torch.randint(-5, 5, scale.shape, dtype=torch.float32, device="cpu")
    else:
        offset = None
    
    print("----op input----")
    if DISPLAY_VARS:
        print("x",x.shape,x.dtype,x)
        print("weight",weight.shape,weight.dtype,weight)
        print("bias",bias.shape,bias.dtype,bias)
        print("scale",scale.shape,scale.dtype,scale_np)
        print("groupList",groupList.shape,groupList.dtype,groupList)
        print("perTokenScale",perTokenScale.shape,perTokenScale.dtype,perTokenScale)
    else:
        print("x",x.shape,x.dtype)
        print("weight",weight.shape,weight.dtype)
        print("bias",bias.shape,bias.dtype)
        print("scale",scale.shape,scale.dtype)
        print("groupList",groupList.shape,groupList.dtype)
        print("perTokenScale",perTokenScale.shape,perTokenScale.dtype)
    
    # out = torch_cpu.cpu_grouped_matmul([x], [weight_quant], bias=[bias], scale=[scale], offset=None, antiquant_scale=None,
    #                                    antiquant_offset=None, per_token_scale=[perTokenScale], group_list=groupList,
    #                                    activation_icput=None, activation_quant_scale=None, activation_quant_offset=None,
    #                                    split_item=3, group_type=0, group_list_type=1, act_type=0, output_dtype=torch.float16)
    
    
    x_in = x_copy.cpu().numpy()
    weight_in = weight.cpu().numpy()
    bias_in = bias.cpu().numpy()
    scale_in = scaleUint64
    groupList_in = groupList.cpu().numpy()
    perTokenScale_in = perTokenScale.cpu().numpy()
    if withOffset:
        offset_in = offset.cpu().numpy()
    else:
        offset_in = None
    
    # print("----golden icput----")
    # print("x_in",x_in.shape,x_in.dtype)
    # print("weight_in",weight_in.shape,weight_in.dtype)
    # print("bias_in",bias_in.shape,bias_in.dtype)
    # print("scale_in",scale_in.shape,scale_in.dtype)
    # print("groupList_in",groupList_in.shape,groupList_in.dtype,groupList_in)
    # print("perTokenScale_in",perTokenScale_in.shape,perTokenScale_in.dtype)
    #np.sum(groupList_in)
    
    print("-----running golden...--------")
    out_golden = torch.tensor([])
    # const bool isMSD = (tuningConfig == 0L || avg_m == 0U || n / avg_m > 4U || withOffset == true) &&
    #             !(biasPtr == nullptr || biasPtr->GetStorageShape().GetShapeSize() == 0);
    if tuningConfig == 0:
        avg_m = M // E
    else:
        avg_m = tuningConfig
        
    isMSD = tuningConfig == 0 or avg_m == 0 or N // avg_m > 4 or withOffset == True
    print("[golden] isMSD:", isMSD, tuningConfig, avg_m, N // avg_m, N)
    if isMSD:
        out_golden = gmm_a8w4_golden(x_in=x_in, weight_in=weight_in, bias_in=bias_in,
                                 scale_in=scale_in, groupList_in=groupList_in, perTokenScale_in=perTokenScale_in, isBF16=isBF16, tuningConfig=tuningConfig, offset=offset_in)
    else:
        out_golden = gmm_a8w4_golden_fakequant(x_in=x_in, weight_in=weight_in, bias_in=bias_in,
                                scale_in=scale_in, groupList_in=groupList_in, perTokenScale_in=perTokenScale_in, isBF16 = isBF16)

    if isBF16:
        print("convert to bf16")
        golden = out_golden.cpu().numpy().astype(bf16)
    else:
        golden = out_golden.cpu().numpy().astype(np.float16)

    if False:
        out_shape = out[0].shape
        golden_shape = out_golden.shape
        
        out_dim1 = out_shape[1]
        
        golden_dim0 = golden_shape[0]
        golden_dim1 = golden_shape[1]
        
        assert out_dim1 == golden_dim1

        result1 = out[0][:golden_dim0, :]
        result2 = out_golden.to(torch.float16).cpu()
        print("compare result:", torch.equal(result1, result2))

    print("-----writing bins...--------")
    weightI4 = pack_4bit_to_bytes(weight_in)
    weightI4Nz = pack_4bit_to_bytes(weight_in.reshape([E, K // 16, 16, N // 64, 64]).transpose([0, 3, 1, 2, 4]))
    weightI4Nz.tofile(f"{data_path}/a8w4_nz_test_weight_nz.bin")

    x_in.tofile(f"{data_path}/a8w4_nz_test_x.bin")
    weightI4.tofile(f"{data_path}/a8w4_nz_test_weight_nd.bin")
    bias_in.tofile(f"{data_path}/a8w4_nz_test_bias.bin")
    scale_in.tofile(f"{data_path}/a8w4_nz_test_scale.bin")
    if withOffset:
        offset_in.tofile(f"{data_path}/a8w4_nz_test_offset.bin")
    scaleUint32.tofile(f"{data_path}/a8w4_nz_test_scale_fp32.bin")
    groupList_in.tofile(f"{data_path}/a8w4_nz_test_group_list.bin")
    perTokenScale_in.tofile(f"{data_path}/a8w4_nz_test_per_token_scale.bin")
    golden.tofile(f"{data_path}/a8w4_nz_test_golden.bin")



E = 5
M = 64
K = 2048
N = 512

quantGroupSize = K # 256
isBF16 = False
G = 0
G_num = 0
withOffset = False

if len(sys.argv) > 7:
    E = int(sys.argv[1])
    M = int(sys.argv[2])
    K = int(sys.argv[3])
    N = int(sys.argv[4])
    isBF16 = sys.argv[5] == "bf16"
    G = int(sys.argv[6])
    G_num = int(sys.argv[7])
if len(sys.argv) > 8:
    quantGroupSize = int(sys.argv[8])
    tuningConfig = int(sys.argv[9])
if len(sys.argv) > 10:
    withOffset = sys.argv[10] == "withOffset" or sys.argv[10] == "true"


print("gen","E",E, "M",M, "K",K, "N",N, "isBF16",isBF16,"G", G,"G_num", G_num, "quantGroupSize",quantGroupSize,"tuningConfig",tuningConfig)

import os

# 获取环境变量，不存在返回 None
path = os.getenv('A8W4_DATA_PATH')
if path:
    data_path = path
else:
    data_path = "data/"

run_test(E, M, K, N, isBF16, G,G_num,quantGroupSize, tuningConfig=tuningConfig, device=3, data_path=data_path, withOffset = withOffset)