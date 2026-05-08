#############################################################
import torch
import torch_npu
torch.npu.utils.set_device(6)
torch_npu.npu.config.allow_internal_format=True
import numpy as np
def prepare_scale_parameter(scale, E, k_group, N):
   """
   准备scale参数，进行特殊的格式转换
   
   Args:
      scale: 原始scale张量
      E: 专家数量
      k_group: k分组数
      N: 输出特征维度
      
   Returns:
      转换后的scale张量
   """
   scale_np = scale.cpu().numpy()
   scale_uint32 = scale_np.astype(np.float32)
   scale_uint32.dtype = np.uint32
   scale_uint32 = scale_uint32.reshape(E, -1, N)
   scale_uint64 = np.zeros((E, k_group, N * 2), dtype=np.uint32)
   scale_uint64[..., ::2] = scale_uint32
   scale_uint64.dtype = np.int64
   return torch.from_numpy(scale_uint64).npu()

def gmm_a8w4(e, m, k, n, token_per_e = None, GroupSize = None, tuningConfig=None):
    # x = torch.randint(low=-128, high=127, size=(m, k), dtype=torch.int16).to(torch.int8).npu()
    #构造全1数据
    x = torch.full((m, k), 25, dtype=torch.int8).npu()
    # x = torch.full((m, k), 25, dtype=torch.int8)
    x[0, 0] = 126    # 第一个元素设为127
    x[-1, -1] = 126
    # x = x.npu()
    #递增
    # x = torch.arange(start=1, end=128, dtype=torch.int8).repeat(m * k // 127 + 1)[:m * k].reshape(m, k).npu()
    # print(x)
    # w = torch.full((e, k, n // 8), 286331153, dtype=torch.int32).npu()
    # w = torch.randint(low=-5, high=5, size=(e, k, n), dtype=torch.int8).npu()
    w = torch.full((e, k, n), 1, dtype=torch.int8).npu()
    # weight_nz = torch_npu.npu_format_cast(w.to(torch.float32), 29)
    weight_nz = torch_npu.npu_format_cast(w.to(torch.float32), 29)
    weight_quant = torch_npu.npu_quantize(
            weight_nz,
            torch.tensor([1.0]).npu(),
            None,
            torch.quint4x2,
            -1,
            False
        )
    w = weight_quant
    # w = torch.randint(low=-2147483648, high=2147483647, size=(e, k, n // 8), dtype=torch.int32).npu()
    if token_per_e == None:
        token_per_e = m // e
    group_list = torch.full((e,), token_per_e, dtype=torch.int64).npu()
    bias = torch.zeros((e, n), dtype=torch.float32).npu()
    if GroupSize == None:
        GroupSize = k
    # scale = torch.randn((e, n), dtype=torch.float32).npu()
    scale = torch.ones((e, n), dtype=torch.float32).npu()
    scale = prepare_scale_parameter(scale, e, 1, n)
    # scale = torch.randint(low=-2147483648, high=2147483647, size=(e, k // GroupSize, n), dtype=torch.int64).npu()
    per_token_scale = torch.rand((m), dtype=torch.float32).npu()
    # w = torch_npu.npu_format_cast(w, 29)
 
    out = torch_npu.npu_grouped_matmul([x], [w], bias = [bias], scale = [scale], per_token_scale = [per_token_scale], group_list_type=1, group_list = group_list, group_type = 0,split_item = 3, output_dtype =torch.bfloat16, tuning_config=tuningConfig)

e = 1
m = 8 
k = 128
n = 64
# gmm_a8w4(e, m, k, n, token_per_e = None, GroupSize = None, tuningConfig=[0,0,-1])
gmm_a8w4(e, m, k, n, token_per_e = None, GroupSize = 64, tuningConfig=[0,0,-1])
