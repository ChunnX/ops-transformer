

#############################################################
import torch
import torch_npu
torch.npu.utils.set_device(6)
torch_npu.npu.config.allow_internal_format=True
def gmm_a8w4(e, m, k, n, token_per_e = None, GroupSize = None, tuningConfig=None):
    # x = torch.randint(low=-128, high=127, size=(m, k), dtype=torch.int16).to(torch.int8).npu()
    #构造全1数据
    x = torch.full((m, k), 25, dtype=torch.int8).npu()
    # 构造：第一个数=127，其余全=25
    # x = torch.full((m, k), 25, dtype=torch.int8)
    x[0, 0] = 126    # 第一个元素设为127
    x[-1, -1] = 126
    # x = x.npu()
    #递增
    # x = torch.arange(start=1, end=128, dtype=torch.int8).repeat(m * k // 127 + 1)[:m * k].reshape(m, k).npu()
    # print(x)
    w = torch.full((e, k, n // 8), 286331153, dtype=torch.int32).npu()
    # w = torch.randint(low=-2147483648, high=2147483647, size=(e, k, n // 8), dtype=torch.int32).npu()
    if token_per_e == None:
        token_per_e = m // e
    group_list = torch.full((e,), token_per_e, dtype=torch.int64).npu()
    bias = torch.rand((e, n), dtype=torch.float32).npu()
    if GroupSize == None:
        GroupSize = k
    scale = torch.randint(low=-2147483648, high=2147483647, size=(e, k // GroupSize, n), dtype=torch.int64).npu()
    per_token_scale = torch.rand((m), dtype=torch.float32).npu()
    w = torch_npu.npu_format_cast(w, 29)
 
    out = torch_npu.npu_grouped_matmul([x], [w], bias = [bias], scale = [scale], per_token_scale = [per_token_scale], group_list_type=1, group_list = group_list, group_type = 0,split_item = 3, output_dtype =torch.bfloat16, tuning_config=tuningConfig)

e = 1
m = 8 
k = 64
n = 64
# gmm_a8w4(e, m, k, n, token_per_e = None, GroupSize = None, tuningConfig=[0,0,-1])
gmm_a8w4(e, m, k, n, token_per_e = None, GroupSize = 64, tuningConfig=[0,0,-1])
