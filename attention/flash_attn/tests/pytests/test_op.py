import subprocess                                                                                                                                                                    
import torch                                                                                                                                                                         
import torch_npu                                                                                                                                                                     
import npu_ops_transformer                                                                                                                                                           
                                                                                                                                                                                    
# 初始化 NPU                                                                                                                                                                         
torch_npu.npu.set_device(0)                                                                                                                                                          
                                                                                                                                                                                    
# 创建输入 tensor（可选参数可以传 None）                                                                                                                                             
batch_size = 4 
max_seqlen_q = 128                                                                                                                                                                   
max_seqlen_kv = 128

num_heads_q = 1                                                                                                                                                                     
num_heads_kv = 1                                                                                                                                                                    
head_dim = 128 

mask_mode = 1  # 0: default mask, 3: rightDownCausal, 4: band mask 
win_left = None
win_right = None
layout_q = "TND"
layout_kv = "TND"
layout_out = "TND"

# 创建实际序列长度 tensor（可选）                                                                                                                                                    
cu_seqlens_q = torch.arange(0, max_seqlen_q * batch_size + max_seqlen_q , max_seqlen_q).to(dtype=torch.int32).npu()                                                                                                    
cu_seqlens_kv = torch.arange(0, max_seqlen_kv * batch_size + max_seqlen_kv , max_seqlen_kv).to(dtype=torch.int32).npu()    
seqused_q = torch.full((batch_size,), max_seqlen_q).to(dtype=torch.int32).npu()                                                                                                    
seqused_kv = torch.full((batch_size,), max_seqlen_kv).to(dtype=torch.int32).npu()  

print("cu_seqlens_q:",cu_seqlens_q)
print("seqused_q",seqused_q)

print("cu_seqlens_kv:",cu_seqlens_kv)
print("seqused_kv",seqused_kv)
print(" num_heads_q", num_heads_q)
print("num_heads_kv", num_heads_kv)
print("seqused_kv",seqused_kv)
print("seqused_kv",seqused_kv)
print("seqused_kv",seqused_kv)
                                                                                                                                                                                    
# 调用算子
# result = npu_ops_transformer.ops.npu_flash_attn_metadata(
result =  torch.ops.npu_ops_transformer.npu_flash_attn_metadata(
    # cu_seqlens_q = cu_seqlens_q,
    # cu_seqlens_kv = cu_seqlens_kv,
    # seqused_q = seqused_q,
    # seqused_kv = seqused_kv,

    cu_seqlens_q = cu_seqlens_q,
    cu_seqlens_kv = cu_seqlens_kv,
    # seqused_q = seqused_q,
    # seqused_kv = seqused_kv,
    

    num_heads_q = num_heads_q,
    num_heads_kv = num_heads_kv,
    head_dim = head_dim,

    batch_size = batch_size,
    # max_seqlen_q = max_seqlen_q,
    # max_seqlen_kv = max_seqlen_kv,
    mask_mode = mask_mode,
    win_left = win_left,
    win_right = win_right,
    layout_q = layout_q,
    layout_kv = layout_kv,
    layout_out = layout_out

    # batch_size = None,
    # max_seqlen_q = None,
    # max_seqlen_kv = None,
    # mask_mode = None,
    # win_left = None,
    # win_right = None,
    # layout_q = None,
    # layout_kv = None,
    # layout_out = None
)

# 验证结果                                                                                                                                                                           
print(f"Result shape: {result.shape}")                                                                                                                                               
print(f"Result dtype: {result.dtype}")                                                                                                                                               
print(f"Result device: {result.device}")                                                                                                                                             
print(f"First 10 values: {result[:10].cpu().tolist()}")                                                                                                                              
                                                                                                                                                                                    
# 断言验证                                                                                                                                                                           
shape_size = (((36 + 72) * batch_size * num_heads_kv + 1) * 16 + 4095) // 4096 * 4096
assert result.shape == (shape_size,), f"Expected shape ({shape_size} ,), got {result.shape}"                                                                                                        
assert result.dtype == torch.int32, f"Expected dtype int32, got {result.dtype}"                                                                                                      
assert result.device.type == 'npu', f"Expected device npu, got {result.device.type}"                                                                                                 
                                                                                                                                                                                    
print("✅ Test passed!") 

result =result.cpu()

print("sectionNum:",result[0])

aicNum = 36
aivNum = 72
faSize = 16
fdSize = 16
sectionNum = result[0]
for sectionId in range(sectionNum):
    print("sectionId:",sectionId)
    for i in range(aicNum):
        print("bn2 start ", result[16 + aicNum * faSize * sectionId + faSize * i + 0])
        print("m start ", result[16 + aicNum * faSize * sectionId + faSize * i + 1])
        print("s2 start ", result[16 + aicNum * faSize * sectionId + faSize * i + 2])
        print("bn2 end ", result[16 + aicNum * faSize * sectionId + faSize * i + 3])
        print("m end ", result[16 + aicNum * faSize * sectionId + faSize * i + 4])
        print("s2 end ", result[16 + aicNum * faSize * sectionId + faSize * i + 5])
        print("first fd data workspace idx ", result[16 + aicNum * faSize * sectionId + faSize * i + 6])
    for i in range(aivNum):
        print("fd bn2 idx ", result[16 + aicNum * faSize * sectionNum + aivNum * fdSize * sectionId + fdSize * i + 0])
        print("fd m idx ", result[16 + aicNum * faSize * sectionNum + aivNum * fdSize * sectionId + fdSize * i + 1])
        print("fd workspace idx ", result[16 + aicNum * faSize * sectionNum + aivNum * fdSize * sectionId + fdSize * i + 2])
        print("fd workspace num ", result[16 + aicNum * faSize * sectionNum + aivNum * fdSize * sectionId + fdSize * i + 3])
        print("m start ", result[16 + aicNum * faSize * sectionNum + aivNum * fdSize * sectionId + fdSize * i + 4])
        print("m num ", result[16 + aicNum * faSize * sectionNum + aivNum * fdSize * sectionId + fdSize * i + 5])
