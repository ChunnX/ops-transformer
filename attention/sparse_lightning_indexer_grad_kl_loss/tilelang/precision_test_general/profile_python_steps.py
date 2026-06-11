import torch, torch_npu, math, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
mod = import_module("0611")
fn = mod.sparse_lightning_indexer_grad_kl_loss

B,S1,S2,N1,N2,Nidx1,Nidx2,DQuery,DQueryIndex,d_rope,K = 1,1,2048,128,1,64,1,512,128,64,2048
G = N1//N2; scale = 1.0/math.sqrt(DQuery); dtype = torch.float16
torch.manual_seed(42)
query = torch.randn(B,S1,N1,DQuery,dtype=dtype).npu()
key = torch.randn(B,S2,N2,DQuery,dtype=dtype).npu()
qi = torch.randn(B,S1,Nidx1,DQueryIndex,dtype=dtype).npu()
ki = torch.randn(B,S2,Nidx2,DQueryIndex,dtype=dtype).npu()
qr = torch.randn(B,S1,N1,d_rope,dtype=dtype).npu()
kr = torch.randn(B,S2,N2,d_rope,dtype=dtype).npu()
w = torch.randn(B,S1,Nidx1,dtype=dtype).npu()
si = torch.zeros(B,S1,Nidx2,K,dtype=torch.int32).npu()
indices = torch.randperm(S2)[:K]
si[0,0,0,:]=indices.int().npu()
# compute softmax_max/sum
with torch.no_grad():
    k_mqa=key.squeeze(2); kr_mqa=kr.squeeze(2)
    qf=torch.cat([query,qr],-1); kf=torch.cat([k_mqa,kr_mqa],-1); kft=kf.transpose(-1,-2).unsqueeze(1)
    row=torch.matmul(qf[:,0:1],kft).float()*scale
    rm=row.max(-1).values; rs=torch.exp(row-rm.unsqueeze(-1)).sum(-1)
    sm_max=rm.reshape(B,S1,N2,G).permute(0,2,1,3).contiguous().float()
    sm_sum=rs.reshape(B,S1,N2,G).permute(0,2,1,3).contiguous().float()

# warmup
for _ in range(3): fn(query,key,qi,ki,w,si,sm_max,sm_sum,scale,query_rope=qr,key_rope=kr,layout_query="BSND",actual_seq_lengths_query=[S1],actual_seq_lengths_key=[S2],sparse_mode=3); torch.npu.synchronize()

# measure total
t0=time.perf_counter(); fn(query,key,qi,ki,w,si,sm_max,sm_sum,scale,query_rope=qr,key_rope=kr,layout_query="BSND",actual_seq_lengths_query=[S1],actual_seq_lengths_key=[S2],sparse_mode=3); torch.npu.synchronize()
total_ms=(time.perf_counter()-t0)*1e3

# measure gather only
from importlib import import_module
m=import_module("0611")
tiling=m.check_param_valid(query,key,qi,ki,w,si,sm_max,sm_sum,sparse_mode=3,layout_query="BSND")
t0=time.perf_counter()
gk,gki=m._preprocess_gather(key,ki,si,tiling.B,tiling.S1,tiling.S2,tiling.N2,tiling.Nidx2,tiling.K,tiling.DQuery,tiling.DQueryIndex)
torch.npu.synchronize()
gather_ms=(time.perf_counter()-t0)*1e3

print(f"Total: {total_ms:.3f}ms, Gather: {gather_ms:.3f}ms, Kernel+Scatter: {total_ms-gather_ms:.3f}ms")