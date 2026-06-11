import torch, torch_npu, math, time, sys, os, importlib.util

spec = importlib.util.spec_from_file_location("mod_0611", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "0611.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
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
with torch.no_grad():
    k_mqa=key.squeeze(2); kr_mqa=kr.squeeze(2)
    qf=torch.cat([query,qr],-1); kf=torch.cat([k_mqa,kr_mqa],-1); kft=kf.transpose(-1,-2).unsqueeze(1)
    row=torch.matmul(qf[:,0:1],kft).float()*scale
    rm=row.max(-1).values; rs=torch.exp(row-rm.unsqueeze(-1)).sum(-1)
    sm_max=rm.reshape(B,S1,N2,G).permute(0,2,1,3).contiguous().float()
    sm_sum=rs.reshape(B,S1,N2,G).permute(0,2,1,3).contiguous().float()

for _ in range(3): fn(query,key,qi,ki,w,si,sm_max,sm_sum,scale,query_rope=qr,key_rope=kr,layout_query="BSND",actual_seq_lengths_query=[S1],actual_seq_lengths_key=[S2],sparse_mode=3); torch.npu.synchronize()

N=5
total_times=[]
gather_times=[]
scatter_times=[]
for _ in range(N):
    t0=time.perf_counter()
    gk,gki=mod._preprocess_gather(key,ki,si,B,S1,S2,N2,Nidx2,K,DQuery,DQueryIndex)
    torch.npu.synchronize()
    gather_times.append((time.perf_counter()-t0)*1e3)

    tiling=mod.check_param_valid(query,key,qi,ki,w,si,sm_max,sm_sum,sparse_mode=3,layout_query="BSND")
    d_rope_orig=qr.shape[-1]
    _sparse_idx_rope=si[:,:,0,:].clamp(min=0)
    key_rope_2d=kr[:,:,0,:]
    idx_rope=_sparse_idx_rope.reshape(B,S1*K).unsqueeze(-1).expand(B,S1*K,d_rope_orig)
    gkr=torch.gather(key_rope_2d,1,idx_rope).reshape(B,S1,K,d_rope_orig).contiguous()
    _vm=(si[:,:,0,:]>=0).unsqueeze(-1).to(dtype)
    gkr.mul_(_vm)
    d_rope_pad=((d_rope_orig+127)//128)*128 if d_rope_orig>0 else 0
    if d_rope_pad>d_rope_orig and d_rope_orig>0:
        qr_p=torch.nn.functional.pad(qr,(0,d_rope_pad-d_rope_orig))
        gkr=torch.nn.functional.pad(gkr,(0,d_rope_pad-d_rope_orig))
    else:
        qr_p=qr
    gkit=gki.permute(0,1,3,2).contiguous()
    cache_key=(B,S1,S2,N1,N2,G,Nidx1,Nidx2,DQuery,DQueryIndex,K,d_rope_pad,"float16","float",scale)
    if cache_key in mod._kernel_cache:
        func=mod._kernel_cache[cache_key]
    else:
        func=mod.tilelang_grad_kl_loss_single_kernel(B,S1,S2,N1,N2,G,Nidx1,Nidx2,DQuery,DQueryIndex,K,d_rope_pad,"float16","float",scale)
        mod._kernel_cache[cache_key]=func

    ws_key=(B,S1,S2,Nidx1,K,DQueryIndex,dtype,query.device)
    if ws_key in mod._workspace_cache:
        ws=mod._workspace_cache[ws_key]
    else:
        bmm1_ws=torch.zeros(B,S1,N1,K,dtype=torch.float32,device=query.device)
        bmm2_ws=torch.zeros(B,S1,Nidx1,K,dtype=torch.float32,device=query.device)
        pr_ws=torch.zeros(B,S1,K,dtype=torch.float32,device=query.device)
        rsy_ws=torch.zeros(B,S1,K,dtype=torch.float32,device=query.device)
        si_ws=torch.zeros(B,S1,K,dtype=torch.float32,device=query.device)
        dqi_ws=torch.zeros(B,S1,Nidx1,DQueryIndex,dtype=torch.float32,device=query.device)
        dqi_out=torch.zeros(B,S1,Nidx1,DQueryIndex,dtype=dtype,device=query.device)
        dw_out=torch.zeros(B,S1,Nidx1,dtype=dtype,device=query.device)
        rg_out=torch.zeros(B,S1,Nidx1,K,dtype=dtype,device=query.device)
        lo_out=torch.zeros(B,S1,16,dtype=torch.float32,device=query.device)
        ws=(bmm1_ws,bmm2_ws,pr_ws,rsy_ws,si_ws,dqi_ws,dqi_out,dw_out,rg_out,lo_out)
        mod._workspace_cache[ws_key]=ws
    bmm1_ws,bmm2_ws,pr_ws,rsy_ws,si_ws,dqi_ws,dqi_out,dw_out,rg_out,lo_out=ws

    t0=time.perf_counter()
    results=func(query,qr_p,gk,gkr,qi,gki,gkit,w,sm_max,sm_sum,bmm1_ws,bmm2_ws,pr_ws,rsy_ws,si_ws,dqi_out,dw_out,rg_out,lo_out,dqi_ws)
    torch.npu.synchronize()
    kernel_ms=(time.perf_counter()-t0)*1e3

    rg=results[2]
    t0=time.perf_counter()
    dki=mod._compute_d_key_index(rg,qi,si,B,S1,S2,Nidx1,Nidx2,K,DQueryIndex,query.device,dtype)
    torch.npu.synchronize()
    scatter_ms=(time.perf_counter()-t0)*1e3

    total_ms=gather_times[-1]+kernel_ms+scatter_ms
    total_times.append(total_ms)
    print(f"  gather={gather_times[-1]:.3f}ms, kernel={kernel_ms:.3f}ms, scatter={scatter_ms:.3f}ms, total={total_ms:.3f}ms")

print(f"\nAvg: gather={sum(gather_times)/N:.3f}ms, kernel={sum([t-g-s for t,g,s in zip(total_times,gather_times,scatter_times)])/N:.3f}ms, scatter={sum(scatter_times)/N:.3f}ms, total={sum(total_times)/N:.3f}ms")
