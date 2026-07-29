#!/usr/bin/env python3
"""Comprehensive multi-shape determinism and precision validation."""

import torch
import torch_npu
import math
import numpy as np
from einops import repeat
import random

torch.set_printoptions(precision=6)
seed = 128; torch.manual_seed(seed); device = "npu:7"; DTYPE = torch.float16
torch.npu.empty_cache()

def compera(golden, out):
    golden = golden.cpu().flatten().to(torch.float32); out = out.cpu().flatten().to(torch.float32)
    res = torch.allclose(golden, out, rtol=1e-03, atol=1e-03)
    if not res:
        d = torch.abs(out - golden)
        return res, d.max().item(), torch.sum(~torch.isclose(golden, out, rtol=1e-3, atol=1e-3)).item()
    return res, 0.0, 0

def deter_check(a, b):
    a = a.cpu().flatten().to(torch.float32); b = b.cpu().flatten().to(torch.float32)
    if torch.equal(a, b): return True, 0.0
    d = torch.abs(a - b)
    return torch.allclose(a, b, rtol=0, atol=0), d.max().item()

def attention_ori_GQA(q_all, k_all, v_all, scale):
    nG = q_all.shape[1] // k_all.shape[1]
    for i in range(q_all.shape[1]):
        g = i // nG
        q = q_all[:, i:i+1].float(); k = k_all[:, g:g+1].float(); v = v_all[:, g:g+1].float()
        bm = torch.matmul(q, torch.transpose(k,2,3)).float() * scale
        mx,_ = torch.max(bm, dim=-1, keepdim=True); ex = torch.exp(bm-mx)
        sm = torch.from_numpy(np.sum(ex.numpy(),-1,keepdims=True)); so = ex/sm
        attn = torch.matmul(so, v); l = mx + torch.log(sm)
        if i == 0: tl=l; ta=attn
        else: tl=torch.cat((tl,l),1); ta=torch.cat((ta,attn),1)
    return tl, ta.to(DTYPE)

def attention_gqa_tnd(q, k, v, scale, cu_q, cu_k):
    q=q.to(torch.float32); k=k.to(torch.float32); v=v.to(torch.float32); dev=q.device
    tq,nh,hd=q.shape; tk,nhk,_=k.shape; bs=len(cu_q)-1; nG=nh//nhk
    out=torch.zeros_like(q); lse=torch.zeros((tq,nh,1),dtype=torch.float32,device=dev)
    for i in range(nh):
        g=i//nG
        for b in range(bs):
            qs,qe=cu_q[b],cu_q[b+1]; ks,ke=cu_k[b],cu_k[b+1]
            qb=q[qs:qe,i:i+1,:].permute(1,0,2); kb=k[ks:ke,g:g+1,:].permute(1,0,2); vb=v[ks:ke,g:g+1,:].permute(1,0,2)
            bm=torch.matmul(qb,torch.transpose(kb,1,2)).float()*scale
            mx,_=torch.max(bm,dim=-1,keepdim=True); ex=torch.exp(bm-mx)
            sm=torch.from_numpy(np.sum(ex.cpu().numpy(),-1,keepdims=True)).to(dev); so=ex/sm
            out[qs:qe,i:i+1,:]=torch.matmul(so,vb).permute(1,0,2); lse[qs:qe,i:i+1,:]=(mx+torch.log(sm)).permute(1,0,2)
    return lse.contiguous(), out.to(DTYPE).contiguous()

def gen_mask(sq,sk,bx,by,B,N,sp):
    nr=(sq+bx-1)//bx; nc=(sk+by-1)//by; m=torch.zeros(B,N,nr,nc,dtype=torch.uint8)
    torch.manual_seed(seed)
    for b in range(B):
        for h in range(N):
            if sp==1.0: m[b][h]=1
            elif sp!=0.0:
                for i in range(nr): m[b][h][i,torch.randperm(nc)[:max(1,int(sp*nc))]]=1
    return m

def tmask(m,sq,sk):
    b,n=m.shape[0],m.shape[1]; m=m[:,:,:sq,:sk]
    p=torch.zeros(b,n,sq,sk,dtype=torch.bool); p[:,:,:m.shape[2],:m.shape[3]]=m; return p.contiguous()

def qk_masked(q,k,mask,bs,bi=-1,hi=-1):
    kt=torch.transpose(k,2,3).contiguous(); S=torch.einsum("bnqd,bndk->bnqk",q,kt)
    fm=repeat(mask,"b n s_m s_n -> b n (s_m d_m) (s_n d_n)",d_m=bs[0],d_n=bs[1])
    fm=tmask(fm,q.shape[2],k.shape[2]); nm=~fm
    if bi!=-1: nm=nm[bi:bi+1]
    if hi!=-1: nm=nm[:,hi:hi+1]
    return S.masked_fill(nm,float('-inf'))

def tbd(a,b):
    bs,n,h,_=a.shape; _,_,_,w=b.shape; a=a.float(); b=b.float()
    r=torch.zeros(bs,n,h,w,dtype=torch.float)
    for i in range(bs):
        for j in range(n): r[i,j]=torch.matmul(a[i,j],b[i,j])
    return r

def cpu_grad_bnsd(dout,q,k,v,out,lse,mask,bs,scale,bi=-1):
    nG=q.shape[1]//k.shape[1]; B,H,_,D=q.shape
    # Match NPU: accumulate in FP32 without scale, apply scale at end
    dq_unsat=torch.zeros(B,H,q.shape[2],D,dtype=torch.float32)
    dk_unsat=torch.zeros(B,k.shape[1],k.shape[2],D,dtype=torch.float32)
    dv_unsat=torch.zeros(B,k.shape[1],k.shape[2],D,dtype=torch.float32)
    for i in range(H):
        g=i//nG; a=out[:,i:i+1].float();ag=dout[:,i:i+1].float();li=lse[:,i:i+1,:,:].float()
        qi=q[:,i:i+1].float();ki=k[:,g:g+1].float();vi=v[:,g:g+1].float()
        b2,h2,ql,_=qi.shape;li=li.view(b2,h2,ql,1)
        Dv=(a*ag).sum(-1,keepdim=True).reshape(b2,h2,ql,1)
        S=qk_masked(qi,ki,mask,bs,bi,i)*scale; dP=tbd(ag,torch.transpose(vi,2,3)).float()
        P=torch.exp(S-li); dS=(dP-Dv)*P
        # NPU stores dS and P as FP16 in workspace, then reads back for Mmad
        dS_fp16=dS.to(DTYPE).float()
        P_fp16=P.to(DTYPE).float()
        # Accumulate WITHOUT scale (NPU applies scale in VecPost after reduction)
        dq_unsat[:,i:i+1]=tbd(dS_fp16,ki)
        dk_unsat[:,g:g+1]+=tbd(torch.transpose(dS_fp16,2,3),qi)
        dv_unsat[:,g:g+1]+=tbd(torch.transpose(P_fp16,2,3),ag)
    # Apply scale at end (matching NPU VecPost: Muls then Cast)
    dq=(dq_unsat*scale).to(DTYPE)
    dk=(dk_unsat*scale).to(DTYPE)
    dv=dv_unsat.to(DTYPE)
    return dq,dk,dv

def cpu_grad_tnd(dout,q,k,v,out,lse,mask,bs,cu_q,cu_kv,scale,sqm,skm):
    tq,H,D=q.shape;tkv,KH,_=k.shape;B=len(cu_q)-1
    q=q.float();k=k.float();v=v.float();dout=dout.float();out=out.float()
    lse=lse.float().reshape(tq,H,1).contiguous()
    dq=torch.zeros_like(q);dk=torch.zeros_like(k);dv=torch.zeros_like(v)
    def t2b(t,a,b,n,h,d): return t[a:b].reshape(1,b-a,h,d).permute(0,2,1,3).contiguous()
    for i in range(B):
        qn=cu_q[i+1]-cu_q[i];kvn=cu_kv[i+1]-cu_kv[i]
        qt=t2b(q,cu_q[i],cu_q[i+1],qn,H,D);dt=t2b(dout,cu_q[i],cu_q[i+1],qn,H,D)
        ot=t2b(out,cu_q[i],cu_q[i+1],qn,H,D);kt=t2b(k,cu_kv[i],cu_kv[i+1],kvn,KH,D)
        vt=t2b(v,cu_kv[i],cu_kv[i+1],kvn,KH,D);lt=t2b(lse,cu_q[i],cu_q[i+1],qn,H,1)
        dqt,dkt,dvt=cpu_grad_bnsd(dt,qt,kt,vt,ot,lt,mask,bs,scale,i)
        dq[cu_q[i]:cu_q[i+1]]=dqt.permute(0,2,1,3)[0:1]
        dk[cu_kv[i]:cu_kv[i+1]]=dkt.permute(0,2,1,3)[0:1]
        dv[cu_kv[i]:cu_kv[i+1]]=dvt.permute(0,2,1,3)[0:1]
    return dq.to(DTYPE),dk.to(DTYPE),dv.to(DTYPE)

def run_test(desc,B,N,S1,S2,D,kvh,sp,fmt,block_shape=(128,128)):
    FMT=fmt.upper(); scale=1.0/math.sqrt(D)
    print(f"\n{'='*70}\nTEST: {desc} | B={B} N={N} KV={kvh} S1={S1} S2={S2} sp={sp} fmt={FMT} g={N//kvh}\n{'='*70}")
    try:
        if fmt=="tnd":
            random.seed(seed);torch.manual_seed(seed)
            sql=[random.randint(1,S1) for _ in range(B)];svl=[random.randint(1,S2) for _ in range(B)]
            aqm=max(sql);akm=max(svl)
            cu_q=[0];[cu_q.append(cu_q[-1]+s) for s in sql]
            cu_kv=[0];[cu_kv.append(cu_kv[-1]+s) for s in svl]
            q=torch.randn(cu_q[-1],N,D,dtype=DTYPE);k=torch.randn(cu_kv[-1],kvh,D,dtype=DTYPE)
            v=torch.randn(cu_kv[-1],kvh,D,dtype=DTYPE);do=torch.randn(cu_q[-1],N,D,dtype=DTYPE)
            mask=gen_mask(aqm,akm,block_shape[0],block_shape[1],B,N,sp)
            lse,attn=attention_gqa_tnd(q,k,v,scale,cu_q,cu_kv)
            cpu_dq,cpu_dk,cpu_dv=cpu_grad_tnd(do,q,k,v,attn,lse,mask,block_shape,cu_q,cu_kv,scale,S1,S2)
        else:
            sql=[S1]*B;svl=[S2]*B
            q=torch.randn(B,N,S1,D,dtype=DTYPE);k=torch.randn(B,kvh,S2,D,dtype=DTYPE)
            v=torch.randn(B,kvh,S2,D,dtype=DTYPE);do=torch.randn(B,N,S1,D,dtype=DTYPE)
            mask=gen_mask(S1,S2,block_shape[0],block_shape[1],B,N,sp)
            lse,attn=attention_ori_GQA(q,k,v,scale)
            cpu_dq,cpu_dk,cpu_dv=cpu_grad_bnsd(do,q,k,v,attn,lse,mask,block_shape,scale)
    except Exception as e:
        print(f"  CPU ERR: {e}");return desc,False,False

    torch.npu.synchronize();torch.npu.empty_cache()
    results=[];outputs=[]
    for j in range(5):
        try:
            dq,dk,dv=torch_npu.npu_block_sparse_attention_backward(
                do.to(device),q.to(device),k.to(device),v.to(device),
                attn.to(device),lse.to(device),mask.to(device),
                block_shape=block_shape,actual_seq_lengths=sql,actual_seq_lengths_kv=svl,
                q_input_layout=FMT,kv_input_layout=FMT,num_key_value_heads=kvh,scale_value=scale)
            torch.npu.synchronize()
            results.append((compera(cpu_dq,dq),compera(cpu_dk,dk),compera(cpu_dv,dv)))
            outputs.append((dq.clone(),dk.clone(),dv.clone()))
        except Exception as e:
            print(f"  NPU iter {j} ERR: {e}")
            results.append(((False,-1,-1),(False,-1,-1),(False,-1,-1)));outputs.append(None)
        torch.npu.empty_cache()

    all_prec=True
    for j,(r1,r2,r3) in enumerate(results):
        ok=r1[0]and r2[0]and r3[0];all_prec=all_prec and ok
        print(f"  iter {j}: [{'PASS' if ok else 'FAIL'}] dq={'OK' if r1[0] else f'F({r1[1]:.4f},{r1[2]})'} dk={'OK' if r2[0] else f'F({r2[1]:.4f},{r2[2]})'} dv={'OK' if r3[0] else f'F({r3[1]:.4f},{r3[2]})'}")

    max_det_diff=0.0;det_exact=True;det_close=True
    if outputs[0]:
        b0=outputs[0]
        for j in range(1,5):
            if outputs[j] is None: det_exact=False;det_close=False;continue
            for i in range(3):
                exact,diff=deter_check(b0[i],outputs[j][i])
                if not exact: det_exact=False
                if diff>1e-3: det_close=False
                max_det_diff=max(max_det_diff,diff)
    else: det_exact=False;det_close=False
    print(f"  >>> PREC={'PASS' if all_prec else 'FAIL'} DET_EXACT={'PASS' if det_exact else 'FAIL'} DET_TOL1e3={'PASS' if det_close else 'FAIL'} max_diff={max_det_diff:.6f}")
    return desc,all_prec,det_exact,det_close,max_det_diff

if __name__=="__main__":
    cases=[
        # GQA BNSD - various group sizes, batch, seq lens, sparsity
        ("GQA BNSD g4 s128",   2,8,  128,128,128,2, 0.5,"bnsd"),
        ("GQA BNSD g4 s256",   4,16, 256,256,128,4, 0.5,"bnsd"),
        ("GQA BNSD g3 s512",   8,12, 512,512,128,4, 0.3,"bnsd"),
        ("GQA BNSD g4 full",   2,8,  128,256,128,2, 1.0,"bnsd"),
        ("GQA BNSD g2 sparse", 1,16, 256,384,128,8, 0.3,"bnsd"),
        ("GQA BNSD g6 s128",   2,12, 128,128,128,2, 0.5,"bnsd"),
        ("GQA BNSD g8 s256",   1,16, 256,256,128,2, 0.5,"bnsd"),
        ("GQA BNSD g2 s384",   2,8,  384,384,128,4, 0.5,"bnsd"),
        ("GQA BNSD g4 b4s64",  4,8,  64, 64, 128,2, 0.5,"bnsd"),
        ("GQA BNSD g4 sp02",   2,8,  256,256,128,2, 0.2,"bnsd"),
        ("GQA BNSD g4 sp08",   2,8,  256,256,128,2, 0.8,"bnsd"),
        # GQA TND
        ("GQA TND g4 s128",    2,8,  128,128,128,2, 0.5,"tnd"),
        ("GQA TND g4 s256",    4,16, 256,256,128,4, 0.5,"tnd"),
        ("GQA TND g3 s512",    8,24, 512,512,128,4, 0.3,"tnd"),
        ("GQA TND g4 full",    3,12, 256,384,128,4, 1.0,"tnd"),
        ("GQA TND g6 s128",    2,12, 128,128,128,2, 0.5,"tnd"),
        ("GQA TND g2 sp03",    2,8,  256,384,128,4, 0.3,"tnd"),
        ("GQA TND g8 s256",    1,16, 256,256,128,2, 0.5,"tnd"),
        # MHA BNSD
        ("MHA BNSD s128",      2,8,  128,128,128,8, 0.5,"bnsd"),
        ("MHA BNSD s256",      1,4,  256,256,128,4, 1.0,"bnsd"),
        ("MHA BNSD s64",       4,8,  64, 64, 128,8, 0.5,"bnsd"),
        ("MHA BNSD sp03",      2,8,  256,256,128,8, 0.3,"bnsd"),
        # MHA TND
        ("MHA TND s128",       3,8,  128,128,128,8, 0.5,"tnd"),
        ("MHA TND full",       2,4,  256,256,128,4, 1.0,"tnd"),
        ("MHA TND sp03",       2,8,  256,256,128,8, 0.3,"tnd"),
    ]
    res=[run_test(*c) for c in cases]
    print(f"\n{'='*70}\nFINAL SUMMARY\n{'='*70}")
    print(f"{'Test':<25} {'Prec':<8} {'DetExact':<10} {'DetTol':<8} {'MaxDiff':<12}")
    print(f"{'-'*25} {'-'*8} {'-'*10} {'-'*8} {'-'*12}")
    for d,p,de,dc,md in res:
        print(f"{d:<25} {'PASS' if p else 'FAIL':<8} {'PASS' if de else 'FAIL':<10} {'PASS' if dc else 'FAIL':<8} {md:<12.6f}")
    total=len(res)
    print(f"\nPrecision: {sum(1 for _,p,_,_,_ in res if p)}/{total}  DetExact: {sum(1 for _,_,de,_,_ in res if de)}/{total}  DetTol1e-3: {sum(1 for _,_,_,dc,_ in res if dc)}/{total}")
