#!/usr/bin/env python3
"""
Generalization validation:
  1. det=True vs det=False: precision consistency (both should match CPU golden)
  2. det=True: bit-exact determinism across iterations
  3. det=False: precision passes (determinism not required)
59 shapes covering edge cases.
"""

import torch
import torch_npu
import math
import numpy as np
from einops import repeat
import random
import os

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
    dq_u=torch.zeros(B,H,q.shape[2],D,dtype=torch.float32)
    dk_u=torch.zeros(B,k.shape[1],k.shape[2],D,dtype=torch.float32)
    dv_u=torch.zeros(B,k.shape[1],k.shape[2],D,dtype=torch.float32)
    for i in range(H):
        g=i//nG; a=out[:,i:i+1].float();ag=dout[:,i:i+1].float();li=lse[:,i:i+1,:,:].float()
        qi=q[:,i:i+1].float();ki=k[:,g:g+1].float();vi=v[:,g:g+1].float()
        b2,h2,ql,_=qi.shape;li=li.view(b2,h2,ql,1)
        Dv=(a*ag).sum(-1,keepdim=True).reshape(b2,h2,ql,1)
        S=qk_masked(qi,ki,mask,bs,bi,i)*scale; dP=tbd(ag,torch.transpose(vi,2,3)).float()
        P=torch.exp(S-li); dS=(dP-Dv)*P
        dS_f=dS.to(DTYPE).float(); P_f=P.to(DTYPE).float()
        dq_u[:,i:i+1]=tbd(dS_f,ki); dk_u[:,g:g+1]+=tbd(torch.transpose(dS_f,2,3),qi)
        dv_u[:,g:g+1]+=tbd(torch.transpose(P_f,2,3),ag)
    return (dq_u*scale).to(DTYPE),(dk_u*scale).to(DTYPE),dv_u.to(DTYPE)

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

def gen_data(B,N,S1,S2,D,kvh,sp,fmt):
    scale=1.0/math.sqrt(D)
    if fmt=="tnd":
        random.seed(seed);torch.manual_seed(seed)
        sql=[random.randint(1,S1) for _ in range(B)];svl=[random.randint(1,S2) for _ in range(B)]
        aqm=max(sql);akm=max(svl)
        cu_q=[0];[cu_q.append(cu_q[-1]+s) for s in sql]
        cu_kv=[0];[cu_kv.append(cu_kv[-1]+s) for s in svl]
        q=torch.randn(cu_q[-1],N,D,dtype=DTYPE);k=torch.randn(cu_kv[-1],kvh,D,dtype=DTYPE)
        v=torch.randn(cu_kv[-1],kvh,D,dtype=DTYPE);do=torch.randn(cu_q[-1],N,D,dtype=DTYPE)
        mask=gen_mask(aqm,akm,128,128,B,N,sp)
        lse,attn=attention_gqa_tnd(q,k,v,scale,cu_q,cu_kv)
        cpu_dq,cpu_dk,cpu_dv=cpu_grad_tnd(do,q,k,v,attn,lse,mask,(128,128),cu_q,cu_kv,scale,S1,S2)
    else:
        sql=[S1]*B;svl=[S2]*B
        q=torch.randn(B,N,S1,D,dtype=DTYPE);k=torch.randn(B,kvh,S2,D,dtype=DTYPE)
        v=torch.randn(B,kvh,S2,D,dtype=DTYPE);do=torch.randn(B,N,S1,D,dtype=DTYPE)
        mask=gen_mask(S1,S2,128,128,B,N,sp)
        lse,attn=attention_ori_GQA(q,k,v,scale)
        cpu_dq,cpu_dk,cpu_dv=cpu_grad_bnsd(do,q,k,v,attn,lse,mask,(128,128),scale)
    return q,k,v,do,attn,lse,mask,sql,svl,kvh,scale,cpu_dq,cpu_dk,cpu_dv,FMT.upper() if (FMT:=fmt) else ""

def npu_run(q,k,v,do,attn,lse,mask,sql,svl,kvh,scale,FMT):
    dq,dk,dv=torch_npu.npu_block_sparse_attention_backward(
        do.to(device),q.to(device),k.to(device),v.to(device),
        attn.to(device),lse.to(device),mask.to(device),
        block_shape=(128,128),actual_seq_lengths=sql,actual_seq_lengths_kv=svl,
        q_input_layout=FMT,kv_input_layout=FMT,num_key_value_heads=kvh,scale_value=scale)
    torch.npu.synchronize()
    return dq,dk,dv

def run_test(desc,B,N,S1,S2,D,kvh,sp,fmt):
    tag=f"B={B} N={N} KV={kvh} g={N//kvh} S1={S1} S2={S2} sp={sp} {fmt.upper()}"
    try:
        data=gen_data(B,N,S1,S2,D,kvh,sp,fmt)
        q,k,v,do,attn,lse,mask,sql,svl,kvh2,scale,cpu_dq,cpu_dk,cpu_dv,FMT=data
    except Exception as e:
        print(f"[ERR] {desc:<28} {tag}  CPU: {str(e)[:50]}")
        return desc,tag,False,False,False

    torch.npu.empty_cache()
    # det=True: 3 iterations (check precision + bit-exact determinism)
    os.environ["BSAG_DETERMINISTIC"]="1"
    det_results=[];det_outputs=[]
    for j in range(3):
        try:
            dq,dk,dv=npu_run(q,k,v,do,attn,lse,mask,sql,svl,kvh2,scale,FMT)
            det_results.append((compera(cpu_dq,dq),compera(cpu_dk,dk),compera(cpu_dv,dv)))
            det_outputs.append((dq.clone(),dk.clone(),dv.clone()))
        except:
            det_results.append(((False,-1,-1),(False,-1,-1),(False,-1,-1)));det_outputs.append(None)
        torch.npu.empty_cache()

    # det=False: 3 iterations (check precision only)
    os.environ["BSAG_DETERMINISTIC"]="0"
    nondet_results=[]
    for j in range(3):
        try:
            dq,dk,dv=npu_run(q,k,v,do,attn,lse,mask,sql,svl,kvh2,scale,FMT)
            nondet_results.append((compera(cpu_dq,dq),compera(cpu_dk,dk),compera(cpu_dv,dv)))
        except:
            nondet_results.append(((False,-1,-1),(False,-1,-1),(False,-1,-1)))
        torch.npu.empty_cache()

    det_prec=all(r[0][0]and r[1][0]and r[2][0] for r in det_results)
    nondet_prec=all(r[0][0]and r[1][0]and r[2][0] for r in nondet_results)

    # det=True bit-exact check
    det_exact=True;max_diff=0.0
    if det_outputs[0]:
        b0=det_outputs[0]
        for j in range(1,3):
            if det_outputs[j] is None:det_exact=False;continue
            for i in range(3):
                ex,d=deter_check(b0[i],det_outputs[j][i])
                if not ex:det_exact=False
                max_diff=max(max_diff,d)
    else:det_exact=False

    ok = det_prec and nondet_prec and det_exact
    status = "OK" if ok else ("FAIL" if not (det_prec and nondet_prec) else "DET_FAIL")
    print(f"[{status:4}] {desc:<28} {tag}  det_prec={'P' if det_prec else 'F'} nondet_prec={'P' if nondet_prec else 'F'} det_exact={'Y' if det_exact else 'N'}")
    return desc,tag,det_prec,nondet_prec,det_exact

if __name__=="__main__":
    cases=[
        ("GQA g16 BNSD",        1,16, 128,128,128,1,  0.5,"bnsd"),
        ("GQA g16 TND",         1,16, 128,128,128,1,  0.5,"tnd"),
        ("GQA g8 BNSD large",   2,16, 256,256,128,2,  0.5,"bnsd"),
        ("GQA g8 TND",          2,16, 256,256,128,2,  0.5,"tnd"),
        ("GQA g4 BNSD xl",      4,16, 512,512,128,4,  0.5,"bnsd"),
        ("GQA g4 TND xl",       4,16, 512,512,128,4,  0.5,"tnd"),
        ("GQA g2 BNSD",         2,8,  256,256,128,4,  0.5,"bnsd"),
        ("GQA g2 TND",          2,8,  256,256,128,4,  0.5,"tnd"),
        ("B1 GQA g4 BNSD",      1,8,  128,128,128,2,  0.5,"bnsd"),
        ("B1 GQA g4 TND",       1,8,  256,256,128,2,  0.5,"tnd"),
        ("B1 MHA BNSD",         1,8,  128,128,128,8,  0.5,"bnsd"),
        ("B1 MHA TND",          1,8,  256,256,128,8,  0.5,"tnd"),
        ("B16 GQA g4 BNSD",     16,8, 128,128,128,2,  0.5,"bnsd"),
        ("B16 GQA g4 TND",      16,8, 128,128,128,2,  0.5,"tnd"),
        ("B12 MHA BNSD",        12,4, 128,128,128,4,  0.5,"bnsd"),
        ("B12 MHA TND",         12,4, 128,128,128,4,  0.5,"tnd"),
        ("GQA g4 S768 BNSD",    2,8,  768,768,128,2,  0.5,"bnsd"),
        ("GQA g4 S768 TND",     2,8,  768,768,128,2,  0.5,"tnd"),
        ("GQA g3 S1024 BNSD",   2,12, 1024,1024,128,4,0.3,"bnsd"),
        ("GQA g4 S1>S2 BNSD",   2,8,  512,128,128,2,  0.5,"bnsd"),
        ("GQA g4 S1<S2 BNSD",   2,8,  128,512,128,2,  0.5,"bnsd"),
        ("GQA g4 S1>S2 TND",    2,8,  512,128,128,2,  0.5,"tnd"),
        ("GQA g4 S1<S2 TND",    2,8,  128,512,128,2,  0.5,"tnd"),
        ("MHA S1>S2 BNSD",      2,8,  384,128,128,8,  0.5,"bnsd"),
        ("MHA S1<S2 TND",       2,8,  128,384,128,8,  0.5,"tnd"),
        ("GQA g4 sp0.1 BNSD",   2,8,  256,256,128,2,  0.1,"bnsd"),
        ("GQA g4 sp0.1 TND",    2,8,  256,256,128,2,  0.1,"tnd"),
        ("GQA g4 sp0.9 BNSD",   2,8,  256,256,128,2,  0.9,"bnsd"),
        ("GQA g4 sp0.9 TND",    2,8,  256,256,128,2,  0.9,"tnd"),
        ("GQA g4 sp1.0 BNSD",   2,8,  256,256,128,2,  1.0,"bnsd"),
        ("GQA g4 sp1.0 TND",    2,8,  256,256,128,2,  1.0,"tnd"),
        ("MHA sp0.1 BNSD",      2,8,  256,256,128,8,  0.1,"bnsd"),
        ("MHA sp1.0 TND",       2,8,  256,256,128,8,  1.0,"tnd"),
        ("GQA g4 S32 BNSD",     2,8,  32, 32, 128,2,  0.5,"bnsd"),
        ("GQA g4 S32 TND",      2,8,  32, 32, 128,2,  0.5,"tnd"),
        ("MHA S16 BNSD",        2,4,  16, 16, 128,4,  1.0,"bnsd"),
        ("MHA S16 TND",         2,4,  16, 16, 128,4,  1.0,"tnd"),
        ("GQA g4 S127 BNSD",    2,8,  127,127,128,2,  0.5,"bnsd"),
        ("GQA g4 S129 BNSD",    2,8,  129,129,128,2,  0.5,"bnsd"),
        ("GQA g4 S257 BNSD",    2,8,  257,257,128,2,  0.5,"bnsd"),
        ("GQA N32 KV4 BNSD",    1,32, 128,128,128,4,  0.5,"bnsd"),
        ("GQA N32 KV2 TND",     1,32, 128,128,128,2,  0.5,"tnd"),
        ("MHA N2 BNSD",         2,2,  128,128,128,2,  0.5,"bnsd"),
        ("MHA N2 TND",          2,2,  128,128,128,2,  0.5,"tnd"),
        ("B8 GQA g4 S128 BNSD", 8,8,  128,128,128,2,  0.5,"bnsd"),
        ("B8 GQA g4 S128 TND",  8,8,  128,128,128,2,  0.5,"tnd"),
        ("B6 MHA S256 BNSD",    6,8,  256,256,128,8,  0.5,"bnsd"),
        ("GQA g4 TND var",      5,8,  512,512,128,2,  0.5,"tnd"),
        ("MHA TND var",         5,8,  512,512,128,8,  0.5,"tnd"),
        ("GQA g8 TND var",      3,16, 256,256,128,2,  0.5,"tnd"),
        ("B4 GQA g6 S384 BNSD", 4,12, 384,384,128,2,  0.3,"bnsd"),
        ("B4 GQA g6 S384 TND",  4,12, 384,384,128,2,  0.3,"tnd"),
        ("B2 MHA N16 S512 BNSD",2,16, 512,512,128,16, 0.5,"bnsd"),
        ("B2 MHA N16 S512 TND", 2,16, 512,512,128,16, 0.5,"tnd"),
    ]

    print(f"Total: {len(cases)} shapes x (det=True: prec+exact, det=False: prec) = {len(cases)} tests\n{'='*110}")
    results=[]
    for desc,B,N,S1,S2,D,kvh,sp,fmt in cases:
        r=run_test(desc,B,N,S1,S2,D,kvh,sp,fmt)
        results.append(r)

    det_prec_ok=sum(1 for r in results if r[2])
    nondet_prec_ok=sum(1 for r in results if r[3])
    det_exact_ok=sum(1 for r in results if r[4])
    total=len(results)
    print(f"\n{'='*110}")
    print(f"SUMMARY: {total} shapes")
    print(f"  det=True  precision:  {det_prec_ok}/{total}")
    print(f"  det=False precision:  {nondet_prec_ok}/{total}")
    print(f"  det=True  bit-exact:  {det_exact_ok}/{total}")

    fails=[r for r in results if not (r[2] and r[3] and r[4])]
    if fails:
        print(f"\n--- Failed cases ({len(fails)}) ---")
        for desc,tag,dp,ndp,de in fails:
            print(f"  {desc:<28} {tag}  det_prec={'P' if dp else 'F'} nondet_prec={'P' if ndp else 'F'} det_exact={'Y' if de else 'N'}")
    else:
        print("\n  ALL PASS!")
