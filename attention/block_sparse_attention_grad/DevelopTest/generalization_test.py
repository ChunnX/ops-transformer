#!/usr/bin/env python3
"""
Generalization stress test: 60+ shapes covering edge cases, extreme params,
and combinatorial coverage for block_sparse_attention_grad determinism.
"""

import torch
import torch_npu
import math
import numpy as np
from einops import repeat
import random
import traceback

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
        dS_fp16=dS.to(DTYPE).float(); P_fp16=P.to(DTYPE).float()
        dq_unsat[:,i:i+1]=tbd(dS_fp16,ki)
        dk_unsat[:,g:g+1]+=tbd(torch.transpose(dS_fp16,2,3),qi)
        dv_unsat[:,g:g+1]+=tbd(torch.transpose(P_fp16,2,3),ag)
    dq=(dq_unsat*scale).to(DTYPE); dk=(dk_unsat*scale).to(DTYPE); dv=dv_unsat.to(DTYPE)
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

def run_test(desc,B,N,S1,S2,D,kvh,sp,fmt,niter=3):
    FMT=fmt.upper(); scale=1.0/math.sqrt(D)
    tag=f"B={B} N={N} KV={kvh} g={N//kvh} S1={S1} S2={S2} sp={sp} {FMT}"
    try:
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
    except Exception as e:
        return desc,tag,"CPU_ERR",str(e),False,False,-1

    torch.npu.synchronize();torch.npu.empty_cache()
    results=[];outputs=[]
    for j in range(niter):
        try:
            dq,dk,dv=torch_npu.npu_block_sparse_attention_backward(
                do.to(device),q.to(device),k.to(device),v.to(device),
                attn.to(device),lse.to(device),mask.to(device),
                block_shape=(128,128),actual_seq_lengths=sql,actual_seq_lengths_kv=svl,
                q_input_layout=FMT,kv_input_layout=FMT,num_key_value_heads=kvh,scale_value=scale)
            torch.npu.synchronize()
            results.append((compera(cpu_dq,dq),compera(cpu_dk,dk),compera(cpu_dv,dv)))
            outputs.append((dq.clone(),dk.clone(),dv.clone()))
        except Exception as e:
            results.append(((False,-1,-1),(False,-1,-1),(False,-1,-1)));outputs.append(None)
        torch.npu.empty_cache()

    all_prec=all(r[0][0]and r[1][0]and r[2][0] for r in results)
    max_det_diff=0.0;det_exact=True
    if outputs[0]:
        b0=outputs[0]
        for j in range(1,len(outputs)):
            if outputs[j] is None: det_exact=False;continue
            for i in range(3):
                exact,diff=deter_check(b0[i],outputs[j][i])
                if not exact: det_exact=False
                max_det_diff=max(max_det_diff,diff)
    else: det_exact=False

    # Collect worst precision diff
    max_prec_diff=0.0
    for r1,r2,r3 in results:
        for r in [r1,r2,r3]:
            if not r[0]: max_prec_diff=max(max_prec_diff,r[1])

    status="OK" if (all_prec and det_exact) else ("PREC_FAIL" if not all_prec else "DET_FAIL")
    return desc,tag,status,"",all_prec,det_exact,max_det_diff

if __name__=="__main__":
    cases=[
        # ====== Extreme group sizes ======
        ("GQA g16 BNSD",        1,16, 128,128,128,1,  0.5,"bnsd"),
        ("GQA g16 TND",         1,16, 128,128,128,1,  0.5,"tnd"),
        ("GQA g12 BNSD",        1,12, 128,128,128,1,  0.5,"bnsd"),
        ("GQA g12 TND",         2,12, 256,256,128,1,  0.5,"tnd"),
        ("GQA g8 BNSD large",   2,16, 256,256,128,2,  0.5,"bnsd"),
        ("GQA g8 TND",          2,16, 256,256,128,2,  0.5,"tnd"),
        ("GQA g4 BNSD xl",      4,16, 512,512,128,4,  0.5,"bnsd"),
        ("GQA g4 TND xl",       4,16, 512,512,128,4,  0.5,"tnd"),
        ("GQA g2 BNSD",         2,8,  256,256,128,4,  0.5,"bnsd"),
        ("GQA g2 TND",          2,8,  256,256,128,4,  0.5,"tnd"),

        # ====== Single batch edge cases ======
        ("B1 GQA g4 BNSD",      1,8,  128,128,128,2,  0.5,"bnsd"),
        ("B1 GQA g4 TND",       1,8,  256,256,128,2,  0.5,"tnd"),
        ("B1 MHA BNSD",         1,8,  128,128,128,8,  0.5,"bnsd"),
        ("B1 MHA TND",          1,8,  256,256,128,8,  0.5,"tnd"),

        # ====== Large batch ======
        ("B16 GQA g4 BNSD",     16,8, 128,128,128,2,  0.5,"bnsd"),
        ("B16 GQA g4 TND",      16,8, 128,128,128,2,  0.5,"tnd"),
        ("B12 MHA BNSD",        12,4, 128,128,128,4,  0.5,"bnsd"),
        ("B12 MHA TND",         12,4, 128,128,128,4,  0.5,"tnd"),

        # ====== Large sequence lengths ======
        ("GQA g4 S1_768 BNSD",  2,8,  768,768,128,2,  0.5,"bnsd"),
        ("GQA g4 S1_768 TND",   2,8,  768,768,128,2,  0.5,"tnd"),
        ("GQA g3 S1_1024 BNSD", 2,12, 1024,1024,128,4,0.3,"bnsd"),

        # ====== Asymmetric S1 vs S2 ======
        ("GQA g4 S1>S2 BNSD",   2,8,  512,128,128,2,  0.5,"bnsd"),
        ("GQA g4 S1<S2 BNSD",   2,8,  128,512,128,2,  0.5,"bnsd"),
        ("GQA g4 S1>S2 TND",    2,8,  512,128,128,2,  0.5,"tnd"),
        ("GQA g4 S1<S2 TND",    2,8,  128,512,128,2,  0.5,"tnd"),
        ("MHA S1>S2 BNSD",      2,8,  384,128,128,8,  0.5,"bnsd"),
        ("MHA S1<S2 TND",       2,8,  128,384,128,8,  0.5,"tnd"),

        # ====== Sparsity extremes ======
        ("GQA g4 sp0.1 BNSD",   2,8,  256,256,128,2,  0.1,"bnsd"),
        ("GQA g4 sp0.1 TND",    2,8,  256,256,128,2,  0.1,"tnd"),
        ("GQA g4 sp0.9 BNSD",   2,8,  256,256,128,2,  0.9,"bnsd"),
        ("GQA g4 sp0.9 TND",    2,8,  256,256,128,2,  0.9,"tnd"),
        ("GQA g4 sp1.0 BNSD",   2,8,  256,256,128,2,  1.0,"bnsd"),
        ("GQA g4 sp1.0 TND",    2,8,  256,256,128,2,  1.0,"tnd"),
        ("MHA sp0.1 BNSD",      2,8,  256,256,128,8,  0.1,"bnsd"),
        ("MHA sp1.0 TND",       2,8,  256,256,128,8,  1.0,"tnd"),

        # ====== Small sequences (edge) ======
        ("GQA g4 S32 BNSD",     2,8,  32, 32, 128,2,  0.5,"bnsd"),
        ("GQA g4 S32 TND",      2,8,  32, 32, 128,2,  0.5,"tnd"),
        ("MHA S16 BNSD",        2,4,  16, 16, 128,4,  1.0,"bnsd"),
        ("MHA S16 TND",         2,4,  16, 16, 128,4,  1.0,"tnd"),

        # ====== Sequences near block boundary ======
        ("GQA g4 S127 BNSD",    2,8,  127,127,128,2,  0.5,"bnsd"),
        ("GQA g4 S129 BNSD",    2,8,  129,129,128,2,  0.5,"bnsd"),
        ("GQA g4 S256 BNSD",    2,8,  256,256,128,2,  0.5,"bnsd"),
        ("GQA g4 S257 BNSD",    2,8,  257,257,128,2,  0.5,"bnsd"),
        ("GQA g4 S128 TND",     2,8,  128,128,128,2,  0.5,"tnd"),

        # ====== Various head counts ======
        ("GQA N32 KV4 BNSD",    1,32, 128,128,128,4,  0.5,"bnsd"),
        ("GQA N32 KV8 BNSD",    1,32, 128,128,128,8,  0.5,"bnsd"),
        ("GQA N32 KV2 TND",     1,32, 128,128,128,2,  0.5,"tnd"),
        ("MHA N2 BNSD",         2,2,  128,128,128,2,  0.5,"bnsd"),
        ("MHA N2 TND",          2,2,  128,128,128,2,  0.5,"tnd"),

        # ====== High batch + moderate seq ======
        ("B8 GQA g4 S128 BNSD", 8,8,  128,128,128,2,  0.5,"bnsd"),
        ("B8 GQA g4 S128 TND",  8,8,  128,128,128,2,  0.5,"tnd"),
        ("B6 MHA S256 BNSD",    6,8,  256,256,128,8,  0.5,"bnsd"),

        # ====== TND with extreme variable lengths ======
        ("GQA g4 TND var",      5,8,  512,512,128,2,  0.5,"tnd"),
        ("MHA TND var",         5,8,  512,512,128,8,  0.5,"tnd"),
        ("GQA g8 TND var",      3,16, 256,256,128,2,  0.5,"tnd"),

        # ====== Combined stress ======
        ("B4 GQA g6 S384 BNSD", 4,12, 384,384,128,2,  0.3,"bnsd"),
        ("B4 GQA g6 S384 TND",  4,12, 384,384,128,2,  0.3,"tnd"),
        ("B2 MHA N16 S512 BNSD",2,16, 512,512,128,16, 0.5,"bnsd"),
        ("B2 MHA N16 S512 TND", 2,16, 512,512,128,16, 0.5,"tnd"),
    ]

    print(f"Total test cases: {len(cases)}")
    print(f"{'='*100}")

    results=[]
    for desc,B,N,S1,S2,D,kvh,sp,fmt in cases:
        ret=run_test(desc,B,N,S1,S2,D,kvh,sp,fmt,niter=3)
        results.append(ret)
        _,tag,status,err,prec,det,md=ret
        if status=="OK":
            print(f"[OK]  {desc:<25} {tag}")
        elif status=="CPU_ERR":
            print(f"[ERR] {desc:<25} {tag}  CPU: {err[:60]}")
        elif status=="PREC_FAIL":
            print(f"[P!]  {desc:<25} {tag}  prec={'PASS' if prec else 'FAIL'} det={'PASS' if det else 'FAIL'}")
        else:
            print(f"[D!]  {desc:<25} {tag}  prec={'PASS' if prec else 'FAIL'} det={'PASS' if det else 'FAIL'}")

    # Summary
    ok=sum(1 for r in results if r[2]=="OK")
    cpu_err=sum(1 for r in results if r[2]=="CPU_ERR")
    prec_fail=sum(1 for r in results if r[2]=="PREC_FAIL")
    det_fail=sum(1 for r in results if r[2]=="DET_FAIL")
    npu_err=sum(1 for r in results if r[2] not in ("OK","CPU_ERR","PREC_FAIL","DET_FAIL"))

    print(f"\n{'='*100}")
    print(f"SUMMARY: total={len(results)}  OK={ok}  PREC_FAIL={prec_fail}  DET_FAIL={det_fail}  CPU_ERR={cpu_err}  NPU_ERR={npu_err}")
    print(f"Precision: {sum(1 for r in results if r[4])}/{len(results)}")
    print(f"Determinism: {sum(1 for r in results if r[5])}/{len(results)}")

    if prec_fail>0 or det_fail>0:
        print("\n--- Failed cases ---")
        for desc,tag,status,err,prec,det,md in results:
            if status!="OK" and status!="CPU_ERR":
                print(f"  {desc:<25} {tag}  status={status} prec={prec} det={det} max_diff={md}")
