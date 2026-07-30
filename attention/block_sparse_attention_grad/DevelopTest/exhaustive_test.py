#!/usr/bin/env python3
"""
Exhaustive test - optimized: flush output, 2 iters, skip shapes with S>768.
"""
import torch, torch_npu, math, numpy as np, random, os, sys
from einops import repeat

sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)  # line-buffered

torch.set_printoptions(precision=6)
seed=128; torch.manual_seed(seed); device="npu:7"; DTYPE=torch.float16
torch.npu.empty_cache()

def compera(g,a):
    g=g.cpu().flatten().to(torch.float32);a=a.cpu().flatten().to(torch.float32)
    r=torch.allclose(g,a,rtol=1e-03,atol=1e-03)
    if not r:
        d=torch.abs(a-g);return r,d.max().item(),torch.sum(~torch.isclose(g,a,rtol=1e-3,atol=1e-3)).item()
    return r,0.0,0

def deter_chk(a,b):
    a=a.cpu().flatten().to(torch.float32);b=b.cpu().flatten().to(torch.float32)
    if torch.equal(a,b):return True,0.0
    d=torch.abs(a-b);return torch.allclose(a,b,rtol=0,atol=0),d.max().item()

def attn_bnsd(qa,ka,va,sc):
    nG=qa.shape[1]//ka.shape[1]
    for i in range(qa.shape[1]):
        g=i//nG;q=qa[:,i:i+1].float();k=ka[:,g:g+1].float();v=va[:,g:g+1].float()
        bm=torch.matmul(q,torch.transpose(k,2,3)).float()*sc
        mx,_=torch.max(bm,dim=-1,keepdim=True);ex=torch.exp(bm-mx)
        sm=torch.from_numpy(np.sum(ex.numpy(),-1,keepdims=True));so=ex/sm
        at=torch.matmul(so,v);l=mx+torch.log(sm)
        if i==0:tl=l;ta=at
        else:tl=torch.cat((tl,l),1);ta=torch.cat((ta,at),1)
    return tl,ta.to(DTYPE)

def attn_tnd(q,k,v,sc,cq,ck):
    q=q.to(torch.float32);k=k.to(torch.float32);v=v.to(torch.float32);dv=q.device
    tq,nh,hd=q.shape;tk,nhk,_=k.shape;bs=len(cq)-1;nG=nh//nhk
    o=torch.zeros_like(q);lse=torch.zeros((tq,nh,1),dtype=torch.float32,device=dv)
    for i in range(nh):
        g=i//nG
        for b in range(bs):
            qs,qe=cq[b],cq[b+1];ks,ke=ck[b],ck[b+1]
            qb=q[qs:qe,i:i+1,:].permute(1,0,2);kb=k[ks:ke,g:g+1,:].permute(1,0,2);vb=v[ks:ke,g:g+1,:].permute(1,0,2)
            bm=torch.matmul(qb,torch.transpose(kb,1,2)).float()*sc
            mx,_=torch.max(bm,dim=-1,keepdim=True);ex=torch.exp(bm-mx)
            sm=torch.from_numpy(np.sum(ex.cpu().numpy(),-1,keepdims=True)).to(dv);so=ex/sm
            o[qs:qe,i:i+1,:]=torch.matmul(so,vb).permute(1,0,2);lse[qs:qe,i:i+1,:]=(mx+torch.log(sm)).permute(1,0,2)
    return lse.contiguous(),o.to(DTYPE).contiguous()

def gmask(sq,sk,bx,by,B,N,sp):
    nr=(sq+bx-1)//bx;nc=(sk+by-1)//by;m=torch.zeros(B,N,nr,nc,dtype=torch.uint8)
    torch.manual_seed(seed)
    for b in range(B):
        for h in range(N):
            if sp==1.0:m[b][h]=1
            elif sp!=0.0:
                for i in range(nr):m[b][h][i,torch.randperm(nc)[:max(1,int(sp*nc))]]=1
    return m

def tmask(m,sq,sk):
    b,n=m.shape[0],m.shape[1];m=m[:,:,:sq,:sk]
    p=torch.zeros(b,n,sq,sk,dtype=torch.bool);p[:,:,:m.shape[2],:m.shape[3]]=m;return p.contiguous()

def qkm(q,k,mask,bs,bi=-1,hi=-1):
    kt=torch.transpose(k,2,3).contiguous();S=torch.einsum("bnqd,bndk->bnqk",q,kt)
    fm=repeat(mask,"b n s_m s_n -> b n (s_m d_m) (s_n d_n)",d_m=bs[0],d_n=bs[1])
    fm=tmask(fm,q.shape[2],k.shape[2]);nm=~fm
    if bi!=-1:nm=nm[bi:bi+1]
    if hi!=-1:nm=nm[:,hi:hi+1]
    return S.masked_fill(nm,float('-inf'))

def tbd(a,b):
    bs,n,h,_=a.shape;_,_,_,w=b.shape;a=a.float();b=b.float()
    r=torch.zeros(bs,n,h,w,dtype=torch.float)
    for i in range(bs):
        for j in range(n):r[i,j]=torch.matmul(a[i,j],b[i,j])
    return r

def cgrad_bnsd(dout,q,k,v,out,lse,mask,bs,sc,bi=-1):
    nG=q.shape[1]//k.shape[1];B,H,_,D=q.shape
    dqu=torch.zeros(B,H,q.shape[2],D,dtype=torch.float32)
    dku=torch.zeros(B,k.shape[1],k.shape[2],D,dtype=torch.float32)
    dvu=torch.zeros(B,k.shape[1],k.shape[2],D,dtype=torch.float32)
    for i in range(H):
        g=i//nG;a=out[:,i:i+1].float();ag=dout[:,i:i+1].float();li=lse[:,i:i+1,:,:].float()
        qi=q[:,i:i+1].float();ki=k[:,g:g+1].float();vi=v[:,g:g+1].float()
        b2,h2,ql,_=qi.shape;li=li.view(b2,h2,ql,1)
        Dv=(a*ag).sum(-1,keepdim=True).reshape(b2,h2,ql,1)
        S=qkm(qi,ki,mask,bs,bi,i)*sc;dP=tbd(ag,torch.transpose(vi,2,3)).float()
        P=torch.exp(S-li);dS=(dP-Dv)*P
        df=dS.to(DTYPE).float();Pf=P.to(DTYPE).float()
        dqu[:,i:i+1]=tbd(df,ki);dku[:,g:g+1]+=tbd(torch.transpose(df,2,3),qi)
        dvu[:,g:g+1]+=tbd(torch.transpose(Pf,2,3),ag)
    return (dqu*sc).to(DTYPE),(dku*sc).to(DTYPE),dvu.to(DTYPE)

def cgrad_tnd(dout,q,k,v,out,lse,mask,bs,cq,ck,sc,sqm,skm):
    tq,H,D=q.shape;tkv,KH,_=k.shape;B=len(cq)-1
    q=q.float();k=k.float();v=v.float();dout=dout.float();out=out.float()
    lse=lse.float().reshape(tq,H,1).contiguous()
    dq=torch.zeros_like(q);dk=torch.zeros_like(k);dv=torch.zeros_like(v)
    def t2(t,a,b,n,h,d):return t[a:b].reshape(1,b-a,h,d).permute(0,2,1,3).contiguous()
    for i in range(B):
        qn=cq[i+1]-cq[i];kvn=ck[i+1]-ck[i]
        qt=t2(q,cq[i],cq[i+1],qn,H,D);dt=t2(dout,cq[i],cq[i+1],qn,H,D)
        ot=t2(out,cq[i],cq[i+1],qn,H,D);kt=t2(k,ck[i],ck[i+1],kvn,KH,D)
        vt=t2(v,ck[i],ck[i+1],kvn,KH,D);lt=t2(lse,cq[i],cq[i+1],qn,H,1)
        dqt,dkt,dvt=cgrad_bnsd(dt,qt,kt,vt,ot,lt,mask,bs,sc,i)
        dq[cq[i]:cq[i+1]]=dqt.permute(0,2,1,3)[0:1]
        dk[ck[i]:ck[i+1]]=dkt.permute(0,2,1,3)[0:1]
        dv[ck[i]:ck[i+1]]=dvt.permute(0,2,1,3)[0:1]
    return dq.to(DTYPE),dk.to(DTYPE),dv.to(DTYPE)

def gen(B,N,S1,S2,D,kvh,sp,fmt):
    sc=1.0/math.sqrt(D)
    if fmt=="tnd":
        random.seed(seed);torch.manual_seed(seed)
        sql=[random.randint(1,S1) for _ in range(B)];svl=[random.randint(1,S2) for _ in range(B)]
        aqm=max(sql);akm=max(svl)
        cq=[0];[cq.append(cq[-1]+s) for s in sql]
        ck=[0];[ck.append(ck[-1]+s) for s in svl]
        q=torch.randn(cq[-1],N,D,dtype=DTYPE);k=torch.randn(ck[-1],kvh,D,dtype=DTYPE)
        v=torch.randn(ck[-1],kvh,D,dtype=DTYPE);do=torch.randn(cq[-1],N,D,dtype=DTYPE)
        mask=gmask(aqm,akm,128,128,B,N,sp)
        lse,attn=attn_tnd(q,k,v,sc,cq,ck)
        cdq,cdk,cdv=cgrad_tnd(do,q,k,v,attn,lse,mask,(128,128),cq,ck,sc,S1,S2)
    else:
        sql=[S1]*B;svl=[S2]*B
        q=torch.randn(B,N,S1,D,dtype=DTYPE);k=torch.randn(B,kvh,S2,D,dtype=DTYPE)
        v=torch.randn(B,kvh,S2,D,dtype=DTYPE);do=torch.randn(B,N,S1,D,dtype=DTYPE)
        mask=gmask(S1,S2,128,128,B,N,sp)
        lse,attn=attn_bnsd(q,k,v,sc)
        cdq,cdk,cdv=cgrad_bnsd(do,q,k,v,attn,lse,mask,(128,128),sc)
    return q,k,v,do,attn,lse,mask,sql,svl,kvh,sc,cdq,cdk,cdv,fmt.upper()

def npu_run(q,k,v,do,attn,lse,mask,sql,svl,kvh,sc,FMT):
    dq,dk,dv=torch_npu.npu_block_sparse_attention_backward(
        do.to(device),q.to(device),k.to(device),v.to(device),attn.to(device),lse.to(device),mask.to(device),
        block_shape=(128,128),actual_seq_lengths=sql,actual_seq_lengths_kv=svl,
        q_input_layout=FMT,kv_input_layout=FMT,num_key_value_heads=kvh,scale_value=sc)
    torch.npu.synchronize();return dq,dk,dv

def test(desc,B,N,S1,S2,D,kvh,sp,fmt):
    tag=f"B={B} N={N} KV={kvh} g={N//kvh} S1={S1} S2={S2} sp={sp} {fmt.upper()}"
    try:
        d=gen(B,N,S1,S2,D,kvh,sp,fmt)
    except Exception as e:
        print(f"[ERR] {desc:<28} {tag}  CPU: {str(e)[:50]}");return desc,tag,False,False,False
    q,k,v,do,attn,lse,mask,sql,svl,kvh2,sc,cdq,cdk,cdv,FMT=d
    torch.npu.empty_cache()
    os.environ["BSAG_DETERMINISTIC"]="1"
    dr=[];dol=[]
    for _ in range(2):
        try:
            dq,dk,dv=npu_run(q,k,v,do,attn,lse,mask,sql,svl,kvh2,sc,FMT)
            dr.append((compera(cdq,dq),compera(cdk,dk),compera(cdv,dv)))
            dol.append((dq.clone(),dk.clone(),dv.clone()))
        except:dr.append(((False,-1,-1),(False,-1,-1),(False,-1,-1)));dol.append(None)
        torch.npu.empty_cache()
    os.environ["BSAG_DETERMINISTIC"]="0"
    ndr=[]
    for _ in range(2):
        try:
            dq,dk,dv=npu_run(q,k,v,do,attn,lse,mask,sql,svl,kvh2,sc,FMT)
            ndr.append((compera(cdq,dq),compera(cdk,dk),compera(cdv,dv)))
        except:ndr.append(((False,-1,-1),(False,-1,-1),(False,-1,-1)))
        torch.npu.empty_cache()
    dp=all(r[0][0]and r[1][0]and r[2][0] for r in dr)
    ndp=all(r[0][0]and r[1][0]and r[2][0] for r in ndr)
    de=True
    if dol[0] and dol[1]:
        for i in range(3):
            ex,_=deter_chk(dol[0][i],dol[1][i])
            if not ex:de=False
    else:de=False
    s="OK" if(dp and ndp and de)else"FAIL"
    print(f"[{s:4}] {desc:<28} {tag}")
    return desc,tag,dp,ndp,de

if __name__=="__main__":
    C=[
        # Extreme group sizes
        ("GQA g16 BNSD s128",     1,16,128,128,128,1,0.5,"bnsd"),
        ("GQA g16 TND s256",      1,16,256,256,128,1,0.5,"tnd"),
        ("GQA g12 BNSD s128",     1,12,128,128,128,1,0.5,"bnsd"),
        ("GQA g8 BNSD s256",      2,16,256,256,128,2,0.5,"bnsd"),
        ("GQA g8 TND s256",       2,16,256,256,128,2,0.5,"tnd"),
        ("GQA g6 BNSD s384",      4,12,384,384,128,2,0.3,"bnsd"),
        ("GQA g6 TND s384",       4,12,384,384,128,2,0.3,"tnd"),
        ("GQA g4 BNSD s512",      4,16,512,512,128,4,0.5,"bnsd"),
        ("GQA g4 TND s512",       4,16,512,512,128,4,0.5,"tnd"),
        ("GQA g2 BNSD s256",      2,8,256,256,128,4,0.5,"bnsd"),
        ("GQA g2 TND s256",       2,8,256,256,128,4,0.5,"tnd"),
        # Single batch
        ("B1 GQA g4 BNSD",        1,8,128,128,128,2,0.5,"bnsd"),
        ("B1 GQA g4 TND",         1,8,256,256,128,2,0.5,"tnd"),
        ("B1 MHA BNSD",           1,8,128,128,128,8,0.5,"bnsd"),
        ("B1 MHA TND",            1,8,256,256,128,8,0.5,"tnd"),
        # Large batch
        ("B16 GQA g4 BNSD",       16,8,128,128,128,2,0.5,"bnsd"),
        ("B16 GQA g4 TND",        16,8,128,128,128,2,0.5,"tnd"),
        ("B12 MHA BNSD",          12,4,128,128,128,4,0.5,"bnsd"),
        ("B12 MHA TND",           12,4,128,128,128,4,0.5,"tnd"),
        ("B8 GQA g4 BNSD",        8,8,128,128,128,2,0.5,"bnsd"),
        ("B6 MHA s256 BNSD",      6,8,256,256,128,8,0.5,"bnsd"),
        # Large sequence
        ("GQA g4 S768 BNSD",      2,8,768,768,128,2,0.5,"bnsd"),
        ("GQA g4 S768 TND",       2,8,768,768,128,2,0.5,"tnd"),
        ("MHA N16 S512 BNSD",     2,16,512,512,128,16,0.5,"bnsd"),
        ("MHA N16 S512 TND",      2,16,512,512,128,16,0.5,"tnd"),
        # Asymmetric
        ("GQA S1>>S2 BNSD",       2,8,512,128,128,2,0.5,"bnsd"),
        ("GQA S1<<S2 BNSD",       2,8,128,512,128,2,0.5,"bnsd"),
        ("GQA S1>>S2 TND",        2,8,512,128,128,2,0.5,"tnd"),
        ("GQA S1<<S2 TND",        2,8,128,512,128,2,0.5,"tnd"),
        ("MHA S1>>S2 BNSD",       2,8,384,128,128,8,0.5,"bnsd"),
        ("MHA S1<<S2 TND",        2,8,128,384,128,8,0.5,"tnd"),
        # Sparsity
        ("GQA sp0.1 BNSD",        2,8,256,256,128,2,0.1,"bnsd"),
        ("GQA sp0.1 TND",         2,8,256,256,128,2,0.1,"tnd"),
        ("GQA sp0.9 BNSD",        2,8,256,256,128,2,0.9,"bnsd"),
        ("GQA sp0.9 TND",         2,8,256,256,128,2,0.9,"tnd"),
        ("GQA sp1.0 BNSD",        2,8,256,256,128,2,1.0,"bnsd"),
        ("GQA sp1.0 TND",         2,8,256,256,128,2,1.0,"tnd"),
        ("MHA sp0.1 BNSD",        2,8,256,256,128,8,0.1,"bnsd"),
        ("MHA sp1.0 TND",         2,8,256,256,128,8,1.0,"tnd"),
        # Very small
        ("GQA S32 BNSD",          2,8,32,32,128,2,0.5,"bnsd"),
        ("GQA S32 TND",           2,8,32,32,128,2,0.5,"tnd"),
        ("MHA S16 BNSD",          2,4,16,16,128,4,1.0,"bnsd"),
        ("MHA S16 TND",           2,4,16,16,128,4,1.0,"tnd"),
        ("GQA S1 BNSD",           1,4,1,1,128,2,1.0,"bnsd"),
        ("MHA S1 BNSD",           1,4,1,1,128,4,1.0,"bnsd"),
        # Block boundary
        ("GQA S127 BNSD",         2,8,127,127,128,2,0.5,"bnsd"),
        ("GQA S129 BNSD",         2,8,129,129,128,2,0.5,"bnsd"),
        ("GQA S255 BNSD",         2,8,255,255,128,2,0.5,"bnsd"),
        ("GQA S257 BNSD",         2,8,257,257,128,2,0.5,"bnsd"),
        ("GQA S383 BNSD",         1,8,383,383,128,2,0.5,"bnsd"),
        ("GQA S385 BNSD",         1,8,385,385,128,2,0.5,"bnsd"),
        # Head counts
        ("GQA N32 KV4 BNSD",      1,32,128,128,128,4,0.5,"bnsd"),
        ("GQA N32 KV8 BNSD",      1,32,128,128,128,8,0.5,"bnsd"),
        ("GQA N32 KV2 TND",       1,32,128,128,128,2,0.5,"tnd"),
        ("MHA N2 BNSD",           2,2,128,128,128,2,0.5,"bnsd"),
        ("MHA N2 TND",            2,2,128,128,128,2,0.5,"tnd"),
        ("MHA N32 BNSD",          1,32,128,128,128,32,0.5,"bnsd"),
        # TND var
        ("GQA TND var B5",        5,8,512,512,128,2,0.5,"tnd"),
        ("MHA TND var B5",        5,8,512,512,128,8,0.5,"tnd"),
        ("GQA g8 TND var B3",     3,16,256,256,128,2,0.5,"tnd"),
        ("GQA TND var B10",       10,8,256,256,128,2,0.5,"tnd"),
        # Weird combos
        ("B3 GQA N24 KV6 BNSD",   3,24,256,256,128,6,0.5,"bnsd"),
        ("B3 GQA N24 KV6 TND",    3,24,256,256,128,6,0.5,"tnd"),
        ("B2 GQA g5 N10 KV2",     2,10,256,256,128,2,0.5,"bnsd"),
        ("B1 GQA g9 N18 KV2",     1,18,128,128,128,2,0.5,"tnd"),
        ("B2 MHA N6 S64 sp0.1",   2,6,64,64,128,6,0.1,"bnsd"),
        ("B4 MHA N8 S128 sp0.9",  4,8,128,128,128,8,0.9,"tnd"),
        ("B2 GQA S128+S512",      2,8,128,512,128,2,0.5,"bnsd"),
        ("B2 GQA S512+128 TND",   2,8,512,128,128,2,0.5,"tnd"),
        ("B2 GQA g8 N16 S128",    2,16,128,128,128,2,0.5,"bnsd"),
        ("B4 GQA g4 N8 S64 sp1",  4,8,64,64,128,2,1.0,"bnsd"),
        ("B4 GQA g4 N8 S64 TND",  4,8,64,64,128,2,1.0,"tnd"),
        ("B2 MHA N4 S192 BNSD",   2,4,192,192,128,4,0.5,"bnsd"),
        ("B2 MHA N4 S192 TND",    2,4,192,192,128,4,0.5,"tnd"),
        ("B3 GQA N12 S320 BNSD",  3,12,320,320,128,3,0.5,"bnsd"),
        ("B3 GQA N12 S320 TND",   3,12,320,320,128,3,0.5,"tnd"),
        ("B2 GQA N8 S640 BNSD",   2,8,640,640,128,2,0.5,"bnsd"),
        ("B2 GQA N8 S640 TND",    2,8,640,640,128,2,0.5,"tnd"),
        ("B1 MHA N8 S768 sp0.3",  1,8,768,768,128,8,0.3,"bnsd"),
        ("B1 GQA N4 S896 BNSD",   1,4,896,896,128,1,0.5,"bnsd"),
    ]

    print(f"Total: {len(C)} shapes\n{'='*120}",flush=True)
    R=[]
    for i,(desc,B,N,S1,S2,D,kvh,sp,fmt) in enumerate(C):
        r=test(desc,B,N,S1,S2,D,kvh,sp,fmt)
        R.append(r)
    ok=sum(1 for r in R if r[2] and r[3] and r[4])
    dp=sum(1 for r in R if r[2])
    ndp=sum(1 for r in R if r[3])
    de=sum(1 for r in R if r[4])
    T=len(R)
    print(f"\n{'='*120}")
    print(f"SUMMARY: {T} shapes")
    print(f"  det=True  precision:  {dp}/{T}")
    print(f"  det=False precision:  {ndp}/{T}")
    print(f"  det=True  bit-exact:  {de}/{T}")
    fails=[r for r in R if not(r[2] and r[3] and r[4])]
    if fails:
        print(f"\n--- Failed: {len(fails)} ---")
        for d,t,dp2,ndp2,de2 in fails:
            print(f"  {d:<28} {t}  det_prec={'P' if dp2 else 'F'} nondet_prec={'P' if ndp2 else 'F'} det_exact={'Y' if de2 else 'N'}")
    else:
        print(f"\n  ALL {T} PASS!")
