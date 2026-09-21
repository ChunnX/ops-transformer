(fzy) root@ubuntu:/home/y00845357/github/flash-linear-attention-main# python tools/benchmark_kda_gpu_kernels.py --device 0 --op-filter chunk_kda_bwd_intra  --warmup 5 --iters 20
Benchmark config: device=cuda:0, warmup=5, iters=20, num_cases=10, case_xlsx=/home/y00845357/github/flash-linear-attention-main/tools/KDA算子启用10个Case汇总.xlsx

Cases:
  chunk_kda_bwd_intra: 10 cases
    row=38 case=01:other-shape6: Varlen,B=1,T=8192,H=32,HV=32,K=128,V=128,BT=64,BC=16,bfloat16,safe_gate=False,seqs=3
    row=39 case=02:shape7: Varlen,B=1,T=8192,H=96,HV=96,K=128,V=128,BT=64,BC=16,bfloat16,safe_gate=True,seqs=3
    row=40 case=03:other-shape8: Varlen,B=1,T=4607,H=32,HV=32,K=128,V=128,BT=64,BC=16,bfloat16,safe_gate=False,seqs=513
    row=41 case=04:shape9: Varlen,B=1,T=4607,H=96,HV=96,K=128,V=128,BT=64,BC=16,bfloat16,safe_gate=True,seqs=513
    row=42 case=05:shape10: Varlen,B=1,T=1023,H=96,HV=96,K=128,V=128,BT=64,BC=16,bfloat16,safe_gate=True,seqs=126
    row=43 case=06:other-shape1: Padded,B=2,T=2048,H=8,HV=8,K=128,V=128,BT=64,BC=16,float16,safe_gate=False
    row=44 case=07:shape2: Padded,B=10,T=1024,H=4,HV=4,K=64,V=64,BT=64,BC=16,float16,safe_gate=True
    row=45 case=08:shape3: Padded,B=15,T=1000,H=3,HV=3,K=64,V=64,BT=64,BC=16,float16,safe_gate=True
    row=46 case=09:shape4: Padded,B=1,T=4096,H=32,HV=32,K=128,V=128,BT=64,BC=16,bfloat16,safe_gate=True
    row=47 case=10:shape5: Padded,B=3,T=4096,H=96,HV=96,K=128,V=128,BT=64,BC=16,bfloat16,safe_gate=True

op                                                       case   mean(ms)        p50        p90        min        max  shape / note
----------------------------------------------------------------------------------------------------------------------------------------------------------------------
Current Triton version 3.2.0 is below the recommended 3.3.0 version. Errors may occur and these issues will not be fixed. Please consider upgrading Triton.
Current Python version 3.10 is below the recommended 3.11 version. It is recommended to upgrade to Python 3.11 or higher for the best experience.
chunk_kda_bwd_intra                           01:other-shape6     5.4936     5.4968     5.5204     5.4651     5.5398  Varlen,B=1,T=8192,H=32,HV=32,K=128,V=128,BT=64,BC=16,bfloat16,safe_gate=False,seqs=3 / Python wrapper, includes temporary allocation and db sum
chunk_kda_bwd_intra                                 02:shape7     FAILED  Varlen,B=1,T=8192,H=96,HV=96,K=128,V=128,BT=64,BC=16,bfloat16,safe_gate=True,seqs=3 / CompilationError: at 178:4:
    p_db = db + o_c * HV

    b_dg2 = b_q * b_dq2
    b_dq2 = b_dq2 + tl.load(p_dq, mask=m_ck, other=0.0)
    tl.store(p_dq2, b_dq2.to(p_dq2.dtype.element_ty), mask=m_ck)
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), mask=m_c)

    tl.debug_barrier()
    b_dkt = tl.zeros([BC, BK], dtype=tl.float32)

    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC - 1:
    ^
AssertionError('initial value for `b_gn` is of type <[1, 32], bf16>, but the then block redefines it as <[1, 32], fp32>')
chunk_kda_bwd_intra                           03:other-shape8     4.5844     4.5834     4.5916     4.5711     4.5988  Varlen,B=1,T=4607,H=32,HV=32,K=128,V=128,BT=64,BC=16,bfloat16,safe_gate=False,seqs=513 / Python wrapper, includes temporary allocation and db sum
chunk_kda_bwd_intra                                 04:shape9     FAILED  Varlen,B=1,T=4607,H=96,HV=96,K=128,V=128,BT=64,BC=16,bfloat16,safe_gate=True,seqs=513 / CompilationError: at 178:4:
    p_db = db + o_c * HV

    b_dg2 = b_q * b_dq2
    b_dq2 = b_dq2 + tl.load(p_dq, mask=m_ck, other=0.0)
    tl.store(p_dq2, b_dq2.to(p_dq2.dtype.element_ty), mask=m_ck)
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), mask=m_c)

    tl.debug_barrier()
    b_dkt = tl.zeros([BC, BK], dtype=tl.float32)

    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC - 1:
    ^
AssertionError('initial value for `b_gn` is of type <[1, 32], bf16>, but the then block redefines it as <[1, 32], fp32>')
chunk_kda_bwd_intra                                05:shape10     FAILED  Varlen,B=1,T=1023,H=96,HV=96,K=128,V=128,BT=64,BC=16,bfloat16,safe_gate=True,seqs=126 / CompilationError: at 178:4:
    p_db = db + o_c * HV

    b_dg2 = b_q * b_dq2
    b_dq2 = b_dq2 + tl.load(p_dq, mask=m_ck, other=0.0)
    tl.store(p_dq2, b_dq2.to(p_dq2.dtype.element_ty), mask=m_ck)
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), mask=m_c)

    tl.debug_barrier()
    b_dkt = tl.zeros([BC, BK], dtype=tl.float32)

    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC - 1:
    ^
AssertionError('initial value for `b_gn` is of type <[1, 32], bf16>, but the then block redefines it as <[1, 32], fp32>')
chunk_kda_bwd_intra                           06:other-shape1     0.9306     0.9124     0.9687     0.8970     1.0127  Padded,B=2,T=2048,H=8,HV=8,K=128,V=128,BT=64,BC=16,float16,safe_gate=False / Python wrapper, includes temporary allocation and db sum
chunk_kda_bwd_intra                                 07:shape2     FAILED  Padded,B=10,T=1024,H=4,HV=4,K=64,V=64,BT=64,BC=16,float16,safe_gate=True / CompilationError: at 178:4:
    p_db = db + o_c * HV

    b_dg2 = b_q * b_dq2
    b_dq2 = b_dq2 + tl.load(p_dq, mask=m_ck, other=0.0)
    tl.store(p_dq2, b_dq2.to(p_dq2.dtype.element_ty), mask=m_ck)
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), mask=m_c)

    tl.debug_barrier()
    b_dkt = tl.zeros([BC, BK], dtype=tl.float32)

    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC - 1:
    ^
AssertionError('initial value for `b_gn` is of type <[1, 32], fp16>, but the then block redefines it as <[1, 32], fp32>')
chunk_kda_bwd_intra                                 08:shape3     FAILED  Padded,B=15,T=1000,H=3,HV=3,K=64,V=64,BT=64,BC=16,float16,safe_gate=True / CompilationError: at 178:4:
    p_db = db + o_c * HV

    b_dg2 = b_q * b_dq2
    b_dq2 = b_dq2 + tl.load(p_dq, mask=m_ck, other=0.0)
    tl.store(p_dq2, b_dq2.to(p_dq2.dtype.element_ty), mask=m_ck)
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), mask=m_c)

    tl.debug_barrier()
    b_dkt = tl.zeros([BC, BK], dtype=tl.float32)

    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC - 1:
    ^
AssertionError('initial value for `b_gn` is of type <[1, 32], fp16>, but the then block redefines it as <[1, 32], fp32>')
chunk_kda_bwd_intra                                 09:shape4     FAILED  Padded,B=1,T=4096,H=32,HV=32,K=128,V=128,BT=64,BC=16,bfloat16,safe_gate=True / CompilationError: at 178:4:
    p_db = db + o_c * HV

    b_dg2 = b_q * b_dq2
    b_dq2 = b_dq2 + tl.load(p_dq, mask=m_ck, other=0.0)
    tl.store(p_dq2, b_dq2.to(p_dq2.dtype.element_ty), mask=m_ck)
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), mask=m_c)

    tl.debug_barrier()
    b_dkt = tl.zeros([BC, BK], dtype=tl.float32)

    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC - 1:
    ^
AssertionError('initial value for `b_gn` is of type <[1, 32], bf16>, but the then block redefines it as <[1, 32], fp32>')
chunk_kda_bwd_intra                                 10:shape5     FAILED  Padded,B=3,T=4096,H=96,HV=96,K=128,V=128,BT=64,BC=16,bfloat16,safe_gate=True / CompilationError: at 178:4:
    p_db = db + o_c * HV

    b_dg2 = b_q * b_dq2
    b_dq2 = b_dq2 + tl.load(p_dq, mask=m_ck, other=0.0)
    tl.store(p_dq2, b_dq2.to(p_dq2.dtype.element_ty), mask=m_ck)
    tl.store(p_db, b_db.to(p_db.dtype.element_ty), mask=m_c)

    tl.debug_barrier()
    b_dkt = tl.zeros([BC, BK], dtype=tl.float32)

    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC - 1:
    ^
AssertionError('initial value for `b_gn` is of type <[1, 32], bf16>, but the then block redefines it as <[1, 32], fp32>')