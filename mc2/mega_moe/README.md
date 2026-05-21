ninja: no work to do.
ninja: no work to do.
ninja: no work to do.
schema entrance ========================

ninja: no work to do.
ninja: no work to do.
===============================================================
cur run case: MOE_Dispatch_FFN_Combine_RedLine046, bs: 512, h: 7168, k: 8, local_moe: 4
dispatch_quant_out_type: 24, quant_dtype: torch.float8_e4m3fn, quant_mode: 4
===============================================================
[INTERLEAVE CHECK in GenInputAndGolden] h=7168 256对齐 ✓, n=3072 256对齐 ✓; tile 数=12, 每 tile = 128 gate + 128 up
The transpose flag is true, need to trans last 2 dim
The transpose flag is true, need to trans last 2 dim
The transpose flag is true, need to trans last 2 dim
The transpose flag is true, need to trans last 2 dim
[INTERLEAVE] 交织前 w1_npu shape=(8, 3072, 7168), weight_scale1_npu_cpu shape=(8, 3072, 112, 2)
[INTERLEAVE] 交织后 w1_npu shape=(8, 3072, 7168), weight_scale1_npu_cpu shape=(8, 3072, 112, 2)
---------------start to caculate Matmul_1 ------------- 
---INF_num = 0, NAN_num = 0, Ratio = 0.0 %
---------------start to process swiglu ------------- 
---------------start to process swiglu_quant ------------- 
The transpose flag is false, no need to trans
---------------start to caculate Matmul_2 ------------- 
single_server scene!!!!!
rank list is: [0, 1]
ninja: no work to do.
ninja: no work to do.
ninja: no work to do.
schema entrance ========================

ninja: no work to do.
ninja: no work to do.
ninja: no work to do.
ninja: no work to do.
ninja: no work to do.
schema entrance ========================

ninja: no work to do.
ninja: no work to do.
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 0/32] [rank=0 DISPATCH bk=0 ct=1 sub=0] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 0/32] [rank=0 DISPATCH bk=0 ct=1 sub=0] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 0/32] [rank=0 DISPATCH bk=0 ct=1 sub=0] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 0/32] [rank=0 DISPATCH bk=0 ct=1 sub=0] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 1/32] [rank=0 AIC bk=1] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 2/32] [rank=0 AIC bk=2] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 3/32] [rank=0 AIC bk=3] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 4/32] [rank=0 AIC bk=4] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 5/32] [rank=0 AIC bk=5] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 6/32] [rank=0 AIC bk=6] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 7/32] [rank=0 AIC bk=7] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 8/32] [rank=0 AIC bk=8] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 9/32] [rank=0 AIC bk=9] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 10/32] [rank=0 AIC bk=10] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 11/32] [rank=0 AIC bk=11] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 12/32] [rank=0 AIC bk=12] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 13/32] [rank=0 AIC bk=13] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 14/32] [rank=0 AIC bk=14] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 15/32] [rank=0 AIC bk=15] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 0/64] [rank=0 DISPATCH bk=0 ct=2 sub=0] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 0/64] [rank=0 SCHED] >>> ENTRY exp=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 0/64] [rank=0 SCHED] TIMEOUT exp=0
[AIV Block 0/64]   wave=0 flag=0 d1=12 d2=0
[AIV Block 0/64]   wave=1 flag=0 d1=12 d2=0
[AIV Block 0/64]   wave=2 flag=0 d1=7 d2=0
[AIV Block 0/64]   wave=3 flag=0 d1=0 d2=0
[AIV Block 0/64]   STUCK bk=1 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=1 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=2 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=2 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=3 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=3 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=4 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=4 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=5 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=5 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=6 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=6 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=7 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=7 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=8 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=8 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=9 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=9 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=10 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=10 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=11 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=11 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=12 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=12 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=13 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=13 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=14 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=14 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=15 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=15 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=16 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=16 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=17 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=17 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=18 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=18 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=19 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=19 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=20 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=20 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=21 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=21 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=22 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=22 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=23 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=23 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=24 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=24 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=25 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=25 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=26 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=26 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=27 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=27 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=28 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=28 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=29 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=29 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=30 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=30 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=31 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=31 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64] [rank=0 SCHED] <<< EXIT exp=0
[AIV Block 0/64] [rank=0 DISPATCH bk=0 ct=2 sub=0] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 0/64] [rank=0 SCHED] >>> ENTRY exp=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 1/64] [rank=0 DISPATCH bk=0 ct=2 sub=1] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 1/64] [rank=0 DISPATCH bk=0 ct=2 sub=1] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 1/64] [rank=0 DISPATCH bk=0 CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 0/32] [rank=1 DISPATCH bk=0 ct=1 sub=0] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 0/32] [rank=1 DISPATCH bk=0 ct=1 sub=0] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 0/32] [rank=1 DISPATCH bk=0 ct=1 sub=0] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 0/32] [rank=1 DISPATCH bk=0 ct=1 sub=0] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 1/32] [rank=1 AIC bk=1] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 2/32] [rank=1 AIC bk=2] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 3/32] [rank=1 AIC bk=3] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 4/32] [rank=1 AIC bk=4] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 5/32] [rank=1 AIC bk=5] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 6/32] [rank=1 AIC bk=6] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 7/32] [rank=1 AIC bk=7] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 8/32] [rank=1 AIC bk=8] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 9/32] [rank=1 AIC bk=9] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 10/32] [rank=1 AIC bk=10] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 11/32] [rank=1 AIC bk=11] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 12/32] [rank=1 AIC bk=12] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 13/32] [rank=1 AIC bk=13] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 14/32] [rank=1 AIC bk=14] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 15/32] [rank=1 AIC bk=15] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 0/64] [rank=1 DISPATCH bk=0 ct=2 sub=0] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 0/64] [rank=1 SCHED] >>> ENTRY exp=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 0/64] [rank=1 SCHED] TIMEOUT exp=0
[AIV Block 0/64]   wave=0 flag=0 d1=12 d2=0
[AIV Block 0/64]   wave=1 flag=0 d1=12 d2=0
[AIV Block 0/64]   wave=2 flag=0 d1=7 d2=0
[AIV Block 0/64]   wave=3 flag=0 d1=0 d2=0
[AIV Block 0/64]   STUCK bk=1 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=1 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=2 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=2 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=3 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=3 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=4 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=4 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=5 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=5 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=6 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=6 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=7 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=7 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=8 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=8 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=9 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=9 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=10 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=10 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=11 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=11 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=12 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=12 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=13 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=13 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=14 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=14 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=15 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=15 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=16 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=16 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=17 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=17 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=18 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=18 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=19 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=19 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=20 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=20 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=21 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=21 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=22 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=22 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=23 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=23 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=24 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=24 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=25 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=25 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=26 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=26 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=27 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=27 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=28 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=28 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=29 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=29 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=30 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=30 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=31 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=31 ch=1 myseq=0 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64] [rank=1 SCHED] <<< EXIT exp=0
[AIV Block 0/64] [rank=1 DISPATCH bk=0 ct=2 sub=0] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 0/64] [rank=1 SCHED] >>> ENTRY exp=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 1/64] [rank=1 DISPATCH bk=0 ct=2 sub=1] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 1/64] [rank=1 DISPATCH bk=0 ct=2 sub=1] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 1/64] [rank=1 DISPATCH bk=0 ct=2 sub=1] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 1/64] [rank=0 DISPATCH bk=0 ct=2 sub=1] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 2/64] [rank=0 AIV0 bk=1] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 3/64] [rank=0 AIV1 bk=1] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 4/64] [rank=0 AIV0 bk=2] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 5/64] [rank=0 AIV1 bk=2] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 6/64] [rank=0 AIV0 bk=3] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 7/64] [rank=0 AIV1 bk=3] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 8/64] [rank=0 AIV0 bk=4] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 9/64] [rank=0 AIV1 bk=4] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 10/64] [rank=0 AIV0 bk=5] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 11/64] [rank=0 AIV1 bk=5] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 12/64] [rank=0 AIV0 bk=6] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 13/64] [rank=0 AIV1 bk=6] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 14/64] [rank=0 AIV0 bk=7] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 15/64] [rank=0 AIV1 bk=7] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 16/64] [rank=0 AIV0 bk=8] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 17/64] [rank=0 AIV1 bk=8] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 18/64] [rank=0 AIV0 bk=9] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 19/64] [rank=0 AIV1 bk=9] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 20/64] [rank=0 AIV0 bk=10] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 21/64] [rank=0 AIV1 bk=10] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 22/64] [rank=0 AIV0 bk=11] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 23/64] [rank=0 AIV1 bk=11] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 24/64] [rank=0 AIV0 bk=12] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 25/64] [rank=0 AIV1 bk=12] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 26/64] [rank=0 AIV0 bk=13] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 27/64] [rank=0 AIV1 bk=13] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 28/64] [rank=0 AIV0 bk=14] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 29/64] [rank=0 AIV1 bk=14] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 30/64] [rank=0 AIV0 bk=15] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 31/64] [rank=0 AIV1 bk=15] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 16/32] [rank=0 AIC bk=16] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 17/32] [rank=0 AIC bk=17] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 18/32] [rank=0 AIC bk=18] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 19/32] [rank=0 AIC bk=19] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 20/32] [rank=0 AIC bk=20] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 21/32] [rank=0 AIC bk=21] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 22/32] [rank=0 AIC bk=22] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 23/32] [rank=0 AIC bk=23] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 24/32] [rank=0 AIC bk=24] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 25/32] [rank=0 DISPATCH bk=25 ct=1 sub=0] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 25/32] [rank=0 AIC bk=25] >>> ENTRY exp=0 tnG1=12 tnG2=28
[AIC Block 25/32] [rank=0 AIC bk=25] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
[AIC Block 25/32] [rank=0 AIC bk=25] <<< EXIT exp=0 (G1=0 G2=0)
[AIC Block 25/32] [rank=0 DISPATCH bk=25 ct=1 sub=0] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 25/32] [rank=0 AIC bk=25] >>> ENTRY exp=1 tnG1=12 tnG2=28
[AIC Block 25/32] [rank=0 AIC bk=25] G1#1 w=2 nT=0 waveM=256
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 26/32] [rank=0 AIC bk=26] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 27/32] [rank=0 AIC bk=27] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 28/32] [rank=0 AIC bk=28] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 29/32] [rank=0 AIC bk=29] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 30/32] [rank=0 AIC bk=30] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 31/32] [rank=0 AIC bk=31] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 32/64] [rank=0 AIV0 bk=16] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 33/64] [rank=0 AIV1 bk=16] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 34/64] [rank=0 AIV0 bk=17] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 35/64] [rank=0 AIV1 bk=17] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 36/64] [rank=0 AIV0 bk=18] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 37/64] [rank=0 AIV1 bk=18] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 38/64] [rank=0 AIV0 bk=19] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 39/64] [rank=0 AIV1 bk=19] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 40/64] [rank=0 AIV0 bk=20] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 41/64] [rank=0 AIV1 bk=20] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 42/64] [rank=0 AIV0 bk=21] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 43/64] [rank=0 AIV1 bk=21] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 44/64] [rank=0 AIV0 bk=22] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 45/64] [rank=0 AIV1 bk=22] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 46/64] [rank=0 AIV0 bk=23] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 47/64] [rank=0 AIV1 bk=23] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 48/64] [rank=0 Act=2 sub=1] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 1/64] [rank=1 DISPATCH bk=0 ct=2 sub=1] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 2/64] [rank=1 AIV0 bk=1] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 3/64] [rank=1 AIV1 bk=1] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 4/64] [rank=1 AIV0 bk=2] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 5/64] [rank=1 AIV1 bk=2] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 6/64] [rank=1 AIV0 bk=3] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 7/64] [rank=1 AIV1 bk=3] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 8/64] [rank=1 AIV0 bk=4] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 9/64] [rank=1 AIV1 bk=4] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 10/64] [rank=1 AIV0 bk=5] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 11/64] [rank=1 AIV1 bk=5] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 12/64] [rank=1 AIV0 bk=6] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 13/64] [rank=1 AIV1 bk=6] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 14/64] [rank=1 AIV0 bk=7] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 15/64] [rank=1 AIV1 bk=7] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 16/64] [rank=1 AIV0 bk=8] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 17/64] [rank=1 AIV1 bk=8] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 18/64] [rank=1 AIV0 bk=9] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 19/64] [rank=1 AIV1 bk=9] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 20/64] [rank=1 AIV0 bk=10] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 21/64] [rank=1 AIV1 bk=10] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 22/64] [rank=1 AIV0 bk=11] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 23/64] [rank=1 AIV1 bk=11] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 24/64] [rank=1 AIV0 bk=12] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 25/64] [rank=1 AIV1 bk=12] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 26/64] [rank=1 AIV0 bk=13] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 27/64] [rank=1 AIV1 bk=13] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 28/64] [rank=1 AIV0 bk=14] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 29/64] [rank=1 AIV1 bk=14] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 30/64] [rank=1 AIV0 bk=15] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 31/64] [rank=1 AIV1 bk=15] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 16/32] [rank=1 AIC bk=16] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 17/32] [rank=1 AIC bk=17] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 18/32] [rank=1 AIC bk=18] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 19/32] [rank=1 AIC bk=19] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 20/32] [rank=1 AIC bk=20] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 21/32] [rank=1 AIC bk=21] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 22/32] [rank=1 AIC bk=22] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 23/32] [rank=1 AIC bk=23] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 24/32] [rank=1 AIC bk=24] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 25/32] [rank=1 DISPATCH bk=25 ct=1 sub=0] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 25/32] [rank=1 AIC bk=25] >>> ENTRY exp=0 tnG1=12 tnG2=28
[AIC Block 25/32] [rank=1 AIC bk=25] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
[AIC Block 25/32] [rank=1 AIC bk=25] <<< EXIT exp=0 (G1=0 G2=0)
[AIC Block 25/32] [rank=1 DISPATCH bk=25 ct=1 sub=0] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 25/32] [rank=1 AIC bk=25] >>> ENTRY exp=1 tnG1=12 tnG2=28
[AIC Block 25/32] [rank=1 AIC bk=25] G1#1 w=2 nT=0 waveM=256
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 26/32] [rank=1 AIC bk=26] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 27/32] [rank=1 AIC bk=27] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 28/32] [rank=1 AIC bk=28] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 29/32] [rank=1 AIC bk=29] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 30/32] [rank=1 AIC bk=30] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 31/32] [rank=1 AIC bk=31] *** TIMEOUT idle exp=0 seqG1=0 seqG2=0 lastG1=0 lastG2=0 g1End=0 g2End=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 32/64] [rank=1 AIV0 bk=16] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 33/64] [rank=1 AIV1 bk=16] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 34/64] [rank=1 AIV0 bk=17] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 35/64] [rank=1 AIV1 bk=17] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 36/64] [rank=1 AIV0 bk=18] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 37/64] [rank=1 AIV1 bk=18] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 38/64] [rank=1 AIV0 bk=19] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 39/64] [rank=1 AIV1 bk=19] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 40/64] [rank=1 AIV0 bk=20] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 41/64] [rank=1 AIV1 bk=20] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 42/64] [rank=1 AIV0 bk=21] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 43/64] [rank=1 AIV1 bk=21] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 44/64] [rank=1 AIV0 bk=22] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 45/64] [rank=1 AIV1 bk=22] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 46/64] [rank=1 AIV0 bk=23] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 47/64] [rank=1 AIV1 bk=23] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 48/64] [rank=1 AIV0 bk=24] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 49/64] [rank=0 AIV1 bk=24] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 50/64] [rank=0 DISPATCH bk=25 ct=2 sub=0] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 50/64] [rank=0 AIV0 bk=25] >>> ENTRY exp=0
[AIV Block 50/64] [rank=0 AIV0 bk=25] *** TIMEOUT exp=0 seq=0 last=0
[AIV Block 50/64] [rank=0 AIV0 bk=25] <<< EXIT (tasks=0)
[AIV Block 50/64] [rank=0 DISPATCH bk=25 ct=2 sub=0] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 50/64] [rank=0 AIV0 bk=25] >>> ENTRY exp=1
[AIV Block 50/64] [rank=0 AIV0 bk=25] #1 w=2 nT=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 51/64] [rank=0 DISPATCH bk=25 ct=2 sub=1] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 51/64] [rank=0 AIV1 bk=25] >>> ENTRY exp=0
[AIV Block 51/64] [rank=0 AIV1 bk=25] *** TIMEOUT exp=0 seq=0 last=0
[AIV Block 51/64] [rank=0 AIV1 bk=25] <<< EXIT (tasks=0)
[AIV Block 51/64] [rank=0 DISPATCH bk=25 ct=2 sub=1] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 51/64] [rank=0 AIV1 bk=25] >>> ENTRY exp=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 52/64] [rank=0 AIV0 bk=26] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 53/64] [rank=0 AIV1 bk=26] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 54/64] [rank=0 AIV0 bk=27] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 55/64] [rank=0 AIV1 bk=27] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 56/64] [rank=0 AIV0 bk=28] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 57/64] [rank=0 AIV1 bk=28] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 58/64] [rank=0 AIV0 bk=29] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 59/64] [rank=0 AIV1 bk=29] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 60/64] [rank=0 AIV0 bk=30] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 61/64] [rank=0 AIV1 bk=30] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 62/64] [rank=0 AIV0 bk=31] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 63/64] [rank=0 AIV1 bk=31] *** TIMEOUT exp=0 seq=0 last=0
[AIC Block 1/32] [rank=0 AIC bk=1] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 1/32] [static GMM2 AIC bk=1] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 2/32] [rank=0 AIC bk=2] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 2/32] [static GMM2 AIC bk=2] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 3/32] [rank=0 AIC bk=3] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 3/32] [static GMM2 AIC bk=3] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 4/32] [rank=0 AIC bk=4] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 4/32] [static GMM2 AIC bk=4] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 5/32] [rank=0 AIC bk=5] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 5/32] [static GMM2 AIC bk=5] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 6/32] [rank=0 AIC bk=6] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 6/32] [static GMM2 AIC bk=6] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 7/32] [rank=0 AIC bk=7] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 7/32] [static GMM2 AIC bk=7] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 8/32] [rank=0 AIC bk=8] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 8/32] [static GMM2 AIC bk=8] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 9/32] [rank=0 AIC bk=9] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 9/32] [static GMM2 AIC bk=9] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 10/32] [rank=0 AIC bk=10] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 10/32] [static GMM2 AIC bk=10] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 11/32] [rank=0 AIC bk=11] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 11/32] [static GMM2 AIC bk=11] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 12/32] [rank=0 AIC bk=12] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 13/32] [rank=0 AIC bk=13] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 13/32] [static GMM2 AIC bk=13] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 14/32] [rank=0 AIC bk=14] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 14/32] [static GMM2 AIC bk=14] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 15/32] [rank=0 AIC bk=15] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 15/32] [static GMM2 AIC bk=15] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIV Block 0/64] [rank=0 SCHED] TIMEOUT exp=1
[AIV Block 0/64]   wave=0 flag=12 d1=0 d2=28
[AIV Block 0/64]   wave=1 flag=12 d1=0 d2=3
[AIV Block 0/64]   wave=2 flag=3 d1=0 d2=0
[AIV Block 0/64]   wave=3 flag=0 d1=0 d2=0
[AIV Block 0/64]   STUCK bk=1 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=1 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=2 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=2 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=3 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=3 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=4 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=4 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=5 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=5 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=6 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=6 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=7 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=7 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=8 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=8 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=9 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=9 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=10 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=10 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=11 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=11 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=12 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=12 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=13 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=13 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=14 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=14 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=15 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=15 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=16 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=16 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=17 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=17 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=18 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/IV0 bk=24] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 49/64] [rank=1 AIV1 bk=24] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 50/64] [rank=1 DISPATCH bk=25 ct=2 sub=0] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 50/64] [rank=1 AIV0 bk=25] >>> ENTRY exp=0
[AIV Block 50/64] [rank=1 AIV0 bk=25] *** TIMEOUT exp=0 seq=0 last=0
[AIV Block 50/64] [rank=1 AIV0 bk=25] <<< EXIT (tasks=0)
[AIV Block 50/64] [rank=1 DISPATCH bk=25 ct=2 sub=0] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 50/64] [rank=1 AIV0 bk=25] >>> ENTRY exp=1
[AIV Block 50/64] [rank=1 AIV0 bk=25] #1 w=2 nT=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 51/64] [rank=1 DISPATCH bk=25 ct=2 sub=1] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 51/64] [rank=1 AIV1 bk=25] >>> ENTRY exp=0
[AIV Block 51/64] [rank=1 AIV1 bk=25] *** TIMEOUT exp=0 seq=0 last=0
[AIV Block 51/64] [rank=1 AIV1 bk=25] <<< EXIT (tasks=0)
[AIV Block 51/64] [rank=1 DISPATCH bk=25 ct=2 sub=1] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 51/64] [rank=1 AIV1 bk=25] >>> ENTRY exp=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 52/64] [rank=1 AIV0 bk=26] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 53/64] [rank=1 AIV1 bk=26] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 54/64] [rank=1 AIV0 bk=27] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 55/64] [rank=1 AIV1 bk=27] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 56/64] [rank=1 AIV0 bk=28] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 57/64] [rank=1 AIV1 bk=28] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 58/64] [rank=1 AIV0 bk=29] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 59/64] [rank=1 AIV1 bk=29] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 60/64] [rank=1 AIV0 bk=30] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 61/64] [rank=1 AIV1 bk=30] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 62/64] [rank=1 AIV0 bk=31] *** TIMEOUT exp=0 seq=0 last=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 63/64] [rank=1 AIV1 bk=31] *** TIMEOUT exp=0 seq=0 last=0
[AIC Block 1/32] [rank=1 AIC bk=1] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 1/32] [static GMM2 AIC bk=1] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 2/32] [rank=1 AIC bk=2] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 2/32] [static GMM2 AIC bk=2] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 3/32] [rank=1 AIC bk=3] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 3/32] [static GMM2 AIC bk=3] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 4/32] [rank=1 AIC bk=4] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 4/32] [static GMM2 AIC bk=4] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 5/32] [rank=1 AIC bk=5] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 5/32] [static GMM2 AIC bk=5] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 6/32] [rank=1 AIC bk=6] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 6/32] [static GMM2 AIC bk=6] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 7/32] [rank=1 AIC bk=7] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 7/32] [static GMM2 AIC bk=7] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 8/32] [rank=1 AIC bk=8] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 8/32] [static GMM2 AIC bk=8] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 9/32] [rank=1 AIC bk=9] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 9/32] [static GMM2 AIC bk=9] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 10/32] [rank=1 AIC bk=10] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 10/32] [static GMM2 AIC bk=10] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 11/32] [rank=1 AIC bk=11] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 11/32] [static GMM2 AIC bk=11] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 12/32] [rank=1 AIC bk=12] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 12/32] [static GMM2 AIC bk=12] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 13/32] [rank=1 AIC bk=13] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 13/32] [static GMM2 AIC bk=13] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 14/32] [rank=1 AIC bk=14] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 14/32] [static GMM2 AIC bk=14] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 15/32] [rank=1 AIC bk=15] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 15/32] [static GMM2 AIC bk=15] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIV Block 0/64] [rank=1 SCHED] TIMEOUT exp=1
[AIV Block 0/64]   wave=0 flag=12 d1=0 d2=28
[AIV Block 0/64]   wave=1 flag=12 d1=0 d2=3
[AIV Block 0/64]   wave=2 flag=7 d1=0 d2=0
[AIV Block 0/64]   wave=3 flag=0 d1=0 d2=0
[AIV Block 0/64]   STUCK bk=1 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=1 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=2 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=2 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=3 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=3 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=4 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=4 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=5 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=5 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=6 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=6 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=7 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=7 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=8 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=8 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=9 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=9 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=10 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=10 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=11 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=11 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=12 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=12 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=13 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=13 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=14 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=14 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=15 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=15 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=16 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=16 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=17 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=17 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea64]   STUCK bk=18 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=19 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=19 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=20 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=20 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=21 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=21 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=22 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=22 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=23 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=23 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=24 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=24 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=25 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=25 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=26 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=26 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=27 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=27 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=28 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=28 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=29 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=29 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=30 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=30 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=31 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=31 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64] [rank=0 SCHED] <<< EXIT exp=1
[AIV Block 0/64] [rank=0 DISPATCH bk=0 ct=2 sub=0] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 0/64] [rank=0 SCHED] >>> ENTRY exp=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 2/64] [rank=0 AIV0 bk=1] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 3/64] [rank=0 AIV1 bk=1] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 3/64] [static GMM2 AIV1 bk=1] WARN long spin rowReadyFlags loopIdx=0 cur=0 (keep spinning)
[AIV Block 4/64] [rank=0 AIV0 bk=2] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 5/64] [rank=0 AIV1 bk=2] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 5/64] [static GMM2 AIV1 bk=2] WARN long spin rowReadyFlags loopIdx=1 cur=0 (keep spinning)
[AIV Block 6/64] [rank=0 AIV0 bk=3] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 7/64] [rank=0 AIV1 bk=3] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 7/64] [static GMM2 AIV1 bk=3] WARN long spin rowReadyFlags loopIdx=2 cur=0 (keep spinning)
[AIV Block 8/64] [rank=0 AIV0 bk=4] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 9/64] [rank=0 AIV1 bk=4] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 9/64] [static GMM2 AIV1 bk=4] WARN long spin rowReadyFlags loopIdx=3 cur=0 (keep spinning)
[AIV Block 10/64] [rank=0 AIV0 bk=5] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 11/64] [rank=0 AIV1 bk=5] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 11/64] [static GMM2 AIV1 bk=5] WARN long spin rowReadyFlags loopIdx=4 cur=0 (keep spinning)
[AIV Block 12/64] [rank=0 AIV0 bk=6] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 13/64] [rank=0 AIV1 bk=6] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 13/64] [static GMM2 AIV1 bk=6] WARN long spin rowReadyFlags loopIdx=5 cur=0 (keep spinning)
[AIV Block 14/64] [rank=0 AIV0 bk=7] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 15/64] [rank=0 AIV1 bk=7] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 15/64] [static GMM2 AIV1 bk=7] WARN long spin rowReadyFlags loopIdx=6 cur=0 (keep spinning)
[AIV Block 16/64] [rank=0 AIV0 bk=8] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 17/64] [rank=0 AIV1 bk=8] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 17/64] [static GMM2 AIV1 bk=8] WARN long spin rowReadyFlags loopIdx=7 cur=0 (keep spinning)
[AIV Block 18/64] [rank=0 AIV0 bk=9] ***=0
[AIV Block 0/64]   STUCK bk=18 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=18 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=19 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=19 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=20 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=20 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=21 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=21 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=22 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=22 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=23 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=23 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=24 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=24 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=25 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=25 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=26 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=26 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=27 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=27 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=28 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=28 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=29 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=29 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=30 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=30 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=31 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=31 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64] [rank=1 SCHED] <<< EXIT exp=1
[AIV Block 0/64] [rank=1 DISPATCH bk=0 ct=2 sub=0] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 0/64] [rank=1 SCHED] >>> ENTRY exp=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 2/64] [rank=1 AIV0 bk=1] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 3/64] [rank=1 AIV1 bk=1] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 3/64] [static GMM2 AIV1 bk=1] WARN long spin rowReadyFlags loopIdx=0 cur=0 (keep spinning)
[AIV Block 4/64] [rank=1 AIV0 bk=2] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 5/64] [rank=1 AIV1 bk=2] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 5/64] [static GMM2 AIV1 bk=2] WARN long spin rowReadyFlags loopIdx=1 cur=0 (keep spinning)
[AIV Block 6/64] [rank=1 AIV0 bk=3] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 7/64] [rank=1 AIV1 bk=3] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 7/64] [static GMM2 AIV1 bk=3] WARN long spin rowReadyFlags loopIdx=2 cur=0 (keep spinning)
[AIV Block 8/64] [rank=1 AIV0 bk=4] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 9/64] [rank=1 AIV1 bk=4] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 9/64] [static GMM2 AIV1 bk=4] WARN long spin rowReadyFlags loopIdx=3 cur=0 (keep spinning)
[AIV Block 10/64] [rank=1 AIV0 bk=5] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 11/64] [rank=1 AIV1 bk=5] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 11/64] [static GMM2 AIV1 bk=5] WARN long spin rowReadyFlags loopIdx=4 cur=0 (keep spinning)
[AIV Block 12/64] [rank=1 AIV0 bk=6] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 13/64] [rank=1 AIV1 bk=6] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 13/64] [static GMM2 AIV1 bk=6] WARN long spin rowReadyFlags loopIdx=5 cur=0 (keep spinning)
[AIV Block 14/64] [rank=1 AIV0 bk=7] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 15/64] [rank=1 AIV1 bk=7] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 15/64] [static GMM2 AIV1 bk=7] WARN long spin rowReadyFlags loopIdx=6 cur=0 (keep spinning)
[AIV Block 16/64] [rank=1 AIV0 bk=8] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 17/64] [rank=1 AIV1 bk=8] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 17/64] [static GMM2 AIV1 bk=8] WARN long spi TIMEOUT exp=1 seq=1 last=1
[AIV Block 19/64] [rank=0 AIV1 bk=9] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 19/64] [static GMM2 AIV1 bk=9] WARN long spin rowReadyFlags loopIdx=8 cur=0 (keep spinning)
[AIV Block 20/64] [rank=0 AIV0 bk=10] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 21/64] [rank=0 AIV1 bk=10] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 21/64] [static GMM2 AIV1 bk=10] WARN long spin rowReadyFlags loopIdx=9 cur=0 (keep spinning)
[AIV Block 22/64] [rank=0 AIV0 bk=11] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 23/64] [rank=0 AIV1 bk=11] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 23/64] [static GMM2 AIV1 bk=11] WARN long spin rowReadyFlags loopIdx=10 cur=0 (keep spinning)
[AIV Block 24/64] [rank=0 AIV0 bk=12] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 25/64] [rank=0 AIV1 bk=12] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 26/64] [rank=0 AIV0 bk=13] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 27/64] [rank=0 AIV1 bk=13] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 28/64] [rank=0 AIV0 bk=14] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 29/64] [rank=0 AIV1 bk=14] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 29/64] [static GMM2 AIV1 bk=14] WARN long spin rowReadyFlags loopIdx=13 cur=0 (keep spinning)
[AIV Block 30/64] [rank=0 AIV0 bk=15] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 31/64] [rank=0 AIV1 bk=15] *** TIMEOUT exp=1 seq=0 last=0
[AIC Block 16/32] [rank=0 AIC bk=16] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 16/32] [static GMM2 AIC bk=16] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 17/32] [rank=0 AIC bk=17] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 17/32] [static GMM2 AIC bk=17] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 18/32] [rank=0 AIC bk=18] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 18/32] [static GMM2 AIC bk=18] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 19/32] [rank=0 AIC bk=19] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 19/32] [static GMM2 AIC bk=19] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 20/32] [rank=0 AIC bk=20] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 20/32] [static GMM2 AIC bk=20] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 21/32] [rank=0 AIC bk=21] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 21/32] [static GMM2 AIC bk=21] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 22/32] [rank=0 AIC bk=22] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 22/32] [static GMM2 AIC bk=22] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 23/32] [rank=0 AIC bk=23] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 23/32] [static GMM2 AIC bk=23] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 24/32] [rank=0 AIC bk=24] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 24/32] [static GMM2 AIC bk=24] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 25/32] [rank=0 AIC bk=25] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 25/32] [rank=0 AIC bk=25] <<< EXIT exp=1 (G1=1 G2=0)
[AIC Block 25/32] [rank=0 DISPATCH bk=25 ct=1 sub=0] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 25/32] [rank=0 AIC bk=25] >>> ENTRY exp=2 tnG1=12 tnG2=28
[AIC Block 25/32] [rank=0 AIC bk=25] G2#1 w=0 nT=24 waveM=256
[AIC Block 25/32] [static GMM2 AIC bk=25] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 26/32] [rank=0 AIC bk=26] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 26/32] [static GMM2 AIC bk=26] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 27/32] [rank=0 AIC bk=27] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 27/32] [static GMM2 n rowReadyFlags loopIdx=7 cur=0 (keep spinning)
[AIV Block 18/64] [rank=1 AIV0 bk=9] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 19/64] [rank=1 AIV1 bk=9] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 19/64] [static GMM2 AIV1 bk=9] WARN long spin rowReadyFlags loopIdx=8 cur=0 (keep spinning)
[AIV Block 20/64] [rank=1 AIV0 bk=10] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 21/64] [rank=1 AIV1 bk=10] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 21/64] [static GMM2 AIV1 bk=10] WARN long spin rowReadyFlags loopIdx=9 cur=0 (keep spinning)
[AIV Block 22/64] [rank=1 AIV0 bk=11] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 23/64] [rank=1 AIV1 bk=11] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 23/64] [static GMM2 AIV1 bk=11] WARN long spin rowReadyFlags loopIdx=10 cur=0 (keep spinning)
[AIV Block 24/64] [rank=1 AIV0 bk=12] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 25/64] [rank=1 AIV1 bk=12] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 25/64] [static GMM2 AIV1 bk=12] WARN long spin rowReadyFlags loopIdx=11 cur=0 (keep spinning)
[AIV Block 26/64] [rank=1 AIV0 bk=13] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 27/64] [rank=1 AIV1 bk=13] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 27/64] [static GMM2 AIV1 bk=13] WARN long spin rowReadyFlags loopIdx=12 cur=0 (keep spinning)
[AIV Block 28/64] [rank=1 AIV0 bk=14] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 29/64] [rank=1 AIV1 bk=14] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 29/64] [static GMM2 AIV1 bk=14] WARN long spin rowReadyFlags loopIdx=13 cur=0 (keep spinning)
[AIV Block 30/64] [rank=1 AIV0 bk=15] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 31/64] [rank=1 AIV1 bk=15] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 31/64] [static GMM2 AIV1 bk=15] WARN long spin rowReadyFlags loopIdx=14 cur=0 (keep spinning)
[AIC Block 16/32] [rank=1 AIC bk=16] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 16/32] [static GMM2 AIC bk=16] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 17/32] [rank=1 AIC bk=17] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 17/32] [static GMM2 AIC bk=17] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 18/32] [rank=1 AIC bk=18] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 18/32] [static GMM2 AIC bk=18] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 19/32] [rank=1 AIC bk=19] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 19/32] [static GMM2 AIC bk=19] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 20/32] [rank=1 AIC bk=20] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 20/32] [static GMM2 AIC bk=20] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 21/32] [rank=1 AIC bk=21] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 21/32] [static GMM2 AIC bk=21] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 22/32] [rank=1 AIC bk=22] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 22/32] [static GMM2 AIC bk=22] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 23/32] [rank=1 AIC bk=23] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 23/32] [static GMM2 AIC bk=23] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 24/32] [rank=1 AIC bk=24] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 24/32] [static GMM2 AIC bk=24] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 25/32] [rank=1 AIC bk=25] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 25/32] [rank=1 AIC bk=25] <<< EXIT exp=1 (G1=1 G2=0)
[AIC Block 25/32] [rank=1 DISPATCH bk=25 ct=1 sub=0] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 25/32] [rank=1 AIC bk=25] >>> ENTRY exp=2 tnG1=12 tnG2=28
[AIC Block 25/32] [rank=1 AIC bk=25] G2#1 w=0 nT=24 waveM=256
[AIC Block 25/32] [static GMM2 AIC bk=25] *** TIAIC bk=27] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 28/32] [rank=0 AIC bk=28] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 28/32] [static GMM2 AIC bk=28] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 29/32] [rank=0 AIC bk=29] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 30/32] [rank=0 AIC bk=30] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 31/32] [rank=0 AIC bk=31] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIV Block 32/64] [rank=0 AIV0 bk=16] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 33/64] [rank=0 AIV1 bk=16] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 33/64] [static GMM2 AIV1 bk=16] WARN long spin rowReadyFlags loopIdx=15 cur=0 (keep spinning)
[AIV Block 34/64] [rank=0 AIV0 bk=17] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 35/64] [rank=0 AIV1 bk=17] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 35/64] [static GMM2 AIV1 bk=17] WARN long spin rowReadyFlags loopIdx=16 cur=0 (keep spinning)
[AIV Block 36/64] [rank=0 AIV0 bk=18] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 37/64] [rank=0 AIV1 bk=18] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 37/64] [static GMM2 AIV1 bk=18] WARN long spin rowReadyFlags loopIdx=17 cur=0 (keep spinning)
[AIV Block 38/64] [rank=0 AIV0 bk=19] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 39/64] [rank=0 AIV1 bk=19] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 39/64] [static GMM2 AIV1 bk=19] WARN long spin rowReadyFlags loopIdx=18 cur=0 (keep spinning)
[AIV Block 40/64] [rank=0 AIV0 bk=20] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 41/64] [rank=0 AIV1 bk=20] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 41/64] [static GMM2 AIV1 bk=20] WARN long spin rowReadyFlags loopIdx=19 cur=0 (keep spinning)
[AIV Block 42/64] [rank=0 AIV0 bk=21] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 43/64] [rank=0 AIV1 bk=21] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 43/64] [static GMM2 AIV1 bk=21] WARN long spin rowReadyFlags loopIdx=20 cur=0 (keep spinning)
[AIV Block 44/64] [rank=0 AIV0 bk=22] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 45/64] [rank=0 AIV1 bk=22] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 45/64] [static GMM2 AIV1 bk=22] WARN long spin rowReadyFlags loopIdx=21 cur=0 (keep spinning)
[AIV Block 46/64] [rank=0 AIV0 bk=23] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 47/64] [rank=0 AIV1 bk=23] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 47/64] [static GMM2 AIV1 bk=23] WARN long spin rowReadyFlags loopIdx=22 cur=0 (keep spinning)
[AIV Block 48/64] [rank=0 AIV0 bk=24] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 49/64] [rank=0 AIV1 bk=24] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 49/64] [static GMM2 AIV1 bk=24] WARN long spin rowReadyFlags loopIdx=23 cur=0 (keep spinning)
[AIV Block 50/64] [rank=0 AIV0 bk=25] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 50/64] [rank=0 AIV0 bk=25] <<< EXIT (tasks=1)
[AIV Block 50/64] [rank=0 DISPATCH bk=25 ct=2 sub=0] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 50/64] [rank=0 AIV0 bk=25] >>> ENTRY exp=2
[AIV Block 51/64] [rank=0 AIV1 bk=25] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 51/64] [rank=0 AIV1 bk=25] <<< EXIT (tasks=0)
[AIV Block 51/64] [rank=0 DISPATCH bk=25 ct=2 sub=1] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 51/64] [rank=0 AIV1 bk=25] >>> ENTRY exp=2
[AIV Block 51/64] [rank=0 AIV1 bk=25] #1 w=0 nT=24
[AIV Block 51/64] [static GMM2 AIV1 bk=25] WARN long spin rowReadyFlags loopIdx=24 cur=0 (keep spinning)
[AIV Block 52/64] [rank=0 AIV0 bk=26] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 53/64] [rank=0 AIV1 bk=26] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 53/64] [static GMM2 AIV1 bk=26] WARN long spin rowReadyFlags loopIdx=25 cur=0 (keep spinning)
[AIV Block 54/64] [rank=0 AIV0 bk=27] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 55/64] [rank=0 AIV1 bk=27] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 55/64] [static GMM2 AIV1 bk=27] WARN long spin rowReadyFlags loopIdx=26 cur=0 (keep spinning)
[AIV Block 56/64] [rank=0 AIV0 bk=28] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block MEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 26/32] [rank=1 AIC bk=26] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 26/32] [static GMM2 AIC bk=26] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 27/32] [rank=1 AIC bk=27] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 27/32] [static GMM2 AIC bk=27] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 28/32] [rank=1 AIC bk=28] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 29/32] [rank=1 AIC bk=29] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 30/32] [rank=1 AIC bk=30] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIC Block 31/32] [rank=1 AIC bk=31] *** TIMEOUT idle exp=1 seqG1=1 seqG2=0 lastG1=1 lastG2=0 g1End=0 g2End=0 g1Pending=1
[AIV Block 32/64] [rank=1 AIV0 bk=16] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 33/64] [rank=1 AIV1 bk=16] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 33/64] [static GMM2 AIV1 bk=16] WARN long spin rowReadyFlags loopIdx=15 cur=0 (keep spinning)
[AIV Block 34/64] [rank=1 AIV0 bk=17] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 35/64] [rank=1 AIV1 bk=17] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 35/64] [static GMM2 AIV1 bk=17] WARN long spin rowReadyFlags loopIdx=16 cur=0 (keep spinning)
[AIV Block 36/64] [rank=1 AIV0 bk=18] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 37/64] [rank=1 AIV1 bk=18] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 37/64] [static GMM2 AIV1 bk=18] WARN long spin rowReadyFlags loopIdx=17 cur=0 (keep spinning)
[AIV Block 38/64] [rank=1 AIV0 bk=19] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 39/64] [rank=1 AIV1 bk=19] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 39/64] [static GMM2 AIV1 bk=19] WARN long spin rowReadyFlags loopIdx=18 cur=0 (keep spinning)
[AIV Block 40/64] [rank=1 AIV0 bk=20] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 41/64] [rank=1 AIV1 bk=20] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 41/64] [static GMM2 AIV1 bk=20] WARN long spin rowReadyFlags loopIdx=19 cur=0 (keep spinning)
[AIV Block 42/64] [rank=1 AIV0 bk=21] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 43/64] [rank=1 AIV1 bk=21] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 43/64] [static GMM2 AIV1 bk=21] WARN long spin rowReadyFlags loopIdx=20 cur=0 (keep spinning)
[AIV Block 44/64] [rank=1 AIV0 bk=22] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 45/64] [rank=1 AIV1 bk=22] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 45/64] [static GMM2 AIV1 bk=22] WARN long spin rowReadyFlags loopIdx=21 cur=0 (keep spinning)
[AIV Block 46/64] [rank=1 AIV0 bk=23] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 47/64] [rank=1 AIV1 bk=23] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 47/64] [static GMM2 AIV1 bk=23] WARN long spin rowReadyFlags loopIdx=22 cur=0 (keep spinning)
[AIV Block 48/64] [rank=1 AIV0 bk=24] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 49/64] [rank=1 AIV1 bk=24] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 49/64] [static GMM2 AIV1 bk=24] WARN long spin rowReadyFlags loopIdx=23 cur=0 (keep spinning)
[AIV Block 50/64] [rank=1 AIV0 bk=25] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 50/64] [rank=1 AIV0 bk=25] <<< EXIT (tasks=1)
[AIV Block 50/64] [rank=1 DISPATCH bk=25 ct=2 sub=0] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 50/64] [rank=1 AIV0 bk=25] >>> ENTRY exp=2
[AIV Block 51/64] [rank=1 AIV1 bk=25] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 51/64] [rank=1 AIV1 bk=25] <<< EXIT (tasks=0)
[AIV Block 51/64] [rank=1 DISPATCH bk=25 ct=2 sub=1] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 51/64] [rank=1 AIV1 bk=25] >>> ENTRY exp=2
[AIV Block 51/64] [rank=1 AIV1 bk=25] #1 w=0 nT=24
[AIV Block 51/64] [static GMM2 AIV1 bk=25] WARN long spin rowReadyFlags loopIdx=24 cur=0 (keep spinning)
[AIV Block 52/64] [rank=1 AIV0 bk=26] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 53/64] [rank=1 AIV1 bk=26] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 53/64] [static GMM2 AIV1 bk=26] WARN long spin rowReadyFlags loopIdx=25 cur=0 (keep spinning)
[AIV Block 54/64] [rank=1 AIV0 bk=27] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 55/64] [rank=1 AIV1 bk=27] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 55/64] [static GMM2 AIV1 bk=27] WARN long spin rowReadyFlags loopIdx=26 cur=0 (keep spinning)
[AIV Block 56/64] [rank=1 AIV0 bk=28] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 57/64] [rank=1 AIV1 bk=28] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 58/64] [rank=1 AIV0 bk=29] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 59/64] [rank=1 AIV1 bk=29] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 60/64] [rank=1 AIV0 bk=30] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 61/64] [rank=1 AIV1 bk=30] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 62/64] [rank=1 AIV0 bk=31] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 63/64] [rank=1 AIV1 bk=31] *** TIMEOUT exp=1 seq=0 last=0
[AIC Block 1/32] [rank=1 AIC bk=1] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 2/32] [rank=1 AIC bk=2] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 3/32] [rank=1 AIC bk=3] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 4/32] [rank=1 AIC bk=4] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 5/32] [rank=1 AIC bk=5] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 6/32] [rank=1 AIC bk=6] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 7/32] [rank=1 AIC bk=7] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 8/32] [rank=1 AIC bk=8] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 9/32] [rank=1 AIC bk=9] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 10/32] [rank=1 AIC bk=10] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 11/32] [rank=1 AIC bk=11] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 12/32] [rank=1 AIC bk=12] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 13/32] [rank=1 AIC bk=13] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 14/32] [rank=1 AIC bk=14] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 15/32] [rank=1 AIC bk=15] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIV Block 0/64] [rank=1 SCHED] TIMEOUT exp=2
[AIV Block 0/64]   wave=0 flag=0 d1=0 d2=0
[AIV Block 0/64]   wave=1 flag=0 d1=0 d2=0
[AIV Block 0/64]   wave=2 flag=0 d1=0 d2=0
[AIV Block 0/64]   wave=3 flag=0 d1=0 d2=0
[AIV Block 0/64]   STUCK bk=1 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=1 ch=1 myseq=1 aicAck=1 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=2 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=2 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=3 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=3 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=4 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=4 ch=1 myseq=1 aicAck=1 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=5 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=5 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=6 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=6 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=7 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=7 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=8 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=8 ch=1 myseq=1 aicAck=1 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=9 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=9 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[57/64] [rank=0 AIV1 bk=28] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 58/64] [rank=0 AIV0 bk=29] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 59/64] [rank=0 AIV1 bk=29] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 60/64] [rank=0 AIV0 bk=30] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 61/64] [rank=0 AIV1 bk=30] *** TIMEOUT exp=1 seq=0 last=0
[AIV Block 61/64] [static GMM2 AIV1 bk=30] WARN long spin rowReadyFlags loopIdx=1 cur=0 (keep spinning)
[AIV Block 62/64] [rank=0 AIV0 bk=31] *** TIMEOUT exp=1 seq=1 last=1
[AIV Block 63/64] [rank=0 AIV1 bk=31] *** TIMEOUT exp=1 seq=0 last=0
[AIC Block 1/32] [rank=0 AIC bk=1] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 2/32] [rank=0 AIC bk=2] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 3/32] [rank=0 AIC bk=3] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 4/32] [rank=0 AIC bk=4] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 5/32] [rank=0 AIC bk=5] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 6/32] [rank=0 AIC bk=6] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 7/32] [rank=0 AIC bk=7] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 8/32] [rank=0 AIC bk=8] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 9/32] [rank=0 AIC bk=9] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 10/32] [rank=0 AIC bk=10] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 11/32] [rank=0 AIC bk=11] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 12/32] [rank=0 AIC bk=12] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 13/32] [rank=0 AIC bk=13] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 14/32] [rank=0 AIC bk=14] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 15/32] [rank=0 AIC bk=15] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIV Block 0/64] [rank=0 SCHED] TIMEOUT exp=2
[AIV Block 0/64]   wave=0 flag=0 d1=0 d2=0
[AIV Block 0/64]   wave=1 flag=0 d1=0 d2=0
[AIV Block 0/64]   wave=2 flag=0 d1=0 d2=0
[AIV Block 0/64]   wave=3 flag=0 d1=0 d2=0
[AIV Block 0/64]   STUCK bk=1 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=1 ch=1 myseq=1 aicAck=1 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=2 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=2 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=3 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=3 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=4 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=4 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=5 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=5 ch=1 myseq=1 aicAck=1 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=6 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=6 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=7 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=7 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=8 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=8 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=9 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=9 ch=1 myseq=1 aicAck=1 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=10 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=10 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=11 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 eAIV Block 0/64]   STUCK bk=10 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=10 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=11 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=11 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=12 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=12 ch=1 myseq=1 aicAck=1 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=13 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=13 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=14 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=14 ch=1 myseq=1 aicAck=1 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=15 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=15 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=16 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=16 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=17 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=17 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=18 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=18 ch=1 myseq=1 aicAck=1 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=19 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=19 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=20 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=20 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=21 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=21 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=22 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=22 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=23 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=23 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=24 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=24 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=25 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=25 ch=1 myseq=1 aicAck=1 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=26 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=26 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=27 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=27 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=28 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=28 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=29 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=29 ch=1 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=30 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=30 ch=1 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=31 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=31 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64] [rank=1 SCHED] <<< EXIT exp=2
[AIV Block 0/64] [rank=1 DISPATCH bk=0 ct=2 sub=0] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 0/64] [rank=1 SCHED] >>> ENTRY exp=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 2/64] [rank=1 AIV0 bk=1] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 3/64] [rank=1 AIV1 bk=1] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 4/64] [rank=1 AIV0 bk=2] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 5/64] [rank=1 AIV1 bk=2] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 6/64] [rank=1 AIV0 bk=3] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 7/64] [rank=1 AIV1 bk=3] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 8/64] [rank=1 AIV0 bk=4] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 9/64] [rank=1 AIV1 bk=4] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 10/64] [rank=1 AIV0 bk=5] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 11/64] [rank=1 AIV1 bk=5] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 12/64] [rank=1 AIV0 ba=0
[AIV Block 0/64]   STUCK bk=11 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=12 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=12 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=13 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=13 ch=1 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=14 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=14 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=15 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=15 ch=1 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=16 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=16 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=17 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=17 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=18 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=18 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=19 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=19 ch=1 myseq=1 aicAck=1 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=20 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=20 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=21 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=21 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=22 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=22 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=23 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=23 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=24 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=24 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=25 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=25 ch=1 myseq=1 aicAck=1 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=26 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=26 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=27 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=27 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=28 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=28 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=29 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=29 ch=1 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=30 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=30 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=31 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=31 ch=1 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64] [rank=0 SCHED] <<< EXIT exp=2
[AIV Block 0/64] [rank=0 DISPATCH bk=0 ct=2 sub=0] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 0/64] [rank=0 SCHED] >>> ENTRY exp=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 2/64] [rank=0 AIV0 bk=1] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 3/64] [rank=0 AIV1 bk=1] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 4/64] [rank=0 AIV0 bk=2] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 5/64] [rank=0 AIV1 bk=2] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 6/64] [rank=0 AIV0 bk=3] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 7/64] [rank=0 AIV1 bk=3] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 8/64] [rank=0 AIV0 bk=4] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 9/64] [rank=0 AIV1 bk=4] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 10/64] [rank=0 AIV0 bk=5] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 11/64] [rank=0 AIV1 bk=5] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 12/64] [rank=0 AIV0 bk=6] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 13/64] [rank=0 AIV1 bk=6] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 14/64] [rank=0 AIV0 bk=7] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 15/64] [rank=0 AIV1 bk=7] **k=6] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 13/64] [rank=1 AIV1 bk=6] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 14/64] [rank=1 AIV0 bk=7] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 15/64] [rank=1 AIV1 bk=7] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 16/64] [rank=1 AIV0 bk=8] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 17/64] [rank=1 AIV1 bk=8] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 18/64] [rank=1 AIV0 bk=9] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 19/64] [rank=1 AIV1 bk=9] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 20/64] [rank=1 AIV0 bk=10] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 21/64] [rank=1 AIV1 bk=10] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 22/64] [rank=1 AIV0 bk=11] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 23/64] [rank=1 AIV1 bk=11] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 24/64] [rank=1 AIV0 bk=12] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 25/64] [rank=1 AIV1 bk=12] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 26/64] [rank=1 AIV0 bk=13] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 27/64] [rank=1 AIV1 bk=13] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 28/64] [rank=1 AIV0 bk=14] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 29/64] [rank=1 AIV1 bk=14] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 30/64] [rank=1 AIV0 bk=15] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 31/64] [rank=1 AIV1 bk=15] *** TIMEOUT exp=2 seq=1 last=1
[AIC Block 16/32] [rank=1 AIC bk=16] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 17/32] [rank=1 AIC bk=17] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 18/32] [rank=1 AIC bk=18] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 19/32] [rank=1 AIC bk=19] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 20/32] [rank=1 AIC bk=20] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 21/32] [rank=1 AIC bk=21] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 22/32] [rank=1 AIC bk=22] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 23/32] [rank=1 AIC bk=23] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 24/32] [rank=1 AIC bk=24] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 25/32] [rank=1 AIC bk=25] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 25/32] [rank=1 AIC bk=25] <<< EXIT exp=2 (G1=0 G2=1)
[AIC Block 25/32] [rank=1 DISPATCH bk=25 ct=1 sub=0] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 25/32] [rank=1 AIC bk=25] >>> ENTRY exp=3 tnG1=12 tnG2=28
[AIC Block 26/32] [rank=1 AIC bk=26] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 27/32] [rank=1 AIC bk=27] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 28/32] [rank=1 AIC bk=28] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 29/32] [rank=1 AIC bk=29] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 30/32] [rank=1 AIC bk=30] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 31/32] [rank=1 AIC bk=31] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIV Block 32/64] [rank=1 AIV0 bk=16] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 33/64] [rank=1 AIV1 bk=16] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 34/64] [rank=1 AIV0 bk=17] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 35/64] [rank=1 AIV1 bk=17] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 36/64] [rank=1 AIV0 bk=18] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 37/64] [rank=1 AIV1 bk=18] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 38/64] [rank=1 AIV0 bk=19] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 39/64] [rank=1 AIV1 bk=19] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 40/64] [ran* TIMEOUT exp=2 seq=1 last=1
[AIV Block 16/64] [rank=0 AIV0 bk=8] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 17/64] [rank=0 AIV1 bk=8] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 18/64] [rank=0 AIV0 bk=9] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 19/64] [rank=0 AIV1 bk=9] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 20/64] [rank=0 AIV0 bk=10] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 21/64] [rank=0 AIV1 bk=10] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 22/64] [rank=0 AIV0 bk=11] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 23/64] [rank=0 AIV1 bk=11] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 24/64] [rank=0 AIV0 bk=12] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 25/64] [rank=0 AIV1 bk=12] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 26/64] [rank=0 AIV0 bk=13] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 27/64] [rank=0 AIV1 bk=13] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 28/64] [rank=0 AIV0 bk=14] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 29/64] [rank=0 AIV1 bk=14] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 30/64] [rank=0 AIV0 bk=15] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 31/64] [rank=0 AIV1 bk=15] *** TIMEOUT exp=2 seq=1 last=1
[AIC Block 16/32] [rank=0 AIC bk=16] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 17/32] [rank=0 AIC bk=17] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 18/32] [rank=0 AIC bk=18] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 19/32] [rank=0 AIC bk=19] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 20/32] [rank=0 AIC bk=20] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 21/32] [rank=0 AIC bk=21] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 22/32] [rank=0 AIC bk=22] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 23/32] [rank=0 AIC bk=23] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 24/32] [rank=0 AIC bk=24] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 25/32] [rank=0 AIC bk=25] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 25/32] [rank=0 AIC bk=25] <<< EXIT exp=2 (G1=0 G2=1)
[AIC Block 25/32] [rank=0 DISPATCH bk=25 ct=1 sub=0] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 25/32] [rank=0 AIC bk=25] >>> ENTRY exp=3 tnG1=12 tnG2=28
[AIC Block 26/32] [rank=0 AIC bk=26] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 27/32] [rank=0 AIC bk=27] *** TIMEOUT idle exp=2 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIV Block 32/64] [rank=0 AIV0 bk=16] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 33/64] [rank=0 AIV1 bk=16] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 34/64] [rank=0 AIV0 bk=17] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 35/64] [rank=0 AIV1 bk=17] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 36/64] [rank=0 AIV0 bk=18] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 37/64] [rank=0 AIV1 bk=18] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 38/64] [rank=0 AIV0 bk=19] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 39/64] [rank=0 AIV1 bk=19] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 40/64] [rank=0 AIV0 bk=20] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 41/64] [rank=0 AIV1 bk=20] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 42/64] [rank=0 AIV0 bk=21] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 43/64] [rank=0 AIV1 bk=21] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 44/64] [rank=0 AIV0 bk=22] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 45/64] [rank=0 AIV1 bk=22] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 46/64] [rank=0 AIV0 bk=23] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 47/64] [rank=0 AIV1 bk=23] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 48/64] [rank=0 AIV0 bk=24] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 49/64] [rank=0 AIV1 bk=24] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 50/64] [rank=0 AIV0 k=1 AIV0 bk=20] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 41/64] [rank=1 AIV1 bk=20] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 42/64] [rank=1 AIV0 bk=21] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 43/64] [rank=1 AIV1 bk=21] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 44/64] [rank=1 AIV0 bk=22] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 45/64] [rank=1 AIV1 bk=22] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 46/64] [rank=1 AIV0 bk=23] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 47/64] [rank=1 AIV1 bk=23] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 48/64] [rank=1 AIV0 bk=24] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 49/64] [rank=1 AIV1 bk=24] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 50/64] [rank=1 AIV0 bk=25] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 50/64] [rank=1 AIV0 bk=25] <<< EXIT (tasks=0)
[AIV Block 50/64] [rank=1 DISPATCH bk=25 ct=2 sub=0] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 50/64] [rank=1 AIV0 bk=25] >>> ENTRY exp=3
[AIV Block 51/64] [rank=1 AIV1 bk=25] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 51/64] [rank=1 AIV1 bk=25] <<< EXIT (tasks=1)
[AIV Block 51/64] [rank=1 DISPATCH bk=25 ct=2 sub=1] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 51/64] [rank=1 AIV1 bk=25] >>> ENTRY exp=3
[AIV Block 52/64] [rank=1 AIV0 bk=26] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 53/64] [rank=1 AIV1 bk=26] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 54/64] [rank=1 AIV0 bk=27] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 55/64] [rank=1 AIV1 bk=27] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 56/64] [rank=1 AIV0 bk=28] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 57/64] [rank=1 AIV1 bk=28] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 58/64] [rank=1 AIV0 bk=29] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 59/64] [rank=1 AIV1 bk=29] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 60/64] [rank=1 AIV0 bk=30] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 61/64] [rank=1 AIV1 bk=30] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 62/64] [rank=1 AIV0 bk=31] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 63/64] [rank=1 AIV1 bk=31] *** TIMEOUT exp=2 seq=1 last=1
[AIC Block 1/32] [rank=1 AIC bk=1] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 2/32] [rank=1 AIC bk=2] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 3/32] [rank=1 AIC bk=3] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 4/32] [rank=1 AIC bk=4] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 5/32] [rank=1 AIC bk=5] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 6/32] [rank=1 AIC bk=6] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 7/32] [rank=1 AIC bk=7] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 8/32] [rank=1 AIC bk=8] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 9/32] [rank=1 AIC bk=9] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 10/32] [rank=1 AIC bk=10] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 11/32] [rank=1 AIC bk=11] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 12/32] [rank=1 AIC bk=12] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 13/32] [rank=1 AIC bk=13] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 14/32] [rank=1 AIC bk=14] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 15/32] [rank=1 AIC bk=15] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIV Block 0/64] [rank=1 SCHED] TIMEOUT exp=3
[AIV Block 0/64]   wave=0 flag=0 d1=0 d2=0
[AIV Block 0/64]   wave=1 flag=0 d1=0 d2=0
[AIV Block 0/64]   wave=2 flag=0 d1=0 d2=0
[AIV Block 0/64]   wave=3 flag=0 d1=0 d2=0
[AIV Block 0/64]bk=25] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 50/64] [rank=0 AIV0 bk=25] <<< EXIT (tasks=0)
[AIV Block 50/64] [rank=0 DISPATCH bk=25 ct=2 sub=0] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 50/64] [rank=0 AIV0 bk=25] >>> ENTRY exp=3
[AIV Block 51/64] [rank=0 AIV1 bk=25] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 51/64] [rank=0 AIV1 bk=25] <<< EXIT (tasks=1)
[AIV Block 51/64] [rank=0 DISPATCH bk=25 ct=2 sub=1] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 51/64] [rank=0 AIV1 bk=25] >>> ENTRY exp=3
[AIV Block 52/64] [rank=0 AIV0 bk=26] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 53/64] [rank=0 AIV1 bk=26] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 54/64] [rank=0 AIV0 bk=27] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 55/64] [rank=0 AIV1 bk=27] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 56/64] [rank=0 AIV0 bk=28] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 57/64] [rank=0 AIV1 bk=28] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 58/64] [rank=0 AIV0 bk=29] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 59/64] [rank=0 AIV1 bk=29] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 60/64] [rank=0 AIV0 bk=30] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 62/64] [rank=0 AIV0 bk=31] *** TIMEOUT exp=2 seq=1 last=1
[AIV Block 63/64] [rank=0 AIV1 bk=31] *** TIMEOUT exp=2 seq=1 last=1
[AIC Block 1/32] [rank=0 AIC bk=1] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 2/32] [rank=0 AIC bk=2] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 3/32] [rank=0 AIC bk=3] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 4/32] [rank=0 AIC bk=4] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 5/32] [rank=0 AIC bk=5] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 6/32] [rank=0 AIC bk=6] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 7/32] [rank=0 AIC bk=7] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 8/32] [rank=0 AIC bk=8] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 9/32] [rank=0 AIC bk=9] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 10/32] [rank=0 AIC bk=10] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 11/32] [rank=0 AIC bk=11] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 12/32] [rank=0 AIC bk=12] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 13/32] [rank=0 AIC bk=13] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 14/32] [rank=0 AIC bk=14] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 15/32] [rank=0 AIC bk=15] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIV Block 0/64] [rank=0 SCHED] TIMEOUT exp=3
[AIV Block 0/64]   wave=0 flag=0 d1=0 d2=0
[AIV Block 0/64]   wave=1 flag=0 d1=0 d2=0
[AIV Block 0/64]   wave=2 flag=0 d1=0 d2=0
[AIV Block 0/64]   wave=3 flag=0 d1=0 d2=0
[AIV Block 0/64]   STUCK bk=1 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=1 ch=1 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=2 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=2 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=3 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=3 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=4 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=4 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=5 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=5 ch=1 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=6 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AI   STUCK bk=1 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=1 ch=1 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=2 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=2 ch=1 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=3 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=3 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=4 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=4 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=5 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=5 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=6 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=6 ch=1 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=7 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=7 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=8 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=8 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=9 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=9 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=10 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=10 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=11 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=11 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=12 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=12 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=13 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=13 ch=1 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=14 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=14 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=15 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=15 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=16 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=16 ch=1 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=17 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=17 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=18 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=18 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=19 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=19 ch=1 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=20 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=20 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=21 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=21 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=22 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=22 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=23 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=23 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=24 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=24 ch=1 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=25 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=25 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=26 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=26 ch=1 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=27 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=27 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=28 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=28 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=29 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bV Block 0/64]   STUCK bk=6 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=7 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=7 ch=1 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=8 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=8 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=9 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=9 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=10 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=10 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=11 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=11 ch=1 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=12 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=12 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=13 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=13 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=14 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=14 ch=1 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=15 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=15 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=16 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=16 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=17 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=17 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=18 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=18 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=19 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=19 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=20 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=20 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=21 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=21 ch=1 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=22 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=22 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=23 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=23 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=24 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=24 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=25 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=25 ch=1 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=26 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=26 ch=1 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=27 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=27 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=28 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=28 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=29 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=29 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=30 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=30 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=31 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=31 ch=1 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64] [rank=0 SCHED] <<< EXIT exp=3
[AIV Block 2/64] [rank=0 AIV0 bk=1] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 3/64] [rank=0 AIV1 bk=1] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 4/64] [rank=0 AIV0 bk=2] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 5/64] [rank=0 AIV1 bk=2] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 6/64] [rank=0 AIV0 bk=3] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 7/64] [rank=0 AIV1 bk=3] *** TIMEOUT exp=k=29 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=30 ch=0 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=30 ch=1 myseq=1 aicAck=0 aivAck=1 es=0 ea=0
[AIV Block 0/64]   STUCK bk=31 ch=0 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64]   STUCK bk=31 ch=1 myseq=1 aicAck=0 aivAck=0 es=0 ea=0
[AIV Block 0/64] [rank=1 SCHED] <<< EXIT exp=3
[AIV Block 2/64] [rank=1 AIV0 bk=1] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 3/64] [rank=1 AIV1 bk=1] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 4/64] [rank=1 AIV0 bk=2] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 5/64] [rank=1 AIV1 bk=2] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 6/64] [rank=1 AIV0 bk=3] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 7/64] [rank=1 AIV1 bk=3] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 8/64] [rank=1 AIV0 bk=4] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 9/64] [rank=1 AIV1 bk=4] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 10/64] [rank=1 AIV0 bk=5] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 11/64] [rank=1 AIV1 bk=5] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 12/64] [rank=1 AIV0 bk=6] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 13/64] [rank=1 AIV1 bk=6] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 14/64] [rank=1 AIV0 bk=7] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 15/64] [rank=1 AIV1 bk=7] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 16/64] [rank=1 AIV0 bk=8] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 17/64] [rank=1 AIV1 bk=8] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 18/64] [rank=1 AIV0 bk=9] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 19/64] [rank=1 AIV1 bk=9] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 20/64] [rank=1 AIV0 bk=10] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 21/64] [rank=1 AIV1 bk=10] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 22/64] [rank=1 AIV0 bk=11] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 23/64] [rank=1 AIV1 bk=11] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 24/64] [rank=1 AIV0 bk=12] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 25/64] [rank=1 AIV1 bk=12] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 26/64] [rank=1 AIV0 bk=13] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 27/64] [rank=1 AIV1 bk=13] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 28/64] [rank=1 AIV0 bk=14] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 29/64] [rank=1 AIV1 bk=14] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 30/64] [rank=1 AIV0 bk=15] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 31/64] [rank=1 AIV1 bk=15] *** TIMEOUT exp=3 seq=1 last=1
[AIC Block 16/32] [rank=1 AIC bk=16] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 17/32] [rank=1 AIC bk=17] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 18/32] [rank=1 AIC bk=18] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 19/32] [rank=1 AIC bk=19] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 20/32] [rank=1 AIC bk=20] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 21/32] [rank=1 AIC bk=21] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 22/32] [rank=1 AIC bk=22] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 23/32] [rank=1 AIC bk=23] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 24/32] [rank=1 AIC bk=24] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 25/32] [rank=1 AIC bk=25] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 25/32] [rank=1 AIC bk=25] <<< EXIT exp=3 (G1=0 G2=0)
[AIC Block 26/32] [rank=1 AIC bk=26] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 27/32] [rank=1 AIC bk=27] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 28/32] [rank=1 AIC bk=28] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 29/32] [ra3 seq=1 last=1
[AIV Block 8/64] [rank=0 AIV0 bk=4] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 9/64] [rank=0 AIV1 bk=4] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 10/64] [rank=0 AIV0 bk=5] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 11/64] [rank=0 AIV1 bk=5] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 12/64] [rank=0 AIV0 bk=6] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 13/64] [rank=0 AIV1 bk=6] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 14/64] [rank=0 AIV0 bk=7] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 15/64] [rank=0 AIV1 bk=7] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 16/64] [rank=0 AIV0 bk=8] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 17/64] [rank=0 AIV1 bk=8] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 18/64] [rank=0 AIV0 bk=9] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 19/64] [rank=0 AIV1 bk=9] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 20/64] [rank=0 AIV0 bk=10] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 21/64] [rank=0 AIV1 bk=10] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 22/64] [rank=0 AIV0 bk=11] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 23/64] [rank=0 AIV1 bk=11] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 24/64] [rank=0 AIV0 bk=12] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 25/64] [rank=0 AIV1 bk=12] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 26/64] [rank=0 AIV0 bk=13] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 27/64] [rank=0 AIV1 bk=13] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 28/64] [rank=0 AIV0 bk=14] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 29/64] [rank=0 AIV1 bk=14] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 30/64] [rank=0 AIV0 bk=15] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 31/64] [rank=0 AIV1 bk=15] *** TIMEOUT exp=3 seq=1 last=1
[AIC Block 16/32] [rank=0 AIC bk=16] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 17/32] [rank=0 AIC bk=17] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 18/32] [rank=0 AIC bk=18] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 19/32] [rank=0 AIC bk=19] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 20/32] [rank=0 AIC bk=20] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 21/32] [rank=0 AIC bk=21] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 22/32] [rank=0 AIC bk=22] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 23/32] [rank=0 AIC bk=23] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 24/32] [rank=0 AIC bk=24] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 25/32] [rank=0 AIC bk=25] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 25/32] [rank=0 AIC bk=25] <<< EXIT exp=3 (G1=0 G2=0)
[AIC Block 26/32] [rank=0 AIC bk=26] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 27/32] [rank=0 AIC bk=27] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIV Block 32/64] [rank=0 AIV0 bk=16] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 33/64] [rank=0 AIV1 bk=16] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 34/64] [rank=0 AIV0 bk=17] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 35/64] [rank=0 AIV1 bk=17] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 36/64] [rank=0 AIV0 bk=18] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 37/64] [rank=0 AIV1 bk=18] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 38/64] [rank=0 AIV0 bk=19] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 39/64] [rank=0 AIV1 bk=19] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 40/64] [rank=0 AIV0 bk=20] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 41/64] [rank=0 AIV1 bk=20] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 42/64] [rank=0 AIV0 bk=21] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 43/64] [rank=0 AIV1 bk=21] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 44/64] [rank=0 AIV0 bk=22] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 45/===============================================================
cur run case: MOE_Dispatch_FFN_Combine_RedLine046, bs: 512, h: 7168, k: 8, local_moe: 4
dispatch_quant_out_type: 24, quant_dtype: torch.float8_e4m3fn, quant_mode: 4
===============================================================
current device set: 1
[INFO] device_1 创建HCCL通信链路
device_1 init_process_group success
device 1 初始化EP域
device 1 初始化TP域
jjjjjjjjjjjjjjj
<npu_ops_transformer.ops.mega_moe.SymmBuffer object at 0x7fbfced88a30>
rank:1 topk_idstensor([[0, 1, 2,  ..., 5, 6, 7],
        [1, 2, 3,  ..., 6, 7, 0],
        [2, 3, 4,  ..., 7, 0, 1],
        ...,
        [5, 6, 7,  ..., 2, 3, 4],
        [6, 7, 0,  ..., 3, 4, 5],
        [7, 0, 1,  ..., 4, 5, 6]], device='npu:1', dtype=torch.int32)
rank:1 构造算子输入finished!
nk=1 AIC bk=29] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 30/32] [rank=1 AIC bk=30] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIC Block 31/32] [rank=1 AIC bk=31] *** TIMEOUT idle exp=3 seqG1=1 seqG2=1 lastG1=1 lastG2=1 g1End=0 g2End=0 g1Pending=0
[AIV Block 32/64] [rank=1 AIV0 bk=16] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 33/64] [rank=1 AIV1 bk=16] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 34/64] [rank=1 AIV0 bk=17] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 35/64] [rank=1 AIV1 bk=17] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 36/64] [rank=1 AIV0 bk=18] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 37/64] [rank=1 AIV1 bk=18] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 38/64] [rank=1 AIV0 bk=19] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 39/64] [rank=1 AIV1 bk=19] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 40/64] [rank=1 AIV0 bk=20] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 41/64] [rank=1 AIV1 bk=20] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 42/64] [rank=1 AIV0 bk=21] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 43/64] [rank=1 AIV1 bk=21] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 44/64] [rank=1 AIV0 bk=22] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 45/64] [rank=1 AIV1 bk=22] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 46/64] [rank=1 AIV0 bk=23] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 47/64] [rank=1 AIV1 bk=23] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 48/64] [rank=1 AIV0 bk=24] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 49/64] [rank=1 AIV1 bk=24] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 50/64] [rank=1 AIV0 bk=25] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 50/64] [rank=1 AIV0 bk=25] <<< EXIT (tasks=0)
[AIV Block 51/64] [rank=1 AIV1 bk=25] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 51/64] [rank=1 AIV1 bk=25] <<< EXIT (tasks=0)
[AIV Block 52/64] [rank=1 AIV0 bk=26] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 53/64] [rank=1 AIV1 bk=26] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 54/64] [rank=1 AIV0 bk=27] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 55/64] [rank=1 AIV1 bk=27] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 56/64] [rank=1 AIV0 bk=28] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 57/64] [rank=1 AIV1 bk=28] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 58/64] [rank=1 AIV0 bk=29] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 59/64] [rank=1 AIV1 bk=29] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 60/64] [rank=1 AIV0 bk=30] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 61/64] [rank=1 AIV1 bk=30] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 62/64] [rank=1 AIV0 bk=31] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 63/64] [rank=1 AIV1 bk=31] *** TIMEOUT exp=3 seq=1 last=1
===============================================================
cur run case: MOE_Dispatch_FFN_Combine_RedLine046, bs: 512, h: 7168, k: 8, local_moe: 4
dispatch_quant_out_type: 24, quant_dtype: torch.float8_e4m3fn, quant_mode: 4
===============================================================
current device set: 0
[INFO] device_0 创建HCCL通信链路
device_0 init_process_group success
device 0 初始化EP域
device 0 初始化TP域
jjjjjjjjjjjjjjj
<npu_ops_transformer.ops.mega_moe.SymmBuffer object at 0x7effcc7d4a30>
rank:0 topk_idstensor([[0, 1, 2,  ..., 5, 6, 7],
        [1, 2, 3,  ..., 6, 7, 0],
        [2, 3, 4,  ..., 7, 0, 1],
        ...,
        [5, 6, 7,  ..., 2, 3, 4],
        [6, 7, 0,  ..., 3, 4, 5],
        [7, 0, 1,  ..., 4, 5, 6]], device='npu:0', dtype=torch.int32)
rank:0 构造算子输入finished!
64] [rank=0 AIV1 bk=22] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 46/64] [rank=0 AIV0 bk=23] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 47/64] [rank=0 AIV1 bk=23] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 48/64] [rank=0 AIV0 bk=24] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 49/64] [rank=0 AIV1 bk=24] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 50/64] [rank=0 AIV0 bk=25] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 50/64] [rank=0 AIV0 bk=25] <<< EXIT (tasks=0)
[AIV Block 51/64] [rank=0 AIV1 bk=25] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 51/64] [rank=0 AIV1 bk=25] <<< EXIT (tasks=0)
[AIV Block 52/64] [rank=0 AIV0 bk=26] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 53/64] [rank=0 AIV1 bk=26] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 54/64] [rank=0 AIV0 bk=27] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 55/64] [rank=0 AIV1 bk=27] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 56/64] [rank=0 AIV0 bk=28] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 57/64] [rank=0 AIV1 bk=28] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 58/64] [rank=0 AIV0 bk=29] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 59/64] [rank=0 AIV1 bk=29] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 60/64] [rank=0 AIV0 bk=30] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 62/64] [rank=0 AIV0 bk=31] *** TIMEOUT exp=3 seq=1 last=1
[AIV Block 63/64] [rank=0 AIV1 bk=31] *** TIMEOUT exp=3 seq=1 last=1
