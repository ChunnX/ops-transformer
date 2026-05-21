========== [rank 3] MegaMoe TIMESTAMPS (unit = us) ==========
  active AIC : 32/36, active AIV0: 32/36, active AIV1: 32/36

  --- [AIC] STAGE durations (per-core delta) ---
    KERNEL_ENTRY -> ROUTING_DONE          n=  32  min=       0  avg=       0  max=       1  (us)
    ROUTING_DONE -> CROSSRANK_DONE        n=  32  min=      74  avg=      75  max=      77  (us)  <-- HOT
    CROSSRANK_DONE -> EXPERT_LOOP_END       n=  32  min=     387  avg=     399  max=     421  (us)  <-- HOT
    EXPERT_LOOP_END -> PROCESS_END           n=  32  min=       0  avg=       0  max=       1  (us)

  --- [AIV0] STAGE durations (per-core delta) ---
    KERNEL_ENTRY -> ROUTING_DONE          n=  32  min=      68  avg=      68  max=      69  (us)  <-- HOT
    ROUTING_DONE -> CROSSRANK_DONE        n=  32  min=       6  avg=       6  max=       7  (us)  <-- HOT
    CROSSRANK_DONE -> EXPERT_LOOP_END       n=  32  min=     355  avg=     363  max=     381  (us)  <-- HOT
    EXPERT_LOOP_END -> PROCESS_END           n=  32  min=      57  avg=      74  max=      83  (us)  <-- HOT

  --- [AIV1] STAGE durations (per-core delta) ---
    KERNEL_ENTRY -> ROUTING_DONE          n=  32  min=      68  avg=      68  max=      69  (us)  <-- HOT
    ROUTING_DONE -> CROSSRANK_DONE        n=  32  min=       6  avg=       6  max=       7  (us)  <-- HOT
    CROSSRANK_DONE -> EXPERT_LOOP_END       n=  32  min=     392  avg=     412  max=     438  (us)  <-- HOT
    EXPERT_LOOP_END -> PROCESS_END           n=  32  min=       0  avg=      25  max=      46  (us)  <-- HOT

  --- [AIV1] per-expert DISPATCH duration ---
    expert[0] dispatch  n=  32  min=       4  avg=      15  max=      19  (us)  <-- HOT
    expert[1] dispatch  n=  32  min=       4  avg=      14  max=      17  (us)  <-- HOT
    expert[2] dispatch  n=  32  min=       7  avg=      15  max=      26  (us)  <-- HOT
    expert[3] dispatch  n=  32  min=       8  avg=      16  max=      27  (us)  <-- HOT

  --- [AIC] WAVE-by-WAVE  (us, avg across active cores) ---
    exp  wave | entr->g1B |      wDsp |    g1Frst |     g1Mid |     g1Drn |    g12gap |      wSwi |    g2Frst |     g2Mid |     g2Drn |    g2->ex |     TOTAL
    ---------------------------------------------------------------------------------------------------------------------------------------------------------
      1     0 |         0 |        19 |        24 |        23 |        10 |         0 |         2 |        15 |        20 |         0 |         0 |       115
      2     1 |         0 |         0 |        24 |        23 |        10 |         0 |         1 |        14 |        19 |         0 |         0 |        94
      3     2 |         0 |         0 |        24 |        23 |        10 |         0 |         1 |        15 |        19 |         0 |         0 |        94
      3     3 |         0 |         0 |        23 |        22 |        10 |         0 |         1 |        14 |        19 |         0 |         0 |        93
    ---------------------------------------------------------------------------------------------------------------------------------------------------------
          avg |         0 |         4 |        24 |        23 |        10 |         0 |         1 |        15 |        19 |         0 |         0 |        99
          max |         1 |        19 |        26 |        25 |        11 |         1 |         7 |        17 |        30 |         1 |         1 |       122

  --- [AIV0] WAVE-by-WAVE  (us, avg across active cores) ---
    exp  wave | entr->g1B |      wCub |     epiFr |     epiMd |     epiDr |    g1->ex |     TOTAL
    ---------------------------------------------------------------------------------------------
      1     0 |         0 |        41 |        10 |        22 |         0 |         0 |        75
      2     1 |         0 |        61 |        11 |        22 |         0 |         0 |        95
      3     2 |         0 |        60 |        10 |        22 |         0 |         0 |        94
      3     3 |         0 |        60 |        10 |        22 |         0 |         0 |        94
    ---------------------------------------------------------------------------------------------
          avg |         0 |        56 |        10 |        22 |         0 |         0 |        89
          max |         1 |        73 |        12 |        24 |         1 |         1 |       106

  --- [AIV1] WAVE-by-WAVE  (us, avg across active cores) ---
    exp  wave | entr->g2B |     wRowR |      cbFr |      cbMd |     cbEnd |    g2->ex |     TOTAL
    ---------------------------------------------------------------------------------------------
      1     0 |         0 |        64 |        18 |        23 |         0 |         0 |       106
      2     1 |         0 |        37 |        18 |        19 |         0 |         0 |        75
      3     2 |         0 |        40 |        16 |        17 |         0 |         0 |        74
      3     3 |         0 |        59 |        16 |        14 |         0 |         0 |        90
    ---------------------------------------------------------------------------------------------
          avg |         0 |        50 |        17 |        18 |         0 |         0 |        86
          max |         1 |        86 |        31 |        31 |         1 |         1 |       118
          
          
          
          
          
          
          
          
          
          
          
          ========== [rank 4] MegaMoe TIMESTAMPS (unit = us) ==========
  active AIC : 32/36, active AIV0: 32/36, active AIV1: 32/36

  --- [AIC] STAGE durations (per-core delta) ---
    KERNEL_ENTRY -> ROUTING_DONE          n=  32  min=       0  avg=       0  max=       1  (us)
    ROUTING_DONE -> CROSSRANK_DONE        n=  32  min=      76  avg=      77  max=      79  (us)  <-- HOT
    CROSSRANK_DONE -> EXPERT_LOOP_END       n=  32  min=     471  avg=     473  max=     479  (us)  <-- HOT
    EXPERT_LOOP_END -> PROCESS_END           n=  32  min=       0  avg=       0  max=       1  (us)

  --- [AIV0] STAGE durations (per-core delta) ---
    KERNEL_ENTRY -> ROUTING_DONE          n=  32  min=      68  avg=      68  max=      69  (us)  <-- HOT
    ROUTING_DONE -> CROSSRANK_DONE        n=  32  min=       9  avg=       9  max=       9  (us)  <-- HOT
    CROSSRANK_DONE -> EXPERT_LOOP_END       n=  32  min=     428  avg=     429  max=     431  (us)  <-- HOT
    EXPERT_LOOP_END -> PROCESS_END           n=  32  min=      63  avg=      64  max=      66  (us)  <-- HOT

  --- [AIV1] STAGE durations (per-core delta) ---
    KERNEL_ENTRY -> ROUTING_DONE          n=  32  min=      68  avg=      68  max=      69  (us)  <-- HOT
    ROUTING_DONE -> CROSSRANK_DONE        n=  32  min=       9  avg=       9  max=       9  (us)  <-- HOT
    CROSSRANK_DONE -> EXPERT_LOOP_END       n=  32  min=     474  avg=     482  max=     493  (us)  <-- HOT
    EXPERT_LOOP_END -> PROCESS_END           n=  32  min=       1  avg=      11  max=      20  (us)  <-- HOT

  --- [AIV1] per-expert DISPATCH duration ---
    expert[0] dispatch  n=  32  min=       5  avg=      16  max=      19  (us)  <-- HOT
    expert[1] dispatch  n=  32  min=       4  avg=      14  max=      17  (us)  <-- HOT
    expert[2] dispatch  n=  32  min=       6  avg=      16  max=      20  (us)  <-- HOT
    expert[3] dispatch  n=  32  min=       7  avg=      16  max=      23  (us)  <-- HOT

  --- [AIC] WAVE-by-WAVE  (us, avg across active cores) ---
    exp  wave | entr->g1B |      wDsp |    g1Frst |     g1Mid |     g1Drn |    g12gap |      wSwi |    g2Frst |     g2Mid |     g2Drn |    g2->ex |     TOTAL
    ---------------------------------------------------------------------------------------------------------------------------------------------------------
      1     0 |         0 |        18 |        17 |        46 |         5 |         0 |         1 |        10 |        32 |         0 |         0 |       132
      2     1 |         0 |         0 |        16 |        46 |         5 |         0 |         1 |        10 |        33 |         0 |         0 |       113
      3     2 |         0 |         0 |        16 |        46 |         5 |         0 |         2 |        10 |        31 |         0 |         0 |       114
      3     3 |         0 |         0 |        16 |        45 |         5 |         0 |         2 |         9 |        33 |         0 |         0 |       112
    ---------------------------------------------------------------------------------------------------------------------------------------------------------
          avg |         0 |         4 |        16 |        46 |         5 |         0 |         1 |        10 |        32 |         0 |         0 |       118
          max |         1 |        19 |        19 |        48 |         6 |         1 |         4 |        12 |        39 |         1 |         1 |       137

  --- [AIV0] WAVE-by-WAVE  (us, avg across active cores) ---
    exp  wave | entr->g1B |      wCub |     epiFr |     epiMd |     epiDr |    g1->ex |     TOTAL
    ---------------------------------------------------------------------------------------------
      1     0 |         0 |        32 |         6 |        44 |         0 |         0 |        85
      2     1 |         0 |        61 |         5 |        45 |         0 |         0 |       113
      3     2 |         0 |        62 |         5 |        46 |         0 |         0 |       114
      3     3 |         0 |        60 |         5 |        45 |         0 |         0 |       112
    ---------------------------------------------------------------------------------------------
          avg |         0 |        54 |         5 |        45 |         0 |         0 |       106
          max |         1 |        64 |         7 |        49 |         1 |         1 |       116

  --- [AIV1] WAVE-by-WAVE  (us, avg across active cores) ---
    exp  wave | entr->g2B |     wRowR |      cbFr |      cbMd |     cbEnd |    g2->ex |     TOTAL
    ---------------------------------------------------------------------------------------------
      1     0 |         0 |        68 |         8 |        31 |         0 |         0 |       109
      2     1 |         0 |        56 |         7 |        34 |         0 |         0 |        98
      3     2 |         0 |        56 |         6 |        32 |         0 |         0 |        96
      3     3 |         0 |        71 |         8 |        33 |         0 |         0 |       112
    ---------------------------------------------------------------------------------------------
          avg |         0 |        63 |         7 |        32 |         0 |         0 |       104
          max |         1 |        91 |        14 |        44 |         1 |         1 |       124

  --- LEGEND  AIC phases ---
    legend:
      entr->g1B  =  enter -> GMM1 call
           wDsp  =  GMM1 wait dispatch flag
         g1Frst  =  GMM1 first tile mmad
          g1Mid  =  GMM1 mid+last tiles
          g1Drn  =  GMM1 end-drain (EndSync)
         g12gap  =  between GMM1 & GMM2 (mega_moe.h gap)
           wSwi  =  GMM2 wait swiglu flag
         g2Frst  =  GMM2 first tile mmad
          g2Mid  =  GMM2 mid+last tiles
          g2Drn  =  GMM2 end-drain
         g2->ex  =  GMM2 end -> wave end
          TOTAL  =  TOTAL wave

  --- LEGEND  AIV0 phases ---
    legend:
      entr->g1B  =  enter -> GMM1 call
           wCub  =  epi wait cube
          epiFr  =  epi first tile
          epiMd  =  epi mid+last tiles
          epiDr  =  epi end-drain
         g1->ex  =  GMM1 end -> wave end (sub0 idle)
          TOTAL  =  TOTAL wave

  --- LEGEND  AIV1 phases ---
    legend:
      entr->g2B  =  enter -> GMM2 call (sub1 idle)
          wRowR  =  combine wait rowReady
           cbFr  =  combine first tile
           cbMd  =  combine mid+last tiles
          cbEnd  =  combine end -> GMM2 end
         g2->ex  =  GMM2 end -> wave end
          TOTAL  =  TOTAL wave
========== [rank 4] END TIMESTAMPS ==========

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
[AIV Block 0/64] [rank=0 DISPATCH bk=0 ct=2 sub=0] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 0/64] [rank=0 SCHED] >>> ENTRY exp=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024 blockNum=32
[AIV Block 0/64] [rank=0 SCHED] <<< EXIT exp=0
[AIV Block 0/64] [rank=0 DISPATCH bk=0 ct=2 sub=0] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 0/64] [rank=0 SCHED] >>> ENTRY exp=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024 blockNum=32
[AIV Block 0/64] [rank=0 SCHED] <<< EXIT exp=1
[AIV Block 0/64] [rank=0 DISPATCH bk=0 ct=2 sub=0] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 0/64] [rank=0 SCHED] >>> ENTRY exp=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024 blockNum=32
[AIV Block 0/64] [rank=0 SCHED] <<< EXIT exp=2
[AIV Block 0/64] [rank=0 DISPATCH bk=0 ct=2 sub=0] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 0/64] [rank=0 SCHED] >>> ENTRY exp=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024 blockNum=32
[AIV Block 0/64] [rank=0 SCHED] <<< EXIT exp=3
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 1/64] [rank=0 DISPATCH bk=0 ct=2 sub=1] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 1/64] [rank=0 DISPATCH bk=0 ct=2 sub=1] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 1/64] [rank=0 DISPATCH bk=0 ct=2 sub=1] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 1/64] [rank=0 DISPATCH bk=0 ct=2 sub=1] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 25/32] [rank=0 DISPATCH bk=25 ct=1 sub=0] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 25/32] [rank=0 AIC bk=25] >>> ENTRY exp=0 tnG1=12 tnG2=28
[AIC Block 25/32] [rank=0 AIC bk=25] G1#1 w=2 nT=0 waveM=256
[AIC Block 25/32] [rank=0 AIC bk=25] G1 END seen
[AIC Block 25/32] [rank=0 AIC bk=25] G2#1 w=0 nT=6 waveM=256
[AIC Block 25/32] [rank=0 AIC bk=25] G2#2 w=1 nT=9 waveM=256
[AIC Block 25/32] [rank=0 AIC bk=25] G2#3 w=2 nT=12 waveM=256
[AIC Block 25/32] [rank=0 AIC bk=25] G2 END seen
[AIC Block 25/32] [rank=0 AIC bk=25] final WaitForVector(0) drain
[AIC Block 25/32] [rank=0 AIC bk=25] <<< EXIT exp=0 (G1=1 G2=4)
[AIC Block 25/32] [rank=0 DISPATCH bk=25 ct=1 sub=0] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 25/32] [rank=0 AIC bk=25] >>> ENTRY exp=1 tnG1=12 tnG2=28
[AIC Block 25/32] [rank=0 AIC bk=25] G1#1 w=2 nT=0 waveM=256
[AIC Block 25/32] [rank=0 AIC bk=25] G1 END seen
[AIC Block 25/32] [rank=0 AIC bk=25] G2#1 w=0 nT=8 waveM=256
[AIC Block 25/32] [rank=0 AIC bk=25] G2#2 w=3 nT=15 waveM=256
[AIC Block 25/32] [rank=0 AIC bk=25] G2 END seen
[AIC Block 25/32] [rank=0 AIC bk=25] final WaitForVector(0) drain
[AIC Block 25/32] [rank=0 AIC bk=25] <<< EXIT exp=1 (G1=1 G2=2)
[AIC Block 25/32] [rank=0 DISPATCH bk=25 ct=1 sub=0] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 25/32] [rank=0 AIC bk=25] >>> ENTRY exp=2 tnG1=12 tnG2=28
[AIC Block 25/32] [rank=0 AIC bk=25] G1#1 w=2 nT=0 waveM=256
[AIC Block 25/32] [rank=0 AIC bk=25] G1 END seen
[AIC Block 25/32] [rank=0 AIC bk=25] G2#1 w=0 nT=7 waveM=256
[AIC Block 25/32] [rank=0 AIC bk=25] G2#2 w=1 nT=10 waveM=256
[AIC Block 25/32] [rank=0 AIC bk=25] G2#3 w=2 nT=13 waveM=256
[AIC Block 25/32] [rank=0 AIC bk=25] G2 END seen
[AIC Block 25/32] [rank=0 AIC bk=25] final WaitForVector(0) drain
[AIC Block 25/32] [rank=0 AIC bk=25] <<< EXIT exp=2 (G1=1 G2=4)
[AIC Block 25/32] [rank=0 DISPATCH bk=25 ct=1 sub=0] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 25/32] [rank=0 AIC bk=25] >>> ENTRY exp=3 tnG1=12 tnG2=28
[AIC Block 25/32] [rank=0 AIC bk=25] G1#1 w=2 nT=0 waveM=256
[AIC Block 25/32] [rank=0 AIC bk=25] G1 END seen
[AIC Block 25/32] [rank=0 AIC bk=25] G2#1 w=0 nT=13 waveM=256
[AIC Block 25/32] [rank=0 AIC bk=25] CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 0/32] [rank=1 DISPATCH bk=0 ct=1 sub=0] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 0/32] [rank=1 DISPATCH bk=0 ct=1 sub=0] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 0/32] [rank=1 DISPATCH bk=0 ct=1 sub=0] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 0/32] [rank=1 DISPATCH bk=0 ct=1 sub=0] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 1/32] [rank=1 AIC bk=1] *** TIMEOUT idle exp=3 g1End=0 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 2/32] [rank=1 AIC bk=2] *** TIMEOUT idle exp=3 g1End=0 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 3/32] [rank=1 AIC bk=3] *** TIMEOUT idle exp=3 g1End=0 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 4/32] [rank=1 AIC bk=4] *** TIMEOUT idle exp=3 g1End=0 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 5/32] [rank=1 AIC bk=5] *** TIMEOUT idle exp=3 g1End=0 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 6/32] [rank=1 AIC bk=6] *** TIMEOUT idle exp=3 g1End=0 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 7/32] [rank=1 AIC bk=7] *** TIMEOUT idle exp=3 g1End=0 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 8/32] [rank=1 AIC bk=8] *** TIMEOUT idle exp=3 g1End=0 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 9/32] [rank=1 AIC bk=9] *** TIMEOUT idle exp=3 g1End=0 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 10/32] [rank=1 AIC bk=10] *** TIMEOUT idle exp=3 g1End=0 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 11/32] [rank=1 AIC bk=11] *** TIMEOUT idle exp=3 g1End=0 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 12/32] [rank=1 AIC bk=12] *** TIMEOUT idle exp=3 g1End=0 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 13/32] [rank=1 AIC bk=13] *** TIMEOUT idle exp=3 g1End=0 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 14/32] [rank=1 AIC bk=14] *** TIMEOUT idle exp=3 g1End=0 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 15/32] [rank=1 AIC bk=15] *** TIMEOUT idle exp=3 g1End=0 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 0/64] [rank=1 DISPATCH bk=0 ct=2 sub=0] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 0/64] [rank=1 SCHED] >>> ENTRY exp=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024 blockNum=32
[AIV Block 0/64] [rank=1 SCHED] <<< EXIT exp=0
[AIV Block 0/64] [rank=1 DISPATCH bk=0 ct=2 sub=0] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 0/64] [rank=1 SCHED] >>> ENTRY exp=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024 blockNum=32
[AIV Block 0/64] [rank=1 SCHED] <<< EXIT exp=1
[AIV Block 0/64] [rank=1 DISPATCH bk=0 ct=2 sub=0] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 0/64] [rank=1 SCHED] >>> ENTRY exp=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024 blockNum=32
[AIV Block 0/64] [rank=1 SCHED] TIMEOUT main loop exp=2
[AIV Block 0/64]   wave=0 flag=12 d1=12 d2=28
[AIV Block 0/64]   wave=1 flag=12 d1=12 d2=28
[AIV Block 0/64]   wave=2 flag=12 d1=12 d2=28
[AIV Block 0/64]   wave=3 flag=12 d1=12 d2=28
[AIV Block 0/64]   bk=1 slot_g1aic=0 _aiv=0 slot_g2aic=0 _aiv=0 e1s=1 e1a=1 e2s=1 e2a=1
[AIV Block 0/64]   bk=2 slot_g1aic=0 _aiv=0 slot_g2aic=0 _aiv=0 e1s=1 e1a=1 e2s=1 e2a=1
[AIV Block 0/64]   bk=3 slot_g1aic=0 _aiv=0 slot_g2aic=0 _aiv=0 e1s=1 e1a=1 e2s=1 e2a=1
[AIV Block 0/64]   bk=4 slot_g1aic=0 _aiv=0 slot_g2aic=0 _aiv=0 e1s=1 e1a=1 e2s=1 e2a=1
[AIV Block 0/64]   bk=5 slot_g1aic=0 _aiv=0 slot_g2aic=0 _aiv=0 e1s=1 e1a=1 e2s=1 e2a=1
[AIV Block 0/64]   bk=6 slot_g1aic=0 _aiv=0 slot_g2aic=0 _aiv=0 e1s=1 e1a=1 e2s=1 e2a=1
[AIV Block 0/64]   bk=7 slot_g1aic=0 _aiv=0 slot_g2aic=0 _aiv=0 e1s=1 e1a=1 e2s=1 e2a=1
[AIV Block 0/64] [rank=1 SCHED] <<< EXIT exp=2
[AIV Block 0/64] [rank=1 DISPATCH bk=0 ct=2 sub=0] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 0/64] [rank=1 SCHED] >>> ENTRY exp=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024 blockNum=32
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 1/64] [rank=1 DISPATCH bk=0 ct=2 sub=1] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 1/64] [rank=1 DISPATCH bk=0 ct=2 sub=1] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 1/64] [rank=1 DISPATCH bk=0 ct=2 sub=1] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 1/64] [rank=1 DISPATCH bk=0 ct=2 sub=1] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 2/64] [rank=1 AIV0 bk=1] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 3/64] [rank=1 AIV1 bk=1] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 4/64] [rank=1 AIV0 bk=2] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 5/64] [rank=1 AIV1 bk=2] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 6/64] [rank=1 AIV0 bk=3] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 7/64] [rank=1 AIV1 bk=3] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 8/64] [rank=1 AIV0 bk=4] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 9/64] [rank=1 AIV1 bk=4] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 10/64] [rank=1 AIV0 bk=5] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 11/64] [rank=1 AIV1 bk=5] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 12/64] [rank=1 AIV0 bk=6] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 13/64] [rank=1 AIV1 bk=6] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 14/64] [rank=1 AIV0 bk=7] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 15/64] [rank=1 AIV1 bk=7] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 16/64] [rank=1 AIV0 bk=8] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 17/64] [rank=1 AIV1 bk=8] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 18/64] [rank=1 AIV0 bk=9] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 19/64] [rank=1 AIV1 bk=9] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 20/64] [rank=1 AIV0 bk=10] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 21/64] [rank=1 AIV1 bk=10] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 22/64] [rank=1 AIV0 bk=11] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 23/64] [rank=1 AIV1 bk=11] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 24/64] [rank=1 AIV0 bk=12] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 25/64] [rank=1 AIV1 bk=12] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 26/64] [rank=1 AIV0 bk=13] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 28/64] [rank=1 AIV0 bk=14] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 29/64] [rank=1 AIV1 bk=14] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 30/64] [rank=1 AIV0 bk=15] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 31/64] [rank=1 AIV1 bk=15] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 16/32] [rank=1 AIC bk=16] *** TIMEOUT idle exp=3 g1End=0 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 17/32] [rank=1 AIC bk=17] *** TIMEOUT idle exp=3 g1End=0 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 18/32] [rank=1 AIC bk=18] *** TIMEOUT idle exp=3 g1End=0 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 19/32] [rank=1 AIC bk=19] *** TIMEOUT idle exp=3 g1End=0 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 20/32] [rank=1 AIC bk=20] *** TIMEOUT idle exp=3 g1End=0 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 21/32] [rank=1 AIC bk=21] *** TIMEOUT idle exp=3 g1End=0 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 22/32] [rank=1 AIC bk=22] *** TIMEOUT idle exp=3 g1End=0 g2End=0 g1=0 g2=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 23/32] [rank=1 AIC bk=23] *** TIMEOUT idle exp=3 g1End=0 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 24/32] [rank=1 AIC bk=24] *** TIMEOUT idle exp=3 g1End=0 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 25/32] [rank=1 DISPATCH bk=25 ct=1 sub=0] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 25/32] [rank=1 AIC bk=25] >>> ENTRY exp=0 tnG1=12 tnG2=28
[AIC Block 25/32] [rank=1 AIC bk=25] G1#1 w=2 nT=0 waveM=256
[AIC Block 25/32] [rank=1 AIC bk=25] G1 END seen
[AIC Block 25/32] [rank=1 AIC bk=25] G2#1 w=0 nT=6 waveM=256
[AIC Block 25/32] [rank=1 AIC bk=25] G2#2 w=1 nT=9 waveM=256
[AIC Block 25/32] [rank=1 AIC bk=25] G2#3 w=2 nT=12 waveM=256
[AIC Block 25/32] [rank=1 AIC bk=25] G2 END seen
[AIC Block 25/32] [rank=1 AIC bk=25] final WaitForVector(0) drain
[AIC Block 25/32] [rank=1 AIC bk=25] <<< EXIT exp=0 (G1=1 G2=4)
[AIC Block 25/32] [rank=1 DISPATCH bk=25 ct=1 sub=0] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 25/32] [rank=1 AIC bk=25] >>> ENTRY exp=1 tnG1=12 tnG2=28
[AIC Block 25/32] [rank=1 AIC bk=25] G1#1 w=2 nT=0 waveM=256
[AIC Block 25/32] [rank=1 AIC bk=25] G1 END seen
[AIC Block 25/32] [rank=1 AIC bk=25] G2#1 w=0 nT=3 waveM=256
[AIC Block 25/32] [rank=1 AIC bk=25] G2#2 w=1 nT=6 waveM=256
[AIC Block 25/32] [rank=1 AIC bk=25] G2#3 w=1 nT=25 waveM=256
[AIC Block 25/32] [rank=1 AIC bk=25] G2 END seen
[AIC Block 25/32] [rank=1 AIC bk=25] final WaitForVector(0) drain
[AIC Block 25/32] [rank=1 AIC bk=25] <<< EXIT exp=1 (G1=1 G2=4)
[AIC Block 25/32] [rank=1 DISPATCH bk=25 ct=1 sub=0] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 25/32] [rank=1 AIC bk=25] >>> ENTRY exp=2 tnG1=12 tnG2=28
[AIC Block 25/32] [rank=1 AIC bk=25] G1#1 w=2 nT=0 waveM=256
[AIC Block 25/32] [rank=1 AIC bk=25] G1 END seen
[AIC Block 25/32] [rank=1 AIC bk=25] G2#1 w=0 nT=11 waveM=256
[AIC Block 25/32] [rank=1 AIC bk=25] G2#2 w=1 nT=14 waveM=256
[AIC Block 25/32] [rank=1 AIC bk=25] G2#3 w=2 nT=17 waveM=256
[AIC Block 25/32] [rank=1 AIC bk=25] G2 END seen
[AIC Block 25/32] [rank=1 AIC bk=25] final WaitForVector(0) drain
[AIC Block 25/32] [rank=1 AIC bk=25] <<< EXIT exp=2 (G1=1 G2=3)
[AIC Block 25/32] [rank=1 DISPATCH bk=25 ct=1 sub=0] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 25/32] [rank=1 AIC bk=25] >>> ENTRY exp=3 tnG1=12 tnG2=28
[AIC Block 25/32] [rank=1 AIC bk=25] G1#1 w=1 nT=11 waveM=256
[AIC Block 25/32] [rank=1 AIC bk=25] *** TIMEOUT idle exp=3 g1End=0 g2End=0 g1=0 g2=0 g1Pending=1
[AIC Block 25/32] [rank=1 AIC bk=25] final WaitForVector(0) drain
[AIC Block 25/32] [rank=1 AIC bk=25] <<< EXIT exp=3 (G1=1 G2=0)
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 26/32] [rank=1 AIC bk=26] *** TIMEOUT idle exp=3 g1End=0 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 27/32] [rank=1 AIC bk=27] *** TIMEOUT idle exp=3 g1End=0 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 28/32] [rank=1 AIC bk=28] *** TIMEOUT idle exp=3 g1End=0 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 29/32] [rank=1 AIC bk=29] *** TIMEOUT idle exp=3 g1End=0 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 30/32] [rank=1 AIC bk=30] *** TIMEOUT idle exp=3 g1End=0 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 31/32] [rank=1 AIC bk=31] *** TIMEOUT idle exp=3 g1End=0 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 32/64] [rank=1 AIV0 bk=16] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 33/64] [rank=1 AIV1 bk=16] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 34/64] [rank=1 AIV0 bk=17] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 35/64] [rank=1 AIV1 bk=17] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 36/64] [rank=1 AIV0 bk=18] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 37/64] [rank=1 AIV1 bk=18] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 38/64] [rank=1 AIV0 bk=19] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 39/64] [rank=1 AIV1 bk=19] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 40/64] [rank=1 AIV0 bk=20] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 41/64] [rank=1 AIV1 bk=20] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 42/64] [rank=1 AIV0 bk=21] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 43/64] [rank=1 AIV1 bk=21] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 45/64] [rank=1 AIV1 bk=22] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 46/64] [rank=1 AIV0 bk=23] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 47/64] [rank=1 AIV1 bk=23] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 48/64] [rank=1 AIV0 bk=24] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 49/64] [rank=1 AIV1 bk=24] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 50/64] [rank=1 DISPATCH bk=25 ct=2 sub=0] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 50/64] [rank=1 AIV0 bk=25] >>> ENTRY exp=0
[AIV Block 50/64] [rank=1 AIV0 bk=25] #1 w=2 nT=0
[AIV Block 50/64] [rank=1 AIV0 bk=25] <<< EXIT (tasks=1)
[AIV Block 50/64] [rank=1 DISPATCH bk=25 ct=2 sub=0] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 50/64] [rank=1 AIV0 bk=25] >>> ENTRY exp=1
[AIV Block 50/64] [rank=1 AIV0 bk=25] #1 w=2 nT=0
[AIV Block 50/64] [rank=1 AIV0 bk=25] <<< EXIT (tasks=1)
[AIV Block 50/64] [rank=1 DISPATCH bk=25 ct=2 sub=0] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 50/64] [rank=1 AIV0 bk=25] >>> ENTRY exp=2
[AIV Block 50/64] [rank=1 AIV0 bk=25] #1 w=2 nT=0
[AIV Block 50/64] [rank=1 AIV0 bk=25] <<< EXIT (tasks=1)
[AIV Block 50/64] [rank=1 DISPATCH bk=25 ct=2 sub=0] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 50/64] [rank=1 AIV0 bk=25] >>> ENTRY exp=3
[AIV Block 50/64] [rank=1 AIV0 bk=25] *** TIMEOUT spin slot exp=3 seq=1
[AIV Block 50/64] [rank=1 AIV0 bk=25] <<< EXIT (tasks=0)
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 51/64] [rank=1 DISPATCH bk=25 ct=2 sub=1] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 51/64] [rank=1 AIV1 bk=25] >>> ENTRY exp=0
[AIV Block 51/64] [rank=1 AIV1 bk=25] #1 w=0 nT=6
[AIV Block 51/64] [rank=1 AIV1 bk=25] #2 w=1 nT=9
[AIV Block 51/64] [rank=1 AIV1 bk=25] #3 w=2 nT=12
[AIV Block 51/64] [rank=1 AIV1 bk=25] <<< EXIT (tasks=4)
[AIV Block 51/64] [rank=1 DISPATCH bk=25 ct=2 sub=1] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 51/64] [rank=1 AIV1 bk=25] >>> ENTRY exp=1
[AIV Block 51/64] [rank=1 AIV1 bk=25] #1 w=0 nT=3
[AIV Block 51/64] [rank=1 AIV1 bk=25] #2 w=1 nT=6
[AIV Block 51/64] [rank=1 AIV1 bk=25] #3 w=1 nT=25
[AIV Block 51/64] [rank=1 AIV1 bk=25] <<< EXIT (tasks=4)
[AIV Block 51/64] [rank=1 DISPATCH bk=25 ct=2 sub=1] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 51/64] [rank=1 AIV1 bk=25] >>> ENTRY exp=2
[AIV Block 51/64] [rank=1 AIV1 bk=25] #1 w=0 nT=11
[AIV Block 51/64] [rank=1 AIV1 bk=25] #2 w=1 nT=14
[AIV Block 51/64] [rank=1 AIV1 bk=25] #3 w=2 nT=17
[AIV Block===============================================================
cur run case: MOE_Dispatch_FFN_Combine_RedLine046, bs: 512, h: 7168, k: 8, local_moe: 4
dispatch_quant_out_type: 24, quant_dtype: torch.float8_e4m3fn, quant_mode: 4
===============================================================
current device set: 0
[INFO] device_0 创建HCCL通信链路
device_0 init_process_group success
device 0 初始化EP域
device 0 初始化TP域
jjjjjjjjjjjjjjj
<npu_ops_transformer.ops.mega_moe.SymmBuffer object at 0x7f4028550a30>
rank:0 topk_idstensor([[0, 1, 2,  ..., 5, 6, 7],
        [1, 2, 3,  ..., 6, 7, 0],
        [2, 3, 4,  ..., 7, 0, 1],
        ...,
        [5, 6, 7,  ..., 2, 3, 4],
        [6, 7, 0,  ..., 3, 4, 5],
        [7, 0, 1,  ..., 4, 5, 6]], device='npu:0', dtype=torch.int32)
rank:0 构造算子输入finished!
rank 0 epid 0 tpid 0 npu finished! 

G2#2 w=1 nT=16 waveM=256
[AIC Block 25/32] [rank=0 AIC bk=25] G2#3 w=2 nT=19 waveM=256
[AIC Block 25/32] [rank=0 AIC bk=25] G2 END seen
[AIC Block 25/32] [rank=0 AIC bk=25] final WaitForVector(0) drain
[AIC Block 25/32] [rank=0 AIC bk=25] <<< EXIT exp=3 (G1=1 G2=4)
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 50/64] [rank=0 DISPATCH bk=25 ct=2 sub=0] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 50/64] [rank=0 AIV0 bk=25] >>> ENTRY exp=0
[AIV Block 50/64] [rank=0 AIV0 bk=25] #1 w=2 nT=0
[AIV Block 50/64] [rank=0 AIV0 bk=25] <<< EXIT (tasks=1)
[AIV Block 50/64] [rank=0 DISPATCH bk=25 ct=2 sub=0] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 50/64] [rank=0 AIV0 bk=25] >>> ENTRY exp=1
[AIV Block 50/64] [rank=0 AIV0 bk=25] #1 w=2 nT=0
[AIV Block 50/64] [rank=0 AIV0 bk=25] <<< EXIT (tasks=1)
[AIV Block 50/64] [rank=0 DISPATCH bk=25 ct=2 sub=0] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 50/64] [rank=0 AIV0 bk=25] >>> ENTRY exp=2
[AIV Block 50/64] [rank=0 AIV0 bk=25] #1 w=2 nT=0
[AIV Block 50/64] [rank=0 AIV0 bk=25] <<< EXIT (tasks=1)
[AIV Block 50/64] [rank=0 DISPATCH bk=25 ct=2 sub=0] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 50/64] [rank=0 AIV0 bk=25] >>> ENTRY exp=3
[AIV Block 50/64] [rank=0 AIV0 bk=25] #1 w=2 nT=0
[AIV Block 50/64] [rank=0 AIV0 bk=25] <<< EXIT (tasks=1)
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 51/64] [rank=0 DISPATCH bk=25 ct=2 sub=1] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 51/64] [rank=0 AIV1 bk=25] >>> ENTRY exp=0
[AIV Block 51/64] [rank=0 AIV1 bk=25] #1 w=0 nT=6
[AIV Block 51/64] [rank=0 AIV1 bk=25] #2 w=1 nT=9
[AIV Block 51/64] [rank=0 AIV1 bk=25] #3 w=2 nT=12
[AIV Block 51/64] [rank=0 AIV1 bk=25] <<< EXIT (tasks=4)
[AIV Block 51/64] [rank=0 DISPATCH bk=25 ct=2 sub=1] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 51/64] [rank=0 AIV1 bk=25] >>> ENTRY exp=1
[AIV Block 51/64] [rank=0 AIV1 bk=25] #1 w=0 nT=8
[AIV Block 51/64] [rank=0 AIV1 bk=25] #2 w=3 nT=15
[AIV Block 51/64] [rank=0 AIV1 bk=25] <<< EXIT (tasks=2)
[AIV Block 51/64] [rank=0 DISPATCH bk=25 ct=2 sub=1] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 51/64] [rank=0 AIV1 bk=25] >>> ENTRY exp=2
[AIV Block 51/64] [rank=0 AIV1 bk=25] #1 w=0 nT=7
[AIV Block 51/64] [rank=0 AIV1 bk=25] #2 w=1 nT=10
[AIV Block 51/64] [rank=0 AIV1 bk=25] #3 w=2 nT=13
[AIV Block 51/64] [rank=0 AIV1 bk=25] <<< EXIT (tasks=4)
[AIV Block 51/64] [rank=0 DISPATCH bk=25 ct=2 sub=1] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 51/64] [rank=0 AIV1 bk=25] >>> ENTRY exp=3
[AIV Block 51/64] [rank=0 AIV1 bk=25] #1 w=0 nT=13
[AIV Block 51/64] [rank=0 AIV1 bk=25] #2 w=1 nT=16
[AIV Block 51/64] [rank=0 AIV1 bk=25] #3 w=2 nT=19
[AIV Block 51/64] [rank=0 AIV1 bk=25] <<< EXIT (tasks=4)
===============================================================
cur run case: MOE_Dispatch_FFN_Combine_RedLine046, bs: 512, h: 7168, k: 8, local_moe: 4
dispatch_quant_out_type: 24, quant_dtype: torch.float8_e4m3fn, quant_mode: 4
===============================================================
current device set: 1
[INFO] device_1 创建HCCL通信链路
device_1 init_process_group success
device 1 初始化EP域
device 1 初始化TP域
jjjjjjjjjjjjjjj
<npu_ops_transformer.ops.mega_moe.SymmBuffer object at 0x7f4c19bc4a30>
rank:1 topk_idstensor([[0, 1, 2,  ..., 5, 6, 7],
        [1, 2, 3,  ..., 6, 7, 0],
        [2, 3, 4,  ..., 7, 0, 1],
        ...,
        [5, 6, 7,  ..., 2, 3, 4],
        [6, 7, 0,  ..., 3, 4, 5],
        [7, 0, 1,  ..., 4, 5, 6]], device='npu:1', dtype=torch.int32)
rank:1 构造算子输入finished!
 51/64] [rank=1 AIV1 bk=25] <<< EXIT (tasks=3)
[AIV Block 51/64] [rank=1 DISPATCH bk=25 ct=2 sub=1] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 51/64] [rank=1 AIV1 bk=25] >>> ENTRY exp=3
[AIV Block 51/64] [rank=1 AIV1 bk=25] *** TIMEOUT spin slot exp=3 seq=1
[AIV Block 51/64] [rank=1 AIV1 bk=25] <<< EXIT (tasks=0)
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 52/64] [rank=1 AIV0 bk=26] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 53/64] [rank=1 AIV1 bk=26] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 54/64] [rank=1 AIV0 bk=27] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 55/64] [rank=1 AIV1 bk=27] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 56/64] [rank=1 AIV0 bk=28] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 57/64] [rank=1 AIV1 bk=28] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 58/64] [rank=1 AIV0 bk=29] *** TIMEOUT spin slot exp=3 seq=1

CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 59/64] [rank=1 AIV1 bk=29] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 60/64] [rank=1 AIV0 bk=30] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 61/64] [rank=1 AIV1 bk=30] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 62/64] [rank=1 AIV0 bk=31] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 63/64] [rank=1 AIV1 bk=31] *** TIMEOUT spin slot exp=3 seq=1
[AIV Block 0/64] [rank=1 SCHED] TIMEOUT main loop exp=3
[AIV Block 0/64]   wave=0 flag=0 d1=12 d2=0
[AIV Block 0/64]   wave=1 flag=0 d1=12 d2=0
[AIV Block 0/64]   wave=2 flag=0 d1=6 d2=0
[AIV Block 0/64]   wave=3 flag=0 d1=0 d2=0
[AIV Block 0/64]   bk=1 slot_g1aic=0 _aiv=536870912 slot_g2aic=0 _aiv=0 e1s=0 e1a=0 e2s=0 e2a=0
[AIV Block 0/64]   bk=2 slot_g1aic=0 _aiv=536871936 slot_g2aic=0 _aiv=0 e1s=0 e1a=0 e2s=0 e2a=0
[AIV Block 0/64]   bk=3 slot_g1aic=0 _aiv=536872960 slot_g2aic=0 _aiv=0 e1s=0 e1a=0 e2s=0 e2a=0
[AIV Block 0/64]   bk=4 slot_g1aic=0 _aiv=536873984 slot_g2aic=0 _aiv=0 e1s=0 e1a=0 e2s=0 e2a=0
[AIV Block 0/64]   bk=5 slot_g1aic=0 _aiv=536875008 slot_g2aic=0 _aiv=0 e1s=0 e1a=0 e2s=0 e2a=0
[AIV Block 0/64]   bk=6 slot_g1aic=0 _aiv=536876032 slot_g2aic=0 _aiv=0 e1s=0 e1a=0 e2s=0 e2a=0
[AIV Block 0/64]   bk=7 slot_g1aic=0 _aiv=536877056 slot_g2aic=0 _aiv=0 e1s=0 e1a=0 e2s=0 e2a=0
[AIV Block 0/64] [rank=1 SCHED] <<< EXIT exp=3
