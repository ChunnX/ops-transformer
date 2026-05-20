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
<npu_ops_transformer.ops.mega_moe.SymmBuffer object at 0x7fd6b38eca30>
rank:1 topk_idstensor([[0, 1, 2,  ..., 5, 6, 7],
        [1, 2, 3,  ..., 6, 7, 0],
        [2, 3, 4,  ..., 7, 0, 1],
        ...,
        [5, 6, 7,  ..., 2, 3, 4],
        [6, 7, 0,  ..., 3, 4, 5],
        [7, 0, 1,  ..., 4, 5, 6]], device='npu:1', dtype=torch.int32)
rank:1 构造算子输入finished!
rank 1 epid 1 tpid 0 npu finished! 

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
<npu_ops_transformer.ops.mega_moe.SymmBuffer object at 0x7f80c17eca30>
rank:0 topk_idstensor([[0, 1, 2,  ..., 5, 6, 7],
        [1, 2, 3,  ..., 6, 7, 0],
        [2, 3, 4,  ..., 7, 0, 1],
        ...,
        [5, 6, 7,  ..., 2, 3, 4],
        [6, 7, 0,  ..., 3, 4, 5],
        [7, 0, 1,  ..., 4, 5, 6]], device='npu:0', dtype=torch.int32)
rank:0 构造算子输入finished!
rank 0 epid 0 tpid 0 npu finished! 

CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 0/32] [rank=0 DISPATCH bk=0 ct=1 sub=0] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 0/32] [rank=0 AIC bk=0] >>> DynamicAicLoop ENTRY exp=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 0/32] [rank=0 AIC bk=0] dispatch flag DONE exp=0
[AIC Block 0/32] [rank=0 AIC bk=0] iter#1 CLAIM op=0 wave=2 nTile=1 waveM=256
[AIC Block 0/32] [rank=0 AIC bk=0] iter#1 DONE (op=0)
[AIC Block 0/32] [rank=0 AIC bk=0] iter#2 CLAIM op=1 wave=0 nTile=14 waveM=256
[AIC Block 0/32] [rank=0 AIC bk=0] iter#2 DONE (op=1)
[AIC Block 0/32] [rank=0 AIC bk=0] iter#3 CLAIM op=1 wave=2 nTile=25 waveM=256
[AIC Block 0/32] [rank=0 AIC bk=0] iter#3 DONE (op=1)
[AIC Block 0/32] [rank=0 AIC bk=0] main loop DONE (3 tasks), sending END
[AIC Block 0/32] [rank=0 AIC bk=0] G1 final drain WaitForVector(0)...
[AIC Block 0/32] [rank=0 AIC bk=0] write GMM1 END
[AIC Block 0/32] [rank=0 AIC bk=0] G2 final drain (spin slot==0)...
[AIC Block 0/32] [rank=0 AIC bk=0] write GMM2 END
[AIC Block 0/32] [rank=0 AIC bk=0] <<< DynamicAicLoop EXIT exp=0
[AIC Block 0/32] [rank=0 DISPATCH bk=0 ct=1 sub=0] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 0/32] [rank=0 AIC bk=0] >>> DynamicAicLoop ENTRY exp=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 0/32] [rank=0 AIC bk=0] dispatch flag DONE exp=1
[AIC Block 0/32] [rank=0 AIC bk=0] iter#1 CLAIM op=1 wave=0 nTile=14 waveM=256
[AIC Block 0/32] [rank=0 AIC bk=0] iter#1 DONE (op=1)
[AIC Block 0/32] [rank=0 AIC bk=0] iter#2 CLAIM op=1 wave=2 nTile=15 waveM=256
[AIC Block 0/32] [rank=0 AIC bk=0] iter#2 DONE (op=1)
[AIC Block 0/32] [rank=0 AIC bk=0] iter#3 CLAIM op=1 wave=1 nTile=11 waveM=256
[AIC Block 0/32] [rank=0 AIC bk=0] iter#3 DONE (op=1)
[AIC Block 0/32] [rank=0 AIC bk=0] iter#4 CLAIM op=1 wave=3 nTile=19 waveM=256
[AIC Block 0/32] [rank=0 AIC bk=0] iter#4 DONE (op=1)
[AIC Block 0/32] [rank=0 AIC bk=0] main loop DONE (4 tasks), sending END
[AIC Block 0/32] [rank=0 AIC bk=0] write GMM1 END
[AIC Block 0/32] [rank=0 AIC bk=0] G2 final drain (spin slot==0)...
[AIC Block 0/32] [rank=0 AIC bk=0] write GMM2 END
[AIC Block 0/32] [rank=0 AIC bk=0] <<< DynamicAicLoop EXIT exp=1
[AIC Block 0/32] [rank=0 DISPATCH bk=0 ct=1 sub=0] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 0/32] [rank=0 AIC bk=0] >>> DynamicAicLoop ENTRY exp=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 0/32] [rank=0 AIC bk=0] dispatch flag DONE exp=2
[AIC Block 0/32] [rank=0 AIC bk=0] iter#1 CLAIM op=1 wave=0 nTile=0 waveM=256
[AIC Block 0/32] [rank=0 AIC bk=0] iter#1 DONE (op=1)
[AIC Block 0/32] [rank=0 AIC bk=0] iter#2 CLAIM op=1 wave=1 nTile=11 waveM=256
[AIC Block 0/32] [rank=0 AIC bk=0] iter#2 DONE (op=1)
[AIC Block 0/32] [rank=0 AIC bk=0] iter#3 CLAIM op=1 wave=2 nTile=16 waveM=256
[AIC Block 0/32] [rank=0 AIC bk=0] iter#3 DONE (op=1)
[AIC Block 0/32] [rank=0 AIC bk=0] iter#4 CLAIM op=1 wave=3 nTile=9 waveM=256
[AIC Block 0/32] [rank=0 AIC bk=0] iter#4 DONE (op=1)
[AIC Block 0/32] [rank=0 AIC bk=0] main loop DONE (4 tasks), sending END
[AIC Block 0/32] [rank=0 AIC bk=0] write GMM1 END
[AIC Block 0/32] [rank=0 AIC bk=0] G2 final drain (spin slot==0)...
[AIC Block 0/32] [rank=0 AIC bk=0] write GMM2 END
[AIC Block 0/32] [rank=0 AIC bk=0] <<< DynamicAicLoop EXIT exp=2
[AIC Block 0/32] [rank=0 DISPATCH bk=0 ct=1 sub=0] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 0/32] [rank=0 AIC bk=0] >>> DynamicAicLoop ENTRY exp=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 0/32] [rank=0 AIC bk=0] dispatch flag DONE exp=3
[AIC Block 0/32] [rank=0 AIC bk=0] iter#1 CLAIM op=1 wave=1 nTile=1 waveM=256
[AIC Block 0/32] [rank=0 AIC bk=0] iter#1 DONE (op=1)
[AIC Block 0/32] [rank=0 AIC bk=0] iter#2 CLAIM op=1 wave=2 nTile=27 waveM=256
[AIC Block 0/32] [rank=0 AIC bk=0] iter#2 DONE (op=1)
[AIC Block 0/32] [rank=0 AIC bk=0] main loop DONE (2 tasks), sending END
[AIC Block 0/32] [rank=0 AIC bk=0] write GMM1 END
[AIC Block 0/32] [rank=0 AIC bk=0] G2 final drain (spin slot==0)...
[AIC Block 0/32] [rank=0 AIC bk=0] write GMM2 END
[AIC Block 0/32] [rank=0 AIC bk=0] <<< DynamicAicLoop EXIT exp=3
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 0/64] [rank=0 DISPATCH bk=0 ct=2 sub=0] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 0/64] [rank=0 AIV0 bk=0] >>> DynamicAiv0Loop ENTRY exp=0 tnG1=12
[AIV Block 0/64] [rank=0 AIV0 bk=0] iter#1 SLOT READ packed=536871944 op=0 wave=2 nTile=1 waveM=256
[AIV Block 0/64] [rank=0 AIV0 bk=0] iter#1 DONE
[AIV Block 0/64] [rank=0 AIV0 bk=0] iter#2 SLOT READ packed=2 op=2 wave=0 nTile=0 waveM=0
[AIV Block 0/64] [rank=0 AIV0 bk=0] <<< DynamicAiv0Loop EXIT
[AIV Block 0/64] [rank=0 DISPATCH bk=0 ct=2 sub=0] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 0/64] [rank=0 AIV0 bk=0] >>> DynamicAiv0Loop ENTRY exp=1 tnG1=12
[AIV Block 0/64] [rank=0 AIV0 bk=0] iter#1 SLOT READ packed=2 op=2 wave=0 nTile=0 waveM=0
[AIV Block 0/64] [rank=0 AIV0 bk=0] <<< DynamicAiv0Loop EXIT
[AIV Block 0/64] [rank=0 DISPATCH bk=0 ct=2 sub=0] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 0/64] [rank=0 AIV0 bk=0] >>> DynamicAiv0Loop ENTRY exp=2 tnG1=12
[AIV Block 0/64] [rank=0 AIV0 bk=0] iter#1 SLOT READ packed=2 op=2 wave=0 nTile=0 waveM=0
[AIV Block 0/64] [rank=0 AIV0 bk=0] <<< DynamicAiv0Loop EXIT
[AIV Block 0/64] [rank=0 DISPATCH bk=0 ct=2 sub=0] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 0/64] [rank=0 AIV0 bk=0] >>> DynamicAiv0Loop ENTRY exp=3 tnG1=12
[AIV Block 0/64] [rank=0 AIV0 bk=0] iter#1 SLOT READ packed=2 op=2 wave=0 nTile=0 waveM=0
[AIV Block 0/64] [rank=0 AIV0 bk=0] <<< DynamicAiv0Loop EXIT
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 1/64] [rank=0 DISPATCH bk=0 ct=2 sub=1] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 1/64] [rank=0 AIV1 bk=0] >>> DynamicAiv1Loop ENTRY exp=0 tnG2=28
[AIV Block 1/64] [rank=0 AIV1 bk=0] iter#1 SLOT READ packed=536885249 op=1 wave=0 nTile=14 waveM=256
[AIV Block 1/64] [rank=0 AIV1 bk=0] iter#1 DONE
[AIV Block 1/64] [rank=0 AIV1 bk=0] iter#2 SLOT READ packed=536896521 op=1 wave=2 nTile=25 waveM=256
[AIV Block 1/64] [rank=0 AIV1 bk=0] iter#2 DONE
[AIV Block 1/64] [rank=0 AIV1 bk=0] iter#3 SLOT READ packed=2 op=2 wave=0 nTile=0 waveM=0
[AIV Block 1/64] [rank=0 AIV1 bk=0] <<< DynamicAiv1Loop EXIT
[AIV Block 1/64] [rank=0 DISPATCH bk=0 ct=2 sub=1] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 1/64] [rank=0 AIV1 bk=0] >>> DynamicAiv1Loop ENTRY exp=1 tnG2=28
[AIV Block 1/64] [rank=0 AIV1 bk=0] iter#1 SLOT READ packed=536885249 op=1 wave=0 nTile=14 waveM=256
[AIV Block 1/64] [rank=0 AIV1 bk=0] iter#1 DONE
[AIV Block 1/64] [rank=0 AIV1 bk=0] iter#2 SLOT READ packed=536886281 op=1 wave=2 nTile=15 waveM=256
[AIV Block 1/64] [rank=0 AIV1 bk=0] iter#2 DONE
[AIV Block 1/64] [rank=0 AIV1 bk=0] iter#3 SLOT READ packed=536882181 op=1 wave=1 nTile=11 waveM=256
[AIV Block 1/64] [rank=0 AIV1 bk=0] iter#3 DONE
[AIV Block 1/64] [rank=0 AIV1 bk=0] iter#4 SLOT READ packed=536890381 op=1 wave=3 nTile=19 waveM=256
[AIV Block 1/64] [rank=0 AIV1 bk=0] iter#4 DONE
[AIV Block 1/64] [rank=0 AIV1 bk=0] iter#5 SLOT READ packed=2 op=2 wave=0 nTile=0 waveM=0
[AIV Block 1/64] [rank=0 AIV1 bk=0] <<< DynamicAiv1Loop EXIT
[AIV Block 1/64] [rank=0 DISPATCH bk=0 ct=2 sub=1] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 1/64] [rank=0 AIV1 bk=0] >>> DynamicAiv1Loop ENTRY exp=2 tnG2=28
[AIV Block 1/64] [rank=0 AIV1 bk=0] iter#1 SLOT READ packed=536870913 op=1 wave=0 nTile=0 waveM=256
[AIV Block 1/64] [rank=0 AIV1 bk=0] iter#1 DONE
[AIV Block 1/64] [rank=0 AIV1 bk=0] iter#2 SLOT READ packed=536882181 op=1 wave=1 nTile=11 waveM=256
[AIV Block 1/64] [rank=0 AIV1 bk=0] iter#2 DONE
[AIV Block 1/64] [rank=0 AIV1 bk=0] iter#3 SLOT READ packed=536887305 op=1 wave=2 nTile=16 waveM=256
[AIV Block 1/64] [rank=0 AIV1 bk=0] iter#3 DONE
[AIV Block 1/64] [rank=0 AIV1 bk=0] iter#4 SLOT READ packed=536880141 op=1 wave=3 nTile=9 waveM=256
[AIV Block 1/64] [rank=0 AIV1 bk=0] iter#4 DONE
[AIV Block 1/64] [rank=0 AIV1 bk=0] iter#5 SLOT READ packed=2 op=2 wave=0 nTile=0 waveM=0
[AIV Block 1/64] [rank=0 AIV1 bk=0] <<< DynamicAiv1Loop EXIT
[AIV Block 1/64] [rank=0 DISPATCH bk=0 ct=2 sub=1] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 1/64] [rank=0 AIV1 bk=0] >>> DynamicAiv1Loop ENTRY exp=3 tnG2=28
[AIV Block 1/64] [rank=0 AIV1 bk=0] iter#1 SLOT READ packed=536871941 op=1 wave=1 nTile=1 waveM=256
[AIV Block 1/64] [rank=0 AIV1 bk=0] iter#1 DONE
[AIV Block 1/64] [rank=0 AIV1 bk=0] iter#2 SLOT READ packed=536898569 op=1 wave=2 nTile=27 waveM=256
[AIV Block 1/64] [rank=0 AIV1 bk=0] iter#2 DONE
[AIV Block 1/64] [rank=0 AIV1 bk=0] iter#3 SLOT READ packed=2 op=2 wave=0 nTile=0 waveM=0
[AIV Block 1/64] [rank=0 AIV1 bk=0] <<< DynamicAiv1Loop EXIT
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 25/32] [rank=0 AIC bk=25] *** TIMEOUT end-of-exp G1 END drain exp=2 slot=-2
[AIC Block 25/32] [rank=0 AIC bk=25] *** TIMEOUT end-of-exp G2 END drain exp=3 slot=2
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 30/32] [rank=0 DISPATCH bk=30 ct=1 sub=0] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 30/32] [rank=0 AIC bk=30] >>> DynamicAicLoop ENTRY exp=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 30/32] [rank=0 AIC bk=30] dispatch flag DONE exp=0
[AIC Block 30/32] [rank=0 AIC bk=30] iter#1 CLAIM op=0 wave=2 nTile=3 waveM=256
[AIC Block 30/32] [rank=0 AIC bk=30] iter#1 DONE (op=0)
[AIC Block 30/32] [rank=0 AIC bk=30] iter#2 CLAIM op=1 wave=0 nTile=15 waveM=256
[AIC Block 30/32] [rank=0 AIC bk=30] iter#2 DONE (op=1)
[AIC Block 30/32] [rank=0 AIC bk=30] iter#3 CLAIM op=1 wave=2 nTile=13 waveM=256
[AIC Block 30/32] [rank=0 AIC bk=30] iter#3 DONE (op=1)
[AIC Block 30/32] [rank=0 AIC bk=30] main loop DONE (3 tasks), sending END
[AIC Block 30/32] [rank=0 AIC bk=30] G1 final drain WaitForVector(0)...
[AIC Block 30/32] [rank=0 AIC bk=30] write GMM1 END
[AIC Block 30/32] [rank=0 AIC bk=30] G2 final drain (spin slot==0)...
[AIC Block 30/32] [rank=0 AIC bk=30] write GMM2 END
[AIC Block 30/32] [rank=0 AIC bk=30] <<< DynamicAicLoop EXIT exp=0
[AIC Block 30/32] [rank=0 DISPATCH bk=30 ct=1 sub=0] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 30/32] [rank=0 AIC bk=30] >>> DynamicAicLoop ENTRY exp=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 30/32] [rank=0 AIC bk=30] dispatch flag DONE exp=1
[AIC Block 30/32] [rank=0 AIC bk=30] iter#1 CLAIM op=1 wave=1 nTile=0 waveM=256
[AIC Block 30/32] [rank=0 AIC bk=30] iter#1 DONE (op=1)
[AIC Block 30/32] [rank=0 AIC bk=30] iter#2 CLAIM op=1 wave=1 nTile=20 waveM=256
[AIC Block 30/32] [rank=0 AIC bk=30] iter#2 DONE (op=1)
[AIC Block 30/32] [rank=0 AIC bk=30] main loop DONE (2 tasks), sending END
[AIC Block 30/32] [rank=0 AIC bk=30] write GMM1 END
[AIC Block 30/32] [rank=0 AIC bk=30] G2 final drain (spin slot==0)...
[AIC Block 30/32] [rank=0 AIC bk=30] write GMM2 END
[AIC Block 30/32] [rank=0 AIC bk=30] <<< DynamicAicLoop EXIT exp=1
[AIC Block 30/32] [rank=0 DISPATCH bk=30 ct=1 sub=0] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 30/32] [rank=0 AIC bk=30] >>> DynamicAicLoop ENTRY exp=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 30/32] [rank=0 AIC bk=30] dispatch flag DONE exp=2
[AIC Block 30/32] [rank=0 AIC bk=30] iter#1 CLAIM op=0 wave=2 nTile=6 waveM=256
[AIC Block 30/32] [rank=0 AIC bk=30] iter#1 DONE (op=0)
[AIC Block 30/32] [rank=0 AIC bk=30] iter#2 CLAIM op=1 wave=0 nTile=17 waveM=256
[AIC Block 30/32] [rank=0 AIC bk=30] iter#2 DONE (op=1)
[AIC Block 30/32] [rank=0 AIC bk=30] iter#3 CLAIM op=1 wave=1 nTile=25 waveM=256
[AIC Block 30/32] [rank=0 AIC bk=30] iter#3 DONE (op=1)
[AIC Block 30/32] [rank=0 AIC bk=30] main loop DONE (3 tasks), sending END
[AIC Block 30/32] [rank=0 AIC bk=30] G1 final drain WaitForVector(0)...
[AIC Block 30/32] [rank=0 AIC bk=30] write GMM1 END
[AIC Block 30/32] [rank=0 AIC bk=30] G2 final drain (spin slot==0)...
[AIC Block 30/32] [rank=0 AIC bk=30] write GMM2 END
[AIC Block 30/32] [rank=0 AIC bk=30] <<< DynamicAicLoop EXIT exp=2
[AIC Block 30/32] [rank=0 DISPATCH bk=30 ct=1 sub=0] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 30/32] [rank=0 AIC bk=30] >>> DynamicAicLoop ENTRY exp=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 30/32] [rank=0 AIC bk=30] dispatch flag DONE exp=3
[AIC Block 30/32] [rank=0 AIC bk=30] iter#1 CLAIM op=0 wave=2 nTile=6 waveM=256
[AIC Block 30/32] [rank=0 AIC bk=30] iter#1 DONE (op=0)
[AIC Block 30/32] [rank=0 AIC bk=30] iter#2 CLAIM op=1 wave=0 nTile=23 waveM=256
[AIC Block 30/32] [rank=0 AIC bk=30] iter#2 DONE (op=1)
[AIC Block 30/32] [rank=0 AIC bk=30] iter#3 CLAIM op=1 wave=3 nTile=1 waveM=256
[AIC Block 30/32] [rank=0 AIC bk=30] iter#3 DONE (op=1)
[AIC Block 30/32] [rank=0 AIC bk=30] main loop DONE (3 tasks), sending END
[AIC Block 30/32] [rank=0 AIC bk=30] G1 final drain WaitForVector(0)...
[AIC Block 30/32] [rank=0 AIC bk=30] write GMM1 END
[AIC Block 30/32] [rank=0 AIC bk=30] G2 final drain (spin slot==0)...
[AIC Block 30/32] [rank=0 AIC bk=30] write GMM2 END
[AIC Block 30/32] [rank=0 AIC bk=30] <<< DynamicAicLoop EXIT exp=3
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 51/64] [rank=0 AIV1 bk=25] *** TIMEOUT spin slot exp=2 iter=4 slot=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 60/64] [rank=0 DISPATCH bk=30 ct=2 sub=0] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 60/64] [rank=0 AIV0 bk=30] >>> DynamicAiv0Loop ENTRY exp=0 tnG1=12
[AIV Block 60/64] [rank=0 AIV0 bk=30] iter#1 SLOT READ packed=536873992 op=0 wave=2 nTile=3 waveM=256
[AIV Block 60/64] [rank=0 AIV0 bk=30] iter#1 DONE
[AIV Block 60/64] [rank=0 AIV0 bk=30] iter#2 SLOT READ packed=2 op=2 wave=0 nTile=0 waveM=0
[AIV Block 60/64] [rank=0 AIV0 bk=30] <<< DynamicAiv0Loop EXIT
[AIV Block 60/64] [rank=0 DISPATCH bk=30 ct=2 sub=0] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 60/64] [rank=0 AIV0 bk=30] >>> DynamicAiv0Loop ENTRY exp=1 tnG1=12
[AIV Block 60/64] [rank=0 AIV0 bk=30] iter#1 SLOT READ packed=2 op=2 wave=0 nTile=0 waveM=0
[AIV Block 60/64] [rank=0 AIV0 bk=30] <<< DynamicAiv0Loop EXIT
[AIV Block 60/64] [rank=0 DISPATCH bk=30 ct=2 sub=0] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 60/64] [rank=0 AIV0 bk=30] >>> DynamicAiv0Loop ENTRY exp=2 tnG1=12
[AIV Block 60/64] [rank=0 AIV0 bk=30] iter#1 SLOT READ packed=536877064 op=0 wave=2 nTile=6 waveM=256
[AIV Block 60/64] [rank=0 AIV0 bk=30] iter#1 DONE
[AIV Block 60/64] [rank=0 AIV0 bk=30] iter#2 SLOT READ packed=2 op=2 wave=0 nTile=0 waveM=0
[AIV Block 60/64] [rank=0 AIV0 bk=30] <<< DynamicAiv0Loop EXIT
[AIV Block 60/64] [rank=0 DISPATCH bk=30 ct=2 sub=0] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 60/64] [rank=0 AIV0 bk=30] >>> DynamicAiv0Loop ENTRY exp=3 tnG1=12
[AIV Block 60/64] [rank=0 AIV0 bk=30] iter#1 SLOT READ packed=536877064 op=0 wave=2 nTile=6 waveM=256
[AIV Block 60/64] [rank=0 AIV0 bk=30] iter#1 DONE
[AIV Block 60/64] [rank=0 AIV0 bk=30] iter#2 SLOT READ packed=2 op=2 wave=0 nTile=0 waveM=0
[AIV Block 60/64] [rank=0 AIV0 bk=30] <<< DynamicAiv0Loop EXIT
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 61/64] [rank=0 DISPATCH bk=30 ct=2 sub=1] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 61/64] [rank=0 AIV1 bk=30] >>> DynamicAiv1Loop ENTRY exp=0 tnG2=28
[AIV Block 61/64] [rank=0 AIV1 bk=30] iter#1 SLOT READ packed=536886273 op=1 wave=0 nTile=15 waveM=256
[AIV Block 61/64] [rank=0 AIV1 bk=30] iter#1 DONE
[AIV Block 61/64] [rank=0 AIV1 bk=30] iter#2 SLOT READ packed=536884233 op=1 wave=2 nTile=13 waveM=256
[AIV Block 61/64] [rank=0 AIV1 bk=30] iter#2 DONE
[AIV Block 61/64] [rank=0 AIV1 bk=30] iter#3 SLOT READ packed=2 op=2 wave=0 nTile=0 waveM=0
[AIV Block 61/64] [rank=0 AIV1 bk=30] <<< DynamicAiv1Loop EXIT
[AIV Block 61/64] [rank=0 DISPATCH bk=30 ct=2 sub=1] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 61/64] [rank=0 AIV1 bk=30] >>> DynamicAiv1Loop ENTRY exp=1 tnG2=28
[AIV Block 61/64] [rank=0 AIV1 bk=30] iter#1 SLOT READ packed=536870917 op=1 wave=1 nTile=0 waveM=256
[AIV Block 61/64] [rank=0 AIV1 bk=30] iter#1 DONE
[AIV Block 61/64] [rank=0 AIV1 bk=30] iter#2 SLOT READ packed=536891397 op=1 wave=1 nTile=20 waveM=256
[AIV Block 61/64] [rank=0 AIV1 bk=30] iter#2 DONE
[AIV Block 61/64] [rank=0 AIV1 bk=30] iter#3 SLOT READ packed=2 op=2 wave=0 nTile=0 waveM=0
[AIV Block 61/64] [rank=0 AIV1 bk=30] <<< DynamicAiv1Loop EXIT
[AIV Block 61/64] [rank=0 DISPATCH bk=30 ct=2 sub=1] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 61/64] [rank=0 AIV1 bk=30] >>> DynamicAiv1Loop ENTRY exp=2 tnG2=28
[AIV Block 61/6CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 0/32] [rank=1 DISPATCH bk=0 ct=1 sub=0] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 0/32] [rank=1 AIC bk=0] >>> DynamicAicLoop ENTRY exp=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 0/32] [rank=1 AIC bk=0] dispatch flag DONE exp=0
[AIC Block 0/32] [rank=1 AIC bk=0] iter#1 CLAIM op=0 wave=2 nTile=1 waveM=256
[AIC Block 0/32] [rank=1 AIC bk=0] iter#1 DONE (op=0)
[AIC Block 0/32] [rank=1 AIC bk=0] iter#2 CLAIM op=1 wave=0 nTile=14 waveM=256
[AIC Block 0/32] [rank=1 AIC bk=0] iter#2 DONE (op=1)
[AIC Block 0/32] [rank=1 AIC bk=0] iter#3 CLAIM op=1 wave=1 nTile=18 waveM=256
[AIC Block 0/32] [rank=1 AIC bk=0] iter#3 DONE (op=1)
[AIC Block 0/32] [rank=1 AIC bk=0] main loop DONE (3 tasks), sending END
[AIC Block 0/32] [rank=1 AIC bk=0] G1 final drain WaitForVector(0)...
[AIC Block 0/32] [rank=1 AIC bk=0] write GMM1 END
[AIC Block 0/32] [rank=1 AIC bk=0] G2 final drain (spin slot==0)...
[AIC Block 0/32] [rank=1 AIC bk=0] write GMM2 END
[AIC Block 0/32] [rank=1 AIC bk=0] <<< DynamicAicLoop EXIT exp=0
[AIC Block 0/32] [rank=1 DISPATCH bk=0 ct=1 sub=0] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 0/32] [rank=1 AIC bk=0] >>> DynamicAicLoop ENTRY exp=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 0/32] [rank=1 AIC bk=0] dispatch flag DONE exp=1
[AIC Block 0/32] [rank=1 AIC bk=0] iter#1 CLAIM op=1 wave=1 nTile=9 waveM=256
[AIC Block 0/32] [rank=1 AIC bk=0] iter#1 DONE (op=1)
[AIC Block 0/32] [rank=1 AIC bk=0] iter#2 CLAIM op=1 wave=2 nTile=6 waveM=256
[AIC Block 0/32] [rank=1 AIC bk=0] iter#2 DONE (op=1)
[AIC Block 0/32] [rank=1 AIC bk=0] main loop DONE (2 tasks), sending END
[AIC Block 0/32] [rank=1 AIC bk=0] write GMM1 END
[AIC Block 0/32] [rank=1 AIC bk=0] G2 final drain (spin slot==0)...
[AIC Block 0/32] [rank=1 AIC bk=0] write GMM2 END
[AIC Block 0/32] [rank=1 AIC bk=0] <<< DynamicAicLoop EXIT exp=1
[AIC Block 0/32] [rank=1 DISPATCH bk=0 ct=1 sub=0] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 0/32] [rank=1 AIC bk=0] >>> DynamicAicLoop ENTRY exp=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 0/32] [rank=1 AIC bk=0] dispatch flag DONE exp=2
[AIC Block 0/32] [rank=1 AIC bk=0] iter#1 CLAIM op=1 wave=2 nTile=10 waveM=256
[AIC Block 0/32] [rank=1 AIC bk=0] iter#1 DONE (op=1)
[AIC Block 0/32] [rank=1 AIC bk=0] iter#2 CLAIM op=1 wave=2 nTile=11 waveM=256
[AIC Block 0/32] [rank=1 AIC bk=0] iter#2 DONE (op=1)
[AIC Block 0/32] [rank=1 AIC bk=0] main loop DONE (2 tasks), sending END
[AIC Block 0/32] [rank=1 AIC bk=0] write GMM1 END
[AIC Block 0/32] [rank=1 AIC bk=0] G2 final drain (spin slot==0)...
[AIC Block 0/32] [rank=1 AIC bk=0] write GMM2 END
[AIC Block 0/32] [rank=1 AIC bk=0] <<< DynamicAicLoop EXIT exp=2
[AIC Block 0/32] [rank=1 DISPATCH bk=0 ct=1 sub=0] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 0/32] [rank=1 AIC bk=0] >>> DynamicAicLoop ENTRY exp=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 0/32] [rank=1 AIC bk=0] dispatch flag DONE exp=3
[AIC Block 0/32] [rank=1 AIC bk=0] iter#1 CLAIM op=1 wave=2 nTile=23 waveM=256
[AIC Block 0/32] [rank=1 AIC bk=0] iter#1 DONE (op=1)
[AIC Block 0/32] [rank=1 AIC bk=0] iter#2 CLAIM op=1 wave=3 nTile=1 waveM=256
[AIC Block 0/32] [rank=1 AIC bk=0] iter#2 DONE (op=1)
[AIC Block 0/32] [rank=1 AIC bk=0] iter#3 CLAIM op=1 wave=3 nTile=16 waveM=256
[AIC Block 0/32] [rank=1 AIC bk=0] iter#3 DONE (op=1)
[AIC Block 0/32] [rank=1 AIC bk=0] main loop DONE (3 tasks), sending END
[AIC Block 0/32] [rank=1 AIC bk=0] write GMM1 END
[AIC Block 0/32] [rank=1 AIC bk=0] G2 final drain (spin slot==0)...
[AIC Block 0/32] [rank=1 AIC bk=0] write GMM2 END
[AIC Block 0/32] [rank=1 AIC bk=0] <<< DynamicAicLoop EXIT exp=3
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 0/64] [rank=1 DISPATCH bk=0 ct=2 sub=0] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 0/64] [rank=1 AIV0 bk=0] >>> DynamicAiv0Loop ENTRY exp=0 tnG1=12
[AIV Block 0/64] [rank=1 AIV0 bk=0] iter#1 SLOT READ packed=536871944 op=0 wave=2 nTile=1 waveM=256
[AIV Block 0/64] [rank=1 AIV0 bk=0] iter#1 DONE
[AIV Block 0/64] [rank=1 AIV0 bk=0] iter#2 SLOT READ packed=2 op=2 wave=0 nTile=0 waveM=0
[AIV Block 0/64] [rank=1 AIV0 bk=0] <<< DynamicAiv0Loop EXIT
[AIV Block 0/64] [rank=1 DISPATCH bk=0 ct=2 sub=0] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 0/64] [rank=1 AIV0 bk=0] >>> DynamicAiv0Loop ENTRY exp=1 tnG1=12
[AIV Block 0/64] [rank=1 AIV0 bk=0] iter#1 SLOT READ packed=2 op=2 wave=0 nTile=0 waveM=0
[AIV Block 0/64] [rank=1 AIV0 bk=0] <<< DynamicAiv0Loop EXIT
[AIV Block 0/64] [rank=1 DISPATCH bk=0 ct=2 sub=0] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 0/64] [rank=1 AIV0 bk=0] >>> DynamicAiv0Loop ENTRY exp=2 tnG1=12
[AIV Block 0/64] [rank=1 AIV0 bk=0] iter#1 SLOT READ packed=2 op=2 wave=0 nTile=0 waveM=0
[AIV Block 0/64] [rank=1 AIV0 bk=0] <<< DynamicAiv0Loop EXIT
[AIV Block 0/64] [rank=1 DISPATCH bk=0 ct=2 sub=0] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 0/64] [rank=1 AIV0 bk=0] >>> DynamicAiv0Loop ENTRY exp=3 tnG1=12
[AIV Block 0/64] [rank=1 AIV0 bk=0] iter#1 SLOT READ packed=2 op=2 wave=0 nTile=0 waveM=0
[AIV Block 0/64] [rank=1 AIV0 bk=0] <<< DynamicAiv0Loop EXIT
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 1/64] [rank=1 DISPATCH bk=0 ct=2 sub=1] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 1/64] [rank=1 AIV1 bk=0] >>> DynamicAiv1Loop ENTRY exp=0 tnG2=28
[AIV Block 1/64] [rank=1 AIV1 bk=0] iter#1 SLOT READ packed=536885249 op=1 wave=0 nTile=14 waveM=256
[AIV Block 1/64] [rank=1 AIV1 bk=0] iter#1 DONE
[AIV Block 1/64] [rank=1 AIV1 bk=0] iter#2 SLOT READ packed=536889349 op=1 wave=1 nTile=18 waveM=256
[AIV Block 1/64] [rank=1 AIV1 bk=0] iter#2 DONE
[AIV Block 1/64] [rank=1 AIV1 bk=0] iter#3 SLOT READ packed=2 op=2 wave=0 nTile=0 waveM=0
[AIV Block 1/64] [rank=1 AIV1 bk=0] <<< DynamicAiv1Loop EXIT
[AIV Block 1/64] [rank=1 DISPATCH bk=0 ct=2 sub=1] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 1/64] [rank=1 AIV1 bk=0] >>> DynamicAiv1Loop ENTRY exp=1 tnG2=28
[AIV Block 1/64] [rank=1 AIV1 bk=0] iter#1 SLOT READ packed=536880133 op=1 wave=1 nTile=9 waveM=256
[AIV Block 1/64] [rank=1 AIV1 bk=0] iter#1 DONE
[AIV Block 1/64] [rank=1 AIV1 bk=0] iter#2 SLOT READ packed=536877065 op=1 wave=2 nTile=6 waveM=256
[AIV Block 1/64] [rank=1 AIV1 bk=0] iter#2 DONE
[AIV Block 1/64] [rank=1 AIV1 bk=0] iter#3 SLOT READ packed=2 op=2 wave=0 nTile=0 waveM=0
[AIV Block 1/64] [rank=1 AIV1 bk=0] <<< DynamicAiv1Loop EXIT
[AIV Block 1/64] [rank=1 DISPATCH bk=0 ct=2 sub=1] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 1/64] [rank=1 AIV1 bk=0] >>> DynamicAiv1Loop ENTRY exp=2 tnG2=28
[AIV Block 1/64] [rank=1 AIV1 bk=0] iter#1 SLOT READ packed=536881161 op=1 wave=2 nTile=10 waveM=256
[AIV Block 1/64] [rank=1 AIV1 bk=0] iter#1 DONE
[AIV Block 1/64] [rank=1 AIV1 bk=0] iter#2 SLOT READ packed=536882185 op=1 wave=2 nTile=11 waveM=256
[AIV Block 1/64] [rank=1 AIV1 bk=0] iter#2 DONE
[AIV Block 1/64] [rank=1 AIV1 bk=0] iter#3 SLOT READ packed=2 op=2 wave=0 nTile=0 waveM=0
[AIV Block 1/64] [rank=1 AIV1 bk=0] <<< DynamicAiv1Loop EXIT
[AIV Block 1/64] [rank=1 DISPATCH bk=0 ct=2 sub=1] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 1/64] [rank=1 AIV1 bk=0] >>> DynamicAiv1Loop ENTRY exp=3 tnG2=28
[AIV Block 1/64] [rank=1 AIV1 bk=0] iter#1 SLOT READ packed=536894473 op=1 wave=2 nTile=23 waveM=256
[AIV Block 1/64] [rank=1 AIV1 bk=0] iter#1 DONE
[AIV Block 1/64] [rank=1 AIV1 bk=0] iter#2 SLOT READ packed=536871949 op=1 wave=3 nTile=1 waveM=256
[AIV Block 1/64] [rank=1 AIV1 bk=0] iter#2 DONE
[AIV Block 1/64] [rank=1 AIV1 bk=0] iter#3 SLOT READ packed=536887309 op=1 wave=3 nTile=16 waveM=256
[AIV Block 1/64] [rank=1 AIV1 bk=0] iter#3 DONE
[AIV Block 1/64] [rank=1 AIV1 bk=0] iter#4 SLOT READ packed=2 op=2 wave=0 nTile=0 waveM=0
[AIV Block 1/64] [rank=1 AIV1 bk=0] <<< DynamicAiv1Loop EXIT
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 30/32] [rank=1 DISPATCH bk=30 ct=1 sub=0] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 30/32] [rank=1 AIC bk=30] >>> DynamicAicLoop ENTRY exp=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 30/32] [rank=1 AIC bk=30] dispatch flag DONE exp=0
[AIC Block 30/32] [rank=1 AIC bk=30] iter#1 CLAIM op=0 wave=2 nTile=6 waveM=256
[AIC Block 30/32] [rank=1 AIC bk=30] iter#1 DONE (op=0)
[AIC Block 30/32] [rank=1 AIC bk=30] iter#2 CLAIM op=1 wave=0 nTile=15 waveM=256
[AIC Block 30/32] [rank=1 AIC bk=30] iter#2 DONE (op=1)
[AIC Block 30/32] [rank=1 AIC bk=30] iter#3 CLAIM op=1 wave=1 nTile=19 waveM=256
[AIC Block 30/32] [rank=1 AIC bk=30] iter#3 DONE (op=1)
[AIC Block 30/32] [rank=1 AIC bk=30] iter#4 CLAIM op=1 wave=3 nTile=2 waveM=256
[AIC Block 30/32] [rank=1 AIC bk=30] iter#4 DONE (op=1)
[AIC Block 30/32] [rank=1 AIC bk=30] main loop DONE (4 tasks), sending END
[AIC Block 30/32] [rank=1 AIC bk=30] G1 final drain WaitForVector(0)...
[AIC Block 30/32] [rank=1 AIC bk=30] write GMM1 END
[AIC Block 30/32] [rank=1 AIC bk=30] G2 final drain (spin slot==0)...
[AIC Block 30/32] [rank=1 AIC bk=30] write GMM2 END
[AIC Block 30/32] [rank=1 AIC bk=30] <<< DynamicAicLoop EXIT exp=0
[AIC Block 30/32] [rank=1 DISPATCH bk=30 ct=1 sub=0] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 30/32] [rank=1 AIC bk=30] >>> DynamicAicLoop ENTRY exp=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 30/32] [rank=1 AIC bk=30] dispatch flag DONE exp=1
[AIC Block 30/32] [rank=1 AIC bk=30] iter#1 CLAIM op=1 wave=2 nTile=5 waveM=256
[AIC Block 30/32] [rank=1 AIC bk=30] iter#1 DONE (op=1)
[AIC Block 30/32] [rank=1 AIC bk=30] iter#2 CLAIM op=1 wave=2 nTile=9 waveM=256
[AIC Block 30/32] [rank=1 AIC bk=30] iter#2 DONE (op=1)
[AIC Block 30/32] [rank=1 AIC bk=30] iter#3 CLAIM op=1 wave=2 nTile=20 waveM=256
[AIC Block 30/32] [rank=1 AIC bk=30] iter#3 DONE (op=1)
[AIC Block 30/32] [rank=1 AIC bk=30] iter#4 CLAIM op=1 wave=3 nTile=1 waveM=256
[AIC Block 30/32] [rank=1 AIC bk=30] iter#4 DONE (op=1)
[AIC Block 30/32] [rank=1 AIC bk=30] iter#5 CLAIM op=1 wave=3 nTile=12 waveM=256
[AIC Block 30/32] [rank=1 AIC bk=30] iter#5 DONE (op=1)
[AIC Block 30/32] [rank=1 AIC bk=30] main loop DONE (6 tasks), sending END
[AIC Block 30/32] [rank=1 AIC bk=30] write GMM1 END
[AIC Block 30/32] [rank=1 AIC bk=30] G2 final drain (spin slot==0)...
[AIC Block 30/32] [rank=1 AIC bk=30] write GMM2 END
[AIC Block 30/32] [rank=1 AIC bk=30] <<< DynamicAicLoop EXIT exp=1
[AIC Block 30/32] [rank=1 DISPATCH bk=30 ct=1 sub=0] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 30/32] [rank=1 AIC bk=30] >>> DynamicAicLoop ENTRY exp=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 30/32] [rank=1 AIC bk=30] dispatch flag DONE exp=2
[AIC Block 30/32] [rank=1 AIC bk=30] iter#1 CLAIM op=0 wave=1 nTile=9 waveM=256
[AIC Block 30/32] [rank=1 AIC bk=30] iter#1 DONE (op=0)
[AIC Block 30/32] [rank=1 AIC bk=30] iter#2 CLAIM op=1 wave=0 nTile=12 waveM=256
[AIC Block 30/32] [rank=1 AIC bk=30] iter#2 DONE (op=1)
[AIC Block 30/32] [rank=1 AIC bk=30] iter#3 CLAIM op=1 wave=2 nTile=3 waveM=256
[AIC Block 30/32] [rank=1 AIC bk=30] iter#3 DONE (op=1)
[AIC Block 30/32] [rank=1 AIC bk=30] iter#4 CLAIM op=1 wave=3 nTile=2 waveM=256
[AIC Block 30/32] [rank=1 AIC bk=30] iter#4 DONE (op=1)
[AIC Block 30/32] [rank=1 AIC bk=30] iter#5 CLAIM op=1 wave=3 nTile=15 waveM=256
[AIC Block 30/32] [rank=1 AIC bk=30] iter#5 DONE (op=1)
[AIC Block 30/32] [rank=1 AIC bk=30] main loop DONE (7 tasks), sending END
[AIC Block 30/32] [rank=1 AIC bk=30] G1 final drain WaitForVector(0)...
[AIC Block 30/32] [rank=1 AIC bk=30] write GMM1 END
[AIC Block 30/32] [rank=1 AIC bk=30] G2 final drain (spin slot==0)...
[AIC Block 30/32] [rank=1 AIC bk=30] write GMM2 END
[AIC Block 30/32] [rank=1 AIC bk=30] <<< DynamicAicLoop EXIT exp=2
[AIC Block 30/32] [rank=1 DISPATCH bk=30 ct=1 sub=0] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 30/32] [rank=1 AIC bk=30] >>> DynamicAicLoop ENTRY exp=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 30/32] [rank=1 AIC bk=30] dispatch flag DONE exp=3
[AIC Block 30/32] [rank=1 AIC bk=30] iter#1 CLAIM op=1 wave=0 nTile=9 waveM=256
[AIC Block 30/32] [rank=1 AIC bk=30] iter#1 DONE (op=1)
[AIC Block 30/32] [rank=1 AIC bk=30] iter#2 CLAIM op=1 wave=1 nTile=17 waveM=256
[AIC Block 30/32] [rank=1 AIC bk=30] iter#2 DONE (op=1)
[AIC Block 30/32] [rank=1 AIC bk=30] iter#3 CLAIM op=1 wave=3 nTile=3 waveM=256
[AIC Block 30/32] [rank=1 AIC bk=30] iter#3 DONE (op=1)
[AIC Block 30/32] [rank=1 AIC bk=30] main loop DONE (3 tasks), sending END
[AIC Block 30/32] [rank=1 AIC bk=30] write GMM1 END
[AIC Block 30/32] [rank=1 AIC bk=30] G2 final drain (spin slot==0)...
[AIC Block 30/32] [rank=1 AIC bk=30] write GMM2 END
[AIC Block 30/32] [rank=1 AIC bk=30] <<< DynamicAicLoop EXIT exp=3
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 60/64] [rank=1 DISPATCH bk=30 ct=2 sub=0] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 60/64] [rank=1 AIV0 bk=30] >>> DynamicAiv0Loop ENTRY exp=0 tnG1=12
[AIV Block 60/64] [rank=1 AIV0 bk=30] iter#1 SLOT READ packed=536877064 op=0 wave=2 nTile=6 waveM=256
[AIV Block 60/64] [rank=1 AIV0 bk=30] iter#1 DONE
[AIV Block 60/64] [rank=1 AIV0 bk=30] iter#2 SLOT READ packed=2 op=2 wave=0 nTile=0 waveM=0
[AIV Block 60/64] [rank=1 AIV0 bk=30] <<< DynamicAiv0Loop EXIT
[AIV Block 60/64] [rank=1 DISPATCH bk=30 ct=2 sub=0] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 60/64] [rank=1 AIV0 bk=30] >>> DynamicAiv0Loop ENTRY exp=1 tnG1=12
[AIV Block 60/64] [rank=1 AIV0 bk=30] iter#1 SLOT READ packed=2 op=2 wave=0 nTile=0 waveM=0
[AIV Block 60/64] [rank=1 AIV0 bk=30] <<< DynamicAiv0Loop EXIT
[AIV Block 60/64] [rank=1 DISPATCH bk=30 ct=2 sub=0] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 60/64] [rank=1 AIV0 bk=30] >>> DynamicAiv0Loop ENTRY exp=2 tnG1=12
[AIV Block 60/64] [rank=1 AIV0 bk=30] iter#1 SLOT READ packed=536880132 op=0 wave=1 nTile=9 waveM=256
[AIV Block 60/64] [rank=1 AIV0 bk=30] iter#1 DONE
[AIV Block 60/64] [rank=1 AIV0 bk=30] iter#2 SLOT READ packed=2 op=2 wave=0 nTile=0 waveM=0
[AIV Block 60/64] [rank=1 AIV0 bk=30] <<< DynamicAiv0Loop EXIT
[AIV Block 60/64] [rank=1 DISPATCH bk=30 ct=2 sub=0] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 60/64] [rank=1 AIV0 bk=30] >>> DynamicAiv0Loop ENTRY exp=3 tnG1=12
[AIV Block 60/64] [rank=1 AIV0 bk=30] iter#1 SLOT READ packed=2 op=2 wave=0 nTile=0 waveM=0
[AIV Block 60/64] [rank=1 AIV0 bk=30] <<< DynamicAiv0Loop EXIT
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 61/64] [rank=1 DISPATCH bk=30 ct=2 sub=1] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 61/64] [rank=1 AIV1 bk=30] >>> DynamicAiv1Loop ENTRY exp=0 tnG2=28
[AIV Block 61/64] [rank=1 AIV1 bk=30] iter#1 SLOT READ packed=536886273 op=1 wave=0 nTile=15 waveM=256
[AIV Block 61/64] [rank=1 AIV1 bk=30] iter#1 DONE
[AIV Block 61/64] [rank=1 AIV1 bk=30] iter#2 SLOT READ packed=536890373 op=1 wave=1 nTile=19 waveM=256
[AIV Block 61/64] [rank=1 AIV1 bk=30] iter#2 DONE
[AIV Block 61/64] [rank=1 AIV1 bk=30] iter#3 SLOT READ packed=536872973 op=1 wave=3 nTile=2 waveM=256
[AIV Block 61/64] [rank=1 AIV1 bk=30] iter#3 DONE
[AIV Block 61/64] [rank=1 AIV1 bk=30] iter#4 SLOT READ packed=2 op=2 wave=0 nTile=0 waveM=0
[AIV Block 61/64] [rank=1 AIV1 bk=30] <<< DynamicAiv1Loop EXIT
[AIV Block 61/64] [rank=1 DISPATCH bk=30 ct=2 sub=1] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 61/64] [rank=1 AIV1 bk=30] >>> DynamicAiv1Loop ENTRY exp=1 tnG2=28
[AIV Block 61/64] [rank=1 AIV1 bk=30] iter#1 SLOT READ packed=536876041 op=1 wave=2 nTile=5 waveM=256
[AIV Block 61/64] [rank=1 AIV1 bk=30] iter#1 DONE
[AIV Block 61/64] [rank=1 AIV1 bk=30] iter#2 SLOT READ packed=536880137 op=1 wave=2 nTile=9 waveM=256
[AIV Block 61/64] [rank=1 AIV1 bk=30] iter#2 DONE
[AIV Block 61/64] [rank=1 AIV1 bk=30] iter#3 SLOT READ packed=536891401 op=1 wave=2 nTile=20 waveM=256
[AIV Block 61/64] [rank=1 AIV1 bk=30] iter#3 DONE
[AIV Block 61/64] [rank=1 AIV1 bk=30] iter#4 SLOT READ packed=536871949 op=1 wave=3 nTile=1 waveM=256
[AIV Block 61/64] [rank=1 AIV1 bk=30] iter#4 DONE
[AIV Block 61/64] [rank=1 AIV1 bk=30] iter#5 SLOT READ packed=536883213 op=1 wave=3 nTile=12 waveM=256
[AIV Block 61/64] [rank=1 AIV1 bk=30] iter#5 DONE
[AIV Block 61/64] [rank=1 AIV1 bk=30] <<< DynamicAiv1Loop EXIT
[AIV Block 61/64] [rank=1 DISPATCH bk=30 ct=2 sub=1] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 61/64] [rank=1 AIV1 bk=30] >>> DynamicAiv1Loop ENTRY exp=2 tnG2=28
[AIV Block 61/64] [rank=1 AIV1 bk=30] iter#1 SLOT READ packed=536883201 op=1 wave=0 nTile=12 waveM=256
[AIV Block 61/64] [rank=1 AIV1 bk=30] iter#1 DONE
[AIV Block 61/64] [rank=1 AIV1 bk=30] iter#2 SLOT READ packed=536873993 op=1 wave=2 nTile=3 waveM=256
[AIV Block 61/64] [rank=1 AIV1 bk=30] iter#2 DONE
[AIV Block 61/64] [rank=1 AIV1 bk=30] iter#3 SLOT READ packed=536872973 op=1 wave=3 nTile=2 waveM=256
[AIV Block 61/64] [rank=1 AIV1 bk=30] iter#3 DONE
[AIV Block 61/64] [rank=1 AIV1 bk=30] iter#4 SLOT READ packed=536886285 op=1 wave=3 nTile=15 waveM=256
[AIV Block 61/64] [rank=1 AIV1 bk=30] iter#4 DONE
[AIV Block 61/64] [rank=1 AIV1 bk=30] iter#5 SLOT READ packed=536894477 op=1 wave=3 nTile=23 waveM=256
[AIV Block 61/64] [rank=1 AIV1 bk=30] iter#5 DONE
[AIV Block 61/64] [rank=1 AIV1 bk=30] <<< DynamicAiv1Loop EXIT
[AIV Block 61/64] [rank=1 DISPATCH bk=30 ct=2 sub=1] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 61/64] [rank=1 AIV1 bk=30] >>> DynamicAiv1Loop ENTRY exp=3 tnG2=28
[AIV Block 61/64] [rank=1 AIV1 bk=30] iter#1 SLOT READ packed=536880129 op=1 wave=0 nTile=9 waveM=256
[AIV Block 61/64] [rank=1 AIV1 bk=30] iter#1 DONE
[AIV Block 61/64] [rank=1 AIV1 bk=30] iter#2 SLOT READ packed=536888325 op=1 wave=1 nTile=17 waveM=256
[AIV Block 61/64] [rank=1 AIV1 bk=30] iter#2 DONE
[AIV Block 61/64] [rank=1 AIV1 bk=30] iter#3 SLOT READ packed=536873997 op=1 wave=3 nTile=3 waveM=256
[AIV Block 61/64] [rank=1 AIV1 bk=30] iter#3 DONE
[AIV Block 61/64] [rank=1 AIV1 bk=30] iter#4 SLOT READ packed=2 op=2 wave=0 nTile=0 waveM=0
[AIV Block 61/64] [rank=1 AIV1 bk=30] <<< DynamicAiv1Loop EXIT
4] [rank=0 AIV1 bk=30] iter#1 SLOT READ packed=536888321 op=1 wave=0 nTile=17 waveM=256
[AIV Block 61/64] [rank=0 AIV1 bk=30] iter#1 DONE
[AIV Block 61/64] [rank=0 AIV1 bk=30] iter#2 SLOT READ packed=536896517 op=1 wave=1 nTile=25 waveM=256
[AIV Block 61/64] [rank=0 AIV1 bk=30] iter#2 DONE
[AIV Block 61/64] [rank=0 AIV1 bk=30] iter#3 SLOT READ packed=2 op=2 wave=0 nTile=0 waveM=0
[AIV Block 61/64] [rank=0 AIV1 bk=30] <<< DynamicAiv1Loop EXIT
[AIV Block 61/64] [rank=0 DISPATCH bk=30 ct=2 sub=1] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 61/64] [rank=0 AIV1 bk=30] >>> DynamicAiv1Loop ENTRY exp=3 tnG2=28
[AIV Block 61/64] [rank=0 AIV1 bk=30] iter#1 SLOT READ packed=536894465 op=1 wave=0 nTile=23 waveM=256
[AIV Block 61/64] [rank=0 AIV1 bk=30] iter#1 DONE
[AIV Block 61/64] [rank=0 AIV1 bk=30] iter#2 SLOT READ packed=536871949 op=1 wave=3 nTile=1 waveM=256
[AIV Block 61/64] [rank=0 AIV1 bk=30] iter#2 DONE
[AIV Block 61/64] [rank=0 AIV1 bk=30] iter#3 SLOT READ packed=2 op=2 wave=0 nTile=0 waveM=0
[AIV Block 61/64] [rank=0 AIV1 bk=30] <<< DynamicAiv1Loop EXIT
##################Compare X in rank 0 epId 0 tpId 0 ##################
local_rank_id2374:0
[2026/05/20 19:32:36] [INFO] ---------------------------------------------------------------------------------------
[2026/05/20 19:32:36] [INFO] Loop 	         cascade_npu_out 	             golden 	                   npu_out 	           RateDiff
[2026/05/20 19:32:36] [INFO] ---------------------------------------------------------------------------------------
[2026/05/20 19:32:36] [INFO] 00000001 	              0.2421875 	              0.2428483 	              0.2421875 	 0.0006608
[2026/05/20 19:32:36] [INFO] 00000002 	             -1.1328125 	             -1.1348476 	             -1.1328125 	 0.0017933
[2026/05/20 19:32:36] [INFO] 00000003 	              1.5234375 	              1.5227127 	              1.5234375 	 0.0007248
[2026/05/20 19:32:36] [INFO] 00000004 	              1.1796875 	              1.1770338 	              1.1796875 	 0.0022495
[2026/05/20 19:32:36] [INFO] 00000005 	             -0.5468750 	             -0.5438833 	             -0.5468750 	 0.0054705
[2026/05/20 19:32:36] [INFO] 00000006 	             -0.0456543 	             -0.0455816 	             -0.0456543 	 0.0000727
[2026/05/20 19:32:36] [INFO] 00000007 	              0.3437500 	              0.3441011 	              0.3437500 	 0.0003511
[2026/05/20 19:32:36] [INFO] 00000008 	              0.2773438 	              0.2782815 	              0.2773438 	 0.0009377
[2026/05/20 19:32:36] [INFO] 00000009 	             -0.0698242 	             -0.0695252 	             -0.0698242 	 0.0002990
[2026/05/20 19:32:36] [INFO] 00000010 	              1.0234375 	              1.0249438 	              1.0234375 	 0.0014697
[2026/05/20 19:32:36] [INFO] ...   	   ...   	   ...   	   ...    	   ...
[2026/05/20 19:32:36] [INFO] 03670007 	             -0.6328125 	             -0.6314448 	             -0.6328125 	 0.0021613
[2026/05/20 19:32:36] [INFO] 03670008 	              0.3710938 	              0.3725935 	              0.3710938 	 0.0040251
[2026/05/20 19:32:36] [INFO] 03670009 	              0.1552734 	              0.1535522 	              0.1552734 	 0.0110850
[2026/05/20 19:32:36] [INFO] 03670010 	             -0.8828125 	             -0.8840220 	             -0.8828125 	 0.0013682
[2026/05/20 19:32:36] [INFO] 03670011 	              0.4101562 	              0.4096218 	              0.4101562 	 0.0005344
[2026/05/20 19:32:36] [INFO] 03670012 	             -0.1416016 	             -0.1429626 	             -0.1416016 	 0.0095203
[2026/05/20 19:32:36] [INFO] 03670013 	              0.4199219 	              0.4195377 	              0.4199219 	 0.0003842
[2026/05/20 19:32:36] [INFO] 03670014 	             -0.1875000 	             -0.1874906 	             -0.1875000 	 0.0000094
[2026/05/20 19:32:36] [INFO] 03670015 	             -0.0869141 	             -0.0855013 	             -0.0869141 	 0.0162547
[2026/05/20 19:32:36] [INFO] 03670016 	              0.7617188 	              0.7604597 	              0.7617188 	 0.0016529
##################Compare X in rank 1 epId 1 tpId 0 ##################
local_rank_id2374:1
[2026/05/20 19:32:36] [INFO] ---------------------------------------------------------------------------------------
[2026/05/20 19:32:36] [INFO] Loop 	         cascade_npu_out 	             golden 	                   npu_out 	           RateDiff
[2026/05/20 19:32:36] [INFO] ---------------------------------------------------------------------------------------
[2026/05/20 19:32:36] [INFO] 00000001 	             -0.7617188 	             -0.7617300 	             -0.7617188 	 0.0000112
[2026/05/20 19:32:36] [INFO] 00000002 	              0.8554688 	              0.8577936 	              0.8554688 	 0.0027102
[2026/05/20 19:32:36] [INFO] 00000003 	              0.3242188 	              0.3244100 	              0.3242188 	 0.0001912
[2026/05/20 19:32:36] [INFO] 00000004 	              0.2675781 	              0.2683774 	              0.2675781 	 0.0007993
[2026/05/20 19:32:36] [INFO] 00000005 	             -0.7031250 	             -0.7012520 	             -0.7031250 	 0.0026638
[2026/05/20 19:32:36] [INFO] 00000006 	             -0.5898438 	             -0.5917263 	             -0.5898438 	 0.0031815
[2026/05/20 19:32:36] [INFO] 00000007 	             -0.1972656 	             -0.1957367 	             -0.1972656 	 0.0077505
[2026/05/20 19:32:36] [INFO] 00000008 	             -0.5390625 	             -0.5396764 	             -0.5390625 	 0.0006139
[2026/05/20 19:32:36] [INFO] 00000009 	              0.4414062 	              0.4434330 	              0.4414062 	 0.0045706
[2026/05/20 19:32:36] [INFO] 00000010 	             -0.3085938 	             -0.3079789 	             -0.3085938 	 0.0006149
[2026/05/20 19:32:36] [INFO] ...   	   ...   	   ...   	   ...    	   ...
[2026/05/20 19:32:36] [INFO] 03670007 	              0.8007812 	              0.8019012 	              0.8007812 	 0.0013966
[2026/05/20 19:32:36] [INFO] 03670008 	             -1.2656250 	             -1.2691318 	             -1.2656250 	 0.0027631
[2026/05/20 19:32:36] [INFO] 03670009 	             -0.4667969 	             -0.4652840 	             -0.4667969 	 0.0032410
[2026/05/20 19:32:36] [INFO] 03670010 	              0.1396484 	              0.1388095 	              0.1396484 	 0.0008389
[2026/05/20 19:32:36] [INFO] 03670011 	             -0.0520020 	             -0.0533563 	             -0.0520020 	 0.0253822
[2026/05/20 19:32:36] [INFO] 03670012 	              0.8437500 	              0.8435734 	              0.8437500 	 0.0001766
[2026/05/20 19:32:36] [INFO] 03670013 	              0.6640625 	              0.6656507 	              0.6640625 	 0.0023860
[2026/05/20 19:32:36] [INFO] 03670014 	             -0.5898438 	             -0.5896969 	             -0.5898438 	 0.0001468
[2026/05/20 19:32:36] [INFO] 03670015 	              0.1206055 	              0.1210413 	              0.1206055 	 0.0004358
[2026/05/20 19:32:36] [INFO] 03670016 	             -0.9609375 	             -0.9614851 	             -0.9609375 	 0.0005476
##################Compare EXPERT TOKEN NUMS##################
[2026/05/20 19:32:36] [INFO] split_count:8.0; max_diff_hd:0.0;
[2026/05/20 19:32:36] [INFO] ---------------------------------------------------------------------------------------
[2026/05/20 19:32:36] [INFO] Loop 	 ExpectOut 	 RealOut 	 FpDiff 	 RateDiff
[2026/05/20 19:32:36] [INFO] ---------------------------------------------------------------------------------------
[2026/05/20 19:32:36] [INFO] 00000001 	 1024.0000000 	 1024.0000000 	 0.0000000 	 0.0000000
[2026/05/20 19:32:36] [INFO] 00000002 	 1024.0000000 	 1024.0000000 	 0.0000000 	 0.0000000
[2026/05/20 19:32:36] [INFO] 00000003 	 1024.0000000 	 1024.0000000 	 0.0000000 	 0.0000000
[2026/05/20 19:32:36] [INFO] 00000004 	 1024.0000000 	 1024.0000000 	 0.0000000 	 0.0000000
[2026/05/20 19:32:36] [INFO] 00000005 	 1024.0000000 	 1024.0000000 	 0.0000000 	 0.0000000
[2026/05/20 19:32:36] [INFO] 00000006 	 1024.0000000 	 1024.0000000 	 0.0000000 	 0.0000000
[2026/05/20 19:32:36] [INFO] 00000007 	 1024.0000000 	 1024.0000000 	 0.0000000 	 0.0000000
[2026/05/20 19:32:36] [INFO] 00000008 	 1024.0000000 	 1024.0000000 	 0.0000000 	 0.0000000
[2026/05/20 19:32:36] [INFO] ---------------------------------------------------------------------------------------
[2026/05/20 19:32:36] [INFO] DiffThd  	 PctThd   	 PctRlt   	 Result
[2026/05/20 19:32:36] [INFO] ---------------------------------------------------------------------------------------
[2026/05/20 19:32:36] [INFO] 0.0010     	 99.90%   	 100.000000%   	 PASS
[INFO] tensor expert_token_nums output测试的精度结果为：('PASS', 100.0)
##################Compare x in rank 0 epId 0 tpId 0 ##################
info：开始计算 npu_x 精度。
处理golden中，超过value表示范围的值，赋值为inf/-inf
info：开始计算 npu_golden_x 精度。
处理golden中，超过value表示范围的值，赋值为inf/-inf
comparing result: npu_x VS npu_golden_x
npu_xcompare result: ok
[INFO] tensor x output 精度结果为：PASS
##################Compare x in rank 1 epId 1 tpId 0 ##################
info：开始计算 npu_x 精度。
处理golden中，超过value表示范围的值，赋值为inf/-inf
info：开始计算 npu_golden_x 精度。
处理golden中，超过value表示范围的值，赋值为inf/-inf
comparing result: npu_x VS npu_golden_x
npu_xcompare result: ok
[INFO] tensor x output 精度结果为：PASS

##################expert_token_nums result##################
expert_token_nums:  {'expert_token_nums': True}
