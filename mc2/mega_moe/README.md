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
current device set: 0
[INFO] device_0 创建HCCL通信链路
device_0 init_process_group success
device 0 初始化EP域
device 0 初始化TP域
jjjjjjjjjjjjjjj
<npu_ops_transformer.ops.mega_moe.SymmBuffer object at 0x7f6eebb94a30>
rank:0 topk_idstensor([[0, 1, 2,  ..., 5, 6, 7],
        [1, 2, 3,  ..., 6, 7, 0],
        [2, 3, 4,  ..., 7, 0, 1],
        ...,
        [5, 6, 7,  ..., 2, 3, 4],
        [6, 7, 0,  ..., 3, 4, 5],
        [7, 0, 1,  ..., 4, 5, 6]], device='npu:0', dtype=torch.int32)
rank:0 构造算子输入finished!
[2026-05-21 10:54:06] [WARNING] [68384] profiler.py: Profiler won't be using warmup, this can skew profiler results
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
<npu_ops_transformer.ops.mega_moe.SymmBuffer object at 0x7f87f5fc0a30>
rank:1 topk_idstensor([[0, 1, 2,  ..., 5, 6, 7],
        [1, 2, 3,  ..., 6, 7, 0],
        [2, 3, 4,  ..., 7, 0, 1],
        ...,
        [5, 6, 7,  ..., 2, 3, 4],
        [6, 7, 0,  ..., 3, 4, 5],
        [7, 0, 1,  ..., 4, 5, 6]], device='npu:1', dtype=torch.int32)
rank:1 构造算子输入finished!
[2026-05-21 10:54:06] [WARNING] [68700] profiler.py: Profiler won't be using warmup, this can skew profiler results
[2026-05-21 10:54:09] [INFO] [69493] profiler.py: Start parsing profiling data in sync mode at: /home/c00946619/script/Profiling_Result_mc2_cascade_16p/huawei_68384_20260521105406477_ascend_pt
|[2026-05-21 10:54:09] [INFO] [69515] profiler.py: Start parsing profiling data in sync mode at: /home/c00946619/script/Profiling_Result_mc2_cascade_16p/huawei_68700_20260521105406536_ascend_pt
|[2026-05-21 10:54:09] [WARNING] [69493] profiler.py: 20 memory record(s) are incomplete.
[2026-05-21 10:54:09] [WARNING] [69515] profiler.py: 20 memory record(s) are incomplete.
//--\\||//--\\||/[2026-05-21 10:54:18] [INFO] [69508] profiler.py: CANN profiling data parsed in a total time of 0:00:09.037925
/[2026-05-21 10:54:18] [INFO] [69536] profiler.py: CANN profiling data parsed in a total time of 0:00:09.032900
--[2026-05-21 10:54:19] [INFO] [69493] profiler.py: All profiling data parsed in a total time of 0:00:10.542571
[2026-05-21 10:54:19] [INFO] [69515] profiler.py: All profiling data parsed in a total time of 0:00:10.521896
[2026-05-21 10:54:20] [WARNING] [68384] profiler.py: Profiler won't be using warmup, this can skew profiler results
[2026-05-21 10:54:20] [WARNING] [68700] profiler.py: Profiler won't be using warmup, this can skew profiler results
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 1/32] [rank=0 AIC bk=1] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 2/32] [rank=0 AIC bk=2] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 3/32] [rank=0 AIC bk=3] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 4/32] [rank=0 AIC bk=4] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 6/32] [rank=0 AIC bk=6] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 7/32] [rank=0 AIC bk=7] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 8/32] [rank=0 AIC bk=8] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 9/32] [rank=0 AIC bk=9] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 11/32] [rank=0 AIC bk=11] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 12/32] [rank=0 AIC bk=12] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 13/32] [rank=0 AIC bk=13] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 14/32] [rank=0 AIC bk=14] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 15/32] [rank=0 AIC bk=15] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 0/64] [rank=0 SCHED] TIMEOUT main loop exp=2
[AIV Block 0/64]   wave=0 flag=12 d1=12 d2=28
[AIV Block 0/64]   wave=1 flag=12 d1=12 d2=28
[AIV Block 0/64]   wave=2 flag=12 d1=12 d2=28
[AIV Block 0/64]   wave=3 flag=12 d1=12 d2=28
[AIV Block 0/64]   STUCK bk=22 slot_g1aic=0 _aiv=0 slot_g2aic=-2 _aiv=0 e1s=1 e1a=1 e2s=1 e2a=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 3/64] [static GMM2 AIV1 bk=1] *** TIMEOUT spin rowReadyFlags loopIdx=17 cur=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 5/64] [static GMM2 AIV1 bk=2] *** TIMEOUT spin rowReadyFlags loopIdx=18 cur=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 7/64] [static GMM2 AIV1 bk=3] *** TIMEOUT spin rowReadyFlags loopIdx=19 cur=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 9/64] [static GMM2 AIV1 bk=4] *** TIMEOUT spin rowReadyFlags loopIdx=20 cur=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 13/64] [static GMM2 AIV1 bk=6] *** TIMEOUT spin rowReadyFlags loopIdx=22 cur=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 15/64] [static GMM2 AIV1 bk=7] *** TIMEOUT spin rowReadyFlags loopIdx=23 cur=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 17/64] [static GMM2 AIV1 bk=8] *** TIMEOUT spin rowReadyFlags loopIdx=24 cur=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 19/64] [static GMM2 AIV1 bk=9] *** TIMEOUT spin rowReadyFlags loopIdx=25 cur=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 23/64] [static GMM2 AIV1 bk=11] *** TIMEOUT spin rowReadyFlags loopIdx=27 cur=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 25/64] [static GMM2 AIV1 bk=12] *** TIMEOUT spin rowReadyFlags loopIdx=6 cur=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 27/64] [static GMM2 AIV1 bk=13] *** TIMEOUT spin rowReadyFlags loopIdx=0 cur=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 29/64] [static GMM2 AIV1 bk=14] *** TIMEOUT spin rowReadyFlags loopIdx=1 cur=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 31/64] [static GMM2 AIV1 bk=15] *** TIMEOUT spin rowReadyFlags loopIdx=2 cur=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 18/32] [rank=0 AIC bk=18] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 21/32] [rank=0 AIC bk=21] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 23/32] [rank=0 AIC bk=23] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 24/32] [rank=0 AIC bk=24] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 25/32] [rank=0 AIC bk=25] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 26/32] [rank=0 AIC bk=26] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 27/32] [rank=0 AIC bk=27] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 28/32] [rank=0 AIC bk=28] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 29/32] [rank=0 AIC bk=29] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 30/32] [rank=0 AIC bk=30] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 31/32] [rank=0 AIC bk=31] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 37/64] [static GMM2 AIV1 bk=18] *** TIMEOUT spin rowReadyFlags loopIdx=5 cur=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 43/64] [static GMM2 AIV1 bk=21] *** TIMEOUT spin rowReadyFlags loopIdx=13 cur=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 45/64] [static GMM2 AIV1 bk=22] *** TIMEOUT spin rowReadyFlags loopIdx=11 cur=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 47/64] [static GMM2 AIV1 bk=23] *** TIMEOUT spin rowReadyFlags loopIdx=14 cur=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 49/64] [static GMM2 AIV1 bk=24] *** TIMEOUT spin rowReadyFlags loopIdx=15 cur=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 51/64] [static GMM2 AIV1 bk=25] *** TIMEOUT spin rowReadyFlags loopIdx=16 cur=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 53/64] [static GMM2 AIV1 bk=26] *** TIMEOUT spin rowReadyFlags loopIdx=17 cur=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 55/64] [static GMM2 AIV1 bk=27] *** TIMEOUT spin rowReadyFlags loopIdx=18 cur=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 57/64] [static GMM2 AIV1 bk=28] *** TIMEOUT spin rowReadyFlags loopIdx=19 cur=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 59/64] [static GMM2 AIV1 bk=29] *** TIMEOUT spin rowReadyFlags loopIdx=15 cur=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 61/64] [static GMM2 AIV1 bk=30] *** TIMEOUT spin rowReadyFlags loopIdx=16 cur=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 63/64] [static GMM2 AIV1 bk=31] *** TIMEOUT spin rowReadyFlags loopIdx=22 cur=0
[AIV Block 0/64] [rank=0 SCHED] TIMEOUT main loop exp=3
[AIV Block 0/64]   wave=0 flag=12 d1=12 d2=28
[AIV Block 0/64]   wave=1 flag=12 d1=12 d2=28
[AIV Block 0/64]   wave=2 flag=12 d1=12 d2=28
[AIV Block 0/64]   wave=3 flag=12 d1=12 d2=28
[AIV Block 0/64]   STUCK bk=1 slot_g1aic=0 _aiv=0 slot_g2aic=536888329 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=2 slot_g1aic=0 _aiv=0 slot_g2aic=536889353 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=3 slot_g1aic=0 _aiv=0 slot_g2aic=536890377 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=4 slot_g1aic=0 _aiv=0 slot_g2aic=536891401 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=6 slot_g1aic=0 _aiv=0 slot_g2aic=536893449 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=7 slot_g1aic=0 _aiv=0 slot_g2aic=536894473 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=8 slot_g1aic=0 _aiv=0 slot_g2aic=536895497 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=9 slot_g1aic=0 _aiv=0 slot_g2aic=536896521 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=11 slot_g1aic=0 _aiv=0 slot_g2aic=536898569 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=12 slot_g1aic=0 _aiv=0 slot_g2aic=536877065 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=13 slot_g1aic=0 _aiv=0 slot_g2aic=536870925 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=14 slot_g1aic=0 _aiv=0 slot_g2aic=536871949 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=15 slot_g1aic=0 _aiv=0 slot_g2aic=536872973 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=18 slot_g1aic=0 _aiv=0 slot_g2aic=536876045 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=21 slot_g1aic=0 _aiv=0 slot_g2aic=536884229 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=22 slot_g1aic=0 _aiv=0 slot_g2aic=536882177 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=23 slot_g1aic=0 _aiv=0 slot_g2aic=536885253 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=24 slot_g1aic=0 _aiv=0 slot_g2aic=536886277 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=25 slot_g1aic=0 _aiv=0 slot_g2aic=536887301 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=26 slot_g1aic=0 _aiv=0 slot_g2aic=536888325 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=27 slot_g1aic=0 _aiv=0 slot_g2aic=536889349 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=28 slot_g1aic=0 _aiv=0 slot_g2aic=536890373 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=29 slot_g1aic=0 _aiv=0 slot_g2aic=536886281 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=30 slot_g1aic=0 _aiv=0 slot_g2aic=536887305 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=31 slot_g1aic=0 _aiv=0 slot_g2aic=536893445 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 3/64] [rank=0 AIV1 bk=1] *** TIMEOUT spin slot exp=3 seq=4
[AIV Block 5/64] [rank=0 AIV1 bk=2] *** TIMEOUT spin slot exp=3 seq=4
[AIV Block 7/64] [rank=0 AIV1 bk=3] *** TIMEOUT spin slot exp=3 seq=4
[AIV Block 9/64] [rank=0 AIV1 bk=4] *** TIMEOUT spin slot exp=3 seq=4
[AIV Block 13/64] [rank=0 AIV1 bk=6] *** TIMEOUT spin slot exp=3 seq=4
[AIV Block 15/64] [rank=0 AIV1 bk=7] *** TIMEOUT spin slot exp=3 seq=4
[AIV Block 17/64] [rank=0 AIV1 bk=8] *** TIMEOUT spin slot exp=3 seq=4
[AIV Block 19/64] [rank=0 AIV1 bk=9] *** TIMEOUT spin slot exp=3 seq=4
[AIV Block 23/64] [rank=0 AIV1 bk=11] *** TIMEOUT spin slot exp=3 seq=5
[AIV Block 25/64] [rank=0 AIV1 bk=12] *** TIMEOUT spin slot exp=3 seq=4
[AIV Block 27/64] [rank=0 AIV1 bk=13] *** TIMEOUT spin slot exp=3 seq=5
[AIV Block 29/64] [rank=0 AIV1 bk=14] *** TIMEOUT spin slot exp=3 seq=5
[AIV Block 31/64] [rank=0 AIV1 bk=15] *** TIMEOUT spin slot exp=3 seq=5
[AIV Block 37/64] [rank=0 AIV1 bk=18] *** TIMEOUT spin slot exp=3 seq=5
[AIV Block 43/64] [rank=0 AIV1 bk=21] *** TIMEOUT spin slot exp=3 seq=3
[AIV Block 45/64] [rank=0 AIV1 bk=22] *** TIMEOUT spin slot exp=3 seq=2
[AIV Block 47/64] [rank=0 AIV1 bk=23] *** TIMEOUT spin slot exp=3 seq=3
[AIV Block 49/64] [rank=0 AIV1 bk=24] *** TIMEOUT spin slot exp=3 seq=3
[AIV Block 51/64] [rank=0 AIV1 bk=25] *** TIMEOUT spin slot exp=3 seq=3
[AIV Block 53/64] [rank=0 AIV1 bk=26] *** TIMEOUT spin slot exp=3 seq=3
[AIV Block 55/64] [rank=0 AIV1 bk=27] *** TIMEOUT spin slot exp=3 seq=3
[AIV Block 57/64] [rank=0 AIV1 bk=28] *** TIMEOUT spin slot exp=3 seq=3
[AIV Block 59/64] [rank=0 AIV1 bk=29] *** TIMEOUT spin slot exp=3 seq=4
[AIV Block 61/64] [rank=0 AIV1 bk=30] *** TIMEOUT spin slot exp=3 seq=4
[AIV Block 63/64] [rank=0 AIV1 bk=31] *** TIMEOUT spin slot exp=3 seq=3
[AIC Block 6/32] [static GMM2 AIC bk=6] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 6/32] [static GMM2 AIC bk=6] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 6/32] [static GMM2 AIC bk=6] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 6/32] [static GMM2 AIC bk=6] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 7/32] [static GMM2 AIC bk=7] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 7/32] [static GMM2 AIC bk=7] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 7/32] [static GMM2 AIC bk=7] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 7/32] [static GMM2 AIC bk=7] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 8/32] [rank=0 AIC bk=8] *** TIMEOUT idle exp=1 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
[AIC Block 8/32] [static GMM2 AIC bk=8] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 8/32] [static GMM2 AIC bk=8] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 8/32] [static GMM2 AIC bk=8] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 8/32] [static GMM2 AIC bk=8] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 9/32] [rank=0 AIC bk=9] *** TIMEOUT idle exp=1 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
[AIC Block 9/32] [static GMM2 AIC bk=9] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 9/32] [static GMM2 AIC bk=9] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 9/32] [static GMM2 AIC bk=9] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 9/32] [static GMM2 AIC bk=9] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 10/32] [rank=0 AIC bk=10] *** TIMEOUT idle exp=1 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
[AIC Block 10/32] [static GMM2 AIC bk=10] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 10/32] [static GMM2 AIC bk=10] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 10/32] [static GMM2 AIC bk=10] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 10/32] [static GMM2 AIC bk=10] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 11/32] [rank=0 AIC bk=11] *** TIMEOUT idle exp=1 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
[AIC Block 11/32] [static GMM2 AIC bk=11] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 11/32] [static GMM2 AIC bk=11] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 11/32] [static GMM2 AIC bk=11] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 11/32] [static GMM2 AIC bk=11] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 12/32] [rank=0 AIC bk=12] *** TIMEOUT idle exp=1 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
[AIC Block 12/32] [static GMM2 AIC bk=12] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 12/32] [static GMM2 AIC bk=12] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 12/32] [static GMM2 AIC bk=12] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 12/32] [static GMM2 AIC bk=12] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 13/32] [rank=0 AIC bk=13] *** TIMEOUT idle exp=1 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
[AIC Block 13/32] [static GMM2 AIC bk=13] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 13/32] [static GMM2 AIC bk=13] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 13/32] [static GMM2 AIC bk=13] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 13/32] [static GMM2 AIC bk=13] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 14/32] [rank=0 AIC bk=14] *** TIMEOUT idle exp=1 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
[AIC Block 14/32] [static GMM2 AIC bk=14] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 14/32] [static GMM2 AIC bk=14] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 14/32] [static GMM2 AIC bk=14] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 14/32] [static GMM2 AIC bk=14] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 15/32] [rank=0 AIC bk=15] *** TIMEOUT idle exp=1 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
[AIC Block 15/32] [static GMM2 AIC bk=15] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 15/32] [static GMM2 AIC bk=15] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 15/32] [static GMM2 AIC bk=15] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIV Block 0/64] [rank=0 SCHED] TIMEOUT main loop exp=0
[AIV Block 0/64]   wave=0 flag=12 d1=12 d2=28
[AIV Block 0/64]   wave=1 flag=12 d1=12 d2=28
[AIV Block 0/64]   wave=2 flag=12 d1=12 d2=28
[AIV Block 0/64]   wave=3 flag=12 d1=12 d2=28
[AIV Block 0/64]   STUCK bk=6 slot_g1aic=0 _aiv=0 slot_g2aic=-2 _aiv=0 e1s=1 e1a=1 e2s=1 e2a=0
[AIV Block 0/64]   STUCK bk=7 slot_g1aic=0 _aiv=0 slot_g2aic=-2 _aiv=0 e1s=1 e1a=1 e2s=1 e2a=0
[AIV Block 13/64] [static GMM2 AIV1 bk=6] *** TIMEOUT spin rowReadyFlags loopIdx=16 cur=0
[AIV Block 13/64] [static GMM2 AIV1 bk=6] *** TIMEOUT spin rowReadyFlags loopIdx=19 cur=0
[AIV Block 13/64] [static GMM2 AIV1 bk=6] *** TIMEOUT spin rowReadyFlags loopIdx=18 cur=0
[AIV Block 13/64] [static GMM2 AIV1 bk=6] *** TIMEOUT spin rowReadyFlags loopIdx=21 cur=0
[AIV Block 15/64] [static GMM2 AIV1 bk=7] *** TIMEOUT spin rowReadyFlags loopIdx=17 cur=0
[AIV Block 15/64] [static GMM2 AIV1 bk=7] *** TIMEOUT spin rowReadyFlags loopIdx=20 cur=0
[AIV Block 15/64] [static GMM2 AIV1 bk=7] *** TIMEOUT spin rowReadyFlags loopIdx=19 cur=0
[AIV Block 15/64] [static GMM2 AIV1 bk=7] *** TIMEOUT spin rowReadyFlags loopIdx=22 cur=0
[AIV Block 17/64] [static GMM2 AIV1 bk=8] *** TIMEOUT spin rowReadyFlags loopIdx=21 cur=0
[AIV Block 17/64] [static GMM2 AIV1 bk=8] *** TIMEOUT spin rowReadyFlags loopIdx=22 cur=0
[AIV Block 17/64] [static GMM2 AIV1 bk=8] *** TIMEOUT spin rowReadyFlags loopIdx=20 cur=0
[AIV Block 17/64] [static GMM2 AIV1 bk=8] *** TIMEOUT spin rowReadyFlags loopIdx=23 cur=0
[AIV Block 19/64] [static GMM2 AIV1 bk=9] *** TIMEOUT spin rowReadyFlags loopIdx=22 cur=0
[AIV Block 19/64] [static GMM2 AIV1 bk=9] *** TIMEOUT spin rowReadyFlags loopIdx=23 cur=0
[AIV Block 19/64] [static GMM2 AIV1 bk=9] *** TIMEOUT spin rowReadyFlags loopIdx=21 cur=0
[AIV Block 19/64] [static GMM2 AIV1 bk=9] *** TIMEOUT spin rowReadyFlags loopIdx=24 cur=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 21/64] [static GMM2 AIV1 bk=10] *** TIMEOUT spin rowReadyFlags loopIdx=23 cur=0
[AIV Block 21/64] [static GMM2 AIV1 bk=10] *** TIMEOUT spin rowReadyFlags loopIdx=24 cur=0
[AIV Block 21/64] [static GMM2 AIV1 bk=10] *** TIMEOUT spin rowReadyFlags loopIdx=22 cur=0
[AIV Block 21/64] [static GMM2 AIV1 bk=10] *** TIMEOUT spin rowReadyFlags loopIdx=25 cur=0
[AIV Block 23/64] [static GMM2 AIV1 bk=11] *** TIMEOUT spin rowReadyFlags loopIdx=24 cur=0
[AIV Block 23/64] [static GMM2 AIV1 bk=11] *** TIMEOUT spin rowReadyFlags loopIdx=25 cur=0
[AIV Block 23/64] [static GMM2 AIV1 bk=11] *** TIMEOUT spin rowReadyFlags loopIdx=23 cur=0
[AIV Block 23/64] [static GMM2 AIV1 bk=11] *** TIMEOUT spin rowReadyFlags loopIdx=26 cur=0
[AIV Block 25/64] [static GMM2 AIV1 bk=12] *** TIMEOUT spin rowReadyFlags loopIdx=25 cur=0
[AIV Block 25/64] [static GMM2 AIV1 bk=12] *** TIMEOUT spin rowReadyFlags loopIdx=26 cur=0
[AIV Block 25/64] [static GMM2 AIV1 bk=12] *** TIMEOUT spin rowReadyFlags loopIdx=24 cur=0
[AIV Block 25/64] [static GMM2 AIV1 bk=12] *** TIMEOUT spin rowReadyFlags loopIdx=27 cur=0
[AIV Block 27/64] [static GMM2 AIV1 bk=13] *** TIMEOUT spin rowReadyFlags loopIdx=26 cur=0
[AIV Block 27/64] [static GMM2 AIV1 bk=13] *** TIMEOUT spin rowReadyFlags loopIdx=27 cur=0
[AIV Block 27/64] [static GMM2 AIV1 bk=13] *** TIMEOUT spin rowReadyFlags loopIdx=25 cur=0
[AIV Block 27/64] [static GMM2 AIV1 bk=13] *** TIMEOUT spin rowReadyFlags loopIdx=0 cur=0
[AIV Block 29/64] [static GMM2 AIV1 bk=14] *** TIMEOUT spin rowReadyFlags loopIdx=0 cur=0
[AIV Block 29/64] [static GMM2 AIV1 bk=14] *** TIMEOUT spin rowReadyFlags loopIdx=23 cur=0
[AIV Block 29/64] [static GMM2 AIV1 bk=14] *** TIMEOUT spin rowReadyFlags loopIdx=26 cur=0
[AIV Block 29/64] [static GMM2 AIV1 bk=14] *** TIMEOUT spin rowReadyFlags loopIdx=1 cur=0
[AIV Block 31/64] [static GMM2 AIV1 bk=15] *** TIMEOUT spin rowReadyFlags loopIdx=27 cur=0
[AIV Block 31/64] [static GMM2 AIV1 bk=15] *** TIMEOUT spin rowReadyFlags loopIdx=2 cur=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 17/32] [rank=0 AIC bk=17] *** TIMEOUT idle exp=1 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
[AIC Block 17/32] [static GMM2 AIC bk=17] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 17/32] [static GMM2 AIC bk=17] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 18/32] [rank=0 AIC bk=18] *** TIMEOUT idle exp=1 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
[AIC Block 18/32] [static GMM2 AIC bk=18] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
[AIC Block 18/32] [static GMM2 AIC bk=18] *** TIMEOUT swiglu->gmm2 flag target=12 cur=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 35/64] [static GMM2 AIV1 bk=17] *** TIMEOUT spin rowReadyFlags loopIdx=1 cur=0
[AIV Block 35/64] [static GMM2 AIV1 bk=17] *** TIMEOUT spin rowReadyFlags loopIdx=4 cur=0
[AIV Block 37/64] [static GMM2 AIV1 bk=18] *** TIMEOUT spin rowReadyFlags loopIdx=2 cur=0
[AIV Block 37/64] [static GMM2 AIV1 bk=18] *** TIMEOUT spin rowReadyFlags loopIdx=5 cur=0
[AIC Block 1/32] [rank=0 AIC bk=1] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
[AIC Block 2/32] [rank=0 AIC bk=2] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
[AIC Block 3/32] [rank=0 AIC bk=3] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
[AIC Block 4/32] [rank=0 AIC bk=4] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 5/32] [rank=0 AIC bk=5] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
[AIC Block 6/32] [rank=0 AIC bk=6] *** TIMEOUT idle exp=3 g1End=0 g2End=1 g1=0 g2=0 g1Pending=1
[AIC Block 8/32] [rank=0 AIC bk=8] *** TIMEOUT idle exp=3 g1End=0 g2End=1 g1=0 g2=0 g1Pending=1
[AIC Block 9/32] [rank=0 AIC bk=9] *** TIMEOUT idle exp=3 g1End=0 g2End=1 g1=0 g2=0 g1Pending=1
[AIC Block 10/32] [rank=0 AIC bk=10] *** TIMEOUT idle exp=3 g1End=0 g2End=1 g1=0 g2=0 g1Pending=1
[AIC Block 11/32] [rank=0 AIC bk=11] *** TIMEOUT idle exp=3 g1End=0 g2End=1 g1=0 g2=0 g1Pending=1
[AIC Block 12/32] [rank=0 AIC bk=12] *** TIMEOUT idle exp=3 g1End=0 g2End=1 g1=0 g2=0 g1Pending=1
[AIC Block 13/32] [rank=0 AIC bk=13] *** TIMEOUT idle exp=3 g1End=0 g2End=1 g1=0 g2=0 g1Pending=1
[AIC Block 14/32] [rank=0 AIC bk=14] *** TIMEOUT idle exp=3 g1End=0 g2End=1 g1=0 g2=0 g1Pending=1
[AIC Block 15/32] [rank=0 AIC bk=15] *** TIMEOUT idle exp=3 g1End=0 g2End=1 g1=0 g2=0 g1Pending=1
[AIV Block 0/64] [rank=0 SCHED] TIMEOUT main loop exp=2
[AIV Block 0/64]   wave=0 flag=12 d1=12 d2=28
[AIV Block 0/64]   wave=1 flag=12 d1=12 d2=28
[AIV Block 0/64]   wave=2 flag=12 d1=12 d2=28
[AIV Block 0/64]   wave=3 flag=12 d1=12 d2=28
[AIV Block 0/64]   STUCK bk=7 slot_g1aic=-2 _aiv=0 slot_g2aic=0 _aiv=0 e1s=1 e1a=0 e2s=1 e2a=1
[AIV Block 13/64] [static GMM2 AIV1 bk=6] *** TIMEOUT spin rowReadyFlags loopIdx=0 cur=0
[AIV Block 15/64] [static GMM2 AIV1 bk=7] *** TIMEOUT spin rowReadyFlags loopIdx=1 cur=0
[AIV Block 17/64] [static GMM2 AIV1 bk=8] *** TIMEOUT spin rowReadyFlags loopIdx=2 cur=0
[AIV Block 19/64] [static GMM2 AIV1 bk=9] *** TIMEOUT spin rowReadyFlags loopIdx=3 cur=0
[AIV Block 21/64] [static GMM2 AIV1 bk=10] *** TIMEOUT spin rowReadyFlags loopIdx=4 cur=0
[AIV Block 23/64] [static GMM2 AIV1 bk=11] *** TIMEOUT spin rowReadyFlags loopIdx=5 cur=0
[AIV Block 25/64] [static GMM2 AIV1 bk=12] *** TIMEOUT spin rowReadyFlags loopIdx=6 cur=0
[AIV Block 27/64] [static GMM2 AIV1 bk=13] *** TIMEOUT spin rowReadyFlags loopIdx=7 cur=0
[AIV Block 29/64] [static GMM2 AIV1 bk=14] *** TIMEOUT spin rowReadyFlags loopIdx=8 cur=0
[AIV Block 31/64] [static GMM2 AIV1 bk=15] *** TIMEOUT spin rowReadyFlags loopIdx=9 cur=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 16/32] [rank=0 AIC bk=16] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
[AIC Block 17/32] [rank=0 AIC bk=17] *** TIMEOUT idle exp=3 g1End=0 g2End=1 g1=0 g2=0 g1Pending=1
[AIC Block 18/32] [rank=0 AIC bk=18] *** TIMEOUT idle exp=3 g1End=0 g2End=1 g1=0 g2=0 g1Pending=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 19/32] [rank=0 AIC bk=19] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 20/32] [rank=0 AIC bk=20] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
[AIC Block 21/32] [rank=0 AIC bk=21] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 22/32] [rank=0 AIC bk=22] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
[AIC Block 23/32] [rank=0 AIC bk=23] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
[AIC Block 24/32] [rank=0 AIC bk=24] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
[AIC Block 25/32] [rank=0 AIC bk=25] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
[AIC Block 26/32] [rank=0 AIC bk=26] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
[AIC Block 27/32] [rank=0 AIC bk=27] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
[AIC Block 28/32] [rank=0 AIC bk=28] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
[AIC Block 29/32] [rank=0 AIC bk=29] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
[AIC Block 30/32] [rank=0 AIC bk=30] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
[AIC Block 31/32] [rank=0 AIC bk=31] *** TIMEOUT idle exp=3 g1End=1 g2End=0 g1=0 g2=0 g1Pending=1
[AIV Block 35/64] [static GMM2 AIV1 bk=17] *** TIMEOUT spin rowReadyFlags loopIdx=11 cur=0
[AIV Block 37/64] [static GMM2 AIV1 bk=18] *** TIMEOUT spin rowReadyFlags loopIdx=12 cur=0
[AIV Block 0/64] [rank=0 SCHED] TIMEOUT main loop exp=3
[AIV Block 0/64]   wave=0 flag=11 d1=12 d2=0
[AIV Block 0/64]   wave=1 flag=11 d1=12 d2=0
[AIV Block 0/64]   wave=2 flag=12 d1=12 d2=28
[AIV Block 0/64]   wave=3 flag=2 d1=12 d2=0
[AIV Block 0/64]   STUCK bk=1 slot_g1aic=0 _aiv=0 slot_g2aic=0 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=2 slot_g1aic=0 _aiv=0 slot_g2aic=0 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=3 slot_g1aic=0 _aiv=0 slot_g2aic=0 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=4 slot_g1aic=0 _aiv=0 slot_g2aic=0 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=5 slot_g1aic=0 _aiv=0 slot_g2aic=0 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=6 slot_g1aic=536870924 _aiv=0 slot_g2aic=536870921 _aiv=0 e1s=0 e1a=0 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=7 slot_g1aic=536877056 _aiv=0 slot_g2aic=536871945 _aiv=0 e1s=0 e1a=0 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=8 slot_g1aic=536871948 _aiv=0 slot_g2aic=536872969 _aiv=0 e1s=0 e1a=0 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=9 slot_g1aic=536872972 _aiv=0 slot_g2aic=536873993 _aiv=0 e1s=0 e1a=0 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=10 slot_g1aic=536873996 _aiv=0 slot_g2aic=536875017 _aiv=0 e1s=0 e1a=0 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=11 slot_g1aic=536875020 _aiv=0 slot_g2aic=536876041 _aiv=0 e1s=0 e1a=0 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=12 slot_g1aic=536876044 _aiv=0 slot_g2aic=536877065 _aiv=0 e1s=0 e1a=0 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=13 slot_g1aic=536877068 _aiv=0 slot_g2aic=536878089 _aiv=0 e1s=0 e1a=0 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=14 slot_g1aic=536878092 _aiv=0 slot_g2aic=536879113 _aiv=0 e1s=0 e1a=0 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=15 slot_g1aic=536879116 _aiv=0 slot_g2aic=536880137 _aiv=0 e1s=0 e1a=0 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=16 slot_g1aic=0 _aiv=0 slot_g2aic=0 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=17 slot_g1aic=536881164 _aiv=0 slot_g2aic=536882185 _aiv=0 e1s=0 e1a=0 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=18 slot_g1aic=536876036 _aiv=0 slot_g2aic=536883209 _aiv=0 e1s=0 e1a=0 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=19 slot_g1aic=0 _aiv=0 slot_g2aic=0 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=20 slot_g1aic=0 _aiv=0 slot_g2aic=0 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=21 slot_g1aic=0 _aiv=0 slot_g2aic=0 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=22 slot_g1aic=0 _aiv=0 slot_g2aic=0 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=23 slot_g1aic=0 _aiv=0 slot_g2aic=0 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=24 slot_g1aic=0 _aiv=0 slot_g2aic=0 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=25 slot_g1aic=0 _aiv=0 slot_g2aic=0 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=26 slot_g1aic=0 _aiv=0 slot_g2aic=0 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=27 slot_g1aic=0 _aiv=0 slot_g2aic=0 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=28 slot_g1aic=0 _aiv=0 slot_g2aic=0 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=29 slot_g1aic=0 _aiv=0 slot_g2aic=0 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=30 slot_g1aic=0 _aiv=0 slot_g2aic=0 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 0/64]   STUCK bk=31 slot_g1aic=0 _aiv=0 slot_g2aic=0 _aiv=0 e1s=1 e1a=1 e2s=0 e2a=0
[AIV Block 3/64] [rank=0 AIV1 bk=1] *** TIMEOUT spin slot exp=3 seq=2
[AIV Block 5/64] [rank=0 AIV1 bk=2] *** TIMEOUT spin slot exp=3 seq=2
[AIV Block 7/64] [rank=0 AIV1 bk=3] *** TIMEOUT s[2026-05-21 11:03:32] [ERROR] [68700] profiler.py: Call step failed. Exception: npuSynchronizeDevice:../torch_npu/csrc/core/npu/NPUStream.cpp:576 NPU function error: AclrtSynchronizeDeviceWithTimeout, error code is 507014
[ERROR] 2026-05-21-11:03:32 (PID:68700, Device:1, RankID:-1) ERR00100 PTA call acl api failed
[Error]: The aicore execution times out. 
        Rectify the fault based on the error information in the ascend log.
EZ9999: Inner Error!
EZ9999[PID: 68700] 2026-05-21-11:03:31.862.928 (EZ9999):  The error from device(chipId:0, dieId:0), serial number is 71, there is an aicore error exception, core id is 0, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0x7ffeff7d1f9cadff,0x9dfc6f5dd800ffff, mte error info: 0x3beffdff0007cffd, vec error info: 0, cube error info: 0, l1 error info: 0xfddf001ff52f, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
        TraceBack (most recent call last):
       The extend info: errcode:(0) errorStr: timeout or trap error. subErrType: 0x4.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:643]
       The error from device(chipId:0, dieId:0), serial number is 71, there is an aicore error exception, core id is 1, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xf7f3bf7f1b9acfed,0xc5fefcf1b800eb7f, mte error info: 0x8fc7ed7900037bbe, vec error info: 0, cube error info: 0, l1 error info: 0xb7bf000a9874, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 71, there is an aicore error exception, core id is 2, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xfabbffff13de29e6,0x7bfd7d4f78007fff, mte error info: 0xebbed7950007b7bf, vec error info: 0, cube error info: 0, l1 error info: 0xfe9b001ffddb, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 71, there is an aicore error exception, core id is 3, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xfffffeff19f8ad8f,0xc63995fde800bfbd, mte error info: 0x8f18c1fd0001cf8f, vec error info: 0, cube error info: 0, l1 error info: 0xfffb00178f7f, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 71, there is an aicore error exception, core id is 4, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0x97ffffef047eefd7,0xdff5d7ddf800ffd7, mte error info: 0xafde967f0007eff5, vec error info: 0, cube error info: 0, l1 error info: 0xbefb00139df5, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 71, there is an aicore error exception, core id is 5, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xf7ffffef1dfcecfe,0xcbcfedf6f800fffb, mte error info: 0x9ffffc7a00073dfd, vec error info: 0, cube error info: 0, l1 error info: 0xfdbf0017d4c4, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 71, there is an aicore error exception, core id is 6, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xffffffef0cf8bedf,0xfdfbdfeff800efff, mte error info: 0xdffff2ee0007bfff, vec error info: 0, cube error info: 0, l1 error info: 0xdebf000f77ef, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 71, there is an aicore error exception, core id is 7, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xffffd7e5158efead,0x4a5cfc2ff800deb7, mte error info: 0x9dffefff00029ffb, vec error info: 0, cube error info: 0, l1 error info: 0x5393000fffff, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 71, there is an aicore error exception, core id is 8, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xffbfb7ff1ba8f1de,0xffffe6f0f000ffef, mte error info: 0xbfad9ccc0003bff1, vec error info: 0, cube error info: 0, l1 error info: 0xfcb5000d7b5c, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 71, there is an aicore error exception, core id is 9, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0x5fe5ffaf1a4effbb,0xf7feedd7b000fdff, mte error info: 0xd065f0fc0006f3ab, vec error info: 0, cube error info: 0, l1 error info: 0xdbff001fc9cd, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 71, there is an aicore error exception, core id is 10, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xffff7cff16e4cff6,0x9f9dc71fd800fffe, mte error info: 0xfcf5d77f0007f2bb, vec error info: 0, cube error info: 0, l1 error info: 0xbfcb0002f55e, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 71, there is an aicore error exception, core id is 11, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xfff7afff19feff7c,0x7f97977d7800ffff, mte error info: 0x8b3c0eff0007b6b2, vec error info: 0, cube error info: 0, l1 error info: 0x7fbd0003f4de, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 71, there is an aicore error exception, core id is 12, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xfffffff91d98f79f,0x6f1feaccf8007ffd, mte error info: 0x83cccdfb0006a3d2, vec error info: 0, cube error info: 0, l1 error info: 0xbdfe0015f55d, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 71, there is an aicore error exception, core id is 13, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xfdeffffd0ef0aff9,0xcdf4ff6af800dfff, mte error info: 0xfbbcdfcf0007fbb3, vec error info: 0, cube error info: 0, l1 error info: 0xbfb9001a9f77, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 71, there is an aicore error exception, core id is 14, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xffff7f981e3cedfd,0xb4fbe5ed7800fffd, mte error info: 0x8a0320e60004ffff, vec error info: 0, cube error info: 0, l1 error info: 0xffbf001e9fd2, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 71, there is an aicore error exception, core id is 15, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xfe77ffa319ee3fdc,0xf5daed5678005d7e, mte error info: 0xe3efcf5b0003efff, vec error info: 0, cube error info: 0, l1 error info: 0xd9e30005f3ec, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 71, there is an aicore error exception, core id is 19, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xffffffff1f32a8f7,0xb3dfd9d1f800fbff, mte error info: 0xeee924e900074cbf, vec error info: 0, cube error info: 0, l1 error info: 0x9fdb00032f77, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 71, there is an aicore error exception, core id is 20, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xffbffffd193afffb,0xa4dfdf7db800bfff, mte error info: 0xfbd7fdee0002ffff, vec error info: 0, cube error info: 0, l1 error info: 0xd9bf0017e70c, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 71, there is an aicore error exception, core id is 21, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xfffffefb0212fdfa,0xfdfd1fd4f800ffbf, mte error info: 0x242df4ab00022ade, vec error info: 0, cube error info: 0, l1 error info: 0x16f800170df9, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 71, there is an aicore error exception, core id is 22, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xfb3ffffd1ee0ffe5,0xcdf7ed1cf800ffff, mte error info: 0x9db5c5c50001eb3d, vec error info: 0, cube error info: 0, l1 error info: 0x5fb8001d7ddd, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 71, there is an aicore error exception, core id is 23, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xfb3fff9513bafe2b,0x2bdf2dedf000dfff, mte error info: 0xffff7bce0002bf1b, vec error info: 0, cube error info: 0, l1 error info: 0xf38f000d8dc4, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 71, there is an aicore error exception, core id is 24, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xe7ffefed0fceadda,0x6b576d5ef800ffff, mte error info: 0x9be0f0be0004fbfb, vec error info: 0, cube error info: 0, l1 error info: 0xdbbd000f9ed4, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 71, there is an aicore error exception, core id is 25, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xdffffff512bcff1e,0xdc9ebd8cf800ffff, mte error info: 0x4f3996ed0007e1b3, vec error info: 0, cube error info: 0, l1 error info: 0xfbbb001b8cd0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 71, there is an aicore error exception, core id is 26, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xfbfdfeff093a7f60,0xe5fbbd99f8007fff, mte error info: 0x3efd8db500002a6a, vec error info: 0, cube error info: 0, l1 error info: 0x8edf001b6ef7, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 71, there is an aicore error exception, core id is 27, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0x7ffffeff1b8afcfc,0xcdedd6f5b800ffff, mte error info: 0x5fdd958e0007d7b3, vec error info: 0, cube error info: 0, l1 error info: 0x7fe0001329cd, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 71, there is an aicore error exception, core id is 28, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xffdfffef12a6bd9d,0xa8ddbd1ff800f7ff, mte error info: 0x71b7d7dc0004cdf6, vec error info: 0, cube error info: 0, l1 error info: 0xdddf001b7fdf, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 71, there is an aicore error exception, core id is 29, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xffffdefd1a489bdf,0x359d9c4fe800ffff, mte error info: 0xdbe0a9db0007fb36, vec error info: 0, cube error info: 0, l1 error info: 0xfdff00075bff, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 71, there is an aicore error exception, core id is 30, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xfffffffd03a6cdc7,0xfdddf6ecf800ffbf, mte error info: 0x5e1b87dc0005de5f, vec error info: 0, cube error info: 0, l1 error info: 0x97df001ef8d6, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 71, there is an aicore error exception, core id is 31, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xffbff5ff1fbe9ffc,0xeee197c0f800f5ff, mte error info: 0x6579edb900073fee, vec error info: 0, cube error info: 0, l1 error info: 0x7fd400167dc9, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 71, there is an aicore error exception, core id is 32, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xffefd4fb1fbefcff,0x9fdeb68be800dd7f, mte error info: 0xfa34cff40007bee6, vec error info: 0, cube error info: 0, l1 error info: 0xd5090017fdf5, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 71, there is an aicore error exception, core id is 33, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xfeff7fff1dbe3098,0xefef6e7cf800ffff, mte error info: 0x9d1ff3220000d7bb, vec error info: 0, cube error info: 0, l1 error info: 0x92ad001a3fdd, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 71, there is an aicore error exception, core id is 34, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xfeffffee1f9ed1fe,0xfbfa99edf800fffb, mte error info: 0x7f8ccff80001bbda, vec error info: 0, cube error info: 0, l1 error info: 0xb9bb001f5fcf, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 0, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b04, sc error info: 0xffffffffffff, su error info: 0xbfaffa3802fe94c3,0xdd1df7fcf800ff7c, mte error info: 0x25d12, vec error info: 0xf717bfc5007defda, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 1, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0x3f5753fb2dbe931d,0xfecffbf57800d7f4, mte error info: 0x840427a00002931d, vec error info: 0x7f9fba6d007bbfc7, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 2, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xfff8e7843bbe94c3,0xfcfdc2fc5000f95f, mte error info: 0x25d12, vec error info: 0xb373afe400643d0f, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 3, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xee7fd7dd13d2931d,0x166d1cfd6800ddf7, mte error info: 0x84052fa00002931d, vec error info: 0xff37dbde005771cf, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 4, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0x33fe87ee369e94c3,0x3777c6ed5000f6f9, mte error info: 0x25d12, vec error info: 0x477d89dd0074bfec, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 5, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xfffb66bf33a8931d,0xdf4da7ffb000ffdf, mte error info: 0x840637a00002931d, vec error info: 0x7d7ebfdf007dfef5, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 6, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xba88eee331ea94c3,0xcbcc5f79980036eb, mte error info: 0x25d12, vec error info: 0x7fc2742400318bc7, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 7, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xc9efafa5213e931d,0xd6f5da45c8002f9a, mte error info: 0x84073fa00002931d, vec error info: 0x517f78780009f5c7, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 8, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0x7bffde75029c94c3,0xd0993afff800f5ff, mte error info: 0x25d12, vec error info: 0x7728effe005471e7, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 9, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0x75dd7bdf19f8931d,0x17e6641ff800ffff, mte error info: 0x840847a00002931d, vec error info: 0x367fbad500527fff, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 10, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xf7f7f7ec1ffe94c3,0x705fc75d7800923b, mte error info: 0x25d12, vec error info: 0x7fc2742400318bce, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 11, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xfd7eb7f03bde931d,0xff31c3f3e000d7df, mte error info: 0x84094fa00002931d, vec error info: 0x37fe997d007fbdff, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 12, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0x74bdbbcd18de94c3,0xc1ddd1df0007ffe, mte error info: 0x25d12, vec error info: 0x7fc2742000318bc7, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 13, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xbd667fb91cba931d,0x7caf6f5d2800fedd, mte error info: 0x840a57a00002931d, vec error info: 0x7f3ffa5d00743d9c, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 14, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xfef3fbee16b094c3,0xf5d5d5dde800fbeb, mte error info: 0x25d12, vec error info: 0x7fc2742400318bc7, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 15, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xf10f7f6d3b8e931d,0x7ace9fd7a000a3f7, mte error info: 0x840b5fa00002931d, vec error info: 0xff7df5fe007efffc, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 16, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xefff7db62a3c94c3,0xfeddf1ffe80057fc, mte error info: 0x25d12, vec error info: 0x7fc2742400318bce, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 17, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0x3fe7ff9c3fbc931d,0xcda7423b9000b3dd, mte error info: 0x840c67a00002931d, vec error info: 0xbd5fbdff0038f9f7, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 18, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xcffffe791fca94c3,0xffdff7dfb000a79f, mte error info: 0x25d12, vec error info: 0x7fc2742000318bce, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 19, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0x7fffdf7d3f9e931d,0x5fd1f0f76800fafe, mte error info: 0x840d6fa00002931d, vec error info: 0x7dd3ebd8007cddb9, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 20, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xdf3b9e3037fa94c3,0xfbdfc7f17800c5d3, mte error info: 0x25d12, vec error info: 0x7fc2742000318bc7, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 21, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xfbfdbffc3ad2931d,0x36cd964d0800fbff, mte error info: 0x840e77a00002931d, vec error info: 0x6716ddff007f5e4c, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 22, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0x82ff62db22d894c3,0x8edf9f5ed800bfdf, mte error info: 0x25d12, vec error info: 0xdf07b88f006dc9dd, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 23, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xffcdf7b01f3c931d,0xf17c51d3c800ffff, mte error info: 0x840f7fa00002931d, vec error info: 0x6f439af7005c5ffc, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 24, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xddeb7bcf2a3094c3,0xfcdedd9f78006a53, mte error info: 0x25d12, vec error info: 0x7fc2742000318bce, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 25, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xfebbefdb3aec931d,0x9bf5cdf73800db4e, mte error info: 0x841087a00002931d, vec error info: 0x73fabbe9001d6fdd, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 26, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xefed70f93bfe94c3,0xfcf7d7ffd800ffbf, mte error info: 0x25d12, vec error info: 0x7fc2742000318bc7, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 27, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0x7fbf579c39d8931d,0xff97fffff8007dea, mte error info: 0x84118fa00002931d, vec error info: 0xf7f79f8f007dd2cc, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 28, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xffff5ffb34de94c3,0xfffdf7d5d800eff7, mte error info: 0x25d12, vec error info: 0x7fc2742000318bc7, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 29, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xb8c1bd2c2fdc931d,0xef6d7ee41000fe6b, mte error info: 0x841297a00002931d, vec error info: 0xd7935f170079bdf7, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 30, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xf97fe57f393694c3,0x7769f7fde800fbcf, mte error info: 0x25d12, vec error info: 0x7fc2742000318bc7, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 31, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xfebfcedd1eca931d,0xd67f608768007a77, mte error info: 0x84139fa00002931d, vec error info: 0xfdb7f795007f7fce, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 38, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0x1e11cff43a4e94c3,0xfdfd13f9d000ebf3, mte error info: 0x25d12, vec error info: 0x7fc2742400318bce, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 39, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xb71e8bdf11de931d,0xe3cdc3d878007746, mte error info: 0x931d, vec error info: 0x3daf8ad1006df8d5, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 40, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xffffd9ff17d694c3,0xec9defff800ffdf, mte error info: 0x25d12, vec error info: 0x7fc2742000318bce, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 41, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xffffefef3dfc931d,0x2bfa1e7ff800dfdf, mte error info: 0x931d, vec error info: 0xfdffffff007efffb, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 42, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0x61822eff1fb094c3,0xd9fd507dd800c8bb, mte error info: 0x25d12, vec error info: 0x7fc2742000318bce, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 43, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0x2c46827c03cc931d,0xafb0d5b53800b7c0, mte error info: 0x931d, vec error info: 0xb3eabb0f007f9c7a, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 44, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0x6bb26187107894c3,0x6ec1de775800bdb5, mte error info: 0x25d12, vec error info: 0x7fc2742000318bce, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 45, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0x15ef6a6b38fc931d,0xffd1d2ec480073ef, mte error info: 0x931d, vec error info: 0xcbfaa8c7003ffa87, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 46, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xedafafda2cb894c3,0xf5bffeae8800afec, mte error info: 0x25d12, vec error info: 0x7fc2742400318bce, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 47, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xe4fdfc3604fa931d,0xdcde167d2800f18e, mte error info: 0x931d, vec error info: 0x2df43afc006be64c, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 48, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0x9eb7ecbc23c694c3,0xcecce15fe000f275, mte error info: 0x25d12, vec error info: 0x7fc2742000318bce, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 49, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xd1cfdef435b8931d,0x1f6c5ebf8005c2e, mte error info: 0x931d, vec error info: 0xfd3de7cf0038f5cf, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 50, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xb1f0f6b732a294c3,0xcf8fef331000f3af, mte error info: 0x25d12, vec error info: 0x7fc2742000318bce, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 51, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0x399e0f7831fc931d,0xfed3d1c7080073a8, mte error info: 0x931d, vec error info: 0x84e7a3ef000f4549, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 52, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0x6f9fa7762a9c94c3,0xd7abd357f800dc7e, mte error info: 0x25d12, vec error info: 0x7fc2742000318bc7, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 53, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xdeff5fa7098e931d,0x3e97fbf7c0007cff, mte error info: 0x931d, vec error info: 0x29b7afdd0072f71c, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 54, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xf38c15f5329894c3,0xafdfcdd8d800e54a, mte error info: 0x25d12, vec error info: 0x7fc2742000318bce, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 55, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xf7d7fdb82bf8931d,0x7952d5fce80075c7, mte error info: 0x931d, vec error info: 0x7f37bb5f00794dfe, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 56, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xb6e67dfb31f294c3,0xe76f449fd800c5ff, mte error info: 0x25d12, vec error info: 0x7fc2742000318bce, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 57, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xae9f8f0d199a931d,0xdcd798fdf8004fef, mte error info: 0x931d, vec error info: 0x3377fddf007cd4a7, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 58, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0x6f8bf049031e94c3,0xe706744cc8007eda, mte error info: 0x25d12, vec error info: 0x7fc2742400318bce, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 59, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xaffd649b1ffc931d,0x7e43af99a000df43, mte error info: 0x931d, vec error info: 0x26b9fffc0073deed, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 60, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xc6ffdf1e29da94c3,0xffcfe3ff7800acff, mte error info: 0x25d12, vec error info: 0x7fc2742000318bc7, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 61, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0x2d1dafba1df0931d,0x2bc41e50e80056a0, mte error info: 0x931d, vec error info: 0xff1dfffd001ef7ef, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 62, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0x9d7cb31f1d3494c3,0x4c4f654cd80041fe, mte error info: 0x25d12, vec error info: 0x7fc2742000318bce, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 63, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xd6ff3f2913f8931d,0x566dee7db80053a4, mte error info: 0x931d, vec error info: 0x3717279d006defc7, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 64, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0x60fcfc7107f094c3,0xc2df9df77000fd7f, mte error info: 0x25d12, vec error info: 0x7fc2742400318bce, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 65, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xc7b8eb3c0a26931d,0x98edd9c1a800dfde, mte error info: 0x931d, vec error info: 0x3b31fdd8007a6f7a, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 66, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xb0fa7cff1d4c94c3,0x4fdfccdfa800fd3d, mte error info: 0x25d12, vec error info: 0x7fc2742000318bc7, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 67, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xbf7dfdbb21c0931d,0xdc7645bcd800def4, mte error info: 0x931d, vec error info: 0x110c9813000e637c, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 68, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0x6cfe85cd19bc94c3,0x8c476cf1d0007fef, mte error info: 0x25d12, vec error info: 0x7fc2742000318bc7, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 72, there is an aivec error exception, core id is 69, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe95b40, sc error info: 0xffffffffffff, su error info: 0xcfb59de219b6931d,0xdab49ce9f8003d9c, mte error info: 0x931d, vec error info: 0xdf71df5b003d79fe, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       An error occurred in the kernel task, retCode=0x25, [aicore timeout].[FUNC:PreCheckTaskErr][FILE:davinci_kernel_task.cc][LINE:1267]
       rtDeviceSynchronizeWithTimeout execution failed, reason=aicore timeout[FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:65]
       wait for compute device to finish failed, runtime result = 507014.[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:148]

[2026-05-21 11:03:32] [ERROR] [68384] profiler.py: Call step failed. Exception: npuSynchronizeDevice:../torch_npu/csrc/core/npu/NPUStream.cpp:576 NPU function error: AclrtSynchronizeDeviceWithTimeout, error code is 507014
[ERROR] 2026-05-21-11:03:32 (PID:68384, Device:0, RankID:-1) ERR00100 PTA call acl api failed
[Error]: The aicore execution times out. 
        Rectify the fault based on the error information in the ascend log.
EZ9999: Inner Error!
EZ9999[PID: 68384] 2026-05-21-11:03:32.045.645 (EZ9999):  The error from device(chipId:0, dieId:0), serial number is 75, there is an aicore error exception, core id is 0, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0x7ff1efff07f47d8b,0xb7cd44fcf800fffe, mte error info: 0xa6b3c6530003e9da, vec error info: 0, cube error info: 0, l1 error info: 0x5ffb00075f84, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
        TraceBack (most recent call last):
       The extend info: errcode:(0) errorStr: timeout or trap error. subErrType: 0x4.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:643]
       The error from device(chipId:0, dieId:0), serial number is 75, there is an aicore error exception, core id is 1, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xfbb7ffe71f70bf85,0x7ff7bdddf800f7ff, mte error info: 0xdbdd380f0007faf6, vec error info: 0, cube error info: 0, l1 error info: 0x932f000b2db7, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 75, there is an aicore error exception, core id is 3, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xfeafddff1ab8fdff,0xc98dd1c1f800fdfe, mte error info: 0x5d3b08fd0003ebf8, vec error info: 0, cube error info: 0, l1 error info: 0x5fbf0009d99f, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 75, there is an aicore error exception, core id is 4, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xeffffefb0fdab73f,0xf992e898f800ffe7, mte error info: 0x8b79e4be0002ef7a, vec error info: 0, cube error info: 0, l1 error info: 0x619f0012e7f9, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 75, there is an aicore error exception, core id is 5, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xff95bf7f167effba,0x9f9d1bbff800dff7, mte error info: 0xdfcdf3ff00063da7, vec error info: 0, cube error info: 0, l1 error info: 0xfc9c0013be3b, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 75, there is an aicore error exception, core id is 6, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xf67ffee91c289eaf,0xd8f4dccbf8007ff7, mte error info: 0xfbfeefdf0003dfbf, vec error info: 0, cube error info: 0, l1 error info: 0xfffd0007fffe, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 75, there is an aicore error exception, core id is 7, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xfffff7df1bdadedf,0xf79f445de800ff7f, mte error info: 0xc2c40bee0000e7a8, vec error info: 0, cube error info: 0, l1 error info: 0xda9300017c89, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 75, there is an aicore error exception, core id is 8, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xfefff9fe1ddaaf1f,0x5ed7c452d800ffff, mte error info: 0x5e2db4fd0007beff, vec error info: 0, cube error info: 0, l1 error info: 0xdf1d001fdd9e, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 75, there is an aicore error exception, core id is 9, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xfec7fffe19d2afd9,0xd70721fae000effc, mte error info: 0xfad7e64e0007eb7f, vec error info: 0, cube error info: 0, l1 error info: 0xcfbe000dddcb, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 75, there is an aicore error exception, core id is 10, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xbf7fefe70766bfff,0xdcf9efb1f800ffff, mte error info: 0x5f91e5df0007deff, vec error info: 0, cube error info: 0, l1 error info: 0xa79f001b520f, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 75, there is an aicore error exception, core id is 11, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xfbffedfd1bfe3e27,0x17c0a4dd78006fdf, mte error info: 0xf9e2ff2000041faf, vec error info: 0, cube error info: 0, l1 error info: 0xda9e0015e043, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 75, there is an aicore error exception, core id is 12, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xfef7cfed13defeec,0x6df07cfdf8007ffe, mte error info: 0xe99fb78d0007db9f, vec error info: 0, cube error info: 0, l1 error info: 0xbe53001bddfd, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 75, there is an aicore error exception, core id is 14, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xff9dedff1f7efdbb,0xd77dfbdbf8003cfd, mte error info: 0xd7dffffe0004ffff, vec error info: 0, cube error info: 0, l1 error info: 0x6fff001d7fe5, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 75, there is an aicore error exception, core id is 15, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xfedf2ff30d7cbfef,0xdce9ffe9f800ffbf, mte error info: 0xfa2dd3dd0002a5ca, vec error info: 0, cube error info: 0, l1 error info: 0xf85a0017bf1e, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 75, there is an aicore error exception, core id is 16, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xf7dffffd0e7afb5b,0xfdf3ffe1f800ffff, mte error info: 0x3bd0e7dc0003ff3f, vec error info: 0, cube error info: 0, l1 error info: 0xfd7a0006df0f, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 75, there is an aicore error exception, core id is 17, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xf93fffff13b88bbc,0x5edefffff000f9ff, mte error info: 0x593e3fff0007fe56, vec error info: 0, cube error info: 0, l1 error info: 0x7fd300134fc1, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 75, there is an aicore error exception, core id is 18, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xfeff6ff90f84bcf2,0xecf974f4f800bfff, mte error info: 0xdcddf7f900055fbe, vec error info: 0, cube error info: 0, l1 error info: 0xbfab0006e5fc, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 75, there is an aicore error exception, core id is 19, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xfffffbf717fedddf,0xff63f78d78007eff, mte error info: 0x7aeffddf0007e4fd, vec error info: 0, cube error info: 0, l1 error info: 0xeb9b00173e85, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 75, there is an aicore error exception, core id is 20, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xffffdcfb1c2afeed,0x1ecbefd57800afff, mte error info: 0xfbfef3ff0000bfff, vec error info: 0, cube error info: 0, l1 error info: 0xfde70019797f, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 75, there is an aicore error exception, core id is 21, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xfef7fbf51f66cf9f,0x9ffff91f7800fbfd, mte error info: 0xf7be1eed0004f082, vec error info: 0, cube error info: 0, l1 error info: 0x1aff000a5779, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 75, there is an aicore error exception, core id is 22, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xd7ffffe71bec3bbd,0x6e7ccbdc78007fff, mte error info: 0xabf8fffc0000abab, vec error info: 0, cube error info: 0, l1 error info: 0xdebf0015fff9, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 75, there is an aicore error exception, core id is 24, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xffbf7fff13fa2eeb,0xfff56cfc7800fd9e, mte error info: 0xa2ccdfcf000507ff, vec error info: 0, cube error info: 0, l1 error info: 0xb91f0003e7cc, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 75, there is an aicore error exception, core id is 25, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xfbb5e7ef11eed4ef,0x7c3fdedef000bffd, mte error info: 0x87ca282e0005feff, vec error info: 0, cube error info: 0, l1 error info: 0xff9e001f9efa, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 75, there is an aicore error exception, core id is 26, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0x5ffffdfe1c3edec3,0x93fd47dde800ffff, mte error info: 0xe1c79c7d0005ef3f, vec error info: 0, cube error info: 0, l1 error info: 0x8fff0015e175, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 75, there is an aicore error exception, core id is 27, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xfffeffde1ababff4,0xe537a3fde800dbfd, mte error info: 0xf9b7241d0003b7f7, vec error info: 0, cube error info: 0, l1 error info: 0xba57001e5d9e, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 75, there is an aicore error exception, core id is 28, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xffb37ffb01ecfdee,0x5dcf6ff7f800fffc, mte error info: 0x678f7f7e0007bffb, vec error info: 0, cube error info: 0, l1 error info: 0xf8f9001ff5fe, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 75, there is an aicore error exception, core id is 29, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xfffffffe16fa73ff,0x1f1d41fbb800f7ff, mte error info: 0x3a3fd1df0000f2dc, vec error info: 0, cube error info: 0, l1 error info: 0xbd97001fdcdf, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 75, there is an aicore error exception, core id is 30, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xbbffff7113f2f6bb,0x94f9f7777000eeff, mte error info: 0xf3effbff0002eebd, vec error info: 0, cube error info: 0, l1 error info: 0x5bab001e74d4, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 75, there is an aicore error exception, core id is 31, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0x7bbbbf9f1ff2cebd,0xf3f599d9d000d7df, mte error info: 0x3b1fc9fd0002f7a7, vec error info: 0, cube error info: 0, l1 error info: 0xbfe70006d7ff, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 75, there is an aicore error exception, core id is 32, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xffdffff70f8cffdf,0x9cfcf58ff800dffe, mte error info: 0x8e6ef5f80001ee8c, vec error info: 0, cube error info: 0, l1 error info: 0xa753001d1cf0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 75, there is an aicore error exception, core id is 33, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0xfe7f7d741d9ef9ff,0x45fdeecaf000ffff, mte error info: 0xf8ccfbdf0005ffff, vec error info: 0, cube error info: 0, l1 error info: 0xdab90013b9fd, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 75, there is an aicore error exception, core id is 35, error code = 0, dump info: pc start: 0x10037fe13288, current: 0x10037fe2564c, sc error info: 0xffffffffffff, su error info: 0x7dbff7ef0930cef7,0xf9ffcd14f800ff5e, mte error info: 0x1bf8c57c00076312, vec error info: 0, cube error info: 0, l1 error info: 0x7db7001ee8df, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 0, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0x1bfbffb833fe94c3,0xcfbef5df0005d1f, mte error info: 0x23040, vec error info: 0x3bdc7f3e002c8cf5, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 1, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0xe66febff1ffa931d,0xd7edd0db8800a97f, mte error info: 0x93d00d600002931d, vec error info: 0x77bfb75d007f47c9, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 2, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0xb6fe2fcc1fb694c3,0xe677f386d8007b99, mte error info: 0x23040, vec error info: 0x7f7ffbdc0075efa0, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 3, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0x7f7f1fb21bba931d,0xf1d729f02800da8a, mte error info: 0x93d00c800002931d, vec error info: 0x3ff6f9ff0076d9f9, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 6, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0x4e83b4ee049094c3,0xf5d1defc28002c7c, mte error info: 0x23040, vec error info: 0x7fc2742400318bce, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 7, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0x2a7777eb10e2931d,0x3fcdffdde000f9ec, mte error info: 0x93d000800002931d, vec error info: 0xd55223f2007d0fde, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 8, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0x5eb7dfd338f894c3,0xdc21555fd800f43a, mte error info: 0x23040, vec error info: 0x7fc2742400318bce, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 9, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0xf2fd7fcd3cca931d,0x7d9d9cc5b800b7e5, mte error info: 0x93d000900002931d, vec error info: 0xf758fddf003cd55f, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 10, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0xac35fbbf3fee94c3,0xb47fdbddd800fbde, mte error info: 0x23040, vec error info: 0x7fc2742400318bce, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 11, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0xeeff9cdd3bbe931d,0xcc5dd50ec000f7fa, mte error info: 0x93d000a00002931d, vec error info: 0x2fa7dffd0075decf, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 12, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0x7fdff1ff3ffc94c3,0xffffe37ff800e7f3, mte error info: 0x23040, vec error info: 0x7fc2742000318bce, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 13, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0xf7ffdfc327f8931d,0xd555f69f4000eedd, mte error info: 0x93d000b00002931d, vec error info: 0x7d25bfef007dffcb, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 14, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe71eec, sc error info: 0xffffffffffff, su error info: 0xa89f6aeb23bc94c3,0xc1f19a6dc0008327, mte error info: 0x23040, vec error info: 0x7dfcb1fb005fc577, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 15, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0xd3e6fef11db0931d,0x564be7f57800cabf, mte error info: 0x93d032600002931d, vec error info: 0xe7adbb40007bcb79, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 16, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe71eec, sc error info: 0xffffffffffff, su error info: 0xb2f3d38f320294c3,0xfd07f0af4000f9f7, mte error info: 0x23040, vec error info: 0x7fc2742000318bce, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 17, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0xfff73f7e3db8931d,0xfd7d6fb75800f38f, mte error info: 0x93d032700002931d, vec error info: 0x7f738d9e007a6f5d, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 18, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe71eec, sc error info: 0xffffffffffff, su error info: 0xb61fb5ff13ae94c3,0xddfdf4fec80097af, mte error info: 0x23040, vec error info: 0x7fc2742400318bce, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 19, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0xf3bfaebf3b1a931d,0xc96d35c9e000777d, mte error info: 0x93d032800002931d, vec error info: 0xf76bfbe7003d76ff, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 20, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe71eec, sc error info: 0xffffffffffff, su error info: 0x9af2dffd2f3894c3,0xdff9cddd7800bf98, mte error info: 0x23040, vec error info: 0x7fc2742000318bce, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 21, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0xe8ff5b7633fc931d,0xefe95f49f8005fef, mte error info: 0x93d032900002931d, vec error info: 0x7f31bb5d0078d789, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 22, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe71eec, sc error info: 0xffffffffffff, su error info: 0xc3efee7f3ff494c3,0xfd75dff850002dde, mte error info: 0x23040, vec error info: 0x7fc2742000318bce, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 23, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0x399f1f973b78931d,0x7efa30f568005767, mte error info: 0x93d032a00002931d, vec error info: 0xdf9ff3df0075fbde, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 24, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe71eec, sc error info: 0xffffffffffff, su error info: 0xfeef57e01e3894c3,0x57fe749d8000fff, mte error info: 0x23040, vec error info: 0x7fc2742000318bce, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 25, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0xfff93dec1f50931d,0x7755efdf9800d5b7, mte error info: 0x93d032b00002931d, vec error info: 0xbff7fafc004f7d7e, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 28, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe71eec, sc error info: 0xffffffffffff, su error info: 0xffaf4d770f8e94c3,0xbfffffeef800dffb, mte error info: 0x23040, vec error info: 0x7fc2742000318bc7, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 29, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0xfdfe9fdf1ffe931d,0xcf7f66fff800df72, mte error info: 0x93d0b2600002931d, vec error info: 0xf73a9b5f0028c844, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 30, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe71eec, sc error info: 0xffffffffffff, su error info: 0xd1fef79d09c894c3,0xdef7f9fe1000e3dd, mte error info: 0x23040, vec error info: 0x7fa2742000318bc7, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 31, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0xabfd9fd528bc931d,0x50bd7ab7b80056df, mte error info: 0x93d0b2700002931d, vec error info: 0xdfcdb7dc0057d647, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 32, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe71eec, sc error info: 0xffffffffffff, su error info: 0xffbdfabb2fb694c3,0x6c0f0dbff800fbfd, mte error info: 0x23040, vec error info: 0x7fc2742400318bce, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 33, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0x82bfd4003bba931d,0xdf676dde680035c3, mte error info: 0x93d0b2800002931d, vec error info: 0x94f5bb790074f376, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 34, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe71eec, sc error info: 0xffffffffffff, su error info: 0xbdf9585305fe94c3,0x7f94f7dfe800bbe5, mte error info: 0x23040, vec error info: 0x7fc2742400318bce, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 35, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0xba67e77f1c78931d,0xc7ab9cf578004ed9, mte error info: 0x93d0b2900002931d, vec error info: 0x57e1d5fd0076cd5e, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 36, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0x72d7d7ed2b8e94c3,0x5174da71e800b77b, mte error info: 0x23040, vec error info: 0x7fc2742000318bc7, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 37, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0xf75fce3e19d8931d,0xf6f7f3dc90003de9, mte error info: 0x93d0b2a00002931d, vec error info: 0xef75afde005dffbf, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 38, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe71eec, sc error info: 0xffffffffffff, su error info: 0xed59ef7f0af094c3,0x2ccdfff14800efd7, mte error info: 0x23040, vec error info: 0x7fc2742000318bce, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 39, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0x7eddef6f0d7c931d,0x5e7f57132800ffec, mte error info: 0x93d0b2b00002931d, vec error info: 0xf927cfff004f9fdd, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 40, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe71eec, sc error info: 0xffffffffffff, su error info: 0x33fdffdf3ffe94c3,0x7edfbeffe800fdbf, mte error info: 0x23040, vec error info: 0x7fa2742000318bc7, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 41, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0x6fd9fdfe3dfc931d,0xf7ef72fff800efee, mte error info: 0x93d132600002931d, vec error info: 0xff0ba5780033e485, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 42, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0x4ddfb3ef3d3c94c3,0x2f5933bf0800de5c, mte error info: 0x23040, vec error info: 0x7fc2742000318bce, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 43, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0xfe89eb5f3b92931d,0xbf9c272b600071cb, mte error info: 0x93d132700002931d, vec error info: 0xfaffbbdb005d5bf4, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 44, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0xd7e6ca3c014294c3,0xcde9d1fd48009f0d, mte error info: 0x23040, vec error info: 0x7fc2742000318bce, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 45, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0xaffdfe793e7c931d,0xfffefdf8c800c3be, mte error info: 0x93d132800002931d, vec error info: 0xb7ffb4c7007f2aff, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 48, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0x75ffcfe60ffe94c3,0xf7ec17dfe8003ff9, mte error info: 0x23040, vec error info: 0x7fc2742000318bc7, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 49, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0xedd2f66d3f32931d,0x5dc7c9dfb8007bbd, mte error info: 0x93d132900002931d, vec error info: 0xf635db0d0058857f, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 50, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0x66d7e18c20fc94c3,0xe0ffffff1800fc69, mte error info: 0x23040, vec error info: 0x7fc2742000318bce, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 51, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0x7cd1d32f31b0931d,0xf3abecb71800ddef, mte error info: 0x93d132a00002931d, vec error info: 0xf3f78fdc007bffd5, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 52, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0x9ddf9ee917f894c3,0x981f8b7e9000fe75, mte error info: 0x23040, vec error info: 0x7fc2742400318bc7, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 53, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0xf1ff7dee19bc931d,0xf6df02ccf800b7ef, mte error info: 0x93d132b00002931d, vec error info: 0xe17a79c50078434d, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 54, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0xd696fbbd1fee94c3,0xfcffdd6cf8007f78, mte error info: 0x23040, vec error info: 0x7fa2742000318bc7, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 55, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0x6e7ebfcd067e931d,0xbfdd8508d0007df6, mte error info: 0x93d1b2600002931d, vec error info: 0x79ff2b910035bffd, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 56, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0xb5f7effe1fb894c3,0x7f77d177f800778d, mte error info: 0x2309d, vec error info: 0x7fc2742000318bc7, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 57, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0xffd3f70f3dfe931d,0x767770bf2800ffff, mte error info: 0x93d1b2700002931d, vec error info: 0xe5f209d600655dd0, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 58, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0xefffffcd3ffe94c3,0x910de5f17000b7dd, mte error info: 0x23040, vec error info: 0x7fc2742400318bc7, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 59, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0x9afff1321ef0931d,0xcdde971ce8008d7d, mte error info: 0x93d1b2800002931d, vec error info: 0xd513f9fd003c9ed8, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 60, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0xe5ddf4653b8e94c3,0xe964fcdbc800baff, mte error info: 0x23040, vec error info: 0x7fc2742400318bc7, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 61, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0x815f7bdf3ffe931d,0xffe7ffff8800f8fb, mte error info: 0x93d1b2900002931d, vec error info: 0xdfe39df900362ffd, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 62, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0x6f787fe01ffa94c3,0x7cddebd778003381, mte error info: 0x23040, vec error info: 0x7fc2742000318bc7, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 63, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0xebbf2fff0e68931d,0x6d77982ff800a6ff, mte error info: 0x93d1b2a00002931d, vec error info: 0x333e3df7003cdfff, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 64, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0xfaf2bea603f494c3,0xddcbf439e000e7e9, mte error info: 0x23040, vec error info: 0x7fc2742000318bc7, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 65, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0x1add4b4d2a18931d,0x74677ff5f8009bf7, mte error info: 0x93d1b2b00002931d, vec error info: 0xc717bc79007eddc7, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 66, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0x8e9cffb632b094c3,0x758fc7dd58007edb, mte error info: 0x23040, vec error info: 0x7fc2742400318bc7, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 67, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0xfdd767fe33be931d,0xffdddd3428009a7c, mte error info: 0x2931d, vec error info: 0x479beb9c007dddce, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 70, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0x6af7dbf41b3894c3,0x715ef3fdd800e7f3, mte error info: 0x23040, vec error info: 0x7fc2742000318bc7, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       The error from device(chipId:0, dieId:0), serial number is 76, there is an aivec error exception, core id is 71, error code = 0, dump info: pc start: 0x10037fe5fc00, current: 0x10037fe6b65c, sc error info: 0xffffffffffff, su error info: 0xcf9f7d1527ba931d,0xba4877f8c000e2fb, mte error info: 0x2931d, vec error info: 0x36fba7d6007f2cff, cube error info: 0, l1 error info: 0, aic error mask: 0x395856, para base: 0x10036e62ddd8, mte error: 0.[FUNC:ProcessDavidStarsCoreErrorInfo][FILE:device_error_proc_c.cc][LINE:640]
       An error occurred in the kernel task, retCode=0x25, [aicore timeout].[FUNC:PreCheckTaskErr][FILE:davinci_kernel_task.cc][LINE:1267]
       rtDeviceSynchronizeWithTimeout execution failed, reason=aicore timeout[FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:65]
       wait for compute device to finish failed, runtime result = 507014.[FUNC:ReportCallError][FILE:log_inner.cpp][LINE:148]

pin slot exp=3 seq=1
[AIV Block 9/64] [rank=0 AIV1 bk=4] *** TIMEOUT spin slot exp=3 seq=1
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 11/64] [rank=0 AIV1 bk=5] *** TIMEOUT spin slot exp=3 seq=1
[AIV Block 13/64] [rank=0 AIV1 bk=6] *** TIMEOUT spin slot exp=3 seq=2
[AIV Block 15/64] [rank=0 AIV1 bk=7] *** TIMEOUT spin slot exp=3 seq=2
[AIV Block 17/64] [rank=0 AIV1 bk=8] *** TIMEOUT spin slot exp=3 seq=2
[AIV Block 19/64] [rank=0 AIV1 bk=9] *** TIMEOUT spin slot exp=3 seq=2
[AIV Block 21/64] [rank=0 AIV1 bk=10] *** TIMEOUT spin slot exp=3 seq=2
[AIV Block 23/64] [rank=0 AIV1 bk=11] *** TIMEOUT spin slot exp=3 seq=2
[AIV Block 25/64] [rank=0 AIV1 bk=12] *** TIMEOUT spin slot exp=3 seq=2
[AIV Block 27/64] [rank=0 AIV1 bk=13] *** TIMEOUT spin slot exp=3 seq=2
[AIV Block 29/64] [rank=0 AIV1 bk=14] *** TIMEOUT spin slot exp=3 seq=2
[AIV Block 31/64] [rank=0 AIV1 bk=15] *** TIMEOUT spin slot exp=3 seq=2
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 33/64] [rank=0 AIV1 bk=16] *** TIMEOUT spin slot exp=3 seq=2
[AIV Block 35/64] [rank=0 AIV1 bk=17] *** TIMEOUT spin slot exp=3 seq=2
[AIV Block 37/64] [rank=0 AIV1 bk=18] *** TIMEOUT spin slot exp=3 seq=2
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 39/64] [rank=0 AIV1 bk=19] *** TIMEOUT spin slot exp=3 seq=2
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 41/64] [rank=0 AIV1 bk=20] *** TIMEOUT spin slot exp=3 seq=2
[AIV Block 43/64] [rank=0 AIV1 bk=21] *** TIMEOUT spin slot exp=3 seq=2
[AIV Block 45/64] [rank=0 AIV1 bk=22] *** TIMEOUT spin slot exp=3 seq=2
[AIV Block 47/64] [rank=0 AIV1 bk=23] *** TIMEOUT spin slot exp=3 seq=2
[AIV Block 49/64] [rank=0 AIV1 bk=24] *** TIMEOUT spin slot exp=3 seq=2
[AIV Block 51/64] [rank=0 AIV1 bk=25] *** TIMEOUT spin slot exp=3 seq=2
[AIV Block 53/64] [rank=0 AIV1 bk=26] *** TIMEOUT spin slot exp=3 seq=2
[AIV Block 55/64] [rank=0 AIV1 bk=27] *** TIMEOUT spin slot exp=3 seq=2
[AIV Block 57/64] [rank=0 AIV1 bk=28] *** TIMEOUT spin slot exp=3 seq=2
[AIV Block 59/64] [rank=0 AIV1 bk=29] *** TIMEOUT spin slot exp=3 seq=2
[AIV Block 61/64] [rank=0 AIV1 bk=30] *** TIMEOUT spin slot exp=3 seq=2
[AIV Block 63/64] [rank=0 AIV1 bk=31] *** TIMEOUT spin slot exp=3 seq=2







*************************************

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
<npu_ops_transformer.ops.mega_moe.SymmBuffer object at 0x7f9a775a0a30>
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
<npu_ops_transformer.ops.mega_moe.SymmBuffer object at 0x7fea01cf8a30>
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
[AIC Block 25/32] [rank=0 AIC bk=25] G2#1 w=0 nT=6 waveM=256
[AIC Block 25/32] [rank=0 AIC bk=25] G2#2 w=1 nT=9 waveM=256
[AIC Block 25/32] [rank=0 AIC bk=25] G2#3 w=2 nT=12 waveM=256
[AIC Block 25/32] [rank=0 AIC bk=25] G2 END seen
[AIC Block 25/32] [rank=0 AIC bk=25] final WaitForVector(0) drain
[AIC Block 25/32] [rank=0 AIC bk=25] <<< EXIT exp=1 (G1=1 G2=4)
[AIC Block 25/32] [rank=0 DISPATCH bk=25 ct=1 sub=0] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 25/32] [rank=0 AIC bk=25] >>> ENTRY exp=2 tnG1=12 tnG2=28
[AIC Block 25/32] [rank=0 AIC bk=25] G1#1 w=2 nT=0 waveM=256
[AIC Block 25/32] [rank=0 AIC bk=25] G1 END seen
[AIC Block 25/32] [rank=0 AIC bk=25] G2#1 w=0 nT=17 waveM=256
[AIC Block 25/32] [rank=0 AIC bk=25] G2#2 w=1 nT=20 waveM=256
[AIC Block 25/32] [rank=0 AIC bk=25] G2#3 w=2 nT=23 waveM=256
[AIC Block 25/32] [rank=0 AIC bk=25] G2 END seen
[AIC Block 25/32] [rank=0 AIC bk=25] final WaitForVector(0) drain
[AIC Block 25/32] [rank=0 AIC bk=25] <<< EXIT exp=2 (G1=1 G2=4)
[AIC Block 25/32] [rank=0 DISPATCH bk=25 ct=1 sub=0] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 25/32] [rank=0 AIC bk=25] >>> ENTRY exp=3 tnG1=12 tnG2=28
[AIC Block 25/32] [rank=0 AIC bk=25] G1#1 w=2 nT=0 waveM=256
[AIC Block 25/32] [rank=0 AIC bk=25] G1 END seen
[AIC Block 25/32] [rank=0 AIC bk=25] CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 0/32] [rank=1 DISPATCH bk=0 ct=1 sub=0] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 0/32] [rank=1 DISPATCH bk=0 ct=1 sub=0] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 0/32] [rank=1 DISPATCH bk=0 ct=1 sub=0] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 0/32] [rank=1 DISPATCH bk=0 ct=1 sub=0] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 0/64] [rank=1 DISPATCH bk=0 ct=2 sub=0] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 0/64] [rank=1 SCHED] >>> ENTRY exp=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024 blockNum=32
[AIV Block 0/64] [rank=1 SCHED] <<< EXIT exp=0
[AIV Block 0/64] [rank=1 DISPATCH bk=0 ct=2 sub=0] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 0/64] [rank=1 SCHED] >>> ENTRY exp=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024 blockNum=32
[AIV Block 0/64] [rank=1 SCHED] <<< EXIT exp=1
[AIV Block 0/64] [rank=1 DISPATCH bk=0 ct=2 sub=0] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 0/64] [rank=1 SCHED] >>> ENTRY exp=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024 blockNum=32
[AIV Block 0/64] [rank=1 SCHED] <<< EXIT exp=2
[AIV Block 0/64] [rank=1 DISPATCH bk=0 ct=2 sub=0] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 0/64] [rank=1 SCHED] >>> ENTRY exp=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024 blockNum=32
[AIV Block 0/64] [rank=1 SCHED] <<< EXIT exp=3
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 1/64] [rank=1 DISPATCH bk=0 ct=2 sub=1] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 1/64] [rank=1 DISPATCH bk=0 ct=2 sub=1] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 1/64] [rank=1 DISPATCH bk=0 ct=2 sub=1] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 1/64] [rank=1 DISPATCH bk=0 ct=2 sub=1] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 21/64] [static GMM2 AIV1 bk=10] *** TIMEOUT spin rowReadyFlags loopIdx=21 cur=0
[AIV Block 21/64] [static GMM2 AIV1 bk=10] *** TIMEOUT spin rowReadyFlags loopIdx=4 cur=0
[AIV Block 21/64] [static GMM2 AIV1 bk=10] *** TIMEOUT spin rowReadyFlags loopIdx=8 cur=0
CANN Version: 1.0.0, TimeStamp: 202604
[AIC Block 25/32] [rank=1 DISPATCH bk=25 ct=1 sub=0] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 25/32] [rank=1 AIC bk=25] >>> ENTRY exp=0 tnG1=12 tnG2=28
[AIC Block 25/32] [rank=1 AIC bk=25] G1#1 w=2 nT=0 waveM=256
[AIC Block 25/32] [rank=1 AIC bk=25] G1 END seen
[AIC Block 25/32] [rank=1 AIC bk=25] G2#1 w=0 nT=7 waveM=256
[AIC Block 25/32] [rank=1 AIC bk=25] G2#2 w=1 nT=10 waveM=256
[AIC Block 25/32] [rank=1 AIC bk=25] G2#3 w=2 nT=13 waveM=256
[AIC Block 25/32] [rank=1 AIC bk=25] G2 END seen
[AIC Block 25/32] [rank=1 AIC bk=25] final WaitForVector(0) drain
[AIC Block 25/32] [rank=1 AIC bk=25] <<< EXIT exp=0 (G1=1 G2=3)
[AIC Block 25/32] [rank=1 DISPATCH bk=25 ct=1 sub=0] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 25/32] [rank=1 AIC bk=25] >>> ENTRY exp=1 tnG1=12 tnG2=28
[AIC Block 25/32] [rank=1 AIC bk=25] G1#1 w=2 nT=0 waveM=256
[AIC Block 25/32] [rank=1 AIC bk=25] G1 END seen
[AIC Block 25/32] [rank=1 AIC bk=25] G2#1 w=0 nT=5 waveM=256
[AIC Block 25/32] [rank=1 AIC bk=25] G2#2 w=1 nT=8 waveM=256
[AIC Block 25/32] [rank=1 AIC bk=25] G2#3 w=2 nT=9 waveM=256
[AIC Block 25/32] [rank=1 AIC bk=25] G2 END seen
[AIC Block 25/32] [rank=1 AIC bk=25] final WaitForVector(0) drain
[AIC Block 25/32] [rank=1 AIC bk=25] <<< EXIT exp=1 (G1=1 G2=3)
[AIC Block 25/32] [rank=1 DISPATCH bk=25 ct=1 sub=0] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 25/32] [rank=1 AIC bk=25] >>> ENTRY exp=2 tnG1=12 tnG2=28
[AIC Block 25/32] [rank=1 AIC bk=25] G1#1 w=2 nT=0 waveM=256
[AIC Block 25/32] [rank=1 AIC bk=25] G1 END seen
[AIC Block 25/32] [rank=1 AIC bk=25] G2#1 w=0 nT=10 waveM=256
[AIC Block 25/32] [rank=1 AIC bk=25] G2#2 w=1 nT=13 waveM=256
[AIC Block 25/32] [rank=1 AIC bk=25] G2#3 w=2 nT=16 waveM=256
[AIC Block 25/32] [rank=1 AIC bk=25] G2 END seen
[AIC Block 25/32] [rank=1 AIC bk=25] final WaitForVector(0) drain
[AIC Block 25/32] [rank=1 AIC bk=25] <<< EXIT exp=2 (G1=1 G2=4)
[AIC Block 25/32] [rank=1 DISPATCH bk=25 ct=1 sub=0] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIC Block 25/32] [rank=1 AIC bk=25] >>> ENTRY exp=3 tnG1=12 tnG2=28
[AIC Block 25/32] [rank=1 AIC bk=25] G1#1 w=2 nT=0 waveM=256
[AIC Block 25/32] [rank=1 AIC bk=25] G1 END seen
[AIC Block 25/32] [rank=1 AIC bk=25] G2#1 w=0 nT=14 waveM=256
[AIC Block 25/32] [rank=1 AIC bk=25] G2#2 w=1 nT=17 waveM=256
[AIC Block 25/32] [rank=1 AIC bk=25] G2#3 w=2 nT=20 waveM=256
[AIC Block 25/32] [rank=1 AIC bk=25] G2 END seen
[AIC Block 25/32] [rank=1 AIC bk=25] final WaitForVector(0) drain
[AIC Block 25/32] [rank=1 AIC bk=25] <<< EXIT exp=3 (G1=1 G2=4)
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
[AIV Block 50/64] [rank=1 AIV0 bk=25] #1 w=2 nT=0
[AIV Block 50/64] [rank=1 AIV0 bk=25] <<< EXIT (tasks=1)
CANN Version: 1.0.0, TimeStamp: 202604
[AIV Block 51/64] [rank=1 DISPATCH bk=25 ct=2 sub=1] expert=0 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 51/64] [rank=1 AIV1 bk=25] >>> ENTRY exp=0
[AIV Block 51/64] [rank=1 AIV1 bk=25] #1 w=0 nT=7
[AIV Block 51/64] [rank=1 AIV1 bk=25] #2 w=1 nT=10
[AIV Block 51/64] [rank=1 AIV1 bk=25] #3 w=2 nT=13
[AIV Block 51/64] [rank=1 AIV1 bk=25] <<< EXIT (tasks=3)
[AIV Block 51/64] [rank=1 DISPATCH bk=25 ct=2 sub=1] expert=1 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 51/64] [rank=1 AIV1 bk=25] >>> ENTRY exp=1
[AIV Block 51/64] [rank=1 AIV1 bk=25] #1 w=0 nT=5
[AIV Block 51/64] [rank=1 AIV1 bk=25] #2 w=1 nT=8
[AIV Block 51/64] [rank=1 AIV1 bk=25] #3 w=2 nT=9
[AIV Block 51/64] [rank=1 AIV1 bk=25] <<< EXIT (tasks=3)
[AIV Block 51/64] [rank=1 DISPATCH bk=25 ct=2 sub=1] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 51/64] [rank=1 AIV1 bk=25] >>> ENTRY exp=2
[AIV Block 51/64] [rank=1 AIV1 bk=25] #1 w=0 nT=10
[AIV Block 51/64] [rank=1 AIV1 bk=25] #2 w=1 nT=13
[AIV Block 51/64] [rank=1 AIV1 bk=25] #3 w=2 nT=16
[AIV Block 51/64] [rank=1 AIV1 bk=25] <<< EXIT (tasks=4)
[AIV Block 51/64] [rank=1 DISPATCH bk=25 ct=2 sub=1] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 51/64] [rank=1 AIV1 bk=25] >>> ENTRY exp=3
[AIV Block 51/64] [rank=1 AIV1 bk=25] #1 w=0 nT=14
[AIV Block 51/64] [rank=1 AIV1 bk=25] #2 w=1 nT=17
[AIV Block 51/64] [rank=1 AIV1 bk=25] #3 w=2 nT=20
[AIV Block 51/64] [rank=1 AIV1 bk=25] <<< EXIT (tasks=4)
G2#1 w=0 nT=19 waveM=256
[AIC Block 25/32] [rank=0 AIC bk=25] G2#2 w=1 nT=22 waveM=256
[AIC Block 25/32] [rank=0 AIC bk=25] G2#3 w=2 nT=25 waveM=256
[AIC Block 25/32] [rank=0 AIC bk=25] G2 END seen
[AIC Block 25/32] [rank=0 AIC bk=25] final WaitForVector(0) drain
[AIC Block 25/32] [rank=0 AIC bk=25] <<< EXIT exp=3 (G1=1 G2=3)
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
[AIV Block 51/64] [rank=0 AIV1 bk=25] #1 w=0 nT=6
[AIV Block 51/64] [rank=0 AIV1 bk=25] #2 w=1 nT=9
[AIV Block 51/64] [rank=0 AIV1 bk=25] #3 w=2 nT=12
[AIV Block 51/64] [rank=0 AIV1 bk=25] <<< EXIT (tasks=4)
[AIV Block 51/64] [rank=0 DISPATCH bk=25 ct=2 sub=1] expert=2 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 51/64] [rank=0 AIV1 bk=25] >>> ENTRY exp=2
[AIV Block 51/64] [rank=0 AIV1 bk=25] #1 w=0 nT=17
[AIV Block 51/64] [rank=0 AIV1 bk=25] #2 w=1 nT=20
[AIV Block 51/64] [rank=0 AIV1 bk=25] #3 w=2 nT=23
[AIV Block 51/64] [rank=0 AIV1 bk=25] <<< EXIT (tasks=4)
[AIV Block 51/64] [rank=0 DISPATCH bk=25 ct=2 sub=1] expert=3 tnG1=12 tnG2=28 waveCnt=4 m_i=1024
[AIV Block 51/64] [rank=0 AIV1 bk=25] >>> ENTRY exp=3
[AIV Block 51/64] [rank=0 AIV1 bk=25] #1 w=0 nT=19
[AIV Block 51/64] [rank=0 AIV1 bk=25] #2 w=1 nT=22
[AIV Block 51/64] [rank=0 AIV1 bk=25] #3 w=2 nT=25
[AIV Block 51/64] [rank=0 AIV1 bk=25] <<< EXIT (tasks=3)
##################Compare X in rank 0 epId 0 tpId 0 ##################
local_rank_id2374:0
[2026/05/21 10:13:10] [INFO] ---------------------------------------------------------------------------------------
[2026/05/21 10:13:10] [INFO] Loop 	         cascade_npu_out 	             golden 	                   npu_out 	           RateDiff
[2026/05/21 10:13:10] [INFO] ---------------------------------------------------------------------------------------
[2026/05/21 10:13:10] [INFO] 00000001 	              0.2421875 	              0.2428483 	              0.2421875 	 0.0006608
[2026/05/21 10:13:10] [INFO] 00000002 	             -1.1328125 	             -1.1348476 	             -1.1328125 	 0.0017933
[2026/05/21 10:13:10] [INFO] 00000003 	              1.5234375 	              1.5227127 	              1.5234375 	 0.0007248
[2026/05/21 10:13:10] [INFO] 00000004 	              1.1796875 	              1.1770338 	              1.1796875 	 0.0022495
[2026/05/21 10:13:10] [INFO] 00000005 	             -0.5468750 	             -0.5438833 	             -0.5468750 	 0.0054705
[2026/05/21 10:13:10] [INFO] 00000006 	             -0.0456543 	             -0.0455816 	             -0.0456543 	 0.0000727
[2026/05/21 10:13:10] [INFO] 00000007 	              0.3437500 	              0.3441011 	              0.3437500 	 0.0003511
[2026/05/21 10:13:10] [INFO] 00000008 	              0.2773438 	              0.2782815 	              0.2773438 	 0.0009377
[2026/05/21 10:13:10] [INFO] 00000009 	             -0.0698242 	             -0.0695252 	             -0.0698242 	 0.0002990
[2026/05/21 10:13:10] [INFO] 00000010 	              1.0234375 	              1.0249438 	              1.0234375 	 0.0014697
[2026/05/21 10:13:10] [INFO] ...   	   ...   	   ...   	   ...    	   ...
[2026/05/21 10:13:10] [INFO] 03670007 	             -0.6328125 	             -0.6314448 	             -0.6328125 	 0.0021613
[2026/05/21 10:13:10] [INFO] 03670008 	              0.3710938 	              0.3725935 	              0.3710938 	 0.0040251
[2026/05/21 10:13:10] [INFO] 03670009 	              0.1552734 	              0.1535522 	              0.1552734 	 0.0110850
[2026/05/21 10:13:10] [INFO] 03670010 	             -0.8828125 	             -0.8840220 	             -0.8828125 	 0.0013682
[2026/05/21 10:13:10] [INFO] 03670011 	              0.4101562 	              0.4096218 	              0.4101562 	 0.0005344
[2026/05/21 10:13:10] [INFO] 03670012 	             -0.1416016 	             -0.1429626 	             -0.1416016 	 0.0095203
[2026/05/21 10:13:10] [INFO] 03670013 	              0.4199219 	              0.4195377 	              0.4199219 	 0.0003842
[2026/05/21 10:13:10] [INFO] 03670014 	             -0.1875000 	             -0.1874906 	             -0.1875000 	 0.0000094
[2026/05/21 10:13:10] [INFO] 03670015 	             -0.0869141 	             -0.0855013 	             -0.0869141 	 0.0162547
[2026/05/21 10:13:10] [INFO] 03670016 	              0.7617188 	              0.7604597 	              0.7617188 	 0.0016529
##################Compare X in rank 1 epId 1 tpId 0 ##################
local_rank_id2374:1
[2026/05/21 10:13:10] [INFO] ---------------------------------------------------------------------------------------
[2026/05/21 10:13:10] [INFO] Loop 	         cascade_npu_out 	             golden 	                   npu_out 	           RateDiff
[2026/05/21 10:13:10] [INFO] ---------------------------------------------------------------------------------------
[2026/05/21 10:13:10] [INFO] 00000001 	             -0.7617188 	             -0.7617300 	             -0.7617188 	 0.0000112
[2026/05/21 10:13:10] [INFO] 00000002 	              0.8554688 	              0.8577936 	              0.8554688 	 0.0027102
[2026/05/21 10:13:10] [INFO] 00000003 	              0.3242188 	              0.3244100 	              0.3242188 	 0.0001912
[2026/05/21 10:13:10] [INFO] 00000004 	              0.2675781 	              0.2683774 	              0.2675781 	 0.0007993
[2026/05/21 10:13:10] [INFO] 00000005 	             -0.7031250 	             -0.7012520 	             -0.7031250 	 0.0026638
[2026/05/21 10:13:10] [INFO] 00000006 	             -0.5898438 	             -0.5917263 	             -0.5898438 	 0.0031815
[2026/05/21 10:13:10] [INFO] 00000007 	             -0.1972656 	             -0.1957367 	             -0.1972656 	 0.0077505
[2026/05/21 10:13:10] [INFO] 00000008 	             -0.5390625 	             -0.5396764 	             -0.5390625 	 0.0006139
[2026/05/21 10:13:10] [INFO] 00000009 	              0.4414062 	              0.4434330 	              0.4414062 	 0.0045706
[2026/05/21 10:13:10] [INFO] 00000010 	             -0.3085938 	             -0.3079789 	             -0.3085938 	 0.0006149
[2026/05/21 10:13:10] [INFO] ...   	   ...   	   ...   	   ...    	   ...
[2026/05/21 10:13:10] [INFO] 03670007 	              0.8007812 	              0.8019012 	              0.8007812 	 0.0013966
[2026/05/21 10:13:10] [INFO] 03670008 	             -1.2656250 	             -1.2691318 	             -1.2656250 	 0.0027631
[2026/05/21 10:13:10] [INFO] 03670009 	             -0.4667969 	             -0.4652840 	             -0.4667969 	 0.0032410
[2026/05/21 10:13:10] [INFO] 03670010 	              0.1396484 	              0.1388095 	              0.1396484 	 0.0008389
[2026/05/21 10:13:10] [INFO] 03670011 	             -0.0520020 	             -0.0533563 	             -0.0520020 	 0.0253822
[2026/05/21 10:13:10] [INFO] 03670012 	              0.8437500 	              0.8435734 	              0.8437500 	 0.0001766
[2026/05/21 10:13:10] [INFO] 03670013 	              0.6640625 	              0.6656507 	              0.6640625 	 0.0023860
[2026/05/21 10:13:10] [INFO] 03670014 	             -0.5898438 	             -0.5896969 	             -0.5898438 	 0.0001468
[2026/05/21 10:13:10] [INFO] 03670015 	              0.1206055 	              0.1210413 	              0.1206055 	 0.0004358
[2026/05/21 10:13:10] [INFO] 03670016 	             -0.9609375 	             -0.9614851 	             -0.9609375 	 0.0005476
##################Compare EXPERT TOKEN NUMS##################
[2026/05/21 10:13:10] [INFO] split_count:8.0; max_diff_hd:0.0;
[2026/05/21 10:13:10] [INFO] ---------------------------------------------------------------------------------------
[2026/05/21 10:13:10] [INFO] Loop 	 ExpectOut 	 RealOut 	 FpDiff 	 RateDiff
[2026/05/21 10:13:10] [INFO] ---------------------------------------------------------------------------------------
[2026/05/21 10:13:10] [INFO] 00000001 	 1024.0000000 	 1024.0000000 	 0.0000000 	 0.0000000
[2026/05/21 10:13:10] [INFO] 00000002 	 1024.0000000 	 1024.0000000 	 0.0000000 	 0.0000000
[2026/05/21 10:13:10] [INFO] 00000003 	 1024.0000000 	 1024.0000000 	 0.0000000 	 0.0000000
[2026/05/21 10:13:10] [INFO] 00000004 	 1024.0000000 	 1024.0000000 	 0.0000000 	 0.0000000
[2026/05/21 10:13:10] [INFO] 00000005 	 1024.0000000 	 1024.0000000 	 0.0000000 	 0.0000000
[2026/05/21 10:13:10] [INFO] 00000006 	 1024.0000000 	 1024.0000000 	 0.0000000 	 0.0000000
[2026/05/21 10:13:10] [INFO] 00000007 	 1024.0000000 	 1024.0000000 	 0.0000000 	 0.0000000
[2026/05/21 10:13:10] [INFO] 00000008 	 1024.0000000 	 1024.0000000 	 0.0000000 	 0.0000000
[2026/05/21 10:13:10] [INFO] ---------------------------------------------------------------------------------------
[2026/05/21 10:13:10] [INFO] DiffThd  	 PctThd   	 PctRlt   	 Result
[2026/05/21 10:13:10] [INFO] ---------------------------------------------------------------------------------------
[2026/05/21 10:13:10] [INFO] 0.0010     	 99.90%   	 100.000000%   	 PASS
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
