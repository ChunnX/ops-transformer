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

