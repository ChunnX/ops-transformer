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