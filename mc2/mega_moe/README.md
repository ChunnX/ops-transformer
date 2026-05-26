========== [rank 0] MegaMoe TIMESTAMPS (unit = us) ==========
  active AIC : 36/36, active AIV0: 36/36, active AIV1: 36/36

  --- [AIC] STAGE durations (per-core delta) ---
    KERNEL_ENTRY -> INIT_DONE             n=  36  min=       0  avg=       0  max=       1  (us)
    INIT_DONE -> QUANT_DONE            n=  36  min=       0  avg=       0  max=       0  (us)
    QUANT_DONE -> EXPERT_LOOP_END       n=  36  min=     154  avg=     411  max=     445  (us)  <-- HOT
    EXPERT_LOOP_END -> PROCESS_END           n=  36  min=       0  avg=       0  max=       1  (us)

  --- [AIV0] STAGE durations (per-core delta) ---
    KERNEL_ENTRY -> INIT_DONE             n=  36  min=       0  avg=       0  max=       1  (us)
    INIT_DONE -> QUANT_DONE            n=  36  min=       5  avg=       6  max=       8  (us)  <-- HOT
    QUANT_DONE -> EXPERT_LOOP_END       n=  36  min=     146  avg=     390  max=     425  (us)  <-- HOT
    EXPERT_LOOP_END -> PROCESS_END           n=  36  min=      72  avg=     132  max=     382  (us)  <-- HOT

  --- [AIV1] STAGE durations (per-core delta) ---
    KERNEL_ENTRY -> INIT_DONE             n=  36  min=       0  avg=       0  max=       1  (us)
    INIT_DONE -> QUANT_DONE            n=  36  min=       6  avg=       7  max=       8  (us)  <-- HOT
    QUANT_DONE -> EXPERT_LOOP_END       n=  36  min=     197  avg=     442  max=     491  (us)  <-- HOT
    EXPERT_LOOP_END -> PROCESS_END           n=  36  min=       5  avg=      81  max=     332  (us)  <-- HOT

  --- [AIV1] per-expert DISPATCH duration ---
    expert[0] dispatch  n=  36  min=      20  avg=      24  max=      31  (us)  <-- HOT
    expert[1] dispatch  n=  36  min=      21  avg=      25  max=      28  (us)  <-- HOT
    expert[2] dispatch  n=  36  min=      19  avg=      31  max=      44  (us)  <-- HOT
    expert[3] dispatch  n=  36  min=      25  avg=      32  max=      55  (us)  <-- HOT

  --- [AIC] WAVE-by-WAVE  (us, avg across active cores) ---
    exp  wave | entr->g1B |      wDsp |    g1Frst |     g1Mid |     g1Drn |    g12gap |      wSwi |    g2Frst |     g2Mid |     g2Drn |    g2->ex |     TOTAL
    ---------------------------------------------------------------------------------------------------------------------------------------------------------
      1     0 |         0 |        20 |        27 |         0 |         0 |        21 |         0 |        14 |         0 |         0 |         0 |        77
      1     1 |         0 |         0 |        23 |         0 |         0 |        13 |         0 |        13 |         0 |         0 |         0 |        46
      2     2 |         0 |         0 |        29 |         0 |         0 |        21 |         0 |        14 |         0 |         0 |         0 |        61
      2     3 |         0 |         0 |        23 |         0 |         0 |        12 |         0 |        12 |         0 |         0 |         0 |        45
      3     4 |         0 |        35 |        27 |         0 |         0 |        21 |         0 |        14 |         0 |         0 |         0 |        91
      3     5 |         0 |         0 |        23 |         0 |         0 |        13 |         0 |        12 |         0 |         0 |         0 |        46
      3     6 |         0 |        11 |        25 |         0 |         0 |        21 |         0 |        14 |         0 |         0 |         0 |        67
      3     7 |         0 |         0 |        22 |         0 |         0 |        13 |         0 |        13 |         0 |         0 |         0 |        46
    ---------------------------------------------------------------------------------------------------------------------------------------------------------
          avg |         0 |         8 |        25 |         0 |         0 |        17 |         0 |        13 |         0 |         0 |         0 |        60
          max |         1 |        49 |        31 |         1 |         1 |        26 |         3 |        17 |         1 |         1 |         1 |       116

  --- [AIV0] WAVE-by-WAVE  (us, avg across active cores) ---
    exp  wave | entr->g1B |      wCub |     epiFr |     epiMd |     epiDr |    g1->ex |     TOTAL
    ---------------------------------------------------------------------------------------------
      1     0 |         0 |        47 |        12 |         0 |         0 |        21 |        75
      1     1 |         0 |        11 |        12 |         0 |         0 |         0 |        21
      2     2 |         0 |        29 |        11 |         0 |         0 |        22 |        58
      2     3 |         0 |        11 |        12 |         0 |         0 |         0 |        22
      3     4 |         0 |        62 |        12 |         0 |         0 |        21 |        89
      3     5 |         0 |        11 |        12 |         0 |         0 |         0 |        22
      3     6 |         0 |        36 |        12 |         0 |         0 |        21 |        65
      3     7 |         0 |        11 |        12 |         0 |         0 |         0 |        22
    ---------------------------------------------------------------------------------------------
          avg |         0 |        27 |        12 |         0 |         0 |        11 |        47
          max |         2 |        76 |        14 |         2 |         1 |        29 |       114

  --- [AIV1] WAVE-by-WAVE  (us, avg across active cores) ---
    exp  wave | entr->g2B |     wRowR |      cbFr |      cbMd |     cbEnd |    g2->ex |     TOTAL
    ---------------------------------------------------------------------------------------------
      1     0 |         1 |        19 |        41 |         0 |         0 |         0 |        55
      1     1 |        54 |         0 |        42 |         0 |         0 |         0 |        92
      2     2 |         0 |         0 |        17 |         0 |         0 |         0 |        16
      2     3 |        16 |         0 |        12 |         0 |         0 |         0 |        28
      3     4 |         0 |         3 |        13 |         0 |         0 |         0 |        15
      3     5 |        15 |         0 |        15 |         0 |         0 |         0 |        29
      3     6 |         0 |        44 |        29 |         0 |         0 |         0 |        67
      3     7 |        67 |         0 |        20 |         0 |         0 |         0 |        85
    ---------------------------------------------------------------------------------------------
          avg |        19 |         8 |        24 |         0 |         0 |         0 |        48
          max |       106 |        66 |        56 |         1 |         1 |         1 |       130
