========== [rank 0] MegaMoe TIMESTAMPS (unit = us) ==========
  active AIC : 36/36, active AIV0: 36/36, active AIV1: 36/36

  --- [AIC] STAGE durations (per-core delta) ---
    KERNEL_ENTRY -> INIT_DONE             n=  36  min=       0  avg=       0  max=       1  (us)
    INIT_DONE -> QUANT_DONE            n=  36  min=       0  avg=       0  max=       1  (us)
    QUANT_DONE -> EXPERT_LOOP_END       n=  36  min=     113  avg=     376  max=     411  (us)  <-- HOT
    EXPERT_LOOP_END -> PROCESS_END           n=  36  min=       0  avg=       0  max=       1  (us)

  --- [AIV0] STAGE durations (per-core delta) ---
    KERNEL_ENTRY -> INIT_DONE             n=  36  min=       0  avg=       0  max=       1  (us)
    INIT_DONE -> QUANT_DONE            n=  36  min=       6  avg=       6  max=       8  (us)  <-- HOT
    QUANT_DONE -> EXPERT_LOOP_END       n=  36  min=     106  avg=     355  max=     388  (us)  <-- HOT
    EXPERT_LOOP_END -> PROCESS_END           n=  36  min=      78  avg=     136  max=     379  (us)  <-- HOT

  --- [AIV1] STAGE durations (per-core delta) ---
    KERNEL_ENTRY -> INIT_DONE             n=  36  min=       0  avg=       0  max=       1  (us)
    INIT_DONE -> QUANT_DONE            n=  36  min=       6  avg=       6  max=       8  (us)  <-- HOT
    QUANT_DONE -> EXPERT_LOOP_END       n=  36  min=     122  avg=     406  max=     456  (us)  <-- HOT
    EXPERT_LOOP_END -> PROCESS_END           n=  36  min=      18  avg=      86  max=     361  (us)  <-- HOT

  --- [AIV1] per-expert DISPATCH duration ---
    expert[0] sendInfoCal  n=  36  min=      10  avg=      24  max=      31  (us)  <-- HOT
    expert[0] tripleDisp   n=  36  min=      13  avg=      23  max=      36  (us)  <-- HOT
    expert[0] TOTAL        n=  36  min=      31  avg=      48  max=      66  (us)  <-- HOT
    expert[1] sendInfoCal  n=  36  min=       7  avg=      10  max=      17  (us)  <-- HOT
    expert[1] tripleDisp   n=  36  min=      14  avg=      20  max=      32  (us)  <-- HOT
    expert[1] TOTAL        n=  36  min=      23  avg=      30  max=      41  (us)  <-- HOT
    expert[2] sendInfoCal  n=  36  min=       6  avg=      10  max=      17  (us)  <-- HOT
    expert[2] tripleDisp   n=  36  min=      12  avg=      19  max=      29  (us)  <-- HOT
    expert[2] TOTAL        n=  36  min=      19  avg=      30  max=      42  (us)  <-- HOT
    expert[3] sendInfoCal  n=  36  min=       6  avg=       9  max=      19  (us)  <-- HOT
    expert[3] tripleDisp   n=  36  min=      13  avg=      18  max=      28  (us)  <-- HOT
    expert[3] TOTAL        n=  36  min=      20  avg=      28  max=      40  (us)  <-- HOT

  --- [AIC] WAVE-by-WAVE  (us, avg across active cores) ---
    exp  wave | entr->g1B |      wDsp |    g1Frst |     g1Mid |     g1Drn |    g12gap |      wSwi |    g2Frst |     g2Mid |     g2Drn |    g2->ex |     TOTAL
    ---------------------------------------------------------------------------------------------------------------------------------------------------------
      1     0 |         0 |        24 |        26 |         0 |         0 |        21 |         0 |        13 |         0 |         0 |         0 |        79
      1     1 |         0 |         0 |        22 |         0 |         0 |        12 |         0 |        13 |         0 |         0 |         0 |        46
      2     2 |         0 |         0 |        28 |         0 |         0 |        22 |         0 |        14 |         0 |         0 |         0 |        61
      2     3 |         0 |         0 |        24 |         0 |         0 |        13 |         0 |        12 |         0 |         0 |         0 |        47
      3     4 |         0 |        30 |        26 |         0 |         0 |        21 |         0 |        14 |         0 |         0 |         0 |        84
      3     5 |         0 |         0 |        23 |         0 |         0 |        13 |         0 |        12 |         0 |         0 |         0 |        46
      3     6 |         0 |         0 |        26 |         0 |         0 |        20 |         0 |        14 |         0 |         0 |         0 |        58
      3     7 |         0 |         0 |        22 |         0 |         0 |        13 |         0 |        12 |         0 |         0 |         0 |        46
    ---------------------------------------------------------------------------------------------------------------------------------------------------------
          avg |         0 |         7 |        25 |         0 |         0 |        17 |         0 |        13 |         0 |         0 |         0 |        58
          max |         1 |        43 |        32 |         1 |         1 |        26 |         2 |        17 |         1 |         1 |         1 |       106

  --- [AIV0] WAVE-by-WAVE  (us, avg across active cores) ---
    exp  wave | entr->g1B |      wCub |     epiFr |     epiMd |     epiDr |    g1->ex |     TOTAL
    ---------------------------------------------------------------------------------------------
      1     0 |         0 |        50 |        11 |         0 |         0 |        22 |        78
      1     1 |         0 |        11 |        12 |         0 |         0 |         0 |        22
      2     2 |         0 |        28 |        12 |         0 |         0 |        22 |        59
      2     3 |         0 |        11 |        12 |         0 |         0 |         0 |        22
      3     4 |         0 |        56 |        11 |         0 |         0 |        21 |        82
      3     5 |         0 |        11 |        12 |         0 |         0 |         0 |        21
      3     6 |         0 |        26 |        10 |         0 |         0 |        22 |        55
      3     7 |         0 |        12 |        11 |         0 |         0 |         0 |        22
    ---------------------------------------------------------------------------------------------
          avg |         0 |        26 |        11 |         0 |         0 |        11 |        45
          max |         1 |        67 |        14 |         1 |         1 |        28 |       106

  --- [AIV1] WAVE-by-WAVE  (us, avg across active cores) ---
    exp  wave | entr->g2B |     wRowR |     cbCpy |    cbTrip |     cbSnd |      cbMd |     cbEnd |    g2->ex |     TOTAL
    ---------------------------------------------------------------------------------------------------------------------
      1     0 |         1 |        33 |         1 |         0 |        35 |         0 |         0 |         0 |        64
      1     1 |        64 |         0 |         1 |         0 |        27 |         0 |         0 |         0 |        90
      2     2 |         0 |         1 |         0 |         0 |        25 |         0 |         0 |         0 |        25
      2     3 |        25 |         0 |         1 |         0 |        18 |         0 |         0 |         0 |        43
      3     4 |         0 |        17 |         1 |         0 |        26 |         0 |         0 |         0 |        40
      3     5 |        40 |         0 |         1 |         0 |        21 |         0 |         0 |         0 |        61
      3     6 |         0 |        26 |         1 |         0 |        27 |         0 |         0 |         0 |        49
      3     7 |        49 |         0 |         0 |         0 |        20 |         0 |         0 |         0 |        69
    ---------------------------------------------------------------------------------------------------------------------
          avg |        22 |         9 |         1 |         0 |        25 |         0 |         0 |         0 |        55
          max |       102 |        50 |         3 |         1 |        52 |         1 |         1 |         1 |       141
