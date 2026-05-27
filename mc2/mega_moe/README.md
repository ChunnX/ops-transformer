========== [rank 0] MegaMoe TIMESTAMPS (unit = us) ==========
  active AIC : 36/36, active AIV0: 36/36, active AIV1: 36/36

  --- [AIC] STAGE durations (per-core delta) ---
    KERNEL_ENTRY -> INIT_DONE             n=  36  min=       0  avg=       0  max=       1  (us)
    INIT_DONE -> QUANT_DONE            n=  36  min=       0  avg=       0  max=       0  (us)
    QUANT_DONE -> EXPERT_LOOP_END       n=  36  min=     173  avg=     376  max=     404  (us)  <-- HOT
    EXPERT_LOOP_END -> PROCESS_END           n=  36  min=       0  avg=       0  max=       1  (us)

  --- [AIV0] STAGE durations (per-core delta) ---
    KERNEL_ENTRY -> INIT_DONE             n=  36  min=       0  avg=       0  max=       1  (us)
    INIT_DONE -> QUANT_DONE            n=  36  min=       6  avg=       6  max=       8  (us)  <-- HOT
    QUANT_DONE -> EXPERT_LOOP_END       n=  36  min=     166  avg=     354  max=     383  (us)  <-- HOT
    EXPERT_LOOP_END -> PROCESS_END           n=  36  min=     106  avg=     158  max=     355  (us)  <-- HOT

  --- [AIV1] STAGE durations (per-core delta) ---
    KERNEL_ENTRY -> INIT_DONE             n=  36  min=       0  avg=       0  max=       1  (us)
    INIT_DONE -> QUANT_DONE            n=  36  min=       6  avg=       6  max=       8  (us)  <-- HOT
    QUANT_DONE -> EXPERT_LOOP_END       n=  36  min=     201  avg=     414  max=     458  (us)  <-- HOT
    EXPERT_LOOP_END -> PROCESS_END           n=  36  min=      32  avg=      99  max=     318  (us)  <-- HOT

  --- [AIV1] per-expert DISPATCH duration ---
    expert[0] sendInfoCal  n=  36  min=      16  avg=      31  max=      45  (us)  <-- HOT
    expert[0] tripleDisp   n=  36  min=      21  avg=      28  max=      38  (us)  <-- HOT
    expert[0] TOTAL        n=  36  min=      52  avg=      60  max=      67  (us)  <-- HOT
    expert[1] sendInfoCal  n=  36  min=      13  avg=      16  max=      20  (us)  <-- HOT
    expert[1] tripleDisp   n=  36  min=      22  avg=      24  max=      27  (us)  <-- HOT
    expert[1] TOTAL        n=  36  min=      38  avg=      41  max=      44  (us)  <-- HOT
    expert[2] sendInfoCal  n=  36  min=      12  avg=      19  max=      31  (us)  <-- HOT
    expert[2] tripleDisp   n=  36  min=      19  avg=      25  max=      32  (us)  <-- HOT
    expert[2] TOTAL        n=  36  min=      31  avg=      45  max=      56  (us)  <-- HOT
    expert[3] sendInfoCal  n=  36  min=      14  avg=      20  max=      36  (us)  <-- HOT
    expert[3] tripleDisp   n=  36  min=      22  avg=      27  max=      40  (us)  <-- HOT
    expert[3] TOTAL        n=  36  min=      36  avg=      48  max=      71  (us)  <-- HOT

  --- [AIC] WAVE-by-WAVE  (us, avg across active cores) ---
    exp  wave | entr->g1B |      wDsp |    g1Frst |     g1Mid |     g1Drn |    g12gap |      wSwi |    g2Frst |     g2Mid |     g2Drn |    g2->ex |     TOTAL
    ---------------------------------------------------------------------------------------------------------------------------------------------------------
      1     0 |         0 |        21 |        26 |         0 |         0 |        21 |         0 |        13 |         0 |         0 |         0 |        76
      1     1 |         0 |         0 |        23 |         0 |         0 |        12 |         0 |        12 |         0 |         0 |         0 |        45
      2     2 |         0 |         0 |        28 |         0 |         0 |        21 |         0 |        14 |         0 |         0 |         0 |        60
      2     3 |         0 |         0 |        23 |         0 |         0 |        14 |         0 |        12 |         0 |         0 |         0 |        46
      3     4 |         0 |        13 |        26 |         0 |         0 |        21 |         0 |        13 |         0 |         0 |         0 |        69
      3     5 |         0 |         0 |        22 |         0 |         0 |        12 |         0 |        12 |         0 |         0 |         0 |        45
      3     6 |         0 |         6 |        27 |         0 |         0 |        21 |         0 |        14 |         0 |         0 |         0 |        64
      3     7 |         0 |         0 |        23 |         0 |         0 |        13 |         0 |        13 |         0 |         0 |         0 |        47
    ---------------------------------------------------------------------------------------------------------------------------------------------------------
          avg |         0 |         5 |        25 |         0 |         0 |        17 |         0 |        13 |         0 |         0 |         0 |        57
          max |         1 |        36 |        31 |         1 |         1 |        25 |         2 |        16 |         1 |         1 |         1 |       102

  --- [AIV0] WAVE-by-WAVE  (us, avg across active cores) ---
    exp  wave | entr->g1B |      wCub |     epiFr |     epiMd |     epiDr |    g1->ex |     TOTAL
    ---------------------------------------------------------------------------------------------
      1     0 |         0 |        47 |        11 |         0 |         0 |        21 |        74
      1     1 |         0 |        11 |        11 |         0 |         0 |         0 |        21
      2     2 |         0 |        28 |        12 |         0 |         0 |        21 |        58
      2     3 |         0 |        11 |        12 |         0 |         0 |         0 |        22
      3     4 |         0 |        39 |        11 |         0 |         0 |        21 |        67
      3     5 |         0 |        11 |        11 |         0 |         0 |         0 |        21
      3     6 |         0 |        32 |        11 |         0 |         0 |        22 |        61
      3     7 |         0 |        11 |        12 |         0 |         0 |         0 |        22
    ---------------------------------------------------------------------------------------------
          avg |         0 |        24 |        11 |         0 |         0 |        11 |        43
          max |         1 |        63 |        14 |         2 |         1 |        27 |       100

  --- [AIV1] WAVE-by-WAVE  (us, avg across active cores) ---
    exp  wave | entr->g2B |     wRowR |     cbCpy |    cbTrip |     cbSnd |      cbMd |     cbEnd |    g2->ex |     TOTAL
    ---------------------------------------------------------------------------------------------------------------------
      1     0 |         1 |        15 |         1 |         0 |        31 |         0 |         0 |         0 |        45
      1     1 |        44 |         0 |         1 |         0 |        23 |         0 |         0 |         0 |        67
      2     2 |         1 |         0 |         1 |         0 |        16 |         0 |         0 |         0 |        17
      2     3 |        17 |         0 |         1 |         0 |        13 |         0 |         0 |         0 |        31
      3     4 |         0 |         0 |         1 |         0 |        16 |         0 |         0 |         0 |        17
      3     5 |        16 |         0 |         1 |         0 |        13 |         0 |         0 |         0 |        30
      3     6 |         0 |        37 |         1 |         0 |        33 |         0 |         0 |         0 |        65
      3     7 |        64 |         0 |         0 |         0 |        23 |         0 |         0 |         0 |        87
    ---------------------------------------------------------------------------------------------------------------------
          avg |        18 |         6 |         1 |         0 |        21 |         0 |         0 |         0 |        45
          max |       104 |        60 |         3 |         2 |        47 |         1 |         1 |         1 |       134

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
          cbCpy  =  CombineTile: CopyTileFromGM (GMM2 out → UB)
         cbTrip  =  CombineTile: load triple tensor
          cbSnd  =  CombineTokensOptimize (UB → remote win)
           cbMd  =  combine mid+last tiles
          cbEnd  =  combine end -> GMM2 end
         g2->ex  =  GMM2 end -> wave end
          TOTAL  =  TOTAL wave
========== [rank 0] END TIMESTAMPS ==========