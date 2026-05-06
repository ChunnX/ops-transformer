import csv

def parse_timestamps(csv_file="timestamps.csv"):
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        data = {row['core_name']: [int(row[f'point_{i}']) for i in range(40)] for row in reader}
    
    NUM_GROUPS = 10
    AIV_OFFSET = NUM_GROUPS  # AIV 的 globalCoreIdx 从 10 开始
    NUM_EXPERTS = 4
    
    # 常量（与 kernel 一致）
    GMM1_BASE = 2
    GMM2_BASE = 2 + 2 * NUM_EXPERTS  # = 10
    COMBINE_BASE = 2 + 4 * NUM_EXPERTS  # = 18
    INIT_TIME_POS = 39  # 进入Process的初始时间
    
    print(f"=== 前10组group各模块耗时 (us) ===")
    print(f"布局: init(39), routing(0-1), gmm1+dispatch({GMM1_BASE}-{GMM1_BASE+2*NUM_EXPERTS-1}), "
          f"gmm2({GMM2_BASE}-{GMM2_BASE+2*NUM_EXPERTS-1}), combine({COMBINE_BASE}-{COMBINE_BASE+1})\n")
    
    for group_idx in range(NUM_GROUPS):
        aic_key = f"AIC_core{group_idx}"
        aiv_sub0_key = f"AIV_core{AIV_OFFSET + 2 * group_idx}"
        aiv_sub1_key = f"AIV_core{AIV_OFFSET + 2 * group_idx + 1}"
        
        if aic_key not in data or aiv_sub0_key not in data or aiv_sub1_key not in data:
            print(f"Group {group_idx}: 数据缺失")
            continue
        
        aic = data[aic_key]
        aiv_sub0 = data[aiv_sub0_key]
        aiv_sub1 = data[aiv_sub1_key]
        
        # 检查初始时间
        init_diff_sub0 = aiv_sub0[INIT_TIME_POS] - aic[INIT_TIME_POS]
        init_diff_sub1 = aiv_sub1[INIT_TIME_POS] - aic[INIT_TIME_POS]
        
        routing = aiv_sub0[1] - aiv_sub0[0]
        combine = aiv_sub0[COMBINE_BASE + 1] - aiv_sub0[COMBINE_BASE]
        
        print(f"--- Group {group_idx} ---")
        print(f"  init_time: AIC={aic[INIT_TIME_POS]}, AIV_sub0={aiv_sub0[INIT_TIME_POS]}, AIV_sub1={aiv_sub1[INIT_TIME_POS]}")
        print(f"  init_diff: sub0-aic={init_diff_sub0}, sub1-aic={init_diff_sub1}")
        print(f"  routing: {routing}, combine: {combine}")
        
        dispatch_total = 0
        gmm1_total = 0
        gmm2_total = 0
        
        for i in range(NUM_EXPERTS):
            d = aic[GMM1_BASE + 2*i + 1] - aiv_sub1[2*i]
            g1 = aic[GMM2_BASE + 2*i] - aic[GMM1_BASE + 2*i]
            g2 = aic[GMM2_BASE + 2*i + 1] - aic[GMM2_BASE + 2*i]
            
            dispatch_total += d
            gmm1_total += g1
            gmm2_total += g2
            
            print(f"  expert_{i}: dispatch={d}, gmm1={g1}, gmm2={g2}")
        
        print(f"  --- 汇总 ---")
        print(f"  dispatch_total={dispatch_total}, gmm1_total={gmm1_total}, gmm2_total={gmm2_total}")
        print()

parse_timestamps()
